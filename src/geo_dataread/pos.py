"""PBO ``.pos`` station time series (GLOBK ``tssum`` output), numpy only.

The daily SINEX job on okada (``make_daily_sinex.sh``) keeps the glorg output
with ``PBOP`` lines, and ``tssum`` turns those into one file per station:
``/D/SNX/pos/<SITE>.imo.daily_itr08.pos`` (read-only mount on the laptop:
``/mnt_data/okada/D/SNX/pos``). Each row is one daily solution with

- the absolute ECEF position X/Y/Z, its sigmas and the X–Y/X–Z/Y–Z correlations;
- geodetic latitude/longitude/height;
- dN/dE/dU from the file's NEU reference position, their sigmas and correlations.

This replaces the ``mb_<STA>_TOT.dat{1,2,3}`` files: absolute coordinates (no
±10 m wrap), the full 3×3 per-station covariance, and a documented format.
Like ``mb_`` files it has no covariance BETWEEN stations; that lives in the
daily SINEX (:mod:`geo_dataread.sinex`).

Pieces:

- :func:`read_pos` — path (``.pos`` or ``.pos.gz``) → :class:`PosSeries`.
- :func:`pos_path` — station id + directory → the file.
- :func:`pos_local` — N/E/U (and their 3×3 covariance) computed from the
  ECEF X/Y/Z and covariance relative to a reference position WE choose
  (:data:`REFERENCE_MODES`), after the reviewed junction offsets
  (``junction_offsets.csv``, e.g. GUSK's antenna height) are applied in
  ECEF. tssum's own dN/dE/dU columns are the GLOBK ``Loc.`` convention
  relative to the file's header position; they are kept in
  :class:`PosSeries` but not used by default.
- :func:`open_pos_times` — ``(yearf, data, ddata)`` in the shape and units of
  :func:`geo_dataread.gps_read.openGlobkTimes` (metres, rows N/E/U), so the
  existing chain reads ``.pos`` with ``tType="POS"``.

Time: :attr:`PosSeries.yearf` follows GLOBK — ``year + (doy − 1 + hours/24) /
days_in_year`` at the solution epoch (11:59 for the daily runs) — so it equals
the ``mb_`` fractional years. The conversions are gtimes' (``dTimetoYearf``;
``gtimes.mjd_to_yearf`` etc. for the ``mjd`` column).
"""

from __future__ import annotations

import dataclasses
import datetime
import gzip
from pathlib import Path

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from geofunc import local
from gtimes.timefunc import dTimetoYearf

if TYPE_CHECKING:
    from geo_dataread.globk_tot import JunctionOffset

__all__ = [
    "DEFAULT_FRAME",
    "DEFAULT_POS_DIR",
    "DEFAULT_PROD_ID",
    "DEFAULT_SESSION",
    "FRAMES",
    "LEGACY_IDS",
    "LEGACY_PROD_ID",
    "SESSIONS",
    "REFERENCE_MODES",
    "PosError",
    "PosSeries",
    "apriori_xyz",
    "junction_dlocal",
    "merge_pos",
    "open_pos_times",
    "pos_local",
    "product_junctions",
    "pos_frame",
    "frame_name",
    "pos_path",
    "POS_PRODUCTS",
    "pos_product",
    "product_id",
    "tssum_id",
    "read_pos",
    "write_pos",
]

FloatArray = npt.NDArray[np.float64]

#: okada's archive as seen through the read-only mount on the laptop
DEFAULT_POS_DIR = "/mnt_data/okada/D/SNX/pos"
#: the reference frames a ``.pos`` set comes in, and the frame of the plate-motion
#: model removed from it (``--ref plate``). itrf14 is IGb14 and itrf20 is IGb20,
#: the IGS's GNSS realizations of those ITRFs; the labels (plot titles) name
#: the ITRF only.
FRAMES: dict[str, tuple[str, str]] = {
    "itrf08": ("ITRF2008", "ITRF2008"),
    "itrf14": ("ITRF2014", "ITRF2014"),
    "itrf20": ("ITRF2020", "ITRF2020"),
}
#: solution session → the ``<series>`` part of the product id (also tssum's
#: solution-type column, ``24hrs``/``08hrs``)
SESSIONS: dict[str, str] = {"24h": "24hr", "8h": "08hr"}
DEFAULT_FRAME = "itrf08"
DEFAULT_SESSION = "24h"


def product_id(frame: str = DEFAULT_FRAME, session: str = DEFAULT_SESSION) -> str:
    """``imo.<session>_<frame>``: ``("itrf14", "24h")`` → ``"imo.24hr_itrf14"``.

    15 characters, the width tssum keeps; tssum is run with
    :func:`tssum_id` so its solution-type column says the session.
    """
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; one of {', '.join(FRAMES)}")
    if session not in SESSIONS:
        raise ValueError(f"unknown session {session!r}; one of {', '.join(SESSIONS)}")
    return f"imo.{SESSIONS[session]}_{frame}"


def tssum_id(prod_id: str) -> str:
    """The id to give tssum: ``imo.24hr_itrf14`` → ``imo.24hr_itrf14+24hrs``.

    tssum names the files after the first 15 characters and takes characters
    17–21 as the solution type, written in the last column of every row.
    """
    if len(prod_id) != 15:
        raise ValueError(f"{prod_id!r}: a tssum product id has 15 characters")
    return f"{prod_id}+{prod_id[4:8]}s"


#: product id → (frame label, plate-motion-model frame), side by side in ``pos_dir``
POS_PRODUCTS: dict[str, tuple[str, str]] = {
    product_id(f, s): FRAMES[f] for f in FRAMES for s in SESSIONS
}
#: product id written into the file name (``<SITE>.<prod_id>.pos``) by default
DEFAULT_PROD_ID = product_id()
#: earlier names of the same sets, tried in order when the new name has no file:
#: okada's cron still writes ``imo.daily_itr08`` (``imo.final_itr08`` until
#: 2026-10-11); the laptop's sets were ``imo.daily_igb14`` / ``imo.final_igb20``
#: until 2026-10-13
LEGACY_IDS: dict[str, tuple[str, ...]] = {
    "imo.24hr_itrf08": ("imo.daily_itr08", "imo.final_itr08"),
    "imo.24hr_itrf14": ("imo.daily_igb14",),
    "imo.24hr_itrf20": ("imo.final_igb20",),
}
LEGACY_PROD_ID = "imo.final_itr08"
_RENAMED = {old: new for new, olds in LEGACY_IDS.items() for old in olds}
#: short names → frame
_FRAME_ALIASES = {
    "itrf08": "itrf08",
    "itr08": "itrf08",
    "itrf2008": "itrf08",
    "itrf14": "itrf14",
    "igb14": "itrf14",
    "itrf2014": "itrf14",
    "itrf20": "itrf20",
    "igb20": "itrf20",
    "itrf2020": "itrf20",
}


def pos_frame(name: str | None) -> str:
    """Frame key of a short name, frame or product id: ``"igb14"`` → ``"itrf14"``."""
    if name is None:
        return DEFAULT_FRAME
    if "." in name:
        pid = _RENAMED.get(name, name)
        frame = pid.rsplit("_", 1)[-1]
        if frame not in FRAMES:
            raise ValueError(f"{name!r}: not a frame product ({', '.join(FRAMES)})")
        return frame
    try:
        return _FRAME_ALIASES[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown frame {name!r}; one of itrf2008, itrf2014, itrf2020 "
            "(or itrf08/itrf14/itrf20, igb14/igb20)"
        ) from None


def frame_name(name: str | None) -> str:
    """Full user-facing frame name: ``"itrf14"``/``"igb14"`` → ``"itrf2014"``.

    The short keys (``itrf14``) stay inside product ids and file names of the
    .pos sets, where tssum's 15-character limit leaves no room; figure names,
    titles, logs and help use this full name.
    """
    return FRAMES[pos_frame(name)][1].lower()


def pos_product(name: str | None, session: str = DEFAULT_SESSION) -> str:
    """Full product id from a frame name: ``"igb14"`` → ``"imo.24hr_itrf14"``.

    ``None`` is the default product. Accepts the frames ``itrf08``/``itrf14``/
    ``itrf20`` and their aliases (``itr08``/``itrf2008``, ``igb14``/``itrf2014``,
    ``igb20``/``itrf2020``), or any full id (containing a dot; an earlier name
    is mapped to its new one).
    """
    if name is not None and "." in name:
        return _RENAMED.get(name, name)
    return product_id(pos_frame(name), session)


#: how :func:`pos_local` picks the reference position: a station a-priori
#: coordinate (``station_coord.xyz``), a statistic of the series
#: (:func:`geofunc.local.reference_position`), or ``"tssum"`` — tssum's own
#: dN/dE/dU columns, unchanged (GLOBK ``Loc.`` convention, header reference)
REFERENCE_MODES = ("wmean", "mean", "median", "first", "apriori", "tssum")
ReferenceMode = Literal["wmean", "mean", "median", "first", "apriori", "tssum"]

_NCOL = 25  # date time mjd  X Y Z  Sx Sy Sz  Rxy Rxz Ryz  lat lon h  dN dE dU  Sn Se Su  Rne Rnu Reu  soln


class PosError(ValueError):
    """A ``.pos`` file that does not parse."""


@dataclasses.dataclass(frozen=True)
class PosSeries:
    """One station's ``.pos`` time series; arrays have one row per epoch.

    ``xyz``/``sxyz`` and ``dneu``/``sneu`` are in metres; ``rxyz`` holds the
    correlations (XY, XZ, YZ) and ``rneu`` (NE, NU, EU). ``llh`` is latitude,
    longitude (degrees, longitude in 0…360 as written) and height (m).
    Fields tssum could not fit (Fortran ``*****`` overflow, e.g. dN/dE when the
    reference position is far from the site) are NaN.
    """

    station: str
    epochs: npt.NDArray[np.datetime64]
    mjd: FloatArray
    xyz: FloatArray
    sxyz: FloatArray
    rxyz: FloatArray
    llh: FloatArray
    dneu: FloatArray
    sneu: FloatArray
    rneu: FloatArray
    soln: npt.NDArray[np.str_]
    ref_xyz: FloatArray
    ref_llh: FloatArray
    header: dict[str, str]

    def __len__(self) -> int:
        return len(self.mjd)

    @property
    def yearf(self) -> FloatArray:
        """GLOBK fractional year of each epoch (equals the ``mb_`` time column)."""
        return np.array(
            [
                dTimetoYearf(t)
                for t in self.epochs.astype("datetime64[s]").astype(datetime.datetime)
            ],
            dtype=float,
        )

    def cov_xyz(self) -> FloatArray:
        """Per-epoch 3×3 X/Y/Z covariance, shape ``(n, 3, 3)``, m²."""
        return _cov(self.sxyz, self.rxyz)

    def cov_neu(self) -> FloatArray:
        """Per-epoch 3×3 N/E/U covariance, shape ``(n, 3, 3)``, m²."""
        return _cov(self.sneu, self.rneu)


def _cov(s: FloatArray, r: FloatArray) -> FloatArray:
    c: FloatArray = np.einsum("ni,nj->nij", s, s)
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        c[:, i, j] *= r[:, k]
        c[:, j, i] *= r[:, k]
    return c


def _open_text(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        return f.read().splitlines()


def _num(tok: str) -> float:
    """Fortran overflow (``*****``) → NaN; tssum writes it for |dN/dE| ≥ 10⁴ m."""
    return float("nan") if tok.startswith("*") else float(tok)


def _ref(value: str, path: Path, what: str) -> FloatArray:
    try:
        return np.array([_num(x) for x in value.split()[:3]])
    except (ValueError, IndexError) as exc:
        raise PosError(f"{path}: bad {what} {value!r}") from exc


def read_pos(path: str | Path) -> PosSeries:
    """Read one ``.pos`` file (plain or gzipped)."""
    path = Path(path)
    lines = _open_text(path)
    header: dict[str, str] = {}
    start = None
    for i, line in enumerate(lines):
        if line.startswith("*YYYYMMDD"):
            start = i + 1
            break
        key, sep, value = line.partition(":")
        if sep and key.strip() and not line.startswith(" "):
            header.setdefault(key.strip(), value.strip())
    if (
        start is None
        or not lines
        or not lines[0].startswith("PBO Station Position Time Series")
    ):
        raise PosError(f"{path}: not a PBO .pos file")
    station = header.get("4-character ID", path.name[:4]).strip()

    rows = [ln.split() for ln in lines[start:] if ln.strip()]
    bad = [k for k, r in enumerate(rows) if len(r) != _NCOL]
    if bad:
        raise PosError(
            f"{path}: row {start + bad[0] + 1} has {len(rows[bad[0]])} columns, expected {_NCOL}"
        )
    if rows:
        try:
            num = np.array([[_num(x) for x in r[2:24]] for r in rows])
        except ValueError as exc:
            raise PosError(f"{path}: {exc}") from exc
        stamps = [
            f"{r[0][:4]}-{r[0][4:6]}-{r[0][6:8]}T{r[1][:2]}:{r[1][2:4]}:{r[1][4:6]}"
            for r in rows
        ]
        epochs = np.array(stamps, dtype="datetime64[s]")
        soln = np.array([r[24] for r in rows])
    else:
        num = np.empty((0, 22))
        epochs = np.array([], dtype="datetime64[s]")
        soln = np.array([], dtype=str)

    return PosSeries(
        station=station,
        epochs=epochs,
        mjd=num[:, 0],
        xyz=num[:, 1:4],
        sxyz=num[:, 4:7],
        rxyz=num[:, 7:10],
        llh=num[:, 10:13],
        dneu=num[:, 13:16],
        sneu=num[:, 16:19],
        rneu=num[:, 19:22],
        soln=soln,
        ref_xyz=_ref(header.get("XYZ Reference position", ""), path, "XYZ reference"),
        ref_llh=_ref(header.get("NEU Reference position", ""), path, "NEU reference"),
        header=header,
    )


def pos_path(
    sta: str, directory: str | Path | None = None, prod_id: str = DEFAULT_PROD_ID
) -> Path:
    """``<directory>/<STA>.<prod_id>.pos`` (or ``.pos.gz``); FileNotFoundError if none exists.

    When there is none under ``prod_id``, a file under one of its earlier names
    (:data:`LEGACY_IDS`) is used: okada's set, and sets built before a rename.
    """
    d = Path(directory) if directory is not None else Path(DEFAULT_POS_DIR)
    prod_id = _RENAMED.get(prod_id, prod_id)
    ids = [prod_id, *LEGACY_IDS.get(prod_id, ())]
    for pid in ids:
        for name in (f"{sta.upper()}.{pid}.pos", f"{sta.upper()}.{pid}.pos.gz"):
            if (d / name).is_file():
                return d / name
    raise FileNotFoundError(
        f"no .pos time series for station '{sta}' ({prod_id}) in {d}"
    )


def _header_lonlat(path: Path) -> tuple[float, float] | None:
    """(lon, lat) in degrees, lon in [-180, 180), from a ``.pos`` header's
    ``NEU Reference position`` -- the header only, never the rows."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        for _ in range(40):
            line = fh.readline()
            if not line or line.startswith("*YYYYMMDD"):
                break
            if line.startswith("NEU Reference position"):
                try:
                    lat, lon = (float(x) for x in line.partition(":")[2].split()[:2])
                except ValueError:
                    return None
                return ((lon + 180.0) % 360.0) - 180.0, lat
    return None


def pos_stations(
    prod_id: str = DEFAULT_PROD_ID,
    directory: str | Path | None = None,
    region: tuple[float, float, float, float] | None = None,
) -> list[tuple[str, float, float]]:
    """Every station with a ``.pos`` series of ``prod_id`` in ``directory``.

    ``(STA, lon, lat)`` sorted by station, the position read from each file's
    header (cheap: no rows are parsed). Files under the product's earlier names
    (:data:`LEGACY_IDS`) count, as in :func:`pos_path`. ``region`` (W, E, S, N
    in degrees, lon in [-180, 180)) keeps only stations inside it; a file
    whose header has no readable position is then left out.
    """
    d = Path(directory) if directory is not None else Path(DEFAULT_POS_DIR)
    prod_id = _RENAMED.get(prod_id, prod_id)
    found: dict[str, Path] = {}
    for pid in (prod_id, *LEGACY_IDS.get(prod_id, ())):
        for suffix in (".pos", ".pos.gz"):
            for path in sorted(d.glob(f"*.{pid}{suffix}")):
                found.setdefault(path.name.split(".", 1)[0].upper(), path)
    out: list[tuple[str, float, float]] = []
    for sta in sorted(found):
        ll = _header_lonlat(found[sta])
        lon, lat = ll if ll is not None else (float("nan"), float("nan"))
        if region is not None:
            w, e, s_, n = region
            if ll is None or not (w <= lon <= e and s_ <= lat <= n):
                continue
        out.append((sta, lon, lat))
    return out


def apriori_xyz(sta: str, path: str | Path | None = None) -> FloatArray | None:
    """The station's a-priori ECEF coordinate ``(3,)`` from ``station_coord.xyz``.

    ``path`` defaults to postprocess.cfg ``[PATHS] coord_file`` (via
    gps_parser). Lines are ``X Y Z SITE``. None when the station is not listed.
    """
    if path is None:
        from gps_parser import ConfigParser

        path = ConfigParser().getPostProcessDir("coord_file")
    with open(Path(path).expanduser()) as f:
        for line in f:
            tok = line.split()
            if len(tok) >= 4 and tok[3].upper() == sta.upper():
                return np.array([float(v) for v in tok[:3]])
    return None


def junction_dlocal(
    yearf: npt.ArrayLike, rules: Sequence[JunctionOffset]
) -> FloatArray:
    """Per-epoch N/E/U offsets ``(n, 3)`` [m] from reviewed junction rules.

    Each rule adds ``offset_m`` to its component for the epochs strictly
    before ``before_year`` — the same rule ``gps-globk-tot`` applies to the
    TOT series, here as a local vector for
    :func:`geofunc.local.apply_local_offsets`.
    """
    y = np.asarray(yearf, dtype=float)
    d = np.zeros((len(y), 3))
    for rule in rules:
        d[y < rule.before_year, "NEU".index(rule.component.upper())] += rule.offset_m
    return d


def _default_junctions(sta: str) -> tuple[JunctionOffset, ...]:
    from geo_dataread.globk_tot import resolve_junction_offsets

    rules, _ = resolve_junction_offsets(None)
    return rules.get(sta.upper(), ())


#: frames whose ``.pos`` products are built from okada's GAMIT h-files: the
#: reviewed junction offsets (``junction_offsets.csv``) correct okada's processing
#: (e.g. GUSK's stale 1.093 m station.info height before 2023), so they apply to
#: these only. itrf20 and the 8 h sets come from gamit-p01's processing, whose
#: station.info has the right heights; a junction applied there makes the step.
OKADA_FRAMES = ("itrf08", "itrf14")


def product_junctions(prod_id: str) -> Sequence[JunctionOffset] | None:
    """Junction rules for a product: the deployed ones (None) or none (``()``).

    None (= :func:`pos_local`'s default, the deployed catalog) for okada's
    products — and for any id that is not a frame product, as before; ``()``
    for products processed elsewhere (itrf20, the 8 h sessions).
    """
    try:
        frame = pos_frame(prod_id)
    except ValueError:
        return None
    if frame in OKADA_FRAMES and not _RENAMED.get(prod_id, prod_id).startswith(
        "imo.08hr"
    ):
        return None
    return ()


def pos_local(
    s: PosSeries,
    ref: ReferenceMode | npt.ArrayLike = "wmean",
    *,
    junctions: (
        Sequence[JunctionOffset] | Mapping[str, Sequence[JunctionOffset]] | None
    ) = None,
    convention: local.Convention = "geodetic",
    coord_file: str | Path | None = None,
) -> tuple[local.Local, FloatArray | None]:
    """N/E/U displacements ``(n, 3)`` [m] + covariance ``(n, 3, 3)`` from the X/Y/Z.

    1. The junction offsets (``junctions``: rules for this station, a
       ``{station: rules}`` mapping, or None = the deployed
       ``junction_offsets.csv``) are added in ECEF along the station's own
       axes, so every product derived from the coordinates includes them.
    2. The reference position ``ref``: an ECEF point ``(3,)``, ``"apriori"``
       (``station_coord.xyz``; falls back to ``"wmean"`` when the station is
       not listed) or a statistic of the corrected series — ``"wmean"`` (the
       least-squares mean with the full 3×3 covariances), ``"mean"``,
       ``"median"``, ``"first"``.
    3. :func:`geofunc.local.displacements` with ``convention`` —
       ``"geodetic"`` (true local level) or ``"globk"`` (glorg ``Loc.``, the
       TOT/tssum convention).

    ``ref="tssum"`` returns tssum's own columns untouched (no junction
    offsets, header reference, GLOBK convention, σ only on the diagonal).
    Returns ``(Local(d, cov), ref_xyz)``; ``ref_xyz`` is None for "tssum".
    """
    if isinstance(ref, str) and ref == "tssum":
        cov = np.zeros((len(s), 3, 3))
        idx = np.arange(3)
        cov[:, idx, idx] = s.sneu**2
        return local.Local(s.dneu.copy(), cov), None
    if junctions is None:
        rules: Sequence[JunctionOffset] = _default_junctions(s.station)
    elif isinstance(junctions, Mapping):
        rules = tuple(junctions.get(s.station.upper(), ()))
    else:
        rules = junctions
    xyz = s.xyz
    if rules:
        xyz = local.apply_local_offsets(xyz, junction_dlocal(s.yearf, rules))
    cov = s.cov_xyz()
    if isinstance(ref, str):
        if ref not in REFERENCE_MODES:
            raise ValueError(f"ref must be one of {REFERENCE_MODES} or an ECEF point")
        ref_xyz = apriori_xyz(s.station, coord_file) if ref == "apriori" else None
        if ref_xyz is None:
            how = "wmean" if ref == "apriori" else ref
            ref_xyz = local.reference_position(xyz, cov=cov, how=how)  # type: ignore[arg-type]
    else:
        ref_xyz = np.asarray(ref, dtype=float)
    return local.displacements(xyz, ref_xyz, cov=cov, convention=convention), ref_xyz


def open_pos_times(
    sta: str,
    directory: str | Path | None = None,
    prod_id: str = DEFAULT_PROD_ID,
    *,
    ref: ReferenceMode | npt.ArrayLike = "wmean",
    convention: local.Convention = "geodetic",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(yearf, data, ddata)`` like :func:`geo_dataread.gps_read.openGlobkTimes`.

    ``data``/``ddata`` are ``(3, n)``: rows dN, dE, dU and their sigmas, in
    metres, from :func:`pos_local` (junction offsets applied, relative to
    ``ref``, ``convention``). ``ref="tssum"`` gives tssum's own columns.
    """
    s = read_pos(pos_path(sta, directory, prod_id))
    out, _ = pos_local(
        s, ref, junctions=product_junctions(prod_id), convention=convention
    )
    sig = np.sqrt(np.diagonal(out.cov, axis1=-2, axis2=-1))  # type: ignore[arg-type]
    return s.yearf, out.d.T.copy(), sig.T.copy()


_FIELDS = """Start Field Description
YYYYMMDD      Year, month, day for the given position epoch
HHMMSS        Hour, minute, second for the given position epoch
JJJJJ.JJJJJ   Modified Julian day for the given position epoch
X             X coordinate, Specified Reference Frame, meters
Y             Y coordinate, Specified Reference Frame, meters
Z             Z coordinate, Specified Reference Frame, meters
Sx            Standard deviation of the X position, meters
Sy            Standard deviation of the Y position, meters
Sz            Standard deviation of the Z position, meters
Rxy           Correlation of the X and Y position
Rxz           Correlation of the X and Z position
Ryz           Correlation of the Y and Z position
Nlat          North latitude, WGS-84 ellipsoid, decimal degrees
Elong         East longitude, WGS-84 ellipsoid, decimal degrees
Height (Up)   Height relative to WGS-84 ellipsoid, m
dN            Difference in North component from NEU reference position, meters
dE            Difference in East component from NEU reference position, meters
du            Difference in vertical component from NEU reference position, meters
Sn            Standard deviation of dN, meters
Se            Standard deviation of dE, meters
Su            Standard deviation of dU, meters
Rne           Correlation of dN and dE
Rnu           Correlation of dN and dU
Reu           Correlation of dEand dU
Soln          "rapid", "final", "suppl/suppf", "campd", or "repro" corresponding to products  generated with rapid or final orbit products, in supplemental processing, campaign data processing or reprocessing
End Field Description
*YYYYMMDD HHMMSS JJJJJ.JJJJ         X             Y             Z            Sx        Sy       Sz     Rxy   Rxz    Ryz            NLat         Elong         Height         dN        dE        dU         Sn       Se       Su      Rne    Rnu    Reu  Soln"""

_MJD0 = np.datetime64("1858-11-17T00:00:00", "s")


def _sig_corr(c: FloatArray) -> tuple[FloatArray, FloatArray]:
    sd = np.sqrt(np.diagonal(c, axis1=-2, axis2=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.stack(
            [
                c[..., 0, 1] / (sd[..., 0] * sd[..., 1]),
                c[..., 0, 2] / (sd[..., 0] * sd[..., 2]),
                c[..., 1, 2] / (sd[..., 1] * sd[..., 2]),
            ],
            axis=-1,
        )
    return sd, np.nan_to_num(r)


def write_pos(
    path: str | Path,
    station: str,
    epochs: npt.ArrayLike,
    xyz: npt.ArrayLike,
    cov: npt.ArrayLike,
    *,
    frame: str,
    soln: str = "final",
    ref_xyz: npt.ArrayLike | None = None,
    name: str | None = None,
    released: datetime.datetime | None = None,
) -> Path:
    """Write a PBO ``.pos`` (format 1.1.1, as tssum) that :func:`read_pos` reads.

    ``epochs`` are datetimes; like tssum they are written truncated to the
    minute (GLOBK's 11:59:45 day epoch → ``115900``, MJD ``….4993``), which
    is what keeps ``yearf`` equal to the TOT time column. ``cov (n, 3, 3)``
    is the X/Y/Z covariance [m²]. dN/dE/dU and their sigmas/correlations are
    in the GLOBK ``Loc.`` convention relative to ``ref_xyz`` (default: the
    covariance-weighted mean position), lat/lon/height on WGS-84.
    """
    t = (
        np.asarray(epochs, dtype="datetime64[s]")
        .astype("datetime64[m]")
        .astype("datetime64[s]")
    )
    x = np.asarray(xyz, dtype=np.float64)
    c = np.asarray(cov, dtype=np.float64)
    if not (len(t) == len(x) == len(c)) or len(t) == 0:
        raise ValueError("epochs, xyz and cov must be non-empty and the same length")
    order = np.argsort(t, kind="stable")
    t, x, c = t[order], x[order], c[order]
    ref = (
        local.reference_position(x, cov=c, how="wmean")
        if ref_xyz is None
        else np.asarray(ref_xyz, dtype=np.float64)
    )
    sxyz, rxyz = _sig_corr(c)
    g = local.ecef_to_geodetic(x, ellipsoid=local.WGS84)
    g0 = local.ecef_to_geodetic(ref, ellipsoid=local.WGS84)
    loc = local.displacements(x, ref, cov=c, convention="globk")
    assert loc.cov is not None
    sneu, rneu = _sig_corr(loc.cov)
    mjd = (t - _MJD0).astype(np.float64) / 86400.0
    stamp = [str(v).replace("-", "").replace(":", "").split("T") for v in t]
    released = released or datetime.datetime.now(datetime.timezone.utc)
    deg = np.degrees
    head = [
        f"PBO Station Position Time Series. Reference Frame : {frame}",
        "Format Version: 1.1.1",
        f"4-character ID: {station.upper()[:4]}",
        f"Station name  : {(name or station.upper()[:4] + '_GPS'):<16}",
        f"First Epoch   : {stamp[0][0]} {stamp[0][1]}",
        f"Last Epoch    : {stamp[-1][0]} {stamp[-1][1]}",
        f"Release Date  : {released:%Y%m%d %H%M%S}",
        f"XYZ Reference position : {ref[0]:15.5f}{ref[1]:15.5f}{ref[2]:15.5f} ({frame})",
        f"NEU Reference position : {deg(g0.lat):16.10f}{deg(g0.lon) % 360:16.10f}{g0.h:11.5f} ({frame}/WGS84)",
        _FIELDS,
    ]
    rows = []
    for k in range(len(t)):
        rows.append(
            f" {stamp[k][0]} {stamp[k][1]} {mjd[k]:10.4f} "
            f"{x[k, 0]:14.5f} {x[k, 1]:14.5f} {x[k, 2]:14.5f} "
            f"{sxyz[k, 0]:8.5f} {sxyz[k, 1]:8.5f} {sxyz[k, 2]:8.5f} "
            f"{rxyz[k, 0]:6.3f} {rxyz[k, 1]:6.3f} {rxyz[k, 2]:6.3f}     "
            f"{deg(g.lat[k]):15.10f} {deg(g.lon[k]) % 360:15.10f} {g.h[k]:10.5f}   "
            f"{loc.d[k, 0]:9.5f} {loc.d[k, 1]:9.5f} {loc.d[k, 2]:9.5f}   "
            f"{sneu[k, 0]:8.5f} {sneu[k, 1]:8.5f} {sneu[k, 2]:8.5f} "
            f"{rneu[k, 0]:6.3f} {rneu[k, 1]:6.3f} {rneu[k, 2]:6.3f} {soln}"
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("\n".join(head + rows) + "\n")
    tmp.replace(out)
    return out


def merge_pos(
    path: str | Path, new: str | Path | None, replace: Iterable[str]
) -> Path | None:
    """Fold the rows of ``new`` into the ``.pos`` at ``path`` (in place).

    Rows of ``path`` whose date (``YYYYMMDD``) is in ``replace`` are dropped,
    every row of ``new`` (``None``: none) is added, and the rows are re-sorted.
    Rows outside ``replace`` keep their text byte for byte; the header is
    ``new``'s (write it with ``ref_xyz`` = the old reference so dN/dE/dU stay
    comparable), else the old one, with the first/last epoch of the merged
    series. Without ``path``, ``new`` is moved there; a merge leaving no rows
    removes the file (returns ``None``). ``new`` is consumed.
    """
    path = Path(path)
    if new is not None and not path.exists():
        return Path(Path(new).replace(path))
    head, old = _split_rows(path.read_text().splitlines(), path)
    rows: list[str] = []
    if new is not None:
        head, rows = _split_rows(Path(new).read_text().splitlines(), Path(new))
        Path(new).unlink()
    gone = set(replace)
    merged = sorted(
        [ln for ln in old if ln.split()[0] not in gone] + rows,
        key=lambda ln: ln.split()[:2],
    )
    if not merged:
        path.unlink()
        return None
    for k, ln in enumerate(head):
        if ln.startswith("First Epoch"):
            head[k] = "First Epoch   : " + " ".join(merged[0].split()[:2])
        elif ln.startswith("Last Epoch"):
            head[k] = "Last Epoch    : " + " ".join(merged[-1].split()[:2])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(head + merged) + "\n")
    tmp.replace(path)
    return path


def _split_rows(lines: list[str], path: Path) -> tuple[list[str], list[str]]:
    for i, line in enumerate(lines):
        if line.startswith("*YYYYMMDD"):
            return lines[: i + 1], [ln for ln in lines[i + 1 :] if ln.strip()]
    raise PosError(f"{path}: not a PBO .pos file")
