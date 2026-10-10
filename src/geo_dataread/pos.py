"""PBO ``.pos`` station time series (GLOBK ``tssum`` output), numpy only.

The daily SINEX job on okada (``make_daily_sinex.sh``) keeps the glorg output
with ``PBOP`` lines, and ``tssum`` turns those into one file per station:
``/D/SNX/pos/<SITE>.imo.final_itr08.pos`` (read-only mount on the laptop:
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
- :func:`open_pos_times` — ``(yearf, data, ddata)`` in the shape and units of
  :func:`geo_dataread.gps_read.openGlobkTimes` (metres, rows N/E/U), so the
  existing chain reads ``.pos`` with ``tType="POS"``.

Time: :attr:`PosSeries.yearf` follows GLOBK — ``year + (doy − 1 + hours/24) /
days_in_year`` at the solution epoch (11:59 for the daily runs) — so it equals
the ``mb_`` fractional years.
"""

from __future__ import annotations

import dataclasses
import datetime
import gzip
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_POS_DIR",
    "DEFAULT_PROD_ID",
    "PosError",
    "PosSeries",
    "open_pos_times",
    "pos_path",
    "read_pos",
]

FloatArray = npt.NDArray[np.float64]

#: okada's archive as seen through the read-only mount on the laptop
DEFAULT_POS_DIR = "/mnt_data/okada/D/SNX/pos"
#: product id ``tssum`` writes into the file name (``<SITE>.<prod_id>.pos``)
DEFAULT_PROD_ID = "imo.final_itr08"

_NCOL = 25  # date time mjd  X Y Z  Sx Sy Sz  Rxy Rxz Ryz  lat lon h  dN dE dU  Sn Se Su  Rne Rnu Reu  soln


class PosError(ValueError):
    """A ``.pos`` file that does not parse."""


@dataclasses.dataclass(frozen=True)
class PosSeries:
    """One station's ``.pos`` time series; arrays have one row per epoch.

    ``xyz``/``sxyz`` and ``dneu``/``sneu`` are in metres; ``rxyz`` holds the
    correlations (XY, XZ, YZ) and ``rneu`` (NE, NU, EU). ``llh`` is latitude,
    longitude (degrees, longitude in 0…360 as written) and height (m).
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
        out = np.empty(len(self))
        for i, t in enumerate(
            self.epochs.astype("datetime64[s]").astype(datetime.datetime)
        ):
            start = datetime.datetime(t.year, 1, 1)
            ndays = (datetime.datetime(t.year + 1, 1, 1) - start).days
            out[i] = t.year + (t - start).total_seconds() / 86400.0 / ndays
        return out

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


def _ref(value: str, path: Path, what: str) -> FloatArray:
    try:
        return np.array([float(x) for x in value.split()[:3]])
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
        num = np.array([r[2:24] for r in rows], dtype=float)
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
    """``<directory>/<STA>.<prod_id>.pos`` (or ``.pos.gz``); FileNotFoundError if neither exists."""
    d = Path(directory) if directory is not None else Path(DEFAULT_POS_DIR)
    for name in (f"{sta.upper()}.{prod_id}.pos", f"{sta.upper()}.{prod_id}.pos.gz"):
        if (d / name).is_file():
            return d / name
    raise FileNotFoundError(
        f"no .pos time series for station '{sta}' ({prod_id}) in {d}"
    )


def open_pos_times(
    sta: str, directory: str | Path | None = None, prod_id: str = DEFAULT_PROD_ID
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(yearf, data, ddata)`` like :func:`geo_dataread.gps_read.openGlobkTimes`.

    ``data``/``ddata`` are ``(3, n)``: rows dN, dE, dU and their sigmas, in
    metres, relative to the file's NEU reference position (the caller removes
    its own offset, as for the ``mb_`` files).
    """
    s = read_pos(pos_path(sta, directory, prod_id))
    return s.yearf, s.dneu.T.copy(), s.sneu.T.copy()
