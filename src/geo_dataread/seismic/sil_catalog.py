"""SIL day and period assembly onto the common schema (ticket T3, PLAN-seismicity.md).

:func:`read_sil_day` turns one SIL day directory (``<base>/<yyyy>/<mon>/<dd>/``)
into an events frame of :mod:`geo_dataread.seismic.schema`;
:func:`read_sil_catalog` walks the day directories of a period. Both return
``(events, report)``; the :class:`SilReadReport` counts every join, fallback,
dropped row and missing file, and rides on ``events.attrs["catalog"]``.

**Which files a day has decides what it holds, never its date.** Reviewed
events come from ``events.fps`` / ``events.lib`` / ``lib.mag`` when any of them
exists (they still appear on scattered days after February 2026); automatic
events from ``cleanaut.mag``, else ``aut.mag``.

Reviewed joins (never by row order: the row counts of the three files differ):

* **spine** = ``events.fps`` outer-joined with ``events.lib`` on ``id`` +
  ``time``. Rows that agree on ``id`` but not on ``time`` are joined by id
  when the id is unique on both sides (``fps_lib_time_conflict``). fps rows
  that are not in events.lib (the 1995 ``FILE`` rows) stay in
  (``fps_only``); 1990 has no fps, so events.lib alone is the spine there.
* ``lib.mag`` has no id: it joins the spine on ``time`` + latitude +
  longitude (rounded to 0.001°). A lib.mag row that joins nothing has no id
  to stand on, so it is dropped and counted (``lib_mag_unjoined``).

Column sources (reviewed events; first source that has a value wins):

=====================  ===================================================
Column                 Source
=====================  ===================================================
``time``               the spine (fps and events.lib agree)
``event_id``           the file id (``sil_path``)
lat, lon, depth        events.fps → events.lib → lib.mag
``mag_mlw``            lib.mag → events.lib (0.01) → events.fps (0.1)
``mag_ml``             lib.mag ``ml``
``magnitude``          ``mag_mlw`` (``Mlw``), else ``mag_ml`` (``ML``)
``mag_auto``           the matched automatic row's ML
``moment_nm``          events.fps ``seismic_moment``
``h_err_km``           :func:`horizontal_error_km` of lib.mag → fps errors
``depth_err_km``       lib.mag ``d_depth`` only (fps keeps it mod 100: NA
                       and ``depth_err_unsafe`` when lib.mag is missing)
``event_type(_code)``  lib.mag → events.fps → events.lib (events.lib is
                       blank for early rows)
``n_phases``           lib.mag ``num_of_p_pha + num_of_s_pha`` (= SeisComP
                       ``usedPhaseCount`` of the back-import, 16 on
                       2024-08-22 #1)
``azimuthal_gap``      lib.mag ``max_azim_gap`` (= SeisComP 102 on #1)
``quality``            NA (Q is a property of automatic solutions)
``sil_aut_path``       the matched automatic id (cleanaut, else aut.mag)
=====================  ===================================================

Automatic events (``evaluation_mode="automatic"``) are, by default, the
``aut.mag`` rows that match no reviewed event (``mode="catalog"``: an
automatic row stands until its event is reviewed): ``magnitude`` =
``mag_auto`` = their ML labelled ``aut.mag``, ``quality`` = Q,
``event_type`` ``earthquake`` (as the Quakes API types them; no code in the
file). ``cleanaut.mag`` is aut.mag with some rows replaced by alert2sil rows
(PGV magnitude, Q 50.50 as a marker, position rounded, id ``/2000/aaa/…``);
their PGV magnitude goes to ``mag_pgv`` of the aut.mag row they replaced
(same origin time). ``automatic_file="cleanaut.mag"`` reads that file as
written instead — the Quakes API's SIL ingest appears to read it (T9).
``mode="automatic"`` returns every automatic row, reviewed or not;
``mode="reviewed"`` none. Automatic rows below ``min_quality`` (default
Q 10) are dropped and the rest get ``possible_duplicate``
(:func:`duplicate_flags`). Layouts without an id (pre-1996) get
``event_id`` ``/<yyyy>/<mon>/<dd>/aut.mag#<num>`` and ``sil_path`` NA.

Reviewed ↔ automatic matching: first by id; then, among the rows left on both
sides, greedily one-to-one by time and distance (best score first), within
``match_tol`` when the automatic row has an id and ``match_tol_no_id``
before 1996. A reviewed id can differ from its automatic id by 1 ms or in the
5-minute directory, and reviewed manual ids (``…/62:25:000``) carry no
automatic id at all; on 2023-01..03 the id pass matched 4471 of 4758 and
the fallback pairs sit at dt < 0.7 s / < 8 km, while the same-event pairs of
1995 (no ids) are up to 4.3 s / 15 km apart.

Values outside the schema's ranges (aut.mag ML 12.6, say) become NA and are
counted (``out_of_range:<column>``); a row with an impossible latitude or
longitude is dropped and counted.
"""

from __future__ import annotations

import dataclasses
import os
import time as _time
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from geo_dataread.seismic.schema import (
    EVENT_COLUMNS,
    catalog_provenance,
    coerce_catalog,
    sil_event_type,
    with_provenance,
)
from geo_dataread.seismic.sil_files import FileKind, ParseReport, read_sil_file

SOURCE: Final = "sil-files"

#: Environment variable naming the SIL archive root.
ENV_SIL_DIR: Final = "GPS_SIL_DIR"

#: ``postprocess.cfg`` ``[PATHS]`` key naming the SIL archive root.
CONFIG_KEY: Final = "sil_dir"

#: Lower-case English month directory names, fixed (``strftime("%b")`` is
#: locale dependent).
MONTHS: Final = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)

#: km per degree of latitude (and of longitude at the equator), the value
#: SeisComP's SIL back-import uses.
KM_PER_DEG: Final = 111.19

#: Position rounding (degrees) of the lib.mag join key.
_POS_DECIMALS: Final = 3

_REVIEWED_KINDS: Final[tuple[FileKind, ...]] = ("events.fps", "events.lib", "lib.mag")


@dataclasses.dataclass(frozen=True)
class MatchTolerance:
    """Time/space window of the reviewed ↔ automatic fallback match."""

    seconds: float
    km: float


#: Fallback window when the automatic row has an id (1996 →): Q6's first
#: tolerance. ``MatchTolerance(0, 0)`` switches the fallback off. Measured 2023-01..03: id-mismatched twins at dt < 0.7 s.
DEFAULT_MATCH_TOL: Final = MatchTolerance(2.0, 10.0)

#: Fallback window when the automatic row has no id (before 1996), where it
#: is the only match: Q6's second tolerance (1995-06-15 pairs: ≤ 4.3 s, 15 km).
DEFAULT_MATCH_TOL_NO_ID: Final = MatchTolerance(5.0, 25.0)

#: What a catalogue read returns (PLAN-seismicity §9e):
#: ``"catalog"`` — reviewed events, plus the automatic solutions of events
#: not (yet) reviewed: an automatic row stands until its event is reviewed,
#: then the reviewed row replaces it; ``"automatic"`` — every automatic
#: solution, reviewed or not, the same kind of row in every period (a
#: series that does not change character at the February 2026 seam);
#: ``"reviewed"`` — reviewed events only.
Mode = Literal["catalog", "automatic", "reviewed"]
MODES: Final[tuple[Mode, ...]] = ("catalog", "automatic", "reviewed")

#: Automatic file of a catalogue read. ``aut.mag`` (default) is SIL's own
#: automatic solutions; ``cleanaut.mag`` is aut.mag with some rows replaced
#: by alert2sil rows (PGV magnitude, Q 50.50 as a marker, position rounded
#: to 0.01°, fake id ``/2000/aaa/…``), the file the Quakes API ingests.
CatalogAutomaticFile = Literal["aut.mag", "cleanaut.mag"]

#: Default lower bound on automatic Q (inclusive). aut.mag holds Q ≥ 10 only
#: from about 2012 (1996: 79 % of its rows are below 10), so the default
#: applies one cut to every period. 2026 waveform check: Q ≤ 10 ~90 % junk,
#: Q 10–40 ~78 % real, Q > 40 ~81 %. Raise it in swarms (PLAN §9e).
DEFAULT_MIN_QUALITY: Final = 10.0

#: Window in which an automatic row counts as a possible duplicate of a
#: better solution (Q6's first tolerance). aut.mag Q > 40, Feb–Nov 2026:
#: 35 neighbours within 2 s / 10 km of 10,846 rows; events.ref best
#: solutions 264 of 11,368.
DEFAULT_DUPLICATE_TOL: Final = MatchTolerance(2.0, 10.0)

#: Id prefix of the alert2sil rows in cleanaut.mag.
ALERT2SIL_PREFIX: Final = "/2000/aaa/"

#: alert2sil row ↔ aut.mag row: same origin time (to 0.1 s), position
#: rounded to 0.01° (2026-06-13: dt 0.0 s, ≤ 0.6 km).
ALERT2SIL_TOL: Final = MatchTolerance(0.5, 5.0)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SilReadReport:
    """Counts of what an assembly did; reports of several days add up.

    ``counts`` keys (absent = 0): ``days_read``, ``days_missing``,
    ``days_reviewed``, ``days_automatic_only``, ``days_empty``;
    ``file_missing:<kind>``, ``file_empty:<kind>``, ``rows:<kind>``,
    ``malformed:<kind>``, ``field_warnings:<kind>``, ``duplicates:<kind>``;
    the reviewed joins ``fps_lib_joined``, ``fps_only``, ``lib_only``,
    ``fps_lib_time_conflict``, ``lib_mag_joined``, ``lib_mag_unjoined``,
    ``reviewed_without_lib_mag``; column choices ``mlw_from:<kind>``,
    ``magnitude_ml_fallback``, ``magnitude_missing``, ``mlw_disagree``
    (lib.mag vs events.lib by more than 0.005), ``event_type_disagree``,
    ``depth_err_unsafe``; the automatic side ``automatic_from:<kind>``,
    ``match_id``, ``match_id_aut_mag``, ``match_time_space``,
    ``reviewed_unmatched``, ``automatic_added``, ``aut_mag_only``;
    ``out_of_range:<column>``, ``dropped_bad_position``; ``off_day``
    (event date differs from its directory's date); and the totals
    ``events_reviewed`` / ``events_automatic`` over every directory read
    (a period read includes one padding day each side), and, for a period,
    ``window_reviewed`` / ``window_automatic``: the rows returned. ``issues`` holds one line per
    malformed file line (``path:line: reason``). ``fallback_dt_s`` the |dt|
    of each time/space match, for checking the tolerance.
    """

    counts: Mapping[str, int] = dataclasses.field(default_factory=dict)
    issues: tuple[str, ...] = ()
    fallback_dt_s: tuple[float, ...] = ()
    seconds: float = 0.0

    def __add__(self, other: SilReadReport) -> SilReadReport:
        c: Counter[str] = Counter(self.counts)
        c.update(other.counts)
        return SilReadReport(
            dict(c),
            self.issues + other.issues,
            self.fallback_dt_s + other.fallback_dt_s,
            self.seconds + other.seconds,
        )

    def get(self, key: str) -> int:
        """A count (0 when absent)."""
        return int(self.counts.get(key, 0))

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary (counts, issue count, a few issues)."""
        return {
            "counts": {k: int(v) for k, v in sorted(self.counts.items())},
            "n_issues": len(self.issues),
            "issues": list(self.issues[:20]),
            "n_fallback": len(self.fallback_dt_s),
            "seconds": round(self.seconds, 3),
        }

    def summary(self) -> str:
        """One line for a log."""
        return (
            f"sil-files: {self.get('days_read')} days "
            f"({self.get('days_missing')} missing), "
            f"{self.get('events_reviewed')} reviewed + "
            f"{self.get('events_automatic')} automatic, "
            f"match id/aut.mag/time-space {self.get('match_id')}/"
            f"{self.get('match_id_aut_mag')}/{self.get('match_time_space')}, "
            f"{len(self.issues)} malformed lines"
        )


class _Tally:
    def __init__(self) -> None:
        self.c: Counter[str] = Counter()
        self.issues: list[str] = []
        self.fallback_dt: list[float] = []

    def add(self, key: str, n: int = 1) -> None:
        if n:
            self.c[key] += int(n)

    def report(self, seconds: float = 0.0) -> SilReadReport:
        return SilReadReport(
            dict(self.c), tuple(self.issues), tuple(self.fallback_dt), seconds
        )


# --------------------------------------------------------------------------
# Base path and dates
# --------------------------------------------------------------------------


def _config_sil_dir() -> str | None:
    try:
        from gps_parser import ConfigParser

        value = ConfigParser().getPostProcessDir(CONFIG_KEY)
    except Exception:  # noqa: BLE001 - no gpsconfig, or the key is not deployed
        return None
    return str(value) if value else None


def sil_dir(base: str | os.PathLike[str] | None = None) -> Path:
    """The SIL archive root: ``base``, ``$GPS_SIL_DIR``, then ``postprocess.cfg``
    ``[PATHS] sil_dir``. Raises :class:`FileNotFoundError` naming all three
    when none is set (there is no built-in default path)."""
    if base:
        return Path(base).expanduser()
    env = os.environ.get(ENV_SIL_DIR)
    if env:
        return Path(env).expanduser()
    cfg = _config_sil_dir()
    if cfg:
        return Path(cfg).expanduser()
    raise FileNotFoundError(
        "no SIL archive path: pass base=, set $GPS_SIL_DIR, or set "
        f"[PATHS] {CONFIG_KEY} in postprocess.cfg (e.g. /mnt_data/sildata)"
    )


def day_dir(base: str | os.PathLike[str], day: date) -> Path:
    """``<base>/<yyyy>/<mon>/<dd>`` for ``day``."""
    return Path(base) / f"{day.year:04d}" / MONTHS[day.month - 1] / f"{day.day:02d}"


def _dir_date(path: Path) -> date | None:
    try:
        dd, mon, yyyy = path.parts[-1], path.parts[-2], path.parts[-3]
        return date(int(yyyy), MONTHS.index(mon) + 1, int(dd))
    except (IndexError, ValueError):
        return None


def _utc(t: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _days(first: date, last: date) -> Iterator[date]:
    d = first
    while d <= last:
        yield d
        d += timedelta(days=1)


# --------------------------------------------------------------------------
# File reading
# --------------------------------------------------------------------------


def _read(ddir: Path, kind: FileKind, tally: _Tally) -> pd.DataFrame | None:
    """A day file's frame, ``None`` when the file is absent."""
    path = ddir / kind
    if not path.is_file():
        tally.add(f"file_missing:{kind}")
        return None
    frame, rep = read_sil_file(path)
    _tally_parse(rep, tally)
    if rep.lines == 0:
        tally.add(f"file_empty:{kind}")
    return frame


def _tally_parse(rep: ParseReport, tally: _Tally) -> None:
    tally.add(f"rows:{rep.kind}", rep.rows)
    tally.add(f"malformed:{rep.kind}", rep.n_malformed)
    tally.add(f"field_warnings:{rep.kind}", len(rep.field_warnings))
    tally.issues.extend(f"{rep.path}:{m.line_no}: {m.reason}" for m in rep.malformed)


def _dedupe(df: pd.DataFrame, keys: list[str], kind: str, tally: _Tally) -> Any:
    """Drop exact repeats of ``keys`` (a file listing an event twice)."""
    if not len(df):
        return df
    dup = df.duplicated(keys, keep="first")
    tally.add(f"duplicates:{kind}", int(dup.sum()))
    return df.loc[~dup.to_numpy()]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def horizontal_error_km(dlat_deg: Any, dlon_deg: Any, latitude: Any) -> Any:
    """Horizontal 1σ (km) from SIL's latitude/longitude errors in degrees.

    The larger of the two axes, ``Dlat·111.19`` and ``Dlon·111.19·cos(lat)``.
    SeisComP's SIL back-import converts both axes with 111.19 and no
    ``cos(lat)`` (so its longitude axis is about 2.2× larger in Iceland);
    compare with that in mind.
    """
    dlat = np.asarray(dlat_deg, dtype="float64") * KM_PER_DEG
    dlon = (
        np.asarray(dlon_deg, dtype="float64")
        * KM_PER_DEG
        * np.cos(np.radians(np.asarray(latitude, dtype="float64")))
    )
    with np.errstate(invalid="ignore"):
        return np.fmax(dlat, dlon)


def _distance_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Equirectangular distance (km), plenty for a few tens of km."""
    lat1, lon1, lat2, lon2 = (
        np.asarray(v, dtype="float64") for v in (lat1, lon1, lat2, lon2)
    )
    mid = np.radians((lat1 + lat2) / 2.0)
    dy = (lat2 - lat1) * KM_PER_DEG
    dx = (lon2 - lon1) * KM_PER_DEG * np.cos(mid)
    return np.hypot(dx, dy)


def _first(*cols: Any) -> Any:
    """Element-wise first non-NA of aligned Series."""
    out = cols[0].copy()
    for c in cols[1:]:
        out = out.where(out.notna(), c)
    return out


def _col(df: pd.DataFrame, name: str, dtype: Any = "float64") -> Any:
    if name in df.columns:
        return df[name]
    return pd.Series(pd.NA if dtype == "Int64" else np.nan, index=df.index, dtype=dtype)


def _prefixed(df: pd.DataFrame | None, prefix: str) -> pd.DataFrame | None:
    if df is None or not len(df):
        return None
    keep = {c: f"{prefix}{c}" for c in df.columns if c not in ("id", "time")}
    return df.rename(columns=keep)


# --------------------------------------------------------------------------
# Reviewed events
# --------------------------------------------------------------------------


def _spine(fps: pd.DataFrame | None, lib: pd.DataFrame | None, tally: _Tally) -> Any:
    """events.fps ⟗ events.lib on id + time (by id alone when only time differs)."""
    f = _prefixed(fps, "f_")
    lb = _prefixed(lib, "l_")
    if f is None and lb is None:
        return None
    if f is None or lb is None:
        only = f if f is not None else lb
        assert only is not None
        tally.add("fps_only" if lb is None else "lib_only", len(only))
        return only.reset_index(drop=True)
    m = f.merge(lb, on=["id", "time"], how="outer", indicator=True)
    left = m["_merge"] == "left_only"
    right = m["_merge"] == "right_only"
    # same id, different time: join by id when it is unique on both sides
    ids_l = m.loc[left, "id"]
    ids_r = m.loc[right, "id"]
    common = set(ids_l[~ids_l.duplicated(keep=False)]) & set(
        ids_r[~ids_r.duplicated(keep=False)]
    )
    common.discard(None)
    if common:
        fl = m.loc[left & m["id"].isin(common)].set_index("id")
        fr = m.loc[right & m["id"].isin(common)].set_index("id")
        lcols = [c for c in m.columns if c.startswith("l_")]
        fl[lcols] = fr.loc[fl.index, lcols]
        fl["_merge"] = "both"
        fl = fl.reset_index()
        rest = m.loc[~(m["id"].isin(common) & (left | right))]
        m = pd.concat([rest, fl], ignore_index=True)
        tally.add("fps_lib_time_conflict", len(common))
    state = m["_merge"].astype(str)
    tally.add("fps_lib_joined", int((state == "both").sum()) - len(common))
    tally.add("fps_only", int((state == "left_only").sum()))
    tally.add("lib_only", int((state == "right_only").sum()))
    return m.drop(columns="_merge").reset_index(drop=True)


def _join_lib_mag(spine: pd.DataFrame, mag: pd.DataFrame | None, tally: _Tally) -> Any:
    """Add lib.mag columns (``m_*``) by time + rounded position."""
    if mag is None or not len(mag):
        tally.add("reviewed_without_lib_mag", len(spine))
        return spine
    lat = _first(_col(spine, "f_latitude"), _col(spine, "l_latitude"))
    lon = _first(_col(spine, "f_longitude"), _col(spine, "l_longitude"))
    s = spine.assign(
        _klat=lat.round(_POS_DECIMALS).to_numpy(),
        _klon=lon.round(_POS_DECIMALS).to_numpy(),
    )
    s["_k"] = s.groupby(["time", "_klat", "_klon"], dropna=False).cumcount()
    mg = mag.rename(columns={c: f"m_{c}" for c in mag.columns if c != "time"})
    mg = mg.assign(
        _klat=mag["latitude"].round(_POS_DECIMALS).to_numpy(),
        _klon=mag["longitude"].round(_POS_DECIMALS).to_numpy(),
    )
    mg["_k"] = mg.groupby(["time", "_klat", "_klon"], dropna=False).cumcount()
    keys = ["time", "_klat", "_klon", "_k"]
    out = s.merge(mg, on=keys, how="left", indicator=True)
    joined = out["_merge"] == "both"
    tally.add("lib_mag_joined", int(joined.sum()))
    tally.add("reviewed_without_lib_mag", int((~joined).sum()))
    tally.add("lib_mag_unjoined", len(mag) - int(joined.sum()))
    return out.drop(columns=[*keys[1:], "_merge"])


def _reviewed(ddir: Path, tally: _Tally) -> pd.DataFrame | None:
    fps = _read(ddir, "events.fps", tally)
    lib = _read(ddir, "events.lib", tally)
    mag = _read(ddir, "lib.mag", tally)
    if fps is not None:
        fps = _dedupe(fps, ["id", "time"], "events.fps", tally)
    if lib is not None:
        lib = _dedupe(lib, ["id", "time"], "events.lib", tally)
    spine = _spine(fps, lib, tally)
    if spine is None:
        if mag is not None and len(mag):
            tally.add("lib_mag_unjoined", len(mag))
        return None
    s = _join_lib_mag(spine, mag, tally)
    n = len(s)

    lat = _first(_col(s, "f_latitude"), _col(s, "l_latitude"), _col(s, "m_latitude"))
    lon = _first(_col(s, "f_longitude"), _col(s, "l_longitude"), _col(s, "m_longitude"))
    depth = _first(_col(s, "f_depth"), _col(s, "l_depth"), _col(s, "m_depth"))

    m_mlw, l_mlw, f_mlw = _col(s, "m_mlw"), _col(s, "l_mlw"), _col(s, "f_mlw")
    for kind, col, prior in (
        ("lib.mag", m_mlw, None),
        ("events.lib", l_mlw, m_mlw),
        ("events.fps", f_mlw, _first(m_mlw, l_mlw)),
    ):
        took = col.notna() if prior is None else (col.notna() & prior.isna())
        tally.add(f"mlw_from:{kind}", int(took.sum()))
    mlw = _first(m_mlw, l_mlw, f_mlw)
    with np.errstate(invalid="ignore"):
        tally.add("mlw_disagree", int(((m_mlw - l_mlw).abs() > 0.005).sum()))
    ml = _col(s, "m_ml")
    magnitude = mlw.where(mlw.notna(), ml)
    mag_type = pd.Series(
        [
            "Mlw" if a else ("ML" if b else None)
            for a, b in zip(mlw.notna(), ml.notna(), strict=True)
        ],
        index=s.index,
        dtype=object,
    )
    tally.add("magnitude_ml_fallback", int((mlw.isna() & ml.notna()).sum()))
    tally.add("magnitude_missing", int(magnitude.isna().sum()))

    blank: Any = pd.Series(np.nan, index=s.index, dtype=object)
    codes = [
        _col(s, c, object) if c in s.columns else blank
        for c in ("m_event_type", "f_event_type", "l_event_type")
    ]
    code = _first(*codes)
    present = [c for c in codes if c is not blank]
    if len(present) > 1:
        a = present[0].astype(object)
        for b in present[1:]:
            both = a.notna() & b.notna()
            tally.add("event_type_disagree", int((both & (a != b)).sum()))
    etype = code.map(sil_event_type)

    dlat = _first(_col(s, "m_d_latitude"), _col(s, "f_d_latitude"))
    dlon = _first(_col(s, "m_d_longitude"), _col(s, "f_d_longitude"))
    depth_err = _col(s, "m_d_depth")
    tally.add(
        "depth_err_unsafe",
        int((depth_err.isna() & _col(s, "f_d_depth_mod100").notna()).sum()),
    )
    n_p = _col(s, "m_num_of_p_pha", "Int64")
    n_s = _col(s, "m_num_of_s_pha", "Int64")

    return pd.DataFrame(
        {
            "time": s["time"].to_numpy(),
            "event_id": s["id"].astype(object).to_numpy(),
            "sil_path": s["id"].astype(object).to_numpy(),
            "sil_aut_path": np.full(n, None, dtype=object),
            "latitude": lat.to_numpy(dtype="float64", na_value=np.nan),
            "longitude": lon.to_numpy(dtype="float64", na_value=np.nan),
            "depth_km": depth.to_numpy(dtype="float64", na_value=np.nan),
            "h_err_km": horizontal_error_km(dlat, dlon, lat),
            "depth_err_km": depth_err.to_numpy(dtype="float64", na_value=np.nan),
            "magnitude": magnitude.to_numpy(dtype="float64", na_value=np.nan),
            "magnitude_type": mag_type.to_numpy(),
            "mag_mlw": mlw.to_numpy(dtype="float64", na_value=np.nan),
            "mag_ml": ml.to_numpy(dtype="float64", na_value=np.nan),
            "mag_auto": np.full(n, np.nan),
            "moment_nm": _col(s, "f_seismic_moment").to_numpy(
                dtype="float64", na_value=np.nan
            ),
            "evaluation_mode": "manual",
            "event_type": etype.astype(object).to_numpy(),
            "event_type_code": code.astype(object).to_numpy(),
            "quality": np.full(n, np.nan),
            "n_phases": (n_p + n_s).to_numpy(dtype=object, na_value=None),
            "azimuthal_gap": _col(s, "m_max_azim_gap").to_numpy(
                dtype="float64", na_value=np.nan
            ),
        }
    )


# --------------------------------------------------------------------------
# Automatic events and matching
# --------------------------------------------------------------------------


def _is_alert2sil(ids: Any) -> Any:
    return np.array(
        [isinstance(v, str) and v.startswith(ALERT2SIL_PREFIX) for v in ids],
        dtype=bool,
    )


def _alert_pgv(aut: pd.DataFrame, clean: pd.DataFrame | None, tally: _Tally) -> Any:
    """``mag_pgv`` for the aut.mag rows that cleanaut.mag replaced by an
    alert2sil row (nearest within :data:`ALERT2SIL_TOL`)."""
    out = np.full(len(aut), np.nan)
    if clean is None or not len(clean) or not len(aut):
        return out
    alert = clean.loc[_is_alert2sil(clean["id"].astype(object))]
    if not len(alert):
        return out
    t_a = pd.DatetimeIndex(aut["time"]).as_unit("ns").asi8 / 1e9
    lat_a = aut["latitude"].to_numpy(dtype="float64")
    lon_a = aut["longitude"].to_numpy(dtype="float64")
    t_x = pd.DatetimeIndex(alert["time"]).as_unit("ns").asi8 / 1e9
    for k in range(len(alert)):
        near = np.flatnonzero(np.abs(t_a - t_x[k]) <= ALERT2SIL_TOL.seconds)
        if len(near):
            d = _distance_km(
                lat_a[near],
                lon_a[near],
                alert["latitude"].iat[k],
                alert["longitude"].iat[k],
            )
            ok = d <= ALERT2SIL_TOL.km
            if ok.any():
                j = int(near[ok][np.argmin(d[ok])])
                out[j] = float(alert["ml"].iat[k])
                tally.add("mag_pgv_from_alert2sil")
                continue
        tally.add("alert2sil_unmatched")
    return out


def _automatic(
    ddir: Path, tally: _Tally, file: CatalogAutomaticFile = "aut.mag"
) -> tuple[Any, Any]:
    """``(primary, aut_mag)``: the automatic rows of ``file`` (the other file
    when the day has no ``file``), and aut.mag for the id-only match when the
    primary is cleanaut.mag. Both carry ``mag_pgv``."""
    clean = _read(ddir, "cleanaut.mag", tally)
    aut = _read(ddir, "aut.mag", tally)
    if aut is not None:
        aut = _dedupe(aut, ["num", "time"], "aut.mag", tally).reset_index(drop=True)
    if clean is not None:
        clean = _dedupe(clean, ["id", "time"], "cleanaut.mag", tally).reset_index(
            drop=True
        )
    if aut is not None and (file == "aut.mag" or clean is None):
        tally.add("automatic_from:aut.mag", len(aut))
        aut["mag_pgv"] = _alert_pgv(aut, clean, tally)
        return aut, None
    if clean is not None:
        tally.add("automatic_from:cleanaut.mag", len(clean))
        is_alert = _is_alert2sil(clean["id"].astype(object))
        clean["mag_pgv"] = np.where(
            is_alert, clean["ml"].to_numpy(dtype="float64", na_value=np.nan), np.nan
        )
        tally.add("alert2sil_rows", int(is_alert.sum()))
        return clean, aut
    return None, None


def _greedy_match(
    rev: pd.DataFrame,
    aut: pd.DataFrame,
    tol: MatchTolerance,
    tol_no_id: MatchTolerance,
) -> list[tuple[int, int, float]]:
    """One-to-one (rev position, aut position, |dt| s) pairs, best score first."""
    if not len(rev) or not len(aut):
        return []
    rt = pd.DatetimeIndex(rev["time"]).as_unit("ns").asi8
    at = pd.DatetimeIndex(aut["time"]).as_unit("ns").asi8
    has_id = aut["id"].notna().to_numpy(dtype=bool)
    tol_s = np.where(has_id, tol.seconds, tol_no_id.seconds)
    tol_km = np.where(has_id, tol.km, tol_no_id.km)
    order = np.argsort(at, kind="stable")
    ats = at[order]
    wide = int(max(tol.seconds, tol_no_id.seconds) * 1e9)
    lo = np.searchsorted(ats, rt - wide, side="left")
    hi = np.searchsorted(ats, rt + wide, side="right")
    rlat = rev["latitude"].to_numpy(dtype="float64")
    rlon = rev["longitude"].to_numpy(dtype="float64")
    alat = aut["latitude"].to_numpy(dtype="float64")
    alon = aut["longitude"].to_numpy(dtype="float64")
    cand: list[tuple[float, int, int, float]] = []
    for i in range(len(rt)):
        js = order[lo[i] : hi[i]]
        if not len(js):
            continue
        dt = np.abs(at[js] - rt[i]) / 1e9
        dist = _distance_km(rlat[i], rlon[i], alat[js], alon[js])
        # a tolerance <= 0 switches the fallback off (id-only join)
        ok = (
            (tol_s[js] > 0)
            & (tol_km[js] > 0)
            & (dt <= tol_s[js])
            & (dist <= tol_km[js])
        )
        for j, d_t, d_x in zip(js[ok], dt[ok], dist[ok], strict=True):
            score = d_t / tol_s[j] + d_x / tol_km[j]
            cand.append((float(score), i, int(j), float(d_t)))
    cand.sort()
    used_r: set[int] = set()
    used_a: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for _, i, j, d_t in cand:
        if i in used_r or j in used_a:
            continue
        used_r.add(i)
        used_a.add(j)
        out.append((i, j, d_t))
    return out


def _match(
    rev: pd.DataFrame,
    aut: pd.DataFrame,
    aut_mag: pd.DataFrame | None,
    tol: MatchTolerance,
    tol_no_id: MatchTolerance,
    tally: _Tally,
) -> Any:
    """Fill ``sil_aut_path``/``mag_auto``/``mag_pgv`` of matched reviewed
    rows; return the automatic rows that match nothing."""
    if "mag_pgv" not in rev.columns:
        rev["mag_pgv"] = np.nan
    pgv_col = rev.columns.get_loc("mag_pgv")
    aut_pgv = (
        aut["mag_pgv"].to_numpy(dtype="float64")
        if "mag_pgv" in aut.columns
        else np.full(len(aut), np.nan)
    )
    matched_a = np.zeros(len(aut), dtype=bool)
    matched_r = np.zeros(len(rev), dtype=bool)
    aut_ids = aut["id"].astype(object)
    if len(rev) and len(aut):
        pos = {
            v: k for k, v in reversed(list(enumerate(aut_ids))) if isinstance(v, str)
        }
        for i, rid in enumerate(rev["sil_path"]):
            j = pos.get(rid) if isinstance(rid, str) else None
            if j is not None and not matched_a[j]:
                matched_a[j] = matched_r[i] = True
                rev.iat[i, rev.columns.get_loc("sil_aut_path")] = rid
                rev.iat[i, rev.columns.get_loc("mag_auto")] = aut["ml"].iat[j]
                rev.iat[i, pgv_col] = aut_pgv[j]
        tally.add("match_id", int(matched_r.sum()))
        r_left = np.flatnonzero(~matched_r)
        a_left = np.flatnonzero(~matched_a)
        pairs = _greedy_match(rev.iloc[r_left], aut.iloc[a_left], tol, tol_no_id)
        for i, j, d_t in pairs:
            ri, aj = int(r_left[i]), int(a_left[j])
            matched_r[ri] = matched_a[aj] = True
            rev.iat[ri, rev.columns.get_loc("sil_aut_path")] = aut_ids.iat[aj]
            rev.iat[ri, rev.columns.get_loc("mag_auto")] = aut["ml"].iat[aj]
            rev.iat[ri, pgv_col] = aut_pgv[aj]
            tally.fallback_dt.append(round(d_t, 4))
        tally.add("match_time_space", len(pairs))
    if aut_mag is not None and len(aut_mag):
        # reviewed events whose automatic row only aut.mag kept (id only)
        amag_ids = aut_mag["id"].astype(object)
        apos = {
            v: k for k, v in reversed(list(enumerate(amag_ids))) if isinstance(v, str)
        }
        for pos_r in np.flatnonzero(~matched_r):
            i = int(pos_r)
            rid = rev["sil_path"].iat[i]
            j = apos.get(rid) if isinstance(rid, str) else None
            if j is not None:
                matched_r[i] = True
                rev.iat[i, rev.columns.get_loc("sil_aut_path")] = rid
                rev.iat[i, rev.columns.get_loc("mag_auto")] = aut_mag["ml"].iat[j]
                if "mag_pgv" in aut_mag.columns:
                    rev.iat[i, pgv_col] = aut_mag["mag_pgv"].iat[j]
                tally.add("match_id_aut_mag")
        clean_ids = set(aut_ids.dropna())
        rev_aut = set(rev["sil_aut_path"].dropna()) | set(rev["sil_path"].dropna())
        only = ~amag_ids.isin(clean_ids) & ~amag_ids.isin(rev_aut)
        tally.add("aut_mag_only", int(only.sum()))
    tally.add("reviewed_unmatched", int((~matched_r).sum()))
    return aut.loc[~matched_a]


def _automatic_rows(aut: pd.DataFrame, prefix: str) -> pd.DataFrame:
    n = len(aut)
    ids = aut["id"].astype(object)
    synth = [f"{prefix}/aut.mag#{num}" for num in aut["num"].astype(object)]
    event_id = [i if isinstance(i, str) else s for i, s in zip(ids, synth, strict=True)]
    ml = aut["ml"].to_numpy(dtype="float64", na_value=np.nan)
    return pd.DataFrame(
        {
            "time": aut["time"].to_numpy(),
            "event_id": event_id,
            "sil_path": ids.to_numpy(),
            "sil_aut_path": ids.to_numpy(),
            "latitude": aut["latitude"].to_numpy(dtype="float64"),
            "longitude": aut["longitude"].to_numpy(dtype="float64"),
            "depth_km": aut["depth"].to_numpy(dtype="float64", na_value=np.nan),
            "h_err_km": np.full(n, np.nan),
            "depth_err_km": np.full(n, np.nan),
            "magnitude": ml,
            "magnitude_type": [None if np.isnan(v) else "aut.mag" for v in ml],
            "mag_mlw": np.full(n, np.nan),
            "mag_ml": np.full(n, np.nan),
            "mag_auto": ml,
            "mag_pgv": (
                aut["mag_pgv"].to_numpy(dtype="float64")
                if "mag_pgv" in aut.columns
                else np.full(n, np.nan)
            ),
            "moment_nm": np.full(n, np.nan),
            "evaluation_mode": "automatic",
            "event_type": "earthquake",
            "event_type_code": np.full(n, None, dtype=object),
            "quality": aut["q"].to_numpy(dtype="float64", na_value=np.nan),
            "n_phases": np.full(n, None, dtype=object),
            "azimuthal_gap": np.full(n, np.nan),
        }
    )


# --------------------------------------------------------------------------
# Range guard
# --------------------------------------------------------------------------

_RANGED: Final = tuple(
    c
    for c in EVENT_COLUMNS
    if c.kind in ("float64", "Int64")
    and (c.minimum is not None or c.maximum is not None)
)


def _range_guard(df: pd.DataFrame, tally: _Tally) -> pd.DataFrame:
    """NA out-of-range values (count them); drop rows with impossible positions."""
    drop = np.zeros(len(df), dtype=bool)
    for spec in _RANGED:
        if spec.name not in df.columns:
            continue
        num = pd.to_numeric(df[spec.name], errors="coerce").astype("float64").to_numpy()
        with np.errstate(invalid="ignore"):
            bad = np.zeros(len(df), dtype=bool)
            if spec.minimum is not None:
                bad |= num < spec.minimum
            if spec.maximum is not None:
                bad |= num > spec.maximum
        if spec.name in ("latitude", "longitude"):
            bad |= np.isnan(num)
        if not bad.any():
            continue
        tally.add(f"out_of_range:{spec.name}", int(bad.sum()))
        if not spec.nullable:
            drop |= bad
            continue
        df.loc[bad, spec.name] = None if spec.kind == "Int64" else np.nan
        if spec.name == "magnitude":
            df.loc[bad, "magnitude_type"] = None
    if drop.any():
        tally.add("dropped_bad_position", int(drop.sum()))
        df = df.loc[~drop]
    return df


# --------------------------------------------------------------------------
# Day assembly
# --------------------------------------------------------------------------


def _assemble_day(
    ddir: Path,
    *,
    mode: Mode,
    automatic_file: CatalogAutomaticFile,
    match_tol: MatchTolerance,
    match_tol_no_id: MatchTolerance,
    tally: _Tally,
) -> pd.DataFrame | None:
    """The raw (not yet coerced) events of one day directory."""
    tally.add("days_read")
    day = _dir_date(ddir)
    prefix = (
        f"/{day.year:04d}/{MONTHS[day.month - 1]}/{day.day:02d}"
        if day
        else (f"/{ddir.name}")
    )
    rev = _reviewed(ddir, tally) if mode != "automatic" else None
    parts: list[pd.DataFrame] = []
    if rev is not None and len(rev):
        tally.add("days_reviewed")
    if mode == "automatic":
        aut, _ = _automatic(ddir, tally, automatic_file)
        if aut is not None and len(aut):
            parts.append(_automatic_rows(aut, prefix))
    elif mode == "catalog":
        aut, aut_mag = _automatic(ddir, tally, automatic_file)
        if aut is not None and len(aut):
            rest = (
                _match(rev, aut, aut_mag, match_tol, match_tol_no_id, tally)
                if rev is not None
                else aut
            )
            if rev is None or not len(rev):
                tally.add("days_automatic_only")
            if len(rest):
                parts.append(_automatic_rows(rest, prefix))
        elif rev is not None:
            tally.add("reviewed_unmatched", len(rev))
    if rev is not None and len(rev):
        parts.insert(0, rev)
    if not parts:
        tally.add("days_empty")
        return None
    df = pd.concat(parts, ignore_index=True)
    df = _range_guard(df, tally)
    is_aut = (df["evaluation_mode"] == "automatic").to_numpy()
    tally.add("events_reviewed", int((~is_aut).sum()))
    tally.add("events_automatic", int(is_aut.sum()))
    tally.add("automatic_added", int(is_aut.sum()))
    if day is not None:
        dates = pd.DatetimeIndex(df["time"]).date
        tally.add("off_day", int(sum(d != day for d in dates)))
    return df


def _resolve_mode(mode: Mode, automatic: bool | None) -> Mode:
    """``mode``, with the older ``automatic`` flag: False = ``"reviewed"``."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}: one of {MODES}")
    if automatic is None:
        return mode
    if not automatic:
        if mode == "automatic":
            raise ValueError("automatic=False contradicts mode='automatic'")
        return "reviewed"
    return mode


def duplicate_flags(
    times: Any,
    latitude: Any,
    longitude: Any,
    automatic: Any,
    quality: Any,
    tol: MatchTolerance = DEFAULT_DUPLICATE_TOL,
) -> Any:
    """Flag automatic rows that have a better solution within ``tol``.

    A pair of rows within ``tol.seconds`` and ``tol.km`` is one event seen
    twice when at least one of them is automatic: the automatic one is
    flagged when the other is reviewed; of two automatic rows the lower Q is
    flagged (missing Q counts lowest; a tie flags the later row). Nothing is
    dropped — two real swarm events can be that close. Returns a nullable
    boolean array in the input order: True/False on automatic rows, NA on
    the others.
    """
    t = np.asarray(pd.DatetimeIndex(times).as_unit("ns").asi8, dtype="int64")
    auto = np.asarray(automatic, dtype=bool)
    n = len(t)
    worse = np.zeros(n, dtype=bool)
    if n > 1 and auto.any():
        order = np.argsort(t, kind="stable")
        ts = t[order] / 1e9
        la = np.asarray(latitude, dtype="float64")[order]
        lo = np.asarray(longitude, dtype="float64")[order]
        au = auto[order]
        q = np.asarray(pd.to_numeric(pd.Series(quality)), dtype="float64")[order]
        q = np.where(np.isnan(q), -np.inf, q)
        w = np.zeros(n, dtype=bool)
        k = 1
        while k < n:
            close = (ts[k:] - ts[:-k]) <= tol.seconds
            if not close.any():
                break
            i = np.flatnonzero(close)
            j = i + k
            near = _distance_km(la[i], lo[i], la[j], lo[j]) <= tol.km
            i, j = i[near], j[near]
            ai, aj = au[i], au[j]
            w[j[aj & (~ai | (q[j] <= q[i]))]] = True
            w[i[ai & (~aj | (q[i] < q[j]))]] = True
            k += 1
        worse[order] = w
    out = pd.array(worse, dtype="boolean")
    out[~auto] = pd.NA
    return out


def _select(
    df: pd.DataFrame,
    *,
    min_quality: float | None,
    duplicate_tol: MatchTolerance | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the Q floor to automatic rows, then flag possible duplicates.

    Works on a raw frame (``time`` column) or a schema frame (time index).
    Reviewed rows are never dropped. Returns the frame and its counts
    (``below_min_quality``, ``quality_missing_kept``, ``possible_duplicate``).
    """
    counts: dict[str, int] = {}
    if not len(df):
        return df, counts
    auto = (df["evaluation_mode"].astype(object) == "automatic").to_numpy()
    if min_quality is not None:
        q = pd.to_numeric(df["quality"]).to_numpy(dtype="float64", na_value=np.nan)
        with np.errstate(invalid="ignore"):
            low = auto & (q < float(min_quality))
        counts["below_min_quality"] = int(low.sum())
        counts["quality_missing_kept"] = int((auto & np.isnan(q)).sum())
        if low.any():
            df = df.loc[~low]
            auto = auto[~low]
    if duplicate_tol is not None:
        times = df["time"] if "time" in df.columns else df.index
        flags = duplicate_flags(
            times,
            df["latitude"].to_numpy(dtype="float64"),
            df["longitude"].to_numpy(dtype="float64"),
            auto,
            df["quality"].to_numpy(dtype="float64", na_value=np.nan),
            duplicate_tol,
        )
        df = df.copy()
        df["possible_duplicate"] = flags
        counts["possible_duplicate"] = int(np.asarray(flags.fillna(False)).sum())
    return df, {k: v for k, v in counts.items() if v}


def _finish(
    raw: list[pd.DataFrame],
    *,
    location: str,
    start: Any,
    end: Any,
    filters: Mapping[str, Any],
    report: SilReadReport,
) -> pd.DataFrame:
    frames = [f for f in raw if len(f)]
    if frames:
        df = pd.concat(frames, ignore_index=True).set_index("time")
    else:
        df = pd.DataFrame({"time": pd.DatetimeIndex([], tz="UTC")}).set_index("time")
    out = coerce_catalog(df, source=SOURCE)
    prov = catalog_provenance(
        source=SOURCE,
        location=location,
        start=start,
        end=end,
        filters=filters,
        extra={"report": report.as_dict()},
    )
    return with_provenance(out, prov)


def _filters(
    mode: Mode,
    automatic_file: CatalogAutomaticFile,
    min_quality: float | None,
    duplicate_tol: MatchTolerance | None,
    tol: MatchTolerance,
    tol_no_id: MatchTolerance,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "automatic_file": automatic_file,
        "min_quality": min_quality,
        "duplicate_tol": (
            None if duplicate_tol is None else dataclasses.asdict(duplicate_tol)
        ),
        "match_tol": dataclasses.asdict(tol),
        "match_tol_no_id": dataclasses.asdict(tol_no_id),
    }


def read_sil_day(
    day_dir: str | os.PathLike[str],
    *,
    mode: Mode = "catalog",
    automatic: bool | None = None,
    automatic_file: CatalogAutomaticFile = "aut.mag",
    min_quality: float | None = DEFAULT_MIN_QUALITY,
    duplicate_tol: MatchTolerance | None = DEFAULT_DUPLICATE_TOL,
    match_tol: MatchTolerance = DEFAULT_MATCH_TOL,
    match_tol_no_id: MatchTolerance = DEFAULT_MATCH_TOL_NO_ID,
) -> tuple[pd.DataFrame, SilReadReport]:
    """Read one SIL day directory into a schema events frame.

    ``mode`` (:data:`Mode`): ``"catalog"`` (default, Q8) — reviewed events
    from whichever of ``events.fps`` / ``events.lib`` / ``lib.mag`` exist,
    joined on id + time (see the module docstring for the column sources),
    plus the automatic rows of ``automatic_file`` that match no reviewed
    event, labelled ``evaluation_mode="automatic"``; ``"automatic"`` — every
    automatic row of ``automatic_file``, reviewed or not; ``"reviewed"`` —
    the reviewed events only (``sil_aut_path`` / ``mag_auto`` / ``mag_pgv``
    stay NA). ``automatic=False`` is the older spelling of ``"reviewed"``.

    ``automatic_file`` ``"aut.mag"`` (default) keeps SIL's own automatic
    solutions and puts the alert2sil PGV magnitude of cleanaut.mag in
    ``mag_pgv``; ``"cleanaut.mag"`` reads that file as written (Quakes API
    rows: the alert2sil rows keep their PGV magnitude and Q 50.50 marker).
    A day without the chosen file falls back to the other.

    ``min_quality`` drops automatic rows with Q below it (inclusive bound;
    missing Q is kept and counted; ``None`` keeps all); reviewed rows are
    never dropped. ``duplicate_tol`` sets ``possible_duplicate``
    (:func:`duplicate_flags`; ``None`` leaves it NA).

    Returns every event the directory holds, even one whose time falls on
    another date (``off_day``). Raises :class:`FileNotFoundError` when
    ``day_dir`` is not a directory. Missing and 0-byte files are fine.
    """
    t0 = _time.perf_counter()
    ddir = Path(day_dir)
    if not ddir.is_dir():
        raise FileNotFoundError(f"no SIL day directory {ddir}")
    mode = _resolve_mode(mode, automatic)
    tally = _Tally()
    raw = _assemble_day(
        ddir,
        mode=mode,
        automatic_file=automatic_file,
        match_tol=match_tol,
        match_tol_no_id=match_tol_no_id,
        tally=tally,
    )
    if raw is not None:
        raw, sel = _select(raw, min_quality=min_quality, duplicate_tol=duplicate_tol)
        for k, v in sel.items():
            tally.add(k, v)
    report = tally.report(_time.perf_counter() - t0)
    df = _finish(
        [] if raw is None else [raw],
        location=str(ddir),
        start=None,
        end=None,
        filters=_filters(
            mode,
            automatic_file,
            min_quality,
            duplicate_tol,
            match_tol,
            match_tol_no_id,
        ),
        report=report,
    )
    return df, report


def read_sil_catalog(
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
    *,
    base: str | os.PathLike[str] | None = None,
    mode: Mode = "catalog",
    automatic: bool | None = None,
    automatic_file: CatalogAutomaticFile = "aut.mag",
    min_quality: float | None = DEFAULT_MIN_QUALITY,
    duplicate_tol: MatchTolerance | None = DEFAULT_DUPLICATE_TOL,
    match_tol: MatchTolerance = DEFAULT_MATCH_TOL,
    match_tol_no_id: MatchTolerance = DEFAULT_MATCH_TOL_NO_ID,
) -> tuple[pd.DataFrame, SilReadReport]:
    """Read the SIL archive for the half-open window ``[start, end)``.

    ``mode``, ``automatic_file``, ``min_quality`` and ``duplicate_tol`` as
    :func:`read_sil_day`; the Q floor and the duplicate flags are applied
    to the whole window after the cut. SIL reviewed events end in early
    February 2026 (SeisComP took review): a ``"catalog"`` window across the
    seam changes from reviewed to automatic rows there; ``"automatic"``
    gives one kind of row throughout.

    Times without a zone are UTC. The archive root is ``base``, else
    ``$GPS_SIL_DIR``, else ``postprocess.cfg`` ``[PATHS] sil_dir``
    (:func:`sil_dir`). One extra day directory is read on each side (an
    event's time can fall on the day next to its directory's), then the
    events are cut to the window by origin time. A missing day directory is
    counted (``days_missing``), not an error. Returns the events frame
    (validated against the schema) and the summed :class:`SilReadReport`.
    No caching here (T4).
    """
    t0 = _time.perf_counter()
    root = sil_dir(base)
    lo, hi = _utc(start), _utc(end)
    if hi < lo:
        raise ValueError(f"end {hi} is before start {lo}")
    first = (lo - pd.Timedelta(days=1)).date()
    last = (max(hi - pd.Timedelta(1, "ns"), lo) + pd.Timedelta(days=1)).date()
    mode = _resolve_mode(mode, automatic)
    tally = _Tally()
    raw: list[pd.DataFrame] = []
    for day in _days(first, last):
        ddir = day_dir(root, day)
        if not ddir.is_dir():
            tally.add("days_missing")
            continue
        df = _assemble_day(
            ddir,
            mode=mode,
            automatic_file=automatic_file,
            match_tol=match_tol,
            match_tol_no_id=match_tol_no_id,
            tally=tally,
        )
        if df is not None:
            t = pd.DatetimeIndex(df["time"])
            keep = np.asarray((t >= lo) & (t < hi), dtype=bool)
            raw.append(df.loc[keep])
    frames = [f for f in raw if len(f)]
    if frames:
        whole, sel = _select(
            pd.concat(frames, ignore_index=True),
            min_quality=min_quality,
            duplicate_tol=duplicate_tol,
        )
        for k, v in sel.items():
            tally.add(k, v)
        is_aut = (whole["evaluation_mode"] == "automatic").to_numpy()
        tally.add("window_reviewed", int((~is_aut).sum()))
        tally.add("window_automatic", int(is_aut.sum()))
        frames = [whole]
    report = tally.report(_time.perf_counter() - t0)
    out = _finish(
        frames,
        location=str(root),
        start=lo.to_pydatetime(),
        end=hi.to_pydatetime(),
        filters=_filters(
            mode,
            automatic_file,
            min_quality,
            duplicate_tol,
            match_tol,
            match_tol_no_id,
        ),
        report=report,
    )
    return out, report


#: :func:`read_sil_automatic` sources
AutomaticFile = Literal["events.ref", "aut.mag", "cleanaut.mag"]
#: :func:`read_sil_automatic` ``solutions``
Solutions = Literal["best", "all"]


def _automatic_raw(
    frame: pd.DataFrame, file: AutomaticFile, solutions: Solutions, tally: _Tally
) -> pd.DataFrame:
    """Raw automatic rows of one file's frame, before coercion."""
    if file == "events.ref":
        key = frame["ref_id"].astype(object).where(frame["ref_id"].notna(), frame["id"])
        if solutions == "best":
            # the highest-Q solution of each event (ties: the last written)
            q = frame["q"].to_numpy(dtype="float64", na_value=-np.inf)
            order = np.lexsort((frame["line_no"].to_numpy(), q))
            pick = (
                pd.Series(order, index=key.iloc[order].to_numpy())
                .groupby(level=0, sort=False)
                .last()
                .to_numpy()
            )
            tally.add("solutions_collapsed", len(frame) - len(pick))
            # a best solution without a magnitude takes the highest-Q one of
            # the same event that has one (2026-06-13 20:14 M4.8: Q 90 had none)
            has = frame["ml"].notna().to_numpy()
            mag_by_key = (
                pd.Series(
                    frame["ml"].to_numpy(dtype="float64", na_value=np.nan)[order],
                    index=key.iloc[order].to_numpy(),
                )[has[order]]
                .groupby(level=0, sort=False)
                .last()
            )
            frame = frame.iloc[np.sort(pick)].copy()
            key = key.iloc[np.sort(pick)]
            fill = frame["ml"].isna().to_numpy() & key.isin(mag_by_key.index).to_numpy()
            if fill.any():
                frame.loc[fill, "ml"] = mag_by_key.reindex(key[fill]).to_numpy()
                tally.add("magnitude_from_other_solution", int(fill.sum()))
        mag_type = "events.ref"
        event_id = key.to_numpy(dtype=object)
    else:
        mag_type = "aut.mag"
        ids = frame["id"].astype(object)
        event_id = ids.where(ids.notna(), None).to_numpy(dtype=object)
    n = len(frame)
    ml = frame["ml"].to_numpy(dtype="float64", na_value=np.nan)
    ids = frame["id"].astype(object).where(frame["id"].notna(), None).to_numpy()
    return pd.DataFrame(
        {
            "time": frame["time"].to_numpy(),
            "event_id": event_id,
            "sil_path": np.full(n, None, dtype=object),
            "sil_aut_path": ids,
            "latitude": frame["latitude"].to_numpy(dtype="float64"),
            "longitude": frame["longitude"].to_numpy(dtype="float64"),
            "depth_km": frame["depth"].to_numpy(dtype="float64", na_value=np.nan),
            "h_err_km": np.full(n, np.nan),
            "depth_err_km": np.full(n, np.nan),
            "magnitude": ml,
            "magnitude_type": [None if np.isnan(v) else mag_type for v in ml],
            "mag_mlw": np.full(n, np.nan),
            "mag_ml": np.full(n, np.nan),
            "mag_auto": ml,
            "moment_nm": np.full(n, np.nan),
            "evaluation_mode": "automatic",
            "event_type": "earthquake",
            "event_type_code": np.full(n, None, dtype=object),
            "quality": frame["q"].to_numpy(dtype="float64", na_value=np.nan),
            "n_phases": np.full(n, None, dtype=object),
            "azimuthal_gap": np.full(n, np.nan),
        }
    )


def read_sil_automatic(
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
    *,
    base: str | os.PathLike[str] | None = None,
    file: AutomaticFile = "events.ref",
    solutions: Solutions = "best",
) -> tuple[pd.DataFrame, SilReadReport]:
    """Every automatic SIL detection of ``[start, end)``, reviewed or not.

    :func:`read_sil_catalog` keeps an automatic row only when no reviewed
    event matches it, and reads ``cleanaut.mag``: right for a catalogue, wrong
    for judging the automatic system itself (a reviewed event's automatic
    location disappears, and cleanaut.mag rows replaced by alert2sil carry
    Q 50.50 and a PGV magnitude). This reads one automatic file as written:

    * ``file="events.ref"`` (default): every solution SIL made, every Q —
      ``aut.mag`` is only its Q ≥ 10 part. ``solutions="best"`` keeps the
      highest-Q solution of each event (``event_id`` = its ``ref_id``; a
      best solution without a magnitude takes the magnitude of the highest-Q
      solution that has one, ``magnitude_from_other_solution``);
      ``"all"`` keeps every solution (``event_id`` repeats).
    * ``file="aut.mag"`` / ``"cleanaut.mag"``: those files' rows
      (``event_id`` = their id path).

    Every row is ``evaluation_mode="automatic"``; ``magnitude_type`` names
    the file's scale (``events.ref`` or ``aut.mag``). One extra day directory
    is read on each side (a day file holds events of the day before) and an
    event seen in two day files is kept once (``duplicates:<file>``).
    Values outside the schema ranges become NA (early-1995 negative Q).
    """
    t0 = _time.perf_counter()
    root = sil_dir(base)
    lo, hi = _utc(start), _utc(end)
    if hi < lo:
        raise ValueError(f"end {hi} is before start {lo}")
    if file not in ("events.ref", "aut.mag", "cleanaut.mag"):
        raise ValueError(f"file {file!r}: events.ref, aut.mag or cleanaut.mag")
    if solutions not in ("best", "all"):
        raise ValueError(f"solutions {solutions!r}: best or all")
    first = (lo - pd.Timedelta(days=1)).date()
    last = (max(hi - pd.Timedelta(1, "ns"), lo) + pd.Timedelta(days=1)).date()
    tally = _Tally()
    frames: list[pd.DataFrame] = []
    for day in _days(first, last):
        ddir = day_dir(root, day)
        if not ddir.is_dir():
            tally.add("days_missing")
            continue
        tally.add("days_read")
        frame = _read(ddir, file, tally)
        if frame is not None and len(frame):
            frames.append(frame.assign(_day=str(day)))
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        allf["line_no"] = np.arange(len(allf))  # global order for tie-breaks
        if file == "events.ref" and solutions == "all":
            allf = _dedupe(allf, ["ref_id", "id", "time", "q"], file, tally)
        elif file != "events.ref":
            allf = _dedupe(allf, ["id", "time"], file, tally)
        raw = _automatic_raw(allf.drop(columns="_day"), file, solutions, tally)
        t = pd.DatetimeIndex(raw["time"])
        raw = raw.loc[np.asarray((t >= lo) & (t < hi), dtype=bool)]
        raw = _range_guard(raw, tally)
        tally.add("window_automatic", len(raw))
        parts = [raw]
    else:
        parts = []
    report = tally.report(_time.perf_counter() - t0)
    out = _finish(
        parts,
        location=str(root),
        start=lo.to_pydatetime(),
        end=hi.to_pydatetime(),
        filters={"automatic_file": file, "solutions": solutions, "reviewed": False},
        report=report,
    )
    return out, report


__all__ = [
    "ALERT2SIL_PREFIX",
    "ALERT2SIL_TOL",
    "AutomaticFile",
    "CONFIG_KEY",
    "CatalogAutomaticFile",
    "DEFAULT_DUPLICATE_TOL",
    "DEFAULT_MATCH_TOL",
    "DEFAULT_MATCH_TOL_NO_ID",
    "DEFAULT_MIN_QUALITY",
    "MODES",
    "Mode",
    "ENV_SIL_DIR",
    "KM_PER_DEG",
    "MONTHS",
    "SOURCE",
    "MatchTolerance",
    "SilReadReport",
    "Solutions",
    "day_dir",
    "duplicate_flags",
    "horizontal_error_km",
    "read_sil_automatic",
    "read_sil_catalog",
    "read_sil_day",
    "sil_dir",
]
