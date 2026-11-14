"""PGV magnitudes from the alertMap early-warning files (ticket T7b).

PLAN-seismicity.md §7 (T7b) and §9c (B3, B4); the discovery is
``docs/seismic/sil_alert_discovery.md``. This module has no I/O beyond
reading a local file and no network: the sources (HTTP, mounted directory),
the association over a day's detections and the ``sil_read`` shim are T7c.

**The relation.** IMO report VÍ 2009-012 (Pétursson & Vogfjörð, SAFER D5.2),
equation (C), p. 25, standard error 0.223 (log10)::

    log10(PGV) = -1.69 · log10(r + 0.00299 · 10^(0.621 · M)) + 1.05 · M - 4.96

with PGV in **m/s** (peak of the 3-component vector sum), ``r`` the
**epicentral** distance in km and ``M`` the report's Mw. The alertMap files
give PGV in **µm/s**, so the inverse functions take µm/s.
:func:`pgv_model_c` is the forward relation, :func:`pgv_magnitude` its exact
inverse (bracketed bisection; unique because PGV grows with M for r > 0),
:func:`pgv_magnitude_farfield` the legacy ``M_PGV`` (near-field term dropped)
and :func:`nearfield_bias` the magnitude error that drop makes.
:func:`pgverror` keeps the legacy quantity (the relative change of the
distance term, **not** a magnitude error).

**Parsers** (pure text in, explicit dtypes, UTC times, no ``locale``):
:func:`parse_alert_event` (``NNN.event``), :func:`parse_alert_day_index`
(day ``index.html``), :func:`parse_alert_mag` (alertMap day ``alert.mag``,
not the SIL alarm file of the same name) and :func:`parse_evb` (SIL
``.evb``). Each has a ``read_*`` twin that reads a path.

**The method** (B3, the Mpgv page's): geodesic distances from the alert-file
station coordinates (``distance="evb"`` takes the evb ``dist`` instead, the
legacy ``find_alert`` path), every alert station, ``min_km < r <= 230 km``
with the near-field cut from the page's table, station ``ada`` removed and
field 6 code **22** removed (B4: that code marks ``hei`` (Lennartz 1 Hz),
``gha`` and ``ada`` on the golden event; the code is **not** a fixed
station property, see the T7b notes in ``tests/fixtures/alert/README.md``).
Station magnitudes come from the **exact** inverse by default (it reproduces
the page's median and mean; the far field does not). The event value is the
**median** of the station magnitudes (``magnitude_type="Mpgv_alert"``),
reported with the mid-third mean. :func:`travel_time_fit` is the legacy
``check_event_fit`` rule on full datetimes, pure, for T7c to wire.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from geo_dataread.seismic.schema import (
    MAGNITUDE_TYPES,
    STRING_DTYPE,
    TIME_DTYPE,
    FloatArray,
)

__all__ = [
    "ALERT_EVENT_COLUMNS",
    "ALERT_MAG_COLUMNS",
    "DAY_INDEX_COLUMNS",
    "DEFAULT_FIT",
    "DEFAULT_PGV",
    "EVB_PHASE_COLUMNS",
    "MAGNITUDE_TYPE",
    "MODEL_C",
    "PGV_SOURCE",
    "PGV_STATION_COLUMNS",
    "AlertEvent",
    "AlertFormatError",
    "AlertHeader",
    "DistanceSource",
    "EventFit",
    "EvbFile",
    "FitSettings",
    "Inverse",
    "ModelC",
    "Origin",
    "PgvResult",
    "PgvSettings",
    "SpecialRegion",
    "estimate_pgv_magnitude",
    "mid_third_mean",
    "nearfield_bias",
    "nearfield_min_km",
    "parse_alert_day_index",
    "parse_alert_event",
    "parse_alert_mag",
    "parse_evb",
    "pgv_magnitude",
    "pgv_magnitude_farfield",
    "pgv_model_c",
    "pgv_stations",
    "pgverror",
    "read_alert_day_index",
    "read_alert_event",
    "read_alert_mag",
    "read_evb",
    "station_distances_km",
    "travel_time_fit",
]

#: The schema ``magnitude_type`` of the event value (PLAN B5).
MAGNITUDE_TYPE: Final = "Mpgv_alert"
if MAGNITUDE_TYPE not in MAGNITUDE_TYPES:  # pragma: no cover - schema v3 has it
    raise ImportError(
        f"schema MAGNITUDE_TYPES lacks {MAGNITUDE_TYPE!r} (needs schema v3)"
    )

#: ``source`` value of the ``pgv_stations`` rows.
PGV_SOURCE: Final = "alertmap"

#: How a station distance is obtained (B3).
DistanceSource: TypeAlias = Literal["geodesic", "evb"]
#: Which inverse of model C gives the station magnitude.
Inverse: TypeAlias = Literal["farfield", "exact"]

#: English month abbreviations, fixed (``strptime("%b")`` is locale dependent).
_MONTHS: Final = (
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


class AlertFormatError(ValueError):
    """A file does not have the expected alertMap or evb layout."""


# --------------------------------------------------------------------------
# VÍ 2009-012 equation (C)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelC:
    """Coefficients of VÍ 2009-012 eq. (C), p. 25 (PGV in m/s, r in km).

    ``log10(PGV) = a · log10(r + c · 10^(g · M)) + b · M + d``. ``g`` is
    constrained to ``-b/a``, so PGV is (almost) independent of M at r = 0.
    """

    a: float = -1.69
    c: float = 0.00299
    g: float = 0.621
    b: float = 1.05
    d: float = -4.96
    sigma_log10: float = 0.223
    reference: str = (
        "Pétursson & Vogfjörð, VÍ 2009-012 (SAFER D5.2), eq. (C), p. 25; "
        "https://www.vedur.is/media/vedurstofan/utgafa/skyrslur/2009/VI_2009_012.pdf"
    )


#: The published coefficients.
MODEL_C: Final = ModelC()

#: Bracket of the exact inverse (magnitudes outside give NaN).
_M_BRACKET: Final = (-3.0, 10.0)
_BISECT_ITER: Final = 64


def _arr(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def pgv_model_c(
    m: npt.ArrayLike, r_km: npt.ArrayLike, *, model: ModelC = MODEL_C
) -> FloatArray:
    """Forward model C: PGV in **m/s** for magnitude ``m`` at ``r_km``.

    Broadcasts. ``r_km < 0`` gives NaN; ``r_km = 0`` is allowed (the
    near-field term keeps it finite).
    """
    mm, rr = np.broadcast_arrays(_arr(m), _arr(r_km))
    with np.errstate(invalid="ignore", divide="ignore"):
        log_pgv = (
            model.a * np.log10(rr + model.c * 10.0 ** (model.g * mm))
            + model.b * mm
            + model.d
        )
        out = 10.0**log_pgv
    return np.where(rr >= 0.0, out, np.nan)


def pgv_magnitude_farfield(
    pgv_um_s: npt.ArrayLike, r_km: npt.ArrayLike, *, model: ModelC = MODEL_C
) -> FloatArray:
    """Far-field inverse of model C (the legacy ``M_PGV``): r + near term ≈ r.

    ``M = (-a · log10(r) + log10(pgv · 1e-6) - d) / b`` with ``pgv`` in
    **µm/s** and ``r`` in km. ``r <= 0`` or ``pgv <= 0`` (or NaN) gives NaN.
    """
    pp, rr = np.broadcast_arrays(_arr(pgv_um_s), _arr(r_km))
    ok = (pp > 0.0) & (rr > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (-model.a * np.log10(rr) + np.log10(pp * 1e-6) - model.d) / model.b
    return np.where(ok, m, np.nan)


def pgv_magnitude(
    pgv_um_s: npt.ArrayLike, r_km: npt.ArrayLike, *, model: ModelC = MODEL_C
) -> FloatArray:
    """Exact inverse of model C: the M whose predicted PGV at ``r_km`` is ``pgv``.

    ``pgv`` in **µm/s**, ``r`` in km. Vectorised bisection on
    ``[-3, 10]`` (64 halvings, i.e. to rounding). PGV(M) is strictly
    increasing for r > 0, so the root is unique. NaN where ``r <= 0``,
    ``pgv <= 0`` or the root lies outside the bracket (a PGV the relation
    cannot reach at that distance).
    """
    pp, rr = np.broadcast_arrays(_arr(pgv_um_s), _arr(r_km))
    ok = (pp > 0.0) & (rr > 0.0) & np.isfinite(pp) & np.isfinite(rr)
    target = np.where(ok, np.log10(np.where(ok, pp, 1.0) * 1e-6), 0.0)
    r = np.where(ok, rr, 1.0)

    def f(m: FloatArray) -> FloatArray:
        out: FloatArray = (
            model.a * np.log10(r + model.c * 10.0 ** (model.g * m))
            + model.b * m
            + model.d
            - target
        )
        return out

    lo = np.full(pp.shape, _M_BRACKET[0])
    hi = np.full(pp.shape, _M_BRACKET[1])
    inside = (f(lo) <= 0.0) & (f(hi) >= 0.0)
    for _ in range(_BISECT_ITER):
        mid = 0.5 * (lo + hi)
        up = f(mid) < 0.0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    m = 0.5 * (lo + hi)
    return np.where(ok & inside, m, np.nan)


def nearfield_bias(
    r_km: npt.ArrayLike, m: npt.ArrayLike, *, model: ModelC = MODEL_C
) -> FloatArray:
    """How much the far-field inverse underestimates M (a positive number).

    ``ΔM = (-a/b) · (log10(r + c·10^(gM)) - log10(r)) = M - M_farfield`` for
    an event of magnitude ``m`` seen at ``r_km`` (0.2255 at 10 km, M5).
    NaN for r <= 0.
    """
    mm, rr = np.broadcast_arrays(_arr(m), _arr(r_km))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (-model.a / model.b) * (
            np.log10(rr + model.c * 10.0 ** (model.g * mm)) - np.log10(rr)
        )
    return np.where(rr > 0.0, d, np.nan)


def pgverror(
    r_km: npt.ArrayLike, m: npt.ArrayLike, *, model: ModelC = MODEL_C
) -> FloatArray:
    """The legacy ``pgverror``: relative change of the distance term.

    ``(log10(r + c·10^(gM)) - log10 r) / log10 r``. Kept for compatibility
    only: it is infinite at r = 1 km and negative below; use
    :func:`nearfield_bias` for the magnitude error.
    """
    mm, rr = np.broadcast_arrays(_arr(m), _arr(r_km))
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log10(rr)
        out: FloatArray = (np.log10(rr + model.c * 10.0 ** (model.g * mm)) - lr) / lr
    return out


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------


def _read_text(path: str | os.PathLike[str]) -> str:
    # The alertMap pages declare ISO-8859-1; the data files are ASCII.
    return Path(path).read_bytes().decode("latin-1")


def _month(token: str, *, what: str) -> int:
    try:
        return _MONTHS.index(token.strip().lower()[:3]) + 1
    except ValueError:
        raise AlertFormatError(f"{what}: unknown month {token!r}") from None


def _utc(
    year: int, month: int, day: int, hour: int, minute: int, second: float, *, what: str
) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(
            year=year, month=month, day=day, hour=hour, minute=minute, tz="UTC"
        )
    except (TypeError, ValueError) as exc:
        raise AlertFormatError(f"{what}: bad date/time ({exc})") from exc
    return ts + pd.Timedelta(seconds=second)


@dataclasses.dataclass(frozen=True)
class AlertHeader:
    """The first line of an alertMap ``NNN.event``.

    ``076 8023 11330 11330 Thu Nov 12 21:53:45 2026 3307 66``: ``alert_id``
    (the file number), three integers of unknown meaning (``counters``), the
    detection time (UTC, whole seconds), an integer the day index also shows
    (``index_value``; unknown) and the number of station rows.
    """

    alert_id: int
    counters: tuple[int, int, int]
    time: pd.Timestamp
    index_value: int
    n_stations: int
    raw: str


#: Columns of the ``NNN.event`` station frame, in order.
ALERT_EVENT_COLUMNS: Final = (
    "station",
    "lon",
    "lat",
    "offset_s",
    "pgv_um_s",
    "fb",
    "class_code",
    "aux7",
    "second_of_minute",
    "aux9",
    "aux10",
    "aux11",
    "aux12",
    "arrival_time",
)


class AlertEvent(NamedTuple):
    """A parsed ``NNN.event``: header plus one row per station.

    Frame columns (:data:`ALERT_EVENT_COLUMNS`): ``station`` (SIL code),
    ``lon``/``lat`` (degrees), ``offset_s`` (PGV time after the header time,
    s; the file has centiseconds), ``pgv_um_s`` (µm/s), ``fb`` (unknown,
    perhaps a frequency), ``class_code`` (field 6; 22 = excluded, B4),
    ``aux7``/``aux9`` (unknown integers), ``second_of_minute`` (field 8),
    ``aux10``–``aux12`` (unknown floats, ``--.--`` → NaN) and
    ``arrival_time`` (UTC, header time + offset).
    """

    header: AlertHeader
    frame: pd.DataFrame


def _parse_alert_header(line: str, *, what: str) -> AlertHeader:
    tok = line.split()
    if len(tok) != 11:
        raise AlertFormatError(
            f"{what}: header has {len(tok)} fields, expected 11: {line!r}"
        )
    try:
        ints = [int(t) for t in (*tok[0:4], tok[9], tok[10])]
        hh, mi, ss = (int(x) for x in tok[7].split(":"))
        day, year = int(tok[6]), int(tok[8])
    except ValueError as exc:
        raise AlertFormatError(f"{what}: bad header {line!r}") from exc
    month = _month(tok[5], what=what)
    return AlertHeader(
        alert_id=ints[0],
        counters=(ints[1], ints[2], ints[3]),
        time=_utc(year, month, day, hh, mi, float(ss), what=what),
        index_value=ints[4],
        n_stations=ints[5],
        raw=line.rstrip("\n"),
    )


def _float_or_nan(tok: str) -> float:
    return math.nan if set(tok) <= {"-", "."} else float(tok)


def _empty_alert_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": pd.Series([], dtype=STRING_DTYPE),
            **{
                c: pd.Series([], dtype="float64")
                for c in ("lon", "lat", "offset_s", "pgv_um_s", "fb")
            },
            "class_code": pd.Series([], dtype="Int64"),
            "aux7": pd.Series([], dtype="Int64"),
            "second_of_minute": pd.Series([], dtype="float64"),
            "aux9": pd.Series([], dtype="Int64"),
            **{c: pd.Series([], dtype="float64") for c in ("aux10", "aux11", "aux12")},
            "arrival_time": pd.Series([], dtype=TIME_DTYPE),
        }
    )


def parse_alert_event(
    text: str, *, source: str = "<text>", strict: bool = True
) -> AlertEvent:
    """Parse the text of an alertMap ``NNN.event`` (see :class:`AlertEvent`).

    Every row must have 13 whitespace-separated fields; fields 0–5 must be
    numbers (station code at 4), fields 10–12 may be ``--.--`` (NaN). A
    malformed row raises :class:`AlertFormatError` naming the line. A
    header-only file gives an empty frame. With ``strict`` (default) the
    row count must equal the header's station count and the stations must
    be unique.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise AlertFormatError(f"{source}: empty file")
    header = _parse_alert_header(lines[0], what=source)
    rows: list[dict[str, Any]] = []
    for no, line in enumerate(lines[1:], start=2):
        tok = line.split()
        if len(tok) != 13:
            raise AlertFormatError(
                f"{source}:{no}: {len(tok)} fields, expected 13: {line!r}"
            )
        try:
            rows.append(
                {
                    "station": tok[4],
                    "lon": float(tok[0]),
                    "lat": float(tok[1]),
                    "offset_s": int(tok[2]) / 100.0,
                    "pgv_um_s": float(tok[3]),
                    "fb": float(tok[5]),
                    "class_code": int(tok[6]),
                    "aux7": int(tok[7]),
                    "second_of_minute": float(tok[8]),
                    "aux9": int(tok[9]),
                    "aux10": _float_or_nan(tok[10]),
                    "aux11": _float_or_nan(tok[11]),
                    "aux12": _float_or_nan(tok[12]),
                }
            )
        except ValueError as exc:
            raise AlertFormatError(
                f"{source}:{no}: bad field ({exc}): {line!r}"
            ) from exc
    if strict and len(rows) != header.n_stations:
        raise AlertFormatError(
            f"{source}: {len(rows)} station rows, header says {header.n_stations}"
        )
    frame = _empty_alert_frame()
    if rows:
        raw = pd.DataFrame(rows)
        frame = pd.DataFrame(
            {
                c: raw[c].astype(frame[c].dtype)
                for c in ALERT_EVENT_COLUMNS
                if c != "arrival_time"
            }
        )
        frame["arrival_time"] = (
            header.time + pd.to_timedelta(raw["offset_s"], unit="s")
        ).astype(TIME_DTYPE)
        dup = frame["station"].duplicated()
        if strict and dup.any():
            names = sorted(set(frame.loc[dup, "station"]))
            raise AlertFormatError(f"{source}: duplicated station rows {names}")
    return AlertEvent(header, frame[list(ALERT_EVENT_COLUMNS)])


def read_alert_event(
    path: str | os.PathLike[str], *, strict: bool = True
) -> AlertEvent:
    """Read one alertMap ``NNN.event`` file (:func:`parse_alert_event`)."""
    return parse_alert_event(_read_text(path), source=str(path), strict=strict)


#: Columns of the day ``index.html`` frame.
DAY_INDEX_COLUMNS: Final = ("alert_id", "time", "n_stations", "index_value")

_INDEX_ROW: Final = re.compile(
    r'<a\s+href="(\d+)\.html">\s*(\d{1,2}):(\d{2}):(\d{2})\s*</a>\s*\]?\s*'
    r"</td>\s*<td>\s*(-?\d+)\s*</td>\s*<td>\s*(-?\d+)\s*</td>",
    re.IGNORECASE,
)
_INDEX_TITLE: Final = re.compile(
    r"<title>\s*Alert Map\s+(\d{4})\s+(\w+)\s+(\d{1,2})\s*</title>", re.I
)


def parse_alert_day_index(
    text: str, day: date | None = None, *, source: str = "<text>"
) -> pd.DataFrame:
    """Parse an alertMap day ``index.html`` into one row per detection.

    Columns :data:`DAY_INDEX_COLUMNS`: ``alert_id``, ``time`` (UTC: the day
    plus the listed ``HH:MM:SS``), ``n_stations`` (the first number; equals
    the ``.event`` header's last field) and ``index_value`` (the second; the
    header's field 9). Sorted by ``alert_id``. ``day`` defaults to the
    page title's date (``Alert Map 2026 Nov 12``); a page with neither
    raises :class:`AlertFormatError`, as does a given ``day`` that
    contradicts the title. A page without rows gives an empty frame.
    """
    title = _INDEX_TITLE.search(text)
    title_day: date | None = None
    if title:
        title_day = date(
            int(title.group(1)),
            _month(title.group(2), what=source),
            int(title.group(3)),
        )
    if day is None:
        if title_day is None:
            raise AlertFormatError(f"{source}: no day given and no 'Alert Map' title")
        day = title_day
    elif title_day is not None and title_day != day:
        raise AlertFormatError(f"{source}: title says {title_day}, caller says {day}")
    recs = [
        (
            int(m.group(1)),
            _utc(
                day.year,
                day.month,
                day.day,
                int(m.group(2)),
                int(m.group(3)),
                float(m.group(4)),
                what=source,
            ),
            int(m.group(5)),
            int(m.group(6)),
        )
        for m in _INDEX_ROW.finditer(text)
    ]
    recs.sort(key=lambda r: r[0])
    return pd.DataFrame(
        {
            "alert_id": pd.Series([r[0] for r in recs], dtype="int64"),
            "time": pd.Series([r[1] for r in recs], dtype=TIME_DTYPE),
            "n_stations": pd.Series([r[2] for r in recs], dtype="int64"),
            "index_value": pd.Series([r[3] for r in recs], dtype="int64"),
        }
    )


def read_alert_day_index(
    path: str | os.PathLike[str], day: date | None = None
) -> pd.DataFrame:
    """Read a day ``index.html`` file (:func:`parse_alert_day_index`)."""
    return parse_alert_day_index(_read_text(path), day, source=str(path))


#: Columns of the alertMap day ``alert.mag`` frame (``fNN`` = token NN,
#: meaning unknown).
ALERT_MAG_COLUMNS: Final = (
    "alert_id",
    "time",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "class_letter",
    "code",
    "f09",
    "f10",
    "f11",
    "n_stations",
    "f13",
    "f14",
)


def parse_alert_mag(text: str, *, source: str = "<text>") -> pd.DataFrame:
    """Parse an alertMap day ``alert.mag`` (one line per located detection).

    ``076 20261112 215349.1  64.089  -17.052 4.000  4.29 T 000  0.49 ...``:
    15 whitespace fields → :data:`ALERT_MAG_COLUMNS`. ``time`` is UTC
    (*guess*, as for the ``.event`` header); ``depth_km`` is a fixed 4.000;
    the meaning of ``magnitude`` (the alert system's own estimate) and of
    the ``fNN`` fields is not documented. ``code`` stays text (``"011"``).
    Not the SIL alarm ``alert.mag`` (that is the ``cleanaut.mag`` layout).
    """
    recs: list[dict[str, Any]] = []
    for no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        tok = line.split()
        if len(tok) != 15:
            raise AlertFormatError(
                f"{source}:{no}: {len(tok)} fields, expected 15: {line!r}"
            )
        try:
            ymd, hms = tok[1], tok[2]
            if len(ymd) != 8 or "." not in hms or len(hms.split(".")[0]) != 6:
                raise ValueError("date/time layout")
            t = _utc(
                int(ymd[:4]),
                int(ymd[4:6]),
                int(ymd[6:]),
                int(hms[:2]),
                int(hms[2:4]),
                float(hms[4:]),
                what=f"{source}:{no}",
            )
            recs.append(
                {
                    "alert_id": int(tok[0]),
                    "time": t,
                    "latitude": float(tok[3]),
                    "longitude": float(tok[4]),
                    "depth_km": float(tok[5]),
                    "magnitude": float(tok[6]),
                    "class_letter": tok[7],
                    "code": tok[8],
                    "f09": float(tok[9]),
                    "f10": float(tok[10]),
                    "f11": float(tok[11]),
                    "n_stations": int(tok[12]),
                    "f13": float(tok[13]),
                    "f14": int(tok[14]),
                }
            )
        except ValueError as exc:
            raise AlertFormatError(
                f"{source}:{no}: bad field ({exc}): {line!r}"
            ) from exc
    dtypes: Mapping[str, Any] = {
        "alert_id": "int64",
        "time": TIME_DTYPE,
        "latitude": "float64",
        "longitude": "float64",
        "depth_km": "float64",
        "magnitude": "float64",
        "class_letter": STRING_DTYPE,
        "code": STRING_DTYPE,
        "f09": "float64",
        "f10": "float64",
        "f11": "float64",
        "n_stations": "int64",
        "f13": "float64",
        "f14": "int64",
    }
    return pd.DataFrame(
        {c: pd.Series([r[c] for r in recs], dtype=dtypes[c]) for c in ALERT_MAG_COLUMNS}
    )


def read_alert_mag(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read an alertMap day ``alert.mag`` file (:func:`parse_alert_mag`)."""
    return parse_alert_mag(_read_text(path), source=str(path))


# --------------------------------------------------------------------------
# SIL .evb
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Origin:
    """An earthquake origin: UTC time, epicentre, optional depth."""

    time: pd.Timestamp
    latitude: float
    longitude: float
    depth_km: float | None = None

    def __post_init__(self) -> None:
        ts = pd.Timestamp(self.time)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        object.__setattr__(self, "time", ts)
        if not (math.isfinite(self.latitude) and math.isfinite(self.longitude)):
            raise ValueError(
                f"Origin: bad epicentre ({self.latitude}, {self.longitude})"
            )
        if abs(self.latitude) > 90.0:
            raise ValueError(f"Origin: latitude {self.latitude} out of range")


#: Columns of the evb phase frame.
EVB_PHASE_COLUMNS: Final = (
    "station",
    "phase",
    "arrival_time",
    "residual_s",
    "weight",
    "distance_km",
    "azimuth_deg",
    "azimuth_obs_deg",
    "azimuth_res_deg",
    "used",
)

#: (column, start, end) of an evb phase row (the legacy colspecs).
_EVB_SLICES: Final = (
    ("station", 0, 4),
    ("phase", 4, 6),
    ("time", 6, 18),
    ("residual_s", 18, 24),
    ("weight", 24, 30),
    ("distance_km", 30, 36),
    ("azimuth_deg", 36, 42),
    ("azimuth_obs_deg", 42, 49),
    ("azimuth_res_deg", 49, 55),
    ("used", 55, 62),
)

_EVB_ORIGIN: Final = re.compile(
    r"origin time\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})h\s*(\d{1,2})m\s*([\d.]+)s"
    r"(?:\s*\+/-\s*([\d.]+)s)?"
)
_EVB_VALUE: Final = {
    "latitude": re.compile(r"latitude\s*(-?[\d.]+)\s*\+/-\s*([\d.]+)"),
    "longitude": re.compile(r"longitude\s*(-?[\d.]+)\s*\+/-\s*([\d.]+)"),
    "depth": re.compile(r"focal depth\s*(-?[\d.]+)\s*\+/-\s*([\d.]+)"),
}
_EVB_MODEL: Final = re.compile(r"^\s*model:\s*(\d+)\s+(-?[\d.]+)")


@dataclasses.dataclass(frozen=True)
class EvbFile:
    """A parsed SIL ``.evb``: the origin, its errors and the phase rows.

    ``phases`` (:data:`EVB_PHASE_COLUMNS`) keeps p and s picks as separate
    rows (never a duplicated index): ``station``, ``phase`` (``p``/``s``),
    ``arrival_time`` (UTC, full datetime; rolls into the next day when the
    time of day is far before the origin's), ``residual_s``, ``weight``,
    ``distance_km`` (epicentral), ``azimuth_deg``, ``azimuth_obs_deg``
    (0 = none), ``azimuth_res_deg`` and ``used`` (1.0/0.0, *guess*).
    ``quality`` is the footer number (the event's Q in ``cleanaut.mag``),
    ``model`` the velocity-model number.
    """

    origin: Origin
    time_error_s: float | None
    latitude_error_deg: float | None
    longitude_error_deg: float | None
    depth_error_km: float | None
    model: int | None
    quality: float | None
    phases: pd.DataFrame

    def station_distances(self) -> pd.Series:
        """Epicentral distance (km) per station, one value per station."""
        p = self.phases.drop_duplicates("station", keep="first")
        return pd.Series(
            p["distance_km"].to_numpy(dtype=np.float64),
            index=pd.Index(p["station"].astype(str), name="station"),
            name="distance_km",
        )


def parse_evb(text: str, *, source: str = "<text>", year: int | None = None) -> EvbFile:
    """Parse the text of a SIL ``.evb`` (see :class:`EvbFile`).

    The origin line has a 2-digit year: ``year`` overrides it (pass the
    archive path's year); otherwise ``YY >= 70`` is 19YY, else 20YY. Rows
    are cut at the legacy fixed columns; a row whose numbers do not parse
    raises :class:`AlertFormatError` with its line number.
    """
    lines = text.splitlines()
    if len(lines) < 5:
        raise AlertFormatError(f"{source}: {len(lines)} lines, too short for an evb")
    m = _EVB_ORIGIN.search(lines[0])
    if not m:
        raise AlertFormatError(f"{source}:1: no origin time: {lines[0]!r}")
    yy, mo, dd, hh, mi = (int(m.group(i)) for i in range(1, 6))
    yr = year if year is not None else (1900 + yy if yy >= 70 else 2000 + yy)
    if year is not None and year % 100 != yy:
        raise AlertFormatError(f"{source}:1: year {year} does not match '{yy:02d}'")
    t0 = _utc(yr, mo, dd, hh, mi, float(m.group(6)), what=source)
    terr = float(m.group(7)) if m.group(7) else None
    vals: dict[str, tuple[float, float]] = {}
    for (key, rx), line_no in zip(_EVB_VALUE.items(), (2, 3, 4), strict=True):
        mv = rx.search(lines[line_no - 1])
        if not mv:
            raise AlertFormatError(
                f"{source}:{line_no}: no {key}: {lines[line_no - 1]!r}"
            )
        vals[key] = (float(mv.group(1)), float(mv.group(2)))
    if not lines[4].lstrip("0 ").startswith("sta"):
        raise AlertFormatError(f"{source}:5: not the phase header: {lines[4]!r}")
    model: int | None = None
    quality: float | None = None
    recs: list[dict[str, Any]] = []
    for no, line in enumerate(lines[5:], start=6):
        if not line.strip():
            continue
        mm = _EVB_MODEL.match(line)
        if mm:
            model, quality = int(mm.group(1)), float(mm.group(2))
            break
        cut = {name: line[a:b] for name, a, b in _EVB_SLICES}
        try:
            h, mn, s = cut["time"].split()
            tod = pd.Timedelta(hours=int(h), minutes=int(mn), seconds=float(s))
            rec: dict[str, Any] = {
                "station": cut["station"].strip(),
                "phase": cut["phase"].strip(),
                "tod": tod,
            }
            for name in EVB_PHASE_COLUMNS[3:]:
                rec[name] = float(cut[name])
        except ValueError as exc:
            raise AlertFormatError(
                f"{source}:{no}: bad phase row ({exc}): {line!r}"
            ) from exc
        if not rec["station"] or rec["phase"] not in ("p", "s"):
            raise AlertFormatError(f"{source}:{no}: bad station/phase: {line!r}")
        recs.append(rec)
    day0 = t0.normalize()
    t0_tod = t0 - day0
    arrivals = []
    for r in recs:
        t = day0 + r["tod"]
        # A pick 12 h or more before the origin's time of day is after midnight.
        if r["tod"] < t0_tod - pd.Timedelta(hours=12):
            t += pd.Timedelta(days=1)
        arrivals.append(t)
    phases = pd.DataFrame(
        {
            "station": pd.Series([r["station"] for r in recs], dtype=STRING_DTYPE),
            "phase": pd.Series([r["phase"] for r in recs], dtype=STRING_DTYPE),
            "arrival_time": pd.Series(arrivals, dtype=TIME_DTYPE),
            **{
                c: pd.Series([r[c] for r in recs], dtype="float64")
                for c in EVB_PHASE_COLUMNS[3:]
            },
        }
    )
    return EvbFile(
        origin=Origin(t0, vals["latitude"][0], vals["longitude"][0], vals["depth"][0]),
        time_error_s=terr,
        latitude_error_deg=vals["latitude"][1],
        longitude_error_deg=vals["longitude"][1],
        depth_error_km=vals["depth"][1],
        model=model,
        quality=quality,
        phases=phases,
    )


def read_evb(path: str | os.PathLike[str], *, year: int | None = None) -> EvbFile:
    """Read one SIL ``.evb`` file (:func:`parse_evb`).

    Without ``year``, the year is taken from a ``/YYYY/mon/DD/`` part of the
    path when there is one, else from the 2-digit year (pivot 70).
    """
    if year is None:
        mpath = re.search(r"(?:^|/)(\d{4})/[a-z]{3}/\d{2}/", Path(path).as_posix())
        year = int(mpath.group(1)) if mpath else None
    return parse_evb(_read_text(path), source=str(path), year=year)


# --------------------------------------------------------------------------
# Distances and the travel-time fit (legacy check_event_fit)
# --------------------------------------------------------------------------


def station_distances_km(
    lat: float,
    lon: float,
    sta_lat: npt.ArrayLike,
    sta_lon: npt.ArrayLike,
    *,
    ellps: str = "GRS80",
) -> FloatArray:
    """Geodesic distance (km) from ``(lat, lon)`` to each station.

    ``geofunc.frames.geodesic`` (pyproj Karney), as
    :func:`geo_dataread.seismic.filters.distance_km`; NaN where a station
    coordinate is NaN.
    """
    from geofunc.frames import geodesic

    la, lo = np.broadcast_arrays(_arr(sta_lat), _arr(sta_lon))
    out = np.full(la.shape, np.nan)
    ok = np.isfinite(la) & np.isfinite(lo)
    if ok.any():
        n = int(ok.sum())
        _, _, dist_m = geodesic(ellps).inv(
            np.full(n, float(lon)), np.full(n, float(lat)), lo[ok], la[ok]
        )
        out[ok] = np.asarray(dist_m, dtype=np.float64) / 1000.0
    return out


@dataclasses.dataclass(frozen=True)
class SpecialRegion:
    """An epicentre box with its own late-arrival allowance (``post_s``)."""

    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    post_s: float

    def contains(self, lat: float, lon: float) -> bool:
        """Inclusive box test on the epicentre."""
        return (
            self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max
        )


@dataclasses.dataclass(frozen=True)
class FitSettings:
    """The legacy ``check_event_fit`` rule (defaults = the legacy literals).

    A station fits when ``origin + r/vp - pre_s < arrival < origin + r/vs +
    post_s`` (strict), with ``post_s`` from the first special region holding
    the epicentre (Bárðarbunga: 25 s, added 2021–22). The event fits when
    ``n > 0``, ``n_fit >= min_stations`` and ``n_fit / n >= min_fraction``.
    """

    vp_km_s: float = 7.3
    vs_km_s: float = 1.9
    pre_s: float = 2.0
    post_s: float = 12.0
    min_stations: int = 10
    min_fraction: float = 0.68
    special_regions: tuple[SpecialRegion, ...] = (
        SpecialRegion("bardarbunga", 64.55, 64.70, -17.65, -17.10, post_s=25.0),
    )


#: The legacy fit rule.
DEFAULT_FIT: Final = FitSettings()


@dataclasses.dataclass(frozen=True)
class EventFit:
    """Result of :func:`travel_time_fit`.

    ``fits`` is per station (in input order). ``label`` is the Mpgv page's
    ``alertfit`` text, e.g. ``"61/66 (92%)"`` (the legacy string without
    its trailing space); ``""`` when there are no stations.
    """

    accepted: bool
    n_fit: int
    n_total: int
    post_s: float
    region: str | None
    fits: npt.NDArray[np.bool_]

    @property
    def fraction(self) -> float:
        """``n_fit / n_total`` (NaN with no stations)."""
        return self.n_fit / self.n_total if self.n_total else math.nan

    @property
    def label(self) -> str:
        """``"i/n (p%)"`` as on the Mpgv page."""
        if not self.n_total:
            return ""
        return f"{self.n_fit}/{self.n_total} ({self.fraction * 100:.0f}%)"


def travel_time_fit(
    origin: Origin,
    arrival_times: Sequence[Any] | pd.Series | pd.DatetimeIndex,
    distances_km: npt.ArrayLike,
    *,
    settings: FitSettings = DEFAULT_FIT,
) -> EventFit:
    """Do the PGV arrival times fit the origin? (legacy ``check_event_fit``).

    Pure: full UTC datetimes (naive = UTC), so a window across midnight
    works; no I/O, no printing, nothing mutated. ``distances_km`` are the
    epicentral distances (the legacy code used geodesic distances from the
    alert-file coordinates; see :func:`station_distances_km`). A NaN
    distance or arrival never fits but counts in ``n_total``, as every
    alert row did.
    """
    arr = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(arrival_times)), utc=True))
    dist = _arr(distances_km).reshape(-1)
    if len(arr) != dist.size:
        raise ValueError(
            f"travel_time_fit: {len(arr)} arrivals but {dist.size} distances"
        )
    region = next(
        (
            r
            for r in settings.special_regions
            if r.contains(origin.latitude, origin.longitude)
        ),
        None,
    )
    post = region.post_s if region else settings.post_s
    t0 = origin.time
    with np.errstate(invalid="ignore"):
        lo_s = dist / settings.vp_km_s - settings.pre_s
        hi_s = dist / settings.vs_km_s + post
    rel = (
        (arr - t0).total_seconds().to_numpy(dtype=np.float64)
        if len(arr)
        else np.zeros(0)
    )
    with np.errstate(invalid="ignore"):
        fits = np.asarray((rel > lo_s) & (rel < hi_s), dtype=bool)
    n, i = int(dist.size), int(fits.sum())
    accepted = n > 0 and i >= settings.min_stations and i / n >= settings.min_fraction
    return EventFit(
        accepted=accepted,
        n_fit=i,
        n_total=n,
        post_s=float(post),
        region=region.name if region else None,
        fits=fits,
    )


# --------------------------------------------------------------------------
# Filters and the robust estimate
# --------------------------------------------------------------------------

#: The Mpgv page's near-field cut: (upper magnitude bound, cut km): 10 km
#: below Mw 4, 15 km for 4–5, 30 km for 5–6, 50 km above 6.
_NEARFIELD: Final = ((4.0, 10.0), (5.0, 15.0), (6.0, 30.0), (math.inf, 50.0))


@dataclasses.dataclass(frozen=True)
class PgvSettings:
    """Station filters and estimator of the PGV magnitude (B3/B4 defaults).

    * ``max_km``: keep ``r <= max_km`` (230, the page).
    * ``min_km``: keep ``r > min_km``. ``None`` (default) takes the page's
      near-field table ``nearfield_km`` (upper-bound magnitude → km, lower
      bound inclusive), indexed by a first-pass median of the stations that
      pass every other filter (see :func:`nearfield_min_km`).
    * ``exclude_stations``: removed codes (``ada``, the page).
    * ``exclude_class_codes``: removed ``.event`` field-6 codes (22, B4).
    * ``distance``: ``"geodesic"`` from the alert-file station coordinates
      (B3, the page) or ``"evb"`` (the evb ``dist``; stations missing in
      the evb are dropped).
    * ``inverse``: which inverse of model C gives the station magnitude
      used by the estimate (both are always in the frame). ``"exact"``
      (default) reproduces the Mpgv page's median and mean (golden event
      4.15 / 4.17; far field gives 4.15 / 4.16); ``"farfield"`` is the
      legacy ``M_PGV``.
    """

    max_km: float = 230.0
    min_km: float | None = None
    nearfield_km: tuple[tuple[float, float], ...] = _NEARFIELD
    exclude_stations: tuple[str, ...] = ("ada",)
    exclude_class_codes: tuple[int, ...] = (22,)
    distance: DistanceSource = "geodesic"
    inverse: Inverse = "exact"
    ellps: str = "GRS80"


#: The B3/B4 method.
DEFAULT_PGV: Final = PgvSettings()


def nearfield_min_km(
    m: float, table: Iterable[tuple[float, float]] = _NEARFIELD
) -> float:
    """The near-field cut (km) for magnitude ``m`` from ``(upper, km)`` rows.

    The first row with ``m < upper`` wins (so Mw 4.0 is in the 4–5 class);
    NaN ``m`` takes the first row.
    """
    rows = list(table)
    if not rows:
        raise ValueError("nearfield_min_km: empty table")
    if not math.isfinite(m):
        return float(rows[0][1])
    for upper, km in rows:
        if m < upper:
            return float(km)
    return float(rows[-1][1])


def mid_third_mean(values: npt.ArrayLike, *, min_values: int = 3) -> float:
    """Mean of the middle third of the sorted values (NaN ignored).

    Drops ``floor(n/3)`` values at each end and averages the rest. NaN with
    fewer than ``min_values`` values. Both choices reproduce the Mpgv page
    ("MW mean"): n = 8 (2026-11-03 #009: 3.01, where ``round(n/3)`` gives
    3.10) and n = 2 (2026-11-01 #050: blank on the page).
    """
    v = np.sort(_arr(values).reshape(-1))
    v = v[np.isfinite(v)]
    if v.size == 0 or v.size < min_values:
        return math.nan
    k = v.size // 3
    return float(np.mean(v[k : v.size - k]))


#: Columns of the ``pgv_stations`` frame, in order.
PGV_STATION_COLUMNS: Final = (
    "source",
    "event_id",
    "alert_day",
    "alert_id",
    "station",
    "lon",
    "lat",
    "class_code",
    "pgv_um_s",
    "arrival_time",
    "distance_km",
    "distance_source",
    "fits_window",
    "kept",
    "drop_reason",
    "mag_farfield",
    "mag_exact",
)


def pgv_stations(
    event: AlertEvent,
    origin: Origin,
    *,
    evb: EvbFile | None = None,
    settings: PgvSettings = DEFAULT_PGV,
    fit: FitSettings | None = DEFAULT_FIT,
    event_id: str | None = None,
) -> pd.DataFrame:
    """One row per alert station: distance, fit, filter verdict, magnitudes.

    Columns :data:`PGV_STATION_COLUMNS`. ``distance_km`` follows
    ``settings.distance`` (``"evb"`` needs ``evb``); ``fits_window`` is the
    station's :func:`travel_time_fit` verdict with ``fit`` (NA when ``fit``
    is ``None``), informational only: it never drops a station (the page
    does not). ``kept`` / ``drop_reason`` give the first failing filter, in
    this order: ``excluded_station``, ``excluded_class_code``,
    ``no_distance``, ``bad_pgv``, ``beyond_max_km``, ``nearfield``.
    ``mag_farfield`` / ``mag_exact`` are computed for every row with a
    distance. The near-field cut used is in ``df.attrs["min_km"]``, the
    whole-event :class:`EventFit` (geodesic distances, always) in
    ``df.attrs["fit"]``.
    """
    f = event.frame
    n = len(f)
    geo = station_distances_km(
        origin.latitude,
        origin.longitude,
        f["lat"].to_numpy(dtype=np.float64),
        f["lon"].to_numpy(dtype=np.float64),
        ellps=settings.ellps,
    )
    if settings.distance == "geodesic":
        dist = geo
    elif settings.distance == "evb":
        if evb is None:
            raise ValueError("pgv_stations: distance='evb' needs the evb file")
        dist = (
            evb.station_distances()
            .reindex(f["station"].astype(str).to_numpy())
            .to_numpy(dtype=np.float64)
        )
    else:
        raise ValueError(f"pgv_stations: unknown distance source {settings.distance!r}")
    pgv = f["pgv_um_s"].to_numpy(dtype=np.float64)
    m_far = pgv_magnitude_farfield(pgv, dist)
    m_exact = pgv_magnitude(pgv, dist)
    m_use = m_far if settings.inverse == "farfield" else m_exact

    stations = f["station"].astype(str).to_numpy()
    codes = f["class_code"].astype("Int64")
    reason = np.full(n, None, dtype=object)
    dropped = np.zeros(n, dtype=bool)

    def mark(mask: npt.NDArray[np.bool_], why: str) -> None:
        sel = np.asarray(mask, dtype=bool) & ~dropped
        reason[sel] = why
        dropped[sel] = True

    mark(np.isin(stations, list(settings.exclude_stations)), "excluded_station")
    mark(
        codes.isin(list(settings.exclude_class_codes))
        .fillna(False)
        .to_numpy(dtype=bool),
        "excluded_class_code",
    )
    mark(~np.isfinite(dist), "no_distance")
    mark(~(np.isfinite(pgv) & (pgv > 0.0)), "bad_pgv")
    with np.errstate(invalid="ignore"):
        mark(dist > settings.max_km, "beyond_max_km")
    pre = ~dropped
    if settings.min_km is not None:
        min_km = float(settings.min_km)
    else:
        first = m_use[pre & np.isfinite(m_use)]
        m0 = float(np.median(first)) if first.size else math.nan
        min_km = nearfield_min_km(m0, settings.nearfield_km)
    with np.errstate(invalid="ignore"):
        mark(dist <= min_km, "nearfield")
    kept = ~dropped

    if fit is not None:
        # The association rule always uses the geodesic distance from the
        # alert coordinates (legacy check_event_fit), whatever ``distance``.
        fit_res = travel_time_fit(origin, f["arrival_time"], geo, settings=fit)
        fits = pd.array(fit_res.fits, dtype="boolean")
        out_fit: EventFit | None = fit_res
    else:
        out_fit = None
        fits = pd.array([pd.NA] * n, dtype="boolean")
    day = event.header.time.date().isoformat()
    out = pd.DataFrame(
        {
            "source": pd.Series([PGV_SOURCE] * n, dtype=STRING_DTYPE),
            "event_id": pd.Series([event_id] * n, dtype=STRING_DTYPE),
            "alert_day": pd.Series([day] * n, dtype=STRING_DTYPE),
            "alert_id": pd.Series([event.header.alert_id] * n, dtype="int64"),
            "station": pd.Series(stations, dtype=STRING_DTYPE),
            "lon": f["lon"].to_numpy(dtype=np.float64),
            "lat": f["lat"].to_numpy(dtype=np.float64),
            "class_code": codes.to_numpy(),
            "pgv_um_s": pgv,
            "arrival_time": f["arrival_time"].astype(TIME_DTYPE).to_numpy(),
            "distance_km": dist,
            "distance_source": pd.Series([settings.distance] * n, dtype=STRING_DTYPE),
            "fits_window": fits,
            "kept": kept,
            "drop_reason": pd.Series(list(reason), dtype=STRING_DTYPE),
            "mag_farfield": m_far,
            "mag_exact": m_exact,
        }
    )
    out["class_code"] = out["class_code"].astype("Int64")
    out["arrival_time"] = out["arrival_time"].astype(TIME_DTYPE)
    out = out[list(PGV_STATION_COLUMNS)]
    out.attrs["min_km"] = min_km
    out.attrs["fit"] = out_fit
    return out


@dataclasses.dataclass(frozen=True)
class PgvResult:
    """The PGV magnitude of one event (``magnitude_type="Mpgv_alert"``).

    ``magnitude`` is the median of the kept stations' magnitudes (the main
    value, B3), ``robust_mean`` their :func:`mid_third_mean`; both NaN with
    no kept station. ``n_alert`` = stations in the file, ``n_kept`` = the
    page's "filter sta", ``fit`` the :func:`travel_time_fit` of all alert
    stations (``None`` when not run). ``stations`` is the
    :func:`pgv_stations` frame.
    """

    magnitude: float
    robust_mean: float
    n_alert: int
    n_kept: int
    min_km: float
    inverse: Inverse
    distance: DistanceSource
    fit: EventFit | None
    stations: pd.DataFrame
    alert_id: int
    alert_time: pd.Timestamp
    magnitude_type: str = MAGNITUDE_TYPE

    def summary(self) -> dict[str, Any]:
        """The event-level numbers as a plain dict (the Mpgv page row)."""
        return {
            "magnitude_type": self.magnitude_type,
            "magnitude": self.magnitude,
            "robust_mean": self.robust_mean,
            "n_alert": self.n_alert,
            "n_fit": self.fit.n_fit if self.fit else None,
            "n_kept": self.n_kept,
            "alertfit": self.fit.label if self.fit else None,
            "accepted": self.fit.accepted if self.fit else None,
            "alert_id": self.alert_id,
            "alert_time": self.alert_time,
            "min_km": self.min_km,
            "inverse": self.inverse,
            "distance": self.distance,
        }


def estimate_pgv_magnitude(
    event: AlertEvent,
    origin: Origin,
    *,
    evb: EvbFile | None = None,
    settings: PgvSettings = DEFAULT_PGV,
    fit: FitSettings | None = DEFAULT_FIT,
    event_id: str | None = None,
) -> PgvResult:
    """PGV magnitude of ``origin`` from one alertMap detection (B3 method).

    Builds :func:`pgv_stations`, then the median and mid-third mean of the
    kept stations' magnitudes (``settings.inverse``). The fit verdict is
    reported, not enforced: choosing the detection (and rejecting a
    non-fitting one) is the association step of T7c.
    """
    st = pgv_stations(
        event, origin, evb=evb, settings=settings, fit=fit, event_id=event_id
    )
    col = "mag_farfield" if settings.inverse == "farfield" else "mag_exact"
    vals = st.loc[st["kept"], col].to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    fit_res: EventFit | None = st.attrs.get("fit")
    return PgvResult(
        magnitude=float(np.median(vals)) if vals.size else math.nan,
        robust_mean=mid_third_mean(vals),
        n_alert=len(st),
        n_kept=int(st["kept"].sum()),
        min_km=float(st.attrs["min_km"]),
        inverse=settings.inverse,
        distance=settings.distance,
        fit=fit_res,
        stations=st,
        alert_id=event.header.alert_id,
        alert_time=event.header.time,
    )
