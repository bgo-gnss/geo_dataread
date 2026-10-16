"""The velocity product: one estimator, one versioned record, three formats.

Velocity estimation is its own process and its output a data product
(BGÓ 2026-10-16): the map's vectors, the rates on time-series plots and the
velocity tables fed to deformation models all read the SAME numbers, from
:func:`estimate_station_velocity` or from the files it writes.

Phase 1 is the **windowed** velocity (days to months, the unrest lane): a
line fitted over a chosen window of a plate-removed series, in one frame.

Reference modes (``ref``), all plate-removed first:

- ``plate-periodic`` — the station's stored seasonal removed, then the
  rate: background + whatever happens in the window.
- ``background`` — the stored secular + seasonal (+ steps) removed: the
  rate is the transient alone, relative to the station's background.
- ``plate`` — nothing more removed (``background:none``).

A station with no stored detrend record cannot take the first two; by
default it falls back to ``plate`` and says so (``fallback:plate``,
``background:none``), or is refused (``no_record="refuse"``).

Uncertainty: the WLS formal σ of the window rate times the station's
empirical rate scale k(T) (:func:`gps_analysis.empirical_rate_scale`),
measured on the station's own quiet span (its detrend record window, the
background removed) for windows of the same length. A station without
one takes the median k of the stations in the same run, else
:data:`DEFAULT_K`; ``sigma_method`` says which.

Refusals (``refused``, no rate): fewer than 20 epochs, less than 80 % of
the window spanned, a declared step inside the window (``allow_steps``
turns that into a flag), a failed background removal, no record when one
is required. The model is always a line: a seasonal cannot be estimated
inside a window shorter than ~2.5 yr, it is removed from the record or not
at all.

Files (:func:`write_velocity_set`), all sorted by station, no timestamps:

- ``<name>.gmt`` — ``lon lat ve vn sve svn corr_en station``: GMT
  ``velo -Se`` input, accepted rows only;
- ``<name>.csv`` — every row incl. refused ones, all columns;
- ``<name>.jsonl`` — the full :class:`VelocityRecord` per line.

A set holds ONE frame; mixing frames is refused at write time.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gtimes.timefunc import TimetoYearf

SCHEMA_VERSION = 1
REFS = ("plate-periodic", "background", "plate")
VIEWS = ("cleaned", "raw")
NO_RECORD_POLICIES = ("plate", "refuse")
#: the stored-record terms each ref removes (read_gps_view ``terms``)
_TERMS = {"plate-periodic": "periodic", "background": "all"}
COMPONENTS = ("north", "east", "up")
#: k when neither the station nor the run gives one (NYLA measured 4-7)
DEFAULT_K = 5.0
DEFAULT_FRAME = "itrf2014"
DEFAULT_UNCERT = 10.0
DEFAULT_VELOCITY_DIR = "~/gps-data/velocities"
#: decimals written to the text formats
_RATE_DECIMALS = 3
_COORD_DECIMALS = 6

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VelocityRecord:
    """One station's velocity over one window: the product's row.

    Rates and σ are N/E/U [mm/yr]; NaN (``null`` in JSON) when refused.
    ``sigma`` is what consumers use; ``sigma_formal`` the WLS formal σ it
    was scaled from by ``k``.
    """

    station: str
    lon: float
    lat: float
    frame: str
    plate: str | None
    ref: str
    ref_requested: str
    series: str
    view: str
    uncert: float
    window: tuple[float, float]
    span_used: tuple[float, float] | None
    n_epochs: tuple[int, int, int]
    largest_gap: float | None
    method: str
    model: str
    rate: tuple[float, float, float]
    sigma: tuple[float, float, float]
    sigma_formal: tuple[float, float, float]
    sigma_method: str
    k: tuple[float, float, float]
    k_source: dict[str, Any]
    corr_en: float
    record: dict[str, Any] | None
    steps_in_window: tuple[float, ...]
    outlier_abort: bool
    n_flagged: int
    flags: tuple[str, ...]
    refused: str | None
    software: dict[str, str]
    input_fingerprint: str
    schema_version: int = SCHEMA_VERSION
    kind: str = "window"

    @property
    def accepted(self) -> bool:
        return self.refused is None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: NaN → None, tuples → lists."""
        return {k: _jsonable(v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VelocityRecord:
        """Inverse of :meth:`to_dict`.

        Raises:
            ValueError: an unknown ``schema_version``.
        """
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"velocity record schema {d.get('schema_version')!r}, "
                f"this reader knows {SCHEMA_VERSION}"
            )
        kw = dict(d)
        for key in ("rate", "sigma", "sigma_formal", "k"):
            kw[key] = tuple(np.nan if v is None else float(v) for v in kw[key])
        for key in ("window", "n_epochs", "steps_in_window", "flags"):
            kw[key] = tuple(kw[key])
        if kw.get("span_used") is not None:
            kw["span_used"] = tuple(kw["span_used"])
        kw["corr_en"] = np.nan if kw["corr_en"] is None else float(kw["corr_en"])
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in kw.items() if k in names})


def _jsonable(v: Any) -> Any:
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.floating, np.integer)):
        return _jsonable(v.item())
    return v


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VelocitySettings:
    """How a set of velocities is estimated: what one product run means."""

    frame: str = DEFAULT_FRAME
    ref: str = "plate-periodic"
    view: str = "cleaned"
    uncert: float = DEFAULT_UNCERT
    no_record: str = "plate"
    allow_steps: bool = False
    min_obs: int = 20
    min_cover: float = 0.8
    default_k: float = DEFAULT_K
    pos_dir: str | None = None
    params: str | None = None
    steps: str | None = None

    def __post_init__(self) -> None:
        if self.ref not in REFS:
            raise ValueError(f"ref must be one of {REFS}, got {self.ref!r}")
        if self.view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {self.view!r}")
        if self.no_record not in NO_RECORD_POLICIES:
            raise ValueError(
                f"no_record must be one of {NO_RECORD_POLICIES}, got {self.no_record!r}"
            )


# ---------------------------------------------------------------------------
# the pure core
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WindowRate:
    """:func:`velocity_from_arrays`' result, before any provenance."""

    rate: FloatArray
    sigma_formal: FloatArray
    n_epochs: tuple[int, int, int]
    span_used: tuple[float, float] | None
    largest_gap: float | None
    reasons: tuple[str, ...]
    steps_in_window: tuple[float, ...]


def velocity_from_arrays(
    t: npt.ArrayLike,
    y: npt.ArrayLike,
    sigma: npt.ArrayLike,
    window: tuple[float, float],
    *,
    step_epochs: Sequence[float] = (),
    allow_steps: bool = False,
    min_obs: int = 20,
    min_cover: float = 0.8,
) -> WindowRate:
    """The rate of a line through each component inside ``window``.

    ``y`` and ``sigma`` are (3, N) [mm], NaN where an epoch is dropped
    (outliers); each component is fitted on its own finite epochs. The
    gates (:func:`gps_analysis.window_gates`) apply to the epochs finite
    in ALL components; any reason refuses all three rates.

    Returns:
        The :class:`WindowRate`; ``reasons`` empty when the rate stands.
        With ``allow_steps``, a step inside the window is listed in
        ``steps_in_window`` but does not refuse.
    """
    from gps_analysis import estimate_velocity, largest_gap, window_gates

    tt = np.asarray(t, dtype=np.float64)
    yy = np.atleast_2d(np.asarray(y, dtype=np.float64))
    ss = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
    start, end = float(window[0]), float(window[1])
    inside = np.isfinite(tt) & (tt >= start) & (tt <= end)
    finite = (
        inside
        & np.all(np.isfinite(yy), axis=0)
        & np.all(np.isfinite(ss) & (ss > 0), axis=0)
    )
    steps = tuple(sorted(float(e) for e in step_epochs if start < float(e) < end))
    reasons = list(
        window_gates(tt[finite], (start, end), step_epochs=step_epochs,
                     min_obs=min_obs, min_cover=min_cover)
    )  # fmt: skip
    if allow_steps:
        reasons = [r for r in reasons if not r.startswith("step_in_window")]
    rate = np.full(3, np.nan)
    sig = np.full(3, np.nan)
    counts = [0, 0, 0]
    for c in range(3):
        ok = inside & np.isfinite(yy[c]) & np.isfinite(ss[c]) & (ss[c] > 0)
        counts[c] = int(ok.sum())
        if reasons or counts[c] < 3:
            continue
        est = estimate_velocity(tt[ok], yy[c, ok], ss[c, ok], model="linear")
        rate[c] = float(est.rates[0])
        sig[c] = float(est.sigmas[0])
    used = tt[finite]
    span = (float(used.min()), float(used.max())) if used.size else None
    gap = largest_gap(used) if used.size else None
    return WindowRate(rate, sig, (counts[0], counts[1], counts[2]), span, gap,
                      tuple(reasons), steps)  # fmt: skip


def network_k(records: Sequence[VelocityRecord]) -> FloatArray | None:
    """The per-component median k of the records whose k is their own."""
    own = [r.k for r in records if r.sigma_method == "empirical_k"]
    if not own:
        return None
    return np.asarray(np.nanmedian(np.asarray(own, dtype=float), axis=0), dtype=float)


# ---------------------------------------------------------------------------
# reading a station
# ---------------------------------------------------------------------------


def yearf(value: dt.date | dt.datetime | float) -> float:
    """A date (its midnight), a datetime or a fractional year as a
    fractional year."""
    if isinstance(value, dt.datetime):
        return float(TimetoYearf(*value.timetuple()[:6]))
    if isinstance(value, dt.date):
        return float(TimetoYearf(value.year, value.month, value.day, 0))
    return float(value)


def window_bounds(
    start: dt.date | dt.datetime | float, end: dt.date | dt.datetime | float
) -> tuple[float, float]:
    """A window as fractional years, whole days for dates: from the start
    date's midnight to the midnight AFTER the end date, so both end days
    are in (a datetime or a fractional year is taken as given)."""
    if isinstance(end, dt.date) and not isinstance(end, dt.datetime):
        end = end + dt.timedelta(days=1)
    return yearf(start), yearf(end)


def _pos_dir(settings: VelocitySettings) -> str:
    if settings.pos_dir is not None:
        return str(Path(settings.pos_dir).expanduser())
    from gps_parser import ConfigParser

    return str(Path(str(ConfigParser().getPostProcessDir("pos_dir"))).expanduser())


def _product(settings: VelocitySettings) -> tuple[str, str]:
    """(pos product id, the frame's full name) of the settings."""
    from geo_dataread.pos import frame_name, pos_product

    return pos_product(settings.frame), frame_name(settings.frame)


def _params_doc(settings: VelocitySettings) -> Mapping[str, Any] | None:
    from geo_dataread.gps_views import default_params_path, read_detrend_params

    pid, _ = _product(settings)
    path = settings.params or default_params_path(pid)
    if path is None or not Path(path).is_file():
        return None
    return read_detrend_params(path)


def _record(doc: Mapping[str, Any] | None, sta: str) -> dict[str, Any] | None:
    if doc is None:
        return None
    from geo_dataread.gps_views import station_detrend_record

    try:
        record, _source = station_detrend_record(doc, sta)
    except (FileNotFoundError, ValueError):
        return None
    return dict(record) if record is not None else None


def _read(
    sta: str,
    start: float,
    end: float,
    settings: VelocitySettings,
    *,
    ref: str,
    doc: Mapping[str, Any] | None,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
    """(t, y (3,N), σ (3,N), view attrs) of one station: plate removed, the
    ``ref`` terms of its stored record removed, outliers NaN (cleaned)."""
    from geo_dataread.gps_views import read_gps_view

    pid, _ = _product(settings)
    clean = settings.view == "cleaned"
    detrend = ref in _TERMS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = read_gps_view(
            sta, start, end,
            view="detrended" if detrend else "raw",
            terms=_TERMS.get(ref, "all"),
            clean=clean,
            params=doc,
            steps=settings.steps,
            tType="POS", pos_id=pid, Dir=_pos_dir(settings),
            uncert=settings.uncert,
        )  # fmt: skip
    attrs = dict(df.attrs.get("gps_view", {}))
    t = df["yearf"].to_numpy(dtype=np.float64)
    cols = [f"{c}_detrended" if detrend else c for c in COMPONENTS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{sta}: background not removed ({attrs.get('degrade_reason') or missing})"
        )
    y = np.vstack([df[c].to_numpy(dtype=np.float64) for c in cols])
    s = np.vstack([df[f"D{c}"].to_numpy(dtype=np.float64) for c in COMPONENTS])
    if clean and not attrs.get("outlier_abort"):
        for i, c in enumerate(COMPONENTS):
            mask = df.get(f"{c}_outlier")
            if mask is not None:
                y[i, mask.to_numpy(dtype=bool)] = np.nan
    return t, y, s, attrs


def station_rate_scale(
    sta: str,
    window_years: float,
    settings: VelocitySettings,
    *,
    doc: Mapping[str, Any] | None = None,
) -> tuple[FloatArray | None, dict[str, Any]]:
    """The station's k(T) for windows of ``window_years``, measured on its
    detrend record's window with the whole stored model removed; ``(None,
    info)`` when it has no record or too short a quiet span."""
    from gps_analysis import empirical_rate_scale

    record = _record(doc, sta)
    if record is None or not record.get("window"):
        return None, {"reason": "no record"}
    w0, w1 = (float(v) for v in record["window"])
    try:
        t, y, s, _ = _read(sta, w0, w1, settings, ref="background", doc=doc)
    except ValueError as exc:
        return None, {"reason": str(exc)}
    rs = empirical_rate_scale(t, y, s, window_years=window_years)
    info = {"quiet_span": [w0, w1], "n_windows": [int(n) for n in rs.n_windows]}
    if not np.all(np.isfinite(rs.k)):
        info["reason"] = "too few windows on the quiet span"
        return None, info
    return np.asarray(rs.k, dtype=float), info


# ---------------------------------------------------------------------------
# one station
# ---------------------------------------------------------------------------


def _software() -> dict[str, str]:
    out = {}
    for pkg in ("geo_dataread", "gps_analysis"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def _fingerprint(
    t: FloatArray, y: FloatArray, s: FloatArray, extra: Mapping[str, Any]
) -> str:
    h = hashlib.sha256()
    for a in (t, y, s):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    h.update(json.dumps(_jsonable(dict(extra)), sort_keys=True).encode())
    return h.hexdigest()


def _coords(sta: str) -> tuple[float, float]:
    from gps_parser import ConfigParser

    info = dict(dict(ConfigParser().getStationInfo(sta))["station"])
    return float(info["longitude"]), float(info["latitude"])


def _record_summary(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    borrowed = record.get("borrowed")
    return {
        "frame": record.get("frame"),
        "window": list(record.get("window") or []),
        "fitted_at": record.get("fitted_at"),
        "detrend_method": record.get("detrend_method"),
        "borrowed": (borrowed or {}).get("from") if borrowed else None,
    }


def estimate_station_velocity(
    sta: str,
    start: dt.date | dt.datetime | float,
    end: dt.date | dt.datetime | float,
    settings: VelocitySettings | None = None,
    *,
    doc: Mapping[str, Any] | None = None,
    k: npt.ArrayLike | None = None,
    k_method: str | None = None,
    coords: tuple[float, float] | None = None,
) -> VelocityRecord:
    """The velocity of ``sta`` over [start, end]: THE product's estimator.

    Args:
        sta: Station marker.
        start, end: The window: dates (whole days, both included,
            :func:`window_bounds`), datetimes or fractional years.
        settings: How (:class:`VelocitySettings`); default settings when
            None.
        doc: The detrend parameter document of the settings' frame; read
            when None.
        k: The rate scale to use, shape (3,); None measures the station's
            own (:func:`station_rate_scale`), falling back to
            ``settings.default_k``.
        k_method: The ``sigma_method`` a given ``k`` carries
            (``empirical_k_network`` for a run median).
        coords: (lon, lat); stations.cfg when None.

    Never raises for a station-level reason: a station that cannot give a
    rate comes back ``refused``, with the reason. Raises only on bad
    arguments.
    """
    settings = settings or VelocitySettings()
    sta = sta.upper()
    t0, t1 = window_bounds(start, end)
    if not t1 > t0:
        raise ValueError(f"window end {t1} is not after its start {t0}")
    pid, frame = _product(settings)
    if doc is None:
        doc = _params_doc(settings)
    record = _record(doc, sta)
    flags: list[str] = []
    ref = settings.ref
    refused: str | None = None
    if ref in _TERMS and record is None:
        if settings.no_record == "plate":
            ref = "plate"
            flags.append("fallback:plate")
        else:
            refused = "no_record"
    if ref == "plate":
        flags.append("background:none")
    if record is not None and record.get("borrowed") and ref in _TERMS:
        flags.append(f"borrowed:{(record['borrowed'] or {}).get('from')}")
    try:
        lon, lat = coords if coords is not None else _coords(sta)
    except (KeyError, TypeError, ValueError):
        lon, lat = np.nan, np.nan
        flags.append("no_coordinates")
    from geo_dataread.gps_views import station_step_epochs
    from geo_dataread.station_catalog import station_plate

    try:
        step_epochs, _src = station_step_epochs(sta, steps=settings.steps)
    except (FileNotFoundError, ValueError):
        step_epochs = np.zeros(0)
    t = y = s = np.zeros((0,))
    attrs: dict[str, Any] = {}
    wr = None
    if refused is None:
        try:
            t, y, s, attrs = _read(sta, t0, t1, settings, ref=ref, doc=doc)
        except ValueError as exc:
            refused = f"read: {exc}"
    if refused is None:
        if attrs.get("outlier_abort"):
            flags.append("outlier_abort")
        wr = velocity_from_arrays(
            t, y, s, (t0, t1), step_epochs=step_epochs.tolist(),
            allow_steps=settings.allow_steps, min_obs=settings.min_obs,
            min_cover=settings.min_cover,
        )  # fmt: skip
        if wr.reasons:
            refused = "; ".join(wr.reasons)
        elif wr.steps_in_window:
            flags.append("steps_in_window")
    k_source: dict[str, Any] = {}
    if k is not None:
        kk = np.asarray(k, dtype=float).reshape(3)
        sigma_method = k_method or "empirical_k"
    else:
        own: FloatArray | None = None
        if refused is None and record is not None:
            own, k_source = station_rate_scale(sta, t1 - t0, settings, doc=doc)
        if own is not None:
            kk, sigma_method = own, "empirical_k"
        else:
            kk, sigma_method = np.full(3, settings.default_k), "default_k"
    nan3 = (np.nan, np.nan, np.nan)
    rate = tuple(float(v) for v in wr.rate) if wr and refused is None else nan3
    formal = (
        tuple(float(v) for v in wr.sigma_formal) if wr and refused is None else nan3
    )
    sigma = tuple(float(a * b) for a, b in zip(formal, kk, strict=True))
    fp_extra = {"settings": dataclasses.asdict(settings), "ref": ref,
                "window": [t0, t1], "record": _record_summary(record)}  # fmt: skip
    return VelocityRecord(
        station=sta,
        lon=float(lon),
        lat=float(lat),
        frame=frame,
        plate=station_plate(sta),
        ref=ref,
        ref_requested=settings.ref,
        series=pid,
        view=settings.view,
        uncert=float(settings.uncert),
        window=(t0, t1),
        span_used=wr.span_used if wr else None,
        n_epochs=wr.n_epochs if wr else (0, 0, 0),
        largest_gap=wr.largest_gap if wr else None,
        method="wls",
        model="linear",
        rate=rate,  # type: ignore[arg-type]
        sigma=sigma,  # type: ignore[arg-type]
        sigma_formal=formal,  # type: ignore[arg-type]
        sigma_method=sigma_method,
        k=tuple(float(v) for v in kk),  # type: ignore[arg-type]
        k_source=k_source,
        corr_en=0.0 if refused is None else np.nan,
        record=_record_summary(record) if ref in _TERMS else None,
        steps_in_window=wr.steps_in_window if wr else (),
        outlier_abort=bool(attrs.get("outlier_abort", False)),
        n_flagged=int(attrs.get("n_flagged") or 0),
        flags=tuple(flags),
        refused=refused,
        software=_software(),
        input_fingerprint=_fingerprint(t, y, s, fp_extra),
    )


# ---------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------


def estimate_velocities(
    stations: Sequence[str],
    start: dt.date | dt.datetime | float,
    end: dt.date | dt.datetime | float,
    settings: VelocitySettings | None = None,
    *,
    progress: Any = None,
) -> list[VelocityRecord]:
    """Every station over one window, sorted by station.

    Stations with a quiet span measure their own k first; the others take
    the per-component median of those (``empirical_k_network``), else
    ``settings.default_k``.
    """
    settings = settings or VelocitySettings()
    doc = _params_doc(settings)
    first = [
        estimate_station_velocity(sta, start, end, settings, doc=doc)
        for sta in sorted({x.upper() for x in stations})
    ]
    median = network_k(first)
    out = []
    for r in first:
        if r.sigma_method == "default_k" and median is not None and r.accepted:
            scaled = [float(a * b) for a, b in zip(r.sigma_formal, median, strict=True)]
            r = dataclasses.replace(
                r,
                k=(float(median[0]), float(median[1]), float(median[2])),
                sigma=(scaled[0], scaled[1], scaled[2]),
                sigma_method="empirical_k_network",
            )
        out.append(r)
        if progress is not None:
            progress(r)
    return out


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def _fmt(v: float, decimals: int) -> str:
    return "" if not math.isfinite(v) else f"{v:.{decimals}f}"


CSV_COLUMNS = (
    "station", "lon", "lat", "ve", "vn", "vu", "sve", "svn", "svu", "corr_en",
    "n_epochs", "window_start", "window_end", "ref", "frame", "plate",
    "sigma_method", "k_e", "k_n", "k_u", "flags", "refused",
)  # fmt: skip


def write_velocity_set(
    records: Sequence[VelocityRecord], directory: str | Path, name: str
) -> dict[str, Path]:
    """Write ``<name>.gmt``, ``.csv`` and ``.jsonl`` under ``directory``.

    Byte-stable: rows sorted by station, fixed decimals, no timestamps.

    Raises:
        ValueError: records in more than one frame, or none.
    """
    if not records:
        raise ValueError("no velocity records to write")
    frames = {r.frame for r in records}
    if len(frames) > 1:
        raise ValueError(f"a velocity set holds one frame, got {sorted(frames)}")
    rows = sorted(records, key=lambda r: (r.station, r.window))
    out_dir = Path(directory).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (frame,) = frames
    head = (
        f"# gps-velocities schema {SCHEMA_VERSION}, set {name}, frame {frame}, "
        f"{sum(r.accepted for r in rows)} of {len(rows)} stations accepted\n"
    )
    paths = {
        "gmt": out_dir / f"{name}.gmt",
        "csv": out_dir / f"{name}.csv",
        "jsonl": out_dir / f"{name}.jsonl",
    }
    d, c = _RATE_DECIMALS, _COORD_DECIMALS
    with open(paths["gmt"], "w", encoding="utf-8") as fh:
        fh.write(head + "# lon lat ve vn sve svn corr_en station  [mm/yr]\n")
        for r in rows:
            if not r.accepted:
                continue
            vn, ve, _vu = r.rate
            sn, se, _su = r.sigma
            fh.write(
                f"{r.lon:.{c}f} {r.lat:.{c}f} {ve:.{d}f} {vn:.{d}f} {se:.{d}f} "
                f"{sn:.{d}f} {r.corr_en:.{d}f} {r.station}\n"
            )
    with open(paths["csv"], "w", encoding="utf-8") as fh:
        fh.write(head + ",".join(CSV_COLUMNS) + "\n")
        for r in rows:
            vn, ve, vu = r.rate
            sn, se, su = r.sigma
            kn, ke, ku = r.k
            cells = [
                r.station, _fmt(r.lon, c), _fmt(r.lat, c),
                _fmt(ve, d), _fmt(vn, d), _fmt(vu, d),
                _fmt(se, d), _fmt(sn, d), _fmt(su, d), _fmt(r.corr_en, d),
                str(min(r.n_epochs)), f"{r.window[0]:.4f}", f"{r.window[1]:.4f}",
                r.ref, r.frame, r.plate or "", r.sigma_method,
                _fmt(ke, 2), _fmt(kn, 2), _fmt(ku, 2),
                "|".join(r.flags), (r.refused or "").replace(",", ";"),
            ]  # fmt: skip
            fh.write(",".join(cells) + "\n")
    with open(paths["jsonl"], "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    return paths


def read_velocity_records(path: str | Path) -> list[VelocityRecord]:
    """The records of a ``.jsonl`` velocity set (a ``.csv``/``.gmt`` path
    reads its ``.jsonl`` sibling)."""
    p = Path(path).expanduser()
    if p.suffix != ".jsonl":
        p = p.with_suffix(".jsonl")
    with open(p, encoding="utf-8") as fh:
        return [
            VelocityRecord.from_dict(json.loads(line)) for line in fh if line.strip()
        ]


def velocity_dir() -> Path:
    """``velocity_dir`` of postprocess.cfg [PATHS], else
    :data:`DEFAULT_VELOCITY_DIR`."""
    try:
        from gps_parser import ConfigParser

        found = ConfigParser().getPostProcessDir("velocity_dir")
        if found:
            return Path(str(found)).expanduser()
    except Exception:  # noqa: BLE001 - optional key
        pass
    return Path(DEFAULT_VELOCITY_DIR).expanduser()


def set_name(settings: VelocitySettings, start: dt.date, end: dt.date) -> str:
    """``window-<start>-<end>_<ref>_<frame>``, the dates as YYYYMMDD
    (both included)."""
    from geo_dataread.pos import pos_frame

    return (
        f"window-{start:%Y%m%d}-{end:%Y%m%d}_{settings.ref}_{pos_frame(settings.frame)}"
    )


# ---------------------------------------------------------------------------
# command line: gps-velocities
# ---------------------------------------------------------------------------


def _date(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: use YYYYMMDD") from None


def _window(value: str) -> tuple[dt.date, dt.date]:
    a, sep, b = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"{value!r}: use START:END (YYYYMMDD)")
    return _date(a), _date(b)


def _box(value: str) -> tuple[float, float, float, float]:
    try:
        w, e, s, n = (float(v) for v in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: use W/E/S/N") from None
    return w, e, s, n


def select_stations(
    patterns: Sequence[str], region: tuple[float, float, float, float] | None
) -> list[str]:
    """stations.cfg markers matching ``patterns`` (all when empty) and
    inside ``region`` (W/E/S/N) when given."""
    from gps_parser import ConfigParser

    cfg = ConfigParser()
    names = [str(s) for s in cfg.getStationInfo()]
    pats = [p.upper() for p in patterns] or ["*"]
    picked = [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in pats)]
    if region is None:
        return sorted(picked)
    w, e, s, n = region
    out = []
    for sta in picked:
        try:
            lon, lat = _coords(sta)
        except (KeyError, TypeError, ValueError):
            continue
        if w <= lon <= e and s <= lat <= n:
            out.append(sta)
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gps-velocities",
        description="The velocity product: estimate windowed station velocities "
        "into a versioned set (.gmt/.csv/.jsonl), or show one.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="estimate a set over one window")
    e.add_argument("stations", nargs="*", help="markers or patterns (SEY?, *)")
    e.add_argument("--region", type=_box, default=None, metavar="W/E/S/N")
    e.add_argument("--window", type=_window, default=None, metavar="START:END")
    e.add_argument("--last", type=int, default=None, metavar="DAYS")
    e.add_argument("-e", "--end", type=_date, default=None,
                   help="YYYYMMDD: the end of --last (default today)")  # fmt: skip
    e.add_argument("--ref", choices=REFS, default="plate-periodic",
                   help="plate-periodic: seasonal removed; background: secular + "
                   "seasonal removed (the transient); plate: nothing more")  # fmt: skip
    e.add_argument("--view", choices=VIEWS, default="cleaned")
    e.add_argument("--frame", default=DEFAULT_FRAME)
    e.add_argument("-u", "--uncert", type=float, default=DEFAULT_UNCERT)
    e.add_argument("--no-record", choices=NO_RECORD_POLICIES, default="plate",
                   help="a station without a detrend record: fall back to ref "
                   "plate (flagged) or refuse")  # fmt: skip
    e.add_argument("--allow-steps", action="store_true",
                   help="keep a window with a declared step inside (flagged)")  # fmt: skip
    e.add_argument("--min-obs", type=int, default=20)
    e.add_argument("--min-cover", type=float, default=0.8)
    e.add_argument("--default-k", type=float, default=DEFAULT_K)
    e.add_argument("--pos-dir", default=None)
    e.add_argument("--params", default=None, help="detrend parameter document")
    e.add_argument("--steps", default=None, help="steps.yaml")
    e.add_argument("--out", default=None,
                   help="directory (default: velocity_dir of postprocess.cfg, "
                   f"else {DEFAULT_VELOCITY_DIR})")  # fmt: skip
    e.add_argument(
        "--name", default=None, help="set name (default from window/ref/frame)"
    )
    s = sub.add_parser("show", help="print a set")
    s.add_argument("path")
    s.add_argument("--all", action="store_true", help="refused rows too")
    return p


def _show(records: Sequence[VelocityRecord], show_all: bool) -> None:
    print(f"{'sta':4} {'vn':>9} {'ve':>9} {'vu':>9} {'sn':>7} {'se':>7} {'su':>7} "
          f"{'n':>4} {'σ':>18} ref/flags")  # fmt: skip
    for r in records:
        if not (r.accepted or show_all):
            continue
        if r.accepted:
            vn, ve, vu = r.rate
            sn, se, su = r.sigma
            nums = f"{vn:9.2f} {ve:9.2f} {vu:9.2f} {sn:7.2f} {se:7.2f} {su:7.2f}"
        else:
            nums = f"{'refused':>9} {'':9} {'':9} {'':7} {'':7} {'':7}"
        note = r.refused or "|".join((r.ref, *r.flags))
        print(f"{r.station:4} {nums} {min(r.n_epochs):4d} {r.sigma_method:>18} {note}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "show":
        _show(read_velocity_records(args.path), args.all)
        return 0
    if (args.window is None) == (args.last is None):
        print("error: give --window START:END or --last DAYS", file=sys.stderr)
        return 2
    if args.window is not None:
        start, end = args.window
    else:
        end = args.end or dt.date.today()
        start = end - dt.timedelta(days=args.last)
    try:
        settings = VelocitySettings(
            frame=args.frame, ref=args.ref, view=args.view, uncert=args.uncert,
            no_record=args.no_record, allow_steps=args.allow_steps,
            min_obs=args.min_obs, min_cover=args.min_cover,
            default_k=args.default_k, pos_dir=args.pos_dir, params=args.params,
            steps=args.steps,
        )  # fmt: skip
        stations = select_stations(args.stations, args.region)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not stations:
        print("error: no station selected", file=sys.stderr)
        return 2

    def report(r: VelocityRecord) -> None:
        state = "ok" if r.accepted else f"refused: {r.refused}"
        print(f"{r.station}: [{state}] {r.ref} {r.sigma_method}", file=sys.stderr)

    records = estimate_velocities(stations, start, end, settings, progress=report)
    name = args.name or set_name(settings, start, end)
    out = Path(args.out).expanduser() if args.out else velocity_dir()
    paths = write_velocity_set(records, out, name)
    n_ok = sum(r.accepted for r in records)
    print(
        f"{n_ok} of {len(records)} stations -> {paths['csv'].with_suffix('')}.{{gmt,csv,jsonl}}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPONENTS", "DEFAULT_K", "REFS", "SCHEMA_VERSION", "VIEWS",
    "VelocityRecord", "VelocitySettings", "WindowRate",
    "estimate_station_velocity", "estimate_velocities", "network_k",
    "read_velocity_records", "select_stations", "set_name",
    "station_rate_scale", "velocity_dir", "velocity_from_arrays",
    "window_bounds", "write_velocity_set", "yearf",
]  # fmt: skip
