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
one takes the stored k(T) table's (:func:`load_k_table`), else
:data:`DEFAULT_K`; ``sigma_method`` says which. Never a median of the run:
a record depends on its station, the settings and the table alone.

Refusals (``refused``, no rate): fewer than 5 epochs, less than 80 % of
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
from gps_analysis.rate_quality import NONLINEAR_F, SHORT_SNR, SHORT_SNR_PAIR
from gtimes.timefunc import TimetoYearf

SCHEMA_VERSION = 2
#: the record schemas this reader takes (1: no tier/snr/parts, read with
#: the standard-tier defaults)
READ_SCHEMAS = (1, 2)
REFS = ("plate-periodic", "background", "plate")
#: declared: the raw series less the operator-declared blunder epochs
#: (excluded_epochs.csv), no detection, the default; cleaned: outliers
#: detected on the window (flags real signal in unrest, see the docs); raw
VIEWS = ("declared", "cleaned", "raw")
DEFAULT_VIEW = "declared"
#: k is looked up at this window length at least: below a week the
#: sliding-window k is not measured (the table holds its 7 d value)
K_MIN_DAYS = 7
TIERS = ("standard", "short")
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
    #: "standard": the count/cover gates passed; "short": they did not, and
    #: the parts that stand do so on |v|/σ (``snr``)
    tier: str = "standard"
    #: (|v_h|/σ_h, |v_u|/σ_u) of the rate (NaN when refused)
    snr: tuple[float, float] = (math.nan, math.nan)
    #: χ²/ν and the quadratic's F per component (NaN when not fitted)
    chi2nu: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    f_quadratic: tuple[float, float, float] = (math.nan, math.nan, math.nan)
    #: the parts of an accepted short-tier record that did not stand
    #: ("horizontal", "up") and why; their rates are NaN
    parts_refused: dict[str, str] = dataclasses.field(default_factory=dict)
    #: the station's last epoch when it is before the window end [yr]
    data_end: float | None = None

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
        if d.get("schema_version") not in READ_SCHEMAS:
            raise ValueError(
                f"velocity record schema {d.get('schema_version')!r}, "
                f"this reader knows {READ_SCHEMAS}"
            )
        kw = dict(d)
        for key in ("rate", "sigma", "sigma_formal", "k", "snr", "chi2nu",
                    "f_quadratic"):  # fmt: skip
            if key in kw:
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
    view: str = DEFAULT_VIEW
    uncert: float = DEFAULT_UNCERT
    no_record: str = "plate"
    allow_steps: bool = False
    min_obs: int = 5
    min_cover: float = 0.8
    default_k: float = DEFAULT_K
    pos_dir: str | None = None
    params: str | None = None
    steps: str | None = None
    #: the k(T) table (:func:`build_k_table`): a path, None for the
    #: deployed one when it exists, "none" for no table
    k_table: str | None = None
    #: the short tier: a window failing only the count/cover gates keeps
    #: the parts whose |v|/σ reaches short_snr (short_snr_pair on 2 epochs)
    short_tier: bool = True
    short_snr: float = SHORT_SNR
    short_snr_pair: float = SHORT_SNR_PAIR

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
    #: the short-tier σ (a priori × √max(1, χ²/ν)) [mm/yr]
    sigma_short: FloatArray = dataclasses.field(
        default_factory=lambda: np.full(3, np.nan)
    )
    chi2nu: FloatArray = dataclasses.field(default_factory=lambda: np.full(3, np.nan))
    f_quadratic: FloatArray = dataclasses.field(
        default_factory=lambda: np.full(3, np.nan)
    )
    #: only the count/cover gates failed, with ≥ 2 epochs: the rates are
    #: fitted, for the short tier to judge
    short: bool = False


def velocity_from_arrays(
    t: npt.ArrayLike,
    y: npt.ArrayLike,
    sigma: npt.ArrayLike,
    window: tuple[float, float],
    *,
    step_epochs: Sequence[float] = (),
    allow_steps: bool = False,
    min_obs: int = 5,
    min_cover: float = 0.8,
    gate_end: float | None = None,
    epoch_span: float = 1 / 365.25,
) -> WindowRate:
    """The rate of a line through each component inside ``window``.

    ``y`` and ``sigma`` are (3, N) [mm], NaN where an epoch is dropped
    (outliers); each component is fitted on its own finite epochs
    (:func:`gps_analysis.line_fit`). The gates
    (:func:`gps_analysis.window_gates`) apply to the epochs finite in ALL
    components; any reason refuses all three rates, except that a window
    failing only the count/cover gates with ≥ 2 epochs is still fitted
    and marked ``short``, for the caller's short tier to judge.

    Args:
        gate_end: The end the cover gate measures against, when the data
            end before ``window`` does (default the window end).
        epoch_span: The time one epoch stands for, in the cover gate
            (a daily solution: its day).

    Returns:
        The :class:`WindowRate`; ``reasons`` empty when the rate stands.
        With ``allow_steps``, a step inside the window is listed in
        ``steps_in_window`` but does not refuse.
    """
    from gps_analysis import largest_gap, line_fit, window_gates

    tt = np.asarray(t, dtype=np.float64)
    yy = np.atleast_2d(np.asarray(y, dtype=np.float64))
    ss = np.atleast_2d(np.asarray(sigma, dtype=np.float64))
    start, end = float(window[0]), float(window[1])
    inside = np.isfinite(tt) & (tt >= start) & (tt < end)
    finite = (
        inside
        & np.all(np.isfinite(yy), axis=0)
        & np.all(np.isfinite(ss) & (ss > 0), axis=0)
    )
    steps = tuple(sorted(float(e) for e in step_epochs if start < float(e) < end))
    g_end = end if gate_end is None else min(end, float(gate_end))
    reasons = list(
        window_gates(tt[finite], (start, max(g_end, start + 1e-9)),
                     step_epochs=step_epochs, min_obs=min_obs,
                     min_cover=min_cover, epoch_span=epoch_span)
    )  # fmt: skip
    if allow_steps:
        reasons = [r for r in reasons if not r.startswith("step_in_window")]
    hard = any(r.startswith("step_in_window") for r in reasons)
    short = bool(reasons) and not hard and np.unique(tt[finite]).size >= 2
    rate, sig, sig_short, chi, fq = (np.full(3, np.nan) for _ in range(5))
    counts = [0, 0, 0]
    for c in range(3):
        ok = inside & np.isfinite(yy[c]) & np.isfinite(ss[c]) & (ss[c] > 0)
        counts[c] = int(ok.sum())
        if (reasons and not short) or counts[c] < 2 or np.ptp(tt[ok]) <= 0:
            continue
        if counts[c] < 3 and not short:
            continue
        fit = line_fit(tt[ok], yy[c, ok], ss[c, ok])
        rate[c], sig[c], sig_short[c] = fit.rate, fit.sigma_scaled, fit.sigma_short
        chi[c], fq[c] = fit.chi2nu, fit.f_quadratic
    used = tt[finite]
    span = (float(used.min()), float(used.max())) if used.size else None
    gap = largest_gap(used) if used.size else None
    return WindowRate(rate, sig, (counts[0], counts[1], counts[2]), span, gap,
                      tuple(reasons), steps, sig_short, chi, fq,
                      short)  # fmt: skip


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


def last_days_start(end: dt.date, days: int) -> dt.date:
    """The first day of the ``days`` days ending with ``end`` (both in):
    ``--last 7`` is seven whole days."""
    if days < 1:
        raise ValueError(f"--last {days}: at least one day")
    return end - dt.timedelta(days=days - 1)


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
    terms: str | None,
    doc: Mapping[str, Any] | None,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
    """(t, y (3,N), σ (3,N), view attrs) of one station: plate removed, the
    ``terms`` of its stored record removed (None: none), outliers NaN
    (cleaned) or the declared blunder epochs NaN (declared)."""
    from geo_dataread.gps_views import read_gps_view

    pid, _ = _product(settings)
    clean = settings.view == "cleaned"
    detrend = terms is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = read_gps_view(
            sta, start, end,
            view="detrended" if detrend else "raw",
            terms=terms or "all",
            clean=clean,
            params=doc,
            steps=settings.steps,
            tType="POS", pos_id=pid, Dir=_pos_dir(settings),
            uncert=settings.uncert,
        )  # fmt: skip
    attrs = dict(df.attrs.get("gps_view", {}))
    if settings.view == "declared":
        from gps_parser import outlier_catalogs as _oc

        from geo_dataread.gps_views import station_excluded_epochs

        xepochs, xsource = station_excluded_epochs(sta)
        tt = df["yearf"].to_numpy(dtype=np.float64)
        idx = list(_oc.match_declared_epochs(tt, np.asarray(xepochs, float)))
        attrs["excluded_epochs_source"] = xsource
        attrs["n_flagged"] = len(idx)
    t = df["yearf"].to_numpy(dtype=np.float64)
    cols = [f"{c}_detrended" if detrend else c for c in COMPONENTS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{sta}: background not removed ({attrs.get('degrade_reason') or missing})"
        )
    y = np.vstack([df[c].to_numpy(dtype=np.float64) for c in cols])
    s = np.vstack([df[f"D{c}"].to_numpy(dtype=np.float64) for c in COMPONENTS])
    if settings.view == "declared" and idx:
        y[:, idx] = np.nan
    if clean and not attrs.get("outlier_abort"):
        for i, c in enumerate(COMPONENTS):
            mask = df.get(f"{c}_outlier")
            if mask is not None:
                y[i, mask.to_numpy(dtype=bool)] = np.nan
    return t, y, s, attrs


#: the residual k is measured on, by ref: a plate rate still carries the
#: seasonal slope, so its k must too (only the secular removed)
K_RESIDUAL = {"plate-periodic": "all", "background": "all", "plate": "secular"}


def station_rate_scales(
    sta: str,
    years: Sequence[float],
    settings: VelocitySettings,
    *,
    doc: Mapping[str, Any] | None = None,
    residual: str = "all",
) -> tuple[dict[float, FloatArray], dict[str, Any]]:
    """The station's k(T) for each window length of ``years``, measured on
    its detrend record's window with ``residual`` of the stored model
    removed (``"all"``, or ``"secular"`` to keep the seasonal in);
    ``({}, info)`` when it has no record. A length with too few windows
    on the quiet span is left out."""
    from gps_analysis import empirical_rate_scale

    record = _record(doc, sta)
    if record is None or not record.get("window"):
        return {}, {"reason": "no record"}
    w0, w1 = (float(v) for v in record["window"])
    try:
        t, y, s, _ = _read(sta, w0, w1, settings, terms=residual, doc=doc)
    except ValueError as exc:
        return {}, {"reason": str(exc)}
    out: dict[float, FloatArray] = {}
    for T in years:
        rs = empirical_rate_scale(t, y, s, window_years=float(T))
        if np.all(np.isfinite(rs.k)):
            out[float(T)] = np.asarray(rs.k, dtype=float)
    return out, {"quiet_span": [w0, w1], "residual": residual}


def station_rate_scale(
    sta: str,
    window_years: float,
    settings: VelocitySettings,
    *,
    doc: Mapping[str, Any] | None = None,
    residual: str = "all",
) -> tuple[FloatArray | None, dict[str, Any]]:
    """:func:`station_rate_scales` for one window length; ``(None, info)``
    when there is no k."""
    got, info = station_rate_scales(
        sta, [window_years], settings, doc=doc, residual=residual
    )
    if not got:
        info.setdefault("reason", "too few windows on the quiet span")
        return None, info
    return got[float(window_years)], info


# ---------------------------------------------------------------------------
# the k(T) table: the fallback that does not depend on the run
# ---------------------------------------------------------------------------

#: window lengths of the k(T) table [days]
K_TABLE_DAYS = (7, 14, 30, 61, 91, 182, 365, 730)
K_TABLE_SCHEMA = 1


def k_table_path(settings: VelocitySettings) -> Path:
    """``k_table_<frame>.json`` in :func:`velocity_dir`."""
    from geo_dataread.pos import pos_frame

    return velocity_dir() / f"k_table_{pos_frame(settings.frame)}.json"


def build_k_table(
    stations: Sequence[str],
    settings: VelocitySettings,
    *,
    days: Sequence[int] = K_TABLE_DAYS,
    progress: Any = None,
) -> dict[str, Any]:
    """The per-component median k(T) over every station with a record:
    for both residuals (``all``: the seasonal removed; ``secular``: kept),
    at the lengths of ``days``. Deterministic for the same inputs; what a
    station without a quiet span of its own takes (sigma_method
    ``empirical_k_table``)."""
    doc = _params_doc(settings)
    years = [d / 365.25 for d in days]
    per: dict[str, dict[float, list[FloatArray]]] = {"all": {}, "secular": {}}
    used: list[str] = []
    for sta in sorted({x.upper() for x in stations}):
        if _record(doc, sta) is None:
            continue
        got_any = False
        for residual in per:
            got, _info = station_rate_scales(
                sta, years, settings, doc=doc, residual=residual
            )
            for T, k in got.items():
                per[residual].setdefault(T, []).append(k)
            got_any |= bool(got)
        if got_any:
            used.append(sta)
        if progress is not None:
            progress(sta)
    residuals = {}
    for residual, by_t in per.items():
        rows = [(T, np.asarray(by_t[T])) for T in years if T in by_t]
        residuals[residual] = {
            "days": [round(T * 365.25) for T, _ in rows],
            "k": [
                [round(float(v), 4) for v in np.nanmedian(a, axis=0)] for _, a in rows
            ],
            "n_stations": [int(a.shape[0]) for _, a in rows],
        }
    pid, frame = _product(settings)
    return {
        "schema": K_TABLE_SCHEMA,
        "frame": frame,
        "series": pid,
        "view": settings.view,
        "uncert": settings.uncert,
        "stations": used,
        "residuals": residuals,
    }


def table_k(
    table: Mapping[str, Any], residual: str, window_years: float
) -> FloatArray | None:
    """k at ``window_years`` from a k(T) table, or None when it has no
    entry for ``residual``."""
    from gps_analysis import rate_scale_at

    entry = (table.get("residuals") or {}).get(residual)
    if not entry or not entry.get("days"):
        return None
    k = rate_scale_at(
        window_years, np.asarray(entry["days"], float) / 365.25, entry["k"]
    )
    return k if np.all(np.isfinite(k)) else None


def load_k_table(
    settings: VelocitySettings,
) -> tuple[dict[str, Any] | None, str | None]:
    """(table, its path + sha256) of the settings, or (None, None)."""
    if settings.k_table == "none":
        return None, None
    path = (
        Path(settings.k_table).expanduser()
        if settings.k_table
        else k_table_path(settings)
    )
    if not path.is_file():
        if settings.k_table:
            raise FileNotFoundError(f"k table {path} not found")
        return None, None
    raw = path.read_bytes()
    table = json.loads(raw)
    if table.get("schema") != K_TABLE_SCHEMA:
        raise ValueError(f"{path}: k table schema {table.get('schema')!r}")
    _pid, frame = _product(settings)
    if table.get("frame") != frame:
        raise ValueError(f"{path}: k table frame {table.get('frame')!r}, not {frame!r}")
    return table, f"{path}#sha256:{hashlib.sha256(raw).hexdigest()[:16]}"


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
    k_table: tuple[Mapping[str, Any] | None, str | None] | None = None,
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
        k: The rate scale to use, shape (3,); None takes, in order, the
            station's own (:func:`station_rate_scale`, on the residual of
            its ref, :data:`K_RESIDUAL`), the k(T) table's
            (:func:`load_k_table`), ``settings.default_k``.
        k_method: The ``sigma_method`` a given ``k`` carries.
        k_table: A loaded ``(table, source)``; None loads the settings'.
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
    except Exception:  # gps_parser raises bare Exception for an unknown station
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
    data_end: float | None = None
    if refused is None:
        try:
            t, y, s, attrs = _read(
                sta, t0, t1, settings, terms=_TERMS.get(ref), doc=doc
            )
        except (OSError, ValueError) as exc:  # no .pos: FileNotFoundError
            refused = f"read: {exc}"
    gate_end = None
    if refused is None:
        if attrs.get("outlier_abort"):
            flags.append("outlier_abort")
        data_end = _data_end(sta, t, t1, settings)
        if data_end is not None:
            # the data stop inside the window (the processing lag): cover
            # and T against the data end, not the calendar
            gate_end = data_end + _HALF_DAY
            flags.append(f"truncated@{data_end:.4f}")
        wr = velocity_from_arrays(
            t, y, s, (t0, t1), step_epochs=step_epochs.tolist(),
            allow_steps=settings.allow_steps, min_obs=settings.min_obs,
            min_cover=settings.min_cover, gate_end=gate_end,
        )  # fmt: skip
        short = wr.short and settings.short_tier
        if wr.reasons and not short:
            refused = "; ".join(wr.reasons)
        elif wr.steps_in_window:
            flags.append("steps_in_window")
    # k at the window's length (its data span when truncated), a week at
    # least: below that the sliding k is not measured
    t_len = (min(t1, gate_end) if gate_end is not None else t1) - t0
    k_years = max(t_len, K_MIN_DAYS / 365.25)
    k_source: dict[str, Any] = {}
    residual = K_RESIDUAL[ref]
    if k is not None:
        kk = np.asarray(k, dtype=float).reshape(3)
        sigma_method = k_method or "given_k"
        k_source = {"given": True}
    else:
        own: FloatArray | None = None
        if refused is None and record is not None:
            own, k_source = station_rate_scale(
                sta, k_years, settings, doc=doc, residual=residual
            )
        table, table_source = k_table if k_table is not None else load_k_table(settings)
        from_table = table_k(table, residual, k_years) if table is not None else None
        if own is not None:
            kk, sigma_method = own, "empirical_k"
        elif from_table is not None:
            kk, sigma_method = from_table, "empirical_k_table"
            k_source = {"table": table_source, "residual": residual,
                        "own": k_source.get("reason")}  # fmt: skip
            assert table is not None
            differ = {
                key: table.get(key)
                for key, value in (("view", settings.view), ("uncert", settings.uncert))
                if table.get(key) is not None and table.get(key) != value
            }
            if differ:  # a table built on other settings: usable, but say so
                k_source["table_settings"] = differ
                flags.append("k_table_settings_differ")
        else:
            kk, sigma_method = np.full(3, settings.default_k), "default_k"
            k_source = {"default_k": settings.default_k, "own": k_source.get("reason")}
    from gps_analysis import signal_to_noise

    nan3 = np.full(3, np.nan)
    tier = "standard"
    parts_refused: dict[str, str] = {}
    rate_a, formal_a = nan3.copy(), nan3.copy()
    if wr is not None and refused is None:
        rate_a = np.asarray(wr.rate, dtype=float).copy()
        if wr.reasons:  # the short tier: judged on |v|/σ, part by part
            tier = "short"
            formal_a = np.asarray(wr.sigma_short, dtype=float).copy()
        else:
            formal_a = np.asarray(wr.sigma_formal, dtype=float).copy()
    sigma_a = formal_a * kk
    snr = signal_to_noise(rate_a, sigma_a)
    if tier == "short":
        assert wr is not None
        n = int(np.unique(t[np.all(np.isfinite(y), axis=0)]).size)
        th = settings.short_snr_pair if n < 3 else settings.short_snr
        for part, idx, value in (("horizontal", [0, 1], snr[0]), ("up", [2], snr[1])):
            if not (value >= th):  # NaN fails too
                parts_refused[part] = f"snr={value:.1f}<{th:g}"
                rate_a[idx] = sigma_a[idx] = formal_a[idx] = np.nan
        if len(parts_refused) == 2:
            refused = "; ".join(
                (*wr.reasons, *(f"{p} {r}" for p, r in parts_refused.items()))
            )
            parts_refused = {}
            tier = "standard"
            snr = (np.nan, np.nan)
        else:
            flags.append(f"short:n={n}")
    if refused is None and wr is not None:
        comps = [name for name, f in zip("NEU", wr.f_quadratic, strict=True)
                 if np.isfinite(rate_a["NEU".index(name)]) and f >= NONLINEAR_F]  # fmt: skip
        if comps:
            flags.append(f"nonlinear:{''.join(comps)}")
    else:
        rate_a, formal_a, sigma_a = nan3.copy(), nan3.copy(), nan3.copy()
    rate = tuple(float(v) for v in rate_a)
    formal = tuple(float(v) for v in formal_a)
    sigma = tuple(float(v) for v in sigma_a)
    fitted = wr is not None and refused is None
    fp_extra = {"settings": dataclasses.asdict(settings), "ref": ref,
                "window": [t0, t1], "record": _record_summary(record),
                "k": [float(v) for v in kk], "sigma_method": sigma_method,
                "k_source": k_source, "data_end": data_end}  # fmt: skip
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
        tier=tier,
        snr=(float(snr[0]), float(snr[1])) if fitted else (np.nan, np.nan),
        chi2nu=tuple(float(v) for v in wr.chi2nu) if fitted and wr else (np.nan,) * 3,  # type: ignore[arg-type]
        f_quadratic=(
            tuple(float(v) for v in wr.f_quadratic) if fitted and wr else (np.nan,) * 3  # type: ignore[arg-type]
        ),
        parts_refused=parts_refused,
        data_end=data_end,
    )


#: half a day [yr]: a noon epoch's day ends this much after it
_HALF_DAY = 0.5 / 365.25


def _data_end(
    sta: str, t: FloatArray, end: float, settings: VelocitySettings
) -> float | None:
    """The station's last epoch when the data stop before ``end`` by more
    than a day (the processing lag at the head of the series), else None.
    Reads past the window only when its epochs stop early."""
    last = float(np.nanmax(t)) if np.size(t) and np.isfinite(t).any() else -np.inf
    if last + 1.5 / 365.25 >= end:
        return None
    import geo_dataread.gps_read as gpsr

    pid, _ = _product(settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tt, *_ = gpsr.getData(  # type: ignore[no-untyped-call]
                sta, fstart=end - 1.5 / 365.25 if last == -np.inf else last,
                fend=None, ref="plate", Dir=_pos_dir(settings), tType="POS",
                uncert=settings.uncert, pos_id=pid,
            )  # fmt: skip
        except (OSError, ValueError):
            return None
    later = np.asarray(tt if tt is not None else [], dtype=float)
    later = later[np.isfinite(later) & (later > last)]
    if later.size:  # data go on after the window's last epoch: a gap, not the end
        return None
    return last if np.isfinite(last) else None


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

    Each record depends on its station, the settings and the k(T) table
    named in its ``k_source`` (:func:`build_k_table`), never on which other
    stations are in the run. The table is a file: a host without it falls
    back to ``default_k``, which ``sigma_method`` says.
    """
    settings = settings or VelocitySettings()
    doc = _params_doc(settings)
    table = load_k_table(settings)
    out = []
    for sta in sorted({x.upper() for x in stations}):
        r = estimate_station_velocity(sta, start, end, settings, doc=doc, k_table=table)
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
    "tier", "snr_h", "snr_u", "parts_refused",
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
            if not r.accepted or not np.isfinite(r.rate[:2]).all():
                continue  # refused, or a short-tier record with up alone
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
                r.tier, _fmt(r.snr[0], 1), _fmt(r.snr[1], 1),
                "|".join(f"{k}:{v}" for k, v in sorted(r.parts_refused.items())),
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
    names = [str(s) for s in cfg.getStationInfo() if str(s) != "DEFAULTS"]
    pats = [p.upper() for p in patterns] or ["*"]
    picked = [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in pats)]
    if region is None:
        return sorted(picked)
    w, e, s, n = region
    out = []
    for sta in picked:
        try:
            lon, lat = _coords(sta)
        except Exception:  # an unknown station: not in the region
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
                   help="YYYYMMDD: the last day of --last (default today)")  # fmt: skip
    e.add_argument("--ref", choices=REFS, default="plate-periodic",
                   help="plate-periodic: seasonal removed; background: secular + "
                   "seasonal removed (the transient); plate: nothing more")  # fmt: skip
    e.add_argument("--view", choices=VIEWS, default=DEFAULT_VIEW)
    e.add_argument("--frame", default=DEFAULT_FRAME)
    e.add_argument("-u", "--uncert", type=float, default=DEFAULT_UNCERT)
    e.add_argument("--no-record", choices=NO_RECORD_POLICIES, default="plate",
                   help="a station without a detrend record: fall back to ref "
                   "plate (flagged) or refuse")  # fmt: skip
    e.add_argument("--allow-steps", action="store_true",
                   help="keep a window with a declared step inside (flagged)")  # fmt: skip
    e.add_argument("--min-obs", type=int, default=VelocitySettings.min_obs)
    e.add_argument("--min-cover", type=float, default=VelocitySettings.min_cover)
    e.add_argument(
        "--no-short-tier",
        dest="short_tier",
        action="store_false",
        help="refuse every window failing the count/cover gates "
        "(default: keep the parts whose |v|/sigma reaches --short-snr)",
    )
    e.add_argument("--short-snr", type=float, default=VelocitySettings.short_snr)
    e.add_argument("--short-snr-pair", type=float,
                   default=VelocitySettings.short_snr_pair,
                   help="the short tier's |v|/sigma on 2 epochs")  # fmt: skip
    e.add_argument("--default-k", type=float, default=DEFAULT_K)
    e.add_argument(
        "--k-table", default=None,
        help="k(T) table for stations without a quiet span of their own "
        "(default: k_table_<frame>.json in the velocity dir when built; "
        "'none' = default-k)",
    )  # fmt: skip
    e.add_argument("--pos-dir", default=None)
    e.add_argument("--params", default=None, help="detrend parameter document")
    e.add_argument("--steps", default=None, help="steps.yaml")
    e.add_argument("--out", default=None,
                   help="directory (default: velocity_dir of postprocess.cfg, "
                   f"else {DEFAULT_VELOCITY_DIR})")  # fmt: skip
    e.add_argument(
        "--name", default=None, help="set name (default from window/ref/frame)"
    )
    t = sub.add_parser(
        "k-table",
        help="build the k(T) table: the median empirical rate scale over every "
        "station with a detrend record, the fallback of stations without one",
    )
    t.add_argument("stations", nargs="*", help="markers or patterns (default all)")
    t.add_argument("--frame", default=DEFAULT_FRAME)
    t.add_argument("--view", choices=VIEWS, default=DEFAULT_VIEW)
    t.add_argument("-u", "--uncert", type=float, default=DEFAULT_UNCERT)
    t.add_argument("--pos-dir", default=None)
    t.add_argument("--params", default=None)
    t.add_argument("--out", default=None, help="file (default k_table_<frame>.json "
                   "in the velocity dir)")  # fmt: skip
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


def _k_table_main(args: argparse.Namespace) -> int:
    settings = VelocitySettings(frame=args.frame, view=args.view,
                                uncert=args.uncert, pos_dir=args.pos_dir,
                                params=args.params)  # fmt: skip
    stations = select_stations(args.stations, None)
    table = build_k_table(
        stations, settings, progress=lambda sta: print(sta, file=sys.stderr, end=" ")
    )
    out = Path(args.out).expanduser() if args.out else k_table_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{len(table['stations'])} stations -> {out}")
    for residual, entry in table["residuals"].items():
        for d, k, n in zip(entry["days"], entry["k"], entry["n_stations"], strict=True):
            print(f"  {residual:8} {d:4d} d  k N/E/U {k[0]:5.2f} {k[1]:5.2f} {k[2]:5.2f}"
                  f"  ({n} stations)")  # fmt: skip
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "show":
        _show(read_velocity_records(args.path), args.all)
        return 0
    if args.cmd == "k-table":
        return _k_table_main(args)
    if (args.window is None) == (args.last is None):
        print("error: give --window START:END or --last DAYS", file=sys.stderr)
        return 2
    if args.window is not None:
        start, end = args.window
    else:
        end = args.end or dt.date.today()
        start = last_days_start(end, args.last)
    try:
        settings = VelocitySettings(
            frame=args.frame, ref=args.ref, view=args.view, uncert=args.uncert,
            no_record=args.no_record, allow_steps=args.allow_steps,
            min_obs=args.min_obs, min_cover=args.min_cover,
            short_tier=args.short_tier, short_snr=args.short_snr,
            short_snr_pair=args.short_snr_pair,
            default_k=args.default_k, pos_dir=args.pos_dir, params=args.params,
            steps=args.steps, k_table=args.k_table,
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
    "COMPONENTS", "DEFAULT_K", "K_RESIDUAL", "K_TABLE_DAYS", "REFS",
    "SCHEMA_VERSION", "VIEWS", "VelocityRecord", "VelocitySettings",
    "WindowRate", "build_k_table", "estimate_station_velocity",
    "estimate_velocities", "k_table_path", "load_k_table",
    "read_velocity_records", "select_stations", "set_name",
    "station_rate_scale", "station_rate_scales", "table_k", "velocity_dir",
    "velocity_from_arrays", "window_bounds", "write_velocity_set", "yearf",
]  # fmt: skip
