"""The transient velocity product: the rate on a date from a fitted recharge curve.

After an event (an eruption that drained a reservoir) the stations move on a
curve, fast at first and slowing towards a steady rate. A straight line over
a long window misfits it; a short window is noisy. This product fits every
station and component, from the event on, with

    x(t) = offset + v·t + Φ·(1 − e^(−(t − t₀)/τ))

(:class:`gps_analysis.terms.ExpTransient`; ``models.exp_linear`` with
``A = −Φ``, ``k = 1/τ``) and reports the derivative — the velocity — on
chosen dates (``at``), plus v, the steady rate the curve approaches.

Two stages, the k(T)-table pattern of the velocity product:

1. **Decay** — one τ shared by the stations of a run (one source), estimated
   jointly (:func:`gps_analysis.profile_shared_tau`) and written to a small
   decay file (τ, σ_τ, its interval, t₀, the stations, the settings). Or a
   fixed τ, or an earlier decay file.
2. **Station** — each station fitted with τ fixed
   (:func:`gps_analysis.transient_rate`); σ_τ is carried into the rate's σ.
   A station's record depends on its own data and the decay (named in
   ``k_source``), never on which other stations ran with it.

σ is formal × √(χ²/ν) (``sigma_method`` ``formal_chi2``, flag
``sigma:formal``): no coloured-noise scale is measured for curves yet, so it
is smaller than the velocity product's empirical σ.

The library of gpsprod ``transient`` (docs/gpsprod.md).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from geo_dataread.products.velocity_core import (
    _TERMS,
    FloatArray,
    VelocityRecord,
    VelocitySettings,
    _coords,
    _data_end,
    _fingerprint,
    _fingerprint_settings,
    _params_doc,
    _product,
    _read,
    _record,
    _record_summary,
    _software,
)

DECAY_SCHEMA = 1
#: τ search box for the shared fit [yr] (half-life 2.5 d .. 4 yr)
TAU_BOUNDS = (0.01, 6.0)
#: residuals beyond this many robust σ (MAD of the normalised residual) are
#: dropped, per component, before the decay is re-estimated
REJECT_N_SIGMA = 4.0
REJECT_ROUNDS = 3
#: a station needs this much data after t₀ [days] and this many epochs
MIN_SPAN_DAYS = 90
MIN_EPOCHS = 30
#: a station's own τ is checked when its largest transient has |Φ|/σ_Φ ≥
#: this; flagged when its θ = ln τ is > TAU_DIFFER σ from the shared one
OWN_TAU_SNR = 10.0
TAU_DIFFER = 5.0
#: the initial τ for the first outlier pass [yr] (≈ 2 months half-life)
TAU_START = 0.25
_HALF_DAY = 0.5 / 365.25


def half_life_days(tau: float) -> float:
    """The half-life of e^(−t/τ) in days."""
    return math.log(2.0) * tau * 365.25


def tau_from_half_life(days: float) -> float:
    """τ [yr] of a half-life in days."""
    return days / 365.25 / math.log(2.0)


@dataclasses.dataclass(frozen=True)
class Decay:
    """One τ for a run: estimated jointly, fixed by hand, or read back.

    ``source`` is "shared" (estimated from ``stations``), "fixed" (given) or
    a decay file's path; ``fingerprint`` names the content.
    """

    t0: float
    tau: float
    tau_sigma: float
    tau_interval: tuple[float, float] | None = None
    interval_open: tuple[bool, bool] = (False, False)
    stations: tuple[str, ...] = ()
    n_obs: int = 0
    chi2nu: float = math.nan
    end: float | None = None
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)
    source: str = "fixed"
    schema: int = DECAY_SCHEMA

    @property
    def half_life_days(self) -> float:
        return half_life_days(self.tau)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["half_life_days"] = round(self.half_life_days, 3)
        d["decay_rate_per_yr"] = 1.0 / self.tau
        d.pop("source")
        return d

    @property
    def fingerprint(self) -> str:
        body = json.dumps(self.to_dict(), sort_keys=True, default=float)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def k_source(self) -> dict[str, Any]:
        return {"decay": self.source, "decay_fingerprint": self.fingerprint,
                "tau": self.tau, "tau_sigma": self.tau_sigma}  # fmt: skip


def write_decay(decay: Decay, path: str | Path) -> Path:
    """Write a decay file (JSON, byte-stable, no timestamps)."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(decay.to_dict(), indent=1, sort_keys=True) + "\n")
    return p


def read_decay(path: str | Path) -> Decay:
    """Read a decay file back (``source`` = the path).

    Raises:
        ValueError: an unknown schema.
    """
    p = Path(path).expanduser()
    d = json.loads(p.read_text())
    if d.get("schema") != DECAY_SCHEMA:
        raise ValueError(f"{p}: decay schema {d.get('schema')!r}, know {DECAY_SCHEMA}")
    names = {f.name for f in dataclasses.fields(Decay)}
    kw = {k: v for k, v in d.items() if k in names}
    for key in ("tau_interval", "interval_open", "stations"):
        if kw.get(key) is not None:
            kw[key] = tuple(kw[key])
    return Decay(**kw, source=str(p))


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _StationData:
    """One station's series after t₀, and why it may not be fitted."""

    sta: str
    t: FloatArray
    y: FloatArray  # (3, N), NaN = not used
    s: FloatArray
    ref: str
    flags: list[str]
    refused: str | None
    record: Mapping[str, Any] | None
    steps: tuple[float, ...]
    data_end: float | None
    attrs: dict[str, Any]
    lonlat: tuple[float, float]
    n_rejected: int = 0
    #: the series as read, so the final rejection starts from it
    y_read: FloatArray | None = None


def _read_station(
    sta: str, t0: float, t1: float, settings: VelocitySettings, doc: Any
) -> _StationData:
    """The velocity product's read of one station on [t₀, t₁) (same ref,
    view, record fallback and step catalogue), gated for a curve fit."""
    from geo_dataread.gps_views import station_step_epochs

    sta = sta.upper()
    record = _record(doc, sta)
    flags: list[str] = []
    ref, refused = settings.ref, None
    if ref == "pre-onset":
        raise ValueError("the transient product takes no ref pre-onset")
    if ref in _TERMS and record is None:
        if settings.no_record == "plate":
            ref = "plate"
            flags.append("fallback:plate")
        else:
            refused = "no_record"
    regional = record is not None and bool(record.get("seasonal_only"))
    if regional and ref in _TERMS:
        if _TERMS[ref] != "periodic":
            refused = f"seasonal_only: {ref} needs a background of the station's own"
        else:
            flags.append("seasonal:regional")
    try:
        lonlat = _coords(sta)
    except Exception:  # gps_parser raises bare Exception for an unknown station
        lonlat = (math.nan, math.nan)
        flags.append("no_coordinates")
    try:
        step_epochs, _src = station_step_epochs(sta, steps=settings.steps)
    except (FileNotFoundError, ValueError):
        step_epochs = np.zeros(0)
    steps = tuple(float(e) for e in step_epochs if t0 < e < t1)
    empty = np.zeros((3, 0))
    attrs: dict[str, Any] = {}
    t, y, s = np.zeros(0), empty, empty
    if refused is None:
        try:
            t, y, s, attrs = _read(
                sta, t0, t1, settings, terms=_TERMS.get(ref), doc=doc
            )
        except (OSError, ValueError) as exc:
            refused = f"read: {exc}"
    data_end = None
    if refused is None:
        ok = np.all(np.isfinite(y), axis=0) & np.all(np.isfinite(s), axis=0)
        n = int(ok.sum())
        span = (t[ok].max() - t[ok].min()) * 365.25 if n else 0.0
        if n < MIN_EPOCHS:
            refused = f"too few epochs after t0 ({n} < {MIN_EPOCHS})"
        elif span < MIN_SPAN_DAYS:
            refused = f"too short after t0 ({span:.0f} d < {MIN_SPAN_DAYS} d)"
        elif steps and not settings.allow_steps:
            refused = "steps_in_window: " + ", ".join(f"{e:.4f}" for e in steps)
        else:
            if steps:
                flags.append("steps_in_window")
            # recorded, not flagged: the processing lag makes every series
            # end early; a rate dated past the data says extrapolated:Nd
            data_end = _data_end(sta, t, t1, settings)
    return _StationData(sta, t, y, s, ref, flags, refused, record, steps,
                        data_end, attrs, lonlat, y_read=y.copy())  # fmt: skip


def _model(t0: float, tau: float, steps: Sequence[float] = ()) -> Any:
    from gps_analysis import ExpTransient, Polynomial, Step, TrajectoryModel

    terms: list[Any] = [Polynomial(1), ExpTransient(epoch=t0, tau=tau)]
    terms += [Step(epoch=e) for e in steps]
    return TrajectoryModel(terms)


def _amp_index(model: Any) -> int:
    """The exponential amplitude's index (TrajectoryModel numbers it)."""
    names = list(model.param_names)
    return next(i for i, n in enumerate(names) if n.startswith("exp_amp"))


def _series(d: _StationData) -> list[tuple[FloatArray, FloatArray, FloatArray]]:
    return [(d.t, d.y[c], d.s[c]) for c in range(3)]


def _reject(d: _StationData, t0: float, tau: float) -> int:
    """NaN the residuals beyond REJECT_N_SIGMA robust σ at τ; count them."""
    from gps_analysis import transient_rate

    dropped = 0
    m = _model(t0, tau, d.steps)
    for c in range(3):
        ok = np.isfinite(d.y[c]) & np.isfinite(d.s[c])
        fit = transient_rate(m, d.t, d.y[c], d.s[c], [d.t[ok][-1]], tau=tau)
        r = (d.y[c] - fit.model.design(d.t) @ fit.params) / d.s[c]
        med = np.nanmedian(r[ok])
        mad = 1.4826 * np.nanmedian(np.abs(r[ok] - med))
        bad = ok & (np.abs(r - med) > REJECT_N_SIGMA * mad)
        d.y[c, bad] = np.nan
        dropped += int(bad.sum())
    d.n_rejected += dropped
    return dropped


def _reject_at(d: _StationData, t0: float, tau: float) -> None:
    """Outlier rounds at a fixed τ, starting from the series as read."""
    assert d.y_read is not None
    d.y = d.y_read.copy()
    d.n_rejected = 0
    for _ in range(REJECT_ROUNDS):
        if _reject(d, t0, tau) == 0:
            break


# ---------------------------------------------------------------------------
# stage 1: the decay
# ---------------------------------------------------------------------------


def _shared(data: Sequence[_StationData], t0: float) -> Any:
    from gps_analysis import profile_shared_tau

    # one model per step pattern is not possible in one profile; a station
    # with steps joins with its steps as extra terms via its own series
    # list, so the steps-free model is used and stepped stations sit out
    series = [s for d in data if not d.steps for s in _series(d)]
    if not series:
        raise ValueError("no station without steps to estimate the decay from")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return profile_shared_tau(_model(t0, TAU_START), series, tau_bounds=TAU_BOUNDS)


def estimate_decay(
    data: Sequence[_StationData],
    t0: float,
    t1: float,
    settings: VelocitySettings,
) -> Decay:
    """τ shared by the usable stations, with outlier rounds in between."""
    usable = [d for d in data if d.refused is None]
    if not usable:
        raise ValueError("no usable station to estimate the decay from")
    tau = TAU_START
    fit = None
    for _ in range(REJECT_ROUNDS):
        dropped = sum(_reject(d, t0, tau) for d in usable)
        fit = _shared(usable, t0)
        tau = fit.tau
        if dropped == 0:
            break
    assert fit is not None
    return Decay(
        t0=t0, tau=fit.tau, tau_sigma=fit.tau_sigma,
        tau_interval=fit.tau_interval,
        interval_open=(fit.interval_open_lower, fit.interval_open_upper),
        stations=tuple(sorted(d.sta for d in usable if not d.steps)),
        n_obs=fit.n_obs, chi2nu=fit.scale_sq, end=t1,
        settings=_fingerprint_settings(settings), source="shared",
    )  # fmt: skip


# ---------------------------------------------------------------------------
# stage 2: one station
# ---------------------------------------------------------------------------


def _own_tau_flag(d: _StationData, t0: float, decay: Decay) -> str | None:
    """``tau_differs:<half-life>d`` when the station's own τ is well measured
    and far from the shared one."""
    from gps_analysis import profile_shared_tau, transient_rate

    m = _model(t0, decay.tau, d.steps)
    j = _amp_index(m)
    snr = 0.0
    for c in range(3):
        f = transient_rate(m, d.t, d.y[c], d.s[c], [t0 + 1.0], tau=decay.tau)
        sd = math.sqrt(f.covariance[j, j])
        snr = max(snr, abs(f.params[j]) / sd if sd > 0 else 0.0)
    if snr < OWN_TAU_SNR:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        own = profile_shared_tau(m, _series(d), tau_bounds=TAU_BOUNDS)
    dth = abs(math.log(own.tau) - math.log(decay.tau))
    sth = math.hypot(own.theta_sigma, decay.tau_sigma / decay.tau)
    if dth > TAU_DIFFER * sth:
        return f"tau_differs:{half_life_days(own.tau):.0f}d"
    return None


def station_transient(
    d: _StationData,
    t0: float,
    t1: float,
    at: Sequence[float],
    decay: Decay,
    settings: VelocitySettings,
) -> list[VelocityRecord]:
    """One record per ``at`` epoch for one read station."""
    from gps_analysis import signal_to_noise, transient_rate

    pid, frame = _product(settings)
    flags = list(d.flags)
    refused = d.refused
    nan3 = (math.nan,) * 3
    rates = np.full((len(at), 3), np.nan)
    sig = np.full((len(at), 3), np.nan)
    sig_fixed = np.full((len(at), 3), np.nan)
    chi2nu = [math.nan] * 3
    n_epochs = (0, 0, 0)
    span = None
    transient: dict[str, Any] | None = None
    if refused is None:
        own = _own_tau_flag(d, t0, decay)
        if own:
            flags.append(own)
        m = _model(t0, decay.tau, d.steps)
        names = list(m.param_names)
        comp: dict[str, Any] = {"v": [], "sigma_v": [], "phi": [], "A": []}
        counts = []
        for c in range(3):
            f = transient_rate(m, d.t, d.y[c], d.s[c], at, tau=decay.tau,
                               tau_sigma=decay.tau_sigma)  # fmt: skip
            rates[:, c], sig[:, c], sig_fixed[:, c] = f.rate, f.sigma, f.sigma_fixed_tau
            chi2nu[c] = f.chi2nu
            counts.append(f.n_obs)
            iv, ia = names.index("rate"), _amp_index(m)
            comp["v"].append(float(f.params[iv]))
            comp["sigma_v"].append(float(math.sqrt(f.covariance[iv, iv])))
            comp["phi"].append(float(f.params[ia]))
            comp["A"].append(-float(f.params[ia]))
        n_epochs = (counts[0], counts[1], counts[2])
        ok = np.all(np.isfinite(d.y), axis=0)
        span = (float(d.t[ok].min()), float(d.t[ok].max()))
        transient = {
            "t0": t0, "tau": decay.tau, "tau_sigma": decay.tau_sigma,
            "half_life_days": decay.half_life_days, "decay": decay.source,
            "components": ("north", "east", "up"), **comp,
            "steps": list(d.steps), "n_rejected": d.n_rejected,
        }  # fmt: skip
        flags.append("sigma:formal")
        if d.n_rejected:
            flags.append(f"rejected:{d.n_rejected}")
    out = []
    for i, a in enumerate(at):
        f_i = list(flags)
        last = span[1] if span else None
        if refused is None and last is not None and a > last + 1.0 / 365.25:
            f_i.append(f"extrapolated:{(a - last) * 365.25:.0f}d")
        fp_extra = {"settings": _fingerprint_settings(settings), "ref": d.ref,
                    "window": [t0, t1], "at": a, "decay": decay.fingerprint,
                    "record": _record_summary(d.record)}  # fmt: skip
        fitted = refused is None
        rate = tuple(float(v) for v in rates[i]) if fitted else nan3
        sigma = tuple(float(v) for v in sig[i]) if fitted else nan3
        snr = (
            signal_to_noise(np.asarray(rate), np.asarray(sigma))
            if fitted
            else (math.nan,) * 2
        )
        out.append(VelocityRecord(
            station=d.sta, lon=float(d.lonlat[0]), lat=float(d.lonlat[1]),
            frame=frame, plate=_plate(d.sta), ref=d.ref, ref_requested=settings.ref,
            series=pid, view=settings.view, uncert=float(settings.uncert),
            window=(t0, t1), span_used=span, n_epochs=n_epochs,
            largest_gap=_largest_gap(d) if fitted else None,
            method="wls", model="exp_linear",
            rate=rate,  # type: ignore[arg-type]
            sigma=sigma,  # type: ignore[arg-type]
            sigma_formal=tuple(float(v) for v in sig_fixed[i]) if fitted else nan3,  # type: ignore[arg-type]
            sigma_method="formal_chi2", k=(1.0, 1.0, 1.0),
            k_source=decay.k_source(), corr_en=0.0 if fitted else math.nan,
            record=_record_summary(d.record) if d.ref in _TERMS else None,
            steps_in_window=d.steps, outlier_abort=bool(d.attrs.get("outlier_abort")),
            n_flagged=int(d.attrs.get("n_flagged") or 0), flags=tuple(f_i),
            refused=refused, software=_software(),
            input_fingerprint=_fingerprint(d.t, d.y, d.s, fp_extra),
            session=settings.session, kind="transient",
            snr=(float(snr[0]), float(snr[1])),
            chi2nu=tuple(chi2nu) if fitted else nan3,  # type: ignore[arg-type]
            data_end=d.data_end, at=a, transient=transient,
        ))  # fmt: skip
    return out


def _plate(sta: str) -> str | None:
    from geo_dataread.station_catalog import station_plate

    return station_plate(sta)


def _largest_gap(d: _StationData) -> float | None:
    ok = np.all(np.isfinite(d.y), axis=0)
    tt = d.t[ok]
    return float(np.max(np.diff(tt))) if tt.size > 1 else None


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def estimate_transients(
    stations: Sequence[str],
    t0: float,
    t1: float,
    at: Sequence[float] | None = None,
    settings: VelocitySettings | None = None,
    *,
    decay: Decay | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[VelocityRecord], Decay]:
    """Every station's rate on each ``at`` date, and the decay used.

    Args:
        stations: Station markers.
        t0, t1: The fit window [yr]: t₀ is the transient's onset (the day
            after the event), t₁ its end.
        at: Epochs [yr] to give the rate on; None: the last epoch any
            usable station has (the newest data, not the calendar end —
            the processing lag would make every rate an extrapolation).
        settings: Read settings (frame, ref, view, σ screen, steps, …);
            ref pre-onset is refused.
        decay: A fixed τ (:class:`Decay`, e.g. :func:`read_decay`); None
            estimates one shared by the usable stations.
        progress: Called with each station as it is read.

    Returns:
        (records sorted by station then ``at``, the decay).

    Raises:
        ValueError: bad window, no usable station for a shared decay, or a
            decay whose t₀ differs from ``t0``.
    """
    settings = settings or VelocitySettings()
    if not t1 > t0:
        raise ValueError(f"end {t1} is not after t0 {t0}")
    if at is not None and not len(at):
        raise ValueError("no epoch to give the rate on")
    if decay is not None and abs(decay.t0 - t0) > _HALF_DAY:
        raise ValueError(f"the decay's t0 {decay.t0:.4f} is not this run's {t0:.4f}")
    doc = _params_doc(settings)
    data = []
    for sta in sorted({s.upper() for s in stations}):
        if progress is not None:
            progress(sta)
        data.append(_read_station(sta, t0, t1, settings, doc))
    if at is None:
        ends = [float(d.t[np.all(np.isfinite(d.y), axis=0)].max())
                for d in data if d.refused is None]  # fmt: skip
        if not ends:
            raise ValueError("no usable station: no data to date the rate by")
        at = [max(ends)]
    if decay is None:
        decay = estimate_decay(data, t0, t1, settings)
    for d in data:
        # from the series as read, at the decay's τ: the same epochs go
        # whether τ was estimated in this run or given, so a station's
        # record depends on its data and the decay only
        if d.refused is None:
            _reject_at(d, t0, decay.tau)
    records = [
        r for d in data for r in station_transient(d, t0, t1, at, decay, settings)
    ]
    return sorted(records, key=lambda r: (r.station, r.at or 0.0)), decay
