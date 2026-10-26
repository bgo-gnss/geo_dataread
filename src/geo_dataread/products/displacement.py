"""The displacement product: how far a station moved across a period.

Method ``ends`` (the default): the level at each end of the period is the
weighted mean of the series in a window of ``ends_days`` whole days inside
the period (``[P0, P0 + N d)`` and ``[P1 − N d, P1)``), and the
displacement is the difference. σ of a level is the largest of the formal
σ scaled by the station's daily factor, the in-window scatter, and the
station's day-to-day scatter averaged down (:mod:`gps_analysis.displacement`);
both factors are measured on the station's own series around the period
(on the differenced series for ``relative_to``). With ``ends_days ≤ 2`` each
end is also tested against the line through the next days
(:func:`gps_analysis.neighbour_check`) and flagged ``end_disputed`` when
it disagrees.

The end window stops at the station's last epoch when the data stop before
the period does (the processing lag), flagged ``end_early`` past 1.5 days.
A declared step inside an end window refuses the station; one inside the
period is the point of a displacement and only flagged.

Method ``points``: the last epoch of the period minus the first (all three
components present), no averaging — for fast motion, where a mean over
days moves the effective epoch. The epochs used are in ``ends``; when they
are not the period's first/last day the record says ``start_at``/``end_at``.
σ as an end level of one epoch; each end is checked as ``ends_days ≤ 2``.

The end check (``ends_days ≤ 2`` and ``points``) fits its line through the
``neighbour_days`` OUTSIDE the period — before the start, after the end —
and only falls back to the days inside when outside is too thin, so it
never draws a line across the motion being measured (or a declared step).

Method ``step``: the jump across one event. A line is fitted to the
``step_window_days`` before the event's onset and another to the days after
its last day (the event's own days are in neither), each evaluated at the
event's edge: ``Δ = f_after(last day + 1) − f_before(onset)``, so a
multi-day eruption's displacement includes the motion during it. The level
σ of each line is scaled like an end level, its rate σ by k(T) of the
window (the velocity product's: the station's own, else the k(T) table,
else 5). The event is ``--event`` (an events.yaml id or part of one),
``--at DATE`` (a one-day event), or the station's one dike/eruption inside
the period.

Method ``rate``: the velocity product's rate over the period
(:func:`geo_dataread.velocity.estimate_station_velocity`) times its length
— velocity → displacement, valid for steady motion (its nonlinear flags
are kept). :func:`to_displacement` and :func:`to_velocity` are the two
conversions on records.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from gps_analysis.displacement import (
    displacement_from_rate,
    end_displacement,
    end_level,
    neighbour_check,
    rate_from_displacement,
    step_displacement,
)
from gps_analysis.empirical_sigma import daily_scatter, sigma_scale_factor

from geo_dataread.products.common import (
    COMPONENTS,
    DisplacementRecord,
    RecordSet,
    jsonable,
    software,
)
from geo_dataread.products.loader import DAY_YEARS, Source, StationView, load

if TYPE_CHECKING:
    from geo_dataread.events import Event
    from geo_dataread.velocity import VelocityRecord, VelocitySettings

METHODS = ("ends", "points", "step", "rate")
#: days read beyond the period on each side: the daily σ factors and the
#: cleaned view's detector see context, not just the period
PAD_DAYS = 60
#: the data lag at the head of a series that is not worth a flag [days]
END_LAG_DAYS = 1.5
SIGMA_METHOD = "max(k_day*formal, formal*sqrt(chi2nu), s_day/sqrt(n))"
STEP_SIGMA = "step: level max(k_day, sqrt(chi2nu))*formal, rate k(T)*formal"
#: the velocity product's k(T) fallback and shortest measured window
DEFAULT_K = 5.0
K_MIN_DAYS = 7


@dataclasses.dataclass(frozen=True)
class DisplacementSettings:
    """How a set of displacements is estimated."""

    source: Source = Source()
    method: str = "ends"
    ends_days: int = 5
    neighbour_days: int = 5
    #: step: days of line on each side of the event
    step_window_days: int = 30
    #: step: the event (an events.yaml id or part of one); None: the
    #: station's one dike/eruption inside the period
    event: str | None = None
    #: step: a one-day event on this date (YYYYMMDD), instead of ``event``
    at: str | None = None
    #: step: the gates of each side's line (as the velocity product's)
    min_obs: int = 5
    min_cover: float = 0.8

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.ends_days < 1:
            raise ValueError(f"ends_days must be ≥ 1, got {self.ends_days}")
        if self.neighbour_days < 2:
            raise ValueError(f"neighbour_days must be ≥ 2, got {self.neighbour_days}")
        if self.step_window_days < 2:
            raise ValueError(
                f"step_window_days must be ≥ 2, got {self.step_window_days}"
            )
        if self.event is not None and self.at is not None:
            raise ValueError("give the step's event or its date (at), not both")


def _fingerprint(view: StationView, extra: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for a in (view.t, view.y, view.sigma):
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    h.update(json.dumps(jsonable(dict(extra)), sort_keys=True).encode())
    return h.hexdigest()


def _daily_factors(view: StationView, step: float) -> tuple[list[float], list[float]]:
    """(k_day, s_day) per component on the loaded series; NaN where too few
    consecutive pairs (:func:`level_sigma` then skips the term)."""
    if view.t.size < 2:
        return [math.nan] * 3, [math.nan] * 3
    k = sigma_scale_factor(view.t, view.y, view.sigma, step=step)
    s = daily_scatter(view.t, view.y, step=step)
    return [float(v) for v in k], [float(v) for v in s]


def _iso(t: float) -> str:
    from gtimes.timefunc import TimefromYearf

    # a midnight as a year fraction can round to 23:59:59 the day before
    return str(TimefromYearf(t + 1e-6, "%Y-%m-%d"))


def displacement_from_view(
    view: StationView,
    period: tuple[float, float],
    settings: DisplacementSettings,
    *,
    step_epochs: Sequence[float] = (),
) -> tuple[dict[str, Any], list[str], str | None]:
    """The ``ends`` displacement of a loaded series: ``(values, flags, refused)``.

    ``values`` holds ``displacement``/``sigma`` (N, E, U) and the ``ends``
    provenance. Pure apart from the series: testable on synthetic views.
    """
    p0, p1 = period
    step = settings.source.session_years
    n_days = settings.ends_days
    flags: list[str] = []
    k_day, s_day = _daily_factors(view, step)
    finite = view.t[np.isfinite(view.y).any(axis=0)]
    in_period = finite[(finite >= p0) & (finite < p1)]
    if in_period.size == 0:
        before = finite[finite < p0]
        hint = f" (last data {_iso(float(before[-1]))})" if before.size else ""
        return {}, flags, f"no epochs in the period{hint}"
    last = float(in_period[-1])
    # the end window closes at the last epoch: the newest days of a period
    # running to today are not there yet (processing lag, flagged past
    # END_LAG_DAYS), and an empty tail must not shrink the window
    end_edge = min(p1, last + 0.5 * step)
    if p1 - end_edge > (END_LAG_DAYS + 0.5) * DAY_YEARS:
        flags.append(f"end_early:{(p1 - end_edge) / DAY_YEARS:.0f}d")
    if settings.method == "points":
        # the first and the last epoch of the period with all three
        # components: end point minus start point, no averaging
        whole = view.t[np.isfinite(view.y).all(axis=0)]
        whole = whole[(whole >= p0) & (whole < end_edge)]
        if whole.size < 2:
            return {}, flags, "points: fewer than two epochs in the period"
        t_s, t_e = float(whole[0]), float(whole[-1])
        half = 0.5 * step
        w_start = (t_s - half, t_s + half)
        w_end = (t_e - half, t_e + half)
        n_days = 1
        if t_s - p0 > step:
            flags.append(f"start_at:{_iso(t_s)}")
        if end_edge - t_e > step:
            flags.append(f"end_at:{_iso(t_e)}")
    else:
        w_start = (p0, p0 + n_days * DAY_YEARS)
        w_end = (end_edge - n_days * DAY_YEARS, end_edge)
        if w_end[0] < w_start[1]:
            return {}, flags, f"period shorter than two {n_days}-day end windows"
    # a step inside an end window matters only when the window has epochs on
    # both sides of it (its mean would straddle the step): a one-day window
    # holds one solution, before or after
    for e in sorted(float(x) for x in step_epochs):
        for lo, hi in (w_start, w_end):
            inside = finite[(finite >= lo) & (finite < hi)]
            if (inside < e).any() and (inside > e).any():
                return {}, flags, f"step_in_end_window@{_iso(e)}"
    if any(p0 < float(e) < p1 for e in step_epochs):
        flags.append("steps_in_period")
    for key, (lo, hi) in (("start", w_start), ("end", w_end)):
        if not ((finite >= lo) & (finite < hi)).any():
            if key == "start":
                near = finite[finite >= lo]
                hint = (
                    f"; first data after it {_iso(float(near[0]))}" if near.size else ""
                )
            else:
                near = finite[finite < hi]
                hint = (
                    f"; last data before it {_iso(float(near[-1]))}"
                    if near.size
                    else ""
                )
            return (
                {},
                flags,
                (
                    f"no epochs in the {key} window "
                    f"{_iso(lo)}..{_iso(hi - 0.5 * DAY_YEARS)}{hint}"
                ),
            )
    disp, sig = [], []
    ends: dict[str, Any] = {
        "days": n_days if settings.method == "ends" else 0,
        "window_start": list(w_start),
        "window_end": list(w_end),
        "k_day": k_day,
        "s_day": s_day,
        "start": {},
        "end": {},
    }
    for i, comp in enumerate(COMPONENTS):
        try:
            a = end_level(
                view.t, view.y[i], view.sigma[i],
                window=w_start, k_day=k_day[i], s_day=s_day[i],
            )  # fmt: skip
        except ValueError:
            return {}, flags, f"no {comp} epochs in the start window"
        try:
            b = end_level(
                view.t, view.y[i], view.sigma[i],
                window=w_end, k_day=k_day[i], s_day=s_day[i],
            )  # fmt: skip
        except ValueError:
            return {}, flags, f"no {comp} epochs in the end window"
        d, s = end_displacement(a, b)
        disp.append(d)
        sig.append(s)
        for key, lv in (("start", a), ("end", b)):
            ends[key][comp] = {
                "level": lv.level, "sigma": lv.sigma, "n": lv.n,
                "t_mean": lv.t_mean, "chi2nu": lv.chi2nu,
            }  # fmt: skip
        if n_days <= 2:
            m = settings.neighbour_days * DAY_YEARS
            # the days OUTSIDE the period first (before the start, after the
            # end): a line through the days inside would cross the very
            # motion being measured; inside only when outside is too thin
            for key, lv, sides in (
                (
                    "start",
                    a,
                    ((w_start[0] - m, w_start[0]), (w_start[1], w_start[1] + m)),
                ),
                ("end", b, ((w_end[1], w_end[1] + m), (w_end[0] - m, w_end[0]))),
            ):
                chk = None
                for nb in sides:
                    if any(nb[0] <= float(e) < nb[1] for e in step_epochs):
                        continue  # a declared step in the neighbour days
                    chk = neighbour_check(
                        view.t, view.y[i], view.sigma[i],
                        end=lv, neighbour=nb, k_day=k_day[i],
                    )  # fmt: skip
                    if chk.disputed is not None:
                        break
                if chk is not None and chk.disputed:
                    flags.append(f"end_disputed:{key}:{comp[0].upper()}")
                elif chk is None or chk.disputed is None:
                    flags.append(f"end_unchecked:{key}:{comp[0].upper()}")
    values = {"displacement": tuple(disp), "sigma": tuple(sig), "ends": ends}
    return values, flags, None


def _velocity_settings(src: Source, ref: str | None = None) -> VelocitySettings:
    from geo_dataread.velocity import VelocitySettings

    return VelocitySettings(
        frame=src.frame, ref=ref or "plate-periodic", view=src.view,
        uncert=src.uncert, pos_dir=src.pos_dir, params=src.params,
        steps=src.steps,
    )  # fmt: skip


def rate_scale(
    sta: str,
    years: Sequence[float],
    src: Source,
    ref: str,
    *,
    doc: Mapping[str, Any] | None,
) -> tuple[list[np.ndarray], str, dict[str, Any], list[str]]:
    """k(T) for each window length of ``years`` (≥ a week), the velocity
    product's order: the station's own on its quiet span, else the k(T)
    table, else :data:`DEFAULT_K`. ``(k per length, sigma method, source,
    flags)``. A baseline (``relative_to``) or an 8 h series has no measured
    k of its own: the table (24 h, single station: conservative) or 5."""
    from geo_dataread.velocity import (
        K_RESIDUAL,
        load_k_table,
        station_rate_scales,
        table_k,
    )

    yrs = [max(float(T), K_MIN_DAYS / 365.25) for T in years]
    vs = _velocity_settings(src)
    residual = K_RESIDUAL.get(ref, "secular")
    flags: list[str] = []
    own: dict[float, np.ndarray] = {}
    info: dict[str, Any] = {}
    if src.relative_to:
        flags.append("k:baseline_unmeasured")
    elif src.session != "24h":
        flags.append("k:session_unmeasured")
    elif station_record_exists(sta, doc):
        own, info = station_rate_scales(sta, yrs, vs, doc=doc, residual=residual)
    if own and all(T in own for T in yrs):
        return [own[T] for T in yrs], "empirical_k", info, flags
    try:
        table, table_source = load_k_table(vs)
    except (OSError, ValueError):
        table, table_source = None, None
    if table is not None:
        got = [table_k(table, residual, T) for T in yrs]
        if all(k is not None for k in got):
            src_info = {"table": table_source, "residual": residual}
            return [np.asarray(k) for k in got], "empirical_k_table", src_info, flags
    return [np.full(3, DEFAULT_K) for _ in yrs], "default_k", {}, flags


def station_record_exists(sta: str, doc: Mapping[str, Any] | None) -> bool:
    from geo_dataread.products.loader import station_record

    return station_record(doc, sta) is not None


def event_bounds(
    sta: str,
    period: tuple[float, float],
    settings: DisplacementSettings,
    *,
    events: Sequence[Event] | None,
    areas: Mapping[str, frozenset[str]] | None,
) -> tuple[float, float, str] | str:
    """The step's event as (onset, the midnight after its last day, name),
    or the refusal reason."""
    from geo_dataread.products.period import find_event, parse_date
    from geo_dataread.velocity import window_bounds

    if settings.at is not None:
        day = parse_date(settings.at)
        t0, t1 = window_bounds(day, day)
        return t0, t1, day.isoformat()
    if events is None:
        return "step: no events.yaml (give --at DATE)"
    if settings.event is not None:
        try:
            e = find_event(settings.event, events)
        except ValueError as err:
            return f"step: {err}"
    else:
        from geo_dataread.events import station_events

        p0, p1 = period
        inside = [
            e
            for e in station_events(sta, events, areas or {})
            if p0 <= window_bounds(e.onset, e.onset)[0] < p1
        ]
        if len(inside) != 1:
            names = ", ".join(e.id for e in inside) or "none"
            return f"step: {len(inside)} dikes/eruptions in the period ({names}); give --event"
        e = inside[0]
    t0, _ = window_bounds(e.onset, e.onset)
    _, t1 = window_bounds(e.last_day, e.last_day)
    return t0, t1, e.id


def step_from_view(
    view: StationView,
    period: tuple[float, float],
    event: tuple[float, float, str],
    settings: DisplacementSettings,
    *,
    step_epochs: Sequence[float] = (),
    k_rate: tuple[np.ndarray, np.ndarray] = (np.full(3, 1.0), np.full(3, 1.0)),
) -> tuple[dict[str, Any], list[str], str | None]:
    """The ``step`` displacement of a loaded series across ``event``
    (onset, midnight after the last day, name): ``(values, flags, refused)``."""
    from gps_analysis.rate_quality import NONLINEAR_F, window_gates

    p0, p1 = period
    t_on, t_off, name = event
    w = settings.step_window_days * DAY_YEARS
    before = (max(p0, t_on - w), t_on)
    after = (t_off, min(p1, t_off + w))
    flags: list[str] = []
    if not (before[1] > before[0] and after[1] > after[0]):
        return {}, flags, f"step: the event {name} is not inside the period"
    k_day, _s_day = _daily_factors(view, settings.source.session_years)
    finite = view.t[np.isfinite(view.y).all(axis=0)]
    for side, win in (("before", before), ("after", after)):
        tt = finite[(finite >= win[0]) & (finite < win[1])]
        reasons = window_gates(
            tt, win, step_epochs=step_epochs, min_obs=settings.min_obs,
            min_cover=settings.min_cover, epoch_span=settings.source.session_years,
        )  # fmt: skip
        if reasons:
            return {}, flags, f"step: {side} {'; '.join(reasons)}"
    disp, sig = [], []
    lines: dict[str, Any] = {"before": {}, "after": {}}
    for i, comp in enumerate(COMPONENTS):
        try:
            est = step_displacement(
                view.t, view.y[i], view.sigma[i], before=before, after=after,
                k_day=k_day[i], k_rate=(float(k_rate[0][i]), float(k_rate[1][i])),
            )  # fmt: skip
        except ValueError as err:
            return {}, flags, f"step: {comp}: {err}"
        disp.append(est.step)
        sig.append(est.sigma)
        for side, fit in (("before", est.before), ("after", est.after)):
            lines[side][comp] = {
                "rate": fit.rate, "level": fit.level, "t_mean": fit.t_mean,
                "n": fit.n, "chi2nu": fit.chi2nu, "f_quadratic": fit.f_quadratic,
            }  # fmt: skip
            if math.isfinite(fit.f_quadratic) and fit.f_quadratic > NONLINEAR_F:
                flags.append(f"step_nonlinear:{side}:{comp[0].upper()}")
    step = {
        "event": name,
        "at": [t_on, t_off],
        "window_before": list(before),
        "window_after": list(after),
        "k_day": k_day,
        "k_rate_before": [float(v) for v in k_rate[0]],
        "k_rate_after": [float(v) for v in k_rate[1]],
        "lines": lines,
    }
    return {"displacement": tuple(disp), "sigma": tuple(sig), "step": step}, flags, None


def to_displacement(
    rec: VelocityRecord,
    period: tuple[float, float] | None = None,
    *,
    period_dates: tuple[str, str] | None = None,
) -> DisplacementRecord:
    """A velocity record projected through a period: ``d = v·T``, ``σ = σ_v·T``.

    ``period`` defaults to the record's window. A refused velocity gives a
    refused displacement with the same reason.
    """
    p0, p1 = period if period is not None else rec.window
    d, s = displacement_from_rate(rec.rate, rec.sigma, p1 - p0)
    summary = {
        "window": list(rec.window), "rate": list(rec.rate), "sigma": list(rec.sigma),
        "sigma_method": rec.sigma_method, "k": list(rec.k), "tier": rec.tier,
        "ref": rec.ref, "input_fingerprint": rec.input_fingerprint,
    }  # fmt: skip
    return DisplacementRecord(
        station=rec.station, lon=rec.lon, lat=rec.lat, period=(p0, p1),
        period_dates=period_dates or (_iso(p0), _iso(p1 - DAY_YEARS)),
        frame=rec.frame, series=rec.series, session="24h", ref=rec.ref,
        ref_requested=rec.ref_requested, view=rec.view, method="rate",
        sigma_method=f"rate({rec.sigma_method})*T",
        displacement=(float(d[0]), float(d[1]), float(d[2])),
        sigma=(float(s[0]), float(s[1]), float(s[2])),
        uncert=rec.uncert, rate=jsonable(summary), flags=rec.flags,
        refused=rec.refused, input_fingerprint=rec.input_fingerprint,
        software=rec.software,
    )  # fmt: skip


def displacement_span(rec: DisplacementRecord) -> float:
    """The years a displacement spans: between the end-window centres for
    ``ends`` (where the levels are), else the period."""
    if rec.ends:
        start, end = rec.ends.get("start") or {}, rec.ends.get("end") or {}
        spans = [
            end[c]["t_mean"] - start[c]["t_mean"]
            for c in COMPONENTS
            if c in start and c in end
        ]
        if spans:
            return float(np.mean(spans))
    return rec.period[1] - rec.period[0]


def to_velocity(rec: DisplacementRecord) -> VelocityRecord:
    """The average velocity of a displacement: ``v = d/T``, ``σ = σ_d/T``
    (T by :func:`displacement_span`)."""
    from geo_dataread.velocity import VelocityRecord

    span = displacement_span(rec)
    v, s = rate_from_displacement(rec.displacement, rec.sigma, span)
    nan3 = (math.nan, math.nan, math.nan)
    return VelocityRecord(
        station=rec.station, lon=rec.lon, lat=rec.lat, frame=rec.frame,
        plate=None, ref=rec.ref, ref_requested=rec.ref_requested,
        series=rec.series, view=rec.view, uncert=rec.uncert,
        window=rec.period, span_used=None, n_epochs=(0, 0, 0),
        largest_gap=None, method=f"from_displacement:{rec.method}",
        model="displacement",
        rate=(float(v[0]), float(v[1]), float(v[2])),
        sigma=(float(s[0]), float(s[1]), float(s[2])),
        sigma_formal=nan3, sigma_method=f"({rec.sigma_method})/T",
        k=nan3, k_source={"span_years": span}, corr_en=0.0, record=None,
        steps_in_window=(), outlier_abort=False, n_flagged=0,
        flags=rec.flags, refused=rec.refused, software=rec.software,
        input_fingerprint=rec.input_fingerprint, kind="from_displacement",
    )  # fmt: skip


def estimate_station_displacement(
    sta: str,
    period: tuple[float, float],
    settings: DisplacementSettings | None = None,
    *,
    doc: Mapping[str, Any] | None = None,
    coords: tuple[float, float] | None = None,
    period_dates: tuple[str, str] | None = None,
    period_names: tuple[str, ...] = (),
    events: Sequence[Event] | None = None,
    areas: Mapping[str, frozenset[str]] | None = None,
) -> DisplacementRecord:
    """The displacement of ``sta`` across ``period`` (fractional years,
    half open) by ``settings.method``. Never raises for a station-level
    reason: the record comes back ``refused`` with it. ``events`` and
    ``areas`` (the catalogs) are needed by ``step`` only."""
    settings = settings or DisplacementSettings()
    src = settings.source
    sta = sta.upper()
    p0, p1 = period
    if not p1 > p0:
        raise ValueError(f"period end {p1} is not after its start {p0}")
    if coords is None:
        try:
            coords = station_coords(sta)
        except Exception:  # unknown to stations.cfg: still a record
            coords = (math.nan, math.nan)
    base: dict[str, Any] = {
        "station": sta,
        "lon": coords[0],
        "lat": coords[1],
        "period": (p0, p1),
        "period_dates": period_dates or (_iso(p0), _iso(p1 - DAY_YEARS)),
        "frame": src.frame_name,
        "series": src.pos_id,
        "session": src.session,
        "ref_requested": src.ref,
        "view": src.view,
        "method": settings.method,
        "sigma_method": SIGMA_METHOD,
        "uncert": src.uncert,
        "relative_to": src.relative_to.upper() if src.relative_to else None,
        "period_names": period_names,
        "software": software(),
    }
    if settings.method == "rate":
        return _rate_record(sta, (p0, p1), settings, base, doc=doc)
    pad = PAD_DAYS * DAY_YEARS
    try:
        view = load(sta, p0 - pad, p1 + pad, src, doc=doc)
    except (OSError, ValueError, KeyError) as err:
        return DisplacementRecord(**base, ref=src.ref, refused=f"read: {err}")
    from geo_dataread.gps_views import station_step_epochs

    steps: list[float] = []
    for s in (sta, src.relative_to):
        if s:
            try:
                epochs, _src = station_step_epochs(s.upper(), steps=src.steps)
                steps.extend(float(e) for e in epochs)
            except Exception:  # the catalog is an enhancement, never fatal
                pass
    if settings.method == "step":
        ev = event_bounds(sta, (p0, p1), settings, events=events, areas=areas)
        if isinstance(ev, str):
            return DisplacementRecord(
                **base, ref=view.ref, flags=view.flags, refused=ev
            )
        w_yrs = settings.step_window_days * DAY_YEARS
        before_len = ev[0] - max(p0, ev[0] - w_yrs)
        after_len = min(p1, ev[1] + w_yrs) - ev[1]
        ks, k_method, k_source, k_flags = rate_scale(
            sta, [before_len, after_len], src, view.ref, doc=doc
        )
        values, flags, refused = step_from_view(
            view, (p0, p1), ev, settings, step_epochs=steps, k_rate=(ks[0], ks[1])
        )
        flags = k_flags + flags
        base["sigma_method"] = f"{STEP_SIGMA} [{k_method}]"
        if values:
            values["step"]["k_source"] = jsonable(k_source)
    else:
        values, flags, refused = displacement_from_view(
            view, (p0, p1), settings, step_epochs=steps
        )
    fp = _fingerprint(
        view.sliced(p0, p1),
        {"settings": dataclasses.asdict(settings), "steps": sorted(steps)},
    )
    return DisplacementRecord(
        **base,
        ref=view.ref,
        flags=tuple(view.flags) + tuple(flags),
        refused=refused,
        input_fingerprint=fp,
        **values,
    )


def _rate_record(
    sta: str,
    period: tuple[float, float],
    settings: DisplacementSettings,
    base: dict[str, Any],
    *,
    doc: Mapping[str, Any] | None,
) -> DisplacementRecord:
    """Method ``rate``: the velocity product over the period, × its length."""
    from geo_dataread.velocity import REFS as VELOCITY_REFS
    from geo_dataread.velocity import estimate_station_velocity

    src = settings.source
    why = None
    if src.ref not in VELOCITY_REFS:
        why = f"rate: ref {src.ref} is not a velocity product ref {VELOCITY_REFS}"
    elif src.session != "24h" or src.relative_to:
        why = "rate: 8 h series and --relative-to arrive with gpsprod step 3"
    if why is not None:
        return DisplacementRecord(**base, ref=src.ref, refused=why)
    vrec = estimate_station_velocity(
        sta, period[0], period[1], _velocity_settings(src, src.ref), doc=doc,
        coords=(base["lon"], base["lat"]),
    )  # fmt: skip
    return dataclasses.replace(
        to_displacement(vrec, period, period_dates=base["period_dates"]),
        period_names=base["period_names"],
    )


def station_coords(sta: str) -> tuple[float, float]:
    """(lon, lat) from stations.cfg."""
    from gps_parser import ConfigParser

    info = dict(dict(ConfigParser().getStationInfo(sta))["station"])
    return float(info["longitude"]), float(info["latitude"])


def method_meta(settings: DisplacementSettings) -> dict[str, Any]:
    """The method's own settings, for a set's provenance header."""
    if settings.method == "ends":
        return {"ends_days": settings.ends_days, "sigma": SIGMA_METHOD}
    if settings.method == "points":
        return {"ends": "first and last epoch", "sigma": SIGMA_METHOD}
    if settings.method == "step":
        return {
            "event": settings.event or settings.at or "the one in the period",
            "step_window_days": settings.step_window_days,
            "sigma": STEP_SIGMA,
        }
    return {"sigma": "velocity product sigma × T"}


def estimate_displacements(
    stations: Sequence[str],
    period: tuple[float, float],
    settings: DisplacementSettings | None = None,
    *,
    period_dates: tuple[str, str] | None = None,
    period_names: tuple[str, ...] = (),
    progress: Callable[[str], None] | None = None,
) -> RecordSet[DisplacementRecord]:
    """Every station's displacement across one period, in station order."""
    settings = settings or DisplacementSettings()
    doc = settings.source.params_doc()
    events: Sequence[Event] | None = None
    areas: Mapping[str, frozenset[str]] | None = None
    if settings.method == "step" and settings.at is None:
        from geo_dataread.events import read_events, read_station_areas

        try:
            events = read_events()
            areas = read_station_areas()
        except (OSError, ValueError):
            events = areas = None
    out = []
    for sta in sorted({s.upper() for s in stations}):
        if progress is not None:
            progress(sta)
        out.append(
            estimate_station_displacement(
                sta,
                period,
                settings,
                doc=doc,
                period_dates=period_dates,
                period_names=period_names,
                events=events,
                areas=areas,
            )  # fmt: skip
        )
    src = settings.source
    meta = {
        "frame": src.frame_name,
        "series": src.pos_id,
        "session": src.session,
        "ref": src.ref,
        "view": src.view,
        "method": settings.method,
        **method_meta(settings),
        "period": "{}..{}".format(
            *(period_dates or (_iso(period[0]), _iso(period[1] - DAY_YEARS)))
        ),
        "relative_to": src.relative_to or "-",
    }
    return RecordSet(out, "displacement", meta)
