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
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from gps_analysis.displacement import (
    end_displacement,
    end_level,
    neighbour_check,
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

METHODS = ("ends",)
#: days read beyond the period on each side: the daily σ factors and the
#: cleaned view's detector see context, not just the period
PAD_DAYS = 60
#: the data lag at the head of a series that is not worth a flag [days]
END_LAG_DAYS = 1.5
SIGMA_METHOD = "max(k_day*formal, formal*sqrt(chi2nu), s_day/sqrt(n))"


@dataclasses.dataclass(frozen=True)
class DisplacementSettings:
    """How a set of displacements is estimated."""

    source: Source = Source()
    method: str = "ends"
    ends_days: int = 5
    neighbour_days: int = 5

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.ends_days < 1:
            raise ValueError(f"ends_days must be ≥ 1, got {self.ends_days}")
        if self.neighbour_days < 2:
            raise ValueError(f"neighbour_days must be ≥ 2, got {self.neighbour_days}")


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
        return {}, flags, "no epochs in the period"
    last = float(in_period[-1])
    # the end window closes at the last epoch: the newest days of a period
    # running to today are not there yet (processing lag, flagged past
    # END_LAG_DAYS), and an empty tail must not shrink the window
    end_edge = min(p1, last + 0.5 * step)
    if p1 - end_edge > (END_LAG_DAYS + 0.5) * DAY_YEARS:
        flags.append(f"end_early:{(p1 - end_edge) / DAY_YEARS:.0f}d")
    w_start = (p0, p0 + n_days * DAY_YEARS)
    w_end = (end_edge - n_days * DAY_YEARS, end_edge)
    if w_end[0] < w_start[1]:
        return {}, flags, f"period shorter than two {n_days}-day end windows"
    for e in sorted(float(x) for x in step_epochs):
        if w_start[0] <= e < w_start[1] or w_end[0] <= e < w_end[1]:
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
                    f"{_iso(lo)}..{_iso(hi - DAY_YEARS)}{hint}"
                ),
            )
    disp, sig = [], []
    ends: dict[str, Any] = {
        "days": n_days,
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
            for key, lv, nb in (
                ("start", a, (w_start[1], w_start[1] + m)),
                ("end", b, (w_end[0] - m, w_end[0])),
            ):
                chk = neighbour_check(
                    view.t, view.y[i], view.sigma[i],
                    end=lv, neighbour=nb, k_day=k_day[i],
                )  # fmt: skip
                if chk.disputed:
                    flags.append(f"end_disputed:{key}:{comp[0].upper()}")
                elif chk.disputed is None:
                    flags.append(f"end_unchecked:{key}:{comp[0].upper()}")
    values = {"displacement": tuple(disp), "sigma": tuple(sig), "ends": ends}
    return values, flags, None


def estimate_station_displacement(
    sta: str,
    period: tuple[float, float],
    settings: DisplacementSettings | None = None,
    *,
    doc: Mapping[str, Any] | None = None,
    coords: tuple[float, float] | None = None,
    period_dates: tuple[str, str] | None = None,
    period_names: tuple[str, ...] = (),
) -> DisplacementRecord:
    """The displacement of ``sta`` across ``period`` (fractional years,
    half open). Never raises for a station-level reason: the record comes
    back ``refused`` with it."""
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
    pad = PAD_DAYS * DAY_YEARS
    try:
        view = load(sta, p0 - pad, p1 + pad, src, doc=doc)
    except (OSError, ValueError) as err:
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


def station_coords(sta: str) -> tuple[float, float]:
    """(lon, lat) from stations.cfg."""
    from gps_parser import ConfigParser

    info = dict(dict(ConfigParser().getStationInfo(sta))["station"])
    return float(info["longitude"]), float(info["latitude"])


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
        "ends_days": settings.ends_days,
        "sigma": SIGMA_METHOD,
        "period": "{}..{}".format(
            *(period_dates or (_iso(period[0]), _iso(period[1] - DAY_YEARS)))
        ),
        "relative_to": src.relative_to or "-",
    }
    return RecordSet(out, "displacement", meta)
