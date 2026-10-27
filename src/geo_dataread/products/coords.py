"""The coordinate product: where a station is at an epoch (ECEF and lat/lon/h).

Method ``observed`` (the default): the GLS mean of the station's ``.pos``
X/Y/Z solutions within ``window_days`` either side of the epoch, each with its
full 3×3 covariance (:func:`gps_analysis.gls_mean`), after the reviewed
junction offsets and the view (declared/cleaned/raw, the σ screen) — the
same epochs every other product uses. σ = max(k_day, √χ²/ν) × the formal σ
of the mean, k_day the station's daily σ factor (median over N/E/U).

When no solution lies within the window (a gap, a station that started
later, a future epoch) the coordinate is EXTRAPOLATED from the nearest
observed window by the station's trajectory: the change of its stored
detrend record between the two epochs plus the plate motion (horizontal, as
the series' plate removal), rotated to ECEF — flagged ``extrapolated:Nd``.
Method ``model`` does that for every epoch (a "clean" coordinate without the
day-to-day noise); its σ is the observed window's (the model's own σ is not
propagated: flag ``sigma:observed_only``). Method ``apr`` is the GLOBK
apriori (``igb14_comb.apr``/``.eq`` of the GAMIT tables) — for comparing.

``relative_to``: the baseline vector X/Y/Z(A) − X/Y/Z(B) at the epoch, σ in
quadrature, and its N/E/U at A. Coordinates are metres throughout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gps_analysis.displacement import gls_mean
from gps_analysis.empirical_sigma import sigma_scale_factor

from geo_dataread.products.common import NAN3, SCHEMA_VERSION, Vec3, jsonable, software
from geo_dataread.products.loader import DAY_YEARS, Source, load, station_record

FloatArray = npt.NDArray[np.float64]
METHODS = ("observed", "model", "apr")
#: days read beyond the window each side: the daily σ factor and the
#: cleaned view's detector see context
PAD_DAYS = 60
#: χ²/ν of a window above which the station moved within it (an event, fast
#: unrest): the mean is flagged, a shorter --window-days is the remedy
FAST_CHI2NU = 25.0
GAMIT_TABLES = Path(os.environ.get("GG_TABLES", Path.home() / "gg" / "tables"))
#: frame → the GAMIT tables' apriori and earthquake/rename files
APR_FILES = {
    "itrf14": ("igb14_comb.apr", "igb14_comb.eq"),
    "itrf20": ("igb20_comb.apr", "igb20_comb.eq"),
}


@dataclasses.dataclass(frozen=True)
class CoordSettings:
    """How a set of coordinates is made."""

    source: Source = Source(view="cleaned", uncert=100.0)
    method: str = "observed"
    window_days: int = 5
    apr_files: tuple[str, ...] = ()
    eq_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.window_days < 0:
            raise ValueError(f"window_days must be ≥ 0, got {self.window_days}")


@dataclasses.dataclass(frozen=True)
class CoordRecord:
    """One station's coordinate at one epoch (metres).

    ``xyz``/``sigma_xyz`` ECEF; ``llh`` latitude, longitude [deg], ellipsoidal
    height [m]; ``sigma_neu`` its σ in the local frame. With ``relative_to``
    the coordinate is the baseline vector A − B: ``xyz`` is that vector,
    ``dneu`` its N/E/U at A, ``llh`` A's position. ``t_obs`` is the epoch of
    the observed mean the value rests on (equal to ``epoch`` within the
    window; the anchor of an extrapolation otherwise).
    """

    station: str
    epoch: float
    epoch_date: str
    frame: str
    series: str
    session: str
    view: str
    method: str
    xyz: Vec3 = NAN3
    sigma_xyz: Vec3 = NAN3
    cov_xyz: tuple[Vec3, Vec3, Vec3] | None = None
    llh: Vec3 = NAN3
    sigma_neu: Vec3 = NAN3
    dneu: Vec3 | None = None
    n_used: int = 0
    window_days: int = 0
    t_obs: float = math.nan
    chi2nu: float = math.nan
    k: float = math.nan
    sigma_method: str = ""
    uncert: float = math.nan
    relative_to: str | None = None
    flags: tuple[str, ...] = ()
    refused: str | None = None
    input_fingerprint: str = ""
    software: dict[str, str] = dataclasses.field(default_factory=dict)
    units: str = "m"
    schema_version: int = SCHEMA_VERSION
    kind: str = "coords"

    @property
    def accepted(self) -> bool:
        return self.refused is None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = jsonable(dataclasses.asdict(self))
        return d

    def scaled(self, units: str) -> CoordRecord:
        """Coordinates are always metres (``units`` must be "m")."""
        if units not in ("m", "mm"):
            raise ValueError(f"units {units!r}")
        return self


def _vec(a: npt.ArrayLike) -> Vec3:
    x, y, z = (float(v) for v in np.asarray(a, dtype=float).reshape(3))
    return x, y, z


def _iso(t: float) -> str:
    from gtimes.timefunc import TimefromYearf

    return str(TimefromYearf(t + 1e-6, "%Y-%m-%d"))


# ---------------------------------------------------------------------------
# the pure part
# ---------------------------------------------------------------------------


def window_half(window_days: float) -> float:
    """Half a window [yr]: ``window_days`` either side (a day's edge
    included despite year-fraction rounding), half a day at least."""
    return (max(window_days, 0.5) + 1e-6) * DAY_YEARS


def observed_mean(
    t: FloatArray,
    xyz: FloatArray,
    cov: FloatArray,
    epoch: float,
    window_days: float,
    *,
    k_day: float = 1.0,
) -> dict[str, Any] | None:
    """The GLS mean of the solutions within ``window_days`` of ``epoch``:
    ``{xyz, cov, n, t_obs, chi2nu, k}`` with the σ scale applied, or None
    when no solution is that close. ``window_days = 0`` takes the nearest
    single solution within half a day."""
    half = window_half(window_days)
    keep = np.abs(t - epoch) <= half
    if not keep.any():
        return None
    g = gls_mean(xyz[keep], cov[keep])
    scale = k_day if math.isfinite(k_day) and k_day > 0 else 1.0
    if math.isfinite(g.chi2nu):
        scale = max(scale, math.sqrt(g.chi2nu))
    w = 1.0 / np.trace(cov[keep], axis1=1, axis2=2)
    return {
        "xyz": g.mean,
        "cov": g.cov * scale**2,
        "n": g.n,
        "t_obs": float(np.sum(w * t[keep]) / np.sum(w)),
        "chi2nu": g.chi2nu,
        "k": scale,
    }


def cut_at_steps(
    t: FloatArray, epoch: float, half: float, steps: Sequence[float]
) -> tuple[npt.NDArray[np.bool_], list[str]]:
    """The solutions within ``half`` of ``epoch`` on the epoch's side of every
    declared step inside that window: a mean never straddles a step."""
    keep = np.abs(t - epoch) <= half
    flags = []
    for e in sorted(float(x) for x in steps):
        if abs(e - epoch) <= half and keep.any():
            side = t < e if epoch < e else t > e
            if (keep & ~side).any():
                keep &= side
                flags.append(f"window_cut_at_step:{_iso(e)}")
    return keep, flags


def nearest_window(t: FloatArray, epoch: float, window_days: float) -> float | None:
    """The epoch of the solution nearest ``epoch`` (the centre of the window
    an extrapolation is anchored on); None without solutions."""
    if t.size == 0:
        return None
    return float(t[np.argmin(np.abs(t - epoch))])


def local_frame(xyz: FloatArray) -> tuple[Vec3, FloatArray]:
    """``((lat°, lon°, h), R)``: the geodetic position of ``xyz`` and the
    ECEF → N/E/U rotation there."""
    from geofunc import local

    g = local.ecef_to_geodetic(xyz)
    r = local.local_rotation(g.lat, g.lon)
    return (math.degrees(float(g.lat)), math.degrees(float(g.lon)), float(g.h)), r


# ---------------------------------------------------------------------------
# reading a station
# ---------------------------------------------------------------------------


def _pos_series(sta: str, src: Source) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(t, xyz (n,3), cov (n,3,3))`` of the station's ``.pos`` with the
    reviewed junction offsets, as every product sees the coordinates."""
    from geofunc import local

    from geo_dataread import pos

    series = pos.read_pos(pos.pos_path(sta, src.directory(), src.pos_id))
    rules = pos.product_junctions(src.pos_id)
    if rules is None:
        rules = pos._default_junctions(sta)
    xyz = series.xyz
    t = series.yearf
    if rules:
        xyz = local.apply_local_offsets(xyz, pos.junction_dlocal(t, rules))
    return t, np.asarray(xyz, float), np.asarray(series.cov_xyz(), float)


def _view_epochs(
    sta: str, t0: float, t1: float, src: Source, doc: Mapping[str, Any] | None
) -> tuple[FloatArray, float]:
    """The epochs the view keeps in ``[t0, t1)`` (all three components
    finite) and the station's daily σ factor there (median over N/E/U)."""
    plain = dataclasses.replace(src, ref="plate", relative_to=None)
    view = load(sta, t0, t1, plain, doc=doc)
    ok = np.isfinite(view.y).all(axis=0)
    k = math.nan
    if view.t.size > 2:
        kk = sigma_scale_factor(view.t, view.y, view.sigma, step=src.session_years)
        if np.isfinite(kk).any():
            k = float(np.nanmedian(kk))
    return view.t[ok], k


def _match(t_pos: FloatArray, t_view: FloatArray, tol: float) -> npt.NDArray[np.bool_]:
    """Which ``.pos`` epochs are among the view's kept epochs."""
    if t_view.size == 0:
        return np.zeros(t_pos.size, dtype=bool)
    idx = np.clip(np.searchsorted(t_view, t_pos), 1, t_view.size - 1)
    near = np.minimum(np.abs(t_view[idx] - t_pos), np.abs(t_view[idx - 1] - t_pos))
    return near <= tol


def _model_shift(
    sta: str, src: Source, record: Mapping[str, Any] | None, t_from: float, t_to: float
) -> tuple[FloatArray, list[str]]:
    """N/E/U change [m] of the station from ``t_from`` to ``t_to``: its
    detrend record (plate-removed frame) plus the plate motion removed from
    the horizontals, as the series' own plate removal (getData)."""
    from gps_analysis.detrend import evaluate_record

    from geo_dataread import pos
    from geo_dataread.station_catalog import plate_velocity_neu

    flags: list[str] = []
    d = np.zeros(3)
    if record is not None:
        m = evaluate_record(record, np.array([t_from, t_to]))
        d += (m[:, 1] - m[:, 0]) / 1000.0
    else:
        flags.append("model:plate_only")
    pmm = pos.POS_PRODUCTS.get(src.pos_id, (None, "ITRF2008"))[1]
    try:
        v = plate_velocity_neu(sta, reference=pmm)
        d[:2] += np.asarray(v, float)[:2] * (t_to - t_from)
    except KeyError:
        flags.append("model:no_plate")
    return d, flags


# ---------------------------------------------------------------------------
# one station
# ---------------------------------------------------------------------------


def _apr(
    sta: str, epoch: float, settings: CoordSettings
) -> tuple[FloatArray | None, str | None]:
    from geo_dataread import globk_apr
    from geo_dataread.pos import pos_frame

    apr_files = settings.apr_files
    eq_files = settings.eq_files
    if not apr_files:
        names = APR_FILES.get(pos_frame(settings.source.frame))
        if names is None:
            return None, f"apr: no GAMIT apriori for {settings.source.frame_name}"
        apr_files, eq_files = (
            (str(GAMIT_TABLES / names[0]),),
            (str(GAMIT_TABLES / names[1]),),
        )
    try:
        apr = globk_apr.read_apr(apr_files)
        renames = globk_apr.read_renames(eq_files) if eq_files else []
    except OSError as err:
        return None, f"apr: {err}"
    when = globk_apr.decimal_year_to_datetime(epoch)
    got, skipped = globk_apr.apriori_xyz([sta], when, apr, renames)
    if sta not in got:
        return None, f"apr: {skipped.get(sta, 'not found')}"
    return np.asarray(got[sta], float), None


def station_coords(
    sta: str,
    epoch: float,
    settings: CoordSettings,
    *,
    doc: Mapping[str, Any] | None = None,
    event_edges: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """The coordinate of one station: ``{xyz, cov, n, t_obs, chi2nu, k,
    flags, refused, sigma_method, fp}`` (no relative_to). ``event_edges``:
    per station, the onsets and ends of its dikes/eruptions (events.yaml),
    treated like declared steps."""
    event_edges = event_edges or {}
    src = settings.source
    out: dict[str, Any] = {"flags": [], "refused": None}
    if settings.method == "apr":
        xyz, why = _apr(sta, epoch, settings)
        if xyz is None:
            out["refused"] = why
            return out
        out.update(xyz=xyz, cov=np.full((3, 3), np.nan), n=0, t_obs=epoch,
                   chi2nu=math.nan, k=math.nan, sigma_method="apriori (no sigma)")  # fmt: skip
        out["flags"].append("apriori")
        out["fp"] = hashlib.sha256(np.asarray(xyz).tobytes()).hexdigest()
        return out
    try:
        t, xyz, cov = _pos_series(sta, src)
    except (OSError, ValueError) as err:
        out["refused"] = f"read: {err}"
        return out
    pad = (PAD_DAYS + settings.window_days) * DAY_YEARS
    try:
        kept, k_day = _view_epochs(sta, epoch - pad, epoch + pad, src, doc)
    except (OSError, ValueError, KeyError) as err:
        kept, k_day = np.zeros(0), math.nan
        out["flags"].append(f"view:none ({err})")
    in_view = _match(t, kept, 0.3 * src.session_years)
    near = in_view | (
        np.abs(t - epoch) > pad
    )  # beyond the read: the view did not judge
    from geo_dataread.gps_views import station_step_epochs

    try:
        steps = [float(e) for e in station_step_epochs(sta, steps=src.steps)[0]]
    except Exception:  # the catalog is an enhancement, never fatal
        steps = []
    steps += list(event_edges.get(sta, ()))
    half = window_half(settings.window_days)
    side, cut_flags = cut_at_steps(t, epoch, half, steps)
    obs = None
    if settings.method == "observed":
        use = near & (side | (np.abs(t - epoch) > half))
        obs = observed_mean(
            t[use], xyz[use], cov[use], epoch, settings.window_days, k_day=k_day
        )
        if obs is not None:
            out["flags"] += cut_flags
    if obs is not None:
        if math.isfinite(obs["chi2nu"]) and obs["chi2nu"] > FAST_CHI2NU:
            out["flags"].append(f"fast_motion:chi2nu={obs['chi2nu']:.0f}")
        out.update(obs)
        out["sigma_method"] = "max(k_day, sqrt(chi2nu)) * gls"
        out["fp"] = _fp(t[near], xyz[near], settings, epoch)
        return out
    # extrapolate (or method model): anchor on the nearest observed window
    anchor = nearest_window(t[near], epoch, settings.window_days)
    if anchor is None:
        out["refused"] = "no solutions to anchor on"
        return out
    base = observed_mean(
        t[near], xyz[near], cov[near], anchor, settings.window_days, k_day=k_day
    )
    assert base is not None
    record = station_record(doc, sta)
    shift, mflags = _model_shift(sta, src, record, base["t_obs"], epoch)
    (lat, lon, _h), r = local_frame(base["xyz"])
    out.update(base)
    out["xyz"] = base["xyz"] + r.T @ shift
    out["flags"] += mflags + ["sigma:observed_only"]
    lo, hi = sorted((base["t_obs"], epoch))
    if any(lo < e < hi for e in steps):
        out["flags"].append("steps_between")
    gap = abs(epoch - base["t_obs"]) / DAY_YEARS
    if settings.method == "observed" or gap > max(settings.window_days, 0.5):
        out["flags"].append(f"extrapolated:{gap:.0f}d")
    out["sigma_method"] = "observed anchor; model sigma not propagated"
    out["fp"] = _fp(t[near], xyz[near], settings, epoch)
    return out


def _fp(t: FloatArray, xyz: FloatArray, settings: CoordSettings, epoch: float) -> str:
    h = hashlib.sha256()
    for a in (t, xyz):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    extra = {"settings": dataclasses.asdict(settings), "epoch": epoch}
    h.update(json.dumps(jsonable(extra), sort_keys=True).encode())
    return h.hexdigest()


def station_event_edges(stations: Sequence[str]) -> dict[str, list[float]]:
    """Per station: the start (onset midnight) and end (the midnight after its
    last day) of every dike/eruption of events.yaml in its area; empty
    without a catalog."""
    from geo_dataread.events import read_events, read_station_areas, station_events
    from geo_dataread.products.velocity_core import window_bounds

    try:
        events, areas = read_events(), read_station_areas()
    except (OSError, ValueError):
        return {}
    out: dict[str, list[float]] = {}
    for sta in stations:
        for e in station_events(sta, events, areas):
            out.setdefault(sta.upper(), []).extend(
                (
                    window_bounds(e.onset, e.onset)[0],
                    window_bounds(e.last_day, e.last_day)[1],
                )
            )
    return out


def estimate_station_coords(
    sta: str,
    epoch: float,
    settings: CoordSettings | None = None,
    *,
    doc: Mapping[str, Any] | None = None,
    epoch_date: str | None = None,
    event_edges: Mapping[str, Sequence[float]] | None = None,
) -> CoordRecord:
    """The coordinate of ``sta`` at ``epoch`` (fractional year), or the
    baseline vector to ``settings.source.relative_to``. Never raises for a
    station-level reason: the record comes back ``refused``."""
    settings = settings or CoordSettings()
    src = settings.source
    sta = sta.upper()
    base: dict[str, Any] = {
        "station": sta, "epoch": epoch, "epoch_date": epoch_date or _iso(epoch),
        "frame": src.frame_name, "series": src.pos_id, "session": src.session,
        "view": src.view, "method": settings.method,
        "window_days": settings.window_days, "uncert": src.uncert,
        "relative_to": src.relative_to.upper() if src.relative_to else None,
        "software": software(),
    }  # fmt: skip
    a = station_coords(sta, epoch, settings, doc=doc, event_edges=event_edges)
    if a["refused"]:
        return CoordRecord(**base, flags=tuple(a["flags"]), refused=a["refused"])
    xyz, cov = np.asarray(a["xyz"]), np.asarray(a["cov"])
    flags = list(a["flags"])
    fp = a["fp"]
    llh, r = local_frame(xyz)
    dneu = None
    if src.relative_to:
        other = src.relative_to.upper()
        b = station_coords(other, epoch, settings, doc=doc, event_edges=event_edges)
        if b["refused"]:
            return CoordRecord(
                **base, flags=tuple(flags), refused=f"{other}: {b['refused']}"
            )
        flags += [f"{other}:{f}" for f in b["flags"]]
        xyz = xyz - np.asarray(b["xyz"])
        cov = cov + np.asarray(b["cov"])
        dneu = _vec(r @ xyz)
        fp = hashlib.sha256((fp + b["fp"]).encode()).hexdigest()
    cov_neu = r @ cov @ r.T
    return CoordRecord(
        **base,
        xyz=_vec(xyz),
        sigma_xyz=_vec(np.sqrt(np.diag(cov))),
        cov_xyz=(_vec(cov[0]), _vec(cov[1]), _vec(cov[2])),
        llh=llh,
        sigma_neu=_vec(np.sqrt(np.diag(cov_neu))),
        dneu=dneu,
        n_used=int(a["n"]),
        t_obs=float(a["t_obs"]),
        chi2nu=float(a["chi2nu"]),
        k=float(a["k"]),
        sigma_method=a["sigma_method"],
        flags=tuple(flags),
        input_fingerprint=fp,
    )


def estimate_coords(
    stations: Sequence[str],
    epoch: float,
    settings: CoordSettings | None = None,
    *,
    epoch_date: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Any:
    """Every station's coordinate at one epoch, in station order (a
    :class:`~geo_dataread.products.common.RecordSet`)."""
    from geo_dataread.products.common import RecordSet

    settings = settings or CoordSettings()
    doc = settings.source.params_doc() if settings.method != "apr" else None
    names = sorted({s.upper() for s in stations})
    if settings.source.relative_to:
        names_all = [*names, settings.source.relative_to.upper()]
    else:
        names_all = names
    edges = station_event_edges(names_all) if settings.method != "apr" else {}
    out = []
    for sta in names:
        if progress is not None:
            progress(sta)
        out.append(
            estimate_station_coords(
                sta,
                epoch,
                settings,
                doc=doc,
                epoch_date=epoch_date,
                event_edges=edges,
            )  # fmt: skip
        )
    src = settings.source
    meta = {
        "frame": src.frame_name, "series": src.pos_id, "session": src.session,
        "view": src.view, "method": settings.method, "epoch": epoch_date or _iso(epoch),
        "window_days": settings.window_days, "units": "m",
        "relative_to": src.relative_to or "-",
    }  # fmt: skip
    return RecordSet(out, "coords", meta)
