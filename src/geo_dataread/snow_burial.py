"""Snow-burial index: an antenna under snow, told apart from icing.

A radome iced unevenly makes the post-fit residuals depend on azimuth and moves
the horizontals — :func:`geo_dataread.autcln_stats.icing_days` catches that
from the azimuth mean. A radome BURIED in a snow pack delays the signal from
every direction alike: the azimuth mean barely moves (OLAC ×1.5, below the
icing start of 2.0), the elevation-INDEPENDENT noise term ``A`` of
``RMS² = A² + B²/sin²(el)`` doubles while ``B`` stays, and the position shows
a one-sided Up bump of tens to hundreds of mm that grows for weeks and is gone
at melt, the horizontals quiet. Medians on the days concerned (2026-10-20)::

                       A     B    A/B   az mean   residual
    OLAC buried       6.3   3.3   1.85   3.4      Up +99 mm, N/E ~1-3 mm
    OLAC winter quiet 3.0   3.1   0.82   2.3
    GFUM/AUST icing   4.4   3.6   1.25   4.0-4.5  N/E tens of mm

The icing rule caught 47 of OLAC's 96 burial days (Up > 40 mm before the
2021 inflation). This index catches 93 (itrf14 .pos, the resolver's path);
195 of 228 harvested stations get no snow day at all; most: OLAC 9 %, GFUM
5 % of days (beside its icing), AUST 3 %.

The rule, per calendar day (:func:`snow_days`):

1. Up excursion: Up above its own LOWER envelope (the ``envelope_quantile``
   over ±``half_window_days`` -- 60, so a two-month burial does not lift its
   own envelope; snow only ever lifts Up), in units of the local noise
   (first-difference MAD over ``scale_days``);
2. an episode STARTS on a day with the excursion > ``k_start`` σ and
   > ``min_mm``, North and East inside ``k_start`` σ of their own envelope,
   and A/B above ``ab_ratio`` × its trailing ``window_days`` low
   ``quantile`` (the station's own quiet level);
3. it LASTS while the excursion stays > ``k_stay`` σ, bridging gaps of at most
   ``max_gap_days`` (missing days included);
4. it is KEPT only if A/B over the whole episode sits above
   ``episode_ratio`` × the quiet level (a single disturbed day must not turn
   a Svartsengi uplift ramp into "snow": SKSH 292 → 0 days), it does not end
   in a horizontal jump > ``jump_k`` σ (an Up sawtooth that ends at an
   eruption: SENG 2025-03-31, GEVK 2025-07-16), and it is a BUMP: its median
   rises > ``bump_k`` σ (and > ``bump_mm``) above the chord between the Up
   levels over ``flank_days`` before and after it (a trend lies on the chord),
   AND its peak stands that much above the level after it (a melt comes back
   down; a ramp onto a new level -- uplift, a step -- ends at its peak).

An episode without data after it (the live end, or the station stopped) has
no after-level: it is kept on the A/B evidence alone and marked
``provisional``. One without data before it is not kept.

Snow lifts Up only, so the view flags Up only (the whole epoch still stays
out of the outlier detector, like icing). The input is the station's FULL
24 h ``.pos`` series of the frame (:func:`read_daily_up_series`), so every
caller — a 90-day plot, a detrended read, the estimator — gets the same days.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geo_dataread import autcln_stats as _ac

__all__ = [
    "SnowParams",
    "daily_positions",
    "read_daily_up_series",
    "snow_days",
    "snow_epochs",
]


@dataclass(frozen=True)
class SnowParams:
    """Thresholds of :func:`snow_days` (calibrated on OLAC, 2026-10-20)."""

    half_window_days: int = 60
    envelope_quantile: float = 0.2
    scale_days: int = 121
    k_start: float = 5.0
    k_stay: float = 2.0
    min_mm: float = 25.0
    ab_ratio: float = 2.0
    episode_ratio: float = 1.8
    quantile: float = 0.25
    window_days: int = 365
    min_days: int = 60
    max_gap_days: int = 5
    flank_days: int = 20
    bump_k: float = 3.0
    bump_mm: float = 15.0
    jump_k: float = 20.0


def _day_of(t: np.ndarray) -> pd.DatetimeIndex:
    """Calendar day of noon fractional-year epochs (the inverse of
    :func:`geo_dataread.autcln_stats.icing_epochs`)."""
    year = np.floor(t).astype(int)
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    doy = np.floor((t - year) * np.where(leap, 366, 365)).astype(int)
    days = pd.to_datetime(pd.Series(year).astype(str) + "-01-01") + pd.to_timedelta(
        doy, unit="D"
    )
    return pd.DatetimeIndex(days)


def _noon_yearf(days: pd.DatetimeIndex) -> np.ndarray:
    out = [
        d.year + (d.dayofyear - 0.5) / (366 if d.is_leap_year else 365) for d in days
    ]
    return np.asarray(out, dtype=float)


def daily_positions(t: np.ndarray, neu: np.ndarray) -> pd.DataFrame:
    """North/East/Up [mm] indexed by calendar day (a day's epochs averaged)."""
    t = np.asarray(t, dtype=float)
    neu = np.atleast_2d(np.asarray(neu, dtype=float))
    df = pd.DataFrame({"N": neu[0], "E": neu[1], "U": neu[2]}, index=_day_of(t))
    return df[np.isfinite(df).all(axis=1)].groupby(level=0).mean()


def read_daily_up_series(
    sta: str,
    *,
    frame: str | None = None,
    pos_dir: str | Path | None = None,
    uncert: float = 15,
) -> pd.DataFrame:
    """The station's full 24 h ``.pos`` series of ``frame``, per day [mm]."""
    from geo_dataread import gps_read  # deferred: gps_read imports back lazily
    from geo_dataread.pos import pos_product

    yearf, data, _ddata, _offset = gps_read.getData(  # type: ignore[no-untyped-call]
        sta,
        ref="plate",
        Dir=None if pos_dir is None else str(pos_dir),
        tType="POS",
        uncert=uncert,
        pos_id=pos_product(frame),
    )
    return daily_positions(np.asarray(yearf), np.asarray(data))


def _excursion(x: pd.Series, p: SnowParams) -> tuple[pd.Series, pd.Series]:
    """(excursion above the lower envelope [mm], local noise σ [mm])."""
    span = f"{2 * p.half_window_days + 1}D"
    envelope = x.rolling(span, center=True, min_periods=10).quantile(
        p.envelope_quantile
    )
    step = x.diff().abs()
    sigma = (
        1.4826
        * step.rolling(f"{p.scale_days}D", center=True, min_periods=20).median()
        / np.sqrt(2.0)
    )
    return x - envelope, sigma


def _episodes(
    start: np.ndarray, stay: np.ndarray, max_gap: int
) -> list[tuple[int, int]]:
    """Inclusive (first, last) day indices of every episode.

    An episode holds a start day, reaches back over consecutive stay days,
    and forward to the last stay day whose distance from the previous one is
    at most ``max_gap + 1`` (a gap of ``max_gap`` days is bridged).
    """
    out: list[tuple[int, int]] = []
    n = start.size
    i = 0
    while i < n:
        if not start[i]:
            i += 1
            continue
        first = i
        floor = out[-1][1] + 1 if out else 0
        while first > floor and stay[first - 1]:
            first -= 1
        last = i
        k = i + 1
        while k < n and k - last <= max_gap + 1:
            if stay[k] or start[k]:
                last = k
            k += 1
        out.append((first, last))
        i = last + 1
    return out


def _ends_in_a_jump(
    horiz: list[tuple[np.ndarray, np.ndarray]], last: int, p: SnowParams
) -> bool:
    """Whether North or East jumps by more than ``jump_k`` σ where the episode
    ends: deformation (SENG 2025-03-31 E −188 σ, GEVK 2025-07-16 N +56 σ —
    eruption days), never a melt (OLAC, GFUM, AUST ends ≤ 9 σ)."""
    for x, sig in horiz:
        pre = x[max(0, last - 5) : last + 1]
        post = x[last + 1 : last + 8]
        scale = np.nanmedian(sig[max(0, last - 5) : last + 8])
        if np.isfinite(pre).any() and np.isfinite(post).any() and scale > 0:
            if abs(np.nanmedian(post) - np.nanmedian(pre)) > p.jump_k * scale:
                return True
    return False


def snow_days(
    stats: pd.DataFrame,
    positions: pd.DataFrame,
    params: SnowParams | None = None,
) -> pd.DataFrame:
    """Snow-burial days: ``snow`` and ``provisional`` (bool) per calendar day.

    Args:
        stats: :func:`geo_dataread.autcln_stats.read_autcln_stats` (needs
            ``atelv_a``, ``atelv_b``); empty → no snow days.
        positions: :func:`daily_positions` (``N``, ``E``, ``U`` [mm] by day).
        params: thresholds (:class:`SnowParams`).

    Returns:
        Every calendar day of ``positions``; ``provisional`` marks the days of
        an episode with no data after it (kept on the A/B evidence alone).
    """
    p = SnowParams() if params is None else params
    if positions.empty:
        return pd.DataFrame({"snow": [], "provisional": []}, dtype=bool)
    days = pd.date_range(positions.index.min(), positions.index.max(), freq="D")
    empty = pd.DataFrame({"snow": False, "provisional": False}, index=days, dtype=bool)
    if stats.empty or not {"atelv_a", "atelv_b"} <= set(stats.columns):
        return empty

    ex_u, sig_u = _excursion(positions["U"], p)
    z_u = ex_u / sig_u
    quiet_h = pd.Series(True, index=positions.index)
    horiz: list[tuple[np.ndarray, np.ndarray]] = []
    for c in ("N", "E"):
        ex, sig = _excursion(positions[c], p)
        quiet_h &= (ex / sig).abs() < p.k_start
        horiz.append(
            (positions[c].reindex(days).to_numpy(), sig.reindex(days).to_numpy())
        )

    ab = (stats["atelv_a"] / stats["atelv_b"]).replace([np.inf, -np.inf], np.nan)
    ab = ab.dropna()
    base = ab.rolling(f"{p.window_days}D", min_periods=p.min_days).quantile(p.quantile)
    rel = (ab / base).reindex(days)

    start = ((z_u > p.k_start) & (ex_u > p.min_mm) & quiet_h).reindex(days)
    start = start.fillna(False).to_numpy(dtype=bool) & (rel > p.ab_ratio).to_numpy()
    stay = (z_u > p.k_stay).reindex(days).fillna(False).to_numpy(dtype=bool)

    up = positions["U"].reindex(days).to_numpy()
    sig = sig_u.reindex(days).to_numpy()
    rel_v = rel.to_numpy()
    snow = np.zeros(days.size, dtype=bool)
    prov = np.zeros(days.size, dtype=bool)
    for first, last in _episodes(start, stay, p.max_gap_days):
        if not np.nanmedian(rel_v[first : last + 1]) > p.episode_ratio:
            continue
        if _ends_in_a_jump(horiz, last, p):
            continue  # a dike or an eruption, not a melt
        before = up[max(0, first - p.flank_days) : first]
        after = up[last + 1 : last + 1 + p.flank_days]
        inside = up[first : last + 1]
        if np.isfinite(after).sum() == 0:
            snow[first : last + 1] = True
            prov[first : last + 1] = True
            continue
        if np.isfinite(before).sum() == 0:
            continue  # nothing to compare a bump with: leave it unflagged
        # a BUMP: above the chord between the flank levels (a trend lies on
        # it -- OLAC inflating 150 mm/yr under its 2024 snow), and peaking
        # above the level after it (a melt comes back down; a ramp onto a
        # new level -- uplift, a step -- ends at its peak)
        b0, a0 = np.nanmedian(before), np.nanmedian(after)
        frac = (np.arange(first, last + 1) - first + 0.5) / (last + 1 - first)
        rise = np.nanmedian(inside - (b0 + (a0 - b0) * frac))
        peak = np.nanpercentile(inside, 90) - a0
        need = max(p.bump_k * np.nanmedian(sig[first : last + 1]), p.bump_mm)
        if rise > need and peak > need:
            snow[first : last + 1] = True
    have = np.isfinite(up)
    return pd.DataFrame({"snow": snow & have, "provisional": prov & have}, index=days)


def snow_epochs(
    sta: str,
    directory: str | Path,
    positions: pd.DataFrame,
    params: SnowParams | None = None,
) -> np.ndarray:
    """Noon fractional-year epochs of a station's snow days (empty without
    an autcln harvest for it in ``directory``)."""
    try:
        stats = _ac.read_autcln_stats(sta, directory)
    except FileNotFoundError:
        return np.zeros(0)
    flagged = snow_days(stats, positions, params)
    return _noon_yearf(flagged.index[flagged["snow"].to_numpy()])
