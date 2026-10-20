"""Snow-burial index: a one-sided Up bump with quiet horizontals and raised A/B."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geo_dataread import gps_views
from geo_dataread import snow_burial as sb

DAYS = pd.date_range("2020-01-01", periods=600, freq="D")


def _positions(up_extra: np.ndarray | None = None, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pos = pd.DataFrame(
        rng.normal(0.0, 2.0, (DAYS.size, 3)), index=DAYS, columns=list("NEU")
    )
    if up_extra is not None:
        pos["U"] += up_extra
    return pos


def _stats(a_high: slice | None = None) -> pd.DataFrame:
    a = np.full(DAYS.size, 3.0)
    if a_high is not None:
        a[a_high] = 7.0
    return pd.DataFrame({"atelv_a": a, "atelv_b": np.full(DAYS.size, 3.0)}, index=DAYS)


def _bump(first: int, peak: int, last: int, height: float) -> np.ndarray:
    """Snow: builds up to ``height`` over first..peak, melts by ``last``."""
    out = np.zeros(DAYS.size)
    out[first:peak] = np.linspace(0.0, height, peak - first)
    out[peak : last + 1] = np.linspace(height, 0.0, last + 1 - peak)
    return out


def test_a_buried_antenna_is_flagged():
    got = sb.snow_days(_stats(slice(300, 345)), _positions(_bump(300, 335, 345, 120.0)))
    snow = got["snow"].to_numpy()
    assert snow[310:335].all()
    assert not snow[:290].any() and not snow[360:].any()
    assert not got["provisional"].any()


def test_without_raised_statistics_nothing_is_flagged():
    """An Up bump alone is not snow (Svartsengi uplift; a reprocessing glitch)."""
    got = sb.snow_days(_stats(), _positions(_bump(300, 335, 345, 120.0)))
    assert not got["snow"].any()


def test_a_ramp_is_not_a_bump():
    """Uplift that keeps its level lies on the chord between the flanks."""
    ramp = np.zeros(DAYS.size)
    ramp[300:345] = np.linspace(0.0, 120.0, 45)
    ramp[345:] = 120.0
    got = sb.snow_days(_stats(slice(300, 345)), _positions(ramp))
    assert not got["snow"].any()


def test_an_episode_ending_in_a_horizontal_jump_is_deformation():
    """SENG 2025-03-31: Up sawtooth down at an eruption, East −188 σ."""
    pos = _positions(_bump(300, 344, 345, 120.0))
    pos.loc[DAYS[346] :, "E"] += 200.0
    got = sb.snow_days(_stats(slice(300, 345)), pos)
    assert not got["snow"].any()


def test_an_episode_at_the_live_end_is_provisional():
    pos = _positions(_bump(560, 599, 599, 120.0))
    got = sb.snow_days(_stats(slice(560, 600)), pos)
    assert got["snow"].iloc[570:].all()
    assert got["provisional"].iloc[570:].all()


def test_a_short_gap_inside_an_episode_is_bridged():
    pos = _positions(_bump(300, 335, 345, 120.0)).drop(index=DAYS[320:323])
    got = sb.snow_days(_stats(slice(300, 345)), pos)
    assert got["snow"].iloc[310:335].sum() == 22  # the 3 missing days stay unflagged
    assert got["snow"].iloc[323:335].all()


def test_no_statistics_no_snow():
    got = sb.snow_days(_stats().iloc[:0], _positions(_bump(300, 335, 345, 120.0)))
    assert not got["snow"].any()


def test_daily_positions_and_noon_epochs_round_trip():
    days = pd.DatetimeIndex(["2024-02-29", "2025-12-31"])
    t = sb._noon_yearf(days)
    assert list(sb._day_of(t)) == list(days)
    pos = sb.daily_positions(t, np.ones((3, 2)))
    assert list(pos.index) == list(days)


def test_the_view_flags_up_only_and_only_when_removing():
    t = sb._noon_yearf(DAYS)
    rng = np.random.default_rng(2)
    y = rng.normal(0.0, 2.0, (3, DAYS.size))
    y[2, 300:310] += 100.0
    snow = t[300:310]
    keep, prov = gps_views.detect_view_outliers(t, y, snow_epochs=snow)
    assert prov["n_snow"] == 10
    assert prov["snow"][2, 300:310].all() and not prov["snow"][:2].any()
    assert not keep[:, 300:310].any()  # kept: the days were never detected on
    removed, _ = gps_views.detect_view_outliers(
        t, y, snow_epochs=snow, snow_mode="remove"
    )
    assert removed[2, 300:310].all() and not removed[:2, 300:310].any()


def test_snow_mode_is_validated():
    with pytest.raises(ValueError, match="snow_mode"):
        gps_views.detect_view_outliers(np.arange(3.0), np.zeros(3), snow_mode="drop")


def test_resolve_snow_epochs_switch(tmp_path):
    assert gps_views.resolve_snow_epochs("OLAC", False) == ((), None)
    assert gps_views.resolve_snow_epochs("OLAC", [2020.5]) == ((2020.5,), "explicit")
    assert gps_views.resolve_snow_epochs("OLAC", tmp_path) == ((), None)  # no harvest
