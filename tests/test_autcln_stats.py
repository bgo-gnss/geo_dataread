"""GAMIT autcln per-station statistics: parse, harvest, icing index."""

from __future__ import annotations

import datetime as dt
import gzip
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread import autcln_stats as ac

# real lines, okada WEEKS/2439/ice1/277/autcln.post.sum.gz (bins shortened)
SUMMARY = """\
RMS  IT Site   All   01   02   03
RMS  20 AUST   8.4   10    7    9
RMS  20 ALL    9.3   97   79   91
NUM  IT Site   All   01   02   03
NUM  20 AUST 26655  847  699  790
AMS  IT Site   All Ratio    01   02
AMS  20 AUST   6.4   1.32    8    6
ATELV Site    A     B    0-05  5-10 10-15 15-20
ATELV AUST   3.4   3.1    0.0   0.0  15.3  10.7
ELMEAN Site   00    01    02    03    04
ELMEAN AUST  99.9  99.9   1.2   2.3  -1.4
AZMEAN Site   000   005   010   015
AZMEAN AUST   1.2   2.2  -0.8  -1.4
"""


def test_parse_reads_every_statistic():
    got = ac.parse_autcln_post_sum(SUMMARY.splitlines())
    assert set(got) == {"AUST"}  # ALL is the every-site summary, not a station
    s = got["AUST"]
    assert (s.rms, s.num, s.ams, s.ams_ratio) == (8.4, 26655.0, 6.4, 1.32)
    assert (s.atelv_a, s.atelv_b, s.atelv_low) == (3.4, 3.1, 15.3)
    assert s.el_absmean == pytest.approx((1.2 + 2.3 + 1.4) / 3)  # 99.9 = empty
    assert s.az_absmean == pytest.approx((1.2 + 2.2 + 0.8 + 1.4) / 4)


def test_a_value_run_into_the_site_code_is_split():
    """SAFH: an RMS >= 1000 mm fills the field and glues onto the site code."""
    text = "RMS  20 SAFH1001.0   12\nATELV SAFH120.5  30.1   0.0\n"
    got = ac.parse_autcln_post_sum(text.splitlines())
    assert set(got) == {"SAFH"}
    assert (got["SAFH"].rms, got["SAFH"].atelv_a, got["SAFH"].atelv_b) == (
        1001.0,
        120.5,
        30.1,
    )


def test_a_later_iteration_replaces_an_earlier_one():
    text = "RMS  10 AUST   9.9   1\nRMS  20 AUST   8.4   1\n"
    assert ac.parse_autcln_post_sum(text.splitlines())["AUST"].rms == 8.4
    assert math.isnan(ac.parse_autcln_post_sum(text.splitlines())["AUST"].num)


@pytest.mark.parametrize(
    ("week", "doy", "day"),
    [(2439, 277, dt.date(2026, 10, 4)), (2399, 1, dt.date(2026, 1, 1)),
     (2399, 365, dt.date(2025, 12, 31))],
)  # fmt: skip
def test_date_of_week_and_doy(week, doy, day):
    assert ac.date_of(week, doy) == day


def _day(weeks: Path, week: int, net: str, doy: int, text: str) -> Path:
    d = weeks / str(week) / net / f"{doy:03d}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / ac.AUTCLN_FILE
    with gzip.open(f, "wt") as fh:
        fh.write(text)
    return f


def test_harvest_is_incremental_and_replaces_a_remade_day(tmp_path):
    weeks, out = tmp_path / "WEEKS", tmp_path / "pos"
    f = _day(weeks, 2439, "ice1", 277, SUMMARY)
    _day(
        weeks,
        2439,
        "ice1",
        278,
        SUMMARY.replace("RMS  20 AUST   8.4", "RMS  20 AUST   5.8"),
    )
    kw = dict(weeks_dir=weeks, networks=["ice1"], first_week=2439, last_week=2439)
    assert ac.harvest(out, **kw) == {"files": 2, "skipped": 0, "stations": 1}
    assert ac.harvest(out, **kw) == {"files": 0, "skipped": 2, "stations": 0}
    df = ac.read_autcln_stats("AUST", out)
    assert df["rms"].tolist() == [8.4, 5.8]
    # okada re-makes the day: the row is replaced, not duplicated
    with gzip.open(f, "wt") as fh:
        fh.write(SUMMARY.replace("RMS  20 AUST   8.4", "RMS  20 AUST   12.0"))
    os.utime(f, (1, 2_000_000_000))
    assert ac.harvest(out, **kw)["files"] == 1
    assert ac.read_autcln_stats("AUST", out)["rms"].tolist() == [12.0, 5.8]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [ac.STATE_FILE, "AUST.autcln.24hr.csv", ac.NETWORK_FILE]
    )
    assert ac.read_network_median(out).tolist() == [1.4, 1.4]


def test_a_tie_station_in_two_networks_is_their_median(tmp_path):
    weeks, out = tmp_path / "WEEKS", tmp_path / "pos"
    _day(weeks, 2439, "ice1", 277, SUMMARY)
    _day(
        weeks,
        2439,
        "ice2",
        277,
        SUMMARY.replace("RMS  20 AUST   8.4", "RMS  20 AUST   9.4"),
    )
    ac.harvest(
        out, weeks_dir=weeks, networks=["ice1", "ice2"], first_week=2439, last_week=2439
    )
    assert ac.read_autcln_stats("AUST", out)["rms"].tolist() == [pytest.approx(8.9)]


def test_no_harvest_means_no_icing_epochs(tmp_path):
    with pytest.raises(FileNotFoundError):
        ac.read_autcln_stats("AUST", tmp_path)
    assert ac.icing_epochs("AUST", tmp_path).size == 0


def _stats(az: np.ndarray, a: float = 3.0, b: float = 3.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=az.size, freq="D")
    return pd.DataFrame({"az_absmean": az, "atelv_a": a, "atelv_b": b}, index=idx)


def test_icing_days_against_the_stations_own_low_level():
    rng = np.random.default_rng(3)
    az = 1.8 + rng.normal(0.0, 0.2, 400)
    az[200:215] = 7.0  # a two-week episode
    flags = ac.icing_days(_stats(az))
    assert flags.iloc[200:215].all()
    assert flags.sum() == 15


def test_a_long_icing_season_does_not_lift_the_baseline():
    """The low QUANTILE, not a mean: a winter of ice must not raise the bar."""
    az = np.full(400, 1.8)
    az[100:220] = 6.0  # four months of ice
    assert ac.icing_days(_stats(az)).iloc[100:220].all()


def test_the_sigelv_clip_flags_regardless():
    az = np.full(100, 1.8)
    df = _stats(az)
    df.iloc[50, df.columns.get_loc("atelv_a")] = 60.0
    flags = ac.icing_days(df)
    assert flags.iloc[50] and flags.sum() == 1


def test_icing_epochs_are_noon_yearf(tmp_path):
    out = tmp_path
    rows = ["date,net," + ",".join(ac.STAT_FIELDS)]
    for i, day in enumerate(pd.date_range("2026-01-01", periods=120, freq="D")):
        az = 9.0 if i == 60 else 1.5
        rows.append(f"{day.date()},ice1,7,26000,5,1.5,3,3,12,{az},0.8")
    ac.stats_path(out, "AUST").write_text("\n".join(rows) + "\n")
    ep = ac.icing_epochs("AUST", out)
    assert ep.tolist() == [pytest.approx(2026 + (61 - 0.5) / 365)]


def test_a_network_wide_day_cancels_when_normalised():
    """2026-04-30 flagged 66 % of stations un-normalised: ionosphere, not ice."""
    rng = np.random.default_rng(4)
    az = 1.8 + rng.normal(0.0, 0.1, 400)
    net = pd.Series(1.8, index=pd.date_range("2025-01-01", periods=400, freq="D"))
    az[150] = net.iloc[150] = 6.0  # the whole network that day
    az[300:310] = 6.0  # this station alone
    df = _stats(az)
    raw = ac.icing_days(df)
    normed = ac.icing_days(df, network=net)
    assert raw.iloc[150] and not normed.iloc[150]
    assert normed.iloc[300:310].all() and normed.sum() == 10


def test_write_network_median(tmp_path):
    for sta, az in (("AAAA", 1.0), ("BBBB", 2.0), ("CCCC", 9.0)):
        rows = [
            "date,net," + ",".join(ac.STAT_FIELDS),
            f"2026-01-01,ice1,7,26000,5,1.5,3,3,12,{az},0.8",
        ]
        ac.stats_path(tmp_path, sta).write_text("\n".join(rows) + "\n")
    ac.write_network_median(tmp_path)
    assert ac.read_network_median(tmp_path).tolist() == [2.0]


def test_an_episode_lasts_while_above_the_exit_ratio():
    """GFUM 2025-12-06..11: 1.5-2.0x the baseline inside an obvious episode."""
    az = np.full(300, 1.0)
    az[100:105] = 3.0  # starts (> 2x)
    az[105:111] = 1.7  # dips, still > 1.5x: stays an episode
    az[111:115] = 3.0
    az[200:206] = 1.7  # 1.7 alone never STARTS one
    flags = ac.icing_days(_stats(az))
    assert flags.iloc[100:115].all() and flags.sum() == 15


def test_short_gaps_are_bridged_and_missing_days_flagged():
    az = np.full(300, 1.0)
    az[100:110] = 4.0
    az[112:120] = 4.0  # two days back at baseline: bridged
    az[130:135] = 4.0  # ten days later: a new episode, not bridged
    df = _stats(az).drop(
        index=pd.date_range("2025-04-15", periods=1)
    )  # a day without statistics
    flags = ac.icing_days(df)
    assert len(flags) == 300  # every calendar day
    assert flags.iloc[100:120].all() and not flags.iloc[121:130].any()
    assert flags.iloc[130:135].all()


def test_a_day_autcln_kept_no_data_for_is_no_statistic(tmp_path):
    rows = ["date,net," + ",".join(ac.STAT_FIELDS),
            "2025-12-15,ice1,8.3,26000,5,1.5,3,3,12,4.5,0.8",
            "2025-12-16,ice1,0,0,0,0,0,0,,,"]  # fmt: skip
    ac.stats_path(tmp_path, "GFUM").write_text("\n".join(rows) + "\n")
    s = ac.read_autcln_stats("GFUM", tmp_path)
    assert [d.day for d in s.index] == [15]
