"""SIL catalogue modes, Q floor, alert2sil PGV and duplicate flags (PLAN-seismicity §9e)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import sil_catalog as sc
from geo_dataread.seismic.schema import validate

FIXTURES = Path(__file__).parent / "fixtures" / "sil"
AUG22 = FIXTURES / "2024" / "aug" / "22"

#: an alert2sil row as cleanaut.mag writes it (2026-06-13 layout): it replaces
#: aut.mag #76 (14:42:54.6, reviewed 14:40 event), position rounded to 0.01°
ALERT_LINE = (
    "   9 20240822 144254.6  63.950  -21.310   4.0   3.9 50.50 "
    "/2000/aaa/00/00:00:00/00:00:000\n"
)


def modes(df: pd.DataFrame) -> tuple[int, int]:
    m = df["evaluation_mode"].astype(str)
    return int((m == "manual").sum()), int((m == "automatic").sum())


@pytest.fixture()
def alert_day(tmp_path: Path) -> Path:
    dst = tmp_path / "2024" / "aug" / "22"
    shutil.copytree(AUG22, dst)
    with (dst / "cleanaut.mag").open("a") as fh:
        fh.write(ALERT_LINE)
    return dst


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def test_default_is_catalog_from_aut_mag() -> None:
    df, rep = sc.read_sil_day(AUG22)
    assert modes(df) == (8, 3)
    filt = df.attrs["catalog"]["filters"]
    assert filt["mode"] == "catalog" and filt["automatic_file"] == "aut.mag"
    assert filt["min_quality"] == sc.DEFAULT_MIN_QUALITY == 10.0
    assert validate(df).ok


def test_mode_automatic_keeps_reviewed_events_automatic_rows() -> None:
    df, rep = sc.read_sil_day(AUG22, mode="automatic")
    assert modes(df) == (0, 10)  # every aut.mag row, reviewed or not
    assert "events_reviewed" not in rep.counts or rep.get("events_reviewed") == 0
    assert "/2024/aug/22/14:40:00/42:55:609" in set(df["event_id"])
    assert validate(df).ok


def test_mode_reviewed_and_the_automatic_alias() -> None:
    a, _ = sc.read_sil_day(AUG22, mode="reviewed")
    b, _ = sc.read_sil_day(AUG22, automatic=False)
    assert modes(a) == modes(b) == (8, 0)
    assert a["sil_aut_path"].isna().all() and a["mag_pgv"].isna().all()
    with pytest.raises(ValueError, match="contradicts"):
        sc.read_sil_day(AUG22, mode="automatic", automatic=False)
    with pytest.raises(ValueError, match="mode"):
        sc.read_sil_day(AUG22, mode="all")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Q floor
# --------------------------------------------------------------------------


def test_min_quality_drops_automatic_rows_only() -> None:
    df, rep = sc.read_sil_day(AUG22, min_quality=20)
    assert modes(df) == (8, 0)  # the three automatic rows are Q 10.00
    assert rep.get("below_min_quality") == 3
    df, rep = sc.read_sil_day(AUG22, mode="automatic", min_quality=50)
    assert (df["quality"] >= 50).all()
    assert len(df) == 4 and rep.get("below_min_quality") == 6


def test_min_quality_none_keeps_everything() -> None:
    df, rep = sc.read_sil_day(AUG22, mode="automatic", min_quality=None)
    assert len(df) == 10 and "below_min_quality" not in rep.counts


def test_min_quality_applies_to_a_period(tmp_path: Path) -> None:
    root = tmp_path / "sil"
    shutil.copytree(FIXTURES, root, ignore=shutil.ignore_patterns("README.md"))
    lo, hi = "2024-08-22", "2024-08-23"
    all_, _ = sc.read_sil_catalog(lo, hi, base=root, min_quality=None)
    hi_q, rep = sc.read_sil_catalog(lo, hi, base=root, min_quality=40)
    assert len(all_) - len(hi_q) == rep.get("below_min_quality") == 3
    assert rep.get("window_automatic") == 0 and rep.get("window_reviewed") == 8


def test_missing_quality_is_kept_and_counted() -> None:
    # 1993 layout: aut.mag without Q
    df, rep = sc.read_sil_day(FIXTURES / "1993" / "jan" / "29", min_quality=50)
    assert len(df) == 5 and rep.get("quality_missing_kept") == 5


# --------------------------------------------------------------------------
# alert2sil PGV magnitude
# --------------------------------------------------------------------------


def test_alert2sil_magnitude_goes_to_mag_pgv(alert_day: Path) -> None:
    df, rep = sc.read_sil_day(alert_day)
    plain, _ = sc.read_sil_day(AUG22)
    key = "/2024/aug/22/14:40:00/42:55:609"
    r = df[df["event_id"] == key].iloc[0]
    assert r["evaluation_mode"] == "manual"
    assert r["mag_pgv"] == 3.9  # the reviewed magnitude is untouched
    assert r["magnitude"] == plain[plain["event_id"] == key].iloc[0]["magnitude"]
    assert rep.get("mag_pgv_from_alert2sil") == 1
    auto, _ = sc.read_sil_day(alert_day, mode="automatic")
    a = auto[auto["event_id"] == "/2024/aug/22/14:40:00/42:55:609"].iloc[0]
    assert a["mag_pgv"] == 3.9 and a["magnitude"] == 0.6
    assert a["quality"] == 76.85  # SIL's own Q, not the 50.50 marker
    assert auto["mag_pgv"].notna().sum() == 1


def test_cleanaut_file_marks_its_alert_rows(alert_day: Path) -> None:
    df, rep = sc.read_sil_day(
        alert_day, mode="automatic", automatic_file="cleanaut.mag"
    )
    a = df[df["event_id"].astype(str).str.startswith(sc.ALERT2SIL_PREFIX)].iloc[0]
    assert a["mag_pgv"] == 3.9 and a["quality"] == 50.5
    assert rep.get("alert2sil_rows") == 1


def test_unmatched_alert_row_is_counted(alert_day: Path) -> None:
    text = (alert_day / "cleanaut.mag").read_text()
    (alert_day / "cleanaut.mag").write_text(text.replace("144254.6", "150000.0"))
    df, rep = sc.read_sil_day(alert_day)
    assert rep.get("alert2sil_unmatched") == 1 and df["mag_pgv"].isna().all()


# --------------------------------------------------------------------------
# Duplicate flags
# --------------------------------------------------------------------------

T0 = pd.Timestamp("2026-06-13 20:14:11", tz="UTC")


def _flags(
    dt_s: list[float],
    km_east: list[float],
    auto: list[bool],
    q: list[float],
    tol: sc.MatchTolerance = sc.DEFAULT_DUPLICATE_TOL,
) -> list[object]:
    t = pd.DatetimeIndex([T0 + pd.Timedelta(seconds=s) for s in dt_s])
    lat = np.full(len(dt_s), 64.0)
    lon = -17.0 + np.asarray(km_east) / (sc.KM_PER_DEG * np.cos(np.radians(64.0)))
    out = sc.duplicate_flags(t, lat, lon, auto, q, tol)
    return [None if pd.isna(v) else bool(v) for v in out]


def test_lower_q_of_two_automatic_rows_is_flagged() -> None:
    assert _flags([0, 1], [0, 2], [True, True], [30, 60]) == [True, False]


def test_automatic_row_next_to_a_reviewed_event_is_flagged() -> None:
    assert _flags([0, 0.5], [0, 1], [False, True], [np.nan, 99]) == [None, True]


def test_two_reviewed_rows_are_never_flagged() -> None:
    assert _flags([0, 0.5], [0, 1], [False, False], [np.nan, np.nan]) == [None, None]


def test_tie_flags_the_later_row_and_missing_q_is_lowest() -> None:
    assert _flags([0, 1], [0, 1], [True, True], [40, 40]) == [False, True]
    assert _flags([0, 1], [0, 1], [True, True], [np.nan, 12]) == [True, False]


def test_swarm_neighbours_outside_the_tolerance_stay() -> None:
    # 3 s apart, or 15 km apart: two events, not one
    assert _flags([0, 3], [0, 1], [True, True], [30, 60]) == [False, False]
    assert _flags([0, 1], [0, 15], [True, True], [30, 60]) == [False, False]


def test_flags_follow_input_order_and_wider_lags() -> None:
    # unsorted input; the pair (0 s, 1.5 s) is two lags apart in time order
    got = _flags([1.5, 0.0, 0.7], [0, 0, 30], [True, True, True], [20, 70, 50])
    assert got == [True, False, False]


def test_no_duplicate_tol_leaves_na() -> None:
    df, rep = sc.read_sil_day(AUG22, duplicate_tol=None)
    assert df["possible_duplicate"].isna().all()
    assert "possible_duplicate" not in rep.counts
