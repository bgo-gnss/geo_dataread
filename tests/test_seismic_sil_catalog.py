"""SIL day and period assembly (ticket T3 of PLAN-seismicity.md).

Targets are the T0 fixtures (``tests/fixtures/sil/README.md``, "Joins"
table) and the T9 overlap notes (``tests/fixtures/quakes/README.md``).
Period walks run on synthetic trees in ``tmp_path``. No network; the one
real-data test skips without the archive.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import sil_catalog as sc
from geo_dataread.seismic.schema import ATTRS_KEY, TIME_DTYPE, column_names, validate

FIXTURES = Path(__file__).parent / "fixtures" / "sil"

#: day → (total, manual, automatic), from the README Joins table.
EXPECTED: dict[str, tuple[int, int, int]] = {
    "2024/aug/22": (10, 8, 2),
    "2025/may/03": (7, 6, 1),
    "2022/dec/12": (5, 4, 1),
    "1995/jun/15": (5, 4, 1),
    "1993/jan/29": (5, 0, 5),
    "2026/sep/10": (6, 0, 6),
    "2026/sep/11": (0, 0, 0),
}


def day(rel: str, **kw: object) -> tuple[pd.DataFrame, sc.SilReadReport]:
    return sc.read_sil_day(FIXTURES / rel, **kw)  # type: ignore[arg-type]


def at(df: pd.DataFrame, event_id: str) -> pd.Series:
    rows = df[df["event_id"] == event_id]
    assert len(rows) == 1, event_id
    return rows.iloc[0]


def modes(df: pd.DataFrame) -> tuple[int, int]:
    m = df["evaluation_mode"].astype(str)
    return int((m == "manual").sum()), int((m == "automatic").sum())


# --------------------------------------------------------------------------
# Every fixture day
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rel", sorted(EXPECTED))
def test_fixture_day_counts_and_validation(rel: str) -> None:
    df, rep = day(rel)
    total, manual, automatic = EXPECTED[rel]
    assert len(df) == total
    assert modes(df) == (manual, automatic)
    assert rep.get("events_reviewed") == manual
    assert rep.get("events_automatic") == automatic
    v = validate(df)
    assert v.ok, v.summary()
    assert not v.warnings, v.summary()
    assert list(df.columns) == list(column_names())
    assert df.index.dtype == TIME_DTYPE
    assert (df["source"].astype(str) == "sil-files").all()
    assert df.index.is_monotonic_increasing
    prov = df.attrs[ATTRS_KEY]
    assert prov["source"] == "sil-files"
    assert prov["location"].endswith(rel)
    json.dumps(prov)  # JSON-safe for the T4 Parquet cache


@pytest.mark.parametrize("rel", ["2024/aug/22", "2025/may/03", "2022/dec/12"])
def test_sil_path_is_the_file_id(rel: str) -> None:
    df, _ = day(rel)
    assert df["sil_path"].notna().all()
    assert (df["sil_path"] == df["event_id"]).all()
    assert df["sil_path"].str.startswith("/" + rel + "/").all()


def test_reviewed_columns_2024_08_22_row1() -> None:
    df, _ = day("2024/aug/22")
    r = at(df, "/2024/aug/22/00:05:00/10:05:430")
    assert df.index[df["event_id"] == r["event_id"]][0] == pd.Timestamp(
        "2024-08-22 00:10:04.454", tz="UTC"
    )
    assert r["evaluation_mode"] == "manual"
    assert (r["latitude"], r["longitude"], r["depth_km"]) == (
        63.88752,
        -22.37545,
        2.111,
    )
    assert r["magnitude"] == 0.25 and r["magnitude_type"] == "Mlw"  # lib.mag, 0.01
    assert r["mag_mlw"] == 0.25 and r["mag_ml"] == -0.50
    assert r["moment_nm"] == pytest.approx(1.77e10)
    assert r["depth_err_km"] == 0.6
    assert r["n_phases"] == 16  # n_P 7 + n_S 9 = SeisComP usedPhaseCount
    assert r["azimuthal_gap"] == 102.0
    expect_h = max(0.002 * 111.19, 0.003 * 111.19 * math.cos(math.radians(63.88752)))
    assert r["h_err_km"] == pytest.approx(expect_h)
    assert r["event_type"] == "earthquake" and r["event_type_code"] == "qu"
    assert pd.isna(r["quality"])
    # matched automatic (cleanaut) row: same id, its ML kept as mag_auto
    assert r["sil_aut_path"] == r["sil_path"]
    assert r["mag_auto"] == -1.2


def test_lib_mag_no_ml_sentinel_keeps_mlw() -> None:
    df, _ = day("2024/aug/22")
    r = at(df, "/2024/aug/22/07:50:00/54:44:219")  # lib.mag #42, ML -9.99
    assert pd.isna(r["mag_ml"])
    assert r["magnitude"] == 0.60 and r["magnitude_type"] == "Mlw"


def test_automatic_events_labelled() -> None:
    df, rep = day("2024/aug/22")
    aut = df[df["evaluation_mode"].astype(str) == "automatic"]
    assert set(aut["event_id"]) == {
        "/2024/aug/22/00:45:00/50:30:280",
        "/2024/aug/22/03:25:00/31:59:429",
    }
    r = at(df, "/2024/aug/22/00:45:00/50:30:280")  # cleanaut #7, SIL1352541
    assert df.index[df["event_id"] == r["event_id"]][0] == pd.Timestamp(
        "2024-08-22 00:50:29.1", tz="UTC"
    )
    assert r["magnitude"] == -1.9 and r["magnitude_type"] == "aut.mag"
    assert r["mag_auto"] == -1.9 and pd.isna(r["mag_mlw"]) and pd.isna(r["mag_ml"])
    assert r["quality"] == 10.0
    assert r["event_type"] == "earthquake" and pd.isna(r["event_type_code"])
    assert r["sil_path"] == r["sil_aut_path"] == r["event_id"]
    assert pd.isna(r["moment_nm"]) and pd.isna(r["n_phases"])
    assert rep.get("automatic_from:cleanaut.mag") == 8


def test_aut_mag_only_rows_are_counted_not_added() -> None:
    df, rep = day("2024/aug/22")
    # 00:04:11.7 is in aut.mag only and in no API: not an event here
    assert "/2024/aug/22/00:00:00/04:14:727" not in set(df["event_id"])
    assert rep.get("aut_mag_only") == 1
    # 14:40 reviewed: its automatic row is in aut.mag only, matched by id there
    r = at(df, "/2024/aug/22/14:40:00/42:55:609")
    assert r["sil_aut_path"] == "/2024/aug/22/14:40:00/42:55:609"
    assert r["mag_auto"] == 0.6
    assert rep.get("match_id_aut_mag") == 1
    # 21:20 reviewed: no automatic row anywhere
    r = at(df, "/2024/aug/22/21:20:00/26:39:991")
    assert pd.isna(r["sil_aut_path"]) and pd.isna(r["mag_auto"])
    assert rep.get("match_id") == 6
    assert rep.get("reviewed_unmatched") == 1


def test_one_ms_id_mismatch_2025_05_03() -> None:
    df, rep = day("2025/may/03")
    ids = set(df["event_id"])
    assert "/2025/may/03/08:15:00/20:25:007" in ids
    assert "/2025/may/03/08:15:00/20:25:008" not in ids  # not double-counted
    r = at(df, "/2025/may/03/08:15:00/20:25:007")
    assert r["evaluation_mode"] == "manual"
    assert r["sil_aut_path"] == "/2025/may/03/08:15:00/20:25:008"
    assert r["mag_auto"] == -0.1
    assert rep.get("match_id") == 5
    assert rep.get("match_time_space") == 1
    assert len(rep.fallback_dt_s) == 1 and rep.fallback_dt_s[0] < 0.01
    # the other automatic-only event stays
    assert at(df, "/2025/may/03/07:40:00/43:38:852")["evaluation_mode"] == "automatic"


def test_without_fallback_the_twin_is_counted_twice() -> None:
    off = sc.MatchTolerance(0.0, 0.0)
    df, rep = day("2025/may/03", match_tol=off)
    assert len(df) == 8
    assert rep.get("match_time_space") == 0


def test_event_types_2025_and_2022() -> None:
    df, _ = day("2025/may/03")
    codes = dict(zip(df["event_id"], df["event_type_code"], strict=True))
    assert codes["/2025/may/03/00:30:00/38:28:438"] == "ql"
    assert codes["/2025/may/03/08:10:00/15:02:569"] == "sf"
    assert codes["/2025/may/03/08:15:00/18:43:000"] == "te"
    assert codes["/2025/may/03/09:55:00/57:41:257"] == "un"
    r = at(df, "/2025/may/03/00:30:00/38:28:438")
    assert r["event_type"] == "other"
    assert (r["magnitude"], r["mag_ml"]) == (3.09, 2.32)
    df, _ = day("2022/dec/12")
    r = at(df, "/2022/dec/12/15:55:00/57:46:405")
    assert r["event_type"] == "explosion" and r["event_type_code"] == "ex"
    assert r["magnitude"] == 1.25
    # minute-62 manual id: opaque label, no automatic counterpart
    r = at(df, "/2022/dec/12/20:10:00/62:40:000")
    assert r["evaluation_mode"] == "manual" and pd.isna(r["sil_aut_path"])


def test_1995_no_ids_matched_by_time_and_space() -> None:
    df, rep = day("1995/jun/15")
    assert rep.get("automatic_from:aut.mag") == 5
    assert rep.get("file_missing:cleanaut.mag") == 1
    assert rep.get("match_id") == 0
    assert rep.get("match_time_space") == 4  # aut rows 1-4 = reviewed 1-4
    assert max(rep.fallback_dt_s) < 5.0
    rev = df[df["evaluation_mode"].astype(str) == "manual"]
    assert rev["mag_auto"].notna().all()
    assert rev["sil_aut_path"].isna().all()  # no automatic id in 1995
    aut = df[df["evaluation_mode"].astype(str) == "automatic"]
    assert list(aut["event_id"]) == ["/1995/jun/15/aut.mag#10"]
    assert aut["sil_path"].isna().all()
    assert aut["magnitude"].iloc[0] == 0.0 and pd.isna(aut["quality"].iloc[0])
    r = at(df, "/1995/jun/15/01:05:00/07:38:077")
    assert r["depth_err_km"] == 242.5  # from lib.mag; fps writes 42.5
    assert r["magnitude"] == -0.39  # lib.mag, fps has -0.40
    assert r["event_type_code"] == "qu"  # lib.mag (events.lib is blank)
    t = df.index[df["event_id"] == "/1995/jun/15/03:15:00/16:09:422"][0]
    assert t == pd.Timestamp("1995-06-15 03:16:05.701", tz="UTC")


def test_1993_zero_byte_reviewed_files() -> None:
    df, rep = day("1993/jan/29")
    for kind in ("events.fps", "events.lib", "lib.mag"):
        assert rep.get(f"file_empty:{kind}") == 1
    assert rep.get("days_automatic_only") == 1
    assert (df["evaluation_mode"].astype(str) == "automatic").all()
    assert df["magnitude"].isna().all()  # 117 layout: ml NA
    assert df["magnitude_type"].isna().all()
    assert df["event_id"].str.match(r"^/1993/jan/29/aut\.mag#\d+$").all()


def test_2026_automatic_only_day() -> None:
    df, rep = day("2026/sep/10")
    assert rep.get("file_missing:events.fps") == 1
    assert rep.get("file_missing:lib.mag") == 1
    assert rep.get("file_empty:events.lib") == 1
    assert rep.get("days_automatic_only") == 1
    assert (df["magnitude_type"].astype(str) == "aut.mag").all()
    r = at(df, "/2026/sep/10/09:10:00/12:50:312")  # cleanaut #6
    assert r["quality"] == 17.12 and r["magnitude"] == 0.0


def test_empty_day_validates() -> None:
    df, rep = day("2026/sep/11")
    assert len(df) == 0
    assert validate(df).ok
    assert rep.get("days_empty") == 1


def test_automatic_false_reads_reviewed_only() -> None:
    df, rep = day("2024/aug/22", automatic=False)
    assert modes(df) == (8, 0)
    assert df["sil_aut_path"].isna().all()
    assert rep.get("rows:cleanaut.mag") == 0 and rep.get("rows:aut.mag") == 0
    df, _ = day("2026/sep/10", automatic=False)
    assert len(df) == 0 and validate(df).ok


def test_missing_day_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sc.read_sil_day(tmp_path / "2024" / "aug" / "23")


# --------------------------------------------------------------------------
# Synthetic days
# --------------------------------------------------------------------------

_CLEAN_TEMPLATE = (
    "   1 20260910 014006.2  64.586  -20.316   4.8   1.5 90.01 "
    "/2026/sep/10/01:35:00/40:12:180"
)


def aut_line(
    num: int, ymd: str, hms: str, ml: str, ident: str, q: str = "90.01"
) -> str:
    """An 89-character cleanaut/aut.mag line (fields sliced as in sil_files)."""
    line = f"{num:4d} {ymd} {hms:>8s}  64.586  -20.316   4.8{ml:>6s}{q:>6s} {ident}"
    assert len(line) == 89, (len(line), line)
    return line


def test_aut_line_helper_matches_fixture() -> None:
    made = aut_line(1, "20260910", "014006.2", "1.5", "/2026/sep/10/01:35:00/40:12:180")
    assert made == _CLEAN_TEMPLATE


def write_day(
    base: Path, rel: str, lines: list[str], kind: str = "cleanaut.mag"
) -> Path:
    d = base / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / kind).write_text("".join(ln + "\n" for ln in lines))
    return d


def test_out_of_range_magnitude_becomes_na(tmp_path: Path) -> None:
    d = write_day(
        tmp_path,
        "2026/sep/10",
        [
            aut_line(
                1, "20260910", "014006.2", "12.6", "/2026/sep/10/01:35:00/40:12:180"
            ),
            aut_line(
                2, "20260910", "024006.2", "1.0", "/2026/sep/10/02:35:00/40:12:180"
            ),
        ],
    )
    df, rep = sc.read_sil_day(d)
    assert len(df) == 2 and validate(df).ok
    assert rep.get("out_of_range:magnitude") == 1
    assert rep.get("out_of_range:mag_auto") == 1
    first = df.iloc[0]
    assert pd.isna(first["magnitude"]) and pd.isna(first["magnitude_type"])


def test_reviewed_from_events_lib_alone(tmp_path: Path) -> None:
    """No fps and no lib.mag (as in 1990): events.lib is the spine."""
    d = tmp_path / "2024" / "aug" / "22"
    d.mkdir(parents=True)
    shutil.copy(FIXTURES / "2024/aug/22/events.lib", d / "events.lib")
    df, rep = sc.read_sil_day(d)
    assert modes(df) == (8, 0)
    assert validate(df).ok
    assert rep.get("lib_only") == 8
    assert rep.get("reviewed_without_lib_mag") == 8
    assert rep.get("mlw_from:events.lib") == 8
    assert df["depth_err_km"].isna().all() and df["n_phases"].isna().all()
    assert df["moment_nm"].isna().all()


def test_fps_without_lib_mag_flags_unsafe_depth_error(tmp_path: Path) -> None:
    d = tmp_path / "1995" / "jun" / "15"
    d.mkdir(parents=True)
    for kind in ("events.fps", "events.lib"):
        shutil.copy(FIXTURES / "1995/jun/15" / kind, d / kind)
    df, rep = sc.read_sil_day(d)
    assert len(df) == 4
    assert df["depth_err_km"].isna().all()  # fps Ddepth is mod 100: not used
    assert rep.get("depth_err_unsafe") == 4
    assert rep.get("file_missing:lib.mag") == 1
    assert rep.get("mlw_from:events.lib") == 4


def test_lib_mag_without_spine_is_counted(tmp_path: Path) -> None:
    d = tmp_path / "2024" / "aug" / "22"
    d.mkdir(parents=True)
    shutil.copy(FIXTURES / "2024/aug/22/lib.mag", d / "lib.mag")
    df, rep = sc.read_sil_day(d)
    assert len(df) == 0
    assert rep.get("lib_mag_unjoined") == 8


def test_fps_lib_time_conflict_joins_by_id(tmp_path: Path) -> None:
    d = tmp_path / "2024" / "aug" / "22"
    d.mkdir(parents=True)
    for kind in ("events.fps", "lib.mag"):
        shutil.copy(FIXTURES / "2024/aug/22" / kind, d / kind)
    lines = (FIXTURES / "2024/aug/22/events.lib").read_text().splitlines()
    # shift the first event's time by 1 ms in events.lib only
    lines[0] = (
        lines[0][:13]
        + lines[0][13:24].replace("001004.454", "001004.455")
        + lines[0][24:]
    )
    (d / "events.lib").write_text("".join(ln + "\n" for ln in lines))
    df, rep = sc.read_sil_day(d, automatic=False)
    assert len(df) == 8
    assert rep.get("fps_lib_time_conflict") == 1
    assert rep.get("fps_only") == 0 and rep.get("lib_only") == 0


# --------------------------------------------------------------------------
# Period walk
# --------------------------------------------------------------------------


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Three consecutive days across a month boundary, one missing day after."""
    write_day(
        tmp_path,
        "2024/aug/30",
        [aut_line(1, "20240830", "120000.0", "1.0", "/2024/aug/30/11:55:00/00:01:000")],
    )
    write_day(
        tmp_path,
        "2024/aug/31",
        [
            aut_line(
                1, "20240831", "000000.0", "1.1", "/2024/aug/31/23:55:00/00:01:000"
            ),
            aut_line(
                2, "20240831", "235959.9", "1.2", "/2024/aug/31/23:55:00/59:59:000"
            ),
            # an event whose time falls on the next day (off-day)
            aut_line(
                3, "20240901", "000000.5", "1.3", "/2024/aug/31/23:55:00/60:01:000"
            ),
        ],
    )
    write_day(
        tmp_path,
        "2024/sep/01",
        [aut_line(1, "20240901", "060000.0", "1.4", "/2024/sep/01/05:55:00/00:01:000")],
    )
    return tmp_path


def test_period_walk_across_month_boundary(tree: Path) -> None:
    df, rep = sc.read_sil_catalog("2024-08-31", "2024-09-02", base=tree)
    assert list(df["magnitude"]) == [1.1, 1.2, 1.3, 1.4]
    assert validate(df).ok
    # days read: aug 30 (padding), aug 31, sep 1; sep 2 (padding) is missing
    assert rep.get("days_read") == 3
    assert rep.get("days_missing") == 1
    assert rep.get("off_day") == 1
    assert rep.get("window_automatic") == 4 and rep.get("window_reviewed") == 0
    assert rep.get("events_automatic") == 5  # includes the padding day
    prov = df.attrs[ATTRS_KEY]
    assert prov["window"] == {
        "start": "2024-08-31T00:00:00Z",
        "end": "2024-09-02T00:00:00Z",
    }
    assert prov["report"]["counts"]["days_read"] == 3
    json.dumps(prov)


def test_period_window_is_half_open_and_pads_a_day(tree: Path) -> None:
    df, _ = sc.read_sil_catalog("2024-08-31", "2024-09-01", base=tree)
    assert list(df["magnitude"]) == [1.1, 1.2]  # 00:00:00.5 on sep 1 is out
    # the off-day event, filed under aug 31, is found from sep 1's window
    df, _ = sc.read_sil_catalog("2024-09-01", "2024-09-02", base=tree)
    assert list(df["magnitude"]) == [1.3, 1.4]
    # sub-day window with aware times
    df, _ = sc.read_sil_catalog(
        pd.Timestamp("2024-08-31 23:00", tz="UTC"),
        pd.Timestamp("2024-09-01 01:00", tz="UTC"),
        base=tree,
    )
    assert list(df["magnitude"]) == [1.2, 1.3]


def test_period_reports_sum_the_days(tree: Path) -> None:
    _, total = sc.read_sil_catalog("2024-08-30", "2024-09-02", base=tree)
    parts = [
        sc.read_sil_day(tree / rel)[1]
        for rel in ("2024/aug/30", "2024/aug/31", "2024/sep/01")
    ]
    summed = parts[0] + parts[1] + parts[2]
    for key in ("rows:cleanaut.mag", "events_automatic", "off_day"):
        assert total.get(key) == summed.get(key)


def test_period_over_fixture_days(tmp_path: Path) -> None:
    for rel in ("2026/sep/10", "2026/sep/11"):
        shutil.copytree(FIXTURES / rel, tmp_path / rel)
    df, rep = sc.read_sil_catalog("2026-09-10", "2026-09-12", base=tmp_path)
    assert len(df) == 6 and validate(df).ok
    assert rep.get("days_empty") == 1
    df, _ = sc.read_sil_catalog("2026-09-11", "2026-09-12", base=tmp_path)
    assert len(df) == 0 and validate(df).ok


def test_period_end_before_start(tree: Path) -> None:
    with pytest.raises(ValueError):
        sc.read_sil_catalog("2024-09-02", "2024-09-01", base=tree)


# --------------------------------------------------------------------------
# Base path
# --------------------------------------------------------------------------


def test_sil_dir_resolution_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sc, "_config_sil_dir", lambda: str(tmp_path / "cfg"))
    monkeypatch.setenv(sc.ENV_SIL_DIR, str(tmp_path / "env"))
    assert sc.sil_dir(tmp_path / "arg") == tmp_path / "arg"
    assert sc.sil_dir() == tmp_path / "env"
    monkeypatch.delenv(sc.ENV_SIL_DIR)
    assert sc.sil_dir() == tmp_path / "cfg"
    monkeypatch.setattr(sc, "_config_sil_dir", lambda: None)
    with pytest.raises(FileNotFoundError, match="GPS_SIL_DIR"):
        sc.sil_dir()


def test_catalog_uses_env_base(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sc.ENV_SIL_DIR, str(tree))
    df, _ = sc.read_sil_catalog("2024-08-31", "2024-09-01")
    assert len(df) == 2
    assert df.attrs[ATTRS_KEY]["location"] == str(tree)


def test_day_dir_months_are_fixed() -> None:
    assert sc.day_dir("/x", pd.Timestamp("2024-05-03").date()) == Path("/x/2024/may/03")
    assert len(sc.MONTHS) == 12


def test_horizontal_error_km() -> None:
    h = sc.horizontal_error_km([0.002, np.nan], [0.003, 0.001], [60.0, 60.0])
    assert h[0] == pytest.approx(max(0.002 * 111.19, 0.003 * 111.19 * 0.5))
    assert h[1] == pytest.approx(0.001 * 111.19 * 0.5)


# --------------------------------------------------------------------------
# Real data (skips without the archive)
# --------------------------------------------------------------------------

_REAL = os.environ.get(sc.ENV_SIL_DIR) or "/mnt_data/sildata"


@pytest.mark.skipif(
    not (Path(_REAL) / "2024" / "aug" / "22").is_dir(),
    reason="SIL archive not mounted ($GPS_SIL_DIR or /mnt_data/sildata)",
)
def test_real_day_2024_08_22() -> None:
    df, rep = sc.read_sil_day(Path(_REAL) / "2024" / "aug" / "22")
    assert validate(df).ok
    assert rep.get("rows:events.fps") == 250
    manual, automatic = modes(df)
    assert manual == 250
    assert automatic > 0
    assert rep.get("lib_mag_unjoined") == 0
    assert not rep.issues
