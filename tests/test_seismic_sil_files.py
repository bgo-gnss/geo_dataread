"""Per-file SIL parsers (ticket T1 of PLAN-seismicity.md).

Acceptance targets are the T0 fixtures (``tests/fixtures/sil/README.md``) and
the pins in ``tests/test_sil_characterization.py``. Synthetic bad files are
written to ``tmp_path``. No network, no mount.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import sil_files as sf
from geo_dataread.seismic.schema import STRING_DTYPE, TIME_DTYPE

FIXTURES = Path(__file__).parent / "fixtures" / "sil"

KINDS: tuple[sf.FileKind, ...] = sf.FILE_KINDS

#: Rows per file (None = absent), from the T0 README / characterization tests.
EXPECTED_COUNTS: dict[str, tuple[int | None, ...]] = {
    "1993/jan/29": (0, 0, 0, 5, None),
    "1995/jun/15": (4, 4, 4, 5, None),
    "2022/dec/12": (4, 4, 4, 3, 3),
    "2024/aug/22": (8, 8, 8, 10, 8),
    "2025/may/03": (6, 6, 6, 7, 7),
    "2026/sep/10": (None, 0, None, 6, 6),
    "2026/sep/11": (None, 0, None, 0, 0),
}

#: The layout each fixture file must be detected as.
EXPECTED_LAYOUT: dict[str, str] = {
    "1993/jan/29/aut.mag": "aut.mag-117",
    "1995/jun/15/aut.mag": "aut.mag-61",
    "1995/jun/15/events.lib": "events.lib-136",
    "1995/jun/15/lib.mag": "lib.mag-141",
    "2024/aug/22/aut.mag": "aut.mag-89",
    "2024/aug/22/cleanaut.mag": "aut.mag-89",
    "2024/aug/22/events.fps": "events.fps-33",
}


def _cases() -> list[tuple[str, sf.FileKind, int | None]]:
    return [
        (day, kind, n)
        for day, counts in EXPECTED_COUNTS.items()
        for kind, n in zip(KINDS, counts, strict=True)
    ]


def read(day: str, kind: str) -> sf.ParsedFile:
    return sf.read_sil_file(FIXTURES / day / kind)


def by_num(df: pd.DataFrame, num: int) -> pd.Series:
    hit = df[df["num"] == num]
    assert len(hit) == 1
    return hit.iloc[0]


def utc(*args: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(*args, tzinfo=UTC))  # type: ignore[misc]


# --------------------------------------------------------------------------
# Every fixture file: counts, dtypes, columns, clean report
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("day", "kind", "n"), _cases())
def test_every_fixture_file(day: str, kind: sf.FileKind, n: int | None) -> None:
    path = FIXTURES / day / kind
    if n is None:
        assert not path.exists()
        with pytest.raises(FileNotFoundError):
            sf.read_sil_file(path)
        return
    frame, rep = sf.read_sil_file(path)
    assert rep.kind == kind
    assert rep.rows == len(frame) == n
    assert rep.ok and rep.n_malformed == 0
    assert rep.lines == rep.blank_lines + rep.rows + rep.n_malformed
    assert tuple(frame.columns) == sf.output_columns(kind)
    assert frame["time"].dtype == TIME_DTYPE
    assert frame["line_no"].dtype == np.dtype("int64")
    assert frame["num"].dtype == pd.Int64Dtype()
    assert frame["layout"].dtype == STRING_DTYPE
    if "id" in frame:
        assert frame["id"].dtype == STRING_DTYPE
    assert isinstance(frame.index, pd.RangeIndex)
    if n == 0:
        assert rep.layout == "empty" and rep.layout_counts == {}
    else:
        assert frame["time"].notna().all()
        assert frame["line_no"].is_monotonic_increasing
        assert (frame["time"].dt.strftime("%Y/%b/%d").str.lower() == day).all()


@pytest.mark.parametrize(("rel", "layout"), list(EXPECTED_LAYOUT.items()))
def test_layout_detected(rel: str, layout: str) -> None:
    frame, rep = sf.read_sil_file(FIXTURES / rel)
    assert rep.layout == layout
    assert rep.layout_counts == {layout: rep.rows}
    assert set(frame["layout"]) == {layout}


@pytest.mark.parametrize("kind", KINDS)
def test_zero_byte_file_gives_typed_empty_frame(
    tmp_path: Path, kind: sf.FileKind
) -> None:
    path = tmp_path / kind
    path.write_bytes(b"")
    frame, rep = sf.read_sil_file(path)
    assert frame.empty and tuple(frame.columns) == sf.output_columns(kind)
    assert frame["time"].dtype == TIME_DTYPE
    assert (rep.lines, rep.rows, rep.layout, rep.ok) == (0, 0, "empty", True)
    if kind == "events.fps":
        assert frame["has_solution"].dtype == np.dtype("bool")


def test_unknown_file_name_is_refused(tmp_path: Path) -> None:
    (tmp_path / "events.aut").write_text("x\n")
    with pytest.raises(ValueError, match="not a SIL day catalogue file"):
        sf.read_sil_file(tmp_path / "events.aut")


# --------------------------------------------------------------------------
# Known rows
# --------------------------------------------------------------------------


def test_events_fps_known_rows() -> None:
    df, _ = read("2024/aug/22", "events.fps")
    r = by_num(df, 1)
    assert r["time"] == utc(2024, 8, 22, 0, 10, 4, 454000)
    assert (r["latitude"], r["d_latitude"]) == (63.88752, 0.002)
    assert (r["longitude"], r["d_longitude"]) == (-22.37545, 0.003)
    assert (r["depth"], r["d_depth_mod100"], r["mlw"]) == (2.111, 0.6, 0.2)
    assert r["id"] == "/2024/aug/22/00:05:00/10:05:430"
    assert r["seismic_moment"] == 1.77e10
    assert (r["ev_mag_ath"], r["event_type"], r["source_file"]) == (0.19, "qu", "sp3")
    planes = [
        r[c] for c in ("strike_a", "dip_a", "rake_a", "strike_b", "dip_b", "rake_b")
    ]
    assert planes == [341, 74, 170, 74, 81, 16]
    assert (r["num_of_polarities"], r["num_of_amplitudes"]) == (0, 24)
    assert (r["unknown_pacc"], r["is_significant"], r["has_solution"]) == (
        100.0,
        1,
        True,
    )
    assert (r["sh_min"], r["sh_max"], r["azim_comp"]) == (-0.994, 0.905, 27.0)
    assert (r["relative_size"], r["corner_freq"], r["fault_radius"]) == (0.95, 6.5, 122)
    assert (r["stress_drop"], r["peak_slip"]) == (0.004, 0.03)

    r = by_num(df, 53)  # negative strike on a vertical plane, kept as written
    assert (r["strike_a"], r["dip_a"], r["rake_a"]) == (-86, 90, 28)
    assert (r["strike_b"], r["dip_b"], r["rake_b"]) == (184, 62, 180)
    assert (r["mlw"], r["num_of_polarities"], r["num_of_amplitudes"]) == (1.8, 10, 60)

    df, _ = read("1995/jun/15", "events.fps")
    r = by_num(df, 1)
    assert r["time"] == utc(1995, 6, 15, 1, 7, 33, 724000)
    assert (r["d_depth_mod100"], r["mlw"]) == (42.5, -0.4)
    assert (r["unknown_pacc"], r["is_significant"]) == (44.04, 0)


def test_events_lib_known_rows() -> None:
    df, _ = read("2024/aug/22", "events.lib")
    r = by_num(df, 1)
    assert r["time"] == utc(2024, 8, 22, 0, 10, 4, 454000)
    assert (r["latitude"], r["longitude"], r["depth"], r["mlw"]) == (
        63.88752,
        -22.37545,
        2.111,
        0.25,
    )
    assert (r["event_type"], r["id"]) == ("qu", "/2024/aug/22/00:05:00/10:05:430")
    assert (r["sh_min"], r["sh_max"], r["azim_comp"], r["num_of_polarities"]) == (
        -0.994,
        0.905,
        27.6,
        0,
    )
    # 1995: space-padded seconds, blank event type (NA, not ""), run-together Mlw
    df, _ = read("1995/jun/15", "events.lib")
    r = by_num(df, 4)
    assert r["time"] == utc(1995, 6, 15, 3, 16, 5, 701000)
    assert (r["mlw"], r["azim_comp"]) == (0.4, 39.6)
    assert df["event_type"].isna().all()
    r = by_num(df, 1)
    assert (r["depth"], r["mlw"]) == (6.656, -0.39)
    df, _ = read("2022/dec/12", "events.lib")
    r = by_num(df, 46)
    assert (r["event_type"], r["mlw"], r["num_of_polarities"]) == ("ex", 1.25, 4)


def test_lib_mag_known_rows_and_ml_sentinel() -> None:
    df, rep = read("2024/aug/22", "lib.mag")
    r = by_num(df, 1)
    expected = {
        "mlw": 0.25,
        "ml": -0.5,
        "num_of_st_ml": 4,
        "min_ml": -0.74,
        "max_ml": -0.2,
        "num_of_st_pha": 9,
        "num_of_p_pha": 7,
        "num_of_s_pha": 9,
        "num_of_polarities": 0,
        "d_time": 0.02,
        "d_latitude": 0.002,
        "d_longitude": 0.003,
        "d_depth": 0.6,
        "w_rms_res": 0.06,
        "min_dist_sta": 3.9,
        "max_azim_gap": 102.0,
        "vel_model": 1,
        "event_type": "qu",
    }
    assert {k: r[k] for k in expected} == expected
    # No ML: -9.99 / 9.99 / -9.99 become NA; the station count stays 0.
    r = by_num(df, 42)
    assert pd.isna(r["ml"]) and pd.isna(r["min_ml"]) and pd.isna(r["max_ml"])
    assert (r["num_of_st_ml"], r["mlw"]) == (0, 0.6)
    assert rep.sentinels == {"ml": 1, "min_ml": 1, "max_ml": 1}
    assert not ((df[["ml", "min_ml", "max_ml"]].abs() == 9.99).any().any())

    df, _ = read("1995/jun/15", "lib.mag")
    r = by_num(df, 1)
    assert (r["d_depth"], r["max_azim_gap"], r["vel_model"]) == (242.5, 360.0, 1)
    df, _ = read("2025/may/03", "lib.mag")
    r = by_num(df, 8)
    assert (r["mlw"], r["ml"], r["event_type"], r["vel_model"]) == (3.09, 2.32, "ql", 4)


def test_aut_and_cleanaut_known_rows() -> None:
    df, _ = read("2024/aug/22", "cleanaut.mag")
    r = by_num(df, 7)
    assert r["time"] == utc(2024, 8, 22, 0, 50, 29, 100000)
    assert (r["latitude"], r["longitude"], r["depth"], r["ml"], r["q"]) == (
        63.943,
        -22.402,
        5.2,
        -1.9,
        10.0,
    )
    assert r["id"] == "/2024/aug/22/00:45:00/50:30:280"
    df, _ = read("2026/sep/10", "cleanaut.mag")
    r = by_num(df, 6)
    assert r["time"] == utc(2026, 9, 10, 9, 12, 48, 700000)
    assert r["ml"] == 0.0 and math.copysign(1.0, r["ml"]) == -1.0  # "-0.0" is a number
    assert r["q"] == 17.12

    df, _ = read("2024/aug/22", "aut.mag")
    r = by_num(df, 1)
    assert (r["ml"], r["q"], r["id"]) == (1.5, 10.0, "/2024/aug/22/00:00:00/04:14:727")
    assert r["time"] == utc(2024, 8, 22, 0, 4, 11, 700000)


def test_aut_mag_1995_layout_no_q_no_id() -> None:
    df, _ = read("1995/jun/15", "aut.mag")
    r = by_num(df, 10)
    assert r["time"] == utc(1995, 6, 15, 10, 22, 44, 400000)
    assert (r["latitude"], r["longitude"], r["depth"], r["ml"]) == (
        64.156,
        -20.965,
        5.2,
        0.0,
    )
    assert df["q"].isna().all() and df["id"].isna().all()
    assert df[list(sf.AUX_COLUMNS)].isna().all().all()
    # the legacy spec lost a longitude digit here; every value keeps 3 decimals
    assert (df["longitude"] < -18).all()


def test_aut_mag_1993_layout_unidentified_columns() -> None:
    df, _ = read("1993/jan/29", "aut.mag")
    r = by_num(df, 3)
    assert r["time"] == utc(1993, 1, 29, 19, 10, 32, 700000)
    assert (r["latitude"], r["longitude"], r["depth"]) == (64.24, -20.75, 10.0)
    # T0: "0.94 | 1.81 | — | 0.32 | 0.70", at the measured right-aligned edges
    aux = [None if pd.isna(r[c]) else r[c] for c in sf.AUX_COLUMNS]
    assert aux == [0.94, 1.81, None, None, None, 0.32, None, None, 0.7, None]
    assert df["ml"].isna().all() and df["q"].isna().all() and df["id"].isna().all()
    assert list(df["num"]) == [1, 2, 3, 4, 5]  # "0001" zero padded


def test_aut_and_cleanaut_disagree_on_one_event() -> None:
    a, _ = read("2025/may/03", "aut.mag")
    c, _ = read("2025/may/03", "cleanaut.mag")
    ra, rc = by_num(a, 71), by_num(c, 64)
    assert ra["id"] == rc["id"] == "/2025/may/03/08:15:00/18:43:000"
    assert (ra["q"], rc["q"]) == (90.05, 90.04)
    assert rc["time"] - ra["time"] == pd.Timedelta(milliseconds=100)


@pytest.mark.parametrize(
    "day", ["1995/jun/15", "2022/dec/12", "2024/aug/22", "2025/may/03"]
)
def test_reviewed_files_agree(day: str) -> None:
    """The cross-file pins of T0, through the parsers."""
    fps, _ = read(day, "events.fps")
    lib, _ = read(day, "events.lib")
    mag, _ = read(day, "lib.mag")
    assert list(fps["num"]) == list(lib["num"]) == list(mag["num"])
    assert (fps["time"] == lib["time"]).all() and (lib["time"] == mag["time"]).all()
    assert (fps["id"] == lib["id"]).all()
    assert (lib["mlw"] == mag["mlw"]).all()
    assert ((fps["mlw"] - mag["mlw"]).abs() <= 0.05 + 1e-9).all()
    np.testing.assert_allclose(fps["d_depth_mod100"], mag["d_depth"] % 100)
    assert (fps["num_of_polarities"] == mag["num_of_polarities"]).all()
    assert fps["has_solution"].all()


# --------------------------------------------------------------------------
# Synthetic files: malformed lines, sentinels, layouts
# --------------------------------------------------------------------------


def _fixture_lines(rel: str) -> list[str]:
    return (FIXTURES / rel).read_text(encoding="ascii").splitlines()


def test_malformed_lines_are_reported_not_dropped_silently(tmp_path: Path) -> None:
    good = _fixture_lines("2024/aug/22/aut.mag")[:2]
    bad_width = good[0][:70]
    bad_date = good[0][:4] + " 2024x822" + good[0][13:]
    bad_q = good[1][:51] + "  x.xx" + good[1][57:]
    path = tmp_path / "aut.mag"
    path.write_text(
        "\n".join([good[0], bad_width, "", bad_date, bad_q, good[1]]) + "\n"
    )
    frame, rep = sf.read_aut_mag(path)
    assert rep.lines == 6 and rep.blank_lines == 1
    assert [(m.line_no, m.field) for m in rep.malformed] == [(2, None), (4, "date")]
    assert rep.malformed[0].reason == "unrecognized width 70"
    assert rep.malformed[1].text == bad_date
    # an unreadable optional field: the row stays, the field is NA, a warning says so
    assert [(w.line_no, w.field) for w in rep.field_warnings] == [(5, "q")]
    assert list(frame["line_no"]) == [1, 5, 6]
    assert pd.isna(frame.loc[1, "q"]) and frame.loc[1, "ml"] == -1.2
    assert rep.rows == 3 and not rep.ok
    assert rep.lines == rep.blank_lines + rep.rows + rep.n_malformed


def test_bad_time_and_rollover(tmp_path: Path) -> None:
    line = _fixture_lines("2024/aug/22/aut.mag")[0]  # 000411.7
    roll = line[:13] + " 000460.0" + line[22:]
    bad = line[:13] + " 006011.7" + line[22:]
    path = tmp_path / "aut.mag"
    path.write_text("\n".join([roll, bad]) + "\n")
    frame, rep = sf.read_aut_mag(path)
    assert frame.loc[0, "time"] == utc(2024, 8, 22, 0, 5, 0)
    assert rep.time_rollovers == 1
    assert [(m.line_no, m.field) for m in rep.malformed] == [(2, "time")]


def test_fps_wrong_field_count_is_malformed(tmp_path: Path) -> None:
    lines = _fixture_lines("2024/aug/22/events.fps")[:2]
    joined = lines[1].replace("63.87000 0.004", "63.870000.004")
    path = tmp_path / "events.fps"
    path.write_text("\n".join([lines[0], joined]) + "\r\n")
    frame, rep = sf.read_events_fps(path)
    assert len(frame) == 1
    assert rep.malformed[0].line_no == 2
    assert rep.malformed[0].reason == "32 fields, expected 33"


#: A real no-solution row (2023-12-22 #110) and one with real Mlw/moment
#: (1995 pattern), both "FILE".
NO_SOLUTION = (
    "110 20231222 123940.000 64.82910 -9.000 -23.73240 -9.000   5.244  -9.0 -9.00 "
    "/2023/dec/22/12:35:00/39:46:905 1.00e+01 -9.00 qu -9.000 -9.000  -9.0 9.00 "
    "-9.0   -9 -9.000 -9.000 FILE   -9  -9 -999   -9  -9 -999 -9 -9  -9.00 -9"
)
NO_SOLUTION_REAL_MAG = (
    "46 19950430 012659.131 64.06314 0.031 -21.18253 0.076   5.244   0.0 -0.34 "
    "/1995/apr/30/01:20:00/27:02:129 4.57e+09  0.00 qu -9.000 -9.000  -9.0 9.00 "
    "-9.0   -9 -9.000 -9.000 FILE   -9  -9 -999   -9  -9 -999 -9 -9  -9.00 -9"
)
#: A real solution with values that look like sentinels but are not:
#: azim_comp -9.0 (truncated), rake -9 and -99, corner frequency 9.9 Hz.
GENUINE_NINES = (
    "36 20150401 151843.971 64.82800 0.026 -16.89424 0.025   4.442   7.2  0.60 "
    "/2015/apr/01/15:15:00/18:49:828 3.85e+10  0.13 qu -1.000 0.982  -9.0 0.99  "
    "9.9  147 0.005  0.030 sp3   -9  84  -9  216  85  -99  1 12 100.00 1"
)


def test_fps_no_solution_sentinels(tmp_path: Path) -> None:
    path = tmp_path / "events.fps"
    path.write_text(
        "\n".join([NO_SOLUTION, NO_SOLUTION_REAL_MAG, GENUINE_NINES]) + "\n"
    )
    df, rep = sf.read_events_fps(path)
    assert rep.ok and len(df) == 3
    assert list(df["has_solution"]) == [False, False, True]
    block = list(sf.FPS_NO_SOLUTION_SENTINELS)
    assert df.loc[0:1, block].isna().all().all()
    nosol = df.loc[0]
    for col in (
        "mlw",
        "seismic_moment",
        "ev_mag_ath",
        "d_latitude",
        "d_longitude",
        "d_depth_mod100",
    ):
        assert pd.isna(nosol[col]), col
    assert (nosol["source_file"], nosol["depth"]) == ("FILE", 5.244)
    real = df.loc[1]
    assert (real["mlw"], real["seismic_moment"], real["d_depth_mod100"]) == (
        -0.34,
        4.57e9,
        0.0,
    )
    ok = df.loc[2]
    assert (ok["azim_comp"], ok["strike_a"], ok["rake_a"], ok["rake_b"]) == (
        -9.0,
        -9,
        -9,
        -99,
    )
    assert (ok["corner_freq"], ok["relative_size"]) == (9.9, 0.99)
    assert rep.sentinels["rake_a"] == 2 and rep.sentinels["mlw"] == 1
    assert rep.sentinels["seismic_moment"] == 1


def test_lib_mag_mlw_sentinel_and_134_layout(tmp_path: Path) -> None:
    """1990-91 lib.mag: 134 wide (no w_rms_res); Mlw -9.99 = no Mlw."""
    line = (
        "   1 19910115 043205.944 62.43338 -22.77587  0.062 -9.99  1.82    6 -0.84"
        "  2.15  6  6  4   0  0.29 0.015 0.027   0.0   26.6 292.3 3 qu"
    )
    assert len(line) == 134
    full = _fixture_lines("2024/aug/22/lib.mag")[0]
    path = tmp_path / "lib.mag"
    path.write_text(line + "\n" + full + "\n")
    df, rep = sf.read_lib_mag(path)
    assert rep.layout == "mixed"
    assert rep.layout_counts == {"lib.mag-134": 1, "lib.mag-141": 1}
    r = df.loc[0]
    assert pd.isna(r["mlw"]) and r["ml"] == 1.82
    assert pd.isna(r["w_rms_res"])
    assert (r["d_depth"], r["min_dist_sta"], r["max_azim_gap"], r["vel_model"]) == (
        0.0,
        26.6,
        292.3,
        3,
    )
    assert r["event_type"] == "qu"
    assert df.loc[1, "w_rms_res"] == 0.06
    assert rep.sentinels == {"mlw": 1}


def test_aut_mag_57_layout_and_mixed_file(tmp_path: Path) -> None:
    """Late 1995 / early 1996 aut.mag: Q but no id, 57 wide; mixes with 61."""
    l57 = "   1 19960115 001535.2  63.928  -21.915   1.1   1.6 20.54"
    assert len(l57) == 57
    l61 = _fixture_lines("1995/jun/15/aut.mag")[0]
    path = tmp_path / "aut.mag"
    path.write_text(l57 + "\n" + l61 + "\n")
    df, rep = sf.read_aut_mag(path)
    assert rep.layout == "mixed" and rep.ok
    assert rep.layout_counts == {"aut.mag-57": 1, "aut.mag-61": 1}
    r = df.loc[0]
    assert (r["latitude"], r["longitude"], r["depth"], r["ml"], r["q"]) == (
        63.928,
        -21.915,
        1.1,
        1.6,
        20.54,
    )
    assert pd.isna(r["id"]) and r["layout"] == "aut.mag-57"
    assert df.loc[1, "layout"] == "aut.mag-61" and pd.isna(df.loc[1, "q"])


def test_cleanaut_accepts_only_the_modern_layout(tmp_path: Path) -> None:
    path = tmp_path / "cleanaut.mag"
    path.write_text(_fixture_lines("1995/jun/15/aut.mag")[0] + "\n")
    df, rep = sf.read_cleanaut_mag(path)
    assert df.empty and rep.malformed[0].reason == "unrecognized width 61"


def test_stripped_trailing_blanks_are_not_guessed(tmp_path: Path) -> None:
    """A width the table does not know is malformed, never the nearest layout."""
    line = _fixture_lines("2024/aug/22/events.lib")[0].rstrip()
    path = tmp_path / "events.lib"
    path.write_text(line + "\n")
    df, rep = sf.read_events_lib(path)
    assert df.empty and rep.n_malformed == 1
    assert rep.malformed[0].reason == f"unrecognized width {len(line)}"


def test_parse_report_summary() -> None:
    _, rep = read("2024/aug/22", "lib.mag")
    assert rep.summary().startswith("lib.mag ")
    assert "rows=8" in rep.summary() and "sentinels=3" in rep.summary()


def test_ids_are_opaque_strings() -> None:
    df, _ = read("2022/dec/12", "events.fps")  # minute 62 in an id
    assert "/2022/dec/12/20:10:00/62:40:000" in set(df["id"])


def test_aut_mag_117_dense_row_and_sentinel(tmp_path: Path) -> None:
    """A 1992 row with most columns filled, and aux -9.99 = none."""
    dense = (
        "0002 19920801 015527.6   64.48    -20.77     8.3         1.18    1.25"
        "  0.89  1.15  1.55  1.18  1.19        1.04      "
    )
    nodepth = dense[:40] + " " * 8 + "        -9.99" + dense[61:]
    assert len(dense) == len(nodepth) == 117
    path = tmp_path / "aut.mag"
    path.write_text(dense + "\n" + nodepth + "\n")
    df, rep = sf.read_aut_mag(path)
    assert rep.ok and rep.layout == "aut.mag-117"
    aux = [None if pd.isna(v) else v for v in df.loc[0, list(sf.AUX_COLUMNS)]]
    assert aux == [1.18, 1.25, 0.89, 1.15, 1.55, 1.18, 1.19, None, 1.04, None]
    assert pd.isna(df.loc[1, "depth"]) and pd.isna(df.loc[1, "aux1"])
    assert rep.sentinels == {"aux1": 1}


def test_nan_text_overflow_and_aut_ml_sentinel(tmp_path: Path) -> None:
    aut = tmp_path / "aut.mag"
    aut.write_text(
        "  16 19951210 022312.9  66.304  -16.992   7.1   NaN  7.39\n"
        "  17 19951210 022412.9  66.304  -16.992   7.1  -9.0  7.39\n"
    )
    df, rep = sf.read_aut_mag(aut)
    assert rep.ok and df["ml"].isna().all() and list(df["q"]) == [7.39, 7.39]
    assert rep.sentinels == {"ml": 2}
    mag = tmp_path / "lib.mag"
    line = _fixture_lines("2024/aug/22/lib.mag")[0]
    mag.write_text(line[:67] + " *****" + line[73:] + "\n")
    df, rep = sf.read_lib_mag(mag)
    assert rep.ok and pd.isna(df.loc[0, "min_ml"]) and df.loc[0, "max_ml"] == -0.2
    assert rep.overflows == {"min_ml": 1} and rep.sentinels == {}


def test_fps_xx_event_type_is_na(tmp_path: Path) -> None:
    line = _fixture_lines("2024/aug/22/events.fps")[0].replace(" qu ", " XX ")
    path = tmp_path / "events.fps"
    path.write_text(line + "\n")
    df, rep = sf.read_events_fps(path)
    assert pd.isna(df.loc[0, "event_type"]) and rep.sentinels == {"event_type": 1}
