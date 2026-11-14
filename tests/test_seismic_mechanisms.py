"""Fault planes from the SIL files (ticket T3b of PLAN-seismicity.md).

Fixtures: ``tests/fixtures/sil`` (T0) and the legacy Skjálftalísa recordings
in ``tests/fixtures/skjalftalisa`` (T9). Synthetic days (ap3 blocks, a day
without ``events.fps``, a no-solution row) are written to ``tmp_path`` from
real archive lines embedded below. The real-archive check runs only when
``$GPS_SIL_DIR`` is set. No network.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import mechanisms as mm
from geo_dataread.seismic.schema import (
    TIME_DTYPE,
    column_names,
    validate_mechanisms,
)
from geo_dataread.seismic.sil_files import read_events_fps

FIXTURES = Path(__file__).parent / "fixtures"
SIL = FIXTURES / "sil"
LEGACY = FIXTURES / "skjalftalisa"
REVIEWED_DAYS = ("1995/jun/15", "2022/dec/12", "2024/aug/22", "2025/may/03")
ALL_DAYS = (*REVIEWED_DAYS, "1993/jan/29", "2026/sep/10", "2026/sep/11")


def plane(*v: float) -> tuple[float, float, float]:
    s, d, r = mm.sp3_to_aki_richards(*v)
    return float(s), float(d), float(r)


def fixture_blocks() -> list[tuple[Path, pd.Series]]:
    """Every fixture fps row with its block path."""
    out = []
    for day in REVIEWED_DAYS:
        fps = read_events_fps(SIL / day / "events.fps").frame
        for _, row in fps.iterrows():
            out.append((mm.block_path(SIL / day, row["id"], row["source_file"]), row))
    return out


# --------------------------------------------------------------------------
# Conversion, hand cases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # dip > 90: (s, 180 - d, 180 - r)  — 2024-08-22 #1 plane a
        ((-19, 106, 10), (341, 74, 170)),
        ((74, 99, 164), (74, 81, 16)),
        # dip < 90: (s + 180, d, r + 180), rake wrapped — 1995 ap3 plane a
        ((130, 15, 51), (310, 15, -129)),
        # dip = 90 goes the dip >= 90 way — 2024-08-22 #53 (fps -86/90/28)
        ((-86, 90, 152), (274, 90, 28)),
        # strike wrap past 360
        ((200, 40, -30), (20, 40, 150)),
        # strike wrap below 0 (dip >= 90 keeps s)
        ((-370, 120, 0), (350, 60, 180)),
        # rake exactly -180 becomes +180
        ((10, 30, 0), (190, 30, 180)),
        ((10, 150, 360), (10, 30, 180)),
    ],
)
def test_sp3_to_aki_richards_hand_cases(
    raw: tuple[float, float, float], expected: tuple[float, float, float]
) -> None:
    assert plane(*raw) == pytest.approx(expected)


def test_conversion_ranges_and_bad_dip() -> None:
    rng = np.random.default_rng(3)
    s = rng.uniform(-400, 400, 500)
    d = rng.uniform(0, 180, 500)
    r = rng.uniform(-400, 400, 500)
    cs, cd, cr = mm.sp3_to_aki_richards(s, d, r)
    assert ((cs >= 0) & (cs < 360)).all()
    assert ((cd >= 0) & (cd <= 90)).all()
    assert ((cr > -180) & (cr <= 180)).all()
    assert np.isnan(mm.sp3_to_aki_richards(0, 181, 0)).all()
    assert np.isnan(mm.sp3_to_aki_richards(0, -1, 0)).all()
    assert np.isnan(mm.sp3_to_aki_richards(np.nan, 30, 0)).all()


def test_dip90_branches_same_plane_different_numbers() -> None:
    # (s, 90, r) taken through either branch of the rule
    steep = plane(-86, 90, 152)  # dip >= 90 branch
    as_shallow = mm.normalize_plane(-86 + 180, 90, 152 + 180)  # other branch
    assert steep != pytest.approx(tuple(float(x) for x in as_shallow))
    assert float(mm.same_plane_deg(*steep, *as_shallow)) == pytest.approx(0, abs=1e-6)


def test_same_plane_deg_is_plane_identity_not_swap() -> None:
    assert float(mm.same_plane_deg(341, 74, 170, 341 - 360, 74, 170)) < 1e-6
    # fps plane b of 2024 #1 is the auxiliary: not the same plane
    assert float(mm.same_plane_deg(341, 74, 170, 74, 81, 16)) > 45


def test_normalize_plane_keeps_vertical_description() -> None:
    s, d, r = mm.normalize_plane(-86, 90, 28)
    assert (float(s), float(d), float(r)) == (274, 90, 28)


# --------------------------------------------------------------------------
# Conversion against events.fps on every fixture event (all four classes)
# --------------------------------------------------------------------------


def test_every_fixture_block_reproduces_fps_planes() -> None:
    classes: dict[str, int] = {}
    cases = fixture_blocks()
    assert len(cases) == 22
    for path, row in cases:
        parsed = mm.read_sp3(path)
        assert parsed.report.ok, parsed.report
        b = parsed.block
        assert b.has_planes
        (a, bb) = b.planes_aki_richards()
        fa = mm.normalize_plane(row["strike_a"], row["dip_a"], row["rake_a"])
        fb = mm.normalize_plane(row["strike_b"], row["dip_b"], row["rake_b"])
        assert float(mm.same_plane_deg(*a, *fa)) < 1e-6, path
        assert float(mm.same_plane_deg(*bb, *fb)) < 1e-6, path
        # one solution: moment, stress drop, corner frequency agree
        assert b.seismic_moment_nm == pytest.approx(row["seismic_moment"], rel=6e-3)
        assert b.stress_drop_mpa == pytest.approx(row["stress_drop"])
        assert b.corner_freq_hz == pytest.approx(row["corner_freq"])
        assert b.n_amplitudes == row["num_of_amplitudes"]
        assert b.n_polarities == row["num_of_polarities"]
        assert b.faulting_class is not None
        classes[b.faulting_class] = classes.get(b.faulting_class, 0) + 1
    assert classes == {"strike-slip": 13, "normal": 4, "reverse": 4, "dip-slip": 1}


# --------------------------------------------------------------------------
# read_sp3
# --------------------------------------------------------------------------


def test_read_sp3_modern_block() -> None:
    parsed = mm.read_sp3(SIL / "2024/aug/22/00:05:00/10:05:430.sp3")
    b = parsed.block
    assert parsed.report.ok
    assert b.header_path == "/sk2/2024/aug/22/00:05:00/10:05:430.sp3"
    assert (b.lib_mlw, b.lib_sh_min, b.lib_sh_max, b.lib_azim_comp) == (
        0.25,
        -0.994,
        0.905,
        27.6,
    )
    assert b.lib_freq_hz is None
    assert b.seismic_moment_nm == 0.177e11
    assert b.local_magnitude == 0.2
    assert (b.corner_freq_min_hz, b.corner_freq_max_hz, b.corner_freq_hz) == (
        10.0,
        6.5,
        6.5,
    )
    assert (b.fault_radius_min_m, b.fault_radius_max_m, b.fault_radius_m) == (
        122,
        79,
        122,
    )
    assert b.stress_drop_mpa == 0.004 and b.peak_slip_mm == 0.03
    assert (b.p_axis_azimuth, b.p_axis_dip, b.t_axis_azimuth, b.t_axis_dip) == (
        27,
        -4,
        -62,
        18,
    )
    assert (b.strike_a, b.dip_a, b.rake_a) == (-19, 106, 10)
    assert (b.strike_b, b.dip_b, b.rake_b) == (74, 99, 164)
    assert (b.normal_a_azimuth, b.normal_a_dip) == (251, 16)
    assert b.faulting_class == "strike-slip"
    assert (b.n_polarities, b.n_polarities_required, b.n_polarity_misfits) == (0, 0, 0)
    assert (b.n_amplitudes, b.n_amplitude_stations) == (24, 8)
    assert b.pct_acceptable == 4.63
    assert b.pct_acceptable_first_motion == 100.0
    assert b.pct_well_fitting_planes == 40.8
    assert b.amp_mean_error_factor == 2.75
    assert b.significant is True
    assert (b.significance_qualifier, b.significance_level_pct) == ("at", 4.0)
    assert b.f_dof == (23, 20) and b.f_value == 2.10
    assert (b.p_spectrum_misfit, b.s_spectrum_misfit) == (0.21, 0.36)
    assert b.low_freq_limit_hz is None


def test_read_sp3_1995_block() -> None:
    parsed = mm.read_sp3(SIL / "1995/jun/15/01:45:00/47:03:327.sp3")
    b = parsed.block
    assert parsed.report.ok
    assert b.lib_freq_hz == 3.0
    assert (b.strike_a, b.dip_a, b.rake_a) == (-40, 123, 140)
    assert b.faulting_class == "reverse"
    assert b.significant is False and b.significance_level_pct is None
    assert b.pct_acceptable == 22.5
    assert b.low_freq_limit_hz == 3.0


AP3_1995 = """\
  file: /eq1/1995/mar/19/20:40:00/42:51:202.ap3
      950319 204250    0
      lib 0.34   -0.059  0.456  -30.8
      DYNAMIC SOURCE PARAMETERS
      Size measures
      seismic moment:  0.217E+11 Nm
      local magnitude: 0.3
      Shear wave corner frequency range at close distances (900km)
       7.3Hz -11.0Hz     (  9.1Hz)
      Fault radius range   62m -  94m     (  75m)
      Stress drop range  0.011Mpa - 0.038Mpa     ( 0.022Mpa)
      Range of the peak slip at the fault 0.05mm - 0.10mm ( 0.07mm)
      The orientation of the relaxed stress
              azimuth   dip
      P-axis     91.    56. degrees
      T-axis     71.   -32.
      The horizontal deviatoric stress as given by the P- and T-axes
      the azimuth of compression -30 degrees
      the relative size 0.26
      The two possible fault planes
                strike   dip   slip
      plane a   130.    15.    51. degrees
      plane b   169.   102.   260.
      The normal directions of the fault planes
               azimuth   dip
      plane a   220.    75. degrees
      plane b    79.    12.
      Dip-slip faulting dominates
      STATISTICAL INFORMATION
      Of  1 first motion polarity observations
      at least  1 are required to fit
      The optimum mechanism has  0 polarity misfits
      Amplitudes for P and S at  6 stations are used
      Only mechanisms giving an estimated standard
      deviation of the amplitude error factor of less
      than 2.46 for single p-wave observations are
      taken as acceptable and included in the figures
       0.01 % of all mechanisms are acceptable
       50.0 % acceptable due to first motion observations
        0.0 % of these fitted also the amplitudes
      The part of well fitting planes is  0.9%
      The amplitude fit of the optimal mechanism
      gives a mean error factor of 2.34
      This corresponds to a standard deviation factor of 2.63
      for single P-wave observations
      The double couple solution is not significant
      (f-value: f(17,14) =  1.80)
      The measure of the misfit to an earthquake spectrum
      P-waves  0.36
      S-waves  0.26
"""

FPS_AP3_1995 = (
    "25 19950319 204242.754 63.95652 0.039 -21.25575 0.018   0.997   1.5  0.30 "
    "/1995/mar/19/20:40:00/42:51:202 2.17e+10  0.05 qu -0.059 0.456 -30.0 0.26  "
    "9.1   75 0.022  0.070 ap3  310  15 -129  169  78  -80  1 18   0.01 0"
)
FPS_FILE_2008 = (
    "251 20080529 190403.065 63.94287 0.002 -21.27219 0.004   4.772   0.3 -9.00 "
    "/2008/may/29/18:55:00/01:17:704 1.00e+01  0.28 qu -9.000 -9.000  -9.0 9.00 "
    "-9.0   -9 -9.000 -9.000 FILE   -9  -9 -999   -9  -9 -999 -9 -9  -9.00 -9"
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1")
    return path


def test_read_sp3_ap3_variant_and_conversion(tmp_path: Path) -> None:
    parsed = mm.read_sp3(write(tmp_path / "x.ap3", AP3_1995))
    b = parsed.block
    assert parsed.report.ok
    # "Amplitudes for P and S at N stations": no amplitude count
    assert b.n_amplitudes is None and b.n_amplitude_stations == 6
    assert b.faulting_class == "dip-slip"
    a, bb = b.planes_aki_richards()
    assert a == pytest.approx((310, 15, -129))
    assert bb == pytest.approx((169, 78, -80))


def test_read_sp3_odd_lines(tmp_path: Path) -> None:
    text = (
        AP3_1995.replace("lib 0.34", "lib-0.92")
        .replace("local magnitude: 0.3", "local magnitude:-0.9")
        .replace("(f-value: f(17,14) =  1.80)", "(f-value: f(**,**) =  1.63)")
        .replace("the relative size 0.26", "the relative size .26")
        .replace(" 7.3Hz -11.0Hz", "16.4Hz-30.0Hz")
        .replace(
            "      The optimum mechanism has  0 polarity misfits\n",
            "      The optimum mechanism has  1 polarity misfits\n"
            "      at the station(s) vog nyl\n",
        )
        .replace("      plane b   169.", "      plane b  -169.")
    )
    parsed = mm.read_sp3(write(tmp_path / "x.sp3", text + "      nonsense\n"))
    b = parsed.block
    assert b.lib_mlw == -0.92 and b.local_magnitude == -0.9
    assert b.relative_size == 0.26
    assert (b.corner_freq_min_hz, b.corner_freq_max_hz) == (16.4, 30.0)
    assert b.misfit_stations == ("vog", "nyl") and b.n_polarity_misfits == 1
    assert b.strike_b == -169
    assert b.f_dof is None and b.f_value == 1.63
    assert len(parsed.report.field_warnings) == 1
    # a Fortran overflow in a range is NA, with a warning
    over = mm.read_sp3(
        write(
            tmp_path / "o.sp3",
            AP3_1995.replace("62m -  94m", "1150m -****m").replace(
                "at least  1 are", "at least -2 are"
            ),
        )
    )
    assert over.block.fault_radius_max_m is None
    assert over.block.fault_radius_min_m == 1150
    assert over.block.n_polarities_required == -2
    assert [w.reason for w in over.report.field_warnings] == [
        "overflow (****) read as NA"
    ]
    assert not over.report.unparsed
    assert [u.text.strip() for u in parsed.report.unparsed] == ["nonsense"]
    assert not parsed.report.ok


def test_read_sp3_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mm.read_sp3(tmp_path / "none.sp3")


def test_block_path() -> None:
    p = mm.block_path("/x/2024/aug/22", "/2024/aug/22/00:05:00/10:05:430", "ap3")
    assert p == Path("/x/2024/aug/22/00:05:00/10:05:430.ap3")
    with pytest.raises(ValueError):
        mm.block_path("/x", "nope", "sp3")


# --------------------------------------------------------------------------
# mechanisms_from_fps and the day reader
# --------------------------------------------------------------------------


@pytest.mark.parametrize("day", ALL_DAYS)
@pytest.mark.parametrize("extras", [False, True])
def test_fixture_days_validate(day: str, extras: bool) -> None:
    res = mm.read_sil_day_mechanisms(SIL / day, extras=extras)
    report = validate_mechanisms(res.frame)
    assert report.ok, report.summary()
    # T12b: every frame carries the `planes_inconsistent` marker, an extra
    # column until the schema takes it in
    expect = list(mm.output_columns())
    extra_cols = expect != list(column_names("mechanisms"))
    if extras:
        expect += list(mm.EXTRA_COLUMNS)
    if extras or extra_cols:
        assert {i.kind for i in report.issues} == {"extra_column"}
    else:
        assert not report.issues
    assert list(res.frame.columns) == expect
    assert res.frame.index.dtype == TIME_DTYPE
    assert res.report.block_mismatch == () and res.report.blocks_missing == ()
    assert res.report.block_issues == ()


def test_day_2024_rows() -> None:
    res = mm.read_sil_day_mechanisms(SIL / "2024/aug/22")
    df = res.frame
    assert len(df) == 8 and res.report.blocks_read == 8
    assert set(df["method"]) == {"sil_fps"} and set(df["source"]) == {"sil-files"}
    first = df.iloc[0]
    assert first["event_id"] == "/2024/aug/22/00:05:00/10:05:430"
    assert df.index[0] == pd.Timestamp("2024-08-22 00:10:04.454", tz="UTC")
    assert (first["strike1"], first["dip1"], first["rake1"]) == (341, 74, 170)
    assert (first["strike2"], first["dip2"], first["rake2"]) == (74, 81, 16)
    assert first["quality"] == 4.63 and first["misfit"] == 2.75
    assert (first["n_polarities"], first["n_amplitudes"]) == (0, 24)
    assert first["stress_drop_mpa"] == 0.004 and first["corner_freq_hz"] == 6.5
    # the negative fps strike (#53, -86/90/28) is wrapped
    row53 = df[df["event_id"].str.endswith("39:04:992")].iloc[0]
    assert (row53["strike1"], row53["dip1"], row53["rake1"]) == (274, 90, 28)
    assert df["strike1"].between(0, 360, inclusive="left").all()


def test_day_extras() -> None:
    df = mm.read_sil_day_mechanisms(SIL / "2024/aug/22", extras=True).frame
    assert set(df["block"]) == {"sp3"}
    assert df["block_plane_mismatch_deg"].max() < 1e-6
    assert df.iloc[0]["faulting_class"] == "strike-slip"
    assert bool(df.iloc[0]["significant"]) is True


def test_extras_with_unreadable_significance() -> None:
    fps = read_events_fps(SIL / "2024/aug/22/events.fps").frame
    fps.loc[0, "is_significant"] = pd.NA
    df = mm.mechanisms_from_fps(fps, extras=True)
    assert pd.isna(df.iloc[0]["significant"]) and bool(df.iloc[1]["significant"])


def test_fps_without_blocks_still_gives_rows() -> None:
    fps = read_events_fps(SIL / "2024/aug/22/events.fps").frame
    df = mm.mechanisms_from_fps(fps)
    assert len(df) == 8
    assert df["quality"].isna().all() and df["misfit"].isna().all()
    assert validate_mechanisms(df).ok


def ap3_day(tmp_path: Path) -> Path:
    day = tmp_path / "1995/mar/19"
    write(day / "events.fps", FPS_AP3_1995 + "\n" + FPS_FILE_2008 + "\n")
    write(day / "20:40:00/42:51:202.ap3", AP3_1995)
    # a different solution for the same event, as in the archive
    shutil.copy(
        SIL / "1995/jun/15/01:45:00/47:03:327.sp3", day / "20:40:00/42:51:202.sp3"
    )
    # a block next to the no-solution row
    (day / "18:55:00").mkdir()
    shutil.copy(
        SIL / "2024/aug/22/00:05:00/10:05:430.sp3", day / "18:55:00/01:17:704.sp3"
    )
    return day


def test_ap3_row_uses_its_own_block_and_no_solution_row_gives_none(
    tmp_path: Path,
) -> None:
    res = mm.read_sil_day_mechanisms(ap3_day(tmp_path), extras=True)
    df, rep = res.frame, res.report
    assert len(df) == 1  # the FILE row gives none
    row = df.iloc[0]
    assert row["method"] == mm.FPS_BLOCK_METHOD["ap3"] == "sil_ap3"
    assert row["block"] == "ap3"
    assert (row["strike1"], row["dip1"], row["rake1"]) == (310, 15, -129)
    assert row["quality"] == 0.01 and row["misfit"] == 2.34  # from the ap3
    assert row["n_amplitudes"] == 18  # from fps (the ap3 has no count)
    assert rep.no_solution == 1 and rep.block_kinds == {"ap3": 1}
    assert sorted(rep.unused_blocks) == [
        "/1995/mar/19/20:40:00/42:51:202.sp3",
        "/2008/may/29/18:55:00/01:17:704.sp3",
    ]
    assert validate_mechanisms(df).ok


def test_block_disagreeing_with_fps_is_not_attached(tmp_path: Path) -> None:
    day = ap3_day(tmp_path)
    # replace the ap3 by an unrelated block
    shutil.copy(
        SIL / "2024/aug/22/00:05:00/10:05:430.sp3", day / "20:40:00/42:51:202.ap3"
    )
    res = mm.read_sil_day_mechanisms(day)
    assert res.report.block_mismatch == ("/1995/mar/19/20:40:00/42:51:202",)
    assert len(res.frame) == 1 and np.isnan(res.frame.iloc[0]["quality"])


def test_missing_block_is_reported(tmp_path: Path) -> None:
    day = tmp_path / "1995/mar/19"
    write(day / "events.fps", FPS_AP3_1995 + "\n")
    res = mm.read_sil_day_mechanisms(day)
    assert res.report.blocks_missing == ("/1995/mar/19/20:40:00/42:51:202",)
    assert len(res.frame) == 1


LIB_1991 = (
    "   2 19910610 041127.941 64.06940 -21.10586  7.047 0.60      "
    "/1991/jun/10/04:11:15/11:30:741   -0.994  0.827   70.0  3                  "
)
SP3_1991_PLANES = (
    "      plane a   -62.   110.   165. degrees\n      plane b   202.   104.    21.\n"
)


def test_day_without_fps_reads_sp3_from_events_lib(tmp_path: Path) -> None:
    day = tmp_path / "1991/jun/10"
    write(day / "events.lib", LIB_1991 + "\n")
    block = (
        AP3_1995.replace("plane a   130.    15.    51. degrees\n", "")
        .replace("plane b   169.   102.   260.\n", "")
        .replace(
            "                strike   dip   slip\n",
            "                strike   dip   slip\n" + SP3_1991_PLANES,
        )
        .replace("      Amplitudes for P", "       18 amplitudes for P")
    )
    write(day / "04:11:15/11:30:741.sp3", block)
    res = mm.read_sil_day_mechanisms(day)
    df = res.frame
    assert not res.report.has_fps
    assert len(df) == 1 and res.report.rows_by_method == {"sil_sp3": 1}
    row = df.iloc[0]
    assert row["method"] == "sil_sp3"
    assert df.index[0] == pd.Timestamp("1991-06-10 04:11:27.941", tz="UTC")
    assert (row["strike1"], row["dip1"], row["rake1"]) == (298, 70, 15)
    assert (row["strike2"], row["dip2"], row["rake2"]) == (202, 76, 159)
    assert row["n_amplitudes"] == 18 and row["quality"] == 0.01
    assert validate_mechanisms(df).ok


def test_day_without_fps_or_lib_is_empty(tmp_path: Path) -> None:
    res = mm.read_sil_day_mechanisms(tmp_path)
    assert len(res.frame) == 0 and validate_mechanisms(res.frame).ok


# --------------------------------------------------------------------------
# Cross-check against the legacy Skjálftalísa API (T9 recordings)
# --------------------------------------------------------------------------


def test_legacy_detail_sp3_matches_converted_planes() -> None:
    detail = json.loads((LEGACY / "internal_detail_1352381.json").read_text())
    sp3 = json.loads(detail["sp3"])  # JSON inside a string
    eid = detail["slungapath"]
    b = mm.read_sp3(mm.block_path(SIL / "2024/aug/22", eid, "sp3")).block
    # raw planes are the block's as written
    assert (sp3["fpa_strike"], sp3["fpa_dip"], sp3["fpa_slip"]) == (
        b.strike_a,
        b.dip_a,
        b.rake_a,
    )
    assert (sp3["fpb_strike"], sp3["fpb_dip"], sp3["fpb_slip"]) == (
        b.strike_b,
        b.dip_b,
        b.rake_b,
    )
    # the _AR planes are our converted planes, geometrically and here by value
    df = mm.read_sil_day_mechanisms(SIL / "2024/aug/22").frame
    row = df[df["event_id"] == eid].iloc[0]
    for n, ab in ((1, "a"), (2, "b")):
        ar = [sp3[f"fp{ab}_{x}_AR"] for x in ("strike", "dip", "slip")]
        ours = [row[f"{x}{n}"] for x in ("strike", "dip", "rake")]
        assert ours == ar
        assert float(mm.same_plane_deg(*ours, *ar)) < 1e-6
    # scalars that the legacy ETL sliced short (T12b must not use them)
    assert sp3["amplitudeCnt"] == 4 and b.n_amplitudes == 24
    assert sp3["acceptable_mechanisms"] == 0.63 and b.pct_acceptable == 4.63


def test_legacy_list_plane_matches_plane1() -> None:
    feats = json.loads((LEGACY / "internal_20240822T00.json").read_text())["features"]
    legacy = {
        pd.Timestamp(f["properties"]["time"], tz="UTC"): f["properties"]
        for f in feats
        if f["properties"].get("strike") is not None
    }
    df = mm.read_sil_day_mechanisms(SIL / "2024/aug/22").frame
    hits = 0
    for t, row in df.iterrows():
        p = legacy.get(t)
        if p is None:
            continue
        hits += 1
        ours = (row["strike1"], row["dip1"], row["rake1"])
        assert float(mm.same_plane_deg(*ours, p["strike"], p["dip"], p["slip"])) < 1e-6
    assert hits == 3  # 1352381, 1352384, 1352385


# --------------------------------------------------------------------------
# Real archive (only with $GPS_SIL_DIR)
# --------------------------------------------------------------------------

REAL_DAYS = ("1995/mar/19", "2011/jan/04", "2014/aug/29", "2024/aug/22")


_SIL_DIR = os.environ.get("GPS_SIL_DIR")


@pytest.mark.skipif(not _SIL_DIR, reason="set $GPS_SIL_DIR to the SIL archive")
@pytest.mark.parametrize("day", REAL_DAYS)
def test_real_archive_day(day: str) -> None:
    base = str(_SIL_DIR)
    if not (Path(base) / day).is_dir():
        pytest.skip(f"{day} not in $GPS_SIL_DIR")
    res = mm.read_sil_day_mechanisms(Path(base) / day)
    assert res.report.block_mismatch == ()
    assert res.report.blocks_read == res.report.rows
    assert validate_mechanisms(res.frame).ok
