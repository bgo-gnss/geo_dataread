"""Tests for geo_dataread.globk_tot — batch TOT writer CLI + exclusion rules.

Real-data cases reuse the frozen okada fixtures (``fixtures/globk/pre``·
``rap``, see test_globk_join.py for provenance). The seed exclusion rules
(SEY1 name-clash orphan, SUND rap-subset drop) mirror the DEPLOYED catalog
``gps-config-data/analysis-lane/segment_exclusions.csv`` — resolved by
default via ``gps_parser.outlier_catalogs`` (``default_exclusions_path``),
with ``--exclusions`` as the dev override.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import geo_dataread.globk_tot as globk_tot
from geo_dataread.globk_join import (
    GlobkJoinError,
    join_station_component,
    read_mb_segment,
)
from geo_dataread.globk_tot import (
    SegmentExclusion,
    exclusion_reason,
    join_station,
    load_exclusions,
    main,
    resolve_exclusions,
)

FIXTURES = Path(__file__).parent / "fixtures" / "globk"
PRE = FIXTURES / "pre"
RAP = FIXTURES / "rap"

# In-test mirror of the deployed seed catalog (comment lines included, to
# pin the deployed-catalog convention the loader must accept).
SEED_CSV = """\
# segment_exclusions.csv — reviewed per-station GLOBK segment drops.
station,drop_dir,drop_before_year,reason
SEY1,,2020.0,1996 GPS segment is an old reference site that reused the code (name clash) - BGO 2026-07-18
SUND,rap,,rap is a strict subset of pre (0 unique epochs) and the rapid solution is unreliable during the Nov-2023 dike intrusion - lossless drop - BGO 2026-07-18
"""


@pytest.fixture(autouse=True)
def _no_deployed_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from any real deployed segment_exclusions.csv on this host."""
    monkeypatch.setattr(globk_tot, "default_exclusions_path", lambda: None)


def _write_segment(
    path: Path,
    station: str,
    component: str,
    reference: float,
    rows: list[tuple[float, float, float]],
) -> Path:
    lines = [
        "Globk Analysis GGVer 10.71.021 synthetic",
        f"{station}_GPS to {component} Solution  1 +  {reference:.3f} m",
        " ",
    ]
    lines += [f" {t:.5f}     {v:.5f}  {s:.5f}" for t, v, s in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _synthetic_pre_rap(
    tmp_path: Path,
    station: str = "TEST",
    *,
    orphan: bool = False,
) -> tuple[Path, Path]:
    """Build pre/ and rap/ dirs with co-datum segments for all three axes.

    With ``orphan=True`` the pre dir additionally carries a 1996 segment on
    an off-lattice datum (+3 m) — the SEY1-shaped name-clash case that trips
    the loud residual guard unless excluded by rule.
    """
    pre, rap = tmp_path / "pre", tmp_path / "rap"
    pre.mkdir()
    rap.mkdir()
    for axis, comp in ((1, "N"), (2, "E"), (3, "U")):
        _write_segment(
            pre / f"mb_{station}_GPS.dat{axis}",
            station,
            comp,
            1.0,
            [(2024.1, 0.100, 0.005), (2024.2, 0.102, 0.005)],
        )
        _write_segment(
            rap / f"mb_{station}_GPS.dat{axis}",
            station,
            comp,
            1.0,
            [(2024.3, 0.104, 0.005), (2024.4, 0.106, 0.005)],
        )
        if orphan:
            _write_segment(
                pre / f"mb_{station}_1PS.dat{axis}",
                station,
                comp,
                1.0,
                [(1996.1, 3.100, 0.005), (1996.2, 3.102, 0.005)],
            )
    return pre, rap


class TestLoadExclusions:
    def test_parses_seed_rules(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "segment_exclusions.csv"
        csv_path.write_text(SEED_CSV)
        rules = load_exclusions(csv_path)
        assert set(rules) == {"SEY1", "SUND"}
        (sey1,) = rules["SEY1"]
        assert sey1.drop_before_year == 2020.0
        assert sey1.drop_dir is None
        assert "name clash" in sey1.reason
        (sund,) = rules["SUND"]
        assert sund.drop_dir == "rap"
        assert sund.drop_before_year is None
        assert "lossless" in sund.reason

    def test_multiple_rules_per_station(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        csv_path.write_text(
            "station,drop_dir,drop_before_year,reason\nAAAA,rap,,r1\nAAAA,,2000.0,r2\n"
        )
        rules = load_exclusions(csv_path)
        assert len(rules["AAAA"]) == 2

    def test_missing_reason_rejected(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        csv_path.write_text("station,drop_dir,drop_before_year,reason\nAAAA,rap,,\n")
        with pytest.raises(GlobkJoinError, match="missing reason"):
            load_exclusions(csv_path)

    def test_rule_without_criterion_rejected(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        csv_path.write_text("station,drop_dir,drop_before_year,reason\nAAAA,,,why\n")
        with pytest.raises(GlobkJoinError, match="drop_dir and/or drop_before_year"):
            load_exclusions(csv_path)

    def test_bad_year_rejected(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        csv_path.write_text(
            "station,drop_dir,drop_before_year,reason\nAAAA,,soon,why\n"
        )
        with pytest.raises(GlobkJoinError, match="fractional year"):
            load_exclusions(csv_path)

    def test_wrong_columns_rejected(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        csv_path.write_text("sta,dir,year,why\nAAAA,rap,,why\n")
        with pytest.raises(GlobkJoinError, match="columns"):
            load_exclusions(csv_path)

    def test_station_uppercased(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "x.csv"
        csv_path.write_text("station,drop_dir,drop_before_year,reason\nsund,rap,,r\n")
        assert set(load_exclusions(csv_path)) == {"SUND"}


class TestResolveExclusions:
    def test_explicit_path_wins_and_must_exist(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "excl.csv"
        csv_path.write_text(SEED_CSV)
        rules, source = resolve_exclusions(csv_path)
        assert set(rules) == {"SEY1", "SUND"}
        assert source == str(csv_path)
        with pytest.raises(FileNotFoundError):
            resolve_exclusions(tmp_path / "missing.csv")

    def test_deployed_catalog_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deployed = tmp_path / "segment_exclusions.csv"
        deployed.write_text(SEED_CSV)
        monkeypatch.setattr(globk_tot, "default_exclusions_path", lambda: deployed)
        rules, source = resolve_exclusions(None)
        assert set(rules) == {"SEY1", "SUND"}
        assert source == str(deployed)

    def test_absent_deployed_catalog_warns_and_joins_without(self) -> None:
        with pytest.warns(UserWarning, match="WITHOUT per-station exclusions"):
            rules, source = resolve_exclusions(None)
        assert rules == {}
        assert source is None

    def test_cli_uses_deployed_catalog_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pre, rap = _synthetic_pre_rap(tmp_path, station="SUND")
        deployed = tmp_path / "segment_exclusions.csv"
        deployed.write_text(SEED_CSV)
        monkeypatch.setattr(globk_tot, "default_exclusions_path", lambda: deployed)
        out = tmp_path / "TOT"
        rc = main(["SUND", "--pre", str(pre), "--rap", str(rap), "--out", str(out)])
        assert rc == 0
        reread = read_mb_segment(out / "mb_SUND_TOT.dat2")
        np.testing.assert_allclose(reread.epochs, [2024.1, 2024.2])  # rap dropped


class TestExclusionReason:
    def test_drop_dir_matches_parent_dir(self, tmp_path: Path) -> None:
        rap = tmp_path / "rap"
        rap.mkdir()
        path = _write_segment(
            rap / "mb_TEST_GPS.dat2", "TEST", "E", 1.0, [(2024.1, 0.1, 0.005)]
        )
        segment = read_mb_segment(path)
        rule = SegmentExclusion(station="TEST", reason="r", drop_dir="rap")
        assert exclusion_reason([rule], path, segment) is not None
        pre_rule = SegmentExclusion(station="TEST", reason="r", drop_dir="pre")
        assert exclusion_reason([pre_rule], path, segment) is None

    def test_drop_before_year_uses_last_epoch(self, tmp_path: Path) -> None:
        path = _write_segment(
            tmp_path / "mb_TEST_GPS.dat2",
            "TEST",
            "E",
            1.0,
            [(1996.1, 0.1, 0.005), (1996.2, 0.1, 0.005)],
        )
        segment = read_mb_segment(path)
        rule = SegmentExclusion(station="TEST", reason="r", drop_before_year=2020.0)
        reason = exclusion_reason([rule], path, segment)
        assert reason is not None and "1996.200" in reason
        keep = SegmentExclusion(station="TEST", reason="r", drop_before_year=1990.0)
        assert exclusion_reason([keep], path, segment) is None

    def test_no_rules_keeps_everything(self, tmp_path: Path) -> None:
        path = _write_segment(
            tmp_path / "mb_TEST_GPS.dat2", "TEST", "E", 1.0, [(2024.1, 0.1, 0.005)]
        )
        assert exclusion_reason([], path, read_mb_segment(path)) is None


class TestJoinStation:
    def test_writes_all_axes(self, tmp_path: Path) -> None:
        pre, rap = _synthetic_pre_rap(tmp_path)
        out = tmp_path / "TOT"
        out.mkdir()
        results = join_station("TEST", [pre, rap], out)
        assert [r.status for r in results] == ["written", "written", "written"]
        for axis in (1, 2, 3):
            reread = read_mb_segment(out / f"mb_TEST_TOT.dat{axis}")
            np.testing.assert_allclose(reread.epochs, [2024.1, 2024.2, 2024.3, 2024.4])

    def test_drop_dir_rule_excludes_rap_epochs(self, tmp_path: Path) -> None:
        pre, rap = _synthetic_pre_rap(tmp_path)
        out = tmp_path / "TOT"
        out.mkdir()
        rules = {
            "TEST": (SegmentExclusion(station="TEST", reason="r", drop_dir="rap"),)
        }
        results = join_station("TEST", [pre, rap], out, rules)
        assert all(r.status == "written" for r in results)
        assert all("excluded by rule" in r.detail for r in results)
        reread = read_mb_segment(out / "mb_TEST_TOT.dat2")
        np.testing.assert_allclose(reread.epochs, [2024.1, 2024.2])  # pre only

    def test_orphan_trips_guard_without_rule_and_joins_with_it(
        self, tmp_path: Path
    ) -> None:
        # SEY1-shaped case: an off-datum 1996 name-clash orphan makes the
        # join fail LOUDLY; the reviewed drop_before_year rule fixes it.
        pre, rap = _synthetic_pre_rap(tmp_path, orphan=True)
        out = tmp_path / "TOT"
        out.mkdir()
        unruled = join_station("TEST", [pre, rap], out)
        assert all(r.status == "error" for r in unruled)
        assert "wrap lattice" in unruled[0].detail

        rules = {
            "TEST": (
                SegmentExclusion(station="TEST", reason="r", drop_before_year=2020.0),
            )
        }
        ruled = join_station("TEST", [pre, rap], out, rules)
        assert all(r.status == "written" for r in ruled)
        reread = read_mb_segment(out / "mb_TEST_TOT.dat2")
        assert float(reread.epochs[0]) == 2024.1  # orphan gone, anchor moved

    def test_all_excluded_is_skip_not_error(self, tmp_path: Path) -> None:
        pre, rap = _synthetic_pre_rap(tmp_path)
        out = tmp_path / "TOT"
        out.mkdir()
        rules = {
            "TEST": (
                SegmentExclusion(station="TEST", reason="r", drop_dir="pre"),
                SegmentExclusion(station="TEST", reason="r", drop_dir="rap"),
            )
        }
        results = join_station("TEST", [pre, rap], out, rules)
        assert [r.status for r in results] == ["skipped", "skipped", "skipped"]
        assert not list(out.iterdir())

    def test_missing_station_is_error_per_axis(self, tmp_path: Path) -> None:
        pre, rap = _synthetic_pre_rap(tmp_path)
        out = tmp_path / "TOT"
        out.mkdir()
        results = join_station("XXXX", [pre, rap], out)
        assert [r.status for r in results] == ["error", "error", "error"]


class TestCli:
    def test_batch_matches_library_join(self, tmp_path: Path) -> None:
        from geo_dataread.gps_read import openGlobkTimes

        out = tmp_path / "TOT"
        rc = main(["SENG", "--pre", str(PRE), "--rap", str(RAP), "--out", str(out)])
        assert rc == 0
        yearf, data, _ = openGlobkTimes("SENG", Dir=str(out), tType="TOT")
        for axis in (1, 2, 3):
            joined = join_station_component("SENG", axis, [PRE, RAP])
            np.testing.assert_array_equal(yearf, joined.epochs)
            np.testing.assert_allclose(
                data[axis - 1], joined.values, rtol=0, atol=1e-9, equal_nan=True
            )

    def test_out_dir_is_created(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "TOT"
        rc = main(["SENG", "--pre", str(PRE), "--rap", str(RAP), "--out", str(out)])
        assert rc == 0
        assert (out / "mb_SENG_TOT.dat1").is_file()

    def test_missing_station_sets_exit_code(self, tmp_path: Path) -> None:
        rc = main(
            ["XXXX", "--pre", str(PRE), "--rap", str(RAP), "--out", str(tmp_path)]
        )
        assert rc == 1

    def test_batch_continues_past_a_failing_station(self, tmp_path: Path) -> None:
        out = tmp_path / "TOT"
        rc = main(
            ["XXXX", "SENG", "--pre", str(PRE), "--rap", str(RAP), "--out", str(out)]
        )
        assert rc == 1  # XXXX failed ...
        assert (out / "mb_SENG_TOT.dat2").is_file()  # ... but SENG was written

    def test_exclusions_flag_end_to_end(self, tmp_path: Path) -> None:
        pre, rap = _synthetic_pre_rap(tmp_path, station="SUND")
        csv_path = tmp_path / "excl.csv"
        csv_path.write_text(
            "station,drop_dir,drop_before_year,reason\nSUND,rap,,lossless drop\n"
        )
        out = tmp_path / "TOT"
        rc = main(
            [
                "SUND",
                "--pre",
                str(pre),
                "--rap",
                str(rap),
                "--out",
                str(out),
                "--exclusions",
                str(csv_path),
            ]
        )
        assert rc == 0
        reread = read_mb_segment(out / "mb_SUND_TOT.dat2")
        np.testing.assert_allclose(reread.epochs, [2024.1, 2024.2])  # rap dropped
