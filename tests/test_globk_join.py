"""Tests for geo_dataread.globk_join — GLOBK segment reading and joining.

Fixture provenance (frozen 2026-07-17, all read-only from production):

* ``fixtures/globk/pre/``  — okada.vedur.is:/D/GMT/pre/ (prior-years segments)
* ``fixtures/globk/rap/``  — okada.vedur.is:/D/GLOBK/ITRF08/ (RAW current-year
  multibase output, *before* the production ``fixGlobkoffset`` mutation) —
  except ``mb_AJAC_1PS.dat2`` which is from /D/GMT/rap/ (stale, unmutated).
* ``fixtures/globk/TOT/``  — laptop autofs /mnt_data/gpsdata (=
  okada:/D/GMT/TOT, the live joined output, clean; same-day snapshot as rap).

Key real-data cases:

* SENG East (dat2): the canonical +10 m wrap. Pre reference 16542671.519 m,
  raw Rap reference 16542668.898 m (Δref = 2.621 m — NOT the datum shift);
  raw boundary step 2025.99863 → 2026.00137 is 10.0014 m = 1 wrap quantum
  + 1.4 mm motion. Acceptance gate: the join must reproduce the clean live
  TOT exactly.
* THOB East: header references differ by 1.884 m yet the value column is
  continuous — pins the finding that header references do NOT encode the
  datum (a reference-difference rebaseline would corrupt this station).
* AJAC East: multi-segment station (GPS/1PS/2PS markers, overlapping spans).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geo_dataread.globk_join import (
    GlobkJoinError,
    estimate_segment_offset,
    join_segments,
    join_station_component,
    read_mb_segment,
    wrap_correction,
)

FIXTURES = Path(__file__).parent / "fixtures" / "globk"
PRE = FIXTURES / "pre"
RAP = FIXTURES / "rap"
TOT = FIXTURES / "TOT"


def tot_first_occurrence(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Live-TOT target as (epochs, values), first occurrence per epoch.

    The production TOT is a raw concat and contains duplicate epochs (its
    Pre part is itself a historical concat); first-file-occurrence is the
    matching selection rule (same rule as read_mb_segment).
    """
    seg = read_mb_segment(path)
    return seg.epochs, seg.values


class TestHeaderParsing:
    def test_seng_header(self) -> None:
        seg = read_mb_segment(PRE / "mb_SENG_GPS.dat2")
        h = seg.header
        assert (h.station, h.marker, h.component, h.solution) == ("SENG", "GPS", "E", 1)
        assert h.reference_m == pytest.approx(16542671.519)
        assert h.provenance.startswith("Globk Analysis GGVer")

    def test_marker_variant_header(self) -> None:
        h = read_mb_segment(PRE / "mb_AJAC_1PS.dat2").header
        assert (h.station, h.marker, h.component) == ("AJAC", "1PS", "E")

    def test_bad_header_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "mb_XXXX_GPS.dat2"
        bad.write_text("Globk Analysis\nnot a reference line\n\n 2020.5 1.0 0.001\n")
        with pytest.raises(GlobkJoinError, match="reference line"):
            read_mb_segment(bad)


class TestSegmentReading:
    def test_sorted_and_deduplicated(self) -> None:
        seg = read_mb_segment(PRE / "mb_SENG_GPS.dat2")
        assert np.all(np.diff(seg.epochs) > 0)
        # file has 4927 data rows with 2191 duplicated epochs -> 2736 unique
        assert seg.epochs.size == 2736

    def test_keep_first_occurrence(self) -> None:
        # Epoch 2021.63698 appears twice in the Pre file with differing
        # values (1.34380 first, 1.36188 later); first occurrence wins.
        seg = read_mb_segment(PRE / "mb_SENG_GPS.dat2")
        idx = int(np.searchsorted(seg.epochs, 2021.63698))
        assert seg.values[idx] == pytest.approx(1.34380, abs=1e-9)

    def test_sigma_overflow_is_nan(self) -> None:
        # mb_AJAC_GPS.dat2 first row carries a '********' sigma field.
        seg = read_mb_segment(PRE / "mb_AJAC_GPS.dat2")
        assert np.isnan(seg.sigmas[0])
        assert np.isfinite(seg.values).all()


class TestWrapCorrection:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0.0014, 0.0), (10.0014, 10.0), (-9.9987, -10.0), (20.4, 20.0), (-0.3, 0.0)],
    )
    def test_lattice_snap(self, raw: float, expected: float) -> None:
        assert wrap_correction(raw) == expected


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


class TestJoinSynthetic:
    def test_negative_wrap(self, tmp_path: Path) -> None:
        # HVER-style: the later run sits one quantum LOW.
        a = _write_segment(
            tmp_path / "mb_HVXX_GPS.dat3",
            "HVXX",
            "U",
            150.0,
            [(2025.1, 0.100, 0.005), (2025.2, 0.102, 0.005)],
        )
        b = _write_segment(
            tmp_path / "mb_HVXX_0PS.dat3",
            "HVXX",
            "U",
            152.0,
            [(2025.3, -9.897, 0.005), (2025.4, -9.895, 0.005)],
        )
        joined = join_segments([read_mb_segment(a), read_mb_segment(b)])
        assert joined.corrections[1].correction_m == -10.0
        assert joined.values == pytest.approx([0.100, 0.102, 0.103, 0.105], abs=1e-9)

    def test_double_wrap(self, tmp_path: Path) -> None:
        a = _write_segment(
            tmp_path / "a.dat2", "TEST", "E", 1.0, [(2025.1, 0.500, 0.005)]
        )
        b = _write_segment(
            tmp_path / "b.dat2", "TEST", "E", 3.0, [(2025.2, 20.501, 0.005)]
        )
        joined = join_segments([read_mb_segment(a), read_mb_segment(b)])
        assert joined.corrections[1].correction_m == 20.0
        assert joined.values[-1] == pytest.approx(0.501, abs=1e-9)

    def test_overlap_median_beats_boundary_pair(self, tmp_path: Path) -> None:
        # Overlapping epochs must drive the offset estimate (median), not
        # the first-epoch boundary pair.
        a = _write_segment(
            tmp_path / "a.dat2",
            "TEST",
            "E",
            1.0,
            [(2025.1, 0.100, 0.005), (2025.2, 0.110, 0.005), (2025.3, 0.120, 0.005)],
        )
        b = _write_segment(
            tmp_path / "b.dat2",
            "TEST",
            "E",
            1.0,
            [(2025.2, 10.111, 0.005), (2025.3, 10.119, 0.005), (2025.4, 10.130, 0.005)],
        )
        joined = join_segments([read_mb_segment(a), read_mb_segment(b)])
        corr = joined.corrections[1]
        assert corr.n_overlap == 2
        assert corr.correction_m == 10.0
        # only the non-overlapping tail is appended
        assert corr.n_appended == 1
        assert joined.values[-1] == pytest.approx(0.130, abs=1e-9)

    def test_station_mismatch_raises(self, tmp_path: Path) -> None:
        a = _write_segment(
            tmp_path / "a.dat2", "AAAA", "E", 1.0, [(2025.1, 0.1, 0.005)]
        )
        b = _write_segment(
            tmp_path / "b.dat2", "BBBB", "E", 1.0, [(2025.2, 0.1, 0.005)]
        )
        with pytest.raises(GlobkJoinError, match="mismatch"):
            join_segments([read_mb_segment(a), read_mb_segment(b)])

    def test_off_lattice_offset_raises(self, tmp_path: Path) -> None:
        a = _write_segment(
            tmp_path / "a.dat2", "TEST", "E", 1.0, [(2025.1, 0.1, 0.005)]
        )
        b = _write_segment(
            tmp_path / "b.dat2", "TEST", "E", 1.0, [(2025.2, 4.1, 0.005)]
        )
        with pytest.raises(GlobkJoinError, match="wrap lattice"):
            join_segments([read_mb_segment(a), read_mb_segment(b)])

    def test_empty_raises(self) -> None:
        with pytest.raises(GlobkJoinError, match="no segments"):
            join_segments([])

    def test_estimate_requires_prior_coverage(self, tmp_path: Path) -> None:
        seg = read_mb_segment(
            _write_segment(
                tmp_path / "a.dat2", "TEST", "E", 1.0, [(2020.1, 0.1, 0.005)]
            )
        )
        with pytest.raises(GlobkJoinError, match="cannot estimate"):
            estimate_segment_offset(np.array([2025.0]), np.array([0.0]), seg)


class TestSengAcceptanceGate:
    """The join must reproduce the CLEAN live okada TOT for SENG."""

    def test_east_wrap_corrected_and_matches_live_tot(self) -> None:
        joined = join_station_component("SENG", 2, [PRE, RAP])

        # The raw Rap segment sits exactly one wrap quantum high ...
        corr = joined.corrections[1]
        assert corr.correction_m == 10.0
        assert abs(corr.residual_m) < 0.01  # 1.4 mm real motion at the boundary
        # ... and header references do NOT explain it (Δref = 2.621 m).
        pre_ref = read_mb_segment(PRE / "mb_SENG_GPS.dat2").header.reference_m
        rap_ref = read_mb_segment(RAP / "mb_SENG_GPS.dat2").header.reference_m
        assert pre_ref - rap_ref == pytest.approx(2.621, abs=1e-3)

        # No boundary jump: adjacent steps stay far below the wrap quantum.
        assert np.max(np.abs(np.diff(joined.values))) < 2.0

        # Acceptance: per-epoch identity with the live joined TOT.
        tot_epochs, tot_values = tot_first_occurrence(TOT / "mb_SENG_TOT.dat2")
        np.testing.assert_array_equal(joined.epochs, tot_epochs)
        np.testing.assert_allclose(joined.values, tot_values, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("axis", [1, 3])
    def test_north_up_no_correction_and_match(self, axis: int) -> None:
        joined = join_station_component("SENG", axis, [PRE, RAP])
        assert joined.corrections[1].correction_m == 0.0
        tot_epochs, tot_values = tot_first_occurrence(TOT / f"mb_SENG_TOT.dat{axis}")
        np.testing.assert_array_equal(joined.epochs, tot_epochs)
        np.testing.assert_allclose(joined.values, tot_values, rtol=0, atol=1e-9)


class TestRealDataRegressions:
    def test_thob_reference_difference_is_not_a_datum_shift(self) -> None:
        # Header refs differ by 1.884 m; values are continuous, so the
        # correction must be 0 — a reference-difference rebaseline would
        # inject a spurious 1.884 m step here.
        pre_ref = read_mb_segment(PRE / "mb_THOB_GPS.dat2").header.reference_m
        rap_ref = read_mb_segment(RAP / "mb_THOB_GPS.dat2").header.reference_m
        assert abs(pre_ref - rap_ref) > 1.0

        joined = join_station_component("THOB", 2, [PRE, RAP])
        assert joined.corrections[1].correction_m == 0.0
        tot_epochs, tot_values = tot_first_occurrence(TOT / "mb_THOB_TOT.dat2")
        np.testing.assert_array_equal(joined.epochs, tot_epochs)
        np.testing.assert_allclose(joined.values, tot_values, rtol=0, atol=1e-9)

    def test_ajac_multisegment_join(self) -> None:
        # GPS (1995–2008.6), 1PS (2005.4–2013.6), 2PS (2014.0–2025.999)
        # + rap 1PS (stale, fully overlapped) + rap 2PS (2026): five
        # segments, overlapping spans, all on one datum.
        joined = join_station_component("AJAC", 2, [PRE, RAP])
        assert len(joined.corrections) == 5
        assert all(c.correction_m == 0.0 for c in joined.corrections)
        assert np.all(np.diff(joined.epochs) > 0)
        assert joined.epochs[0] == pytest.approx(1995.36301)
        assert joined.epochs[-1] >= 2026.0
        # the stale rap 1PS segment is fully covered -> contributes nothing
        rap_1ps = [c for c in joined.corrections if c.path.name == "mb_AJAC_1PS.dat2"]
        assert any(c.n_appended == 0 and c.n_overlap > 0 for c in rap_1ps)

    def test_discover_missing_station_raises(self) -> None:
        with pytest.raises(GlobkJoinError, match="no mb_XXXX"):
            join_station_component("XXXX", 2, [PRE, RAP])
