"""Tests for geo_dataread.globk_join — GLOBK segment reading and joining.

Fixture provenance (frozen 2026-07-17/18, all read-only from production):

* ``fixtures/globk/pre/``  — okada.vedur.is:/D/GMT/pre/ (prior-years segments)
* ``fixtures/globk/rap/``  — SENG from /D/GLOBK/ITRF08/ (RAW current-year
  multibase output, *before* the production ``fixGlobkoffset`` mutation);
  REYK/HOFN from /D/GMT/rap/ (not in the ``compGLOBK -o`` mutation list, so
  those rap files are the raw copies).
* ``fixtures/globk/TOT/``  — okada:/D/GMT/TOT, the live joined output
  (concat-only, keeps duplicate epochs).

Key real-data cases:

* SENG East (dat2): the canonical +10 m wrap — single Pre + raw wrapped Rap,
  no cross-segment duplicates. Correction must be exactly +10 m; the join
  must reproduce the min-σ-deduped live TOT.
* REYK: 7 overlapping segments (1PS..6PS + GPS, 1995–2026) + Rap 6PS,
  2326 duplicate epochs in TOT. The dedup acceptance case. Segments are
  co-datum (all corrections 0). REYK East carries single-epoch spikes
  (2.76 m / 165 m / 2973 m) that pass through as outliers.
* HOFN: 5 Pre markers (1PS..4PS + GPS) + Rap 3PS/4PS, heavily overlapping.
  Co-datum, no wrap, no guard trip; a mildly stale TOT (sub-mm re-runs of
  Rap 3PS in 2013) makes it a structural/tolerance case, not a byte gate.
* THOB East: header references differ by 1.884 m yet values are continuous —
  pins that header references do NOT encode the datum (correction must be 0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geo_dataread.globk_join import (
    GlobkJoinError,
    dedup_min_sigma,
    estimate_segment_offset,
    join_segments,
    join_station_component,
    read_mb_segment,
    select_min_sigma_indices,
    wrap_correction,
)

FIXTURES = Path(__file__).parent / "fixtures" / "globk"
PRE = FIXTURES / "pre"
RAP = FIXTURES / "rap"
TOT = FIXTURES / "TOT"


def _read_raw_tot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a raw TOT file (duplicates kept) as (epochs, values, sigmas)."""
    seg = read_mb_segment(path)  # read keeps duplicates, only sorts
    return seg.epochs, seg.values, seg.sigmas


def _tot_solution_sets(path: Path) -> dict[float, list[float]]:
    """Map each TOT epoch -> the list of value solutions present at it."""
    e, v, _ = _read_raw_tot(path)
    out: dict[float, list[float]] = {}
    for ep, val in zip(e, v):
        out.setdefault(round(float(ep), 5), []).append(float(val))
    return out


def _value_in_solutions(val: float, solutions: list[float]) -> bool:
    """True if ``val`` matches one TOT solution (masked ↔ masked included)."""
    if np.isnan(val):
        return any(np.isnan(t) for t in solutions)
    return any(np.isfinite(t) and abs(val - t) < 1e-6 for t in solutions)


class TestHeaderParsing:
    def test_seng_header(self) -> None:
        h = read_mb_segment(PRE / "mb_SENG_GPS.dat2").header
        assert (h.station, h.marker, h.component, h.solution) == ("SENG", "GPS", "E", 1)
        assert h.reference_m == pytest.approx(16542671.519)
        assert h.provenance.startswith("Globk Analysis GGVer")

    def test_marker_variant_header(self) -> None:
        h = read_mb_segment(PRE / "mb_REYK_1PS.dat2").header
        assert (h.station, h.marker, h.component) == ("REYK", "1PS", "E")

    def test_bad_header_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "mb_XXXX_GPS.dat2"
        bad.write_text("Globk Analysis\nnot a reference line\n\n 2020.5 1.0 0.001\n")
        with pytest.raises(GlobkJoinError, match="reference line"):
            read_mb_segment(bad)


class TestSegmentReading:
    def test_sorted_ascending(self) -> None:
        seg = read_mb_segment(PRE / "mb_SENG_GPS.dat2")
        assert np.all(np.diff(seg.epochs) >= 0)

    def test_duplicates_are_kept(self) -> None:
        # SENG Pre is a historical concat with internal duplicate epochs; read
        # no longer dedups (global min-σ dedup is deferred to the join).
        seg = read_mb_segment(PRE / "mb_SENG_GPS.dat2")
        assert seg.epochs.size > np.unique(seg.epochs).size

    def test_masked_value_and_sigma_are_nan(self) -> None:
        # REYK TOT carries '********'/'************' overflow markers in the
        # sigma column (and occasionally the value column) -> NaN.
        _, _, s = _read_raw_tot(TOT / "mb_REYK_TOT.dat2")
        assert np.isnan(s).any()


class TestMinSigmaDedup:
    def test_masked_sigma_never_wins(self) -> None:
        epochs = np.array([2020.0, 2020.0])
        values = np.array([1.0, 2.0])
        sigmas = np.array([0.05, np.nan])  # masked second row -> +inf
        e, v, _ = dedup_min_sigma(epochs, values, sigmas)
        assert e.tolist() == [2020.0]
        assert v.tolist() == [1.0]  # the finite-sigma solution survives

    def test_min_sigma_selected(self) -> None:
        epochs = np.array([2020.0, 2020.0, 2020.0])
        values = np.array([1.0, 2.0, 3.0])
        sigmas = np.array([0.30, 0.05, 0.10])
        _, v, s = dedup_min_sigma(epochs, values, sigmas)
        assert v.tolist() == [2.0]
        assert s.tolist() == [0.05]

    def test_equal_sigma_ties_keep_first(self) -> None:
        epochs = np.array([2020.0, 2020.0])
        values = np.array([1.0, 2.0])
        sigmas = np.array([0.05, 0.05])  # tie -> keep first in array order
        _, v, _ = dedup_min_sigma(epochs, values, sigmas)
        assert v.tolist() == [1.0]

    def test_both_masked_tie_keeps_first(self) -> None:
        epochs = np.array([2020.0, 2020.0])
        values = np.array([7.0, 8.0])
        sigmas = np.array([np.nan, np.nan])  # both +inf -> keep first
        _, v, _ = dedup_min_sigma(epochs, values, sigmas)
        assert v.tolist() == [7.0]

    def test_output_epoch_sorted(self) -> None:
        epochs = np.array([2021.0, 2020.0, 2020.0, 2022.0])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        sigmas = np.array([0.1, 0.2, 0.1, 0.1])
        e, v, _ = dedup_min_sigma(epochs, values, sigmas)
        assert e.tolist() == [2020.0, 2021.0, 2022.0]
        assert v.tolist() == [3.0, 1.0, 4.0]  # 2020 keeps min-σ (0.1 -> value 3)

    def test_indices_helper_returns_original_positions(self) -> None:
        epochs = np.array([2020.0, 2020.0, 2021.0])
        sigmas = np.array([0.2, 0.1, 0.3])
        idx = select_min_sigma_indices(epochs, sigmas)
        assert idx.tolist() == [1, 2]


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
    def test_positive_wrap_no_overlap(self, tmp_path: Path) -> None:
        a = _write_segment(
            tmp_path / "a.dat2",
            "TEST",
            "E",
            1.0,
            [(2025.1, 0.100, 0.005), (2025.2, 0.102, 0.005)],
        )
        b = _write_segment(
            tmp_path / "b.dat2",
            "TEST",
            "E",
            1.0,
            [(2025.3, 10.103, 0.005), (2025.4, 10.105, 0.005)],
        )
        joined = join_segments([read_mb_segment(a), read_mb_segment(b)])
        corr = joined.corrections[1]
        assert corr.correction_m == 10.0
        assert corr.overlap_available is False  # gap: boundary pair
        assert corr.raw_offset_m == pytest.approx(10.001, abs=1e-3)
        assert joined.values == pytest.approx([0.100, 0.102, 0.103, 0.105], abs=1e-9)

    def test_negative_wrap(self, tmp_path: Path) -> None:
        a = _write_segment(
            tmp_path / "a.dat3",
            "HVXX",
            "U",
            150.0,
            [(2025.1, 0.100, 0.005), (2025.2, 0.102, 0.005)],
        )
        b = _write_segment(
            tmp_path / "b.dat3",
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

    def test_overlap_dedup_min_sigma(self, tmp_path: Path) -> None:
        # Overlapping epochs: offset estimated over overlap (median), then the
        # duplicated epochs collapse to their min-σ solution.
        a = _write_segment(
            tmp_path / "a.dat2",
            "TEST",
            "E",
            1.0,
            [(2025.1, 0.100, 0.005), (2025.2, 0.110, 0.020), (2025.3, 0.120, 0.005)],
        )
        b = _write_segment(
            tmp_path / "b.dat2",
            "TEST",
            "E",
            1.0,
            [(2025.2, 10.108, 0.004), (2025.3, 10.119, 0.030), (2025.4, 10.130, 0.005)],
        )
        joined = join_segments([read_mb_segment(a), read_mb_segment(b)])
        corr = joined.corrections[1]
        assert corr.correction_m == 10.0
        assert corr.n_overlap == 2
        assert corr.overlap_available is True
        # 2025.2: b wins (σ 0.004 < 0.020) -> 10.108-10 = 0.108
        # 2025.3: a wins (σ 0.005 < 0.030) -> 0.120
        assert joined.epochs == pytest.approx([2025.1, 2025.2, 2025.3, 2025.4])
        assert joined.values == pytest.approx([0.100, 0.108, 0.120, 0.130], abs=1e-9)

    def test_within_segment_outlier_passes_through(self, tmp_path: Path) -> None:
        a = _write_segment(
            tmp_path / "a.dat2",
            "TEST",
            "E",
            1.0,
            [(2025.1, 0.10, 0.005), (2025.2, 2900.0, 0.5), (2025.3, 0.12, 0.005)],
        )
        joined = join_segments([read_mb_segment(a)])
        assert 2900.0 in joined.values  # gross outlier is NOT touched by the join

    def test_station_mismatch_raises(self, tmp_path: Path) -> None:
        a = _write_segment(
            tmp_path / "a.dat2", "AAAA", "E", 1.0, [(2025.1, 0.1, 0.005)]
        )
        b = _write_segment(
            tmp_path / "b.dat2", "BBBB", "E", 1.0, [(2025.2, 0.1, 0.005)]
        )
        with pytest.raises(GlobkJoinError, match="mismatch"):
            join_segments([read_mb_segment(a), read_mb_segment(b)])

    def test_off_lattice_offset_trips_loud_guard(self, tmp_path: Path) -> None:
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

    def test_gap_records_raw_offset(self, tmp_path: Path) -> None:
        # No shared epoch across the boundary -> boundary-pair estimate, flagged
        # overlap_available=False, raw_offset recorded, guard still active.
        a = _write_segment(
            tmp_path / "a.dat2", "TEST", "E", 1.0, [(2025.1, 0.10, 0.005)]
        )
        b = _write_segment(
            tmp_path / "b.dat2", "TEST", "E", 1.0, [(2028.0, 0.35, 0.005)]
        )
        joined = join_segments([read_mb_segment(a), read_mb_segment(b)])
        corr = joined.corrections[1]
        assert corr.overlap_available is False
        assert corr.n_overlap == 0
        assert corr.correction_m == 0.0
        assert corr.raw_offset_m == pytest.approx(0.25, abs=1e-6)

    def test_estimate_requires_prior_coverage(self, tmp_path: Path) -> None:
        seg = read_mb_segment(
            _write_segment(
                tmp_path / "a.dat2", "TEST", "E", 1.0, [(2020.1, 0.1, 0.005)]
            )
        )
        with pytest.raises(GlobkJoinError, match="cannot estimate"):
            estimate_segment_offset(np.array([2025.0]), np.array([0.0]), seg)


class TestSengAcceptanceGate:
    """SENG (1 Pre + raw wrapped Rap): exact reproduction of min-σ TOT."""

    def test_east_wrap_corrected_matches_tot(self) -> None:
        joined = join_station_component("SENG", 2, [PRE, RAP])
        assert joined.corrections[1].correction_m == 10.0  # the +10 m wrap
        # header references do NOT explain the shift (Δref = 2.621 m)
        pre_ref = read_mb_segment(PRE / "mb_SENG_GPS.dat2").header.reference_m
        rap_ref = read_mb_segment(RAP / "mb_SENG_GPS.dat2").header.reference_m
        assert pre_ref - rap_ref == pytest.approx(2.621, abs=1e-3)
        # no boundary jump
        assert np.max(np.abs(np.diff(joined.values))) < 2.0
        # exact per-epoch identity with min-σ-deduped live TOT
        te, tv, ts = _read_raw_tot(TOT / "mb_SENG_TOT.dat2")
        ref_e, ref_v, _ = dedup_min_sigma(te, tv, ts)
        np.testing.assert_array_equal(joined.epochs, ref_e)
        np.testing.assert_allclose(joined.values, ref_v, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("axis", [1, 3])
    def test_north_up_no_correction_and_match(self, axis: int) -> None:
        joined = join_station_component("SENG", axis, [PRE, RAP])
        assert joined.corrections[1].correction_m == 0.0
        te, tv, ts = _read_raw_tot(TOT / f"mb_SENG_TOT.dat{axis}")
        ref_e, ref_v, _ = dedup_min_sigma(te, tv, ts)
        np.testing.assert_array_equal(joined.epochs, ref_e)
        np.testing.assert_allclose(joined.values, ref_v, rtol=0, atol=1e-9)


class TestReykDedupAcceptanceGate:
    """REYK (7 overlapping segments, 2326 dup epochs): the dedup acceptance."""

    @pytest.mark.parametrize("axis", [1, 2, 3])
    def test_join_reconstructs_tot(self, axis: int) -> None:
        joined = join_station_component("REYK", axis, [PRE, RAP])
        # all segments are co-datum: no wrap correction anywhere
        assert all(c.correction_m == 0.0 for c in joined.corrections)
        assert len(joined.corrections) == 8  # 7 Pre markers + Rap 6PS
        assert np.all(np.diff(joined.epochs) > 0)  # strictly increasing, deduped

        tot_sets = _tot_solution_sets(TOT / f"mb_REYK_TOT.dat{axis}")
        # join covers every TOT epoch (it is a superset: newer Rap epochs too)
        assert set(tot_sets).issubset({round(float(e), 5) for e in joined.epochs})
        # every joined value at a TOT epoch IS one of the real GLOBK solutions
        # present in TOT there (min-σ selects a valid solution by construction)
        not_in_tot = 0
        for ep, val in zip(joined.epochs, joined.values):
            key = round(float(ep), 5)
            if key not in tot_sets:
                continue
            if not _value_in_solutions(float(val), tot_sets[key]):
                not_in_tot += 1
        assert not_in_tot == 0

    def test_spikes_pass_through(self) -> None:
        # REYK East single-epoch outliers (~2.76 m against a ~0 m series) are
        # gross blunders for the downstream cleaner, not steps — they survive.
        joined = join_station_component("REYK", 2, [PRE, RAP])
        assert np.nanmax(np.abs(joined.values)) > 2.0


class TestHofnMultiSegmentRegression:
    """HOFN (5 Pre markers + Rap 3PS/4PS): co-datum multi-segment structure."""

    @pytest.mark.parametrize("axis", [1, 2, 3])
    def test_structure_and_tolerance(self, axis: int) -> None:
        joined = join_station_component("HOFN", axis, [PRE, RAP])
        assert len(joined.corrections) == 7
        assert all(c.correction_m == 0.0 for c in joined.corrections)  # no wrap
        assert np.all(np.diff(joined.epochs) > 0)

        tot_sets = _tot_solution_sets(TOT / f"mb_HOFN_TOT.dat{axis}")
        assert set(tot_sets).issubset({round(float(e), 5) for e in joined.epochs})
        # values agree with TOT within a few mm (residual = a mildly stale
        # TOT: sub-mm–mm Rap-3PS re-runs in 2013, documented — not a datum
        # error; the Up component is the noisiest at ~1.5 mm)
        max_dev = 0.0
        for ep, val in zip(joined.epochs, joined.values):
            key = round(float(ep), 5)
            if key not in tot_sets or not np.isfinite(val):
                continue
            finite = [t for t in tot_sets[key] if np.isfinite(t)]
            if finite:
                max_dev = max(max_dev, min(abs(float(val) - t) for t in finite))
        assert max_dev < 2e-3

    def test_no_wrap_and_guard_not_tripped(self) -> None:
        # HOFN references drift smoothly in cm (no 10 m wrap); the largest raw
        # datum offset is a few mm and the residual guard never trips.
        joined = join_station_component("HOFN", 2, [PRE, RAP])
        assert max(abs(c.raw_offset_m) for c in joined.corrections) < 0.01
        assert max(abs(c.residual_m) for c in joined.corrections) < 0.01
        # the only non-overlapping (gap) boundary is Rap 4PS at 2026
        gap = [c for c in joined.corrections if not c.overlap_available]
        assert all(abs(c.raw_offset_m) < 0.01 for c in gap)


class TestReaderDedupSafetyNet:
    """openGlobkTimes gains a guarded min-σ dedup for legacy dup-carrying TOT."""

    def test_dedups_duplicate_carrying_tot(self) -> None:
        from geo_dataread.gps_read import openGlobkTimes

        yearf, data, ddata = openGlobkTimes("REYK", Dir=str(TOT), tType="TOT")
        # duplicates removed, epoch axis strictly increasing
        assert np.unique(yearf).size == yearf.size
        assert np.all(np.diff(yearf) > 0)
        # per-component result equals dedup_min_sigma of that raw component
        for axis, comp in ((1, 0), (2, 1), (3, 2)):
            e, v, s = _read_raw_tot(TOT / f"mb_REYK_TOT.dat{axis}")
            ref_e, ref_v, ref_s = dedup_min_sigma(e, v, s)
            np.testing.assert_array_equal(yearf, ref_e)
            np.testing.assert_allclose(
                data[comp], ref_v, rtol=0, atol=0, equal_nan=True
            )
            np.testing.assert_allclose(
                ddata[comp], ref_s, rtol=0, atol=0, equal_nan=True
            )

    def test_nonduplicate_read_is_unchanged(self, tmp_path: Path) -> None:
        from geo_dataread.gps_read import openGlobkTimes

        rows1 = [(2024.0, 0.10, 0.005), (2024.5, 0.11, 0.005), (2025.0, 0.12, 0.005)]
        for axis in (1, 2, 3):
            _write_segment(tmp_path / f"mb_ZZZZ_TOT.dat{axis}", "ZZZZ", "N", 1.0, rows1)
        yearf, data, ddata = openGlobkTimes("ZZZZ", Dir=str(tmp_path), tType="TOT")
        # no duplicates -> guard skips dedup, rows returned in file order
        np.testing.assert_array_equal(yearf, np.array([2024.0, 2024.5, 2025.0]))
        np.testing.assert_allclose(data[0], np.array([0.10, 0.11, 0.12]))


class TestThobReferenceDivergence:
    def test_reference_difference_is_not_a_datum_shift(self) -> None:
        pre_ref = read_mb_segment(PRE / "mb_THOB_GPS.dat2").header.reference_m
        rap_ref = read_mb_segment(RAP / "mb_THOB_GPS.dat2").header.reference_m
        assert abs(pre_ref - rap_ref) > 1.0  # refs differ ~1.884 m
        joined = join_station_component("THOB", 2, [PRE, RAP])
        assert joined.corrections[1].correction_m == 0.0  # yet no datum shift

    def test_discover_missing_station_raises(self) -> None:
        with pytest.raises(GlobkJoinError, match="no mb_XXXX"):
            join_station_component("XXXX", 2, [PRE, RAP])
