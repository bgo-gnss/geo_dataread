"""Tests for geo_dataread.detrend_estimate — gps-estimate-detrend batch CLI.

Round-trip acceptance mirrors the DYNG validation of the local-TOT
pipeline: estimate on a synthetic plate-removed series -> ``to_record``
-> document -> ``gps_views.read_detrend_params`` ->
``gps_analysis.apply_detrend`` -> the detrended series is FLAT (residual
secular slope ~0) and the fitted rate recovers the injected one. The
fit catalog (``fit_windows.csv``) is pinned against the deployed seed in
``gps-config-data/analysis-lane/fit_windows.csv`` (DYNG: ``max_gap_years
= 1.0`` — real gaps trip the leaf's default 0.5 yr gate).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import geo_dataread.detrend_estimate as de
import geo_dataread.gps_read as gps_read
from geo_dataread.detrend_estimate import (
    FitCatalogRow,
    FitDefaults,
    StationFitSettings,
    UNCERT,
    build_document,
    estimate_station,
    main,
    read_fit_catalog,
    resolve_fit_settings,
    station_record_from_arrays,
)
from geo_dataread.gps_views import read_detrend_params, station_detrend_record
from gps_analysis import apply_detrend

RATE = (4.33, 1.0, -2.0)  # injected secular rates [mm/yr] N/E/U (DYNG-ish N)
FloatArray = Any  # test-local alias; arrays are plain np.ndarray here

# In-test mirror of the deployed seed catalog (comment lines included, to
# pin the deployed-catalog convention the reader must accept).
SEED_FIT_CSV = """\
# fit_windows.csv — per-station detrend-fit windows + gate overrides.
sta,window_start,window_end,max_gap_years,min_epochs,min_span_years,steps,comment
DYNG,,,1.0,,,,gaps up to ~1 yr; default 0.5 gate trips
"""


def _synthetic_series(
    *,
    n: int = 1200,
    t0: float = 2020.0,
    gap: tuple[float, float] | None = None,
    seed: int = 7,
) -> tuple[Any, Any, Any]:
    """Plate-removed-frame synthetic: line + annual, mild noise, mm units."""
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n, dtype=np.float64) / 365.25
    if gap is not None:
        keep = (t < gap[0]) | (t > gap[1])
        t = t[keep]
    y = np.vstack(
        [
            -8.0
            + RATE[c] * (t - t0)
            + 2.0 * np.cos(2 * np.pi * t)
            + rng.normal(0.0, 0.5, t.size)
            for c in range(3)
        ]
    )
    sigma = np.full_like(y, 1.0)
    return t, y, sigma


def _settings(**overrides: Any) -> StationFitSettings:
    """Build settings, accepting ``window=`` as sugar for one segment.

    The dataclass field is ``segments`` and ``window`` is a derived
    property (the hull), so a caller cannot set it — but "one window" is
    what almost every test here means, and spelling it as a 1-tuple at
    every call site would obscure that.
    """
    base = dict(
        segments=((None, None),),
        min_span_years=2.0,
        min_epochs=365,
        max_gap_years=0.5,
        steps=(),  # explicit: no steps, never consult the deployed steps.csv
        window_source="defaults",
    )
    if "window" in overrides:
        base["segments"] = (tuple(overrides.pop("window")),)
    base.update(overrides)
    return StationFitSettings(**base)


def _empty_overrides(tmp_path: Path) -> Path:
    """Header-only outlier_overrides.csv (valid, empty catalog)."""
    path = tmp_path / "outlier_overrides.csv"
    path.write_text(
        "sta,despike,window_order,window_robust_iterations,epoch_policy,"
        "despike_n_sigma,min_outlier_n,min_outlier_e,min_outlier_u,comment\n"
    )
    return path


# ---------------------------------------------------------------------------
# Fit catalog
# ---------------------------------------------------------------------------


class TestReadFitCatalog:
    def test_parses_seed_row(self, tmp_path: Path) -> None:
        path = tmp_path / "fit_windows.csv"
        path.write_text(SEED_FIT_CSV)
        catalog = read_fit_catalog(path)
        assert set(catalog) == {"DYNG"}
        row = catalog["DYNG"]
        assert row.max_gap_years == 1.0
        assert row.window_start is None and row.window_end is None
        assert row.min_epochs is None and row.min_span_years is None
        assert row.steps is None

    def test_full_row_and_steps_parsing(self, tmp_path: Path) -> None:
        path = tmp_path / "fit_windows.csv"
        path.write_text(
            "sta,window_start,window_end,max_gap_years,min_epochs,"
            "min_span_years,steps,comment\n"
            "seng,2015.0,2020.5,0.75,200,1.5,2023.85;2020.71,pre-unrest window\n"
        )
        row = read_fit_catalog(path)["SENG"]  # upper-cased
        assert row.window_start == 2015.0 and row.window_end == 2020.5
        assert row.max_gap_years == 0.75
        assert row.min_epochs == 200
        assert row.min_span_years == 1.5
        assert row.steps == (2020.71, 2023.85)  # sorted

    def test_wrong_columns_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("sta,start,end\nDYNG,,\n")
        with pytest.raises(ValueError, match="columns"):
            read_fit_catalog(path)

    def test_duplicate_station_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text(
            "sta,window_start,window_end,max_gap_years,min_epochs,"
            "min_span_years,steps,comment\nDYNG,,,,,,,\nDYNG,,,,,,,\n"
        )
        with pytest.raises(ValueError, match="duplicate"):
            read_fit_catalog(path)

    def test_bad_number_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text(
            "sta,window_start,window_end,max_gap_years,min_epochs,"
            "min_span_years,steps,comment\nDYNG,,,soon,,,,\n"
        )
        with pytest.raises(ValueError, match="max_gap_years"):
            read_fit_catalog(path)

    def test_inverted_window_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text(
            "sta,window_start,window_end,max_gap_years,min_epochs,"
            "min_span_years,steps,comment\nDYNG,2020.0,2019.0,,,,,\n"
        )
        with pytest.raises(ValueError, match="window_end"):
            read_fit_catalog(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_fit_catalog(tmp_path / "nope.csv")


class TestResolveFitSettings:
    DEFAULTS = FitDefaults()

    def test_no_row_keeps_defaults(self) -> None:
        settings = resolve_fit_settings("DYNG", {}, self.DEFAULTS)
        assert settings.window == (None, None)
        assert settings.max_gap_years == 0.5
        assert settings.min_epochs == 365
        assert settings.min_span_years == 2.0
        assert settings.steps is None
        assert settings.window_source == "defaults"

    def test_row_overrides_only_set_fields(self) -> None:
        catalog = {"DYNG": FitCatalogRow(max_gap_years=1.0)}
        settings = resolve_fit_settings(
            "DYNG", catalog, self.DEFAULTS, catalog_source="cat.csv"
        )
        assert settings.max_gap_years == 1.0  # overridden
        assert settings.min_epochs == 365  # default kept
        assert settings.window == (None, None)
        assert settings.window_source == "cat.csv"

    def test_cli_defaults_flow_through(self) -> None:
        defaults = FitDefaults(min_span_years=1.0, min_epochs=100, max_gap_years=2.0)
        settings = resolve_fit_settings("XXXX", None, defaults)
        assert settings.min_epochs == 100
        assert settings.max_gap_years == 2.0


# ---------------------------------------------------------------------------
# Round trip: estimate -> record -> document -> read -> apply -> flat
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_estimate_apply_roundtrip_is_flat(self, tmp_path: Path) -> None:
        t, y, sigma = _synthetic_series()
        record = station_record_from_arrays(
            "DYNG",
            t,
            y,
            sigma,
            settings=_settings(),
            protect_windows=(),
            outlier_overrides=_empty_overrides(tmp_path),
        )
        assert record is not None
        # fitted secular rate recovers the injected one (the DYNG check:
        # the REAL plate-removed rate, not a stale seed)
        rates = [comp["params"][1] for comp in record["components"]]
        np.testing.assert_allclose(rates, RATE, atol=0.1)

        # document -> reader -> apply
        doc = build_document({"DYNG": record})
        path = tmp_path / "detrend_params.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        read_doc = read_detrend_params(path)
        stored, source = station_detrend_record(read_doc, "DYNG")
        assert source == "DYNG" and stored is not None

        detrended = apply_detrend(stored, t, y, frame="plate_removed")
        for c in range(3):
            slope = float(np.polyfit(t, detrended[c], 1)[0])
            assert abs(slope) < 0.05  # mm/yr — flat, the applied-detrend check

    def test_record_is_reader_valid_schema_v1(self, tmp_path: Path) -> None:
        t, y, sigma = _synthetic_series()
        record = station_record_from_arrays(
            "DYNG",
            t,
            y,
            sigma,
            settings=_settings(),
            protect_windows=(),
            outlier_overrides=_empty_overrides(tmp_path),
        )
        assert record is not None
        doc = build_document({"DYNG": record})
        assert doc["schema_version"] == 1
        assert doc["frame"] == "plate_removed"
        assert doc["units"] == {
            "displacement": "mm",
            "rate": "mm/yr",
            "time": "yearf",
        }
        assert doc["phase_convention"] == "absolute_yearf"
        assert doc["generated_at"] is None
        assert record["frame"] == "plate_removed"
        assert record["record_version"] == 1
        assert record["refs"]["window_source"] == "defaults"
        assert record["refs"]["data"] == "local TOT"

    def test_gap_gate_trips_by_default_and_catalog_unlocks(
        self, tmp_path: Path
    ) -> None:
        # the DYNG scenario: a ~0.7 yr gap trips the leaf default 0.5 gate;
        # the catalog's max_gap_years=1.0 makes the station estimable.
        t, y, sigma = _synthetic_series(n=1600, gap=(2021.0, 2021.7))
        overrides = _empty_overrides(tmp_path)
        with pytest.raises(ValueError, match="max_gap_years"):
            station_record_from_arrays(
                "DYNG",
                t,
                y,
                sigma,
                settings=_settings(),
                protect_windows=(),
                outlier_overrides=overrides,
            )
        record = station_record_from_arrays(
            "DYNG",
            t,
            y,
            sigma,
            settings=_settings(max_gap_years=1.0, window_source="cat.csv"),
            protect_windows=(),
            outlier_overrides=overrides,
        )
        assert record is not None
        assert record["refs"]["window_source"] == "cat.csv"

    def test_nonfinite_epochs_dropped(self, tmp_path: Path) -> None:
        t, y, sigma = _synthetic_series()
        y[1, 10] = np.nan
        sigma[2, 20] = np.inf
        record = station_record_from_arrays(
            "DYNG",
            t,
            y,
            sigma,
            settings=_settings(),
            protect_windows=(),
            outlier_overrides=_empty_overrides(tmp_path),
        )
        assert record is not None
        assert record["refs"]["n_nonfinite_dropped"] == 2

    def test_catalog_steps_augment_the_model(self, tmp_path: Path) -> None:
        t, y, sigma = _synthetic_series()
        step_epoch = 2021.5
        y += 8.0 * (t >= step_epoch)  # known offset in all components
        record = station_record_from_arrays(
            "DYNG",
            t,
            y,
            sigma,
            settings=_settings(steps=(step_epoch,)),
            protect_windows=(),
            outlier_overrides=_empty_overrides(tmp_path),
        )
        assert record is not None
        assert record["step_epochs"] == [step_epoch]
        assert record["param_names"][-1] == "step_amp_1"
        amp = record["components"][0]["params"][-1]
        assert abs(amp - 8.0) < 0.5
        rates = [comp["params"][1] for comp in record["components"]]
        np.testing.assert_allclose(rates, RATE, atol=0.1)

    def test_outlier_abort_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @dataclasses.dataclass
        class _Aborted:
            outlier_abort: bool = True

        monkeypatch.setattr(de, "estimate_detrend", lambda *a, **k: _Aborted())
        t, y, sigma = _synthetic_series(n=400)
        record = station_record_from_arrays(
            "DYNG",
            t,
            y,
            sigma,
            settings=_settings(),
            protect_windows=(),
            outlier_overrides=_empty_overrides(tmp_path),
        )
        assert record is None


class TestStationEstimateFromArrays:
    """The mask channel: ``n_rejected`` says how many, this says WHICH.

    Every assertion here is a count-parity or index-placement check,
    because the failure mode is silent: the mask is lifted across two
    compressions (non-finite drop, then the fit window) and an off-by-N
    index map still produces a plausible-looking figure — grey markers on
    the wrong epochs, with the right total.
    """

    def _estimate(self, tmp_path: Path, t: Any, y: Any, sigma: Any, **over: Any) -> Any:
        return de.station_estimate_from_arrays(
            "DYNG",
            t,
            y,
            sigma,
            settings=_settings(**over),
            protect_windows=(),
            outlier_overrides=_empty_overrides(tmp_path),
        )

    def test_mask_count_matches_the_records_n_rejected(self, tmp_path: Path) -> None:
        t, y, sigma = _synthetic_series()
        y[0, 100] += 40.0  # unambiguous blunders, one per component
        y[1, 200] -= 35.0
        res = self._estimate(tmp_path, t, y, sigma)
        assert res is not None
        assert res.outliers.shape == (3, t.size)
        assert list(res.record["n_rejected"]) == [
            int(v) for v in res.outliers.sum(axis=1)
        ]
        assert res.outliers[0, 100] and res.outliers[1, 200]

    def test_record_wrapper_returns_the_same_record(self, tmp_path: Path) -> None:
        t, y, sigma = _synthetic_series()
        res = self._estimate(tmp_path, t, y, sigma)
        record = station_record_from_arrays(
            "DYNG",
            t,
            y,
            sigma,
            settings=_settings(),
            protect_windows=(),
            outlier_overrides=_empty_overrides(tmp_path),
        )
        assert res is not None
        assert record == res.record

    def test_masks_lift_across_the_window_and_the_nonfinite_drop(
        self, tmp_path: Path
    ) -> None:
        # BOTH compressions at once — the case a single-compression test
        # cannot distinguish from a wrong index map.
        t, y, sigma = _synthetic_series(n=1600)
        holes = np.arange(3, t.size, 97)
        y[1, holes] = np.nan
        window = (float(t[400]), float(t[1200]))
        y[2, 700] += 45.0  # a blunder INSIDE the window
        y[2, 50] += 45.0  # ... and one OUTSIDE it
        res = self._estimate(tmp_path, t, y, sigma, window=window)
        assert res is not None

        assert int(res.in_window.sum()) == res.record["n_epochs"]
        assert list(res.record["n_rejected"]) == [
            int(v) for v in res.outliers.sum(axis=1)
        ]
        # a verdict may exist ONLY where the fit actually looked
        assert not res.outliers[:, ~res.in_window].any()
        assert not res.outliers[:, ~res.finite].any()
        assert res.finite[res.in_window].all()
        # placement, not just counts: the in-window blunder is flagged and
        # the identical out-of-window one is not (it was never judged)
        assert res.outliers[2, 700]
        assert not res.outliers[2, 50]
        judged = t[res.in_window]
        assert judged.min() >= window[0] - 1e-3
        assert judged.max() <= window[1] + 1e-3

    def test_aborted_fit_yields_no_estimate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @dataclasses.dataclass
        class _Aborted:
            outlier_abort: bool = True

        monkeypatch.setattr(de, "estimate_detrend", lambda *a, **k: _Aborted())
        t, y, sigma = _synthetic_series(n=400)
        assert self._estimate(tmp_path, t, y, sigma) is None


# ---------------------------------------------------------------------------
# Driver + CLI
# ---------------------------------------------------------------------------


def _fake_getdata(
    series: dict[str, tuple[Any, Any, Any]],
) -> Any:
    """A getData stand-in serving per-station synthetic plate-removed data."""

    def fake(
        sta: str,
        fstart: Any = None,
        fend: Any = None,
        ref: str = "itrf2008",
        Dir: Any = None,
        tType: str = "TOT",
        uncert: int = 15,
        offset: Any = None,
    ) -> tuple[Any, Any, Any, Any]:
        assert ref == "plate" and tType == "TOT"
        if sta not in series:
            return None, None, None, None
        t, y, sigma = series[sta]
        return t, y, sigma, 0.0

    return fake


class TestEstimateStation:
    def test_estimated_and_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            gps_read, "getData", _fake_getdata({"DYNG": _synthetic_series()})
        )
        overrides = _empty_overrides(tmp_path)
        ok = estimate_station(
            "DYNG",
            settings=_settings(),
            protect_windows=tmp_path / "no_protect.csv",
            outlier_overrides=overrides,
        )
        assert ok.status == "estimated"
        assert ok.record is not None
        assert "north rate 4." in ok.detail
        missing = estimate_station(
            "XXXX",
            settings=_settings(),
            protect_windows=tmp_path / "no_protect.csv",
            outlier_overrides=overrides,
        )
        assert missing.status == "error"
        assert missing.record is None

    def test_uncert_reaches_getdata_and_the_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The read-time sigma screen must be expressible AND recorded.

        It decides WHICH epochs were fitted while leaving no trace in any
        fitted quantity, so two records with identical parameters, window and
        steps can still have been fitted on different data. Before this knob
        existed the estimator took getData's default silently while
        ``gps-detrend-workbench`` used 10, which made batch and workbench
        records indistinguishable but unequal.
        """
        seen: list[int] = []
        base = _fake_getdata({"DYNG": _synthetic_series()})

        def spy(sta: str, *args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs["uncert"])
            return base(sta, *args, **kwargs)

        monkeypatch.setattr(gps_read, "getData", spy)
        overrides = _empty_overrides(tmp_path)
        kw = {
            "settings": _settings(),
            "protect_windows": tmp_path / "no_protect.csv",
            "outlier_overrides": overrides,
        }
        default = estimate_station("DYNG", **kw)  # type: ignore[arg-type]
        assert seen == [UNCERT]
        assert default.record is not None
        assert default.record["refs"]["uncert"] == UNCERT

        screened = estimate_station("DYNG", uncert=10, **kw)  # type: ignore[arg-type]
        assert seen == [UNCERT, 10]
        assert screened.record is not None
        assert screened.record["refs"]["uncert"] == 10

    def test_gate_failure_is_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gappy = _synthetic_series(n=1600, gap=(2021.0, 2021.7))
        monkeypatch.setattr(gps_read, "getData", _fake_getdata({"DYNG": gappy}))
        result = estimate_station(
            "DYNG",
            settings=_settings(),
            protect_windows=tmp_path / "no_protect.csv",
            outlier_overrides=_empty_overrides(tmp_path),
        )
        assert result.status == "error"
        assert "max_gap_years" in result.detail


class TestCli:
    def _cli_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Path, Path, list[str]]:
        gappy = _synthetic_series(n=1600, gap=(2021.0, 2021.7))
        monkeypatch.setattr(gps_read, "getData", _fake_getdata({"DYNG": gappy}))
        catalog = tmp_path / "fit_windows.csv"
        catalog.write_text(SEED_FIT_CSV)
        out = tmp_path / "detrend_params.json"
        extra = [
            "--fit-catalog",
            str(catalog),
            "--out",
            str(out),
            "--steps",
            str(tmp_path / "no_steps.csv"),
            "--protect-windows",
            str(tmp_path / "no_protect.csv"),
            "--outlier-overrides",
            str(_empty_overrides(tmp_path)),
        ]
        return catalog, out, extra

    def test_catalog_row_unlocks_dyng_and_doc_is_readable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _catalog, out, extra = self._cli_env(tmp_path, monkeypatch)
        rc = main(["DYNG", *extra])
        assert rc == 0
        doc = read_detrend_params(out)
        assert set(doc["stations"]) == {"DYNG"}
        assert doc["generator"] == "gps-estimate-detrend"
        assert doc["stations"]["DYNG"]["refs"]["window_source"] == str(_catalog)

    def test_uncert_flag_lands_in_the_document(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`--uncert` is what makes a workbench record reproducible in batch.

        The workbench prints this exact invocation after a commit, so the flag
        existing is part of that contract, not a convenience.
        """
        _catalog, out, extra = self._cli_env(tmp_path, monkeypatch)
        assert main(["DYNG", "--uncert", "10", *extra]) == 0
        assert read_detrend_params(out)["stations"]["DYNG"]["refs"]["uncert"] == 10

        assert main(["DYNG", *extra]) == 0
        assert read_detrend_params(out)["stations"]["DYNG"]["refs"]["uncert"] == UNCERT

    def test_failing_station_sets_exit_code_and_batch_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _catalog, out, extra = self._cli_env(tmp_path, monkeypatch)
        rc = main(["XXXX", "DYNG", *extra])
        assert rc == 1  # XXXX failed ...
        doc = read_detrend_params(out)
        assert set(doc["stations"]) == {"DYNG"}  # ... but DYNG was written

    def test_unstamped_output_is_byte_reproducible(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _catalog, out, extra = self._cli_env(tmp_path, monkeypatch)
        assert main(["DYNG", *extra]) == 0
        first = out.read_bytes()
        assert main(["DYNG", *extra]) == 0
        assert out.read_bytes() == first
        assert json.loads(first)["generated_at"] is None

    def test_stamp_sets_timestamps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _catalog, out, extra = self._cli_env(tmp_path, monkeypatch)
        assert main(["DYNG", "--stamp", *extra]) == 0
        doc = json.loads(out.read_text())
        assert doc["generated_at"] is not None
        assert doc["stations"]["DYNG"]["fitted_at"] == doc["generated_at"]


# ---------------------------------------------------------------------------
# Segments: a union fit domain, expressed in one catalog cell
# ---------------------------------------------------------------------------

_SEG_HEADER = (
    "sta,window_start,window_end,segments,max_gap_years,min_epochs,"
    "min_span_years,steps,comment\n"
)


class TestSegmentsColumn:
    """The 9-column layout, and the 8-column one it must not break."""

    def test_legacy_header_still_reads(self, tmp_path: Path) -> None:
        """The 37 deployed rows predate this column and must keep working.

        The header check is an exact tuple match by design — a bad fit
        catalog silently changes stored science — so adding a column had to
        become an ALLOWLIST rather than a looser check.
        """
        path = tmp_path / "legacy.csv"
        path.write_text(
            "sta,window_start,window_end,max_gap_years,min_epochs,"
            "min_span_years,steps,comment\nDYNG,,,1.0,,,,gaps\n"
        )
        row = read_fit_catalog(path)["DYNG"]
        assert row.segments is None, "absent column means single window"
        assert row.max_gap_years == 1.0

    def test_segments_cell_parses(self, tmp_path: Path) -> None:
        path = tmp_path / "seg.csv"
        path.write_text(
            _SEG_HEADER + "self,,,2002.1:2008.35;2008.7:2019.5,1.5,,,2008.40847,Olfus\n"
        )
        row = read_fit_catalog(path)["SELF"]
        assert row.segments == ((2002.1, 2008.35), (2008.7, 2019.5))
        assert row.steps == (2008.40847,)

    def test_open_bounds_are_expressible(self, tmp_path: Path) -> None:
        """An empty side means open, as the window columns already do."""
        path = tmp_path / "seg.csv"
        path.write_text(_SEG_HEADER + "SELF,,,:2008.35;2008.7:,1.5,,,,\n")
        assert read_fit_catalog(path)["SELF"].segments == (
            (None, 2008.35),
            (2008.7, None),
        )

    def test_segments_and_window_columns_together_are_refused(
        self, tmp_path: Path
    ) -> None:
        """One row must say ONE thing about which epochs are fitted."""
        path = tmp_path / "seg.csv"
        path.write_text(
            _SEG_HEADER + "SELF,2002.0,,2002.1:2008.35;2008.7:2019.5,,,,,\n"
        )
        with pytest.raises(ValueError, match="both set"):
            read_fit_catalog(path)

    @pytest.mark.parametrize(
        "cell, match",
        [
            ("2002.1-2008.35", "start:end"),
            ("2008.35:2002.1", "end 2002.1 <= start 2008.35"),
            ("2002.1:x", "not a number"),
        ],
    )
    def test_malformed_cells_are_hard_errors(
        self, tmp_path: Path, cell: str, match: str
    ) -> None:
        """Strict, like the rest of this reader: never a silent partial read."""
        path = tmp_path / "seg.csv"
        path.write_text(_SEG_HEADER + f"SELF,,,{cell},,,,,\n")
        with pytest.raises(ValueError, match=match):
            read_fit_catalog(path)

    def test_settings_carry_segments_and_derive_the_hull(self, tmp_path: Path) -> None:
        path = tmp_path / "seg.csv"
        path.write_text(_SEG_HEADER + "SELF,,,2002.1:2008.35;2008.7:2019.5,1.5,,,,\n")
        settings = resolve_fit_settings(
            "SELF", read_fit_catalog(path), FitDefaults(), catalog_source=str(path)
        )
        assert settings.segments == ((2002.1, 2008.35), (2008.7, 2019.5))
        # `window` is derived, so it cannot drift out of step with `segments`
        assert settings.window == (2002.1, 2019.5)

    def test_a_station_without_a_row_is_one_open_segment(self) -> None:
        settings = resolve_fit_settings("NONE", None, FitDefaults())
        assert settings.segments == ((None, None),)
        assert settings.window == (None, None)


class TestSegmentedEstimation:
    """End to end: the mask the caller lifts is the mask the fit used."""

    def test_in_window_comes_from_the_leaf_not_a_second_derivation(
        self, tmp_path: Path
    ) -> None:
        """The seam's contract, asserted rather than argued.

        ``station_estimate_from_arrays`` used to re-derive the window mask
        with its own ``slice_window`` call and argue exactness from "both
        sides pass identical bounds and neither overrides tol". Under a
        union of segments that convention would have to hold across J
        intervals. It now lifts ``est.window_mask`` — the mask the fit
        itself used — so there is nothing left to keep in sync.
        """
        t, y, sigma = _synthetic_series(n=2600, t0=2002.0)
        settings = _settings(
            segments=((2002.1, 2005.0), (2006.0, 2009.0)), max_gap_years=1.5
        )
        est = de.station_estimate_from_arrays(
            "TEST",
            t,
            y,
            sigma,
            settings=settings,
            outlier_overrides=str(_empty_overrides(tmp_path)),
        )
        assert est is not None
        in_window = np.asarray(est.in_window, dtype=bool)
        assert in_window.sum() == est.estimate.window_mask.sum()
        # the excised stretch got no verdict, and that is what the workbench
        # renders as "outside the window"
        excised = (t > 2005.0 + 1e-3) & (t < 2006.0 - 1e-3)
        assert not in_window[excised].any()
        assert est.record["segments"] == [[2002.1, 2005.0], [2006.0, 2009.0]]
        assert len(est.record["window"]) == 2
