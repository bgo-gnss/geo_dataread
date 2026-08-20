"""The --term grammar. Refusals matter more than the happy path."""

from __future__ import annotations

import pytest

from geo_dataread.term_spec import build_trajectory_model, parse_term_spec


class TestParseTermSpec:
    def test_log_and_exp(self) -> None:
        assert parse_term_spec("log@2008.4085,tau=1.0") == {
            "kind": "log_transient",
            "epoch": 2008.4085,
            "tau": 1.0,
        }
        assert parse_term_spec("exp@2010.0,tau=0.33") == {
            "kind": "exp_transient",
            "epoch": 2010.0,
            "tau": 0.33,
        }

    def test_tau_is_required_not_defaulted(self) -> None:
        # Bevis & Brown's 1 yr is a STARTING guess for a per-station search,
        # not a value to store silently behind an operator's back.
        with pytest.raises(ValueError, match="needs an explicit tau"):
            parse_term_spec("log@2008.4")

    def test_amplitude_cannot_be_declared(self) -> None:
        with pytest.raises(ValueError, match="amplitude is ESTIMATED"):
            parse_term_spec("log@2008.4,tau=1.0,amp=25")

    @pytest.mark.parametrize(
        "spec, match",
        [
            ("log2008", "must be 'KIND@EPOCH"),
            ("logg@2008,tau=1", "unknown kind"),
            ("log@notayear,tau=1", "not a fractional year"),
            ("log@2008,tau=0", "tau must be positive"),
            ("log@2008,tau=-1", "tau must be positive"),
            ("log@2008,tau=x", "not a number"),
            ("log@2008,tau", "expected 'name=value'"),
            ("log@", "names no epoch"),
        ],
    )
    def test_rejections(self, spec: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            parse_term_spec(spec)


class TestBuildTrajectoryModel:
    def test_no_terms_keeps_the_registry_path(self) -> None:
        # None, not an empty model: a caller without transients must keep the
        # ordinary path and its byte-identical version-1 record.
        assert build_trajectory_model("lineperiodic", None) is None
        assert build_trajectory_model("lineperiodic", []) is None

    def test_composes_model_and_transient_in_canonical_order(self) -> None:
        tm = build_trajectory_model("lineperiodic", ["log@2008.4085,tau=1.0"])
        assert tm.param_names == (
            "offset",
            "rate",
            "cos_annual",
            "sin_annual",
            "cos_semiannual",
            "sin_semiannual",
            "log_amp_1",
        )
        # param_names[1] == "rate" is what keeps velocity._RATE_INDEX correct
        assert tm.param_names[1] == "rate"

    def test_steps_are_not_composed_in(self) -> None:
        # They travel as step_epochs so estimate_detrend's window filter and
        # its non-separability refusal apply. Baking them in bypassed both.
        tm = build_trajectory_model("lineperiodic", ["log@2008.4085,tau=1.0"])
        assert int(tm.group_mask("step").sum()) == 0
        assert not any(n.startswith("step_amp") for n in tm.param_names)

    def test_the_step_group_appears_once_augmented(self) -> None:
        # with_steps is the one augmentation site, shared with the registry
        # path -- and the step group stays addressable by a stage plan.
        from gps_analysis import with_steps
        from gps_analysis.staged import group_parameter_mask

        tm = build_trajectory_model(
            "lineperiodic", ["log@2008.4,tau=1.0", "exp@2012.0,tau=0.5"]
        )
        aug = with_steps(tm.as_modelfunc(), [2008.4])
        assert int(group_parameter_mask(aug, "transient").sum()) == 2
        assert int(group_parameter_mask(aug, "step").sum()) == 1
        assert int(group_parameter_mask(aug, "secular").sum()) == 2

    def test_unknown_model_composition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="what model"):
            build_trajectory_model("bogus", ["log@2008,tau=1"])

    def test_model_without_steps(self) -> None:
        tm = build_trajectory_model("linear", ["exp@2010.0,tau=0.5"])
        assert tm.param_names == ("offset", "rate", "exp_amp_1")


class TestTransientFixesAnAbort:
    """The reason --term exists: an abort is a MODEL-ADEQUACY problem.

    Measured on NYLA 2026-08-03: re-judging the epochs against a staged
    trajectory leaves candidate fractions at [0.089, 0.116, 0.004] and the
    abort standing, because the staged fit is the same model with a
    differently-estimated seasonal. A transient that can follow the signal
    drops them to ~[0.006, 0.004, 0.006].
    """

    def test_transient_reduces_candidates_on_a_deforming_series(self) -> None:
        import numpy as np
        from gps_analysis import detect_outliers
        from gps_analysis.detrend import _resolve_model

        rng = np.random.default_rng(0)
        t = np.linspace(2006.0, 2026.0, 3000)
        dt = np.clip(t - 2018.0, 0.0, None)
        # deformation the straight line cannot follow
        y = 2 * (t - 2006) + 60 * np.log1p(dt / 2.0) + rng.normal(0, 2.0, t.size)
        s = np.full(t.size, 2.0)

        plain, _ = _resolve_model("linear")
        d_plain = detect_outliers(plain, t, y, s)

        tm = build_trajectory_model("linear", ["log@2018.0,tau=2.0"])
        d_term = detect_outliers(tm.as_modelfunc(), t, y, s)

        frac_plain = float(np.atleast_2d(d_plain.candidates)[0].mean())
        frac_term = float(np.atleast_2d(d_term.candidates)[0].mean())
        assert frac_term < frac_plain
        assert not d_term.excess_flag_abort


class TestTermComposesWithStages:
    """--term and --stage together: the composed model is not a registry code."""

    def test_restage_rebuilds_a_composed_model_from_its_spec(self) -> None:
        # Regression: _restage resolved the model by NAME, and a --term
        # model is called "polynomial+seasonal+log_transient", so the two
        # flags together failed with "unknown model".
        import numpy as np

        from geo_dataread.stage_plan import build_stage_plan
        from geo_dataread.detrend_estimate import (
            FitDefaults,
            resolve_fit_settings,
            station_estimate_from_arrays,
        )

        rng = np.random.default_rng(0)
        t = np.linspace(2006.0, 2026.0, 2400)
        dt = np.clip(t - 2018.0, 0.0, None)
        base = 2 * (t - 2006) + 3 * np.cos(2 * np.pi * t) + 40 * np.log1p(dt / 2.0)
        y = np.vstack([base + rng.normal(0, 2.0, t.size) for _ in range(3)])
        s = np.full_like(y, 2.0)

        settings = resolve_fit_settings("TEST", None, FitDefaults(max_gap_years=3.0))
        plan = build_stage_plan(
            ["clean:secular,periodic@2006.0:2018.0", "long:secular,transient"],
            ["long:periodic=stage:clean"],
        )

        est = station_estimate_from_arrays(
            "TEST",
            t,
            y,
            s,
            settings=settings,
            terms=["log@2018.0,tau=2.0"],
            stage_plan=plan,
            lookup_donor=None,
        )
        assert est is not None
        rec = est.record
        assert rec["record_version"] == 2
        assert [x["kind"] for x in rec["terms"]][-1] == "log_transient"
        # the staged plan really ran
        assert "stage_plan" in rec
        assert [s_["name"] for s_ in rec["stage_plan"]] == ["clean", "long"]


class TestStepsAreScreenedUnderTerm:
    """The guards on ``step_epochs`` must apply to the ``--term`` path too.

    Regression for the 2026-08-09 finding: ``build_trajectory_model`` baked
    the declared steps into the composed model and the caller then passed
    ``step_epochs=None``, so ``estimate_detrend``'s two protections -- drop
    steps outside the window hull, refuse two steps with no fitted epoch
    between them -- were silently skipped for exactly the curated stations
    ``--term`` exists for. A step outside the window became an all-ones
    column, collinear with the intercept, and a record metres off the truth
    was stored on nothing louder than an OptimizeWarning.
    """

    @staticmethod
    def _series(seed: int = 0):
        import numpy as np

        rng = np.random.default_rng(seed)
        t = np.linspace(2006.0, 2026.0, 2400)
        dt = np.clip(t - 2018.0, 0.0, None)
        base = 2 * (t - 2006) + 40 * np.log1p(dt / 2.0)
        y = np.vstack([base + rng.normal(0, 2.0, t.size) for _ in range(3)])
        return t, y, np.full_like(y, 2.0)

    @staticmethod
    def _settings(segments, steps):
        from geo_dataread.detrend_estimate import StationFitSettings

        return StationFitSettings(
            segments=segments,
            min_span_years=2.0,
            min_epochs=365,
            max_gap_years=3.0,
            steps=steps,
            window_source="test",
        )

    def test_a_step_outside_the_window_is_dropped_under_term(self) -> None:
        # Declared 2007.0, window opens 2010.0: the registry path drops it
        # (the intercept absorbs it). The --term path must do the same.
        from geo_dataread.detrend_estimate import station_estimate_from_arrays

        t, y, s = self._series()
        settings = self._settings(((2010.0, None),), (2007.0,))

        est = station_estimate_from_arrays(
            "TEST", t, y, s, settings=settings, terms=["log@2018.0,tau=2.0"]
        )
        assert est is not None
        assert list(est.estimate.step_epochs) == []
        names = est.record["param_names"]
        assert not any(n.startswith("step_amp") for n in names)
        # and the fit is sane. Not via the offset -- the polynomial is
        # referenced to t = 0, so a healthy offset is ~ -4012 mm. The
        # collinear-column failure showed as a trajectory metres from the
        # data, so the residual is what states it: sigma is 2.0 by
        # construction.
        assert all(r < 5.0 for r in est.estimate.rms)

    def test_two_inseparable_steps_are_refused_under_term(self) -> None:
        # Both inside one outage: identical Heaviside columns, amplitudes
        # meaningless. The registry path raises; so must this one.
        import numpy as np
        import pytest

        from geo_dataread.detrend_estimate import station_estimate_from_arrays

        t, y, s = self._series()
        keep = ~((t > 2015.0) & (t < 2016.0))
        t, y, s = t[keep], y[:, keep], s[:, keep]
        settings = self._settings(((None, None),), (2015.3, 2015.6))

        with pytest.raises(ValueError, match="no fitted epoch between them"):
            station_estimate_from_arrays(
                "TEST", t, y, s, settings=settings, terms=["log@2018.0,tau=2.0"]
            )
        assert np.isfinite(y).all()  # the refusal, not a data accident

    def test_a_v2_record_with_steps_round_trips(self) -> None:
        # write -> read -> evaluate. If the reader dropped step_epochs the
        # evaluated trajectory would be off by the step amplitude after it.
        import numpy as np

        from gps_analysis.detrend import evaluate_record
        from geo_dataread.detrend_estimate import station_estimate_from_arrays

        t, y, s = self._series()
        y = y + np.where(t >= 2012.0, 100.0, 0.0)  # a step no reader may lose
        settings = self._settings(((None, None),), (2012.0,))

        est = station_estimate_from_arrays(
            "TEST", t, y, s, settings=settings, terms=["log@2018.0,tau=2.0"]
        )
        assert est is not None
        rec = est.record
        assert rec["record_version"] == 2
        # steps live beside the terms, not inside them
        assert [x["kind"] for x in rec["terms"]] == [
            "polynomial",
            "seasonal",
            "log_transient",
        ]
        assert rec["step_epochs"] == [2012.0]
        assert rec["param_names"][-1] == "step_amp_1"

        got = evaluate_record(rec, t)
        assert np.max(np.abs(got[0] - y[0])) < 15.0

        # And the APPLY path, which is what --view detrended renders through.
        # The terms spec is now SHORT by n_steps (it describes the unaugmented
        # model), so a keep-mask built from the spec instead of param_names
        # would be the wrong length -- the same writer/reader asymmetry this
        # fix closed in to_record and trajectory_from_record.
        tt = np.array([2011.0, 2013.0, 2025.0])
        secular = evaluate_record(rec, tt, terms="secular")[0]
        periodic = evaluate_record(rec, tt, terms="periodic")[0]
        assert np.isfinite(secular).all() and secular.shape == tt.shape
        # steps fold into secular (the 37 deployed records depend on it), so
        # the ~100 mm jump survives while the transient does not
        assert 90.0 < float(secular[1] - secular[0]) < 115.0
        assert float(secular[2]) < float(evaluate_record(rec, tt)[0][2]) - 30.0
        assert np.max(np.abs(periodic)) < 10.0

    def test_a_pre_2026_08_09_v2_record_still_reads(self) -> None:
        # Old shape: steps INSIDE terms, step_epochs empty. Nothing on disk
        # today (all 37 deployed records are v1), but the compatibility claim
        # is what lets this fix ship without a migration -- so it is pinned.
        import numpy as np

        from gps_analysis.detrend import evaluate_record, trajectory_from_record

        old = {
            "record_version": 2,
            "model": "polynomial+seasonal+step+log_transient",
            "param_names": [
                "offset",
                "rate",
                "cos_annual",
                "sin_annual",
                "cos_semiannual",
                "sin_semiannual",
                "step_amp_1",
                "log_amp_1",
            ],
            "terms": [
                {"kind": "polynomial", "degree": 1},
                {"kind": "seasonal", "n_harmonics": 2},
                {"kind": "step", "epoch": 2012.0},
                {"kind": "log_transient", "epoch": 2018.0, "tau": 2.0},
            ],
            "step_epochs": [],
            "frame": "plate",
            "detrend_method": "wls",
            "fitted_at": "2026-08-05T00:00:00Z",
            "window": [2006.0, 2026.0],
            "components": [
                {
                    "component": "north",
                    "params": [1.0, 2.0, 0.5, 0.5, 0.1, 0.1, 100.0, 40.0],
                    "rms": 2.0,
                }
            ],
        }
        model, fits = trajectory_from_record(old)
        assert len(fits[0].params) == 8
        t = np.array([2011.0, 2013.0])
        got = evaluate_record(old, t)
        # the baked step still fires: ~100 mm between the two epochs
        assert 90.0 < float(got[0][1] - got[0][0]) < 115.0
