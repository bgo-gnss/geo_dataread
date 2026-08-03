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
        assert build_trajectory_model("lineperiodic", [2008.4], None) is None
        assert build_trajectory_model("lineperiodic", [2008.4], []) is None

    def test_composes_model_steps_and_transient_in_canonical_order(self) -> None:
        tm = build_trajectory_model(
            "lineperiodic", [2008.4085], ["log@2008.4085,tau=1.0"]
        )
        assert tm.param_names == (
            "offset",
            "rate",
            "cos_annual",
            "sin_annual",
            "cos_semiannual",
            "sin_semiannual",
            "step_amp_1",
            "log_amp_1",
        )
        # param_names[1] == "rate" is what keeps velocity._RATE_INDEX correct
        assert tm.param_names[1] == "rate"

    def test_groups_are_addressable(self) -> None:
        tm = build_trajectory_model(
            "lineperiodic", [2008.4], ["log@2008.4,tau=1.0", "exp@2012.0,tau=0.5"]
        )
        assert int(tm.group_mask("transient").sum()) == 2
        assert int(tm.group_mask("step").sum()) == 1

    def test_unknown_model_composition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="what model"):
            build_trajectory_model("bogus", [], ["log@2008,tau=1"])

    def test_model_without_steps(self) -> None:
        tm = build_trajectory_model("linear", [], ["exp@2010.0,tau=0.5"])
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

        tm = build_trajectory_model("linear", [], ["log@2018.0,tau=2.0"])
        d_term = detect_outliers(tm.as_modelfunc(), t, y, s)

        frac_plain = float(np.atleast_2d(d_plain.candidates)[0].mean())
        frac_term = float(np.atleast_2d(d_term.candidates)[0].mean())
        assert frac_term < frac_plain
        assert not d_term.excess_flag_abort
