"""The staged-estimation grammar: happy paths, and the refusals that matter.

Most of these tests pin REFUSALS rather than results.  Each refused spelling
is one that would otherwise change stored science silently — a renamed stage
flipping a borrow into a self-reference, flag order deciding which stage a hold
binds to, or ``@`` meaning either "inherit" or "the full span".
"""

from __future__ import annotations

import pytest

from geo_dataread.stage_plan import (
    DonorRef,
    StagePlan,
    StageRef,
    build_stage_plan,
    parse_hold_spec,
    parse_stage_spec,
    stage_plan_from_config,
    stage_plan_to_config,
)


class TestParseStageSpec:
    def test_groups_and_window(self) -> None:
        st = parse_stage_spec("clean:secular,periodic@2016.6:2019.0")
        assert st.name == "clean"
        assert st.free == ("secular", "periodic")
        assert st.segments == ((2016.6, 2019.0),)

    def test_omitting_at_inherits(self) -> None:
        # No '@' at all: segments is None, which estimate_staged reads as
        # "inherit the caller's domain".
        assert parse_stage_spec("long:secular").segments is None

    def test_colon_only_is_the_full_span(self) -> None:
        # '@:' is BOTH bounds open — distinct from inheriting, and the
        # distinction bites whenever the caller passed --segment.
        assert parse_stage_spec("long:secular@:").segments == ((None, None),)

    def test_bare_at_is_refused(self) -> None:
        # The plan's own example wrote '--stage long:secular@'. It is ambiguous
        # between the two cases above, so it is rejected rather than guessed.
        with pytest.raises(ValueError, match="bare trailing '@' is ambiguous"):
            parse_stage_spec("long:secular@")

    def test_open_bounds_and_multiple_segments(self) -> None:
        st = parse_stage_spec("s:secular@:2008.35;2008.7:")
        assert st.segments == ((None, 2008.35), (2008.7, None))

    @pytest.mark.parametrize(
        "spec, match",
        [
            ("noseparator", "the ':' after the stage name is required"),
            (":secular", "empty stage name"),
            ("s:", "declares no free term group"),
            ("s:secular,secular", "repeats group"),
            ("s:secular@2019:2016", r"end 2016.0 <= start 2019.0"),
            ("s:secular@notayear:2019", "not a fractional year"),
            ("s:secular@2016", "must be 'START:END'"),
        ],
    )
    def test_rejections(self, spec: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            parse_stage_spec(spec)


class TestParseHoldSpec:
    def test_stage_and_donor_kinds(self) -> None:
        assert parse_hold_spec("periodic=stage:clean") == (
            None,
            "periodic",
            StageRef("clean"),
        )
        assert parse_hold_spec("secular=donor:OLAC") == (
            None,
            "secular",
            DonorRef("OLAC"),
        )

    def test_explicit_stage_binding(self) -> None:
        assert parse_hold_spec("long:periodic=stage:clean") == (
            "long",
            "periodic",
            StageRef("clean"),
        )

    def test_bare_value_is_refused_and_names_both_spellings(self) -> None:
        # THE refusal. 'clean' could be a stage name or a station code, and the
        # two produce different record provenance, so renaming a stage must
        # never silently turn a self-reference into a borrow.
        with pytest.raises(ValueError) as exc:
            parse_hold_spec("periodic=clean")
        msg = str(exc.value)
        assert "periodic=stage:" in msg and "periodic=donor:" in msg
        assert "never inferred" in msg

    @pytest.mark.parametrize(
        "spec",
        [
            "periodic",
            "=stage:clean",
            "periodic=",
            "periodic=bogus:clean",
            "periodic=stage:",
            ":periodic=stage:clean",
            "long:=stage:clean",
        ],
    )
    def test_rejections(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_hold_spec(spec)


class TestBuildStagePlan:
    def test_the_askja_manoeuvre(self) -> None:
        # katlafitlong as two stages, straight from the program plan.
        plan = build_stage_plan(
            ["clean:secular,periodic@2001.6:2019.5", "long:secular"],
            ["long:periodic=stage:clean"],
        )
        assert plan.names == ("clean", "long")
        assert plan.stages[0].segments == ((2001.6, 2019.5),)
        assert plan.stages[1].segments is None
        assert plan.stages[1].held == {"periodic": StageRef("clean")}
        assert plan.donors == ()

    def test_borrow_one_liner_needs_no_stage_prefix(self) -> None:
        # JONC borrows OLAC's secular and fits its own seasonal: the stage
        # names what it DOES estimate, and the borrowed group is held.
        plan = build_stage_plan(["fit:periodic"], ["secular=donor:OLAC"])
        assert plan.stages[0].held == {"secular": DonorRef("OLAC")}
        assert plan.donors == ("OLAC",)

    def test_unbound_hold_refused_once_ambiguous(self) -> None:
        # Same hold spelling that is fine with one stage becomes an error with
        # two, rather than binding to whichever --stage came last.
        with pytest.raises(ValueError, match="does not say which stage"):
            build_stage_plan(["a:secular", "b:secular"], ["periodic=stage:a"])

    def test_error_names_the_fix_and_the_reason(self) -> None:
        with pytest.raises(ValueError) as exc:
            build_stage_plan(["a:secular", "b:secular"], ["periodic=stage:a"])
        msg = str(exc.value)
        assert "--hold STAGE:periodic=stage:a" in msg
        assert "ORDER" in msg

    def test_hold_must_reference_an_earlier_stage(self) -> None:
        with pytest.raises(ValueError, match="not earlier than"):
            build_stage_plan(
                ["clean:secular", "long:periodic"],
                ["clean:periodic=stage:long"],
            )

    def test_hold_cannot_reference_itself(self) -> None:
        with pytest.raises(ValueError, match="not earlier than"):
            build_stage_plan(["a:secular"], ["a:periodic=stage:a"])

    def test_stage_ref_must_actually_fit_the_group(self) -> None:
        with pytest.raises(ValueError, match="does not fit group"):
            build_stage_plan(
                ["clean:secular", "long:secular"],
                ["long:periodic=stage:clean"],
            )

    def test_group_cannot_be_free_and_held(self) -> None:
        with pytest.raises(ValueError, match="estimated or held, never both"):
            build_stage_plan(
                ["clean:periodic", "long:secular"],
                ["long:secular=stage:clean"],
            )

    def test_undeclared_stage_names(self) -> None:
        with pytest.raises(ValueError, match="not\n?\\s*declared|not declared"):
            build_stage_plan(["a:secular"], ["a:periodic=stage:nope"])
        with pytest.raises(ValueError, match="not declared"):
            build_stage_plan(["a:secular"], ["nope:periodic=stage:a"])

    def test_duplicate_stage_names(self) -> None:
        with pytest.raises(ValueError, match="duplicate stage name"):
            build_stage_plan(["a:secular", "a:periodic"])

    def test_hold_without_stage_is_refused_with_a_worked_example(self) -> None:
        with pytest.raises(ValueError, match="--stage fit:secular"):
            build_stage_plan([], ["secular=donor:OLAC"])

    def test_group_names_are_not_enumerated(self) -> None:
        # A group this module has never heard of must pass through untouched,
        # so the transient terms become addressable with no CLI change.
        plan = build_stage_plan(
            ["a:transient,secular", "b:secular"], ["b:transient=stage:a"]
        )
        assert plan.stages[0].free == ("transient", "secular")
        assert plan.stages[1].held == {"transient": StageRef("a")}

    def test_empty_plan_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one stage"):
            StagePlan(())


class TestConfigRoundTrip:
    @pytest.mark.parametrize(
        "stages, holds",
        [
            (
                ["clean:secular,periodic@2001.6:2019.5", "long:secular"],
                ["long:periodic=stage:clean"],
            ),
            (["fit:periodic"], ["secular=donor:OLAC"]),
            (["a:secular@:2008.35;2008.7:"], []),
            (["a:secular@:"], []),
            (["a:transient,secular", "b:secular"], ["b:transient=stage:a"]),
        ],
    )
    def test_plan_to_config_and_back(self, stages: list[str], holds: list[str]) -> None:
        plan = build_stage_plan(stages, holds)
        again = stage_plan_from_config(stage_plan_to_config(plan))
        assert again == plan
        # and stable on a second pass
        assert stage_plan_to_config(again) == stage_plan_to_config(plan)

    def test_inherit_survives_the_round_trip(self) -> None:
        # segments=None must not be rendered as '@:' — that would silently
        # convert "inherit the caller's domain" into "the full span".
        cfg = stage_plan_to_config(build_stage_plan(["long:secular"]))
        assert "segments" not in cfg[0]
        assert stage_plan_from_config(cfg).stages[0].segments is None

    def test_donor_stored_as_pointer_not_values(self) -> None:
        cfg = stage_plan_to_config(
            build_stage_plan(["fit:periodic"], ["secular=donor:OLAC"])
        )
        assert cfg[0]["hold"] == {"secular": "donor:OLAC"}

    def test_config_rejects_incomplete_entries(self) -> None:
        with pytest.raises(ValueError, match="needs 'name' and 'free'"):
            stage_plan_from_config([{"free": ["secular"]}])
        with pytest.raises(ValueError, match="needs 'name' and 'free'"):
            stage_plan_from_config([{"name": "a"}])
