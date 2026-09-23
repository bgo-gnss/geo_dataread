"""The staged-estimation grammar: happy paths, and the refusals that matter.

Most of these tests pin REFUSALS rather than results.  Each refused spelling
is one that would otherwise change stored science silently — a renamed stage
flipping a borrow into a self-reference, flag order deciding which stage a hold
binds to, or ``@`` meaning either "inherit" or "the full span".
"""

from __future__ import annotations

import numpy as np
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
            ("s:secular,secular", "repeats group"),
            ("s:secular@2019:2016", r"end 2016.0 <= start 2019.0"),
            ("s:secular@notayear:2019", "not a fractional year"),
            ("s:secular@2016", "must be 'START:END'"),
        ],
    )
    def test_rejections(self, spec: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            parse_stage_spec(spec)

    def test_empty_free_is_legal_grammar(self) -> None:
        # An apply-only stage ('apply:') frees nothing on purpose -- the
        # fully-borrowed station. Whether it HOLDS anything is only knowable
        # once the holds are attached, so the refusal lives in
        # build_stage_plan (tested there).
        assert parse_stage_spec("apply:").free == ()


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
        assert cfg[0]["held"] == {"secular": "donor:OLAC"}

    def test_config_rejects_incomplete_entries(self) -> None:
        with pytest.raises(ValueError, match="needs 'name'"):
            stage_plan_from_config([{"free": ["secular"]}])
        # An entry with a name but neither free groups nor holds is the
        # estimates-nothing-holds-nothing stage, refused in build_stage_plan.
        with pytest.raises(ValueError, match="no free term group and no --hold"):
            stage_plan_from_config([{"name": "a"}])


class TestDonorResolution:
    """Donor holds are POINTERS: re-estimating a donor must propagate."""

    @staticmethod
    def _record(params: list[float], fitted_at: str = "2026-01-01") -> dict:
        return {
            "model": "lineperiodic",
            "fitted_at": fitted_at,
            "components": [{"params": params}],
        }

    def test_slices_the_right_group(self) -> None:
        from geo_dataread.stage_plan import donor_group_values

        rec = self._record([10.0, 2.0, 1.0, -1.0, 0.5, -0.5])
        assert list(donor_group_values(rec, "secular", component=0, donor="OLAC")) == [
            10.0,
            2.0,
        ]
        assert list(donor_group_values(rec, "periodic", component=0, donor="OLAC")) == [
            1.0,
            -1.0,
            0.5,
            -0.5,
        ]

    # A cross-station borrow is RE-ANCHORED (2026-08-26), through EITHER hold
    # kind, so these now pass the borrower's own series. The datum is the
    # borrower's either way; what is under test here is the pointer decision
    # and the provenance, and the RATE is what carries both.
    _ANCHOR = dict(
        station="JONC",
        anchor_t=np.array([2020.0, 2020.5, 2021.0]),
        anchor_y=np.array([0.0, 1.0, 2.0]),
        anchor_window=(2020.0, 2021.0),
    )

    def test_reestimated_donor_propagates(self) -> None:
        # THE test for the pointer decision. The plan is unchanged; only the
        # donor's record changes. A copy-semantics implementation would keep
        # serving the old numbers here.
        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:periodic"], ["secular=donor:OLAC"])
        store = {"OLAC": self._record([10.0, 2.0, 0, 0, 0, 0], "2026-01-01")}

        before = resolve_stage_plan(
            plan, lookup_donor=lambda s: store[s], component=0, **self._ANCHOR
        )
        assert before[0].held["secular"].values[1] == 2.0

        store["OLAC"] = self._record([11.0, 2.5, 0, 0, 0, 0], "2026-07-01")
        after = resolve_stage_plan(
            plan, lookup_donor=lambda s: store[s], component=0, **self._ANCHOR
        )
        assert after[0].held["secular"].values[1] == 2.5

    def test_the_donor_rate_crosses_but_the_datum_does_not(self) -> None:
        """The half of the borrow that must NOT propagate.

        The donor's intercept is the donor's own level; carried verbatim it
        lands in whatever is free. Two donors differing ONLY in intercept
        must therefore give the borrower the same held values.
        """
        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:periodic"], ["secular=donor:OLAC"])
        out = []
        for intercept in (10.0, 4000.0):
            res = resolve_stage_plan(
                plan,
                lookup_donor=lambda s, c=intercept: self._record(
                    [c, 2.0, 0, 0, 0, 0], "2026-01-01"
                ),
                component=0,
                **self._ANCHOR,
            )
            out.append(list(res[0].held["secular"].values))
        assert out[0] == pytest.approx(out[1]), (
            "the donor's datum crossed into the borrower"
        )

    def test_provenance_names_donor_and_vintage(self) -> None:
        # A stored borrow can then be CHECKED against a re-estimated donor
        # rather than merely asserted.
        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:periodic"], ["secular=donor:OLAC"])
        out = resolve_stage_plan(
            plan,
            lookup_donor=lambda s: self._record([1, 2, 0, 0, 0, 0], "2026-07-01"),
            component=0,
            **self._ANCHOR,
        )
        source = out[0].held["secular"].source
        assert source.startswith("donor:OLAC@2026-07-01")
        assert "anchored [" in source

    def test_stage_refs_need_no_lookup(self) -> None:
        from gps_analysis.staged import HeldFromStage

        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(
            ["clean:secular,periodic", "long:secular"],
            ["long:periodic=stage:clean"],
        )

        def _boom(sta: str) -> dict:
            raise AssertionError(f"should not look up {sta}")

        out = resolve_stage_plan(plan, lookup_donor=_boom, component=0)
        assert out[1].held["periodic"] == HeldFromStage("clean")

    def test_missing_donor_is_loud(self) -> None:
        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:periodic"], ["secular=donor:NOPE"])
        with pytest.raises(KeyError):
            resolve_stage_plan(plan, lookup_donor=lambda s: {}[s], component=0)

    def test_group_absent_from_donor_model_is_refused(self) -> None:
        # An empty borrow would hold a group at nothing and still look like a
        # successful fit.
        from geo_dataread.stage_plan import donor_group_values

        rec = {"model": "linear", "components": [{"params": [1.0, 2.0]}]}
        with pytest.raises(ValueError, match="no 'periodic' term"):
            donor_group_values(rec, "periodic", component=0, donor="OLAC")

    def test_malformed_donor_records(self) -> None:
        from geo_dataread.stage_plan import donor_group_values

        with pytest.raises(ValueError, match="no model code"):
            donor_group_values(
                {"components": [{"params": [1.0]}]}, "secular", component=0, donor="X"
            )
        with pytest.raises(ValueError, match="no component 3"):
            donor_group_values(
                self._record([1, 2, 0, 0, 0, 0]), "secular", component=3, donor="X"
            )
        with pytest.raises(ValueError, match="but the record stores"):
            donor_group_values(
                {"model": "lineperiodic", "components": [{"params": [1.0, 2.0]}]},
                "secular",
                component=0,
                donor="X",
            )


class TestAnalysisYamlPersistence:
    """Stage plans live in analysis.yaml beside fit_windows and use_sta."""

    @staticmethod
    def _write(tmp_path, doc: dict):
        import yaml

        p = tmp_path / "analysis.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        return p

    def test_absent_file_block_or_map_all_mean_none(self, tmp_path) -> None:
        from geo_dataread.stage_plan import read_stage_plans

        assert read_stage_plans(tmp_path / "nope.yaml") == {}
        assert read_stage_plans(self._write(tmp_path, {})) == {}
        assert read_stage_plans(self._write(tmp_path, {"detrend": {}})) == {}
        assert (
            read_stage_plans(
                self._write(tmp_path, {"detrend": {"estimation": {"stage_plans": {}}}})
            )
            == {}
        )

    def test_reads_a_plan(self, tmp_path) -> None:
        from geo_dataread.stage_plan import read_stage_plans

        plan = build_stage_plan(
            ["clean:secular,periodic@2001.6:2019.5", "long:secular"],
            ["long:periodic=stage:clean"],
        )
        p = self._write(
            tmp_path,
            {
                "detrend": {
                    "estimation": {"stage_plans": {"OLAC": stage_plan_to_config(plan)}}
                }
            },
        )
        assert read_stage_plans(p) == {"OLAC": plan}

    def test_malformed_plan_names_the_station(self, tmp_path) -> None:
        # A stage plan that silently failed to load would quietly revert the
        # station to single-stage estimation and store different science.
        from geo_dataread.stage_plan import read_stage_plans

        p = self._write(
            tmp_path,
            {"detrend": {"estimation": {"stage_plans": {"OLAC": [{"name": "a"}]}}}},
        )
        with pytest.raises(ValueError, match="OLAC"):
            read_stage_plans(p)

        p = self._write(
            tmp_path, {"detrend": {"estimation": {"stage_plans": {"OLAC": "nope"}}}}
        )
        with pytest.raises(ValueError, match="must be a list of stages"):
            read_stage_plans(p)

    def test_write_preserves_everything_else(self, tmp_path) -> None:
        # The --commit contract: merge ONE station, never rewrite the doc.
        import yaml

        from geo_dataread.stage_plan import read_stage_plans, write_stage_plan

        p = self._write(
            tmp_path,
            {
                "version": 0,
                "outliers": {"window_n_sigma": 4.0},
                "detrend": {
                    "default_model": "lineperiodic",
                    "estimation": {
                        "min_epochs": 365,
                        "use_sta": {"JONC": "OLAC"},
                        "stage_plans": {"KASC": [{"name": "f", "free": ["secular"]}]},
                    },
                },
            },
        )
        plan = build_stage_plan(["clean:secular,periodic"], [])
        write_stage_plan(p, "OLAC", plan)

        doc = yaml.safe_load(p.read_text())
        assert doc["version"] == 0
        assert doc["outliers"] == {"window_n_sigma": 4.0}
        assert doc["detrend"]["default_model"] == "lineperiodic"
        assert doc["detrend"]["estimation"]["min_epochs"] == 365
        assert doc["detrend"]["estimation"]["use_sta"] == {"JONC": "OLAC"}
        # the other station's plan survives
        assert set(read_stage_plans(p)) == {"KASC", "OLAC"}
        assert read_stage_plans(p)["OLAC"] == plan

    def test_write_into_a_file_without_the_blocks(self, tmp_path) -> None:
        from geo_dataread.stage_plan import read_stage_plans, write_stage_plan

        p = self._write(tmp_path, {"version": 0})
        plan = build_stage_plan(["fit:secular"])
        write_stage_plan(p, "SELF", plan)
        assert read_stage_plans(p) == {"SELF": plan}

    def test_removal_returns_a_station_to_single_stage(self, tmp_path) -> None:
        import yaml

        from geo_dataread.stage_plan import read_stage_plans, write_stage_plan

        p = self._write(tmp_path, {"version": 0})
        write_stage_plan(p, "SELF", build_stage_plan(["fit:secular"]))
        write_stage_plan(p, "OLAC", build_stage_plan(["fit:periodic"]))
        write_stage_plan(p, "SELF", None)
        assert set(read_stage_plans(p)) == {"OLAC"}

        write_stage_plan(p, "OLAC", None)
        assert read_stage_plans(p) == {}
        # the emptied block is cleaned up rather than left as a stub
        assert (
            "stage_plans" not in yaml.safe_load(p.read_text())["detrend"]["estimation"]
        )

    def test_write_round_trips_through_the_file(self, tmp_path) -> None:
        from geo_dataread.stage_plan import read_stage_plans, write_stage_plan

        p = self._write(tmp_path, {})
        for sta, stages, holds in [
            (
                "OLAC",
                ["clean:secular,periodic@2001.6:2019.5", "long:secular"],
                ["long:periodic=stage:clean"],
            ),
            ("JONC", ["fit:periodic"], ["secular=donor:OLAC"]),
            ("SELF", ["a:secular@:2008.35;2008.7:"], []),
        ]:
            write_stage_plan(p, sta, build_stage_plan(stages, holds))
        got = read_stage_plans(p)
        assert got["OLAC"] == build_stage_plan(
            ["clean:secular,periodic@2001.6:2019.5", "long:secular"],
            ["long:periodic=stage:clean"],
        )
        assert got["JONC"].donors == ("OLAC",)
        assert got["SELF"].stages[0].segments == ((None, 2008.35), (2008.7, None))


class TestRecordIsValidConfig:
    """A stored record's stage_plan parses back into the plan that made it.

    ``StagedEstimate.to_record_fragment`` and :func:`stage_plan_to_config`
    emit the SAME keys and the same ``stage:``/``donor:`` hold spellings.
    That makes "what you judged is what got stored" checkable by parsing the
    record, rather than asserted in a docstring.
    """

    def test_staged_record_fragment_parses_as_a_plan(self) -> None:
        import numpy as np
        from gps_analysis import HeldFromStage, Stage, estimate_staged

        from geo_dataread.stage_plan import stage_plan_from_config

        rng = np.random.default_rng(0)
        t = np.linspace(2000, 2020, 900)
        y = 5 + 2 * (t - 2000) + 3 * np.cos(2 * np.pi * t) + rng.normal(0, 0.5, t.size)

        cli = ["clean:secular,periodic@2001.6:2012.0", "long:secular"]
        holds = ["long:periodic=stage:clean"]
        declared = build_stage_plan(cli, holds)

        est = estimate_staged(
            "lineperiodic",
            t,
            y,
            plan=[
                Stage("clean", ("secular", "periodic"), segments=[(2001.6, 2012.0)]),
                Stage("long", ("secular",), held={"periodic": HeldFromStage("clean")}),
            ],
        )
        # The record round-trips into exactly the plan the operator declared.
        assert (
            stage_plan_from_config(est.to_record_fragment()["stage_plan"]) == declared
        )

    def test_null_segments_in_a_record_mean_inherit(self) -> None:
        # to_record_fragment writes `"segments": null` explicitly where the
        # config writer omits the key; both must read back as "inherit", not
        # as the full span.
        from geo_dataread.stage_plan import stage_plan_from_config

        plan = stage_plan_from_config(
            [{"name": "long", "free": ["secular"], "segments": None}]
        )
        assert plan.stages[0].segments is None


def _donor_record(components: list[list[float]], fitted_at: str = "2026-01-01") -> dict:
    """A minimal multi-component donor record (``to_record`` shape)."""
    return {
        "model": "lineperiodic",
        "record_version": 1,
        "fitted_at": fitted_at,
        "components": [{"params": p} for p in components],
    }


class TestDonorGroupDigest:
    """The digest is the drift check's whole evidence — pin its sensitivity."""

    def test_stable_across_a_json_round_trip(self) -> None:
        # Records store full-repr floats and JSON round-trips them
        # bit-identically; a digest that changed on write -> deploy -> read
        # would cry drift on every batch run.
        import json

        from geo_dataread.stage_plan import donor_group_digest

        rec = _donor_record([[10.0, 2.0, 1.0, -1.0, 0.5, -0.5]])
        rt = json.loads(json.dumps(rec))
        assert donor_group_digest(rec, "secular", donor="OLAC") == donor_group_digest(
            rt, "secular", donor="OLAC"
        )

    def test_a_change_in_any_component_changes_it(self) -> None:
        # Holds are resolved PER COMPONENT, so a donor whose east moved while
        # north stayed put has drifted — a component-0 digest would miss it.
        from geo_dataread.stage_plan import donor_group_digest

        base = [[10.0, 2.0, 0, 0, 0, 0], [5.0, 1.0, 0, 0, 0, 0]]
        moved_east = [[10.0, 2.0, 0, 0, 0, 0], [5.0, 1.1, 0, 0, 0, 0]]
        assert donor_group_digest(
            _donor_record(base), "secular", donor="OLAC"
        ) != donor_group_digest(_donor_record(moved_east), "secular", donor="OLAC")

    def test_other_groups_do_not_move_it(self) -> None:
        from geo_dataread.stage_plan import donor_group_digest

        a = _donor_record([[10.0, 2.0, 1.0, -1.0, 0.5, -0.5]])
        b = _donor_record([[10.0, 2.0, 9.0, -9.0, 9.0, -9.0]])
        assert donor_group_digest(a, "secular", donor="X") == donor_group_digest(
            b, "secular", donor="X"
        )
        assert donor_group_digest(a, "periodic", donor="X") != donor_group_digest(
            b, "periodic", donor="X"
        )

    def test_an_empty_record_is_refused(self) -> None:
        # A digest of nothing equals another digest of nothing and would hide
        # the malformation it should surface.
        from geo_dataread.stage_plan import donor_group_digest

        with pytest.raises(ValueError, match="no components"):
            donor_group_digest(
                {"model": "lineperiodic", "components": []}, "secular", donor="X"
            )


class TestAnnotateDonorGroups:
    def test_donor_entries_gain_vintage_and_digest(self) -> None:
        from geo_dataread.stage_plan import annotate_donor_groups, donor_group_digest

        donor = _donor_record([[10.0, 2.0, 0, 0, 0, 0]], fitted_at="2026-07-01")
        groups = {
            "secular": {"indices": [0, 1], "provenance": "donor:OLAC@2026-07-01"},
            "periodic": {"indices": [2, 3, 4, 5], "provenance": "self"},
        }
        out = annotate_donor_groups(groups, lookup_donor=lambda s: donor)
        assert out["secular"]["donor"] == "OLAC"
        assert out["secular"]["donor_fitted_at"] == "2026-07-01"
        assert out["secular"]["donor_record_version"] == 1
        assert out["secular"]["donor_digest"] == donor_group_digest(
            donor, "secular", donor="OLAC"
        )
        # a self group gains nothing and the input is not mutated
        assert out["periodic"] == {"indices": [2, 3, 4, 5], "provenance": "self"}
        assert "donor" not in groups["secular"]


class TestDonorDriftWarnings:
    """The pointer's only witness: a pure comparison of two records."""

    @staticmethod
    def _rec(digest: str | None, *, donor: str = "OLAC", vintage: str = "v1") -> dict:
        entry: dict = {"indices": [0, 1], "provenance": f"donor:{donor}@{vintage}"}
        if digest is not None:
            entry.update(donor=donor, donor_fitted_at=vintage, donor_digest=digest)
        return {"groups": {"secular": entry}}

    def test_a_drifted_donor_is_named_with_both_digests(self) -> None:
        from geo_dataread.stage_plan import donor_drift_warnings

        msgs = donor_drift_warnings(
            self._rec("aaa111"), self._rec("bbb222", vintage="v2"), station="JONC"
        )
        assert len(msgs) == 1
        for token in ("JONC", "OLAC", "aaa111", "bbb222", "'secular'"):
            assert token in msgs[0]

    def test_an_unchanged_donor_is_silent(self) -> None:
        from geo_dataread.stage_plan import donor_drift_warnings

        assert (
            donor_drift_warnings(self._rec("aaa111"), self._rec("aaa111"), station="J")
            == []
        )

    def test_a_changed_donor_station_is_a_plan_edit_not_drift(self) -> None:
        from geo_dataread.stage_plan import donor_drift_warnings

        msgs = donor_drift_warnings(
            self._rec("aaa111", donor="OLAC"),
            self._rec("bbb222", donor="VMEY"),
            station="J",
        )
        assert msgs == []

    def test_nothing_recorded_means_nothing_drifted(self) -> None:
        # First commit (no previous record), a legacy previous without a
        # groups block, and a previous whose group was fitted self — none of
        # these has a recorded borrow to have drifted FROM.
        from geo_dataread.stage_plan import donor_drift_warnings

        cur = self._rec("bbb222")
        assert donor_drift_warnings(None, cur, station="J") == []
        assert donor_drift_warnings({"model": "lineperiodic"}, cur, station="J") == []
        assert (
            donor_drift_warnings(
                {"groups": {"secular": {"provenance": "self"}}}, cur, station="J"
            )
            == []
        )


class TestBorrowFromADonorThatHasSteps:
    """Donor borrowing must survive a donor record carrying a declared step.

    Measured 2026-08-23: `donor_group_values` built its mask from the MODEL
    (6 parameters for lineperiodic) and compared it to the RECORD (7, because
    `to_record` appends step_amp_1). Every station in steps.csv was therefore
    unborrowable -- and the workflow that needs it most is holding a
    station's OWN saved background while estimating only the events, where
    the donor is the station itself and very often has a step.
    """

    RECORD = {
        "model": "lineperiodic",
        "param_names": [
            "offset",
            "rate",
            "cos_annual",
            "sin_annual",
            "cos_semiannual",
            "sin_semiannual",
            "step_amp_1",
        ],
        "components": [
            {"params": [10.0, 20.0, 1.0, 2.0, 3.0, 4.0, -150.0]},
            {"params": [11.0, 21.0, 5.0, 6.0, 7.0, 8.0, 140.0]},
        ],
    }

    def test_periodic_borrows_the_seasonal_only(self) -> None:
        from geo_dataread.stage_plan import donor_group_values

        got = donor_group_values(self.RECORD, "periodic", component=0, donor="SELF")
        assert list(got) == [1.0, 2.0, 3.0, 4.0]

    def test_secular_borrows_offset_and_rate_not_the_step(self) -> None:
        from geo_dataread.stage_plan import donor_group_values

        got = donor_group_values(self.RECORD, "secular", component=1, donor="SELF")
        assert list(got) == [11.0, 21.0], "the appended step leaked into secular"

    def test_the_step_itself_is_borrowable(self) -> None:
        from geo_dataread.stage_plan import donor_group_values

        got = donor_group_values(self.RECORD, "step", component=0, donor="SELF")
        assert list(got) == [-150.0]

    def test_a_stepless_donor_still_works(self) -> None:
        """The case that never broke — keep it covered."""
        from geo_dataread.stage_plan import donor_group_values

        rec = {
            "model": "lineperiodic",
            "param_names": self.RECORD["param_names"][:6],
            "components": [{"params": [10.0, 20.0, 1.0, 2.0, 3.0, 4.0]}],
        }
        got = donor_group_values(rec, "periodic", component=0, donor="DYNG")
        assert list(got) == [1.0, 2.0, 3.0, 4.0]


class TestStoreReAnchoring:
    """Cross-station ``store:`` borrows are re-anchored at resolution time.

    The donor's ``offset`` is the intercept at t = 0 in ABSOLUTE fractional
    years (t ≈ 2×10³), so holding it verbatim pins the borrower's LEVEL to
    the donor's — measured at ~30–40 mm of pure datum error on E/U
    (SENG holding SKSH's s(t); the real-data pin lives in
    ``TestReAnchoringRealData``). Only the datum entry is replaced, by NAME
    (never position — ``params[1] == "rate"`` is load-bearing), and
    ``store:self`` stays byte-identical.
    """

    OFFSET, RATE = 1000.0, 2.0
    SEASONAL = (3.0, -2.0, 1.0, -0.5)
    LEVEL = 5.0  # the borrower's true local level

    @staticmethod
    def _entry(offset: float, rate: float, seasonal: tuple, fitted_at="2026-08-01"):
        from geo_dataread.secular_store import SecularEntry

        vec = (offset, rate, *seasonal)
        return SecularEntry(
            model="lineperiodic",
            param_names=(
                "offset",
                "rate",
                "cos_annual",
                "sin_annual",
                "cos_semiannual",
                "sin_semiannual",
            ),
            components={"north": vec, "east": vec, "up": vec},
            fitted_at=fitted_at,
        )

    def _lookup(self, borrower_entry=None):
        entries = {
            "SKSH": self._entry(self.OFFSET, self.RATE, self.SEASONAL),
            None: borrower_entry or self._entry(self.LEVEL, self.RATE, self.SEASONAL),
        }
        return lambda who: entries[who]

    def _series(self, n: int = 731, start: float = 2021.0):
        """Noise-free borrower series with the DONOR's rate+seasonal but its
        own level, over an integer number of years (seasonal mean ~ 0)."""
        import numpy as np

        t = start + np.arange(n) / 365.25
        a, b, c, d = self.SEASONAL
        y = (
            self.LEVEL
            + self.RATE * t
            + a * np.cos(2 * np.pi * t)
            + b * np.sin(2 * np.pi * t)
            + c * np.cos(4 * np.pi * t)
            + d * np.sin(4 * np.pi * t)
        )
        sigma = np.ones_like(t)
        return t, y, sigma

    def _boom(self, sta: str) -> dict:
        raise AssertionError(f"should not look up donor record {sta}")

    def test_store_self_is_byte_identical(self) -> None:
        """Acceptance 1: no re-anchoring on self-holds — even with a series
        and a window supplied, the resolved values and provenance must be
        exactly what a pre-change resolution produced."""
        import numpy as np

        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(
            ["fit:step"], ["secular=store:self", "periodic=store:self"]
        )
        t, y, sigma = self._series()
        own = self._entry(self.LEVEL, self.RATE, self.SEASONAL)
        out = resolve_stage_plan(
            plan,
            lookup_donor=self._boom,
            component=0,
            lookup_secular=self._lookup(own),
            station="SENG",
            anchor_t=t,
            anchor_y=y,
            anchor_sigma=sigma,
            anchor_window=(2021.0, 2021.5),
        )
        held = out[0].held
        np.testing.assert_array_equal(held["secular"].values, [self.LEVEL, self.RATE])
        np.testing.assert_array_equal(held["periodic"].values, self.SEASONAL)
        assert held["secular"].source == "store:SENG@2026-08-01"
        assert "anchored" not in held["secular"].source

    def test_cross_station_offset_is_reanchored(self) -> None:
        import numpy as np

        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(
            ["apply:"], ["secular=store:SKSH", "periodic=store:SKSH"]
        )
        t, y, sigma = self._series()
        out = resolve_stage_plan(
            plan,
            lookup_donor=self._boom,
            component=0,
            lookup_secular=self._lookup(),
            station="SENG",
            anchor_t=t,
            anchor_y=y,
            anchor_sigma=sigma,
        )
        held = out[0].held
        # offset re-anchored to the borrower's level; rate + every periodic
        # coefficient stay the donor's VERBATIM
        assert np.isclose(held["secular"].values[0], self.LEVEL, atol=1e-9)
        assert held["secular"].values[1] == self.RATE
        np.testing.assert_array_equal(held["periodic"].values, self.SEASONAL)
        # provenance records the anchor and the window actually used (the
        # full span here), and stays distinguishable from `donor:`
        src = held["secular"].source
        assert src.startswith("store:SKSH@2026-08-01 anchored [")
        assert held["periodic"].source == "store:SKSH@2026-08-01"

    def test_own_periodic_is_not_subtracted(self) -> None:
        """Holding only the donor SECULAR while estimating the seasonal
        locally: no periodic is subtracted, and the anchor is still well
        posed because the seasonal carries no DC term (the window spans
        integer years, so its mean is ~0)."""
        import numpy as np

        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:periodic"], ["secular=store:SKSH"])
        t, y, sigma = self._series()
        out = resolve_stage_plan(
            plan,
            lookup_donor=self._boom,
            component=0,
            lookup_secular=self._lookup(),
            station="SENG",
            anchor_t=t,
            anchor_y=y,
            anchor_sigma=sigma,
        )
        assert np.isclose(out[0].held["secular"].values[0], self.LEVEL, atol=0.05)

    def test_periodic_only_borrow_needs_no_series(self) -> None:
        """The seasonal group has no datum parameter, so a cross-station
        periodic borrow resolves without the borrower's series."""
        import numpy as np

        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:secular"], ["periodic=store:SKSH"])
        out = resolve_stage_plan(
            plan,
            lookup_donor=self._boom,
            component=0,
            lookup_secular=self._lookup(),
            station="SENG",
        )
        np.testing.assert_array_equal(out[0].held["periodic"].values, self.SEASONAL)
        assert "anchored" not in out[0].held["periodic"].source

    def test_missing_series_is_refused_actionably(self) -> None:
        """Acceptance 4: a cross-station secular borrow with no series must
        RAISE, never silently hold the donor's offset — that fallback IS the
        measured 30–40 mm datum error."""
        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:periodic"], ["secular=store:SKSH"])
        with pytest.raises(ValueError, match="anchor_t/anchor_y"):
            resolve_stage_plan(
                plan,
                lookup_donor=self._boom,
                component=0,
                lookup_secular=self._lookup(),
                station="SENG",
            )

    def test_degenerate_window_is_refused(self) -> None:
        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:periodic"], ["secular=store:SKSH"])
        t, y, sigma = self._series()
        with pytest.raises(ValueError, match="empty or degenerate"):
            resolve_stage_plan(
                plan,
                lookup_donor=self._boom,
                component=0,
                lookup_secular=self._lookup(),
                station="SENG",
                anchor_t=t,
                anchor_y=y,
                anchor_window=(2022.0, 2022.0),
            )

    def test_zero_epoch_window_is_refused_naming_the_station(self) -> None:
        """Acceptance 5."""
        from geo_dataread.stage_plan import resolve_stage_plan

        plan = build_stage_plan(["fit:periodic"], ["secular=store:SKSH"])
        t, y, sigma = self._series()
        with pytest.raises(ValueError, match="SENG.*selects zero"):
            resolve_stage_plan(
                plan,
                lookup_donor=self._boom,
                component=0,
                lookup_secular=self._lookup(),
                station="SENG",
                anchor_t=t,
                anchor_y=y,
                anchor_window=(1990.0, 1991.0),
            )

    def test_store_ref_config_round_trip(self) -> None:
        """`store:` holds must round-trip as store holds. `_spell` used to
        render every non-stage ref `donor:...`, silently flipping a stored
        plan's hold KIND (`store:self` became `donor:None`)."""
        plan = build_stage_plan(
            ["fit:step"], ["secular=store:self", "periodic=store:SKSH"]
        )
        cfg = stage_plan_to_config(plan)
        assert cfg[0]["held"] == {
            "secular": "store:self",
            "periodic": "store:SKSH",
        }
        assert stage_plan_from_config(cfg) == plan

    def test_apply_only_plan_round_trips(self) -> None:
        plan = build_stage_plan(
            ["apply:"], ["secular=store:SKSH", "periodic=store:SKSH"]
        )
        assert plan.stages[0].free == ()
        assert stage_plan_from_config(stage_plan_to_config(plan)) == plan

    def test_apply_only_stage_without_holds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no free term group and no --hold"):
            build_stage_plan(["apply:"], [])


# ---------------------------------------------------------------------------
# real deployed data + store (skipped when either is absent, following the
# gps_api tests/test_detrend_params.py realdata precedent)
# ---------------------------------------------------------------------------


def _deployed_secular_store(*stations: str):
    """The deployed analysis.yaml secular store, or a clean skip."""
    from geo_dataread.secular_store import read_secular
    from geo_dataread.stage_plan import default_analysis_yaml_path

    path = default_analysis_yaml_path()
    if path is None or not path.is_file():
        pytest.skip("no deployed analysis.yaml (no gpsconfig on this host)")
    entries = read_secular(path)
    missing = [sta for sta in stations if sta not in entries]
    if missing:
        pytest.skip(f"deployed secular store has no entry for {missing}")
    return path


def _real_series(sta: str):
    """Real deployed plate-removed TOT series, or a clean skip."""
    import numpy as np

    from geo_dataread import gps_read

    try:
        yearf, data, sigma, _off = gps_read.getData(
            sta, ref="plate", tType="TOT", uncert=15
        )
    except Exception as exc:  # noqa: BLE001 - any read failure means "skip"
        pytest.skip(f"deployed TOT data for {sta} not reachable: {exc}")
    if yearf is None or len(yearf) == 0:
        pytest.skip(f"deployed TOT data for {sta} is empty")
    return (
        np.asarray(yearf, dtype=float),
        np.asarray(data, dtype=float),
        np.asarray(sigma, dtype=float),
    )


@pytest.fixture
def _deployed_config_reachable(monkeypatch):
    """Undo the golden-master suite's session-wide ``GPS_CONFIG_PATH`` pin.

    ``tests/goldenmaster/conftest.py`` points ``GPS_CONFIG_PATH`` at a
    synthetic config dir with SESSION scope, precisely so no golden-master
    test can touch the operator's real one.  Its teardown runs at the end of
    the run, so for every test collected after it the deployed config is
    unreachable — and these tests, whose whole point is the REAL deployed
    store, degraded to a silent skip in the full-suite run while passing when
    run alone.  A guard that only fires under `-k` is not a guard.

    Clearing the variable restores the same resolution the CLI uses
    (``~/.config/gpsconfig``); monkeypatch puts it back per test, so the
    isolation the golden masters rely on is untouched.  A host with no
    gpsconfig at all still resolves to None and still skips, which is the
    case the guard is actually for.
    """
    monkeypatch.delenv("GPS_CONFIG_PATH", raising=False)


@pytest.mark.usefixtures("_deployed_config_reachable")
class TestReAnchoringRealData:
    """Acceptance gates on the real deployed store + TOT data.

    SENG is the test vehicle because it HAS pre-unrest data, so real
    deformation cannot confound the datum measurement; the operational
    borrowers (ELDC, THOB) have data only after the Dec-2019 onset.
    """

    WINDOW = (2015.5, 2019.9)

    def _resolve_seng_holding_sksh(self, component: int, anchored: bool):
        import numpy as np

        from geo_dataread.detrend_estimate import secular_lookup
        from geo_dataread.stage_plan import build_stage_plan, resolve_stage_plan
        from gps_analysis.staged import evaluate_group_values

        yaml_path = _deployed_secular_store("SENG", "SKSH")
        yearf, data, sigma = _real_series("SENG")
        lk = secular_lookup(yaml_path, "SENG")
        donor = "SKSH" if anchored else "self"
        plan = build_stage_plan(
            ["apply:"],
            [f"secular=store:{donor}", f"periodic=store:{donor}"],
        )
        w = (yearf >= self.WINDOW[0]) & (yearf <= self.WINDOW[1])
        stages = resolve_stage_plan(
            plan,
            lookup_donor=lambda s: pytest.fail(f"donor record lookup {s}"),
            component=component,
            lookup_secular=lk,
            station="SENG",
            anchor_t=yearf[w],
            anchor_y=data[component][w],
            anchor_sigma=sigma[component][w],
            anchor_window=self.WINDOW,
        )
        held = stages[0].held
        model = evaluate_group_values(
            ("offset", "rate"), held["secular"].values, yearf[w]
        ) + evaluate_group_values(
            ("cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual"),
            held["periodic"].values,
            yearf[w],
        )

        return float(np.mean(data[component][w] - model)), held

    def test_reanchor_removes_the_datum_error(self) -> None:
        """Acceptance 2: SENG holding SKSH's s(t) over [2015.5, 2019.9].

        Verbatim borrow (pre-change behavior): mean residual
        (−2.94, −30.06, +41.45) mm N/E/U — pure datum error, the donor's
        offset being the donor's level. Re-anchored: ≈ 0 mm on all three.

        The assertion is on the MEAN, deliberately not the RMS: the
        remaining scatter is real physics — SKSH's east rate differs from
        SENG's by 5.785 mm/yr, so the residual against the donor's rate
        tilts through the window — and an RMS assertion would be testing
        that geophysics away, not the datum fix. The mean tolerance of
        1 mm absorbs the (weighted anchor vs unweighted mean) difference
        plus that tilt's asymmetric sampling; the pre-change error it
        guards against is 30–40× larger.
        """

        expected_raw = (-2.94, -30.06, 41.45)  # measured 2026-08-26
        for component in range(3):
            mean_after, held = self._resolve_seng_holding_sksh(component, anchored=True)
            assert abs(mean_after) < 1.0, (
                f"component {component}: re-anchored mean residual "
                f"{mean_after:.2f} mm (verbatim borrow was "
                f"{expected_raw[component]} mm)"
            )
            assert "anchored [2015.5,2019.9]" in held["secular"].source

    def test_verbatim_borrow_error_is_as_measured(self) -> None:
        """Pin the DISEASE too: rebuild the un-anchored borrow by hand (the
        store's raw values) and confirm the documented datum error, so a
        future store re-estimate that invalidates the numbers in these
        docstrings fails HERE and not silently."""
        import numpy as np

        from geo_dataread.detrend_estimate import secular_lookup
        from geo_dataread.secular_store import secular_group_values
        from gps_analysis.staged import evaluate_group_values

        yaml_path = _deployed_secular_store("SENG", "SKSH")
        yearf, data, sigma = _real_series("SENG")
        lk = secular_lookup(yaml_path, "SENG")
        entry = lk("SKSH")
        w = (yearf >= self.WINDOW[0]) & (yearf <= self.WINDOW[1])
        measured = (-2.94, -30.06, 41.45)
        for component in range(3):
            sec = secular_group_values(
                entry, "secular", component=component, station="SKSH"
            )
            per = secular_group_values(
                entry, "periodic", component=component, station="SKSH"
            )
            model = evaluate_group_values(
                ("offset", "rate"), sec, yearf[w]
            ) + evaluate_group_values(
                ("cos_annual", "sin_annual", "cos_semiannual", "sin_semiannual"),
                per,
                yearf[w],
            )
            mean = float(np.mean(data[component][w] - model))
            assert mean == pytest.approx(measured[component], abs=0.2)

    def test_eldc_all_held_apply_path(self) -> None:
        """Acceptance 3: ELDC (window from 2021.0) holding BOTH secular and
        periodic from SENG — the scientifically correct configuration for a
        station with no pre-unrest data — produces a record with no fitted
        parameters, borrowed provenance and a locally anchored offset, and
        does NOT raise (it was unrepresentable before the apply path)."""
        import dataclasses

        from geo_dataread.detrend_estimate import (
            FitDefaults,
            resolve_fit_settings,
            secular_lookup,
        )
        from geo_dataread.detrend_estimate import (
            station_estimate_from_arrays,
        )
        from geo_dataread.stage_plan import build_stage_plan

        yaml_path = _deployed_secular_store("SENG")
        yearf, data, sigma = _real_series("ELDC")
        plan = build_stage_plan(
            ["apply:"], ["secular=store:SENG", "periodic=store:SENG"]
        )
        settings = dataclasses.replace(
            resolve_fit_settings("ELDC", None, FitDefaults()),
            segments=((2021.0, None),),
            window_source="test",
        )
        result = station_estimate_from_arrays(
            "ELDC",
            yearf,
            data,
            sigma,
            settings=settings,
            stage_plan=plan,
            lookup_donor=lambda s: pytest.fail(f"donor record lookup {s}"),
            lookup_secular=secular_lookup(yaml_path, "ELDC"),
        )
        assert result is not None
        record = result.record
        # nothing fitted: every stage is apply-only and says so
        assert [s["free"] for s in record["stages"]] == [[]]
        assert record["stages"][0]["held_covariance"] == "applied"
        # record-level borrowed marker, in the _borrowed_record shape
        assert record["borrowed"]["from"] == "SENG"
        assert record["borrowed"]["terms"] == "all"
        # locally anchored offset: provenance says so, and the held level is
        # ELDC's, not SENG's (their offsets differ by far more than 1 mm)
        prov = record["groups"]["secular"]["provenance"]
        assert prov.startswith("store:SENG@") and "anchored [" in prov
        own_lookup = secular_lookup(yaml_path, "ELDC")
        seng_offset = float(own_lookup("SENG").components["north"][0])
        eldc_offset = float(record["components"][0]["params"][0])
        assert abs(eldc_offset - seng_offset) > 1.0
        # ...and the method tag agrees with the rest of the record. The
        # outlier stage DID run (it screened epochs for the figure), so the
        # estimator's own tag is "step_augmented_robust" -- true of the
        # screening, false of the parameters. `detrend_method` is the field a
        # reader consults to learn how the parameters were made, and
        # gps_views passes it through to the served provenance, so a
        # fully-borrowed record must not claim a fit it never had.
        assert record["detrend_method"] == "borrowed"

    def test_partial_borrow_keeps_its_fitted_method_tag(self) -> None:
        """The contrast that gives the "borrowed" tag its meaning.

        A station that holds ONE group from a donor and estimates the rest
        did have a fit, so its ``detrend_method`` stays the estimator's own
        and ``borrowed`` (a record-level "nothing was estimated here" flag)
        stays absent.  Which group came from where is per-GROUP provenance,
        and lives in the ``groups`` block.  Tagging this record "borrowed"
        would erase a real fit; tagging the all-held one anything else would
        invent one.
        """
        import dataclasses

        from geo_dataread.detrend_estimate import (
            FitDefaults,
            resolve_fit_settings,
            secular_lookup,
            station_estimate_from_arrays,
        )
        from geo_dataread.stage_plan import build_stage_plan

        yaml_path = _deployed_secular_store("SENG")
        yearf, data, sigma = _real_series("SKSH")
        plan = build_stage_plan(["fit:periodic"], ["secular=store:SENG"])
        settings = dataclasses.replace(
            resolve_fit_settings("SKSH", None, FitDefaults()),
            segments=((2013.9, 2019.9),),
            max_gap_years=1.2,
            window_source="test",
        )
        result = station_estimate_from_arrays(
            "SKSH",
            yearf,
            data,
            sigma,
            settings=settings,
            stage_plan=plan,
            lookup_donor=lambda s: pytest.fail(f"donor record lookup {s}"),
            lookup_secular=secular_lookup(yaml_path, "SKSH"),
        )
        assert result is not None
        record = result.record
        assert record["detrend_method"] != "borrowed"
        assert record.get("borrowed") is None
        # the borrow is still recorded -- per group, where it belongs
        assert record["groups"]["secular"]["provenance"].startswith("store:SENG@")


@pytest.mark.usefixtures("_deployed_config_reachable")
class TestDonorHoldsAreAnchoredToo:
    """The datum problem is the HOLD KIND's, not the store's.

    `store:` reads a saved background, `donor:` a finished record -- but the
    offset in either belongs to the station it was fitted on, so a
    cross-station borrow through EITHER carries the donor's level. Measured
    on THOB holding SENG via `donor:` before this was fixed: the borrowed
    curve sat +75.3 / -34.3 / -124.6 mm N/E/U off THOB's own data, and
    --anchor-window was ignored on that path entirely.
    """

    WINDOW = (2021.0, 2021.5)

    def _mismatch(self, holds):
        import json

        import numpy as np

        from geo_dataread.detrend_estimate import secular_lookup
        from geo_dataread.stage_plan import build_stage_plan, resolve_stage_plan

        yaml_path = _deployed_secular_store("SENG")
        params_path = yaml_path.parent / "detrend_params.json"
        if not params_path.is_file():
            pytest.skip("no deployed detrend_params.json")
        params = json.loads(params_path.read_text()).get("stations", {})
        if "SENG" not in params:
            pytest.skip("deployed detrend_params.json has no SENG record")
        yearf, data, sigma = _real_series("THOB")
        sel = (yearf >= self.WINDOW[0]) & (yearf <= self.WINDOW[1])
        if not sel.any():
            pytest.skip("THOB has no epochs in the anchor window")
        plan = build_stage_plan(["apply:"], holds)
        lookup = secular_lookup(yaml_path, "THOB")
        out = []
        for i in range(3):
            stages = resolve_stage_plan(
                plan,
                lookup_donor=lambda s: params[s],
                component=i,
                lookup_secular=lookup,
                station="THOB",
                anchor_t=yearf[sel],
                anchor_y=data[i][sel],
                anchor_sigma=sigma[i][sel],
                anchor_window=self.WINDOW,
            )
            held = stages[0].held["secular"]
            level = held.values[0] + held.values[1] * float(np.mean(yearf[sel]))
            out.append((level - float(np.mean(data[i][sel])), held.source))
        return out

    def test_a_donor_hold_lands_on_this_station_s_level(self) -> None:
        for mismatch, source in self._mismatch(
            ["secular=donor:SENG", "periodic=donor:SENG"]
        ):
            assert abs(mismatch) < 5.0, f"donor borrow off by {mismatch:.1f} mm"
            assert source.startswith("donor:SENG@")
            assert "anchored [" in source

    def test_both_hold_kinds_agree_after_anchoring(self) -> None:
        """Same donor station, same window, two routes to its coefficients:
        the level they land on must not depend on which store was read."""
        donor = self._mismatch(["secular=donor:SENG", "periodic=donor:SENG"])
        store = self._mismatch(["secular=store:SENG", "periodic=store:SENG"])
        for (dm, _), (sm, _) in zip(donor, store, strict=True):
            assert abs(dm - sm) < 1.0

    def test_the_provenance_still_names_which_store_was_read(self) -> None:
        """Anchoring must not blur the kinds together: they resolve against
        different objects and a reader has to be able to tell which."""
        donor = self._mismatch(["secular=donor:SENG", "periodic=donor:SENG"])
        store = self._mismatch(["secular=store:SENG", "periodic=store:SENG"])
        assert all(s.startswith("donor:") for _, s in donor)
        assert all(s.startswith("store:") for _, s in store)
