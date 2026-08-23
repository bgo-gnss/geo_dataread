"""The secular store — s(t) saved as a reusable component."""

from __future__ import annotations

import pytest

from geo_dataread.secular_store import (
    SecularEntry,
    read_secular,
    secular_from_record,
    secular_group_values,
    write_secular,
)

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
    "segments": [[2001.5, 2008.4], [2009.5, 2020.83]],
    "components": [
        {"params": [-5813.3, 2.90, -0.167, 0.078, 0.185, -0.097, -150.8]},
        {"params": [12733.2, -6.36, 0.047, 0.038, -0.181, 0.162, 149.8]},
        {"params": [-3205.1, 1.60, 3.184, -1.400, 0.778, 1.744, 54.6]},
    ],
}


class TestSlicingTheBackgroundOutOfARecord:
    def test_it_takes_the_background_and_leaves_the_events(self) -> None:
        """s(t) is lin+per by definition; a step is an EVENT, never part of it."""
        entry = secular_from_record(RECORD)
        assert "step_amp_1" not in entry.param_names
        assert entry.param_names == (
            "offset",
            "rate",
            "cos_annual",
            "sin_annual",
            "cos_semiannual",
            "sin_semiannual",
        )
        assert len(entry.components["north"]) == 6
        assert entry.components["north"][1] == pytest.approx(2.90)

    def test_the_offset_is_kept(self) -> None:
        """Unlike the legacy CSV, and for a measured reason.

        Detrending does not need an offset -- subtracting a constant changes
        nothing that is looked at. But HOLDING s(t) while estimating a step
        does: with the level unanchored the step is measured against a
        floating background. On SELF a background held without a correctly
        anchored offset produced a step of 0.0 mm against a true -150.8.
        """
        entry = secular_from_record(RECORD)
        assert entry.param_names[0] == "offset"
        assert entry.components["east"][0] == pytest.approx(12733.2)

    def test_the_segments_come_with_it(self) -> None:
        # A union, which the CSV's single Starttime/Endtime cannot spell --
        # and a union is exactly what a background fitted either side of an
        # event needs.
        entry = secular_from_record(RECORD)
        assert entry.segments == ((2001.5, 2008.4), (2009.5, 2020.83))

    def test_a_model_with_no_background_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no background to save"):
            secular_from_record(
                {
                    "model": "lineperiodic",
                    "param_names": ["step_amp_1"],
                    "components": [{"params": [1.0]}],
                }
            )


class TestRoundTrip:
    def test_it_survives_the_yaml(self, tmp_path) -> None:
        path = tmp_path / "analysis.yaml"
        path.write_text("detrend:\n  estimation:\n    enabled: true\n")
        entry = secular_from_record(RECORD, fitted_at="2026-08-23")
        write_secular(path, "SELF", entry)

        back = read_secular(path)["SELF"]
        assert back.model == entry.model
        assert back.param_names == entry.param_names
        assert back.segments == entry.segments
        assert back.fitted_at == "2026-08-23"
        for comp in ("north", "east", "up"):
            assert back.components[comp] == pytest.approx(entry.components[comp])

    def test_writing_one_station_leaves_the_others(self, tmp_path) -> None:
        path = tmp_path / "analysis.yaml"
        path.write_text("detrend:\n  estimation:\n    enabled: true\n")
        write_secular(path, "SELF", secular_from_record(RECORD))
        write_secular(path, "HOFN", secular_from_record(RECORD))
        assert set(read_secular(path)) == {"SELF", "HOFN"}

        write_secular(path, "HOFN", None)
        assert set(read_secular(path)) == {"SELF"}
        import yaml

        assert yaml.safe_load(path.read_text())["detrend"]["estimation"]["enabled"]

    def test_a_missing_block_is_not_an_error(self, tmp_path) -> None:
        path = tmp_path / "analysis.yaml"
        path.write_text("detrend:\n  estimation:\n    enabled: true\n")
        assert read_secular(path) == {}
        assert read_secular(tmp_path / "nope.yaml") == {}


class TestBorrowingFromTheStore:
    def test_it_slices_one_group(self) -> None:
        entry = secular_from_record(RECORD)
        got = secular_group_values(entry, "periodic", component=0, station="SELF")
        assert list(got) == pytest.approx([-0.167, 0.078, 0.185, -0.097])

    def test_secular_is_offset_and_rate(self) -> None:
        entry = secular_from_record(RECORD)
        got = secular_group_values(entry, "secular", component=1, station="SELF")
        assert list(got) == pytest.approx([12733.2, -6.36])

    def test_it_never_yields_a_step(self) -> None:
        entry = secular_from_record(RECORD)
        with pytest.raises(ValueError, match="stores no 'step'"):
            secular_group_values(entry, "step", component=0, station="SELF")

    def test_a_use_sta_row_says_where_to_look(self) -> None:
        """The CSV's UseSTA, carried over: a borrow row stores no values."""
        entry = SecularEntry(model="", use_sta="DYNG", fit="periodic")
        with pytest.raises(ValueError, match="borrows from 'DYNG'"):
            secular_group_values(entry, "periodic", component=0, station="OLAC")


class TestTheStoreHoldKind:
    def test_store_self_and_store_station_parse(self) -> None:
        from geo_dataread.stage_plan import StoreRef, parse_hold_spec

        assert parse_hold_spec("secular=store:self")[2] == StoreRef(None)
        assert parse_hold_spec("periodic=store:VMEY")[2] == StoreRef("VMEY")

    def test_it_is_a_DIFFERENT_kind_from_donor(self) -> None:
        """They resolve against different objects, so the kind is named.

        `donor:` reads a finished record (s(t) + steps + transients);
        `store:` reads the background store. Inferring between them would
        let a rename change which object a stored hold points at.
        """
        from geo_dataread.stage_plan import DonorRef, StoreRef, parse_hold_spec

        assert parse_hold_spec("periodic=donor:VMEY")[2] == DonorRef("VMEY")
        assert parse_hold_spec("periodic=store:VMEY")[2] == StoreRef("VMEY")
        assert DonorRef("VMEY") != StoreRef("VMEY")

    def test_an_unnamed_kind_is_still_refused(self) -> None:
        from geo_dataread.stage_plan import parse_hold_spec

        with pytest.raises(ValueError, match="must name its kind"):
            parse_hold_spec("periodic=VMEY")
