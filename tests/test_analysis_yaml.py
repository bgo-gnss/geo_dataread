"""Tests for geo_dataread.analysis_yaml — per-station model/terms overrides.

The block closes the gap that wiring the stage plans opened: the batch has
ONE global ``--model`` and no ``--term``, so a station curated under either
could not re-estimate as curated. Since e6dd887 that was a loud per-station
error rather than a silent revert — better, but the station still did not
re-estimate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from geo_dataread.analysis_yaml import (
    STATION_MODEL_KEYS,
    StationModel,
    read_station_models,
    station_model_to_config,
    write_station_model,
)


def _write(tmp_path: Path, models: Any) -> Path:
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump({"detrend": {"estimation": {"models": models}}}))
    return path


class TestReadStationModels:
    def test_absent_file_block_and_empty_all_mean_nothing_curated(
        self, tmp_path: Path
    ) -> None:
        assert read_station_models(tmp_path / "nope.yaml") == {}
        empty = tmp_path / "empty.yaml"
        empty.write_text("{}\n")
        assert read_station_models(empty) == {}
        assert read_station_models(_write(tmp_path, {})) == {}

    def test_model_only_terms_only_and_both(self, tmp_path: Path) -> None:
        got = read_station_models(
            _write(
                tmp_path,
                {
                    "SELF": {"model": "periodic"},
                    "RHOF": {"terms": ["log@2008.4085,tau=1.0"]},
                    "NYLA": {
                        "model": "linear",
                        "terms": ["exp@2023.86,tau=0.33"],
                    },
                },
            )
        )
        assert got["SELF"] == StationModel(model="periodic")
        assert got["RHOF"] == StationModel(terms=("log@2008.4085,tau=1.0",))
        assert got["NYLA"] == StationModel(
            model="linear", terms=("exp@2023.86,tau=0.33",)
        )

    def test_an_empty_entry_is_refused(self, tmp_path: Path) -> None:
        """Curation that is not there must not look like curation."""
        with pytest.raises(ValueError, match="must set 'model' or 'terms'"):
            read_station_models(_write(tmp_path, {"SELF": {}}))

    @pytest.mark.parametrize(
        ("entry", "match"),
        [
            (["periodic"], "must be a mapping"),
            ({"modle": "periodic"}, "unknown key"),
            ({"model": 7}, "must be a string"),
            ({"terms": "log@2008.4,tau=1.0"}, "must be a list"),
            ({"terms": ["log@2008.4"]}, "term 'log@2008.4'"),
            ({"terms": ["log@2008.4,tau=nope"]}, "term "),
        ],
    )
    def test_malformed_entries_name_the_station(
        self, tmp_path: Path, entry: Any, match: str
    ) -> None:
        """A typo'd tau is a CONFIG error, not a failure five stations in.

        ``parse_term_spec`` runs at read time for exactly that reason.
        """
        with pytest.raises(ValueError, match=match) as exc:
            read_station_models(_write(tmp_path, {"SELF": entry}))
        assert "SELF" in str(exc.value)

    def test_a_non_mapping_block_raises(self, tmp_path: Path) -> None:
        assert STATION_MODEL_KEYS == ("detrend", "estimation", "models")
        with pytest.raises(ValueError, match="must be a mapping of station"):
            read_station_models(_write(tmp_path, ["SELF"]))


class TestWriteStationModel:
    def test_merge_preserves_other_stations_and_other_blocks(
        self, tmp_path: Path
    ) -> None:
        """One station is curated at a time; the rest must survive it."""
        path = tmp_path / "analysis.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "detrend": {
                        "estimation": {
                            "models": {"KASC": {"model": "linear"}},
                            "stage_plans": {
                                "OLAC": [{"name": "a", "free": ["secular"]}]
                            },
                        }
                    },
                    "unrelated": {"keep": "me"},
                }
            )
        )
        write_station_model(path, "SELF", StationModel(model="periodic"))
        doc = yaml.safe_load(path.read_text())
        assert set(doc["detrend"]["estimation"]["models"]) == {"KASC", "SELF"}
        assert doc["detrend"]["estimation"]["stage_plans"] == {
            "OLAC": [{"name": "a", "free": ["secular"]}]
        }
        assert doc["unrelated"] == {"keep": "me"}

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "analysis.yaml"
        entry = StationModel(model="periodic", terms=("log@2008.4085,tau=1.0",))
        write_station_model(path, "SELF", entry)
        assert read_station_models(path) == {"SELF": entry}
        assert station_model_to_config(entry) == {
            "model": "periodic",
            "terms": ["log@2008.4085,tau=1.0"],
        }

    def test_none_removes_the_entry_and_the_empty_block(self, tmp_path: Path) -> None:
        """How a station returns to the global default without hand-editing."""
        path = tmp_path / "analysis.yaml"
        write_station_model(path, "SELF", StationModel(model="periodic"))
        write_station_model(path, "SELF", None)
        assert read_station_models(path) == {}
        assert "models" not in yaml.safe_load(path.read_text())["detrend"]["estimation"]

    def test_omits_what_was_not_set(self, tmp_path: Path) -> None:
        path = tmp_path / "analysis.yaml"
        write_station_model(path, "SELF", StationModel(terms=("log@2008.4,tau=1.0",)))
        stored = yaml.safe_load(path.read_text())["detrend"]["estimation"]["models"]
        assert stored["SELF"] == {"terms": ["log@2008.4,tau=1.0"]}
