"""The event catalog (events.yaml) and a station's events."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from geo_dataread import events as ev

CATALOG = """
schema_version: 1
events:
- id: b-dike
  kind: dike
  region: reykjanes
  onset: 2023-11-10
  end: null
- id: a-eruption
  kind: eruption
  region: reykjanes
  onset: 2021-03-19
  end: 2021-09-18
- id: north-swarm
  kind: swarm
  region: tjornes
  onset: 2022-01-01
- id: inflation
  kind: inflation
  region: reykjanes
  onset: 2020-01-21
  end: 2020-02-01
"""

AREAS = {
    "regional_areas": {"reykjanes": {"name": "R", "stations": ["NYLA", "GRVM"]}},
    "volcanic_areas": {"tjornes": {"name": "T", "stations": ["HUSA"]}},
    "category_labels": {"regional": "Regions"},
}


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "events.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_events_are_read_by_onset(tmp_path):
    got = ev.read_events(_write(tmp_path, CATALOG))
    assert [e.id for e in got] == ["inflation", "a-eruption", "north-swarm", "b-dike"]
    erupt = got[1]
    assert erupt.onset == dt.date(2021, 3, 19) and erupt.last_day == dt.date(
        2021, 9, 18
    )
    assert got[3].end is None and got[3].last_day == dt.date(2023, 11, 10)


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ("- {id: x, kind: dike}", "no onset"),
        ("- {id: x, kind: volcano, onset: 2020-01-01}", "kind"),
        ("- {kind: dike, onset: 2020-01-01}", "no id"),
        ("- {id: x, kind: dike, onset: '2020-13-01'}", "ISO date"),
        ("- {id: x, kind: dike, onset: 2020-01-01}\n- {id: x, kind: dike, onset: 2020-01-02}",
         "duplicate"),
    ],
)  # fmt: skip
def test_a_corrupt_catalog_is_refused(tmp_path, row, match):
    with pytest.raises(ValueError, match=match):
        ev.read_events(_write(tmp_path, "events:\n" + row + "\n"))


def test_a_missing_catalog_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.read_events(tmp_path / "nope.yaml")


def test_area_stations_reads_every_area_section():
    areas = ev.area_stations(AREAS)
    assert areas == {
        "reykjanes": frozenset({"NYLA", "GRVM"}),
        "tjornes": frozenset({"HUSA"}),
    }


def test_station_events_by_region_and_kind(tmp_path):
    events = ev.read_events(_write(tmp_path, CATALOG))
    areas = ev.area_stations(AREAS)
    assert [e.id for e in ev.station_events("nyla", events, areas)] == [
        "a-eruption",
        "b-dike",
    ]
    assert ev.station_events("HUSA", events, areas) == []  # a swarm: not a step kind
    got = ev.station_events("HUSA", events, areas, kinds=("swarm",))
    assert [e.id for e in got] == ["north-swarm"]
    assert ev.station_events("HOFN", events, areas) == []  # in no event's region
    with pytest.raises(ValueError, match="unknown event kind"):
        ev.station_events("NYLA", events, areas, kinds=("dyke",))


def test_an_event_without_a_known_region_concerns_no_station():
    e = ev.Event("x", "dike", dt.date(2024, 1, 1), region="atlantis")
    assert ev.station_events("NYLA", [e], ev.area_stations(AREAS)) == []
