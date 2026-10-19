"""The volcano-tectonic event catalog (``events.yaml``) and a station's events.

``events.yaml`` (gps-config-data ``analysis-lane/``, deployed to the
gpsconfig dir; BGÓ 2026-10-17) declares the Reykjanes episodes since 2020:
per event an ``id``, a ``kind`` (inflation, deflation, deep_inflation,
swarm, earthquake, dike, eruption), an ``onset`` date, an ``end`` date
(null = open or unknown) and a ``region`` — a key of ``station_areas.yaml``
(any ``*_areas`` section), which says which stations the event concerns.

Used to split a series into the spans between events (``gpsplot ts
--rate-segments events``): a rate is only meaningful between two dikes, not
across one.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EVENTS_FILENAME = "events.yaml"
STATION_AREAS_KEY = "station_areas_file"
STATION_AREAS_FILENAME = "station_areas.yaml"
KINDS = (
    "inflation",
    "deflation",
    "deep_inflation",
    "swarm",
    "earthquake",
    "dike",
    "eruption",
)
#: the kinds that put a step (or a deformation episode of days) in a series
STEP_KINDS = ("dike", "eruption")


@dataclass(frozen=True)
class Event:
    """One catalogued event; ``end`` None when open or not stated."""

    id: str
    kind: str
    onset: dt.date
    end: dt.date | None = None
    region: str | None = None
    system: str | None = None

    @property
    def last_day(self) -> dt.date:
        """The event's last day: ``end``, else its onset."""
        return (
            self.end if self.end is not None and self.end >= self.onset else self.onset
        )


def _date(value: Any, field: str, eid: str) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"event {eid}: {field} {value!r} is not an ISO date") from None


def parse_events(doc: Mapping[str, Any]) -> list[Event]:
    """The events of a loaded ``events.yaml``, by onset.

    Raises:
        ValueError: A missing id/kind/onset, an unknown kind, a bad date or
            a duplicate id — a corrupt catalog is rejected, never thinned.
    """
    out: list[Event] = []
    seen: set[str] = set()
    for i, row in enumerate(doc.get("events") or ()):
        eid = str(row.get("id") or "")
        if not eid:
            raise ValueError(f"event #{i}: no id")
        if eid in seen:
            raise ValueError(f"event {eid}: duplicate id")
        seen.add(eid)
        kind = str(row.get("kind") or "")
        if kind not in KINDS:
            raise ValueError(f"event {eid}: kind {kind!r} not one of {KINDS}")
        onset = _date(row.get("onset"), "onset", eid)
        if onset is None:
            raise ValueError(f"event {eid}: no onset")
        out.append(
            Event(
                id=eid,
                kind=kind,
                onset=onset,
                end=_date(row.get("end"), "end", eid),
                region=row.get("region"),
                system=row.get("system"),
            )
        )
    return sorted(out, key=lambda e: (e.onset, e.id))


def _gpsconfig_file(key: str | None, default_name: str) -> Path | None:
    try:
        from gps_parser import ConfigParser

        cfg = ConfigParser()
    except Exception:  # noqa: BLE001 - no gpsconfig deployed
        return None
    base = Path(str(getattr(cfg, "config_path", "") or "."))
    if key is not None:
        try:
            value = Path(str(cfg.getPostProcessDir(key))).expanduser()
            return value if value.is_absolute() else base / value
        except Exception:  # noqa: BLE001 - key not deployed: the default name
            pass
    return base / default_name


def default_events_path() -> Path | None:
    """The deployed ``events.yaml`` (``[FILES] events`` in postprocess.cfg,
    else ``<gpsconfig dir>/events.yaml``); None without a gpsconfig."""
    from gps_parser import outlier_catalogs as _oc

    path = _oc.catalog_path("events", EVENTS_FILENAME)
    return None if path is None else Path(path)


def read_events(path: str | Path | None = None) -> list[Event]:
    """Read ``events.yaml`` (None: the deployed one).

    Raises:
        FileNotFoundError: No catalog there.
        ValueError: A corrupt catalog (see :func:`parse_events`).
    """
    resolved = default_events_path() if path is None else Path(path)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"no event catalog at {resolved}")
    doc = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return parse_events(doc)


def area_stations(doc: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """Area key → member stations, over every ``*_areas`` section."""
    out: dict[str, frozenset[str]] = {}
    for section, areas in doc.items():
        if not str(section).endswith("_areas") or not isinstance(areas, Mapping):
            continue
        for key, area in areas.items():
            if isinstance(area, Mapping):
                out[str(key)] = frozenset(str(s) for s in area.get("stations") or ())
    return out


def read_station_areas(path: str | Path | None = None) -> dict[str, frozenset[str]]:
    """The deployed ``station_areas.yaml`` as area key → stations.

    Raises:
        FileNotFoundError: No such file.
    """
    resolved = (
        _gpsconfig_file(STATION_AREAS_KEY, STATION_AREAS_FILENAME)
        if path is None
        else Path(path)
    )
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"no station_areas.yaml at {resolved}")
    return area_stations(yaml.safe_load(resolved.read_text(encoding="utf-8")) or {})


def station_events(
    sta: str,
    events: Iterable[Event],
    areas: Mapping[str, frozenset[str]],
    *,
    kinds: Sequence[str] = STEP_KINDS,
) -> list[Event]:
    """The events of ``kinds`` whose region contains ``sta``.

    An event whose region is not an area of ``areas`` concerns no station:
    it is never applied to every station by default.

    Raises:
        ValueError: An unknown kind in ``kinds``.
    """
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise ValueError(f"unknown event kind(s) {bad}; one of {KINDS}")
    code = sta.upper()
    return [
        e
        for e in events
        if e.kind in kinds and e.region is not None and code in areas.get(e.region, ())
    ]
