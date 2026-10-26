"""Periods of the gpsprod products: dates in, half-open fractional years out.

A period is whole days, both ends included: ``20231110:20240110`` runs from
the midnight starting 10 Nov 2023 to the midnight ending 10 Jan 2024
(:func:`geo_dataread.velocity.window_bounds`). Forms:

- ``A:B`` — dates ``YYYYMMDD`` or ``YYYY-MM-DD``;
- ``A:`` — from A to ``today``;
- event names (``events.yaml`` ids, or any part of one that names a single
  event) in place of a date: as a START, the day after the event's last day;
  as an END, the day before its onset — so ``DIKE:ERUPTION`` is the quiet
  between them, ``DIKE:`` (or ``after:DIKE``) from the dike to today;
- ``EVENT`` alone — the event's own days, onset to last day.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

from collections.abc import Sequence
from typing import TYPE_CHECKING

from geo_dataread.velocity import last_days_start, window_bounds

if TYPE_CHECKING:
    from geo_dataread.events import Event


@dataclasses.dataclass(frozen=True)
class Period:
    """A resolved period: fractional-year bounds, the dates, the names."""

    start: float
    end: float
    first_day: dt.date
    last_day: dt.date
    names: tuple[str, ...] = ()

    @property
    def bounds(self) -> tuple[float, float]:
        return self.start, self.end

    @property
    def dates(self) -> tuple[str, str]:
        return self.first_day.isoformat(), self.last_day.isoformat()


def parse_date(value: str) -> dt.date:
    """``YYYYMMDD`` or ``YYYY-MM-DD``."""
    text = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"{value!r}: use YYYYMMDD or YYYY-MM-DD")


def find_event(name: str, events: Sequence[Event]) -> Event:
    """The event ``name`` names: its id, else the one id containing it.

    Raises:
        ValueError: no event, or several, match (the candidates listed).
    """
    key = name.strip()
    for e in events:
        if e.id == key:
            return e
    hits = [e for e in events if key.lower() in e.id.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no event {key!r} in events.yaml")
    raise ValueError(
        f"{key!r} names {len(hits)} events: {', '.join(e.id for e in hits)}"
    )


def _is_date(text: str) -> bool:
    try:
        parse_date(text)
    except ValueError:
        return False
    return True


def from_days(first: dt.date, last: dt.date, names: tuple[str, ...] = ()) -> Period:
    """The period of whole days ``first`` … ``last``."""
    if last < first:
        raise ValueError(f"period ends ({last}) before it starts ({first})")
    start, end = window_bounds(first, last)
    return Period(start, end, first, last, names)


def resolve_period(
    spec: str | None = None,
    *,
    since: str | None = None,
    end: str | None = None,
    last: int | None = None,
    today: dt.date | None = None,
    events: Sequence[Event] | None = None,
    start: str | None = None,
    days: int | None = None,
) -> Period:
    """The period of a ``--period`` spec, or of ``--start``/``--end``/``--days``.

    - ``--start A [--end B]``: A to B (default ``today``);
    - ``--start A --days N``: the N days from A;
    - ``--days N [--end B]``: the N days ending B (default ``today``);
    - ``--period``: dates and event names (module doc), alone.

    ``since`` is ``start`` and ``last`` is ``days`` (the older names).
    ``events`` is the catalog; read from the deployed ``events.yaml`` only
    when a name is used.
    """
    today = today or dt.date.today()
    if since is not None and start is not None:
        raise ValueError("--since is --start; give one")
    if last is not None and days is not None:
        raise ValueError("--last is --days; give one")
    start = start if start is not None else since
    days = days if days is not None else last
    if days is not None and days < 1:
        raise ValueError(f"--days {days}: at least one day")
    if spec is not None:
        if start is not None or days is not None:
            raise ValueError("--period stands alone (not with --start/--days)")
    elif start is None and days is None:
        raise ValueError(
            "give a period: --period, --start (and --end/--days) or --days"
        )
    if start is not None and days is not None and end is not None:
        raise ValueError("--start, --days and --end: give two of them")
    close = parse_date(end) if end else today
    if start is not None:
        first = parse_date(start)
        if days is not None:
            return from_days(first, first + dt.timedelta(days=days - 1))
        return from_days(first, close)
    if days is not None:
        return from_days(last_days_start(close, days), close)
    assert spec is not None
    spec = spec.strip()
    if spec.startswith("after:"):
        spec = spec.removeprefix("after:") + ":"
    names: list[str] = []
    one_day = dt.timedelta(days=1)

    def event(text: str) -> Event:
        nonlocal events
        if events is None:
            from geo_dataread.events import read_events

            events = read_events()
        e = find_event(text, events)
        names.append(e.id)
        return e

    if ":" not in spec:
        if _is_date(spec):
            raise ValueError(f"{spec!r}: a period is A:B (dates), or A: to today")
        e = event(spec)
        return from_days(e.onset, e.last_day, (e.id,))
    a, b = spec.split(":", 1)
    if not a:
        raise ValueError(f"{spec!r}: the period needs a start")
    first = parse_date(a) if _is_date(a) else event(a).last_day + one_day
    if not b:
        last_day = close
    elif _is_date(b):
        last_day = parse_date(b)
    else:
        last_day = event(b).onset - one_day
    return from_days(first, last_day, tuple(names))
