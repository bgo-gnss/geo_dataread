"""Periods of the gpsprod products: dates in, half-open fractional years out.

A period is whole days, both ends included: ``20231110:20240110`` runs from
the midnight starting 10 Nov 2023 to the midnight ending 10 Jan 2024
(:func:`geo_dataread.velocity.window_bounds`). Forms:

- ``A:B`` — dates ``YYYYMMDD`` or ``YYYY-MM-DD``;
- ``A:`` or ``since=A`` — from A to ``today``;
- ``last=N`` — the N whole days ending ``today``.

Event names (``events.yaml`` ids) come in step 2 of PLAN-gpsprod.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

from geo_dataread.velocity import last_days_start, window_bounds


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
) -> Period:
    """The period of a ``--period`` spec, or ``--since``/``--end``/``--last``.

    Exactly one of ``spec``, ``since`` and ``last`` is given; ``end``
    (default ``today``) closes ``since`` and ``last``.
    """
    today = today or dt.date.today()
    given = [x is not None for x in (spec, since, last)]
    if sum(given) != 1:
        raise ValueError("give one of --period, --since, --last")
    close = parse_date(end) if end else today
    if last is not None:
        return from_days(last_days_start(close, last), close)
    if since is not None:
        return from_days(parse_date(since), close)
    assert spec is not None
    if ":" not in spec:
        raise ValueError(f"{spec!r}: a period is A:B (dates), or A: to today")
    a, b = spec.split(":", 1)
    return from_days(parse_date(a), parse_date(b) if b else close)
