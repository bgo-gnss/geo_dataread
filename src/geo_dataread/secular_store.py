"""The secular store — s(t) as a saved, reusable component.

The background of a GNSS trajectory (rate + annual + semiannual, per
component) is estimated ONCE on clean data and then reused: held fixed while
offsets and transients are estimated against it, and borrowed by stations
whose own series is too short or too noisy to constrain a seasonal.

That object already existed, in the legacy ``detrend_itrf2008.csv``:
``Nrate,Nacos,Nasin,Nscos,Nssin`` per component, plus ``Starttime``/
``Endtime`` (the window it was fitted on), ``UseSTA`` (borrow from another
station) and ``Fit``.  65 stations depend on it.  What never existed is a
way for the modern lane to WRITE one — ``gps-detrend-workbench --commit``
stores a finished record and nothing else, so an operator could not fix a
background and come back to it.

Two stores, two purposes, and the distinction is load-bearing:

``detrend.secular`` (here)
    s(t) as a COMPONENT.  Held while events are estimated, borrowed across
    stations.  Never read by production to detrend anything.

``detrend_params.json``
    The FINISHED model f(t) = s(t) + steps + transients, which is what
    ``plot-gps-timeseries`` and the production read path consume.

Keeping them apart is not tidiness.  A background committed into the
finished store would look complete and be wrong: production would detrend
the station with s(t) alone and serve a series with its coseismic offset
still in it, with no error anywhere.  A flag on one shared store would work
until the first reader forgot to check it.

Two differences from the CSV, both deliberate:

* **The offset is stored.**  The CSV omits it because detrending does not
  need one — subtracting a constant changes nothing that is looked at.  But
  HOLDING s(t) while estimating a step does need the level anchored: with
  the offset free the step is measured against a floating background, and
  with it held at a value fitted on the wrong side of the event the step
  cannot be measured at all (measured on SELF, 2026-08-22: step estimated at
  0.0 mm against a true −150.8).
* **Segments, not one window.**  A background is best fitted on the clean
  intervals EITHER SIDE of an event, which is a union; the CSV can only
  spell a single span.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "SECULAR_KEYS",
    "SECULAR_GROUPS",
    "SecularEntry",
    "read_secular",
    "secular_from_record",
    "secular_group_values",
    "secular_to_config",
    "write_secular",
]

#: Home in ``analysis.yaml``, beside ``stage_plans`` and ``models`` — the same
#: kind of thing, an operator's per-station estimation decision.
SECULAR_KEYS = ("detrend", "secular")

#: The term groups s(t) is made of.  Steps and transients are EVENTS: they are
#: what gets estimated against the background, never part of it.
SECULAR_GROUPS = ("secular", "periodic")

#: Component order, as everywhere else in this ecosystem.
_COMPONENTS = ("north", "east", "up")


@dataclasses.dataclass(frozen=True)
class SecularEntry:
    """One station's saved background.

    Attributes:
        model: Registry code the parameters belong to (``lineperiodic`` …).
        param_names: Names of the stored parameters, in order — the record's
            own, sliced to the secular groups.  Stored rather than implied so
            a reader never has to guess which model produced them.
        components: Per-component parameter vectors, ``north``/``east``/``up``.
        segments: The intervals it was fitted on.  A union is expressible;
            None means the fit's own domain was not recorded.
        use_sta: Borrow this station's background instead of the stored one.
            The CSV's ``UseSTA``, carried over.  When set, ``components`` may
            be empty — the values live at the donor.
        fit: Which part to apply when borrowing (``periodic`` /
            ``lineperiodic``).  The CSV's ``Fit``.
        fitted_at: Opaque provenance stamp; the leaf reads no clock.
    """

    model: str
    param_names: tuple[str, ...] = ()
    components: Mapping[str, tuple[float, ...]] = dataclasses.field(
        default_factory=dict
    )
    segments: tuple[tuple[float | None, float | None], ...] | None = None
    use_sta: str | None = None
    fit: str | None = None
    fitted_at: str | None = None


def secular_from_record(
    record: Mapping[str, Any],
    *,
    fitted_at: str | None = None,
) -> SecularEntry:
    """Slice s(t) out of a finished record.

    Only :data:`SECULAR_GROUPS` are taken.  A record carrying a step or a
    transient is not refused — those are events, and the whole point of
    saving a background estimated ALONGSIDE them is that the fit which
    separated them is the one that got the background right.

    The group membership comes from
    :func:`gps_analysis.staged.record_group_mask`, so "secular" keeps one
    definition across the estimator, the borrow path and this store.
    """
    from gps_analysis.staged import record_group_mask

    model = record.get("model")
    if not isinstance(model, str):
        raise ValueError("record has no model code; cannot save a background")
    names = list(record.get("param_names") or ())
    mask = record_group_mask(record, list(SECULAR_GROUPS))
    if not mask.any():
        raise ValueError(
            f"model {model!r} has no {list(SECULAR_GROUPS)} terms, so there is "
            f"no background to save"
        )
    kept_names = tuple(n for n, m in zip(names, mask, strict=False) if m)

    comps = record.get("components") or ()
    values: dict[str, tuple[float, ...]] = {}
    for index, entry in enumerate(comps):
        if index >= len(_COMPONENTS) or not isinstance(entry, Mapping):
            continue
        params = list(entry.get("params") or ())
        values[_COMPONENTS[index]] = tuple(
            float(p) for p, m in zip(params, mask, strict=False) if m
        )

    segments = record.get("segments")
    seg = (
        None
        if not segments
        else tuple(
            (
                None if s[0] is None else float(s[0]),
                None if s[1] is None else float(s[1]),
            )
            for s in segments
        )
    )
    return SecularEntry(
        model=model,
        param_names=kept_names,
        components=values,
        segments=seg,
        fitted_at=fitted_at,
    )


def secular_group_values(
    entry: SecularEntry, group: str, *, component: int, station: str
) -> "Any":
    """The coefficients of one group of a saved background.

    Mirrors :func:`geo_dataread.stage_plan.donor_group_values`, but reads the
    STORE rather than a finished record — which is the whole reason the hold
    kinds are spelled differently.
    """
    import numpy as np
    from gps_analysis.staged import _staged_group_of

    if component >= len(_COMPONENTS):
        raise ValueError(
            f"secular store {station!r}: no component {component} "
            f"(known: {list(_COMPONENTS)})"
        )
    name = _COMPONENTS[component]
    params = entry.components.get(name)
    if not params:
        raise ValueError(
            f"secular store {station!r}: no {name} parameters stored"
            + (
                f" — it borrows from {entry.use_sta!r}, which must be resolved first"
                if entry.use_sta
                else ""
            )
        )
    if len(params) != len(entry.param_names):
        raise ValueError(
            f"secular store {station!r}: {name} has {len(params)} values but "
            f"{len(entry.param_names)} parameter names"
        )
    picked = [
        float(v)
        for v, n in zip(params, entry.param_names, strict=True)
        if _staged_group_of(n) == group
    ]
    if not picked:
        raise ValueError(
            f"secular store {station!r}: stores no {group!r} term "
            f"(has {list(entry.param_names)})"
        )
    return np.asarray(picked, dtype=float)


def secular_to_config(entry: SecularEntry) -> dict[str, Any]:
    """Serialise one entry.  Absent optional keys rather than nulls."""
    out: dict[str, Any] = {"model": entry.model}
    if entry.param_names:
        out["param_names"] = list(entry.param_names)
    for name in _COMPONENTS:
        values = entry.components.get(name)
        if values:
            out[name] = [float(v) for v in values]
    if entry.segments:
        out["segments"] = [[s[0], s[1]] for s in entry.segments]
    if entry.use_sta:
        out["use_sta"] = entry.use_sta
    if entry.fit:
        out["fit"] = entry.fit
    if entry.fitted_at:
        out["fitted_at"] = entry.fitted_at
    return out


def _entry_from_config(raw: Any, sta: str) -> SecularEntry:
    if not isinstance(raw, Mapping):
        raise ValueError(f"secular[{sta!r}] must be a mapping, got {raw!r}")
    model = raw.get("model")
    use_sta = raw.get("use_sta")
    if not isinstance(model, str):
        # A pure borrow row (the CSV's UseSTA with no coefficients of its own)
        # is legal and carries no model of its own.
        if use_sta:
            model = ""
        else:
            raise ValueError(f"secular[{sta!r}]: 'model' is required")
    names = raw.get("param_names") or ()
    if not isinstance(names, Sequence) or isinstance(names, str):
        raise ValueError(f"secular[{sta!r}]: 'param_names' must be a list")

    components: dict[str, tuple[float, ...]] = {}
    for name in _COMPONENTS:
        values = raw.get(name)
        if values is None:
            continue
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ValueError(f"secular[{sta!r}]: {name!r} must be a list")
        try:
            components[name] = tuple(float(v) for v in values)
        except (TypeError, ValueError):
            raise ValueError(
                f"secular[{sta!r}]: {name!r} is not all numeric: {values!r}"
            ) from None

    segments = raw.get("segments")
    seg = None
    if segments is not None:
        if not isinstance(segments, Sequence) or isinstance(segments, str):
            raise ValueError(f"secular[{sta!r}]: 'segments' must be a list")
        parsed: list[tuple[float | None, float | None]] = []
        for item in segments:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError(
                    f"secular[{sta!r}]: each segment must be [start, end], got {item!r}"
                )
            lo, hi = item
            parsed.append(
                (None if lo is None else float(lo), None if hi is None else float(hi))
            )
        seg = tuple(parsed)

    return SecularEntry(
        model=model,
        param_names=tuple(str(n) for n in names),
        components=components,
        segments=seg,
        use_sta=str(use_sta) if use_sta else None,
        fit=str(raw["fit"]) if raw.get("fit") else None,
        fitted_at=str(raw["fitted_at"]) if raw.get("fitted_at") else None,
    )


def read_secular(path: "str | Path") -> dict[str, SecularEntry]:
    """Read every station's saved background from ``analysis.yaml``.

    A missing file, a missing block or an empty mapping all mean "no station
    has a saved background" and return ``{}`` — the store is an optional
    enhancement, exactly like the fit catalog and the stage plans.

    Raises:
        ValueError: if the block exists but is malformed, naming the station.
            A background that silently failed to load would quietly return a
            station to estimating its own, and store different science.
    """
    from geo_dataread.analysis_yaml import read_station_block

    node = read_station_block(Path(path), SECULAR_KEYS, expected="secular entry")
    return {sta: _entry_from_config(raw, sta) for sta, raw in node.items()}


def write_secular(path: "str | Path", station: str, entry: SecularEntry | None) -> None:
    """Merge ONE station's background into ``analysis.yaml``.

    ``entry=None`` removes it, which is how a station goes back to estimating
    its own background without hand-editing YAML.
    """
    from geo_dataread.analysis_yaml import merge_write_station

    merge_write_station(
        path,
        SECULAR_KEYS,
        station,
        None if entry is None else secular_to_config(entry),
    )
