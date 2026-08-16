"""Per-station estimation overrides in ``analysis.yaml``.

Two things live here: the generic per-station block plumbing that
:mod:`geo_dataread.stage_plan` already needed, and the **model/terms** block
that closes the gap wiring the stage plans open.

The gap, precisely: ``gps-estimate-detrend`` has ONE global ``--model`` and no
``--term`` at all, while ``gps-detrend-workbench`` curates both per station.
A plan committed under ``--model periodic`` therefore could not be honoured by
a batch run at the default model -- the group it names may not exist -- and
since the batch began reading stage plans that surfaces as a loud per-station
error instead of a silent revert.  Loud beats silent, but the station still
does not re-estimate.  A per-station model is what makes it re-estimate.

Home is ``detrend.estimation.models``, beside ``stage_plans`` and the
existing ``use_sta``, and NOT a ``fit_windows.csv`` column -- the same
argument :func:`~geo_dataread.stage_plan.read_stage_plans` makes for itself:
that reader is strict by design, and a value structurally richer than a
number (here, a transient spec ``log@2008.4085,tau=1.0``) nested into a cell
would trade its strictness for unreadability.

``terms`` here means ``--term`` transients, the ones STORED in the record and
the reason ``record_version`` 2 exists.  It is not the workbench's
``--terms``, which is the apply-time group selector, deliberately unstored,
and which ``--commit`` refuses to store under a non-default value.  The two
are different decisions and only one of them belongs in a config file.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "STATION_MODEL_KEYS",
    "StationModel",
    "merge_write_station",
    "read_station_block",
    "read_station_models",
    "station_model_to_config",
    "write_station_model",
]

#: Where per-station model/terms overrides live in ``analysis.yaml``.
STATION_MODEL_KEYS = ("detrend", "estimation", "models")


# ---------------------------------------------------------------------------
# Generic per-station block plumbing
# ---------------------------------------------------------------------------


def read_station_block(
    path: "str | Path", keys: Sequence[str], *, expected: str
) -> dict[str, Any]:
    """Raw ``{station: value}`` mapping at ``keys``, or ``{}``.

    A missing file, a missing block or an empty mapping all mean "no station
    has one" -- these blocks are optional enhancements, exactly like the fit
    catalog.  A block that EXISTS but is not a mapping raises, because a
    config that silently fails to load would quietly revert a curated station
    and store different science.

    Args:
        path: The ``analysis.yaml``.
        keys: Nested key path to the block.
        expected: What the values are, for the error message.
    """
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    doc = yaml.safe_load(p.read_text()) or {}
    node: object = doc
    for key in keys:
        if not isinstance(node, Mapping):
            return {}
        node = node.get(key)
        if node is None:
            return {}
    if not isinstance(node, Mapping):
        raise ValueError(
            f"{p}: {'.'.join(keys)} must be a mapping of station to "
            f"{expected}, got {node!r}"
        )
    return {str(sta): value for sta, value in node.items()}


def merge_write_station(
    path: "str | Path", keys: Sequence[str], station: str, value: Any | None
) -> None:
    """Merge ONE station's entry into ``analysis.yaml``, preserving the rest.

    The workbench's ``--commit`` contract for ``detrend_params.json``, applied
    to config: a merge-write of one station, never a rewrite of the document,
    so an operator curating one station cannot drop another's.

    ``value=None`` removes the station's entry (and the enclosing blocks if
    they become empty), which is how a curated station is returned to the
    ordinary defaults without hand-editing YAML.
    """
    import yaml

    p = Path(path)
    doc = (yaml.safe_load(p.read_text()) if p.exists() else None) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: top level must be a mapping, got {type(doc).__name__}")

    node: dict[str, object] = doc
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"{p}: {key!r} must be a mapping, got {child!r}")
        node = child

    leaf = node.get(keys[-1])
    if not isinstance(leaf, dict):
        leaf = {}
    if value is None:
        leaf.pop(station, None)
    else:
        leaf[station] = value

    if leaf:
        node[keys[-1]] = leaf
    else:
        node.pop(keys[-1], None)

    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


# ---------------------------------------------------------------------------
# Per-station model + transients
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StationModel:
    """One station's fit-time model and its declared transients.

    Both are FIT-time decisions stored in the record: ``model`` is the
    registry code, ``terms`` the ``--term`` specs composed on top of it.
    Either may be absent -- a station may want a transient on the default
    model, or a different model with no transient.
    """

    model: str | None = None
    terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model is None and not self.terms:
            raise ValueError(
                "a station model entry must set 'model' or 'terms' (or both); "
                "an empty entry is indistinguishable from having none, and "
                "would look like curation that is not there"
            )


def station_model_to_config(entry: StationModel) -> dict[str, Any]:
    """Serialize one entry, omitting what was not set."""
    out: dict[str, Any] = {}
    if entry.model is not None:
        out["model"] = entry.model
    if entry.terms:
        out["terms"] = list(entry.terms)
    return out


def _station_model_from_config(raw: Any, sta: str) -> StationModel:
    """Parse one entry, validating the term specs eagerly.

    ``parse_term_spec`` runs HERE rather than at estimation time so a typo in
    a tau is a config error naming the station, not a failure five stations
    into a batch.
    """
    from geo_dataread.term_spec import parse_term_spec

    if not isinstance(raw, Mapping):
        raise ValueError(f"models[{sta!r}] must be a mapping, got {raw!r}")
    unknown = set(raw) - {"model", "terms"}
    if unknown:
        raise ValueError(
            f"models[{sta!r}]: unknown key(s) {sorted(unknown)}; "
            f"known: ['model', 'terms']"
        )
    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError(f"models[{sta!r}]: 'model' must be a string, got {model!r}")
    raw_terms = raw.get("terms") or []
    if isinstance(raw_terms, str) or not isinstance(raw_terms, Sequence):
        raise ValueError(
            f"models[{sta!r}]: 'terms' must be a list of term specs, got {raw_terms!r}"
        )
    terms = tuple(str(t) for t in raw_terms)
    for spec in terms:
        try:
            parse_term_spec(spec)
        except ValueError as exc:
            raise ValueError(f"models[{sta!r}]: term {spec!r}: {exc}") from None
    try:
        return StationModel(model=model, terms=terms)
    except ValueError as exc:
        raise ValueError(f"models[{sta!r}]: {exc}") from None


def read_station_models(path: "str | Path") -> dict[str, StationModel]:
    """Read every station's model/terms override from ``analysis.yaml``.

    Raises:
        ValueError: if the block exists but is malformed, naming the station.
    """
    raw = read_station_block(path, STATION_MODEL_KEYS, expected="a model/terms mapping")
    out: dict[str, StationModel] = {}
    for sta, entry in raw.items():
        try:
            out[sta] = _station_model_from_config(entry, sta)
        except ValueError as exc:
            raise ValueError(f"{Path(path)}: {exc}") from None
    return out


def write_station_model(
    path: "str | Path", station: str, entry: StationModel | None
) -> None:
    """Merge ONE station's model/terms into ``analysis.yaml``.

    ``entry=None`` removes it, returning the station to the batch's global
    ``--model`` and no transients.
    """
    merge_write_station(
        path,
        STATION_MODEL_KEYS,
        station,
        None if entry is None else station_model_to_config(entry),
    )
