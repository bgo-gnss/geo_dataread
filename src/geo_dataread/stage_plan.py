"""The staged-estimation grammar — one grammar, several producers.

:func:`gps_analysis.estimate_staged` has shipped since Phase 1 and until now
had **no command-line or config surface at all**: the Askja manoeuvre it was
built for could not be run by an operator.  This module is that surface, and
it is deliberately a separate module from the CLI that first uses it, because
the graphical picker is specified to emit *the same grammar the CLI parses*
(program plan, "Out of scope").  A grammar living inside one CLI's ``argparse``
block is a grammar the picker would have to reimplement — which is how two
grammars are born.

Spellings::

    --stage NAME:GROUP,GROUP@START:END        fit domain optional
    --hold  [STAGE:]GROUP=stage:NAME          hold at what an earlier stage fitted
    --hold  [STAGE:]GROUP=donor:STA           hold at a donor station's value

Three refusals, each preventing a *silent* change of stored science rather than
a mere typo:

1. **The hold kind must be named.**  A bare ``--hold periodic=clean`` is
   refused.  ``stage:`` and ``donor:`` produce different record provenance —
   :class:`~gps_analysis.HeldFromStage` versus
   :class:`~gps_analysis.HeldExplicit` with a ``source`` — so resolving the
   kind by "does it happen to match a declared stage name?" would let *renaming
   a stage* flip a borrow into a self-reference with no error.  Same reasoning
   as the existing ``--segment`` / ``--window-start`` refusal.
2. **The stage binding is explicit once it can be ambiguous.**  ``Stage.held``
   is per-stage, so a hold must say which stage it belongs to whenever more
   than one is declared.  Inferring "the last ``--stage`` seen" would make
   argument *order* semantically significant, and nothing else in these CLIs
   works that way.  With 0 or 1 stages the prefix is optional, which keeps the
   borrow one-liner (``--hold secular=donor:OLAC``) clean.
3. **A bare trailing ``@`` is illegal.**  Omitting ``@`` means *inherit the
   caller's domain* (``segments=None``); ``@:`` means the *full span*
   (one open-ended segment).  Those differ whenever the caller passed
   ``--segment``, so ``@`` with nothing after it is rejected rather than
   silently picking one.

Group names are **not** enumerated here.  ``estimate_staged`` validates them
against the model's actual term groups, so a group added later (``transient``,
when the transient terms land) is addressable with no change to this module or
to any ``argparse`` ``choices=``.

Donor holds are **pointers, not copies**: :class:`DonorRef` stores the station
code and is resolved against the donor's *current* record at estimation time,
so a re-estimated donor propagates to everyone borrowing from it.  This matches
``analysis.yaml``'s existing ``detrend.estimation.use_sta`` mapping.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from gps_analysis.staged import Stage

__all__ = [
    "DonorRef",
    "HoldRef",
    "StagePlan",
    "StageRef",
    "StageSpec",
    "STAGE_PLAN_KEYS",
    "annotate_donor_groups",
    "build_stage_plan",
    "default_analysis_yaml_path",
    "donor_drift_warnings",
    "donor_group_digest",
    "donor_group_values",
    "read_stage_plans",
    "parse_hold_spec",
    "parse_stage_spec",
    "resolve_stage_plan",
    "stage_plan_from_config",
    "stage_plan_to_config",
    "write_stage_plan",
]

#: How a held group's value is spelled on the command line and in config.
_HOLD_KINDS = ("stage", "donor")

Segments = tuple[tuple[float | None, float | None], ...]


@dataclasses.dataclass(frozen=True)
class StageRef:
    """Hold a term group at the value an earlier stage fitted.

    Resolves to :class:`gps_analysis.HeldFromStage`.
    """

    stage: str


@dataclasses.dataclass(frozen=True)
class DonorRef:
    """Hold a term group at a donor station's stored value — a POINTER.

    Only the station code is stored.  The coefficients are looked up from the
    donor's current record when the plan is resolved, so re-estimating the
    donor updates every station borrowing from it.  A donor with no record is
    a loud error at resolution time, never a stale value.
    """

    station: str


HoldRef = StageRef | DonorRef


@dataclasses.dataclass(frozen=True)
class StageSpec:
    """One stage, in the CLI/config vocabulary (pointers, not values).

    The value-carrying counterpart is :class:`gps_analysis.Stage`; this is what
    survives a round trip through a config file, where a donor's numbers must
    not be frozen in.

    Attributes:
        name: Stage label, referenced by :class:`StageRef`.
        free: Term-group names estimated in this stage.
        held: Group name → where its value comes from.
        segments: Fit domain, or None to inherit the caller's.
    """

    name: str
    free: tuple[str, ...]
    held: Mapping[str, HoldRef] = dataclasses.field(default_factory=dict)
    segments: Segments | None = None


@dataclasses.dataclass(frozen=True)
class StagePlan:
    """An ordered list of stages — the whole declarative estimation plan."""

    stages: tuple[StageSpec, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a stage plan needs at least one stage")
        seen: set[str] = set()
        for st in self.stages:
            if st.name in seen:
                raise ValueError(f"duplicate stage name {st.name!r}")
            seen.add(st.name)

    @property
    def names(self) -> tuple[str, ...]:
        """Stage names in declaration order."""
        return tuple(st.name for st in self.stages)

    @property
    def donors(self) -> tuple[str, ...]:
        """Every donor station referenced, deduplicated, in first-use order."""
        out: list[str] = []
        for st in self.stages:
            for ref in st.held.values():
                if isinstance(ref, DonorRef) and ref.station not in out:
                    out.append(ref.station)
        return tuple(out)


def _parse_segments(text: str, *, context: str) -> Segments:
    """Parse ``a:b`` or ``a:b;c:d`` — an empty bound is an open bound.

    Same shape as ``fit_windows.csv``'s ``segments`` cell (``;`` lists, ``:``
    separates bounds, ``-`` deliberately avoided as it is ambiguous against a
    negative fractional year).  Kept as its own parser rather than shared with
    :func:`geo_dataread.detrend_estimate._parse_segments_cell`: that one reads
    a deployed strict-by-design catalog and its error vocabulary is
    station-keyed, while this one reports against a CLI token.
    """
    out: list[tuple[float | None, float | None]] = []
    for token in (tok.strip() for tok in text.split(";")):
        if not token:
            continue
        if token.count(":") != 1:
            raise ValueError(
                f"{context}: segment {token!r} must be 'START:END' "
                f"(either side may be empty for an open bound)"
            )
        lo_raw, _, hi_raw = token.partition(":")

        def _bound(raw: str, which: str) -> float | None:
            raw = raw.strip()
            if not raw:
                return None
            try:
                return float(raw)
            except ValueError:
                raise ValueError(
                    f"{context}: segment {which} {raw!r} is not a fractional year"
                ) from None

        out.append((_bound(lo_raw, "start"), _bound(hi_raw, "end")))
    if not out:
        raise ValueError(f"{context}: no segment in {text!r}")
    for j, (lo, hi) in enumerate(out):
        if lo is not None and hi is not None and hi <= lo:
            raise ValueError(f"{context}: segment {j} end {hi} <= start {lo}")
    return tuple(out)


def parse_stage_spec(spec: str) -> StageSpec:
    """Parse one ``--stage NAME:GROUP,GROUP[@START:END[;START:END]]``.

    The fit domain is optional and its absence is meaningful: no ``@`` means
    *inherit the caller's domain*, while ``@:`` means the full span.  A bare
    trailing ``@`` is refused because those two readings differ.

    Args:
        spec: The raw flag value.

    Returns:
        A :class:`StageSpec` with an empty ``held`` — holds arrive separately
        via :func:`parse_hold_spec` and are attached by
        :func:`build_stage_plan`.
    """
    if ":" not in spec:
        raise ValueError(
            f"--stage {spec!r} must be 'NAME:GROUP[,GROUP][@START:END]' "
            f"(the ':' after the stage name is required)"
        )
    name, _, rest = spec.partition(":")
    name = name.strip()
    if not name:
        raise ValueError(f"--stage {spec!r} has an empty stage name")

    segments: Segments | None = None
    if "@" in rest:
        free_part, _, seg_part = rest.partition("@")
        if not seg_part.strip():
            raise ValueError(
                f"--stage {spec!r}: a bare trailing '@' is ambiguous — omit it "
                f"to inherit the caller's fit domain, or write '@:' for the "
                f"full span"
            )
        segments = _parse_segments(seg_part, context=f"--stage {spec!r}")
    else:
        free_part = rest

    free = tuple(g.strip() for g in free_part.split(",") if g.strip())
    if not free:
        raise ValueError(
            f"--stage {spec!r} declares no free term group; a stage that "
            f"estimates nothing is not a stage"
        )
    dupes = {g for g in free if free.count(g) > 1}
    if dupes:
        raise ValueError(f"--stage {spec!r} repeats group(s) {sorted(dupes)}")
    return StageSpec(name=name, free=free, held={}, segments=segments)


def parse_hold_spec(spec: str) -> tuple[str | None, str, HoldRef]:
    """Parse one ``--hold [STAGE:]GROUP=stage:NAME`` or ``=donor:STA``.

    Args:
        spec: The raw flag value.

    Returns:
        ``(stage_or_None, group, ref)``.  ``stage_or_None`` is the stage the
        hold binds to, or None when the spec did not name one — resolved (and
        required, if ambiguous) by :func:`build_stage_plan`.

    Raises:
        ValueError: if the hold kind is not named.  This is the refusal that
            matters: ``stage:`` and ``donor:`` produce different record
            provenance, so guessing between them could rewrite stored science
            when a stage is renamed.
    """
    if "=" not in spec:
        raise ValueError(
            f"--hold {spec!r} must be '[STAGE:]GROUP=stage:NAME' or "
            f"'[STAGE:]GROUP=donor:STA'"
        )
    left, _, right = spec.partition("=")
    left, right = left.strip(), right.strip()
    if not left:
        raise ValueError(f"--hold {spec!r} names no term group")

    kind, sep, value = right.partition(":")
    kind, value = kind.strip(), value.strip()
    if not sep or kind not in _HOLD_KINDS or not value:
        raise ValueError(
            f"--hold {spec!r}: the held value must name its kind — write "
            f"'{left}=stage:NAME' to hold at what an earlier stage fitted, or "
            f"'{left}=donor:STA' to borrow station STA's value. "
            f"These are stored differently, so the kind is never inferred."
        )
    ref: HoldRef = StageRef(value) if kind == "stage" else DonorRef(value)

    if ":" in left:
        stage, _, group = left.partition(":")
        stage, group = stage.strip(), group.strip()
        if not stage or not group:
            raise ValueError(f"--hold {spec!r}: expected 'STAGE:GROUP' before '='")
        return stage, group, ref
    return None, left, ref


def build_stage_plan(
    stage_specs: Sequence[str], hold_specs: Sequence[str] = ()
) -> StagePlan:
    """Assemble ``--stage`` / ``--hold`` flag values into a :class:`StagePlan`.

    The stage prefix on a hold is **required once two or more stages are
    declared** and optional below that, so the single-stage borrow
    (``--hold secular=donor:OLAC``) stays a one-liner while the ambiguous case
    is made impossible rather than resolved by convention.

    Args:
        stage_specs: Raw ``--stage`` values, in declaration order.
        hold_specs: Raw ``--hold`` values.

    Returns:
        The assembled plan.

    Raises:
        ValueError: on any grammar violation, an unbound hold, a hold naming
            an undeclared stage, a group both free and held in one stage, or a
            :class:`StageRef` pointing at a stage that is not strictly earlier.
    """
    stages = [parse_stage_spec(s) for s in stage_specs]

    if not stages:
        if not hold_specs:
            raise ValueError("build_stage_plan needs at least one --stage or --hold")
        # The pure-borrow case: an implicit single stage that frees nothing
        # explicitly is not expressible, so require the operator to say what
        # the stage estimates rather than inventing a default here.
        raise ValueError(
            "--hold without --stage: declare the stage that does the fitting, "
            "e.g. --stage fit:secular --hold secular=donor:OLAC"
        )

    by_name = {st.name: st for st in stages}
    order = {st.name: i for i, st in enumerate(stages)}
    held: dict[str, dict[str, HoldRef]] = {st.name: {} for st in stages}

    for raw in hold_specs:
        stage_name, group, ref = parse_hold_spec(raw)
        if stage_name is None:
            if len(stages) > 1:
                raise ValueError(
                    f"--hold {raw!r} does not say which stage it belongs to, and "
                    f"{len(stages)} stages are declared "
                    f"({', '.join(st.name for st in stages)}). "
                    f"Write '--hold STAGE:{group}={_spell(ref)}'. "
                    f"Binding to the last --stage seen would make flag ORDER "
                    f"change the science."
                )
            stage_name = stages[0].name
        if stage_name not in by_name:
            raise ValueError(
                f"--hold {raw!r} names stage {stage_name!r}, which is not "
                f"declared (have: {', '.join(by_name)})"
            )
        if group in held[stage_name]:
            raise ValueError(
                f"--hold {raw!r}: group {group!r} is already held in stage "
                f"{stage_name!r}"
            )
        if group in by_name[stage_name].free:
            raise ValueError(
                f"--hold {raw!r}: group {group!r} is also FREE in stage "
                f"{stage_name!r} — a group is estimated or held, never both"
            )
        if isinstance(ref, StageRef):
            if ref.stage not in by_name:
                raise ValueError(
                    f"--hold {raw!r} refers to stage {ref.stage!r}, which is "
                    f"not declared (have: {', '.join(by_name)})"
                )
            if order[ref.stage] >= order[stage_name]:
                raise ValueError(
                    f"--hold {raw!r}: stage {ref.stage!r} is not earlier than "
                    f"{stage_name!r}; a hold can only reuse an ALREADY-fitted "
                    f"value"
                )
            if group not in by_name[ref.stage].free:
                raise ValueError(
                    f"--hold {raw!r}: stage {ref.stage!r} does not fit group "
                    f"{group!r} (it frees {', '.join(by_name[ref.stage].free)}), "
                    f"so there is no value to hold"
                )
        held[stage_name][group] = ref

    return StagePlan(
        tuple(dataclasses.replace(st, held=dict(held[st.name])) for st in stages)
    )


def _spell(ref: HoldRef) -> str:
    """Render a :class:`HoldRef` back into its command-line spelling."""
    if isinstance(ref, StageRef):
        return f"stage:{ref.stage}"
    return f"donor:{ref.station}"


def stage_plan_to_config(plan: StagePlan) -> list[dict[str, object]]:
    """Render a plan into the ``analysis.yaml`` mapping form.

    Config home is ``detrend.overrides.<STA>.stage_plan`` — nested YAML rather
    than a ``fit_windows.csv`` cell.  That catalog's one-cell ``segments``
    encoding exists because its reader is strict by design and an extra ROW
    would parse into different-but-valid science; a stage plan is structurally
    richer than a segment list, and nesting it into a CSV cell would trade that
    strictness for unreadability.

    Donor holds render as ``donor: STA`` — the pointer, never the donor's
    numbers, so a re-estimated donor propagates.

    The key names match :meth:`gps_analysis.StagedEstimate.to_record_fragment`
    exactly (``name`` / ``free`` / ``held`` / ``segments``, holds spelled
    ``stage:NAME`` / ``donor:STA``).  That is not a coincidence worth losing:
    it means a STORED RECORD's ``stage_plan`` is itself valid input to
    :func:`stage_plan_from_config`, so "what was judged is what got stored"
    is checkable by parsing the record back, not merely asserted.
    """
    out: list[dict[str, object]] = []
    for st in plan.stages:
        entry: dict[str, object] = {"name": st.name, "free": list(st.free)}
        if st.held:
            entry["held"] = {g: _spell(r) for g, r in st.held.items()}
        if st.segments is not None:
            entry["segments"] = [list(seg) for seg in st.segments]
        out.append(entry)
    return out


def stage_plan_from_config(entries: Iterable[Mapping[str, object]]) -> StagePlan:
    """Rebuild a plan from ``analysis.yaml``, round-tripping :func:`stage_plan_to_config`.

    Reuses the command-line parsers for the ``hold`` values so the config and
    the CLI can never drift into two grammars — the whole point of this module.
    """
    stage_specs: list[str] = []
    hold_specs: list[str] = []
    for raw in entries:
        name = str(raw.get("name", "")).strip()
        free_raw = raw.get("free")
        free = (
            [str(g).strip() for g in free_raw]
            if isinstance(free_raw, (list, tuple))
            else []
        )
        if not name or not free:
            raise ValueError(f"stage_plan entry {raw!r} needs 'name' and 'free'")
        spec = f"{name}:{','.join(free)}"

        segs_raw = raw.get("segments")
        if segs_raw is not None:
            if not isinstance(segs_raw, (list, tuple)):
                raise ValueError(
                    f"stage_plan entry {name!r}: 'segments' must be a list of "
                    f"[start, end] pairs, got {segs_raw!r}"
                )
            tokens: list[str] = []
            for seg in segs_raw:
                if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                    raise ValueError(
                        f"stage_plan entry {name!r}: segment {seg!r} must be a "
                        f"[start, end] pair (either may be null)"
                    )
                lo, hi = seg
                tokens.append(f"{'' if lo is None else lo}:{'' if hi is None else hi}")
            spec += "@" + ";".join(tokens)
        stage_specs.append(spec)

        hold_raw = raw.get("held")
        if hold_raw is not None:
            if not isinstance(hold_raw, Mapping):
                raise ValueError(
                    f"stage_plan entry {name!r}: 'held' must be a mapping of "
                    f"GROUP to 'stage:NAME' or 'donor:STA', got {hold_raw!r}"
                )
            for group, value in hold_raw.items():
                hold_specs.append(f"{name}:{group}={value}")
    return build_stage_plan(stage_specs, hold_specs)


# ---------------------------------------------------------------------------
# Resolution: pointers -> values, against the donor's CURRENT record
# ---------------------------------------------------------------------------


def donor_group_values(
    record: Mapping[str, object],
    group: str,
    *,
    component: int,
    donor: str,
) -> "np.ndarray":
    """Slice one term group's coefficients out of a donor's stored record.

    The record is self-contained and in the absolute-t parameterization, so a
    donor's coefficients evaluate on any station's series — that is what makes
    borrowing well-defined at all (design §0.6/§2.6).

    Group membership comes from
    :func:`gps_analysis.staged.group_parameter_mask`, NOT from a local list of
    parameter names: "secular" must keep exactly one definition across the
    estimator, ``select_terms`` and this borrow path.

    Args:
        record: The donor's stored detrend record.
        group: Term-group name.
        component: Component index into ``record["components"]``.
        donor: Station code, for error messages only.

    Returns:
        The group's coefficients, shape ``(k,)``.

    Raises:
        ValueError: If the donor's model has no such group, or its record is
            malformed.  Never returns an empty vector — a silent empty borrow
            would hold a group at nothing and look like a successful fit.
    """
    import numpy as np
    from gps_analysis.staged import group_parameter_mask

    model = record.get("model")
    if not isinstance(model, str):
        raise ValueError(f"donor {donor!r}: record has no model code")
    comps = record.get("components")
    if not isinstance(comps, Sequence) or component >= len(comps):
        raise ValueError(
            f"donor {donor!r}: record has no component {component} "
            f"(has {len(comps) if isinstance(comps, Sequence) else 0})"
        )
    entry = comps[component]
    if not isinstance(entry, Mapping):
        raise ValueError(f"donor {donor!r}: component {component} is malformed")
    params = np.asarray(entry.get("params"), dtype=float)

    mask = group_parameter_mask(model, group)
    if mask.size != params.size:
        raise ValueError(
            f"donor {donor!r}: model {model!r} has {mask.size} parameters but "
            f"the record stores {params.size} for component {component}"
        )
    if not mask.any():
        raise ValueError(
            f"donor {donor!r}: model {model!r} has no {group!r} term, so there "
            f"is nothing to borrow"
        )
    return params[mask]


def donor_group_digest(record: Mapping[str, object], group: str, *, donor: str) -> str:
    """Short content digest of the coefficients a donor borrow resolves to.

    Covers the group's coefficients of **every** component, because donor
    holds are resolved per component (:func:`resolve_stage_plan` is called
    once per component in ``_restage``) — a component-0-only digest would
    miss a donor whose east seasonal moved while north stayed put.

    The digest is built from ``repr`` of the floats.  Records store
    full-``repr`` precision and JSON round-trips it bit-identically (the
    ``to_record`` contract in ``gps_analysis.detrend``), so the digest is
    stable across write → deploy → read, and ANY coefficient change — however
    small — changes it.  That is the point: drift detection must not decide
    what magnitude of change "matters"; the operator does.

    Args:
        record: The donor's stored detrend record.
        group: Term-group name (staged vocabulary).
        donor: Station code, for error messages only.

    Returns:
        12 hex chars of SHA-256 — enough to compare, short enough to print
        in a warning.

    Raises:
        ValueError: Whatever :func:`donor_group_values` raises, plus a
            record with no components at all — a digest of nothing would
            compare equal to another digest of nothing and hide the very
            malformation it should surface.
    """
    import hashlib

    comps = record.get("components")
    n_components = len(comps) if isinstance(comps, Sequence) else 0
    if n_components == 0:
        raise ValueError(f"donor {donor!r}: record has no components to digest")
    parts = [
        ",".join(
            repr(float(v))
            for v in donor_group_values(record, group, component=c, donor=donor)
        )
        for c in range(n_components)
    ]
    return hashlib.sha256(";".join(parts).encode("ascii")).hexdigest()[:12]


def annotate_donor_groups(
    groups: Mapping[str, Mapping[str, object]],
    *,
    lookup_donor: "Callable[[str], Mapping[str, object]]",
) -> dict[str, dict[str, object]]:
    """Stamp donor vintage + digest onto the donor-held entries of ``groups``.

    The ``groups`` block is written by
    :meth:`gps_analysis.StagedEstimate.to_record_fragment`, which sees only
    the ``HeldExplicit`` this module built — it cannot know the donor's
    ``record_version``/``fitted_at`` or coefficients beyond one component.
    This side resolved the pointer, so this side records what it resolved TO:
    without that, a re-estimated donor changes every borrower's next record
    with nothing anywhere saying it happened.

    A donor entry is recognised by its provenance spelling
    ``donor:STA@<fitted_at>`` — written by :func:`resolve_stage_plan` in this
    same module, so writer and reader of the spelling cannot drift apart.
    Re-deriving "which groups are donor-held" from the plan instead would
    mean reimplementing ``estimate_staged``'s ownership rule (last hold
    wins), i.e. a second place assembling the same decision.

    Args:
        groups: The fragment's ``groups`` block.
        lookup_donor: Station code → that station's current record — the
            same lookup the holds were resolved against moments earlier, so
            vintage and digest describe the values actually used.

    Returns:
        A new mapping; non-donor entries are copied unchanged.
    """
    out: dict[str, dict[str, object]] = {}
    for group, entry in groups.items():
        annotated = dict(entry)
        provenance = annotated.get("provenance")
        if isinstance(provenance, str) and provenance.startswith("donor:"):
            station = provenance[len("donor:") :].partition("@")[0]
            record = lookup_donor(station)
            annotated["donor"] = station
            annotated["donor_fitted_at"] = record.get("fitted_at")
            annotated["donor_record_version"] = record.get("record_version")
            annotated["donor_digest"] = donor_group_digest(record, group, donor=station)
        out[group] = annotated
    return out


def donor_drift_warnings(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
    *,
    station: str,
) -> list[str]:
    """Name every donor whose values moved between two of a station's records.

    ``DonorRef`` is a pointer by design — re-estimating a donor propagates to
    every borrower — and this is the propagation's only witness: compare the
    ``donor_digest`` the deployed record stored against the one this run just
    resolved.  A pure comparison of two records; no lookup, no I/O, and no
    veto — the caller warns and the NEW value is used regardless (pointer
    semantics preserved).

    A changed ``donor`` station or a group no longer donor-held is a plan
    edit, not drift, and stays silent: the operator did that on purpose.
    A ``previous`` of None (first commit, or no deployed record) has nothing
    to compare and returns no warnings.

    Args:
        previous: The station's deployed record, or None.
        current: The record this run produced.
        station: The borrowing station, for the warning text.

    Returns:
        One message per drifted (group, donor), empty when nothing drifted.
    """
    if previous is None:
        return []
    prev_groups = previous.get("groups")
    cur_groups = current.get("groups")
    if not isinstance(prev_groups, Mapping) or not isinstance(cur_groups, Mapping):
        return []
    out: list[str] = []
    for group, cur in cur_groups.items():
        if not isinstance(cur, Mapping):
            continue
        donor = cur.get("donor")
        digest = cur.get("donor_digest")
        if donor is None or digest is None:
            continue
        prev = prev_groups.get(group)
        if not isinstance(prev, Mapping) or prev.get("donor") != donor:
            continue
        prev_digest = prev.get("donor_digest")
        if prev_digest is None or prev_digest == digest:
            continue
        out.append(
            f"{station}: donor {donor} drifted under held group {group!r} — "
            f"the deployed record borrowed coefficients {prev_digest} "
            f"(donor fitted_at {prev.get('donor_fitted_at')}), this run "
            f"resolved {digest} (donor fitted_at {cur.get('donor_fitted_at')}). "
            f"The NEW values are used (donor holds are pointers); review the "
            f"borrow if {donor}'s re-estimation was not meant to reach "
            f"{station}."
        )
    return out


def resolve_stage_plan(
    plan: StagePlan,
    *,
    lookup_donor: "Callable[[str], Mapping[str, object]]",
    component: int,
) -> "tuple[Stage, ...]":
    """Turn a :class:`StagePlan` into ``gps_analysis.Stage`` objects.

    This is where the pointer semantics are actually paid for:
    :class:`DonorRef` is resolved by calling ``lookup_donor`` **now**, so the
    values used are the donor's current ones.  Re-estimating a donor therefore
    propagates to every station borrowing from it, instead of leaving stale
    numbers frozen in each borrower's record.  A donor with no record raises
    here rather than silently degrading.

    ``HeldExplicit`` is constructed without a covariance, so the borrowed
    group is treated as exactly known and the stage's result is flagged
    conditional — the honest reading, since the donor's uncertainty describes
    the DONOR's series, not this station's.

    Args:
        plan: The parsed plan.
        lookup_donor: Station code → that station's current record.
        component: Which component's coefficients to borrow.

    Returns:
        Stages ready for :func:`gps_analysis.estimate_staged`.
    """
    from gps_analysis.staged import HeldExplicit, HeldFromStage, Stage

    stages: list[Stage] = []
    for spec in plan.stages:
        held: dict[str, object] = {}
        for group, ref in spec.held.items():
            if isinstance(ref, StageRef):
                held[group] = HeldFromStage(ref.stage)
                continue
            record = lookup_donor(ref.station)
            values = donor_group_values(
                record, group, component=component, donor=ref.station
            )
            fitted_at = record.get("fitted_at")
            held[group] = HeldExplicit(
                values=values,
                # Provenance names the donor AND when its record was fitted,
                # so a stored borrow can be checked against a re-estimated
                # donor rather than merely asserted.
                source=f"donor:{ref.station}@{fitted_at}",
            )
        stages.append(
            Stage(
                name=spec.name,
                free=spec.free,
                held=held,  # type: ignore[arg-type]
                segments=spec.segments,
            )
        )
    return tuple(stages)


# ---------------------------------------------------------------------------
# Persistence: analysis.yaml
# ---------------------------------------------------------------------------

#: Where stage plans live in ``analysis.yaml``.
STAGE_PLAN_KEYS = ("detrend", "estimation", "stage_plans")


def default_analysis_yaml_path() -> "Path | None":
    """Resolve the deployed ``analysis.yaml`` via the shared gpsconfig lookup.

    Same mechanism as :func:`geo_dataread.detrend_estimate.default_fit_catalog_path`
    (``postprocess.cfg`` ``[FILES]`` first, then ``<gpsconfig dir>/``), so the
    stage-plan surface is deployed and overridden exactly like every other
    analysis-lane catalog.
    """
    from gps_parser import outlier_catalogs as oc

    return cast("Path | None", oc.catalog_path("analysis", "analysis.yaml"))


def read_stage_plans(path: "str | Path") -> dict[str, StagePlan]:
    """Read every station's stage plan from ``analysis.yaml``.

    Home is ``detrend.estimation.stage_plans`` — beside the existing
    per-station ``fit_windows`` and ``use_sta`` maps, which are the same kind
    of thing (an operator's per-station estimation override).  Deliberately
    NOT a ``fit_windows.csv`` cell: that reader is strict by design and its
    one-cell ``segments`` encoding exists because an extra ROW would parse
    into different-but-valid science.  A stage plan is structurally richer
    than a segment list, and nesting it into a CSV cell would trade that
    strictness for unreadability.

    A missing file, a missing block or an empty mapping all mean "no station
    has a stage plan" and return ``{}`` — a stage plan is an optional
    enhancement, exactly like the fit catalog.

    Raises:
        ValueError: if the block exists but is malformed, naming the station.
            A stage plan that silently fails to load would quietly revert a
            station to single-stage estimation and store different science.
    """
    from geo_dataread.analysis_yaml import read_station_block

    p = Path(path)
    node = read_station_block(p, STAGE_PLAN_KEYS, expected="stage list")
    out: dict[str, StagePlan] = {}
    for sta, entries in node.items():
        if not isinstance(entries, (list, tuple)):
            raise ValueError(
                f"{p}: stage_plans[{sta!r}] must be a list of stages, got {entries!r}"
            )
        try:
            out[str(sta)] = stage_plan_from_config(entries)
        except ValueError as exc:
            raise ValueError(f"{p}: stage_plans[{sta!r}]: {exc}") from None
    return out


def write_stage_plan(path: "str | Path", station: str, plan: StagePlan | None) -> None:
    """Merge ONE station's stage plan into ``analysis.yaml``, preserving the rest.

    Mirrors the workbench's ``--commit`` contract for ``detrend_params.json``:
    a merge-write of one station, never a rewrite of the document, so an
    operator curating one station cannot drop another's configuration.

    ``plan=None`` removes the station's entry (and the enclosing blocks if
    they become empty), which is how a staged station is returned to ordinary
    single-stage estimation without hand-editing YAML.
    """
    from geo_dataread.analysis_yaml import merge_write_station

    merge_write_station(
        path,
        STAGE_PLAN_KEYS,
        station,
        None if plan is None else stage_plan_to_config(plan),
    )
