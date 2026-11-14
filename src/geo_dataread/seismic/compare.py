"""Event matching between two earthquake catalogues (ticket T15).

``PLAN-seismicity.md`` §5 (comparison method), BGÓ's Q5 (the math lives here,
with the readers) and Q6 (tolerances 2 s / 10 km, then 5 s / 25 km,
configurable). Two layers:

* :func:`match_arrays` — the core, on plain arrays (times as int64 ns,
  ``datetime64`` or float seconds; latitude, longitude in degrees). No pandas.
* :func:`match_catalogs` — the frame wrapper for two schema events frames
  (:mod:`geo_dataread.seismic.schema`): an exact id join first, then the
  tolerance steps of the core, and a pairs frame, the unmatched rows of each
  side and a JSON-safe summary.

Steps, in order; an event matched in one step takes part in no later step
(one-to-one throughout):

0. **Id join** (:data:`ID_JOINS`), only where two sources share an id:

   ============  ==========================================  =================
   Join          Key                                         Source pairs
   ============  ==========================================  =================
   sil_path      ``sil_path`` = ``sil_path``                 sil-files ↔
                                                             skjalftalisa
                                                             (detail rows)
   sil_aut_path  ``sil_aut_path`` of one side = ``sil_path``  the same, for a
                 of the other (both directions)              reviewed file
                                                             event whose
                                                             automatic twin
                                                             the API holds
   sil_number    the SIL integer: ``SIL<n>`` (quakes-sil),   quakes-sil ↔
                 ``<n>`` (skjalftalisa)                      skjalftalisa
   ============  ==========================================  =================

   There is **no** shared id between sil-files and quakes-sil (the Quakes
   API list carries no ``sil_path``), nor between any SIL source and SeisComP
   (``IMO…`` ids): those pairs match by tolerance only. A legacy list frame
   has ``sil_path`` NA (only :func:`~geo_dataread.seismic.skjalftalisa_api.
   read_skjalftalisa_detail` fills it), so it joins sil-files by tolerance
   unless the caller filled ``sil_path``. A key value that occurs more than
   once among the still-unmatched rows of either side is *ambiguous*: those
   rows are not id-joined (counted in the summary) and fall through to the
   next step. Id pairs are **not** gated by any tolerance (reviewed/automatic
   twins can be 68 s apart in the legacy API, plan §9d); their Δt and
   distance are reported, and the summary counts the id pairs outside the
   loosest tolerance.

1. **Tolerance steps** (:data:`DEFAULT_TOLERANCES`): candidate pairs with
   ``|Δt| ≤ max_dt_s`` **and** epicentral distance ``≤ max_km`` — both bounds
   **inclusive**; Δt is compared exactly in integer nanoseconds against
   ``round(max_dt_s · 1e9)``. Each step is a one-to-one assignment on the
   cost ``(Δt / max_dt_s)² + (d / max_km)²``, normalised by that step's own
   tolerances, so a cost is in ``[0, 2]``.

Assignment: **global greedy, closest first** (the rule of §5). All candidate
pairs of a step are sorted by ``(cost, |Δt|, distance, position in A,
position in B)`` — a strict total order, so the result is deterministic and
does not depend on input order beyond the positions — and accepted in that
order when neither event is taken yet. Every accepted pair is then mutually
best among the pairs still open: in a swarm with events 1 s apart, the true
pairs (a few 0.1 s and a few hundred metres apart) are taken first and the
wrong neighbours (≈ 1 s off) find their events gone. Greedy does not
maximise the number of pairs or minimise the total cost; a Hungarian
assignment per connected component (``scipy.optimize.linear_sum_assignment``)
would, but a dense swarm makes one component of tens of thousands of events
(O(n³) on a dense matrix), and §5 asks for closest first. Not implemented.

Complexity: candidates come from one sort of B by time and two
``searchsorted`` over A — O((n_A + n_B) log n_B) — expanded without a Python
loop into K pairs; a latitude bound (``|Δφ| · 110.5 km ≤ max_km``, a lower
bound of the geodesic) prunes them before one vectorised geodesic
(``pyproj.Geod.inv`` on GRS80 via :func:`geofunc.frames.geodesic`, the same
as :func:`~geo_dataread.seismic.filters.distance_km`) over the survivors.
The greedy pass is O(K log K) for the sort plus one Python loop over K.
For 100k events per side spread over a month K is a few hundred thousand
and a step takes well under a second; ``max_candidates`` (default 50 M)
stops a window so dense that K would exhaust memory.

Rows with NaT time or NaN position are never candidates (they can still be
id-joined) and are counted.

Signs: ``dt_s``, ``ddepth_km``, ``dmag`` … are **B − A**; ``de_km`` /
``dn_km`` are the east and north parts of the vector from A's epicentre to
B's (from the geodesic azimuth at A). Magnitudes are compared raw and
labelled (Q7): ``dmag`` is ``magnitude`` B − A whatever the scales, read with
``mag_types`` (``"Mlw|ML_SIL"``); ``dmag_mlw`` / ``dmag_ml`` / ``dmag_auto``
compare one scale where both sides carry it.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from typing import Any, Final, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from geo_dataread.seismic.schema import SCHEMA_VERSION, ATTRS_KEY

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ObjArray = npt.NDArray[np.object_]

#: ``attrs`` key of the provenance record on every output frame.
MATCH_ATTRS_KEY: Final = "match"

#: Lower bound of one degree of latitude along a meridian (GRS80: 110.574 km
#: at the equator, more towards the poles), rounded down for the pre-filter.
_KM_PER_DEG_LAT_MIN: Final = 110.5

#: Default cap on the candidate pairs of one step.
DEFAULT_MAX_CANDIDATES: Final = 50_000_000

_NS_PER_S: Final = 1_000_000_000


class MatchError(ValueError):
    """Bad input to the matcher (shapes, tolerances, time index, id joins)."""


# --------------------------------------------------------------------------
# Tolerances
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Tolerance:
    """One tolerance step: ``|Δt| ≤ max_dt_s`` and ``distance ≤ max_km``
    (both inclusive). ``label`` names the step in the output; empty →
    ``"<dt>s/<km>km"``."""

    max_dt_s: float
    max_km: float
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("max_dt_s", "max_km"):
            v = getattr(self, name)
            if not (isinstance(v, int | float) and math.isfinite(v) and v > 0):
                raise MatchError(f"Tolerance.{name} must be finite and > 0, got {v!r}")
        if not self.label:
            object.__setattr__(self, "label", f"{self.max_dt_s:g}s/{self.max_km:g}km")

    @property
    def max_dt_ns(self) -> int:
        """The time bound in integer nanoseconds (rounded)."""
        return int(round(self.max_dt_s * _NS_PER_S))

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "max_dt_s": self.max_dt_s, "max_km": self.max_km}


#: Q6: 2 s / 10 km (``tolerance``), then 5 s / 25 km (``loose``), plan §5.
DEFAULT_TOLERANCES: Final = (
    Tolerance(2.0, 10.0, "tolerance"),
    Tolerance(5.0, 25.0, "loose"),
)

#: The time/space "exact" test of plan §5 step 1 (0.05 s / 0.5 km, observed to
#: hold for SIL files ↔ SIL API and back-imported SeisComP). Not a default;
#: prepend it to label those pairs apart: ``(EXACT_TOLERANCE,
#: *DEFAULT_TOLERANCES)``.
EXACT_TOLERANCE: Final = Tolerance(0.05, 0.5, "exact")

ToleranceLike: TypeAlias = Tolerance | tuple[float, float] | tuple[float, float, str]


def _tolerances(steps: Sequence[ToleranceLike]) -> tuple[Tolerance, ...]:
    out: list[Tolerance] = []
    for s in steps:
        if isinstance(s, Tolerance):
            out.append(s)
        elif isinstance(s, tuple) and len(s) in (2, 3):
            out.append(
                Tolerance(float(s[0]), float(s[1]), str(s[2]) if len(s) == 3 else "")
            )
        else:
            raise MatchError(f"tolerance {s!r}: a Tolerance or (dt_s, km[, label])")
    labels = [t.label for t in out]
    if len(set(labels)) != len(labels):
        raise MatchError(f"tolerance labels must be unique, got {labels}")
    return tuple(out)


# --------------------------------------------------------------------------
# Core on arrays
# --------------------------------------------------------------------------


def as_time_ns(values: npt.ArrayLike) -> tuple[IntArray, npt.NDArray[np.bool_]]:
    """Times → ``(int64 ns, valid mask)``.

    ``datetime64`` (any unit) and integer arrays (taken as ns) are exact;
    float arrays are seconds, rounded to the nearest ns (float64 holds an
    epoch time to about 0.2 µs). NaT / NaN are invalid (their slot is 0).
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise MatchError(f"times must be 1-D, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.datetime64):
        valid = ~np.isnat(arr)
        ns = arr.astype("datetime64[ns]").view(np.int64).copy()
    elif np.issubdtype(arr.dtype, np.integer):
        ns = arr.astype(np.int64, copy=True)
        valid = ns != np.iinfo(np.int64).min
    elif np.issubdtype(arr.dtype, np.floating):
        f = arr.astype(np.float64)
        valid = np.isfinite(f)
        ns = np.zeros(len(f), dtype=np.int64)
        ns[valid] = np.rint(f[valid] * _NS_PER_S).astype(np.int64)
    else:
        raise MatchError(f"times: datetime64, int64 ns or float s, got {arr.dtype}")
    ns[~valid] = 0
    return ns, np.asarray(valid, dtype=bool)


def _coords(values: npt.ArrayLike, n: int, what: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,):
        raise MatchError(f"{what}: shape {arr.shape}, expected ({n},)")
    return arr


@dataclasses.dataclass(frozen=True)
class ArrayMatch:
    """Result of :func:`match_arrays`. Pair ``k`` joins position ``ia[k]`` of
    A to ``ib[k]`` of B in step ``labels[step[k]]``. ``dt_s`` is B − A,
    ``dist_km`` the geodesic distance, ``azimuth_deg`` the forward azimuth at
    A, ``cost`` the normalised cost of that step. Pairs are ordered by step,
    then by A position."""

    ia: IntArray
    ib: IntArray
    step: IntArray
    dt_s: FloatArray
    dist_km: FloatArray
    azimuth_deg: FloatArray
    cost: FloatArray
    labels: tuple[str, ...]
    candidates: tuple[int, ...]
    invalid_a: int
    invalid_b: int

    def __len__(self) -> int:
        return int(len(self.ia))


def _geodesic(
    lon1: FloatArray, lat1: FloatArray, lon2: FloatArray, lat2: FloatArray, ellps: str
) -> tuple[FloatArray, FloatArray]:
    """(distance km, forward azimuth deg at point 1) for paired points."""
    if len(lon1) == 0:
        return np.zeros(0), np.zeros(0)
    from geofunc.frames import geodesic

    az, _, dist_m = geodesic(ellps).inv(lon1, lat1, lon2, lat2)
    return (
        np.asarray(dist_m, dtype=np.float64) / 1000.0,
        np.asarray(az, dtype=np.float64),
    )


def _candidates(
    ta: IntArray,
    tb_sorted: IntArray,
    order_b: IntArray,
    a_pos: IntArray,
    max_dt_ns: int,
    max_candidates: int,
) -> tuple[IntArray, IntArray]:
    """All (A position, B position) with |t_b − t_a| ≤ max_dt_ns.

    ``ta`` are the times of the A rows at ``a_pos``; ``tb_sorted`` the
    sorted times of the open B rows, ``order_b`` their positions.
    """
    lo = np.searchsorted(tb_sorted, ta - max_dt_ns, side="left")
    hi = np.searchsorted(tb_sorted, ta + max_dt_ns, side="right")
    counts = (hi - lo).astype(np.int64)
    total = int(counts.sum())
    if total > max_candidates:
        raise MatchError(
            f"{total} candidate pairs exceed max_candidates={max_candidates}: "
            "match a shorter window or a smaller time tolerance"
        )
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    rows = np.repeat(np.arange(len(ta), dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    offset = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    jb = np.repeat(lo.astype(np.int64), counts) + offset
    return a_pos[rows], order_b[jb]


def _greedy(
    ia: IntArray,
    ib: IntArray,
    cost: FloatArray,
    adt: IntArray,
    dist: FloatArray,
) -> npt.NDArray[np.bool_]:
    """Global greedy one-to-one selection; returns the accepted mask.

    Order: cost, |Δt| (ns), distance, A position, B position (lexsort keys
    are given last-first)."""
    order = np.lexsort((ib, ia, dist, adt, cost))
    used_a: set[int] = set()
    used_b: set[int] = set()
    keep = np.zeros(len(ia), dtype=bool)
    a_list = ia[order].tolist()
    b_list = ib[order].tolist()
    for k, (a, b) in zip(order.tolist(), zip(a_list, b_list, strict=True), strict=True):
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        keep[k] = True
    return keep


def match_arrays(
    time_a: npt.ArrayLike,
    lat_a: npt.ArrayLike,
    lon_a: npt.ArrayLike,
    time_b: npt.ArrayLike,
    lat_b: npt.ArrayLike,
    lon_b: npt.ArrayLike,
    *,
    tolerances: Sequence[ToleranceLike] = DEFAULT_TOLERANCES,
    exclude_a: npt.ArrayLike | None = None,
    exclude_b: npt.ArrayLike | None = None,
    ellps: str = "GRS80",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> ArrayMatch:
    """Match two catalogues on time and epicentre, step by step.

    Times as ``datetime64``, int64 ns or float seconds (both sides the same
    kind); positions in degrees. ``exclude_a`` / ``exclude_b`` (positions or
    boolean masks) are rows already matched elsewhere (the id join): they
    take part in no step. Each step pairs only rows left open by the steps
    before it; see the module docstring for the bounds (inclusive), the cost
    and the greedy assignment with its tie order.
    """
    steps = _tolerances(tolerances)
    ta, va = as_time_ns(time_a)
    tb, vb = as_time_ns(time_b)
    na, nb = len(ta), len(tb)
    la, lo_a = _coords(lat_a, na, "lat_a"), _coords(lon_a, na, "lon_a")
    lb, lo_b = _coords(lat_b, nb, "lat_b"), _coords(lon_b, nb, "lon_b")
    va = va & np.isfinite(la) & np.isfinite(lo_a)
    vb = vb & np.isfinite(lb) & np.isfinite(lo_b)
    invalid_a, invalid_b = int((~va).sum()), int((~vb).sum())
    open_a, open_b = va.copy(), vb.copy()
    for excl, open_, n, what in (
        (exclude_a, open_a, na, "exclude_a"),
        (exclude_b, open_b, nb, "exclude_b"),
    ):
        if excl is None:
            continue
        e = np.asarray(excl)
        if e.dtype == bool:
            if e.shape != (n,):
                raise MatchError(f"{what}: mask shape {e.shape}, expected ({n},)")
            open_[e] = False
        elif len(e):
            open_[e.astype(np.int64)] = False

    parts: list[tuple[IntArray, ...]] = []
    fparts: list[tuple[FloatArray, ...]] = []
    n_cand: list[int] = []
    for k, tol in enumerate(steps):
        a_pos = np.flatnonzero(open_a).astype(np.int64)
        b_pos = np.flatnonzero(open_b).astype(np.int64)
        order = np.argsort(tb[b_pos], kind="stable")
        order_b = b_pos[order]
        ca, cb = _candidates(
            ta[a_pos], tb[order_b], order_b, a_pos, tol.max_dt_ns, max_candidates
        )
        if len(ca):
            near = np.abs(lb[cb] - la[ca]) * _KM_PER_DEG_LAT_MIN <= tol.max_km
            ca, cb = ca[near], cb[near]
        dist, az = _geodesic(lo_a[ca], la[ca], lo_b[cb], lb[cb], ellps)
        ok = dist <= tol.max_km
        ca, cb, dist, az = ca[ok], cb[ok], dist[ok], az[ok]
        n_cand.append(int(len(ca)))
        dt_ns = tb[cb] - ta[ca]
        adt = np.abs(dt_ns)
        cost = (adt / tol.max_dt_ns) ** 2 + (dist / tol.max_km) ** 2
        keep = _greedy(ca, cb, cost, adt, dist)
        ca, cb = ca[keep], cb[keep]
        order_a = np.argsort(ca, kind="stable")
        ca, cb = ca[order_a], cb[order_a]
        sel = np.flatnonzero(keep)[order_a]
        parts.append((ca, cb, np.full(len(ca), k, dtype=np.int64)))
        fparts.append(
            (dt_ns[sel].astype(np.float64) / _NS_PER_S, dist[sel], az[sel], cost[sel])
        )
        open_a[ca] = False
        open_b[cb] = False

    def cat_i(i: int) -> IntArray:
        return (
            np.concatenate([p[i] for p in parts])
            if parts
            else np.zeros(0, dtype=np.int64)
        )

    def cat_f(i: int) -> FloatArray:
        return np.concatenate([p[i] for p in fparts]) if fparts else np.zeros(0)

    return ArrayMatch(
        ia=cat_i(0),
        ib=cat_i(1),
        step=cat_i(2),
        dt_s=cat_f(0),
        dist_km=cat_f(1),
        azimuth_deg=cat_f(2),
        cost=cat_f(3),
        labels=tuple(t.label for t in steps),
        candidates=tuple(n_cand),
        invalid_a=invalid_a,
        invalid_b=invalid_b,
    )


# --------------------------------------------------------------------------
# Id normalisation
# --------------------------------------------------------------------------

_SIL_NUMBER = re.compile(r"^(?:SIL)?0*(\d+)$", re.IGNORECASE)


def _missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and math.isnan(value)


def sil_number(value: object) -> str | None:
    """The SIL integer event id as a canonical string, or ``None``.

    ``"SIL1352381"`` (Quakes API, ``system=sil``), ``"1352381"`` and
    ``1352381`` (legacy Skjálftalísa) all give ``"1352381"`` (leading zeros
    dropped). SeisComP ``IMO…`` ids, SIL file path ids, blanks and NA give
    ``None``.
    """
    if _missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, int | np.integer):
        return str(int(value)) if int(value) >= 0 else None
    m = _SIL_NUMBER.match(str(value).strip())
    if m is None:
        return None
    return str(int(m.group(1)))


def sil_path_key(value: object) -> str | None:
    """A SIL file id (``/2024/aug/22/00:05:00/10:05:430``) as a join key.

    Surrounding blanks are stripped; the id is otherwise kept verbatim (ids
    are opaque labels, plan §9b: a reviewed and an automatic id can differ by
    1 ms, so no rounding). NA, blanks and ids that do not start with ``/``
    give ``None``.
    """
    if _missing(value):
        return None
    text = str(value).strip()
    return text if text.startswith("/") else None


def _column_keys(
    df: pd.DataFrame, column: str, norm: Callable[[object], str | None]
) -> ObjArray:
    if column not in df.columns:
        return np.full(len(df), None, dtype=object)
    return np.array([norm(v) for v in df[column].to_numpy(dtype=object)], dtype=object)


def key_sil_path(df: pd.DataFrame) -> ObjArray:
    """Per-row ``sil_path`` keys (``None`` where absent)."""
    return _column_keys(df, "sil_path", sil_path_key)


def key_sil_aut_path(df: pd.DataFrame) -> ObjArray:
    """Per-row ``sil_aut_path`` keys (``None`` where absent)."""
    return _column_keys(df, "sil_aut_path", sil_path_key)


#: Sources whose ``event_id`` is the SIL integer (``SIL<n>`` / ``<n>``).
SIL_NUMBER_SOURCES: Final = ("quakes-sil", "skjalftalisa")


def key_sil_number(df: pd.DataFrame) -> ObjArray:
    """Per-row SIL integer keys from ``event_id``, for rows whose ``source``
    is in :data:`SIL_NUMBER_SOURCES` (a frame without ``source`` is taken
    as such a source); ``None`` elsewhere."""
    keys = _column_keys(df, "event_id", sil_number)
    if "source" in df.columns:
        src = df["source"].astype(object).to_numpy()
        keys[~np.isin(src, SIL_NUMBER_SOURCES)] = None
    return keys


KeyFunc: TypeAlias = Callable[[pd.DataFrame], ObjArray]


@dataclasses.dataclass(frozen=True)
class IdJoin:
    """One exact join: rows whose ``key_a(A)`` equals ``key_b(B)`` (both not
    ``None`` and unique among the open rows of their side)."""

    label: str
    key_a: KeyFunc
    key_b: KeyFunc


#: The id joins in precedence order (``id_join="auto"``). ``sil_aut_path``
#: runs in both directions under one label.
ID_JOINS: Final = (
    IdJoin("sil_path", key_sil_path, key_sil_path),
    IdJoin("sil_aut_path", key_sil_aut_path, key_sil_path),
    IdJoin("sil_aut_path", key_sil_path, key_sil_aut_path),
    IdJoin("sil_number", key_sil_number, key_sil_number),
)

IdJoinSpec: TypeAlias = Literal["auto", "none"] | None | Sequence[str | IdJoin]


def _id_joins(spec: IdJoinSpec) -> tuple[IdJoin, ...]:
    if spec is None or spec == "none":
        return ()
    if spec == "auto":
        return ID_JOINS
    if isinstance(spec, str):
        spec = (spec,)
    out: list[IdJoin] = []
    for item in spec:
        if isinstance(item, IdJoin):
            out.append(item)
            continue
        hits = [j for j in ID_JOINS if j.label == item]
        if not hits:
            names = sorted({j.label for j in ID_JOINS})
            raise MatchError(f"id join {item!r} is not one of {names} (or an IdJoin)")
        out.extend(hits)
    return tuple(out)


def _join_keys(
    ka: ObjArray,
    kb: ObjArray,
    open_a: npt.NDArray[np.bool_],
    open_b: npt.NDArray[np.bool_],
) -> tuple[IntArray, IntArray, int]:
    """Exact one-to-one pairs on equal keys among open rows; returns
    ``(ia, ib, n_ambiguous_rows)``. A key repeated on either side joins
    nothing; its rows (both sides) are counted as ambiguous."""
    a_idx = [i for i in np.flatnonzero(open_a).tolist() if ka[i] is not None]
    b_idx = [i for i in np.flatnonzero(open_b).tolist() if kb[i] is not None]
    ca = Counter(ka[i] for i in a_idx)
    cb = Counter(kb[i] for i in b_idx)
    common = set(ca) & set(cb)
    dup = {k for k in common if ca[k] > 1 or cb[k] > 1}
    ambiguous = sum(ca[k] + cb[k] for k in dup)
    b_of = {kb[i]: i for i in b_idx if kb[i] in common and kb[i] not in dup}
    pairs = [(i, b_of[ka[i]]) for i in a_idx if ka[i] in b_of]
    ia = np.array([p[0] for p in pairs], dtype=np.int64)
    ib = np.array([p[1] for p in pairs], dtype=np.int64)
    return ia, ib, ambiguous


# --------------------------------------------------------------------------
# Frame wrapper
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CatalogMatch:
    """Result of :func:`match_catalogs`.

    ``pairs``: one row per matched pair (see :func:`match_catalogs`);
    ``unmatched_a`` / ``unmatched_b``: the unmatched rows of each input, as
    ``iloc`` subsets (same index, columns and ``attrs["catalog"]``) plus the
    column ``pos`` (the row's position in its input); ``summary``: counts per
    step and the settings, JSON-safe. Every frame carries the provenance on
    ``attrs["match"]``.
    """

    pairs: pd.DataFrame
    unmatched_a: pd.DataFrame
    unmatched_b: pd.DataFrame
    summary: dict[str, Any]


def _time_index(df: pd.DataFrame, side: str) -> IntArray:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise MatchError(f"catalogue {side}: the index must be a DatetimeIndex (time)")
    if idx.tz is None:
        raise MatchError(f"catalogue {side}: naive time index (catalogues are UTC)")
    ns = idx.tz_convert("UTC").as_unit("ns")
    vals = np.asarray(ns.asi8, dtype=np.int64)
    return vals


def _float_col(df: pd.DataFrame, column: str) -> FloatArray:
    if column not in df.columns:
        return np.full(len(df), np.nan)
    return np.asarray(
        pd.to_numeric(df[column], errors="coerce").astype("float64"), dtype=np.float64
    )


def _obj_col(df: pd.DataFrame, column: str) -> ObjArray:
    if column not in df.columns:
        return np.full(len(df), None, dtype=object)
    vals = df[column].astype(object).to_numpy()
    return np.array([None if _missing(v) else v for v in vals], dtype=object)


def _source_of(df: pd.DataFrame) -> str | None:
    if "source" in df.columns and len(df):
        vals = sorted({str(v) for v in df["source"].dropna().astype(str)})
        if vals:
            return ",".join(vals)
    cat = df.attrs.get(ATTRS_KEY)
    if isinstance(cat, Mapping) and cat.get("source"):
        return str(cat["source"])
    return None


def _catalog_ref(df: pd.DataFrame) -> dict[str, Any]:
    cat = df.attrs.get(ATTRS_KEY)
    ref: dict[str, Any] = {"source": _source_of(df), "n_events": int(len(df))}
    if isinstance(cat, Mapping):
        for key in ("location", "window", "filters", "fetched_at", "schema_version"):
            if key in cat:
                ref[key] = cat[key]
    return ref


def _version() -> str:
    try:
        return metadata.version("geo_dataread")
    except metadata.PackageNotFoundError:
        return "unknown"


_MAG_SCALES: Final = ("mag_mlw", "mag_ml", "mag_auto")


def _pairs_frame(
    a: pd.DataFrame,
    b: pd.DataFrame,
    ia: IntArray,
    ib: IntArray,
    step: list[str],
    ta: IntArray,
    tb: IntArray,
    dist: FloatArray,
    az: FloatArray,
    cost: FloatArray,
    step_order: list[str],
) -> pd.DataFrame:
    ta_ns, tb_ns = ta[ia], tb[ib]
    ok_t = (ta_ns != np.iinfo(np.int64).min) & (tb_ns != np.iinfo(np.int64).min)
    dt = np.where(ok_t, (tb_ns - ta_ns).astype(np.float64) / _NS_PER_S, np.nan)
    rad = np.radians(az)
    ev_a, ev_b = _obj_col(a, "event_id"), _obj_col(b, "event_id")
    mt_a, mt_b = _obj_col(a, "magnitude_type"), _obj_col(b, "magnitude_type")
    mta, mtb = mt_a[ia], mt_b[ib]
    out: dict[str, Any] = {
        "pos_a": ia,
        "pos_b": ib,
        "time_a": pd.DatetimeIndex(a.index[ia]).tz_convert("UTC"),
        "time_b": pd.DatetimeIndex(b.index[ib]).tz_convert("UTC"),
        "event_id_a": pd.array(ev_a[ia], dtype="string"),
        "event_id_b": pd.array(ev_b[ib], dtype="string"),
        "step": pd.Categorical(step, categories=step_order),
        "dt_s": dt,
        "dist_km": dist,
        "de_km": dist * np.sin(rad),
        "dn_km": dist * np.cos(rad),
        "cost": cost,
        "depth_a_km": _float_col(a, "depth_km")[ia],
        "depth_b_km": _float_col(b, "depth_km")[ib],
    }
    out["ddepth_km"] = out["depth_b_km"] - out["depth_a_km"]
    mag_a, mag_b = _float_col(a, "magnitude")[ia], _float_col(b, "magnitude")[ib]
    out["magnitude_a"] = mag_a
    out["magnitude_b"] = mag_b
    out["magnitude_type_a"] = pd.array(mta, dtype="string")
    out["magnitude_type_b"] = pd.array(mtb, dtype="string")
    out["mag_types"] = pd.array(
        [
            None if x is None or y is None else f"{x}|{y}"
            for x, y in zip(mta.tolist(), mtb.tolist(), strict=True)
        ],
        dtype="string",
    )
    out["dmag"] = mag_b - mag_a
    for scale in _MAG_SCALES:
        out[f"d{scale}"] = _float_col(b, scale)[ib] - _float_col(a, scale)[ia]
    for col in ("evaluation_mode", "source"):
        out[f"{col}_a"] = pd.array(_obj_col(a, col)[ia], dtype="string")
        out[f"{col}_b"] = pd.array(_obj_col(b, col)[ib], dtype="string")
    df = pd.DataFrame(out)
    if len(df):
        df = df.sort_values(["pos_a", "pos_b"], kind="stable").reset_index(drop=True)
    return df


def _q(values: FloatArray, q: float) -> float | None:
    v = values[np.isfinite(values)]
    return float(np.quantile(v, q)) if len(v) else None


def match_catalogs(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    tolerances: Sequence[ToleranceLike] = DEFAULT_TOLERANCES,
    id_join: IdJoinSpec = "auto",
    ellps: str = "GRS80",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> CatalogMatch:
    """Match the events of catalogue ``a`` to those of ``b`` one-to-one.

    ``a`` / ``b``: events frames with a tz-aware ``DatetimeIndex`` and
    ``latitude`` / ``longitude`` (schema frames; the other schema columns
    are used where present). ``tolerances``: the tolerance steps (Q6
    defaults; tuples ``(dt_s, km[, label])`` are accepted, e.g.
    ``(EXACT_TOLERANCE, *DEFAULT_TOLERANCES)``). ``id_join``: ``"auto"``
    (every join of :data:`ID_JOINS`, in that order; a join neither frame has
    keys for does nothing), ``"none"``/``None``, or a sequence of join labels
    (``"sil_path"``, ``"sil_aut_path"``, ``"sil_number"``) and/or
    :class:`IdJoin` objects, in precedence order.

    ``pairs`` columns: ``pos_a``, ``pos_b`` (positions in the inputs; the
    time index is not unique), ``time_a``, ``time_b``, ``event_id_a``,
    ``event_id_b``, ``step`` (category: ``id:<join>`` then the tolerance
    labels), ``dt_s``, ``dist_km``, ``de_km``, ``dn_km``, ``cost`` (NaN for id
    pairs), ``depth_a_km``, ``depth_b_km``, ``ddepth_km``, ``magnitude_a``,
    ``magnitude_b``, ``magnitude_type_a``, ``magnitude_type_b``,
    ``mag_types`` (``"<type a>|<type b>"``), ``dmag``, ``dmag_mlw``,
    ``dmag_ml``, ``dmag_auto``, ``evaluation_mode_a/_b``, ``source_a/_b``;
    differences are B − A; sorted by ``pos_a``.

    ``summary``: ``n_a``, ``n_b``, ``matched`` (per step label, every label
    present), ``matched_total``, ``unmatched_a``, ``unmatched_b``,
    ``candidates`` (per tolerance step), ``id_ambiguous`` (rows per id join),
    ``id_outside_tolerance`` (id pairs beyond the loosest tolerance, or with
    a missing time/position), ``invalid_a`` / ``invalid_b`` (NaT time or NaN
    position), ``mag_types`` (pairs per type pair), ``dt_s`` / ``dist_km``
    (median and 95th percentile of |Δt| and distance per step), and the
    settings. Plain ints/floats/strings only.
    """
    steps = _tolerances(tolerances)
    joins = _id_joins(id_join)
    ta, tb = _time_index(a, "a"), _time_index(b, "b")
    na, nb = len(a), len(b)
    lat_a, lon_a = _float_col(a, "latitude"), _float_col(a, "longitude")
    lat_b, lon_b = _float_col(b, "latitude"), _float_col(b, "longitude")
    for side, df in (("a", a), ("b", b)):
        missing = {"latitude", "longitude"} - set(df.columns)
        if missing:
            raise MatchError(f"catalogue {side}: no column(s) {sorted(missing)}")

    open_a, open_b = np.ones(na, dtype=bool), np.ones(nb, dtype=bool)
    id_ia: list[IntArray] = []
    id_ib: list[IntArray] = []
    id_lab: list[str] = []
    ambiguous: dict[str, int] = {}
    for join in joins:
        ia, ib, amb = _join_keys(join.key_a(a), join.key_b(b), open_a, open_b)
        label = f"id:{join.label}"
        ambiguous[label] = ambiguous.get(label, 0) + amb
        open_a[ia] = False
        open_b[ib] = False
        id_ia.append(ia)
        id_ib.append(ib)
        id_lab.extend([label] * len(ia))
    ia_id = np.concatenate(id_ia) if id_ia else np.zeros(0, dtype=np.int64)
    ib_id = np.concatenate(id_ib) if id_ib else np.zeros(0, dtype=np.int64)

    # NaT in a tz-aware index is int64 min, which the core treats as invalid.
    core = match_arrays(
        ta,
        lat_a,
        lon_a,
        tb,
        lat_b,
        lon_b,
        tolerances=steps,
        exclude_a=~open_a,
        exclude_b=~open_b,
        ellps=ellps,
        max_candidates=max_candidates,
    )

    dist_id, az_id = _geodesic(
        lon_a[ia_id], lat_a[ia_id], lon_b[ib_id], lat_b[ib_id], ellps
    )
    ia_all = np.concatenate([ia_id, core.ia])
    ib_all = np.concatenate([ib_id, core.ib])
    step_all = id_lab + [core.labels[k] for k in core.step.tolist()]
    step_order = list(dict.fromkeys([f"id:{j.label}" for j in joins])) + list(
        core.labels
    )
    pairs = _pairs_frame(
        a,
        b,
        ia_all,
        ib_all,
        step_all,
        ta,
        tb,
        np.concatenate([dist_id, core.dist_km]),
        np.concatenate([az_id, core.azimuth_deg]),
        np.concatenate([np.full(len(ia_id), np.nan), core.cost]),
        step_order,
    )

    matched_a = np.zeros(na, dtype=bool)
    matched_b = np.zeros(nb, dtype=bool)
    matched_a[ia_all] = True
    matched_b[ib_all] = True
    um_a = a.iloc[np.flatnonzero(~matched_a)].copy()
    um_a["pos"] = np.flatnonzero(~matched_a).astype(np.int64)
    um_b = b.iloc[np.flatnonzero(~matched_b)].copy()
    um_b["pos"] = np.flatnonzero(~matched_b).astype(np.int64)

    loosest = max(steps, key=lambda t: (t.max_dt_ns, t.max_km)) if steps else None
    id_rows = pairs["step"].astype(str).str.startswith("id:").to_numpy()
    if loosest is not None and id_rows.any():
        dt_id = np.abs(pairs["dt_s"].to_numpy()[id_rows])
        d_id = pairs["dist_km"].to_numpy()[id_rows]
        with np.errstate(invalid="ignore"):
            inside = (np.rint(dt_id * _NS_PER_S) <= loosest.max_dt_ns) & (
                d_id <= loosest.max_km
            )
        outside = int((~inside).sum())
    else:
        outside = 0

    counts = Counter(pairs["step"].astype(str).tolist())
    step_stats: dict[str, dict[str, Any]] = {}
    adt_all = np.abs(pairs["dt_s"].to_numpy(dtype=np.float64))
    dist_all = pairs["dist_km"].to_numpy(dtype=np.float64)
    labels_all = pairs["step"].astype(str).to_numpy()
    for label in step_order:
        sel = labels_all == label
        step_stats[label] = {
            "abs_dt_s_median": _q(adt_all[sel], 0.5),
            "abs_dt_s_p95": _q(adt_all[sel], 0.95),
            "dist_km_median": _q(dist_all[sel], 0.5),
            "dist_km_p95": _q(dist_all[sel], 0.95),
        }
    mag_types = Counter(str(v) for v in pairs["mag_types"].dropna().tolist())
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ticket": "T15",
        "a": _catalog_ref(a),
        "b": _catalog_ref(b),
        "tolerances": [t.as_dict() for t in steps],
        "bounds": "inclusive: |dt| <= max_dt_s (int ns) and dist <= max_km",
        "cost": "(|dt|/max_dt_s)^2 + (dist/max_km)^2 per step",
        "assignment": "global greedy, ties by (cost, |dt|, dist, pos_a, pos_b)",
        "id_joins": [f"id:{j.label}" for j in joins],
        "ellps": ellps,
        "signs": "differences are B - A",
        "software": {"geo_dataread": _version()},
    }
    summary: dict[str, Any] = {
        "source_a": _source_of(a),
        "source_b": _source_of(b),
        "n_a": int(na),
        "n_b": int(nb),
        "matched": {label: int(counts.get(label, 0)) for label in step_order},
        "matched_total": int(len(pairs)),
        "unmatched_a": int(len(um_a)),
        "unmatched_b": int(len(um_b)),
        "candidates": dict(zip(core.labels, core.candidates, strict=True)),
        "id_ambiguous": {k: int(v) for k, v in ambiguous.items()},
        "id_outside_tolerance": outside,
        "invalid_a": core.invalid_a,
        "invalid_b": core.invalid_b,
        "mag_types": dict(sorted(mag_types.items())),
        "steps": step_stats,
        "settings": {
            "tolerances": provenance["tolerances"],
            "id_joins": provenance["id_joins"],
            "ellps": ellps,
        },
    }
    for frame in (pairs, um_a, um_b):
        frame.attrs[MATCH_ATTRS_KEY] = provenance
    return CatalogMatch(
        pairs=pairs, unmatched_a=um_a, unmatched_b=um_b, summary=summary
    )


__all__ = [
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_TOLERANCES",
    "EXACT_TOLERANCE",
    "ID_JOINS",
    "MATCH_ATTRS_KEY",
    "SIL_NUMBER_SOURCES",
    "ArrayMatch",
    "CatalogMatch",
    "IdJoin",
    "MatchError",
    "Tolerance",
    "as_time_ns",
    "key_sil_aut_path",
    "key_sil_number",
    "key_sil_path",
    "match_arrays",
    "match_catalogs",
    "sil_number",
    "sil_path_key",
]
