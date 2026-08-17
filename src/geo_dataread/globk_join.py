"""GLOBK ``multibase`` segment reading and datum-consistent joining.

Replaces the retired ``timesfunc.compGlobkTimes`` (naive byte-level concat)
+ ``fixGlobkoffset`` (hard-coded per-station ``offset=10`` patch) pair that
builds ``mb_STA_TOT.dat[123]`` on okada.vedur.is (``tododaily`` →
``/usr/local/bin/compGLOBK -f TOT -o``).

File format (GAMIT/GLOBK ``multibase`` output, e.g. ``mb_SENG_GPS.dat2``)::

    Globk Analysis GGVer 10.71.021 Wed Sep 28 13:04:52 EDT 2022
    SENG_GPS to E Solution  1 +  16542671.519 m
    <blank line>
     2015.48356      1.51913  0.00156
     ...

Segments are ad-hoc ``<n>PS`` / ``GPS`` pieces (``1PS``..``9PS`` + ``GPS``),
NOT calendar-year slices: most stations have ``GPS`` only, but long records
carry several overlapping pieces out of suffix order (REYK: ``1PS``..``6PS``
+ ``GPS``, 1995–2026, all overlapping). ``GPS`` is simply one piece (often
the earliest), not an assembly.

Data rows are ``epoch  value  sigma``: epoch in fractional years, value and
sigma in metres. The value column sits on a *run-specific datum*: each
``multibase`` run may place the same physical position at a value shifted by
an integer multiple of the GLOBK wrap quantum q = 10 m. Joining segments
from different runs (Pre = prior years, Rap = current year) therefore shows
spurious ±10 m steps at segment boundaries unless each segment is
rebaselined onto a common datum.

Empirical spec (okada.vedur.is recon, 2026-07-17/18 — see the derivation
below before changing anything):

* The header reference constant (``+ 16542671.519 m``) does **not** encode
  the run datum. Verified: THOB East Pre/Rap references differ by 1.884 m
  while the value column is continuous across the boundary to mm level;
  SENG East references differ by 2.621 m while the actual datum shift is
  exactly 10.000 m. The reference line is parsed for station/component
  identification and provenance only — never applied to the values.
* Boundary datum shifts are integer multiples of q = 10 m (SENG East
  2025.99863 → 2026.00137: raw step 10.0014 m = 1·q + 1.4 mm real motion).
* Overlapping segments share epochs; the production concat keeps every
  duplicate (REYK TOT: 2326 dup epochs / 8459 rows, both solutions present),
  so a de-wrapped join must **dedup**. Dedup rule = **minimum σ**: at a
  duplicated epoch keep the solution with the smallest formal uncertainty
  (3rd column). A masked/invalid σ (``********``/``************`` → NaN) is
  treated as +∞ so it never wins — where one overlapping segment has a real
  solution and the other is masked, min-σ auto-keeps the real one. Ties
  (equal σ, including both masked) resolve keep-first in join/accumulation
  order (a deterministic, negligible tie-break — the affected epochs are the
  overlapping-solution epochs, not steps).
* Within-segment gross blunders (single-epoch spikes: REYK East shows 2.76 m,
  165 m, 2973 m up-then-immediately-back-down outliers) are NOT steps or
  wraps — the join must pass them through untouched; the downstream cleaning
  stage removes them.

Derivation chain (top-down):

1. :func:`read_mb_segment` parses one segment file into epoch/value/sigma
   arrays (stable-sorted by epoch; duplicates kept — global dedup is deferred
   to the join so min-σ can see every overlapping segment's solution).
2. :func:`estimate_segment_offset` measures the raw datum offset Δ̂ of a
   segment against the min-σ-deduped accumulated series (median over
   overlapping epochs, else the boundary pair when a gap leaves no overlap).
3. :func:`wrap_correction` snaps Δ̂ to the nearest multiple of q:
   c = q·round(Δ̂/q).
4. :func:`dedup_min_sigma` collapses duplicate epochs by minimum σ.
5. :func:`join_segments` orders segments chronologically, anchors the datum
   on the earliest, pools v′ = v − c for every segment, guards the residual
   |Δ̂ − c| ≤ r_max (loud — a trip is a finding, not a threshold to relax),
   then returns the min-σ-deduped, epoch-sorted series on the anchor datum.

All arithmetic is float64; values and corrections are metres, epochs are
fractional years.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GlobkJoinError",
    "MbHeader",
    "MbSegment",
    "SegmentCorrection",
    "JoinedSeries",
    "read_mb_segment",
    "estimate_segment_offset",
    "wrap_correction",
    "select_min_sigma_indices",
    "dedup_min_sigma",
    "join_segments",
    "discover_segments",
    "join_station_component",
    "format_tot_file",
    "write_joined_series",
]

#: GLOBK wrap quantum in metres — boundary datum shifts are multiples of this.
WRAP_QUANTUM_M: float = 10.0

#: Default guard on the post-correction residual |Δ̂ − c| in metres.
#: Real inter-epoch motion (even across multi-month gaps in rifting
#: episodes) is ≪ q/2 = 5 m; 1 m leaves margin while catching datum errors.
MAX_RESIDUAL_M: float = 1.0

# "SENG_GPS to E Solution  1 +  16542671.519 m"
_HEADER_RE = re.compile(
    r"^(?P<station>\w{4})_(?P<marker>\w+)\s+to\s+(?P<component>[A-Z])"
    r"\s+Solution\s+(?P<solution>\d+)\s*\+\s*(?P<reference>-?\d+(?:\.\d+)?)\s*m$"
)

#: Epochs are written with 5 decimals (≈ 5 min resolution); this key format
#: makes epoch matching exact for identically-formatted files.
_EPOCH_KEY_FMT = "{0:.5f}"


class GlobkJoinError(ValueError):
    """Raised when GLOBK segments cannot be parsed or safely joined."""


@dataclass(frozen=True)
class MbHeader:
    """Parsed ``multibase`` segment header (identification + provenance only).

    The ``reference_m`` constant is *not* the run datum (see module
    docstring) — it is retained for provenance and sanity checks only.
    """

    station: str  #: 4-char station id, e.g. ``"SENG"``.
    marker: str  #: segment label after the underscore, e.g. ``"GPS"``, ``"1PS"``.
    component: str  #: coordinate component letter: ``"N"``, ``"E"`` or ``"U"``.
    solution: int  #: GLOBK solution number from the header line.
    reference_m: float  #: header reference constant [m] (informational).
    provenance: str  #: first header line (Globk Analysis version/date).


@dataclass(frozen=True)
class MbSegment:
    """One ``multibase`` segment time series on its own run datum.

    ``epochs`` [fractional year] are non-decreasing (stable-sorted; duplicate
    epochs are kept — the join dedups globally by min-σ); ``values`` and
    ``sigmas`` are metres. Masked/invalid fields (``********`` /
    ``************`` field overflow) are NaN in both ``values`` and ``sigmas``.
    """

    header: MbHeader
    path: Path
    epochs: NDArray[np.float64]
    values: NDArray[np.float64]
    sigmas: NDArray[np.float64]


@dataclass(frozen=True)
class SegmentCorrection:
    """Datum bookkeeping for one joined segment.

    ``correction_m`` is the amount *subtracted* from the segment's values:
    v′ = v − c with c = q·round(Δ̂/q); ``raw_offset_m`` is Δ̂ and
    ``residual_m`` = Δ̂ − c (real motion + noise across the boundary).
    ``overlap_available`` is False when a gap left no shared epoch and Δ̂ came
    from the boundary pair (mixes wrap + real motion + any step).
    """

    path: Path
    n_overlap: int  #: epochs shared with the accumulated series.
    overlap_available: bool  #: True if Δ̂ used overlap, False if boundary pair.
    raw_offset_m: float
    correction_m: float
    residual_m: float
    n_rows: int  #: data rows this segment contributed to the pool.


@dataclass(frozen=True)
class JoinedSeries:
    """Segments joined onto the datum of the earliest segment.

    ``epochs`` [fractional year] strictly increasing (min-σ-deduped);
    ``values``/``sigmas`` in metres on the anchor segment's datum. ``header``
    is the anchor segment's header; ``corrections`` records the per-segment
    datum arithmetic in join order (anchor first, correction 0 by
    construction).
    """

    station: str
    component: str
    epochs: NDArray[np.float64]
    values: NDArray[np.float64]
    sigmas: NDArray[np.float64]
    header: MbHeader
    corrections: tuple[SegmentCorrection, ...]


def _parse_header(lines: list[str], path: Path) -> MbHeader:
    """Parse the two-line ``multibase`` header into :class:`MbHeader`."""
    if len(lines) < 2:
        raise GlobkJoinError(f"{path}: file too short for a multibase header")
    match = _HEADER_RE.match(lines[1].strip())
    if match is None:
        raise GlobkJoinError(
            f"{path}: unrecognized multibase reference line: {lines[1].strip()!r}"
        )
    return MbHeader(
        station=match["station"],
        marker=match["marker"],
        component=match["component"],
        solution=int(match["solution"]),
        reference_m=float(match["reference"]),
        provenance=lines[0].strip(),
    )


def select_min_sigma_indices(
    epochs: NDArray[np.float64], sigmas: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Index of the minimum-σ solution per unique epoch (masked σ = +∞).

    For the multiset of rows {(tᵢ, σᵢ)}, returns one original index per
    distinct epoch t: argmin_{i: tᵢ=t} σ̃ᵢ where σ̃ = σ if finite else +∞
    (a masked ``********``/``************`` sigma parses to NaN → +∞ so it
    never wins). Ties (equal epoch and equal σ̃, including two masked rows)
    resolve **keep-first in array order** — deterministic; the join passes
    segments in accumulation order so the choice is reproducible. The
    returned indices are in ascending-epoch order.

    Symbols → args: t → ``epochs`` [fractional year]; σ → ``sigmas`` [m].
    Numerical notes: stable ``np.lexsort`` on (epoch, σ̃, original-index);
    duplicate epochs are matched by exact float equality, which is exact for
    the identically-formatted 5-decimal epoch strings that produce them.
    """
    finite_sigma = np.where(np.isnan(sigmas), np.inf, sigmas)
    original_index = np.arange(epochs.size, dtype=np.intp)
    # lexsort: last key is primary → sort by epoch, then σ̃, then keep-first.
    order = np.lexsort((original_index, finite_sigma, epochs))
    epochs_sorted = epochs[order]
    is_first = np.empty(order.size, dtype=bool)
    is_first[:1] = True
    is_first[1:] = epochs_sorted[1:] != epochs_sorted[:-1]
    return np.asarray(order[is_first], dtype=np.intp)


def dedup_min_sigma(
    epochs: NDArray[np.float64],
    values: NDArray[np.float64],
    sigmas: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Collapse duplicate epochs to their minimum-σ solution, epoch-sorted.

    Applies :func:`select_min_sigma_indices` and gathers the chosen rows, so
    each distinct epoch survives once with the value/σ of its smallest-σ
    solution (masked σ never wins). Returns (epochs, values, sigmas) sorted
    ascending in epoch. Pure; float64; units m for values/σ.
    """
    selected = select_min_sigma_indices(epochs, sigmas)
    return epochs[selected], values[selected], sigmas[selected]


def read_mb_segment(path: Path) -> MbSegment:
    """Read one ``multibase`` segment file into arrays.

    Computes the segment series S = {(tᵢ, vᵢ, σᵢ)} from the data rows
    ``epoch value sigma`` (units: fractional year, m, m), stable-sorted
    ascending in t. Duplicate epochs are **kept** (not deduped here): global
    min-σ dedup is deferred to :func:`join_segments` so it can compare every
    overlapping segment's solution at a shared epoch. Pre files are
    themselves historical concats and carry internal duplicates.

    Numerical notes: float64 throughout; a masked value/sigma field
    (``********`` / ``************`` Fortran field overflow) parses to NaN;
    blank lines are skipped.

    Reference: GAMIT/GLOBK ``multibase`` output format (GGVer 10.71),
    okada.vedur.is ``/D/GMT/pre``·``rap`` recon 2026-07-17/18.
    """
    raw_lines = path.read_text().splitlines()
    header = _parse_header(raw_lines, path)

    epochs: list[float] = []
    values: list[float] = []
    sigmas: list[float] = []
    for lineno, line in enumerate(raw_lines[2:], start=3):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 3:
            raise GlobkJoinError(
                f"{path}:{lineno}: expected 'epoch value sigma', got {line!r}"
            )
        epochs.append(float(fields[0]))
        try:
            value = float(fields[1])
        except ValueError:
            value = float("nan")
        values.append(value)
        try:
            sigma = float(fields[2])
        except ValueError:
            sigma = float("nan")
        sigmas.append(sigma)

    if not epochs:
        raise GlobkJoinError(f"{path}: no data rows")

    epochs_arr = np.asarray(epochs, dtype=np.float64)
    order = np.argsort(epochs_arr, kind="stable")

    return MbSegment(
        header=header,
        path=path,
        epochs=epochs_arr[order],
        values=np.asarray(values, dtype=np.float64)[order],
        sigmas=np.asarray(sigmas, dtype=np.float64)[order],
    )


def estimate_segment_offset(
    acc_epochs: NDArray[np.float64],
    acc_values: NDArray[np.float64],
    segment: MbSegment,
) -> tuple[float, int, bool]:
    """Estimate the raw datum offset Δ̂ [m] of a segment vs. the joined series.

    Δ̂ = median_{t ∈ O} (v_seg(t) − v_acc(t)) over the overlapping epochs
    O = T_seg ∩ T_acc (exact match at 5-decimal epoch resolution). When a gap
    leaves O = ∅, Δ̂ falls back to the boundary pair
    Δ̂ = v_seg(t₀) − v_acc(t⁻), with t₀ the segment's first finite-value epoch
    and t⁻ the latest accumulated epoch < t₀ — this mixes wrap + real motion +
    any genuine step, hence the ``overlap_available=False`` flag so callers
    (and the residual guard) can treat it as the less certain estimate.

    Symbols → args: v_acc, T_acc → ``acc_epochs``/``acc_values`` [fractional
    year, m]; v_seg → ``segment``. Returns (Δ̂ [m], |O|, overlap_available).

    Numerical notes: ``nanmedian`` over the overlap ignores masked solutions
    (NaN); requires ``acc_epochs`` sorted ascending. Raises
    :class:`GlobkJoinError` only if there is neither overlap nor any earlier
    accumulated epoch (cannot happen once segments are anchored on the
    earliest, but guarded for direct callers).
    """
    acc_keys = {_EPOCH_KEY_FMT.format(e): i for i, e in enumerate(acc_epochs)}
    seg_idx: list[int] = []
    acc_idx: list[int] = []
    for j, e in enumerate(segment.epochs):
        i = acc_keys.get(_EPOCH_KEY_FMT.format(e))
        if i is not None:
            seg_idx.append(j)
            acc_idx.append(i)

    if seg_idx:
        diffs = segment.values[np.asarray(seg_idx)] - acc_values[np.asarray(acc_idx)]
        finite = diffs[np.isfinite(diffs)]
        if finite.size:
            return float(np.median(finite)), len(seg_idx), True

    # gap: boundary pair from the first finite segment value and the last
    # accumulated epoch strictly before it that HAS a finite value.
    #
    # Requiring the partner to be finite is not defensive tidiness: the
    # accumulated series is the min-sigma dedup of everything joined so far,
    # and the winning row at an epoch can carry a MASKED value (`********`
    # parses to NaN) while still having the smaller sigma. Taking the latest
    # earlier epoch unconditionally then made the offset NaN, which reached
    # `wrap_correction` and died in `round()` as a bare "cannot convert float
    # NaN to integer" -- no station, no segment, no epoch, and it aborted the
    # whole batch rather than the one axis. Measured on okada's DGAR East and
    # Up: boundary partner 1998.03972 is NaN, first finite segment epoch
    # 1998.04246. The overlap branch above already restricts to finite diffs;
    # this is the same rule for the fallback.
    seg_finite = np.nonzero(np.isfinite(segment.values))[0]
    if seg_finite.size == 0:
        raise GlobkJoinError(f"{segment.path}: no finite values to estimate datum")
    first_i = int(seg_finite[0])
    first_epoch = float(segment.epochs[first_i])
    earlier = np.nonzero((acc_epochs < first_epoch) & np.isfinite(acc_values))[0]
    if earlier.size == 0:
        raise GlobkJoinError(
            f"{segment.path}: no overlap and no accumulated epoch with a finite "
            f"value before {first_epoch:.5f}; cannot estimate datum offset"
        )
    raw = float(segment.values[first_i] - acc_values[int(earlier[-1])])
    return raw, 0, False


def wrap_correction(
    raw_offset_m: float, wrap_quantum_m: float = WRAP_QUANTUM_M
) -> float:
    """Snap a raw datum offset to the GLOBK wrap lattice.

    Computes the datum correction c = q·round(Δ̂ / q) [m], the nearest
    integer multiple of the wrap quantum q to the raw offset Δ̂ — the only
    datum shifts GLOBK segment boundaries exhibit (module docstring).

    Symbols → args: Δ̂ → ``raw_offset_m`` [m]; q → ``wrap_quantum_m`` [m].

    Numerical notes: ``round`` is banker's rounding, irrelevant here since
    valid inputs are ≪ q/2 away from the lattice; the |Δ̂ − c| ≤ r_max
    guard in :func:`join_segments` rejects ambiguous offsets.

    Raises:
        GlobkJoinError: on a non-finite Δ̂. ``round()`` would otherwise raise
            a bare ``ValueError: cannot convert float NaN to integer`` naming
            nothing, which is how a single bad boundary pair aborted a
            379-station batch instead of failing one axis. Callers upstream
            should not produce NaN; this makes it impossible for one to do so
            silently.
    """
    if not np.isfinite(raw_offset_m):
        raise GlobkJoinError(
            f"non-finite raw datum offset ({raw_offset_m}); cannot snap to the "
            f"{wrap_quantum_m} m wrap lattice"
        )
    return wrap_quantum_m * round(raw_offset_m / wrap_quantum_m)


def join_segments(
    segments: Sequence[MbSegment],
    *,
    wrap_quantum_m: float = WRAP_QUANTUM_M,
    max_residual_m: float = MAX_RESIDUAL_M,
) -> JoinedSeries:
    """Join GLOBK segments onto the common datum of the earliest segment.

    Segments are ordered by first epoch with the earliest as datum anchor
    (v′ ≡ v). Each later segment k is de-wrapped by v′ₖ = vₖ − cₖ where
    cₖ = q·round(Δ̂ₖ/q) and Δ̂ₖ is its raw offset against the min-σ-deduped
    join built so far (:func:`estimate_segment_offset`). All de-wrapped
    points are pooled; the returned series is the min-σ dedup
    (:func:`dedup_min_sigma`) of that pool, epoch-sorted — so overlapping
    solutions collapse to their smallest-σ representative (masked σ never
    wins) and single-epoch within-segment outliers pass through untouched.
    Header reference constants are validated for station/component identity
    but never applied to values (they do not encode the datum — module
    docstring).

    Symbols → args: q → ``wrap_quantum_m`` [m]; r_max → ``max_residual_m``
    [m]. The guard |Δ̂ₖ − cₖ| ≤ r_max separates real boundary motion (mm–cm)
    from a datum that is not on the wrap lattice; it is **loud** — a trip
    raises rather than being silently absorbed, because it flags either a
    genuine step (→ steps.csv, a human decision) or a boundary-pair estimate
    with no overlap to lean on. The offending segment's raw Δ̂, correction,
    residual, and overlap availability are all in the message.

    Raises :class:`GlobkJoinError` on empty input, station/component
    mismatch, or a residual exceeding r_max.

    Reference: replacement for ``timesfunc.compGlobkTimes`` +
    ``fixGlobkoffset`` (okada ``compGLOBK -o``), recon 2026-07-17/18.
    """
    if not segments:
        raise GlobkJoinError("no segments to join")

    ordered = sorted(segments, key=lambda s: float(s.epochs[0]))
    anchor = ordered[0]
    station, component = anchor.header.station, anchor.header.component
    for seg in ordered[1:]:
        if (seg.header.station, seg.header.component) != (station, component):
            raise GlobkJoinError(
                f"segment mismatch: {seg.path} is "
                f"{seg.header.station}/{seg.header.component}, "
                f"expected {station}/{component}"
            )

    pool_epochs = anchor.epochs.copy()
    pool_values = anchor.values.copy()
    pool_sigmas = anchor.sigmas.copy()
    # running min-σ-deduped accumulation — the reference the next segment's
    # datum offset is measured against.
    acc_epochs, acc_values, _ = dedup_min_sigma(pool_epochs, pool_values, pool_sigmas)
    corrections = [
        SegmentCorrection(
            path=anchor.path,
            n_overlap=0,
            overlap_available=False,
            raw_offset_m=0.0,
            correction_m=0.0,
            residual_m=0.0,
            n_rows=int(anchor.epochs.size),
        )
    ]

    for seg in ordered[1:]:
        raw_offset, n_overlap, overlap_available = estimate_segment_offset(
            acc_epochs, acc_values, seg
        )
        correction = wrap_correction(raw_offset, wrap_quantum_m)
        residual = raw_offset - correction
        if abs(residual) > max_residual_m:
            raise GlobkJoinError(
                f"{seg.path}: datum residual {residual:+.3f} m after correction "
                f"{correction:+.1f} m exceeds max_residual_m={max_residual_m} "
                f"(raw Δ̂={raw_offset:+.3f} m, "
                f"overlap={'yes' if overlap_available else 'no (boundary pair)'}) "
                "— boundary offset is not on the wrap lattice: genuine step or "
                "unreliable gap estimate, needs a human decision"
            )
        pool_epochs = np.concatenate([pool_epochs, seg.epochs])
        pool_values = np.concatenate([pool_values, seg.values - correction])
        pool_sigmas = np.concatenate([pool_sigmas, seg.sigmas])
        acc_epochs, acc_values, _ = dedup_min_sigma(
            pool_epochs, pool_values, pool_sigmas
        )
        corrections.append(
            SegmentCorrection(
                path=seg.path,
                n_overlap=n_overlap,
                overlap_available=overlap_available,
                raw_offset_m=raw_offset,
                correction_m=correction,
                residual_m=residual,
                n_rows=int(seg.epochs.size),
            )
        )

    epochs, values, sigmas = dedup_min_sigma(pool_epochs, pool_values, pool_sigmas)
    return JoinedSeries(
        station=station,
        component=component,
        epochs=epochs,
        values=values,
        sigmas=sigmas,
        header=anchor.header,
        corrections=tuple(corrections),
    )


def discover_segments(
    station: str, axis: int, segment_dirs: Sequence[Path]
) -> list[Path]:
    """List a station/axis' segment files across directories.

    Matches the production glob ``mb_<STA>_*.dat<axis>`` (axis 1/2/3 ↔
    N/E/U) in each directory in order (e.g. Pre then Rap). Purely a path
    operation; the join itself orders segments by epoch.
    """
    if axis not in (1, 2, 3):
        raise GlobkJoinError(f"axis must be 1, 2 or 3 (N/E/U), got {axis}")
    paths: list[Path] = []
    for directory in segment_dirs:
        paths.extend(sorted(directory.glob(f"mb_{station}_*.dat{axis}")))
    if not paths:
        raise GlobkJoinError(
            f"no mb_{station}_*.dat{axis} segments found in {[str(d) for d in segment_dirs]}"
        )
    return paths


def join_station_component(
    station: str,
    axis: int,
    segment_dirs: Sequence[Path],
    *,
    wrap_quantum_m: float = WRAP_QUANTUM_M,
    max_residual_m: float = MAX_RESIDUAL_M,
) -> JoinedSeries:
    """Discover, read and join one station/component's GLOBK segments.

    Thin orchestration over :func:`discover_segments`,
    :func:`read_mb_segment` and :func:`join_segments`; see those for the
    datum arithmetic. ``axis`` is the mb file suffix (1/2/3 ↔ N/E/U);
    ``segment_dirs`` typically ``[pre_dir, rap_dir]``.
    """
    segments = [
        read_mb_segment(p) for p in discover_segments(station, axis, segment_dirs)
    ]
    return join_segments(
        segments, wrap_quantum_m=wrap_quantum_m, max_residual_m=max_residual_m
    )


def format_tot_file(joined: JoinedSeries) -> str:
    """Render a joined series as an ``mb_STA_TOT.dat`` document (str).

    Produces the exact format :func:`read_mb_segment` and
    ``gps_read.openGlobkTimes`` (``skiprows=3``) consume — **exactly three
    header lines**, then data rows::

        Globk Analysis GGVer 10.71.021 Wed Sep 28 13:04:52 EDT 2022
        SENG_GPS to E Solution  1 +  16542671.519 m
        <single space>
         2015.48356      1.51913  0.00156

    Line 1 is the anchor segment's provenance line; line 2 is reconstructed
    from the anchor header (identification/provenance only — the reference
    constant does not encode the datum and is never applied to values, see
    the module docstring); line 3 is the production single-space separator.
    A missing header line would make the ``skiprows=3`` reader silently
    drop the first data epoch, so the three-line contract is load-bearing.

    Data rows are `` epoch value sigma`` in the production field layout
    (``" %10.5f %12.5f %8.5f"``): epochs in fractional years, values and
    sigmas in metres. NaN (masked) fields render as the Fortran-style star
    fill (12 stars for a value, 8 for a sigma), which parses back to NaN in
    both :func:`read_mb_segment` and ``openGlobkTimes``.
    """
    h = joined.header
    ref_line = (
        f"{h.station}_{h.marker} to {h.component} "
        f"Solution  {h.solution} +  {h.reference_m:.3f} m"
    )
    lines = [h.provenance, ref_line, " "]
    for t, v, s in zip(joined.epochs, joined.values, joined.sigmas, strict=True):
        val = "*" * 12 if np.isnan(v) else f"{v:12.5f}"
        sig = "*" * 8 if np.isnan(s) else f"{s:8.5f}"
        lines.append(f" {t:10.5f} {val} {sig}")
    return "\n".join(lines) + "\n"


def write_joined_series(joined: JoinedSeries, path: Path) -> Path:
    """Write a joined series to ``path`` in ``mb_STA_TOT.dat`` format.

    Thin file wrapper over :func:`format_tot_file` (see there for the
    3-line-header format contract). The conventional name for the TOT
    scheme is ``mb_<STA>_TOT.dat<axis>`` with axis 1/2/3 ↔ N/E/U — the
    caller chooses the path; the parent directory must exist. Returns
    ``path`` for chaining.
    """
    path.write_text(format_tot_file(joined))
    return path
