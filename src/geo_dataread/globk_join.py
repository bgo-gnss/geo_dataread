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

Data rows are ``epoch  value  sigma``: epoch in fractional years, value and
sigma in metres. The value column sits on a *run-specific datum*: each
``multibase`` run may place the same physical position at a value shifted by
an integer multiple of the GLOBK wrap quantum q = 10 m. Joining segments
from different runs (Pre = prior years, Rap = current year) therefore shows
spurious ±10 m steps at segment boundaries unless each segment is
rebaselined onto a common datum.

Empirical spec (okada.vedur.is recon, 2026-07-17 — see the derivation
below before changing anything):

* The header reference constant (``+ 16542671.519 m``) does **not** encode
  the run datum. Verified: THOB East Pre/Rap references differ by 1.884 m
  while the value column is continuous across the boundary to mm level;
  SENG East references differ by 2.621 m while the actual datum shift is
  exactly 10.000 m. The reference line is parsed for station/component
  identification and provenance only — never applied to the values.
* Boundary datum shifts are integer multiples of q = 10 m (SENG East
  2025.99863 → 2026.00137: raw step 10.0014 m = 1·q + 1.4 mm real motion).

Derivation chain (top-down):

1. :func:`read_mb_segment` parses one segment file into epoch/value/sigma
   arrays (sorted by epoch, duplicate epochs keep first file occurrence —
   matching the first-occurrence semantics of the production concat).
2. :func:`estimate_segment_offset` measures the raw datum offset Δ̂ of a
   segment against the accumulated series (median over overlapping epochs,
   else the boundary pair).
3. :func:`wrap_correction` snaps Δ̂ to the nearest multiple of q:
   c = q·round(Δ̂/q).
4. :func:`join_segments` orders segments chronologically, anchors the datum
   on the earliest segment, applies v′ = v − c per segment, guards the
   residual |Δ̂ − c| ≤ r_max, and appends only epochs beyond the current
   coverage (production Rap-filter semantics).

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
    "join_segments",
    "discover_segments",
    "join_station_component",
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

    ``epochs`` [fractional year] are strictly increasing (duplicates dropped,
    first file occurrence kept); ``values`` and ``sigmas`` are metres.
    Unreadable sigmas (``********`` field overflow) are NaN.
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
    """

    path: Path
    n_overlap: int  #: epochs shared with the accumulated series.
    raw_offset_m: float
    correction_m: float
    residual_m: float
    n_appended: int  #: data rows this segment contributed to the join.


@dataclass(frozen=True)
class JoinedSeries:
    """Segments joined onto the datum of the earliest segment.

    ``epochs`` [fractional year] strictly increasing; ``values``/``sigmas``
    in metres on the anchor segment's datum. ``header`` is the anchor
    segment's header; ``corrections`` records the per-segment datum
    arithmetic in join order (anchor first, correction 0 by construction).
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


def read_mb_segment(path: Path) -> MbSegment:
    """Read one ``multibase`` segment file into arrays.

    Computes the segment series S = {(tᵢ, vᵢ, σᵢ)} from the data rows
    ``epoch value sigma`` (units: fractional year, m, m), sorted ascending
    in t with duplicate epochs dropped keeping the *first* file occurrence
    (first-occurrence semantics of the production TOT concat; duplicate
    epochs occur in Pre files that are themselves historical concats).

    Numerical notes: float64 throughout; a sigma field of ``********``
    (Fortran field overflow) parses to NaN; blank lines are skipped.

    Reference: GAMIT/GLOBK ``multibase`` output format (GGVer 10.71),
    okada.vedur.is ``/D/GMT/pre``·``rap`` recon 2026-07-17.
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
        values.append(float(fields[1]))
        try:
            sigma = float(fields[2])
        except ValueError:
            sigma = float("nan")
        sigmas.append(sigma)

    if not epochs:
        raise GlobkJoinError(f"{path}: no data rows")

    epochs_arr = np.asarray(epochs, dtype=np.float64)
    order = np.argsort(epochs_arr, kind="stable")
    keep: list[int] = []
    seen: set[str] = set()
    for idx in order:
        key = _EPOCH_KEY_FMT.format(epochs_arr[idx])
        if key in seen:
            continue
        seen.add(key)
        keep.append(int(idx))
    keep_arr = np.asarray(keep, dtype=np.intp)

    return MbSegment(
        header=header,
        path=path,
        epochs=epochs_arr[keep_arr],
        values=np.asarray(values, dtype=np.float64)[keep_arr],
        sigmas=np.asarray(sigmas, dtype=np.float64)[keep_arr],
    )


def estimate_segment_offset(
    acc_epochs: NDArray[np.float64],
    acc_values: NDArray[np.float64],
    segment: MbSegment,
) -> tuple[float, int]:
    """Estimate the raw datum offset Δ̂ [m] of a segment vs. the joined series.

    Δ̂ = median_{t ∈ O} (v_seg(t) − v_acc(t)) over the overlapping epochs
    O = T_seg ∩ T_acc (exact match at 5-decimal epoch resolution); if
    O = ∅, Δ̂ = v_seg(t₀) − v_acc(t⁻) where t₀ is the segment's first epoch
    and t⁻ the latest accumulated epoch with t⁻ < t₀ (boundary pair).

    Symbols → args: v_acc, T_acc → ``acc_epochs``/``acc_values`` [fractional
    year, m]; v_seg → ``segment``. Returns (Δ̂ [m], |O|).

    Numerical notes: the median makes the overlap estimate robust to
    occasional differing duplicate solutions (observed at cm level);
    requires ``acc_epochs`` sorted ascending. Raises
    :class:`GlobkJoinError` if no overlap and no earlier epoch exists.
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
        return float(np.median(diffs)), len(seg_idx)

    first_epoch = float(segment.epochs[0])
    earlier = np.nonzero(acc_epochs < first_epoch)[0]
    if earlier.size == 0:
        raise GlobkJoinError(
            f"{segment.path}: no overlap and no accumulated epoch before "
            f"{first_epoch:.5f}; cannot estimate datum offset"
        )
    return float(segment.values[0] - acc_values[int(earlier[-1])]), 0


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
    """
    return wrap_quantum_m * round(raw_offset_m / wrap_quantum_m)


def join_segments(
    segments: Sequence[MbSegment],
    *,
    wrap_quantum_m: float = WRAP_QUANTUM_M,
    max_residual_m: float = MAX_RESIDUAL_M,
) -> JoinedSeries:
    """Join GLOBK segments onto the common datum of the earliest segment.

    For segments ordered by first epoch, with the earliest as datum anchor
    (v′ ≡ v), each later segment k gets v′ₖ = vₖ − cₖ where
    cₖ = q·round(Δ̂ₖ/q) and Δ̂ₖ is its raw offset against the join built so
    far (:func:`estimate_segment_offset`); it then contributes only epochs
    beyond the current coverage, t > max(T_acc) — the production Rap-filter
    semantics. The header reference constants are validated for
    station/component identity but never applied to values (they do not
    encode the datum — module docstring).

    Symbols → args: q → ``wrap_quantum_m`` [m]; r_max → ``max_residual_m``
    [m], guard |Δ̂ₖ − cₖ| ≤ r_max separating real boundary motion (mm–cm,
    occasionally dm across gaps) from datum errors.

    Raises :class:`GlobkJoinError` on empty input, station/component
    mismatch, or a residual exceeding r_max.

    Reference: replacement for ``timesfunc.compGlobkTimes`` +
    ``fixGlobkoffset`` (okada ``compGLOBK -o``), recon 2026-07-17.
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

    epochs = anchor.epochs.copy()
    values = anchor.values.copy()
    sigmas = anchor.sigmas.copy()
    corrections = [
        SegmentCorrection(
            path=anchor.path,
            n_overlap=0,
            raw_offset_m=0.0,
            correction_m=0.0,
            residual_m=0.0,
            n_appended=int(anchor.epochs.size),
        )
    ]

    for seg in ordered[1:]:
        raw_offset, n_overlap = estimate_segment_offset(epochs, values, seg)
        correction = wrap_correction(raw_offset, wrap_quantum_m)
        residual = raw_offset - correction
        if abs(residual) > max_residual_m:
            raise GlobkJoinError(
                f"{seg.path}: residual {residual:+.3f} m after datum correction "
                f"{correction:+.1f} m exceeds max_residual_m={max_residual_m} — "
                "boundary offset is not on the wrap lattice"
            )
        new = seg.epochs > epochs[-1]
        n_appended = int(np.count_nonzero(new))
        if n_appended:
            epochs = np.concatenate([epochs, seg.epochs[new]])
            values = np.concatenate([values, seg.values[new] - correction])
            sigmas = np.concatenate([sigmas, seg.sigmas[new]])
        corrections.append(
            SegmentCorrection(
                path=seg.path,
                n_overlap=n_overlap,
                raw_offset_m=raw_offset,
                correction_m=correction,
                residual_m=residual,
                n_appended=n_appended,
            )
        )

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
