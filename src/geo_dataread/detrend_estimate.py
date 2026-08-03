"""Batch detrend-parameter estimation over a local TOT directory.

Console script ``gps-estimate-detrend`` — the estimation caller of
``gps_analysis/docs/DESIGN_live_detrending.md`` (estimation is a
*deliberate, occasional act*; application is the cheap pure view). For
each requested station it reads the plate-removed series from the local
TOT directory (:func:`geo_dataread.gps_read.getData` with ``ref="plate"``,
``tType="TOT"`` — the ``gps-globk-tot`` output), fits the stored-detrend
trajectory with the leaf :func:`gps_analysis.estimate_detrend` (window +
validity gates -> step augmentation -> outlier removal BEFORE the fit ->
clean WLS) and assembles the station records into the versioned
``detrend_params.json`` document that
:func:`geo_dataread.gps_views.read_detrend_params` consumes (schema v1).

Per-station fit catalog (``fit_windows.csv``)
---------------------------------------------

Fit windows and validity-gate overrides are per-station *reviewed
decisions*, never hardcoded: they live in the deployed fit catalog,
resolved through the shared :mod:`gps_parser.outlier_catalogs` mechanism
exactly like ``steps.csv`` / ``protect_windows.csv`` (``postprocess.cfg``
``[FILES] fit_windows``, else ``<gpsconfig dir>/fit_windows.csv``; source
of the deployed copy: ``gps-config-data/analysis-lane/fit_windows.csv``).
CLI flags supply the GLOBAL defaults; a station's catalog row overrides
them field by field (blank = keep the default). Canonical example: DYNG
needs ``max_gap_years=1.0`` (real data gaps up to ~1 yr trip the leaf's
default 0.5 yr gate), while unrest stations get explicit pre-unrest
windows.

Determinism: the document is byte-reproducible by default
(``generated_at``/``fitted_at`` are ``None``); pass ``--stamp`` to embed
the wall-clock estimation timestamp (design §6 staleness trigger T1).

Usage::

    gps-estimate-detrend DYNG SENG --tot-dir ~/gps-data/TOT \\
        --out detrend_params.json [--fit-catalog fit_windows.csv] [--stamp]

Exit status: 0 when every requested station was estimated or was an
intentional loud skip (outlier-stage abort — such a record is NOT
stored); 1 when any station failed (no data, validity gate, ...) —
failures are reported per station, the batch continues.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from gps_analysis import OutlierParams, estimate_detrend, evaluate_record
from gps_parser import outlier_catalogs as _oc

from geo_dataread.gps_views import (
    PLATE_REMOVED_FRAME,
    resolve_outlier_detection,
    resolve_protect_windows,
    station_step_epochs,
)

__all__ = [
    "DOC_SCHEMA_VERSION",
    "FIT_CATALOG_FILENAME",
    "FIT_CATALOG_COLUMNS",
    "GENERATOR",
    "FitCatalogRow",
    "FitDefaults",
    "StationFitSettings",
    "StationResult",
    "default_fit_catalog_path",
    "read_fit_catalog",
    "resolve_fit_settings",
    "StationEstimate",
    "station_estimate_from_arrays",
    "station_record_from_arrays",
    "estimate_station",
    "build_document",
    "main",
]

FloatArray = npt.NDArray[np.float64]

DOC_SCHEMA_VERSION = 1
"""``detrend_params.json`` document schema version this writer produces
(must match ``gps_views.DOC_SCHEMA_VERSION`` — the reader rejects others)."""

FIT_CATALOG_FILENAME = "fit_windows.csv"
"""Deployed per-station fit-catalog filename (gpsconfig-owned)."""

GENERATOR = "gps-estimate-detrend"
"""``generator`` tag written into the document."""

MODEL = "lineperiodic"
"""Trajectory model the estimator fits (production default, design §0.1)."""

FRAME = PLATE_REMOVED_FRAME
"""Reference-frame tag of the stored parameters (design §0.5: detrend runs
AFTER plate-velocity removal; ``getData(ref="plate")`` provides exactly
that frame in mm)."""

UNCERT = 15
"""Formal-sigma screen [mm] applied at READ time by :func:`gps_read.getData`.

Matches ``getData``'s own default, which is what this estimator used
implicitly before the knob existed.  It belongs in the record's ``refs``
because it decides WHICH epochs were fitted without leaving a trace in any
fitted quantity: two records with identical parameters, windows and steps can
still have been fitted on different data.  The detrend workbench screens
harder (10) by default, which is exactly why both sides must be able to say
so."""

FIT_CATALOG_COLUMNS = (
    "sta",
    "window_start",
    "window_end",
    "segments",
    "max_gap_years",
    "min_epochs",
    "min_span_years",
    "steps",
    "comment",
)
"""Current ``fit_windows.csv`` column set."""

_LEGACY_FIT_CATALOG_COLUMNS = tuple(c for c in FIT_CATALOG_COLUMNS if c != "segments")
"""The pre-``segments`` column set, still accepted verbatim.

The header check is an exact tuple match on purpose: this catalog changes
stored science parameters, so an unrecognised layout must be a hard error and
never a silent partial read.  That makes adding a column a breaking change
for every deployed file — so the check matches an ALLOWLIST of known layouts
rather than one tuple.  Strictness is unchanged; only the set of layouts
called "known" grew.  A missing ``segments`` cell reads as empty = single
window, which is what every existing row means.
"""

_FIT_CATALOG_HEADERS = (FIT_CATALOG_COLUMNS, _LEGACY_FIT_CATALOG_COLUMNS)


# ---------------------------------------------------------------------------
# Per-station fit catalog (fit_windows.csv)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FitCatalogRow:
    """One station's parsed ``fit_windows.csv`` row.

    Every field except ``steps`` is an *override*: None = the operator left
    it blank = use the global default (open bound for the window fields).
    ``steps`` is the per-station known-step list for the fit window; None =
    fall back to the deployed ``steps.csv`` catalog, an EXPLICIT empty
    tuple cannot be expressed (list at least one epoch or rely on the
    window keeping steps outside).
    """

    window_start: float | None = None
    window_end: float | None = None
    segments: tuple[tuple[float | None, float | None], ...] | None = None
    max_gap_years: float | None = None
    min_epochs: int | None = None
    min_span_years: float | None = None
    steps: tuple[float, ...] | None = None
    comment: str = ""


@dataclasses.dataclass(frozen=True)
class FitDefaults:
    """Global validity-gate defaults (the CLI flags; leaf defaults here)."""

    min_span_years: float = 2.0
    min_epochs: int = 365
    max_gap_years: float = 0.5


@dataclasses.dataclass(frozen=True)
class StationFitSettings:
    """Fully-resolved fit settings for one station (defaults + catalog row).

    ``window_source`` records where the window/gates came from (the
    catalog path or ``"defaults"``) — it rides into the record's ``refs``
    provenance so a stored fit names its configuration.
    """

    segments: tuple[tuple[float | None, float | None], ...]
    min_span_years: float
    min_epochs: int
    max_gap_years: float
    steps: tuple[float, ...] | None
    window_source: str

    @property
    def window(self) -> tuple[float | None, float | None]:
        """The HULL of the segments, for readers that want one interval.

        Kept as a property rather than a second field so there is exactly
        one source of truth: every existing caller that prints or indexes
        ``settings.window`` keeps working, and none of them can drift out of
        step with ``segments``.
        """
        return (self.segments[0][0], self.segments[-1][1])


@dataclasses.dataclass(frozen=True)
class StationResult:
    """Outcome of estimating one station.

    ``status`` is ``"estimated"`` (record produced), ``"skipped"`` (loud
    intentional skip — outlier-stage abort; such parameters are never
    stored, design §0.4) or ``"error"`` (no data / validity gate / read
    failure, reported loudly without aborting the batch).
    """

    station: str
    status: str
    detail: str
    record: dict[str, Any] | None = None


def default_fit_catalog_path() -> Path | None:
    """Resolve the deployed ``fit_windows.csv`` path via gps_parser.

    Resolution order (the shared :func:`gps_parser.outlier_catalogs.catalog_path`
    mechanism — identical to ``steps.csv`` / ``protect_windows.csv``):

    1. ``postprocess.cfg`` ``[FILES] fit_windows``;
    2. ``<gpsconfig dir>/fit_windows.csv`` (the deploy-target default).

    Returns:
        The resolved path (which may not exist yet — the fit catalog is an
        optional per-station enhancement), or None when no gpsconfig is
        reachable.
    """
    return cast("Path | None", _oc.catalog_path("fit_windows", FIT_CATALOG_FILENAME))


def _parse_optional_float(sta: str, field: str, raw: str) -> float | None:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"station {sta}: {field} {raw!r} is not a number") from None


def _parse_segments_cell(
    sta: str, raw: str
) -> tuple[tuple[float | None, float | None], ...] | None:
    """Parse a ``segments`` cell: ``a:b;c:d``, empty bound = open.

    ``;`` already means "list" in this file (the ``steps`` column) and
    ``:`` separates a segment's bounds — deliberately not ``-``, which
    would be ambiguous against a negative fractional year the moment
    anyone tries one.

    The whole segmentation lives in ONE cell rather than one row per
    segment because this reader is strict by design: a mistyped row in a
    many-row layout would still parse, yielding *different but valid*
    science instead of an error, and a bad fit window silently changes
    stored parameters.  One cell fails loudly or not at all.

    Returns None for an empty cell (= no segmentation, use the window
    columns), never an empty tuple — the leaf refuses that, and rightly.
    """
    if not raw:
        return None
    out: list[tuple[float | None, float | None]] = []
    for token in (tok.strip() for tok in raw.split(";")):
        if not token:
            continue
        if token.count(":") != 1:
            raise ValueError(
                f"station {sta}: segment {token!r} must be 'start:end' "
                f"(either side may be empty for an open bound)"
            )
        lo, _, hi = token.partition(":")
        out.append(
            (
                _parse_optional_float(sta, "segment start", lo.strip()),
                _parse_optional_float(sta, "segment end", hi.strip()),
            )
        )
    if not out:
        raise ValueError(f"station {sta}: segments cell {raw!r} holds no segment")
    for j, (lo_v, hi_v) in enumerate(out):
        if lo_v is not None and hi_v is not None and hi_v <= lo_v:
            raise ValueError(f"station {sta}: segment {j} end {hi_v} <= start {lo_v}")
    return tuple(out)


def read_fit_catalog(path: str | Path) -> dict[str, FitCatalogRow]:
    """Read a per-station fit catalog (``fit_windows.csv``).

    Format: exactly the columns :data:`FIT_CATALOG_COLUMNS`, with ``#``
    comment lines and blank lines allowed; one row per station; blank
    field = keep the global default; ``steps`` is a ``;``-separated list
    of fractional-year step epochs.

    Estimation is a deliberate act, so this reader is STRICT — a corrupt
    catalog is a hard error, never a silent skip (contrast the graceful
    read-path catalogs: a bad fit window would silently change stored
    science parameters).

    Args:
        path: Catalog path.

    Returns:
        ``{station: FitCatalogRow}`` (station markers upper-cased).

    Raises:
        FileNotFoundError: When the catalog does not exist.
        ValueError: On a wrong column set, a missing marker, a duplicate
            station row, a non-numeric field, or ``window_end <=
            window_start``.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"fit catalog not found: {resolved}")
    lines = [
        line
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    reader = csv.DictReader(lines)
    if (
        reader.fieldnames is None
        or tuple(reader.fieldnames) not in _FIT_CATALOG_HEADERS
    ):
        raise ValueError(
            f"{resolved}: fit catalog must have exactly the columns "
            f"{','.join(FIT_CATALOG_COLUMNS)!r} (or the pre-segments layout "
            f"{','.join(_LEGACY_FIT_CATALOG_COLUMNS)!r}), "
            f"got {reader.fieldnames!r}"
        )
    catalog: dict[str, FitCatalogRow] = {}
    for row in reader:
        sta = str(row.get("sta") or "").strip().upper()
        if not sta:
            raise ValueError(f"{resolved}: fit-catalog row without a 'sta' marker")
        if sta in catalog:
            raise ValueError(
                f"{resolved}: duplicate row for station {sta} — one row per station"
            )
        try:
            window_start = _parse_optional_float(
                sta, "window_start", str(row.get("window_start") or "").strip()
            )
            window_end = _parse_optional_float(
                sta, "window_end", str(row.get("window_end") or "").strip()
            )
            max_gap = _parse_optional_float(
                sta, "max_gap_years", str(row.get("max_gap_years") or "").strip()
            )
            min_span = _parse_optional_float(
                sta, "min_span_years", str(row.get("min_span_years") or "").strip()
            )
            raw_epochs = str(row.get("min_epochs") or "").strip()
            min_epochs: int | None = None
            if raw_epochs:
                try:
                    min_epochs = int(raw_epochs)
                except ValueError:
                    raise ValueError(
                        f"station {sta}: min_epochs {raw_epochs!r} is not an integer"
                    ) from None
            raw_steps = str(row.get("steps") or "").strip()
            steps: tuple[float, ...] | None = None
            if raw_steps:
                steps = tuple(
                    sorted(
                        float(tok)
                        for tok in (t.strip() for t in raw_steps.split(";"))
                        if tok
                    )
                )
            segments = _parse_segments_cell(sta, str(row.get("segments") or "").strip())
        except ValueError as exc:
            raise ValueError(f"{resolved}: {exc}") from None
        if segments is not None and (
            window_start is not None or window_end is not None
        ):
            raise ValueError(
                f"{resolved}: station {sta}: 'segments' and "
                f"'window_start'/'window_end' both set — one row must say one "
                f"thing about which epochs are fitted"
            )
        if (
            window_start is not None
            and window_end is not None
            and window_end <= window_start
        ):
            raise ValueError(
                f"{resolved}: station {sta}: window_end {window_end} <= "
                f"window_start {window_start}"
            )
        catalog[sta] = FitCatalogRow(
            window_start=window_start,
            window_end=window_end,
            segments=segments,
            max_gap_years=max_gap,
            min_epochs=min_epochs,
            min_span_years=min_span,
            steps=steps,
            comment=str(row.get("comment") or "").strip(),
        )
    return catalog


def resolve_fit_settings(
    sta: str,
    catalog: Mapping[str, FitCatalogRow] | None,
    defaults: FitDefaults,
    *,
    catalog_source: str | None = None,
) -> StationFitSettings:
    """Merge the global defaults with a station's catalog row.

    Field-by-field: a catalog value overrides the default; a blank (None)
    catalog field — or a station without a row — keeps the default (open
    window bounds, leaf gates).
    """
    row = (catalog or {}).get(sta)
    if row is None:
        return StationFitSettings(
            segments=((None, None),),
            min_span_years=defaults.min_span_years,
            min_epochs=defaults.min_epochs,
            max_gap_years=defaults.max_gap_years,
            steps=None,
            window_source="defaults",
        )
    source = catalog_source if catalog_source is not None else "fit_catalog"
    return StationFitSettings(
        segments=(
            row.segments
            if row.segments is not None
            else ((row.window_start, row.window_end),)
        ),
        min_span_years=(
            row.min_span_years
            if row.min_span_years is not None
            else defaults.min_span_years
        ),
        min_epochs=(
            row.min_epochs if row.min_epochs is not None else defaults.min_epochs
        ),
        max_gap_years=(
            row.max_gap_years
            if row.max_gap_years is not None
            else defaults.max_gap_years
        ),
        steps=row.steps,
        window_source=source,
    )


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def _finite_mask(
    yearf: FloatArray, data: FloatArray, sigma: FloatArray
) -> npt.NDArray[np.bool_]:
    """Epochs where the time tag and EVERY component value/sigma are finite."""
    return cast(
        "npt.NDArray[np.bool_]",
        np.isfinite(yearf)
        & np.all(np.isfinite(data), axis=0)
        & np.all(np.isfinite(sigma), axis=0),
    )


def _stage_summary(params: OutlierParams) -> str:
    """Compact record of which detection stages were enabled (§14).

    S1 (robust fit) and S2 (whitening) are structural and always run, so
    only the switchable stages appear.  ``"all"`` is the full pipeline.
    """
    on = []
    if getattr(params, "despike", False):
        on.append("S0")
    if getattr(params, "enable_global", True):
        on.append("S3")
    if getattr(params, "enable_window", True):
        on.append("S4")
    if getattr(params, "enable_protection", True):
        on.append("S5")
    return "all" if on == ["S3", "S4", "S5"] else "+".join(on) or "none"


@dataclasses.dataclass(frozen=True)
class StationEstimate:
    """A station record plus the fit diagnostics the record cannot carry.

    ``to_record`` is deliberately a *parameter* serialization — it stores
    ``n_rejected`` but not WHICH epochs were rejected, because the inlier
    mask is (C, N) per station and ``detrend_params.json`` is a
    fleet-wide document.  A caller that wants to SHOW the rejected epochs
    (the detrend workbench) therefore needs a second return channel, and
    it must be this one: any independently-run detector disagrees with
    the fit by construction — different window, different declared steps,
    an independently-reached excess-candidate abort — so a grey overlay
    derived that way would contradict the ``n_rejected`` printed beside it.

    The masks are lifted back to the CALLER's index space, i.e. the
    length of the ``yearf`` that was passed in.  Two compressions sit
    between that and ``estimate.inliers``: non-finite epochs are dropped
    before estimation, then the fit window subsets again.  Lifting here
    rather than at the call site is what keeps every consumer from
    re-deriving the same two-step index map.

    Attributes:
        record: The station record (``to_record`` shape).
        estimate: The full leaf :class:`~gps_analysis.DetrendEstimate`.
        outliers: (C, N_input) — True where the fit REJECTED that epoch.
            False everywhere outside the window and on dropped epochs:
            those got no verdict, which is not the same as "clean" (see
            ``in_window``).  Per-component sums equal ``record["n_rejected"]``.
        in_window: (N_input,) — True for epochs the fit actually judged.
        finite: (N_input,) — True for epochs that survived the
            non-finite drop.
    """

    record: dict[str, Any]
    estimate: Any
    outliers: npt.NDArray[np.bool_]
    in_window: npt.NDArray[np.bool_]
    finite: npt.NDArray[np.bool_]


def station_record_from_arrays(
    sta: str,
    yearf: FloatArray,
    data: FloatArray,
    sigma: FloatArray,
    *,
    settings: StationFitSettings,
    model: str = MODEL,
    frame: str = FRAME,
    steps_catalog: str | Path | None = None,
    protect_windows: str | Path | Sequence[tuple[float, float]] | None = None,
    outlier_overrides: str | Path | None = None,
    outlier_params: OutlierParams | None = None,
    fitted_at: str | None = None,
    refs: Mapping[str, Any] | None = None,
    stage_plan: Any | None = None,
    lookup_donor: Any | None = None,
    terms: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Estimate one station's stored-detrend record from ready arrays.

    The testable core of the driver: window/gates per ``settings``, steps
    from the settings (fit-catalog row) or the deployed ``steps.csv``
    (graceful), protect windows and per-station outlier levers through the
    shared graceful resolvers (the same wiring as the cleaned-``.NEU``
    writer), then the leaf :func:`gps_analysis.estimate_detrend` and
    :meth:`~gps_analysis.DetrendEstimate.to_record`.

    Non-finite epochs (NaN value or sigma in any component) are dropped
    before estimation — the leaf requires finite input; the drop count is
    recorded in the returned record's ``refs``.

    Args:
        sta: Station four-letter name.
        yearf: Epochs, fractional years, shape (N,).
        data: Plate-removed observations [mm], shape (3, N).
        sigma: 1-sigma uncertainties [mm], shape of ``data``.
        settings: Resolved window/gates/steps for this station.
        model: Trajectory model registry code.
        frame: Reference-frame tag stored on the record.
        steps_catalog: Explicit ``steps.csv`` path (dev override); None =
            deployed default. Ignored when the fit-catalog row lists steps.
        protect_windows: As :func:`gps_views.resolve_protect_windows`.
        outlier_overrides: Explicit ``outlier_overrides.csv`` path; None =
            deployed default.
        outlier_params: Explicit :class:`gps_analysis.OutlierParams`; None
            defers to the station's catalog row, else the spec defaults
            (the precedence :func:`gps_views.resolve_outlier_detection`
            already documents).  This is the ONLY route to a stage-isolated
            detection here (§14): the per-station catalog cannot express
            it, because ``OUTLIER_OVERRIDE_COLUMNS`` carries no ``enable_*``
            entries and ``read_outlier_overrides`` rejects unknown columns.
            Without it an operator workbench wanting S0-only detection
            would have to bypass this function and call the leaf directly,
            which is exactly how the workbench and the batch estimator
            would come to disagree about what a record means.
        fitted_at: Estimation timestamp for the record (None = unstamped,
            deterministic output).
        refs: Extra provenance merged into the record's ``refs``.

    Returns:
        The station record (:meth:`DetrendEstimate.to_record` shape), or
        None when the outlier stage aborted — an aborted fit is never
        stored (design §0.4); the caller reports the loud skip.

    Raises:
        ValueError: From the leaf on a failed validity gate (names the
            gate) or invalid input — estimation errors are hard.
    """
    result = station_estimate_from_arrays(
        sta,
        yearf,
        data,
        sigma,
        settings=settings,
        model=model,
        frame=frame,
        steps_catalog=steps_catalog,
        protect_windows=protect_windows,
        outlier_overrides=outlier_overrides,
        outlier_params=outlier_params,
        fitted_at=fitted_at,
        refs=refs,
        stage_plan=stage_plan,
        lookup_donor=lookup_donor,
        terms=terms,
    )
    return None if result is None else result.record


def _stage_domain_mask(t: FloatArray, result: Any, settings: StationFitSettings) -> Any:
    """Boolean mask of the epochs one executed stage actually fitted."""
    from gps_analysis.baseline import slice_windows

    segs = getattr(result, "segments", None) or settings.segments
    try:
        return np.asarray(slice_windows(t, segs), dtype=bool)
    except Exception:
        return np.ones(t.size, dtype=bool)


def _restage(
    est: Any,
    yearf: FloatArray,
    data: FloatArray,
    sigma: FloatArray,
    settings: StationFitSettings,
    plan: Any,
    lookup_donor: Any,
) -> tuple[Any, dict[str, Any]]:
    """Re-fit an already-detected estimate under a staged plan.

    Composition, per the operator decision of 2026-08-02: **detection runs
    ONCE over the whole fit window** exactly as the single-stage path does,
    and the staged fit then sees only the surviving epochs.  One verdict per
    epoch, so the grey markers on a figure mean one thing and a staged record
    stays comparable with a single-stage one.

    Consequence worth knowing before writing a plan: detection — and hence
    this whole path — is bounded by the station's fit window, so **stages
    SUBDIVIDE that window rather than reach outside it.**  For the Askja
    manoeuvre that is the right shape: let the window be the long span and
    give the first stage an explicit narrow ``@`` sub-window.

    The detection result is reused, never recomputed: only ``fits`` and
    ``rms`` are replaced, so every other field of the estimate (inliers,
    window_mask, span, step epochs, method) still describes the same
    detection pass and ``to_record`` needs no parallel implementation.

    **Measured safe** (2026-08-03, SELF/RHOF/VMEY/HOFN/AKUR): judging the
    epochs against the single-stage fit while storing the staged one moves
    stored rates by at most 0.0009 mm/yr (0.22 sigma) and step amplitudes by
    at most 0.05 mm -- 1-2x the Huber-vs-WLS reference mismatch the unstaged
    path already carries, since the detector judges against its own robust
    fit rather than the clean WLS one.

    ⚠ **Do not add an iteration loop here** (detect -> stage -> re-detect)
    without a cycle guard. Measured: 2-4 rounds to a fixed point on four
    stations, but AKUR does NOT converge -- a period-2 limit cycle in which
    one borderline east epoch flips in and out forever. Science impact is
    nil, which is precisely why it would survive review and waste a day in
    production.

    ⚠ **Restaging does not rescue an aborted station, and it was tried.**
    The tempting fix -- on abort, re-detect against the staged trajectory --
    was implemented and REVERTED on 2026-08-03 because it cannot work: the
    staged trajectory sits within ~0.005 mm/yr of the joint fit (that is why
    verdict drift is only 0.02-0.4 %), so it is the same model with a
    differently-estimated seasonal, not a better model of the flank signal.
    Measured on NYLA with per-station params, floors and protect windows:
    candidate fractions [0.089, 0.116, 0.004] against the staged trajectory
    versus [0.079, 0.092, 0.004] against the single-stage one -- still
    aborting, staged residual rms 42/65/9 mm, because a straight line cannot
    follow that deformation. The abort is a MODEL-ADEQUACY problem, not a
    reference-model problem; the fix is a term that can follow the signal (a
    transient), not a different judge.
    """
    import dataclasses as _dc

    from gps_analysis import estimate_staged, with_steps
    from gps_analysis.detrend import _resolve_model
    from gps_analysis.fitting import _resolve_linear_design

    from geo_dataread.stage_plan import resolve_stage_plan

    finite = np.isfinite(yearf)
    t_win = yearf[finite][np.asarray(est.window_mask, dtype=bool)]
    inliers = np.atleast_2d(np.asarray(est.inliers, dtype=bool))
    y_win = np.atleast_2d(data)[:, finite][:, np.asarray(est.window_mask, dtype=bool)]
    s_win = np.atleast_2d(sigma)[:, finite][:, np.asarray(est.window_mask, dtype=bool)]

    # The staged fit must use the SAME model the detection pass fitted,
    # including its step augmentation. Passing est.model alone would silently
    # drop every declared step -- caught on SELF, whose 2008 step made the
    # staged param_names disagree with the record's.
    step_epochs = np.asarray(est.step_epochs, dtype=float).ravel()
    if est.term_spec is not None:
        # A --term model is not a registry code (its name is
        # "polynomial+seasonal+log_transient"), so rebuild it from the spec
        # the estimate carries. Resolving the NAME here is what broke
        # --term together with --stage.
        from gps_analysis import TrajectoryModel

        fit_model = TrajectoryModel.from_spec(est.term_spec).as_modelfunc()
    else:
        base_func, _ = _resolve_model(est.model)
        fit_model = (
            with_steps(base_func, step_epochs) if step_epochs.size else base_func
        )

    # Guard the group vocabulary BEFORE fitting. Naming a group this MODEL
    # has no parameters for (e.g. "step" on a station with no declared steps,
    # or "transient" before a transient term is added) would produce an empty
    # mask and be a silent no-op -- the failure the stage grammar's refusals
    # exist to prevent. Group NAMES are validated by group_parameter_mask
    # itself against terms.GROUP_ORDER; this checks they are POPULATED here.
    from gps_analysis.staged import group_parameter_mask

    named = {g for st in plan.stages for g in st.free} | {
        g for st in plan.stages for g in st.held
    }
    empty = sorted(g for g in named if not group_parameter_mask(fit_model, g).any())
    if empty:
        from gps_analysis import GROUP_ORDER

        populated = sorted(
            g for g in GROUP_ORDER if group_parameter_mask(fit_model, g).any()
        )
        raise ValueError(
            f"stage plan names term group(s) {empty} which this model has no "
            f"parameters for, so holding or freeing them would do nothing. "
            f"Populated groups for this station's model: {populated}."
        )

    fits = []
    rms: list[float] = []
    fragment: dict[str, Any] = {}
    for c in range(inliers.shape[0]):
        keep = inliers[c]
        # Resolved PER COMPONENT: a donor hold borrows that component's
        # coefficients, so one resolution for all three would borrow north's
        # numbers into east and up.
        stages = resolve_stage_plan(plan, lookup_donor=lookup_donor, component=c)
        staged = estimate_staged(
            fit_model,
            t_win[keep],
            y_win[c][keep],
            s_win[c][keep],
            plan=stages,
            segments=settings.segments,
        )
        cov = np.asarray(staged.fits[0].covariance, dtype=float)
        if not np.all(np.isfinite(cov)):
            # A stage whose design is rank-deficient yields inf/NaN covariance
            # (_wls_solve's documented convention) and would otherwise be
            # COMMITTED as a record with no usable uncertainties. Diagnose the
            # usual cause: a column that is identically zero inside a stage's
            # own domain -- e.g. a step epoch lying outside it, which is easy
            # to write by accident because "secular" here includes the step
            # amplitudes.
            culprits = []
            for r in staged.stages:
                dom = t_win[keep]
                sub = _stage_domain_mask(dom, r, settings)
                design = _resolve_linear_design(fit_model)
                if design is None:
                    break
                cols = np.asarray(design.build(dom[sub]), dtype=float)
                for j, nm in enumerate(staged.param_names):
                    if r.free_mask[j] and not np.any(cols[:, j] != 0.0):
                        culprits.append(
                            f"{nm!r} is identically zero in stage {r.name!r}"
                        )
            detail = "; ".join(culprits) or "rank-deficient stage design"
            raise ValueError(
                f"staged fit produced a non-finite covariance for component "
                f"{c}: {detail}. A stage can only estimate parameters its own "
                f"domain constrains -- narrow the plan, or free that group in "
                f"a stage whose window contains the relevant epochs. Refusing "
                f"rather than storing a record with unusable uncertainties."
            )
        fits.append(_dc.replace(staged.fits[0], component=est.fits[c].component))
        resid = y_win[c][keep] - evaluate_record(
            {
                "record_version": 1,
                "model": est.model,
                **({} if est.term_spec is None else {"terms": est.term_spec}),
                "param_names": list(staged.param_names),
                "step_epochs": [float(v) for v in step_epochs],
                "components": [staged.fits[0].to_record()],
            },
            t_win[keep],
        )
        rms.append(float(np.sqrt(np.mean(np.asarray(resid, dtype=float) ** 2))))
        if not fragment:
            fragment = dict(staged.to_record_fragment())
    return _dc.replace(est, fits=tuple(fits), rms=tuple(rms)), fragment


def station_estimate_from_arrays(
    sta: str,
    yearf: FloatArray,
    data: FloatArray,
    sigma: FloatArray,
    *,
    settings: StationFitSettings,
    model: str = MODEL,
    frame: str = FRAME,
    steps_catalog: str | Path | None = None,
    protect_windows: str | Path | Sequence[tuple[float, float]] | None = None,
    outlier_overrides: str | Path | None = None,
    outlier_params: OutlierParams | None = None,
    fitted_at: str | None = None,
    refs: Mapping[str, Any] | None = None,
    stage_plan: Any | None = None,
    lookup_donor: Any | None = None,
    terms: Sequence[str] | None = None,
) -> StationEstimate | None:
    """As :func:`station_record_from_arrays`, keeping the fit diagnostics.

    Same estimation, same arguments, same refusals — the only difference
    is the return type: a :class:`StationEstimate` carrying the record
    AND the inlier mask lifted to the caller's index space.
    :func:`station_record_from_arrays` is a thin wrapper over this, so
    the two can never fit differently.

    Returns:
        The :class:`StationEstimate`, or None when the outlier stage
        aborted (no record is stored — design §0.4).
    """
    yearf_in = np.asarray(yearf, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    finite = _finite_mask(yearf_in, data, sigma)
    n_dropped = int(np.count_nonzero(~finite))
    yearf = yearf_in
    if n_dropped:
        yearf = yearf_in[finite]
        data = data[:, finite]
        sigma = sigma[:, finite]

    if settings.steps is not None:
        step_epochs = np.asarray(settings.steps, dtype=np.float64)
        steps_source: str | None = settings.window_source
    else:
        step_epochs, steps_source = station_step_epochs(sta, steps=steps_catalog)

    pwindows, pw_source = resolve_protect_windows(sta, protect_windows)
    resolved = resolve_outlier_detection(
        sta, outlier_params=outlier_params, outlier_overrides=outlier_overrides
    )

    fit_model: Any = model
    if terms:
        # A --term transient cannot be expressed by a registry code plus step
        # epochs, so the model becomes a composed TrajectoryModel. Its
        # as_modelfunc() is a drop-in, and the resulting record carries the
        # term spec (record version 2) so it can be read back.
        from geo_dataread.term_spec import build_trajectory_model

        traj = build_trajectory_model(
            model, [float(v) for v in np.asarray(step_epochs).ravel()], terms
        )
        if traj is not None:
            fit_model = traj.as_modelfunc()

    est = estimate_detrend(
        fit_model,
        yearf,
        data,
        sigma,
        segments=settings.segments,
        step_epochs=(None if terms else (step_epochs if step_epochs.size else None)),
        min_span_years=settings.min_span_years,
        min_epochs=settings.min_epochs,
        max_gap_years=settings.max_gap_years,
        detect=True,
        outlier_params=resolved.params,
        protect_windows=pwindows,
        min_outlier=resolved.min_outlier,
        frame=frame,
    )
    if est.outlier_abort:
        return None

    stage_fragment: dict[str, Any] | None = None
    if stage_plan is not None:
        est, stage_fragment = _restage(
            est, yearf, data, sigma, settings, stage_plan, lookup_donor
        )

    record_refs: dict[str, Any] = {
        "window_source": settings.window_source,
        "data": "local TOT",
        "steps_source": steps_source,
        "protect_windows_source": pw_source,
        "n_nonfinite_dropped": n_dropped,
        # detrend_method records "step_augmented_robust" vs "plain_wls" but
        # NOT which detection stages ran, so a stage-isolated record and a
        # full-pipeline one are otherwise indistinguishable.
        "outlier_stages": _stage_summary(resolved.params),
    }
    if refs:
        record_refs.update(refs)
    record = est.to_record(fitted_at=fitted_at, refs=record_refs)
    if stage_fragment is not None:
        record.update(stage_fragment)

    # --- lift the inlier mask back to the caller's index space -----------
    # est.inliers is (C, N_window) over the FINITE-filtered, WINDOWED epochs;
    # the caller holds (C, N_input). The mask comes from the FIT ITSELF
    # (`est.window_mask`), not from a second derivation here. The previous
    # version re-ran slice_window with what ought to be identical bounds and
    # argued exactness from a convention -- that convention would now have to
    # hold across a union of segments too, and a convention is exactly the
    # thing a seam should not depend on.
    n_in = int(yearf_in.size)
    in_win_f = np.asarray(est.window_mask, dtype=bool)
    idx_win = np.flatnonzero(finite)[in_win_f]
    inliers = np.atleast_2d(np.asarray(est.inliers, dtype=bool))
    outliers = np.zeros((inliers.shape[0], n_in), dtype=bool)
    outliers[:, idx_win] = ~inliers
    in_window = np.zeros(n_in, dtype=bool)
    in_window[idx_win] = True
    return StationEstimate(
        record=record,
        estimate=est,
        outliers=outliers,
        in_window=in_window,
        finite=finite,
    )


def estimate_station(
    sta: str,
    *,
    settings: StationFitSettings,
    tot_dir: str | None = None,
    model: str = MODEL,
    frame: str = FRAME,
    steps_catalog: str | Path | None = None,
    protect_windows: str | Path | None = None,
    outlier_overrides: str | Path | None = None,
    uncert: int = UNCERT,
    fitted_at: str | None = None,
) -> StationResult:
    """Read one station's local plate-removed TOT series and estimate it.

    The per-station driver step: :func:`geo_dataread.gps_read.getData`
    (``ref="plate"``, ``tType="TOT"``, ``Dir=tot_dir``) then
    :func:`station_record_from_arrays`. Failures are captured as an
    ``"error"`` result (the batch continues); an outlier-stage abort is a
    loud ``"skipped"`` result.

    ``uncert`` is the read-time sigma screen (see :data:`UNCERT`); it rides
    into the record's ``refs`` so a record always says which epochs it could
    have been fitted on.
    """
    from geo_dataread import gps_read  # deferred: gps_read lazily imports back

    try:
        yearf, data, ddata, _offset = gps_read.getData(  # type: ignore[no-untyped-call]
            sta, ref="plate", Dir=tot_dir, tType="TOT", uncert=uncert
        )
    except Exception as exc:  # noqa: BLE001 - per-station isolation, reported
        return StationResult(sta, "error", f"read failed: {exc}")
    if yearf is None or data is None or ddata is None:
        return StationResult(sta, "error", "no data (getData returned empty)")

    refs: dict[str, Any] = {"tot_dir": tot_dir, "uncert": uncert}
    try:
        record = station_record_from_arrays(
            sta,
            np.asarray(yearf, dtype=np.float64),
            np.asarray(data, dtype=np.float64),
            np.asarray(ddata, dtype=np.float64),
            settings=settings,
            model=model,
            frame=frame,
            steps_catalog=steps_catalog,
            protect_windows=protect_windows,
            outlier_overrides=outlier_overrides,
            fitted_at=fitted_at,
            refs=refs,
        )
    except ValueError as exc:
        return StationResult(sta, "error", str(exc))
    if record is None:
        return StationResult(
            sta,
            "skipped",
            "outlier stage aborted (excess-candidate rule) — parameters NOT "
            "stored; review the station before re-running",
        )
    rate = record["components"][0]["params"][1]
    detail = (
        f"window {record['window'][0]:.3f}-{record['window'][1]:.3f}, "
        f"{record['n_epochs']} epochs, north rate {rate:.2f} mm/yr, "
        f"method {record['detrend_method']}"
    )
    return StationResult(sta, "estimated", detail, record=record)


def _package_version(dist: str) -> str:
    """Best-effort installed version of a distribution (for provenance)."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:  # pragma: no cover - dev checkouts
        return "unknown"


def build_document(
    records: Mapping[str, Mapping[str, Any]],
    *,
    frame: str = FRAME,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble station records into the schema-v1 parameter document.

    The multi-station document of design §3.2 —
    :func:`geo_dataread.gps_views.read_detrend_params` is the reader this
    shape must satisfy (``schema_version`` + ``stations`` mapping; each
    record is the self-contained leaf shape). Units are fixed by the
    read path: ``getData(ref="plate")`` series are millimetres on
    fractional-year epochs, so displacement mm / rate mm/yr / time yearf;
    the trig phase convention is the leaf's absolute-``yearf``
    parameterization (design §7.1-7).
    """
    return {
        "schema_version": DOC_SCHEMA_VERSION,
        "frame": frame,
        "units": {"displacement": "mm", "rate": "mm/yr", "time": "yearf"},
        "phase_convention": "absolute_yearf",
        "generated_at": generated_at,
        "generator": GENERATOR,
        "software": {
            "geo_dataread": _package_version("geo-dataread"),
            "gps_analysis": _package_version("gps_analysis"),
        },
        "stations": {sta: dict(record) for sta, record in records.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_catalog(
    explicit: Path | None,
) -> tuple[dict[str, FitCatalogRow], str | None]:
    """Resolve + read the fit catalog (explicit path > deployed default).

    An EXPLICIT ``--fit-catalog`` must exist (hard error). The deployed
    default is an optional enhancement: absent -> a loud warning and
    global defaults for every station; corrupt -> hard error (estimation
    is deliberate — a typo'd catalog must stop the batch, never silently
    fall back).
    """
    if explicit is not None:
        return read_fit_catalog(explicit), str(explicit)
    resolved = default_fit_catalog_path()
    if resolved is None or not resolved.is_file():
        warnings.warn(
            "no deployed fit catalog "
            f"({FIT_CATALOG_FILENAME}); estimating every station with the "
            "global window/gate defaults",
            UserWarning,
            stacklevel=2,
        )
        return {}, None
    return read_fit_catalog(resolved), str(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: batch-estimate detrend parameters -> JSON document."""
    parser = argparse.ArgumentParser(
        prog=GENERATOR,
        description=(
            "Estimate stored-detrend trajectory parameters per station from "
            "a local plate-removed TOT directory and write the "
            "detrend_params.json document (schema v1)."
        ),
    )
    parser.add_argument("stations", nargs="+", help="4-char station codes")
    parser.add_argument(
        "--tot-dir",
        default=None,
        help="local TOT directory (default: config totDir)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("detrend_params.json"),
        help="output document path (default: ./detrend_params.json)",
    )
    parser.add_argument(
        "--fit-catalog",
        type=Path,
        default=None,
        help=(
            "per-station fit catalog (fit_windows.csv) dev override; "
            "default: the deployed catalog via gps_parser"
        ),
    )
    parser.add_argument(
        "--steps",
        type=Path,
        default=None,
        help="steps.csv dev override (default: deployed catalog)",
    )
    parser.add_argument(
        "--protect-windows",
        type=Path,
        default=None,
        help="protect_windows.csv dev override (default: deployed catalog)",
    )
    parser.add_argument(
        "--outlier-overrides",
        type=Path,
        default=None,
        help="outlier_overrides.csv dev override (default: deployed catalog)",
    )
    parser.add_argument(
        "--model", default=MODEL, help=f"trajectory model (default: {MODEL})"
    )
    parser.add_argument(
        "--min-span-years",
        type=float,
        default=FitDefaults.min_span_years,
        help="global window-span gate [yr] (catalog rows override per station)",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=FitDefaults.min_epochs,
        help="global epoch-count gate (catalog rows override per station)",
    )
    parser.add_argument(
        "--max-gap-years",
        type=float,
        default=FitDefaults.max_gap_years,
        help="global largest-gap gate [yr] (catalog rows override per station)",
    )
    parser.add_argument(
        "--uncert",
        type=int,
        default=UNCERT,
        help=(
            f"formal-sigma screen [mm] applied at read time (default: "
            f"{UNCERT}, getData's own). Lower screens harder — it changes "
            f"which epochs are fitted, so it is recorded in refs. "
            f"gps-detrend-workbench defaults to 10"
        ),
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help=(
            "embed the wall-clock timestamp (generated_at + per-record "
            "fitted_at); default output is unstamped and byte-reproducible"
        ),
    )
    args = parser.parse_args(argv)

    catalog, catalog_source = _load_catalog(args.fit_catalog)
    defaults = FitDefaults(
        min_span_years=args.min_span_years,
        min_epochs=args.min_epochs,
        max_gap_years=args.max_gap_years,
    )
    stamp = (
        datetime.now(timezone.utc).isoformat(timespec="seconds") if args.stamp else None
    )

    records: dict[str, dict[str, Any]] = {}
    n_errors = 0
    for station in args.stations:
        sta = station.upper()
        settings = resolve_fit_settings(
            sta, catalog, defaults, catalog_source=catalog_source
        )
        result = estimate_station(
            sta,
            settings=settings,
            tot_dir=args.tot_dir,
            model=args.model,
            steps_catalog=args.steps,
            protect_windows=args.protect_windows,
            outlier_overrides=args.outlier_overrides,
            uncert=args.uncert,
            fitted_at=stamp,
        )
        print(f"{sta}: [{result.status}] {result.detail}")
        if result.status == "error":
            n_errors += 1
        elif result.record is not None:
            records[sta] = result.record

    doc = build_document(records, generated_at=stamp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} station record(s) -> {args.out}")
    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
