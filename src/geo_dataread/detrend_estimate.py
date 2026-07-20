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
from gps_analysis import estimate_detrend
from gps_parser import outlier_catalogs as _oc

from geo_dataread.gps_views import (
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

FRAME = "plate_removed"
"""Reference-frame tag of the stored parameters (design §0.5: detrend runs
AFTER plate-velocity removal; ``getData(ref="plate")`` provides exactly
that frame in mm)."""

FIT_CATALOG_COLUMNS = (
    "sta",
    "window_start",
    "window_end",
    "max_gap_years",
    "min_epochs",
    "min_span_years",
    "steps",
    "comment",
)
"""Exact ``fit_windows.csv`` column set (any other layout is rejected)."""


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

    window: tuple[float | None, float | None]
    min_span_years: float
    min_epochs: int
    max_gap_years: float
    steps: tuple[float, ...] | None
    window_source: str


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
    if reader.fieldnames is None or tuple(reader.fieldnames) != FIT_CATALOG_COLUMNS:
        raise ValueError(
            f"{resolved}: fit catalog must have exactly the columns "
            f"{','.join(FIT_CATALOG_COLUMNS)!r}, got {reader.fieldnames!r}"
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
        except ValueError as exc:
            raise ValueError(f"{resolved}: {exc}") from None
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
            window=(None, None),
            min_span_years=defaults.min_span_years,
            min_epochs=defaults.min_epochs,
            max_gap_years=defaults.max_gap_years,
            steps=None,
            window_source="defaults",
        )
    source = catalog_source if catalog_source is not None else "fit_catalog"
    return StationFitSettings(
        window=(row.window_start, row.window_end),
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
    fitted_at: str | None = None,
    refs: Mapping[str, Any] | None = None,
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
    yearf = np.asarray(yearf, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    finite = _finite_mask(yearf, data, sigma)
    n_dropped = int(np.count_nonzero(~finite))
    if n_dropped:
        yearf = yearf[finite]
        data = data[:, finite]
        sigma = sigma[:, finite]

    if settings.steps is not None:
        step_epochs = np.asarray(settings.steps, dtype=np.float64)
        steps_source: str | None = settings.window_source
    else:
        step_epochs, steps_source = station_step_epochs(sta, steps=steps_catalog)

    pwindows, pw_source = resolve_protect_windows(sta, protect_windows)
    resolved = resolve_outlier_detection(sta, outlier_overrides=outlier_overrides)

    est = estimate_detrend(
        model,
        yearf,
        data,
        sigma,
        window=settings.window,
        step_epochs=step_epochs if step_epochs.size else None,
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

    record_refs: dict[str, Any] = {
        "window_source": settings.window_source,
        "data": "local TOT",
        "steps_source": steps_source,
        "protect_windows_source": pw_source,
        "n_nonfinite_dropped": n_dropped,
    }
    if refs:
        record_refs.update(refs)
    return est.to_record(fitted_at=fitted_at, refs=record_refs)


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
    fitted_at: str | None = None,
) -> StationResult:
    """Read one station's local plate-removed TOT series and estimate it.

    The per-station driver step: :func:`geo_dataread.gps_read.getData`
    (``ref="plate"``, ``tType="TOT"``, ``Dir=tot_dir``) then
    :func:`station_record_from_arrays`. Failures are captured as an
    ``"error"`` result (the batch continues); an outlier-stage abort is a
    loud ``"skipped"`` result.
    """
    from geo_dataread import gps_read  # deferred: gps_read lazily imports back

    try:
        yearf, data, ddata, _offset = gps_read.getData(  # type: ignore[no-untyped-call]
            sta, ref="plate", Dir=tot_dir, tType="TOT"
        )
    except Exception as exc:  # noqa: BLE001 - per-station isolation, reported
        return StationResult(sta, "error", f"read failed: {exc}")
    if yearf is None or data is None or ddata is None:
        return StationResult(sta, "error", "no data (getData returned empty)")

    refs: dict[str, Any] = {"tot_dir": tot_dir}
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
