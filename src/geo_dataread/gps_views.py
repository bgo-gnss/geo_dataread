"""Apply-on-read views of GPS time series: ``raw`` | ``cleaned`` | ``detrended``.

Internal-delivery slice of ``gps_analysis/docs/DESIGN_live_detrending.md``
(§0 locked decisions): `geo_dataread` reads series directly (flat files
today, DB later) and serves three *views* of the same durable record —

- ``raw``       — the series exactly as the legacy read path returns it
                  (bit-identical; raw is ALWAYS retrievable),
- ``cleaned``   — outlier flags from :func:`gps_analysis.detect_outliers`
                  (mask, never a filter: raw columns are preserved),
- ``detrended`` — raw − stored trajectory, evaluated from a versioned
                  parameter record via :func:`gps_analysis.apply_detrend`
                  (pure view, NO re-fit on read, exactly invertible).

Provenance rides on the returned DataFrame (``df.attrs["gps_view"]``):
``detrend_method`` (``"step_augmented_robust"`` | ``"plain_wls"``), frame,
record/params version, ``fitted_at``, borrowed-parameter source (the
``UseSTA`` mechanism), and the graceful-degrade state.

Graceful degrade (design §0.4, CRITICAL): any failure to clean or detrend
emits a ``UserWarning`` (plus a log record) and serves the undetrended /
unflagged series — never a hard failure, never a silent clip. The single
deliberate exception is a reference-frame mismatch between the stored
record and the series, which is a hard ``ValueError`` (design §2.5/§6 T5:
applying parameters across frames must be refused, not fudged).

Plate-first (design §0.5): detrending applies only AFTER plate-velocity
removal — stored parameters live in the plate-removed processing frame.

This module supersedes the first-draft mechanism
(`gps_read.getDetrFit`/`convconst`/`save_detrend_const` +
``detrend_itrf2008.csv``); those remain as legacy shims until their
callers are migrated (design §8 step 5).
"""

import dataclasses
import json
import math
import logging
import warnings
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from gps_analysis import OutlierParams, apply_detrend, detect_outliers
from gps_analysis import models as ga_models
from gps_parser import ConfigParser
from gps_parser import outlier_catalogs as _oc

from gtimes.timefunc import TimetoYearf

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

VIEWS = ("raw", "cleaned", "detrended")
"""Valid values of the first-class ``view`` toggle."""

DOC_SCHEMA_VERSION = 1
"""Supported ``detrend_params.json`` document schema version (design §3.2)."""

PARAMS_FILENAME = "detrend_params.json"
"""Deployed parameter-document filename (design §3.3: gpsconfig-owned)."""

# Catalog filenames + format vocab now live in the shared gps_parser resolver
# (the single source both this package and gps_api read); aliased here for
# back-compat with any importer of the old geo_dataread names.
STEPS_FILENAME = _oc.STEPS_FILENAME
"""Deployed per-station step-catalog filename (gpsconfig-owned)."""

STEP_COMPONENTS = _oc.STEP_COMPONENTS
"""Component tags a ``steps.csv`` row may carry (``ALL`` = every component)."""

PROTECT_WINDOWS_FILENAME = _oc.PROTECT_WINDOWS_FILENAME
"""Deployed per-station protect-window catalog filename (gpsconfig-owned)."""

OUTLIER_OVERRIDES_FILENAME = _oc.OUTLIER_OVERRIDES_FILENAME
"""Deployed per-station outlier-override catalog filename (gpsconfig-owned)."""

_COMPONENTS = ("north", "east", "up")


# ---------------------------------------------------------------------------
# Stored-parameter document (detrend_params.json)
# ---------------------------------------------------------------------------


def default_params_path() -> Path | None:
    """Resolve the deployed ``detrend_params.json`` path via gps_parser.

    Resolution order (design §3.3 wiring key):

    1. ``postprocess.cfg`` ``[FILES] detrend_params`` (resolved against
       the gpsconfig dir by :meth:`gps_parser.ConfigParser.getPostProcessConfig`);
    2. ``<gpsconfig dir>/detrend_params.json`` (the deploy target default).

    Returns:
        The resolved path (which may not exist yet — deployment of the
        parameter document is a separate config task), or None when no
        gpsconfig is reachable at all.
    """
    try:
        config = ConfigParser()
    except Exception:  # pragma: no cover - no gpsconfig deployed at all
        logger.warning("no gpsconfig available; cannot resolve %s", PARAMS_FILENAME)
        return None
    try:
        return Path(str(config.getPostProcessConfig("detrend_params")))
    except Exception:
        # additive key not deployed yet - fall back to the gpsconfig dir
        config_path = getattr(config, "config_path", None)
        if config_path:
            return Path(str(config_path)) / PARAMS_FILENAME
        return None


def read_detrend_params(path: str | Path | None = None) -> dict[str, Any]:
    """Read and validate the stored detrend-parameter document.

    The document is the design §3.2 station collection::

        {"schema_version": 1, ..., "stations": {"SENG": <leaf record>}}

    where each station record is the self-contained
    :meth:`gps_analysis.DetrendEstimate.to_record` shape (validated at
    apply time by the leaf). Absent station = "no background model".

    Args:
        path: Explicit document path; None resolves via
            :func:`default_params_path`.

    Returns:
        The parsed document.

    Raises:
        FileNotFoundError: When the document (or a gpsconfig to resolve
            it from) does not exist.
        ValueError: On an unknown ``schema_version`` or a document
            without a ``stations`` mapping — a reader must reject, never
            fudge (design §3.2 rules).
    """
    resolved = Path(path) if path is not None else default_params_path()
    if resolved is None:
        raise FileNotFoundError(f"no gpsconfig available to resolve {PARAMS_FILENAME}")
    if not resolved.is_file():
        raise FileNotFoundError(f"detrend parameter document not found: {resolved}")
    with open(resolved, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{resolved}: document must be a JSON object")
    version = doc.get("schema_version")
    if version != DOC_SCHEMA_VERSION:
        raise ValueError(
            f"{resolved}: unknown schema_version {version!r}; this reader "
            f"supports {DOC_SCHEMA_VERSION}"
        )
    stations = doc.get("stations")
    if not isinstance(stations, dict):
        raise ValueError(f"{resolved}: document has no 'stations' mapping")
    return doc


def station_detrend_record(
    doc: Mapping[str, Any], sta: str, use_sta: str | None = None
) -> tuple[dict[str, Any] | None, str]:
    """Look up a station's stored detrend record in a parameter document.

    ``use_sta`` is the first-class borrowing switch (design §0.6, the
    ``UseSTA`` mechanism): the named donor station's record is served
    instead of ``sta``'s own — records are self-contained, so a nearby
    pre-activity station's parameters apply cleanly to a station in an
    active area.

    Args:
        doc: Document from :func:`read_detrend_params`.
        sta: Station four-letter name.
        use_sta: Optional donor station whose record to borrow.

    Returns:
        ``(record, source_station)`` — record is None when the source
        station is absent from the document (= no background model).
    """
    source = use_sta if use_sta else sta
    stations = doc.get("stations", {})
    record = stations.get(source)
    if record is not None and not isinstance(record, Mapping):
        raise ValueError(f"station record for {source!r} is not a mapping")
    return (dict(record) if record is not None else None, source)


def _record_provenance(
    record: Mapping[str, Any] | None,
    *,
    station: str,
    params_station: str | None,
    terms: str,
) -> dict[str, Any]:
    """Build the provenance block a detrended view carries (design §0.2)."""
    prov: dict[str, Any] = {
        "station": station,
        "params_station": params_station,
        "terms": terms,
        "applied": False,
        "detrend_method": None,
        "frame": None,
        "record_version": None,
        "fitted_at": None,
        "borrowed": None,
        "degraded": False,
        "degrade_reason": None,
    }
    if record is not None:
        prov.update(
            detrend_method=record.get("detrend_method"),
            frame=record.get("frame"),
            record_version=record.get("record_version"),
            fitted_at=record.get("fitted_at"),
            borrowed=record.get("borrowed"),
        )
        if params_station is not None and params_station != station:
            # explicit read-time borrowing (use_sta) - record it even when
            # the donor record itself was the donor's own fit
            prov["borrowed"] = {
                "from": params_station,
                "terms": terms,
                "donor_fitted_at": record.get("fitted_at"),
                **(
                    {"record_borrowed": record.get("borrowed")}
                    if record.get("borrowed")
                    else {}
                ),
            }
    return prov


def _degrade(prov: dict[str, Any], reason: str) -> dict[str, Any]:
    """Mark a provenance block degraded and warn loudly (design §0.4)."""
    prov["degraded"] = True
    prov["degrade_reason"] = reason
    logger.warning("%s", reason)
    warnings.warn(reason, UserWarning, stacklevel=3)
    return prov


# ---------------------------------------------------------------------------
# Declared step catalog (feeds outlier detection)
# ---------------------------------------------------------------------------


def default_steps_path() -> Path | None:
    """Resolve the deployed ``steps.csv`` path via gps_parser.

    Resolution order (mirrors :func:`default_params_path`):

    1. ``postprocess.cfg`` ``[FILES] steps`` (resolved by
       :meth:`gps_parser.ConfigParser.getPostProcessConfig`);
    2. ``<gpsconfig dir>/steps.csv`` (the deploy-target default).

    Returns:
        The resolved path (which may not exist yet — the step catalog is
        an optional enhancement), or None when no gpsconfig is reachable.
    """
    return cast("Path | None", _oc.catalog_path("steps", _oc.STEPS_FILENAME))


def read_step_catalog(path: str | Path | None = None) -> dict[str, tuple[float, ...]]:
    """Read the deployed per-station step catalog (``steps.csv``).

    This is geo_dataread's OWN reader (Tier 1 must not import ``gps_api``);
    it parses the SAME format as
    ``gps_api.precompute.config.load_step_catalog`` for parity —
    ``sta,epoch_yearf,component,kind,source,comment`` with ``#`` comment
    lines and ``component`` in ``N|E|U|ALL``.

    Because ``detect_outliers`` takes a FLAT ``step_epochs`` array applied
    across all components (amplitudes are estimated per-component), the
    per-station value returned here is the sorted, de-duplicated UNION of
    every declared epoch for the station — a step declared for one
    component only picks up a ~0 estimated amplitude on the others
    (harmless).

    Args:
        path: Explicit catalog path; None resolves via
            :func:`default_steps_path`.

    Returns:
        ``{station: (epoch_yearf, ...)}`` — sorted unique fractional-year
        step epochs per station (only stations with rows appear).

    Raises:
        FileNotFoundError: When the catalog (or a gpsconfig to resolve it
            from) does not exist.
        ValueError: On a malformed row (missing marker, unknown component
            tag, non-numeric epoch) — a corrupt catalog is rejected, never
            silently dropped. The graceful-degrade wrapping lives in
            :func:`station_step_epochs`.
    """
    # Delegate parse + resolution to the shared gps_parser reader (the single
    # source gps_api also reads), then flatten its per-component records to the
    # flat, de-duplicated union step_epochs this package's cleaning path uses.
    records = _oc.read_steps(path)
    return {
        marker: tuple(sorted({record.epoch_yearf for record in recs}))
        for marker, recs in records.items()
    }


def station_step_epochs(
    sta: str, *, steps: str | Path | None = None
) -> tuple[FloatArray, str | None]:
    """Declared step epochs for one station, with graceful degrade.

    The graceful convenience the cleaning paths call: resolves and reads
    the catalog, returns the station's flat union step-epoch array and the
    resolved source path. Steps are an ENHANCEMENT — a missing or
    unreadable catalog (or a corrupt row) must NEVER hard-fail cleaning, so
    ANY problem warns (``UserWarning`` + log) and returns no steps.

    Args:
        sta: Station four-letter name.
        steps: Explicit catalog path; None resolves the deployed default.

    Returns:
        ``(step_epochs, source)`` — a float64 ``(K,)`` array (possibly
        empty) and the resolved catalog path (or None when unavailable /
        degraded).
    """
    empty = np.empty(0, dtype=np.float64)
    resolved = default_steps_path() if steps is None else Path(steps)
    try:
        catalog = read_step_catalog(resolved)
    except FileNotFoundError as exc:
        # common case: no catalog deployed. Generic message (no station
        # name) so the warning filter dedups it to once per run.
        warnings.warn(
            f"no step catalog ({exc}); cleaning WITHOUT declared steps — "
            "active stations may over-flag real signal",
            UserWarning,
            stacklevel=2,
        )
        logger.warning("no step catalog: %s", exc)
        return empty, None
    except (ValueError, OSError) as exc:
        warnings.warn(
            f"{sta}: step catalog unreadable ({exc}); cleaning WITHOUT declared steps",
            UserWarning,
            stacklevel=2,
        )
        logger.warning("%s: step catalog unreadable: %s", sta, exc)
        return empty, None
    epochs = np.asarray(catalog.get(sta, ()), dtype=np.float64)
    return epochs, str(resolved)


# ---------------------------------------------------------------------------
# Declared protect-window catalog (active-unrest cleaning lever)
# ---------------------------------------------------------------------------


def default_protect_windows_path() -> Path | None:
    """Resolve the deployed ``protect_windows.csv`` path via gps_parser.

    Resolution order (mirrors :func:`default_steps_path`):

    1. ``postprocess.cfg`` ``[FILES] protect_windows`` (resolved by
       :meth:`gps_parser.ConfigParser.getPostProcessConfig`);
    2. ``<gpsconfig dir>/protect_windows.csv`` (the deploy-target default).

    Returns:
        The resolved path (which may not exist yet — the protect-window
        catalog is an optional enhancement), or None when no gpsconfig is
        reachable.
    """
    return cast(
        "Path | None",
        _oc.catalog_path("protect_windows", _oc.PROTECT_WINDOWS_FILENAME),
    )


def read_protect_windows(
    path: str | Path | None = None,
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Read the deployed per-station protect-window catalog.

    Operator-declared protect windows are the active-unrest cleaning lever
    (``docs/DESIGN_outlier_detection.md`` §3.4.3): intervals the operator
    marks as "this is real signal, not outliers" so
    :func:`gps_analysis.detect_outliers` excludes them from the robust fit,
    the identifier stages AND the excess-flag abort — an unrest station
    CLEANS instead of degrading.

    This is geo_dataread's OWN reader (Tier 1 must not import ``gps_api``).
    Format: ``sta,start_yearf,end_yearf,comment`` with ``#`` comment lines.

    Args:
        path: Explicit catalog path; None resolves via
            :func:`default_protect_windows_path`.

    Returns:
        ``{station: ((start, end), ...)}`` — per-station tuple of closed
        fractional-year intervals, sorted by start (only stations with rows
        appear).

    Raises:
        FileNotFoundError: When the catalog (or a gpsconfig to resolve it
            from) does not exist.
        ValueError: On a malformed row (missing marker, non-numeric bound,
            or ``end < start``) — a corrupt catalog is rejected, never
            silently dropped. The graceful-degrade wrapping lives in
            :func:`station_protect_windows`.
    """
    # Same shape as the shared reader — direct passthrough (single source).
    return cast(
        "dict[str, tuple[tuple[float, float], ...]]",
        _oc.read_protect_windows(path),
    )


def station_protect_windows(
    sta: str, *, catalog: str | Path | None = None
) -> tuple[tuple[tuple[float, float], ...], str | None]:
    """Declared protect windows for one station, with graceful degrade.

    The graceful convenience the cleaning paths call: resolves and reads
    the catalog, returns the station's protect intervals and the resolved
    source path. Protect windows are an ENHANCEMENT — a missing or
    unreadable catalog (or a corrupt row) must NEVER hard-fail cleaning, so
    ANY problem warns (``UserWarning`` + log) and returns no windows.

    Args:
        sta: Station four-letter name.
        catalog: Explicit catalog path; None resolves the deployed default.

    Returns:
        ``(windows, source)`` — a tuple of ``(start, end)`` intervals
        (possibly empty) and the resolved catalog path (or None when
        unavailable / degraded).
    """
    resolved = default_protect_windows_path() if catalog is None else Path(catalog)
    try:
        windows_by_sta = read_protect_windows(resolved)
    except FileNotFoundError as exc:
        # common case: no catalog deployed. Generic message (no station
        # name) so the warning filter dedups it to once per run.
        warnings.warn(
            f"no protect-window catalog ({exc}); cleaning WITHOUT protect "
            "windows — active-unrest stations may over-flag real signal",
            UserWarning,
            stacklevel=2,
        )
        logger.warning("no protect-window catalog: %s", exc)
        return (), None
    except (ValueError, OSError) as exc:
        warnings.warn(
            f"{sta}: protect-window catalog unreadable ({exc}); cleaning "
            "WITHOUT protect windows",
            UserWarning,
            stacklevel=2,
        )
        logger.warning("%s: protect-window catalog unreadable: %s", sta, exc)
        return (), None
    return windows_by_sta.get(sta, ()), str(resolved)


def resolve_protect_windows(
    sta: str,
    protect_windows: str | Path | Sequence[tuple[float, float]] | None = None,
) -> tuple[tuple[tuple[float, float], ...], str | None]:
    """Normalize the ``protect_windows`` cleaning kwarg to intervals + source.

    The shared resolver both cleaning paths use so their ``protect_windows``
    kwarg behaves identically:

    - ``None`` — resolve the station's windows from the deployed catalog
      (graceful degrade on a missing/unreadable catalog);
    - ``str`` / :class:`~pathlib.Path` — resolve from the catalog at that
      path (same graceful degrade);
    - an explicit sequence of ``(start, end)`` intervals — used directly
      (source ``"explicit"``), the REPL / operator override.

    Returns:
        ``(windows, source)`` — a tuple of ``(start, end)`` intervals
        (possibly empty) and the source (catalog path, ``"explicit"``, or
        None when unavailable / degraded).
    """
    if protect_windows is None:
        return station_protect_windows(sta)
    if isinstance(protect_windows, (str, Path)):
        return station_protect_windows(sta, catalog=protect_windows)
    windows = tuple((float(a), float(b)) for a, b in protect_windows)
    return windows, "explicit"


# ---------------------------------------------------------------------------
# Declared per-station outlier-parameter overrides (stronger detection levers)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OutlierOverride:
    """One station's parsed ``outlier_overrides.csv`` row.

    Splits the two kinds of override the catalog carries:

    - ``params_fields`` — :class:`gps_analysis.OutlierParams` field values
      (``despike``, ``window_order``, …), ready for
      :func:`dataclasses.replace`;
    - ``min_outlier`` — the PER-COMPONENT magnitude floor ``[N, E, U]`` that
      goes to the ``detect_outliers`` ``min_outlier`` kwarg (a separate array
      from the scalar ``OutlierParams.min_outlier``), or None.
    """

    params_fields: dict[str, object]
    min_outlier: tuple[float, float, float] | None


@dataclasses.dataclass(frozen=True)
class ResolvedOutlierConfig:
    """Fully-resolved outlier-detection inputs for one station read.

    The single object both cleaning paths get back from
    :func:`resolve_outlier_detection`, after applying the
    explicit-arg > catalog > default precedence.
    """

    params: OutlierParams
    min_outlier: tuple[float, float, float] | None
    overrides_applied: dict[str, object]
    overrides_source: str | None
    min_outlier_source: str | None


def default_outlier_overrides_path() -> Path | None:
    """Resolve the deployed ``outlier_overrides.csv`` path via gps_parser.

    Resolution order (mirrors :func:`default_steps_path`):

    1. ``postprocess.cfg`` ``[FILES] outlier_overrides`` (resolved by
       :meth:`gps_parser.ConfigParser.getPostProcessConfig`);
    2. ``<gpsconfig dir>/outlier_overrides.csv`` (the deploy-target default).

    Returns:
        The resolved path (which may not exist yet — the override catalog is
        an optional enhancement), or None when no gpsconfig is reachable.
    """
    return cast(
        "Path | None",
        _oc.catalog_path("outlier_overrides", _oc.OUTLIER_OVERRIDES_FILENAME),
    )


def read_outlier_overrides(
    path: str | Path | None = None,
) -> dict[str, OutlierOverride]:
    """Read the deployed per-station outlier-parameter override catalog.

    Lets an operator enable the stronger detection levers PER STATION for
    active/unrest stations (Stage-0 despike, robust local-polynomial
    identifier ``window_order=1``, ``epoch_policy="union"``) while the global
    default stays conservative order-0 — zero regression for quiet stations.

    This is geo_dataread's OWN reader (Tier 1 must not import ``gps_api``).
    Columns (all except ``sta`` optional; blank = leave at the base default;
    a ``comment`` column is allowed and ignored)::

        sta,despike,window_order,window_robust_iterations,epoch_policy,
        despike_n_sigma,min_outlier_n,min_outlier_e,min_outlier_u

    Args:
        path: Explicit catalog path; None resolves via
            :func:`default_outlier_overrides_path`.

    Returns:
        ``{station: OutlierOverride}`` — per station, the supplied
        OutlierParams field overrides plus the per-component ``min_outlier``
        floor. Only stations with rows appear.

    Raises:
        FileNotFoundError: When the catalog (or a gpsconfig to resolve it
            from) does not exist.
        ValueError: On an unknown column, a duplicate station row, a missing
            marker, a bad enum (``window_order``/``epoch_policy``), or a
            non-numeric / negative field — a corrupt catalog is rejected,
            never silently dropped. The graceful-degrade wrapping lives in
            :func:`station_outlier_params`.
    """
    # Delegate parse + resolution to the shared gps_parser reader (single
    # source), then adapt its StationOutlierOverride to this package's
    # OutlierOverride (identical split; the ``.params_fields`` name is kept for
    # the existing public API + provenance callers).
    return {
        marker: OutlierOverride(
            params_fields=override.fields, min_outlier=override.min_outlier
        )
        for marker, override in _oc.read_outlier_overrides(path).items()
    }


def station_outlier_params(
    sta: str, *, base: OutlierParams | None = None, catalog: str | Path | None = None
) -> tuple[OutlierParams, tuple[float, float, float] | None, str | None]:
    """Per-station outlier params + floor, with catalog overrides + degrade.

    Starts from ``base`` (or :class:`gps_analysis.OutlierParams` defaults) and
    applies the station's catalog overrides via :func:`dataclasses.replace`,
    and separately returns the per-component ``min_outlier`` floor (a distinct
    ``detect_outliers`` kwarg, NOT an OutlierParams field). Overrides are an
    ENHANCEMENT — a missing / unreadable / corrupt catalog must NEVER
    hard-fail cleaning, so ANY problem warns (``UserWarning`` + log, deduped
    once) and returns the base with no floor.

    Args:
        sta: Station four-letter name.
        base: Base parameters to override; None = spec defaults.
        catalog: Explicit catalog path; None resolves the deployed default.

    Returns:
        ``(params, min_outlier, source)`` — the resolved parameters, the
        station's per-component floor ``[N, E, U]`` (or None), and the catalog
        path (or None when unavailable / degraded). The base is returned
        unchanged with no floor when the station has no override row.
    """
    default = base if base is not None else OutlierParams()
    resolved = default_outlier_overrides_path() if catalog is None else Path(catalog)
    try:
        overrides_by_sta = read_outlier_overrides(resolved)
    except FileNotFoundError as exc:
        # common case: no catalog deployed. Generic message (no station
        # name) so the warning filter dedups it to once per run.
        warnings.warn(
            f"no outlier-override catalog ({exc}); cleaning with the base "
            "OutlierParams (conservative default levers)",
            UserWarning,
            stacklevel=2,
        )
        logger.warning("no outlier-override catalog: %s", exc)
        return default, None, None
    except (ValueError, OSError) as exc:
        warnings.warn(
            f"{sta}: outlier-override catalog unreadable ({exc}); cleaning "
            "with the base OutlierParams",
            UserWarning,
            stacklevel=2,
        )
        logger.warning("%s: outlier-override catalog unreadable: %s", sta, exc)
        return default, None, None
    override = overrides_by_sta.get(sta)
    if override is None:
        return default, None, str(resolved)
    params = default
    if override.params_fields:
        # values are per-field OutlierParams types (validated in the reader);
        # object-typed for a permissive public API, so narrow for replace.
        params = dataclasses.replace(
            default, **cast("dict[str, Any]", override.params_fields)
        )
    return params, override.min_outlier, str(resolved)


def outlier_override_delta(
    params: OutlierParams, base: OutlierParams | None = None
) -> dict[str, object]:
    """Fields where ``params`` differs from ``base`` (for provenance).

    Since :func:`station_outlier_params` only changes the operator-supplied
    fields, this delta is exactly the applied overrides — an empty dict means
    the base was used unchanged.
    """
    reference = base if base is not None else OutlierParams()
    return {
        f.name: getattr(params, f.name)
        for f in dataclasses.fields(params)
        if getattr(params, f.name) != getattr(reference, f.name)
    }


def _normalize_min_outlier(
    min_outlier: float | Sequence[float],
) -> tuple[float, float, float]:
    """Coerce a scalar or ``[N, E, U]`` floor to a validated 3-tuple.

    A scalar broadcasts to all three components. Values must be finite and
    ``>= 0`` (matching the leaf ``detect_outliers`` contract) — a bad value is
    a hard ``ValueError`` (an explicit floor the caller asked for, not
    optional config).
    """
    arr = np.atleast_1d(np.asarray(min_outlier, dtype=np.float64))
    if arr.size == 1:
        arr = np.full(3, float(arr[0]), dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(
            f"min_outlier must be a scalar or length-3 [N,E,U] sequence, got "
            f"shape {arr.shape}"
        )
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError("min_outlier must be finite and >= 0")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def resolve_outlier_detection(
    sta: str,
    *,
    outlier_params: OutlierParams | None = None,
    min_outlier: float | Sequence[float] | None = None,
    outlier_overrides: str | Path | None = None,
) -> ResolvedOutlierConfig:
    """Resolve the detection params + per-component floor for one station.

    The shared resolver both cleaning paths use, applying the precedence:

    - **params:** explicit ``outlier_params`` arg > catalog override >
      ``OutlierParams()`` default;
    - **min_outlier:** explicit ``min_outlier`` arg > catalog
      ``min_outlier_{n,e,u}`` > None (leaf falls back to
      ``params.min_outlier``).

    The two are INDEPENDENT: an explicit ``outlier_params`` bypasses the
    catalog for the params only — the catalog's per-station ``min_outlier``
    still applies unless an explicit ``min_outlier`` arg is also given. When
    BOTH are explicit the catalog is not consulted at all (no spurious
    "no catalog" warning).

    Returns:
        A :class:`ResolvedOutlierConfig` with the resolved params, the
        per-component floor (or None), the applied param-override delta (for
        provenance), and the params / floor sources.
    """
    explicit_floor = (
        None if min_outlier is None else _normalize_min_outlier(min_outlier)
    )

    # short-circuit: neither params nor floor needs the catalog
    if outlier_params is not None and explicit_floor is not None:
        return ResolvedOutlierConfig(
            params=outlier_params,
            min_outlier=explicit_floor,
            overrides_applied={},
            overrides_source=None,
            min_outlier_source="explicit",
        )

    cat_params, cat_floor, cat_source = station_outlier_params(
        sta, catalog=outlier_overrides
    )

    if outlier_params is not None:
        params = outlier_params
        overrides_applied: dict[str, object] = {}
        overrides_source: str | None = None
    else:
        params = cat_params
        overrides_applied = outlier_override_delta(params)
        overrides_source = cat_source

    if explicit_floor is not None:
        floor = explicit_floor
        floor_source: str | None = "explicit"
    elif cat_floor is not None:
        floor = cat_floor
        floor_source = cat_source
    else:
        floor = None
        floor_source = None

    return ResolvedOutlierConfig(
        params=params,
        min_outlier=floor,
        overrides_applied=overrides_applied,
        overrides_source=overrides_source,
        min_outlier_source=floor_source,
    )


# ---------------------------------------------------------------------------
# Outlier flags (cleaned view)
# ---------------------------------------------------------------------------


#: Default recency bound of the provisional mask [days].  Generous next to
#: the ~3 trailing epochs the step statistic genuinely cannot rule on (it
#: needs 3 post-samples), because the trajectory fit and robust scale also
#: re-estimate as epochs arrive: a verdict within a couple of weeks of the
#: end can still change without any flank being involved.
PROVISIONAL_DAYS: float = 14.0

_DAYS_PER_YEAR: float = 365.25


def _provisional_mask(
    detection: Any,
    t: npt.NDArray[np.float64],
    finite: BoolArray,
    shape: tuple[int, ...],
    *,
    provisional_days: float,
) -> BoolArray:
    """Recent candidates protected on INDETERMINATE step evidence.

    ``detection.suspected_events`` carries the per-cluster step-evidence
    statistic, and ``NaN`` there is precisely "no usable post-flank, so a
    step could not be ruled out" — the leaf already computes and reports
    it, nothing new is derived here.  Cluster indices address the FINITE
    subset detection ran on, so they are mapped back through ``finite``;
    the span is intersected with the cluster's candidates so that
    non-candidate epochs inside a cluster's bounds are not marked.

    The mask is DIAGNOSTIC, so a detection object without the fields it
    needs yields an empty mask rather than an exception — the same
    graceful-degrade rule the rest of this path follows (design §0.4).
    A caller may legitimately supply a reduced detection result, and no
    plot or write should fail over an annotation.
    """
    provisional = np.zeros(shape, dtype=np.bool_)
    if provisional_days <= 0.0:
        return provisional
    events = getattr(detection, "suspected_events", None)
    raw_candidates = getattr(detection, "candidates", None)
    if not events or raw_candidates is None:
        return provisional
    t_fin = t[finite]
    if t_fin.size == 0:
        return provisional

    cutoff = float(t_fin[-1]) - provisional_days / _DAYS_PER_YEAR
    index = np.flatnonzero(finite)
    candidates = np.atleast_2d(raw_candidates)
    for event in events:
        if not math.isnan(event.step_evidence):
            continue  # a MEASURED step, not an unrulable one
        if float(event.t_start) < cutoff:
            continue  # old news: a mid-series gap, not the series end
        span = np.zeros(t_fin.size, dtype=np.bool_)
        span[event.i_start : event.i_end + 1] = True
        span &= candidates[event.component]
        provisional[event.component, index[span]] = True
    return provisional


def detect_view_outliers(
    yearf: npt.ArrayLike,
    data: npt.ArrayLike,
    Ddata: npt.ArrayLike | None = None,
    *,
    outlier_params: OutlierParams | None = None,
    step_epochs: npt.ArrayLike | None = None,
    protect_windows: tuple[tuple[float, float], ...] = (),
    min_outlier: npt.ArrayLike | None = None,
    provisional_days: float = PROVISIONAL_DAYS,
) -> tuple[BoolArray, dict[str, Any]]:
    """Outlier flags for a series view, with graceful degrade.

    Wraps :func:`gps_analysis.detect_outliers` (model-aware, signal-
    protecting detection against a robust ``lineperiodic`` fit) for the
    read path: non-finite epochs are excluded from detection (and never
    flagged), and ANY detection failure — including the leaf's
    excess-candidate abort — degrades to all-False flags with a
    ``UserWarning`` instead of failing the read (design §0.4).

    Args:
        yearf: Epochs, fractional years, shape (N,).
        data: Observations, shape (C, N) or (N,) [caller's unit].
        Ddata: Formal 1-σ uncertainties, shape of ``data``; optional.
        outlier_params: :class:`gps_analysis.OutlierParams` thresholds;
            None = spec defaults.
        step_epochs: Known step epochs [yr] to augment the model with.
        protect_windows: Intervals [yr] where flagging is disabled.
        min_outlier: Outlier magnitude floor(s) [caller's unit].
        provisional_days: Recency bound of the PROVISIONAL mask [d]; 0
            disables it.  See the mask's definition below.

    Returns:
        ``(flags, provenance)`` — flags shaped like ``data`` (True =
        outlier; MASK only, nothing is removed) and a provenance dict
        with ``outlier_abort`` / ``degraded`` / ``degrade_reason`` /
        ``n_flagged`` / ``provisional`` / ``n_provisional``.

    **The provisional mask.**  ``provenance["provisional"]`` is shaped
    like ``flags`` and marks epochs the identifiers flagged but that were
    protected because the step evidence was INDETERMINATE (``D`` NaN — no
    usable post-flank), *and* which lie within ``provisional_days`` of the
    last epoch.  These are the epochs the detector genuinely cannot rule
    on yet: a blunder and the onset of real deformation look identical
    until data follows them.

    It is disjoint from ``flags`` by construction (a protected candidate
    is not flagged) and is DIAGNOSTIC — the series is unchanged, so a
    caller that ignores it behaves exactly as before.  The recency bound
    is what makes it useful: indeterminate clusters also occur at
    mid-series gaps wider than ``step_flank_max_reach_days``, which are
    old news and would otherwise dominate the mask (measured: RHOF has 3
    indeterminate clusters, at 323 / 3400 / 3400 days from the end).

    Verdicts inside the window are UNSTABLE by nature — they resolve as
    epochs accumulate, and the trajectory fit re-estimates too.
    """
    t = np.asarray(yearf, dtype=np.float64)
    y = np.asarray(data, dtype=np.float64)
    y2d = y if y.ndim == 2 else y[np.newaxis, :]
    sigma = None if Ddata is None else np.asarray(Ddata, dtype=np.float64)
    sigma2d = (
        None if sigma is None else (sigma if sigma.ndim == 2 else sigma[np.newaxis, :])
    )

    flags = np.zeros(y.shape, dtype=np.bool_)
    prov: dict[str, Any] = {
        "outlier_abort": False,
        "degraded": False,
        "degrade_reason": None,
        "n_flagged": 0,
        "provisional": np.zeros(y.shape, dtype=np.bool_),
        "n_provisional": 0,
    }

    finite = np.isfinite(t) & np.all(np.isfinite(y2d), axis=0)
    if sigma2d is not None:
        finite &= np.all(np.isfinite(sigma2d) & (sigma2d > 0.0), axis=0)

    try:
        detection = detect_outliers(
            ga_models.lineperiodic,
            t[finite],
            y2d[:, finite],
            None if sigma2d is None else sigma2d[:, finite],
            step_epochs=step_epochs,
            protect_windows=protect_windows,
            min_outlier=min_outlier,
            params=outlier_params,
            names=list(_COMPONENTS) if y2d.shape[0] == 3 else None,
        )
    except Exception as exc:  # never fail the read (design §0.4)
        _degrade(prov, f"outlier detection failed ({exc}); serving unflagged data")
        return flags, prov

    if detection.excess_flag_abort:
        prov["outlier_abort"] = True
        _degrade(
            prov,
            "outlier detection aborted (excess-candidate rule); serving unflagged data",
        )
        return flags, prov

    full = np.zeros(y2d.shape, dtype=np.bool_)
    full[:, finite] = np.atleast_2d(detection.flags)
    flags = full[0] if y.ndim == 1 else full
    prov["n_flagged"] = int(np.count_nonzero(flags))

    provisional = _provisional_mask(
        detection, t, finite, y2d.shape, provisional_days=provisional_days
    )
    prov["provisional"] = provisional[0] if y.ndim == 1 else provisional
    prov["n_provisional"] = int(np.count_nonzero(provisional))
    return flags, prov


# ---------------------------------------------------------------------------
# Stored-parameter detrending (detrended view)
# ---------------------------------------------------------------------------


def apply_stored_detrend(
    record: Mapping[str, Any],
    yearf: npt.ArrayLike,
    data: npt.ArrayLike,
    *,
    terms: str = "all",
    frame: str | None = None,
    data_unit: str = "mm",
) -> FloatArray:
    """Subtract a stored trajectory record from a series (pure view).

    Thin unit-aware shim over :func:`gps_analysis.apply_detrend` — no
    re-fit, valid at ANY epoch including epochs newer than the fit
    window and epochs of a different station (borrowed records). Raises
    on any problem; the graceful-degrade wrapping lives in
    :func:`detrend_arrays` / :func:`read_gps_view`.

    Args:
        record: Self-contained station record (leaf ``to_record`` shape).
        yearf: Epochs, fractional years, shape (N,).
        data: Observations, shape (C, N) or (N,).
        terms: ``"all"`` | ``"secular"`` | ``"periodic"`` — partial views
            from the SAME stored parameters (design §4.2).
        frame: Series frame tag; a mismatch with the record's frame is a
            hard error (design §2.5 — refuse, don't fudge).
        data_unit: ``"mm"`` (production record unit) or ``"m"`` (the .NEU
            meter path); the record is evaluated in mm and scaled.

    Returns:
        Detrended series, float64, new array shaped like ``data``.
    """
    if data_unit not in ("mm", "m"):
        raise ValueError(f"data_unit must be 'mm' or 'm', got {data_unit!r}")
    y = np.asarray(data, dtype=np.float64)
    if data_unit == "m":
        detrended_mm = apply_detrend(
            record, yearf, y * 1000.0, terms=terms, frame=frame
        )
        return np.asarray(detrended_mm / 1000.0, dtype=np.float64)
    return np.asarray(
        apply_detrend(record, yearf, y, terms=terms, frame=frame), dtype=np.float64
    )


def detrend_arrays(
    sta: str,
    yearf: npt.ArrayLike,
    data: npt.ArrayLike,
    *,
    params: str | Path | Mapping[str, Any] | None = None,
    use_sta: str | None = None,
    terms: str = "all",
    frame: str | None = None,
    data_unit: str = "mm",
) -> tuple[FloatArray, dict[str, Any]]:
    """Detrended view of a plate-removed series, with graceful degrade.

    Array-level entry used by the legacy array paths
    (:func:`geo_dataread.gps_read.getData` ``ref="detrend"`` and the
    ``.NEU`` writer) and by :func:`read_gps_view`. Loads the station's
    stored record (own or borrowed via ``use_sta``), applies it purely,
    and on ANY failure other than a frame mismatch warns and returns the
    input series unchanged, with the provenance marking the skip
    (design §0.4). The input is never mutated.

    Args:
        sta: Station four-letter name.
        yearf: Epochs, fractional years, shape (N,).
        data: Plate-removed observations, shape (C, N) or (N,).
        params: Parameter document — a path, an already-loaded document
            mapping, or None for the deployed default.
        use_sta: Borrow this station's stored record (``UseSTA``).
        terms: ``"all"`` | ``"secular"`` | ``"periodic"``.
        frame: Series frame tag (mismatch = hard error).
        data_unit: ``"mm"`` or ``"m"`` (see :func:`apply_stored_detrend`).

    Returns:
        ``(series, provenance)`` — the detrended series (or the input,
        unchanged, when degraded) and the provenance block.

    Raises:
        ValueError: On a record/series reference-frame mismatch only.
    """
    y = np.asarray(data, dtype=np.float64)
    record: dict[str, Any] | None = None
    source: str | None = None
    try:
        if isinstance(params, Mapping):
            doc: Mapping[str, Any] = params
        else:
            doc = read_detrend_params(params)
        record, source = station_detrend_record(doc, sta, use_sta=use_sta)
    except (FileNotFoundError, ValueError) as exc:
        prov = _record_provenance(
            None, station=sta, params_station=use_sta, terms=terms
        )
        _degrade(prov, f"{sta}: no detrend parameters ({exc}); serving raw series")
        return y, prov

    prov = _record_provenance(record, station=sta, params_station=source, terms=terms)
    if record is None:
        _degrade(
            prov,
            f"{sta}: station {source!r} absent from the detrend parameter "
            "document (no background model); serving raw series",
        )
        return y, prov

    # frame integrity is a hard refusal, not a degrade (design §2.5/T5)
    record_frame = record.get("frame")
    if frame is not None and record_frame is not None and record_frame != frame:
        raise ValueError(
            f"frame mismatch: record frame {record_frame!r} != series frame "
            f"{frame!r} - refusing to apply (design §2.5)"
        )

    try:
        detrended = apply_stored_detrend(
            record, yearf, y, terms=terms, frame=frame, data_unit=data_unit
        )
    except Exception as exc:
        _degrade(prov, f"{sta}: detrend application failed ({exc}); serving raw series")
        return y, prov

    prov["applied"] = True
    return detrended, prov


# ---------------------------------------------------------------------------
# First-class toggle: the DataFrame read API
# ---------------------------------------------------------------------------


def _to_yearf(value: datetime | date | float | None) -> float | None:
    """Coerce a date/datetime/fractional-year bound to fractional years."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return float(TimetoYearf(value.year, value.month, value.day))
    return float(value)


def read_gps_view(
    sta: str,
    start: datetime | date | float | None = None,
    end: datetime | date | float | None = None,
    *,
    view: str = "raw",
    ref: str = "plate",
    terms: str = "all",
    clean: bool | None = None,
    params: str | Path | Mapping[str, Any] | None = None,
    use_sta: str | None = None,
    outlier_params: OutlierParams | None = None,
    min_outlier: float | Sequence[float] | None = None,
    steps: str | Path | None = None,
    protect_windows: str | Path | Sequence[tuple[float, float]] | None = None,
    outlier_overrides: str | Path | None = None,
    frame: str | None = None,
    Dir: str | None = None,
    tType: str = "TOT",
    uncert: float = 15,
) -> pd.DataFrame:
    """Read one station's GPS series in a chosen view (the toggle).

    THE first-class raw↔cleaned↔detrended switch of the internal
    delivery path (design §0 locked decision 3) — one kwarg away in a
    nvim/terminal/REPL session::

        df = read_gps_view("SENG")                      # raw (default)
        df = read_gps_view("SENG", view="cleaned")      # + outlier flags
        df = read_gps_view("SENG", view="detrended")    # + stored-params view
        df.attrs["gps_view"]                            # provenance

    The raw columns are ALWAYS present and bit-identical to the legacy
    read (``convGlobktopandas(*getData(...))``); views only ADD columns:

    - ``cleaned``:  ``{north,east,up}_outlier`` (bool, True = flagged)
      and ``{north,east,up}_cleaned`` (raw with flagged epochs NaN),
    - ``detrended``: ``{north,east,up}_detrended`` (raw − stored
      trajectory; pure apply of the deployed parameter record, no
      re-fit on read).

    Graceful degrade (design §0.4): a missing/invalid parameter record
    or a failed detection warns (``UserWarning`` + log) and serves the
    raw columns with ``attrs["gps_view"]["degraded"]`` set — the read
    never hard-fails for a view reason. Exception: a reference-frame
    mismatch raises (design §2.5).

    Args:
        sta: Station four-letter name.
        start: Window start — datetime/date or fractional year.
        end: Window end — datetime/date or fractional year.
        view: ``"raw"`` (default) | ``"cleaned"`` | ``"detrended"``.
        ref: Underlying series reference: ``"plate"`` (default,
            plate-velocity removed — required for ``view="detrended"``,
            design §0.5 plate-first) | ``"itrf2008"`` | explicit plate
            name. ``"detrend"`` is rejected here — that array-path alias
            maps to ``view="detrended"``.
        terms: Detrended view term selection: ``"all"`` | ``"secular"``
            | ``"periodic"`` (same stored params, design §4.2).
        clean: Explicit outlier-flag switch; None = implied by ``view``
            (True for ``"cleaned"``). ``clean=True`` combines with
            ``view="detrended"`` for the cleaned-and-detrended view.
        params: Detrend parameter document (path or loaded mapping);
            None = deployed default.
        use_sta: Borrow the named station's stored parameters (UseSTA).
        outlier_params: Detection thresholds. An EXPLICIT value wins over
            the per-station override catalog (REPL override); None resolves
            per-station via ``outlier_overrides`` then the spec defaults.
        min_outlier: Per-component magnitude floor ``[N, E, U]`` (or a scalar
            broadcast to all three) routed to ``detect_outliers`` — a
            candidate below its component floor is NOT flagged (active
            stations use e.g. ``[5, 5, 10]`` mm, U noisier). An EXPLICIT value
            wins over the catalog ``min_outlier_{n,e,u}``; None resolves from
            the catalog then the leaf default. Independent of
            ``outlier_params`` (it is a separate leaf kwarg, not an
            OutlierParams field).
        outlier_overrides: Per-station outlier-parameter override catalog
            path (``outlier_overrides.csv``) — enables the stronger levers
            (despike, ``window_order=1``, ``epoch_policy="union"``) for
            active stations. None = deployed default; missing / unreadable
            degrades gracefully (warn + base OutlierParams). Ignored when
            ``outlier_params`` is passed explicitly.
        steps: Declared step catalog path (``steps.csv``) fed to outlier
            detection so the trajectory model absorbs known offsets instead
            of over-flagging them; None = deployed default. A missing /
            unreadable catalog degrades gracefully (warn + no steps).
        protect_windows: Active-unrest cleaning lever — operator-declared
            intervals excluded from the fit, the identifiers AND the
            excess-flag abort, so an unrest station CLEANS instead of
            degrading. A catalog path (``protect_windows.csv``), an explicit
            sequence of ``(start, end)`` fractional-year intervals, or None
            (deployed default). A missing / unreadable catalog degrades
            gracefully (warn + no windows).
        frame: Series frame tag for the record integrity check.
        Dir: Series directory override (as :func:`gps_read.getData`).
        tType: GLOBK scheme (as :func:`gps_read.getData`).
        uncert: Maximum formal uncertainty kept [mm] (as ``getData``).

    Returns:
        DataFrame indexed by datetime with the legacy columns
        (``north/east/up``, ``Dnorth/Deast/Dup``, ``yearf``) plus the
        view columns, and provenance in ``df.attrs["gps_view"]``.

    Raises:
        ValueError: On an unknown ``view``, ``ref="detrend"``, a
            detrended view requested on an ITRF series (plate-first
            rule), no data for the station, or a frame mismatch.
    """
    from geo_dataread import gps_read  # deferred: gps_read lazily imports back

    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    if ref == "detrend":
        raise ValueError(
            "ref='detrend' is the legacy array-path alias; use view='detrended'"
        )
    if view == "detrended" and ref == "itrf2008":
        raise ValueError(
            "view='detrended' requires a plate-removed series (ref='plate' or "
            "an explicit plate name): detrend parameters live in the "
            "plate-removed processing frame (design §0.5, plate-first)"
        )

    yearf, data, Ddata, _offset = gps_read.getData(  # type: ignore[no-untyped-call]
        sta,
        fstart=_to_yearf(start),
        fend=_to_yearf(end),
        ref=ref,
        Dir=Dir,
        tType=tType,
        uncert=uncert,
    )
    if yearf is None or len(yearf) == 0:
        raise ValueError(f"no data for station {sta}")

    df = gps_read.convGlobktopandas(  # type: ignore[no-untyped-call]
        yearf, data, Ddata
    )

    attrs: dict[str, Any] = {
        "station": sta,
        "view": view,
        "ref": ref,
        "terms": terms,
        "clean": bool(clean) if clean is not None else view == "cleaned",
        "params_station": None,
        "detrend_method": None,
        "frame": None,
        "record_version": None,
        "fitted_at": None,
        "borrowed": None,
        "degraded": False,
        "degrade_reason": None,
        "outlier_abort": False,
        "n_flagged": 0,
        "step_epochs_applied": 0,
        "steps_source": None,
        "protect_windows_applied": 0,
        "protect_windows_source": None,
        "outlier_overrides_applied": {},
        "outlier_overrides_source": None,
        "min_outlier": None,
        "min_outlier_source": None,
    }

    do_clean = attrs["clean"]
    if do_clean:
        step_epochs, steps_source = station_step_epochs(sta, steps=steps)
        attrs["step_epochs_applied"] = int(step_epochs.size)
        attrs["steps_source"] = steps_source
        pwindows, pw_source = resolve_protect_windows(sta, protect_windows)
        attrs["protect_windows_applied"] = len(pwindows)
        attrs["protect_windows_source"] = pw_source
        # precedence: explicit arg > catalog override > default, resolved for
        # BOTH the params and the (independent) per-component min_outlier floor
        resolved = resolve_outlier_detection(
            sta,
            outlier_params=outlier_params,
            min_outlier=min_outlier,
            outlier_overrides=outlier_overrides,
        )
        attrs["outlier_overrides_source"] = resolved.overrides_source
        attrs["outlier_overrides_applied"] = resolved.overrides_applied
        attrs["min_outlier"] = (
            list(resolved.min_outlier) if resolved.min_outlier is not None else None
        )
        attrs["min_outlier_source"] = resolved.min_outlier_source
        flags, oprov = detect_view_outliers(
            yearf,
            data,
            Ddata,
            outlier_params=resolved.params,
            step_epochs=step_epochs if step_epochs.size else None,
            protect_windows=pwindows,
            min_outlier=resolved.min_outlier,
        )
        flags2d = np.atleast_2d(flags)
        for c, name in enumerate(_COMPONENTS):
            df[f"{name}_outlier"] = flags2d[c]
            df[f"{name}_cleaned"] = np.where(flags2d[c], np.nan, data[c])
        attrs["outlier_abort"] = oprov["outlier_abort"]
        attrs["n_flagged"] = oprov["n_flagged"]
        if oprov["degraded"]:
            attrs["degraded"] = True
            attrs["degrade_reason"] = oprov["degrade_reason"]

    if view == "detrended":
        detrended, dprov = detrend_arrays(
            sta,
            yearf,
            data,
            params=params,
            use_sta=use_sta,
            terms=terms,
            frame=frame,
            data_unit="mm",
        )
        attrs.update(
            params_station=dprov["params_station"],
            detrend_method=dprov["detrend_method"],
            frame=dprov["frame"],
            record_version=dprov["record_version"],
            fitted_at=dprov["fitted_at"],
            borrowed=dprov["borrowed"],
        )
        if dprov["degraded"]:
            attrs["degraded"] = True
            attrs["degrade_reason"] = dprov["degrade_reason"]
        if dprov["applied"]:
            for c, name in enumerate(_COMPONENTS):
                df[f"{name}_detrended"] = detrended[c]

    df.attrs["gps_view"] = attrs
    return df
