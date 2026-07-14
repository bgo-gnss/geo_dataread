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

import csv
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from gps_analysis import OutlierParams, apply_detrend, detect_outliers
from gps_analysis import models as ga_models
from gps_parser import ConfigParser

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

STEPS_FILENAME = "steps.csv"
"""Deployed per-station step-catalog filename (gpsconfig-owned)."""

STEP_COMPONENTS = ("N", "E", "U", "ALL")
"""Component tags a ``steps.csv`` row may carry (``ALL`` = every component)."""

PROTECT_WINDOWS_FILENAME = "protect_windows.csv"
"""Deployed per-station protect-window catalog filename (gpsconfig-owned)."""

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
    try:
        config = ConfigParser()
    except Exception:  # pragma: no cover - no gpsconfig deployed at all
        logger.warning("no gpsconfig available; cannot resolve %s", STEPS_FILENAME)
        return None
    try:
        return Path(str(config.getPostProcessConfig("steps")))
    except Exception:
        # additive key not deployed yet - fall back to the gpsconfig dir
        config_path = getattr(config, "config_path", None)
        if config_path:
            return Path(str(config_path)) / STEPS_FILENAME
        return None


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
    resolved = Path(path) if path is not None else default_steps_path()
    if resolved is None:
        raise FileNotFoundError(f"no gpsconfig available to resolve {STEPS_FILENAME}")
    if not resolved.is_file():
        raise FileNotFoundError(f"step catalog not found: {resolved}")
    lines = [
        line
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    catalog: dict[str, set[float]] = {}
    for row in csv.DictReader(lines):
        marker = str(row.get("sta") or "").strip()
        if not marker:
            raise ValueError(f"{resolved}: steps.csv row without a 'sta' marker: {row}")
        component = str(row.get("component") or "ALL").strip().upper()
        if component not in STEP_COMPONENTS:
            raise ValueError(
                f"{resolved}: station {marker}: component {component!r} — "
                f"must be one of {STEP_COMPONENTS}"
            )
        try:
            epoch = float(str(row.get("epoch_yearf")))
        except (TypeError, ValueError):
            raise ValueError(
                f"{resolved}: station {marker}: epoch_yearf "
                f"{row.get('epoch_yearf')!r} is not a fractional year"
            ) from None
        catalog.setdefault(marker, set()).add(epoch)
    return {marker: tuple(sorted(epochs)) for marker, epochs in catalog.items()}


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
    try:
        config = ConfigParser()
    except Exception:  # pragma: no cover - no gpsconfig deployed at all
        logger.warning(
            "no gpsconfig available; cannot resolve %s", PROTECT_WINDOWS_FILENAME
        )
        return None
    try:
        return Path(str(config.getPostProcessConfig("protect_windows")))
    except Exception:
        # additive key not deployed yet - fall back to the gpsconfig dir
        config_path = getattr(config, "config_path", None)
        if config_path:
            return Path(str(config_path)) / PROTECT_WINDOWS_FILENAME
        return None


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
    resolved = Path(path) if path is not None else default_protect_windows_path()
    if resolved is None:
        raise FileNotFoundError(
            f"no gpsconfig available to resolve {PROTECT_WINDOWS_FILENAME}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"protect-window catalog not found: {resolved}")
    lines = [
        line
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    catalog: dict[str, list[tuple[float, float]]] = {}
    for row in csv.DictReader(lines):
        marker = str(row.get("sta") or "").strip()
        if not marker:
            raise ValueError(
                f"{resolved}: protect_windows.csv row without a 'sta' marker: {row}"
            )
        try:
            start = float(str(row.get("start_yearf")))
            end = float(str(row.get("end_yearf")))
        except (TypeError, ValueError):
            raise ValueError(
                f"{resolved}: station {marker}: start_yearf/end_yearf "
                f"{row.get('start_yearf')!r}/{row.get('end_yearf')!r} "
                "are not fractional years"
            ) from None
        if end < start:
            raise ValueError(
                f"{resolved}: station {marker}: protect window end {end} < "
                f"start {start} — an interval must be (start, end) with "
                "end >= start"
            )
        catalog.setdefault(marker, []).append((start, end))
    return {marker: tuple(sorted(windows)) for marker, windows in catalog.items()}


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
# Outlier flags (cleaned view)
# ---------------------------------------------------------------------------


def detect_view_outliers(
    yearf: npt.ArrayLike,
    data: npt.ArrayLike,
    Ddata: npt.ArrayLike | None = None,
    *,
    outlier_params: OutlierParams | None = None,
    step_epochs: npt.ArrayLike | None = None,
    protect_windows: tuple[tuple[float, float], ...] = (),
    min_outlier: npt.ArrayLike | None = None,
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

    Returns:
        ``(flags, provenance)`` — flags shaped like ``data`` (True =
        outlier; MASK only, nothing is removed) and a provenance dict
        with ``outlier_abort`` / ``degraded`` / ``degrade_reason`` /
        ``n_flagged``.
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
    steps: str | Path | None = None,
    protect_windows: str | Path | Sequence[tuple[float, float]] | None = None,
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
        outlier_params: Detection thresholds; None = spec defaults.
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
    }

    do_clean = attrs["clean"]
    if do_clean:
        step_epochs, steps_source = station_step_epochs(sta, steps=steps)
        attrs["step_epochs_applied"] = int(step_epochs.size)
        attrs["steps_source"] = steps_source
        pwindows, pw_source = resolve_protect_windows(sta, protect_windows)
        attrs["protect_windows_applied"] = len(pwindows)
        attrs["protect_windows_source"] = pw_source
        flags, oprov = detect_view_outliers(
            yearf,
            data,
            Ddata,
            outlier_params=outlier_params,
            step_epochs=step_epochs if step_epochs.size else None,
            protect_windows=pwindows,
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
