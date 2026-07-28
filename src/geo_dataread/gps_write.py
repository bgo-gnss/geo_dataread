"""Cleaned ``.NEU`` writer — outlier-cleaned time series as a file product.

The pragmatic intermediate delivery step (internal-delivery slice): the
aflogun portal and external modeling consume the ``.NEU`` files that
``gps-savetimes`` publishes; this module writes a SECOND file next to the
raw one with the outlier-flagged epochs removed, so neither consumer has
to re-run detection.

Contract:

- The cleaned file is produced by the SAME formatter as the raw file
  (:func:`geo_dataread.gps_read.gamittoFile` on a
  :func:`geo_dataread.gps_read.gamittoNEU` record array) — byte-format
  identical (same header, same column widths), differing ONLY by the
  dropped rows. The raw file is never touched.
- Outlier flags come from :func:`gps_analysis.detect_outliers` via
  :func:`geo_dataread.gps_views.detect_view_outliers` (model-aware,
  signal-protecting detection against a robust ``lineperiodic`` fit).
- Row-wise UNION mask: a ``.NEU`` row is shared across N/E/U, so an epoch
  flagged in ANY component drops the whole row (matches the view slice's
  inlier hard rule — per-component dropping is impossible in this format).
- Graceful degrade (design §0.4): if detection fails or aborts on the
  excess-candidate rule, the FULL raw series is written and the sidecar
  marks the file ``degraded`` — a ``UserWarning`` and a log record are
  emitted, never a silent clip and never a half-cleaned file. Consumers
  MUST honour the sidecar's ``degraded`` flag.
- A provenance sidecar ``<outfile>.prov.json`` records station, reference,
  detector parameters (+ stable ``params_hash``), row counts, per-component
  flag counts, the degrade state, and software versions — this is what lets
  a consumer trust the file without re-running detection.

Detection always runs in millimetres (metre-unit output is scaled up for
the detector) so :class:`gps_analysis.OutlierParams` magnitude thresholds
(``scale_floor``, ``min_outlier``) mean the same thing for ``mm=True`` and
``mm=False`` products.
"""

import dataclasses
import hashlib
import json
import logging
import warnings
from datetime import datetime, timezone
from importlib import metadata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from gps_analysis import OutlierParams

from geo_dataread.gps_views import (
    detect_view_outliers,
    resolve_outlier_detection,
    resolve_protect_windows,
    station_step_epochs,
)

logger = logging.getLogger(__name__)

PROV_SCHEMA_VERSION = 1
"""Schema version of the ``.prov.json`` provenance sidecar."""

DETECTION_MODEL = "lineperiodic"
"""Trajectory model the outlier detector fits (leaf ``gps_analysis``)."""

_COMPONENTS = ("north", "east", "up")
_DATA_FIELDS = tuple(f"{kind}[{c}]" for kind in ("data", "Ddata") for c in range(3))


def _package_version(dist: str) -> str:
    """Best-effort installed version of a distribution (for provenance)."""
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:  # pragma: no cover - dev checkouts
        return "unknown"


def outlier_params_hash(
    params: OutlierParams,
    min_outlier: Sequence[float] | None = None,
    model: str = DETECTION_MODEL,
) -> str:
    """Stable content hash of a detection configuration.

    Hashes the model name plus the full :class:`OutlierParams` field set
    (canonical JSON, sorted keys) — two runs with the same configuration
    produce the same hash, any parameter change produces a new one. When a
    per-component ``min_outlier`` floor is in effect it is folded in too (so
    ``[5,5,10]`` and ``[5,5,5]`` differ); ``None`` reproduces the pre-floor
    payload byte-for-byte, so the default path's hash is unchanged.
    """
    payload: dict[str, object] = {"model": model, "params": dataclasses.asdict(params)}
    if min_outlier is not None:
        payload["min_outlier"] = [float(v) for v in min_outlier]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _numeric_neu(
    neudata: Any,
    sta: str,
    *,
    mm: bool,
    ref: str,
    dstring: str | None,
    reference: str,
    Dir: str | None,
    rhour: bool,
) -> Any:
    """Numeric (``dstring="yearf"``) twin of a formatted ``gamittoNEU`` read.

    When ``dstring`` formats the time tag as a string, ``neudata["yearf"]``
    is unusable as detection epochs — re-read through the SAME pipeline
    with numeric fractional years and assert row alignment (identical
    data/sigma columns) so the flags map 1:1 onto the output rows.
    """
    from geo_dataread import gps_read  # deferred: gps_read lazily imports back

    if dstring == "yearf":
        return neudata

    numdata = gps_read.gamittoNEU(  # type: ignore[no-untyped-call]
        sta,
        mm=mm,
        ref=ref,
        dstring="yearf",
        reference=reference,
        Dir=Dir,
        rhour=rhour,
    )
    if len(numdata) != len(neudata):
        raise RuntimeError(
            f"{sta}: numeric re-read row count {len(numdata)} != formatted "
            f"read {len(neudata)} - cannot align outlier flags"
        )
    for field in _DATA_FIELDS:
        if not np.array_equal(
            np.asarray(neudata[field], dtype=np.float64),
            np.asarray(numdata[field], dtype=np.float64),
            equal_nan=True,
        ):
            raise RuntimeError(
                f"{sta}: numeric re-read column {field!r} differs from the "
                "formatted read - cannot align outlier flags"
            )
    return numdata


def write_cleaned_neu(
    sta: str,
    outfile: str | Path,
    *,
    ref: str = "plate",
    mm: bool = True,
    dstring: str | None = None,
    reference: str = "ITRF2008",
    Dir: str | None = None,
    rhour: bool = False,
    outlier_params: OutlierParams | None = None,
    min_outlier: float | Sequence[float] | None = None,
    steps: str | Path | None = None,
    protect_windows: str | Path | Sequence[tuple[float, float]] | None = None,
    outlier_overrides: str | Path | None = None,
) -> dict[str, Any]:
    """Write an outlier-cleaned ``.NEU`` file plus its provenance sidecar.

    Builds the exact raw-product record array
    (:func:`geo_dataread.gps_read.gamittoNEU` with the caller's arguments),
    detects outliers on the numeric twin of the same read, drops every row
    flagged in ANY component (row-wise union), and writes the survivors
    through :func:`geo_dataread.gps_read.gamittoFile` — so the cleaned file
    is byte-format identical to the raw product, just with fewer rows.

    Declared steps (``steps.csv``) are fed to detection so the trajectory
    model absorbs known offsets (equipment changes, coseismic jumps)
    instead of over-flagging real signal on active stations — the empirical
    SENG lesson. A missing / unreadable catalog degrades gracefully (warn +
    detect without steps); the sidecar records how many step epochs were
    applied and their source.

    Declared protect windows (``protect_windows.csv``) are the active-unrest
    lever that composes with steps: operator-marked intervals are excluded
    from the fit, the identifier stages AND the excess-flag abort, so a
    station in continuous unrest (which a bare stepless model over-flags into
    the abort — SENG on defaults) CLEANS to a plain ``_cleaned.NEU`` instead
    of degrading. A missing / unreadable catalog degrades gracefully; the
    sidecar records how many windows were applied and their source.

    Per-station outlier-parameter overrides (``outlier_overrides.csv``) let
    an operator enable the stronger detection levers (Stage-0 despike, robust
    local-polynomial identifier ``window_order=1``, ``epoch_policy="union"``)
    for active stations while the fleet default stays conservative order-0
    (zero regression for quiet stations). Precedence: an explicit
    ``outlier_params`` arg wins; otherwise the station's catalog overrides
    apply on top of the defaults. The sidecar's ``params`` / ``params_hash``
    echo the RESOLVED parameters actually used, plus the applied-override
    delta and its source.

    Graceful degrade (design §0.4): a failed or aborted detection writes
    the FULL raw series and names the file ``..._cleaned.DEGRADED.NEU``
    (instead of the plain ``..._cleaned.NEU``) so a consumer globbing
    ``*_cleaned.NEU`` can NOT ingest uncleaned data — the sidecar
    ``degraded`` flag stays too, but the filename is now a structural
    signal as well. The sidecar is marked ``degraded`` (with the reason
    and, for the excess-candidate rule, ``excess_flag_abort``) and a
    ``UserWarning`` + log fire. The alternative — refusing to write —
    would break the publishing pipeline on any detector hiccup; a
    full-but-clearly-named file keeps delivery alive while never
    misrepresenting itself. Never a silent clip.

    Args:
        sta: Station four-letter name.
        outfile: Requested cleaned-file path (a real path — not stdout).
            The ACTUAL name is this when clean, or ``.DEGRADED`` inserted
            before the suffix when degraded; the sidecar is written next to
            the actual file as ``<actual>.prov.json``.
        ref: Series reference, as :func:`gamittoNEU` (``"plate"`` |
            ``"detrend"`` | ``"itrf2008"``).
        mm: True writes millimetres, False metres (as :func:`gamittoNEU`).
        dstring: Time-tag format (``None`` = datetime default,
            ``"yearf"`` = decimal year), as :func:`gamittoNEU`.
        reference: Reference frame for the plate-velocity model.
        Dir: Input series directory override (as :func:`gamittoNEU`).
        rhour: Round the time tag (as :func:`gamittoNEU`).
        outlier_params: :class:`gps_analysis.OutlierParams` thresholds. An
            EXPLICIT value wins over the per-station override catalog (REPL
            override); None resolves per-station via ``outlier_overrides``
            then the spec defaults. Magnitude thresholds are in mm regardless
            of ``mm`` (detection always runs in mm).
        min_outlier: Per-component magnitude floor ``[N, E, U]`` (or a scalar)
            routed to ``detect_outliers`` — a candidate below its component
            floor is NOT flagged (active stations use e.g. ``[5, 5, 10]`` mm,
            U noisier). Always in mm. An EXPLICIT value wins over the catalog
            ``min_outlier_{n,e,u}``; None resolves from the catalog then the
            leaf default. Independent of ``outlier_params``.
        outlier_overrides: Per-station outlier-parameter override catalog
            path (``outlier_overrides.csv``) — enables the stronger levers
            (despike, ``window_order=1``, ``epoch_policy="union"``) for
            active stations while the global default stays conservative.
            None = deployed default; missing / unreadable degrades
            gracefully (warn + base OutlierParams). Ignored when
            ``outlier_params`` is passed explicitly.
        steps: Declared step catalog path (``steps.csv``); None = deployed
            default. A missing / unreadable catalog degrades gracefully
            (warn + detect without steps).
        protect_windows: Active-unrest cleaning lever — operator-declared
            intervals excluded from the fit, the identifiers AND the
            excess-flag abort, so an unrest station CLEANS (plain
            ``_cleaned.NEU``) instead of degrading. A catalog path
            (``protect_windows.csv``), an explicit sequence of
            ``(start, end)`` fractional-year intervals, or None (deployed
            default). A missing / unreadable catalog degrades gracefully
            (warn + detect without protect windows).

    Returns:
        The provenance record (the sidecar content) plus ``"outfile"`` and
        ``"sidecar"`` paths (the ACTUAL paths, ``.DEGRADED`` included).
    """
    from geo_dataread import gps_read  # deferred: gps_read lazily imports back

    requested = Path(outfile)

    # precedence (params): explicit outlier_params arg > per-station catalog
    # override > OutlierParams() default — the stronger levers (despike,
    # window_order=1, epoch_policy="union") per active station while the fleet
    # default stays conservative order-0. precedence (floor): explicit
    # min_outlier arg > catalog min_outlier_{n,e,u} > None (leaf default). The
    # two are independent (min_outlier is a separate detect_outliers kwarg).
    resolved = resolve_outlier_detection(
        sta,
        outlier_params=outlier_params,
        min_outlier=min_outlier,
        outlier_overrides=outlier_overrides,
    )
    params = resolved.params
    floor = resolved.min_outlier

    # the record array that gets written - identical to the raw product
    neudata = gps_read.gamittoNEU(  # type: ignore[no-untyped-call]
        sta,
        mm=mm,
        ref=ref,
        dstring=dstring,
        reference=reference,
        Dir=Dir,
        rhour=rhour,
    )
    numdata = _numeric_neu(
        neudata,
        sta,
        mm=mm,
        ref=ref,
        dstring=dstring,
        reference=reference,
        Dir=Dir,
        rhour=rhour,
    )

    t = np.asarray(numdata["yearf"], dtype=np.float64)
    y = np.vstack([np.asarray(numdata[f"data[{c}]"]) for c in range(3)]).astype(
        np.float64
    )
    sigma = np.vstack([np.asarray(numdata[f"Ddata[{c}]"]) for c in range(3)]).astype(
        np.float64
    )

    # declared steps let the model absorb known offsets instead of
    # over-flagging them (SENG lesson); missing catalog degrades gracefully
    step_epochs, steps_source = station_step_epochs(sta, steps=steps)

    # operator-declared protect windows are the active-unrest lever: excluded
    # from the fit, the identifiers AND the excess-flag abort, so an unrest
    # station CLEANS instead of degrading. Composes with step_epochs.
    pwindows, pw_source = resolve_protect_windows(sta, protect_windows)

    # detection always in mm so OutlierParams thresholds AND the per-component
    # min_outlier floor are unit-stable (the floor is authored in mm)
    unit_scale = 1.0 if mm else 1000.0
    flags, oprov = detect_view_outliers(
        t,
        y * unit_scale,
        sigma * unit_scale,
        outlier_params=params,
        step_epochs=step_epochs if step_epochs.size else None,
        protect_windows=pwindows,
        min_outlier=floor,
    )

    degraded = bool(oprov["degraded"])

    # option (b): the filename STRUCTURALLY signals cleanliness — a degraded
    # product is `..._cleaned.DEGRADED.NEU`, so a consumer globbing
    # `*_cleaned.NEU` (Vincent) cannot ingest uncleaned data.
    if degraded:
        out_path = requested.with_name(requested.stem + ".DEGRADED" + requested.suffix)
        # detect_view_outliers already warned about the detection failure;
        # make the FILE consequence explicit (design §0.4: never silent).
        reason = (
            f"{sta}: cleaned .NEU degraded - writing the FULL raw series to "
            f"{out_path.name} (structurally marked); consumers must still "
            "honour the sidecar 'degraded' flag"
        )
        logger.warning("%s", reason)
        warnings.warn(reason, UserWarning, stacklevel=2)
    else:
        out_path = requested

    union: npt.NDArray[np.bool_] = np.asarray(flags.any(axis=0), dtype=np.bool_)
    cleaned = neudata[~union]

    gps_read.gamittoFile(  # type: ignore[no-untyped-call]
        cleaned, str(out_path), mm=mm, ref=ref, dstring=dstring
    )

    prov: dict[str, Any] = {
        "schema_version": PROV_SCHEMA_VERSION,
        "kind": "cleaned_neu",
        "station": sta,
        "ref": ref,
        "reference_frame": reference,
        "unit": "mm" if mm else "m",
        "dstring": dstring,
        "cleaned_file": out_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": {
            "function": "gps_analysis.detect_outliers",
            "model": DETECTION_MODEL,
            "detection_unit": "mm",
            "row_policy": "union",
            # `params` echoes the OutlierParams (its scalar `min_outlier` is
            # the fallback); the load-bearing floors that RAN are the
            # per-component `min_outlier` below (overrides the scalar in the
            # leaf when set). `params_hash` folds the floor in so a consumer
            # can tell [5,5,10] from [5,5,5] / the bare default apart.
            "params": dataclasses.asdict(params),
            "params_hash": outlier_params_hash(params, floor),
            "outlier_overrides_applied": resolved.overrides_applied,
            "outlier_overrides_source": resolved.overrides_source,
            "min_outlier": list(floor) if floor is not None else None,
            "min_outlier_source": resolved.min_outlier_source,
            "step_epochs_applied": int(step_epochs.size),
            "steps_source": steps_source,
            "protect_windows_applied": len(pwindows),
            "protect_windows_source": pw_source,
        },
        "n_total": int(len(neudata)),
        "n_removed": int(np.count_nonzero(union)),
        "n_kept": int(len(cleaned)),
        "n_flagged_by_component": {
            name: int(np.count_nonzero(flags[c])) for c, name in enumerate(_COMPONENTS)
        },
        "excess_flag_abort": bool(oprov["outlier_abort"]),
        # §3.5a: WHICH components aborted, not merely that one did. Without
        # this a partial abort is indistinguishable in the sidecar from a
        # total one, though the first still cleans its healthy components.
        "component_abort": oprov.get("component_abort"),
        "n_aborted_components": oprov.get("n_aborted_components"),
        "degraded": degraded,
        "degrade_reason": oprov["degrade_reason"],
        "versions": {
            "geo_dataread": _package_version("geo-dataread"),
            "gps_analysis": _package_version("gps_analysis"),
        },
    }

    sidecar = Path(str(out_path) + ".prov.json")
    sidecar.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "%s: cleaned .NEU written to %s (%d/%d rows removed, degraded=%s)",
        sta,
        out_path,
        prov["n_removed"],
        prov["n_total"],
        degraded,
    )

    result = dict(prov)
    result["outfile"] = str(out_path)
    result["sidecar"] = str(sidecar)
    return result
