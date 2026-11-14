"""Common schema for earthquake catalogues (events) and fault-plane solutions.

Every catalogue reader in :mod:`geo_dataread.seismic` (SIL files, the Quakes
API for SIL and SeisComP, the legacy Skjálftalísa API, the SeisComP FDSN
service) returns the same two frames, so that catalogues can be compared and
plotted without per-source code:

* **events** — one row per event per source (the preferred solution). Index
  ``time``: origin time, ``datetime64[ns, UTC]``, sorted, *not* required to be
  unique (a swarm can have two events in the same millisecond).
* **mechanisms** — one row per fault-plane solution, keyed by
  ``(source, event_id)`` and the same ``time`` index. An event can carry more
  than one solution (``method`` tells them apart).

Rules (``PLAN-seismicity.md`` §4 and BGÓ's decisions §9a):

* A column a source cannot fill is present and NA, never absent.
* Magnitudes are raw and labelled (Q7): ``magnitude`` is always read with
  ``magnitude_type``; no scale is converted to another here.
* Automatic events are included and labelled (Q8) through
  ``evaluation_mode`` / ``evaluation_status``; consumers filter.
* Derived values are functions, not columns: :func:`moment_magnitude`,
  :func:`cumulative_moment`.
* Provenance rides on ``df.attrs["catalog"]`` (:func:`catalog_provenance`),
  as ``df.attrs["gps_view"]`` does for GPS views.

Readers build a frame, call :func:`coerce_catalog` (or
:func:`coerce_mechanisms`) and then :func:`validate` (or
:func:`validate_mechanisms`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from importlib import metadata
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

#: Version of the column tables below. Bump on any change to a column's name,
#: dtype, nullability, allowed values or meaning; cache keys carry it (T4).
#: v2 (T3): + ``sil_aut_path``, + magnitude type ``M``.
SCHEMA_VERSION: Final = 3

#: Name of the time index of both frames.
INDEX_NAME: Final = "time"

FloatArray = npt.NDArray[np.float64]

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

#: Catalogue sources. The strings are the step-source names of BGÓ's Q11
#: (``step_earthquake_source``), so one vocabulary serves config and data.
#: ``sil-files`` the SIL archive on disk; ``quakes-sil`` / ``quakes-seiscomp``
#: the Quakes API with ``system=sil`` / ``system=seiscomp``; ``skjalftalisa``
#: the legacy Skjálftalísa API (SIL); ``fdsn-seiscomp`` the internal SeisComP
#: FDSN event service (Q1).
SOURCES: Final = (
    "sil-files",
    "quakes-sil",
    "skjalftalisa",
    "quakes-seiscomp",
    "fdsn-seiscomp",
)

#: Magnitude scales, raw as each source labels them (Q7). ``Mlw`` SIL reviewed
#: (moment-calibrated local magnitude); ``ML`` SIL reviewed local magnitude;
#: ``Msil_auto`` the SIL automatic magnitude as SeisComP labels it;
#: ``aut.mag`` the SIL automatic magnitude as the Quakes API (``system=sil``)
#: labels it; ``ML_SIL`` SeisComP's recomputed local magnitude; ``Mpgv_w``
#: SeisComP's PGV magnitude; ``M`` an unlabelled magnitude as the Quakes API
#: gives it (14 events in 2024-25, T9). Whether ``aut.mag`` and ``Msil_auto``
#: are the same number is not verified, so they are kept apart.
MAGNITUDE_TYPES: Final = (
    "Mlw",
    "ML",
    "Msil_auto",
    "aut.mag",
    "ML_SIL",
    "Mpgv_w",
    "M",
    "Mpgv_alert",  # our PGV magnitude from alertMap (median), PLAN B5
)

#: QuakeML evaluation modes.
EVALUATION_MODES: Final = ("manual", "automatic")

#: QuakeML evaluation statuses plus SeisComP's ``reported`` (seen in the
#: internal Quakes API, plan §3.1).
EVALUATION_STATUSES: Final = (
    "preliminary",
    "confirmed",
    "reviewed",
    "final",
    "rejected",
    "reported",
)

#: Event types: the Quakes API vocabulary (OpenAPI 2026-08-06), which is what
#: both SIL and SeisComP are published in. ``other`` covers any type outside
#: the list. The source's own code is kept in ``event_type_code``.
EVENT_TYPES: Final = (
    "earthquake",
    "explosion",
    "not locatable",
    "not existing",
    "other",
)

#: SIL two-letter event-type code → ``event_type``. Mirrors what the Quakes API
#: (``system=sil``) publishes for the same events: ``qu`` → earthquake,
#: ``ex`` → explosion, every other code → ``other`` (checked 2026-11-13 on
#: sample events with ``sm``, ``ql``, ``un``, ``sf``, ``te``, ``sc``). The keys
#: are the codes found in the archive (``events.lib`` 1992–2026: ``qu``,
#: ``ex``, ``sm``, ``un``, ``ql``, ``sf``, ``te``, ``sc``) and the extra
#: values the legacy Skjálftalísa API accepts (``ep``, ``2``, ``EQ``, ``eq``,
#: ``me``, ``qy``, ``qe``, ``ew``; none used since 2010). The meaning of the
#: codes other than ``qu`` and ``ex`` is not documented, so none is guessed.
SIL_EVENT_TYPES: Final[Mapping[str, str]] = {
    "qu": "earthquake",
    "ex": "explosion",
    **{
        code: "other"
        for code in (
            "sm",
            "un",
            "ql",
            "sf",
            "te",
            "sc",
            "ep",
            "2",
            "EQ",
            "eq",
            "me",
            "qy",
            "qe",
            "ew",
        )
    },
}

#: How a fault-plane solution was produced (not which door it came through:
#: SIL's own solution read from the legacy API is still ``sil_fps``).
#: ``sil_fps`` the SIL solution in ``events.fps``; ``sil_sp3`` the SIL ``sp3``
#: fault-plane block; ``first_motion`` / ``moment_tensor`` / ``other`` for
#: other systems (SeisComP, to be checked in T12b).
#: Faulting classes as SIL's ``sp3`` blocks label them (normalised), stored in
#: the nullable ``sil_faulting_class`` column (PLAN R3). The computed,
#: source-independent class is a function of the planes, not a column.
SIL_FAULTING_CLASSES: Final = ("strike-slip", "normal", "reverse", "dip-slip")

MECHANISM_METHODS: Final = (
    "sil_fps",
    "sil_sp3",
    "sil_ap3",
    "first_motion",
    "moment_tensor",
    "other",
)


def sil_event_type(code: object) -> str | None:
    """Map a SIL two-letter event-type code to ``event_type``.

    Unknown codes map to ``"other"`` (as the Quakes API does); a missing code
    (``None``/NaN/blank) maps to ``None``. Surrounding blanks are ignored and
    the code is matched case-sensitively first, then in lower case.
    """
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return None
    text = str(code).strip()
    if not text:
        return None
    hit = SIL_EVENT_TYPES.get(text) or SIL_EVENT_TYPES.get(text.lower())
    return hit if hit is not None else "other"


# --------------------------------------------------------------------------
# Column tables
# --------------------------------------------------------------------------

ColumnKind = Literal["float64", "Int64", "string", "category", "datetime"]

#: Canonical dtypes. Strings use the pandas 3 default ``str`` dtype (NaN for
#: missing); validation accepts any ``StringDtype`` storage.
TIME_DTYPE: Final = pd.DatetimeTZDtype(unit="ns", tz="UTC")
STRING_DTYPE: Final = pd.StringDtype(na_value=np.nan)


@dataclasses.dataclass(frozen=True)
class ColumnSpec:
    """One column of a schema frame.

    ``allowed`` (category columns) is the closed set of values; ``minimum`` /
    ``maximum`` (numeric columns) are inclusive bounds, ``None`` = open.
    """

    name: str
    kind: ColumnKind
    nullable: bool
    meaning: str
    unit: str = ""
    allowed: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @property
    def dtype(self) -> Any:
        """The canonical pandas dtype of this column."""
        if self.kind == "category":
            return pd.CategoricalDtype(list(self.allowed), ordered=False)
        if self.kind == "string":
            return STRING_DTYPE
        if self.kind == "datetime":
            return TIME_DTYPE
        return np.dtype("float64") if self.kind == "float64" else pd.Int64Dtype()


def _mag(name: str, meaning: str) -> ColumnSpec:
    return ColumnSpec(name, "float64", True, meaning, "", (), -3.0, 10.0)


#: The events frame (index ``time``, see :data:`INDEX_NAME`).
EVENT_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("source", "category", False, "catalogue source", allowed=SOURCES),
    ColumnSpec("event_id", "string", False, "the source's native event id"),
    ColumnSpec(
        "sil_path",
        "string",
        True,
        "SIL file id (e.g. /2024/aug/22/21:55:00/59:59:641), the exact join "
        "key between SIL files and the legacy API",
    ),
    ColumnSpec(
        "sil_aut_path",
        "string",
        True,
        "SIL file id of the event's automatic solution where known; it can "
        "differ from `sil_path` (by 1 ms, or in the 5-minute directory) for a "
        "reviewed event",
    ),
    ColumnSpec("latitude", "float64", False, "epicentre", "deg", (), -90.0, 90.0),
    ColumnSpec("longitude", "float64", False, "epicentre", "deg", (), -180.0, 180.0),
    ColumnSpec(
        "depth_km",
        "float64",
        True,
        "hypocentre depth, positive down (datum not verified, so a few km "
        "above zero are allowed)",
        "km",
        (),
        -5.0,
        800.0,
    ),
    ColumnSpec("h_err_km", "float64", True, "horizontal 1σ", "km", (), 0.0, None),
    ColumnSpec("depth_err_km", "float64", True, "depth 1σ", "km", (), 0.0, None),
    _mag("magnitude", "preferred magnitude, raw"),
    ColumnSpec(
        "magnitude_type",
        "category",
        True,
        "scale of `magnitude`",
        allowed=MAGNITUDE_TYPES,
    ),
    _mag("mag_mlw", "SIL Mlw where known"),
    _mag("mag_ml", "SIL ML where known"),
    _mag("mag_auto", "SIL automatic ML where known"),
    ColumnSpec("moment_nm", "float64", True, "seismic moment M0", "N·m", (), 0.0, None),
    ColumnSpec(
        "evaluation_mode",
        "category",
        False,
        "manual (reviewed) or automatic solution",
        allowed=EVALUATION_MODES,
    ),
    ColumnSpec(
        "evaluation_status",
        "category",
        True,
        "QuakeML/SeisComP status where the source has one",
        allowed=EVALUATION_STATUSES,
    ),
    ColumnSpec(
        "event_type",
        "category",
        True,
        "event type, Quakes API vocabulary",
        allowed=EVENT_TYPES,
    ),
    ColumnSpec(
        "event_type_code",
        "string",
        True,
        "the source's own event-type code (SIL two-letter code, or the raw "
        "string), kept so the mapping to `event_type` loses nothing",
    ),
    ColumnSpec(
        "quality", "float64", True, "SIL automatic quality Q", "", (), 0.0, 100.0
    ),
    ColumnSpec("n_phases", "Int64", True, "phases used", "", (), 0.0, None),
    ColumnSpec(
        "azimuthal_gap", "float64", True, "azimuthal gap", "deg", (), 0.0, 360.0
    ),
    ColumnSpec("updated_time", "datetime", True, "last change of the event (UTC)"),
)


def _plane(n: int, *, nullable: bool) -> tuple[ColumnSpec, ...]:
    which = "first" if n == 1 else "second (auxiliary)"
    return (
        ColumnSpec(
            f"strike{n}",
            "float64",
            nullable,
            f"{which} plane strike",
            "deg",
            (),
            0.0,
            360.0,
        ),
        ColumnSpec(
            f"dip{n}", "float64", nullable, f"{which} plane dip", "deg", (), 0.0, 90.0
        ),
        ColumnSpec(
            f"rake{n}",
            "float64",
            nullable,
            f"{which} plane rake",
            "deg",
            (),
            -180.0,
            180.0,
        ),
    )


#: The mechanisms frame (index ``time``, the event's origin time). Angles in
#: the Aki & Richards convention (see :func:`auxiliary_plane`). Plane 2 is
#: nullable because some sources give one plane only.
MECHANISM_COLUMNS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("source", "category", False, "catalogue source", allowed=SOURCES),
    ColumnSpec("event_id", "string", False, "the source's native event id"),
    ColumnSpec(
        "method",
        "category",
        False,
        "how the solution was produced",
        allowed=MECHANISM_METHODS,
    ),
    *_plane(1, nullable=False),
    *_plane(2, nullable=True),
    ColumnSpec(
        "quality",
        "float64",
        True,
        "the source's own quality figure for the solution (scale per source)",
    ),
    ColumnSpec("misfit", "float64", True, "misfit of the solution", "", (), 0.0, None),
    ColumnSpec("n_polarities", "Int64", True, "polarities used", "", (), 0.0, None),
    ColumnSpec("n_amplitudes", "Int64", True, "amplitudes used", "", (), 0.0, None),
    ColumnSpec("stress_drop_mpa", "float64", True, "stress drop", "MPa", (), 0.0, None),
    ColumnSpec(
        "corner_freq_hz",
        "float64",
        True,
        "S-wave corner frequency",
        "Hz",
        (),
        0.0,
        None,
    ),
    ColumnSpec(
        "sil_faulting_class",
        "category",
        True,
        "SIL's own faulting label from the sp3 block (source data, PLAN R3)",
        allowed=SIL_FAULTING_CLASSES,
    ),
)

#: Columns that identify one mechanism row (warned on, not refused, if
#: duplicated: a source may legitimately give two solutions by one method).
MECHANISM_KEY: Final = ("source", "event_id", "method")

#: Default tolerance of the plane-consistency check, degrees. SIL stores
#: angles as whole degrees, so a correct pair can disagree by about a degree.
DEFAULT_PLANE_TOL_DEG: Final = 5.0

FrameName = Literal["events", "mechanisms"]
_TABLES: Final[Mapping[str, tuple[ColumnSpec, ...]]] = {
    "events": EVENT_COLUMNS,
    "mechanisms": MECHANISM_COLUMNS,
}


def column_names(frame: FrameName = "events") -> tuple[str, ...]:
    """Column names of a schema frame, in schema order."""
    return tuple(c.name for c in _TABLES[frame])


def column_spec(name: str, frame: FrameName = "events") -> ColumnSpec:
    """The :class:`ColumnSpec` of one column (``KeyError`` if unknown)."""
    for c in _TABLES[frame]:
        if c.name == name:
            return c
    raise KeyError(f"{name!r} is not a column of the {frame} schema")


# --------------------------------------------------------------------------
# Empty frames and coercion
# --------------------------------------------------------------------------


def _empty(frame: FrameName) -> pd.DataFrame:
    index = pd.DatetimeIndex([], dtype=TIME_DTYPE, name=INDEX_NAME)
    data = {c.name: pd.Series([], dtype=c.dtype, index=index) for c in _TABLES[frame]}
    return pd.DataFrame(data, index=index)


def empty_catalog() -> pd.DataFrame:
    """An events frame with no rows and every column correctly typed."""
    return _empty("events")


def empty_mechanisms() -> pd.DataFrame:
    """A mechanisms frame with no rows and every column correctly typed."""
    return _empty("mechanisms")


class CoercionError(ValueError):
    """A frame cannot be brought to the schema without guessing."""


_TZ_SUFFIX: Final = r"(?:Z|[+-]\d{2}(?::?\d{2})?)\s*$"


def _to_utc(values: Any, *, assume_utc: bool, what: str) -> Any:
    """Parse to a tz-aware UTC ns DatetimeIndex; refuse naive unless told."""
    naive_msg = (
        f"{what}: times have no time zone; pass assume_utc=True only "
        "if the source documents them as UTC"
    )
    if len(values) and bool(pd.isna(pd.Series(values, dtype=object)).all()):
        return pd.DatetimeIndex([pd.NaT] * len(values), dtype=TIME_DTYPE)
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        raise CoercionError(
            f"{what}: numbers are not times (no `time` column or index given?)"
        )
    try:
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            text = pd.Series(values, dtype=object).astype(str)
            present = text[pd.Series(values, dtype=object).notna().to_numpy()]
            aware = present.str.contains(_TZ_SUFFIX, regex=True)
            if not aware.all() and not assume_utc:
                raise CoercionError(naive_msg)
            # every string has a zone, or naive ones are UTC by the caller's word
            parsed = pd.to_datetime(values, format="ISO8601", utc=True)
            return pd.DatetimeIndex(parsed).as_unit("ns")
        idx = pd.DatetimeIndex(pd.to_datetime(values))
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CoercionError):
            raise
        raise CoercionError(f"{what}: cannot be read as times ({exc})") from exc
    if idx.tz is None:
        if not assume_utc:
            raise CoercionError(naive_msg)
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.as_unit("ns")


def _coerce_column(col: Any, spec: ColumnSpec, *, assume_utc: bool) -> Any:
    where = f"column {spec.name!r}"
    if spec.kind == "category":
        present = col[col.notna()].astype(object)
        values = [str(v) for v in present]
        unknown = sorted(set(values) - set(spec.allowed))
        # unknown values are kept as extra categories so validate() reports
        # them; a plain astype(CategoricalDtype) would turn them into NaN
        cats = list(spec.allowed) + unknown
        out = pd.Series(pd.NA, index=col.index, dtype=object)
        out[col.notna()] = values
        return out.astype(pd.CategoricalDtype(cats, ordered=False))
    if spec.kind == "string":
        return col.astype(object).where(col.notna(), None).astype(STRING_DTYPE)
    if spec.kind == "datetime":
        idx = _to_utc(col, assume_utc=assume_utc, what=where)
        return pd.Series(idx, index=col.index, dtype=TIME_DTYPE)
    try:
        num = pd.to_numeric(col, errors="raise")
    except (TypeError, ValueError) as exc:
        raise CoercionError(f"{where}: not numeric ({exc})") from exc
    try:
        return num.astype(spec.dtype)
    except (TypeError, ValueError) as exc:
        raise CoercionError(f"{where}: cannot cast to {spec.kind} ({exc})") from exc


def _coerce(
    df: pd.DataFrame,
    frame: FrameName,
    *,
    source: str | None,
    assume_utc: bool,
    sort: bool,
) -> pd.DataFrame:
    out = df.copy(deep=True)
    attrs = dict(df.attrs)
    if INDEX_NAME in out.columns:
        out = out.set_index(INDEX_NAME)
    if len(out) == 0 and not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.DatetimeIndex([], dtype=TIME_DTYPE)
    else:
        out.index = _to_utc(out.index, assume_utc=assume_utc, what="time index")
    out.index.name = INDEX_NAME

    if source is not None:
        if source not in SOURCES:
            raise CoercionError(f"source {source!r} is not one of {SOURCES}")
        if "source" in out.columns:
            have = {str(v) for v in out["source"].dropna()}
            if have - {source}:
                raise CoercionError(
                    f"source={source!r} given but the frame already holds {sorted(have)}"
                )
            out["source"] = (
                out["source"].astype(object).where(out["source"].notna(), source)
            )
        else:
            out["source"] = source

    specs = _TABLES[frame]
    cols: dict[str, Any] = {}
    for spec in specs:
        if spec.name in out.columns:
            cols[spec.name] = _coerce_column(
                out[spec.name], spec, assume_utc=assume_utc
            )
        else:
            cols[spec.name] = pd.Series(index=out.index, dtype=spec.dtype)
    names = {s.name for s in specs}
    for extra in out.columns:
        if extra not in names:
            cols[extra] = out[extra]
    result = pd.DataFrame(cols, index=out.index)
    if sort:
        result = result.sort_index(kind="stable")
    result.attrs = attrs
    return result


def coerce_catalog(
    df: pd.DataFrame,
    *,
    source: str | None = None,
    assume_utc: bool = False,
    sort: bool = True,
) -> pd.DataFrame:
    """Bring a reader's events frame to the schema. Every reader calls this.

    Returns a new frame (the input is not changed): schema columns in schema
    order, each cast to its canonical dtype; missing columns added as NA;
    extra columns kept after them; the index (or a ``time`` column) parsed to
    ``datetime64[ns, UTC]`` and named ``time``; rows stably sorted by time
    unless ``sort=False``. ``attrs`` are carried over.

    ``source`` fills the ``source`` column (or its NA cells); a conflicting
    value already present raises. Times without a time zone raise
    :class:`CoercionError` unless ``assume_utc=True`` — pass it only for a
    source that documents UTC without a ``Z`` (the legacy Skjálftalísa API),
    never for local naive times. Category values outside the allowed set are
    *kept* (not turned into NaN) so that :func:`validate` reports them.
    Non-numeric text in a numeric column raises :class:`CoercionError`.
    """
    return _coerce(df, "events", source=source, assume_utc=assume_utc, sort=sort)


def coerce_mechanisms(
    df: pd.DataFrame,
    *,
    source: str | None = None,
    assume_utc: bool = False,
    sort: bool = True,
) -> pd.DataFrame:
    """Bring a reader's mechanisms frame to the schema (as :func:`coerce_catalog`)."""
    return _coerce(df, "mechanisms", source=source, assume_utc=assume_utc, sort=sort)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

IssueKind = Literal[
    "missing_column",
    "extra_column",
    "wrong_dtype",
    "null_values",
    "out_of_range",
    "unknown_category",
    "index_not_datetime",
    "index_not_utc",
    "index_unit",
    "index_name",
    "index_nat",
    "index_unsorted",
    "planes_inconsistent",
    "partial_plane",
    "duplicate_key",
]
Severity = Literal["error", "warning"]


@dataclasses.dataclass(frozen=True)
class SchemaIssue:
    """One finding of :func:`validate` / :func:`validate_mechanisms`.

    ``count`` is the number of offending rows (0 for frame-level issues) and
    ``examples`` a few of them (row positions with their time, or values).
    """

    kind: IssueKind
    severity: Severity
    column: str | None
    message: str
    count: int = 0
    examples: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    """Structured result of a validation. ``ok`` means no errors."""

    frame: FrameName
    schema_version: int
    n_rows: int
    issues: tuple[SchemaIssue, ...]

    @property
    def errors(self) -> tuple[SchemaIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[SchemaIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def kinds(self) -> set[str]:
        """The issue kinds present (handy in tests and logs)."""
        return {i.kind for i in self.issues}

    def summary(self) -> str:
        """Readable multi-line summary."""
        head = (
            f"{self.frame} schema v{self.schema_version}: {self.n_rows} rows, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        )
        lines = [head]
        for i in self.issues:
            col = f" [{i.column}]" if i.column else ""
            ex = f" e.g. {', '.join(i.examples)}" if i.examples else ""
            lines.append(f"  {i.severity}: {i.kind}{col}: {i.message}{ex}")
        return "\n".join(lines)

    def raise_for_errors(self) -> None:
        """Raise :class:`SchemaError` if there is any error."""
        if not self.ok:
            raise SchemaError(self)


class SchemaError(ValueError):
    """A frame failed validation; ``report`` holds the details."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.summary())
        self.report = report


_N_EXAMPLES: Final = 5


def _examples(df: pd.DataFrame, mask: npt.NDArray[np.bool_]) -> tuple[str, ...]:
    pos = np.flatnonzero(mask)[:_N_EXAMPLES]
    return tuple(f"row {int(p)} ({df.index[int(p)]})" for p in pos)


def _check_index(df: pd.DataFrame) -> list[SchemaIssue]:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return [
            SchemaIssue(
                "index_not_datetime",
                "error",
                None,
                f"index must be a DatetimeIndex, got {type(idx).__name__}",
            )
        ]
    out: list[SchemaIssue] = []
    if idx.name != INDEX_NAME:
        out.append(
            SchemaIssue(
                "index_name",
                "error",
                None,
                f"index must be named {INDEX_NAME!r}, got {idx.name!r}",
            )
        )
    if idx.tz is None:
        out.append(SchemaIssue("index_not_utc", "error", None, "index is tz-naive"))
    elif str(idx.tz) != "UTC":
        out.append(
            SchemaIssue(
                "index_not_utc", "error", None, f"index time zone is {idx.tz}, not UTC"
            )
        )
    if idx.tz is not None and idx.dtype != TIME_DTYPE:
        out.append(
            SchemaIssue(
                "index_unit",
                "error",
                None,
                f"index dtype {idx.dtype}, expected {TIME_DTYPE}",
            )
        )
    nat = np.asarray(idx.isna())
    if nat.any():
        out.append(
            SchemaIssue(
                "index_nat",
                "error",
                None,
                "index holds NaT",
                int(nat.sum()),
                tuple(f"row {int(p)}" for p in np.flatnonzero(nat)[:_N_EXAMPLES]),
            )
        )
    elif not idx.is_monotonic_increasing:
        i8 = idx.asi8
        bad = np.zeros(len(idx), dtype=bool)
        bad[1:] = i8[1:] < i8[:-1]
        out.append(
            SchemaIssue(
                "index_unsorted",
                "error",
                None,
                "index is not sorted by time",
                int(bad.sum()),
                _examples(df, bad),
            )
        )
    return out


def _dtype_ok(dtype: Any, spec: ColumnSpec) -> bool:
    if spec.kind == "category":
        return isinstance(dtype, pd.CategoricalDtype)
    if spec.kind == "string":
        return isinstance(dtype, pd.StringDtype)
    return bool(dtype == spec.dtype)


def _check_column(df: pd.DataFrame, spec: ColumnSpec) -> list[SchemaIssue]:
    col = df[spec.name]
    out: list[SchemaIssue] = []
    if not _dtype_ok(col.dtype, spec):
        out.append(
            SchemaIssue(
                "wrong_dtype",
                "error",
                spec.name,
                f"dtype {col.dtype}, expected {spec.dtype}",
            )
        )
    nulls = np.asarray(col.isna(), dtype=bool)
    if not spec.nullable and nulls.any():
        out.append(
            SchemaIssue(
                "null_values",
                "error",
                spec.name,
                "column may not hold NA",
                int(nulls.sum()),
                _examples(df, nulls),
            )
        )
    if spec.kind == "category":
        values = col[~nulls].astype(object).astype(str)
        bad = sorted(set(values) - set(spec.allowed))
        if bad:
            n = int(values.isin(bad).sum())
            out.append(
                SchemaIssue(
                    "unknown_category",
                    "error",
                    spec.name,
                    f"values outside {spec.allowed}",
                    n,
                    tuple(bad[:_N_EXAMPLES]),
                )
            )
    if (spec.minimum is not None or spec.maximum is not None) and spec.kind in (
        "float64",
        "Int64",
    ):
        num = pd.to_numeric(col, errors="coerce").astype("float64").to_numpy()
        with np.errstate(invalid="ignore"):
            bad_mask = np.zeros(len(num), dtype=bool)
            if spec.minimum is not None:
                bad_mask |= num < spec.minimum
            if spec.maximum is not None:
                bad_mask |= num > spec.maximum
        if bad_mask.any():
            out.append(
                SchemaIssue(
                    "out_of_range",
                    "error",
                    spec.name,
                    f"values outside [{spec.minimum}, {spec.maximum}] {spec.unit}".rstrip(),
                    int(bad_mask.sum()),
                    _examples(df, bad_mask),
                )
            )
    return out


def _validate(df: pd.DataFrame, frame: FrameName) -> list[SchemaIssue]:
    issues = _check_index(df)
    specs = _TABLES[frame]
    for spec in specs:
        if spec.name not in df.columns:
            issues.append(
                SchemaIssue("missing_column", "error", spec.name, "column is absent")
            )
        else:
            issues.extend(_check_column(df, spec))
    names = {s.name for s in specs}
    extras = [str(c) for c in df.columns if c not in names]
    if extras:
        issues.append(
            SchemaIssue(
                "extra_column",
                "warning",
                None,
                "columns not in the schema (kept, but not part of the contract)",
                len(extras),
                tuple(extras[:_N_EXAMPLES]),
            )
        )
    return issues


def _finish(
    df: pd.DataFrame, frame: FrameName, issues: list[SchemaIssue], strict: bool
) -> ValidationReport:
    report = ValidationReport(frame, SCHEMA_VERSION, len(df), tuple(issues))
    if strict:
        report.raise_for_errors()
    return report


def validate(df: pd.DataFrame, *, strict: bool = False) -> ValidationReport:
    """Check an events frame against the schema.

    Detects missing columns, wrong dtypes, NA in non-nullable columns,
    out-of-range values, unknown category values, and an index that is not a
    sorted, NaT-free ``datetime64[ns, UTC]`` index named ``time`` (duplicate
    times are allowed). Extra columns are a warning. Returns the
    :class:`ValidationReport`; with ``strict=True`` raises
    :class:`SchemaError` instead when there is any error.
    """
    return _finish(df, "events", _validate(df, "events"), strict)


def validate_mechanisms(
    df: pd.DataFrame,
    *,
    strict: bool = False,
    plane_tol_deg: float = DEFAULT_PLANE_TOL_DEG,
) -> ValidationReport:
    """Check a mechanisms frame against the schema.

    Everything :func:`validate` checks, plus: plane 2 given whole or not at
    all; where both planes are given,
    that plane 2 is the auxiliary plane of plane 1 within ``plane_tol_deg``
    (see :func:`plane_mismatch_deg`), and a warning when the key
    ``(source, event_id, method)`` repeats.
    """
    issues = _validate(df, "mechanisms")
    angle_cols = [f"{a}{n}" for n in (1, 2) for a in ("strike", "dip", "rake")]
    if all(c in df.columns for c in angle_cols) and len(df):
        a = {
            c: pd.to_numeric(df[c], errors="coerce").astype("float64").to_numpy()
            for c in angle_cols
        }
        given = np.stack([~np.isnan(a[f"{x}2"]) for x in ("strike", "dip", "rake")])
        partial = given.any(axis=0) & ~given.all(axis=0)
        if partial.any():
            issues.append(
                SchemaIssue(
                    "partial_plane",
                    "error",
                    "strike2",
                    "plane 2 must be given whole (strike2, dip2, rake2) or not at all",
                    int(partial.sum()),
                    _examples(df, partial),
                )
            )
        mis = plane_mismatch_deg(
            a["strike1"], a["dip1"], a["rake1"], a["strike2"], a["dip2"], a["rake2"]
        )
        with np.errstate(invalid="ignore"):
            bad = np.asarray(mis > plane_tol_deg, dtype=bool)
        if bad.any():
            worst = float(np.nanmax(mis[bad]))
            issues.append(
                SchemaIssue(
                    "planes_inconsistent",
                    "error",
                    "strike2",
                    f"plane 2 is not the auxiliary plane of plane 1 within "
                    f"{plane_tol_deg}° (worst {worst:.1f}°)",
                    int(bad.sum()),
                    _examples(df, bad),
                )
            )
    if all(c in df.columns for c in MECHANISM_KEY) and len(df):
        dup = np.asarray(
            df.reset_index(drop=True).duplicated(list(MECHANISM_KEY), keep=False),
            dtype=bool,
        )
        if dup.any():
            issues.append(
                SchemaIssue(
                    "duplicate_key",
                    "warning",
                    None,
                    f"{MECHANISM_KEY} repeats",
                    int(dup.sum()),
                    _examples(df, dup),
                )
            )
    return _finish(df, "mechanisms", issues, strict)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

ATTRS_KEY: Final = "catalog"


def _iso_utc(t: datetime | date | str | None) -> str | None:
    if t is None:
        return None
    ts = pd.Timestamp(t)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return str(ts.isoformat().replace("+00:00", "Z"))


def _version() -> str:
    try:
        return metadata.version("geo_dataread")
    except metadata.PackageNotFoundError:
        return "unknown"


def catalog_provenance(
    *,
    source: str,
    location: str,
    start: datetime | date | str | None = None,
    end: datetime | date | str | None = None,
    filters: Mapping[str, Any] | None = None,
    fetched_at: datetime | None = None,
    frame: FrameName = "events",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``df.attrs["catalog"]`` provenance record.

    ``location`` is the endpoint URL or the base path read; ``start``/``end``
    the query window (naive times are taken as UTC, written ISO with ``Z``);
    ``filters`` the query filters as passed (box, depth, size, type, …);
    ``fetched_at`` defaults to now (UTC). Everything is JSON-safe.
    """
    if source not in SOURCES:
        raise ValueError(f"source {source!r} is not one of {SOURCES}")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "frame": frame,
        "source": source,
        "location": location,
        "window": {"start": _iso_utc(start), "end": _iso_utc(end)},
        "filters": dict(filters or {}),
        "fetched_at": _iso_utc(fetched_at or datetime.now(UTC)),
        "software": {"geo_dataread": _version()},
    }
    if extra:
        record.update(extra)
    return record


def with_provenance(df: pd.DataFrame, provenance: Mapping[str, Any]) -> pd.DataFrame:
    """Set ``df.attrs["catalog"]`` (in place) and return ``df`` for chaining."""
    df.attrs[ATTRS_KEY] = dict(provenance)
    return df


# --------------------------------------------------------------------------
# Derived values
# --------------------------------------------------------------------------

#: Hanks & Kanamori (1979) constant for M0 in N·m:
#: Mw = 2/3·log10(M0[dyn·cm]) − 10.7 = 2/3·log10(M0[N·m]) − 6.0333…
HK79_CONSTANT_NM: Final = 10.7 - 2.0 / 3.0 * 7.0


def moment_magnitude(m0_nm: npt.ArrayLike) -> FloatArray:
    """Moment magnitude from seismic moment in N·m (Hanks & Kanamori 1979).

    ``Mw = (2/3)·log10(M0) − 6.0333`` with ``M0`` in N·m, which is Hanks &
    Kanamori's ``Mw = (2/3)·log10(M0) − 10.7`` with ``M0`` in dyn·cm
    (1 N·m = 10⁷ dyn·cm). Hanks, T. C. & Kanamori, H. (1979), *A moment
    magnitude scale*, J. Geophys. Res. 84(B5), 2348–2350,
    doi:10.1029/JB084iB05p02348.

    Two nearby constants are in use: the legacy ``sil_read.MW`` uses 6.03
    (0.0033 higher Mw), and the IASPEI (2013) standard
    ``Mw = (log10(M0) − 9.1)/1.5`` gives 6.0667 (0.033 lower Mw).

    Accepts a scalar or array; returns a float array of the same shape (0-d
    for a scalar). ``M0 ≤ 0`` or NaN gives NaN, without a warning.
    """
    m0 = np.asarray(m0_nm, dtype=np.float64)
    out = np.full(m0.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(m0) & (m0 > 0)
    out[ok] = 2.0 / 3.0 * np.log10(m0[ok]) - HK79_CONSTANT_NM
    return out


def cumulative_moment(df: pd.DataFrame, column: str = "moment_nm") -> Any:
    """Cumulative seismic moment (N·m) in time order.

    Returns a float Series on the frame's index, stably sorted by time, named
    ``cumulative_moment_nm``. Rows without a moment (NA, e.g. automatic
    events) add nothing, but keep their row (they carry the running total).
    """
    ordered = df.sort_index(kind="stable") if len(df) else df
    m0 = pd.to_numeric(ordered[column], errors="coerce").astype("float64")
    return m0.fillna(0.0).cumsum().rename("cumulative_moment_nm")


# --------------------------------------------------------------------------
# Double couple
# --------------------------------------------------------------------------

_EPS: Final = 1e-9


def _normal_slip(
    strike: FloatArray, dip: FloatArray, rake: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Unit fault normal and slip vector (north, east, down), Aki & Richards."""
    phi, delta, lam = np.radians(strike), np.radians(dip), np.radians(rake)
    n = np.stack(
        [-np.sin(delta) * np.sin(phi), np.sin(delta) * np.cos(phi), -np.cos(delta)]
    )
    s = np.stack(
        [
            np.cos(lam) * np.cos(phi) + np.cos(delta) * np.sin(lam) * np.sin(phi),
            np.cos(lam) * np.sin(phi) - np.cos(delta) * np.sin(lam) * np.cos(phi),
            -np.sin(lam) * np.sin(delta),
        ]
    )
    return n, s


def _wrap_strike(strike: FloatArray) -> FloatArray:
    out = np.mod(strike, 360.0)
    return np.where(np.isclose(out, 360.0, atol=_EPS, rtol=0.0), 0.0, out)


def _wrap_rake(rake: FloatArray) -> FloatArray:
    out = np.mod(rake + 180.0, 360.0) - 180.0
    return np.where(np.isclose(out, -180.0, atol=_EPS, rtol=0.0), 180.0, out)


def auxiliary_plane(
    strike: npt.ArrayLike, dip: npt.ArrayLike, rake: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The auxiliary (conjugate) plane of a double couple, in degrees.

    Aki & Richards (2002) convention: strike clockwise from north, the plane
    dipping to the right of the strike direction, dip 0–90, rake the slip
    direction of the hanging wall measured in the plane from the strike
    direction (90 thrust, −90 normal, 0 left-lateral). The auxiliary plane
    has the slip vector as its normal and the fault normal as its slip.

    Output is normalised: strike in [0, 360), dip in [0, 90], rake in
    (−180, 180]. A vertical plane has two descriptions, (φ, 90, λ) and
    (φ+180, 90, −λ); the one with strike in [0, 180) is returned. For a
    horizontal plane the strike is arbitrary and 0 is returned. Example:
    0/90/0 → 90/90/180; 0/45/90 → 180/45/90.

    Accepts scalars or arrays (broadcast); returns float arrays (0-d for
    scalars). NaN in gives NaN out.
    """
    s1, d1, r1 = np.broadcast_arrays(
        np.asarray(strike, dtype=np.float64),
        np.asarray(dip, dtype=np.float64),
        np.asarray(rake, dtype=np.float64),
    )
    n, s = _normal_slip(s1, d1, r1)
    # auxiliary plane: normal = slip, slip = normal; flip both if the normal
    # points down (M = n·sᵀ + s·nᵀ is unchanged by flipping both)
    n2, s2 = s.copy(), n.copy()
    down = n2[2] > _EPS
    n2[:, down] *= -1.0
    s2[:, down] *= -1.0
    nz = np.clip(-n2[2], -1.0, 1.0)
    dip2 = np.degrees(np.arccos(nz))
    sin_d = np.sin(np.radians(dip2))
    horizontal = sin_d < _EPS
    phi = np.where(horizontal, 0.0, np.arctan2(-n2[0], n2[1]))
    with np.errstate(invalid="ignore", divide="ignore"):
        lam_tilted = np.arctan2(
            -s2[2] / np.where(horizontal, 1.0, sin_d),
            s2[0] * np.cos(phi) + s2[1] * np.sin(phi),
        )
    lam_flat = phi - np.arctan2(s2[1], s2[0])
    lam = np.where(horizontal, lam_flat, lam_tilted)
    strike2 = _wrap_strike(np.degrees(phi))
    rake2 = _wrap_rake(np.degrees(lam))
    vertical = np.isclose(dip2, 90.0, atol=1e-7, rtol=0.0) & (strike2 >= 180.0)
    strike2 = np.where(vertical, strike2 - 180.0, strike2)
    rake2 = np.where(vertical, _wrap_rake(-rake2), rake2)
    dip2 = np.where(np.isclose(dip2, 90.0, atol=1e-7, rtol=0.0), 90.0, dip2)
    missing = np.isnan(s1) | np.isnan(d1) | np.isnan(r1)
    nan = np.float64(np.nan)
    return (
        np.where(missing, nan, strike2),
        np.where(missing, nan, dip2),
        np.where(missing, nan, rake2),
    )


def plane_mismatch_deg(
    strike1: npt.ArrayLike,
    dip1: npt.ArrayLike,
    rake1: npt.ArrayLike,
    strike2: npt.ArrayLike,
    dip2: npt.ArrayLike,
    rake2: npt.ArrayLike,
) -> FloatArray:
    """How far plane 2 is from being the auxiliary plane of plane 1, degrees.

    Compared as vectors, so the strike/rake wrap and the two descriptions of
    a vertical plane do not count as differences: the larger of the angles
    between normal 2 and slip 1, and between slip 2 and normal 1, each taken
    sign-free, with the moment-tensor sign checked through the T axis
    (n + s). 0 for a consistent pair; NaN where any angle is NaN. This is a
    consistency check, not the Kagan angle (that is the comparison, T16b).
    """
    a = [
        np.asarray(v, dtype=np.float64)
        for v in (strike1, dip1, rake1, strike2, dip2, rake2)
    ]
    b = np.broadcast_arrays(*a)
    n1, u1 = _normal_slip(b[0], b[1], b[2])
    n2, u2 = _normal_slip(b[3], b[4], b[5])

    def angle(x: FloatArray, y: FloatArray) -> FloatArray:
        # sign-free angle between unit vectors; atan2 keeps precision near 0
        cross = np.linalg.norm(np.cross(x, y, axis=0), axis=0)
        return np.degrees(np.arctan2(cross, np.abs(np.sum(x * y, axis=0))))

    # sign-free plane/slip agreement, then the T axis (sign of the tensor):
    # for the auxiliary plane T = n2 + u2 = ±(u1 + n1)
    t1 = (n1 + u1) / math.sqrt(2.0)
    t2 = (n2 + u2) / math.sqrt(2.0)
    return np.maximum(np.maximum(angle(n2, u1), angle(u2, n1)), angle(t1, t2))
