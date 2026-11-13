"""Per-file parsers for the SIL day catalogue files (ticket T1, PLAN-seismicity.md).

Each SIL day directory (``<base>/<yyyy>/<mon>/<dd>/``) holds a few catalogue
files. This module reads **one file at a time** and returns that file's own
fields, in their native meaning and units, as a typed DataFrame together with a
:class:`ParseReport`. Mapping onto the common schema
(:mod:`geo_dataread.seismic.schema`) is the day assembly's job (T3); turning
the raw fault planes into the ``mechanisms`` frame is T3b's.

========================  ==========================================  =====================
File                      Content                                     Parser
========================  ==========================================  =====================
``events.fps``            reviewed events with fault-plane solution   :func:`read_events_fps`
``events.lib``            reviewed events, Mlw to 0.01, event type    :func:`read_events_lib`
``lib.mag``               reviewed magnitudes, phases, errors, gap    :func:`read_lib_mag`
``aut.mag``               automatic events, all                       :func:`read_aut_mag`
``cleanaut.mag``          automatic events, cleaned (own row numbers) :func:`read_cleanaut_mag`
========================  ==========================================  =====================

Rules every parser follows:

* **Fixed-width files are sliced by column position, never split on
  whitespace**: fields run together (``6.656-0.39`` is a depth and an Mlw).
  The layout is chosen **per line from its width** (:data:`LAYOUTS`), because
  one year (1993) holds two ``aut.mag`` layouts. A width not in the table is
  malformed ("unrecognized width"); there is no nearest-layout fallback.
  ``events.fps`` is the exception: it is a whitespace-delimited format, so it
  is split and a line without exactly 33 fields is malformed.
* **Every output column is present whatever the layout**; a field the layout
  lacks (``aut.mag`` Q and id before 1996, the 1993 extra columns, ``lib.mag``
  ``w_rms_res`` in 1990-91) is NA. The ``layout`` column names each row's layout.
* **Sentinels become NA** (:data:`SENTINELS`), never kept as numbers.
* **No row disappears silently.** A row whose date, time, latitude or
  longitude cannot be read is dropped and listed in
  :attr:`ParseReport.malformed` with its line number and the reason. Any other
  unreadable field becomes NA and is listed in
  :attr:`ParseReport.field_warnings`. Blank lines are counted.
* **Explicit dtypes**: ``float64``, ``Int64``, ``bool`` and the schema's string
  dtype. ``time`` is ``datetime64[ns, UTC]`` (the files are UTC, checked
  against the API to the millisecond). Ids are opaque strings, never times.
* A 0-byte file gives a correctly typed empty frame; a missing file raises
  :class:`FileNotFoundError` (T3 decides a day's content by which files exist).

Every frame carries ``line_no`` (1-based line in the file) and ``num`` (the
file's own row number), and keeps the file's row order (RangeIndex).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import numpy as np
import pandas as pd

from geo_dataread.seismic.schema import STRING_DTYPE, TIME_DTYPE

FileKind = Literal["events.fps", "events.lib", "lib.mag", "aut.mag", "cleanaut.mag"]

#: Field kinds of the raw columns.
_FieldKind = Literal["int", "float", "str"]

#: Fields whose failure drops the row (anything else becomes NA).
_REQUIRED: Final = frozenset({"date", "time", "latitude", "longitude"})

#: Longest line text kept in a :class:`LineIssue`.
_MAX_TEXT: Final = 200


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LineIssue:
    """One problem on one line of a file.

    ``line_no`` is 1-based. ``field`` names the offending field, or is ``None``
    for a whole-line problem (unrecognized width, wrong field count).
    """

    line_no: int
    reason: str
    text: str
    field: str | None = None


@dataclasses.dataclass(frozen=True)
class ParseReport:
    """What a parser did with one file.

    ``layout`` is the single layout of all data lines, ``"mixed"`` when more
    than one occurs (see ``layout_counts``) and ``"empty"`` when the file has
    no data line. ``lines`` counts every line, ``rows`` the rows returned:
    ``lines == blank_lines + rows + len(malformed)``. ``sentinels`` counts,
    per column, the values that were sentinels (or the text ``NaN``) and
    became NA; ``overflows`` the fields written as ``*****`` (a Fortran
    format overflow, e.g. lib.mag ``min_ml`` 1999-2000), also NA.
    ``time_rollovers`` counts times written with 60 seconds (rolled into the
    next minute).
    """

    path: str
    kind: FileKind
    layout: str
    layout_counts: Mapping[str, int]
    lines: int
    blank_lines: int
    rows: int
    malformed: tuple[LineIssue, ...] = ()
    field_warnings: tuple[LineIssue, ...] = ()
    sentinels: Mapping[str, int] = dataclasses.field(default_factory=dict)
    overflows: Mapping[str, int] = dataclasses.field(default_factory=dict)
    time_rollovers: int = 0

    @property
    def n_malformed(self) -> int:
        """Number of dropped lines."""
        return len(self.malformed)

    @property
    def ok(self) -> bool:
        """True when no line was dropped and no field was unreadable."""
        return not self.malformed and not self.field_warnings

    def summary(self) -> str:
        """One line for a log."""
        return (
            f"{self.kind} {self.path}: layout={self.layout} rows={self.rows} "
            f"malformed={self.n_malformed} field_warnings={len(self.field_warnings)} "
            f"blank={self.blank_lines} sentinels={sum(self.sentinels.values())}"
        )


class ParsedFile(NamedTuple):
    """A parser's result: the file's frame and its report."""

    frame: pd.DataFrame
    report: ParseReport


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------

#: (name, start, end): 0-based, end-exclusive column positions.
_Slice = tuple[str, int, int]


@dataclasses.dataclass(frozen=True)
class Layout:
    """One fixed-width layout, recognized by its line width."""

    name: str
    width: int
    fields: tuple[_Slice, ...]
    note: str = ""


_DT: Final[tuple[_Slice, ...]] = (("num", 0, 4), ("date", 4, 13))

#: events.lib, 136 characters. Fields end at 118; blanks to 136. In 1995 the
#: event type is blank and seconds below 10 are space-padded (``0316 5.701``).
EVENTS_LIB_136: Final = Layout(
    "events.lib-136",
    136,
    (
        *_DT,
        ("time", 13, 24),
        ("latitude", 24, 33),
        ("longitude", 33, 43),
        ("depth", 43, 50),
        ("mlw", 50, 55),
        ("event_type", 55, 59),
        ("id", 59, 92),
        ("sh_min", 92, 101),
        ("sh_max", 101, 108),
        ("azim_comp", 108, 115),
        ("num_of_polarities", 115, 118),
    ),
)

_LIB_MAG_HEAD: Final[tuple[_Slice, ...]] = (
    *_DT,
    ("time", 13, 24),
    ("latitude", 24, 33),
    ("longitude", 33, 43),
    ("depth", 43, 50),
    ("mlw", 50, 56),
    ("ml", 56, 62),
    ("num_of_st_ml", 62, 67),
    ("min_ml", 67, 73),
    ("max_ml", 73, 79),
    ("num_of_st_pha", 79, 82),
    ("num_of_p_pha", 82, 85),
    ("num_of_s_pha", 85, 88),
    ("num_of_polarities", 88, 92),
    ("d_time", 92, 98),
    ("d_latitude", 98, 104),
    ("d_longitude", 104, 110),
    ("d_depth", 110, 116),
)

#: lib.mag, 141 characters (1991 onwards).
LIB_MAG_141: Final = Layout(
    "lib.mag-141",
    141,
    (
        *_LIB_MAG_HEAD,
        ("w_rms_res", 116, 123),
        ("min_dist_sta", 123, 130),
        ("max_azim_gap", 130, 136),
        ("vel_model", 136, 138),
        ("event_type", 138, 141),
    ),
)

#: lib.mag, 134 characters (1990-91): as 141 without ``w_rms_res``.
LIB_MAG_134: Final = Layout(
    "lib.mag-134",
    134,
    (
        *_LIB_MAG_HEAD,
        ("min_dist_sta", 116, 123),
        ("max_azim_gap", 123, 129),
        ("vel_model", 129, 131),
        ("event_type", 131, 134),
    ),
    note="no w_rms_res column",
)

_AUT_HEAD: Final[tuple[_Slice, ...]] = (
    *_DT,
    ("time", 13, 22),
    ("latitude", 22, 30),
)

#: aut.mag / cleanaut.mag, 89 characters (mid-1996 onwards): ML, Q and id.
AUT_MAG_89: Final = Layout(
    "aut.mag-89",
    89,
    (
        *_AUT_HEAD,
        ("longitude", 30, 39),
        ("depth", 39, 45),
        ("ml", 45, 51),
        ("q", 51, 57),
        ("id", 57, 89),
    ),
)

#: aut.mag, 57 characters (late 1995 to early 1996): Q but no id.
AUT_MAG_57: Final = Layout(
    "aut.mag-57",
    57,
    (
        *_AUT_HEAD,
        ("longitude", 30, 39),
        ("depth", 39, 45),
        ("ml", 45, 51),
        ("q", 51, 57),
    ),
    note="no id",
)

#: aut.mag, 61 characters (1993-1995): no Q, no id; ML right-aligned at 58-61.
AUT_MAG_61: Final = Layout(
    "aut.mag-61",
    61,
    (
        *_AUT_HEAD,
        ("longitude", 30, 40),
        ("depth", 40, 48),
        ("ml", 48, 61),
    ),
    note="no Q, no id",
)

#: Unidentified magnitude-like columns of the 117-character aut.mag layout,
#: right-aligned: ``aux1`` ends at 61, ``aux2`` at 69, then one every 6
#: characters to ``aux10`` at 117 (never seen filled). Boundaries measured on
#: 1992-93 rows; ``aux1`` is filled on every row seen, but whether it is the
#: automatic ML is **not verified**, so ``ml`` stays NA in this layout.
AUX_COLUMNS: Final = tuple(f"aux{i}" for i in range(1, 11))
_AUX_EDGES: Final = (48, 61, 69, 75, 81, 87, 93, 99, 105, 111, 117)

#: aut.mag, 117 characters (1992-93): zero-padded num, coordinates to 0.01 deg,
#: then the ten unidentified columns :data:`AUX_COLUMNS`.
AUT_MAG_117: Final = Layout(
    "aut.mag-117",
    117,
    (
        *_AUT_HEAD,
        ("longitude", 30, 40),
        ("depth", 40, 48),
        *(
            (name, a, b)
            for name, a, b in zip(
                AUX_COLUMNS, _AUX_EDGES[:-1], _AUX_EDGES[1:], strict=True
            )
        ),
    ),
    note="unidentified columns aux1-aux10; ml, q and id absent",
)

#: Recognized layouts per file kind, keyed by line width.
LAYOUTS: Final[Mapping[FileKind, Mapping[int, Layout]]] = {
    "events.lib": {EVENTS_LIB_136.width: EVENTS_LIB_136},
    "lib.mag": {lay.width: lay for lay in (LIB_MAG_141, LIB_MAG_134)},
    "aut.mag": {
        lay.width: lay for lay in (AUT_MAG_89, AUT_MAG_57, AUT_MAG_61, AUT_MAG_117)
    },
    "cleanaut.mag": {AUT_MAG_89.width: AUT_MAG_89},
}

#: events.fps fields, in order (whitespace delimited, 33 per line). Names as
#: in T0, except ``d_depth_mod100``: the file loses Ddepth's hundreds digit
#: (242.5 is written 42.5), so the true Ddepth must come from ``lib.mag``.
EVENTS_FPS_FIELDS: Final = (
    "num",
    "date",
    "time",
    "latitude",
    "d_latitude",
    "longitude",
    "d_longitude",
    "depth",
    "d_depth_mod100",
    "mlw",
    "id",
    "seismic_moment",
    "ev_mag_ath",
    "event_type",
    "sh_min",
    "sh_max",
    "azim_comp",
    "relative_size",
    "corner_freq",
    "fault_radius",
    "stress_drop",
    "peak_slip",
    "source_file",
    "strike_a",
    "dip_a",
    "rake_a",
    "strike_b",
    "dip_b",
    "rake_b",
    "num_of_polarities",
    "num_of_amplitudes",
    "unknown_pacc",
    "is_significant",
)
EVENTS_FPS_LAYOUT: Final = "events.fps-33"


# --------------------------------------------------------------------------
# Output columns (stable per kind)
# --------------------------------------------------------------------------

_I: Final = "int"
_F: Final = "float"
_S: Final = "str"

_FPS_KINDS: Final[Mapping[str, _FieldKind]] = {
    "num": _I,
    "latitude": _F,
    "d_latitude": _F,
    "longitude": _F,
    "d_longitude": _F,
    "depth": _F,
    "d_depth_mod100": _F,
    "mlw": _F,
    "id": _S,
    "seismic_moment": _F,
    "ev_mag_ath": _F,
    "event_type": _S,
    "sh_min": _F,
    "sh_max": _F,
    "azim_comp": _F,
    "relative_size": _F,
    "corner_freq": _F,
    "fault_radius": _F,
    "stress_drop": _F,
    "peak_slip": _F,
    "source_file": _S,
    "strike_a": _F,
    "dip_a": _F,
    "rake_a": _F,
    "strike_b": _F,
    "dip_b": _F,
    "rake_b": _F,
    "num_of_polarities": _I,
    "num_of_amplitudes": _I,
    "unknown_pacc": _F,
    "is_significant": _I,
}

_LIB_KINDS: Final[Mapping[str, _FieldKind]] = {
    "num": _I,
    "latitude": _F,
    "longitude": _F,
    "depth": _F,
    "mlw": _F,
    "event_type": _S,
    "id": _S,
    "sh_min": _F,
    "sh_max": _F,
    "azim_comp": _F,
    "num_of_polarities": _I,
}

_LIB_MAG_KINDS: Final[Mapping[str, _FieldKind]] = {
    "num": _I,
    "latitude": _F,
    "longitude": _F,
    "depth": _F,
    "mlw": _F,
    "ml": _F,
    "num_of_st_ml": _I,
    "min_ml": _F,
    "max_ml": _F,
    "num_of_st_pha": _I,
    "num_of_p_pha": _I,
    "num_of_s_pha": _I,
    "num_of_polarities": _I,
    "d_time": _F,
    "d_latitude": _F,
    "d_longitude": _F,
    "d_depth": _F,
    "w_rms_res": _F,
    "min_dist_sta": _F,
    "max_azim_gap": _F,
    "vel_model": _I,
    "event_type": _S,
}

_AUT_KINDS: Final[Mapping[str, _FieldKind]] = {
    "num": _I,
    "latitude": _F,
    "longitude": _F,
    "depth": _F,
    "ml": _F,
    "q": _F,
    "id": _S,
    **{name: _F for name in AUX_COLUMNS},
}

_KINDS: Final[Mapping[FileKind, Mapping[str, _FieldKind]]] = {
    "events.fps": _FPS_KINDS,
    "events.lib": _LIB_KINDS,
    "lib.mag": _LIB_MAG_KINDS,
    "aut.mag": _AUT_KINDS,
    "cleanaut.mag": _AUT_KINDS,
}


def output_columns(kind: FileKind) -> tuple[str, ...]:
    """The columns, in order, of the frame a parser returns for ``kind``."""
    extra = ("has_solution",) if kind == "events.fps" else ()
    return ("line_no", "layout", "time", *_KINDS[kind], *extra)


def _dtype(kind: _FieldKind) -> Any:
    if kind == "int":
        return pd.Int64Dtype()
    if kind == "float":
        return np.dtype("float64")
    return STRING_DTYPE


def _empty_frame(kind: FileKind) -> pd.DataFrame:
    data: dict[str, Any] = {
        "line_no": pd.Series([], dtype="int64"),
        "layout": pd.Series([], dtype=STRING_DTYPE),
        "time": pd.Series([], dtype=TIME_DTYPE),
    }
    for name, fk in _KINDS[kind].items():
        data[name] = pd.Series([], dtype=_dtype(fk))
    if kind == "events.fps":
        data["has_solution"] = pd.Series([], dtype="bool")
    return pd.DataFrame(data)


# --------------------------------------------------------------------------
# Sentinels
# --------------------------------------------------------------------------

#: Values that mean "no value", per kind and column, applied everywhere in the
#: file. (events.fps "no solution" rows have more, see
#: :data:`FPS_NO_SOLUTION_SENTINELS`.)
SENTINELS: Final[Mapping[FileKind, Mapping[str, tuple[float, ...]]]] = {
    "events.fps": {
        "mlw": (-9.0,),
        "ev_mag_ath": (-9.0,),
        "d_latitude": (-9.0,),
        "d_longitude": (-9.0,),
        "d_depth_mod100": (-9.0,),
        # 1.00e+01 N·m (Mw ≈ -5.4) is written on rows without a solution.
        "seismic_moment": (10.0,),
    },
    "events.lib": {"mlw": (-9.99,)},
    "lib.mag": {
        "mlw": (-9.99,),
        "ml": (-9.99,),
        "min_ml": (9.99,),
        "max_ml": (-9.99,),
    },
    "aut.mag": {"ml": (-9.0,), **{name: (-9.99,) for name in AUX_COLUMNS}},
    "cleanaut.mag": {"ml": (-9.0,)},
}

#: String sentinels, per kind and column: events.fps writes ``XX`` where
#: events.lib / lib.mag leave the event type blank.
STRING_SENTINELS: Final[Mapping[FileKind, Mapping[str, tuple[str, ...]]]] = {
    "events.fps": {"event_type": ("XX",)},
}

#: events.fps rows without a fault-plane solution (``source_file`` ``FILE``,
#: planes -9/-9/-999) write these values in the solution fields. They are NA
#: **only on such rows**: elsewhere -9 is a real strike, rake or (truncated)
#: azimuth, and 9.9 Hz a real corner frequency.
FPS_NO_SOLUTION_SENTINELS: Final[Mapping[str, tuple[float, ...]]] = {
    "sh_min": (-9.0,),
    "sh_max": (-9.0,),
    "azim_comp": (-9.0,),
    "relative_size": (9.0,),
    "corner_freq": (-9.0,),
    "fault_radius": (-9.0,),
    "stress_drop": (-9.0,),
    "peak_slip": (-9.0,),
    "strike_a": (-9.0,),
    "dip_a": (-9.0,),
    "rake_a": (-999.0,),
    "strike_b": (-9.0,),
    "dip_b": (-9.0,),
    "rake_b": (-999.0,),
    "num_of_polarities": (-9.0,),
    "num_of_amplitudes": (-9.0,),
    "unknown_pacc": (-9.0,),
    "is_significant": (-9.0,),
}

#: ``source_file`` value of an events.fps row without a solution.
FPS_NO_SOLUTION_SOURCE: Final = "FILE"


def _blank_values(
    df: pd.DataFrame,
    column: str,
    values: Sequence[float],
    rows: Any = None,
) -> int:
    """Set ``column`` to NA where it equals one of ``values``; return the count."""
    col = df[column]
    hit = np.array(col.isin(list(values)).to_numpy(dtype=bool, na_value=False))
    if rows is not None:
        hit &= np.asarray(rows, dtype=bool)
    n = int(hit.sum())
    if n:
        df.loc[hit, column] = pd.NA if isinstance(col.dtype, pd.Int64Dtype) else np.nan
    return n


def _apply_sentinels(df: pd.DataFrame, kind: FileKind) -> dict[str, int]:
    counts: dict[str, int] = {}
    if kind == "events.fps":
        no_sol = (df["source_file"] == FPS_NO_SOLUTION_SOURCE).to_numpy(
            dtype=bool, na_value=False
        ) | ((df["rake_a"] == -999.0) & (df["rake_b"] == -999.0)).to_numpy(
            dtype=bool, na_value=False
        )
        df["has_solution"] = pd.Series(~no_sol, index=df.index, dtype="bool")
        for col, vals in FPS_NO_SOLUTION_SENTINELS.items():
            if n := _blank_values(df, col, vals, rows=no_sol):
                counts[col] = n
    for col, vals in SENTINELS[kind].items():
        if n := _blank_values(df, col, vals):
            counts[col] = counts.get(col, 0) + n
    for col, texts in STRING_SENTINELS.get(kind, {}).items():
        hit = df[col].isin(list(texts)).to_numpy(dtype=bool, na_value=False)
        if n := int(hit.sum()):
            df.loc[hit, col] = np.nan
            counts[col] = counts.get(col, 0) + n
    return counts


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

#: A numeric field written as text "NaN" (aut.mag ML, 1994-97) is NA.
_NAN_TEXT: Final = r"(?i)nan"

_TIME_RE: Final = r"^(\d{1,6})(?:\.(\d*))?$"
_NS_PER_S: Final = 1_000_000_000


class _Issues:
    """Collects per-line issues while converting a block of lines."""

    def __init__(self, line_no: Any, text: Any) -> None:
        self.line_no = np.asarray(line_no, dtype=np.int64)
        self.text = list(text)
        self.drop = np.zeros(len(self.line_no), dtype=bool)
        self.malformed: list[LineIssue] = []
        self.warnings: list[LineIssue] = []
        self.nan_text: dict[str, int] = {}
        self.overflows: dict[str, int] = {}

    def flag(self, mask: Any, field: str, reason: str) -> None:
        bad = np.asarray(mask, dtype=bool)
        required = field in _REQUIRED
        for i in np.flatnonzero(bad):
            issue = LineIssue(
                int(self.line_no[i]), reason, self.text[i][:_MAX_TEXT], field
            )
            if required:
                if not self.drop[i]:
                    self.malformed.append(issue)
                self.drop[i] = True
            else:
                self.warnings.append(issue)


def _numeric(raw: Any, name: str, kind: _FieldKind, issues: _Issues) -> Any:
    """Strings → float64 / Int64; blank → NA; unreadable → NA and an issue."""
    nan_text = raw.str.fullmatch(_NAN_TEXT).to_numpy(dtype=bool, na_value=False)
    stars = raw.str.fullmatch(r"\*+").to_numpy(dtype=bool, na_value=False)
    if n := int(nan_text.sum()):
        issues.nan_text[name] = n
    if n := int(stars.sum()):
        issues.overflows[name] = n
    blank = (raw == "").to_numpy(dtype=bool) | nan_text | stars
    values = pd.to_numeric(raw.where(~blank), errors="coerce").astype("float64")
    bad = values.isna().to_numpy() & ~blank
    if kind == "int":
        frac = (values.notna() & (values != values.round())).to_numpy(dtype=bool)
        bad |= frac
        values[frac] = np.nan
        issues.flag(bad, name, f"{name}: not an integer")
        return values.astype(pd.Int64Dtype())
    issues.flag(bad, name, f"{name}: not a number")
    return values


def _times(date: Any, time: Any, issues: _Issues) -> tuple[Any, int]:
    """``YYYYMMDD`` + ``HHMMSS.sss`` (blanks read as zeros) → UTC ns times."""
    day = pd.to_datetime(date, format="%Y%m%d", errors="coerce")
    issues.flag(day.isna().to_numpy(), "date", "date: not YYYYMMDD")
    parts = time.str.replace(" ", "0", regex=False).str.extract(_TIME_RE)
    hms = parts[0].str.zfill(6)
    ok = hms.notna().to_numpy(dtype=bool)
    hh = pd.to_numeric(hms.str.slice(0, 2), errors="coerce")
    mm = pd.to_numeric(hms.str.slice(2, 4), errors="coerce")
    ss = pd.to_numeric(hms.str.slice(4, 6), errors="coerce")
    frac = parts[1].fillna("").str.ljust(9, "0").str.slice(0, 9)
    frac_ns = pd.to_numeric(frac.where(ok, "0"), errors="coerce")
    in_range = ok & (hh < 24).to_numpy(dtype=bool, na_value=False)
    in_range &= (mm < 60).to_numpy(dtype=bool, na_value=False)
    in_range &= (ss <= 60).to_numpy(dtype=bool, na_value=False)
    issues.flag(~in_range, "time", "time: not HHMMSS[.s] within a day")
    rollovers = int(((ss == 60).to_numpy(dtype=bool, na_value=False) & in_range).sum())
    offset = (hh * 3600 + mm * 60 + ss) * _NS_PER_S + frac_ns
    offset = offset.where(in_range, 0).astype("int64")
    ns = day.to_numpy(dtype="datetime64[ns]") + offset.to_numpy().astype(
        "timedelta64[ns]"
    )
    out = pd.Series(pd.DatetimeIndex(ns).tz_localize("UTC"), index=date.index)
    return out.where(~pd.isna(day).to_numpy()), rollovers


def _convert(
    raw: Mapping[str, Any],
    kind: FileKind,
    layout: str,
    issues: _Issues,
) -> tuple[pd.DataFrame, int]:
    """Stripped string fields of one block of lines → typed columns."""
    n = len(issues.line_no)
    index = pd.RangeIndex(n)
    time, rollovers = _times(raw["date"], raw["time"], issues)
    cols: dict[str, Any] = {
        "line_no": pd.Series(issues.line_no, index=index, dtype="int64"),
        "layout": pd.Series([layout] * n, index=index, dtype=STRING_DTYPE),
        "time": time.astype(TIME_DTYPE),
    }
    for name, fk in _KINDS[kind].items():
        if name not in raw:
            cols[name] = pd.Series(
                [pd.NA if fk == "int" else np.nan] * n,
                index=index,
                dtype=_dtype(fk),
            )
        elif fk == "str":
            s = raw[name]
            cols[name] = s.where(s != "").astype(STRING_DTYPE)
        else:
            cols[name] = _numeric(raw[name], name, fk, issues)
    for req in ("latitude", "longitude"):
        if req in raw:
            missing = cols[req].isna().to_numpy() & (raw[req] == "").to_numpy(
                dtype=bool
            )
            issues.flag(missing, req, f"{req}: blank")
    return pd.DataFrame(cols, index=index), rollovers


def _read_lines(path: Path) -> list[str]:
    """Lines of a file, newline (and a CR before it) removed, blanks kept."""
    text = path.read_bytes().decode("latin-1")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln[:-1] if ln.endswith("\r") else ln for ln in lines]


def _parse(path: str | os.PathLike[str], kind: FileKind) -> ParsedFile:
    p = Path(path)
    lines = _read_lines(p)
    blank = sum(1 for ln in lines if not ln.strip())
    malformed: list[LineIssue] = []
    warnings: list[LineIssue] = []
    layout_counts: dict[str, int] = {}
    blocks: list[pd.DataFrame] = []
    rollovers = 0
    overflows: dict[str, int] = {}
    nan_text: dict[str, int] = {}

    # group the data lines by layout
    groups: dict[str, list[tuple[int, str]]] = {}
    for i, ln in enumerate(lines, start=1):
        if not ln.strip():
            continue
        if kind == "events.fps":
            nf = len(ln.split())
            if nf != len(EVENTS_FPS_FIELDS):
                malformed.append(
                    LineIssue(
                        i,
                        f"{nf} fields, expected {len(EVENTS_FPS_FIELDS)}",
                        ln[:_MAX_TEXT],
                    )
                )
                continue
            groups.setdefault(EVENTS_FPS_LAYOUT, []).append((i, ln))
            continue
        lay = LAYOUTS[kind].get(len(ln))
        if lay is None:
            malformed.append(
                LineIssue(i, f"unrecognized width {len(ln)}", ln[:_MAX_TEXT])
            )
            continue
        groups.setdefault(lay.name, []).append((i, ln))

    by_name = {lay.name: lay for lay in LAYOUTS.get(kind, {}).values()}
    for name, rows in groups.items():
        line_no = [r[0] for r in rows]
        text = [r[1] for r in rows]
        issues = _Issues(line_no, text)
        s = pd.Series(text, dtype=STRING_DTYPE)
        raw: dict[str, Any]
        if kind == "events.fps":
            tokens = pd.DataFrame([t.split() for t in text], columns=EVENTS_FPS_FIELDS)
            raw = {c: tokens[c].astype(STRING_DTYPE) for c in EVENTS_FPS_FIELDS}
        else:
            raw = {
                field: s.str.slice(a, b).str.strip()
                for field, a, b in by_name[name].fields
            }
        block, n_roll = _convert(raw, kind, name, issues)
        keep = ~issues.drop
        blocks.append(block.loc[keep])
        rollovers += n_roll
        malformed.extend(issues.malformed)
        dropped = {int(n) for n in issues.line_no[issues.drop]}
        warnings.extend(w for w in issues.warnings if w.line_no not in dropped)
        for src, dst in ((issues.overflows, overflows), (issues.nan_text, nan_text)):
            for col, n in src.items():
                dst[col] = dst.get(col, 0) + n
        layout_counts[name] = int(keep.sum())

    if blocks:
        frame = pd.concat(blocks).sort_values("line_no", kind="stable")
        frame = frame.reset_index(drop=True)
    else:
        frame = _empty_frame(kind)
    sentinels = _apply_sentinels(frame, kind) if len(frame) else {}
    for col, n in nan_text.items():
        sentinels[col] = sentinels.get(col, 0) + n
    if kind == "events.fps" and not len(frame):
        frame["has_solution"] = frame["has_solution"].astype("bool")
    frame = frame[list(output_columns(kind))]

    used = [k for k, v in layout_counts.items() if v]
    if not used:
        layout = "empty"
    elif len(used) == 1:
        layout = used[0]
    else:
        layout = "mixed"
    report = ParseReport(
        path=str(p),
        kind=kind,
        layout=layout,
        layout_counts={k: v for k, v in layout_counts.items() if v},
        lines=len(lines),
        blank_lines=blank,
        rows=len(frame),
        malformed=tuple(sorted(malformed, key=lambda m: m.line_no)),
        field_warnings=tuple(sorted(warnings, key=lambda m: m.line_no)),
        sentinels=sentinels,
        overflows=overflows,
        time_rollovers=rollovers,
    )
    return ParsedFile(frame, report)


# --------------------------------------------------------------------------
# Public parsers
# --------------------------------------------------------------------------


def read_events_fps(path: str | os.PathLike[str]) -> ParsedFile:
    """Read one ``events.fps`` (reviewed events with fault-plane solution).

    Columns (native meaning; :func:`output_columns`): location and its errors
    in **degrees** (``d_latitude``, ``d_longitude``) and km; ``mlw`` **rounded
    to 0.1** (``events.lib``/``lib.mag`` have 0.01); ``d_depth_mod100``, the
    depth error **without its hundreds digit** (take Ddepth from ``lib.mag``);
    ``seismic_moment`` in N·m; ``ev_mag_ath`` and ``unknown_pacc`` (meaning not
    verified); the 2-letter ``event_type`` code; the stress/source fields;
    ``source_file`` (``sp3``, ``ap3``, or ``FILE`` for no solution); the two
    fault planes **as written** (``strike_a`` … ``rake_b``, Aki-Richards in the
    file, strike may be negative; not converted here, that is T3b); polarity
    and amplitude counts; ``is_significant`` (0/1); and ``has_solution``
    (False on rows without a fault-plane solution, whose solution fields are
    then NA).
    """
    return _parse(path, "events.fps")


def read_events_lib(path: str | os.PathLike[str]) -> ParsedFile:
    """Read one ``events.lib`` (reviewed events: Mlw to 0.01, type, id).

    The 1995 layout has a blank event type (NA here) and space-padded seconds.
    ``azim_comp`` is rounded to 0.1 deg (``events.fps`` truncates it).
    """
    return _parse(path, "events.lib")


def read_lib_mag(path: str | os.PathLike[str]) -> ParsedFile:
    """Read one ``lib.mag`` (reviewed magnitudes, phase counts, errors, gap).

    No id column: it joins to ``events.fps`` / ``events.lib`` by time (and, on
    most days, row order; not on days with no-solution fps rows). ML,
    ``min_ml``, ``max_ml`` and ``mlw`` written as ∓9.99 ("no ML") are NA.
    ``d_depth`` is the full depth error (``events.fps`` truncates it). The
    1990-91 layout has no ``w_rms_res`` (NA).
    """
    return _parse(path, "lib.mag")


def read_aut_mag(path: str | os.PathLike[str]) -> ParsedFile:
    """Read one ``aut.mag`` (automatic events), any of its four layouts.

    ``ml`` is the automatic ML, ``q`` the quality 0-100, ``id`` the event
    path. Layouts (per line, by width): 89 (ML, Q, id), 57 (no id), 61 (no Q,
    no id) and 117 (1992-93: unidentified ``aux1`` … ``aux10``, ``ml`` NA).
    """
    return _parse(path, "aut.mag")


def read_cleanaut_mag(path: str | os.PathLike[str]) -> ParsedFile:
    """Read one ``cleanaut.mag`` (cleaned automatic events, 89-character layout).

    Same fields as :func:`read_aut_mag`, but its own row numbers, and for a
    given id its values can differ from ``aut.mag``'s.
    """
    return _parse(path, "cleanaut.mag")


_READERS: Final[Mapping[str, Callable[[str | os.PathLike[str]], ParsedFile]]] = {
    "events.fps": read_events_fps,
    "events.lib": read_events_lib,
    "lib.mag": read_lib_mag,
    "aut.mag": read_aut_mag,
    "cleanaut.mag": read_cleanaut_mag,
}

#: File names this module can read.
FILE_KINDS: Final[tuple[FileKind, ...]] = (
    "events.fps",
    "events.lib",
    "lib.mag",
    "aut.mag",
    "cleanaut.mag",
)


def read_sil_file(path: str | os.PathLike[str]) -> ParsedFile:
    """Read a SIL day file, choosing the parser from its file name."""
    name = Path(path).name
    reader = _READERS.get(name)
    if reader is None:
        raise ValueError(
            f"not a SIL day catalogue file: {name!r} (known: {FILE_KINDS})"
        )
    return reader(path)


__all__ = [
    "AUT_MAG_57",
    "AUT_MAG_61",
    "AUT_MAG_89",
    "AUT_MAG_117",
    "AUX_COLUMNS",
    "EVENTS_FPS_FIELDS",
    "EVENTS_FPS_LAYOUT",
    "EVENTS_LIB_136",
    "FILE_KINDS",
    "FPS_NO_SOLUTION_SENTINELS",
    "FPS_NO_SOLUTION_SOURCE",
    "LAYOUTS",
    "LIB_MAG_134",
    "LIB_MAG_141",
    "SENTINELS",
    "STRING_SENTINELS",
    "FileKind",
    "Layout",
    "LineIssue",
    "ParseReport",
    "ParsedFile",
    "output_columns",
    "read_aut_mag",
    "read_cleanaut_mag",
    "read_events_fps",
    "read_events_lib",
    "read_lib_mag",
    "read_sil_file",
]
