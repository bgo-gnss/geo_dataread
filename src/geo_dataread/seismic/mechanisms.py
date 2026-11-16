"""Fault planes from the SIL files into the ``mechanisms`` frame (ticket T3b).

Two file kinds carry SIL fault-plane solutions:

* ``events.fps`` (one row per reviewed event, parsed by
  :func:`geo_dataread.seismic.sil_files.read_events_fps`): both planes in the
  Aki & Richards convention as whole degrees, with the strike sometimes
  negative on a vertical plane, plus the solution's moment, stress drop,
  corner frequency and polarity/amplitude counts. Column 23 (``source_file``)
  names the block the row was taken from: ``sp3``, ``ap3`` or ``FILE`` (no
  solution).
* the per-event block ``<day>/<HH:MM:SS>/<MM:SS:mmm>.sp3`` (and, 1995-2020,
  ``.ap3``), the fault-plane program's text output, parsed by
  :func:`read_sp3`. Its planes are **not** Aki & Richards: the strike is
  left-handed and the dip runs to 180. :func:`sp3_to_aki_richards` converts.

Decisions (PLAN-seismicity.md §9b, verified in T3b):

* **fps and its block are one solution.** The fps planes are the block's
  planes converted and rounded (geometric match on every row checked, see
  the T3b report); moment, stress drop and corner frequency are copied. So
  one row is written, ``method="sil_fps"`` (the label the schema also gives
  SIL's solution read from the legacy API), with the block's extra figures
  (``quality``, ``misfit``) attached. The block file is ``<id>.<source_file>``,
  never assumed to be ``.sp3``: an event can have both an ``.sp3`` and an
  ``.ap3`` holding different solutions (1995-03-19 ``42:51:202``).
* **ap3** is the same program's output in the same format (only the
  amplitude-count line may lack its count) and converts by the same rule,
  but its role next to sp3 is not documented in the files. fps rows citing
  it get :data:`FPS_BLOCK_METHOD` ``["ap3"]`` = ``"sil_ap3"``.
* **sp3-only rows** (``method="sil_sp3"``) are written only on days that
  have no ``events.fps`` (1990 to late 1991), from the reviewed ids and
  times in ``events.lib``. On days with an ``events.fps`` the reviewer's
  verdict stands: blocks of ``FILE`` rows, the other block of an ``ap3`` row,
  and blocks of ids that are not in the day's catalogue are counted in the
  report and give no row.

T12b adds :func:`pt_axes` and the computed, source-independent
:func:`faulting_class` (Frohlich 1992 by default, or the World Stress Map
regimes), fills the schema column ``sil_faulting_class`` from the block's
label, and marks (never drops) rows whose two planes are not each other's
auxiliary plane (:data:`PLANES_INCONSISTENT`, :func:`validate_marked_mechanisms`).

Angles out: strike in [0, 360), dip in [0, 90], rake in (-180, 180].
``quality`` is the block's "% of all mechanisms are acceptable" (0-100;
not fps ``unknown_pacc``, which is a flat 100.00 from 2022); ``misfit`` the
block's amplitude "mean error factor" of the optimal mechanism (1 = perfect).
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from geo_dataread.seismic.schema import (
    DEFAULT_PLANE_TOL_DEG,
    INDEX_NAME,
    MECHANISM_METHODS,
    STRING_DTYPE,
    TIME_DTYPE,
    FloatArray,
    ValidationReport,
    auxiliary_plane,
    coerce_mechanisms,
    column_names,
    plane_mismatch_deg,
    validate_mechanisms,
)
from geo_dataread.seismic.sil_files import (
    LineIssue,
    read_events_fps,
    read_events_lib,
)

#: ``source`` of every row this module writes.
SOURCE: Final = "sil-files"

#: fps ``source_file`` (the block the solution came from) → ``method``.
#: ``ap3`` maps to ``sil_ap3`` (same program and block format as sp3; its
#: role next to sp3 is undocumented); a value not in this table maps to ``other``.
FPS_BLOCK_METHOD: Final[Mapping[str, str]] = {"sp3": "sil_fps", "ap3": "sil_ap3"}

#: Block extensions the fault-plane program writes.
BLOCK_KINDS: Final = ("sp3", "ap3")

#: Extra (non-schema) columns written with ``extras=True``. ``faulting_class``
#: is **deprecated** (T12b): it repeats the schema column
#: ``sil_faulting_class`` (SIL's own label) and is kept only so older callers
#: and caches keep working; it is dropped once T18b reads the schema column.
#: The computed, source-independent class is :func:`faulting_class`.
EXTRA_COLUMNS: Final = (
    "block",
    "faulting_class",
    "significant",
    "seismic_moment_nm",
    "relative_size",
    "p_axis_azimuth",
    "p_axis_dip",
    "t_axis_azimuth",
    "t_axis_dip",
    "block_plane_mismatch_deg",
)

_MAX_TEXT: Final = 200

# --------------------------------------------------------------------------
# Plane conventions
# --------------------------------------------------------------------------


def _floats(*values: npt.ArrayLike) -> list[FloatArray]:
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    return [np.asarray(a, dtype=np.float64) for a in np.broadcast_arrays(*arrays)]


def normalize_plane(
    strike: npt.ArrayLike, dip: npt.ArrayLike, rake: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Wrap strike into [0, 360) and rake into (-180, 180]; dip unchanged.

    The plane is not otherwise altered: a vertical plane keeps the
    description it was given (``-86/90/28`` → ``274/90/28``, not the
    equivalent ``94/90/-28``). NaN in gives NaN out.
    """
    s, d, r = _floats(strike, dip, rake)
    s_out = np.mod(s, 360.0)
    s_out = np.where(np.isclose(s_out, 360.0, atol=1e-9, rtol=0.0), 0.0, s_out)
    r_out = np.mod(r + 180.0, 360.0) - 180.0
    r_out = np.where(np.isclose(r_out, -180.0, atol=1e-9, rtol=0.0), 180.0, r_out)
    return s_out, d.copy(), r_out


def sp3_to_aki_richards(
    strike: npt.ArrayLike, dip: npt.ArrayLike, rake: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert a raw sp3/ap3 plane to Aki & Richards (PLAN §9b rule).

    The block writes a left-hand strike with a dip in [0, 180]:

    * dip ≥ 90: ``(s, 180 - d, 180 - r)``;
    * otherwise: ``(s + 180, d, r + 180)``;

    then :func:`normalize_plane`. At dip = 90 both branches give the same
    plane and slip but different numbers, so compare results geometrically
    (:func:`same_plane_deg`), not by value. A dip outside [0, 180], or any NaN, gives an all-NaN plane.
    Scalars or arrays (broadcast); float arrays out.
    """
    s, d, r = _floats(strike, dip, rake)
    steep = d >= 90.0
    s2 = np.where(steep, s, s + 180.0)
    d2 = np.where(steep, 180.0 - d, d)
    r2 = np.where(steep, 180.0 - r, r + 180.0)
    with np.errstate(invalid="ignore"):
        bad = (d < 0.0) | (d > 180.0) | np.isnan(s) | np.isnan(d) | np.isnan(r)
    nan = np.float64(np.nan)
    s2, d2, r2 = (np.where(bad, nan, x) for x in (s2, d2, r2))
    return normalize_plane(s2, d2, r2)


def same_plane_deg(
    strike1: npt.ArrayLike,
    dip1: npt.ArrayLike,
    rake1: npt.ArrayLike,
    strike2: npt.ArrayLike,
    dip2: npt.ArrayLike,
    rake2: npt.ArrayLike,
) -> FloatArray:
    """How far plane 1 (with its slip) is from plane 2, degrees.

    0 when both describe the same plane and slip direction, whatever the
    numbers (strike/rake wrap, the two descriptions of a vertical plane).
    It is :func:`~geo_dataread.seismic.schema.plane_mismatch_deg` of plane 1
    against the auxiliary plane of plane 2. A swapped pair (plane 1 = the
    auxiliary of plane 2) is *not* 0 here: this checks plane identity, not
    double-couple identity.
    """
    s2, d2, r2 = auxiliary_plane(strike2, dip2, rake2)
    return plane_mismatch_deg(strike1, dip1, rake1, s2, d2, r2)


# --------------------------------------------------------------------------
# P, B, T axes and the faulting class (T12b, PLAN R3)
# --------------------------------------------------------------------------


class PBTAxes(NamedTuple):
    """Azimuth (degrees clockwise from north, [0, 360)) and plunge (degrees
    below horizontal, [0, 90]) of the P, B and T axes; float arrays."""

    p_azimuth: FloatArray
    p_plunge: FloatArray
    b_azimuth: FloatArray
    b_plunge: FloatArray
    t_azimuth: FloatArray
    t_plunge: FloatArray


def _normal_and_slip(
    strike: FloatArray, dip: FloatArray, rake: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Unit fault normal and slip (north, east, down), Aki & Richards (2002)
    eq. 4.88 (the same vectors as the schema's double-couple helpers)."""
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


def _trend_plunge(v: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Azimuth/plunge of axes (3, ...) in (north, east, down), lower hemisphere.

    An axis has no sign, so it is turned to point down; a horizontal axis is
    given with its azimuth in [0, 180), a vertical one with azimuth 0.
    """
    norm = np.linalg.norm(v, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = v / norm
    up = u[2] < 0.0
    u = np.where(up, -u, u)
    plunge = np.degrees(np.arcsin(np.clip(u[2], 0.0, 1.0)))
    az = np.mod(np.degrees(np.arctan2(u[1], u[0])), 360.0)
    flat = plunge < 1e-7
    az = np.where(flat & (az >= 180.0 - 1e-9), az - 180.0, az)
    az = np.where(np.isclose(az, 360.0, atol=1e-9, rtol=0.0), 0.0, az)
    az = np.where(plunge > 90.0 - 1e-6, 0.0, az)
    return az, plunge


def pt_axes(strike: npt.ArrayLike, dip: npt.ArrayLike, rake: npt.ArrayLike) -> PBTAxes:
    """The P (pressure), B (null) and T (tension) axes of a double couple.

    With unit normal n and slip s of either nodal plane (Aki & Richards
    convention): T ∝ n + s, P ∝ n − s, B = n × s. The two nodal planes give
    the same axes, so the result does not depend on which plane is passed.
    Plunges are positive down (lower hemisphere), so an axis is described
    once. Scalars or arrays (broadcast), float arrays out (0-d for scalars);
    NaN in any angle gives NaN axes. Example: ``0/90/0`` (left-lateral on a
    N-S plane) has P at 45°, T at 135°, B vertical.
    """
    s, d, r = _floats(strike, dip, rake)
    n, u = _normal_and_slip(s, d, r)
    p_az, p_pl = _trend_plunge(n - u)
    t_az, t_pl = _trend_plunge(n + u)
    b_az, b_pl = _trend_plunge(np.cross(n, u, axis=0))
    missing = np.isnan(s) | np.isnan(d) | np.isnan(r)
    nan = np.float64(np.nan)
    out = (np.where(missing, nan, x) for x in (p_az, p_pl, b_az, b_pl, t_az, t_pl))
    return PBTAxes(*out)


#: Rules of :func:`faulting_class` → the classes each can return.
FAULTING_RULES: Final[Mapping[str, tuple[str, ...]]] = {
    "frohlich": ("strike-slip", "normal", "reverse", "odd"),
    "wsm": (
        "normal",
        "normal-oblique",
        "strike-slip",
        "reverse-oblique",
        "reverse",
        "odd",
    ),
}

#: The rule used when none is named.
DEFAULT_FAULTING_RULE: Final = "frohlich"


def _classes_frohlich(p: FloatArray, b: FloatArray, t: FloatArray) -> Any:
    return np.select(
        [b > 60.0, p > 60.0, t > 50.0],
        ["strike-slip", "normal", "reverse"],
        default="odd",
    )


def _classes_wsm(p: FloatArray, b: FloatArray, t: FloatArray) -> Any:
    return np.select(
        [
            (p >= 52.0) & (t <= 35.0),
            (p >= 40.0) & (p < 52.0) & (t <= 20.0),
            (p < 40.0) & (b >= 45.0) & (t <= 20.0),
            (p <= 20.0) & (b >= 45.0) & (t < 40.0),
            (p <= 20.0) & (t >= 40.0) & (t < 52.0),
            (p <= 35.0) & (t >= 52.0),
        ],
        [
            "normal",
            "normal-oblique",
            "strike-slip",
            "strike-slip",
            "reverse-oblique",
            "reverse",
        ],
        default="odd",
    )


def faulting_class(
    strike: npt.ArrayLike,
    dip: npt.ArrayLike,
    rake: npt.ArrayLike,
    *,
    rule: str = DEFAULT_FAULTING_RULE,
) -> npt.NDArray[np.object_]:
    """The faulting class of a double couple from the plunges of its axes.

    One computed class for every source (PLAN R3); SIL's own label stays
    data (``sil_faulting_class``). Only the plunges of P, B and T
    (:func:`pt_axes`) are used, so the class is the same whichever nodal
    plane is passed and whatever the strike/rake wrap.

    ``rule="frohlich"`` (default), Frohlich (1992), *Phys. Earth Planet.
    Inter.* 75, 193–198 (also Frohlich 2001, *J. Seismol.* 5, 187–196):

    * ``strike-slip`` — B plunge > 60°;
    * ``normal`` — P plunge > 60°;
    * ``reverse`` (Frohlich's "thrust") — T plunge > 50°;
    * ``odd`` — none of these (oblique and mixed mechanisms).

    The three are exclusive (sin² of the three plunges sums to 1).

    ``rule="wsm"``, the World Stress Map regimes, Zoback (1992), *J. Geophys.
    Res.* 97(B8), 11703–11728, table 1 (plunges P, B, T in degrees):
    ``normal`` (NF: P ≥ 52, T ≤ 35), ``normal-oblique`` (NS: 40 ≤ P < 52,
    T ≤ 20), ``strike-slip`` (SS: P < 40, B ≥ 45, T ≤ 20, or P ≤ 20,
    B ≥ 45, T < 40), ``reverse-oblique`` (TS: P ≤ 20, 40 ≤ T < 52),
    ``reverse`` (TF: P ≤ 35, T ≥ 52), else ``odd`` (WSM "U").

    SIL's labels map onto ``strike-slip`` / ``normal`` / ``reverse``; SIL's
    ``dip-slip`` has no counterpart (it lands in ``odd`` or one of the
    oblique classes). Scalars or arrays (broadcast); an object array of
    class names out (0-d for scalars, ``.item()`` gives the string), ``None``
    where an angle is NaN.
    """
    if rule not in FAULTING_RULES:
        raise ValueError(f"rule {rule!r} is not one of {tuple(FAULTING_RULES)}")
    ax = pt_axes(strike, dip, rake)
    fn = _classes_frohlich if rule == "frohlich" else _classes_wsm
    with np.errstate(invalid="ignore"):
        names = np.asarray(fn(ax.p_plunge, ax.b_plunge, ax.t_plunge), dtype=object)
    missing = np.isnan(ax.p_plunge)
    out: npt.NDArray[np.object_] = names.copy()
    out[missing] = None
    return out


def faulting_classes(
    df: pd.DataFrame, *, rule: str = DEFAULT_FAULTING_RULE
) -> pd.Series:
    """:func:`faulting_class` of plane 1 of every row of a mechanisms frame,
    as a categorical Series on the frame's index (categories in
    :data:`FAULTING_RULES` order; NA where plane 1 is missing)."""
    cls = faulting_class(
        df["strike1"].to_numpy(dtype=np.float64),
        df["dip1"].to_numpy(dtype=np.float64),
        df["rake1"].to_numpy(dtype=np.float64),
        rule=rule,
    )
    dtype = pd.CategoricalDtype(list(FAULTING_RULES[rule]), ordered=False)
    return pd.Series(list(cls), index=df.index, dtype=dtype, name="faulting_class")


# --------------------------------------------------------------------------
# Inconsistent plane pairs: kept and marked (BGÓ, T12b)
# --------------------------------------------------------------------------

#: The marker column: True where plane 2 is not the auxiliary plane of plane
#: 1 within the tolerance, False where it is, NA where plane 2 (or 1) is
#: missing. Such rows are **kept** (e.g. the two 2023 fps rows with
#: 180/39/−79 and 14/52/−81, 22° apart). Not yet a schema column: every
#: mechanisms frame of this package carries it after the schema columns
#: (:func:`output_columns`).
PLANES_INCONSISTENT: Final = "planes_inconsistent"


def output_columns() -> tuple[str, ...]:
    """Schema mechanisms columns, then :data:`PLANES_INCONSISTENT` unless the
    schema already has it."""
    cols = column_names("mechanisms")
    return cols if PLANES_INCONSISTENT in cols else (*cols, PLANES_INCONSISTENT)


def planes_inconsistent(
    df: pd.DataFrame, *, plane_tol_deg: float = DEFAULT_PLANE_TOL_DEG
) -> pd.Series:
    """The :data:`PLANES_INCONSISTENT` marker of a mechanisms frame (nullable
    boolean Series on its index), from
    :func:`~geo_dataread.seismic.schema.plane_mismatch_deg`."""
    a = [
        pd.to_numeric(df[f"{x}{n}"], errors="coerce").to_numpy(dtype=np.float64)
        for n in (1, 2)
        for x in ("strike", "dip", "rake")
    ]
    mis = plane_mismatch_deg(*a)
    known = ~np.isnan(mis)
    with np.errstate(invalid="ignore"):
        flag = mis > plane_tol_deg
    values = pd.array(np.where(known, flag, False), dtype="boolean")
    values[~known] = pd.NA
    return pd.Series(values, index=df.index, name=PLANES_INCONSISTENT)


def with_planes_inconsistent(
    df: pd.DataFrame, *, plane_tol_deg: float = DEFAULT_PLANE_TOL_DEG
) -> pd.DataFrame:
    """``df`` with the :data:`PLANES_INCONSISTENT` column set (in place) and
    returned."""
    df[PLANES_INCONSISTENT] = planes_inconsistent(df, plane_tol_deg=plane_tol_deg)
    return df


def validate_marked_mechanisms(
    df: pd.DataFrame,
    *,
    strict: bool = False,
    plane_tol_deg: float = DEFAULT_PLANE_TOL_DEG,
) -> ValidationReport:
    """:func:`~geo_dataread.seismic.schema.validate_mechanisms`, with
    inconsistent plane pairs that are **marked** reported as a warning.

    The schema check calls a pair that misses its auxiliary plane an error.
    Rows whose :data:`PLANES_INCONSISTENT` is True are kept on purpose, so
    when every such row is marked the ``planes_inconsistent`` issue becomes a
    warning and a strict check passes; an unmarked one stays an error.
    """
    base = validate_mechanisms(df, strict=False, plane_tol_deg=plane_tol_deg)
    if PLANES_INCONSISTENT in df.columns and "planes_inconsistent" in base.kinds():
        marked = df[PLANES_INCONSISTENT].astype("boolean").fillna(False).to_numpy()
        found = planes_inconsistent(df, plane_tol_deg=plane_tol_deg)
        found_np = found.fillna(False).to_numpy(dtype=bool)
        if not (found_np & ~marked.astype(bool)).any():
            issues = tuple(
                (
                    dataclasses.replace(
                        i,
                        severity="warning",
                        message=f"{i.message}; kept, marked in {PLANES_INCONSISTENT}",
                    )
                    if i.kind == "planes_inconsistent"
                    else i
                )
                for i in base.issues
            )
            base = dataclasses.replace(base, issues=issues)
    if strict:
        base.raise_for_errors()
    return base


# --------------------------------------------------------------------------
# The sp3 / ap3 block
# --------------------------------------------------------------------------

#: A number as the block writes it (``-19.``, ``.95``, ``0.177E+11``), or
#: ``nan``, which stands in for a Fortran overflow (``****``) by the time
#: the patterns see the line.
_N: Final = r"([-+]?(?:(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?|(?i:nan)))"
_OVERFLOW: Final = re.compile(r"\*+")

FAULTING_CLASSES: Final[Mapping[str, str]] = {
    "Strike slip": "strike-slip",
    "Normal": "normal",
    "Reverse": "reverse",
    "Dip-slip": "dip-slip",
}


@dataclasses.dataclass(frozen=True)
class Sp3Block:
    """One parsed ``.sp3`` / ``.ap3`` block. Every field is ``None`` if absent.

    Planes are **raw** (as written; :func:`sp3_to_aki_richards` converts).
    Ranges are ``(*_min, *_max)`` with the preferred value (in parentheses in
    the file) in the unsuffixed field. Units as named.
    """

    header_path: str | None = None
    date_text: str | None = None
    lib_mlw: float | None = None
    lib_sh_min: float | None = None
    lib_sh_max: float | None = None
    lib_azim_comp: float | None = None
    lib_freq_hz: float | None = None
    seismic_moment_nm: float | None = None
    local_magnitude: float | None = None
    corner_freq_min_hz: float | None = None
    corner_freq_max_hz: float | None = None
    corner_freq_hz: float | None = None
    fault_radius_min_m: float | None = None
    fault_radius_max_m: float | None = None
    fault_radius_m: float | None = None
    stress_drop_min_mpa: float | None = None
    stress_drop_max_mpa: float | None = None
    stress_drop_mpa: float | None = None
    peak_slip_min_mm: float | None = None
    peak_slip_max_mm: float | None = None
    peak_slip_mm: float | None = None
    p_axis_azimuth: float | None = None
    p_axis_dip: float | None = None
    t_axis_azimuth: float | None = None
    t_axis_dip: float | None = None
    azimuth_of_compression: float | None = None
    relative_size: float | None = None
    strike_a: float | None = None
    dip_a: float | None = None
    rake_a: float | None = None
    strike_b: float | None = None
    dip_b: float | None = None
    rake_b: float | None = None
    normal_a_azimuth: float | None = None
    normal_a_dip: float | None = None
    normal_b_azimuth: float | None = None
    normal_b_dip: float | None = None
    faulting_class: str | None = None
    n_polarities: int | None = None
    n_polarities_required: int | None = None
    n_polarity_misfits: int | None = None
    misfit_stations: tuple[str, ...] = ()
    n_amplitudes: int | None = None
    n_amplitude_stations: int | None = None
    amp_error_limit: float | None = None
    pct_acceptable: float | None = None
    pct_acceptable_first_motion: float | None = None
    pct_fitted_amplitudes: float | None = None
    pct_well_fitting_planes: float | None = None
    amp_mean_error_factor: float | None = None
    amp_std_dev_factor: float | None = None
    significant: bool | None = None
    significance_level_pct: float | None = None
    significance_qualifier: str | None = None
    f_dof: tuple[int, int] | None = None
    f_value: float | None = None
    p_spectrum_misfit: float | None = None
    s_spectrum_misfit: float | None = None
    low_freq_limit_hz: float | None = None

    @property
    def has_planes(self) -> bool:
        """Both raw planes were read."""
        return None not in (
            self.strike_a,
            self.dip_a,
            self.rake_a,
            self.strike_b,
            self.dip_b,
            self.rake_b,
        )

    def planes_aki_richards(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Both planes converted (NaN where absent)."""

        def conv(s: float | None, d: float | None, r: float | None) -> Any:
            out = sp3_to_aki_richards(
                np.nan if s is None else s,
                np.nan if d is None else d,
                np.nan if r is None else r,
            )
            return tuple(float(x) for x in out)

        a = conv(self.strike_a, self.dip_a, self.rake_a)
        b = conv(self.strike_b, self.dip_b, self.rake_b)
        return a, b


@dataclasses.dataclass(frozen=True)
class Sp3Report:
    """What :func:`read_sp3` could and could not read."""

    path: str
    lines: int
    #: Non-blank lines that matched no known pattern.
    unparsed: tuple[LineIssue, ...] = ()
    #: Lines that matched a pattern but whose numbers could not be read
    #: (``f(**,**)``).
    field_warnings: tuple[LineIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Nothing unparsed and no field warning."""
        return not self.unparsed and not self.field_warnings


class ParsedSp3(NamedTuple):
    """:func:`read_sp3`'s result."""

    block: Sp3Block
    report: Sp3Report


def _f(text: str) -> float:
    return float(text)


#: Single-line patterns → handler(match) → field updates. Order matters only
#: where two could match the same line (none do).
_Handler = Any
_LINE_PATTERNS: Final[tuple[tuple[re.Pattern[str], _Handler], ...]] = (
    (re.compile(r"^\s*file:\s*(\S+)"), lambda m: {"header_path": m[1]}),
    (
        re.compile(rf"^\s*lib\s*{_N}\s*{_N}\s*{_N}(?:\s*{_N})?(?:\s*{_N}\s*Hz)?\s*$"),
        lambda m: {
            "lib_mlw": _f(m[1]),
            "lib_sh_min": _f(m[2]),
            "lib_sh_max": _f(m[3]),
            "lib_azim_comp": None if m[4] is None else _f(m[4]),
            "lib_freq_hz": None if m[5] is None else _f(m[5]),
        },
    ),
    (
        re.compile(rf"seismic moment:\s*{_N}\s*Nm"),
        lambda m: {"seismic_moment_nm": _f(m[1])},
    ),
    (
        re.compile(rf"local magnitude:\s*{_N}"),
        lambda m: {"local_magnitude": _f(m[1])},
    ),
    (
        re.compile(rf"^\s*{_N}\s*Hz\s*-?\s*{_N}\s*Hz\s*\(\s*{_N}\s*Hz\)"),
        lambda m: {
            "corner_freq_min_hz": abs(_f(m[1])),
            "corner_freq_max_hz": abs(_f(m[2])),
            "corner_freq_hz": _f(m[3]),
        },
    ),
    (
        re.compile(rf"Fault radius range\s*{_N}\s*m\s*-?\s*{_N}\s*m\s*\(\s*{_N}\s*m\)"),
        lambda m: {
            "fault_radius_min_m": abs(_f(m[1])),
            "fault_radius_max_m": abs(_f(m[2])),
            "fault_radius_m": _f(m[3]),
        },
    ),
    (
        re.compile(
            rf"Stress drop range\s*{_N}\s*Mpa\s*-?\s*{_N}\s*Mpa\s*\(\s*{_N}\s*Mpa\)"
        ),
        lambda m: {
            "stress_drop_min_mpa": abs(_f(m[1])),
            "stress_drop_max_mpa": abs(_f(m[2])),
            "stress_drop_mpa": _f(m[3]),
        },
    ),
    (
        re.compile(
            rf"peak slip at the fault\s*{_N}\s*mm\s*-?\s*{_N}\s*mm\s*\(\s*{_N}\s*mm\)"
        ),
        lambda m: {
            "peak_slip_min_mm": abs(_f(m[1])),
            "peak_slip_max_mm": abs(_f(m[2])),
            "peak_slip_mm": _f(m[3]),
        },
    ),
    (
        re.compile(rf"^\s*P-axis\s*{_N}\s+{_N}"),
        lambda m: {"p_axis_azimuth": _f(m[1]), "p_axis_dip": _f(m[2])},
    ),
    (
        re.compile(rf"^\s*T-axis\s*{_N}\s+{_N}"),
        lambda m: {"t_axis_azimuth": _f(m[1]), "t_axis_dip": _f(m[2])},
    ),
    (
        re.compile(rf"azimuth of compression\s*{_N}\s*degrees"),
        lambda m: {"azimuth_of_compression": _f(m[1])},
    ),
    (
        re.compile(rf"the relative size\s*{_N}"),
        lambda m: {"relative_size": _f(m[1])},
    ),
    (
        re.compile(r"^\s*(Strike slip|Normal|Reverse|Dip-slip) faulting dominates"),
        lambda m: {"faulting_class": FAULTING_CLASSES[m[1]]},
    ),
    (
        re.compile(r"Of\s*(\d+) first motion polarity observations"),
        lambda m: {"n_polarities": int(m[1])},
    ),
    (
        re.compile(r"at least\s*(-?\d+) are required to fit"),
        lambda m: {"n_polarities_required": int(m[1])},
    ),
    (
        re.compile(r"optimum mechanism has\s*(\d+) polarity misfits"),
        lambda m: {"n_polarity_misfits": int(m[1])},
    ),
    (
        re.compile(r"at the station\(s\)\s*(.*?)\s*$"),
        lambda m: {"misfit_stations": tuple(m[1].split())},
    ),
    (
        re.compile(
            r"^\s*(\d+)?\s*[Aa]mplitudes for P and S at\s*(\d+) stations are used"
        ),
        lambda m: {
            "n_amplitudes": None if m[1] is None else int(m[1]),
            "n_amplitude_stations": int(m[2]),
        },
    ),
    (
        re.compile(rf"^\s*than\s*{_N} for single p-wave"),
        lambda m: {"amp_error_limit": _f(m[1])},
    ),
    (
        re.compile(rf"^\s*{_N}\s*% of all mechanisms are acceptable"),
        lambda m: {"pct_acceptable": _f(m[1])},
    ),
    (
        re.compile(rf"^\s*{_N}\s*% acceptable due to first motion"),
        lambda m: {"pct_acceptable_first_motion": _f(m[1])},
    ),
    (
        re.compile(rf"^\s*{_N}\s*% of these fitted also the amplitudes"),
        lambda m: {"pct_fitted_amplitudes": _f(m[1])},
    ),
    (
        re.compile(rf"part of well fitting planes is\s*{_N}\s*%"),
        lambda m: {"pct_well_fitting_planes": _f(m[1])},
    ),
    (
        re.compile(rf"mean error factor of\s*{_N}"),
        lambda m: {"amp_mean_error_factor": _f(m[1])},
    ),
    (
        re.compile(rf"standard deviation factor of\s*{_N}"),
        lambda m: {"amp_std_dev_factor": _f(m[1])},
    ),
    (
        re.compile(r"double couple solution is (not )?significant"),
        lambda m: {"significant": m[1] is None},
    ),
    (
        re.compile(rf"^\s*(at better than|not even at|at)\s*{_N}\s*% level"),
        lambda m: {
            "significance_qualifier": m[1],
            "significance_level_pct": _f(m[2]),
        },
    ),
    (
        re.compile(rf"^\s*P-waves\s*{_N}"),
        lambda m: {"p_spectrum_misfit": _f(m[1])},
    ),
    (
        re.compile(rf"^\s*S-waves\s*{_N}"),
        lambda m: {"s_spectrum_misfit": _f(m[1])},
    ),
    (
        re.compile(rf"Low frequency limit\s*{_N}\s*Hz"),
        lambda m: {"low_freq_limit_hz": _f(m[1])},
    ),
)

#: Lines with no value of their own (headers and prose).
_TEXT_LINES: Final = re.compile(
    r"^\s*(DYNAMIC SOURCE PARAMETERS|Size measures|Shear wave corner frequency"
    r"|The orientation of the relaxed stress|azimuth\s+dip"
    r"|The horizontal deviatoric stress|strike\s+dip\s+slip"
    r"|STATISTICAL INFORMATION|Only mechanisms giving|deviation of the amplitude"
    r"|taken as acceptable|The amplitude fit of the optimal|for single P-wave"
    r"|The measure of the misfit)"
)
_PLANES_HEADER: Final = re.compile(r"The two possible fault planes")
_NORMALS_HEADER: Final = re.compile(r"The normal directions of the fault planes")
_PLANE_LINE: Final = re.compile(rf"^\s*plane ([ab])\s*{_N}\s+{_N}(?:\s+{_N})?")
_F_VALUE: Final = re.compile(r"\(f-value:\s*f\(\s*([^,]*),\s*([^)]*)\)\s*=\s*(\S+)\)")
_DATE_LINE: Final = re.compile(r"^\s*\d+(?:\s+\d+)+\s*$")


def read_sp3(path: str | os.PathLike[str]) -> ParsedSp3:
    """Parse one ``.sp3`` / ``.ap3`` fault-plane block.

    Each line is matched by its label (the block is partly fixed width, so
    fields can touch: ``lib-0.92``, ``local magnitude:-0.9``). Every field is
    optional; a line that matches nothing is listed in
    :attr:`Sp3Report.unparsed`, an unreadable number (``f(**,**)``) in
    :attr:`Sp3Report.field_warnings`. A missing file raises
    :class:`FileNotFoundError`.
    """
    p = Path(path)
    lines = p.read_text(encoding="latin-1").splitlines()
    fields: dict[str, Any] = {}
    unparsed: list[LineIssue] = []
    warnings: list[LineIssue] = []
    section: str | None = None
    for i, ln in enumerate(lines, start=1):
        if not ln.strip():
            continue
        if _PLANES_HEADER.search(ln):
            section = "planes"
            continue
        if _NORMALS_HEADER.search(ln):
            section = "normals"
            continue
        m = _PLANE_LINE.match(ln)
        if m and section in ("planes", "normals"):
            which = m[1]
            if section == "planes" and m[4] is not None:
                fields[f"strike_{which}"] = _f(m[2])
                fields[f"dip_{which}"] = _f(m[3])
                fields[f"rake_{which}"] = _f(m[4])
            elif section == "normals":
                fields[f"normal_{which}_azimuth"] = _f(m[2])
                fields[f"normal_{which}_dip"] = _f(m[3])
            else:
                unparsed.append(LineIssue(i, "plane line without rake", ln[:_MAX_TEXT]))
            continue
        fm = _F_VALUE.search(ln)
        if fm:
            try:
                fields["f_dof"] = (int(fm[1]), int(fm[2]))
            except ValueError:
                warnings.append(LineIssue(i, "f-value degrees of freedom", ln))
            try:
                fields["f_value"] = float(fm[3])
            except ValueError:
                warnings.append(LineIssue(i, "f-value", ln))
            continue
        if "*" in ln:
            warnings.append(LineIssue(i, "overflow (****) read as NA", ln[:_MAX_TEXT]))
            ln = _OVERFLOW.sub("nan", ln)
        for pattern, handler in _LINE_PATTERNS:
            hit = pattern.search(ln)
            if hit:
                fields.update(handler(hit))
                break
        else:
            if fields.get("date_text") is None and _DATE_LINE.match(ln):
                fields["date_text"] = ln.strip()
            elif not _TEXT_LINES.search(ln):
                unparsed.append(LineIssue(i, "no pattern matched", ln[:_MAX_TEXT]))
    fields = {
        k: None if isinstance(v, float) and np.isnan(v) else v
        for k, v in fields.items()
    }
    block = Sp3Block(**fields)
    report = Sp3Report(str(p), len(lines), tuple(unparsed), tuple(warnings))
    return ParsedSp3(block, report)


# --------------------------------------------------------------------------
# fps → mechanisms
# --------------------------------------------------------------------------


def block_path(day_dir: str | os.PathLike[str], event_id: str, kind: str) -> Path:
    """``<day_dir>/<HH:MM:SS>/<MM:SS:mmm>.<kind>`` for a SIL file id.

    Only the id's last two components are used, so the day directory may sit
    under any base path (archive or fixtures).
    """
    parts = [x for x in str(event_id).split("/") if x]
    if len(parts) < 2:
        raise ValueError(f"not a SIL file id: {event_id!r}")
    return Path(day_dir) / parts[-2] / f"{parts[-1]}.{kind}"


def _method(source_file: Any) -> str:
    method = FPS_BLOCK_METHOD.get(str(source_file), "other")
    return method if method in MECHANISM_METHODS else "other"


def _opt(value: Any) -> float:
    return np.nan if value is None else float(value)


def mechanisms_from_fps(
    fps: pd.DataFrame,
    *,
    blocks: Mapping[str, Sp3Block] | None = None,
    extras: bool = False,
    plane_tol_deg: float = DEFAULT_PLANE_TOL_DEG,
) -> pd.DataFrame:
    """The ``mechanisms`` frame from a :func:`read_events_fps` frame.

    One row per fps row with a solution (``has_solution``); no-solution rows
    give none. Planes: fps plane a → plane 1, plane b → plane 2, normalised
    (:func:`normalize_plane`; a negative strike is wrapped). ``method`` from
    ``source_file`` (:data:`FPS_BLOCK_METHOD`). Counts, stress drop and corner
    frequency come from fps; ``quality``, ``misfit`` and
    ``sil_faulting_class`` (the block's "… faulting dominates" label) from the
    matching block in ``blocks`` (keyed by fps ``id``; the caller must pass the block
    named by ``source_file``), NaN without one. ``extras=True`` adds
    :data:`EXTRA_COLUMNS`. The result is coerced to the schema
    (``source="sil-files"``, index ``time``, sorted).
    """
    blocks = blocks or {}
    rows = fps.loc[fps["has_solution"].to_numpy(dtype=bool)]
    s1, d1, r1 = normalize_plane(rows["strike_a"], rows["dip_a"], rows["rake_a"])
    s2, d2, r2 = normalize_plane(rows["strike_b"], rows["dip_b"], rows["rake_b"])
    ids = [str(x) for x in rows["id"]]
    blk = [blocks.get(i) for i in ids]
    data: dict[str, Any] = {
        INDEX_NAME: pd.DatetimeIndex(rows["time"], dtype=TIME_DTYPE),
        "source": SOURCE,
        "event_id": pd.array(ids, dtype=STRING_DTYPE),
        "method": [_method(x) for x in rows["source_file"]],
        "strike1": s1,
        "dip1": d1,
        "rake1": r1,
        "strike2": s2,
        "dip2": d2,
        "rake2": r2,
        "quality": [_opt(b.pct_acceptable if b else None) for b in blk],
        "misfit": [_opt(b.amp_mean_error_factor if b else None) for b in blk],
        "n_polarities": rows["num_of_polarities"].to_numpy(),
        "n_amplitudes": rows["num_of_amplitudes"].to_numpy(),
        "stress_drop_mpa": rows["stress_drop"].to_numpy(dtype=np.float64),
        "corner_freq_hz": rows["corner_freq"].to_numpy(dtype=np.float64),
        "sil_faulting_class": [b.faulting_class if b else None for b in blk],
    }
    if extras:
        mism = []
        for b, sa, da, ra in zip(blk, s1, d1, r1, strict=True):
            if b is None or not b.has_planes:
                mism.append(np.nan)
                continue
            (cs, cd, cr), _ = b.planes_aki_richards()
            mism.append(float(same_plane_deg(cs, cd, cr, sa, da, ra)))
        data.update(
            {
                "block": pd.array(
                    [str(x) for x in rows["source_file"]], dtype=STRING_DTYPE
                ),
                "faulting_class": pd.array(
                    [b.faulting_class if b else None for b in blk], dtype=STRING_DTYPE
                ),
                "significant": pd.array(
                    [pd.NA if pd.isna(v) else bool(v) for v in rows["is_significant"]],
                    dtype="boolean",
                ),
                "seismic_moment_nm": rows["seismic_moment"].to_numpy(np.float64),
                "relative_size": rows["relative_size"].to_numpy(np.float64),
                "p_axis_azimuth": [_opt(b.p_axis_azimuth if b else None) for b in blk],
                "p_axis_dip": [_opt(b.p_axis_dip if b else None) for b in blk],
                "t_axis_azimuth": [_opt(b.t_axis_azimuth if b else None) for b in blk],
                "t_axis_dip": [_opt(b.t_axis_dip if b else None) for b in blk],
                "block_plane_mismatch_deg": mism,
            }
        )
    return _finish_frame(data, len(rows), extras, plane_tol_deg)


def mechanisms_from_blocks(
    events: pd.DataFrame,
    blocks: Mapping[str, Sp3Block],
    *,
    method: str = "sil_sp3",
    extras: bool = False,
    plane_tol_deg: float = DEFAULT_PLANE_TOL_DEG,
) -> pd.DataFrame:
    """``mechanisms`` rows straight from blocks (days without ``events.fps``).

    ``events`` gives ``id`` and ``time`` (an :func:`read_events_lib` frame);
    one row per event whose id is in ``blocks`` and whose block has both
    planes, converted with :func:`sp3_to_aki_richards`. Counts, stress drop,
    corner frequency, ``quality``, ``misfit`` and ``sil_faulting_class`` from
    the block.
    """
    if method not in MECHANISM_METHODS:
        raise ValueError(f"method {method!r} is not one of {MECHANISM_METHODS}")
    times: list[Any] = []
    ids: list[str] = []
    use: list[Sp3Block] = []
    for t, i in zip(events["time"], events["id"], strict=True):
        b = blocks.get(str(i))
        if b is not None and b.has_planes:
            times.append(t)
            ids.append(str(i))
            use.append(b)
    planes = [b.planes_aki_richards() for b in use]
    col = {
        name: [p[k][j] for p in planes]
        for k, n in ((0, 1), (1, 2))
        for j, name in enumerate((f"strike{n}", f"dip{n}", f"rake{n}"))
    }

    def ints(values: Iterable[int | None]) -> Any:
        return pd.array(list(values), dtype="Int64")

    data: dict[str, Any] = {
        INDEX_NAME: pd.DatetimeIndex(times, dtype=TIME_DTYPE),
        "source": SOURCE,
        "event_id": pd.array(ids, dtype=STRING_DTYPE),
        "method": method,
        **col,
        "quality": [_opt(b.pct_acceptable) for b in use],
        "misfit": [_opt(b.amp_mean_error_factor) for b in use],
        "n_polarities": ints(b.n_polarities for b in use),
        "n_amplitudes": ints(b.n_amplitudes for b in use),
        "stress_drop_mpa": [_opt(b.stress_drop_mpa) for b in use],
        "corner_freq_hz": [_opt(b.corner_freq_hz) for b in use],
        "sil_faulting_class": [b.faulting_class for b in use],
    }
    if extras:
        data.update(
            {
                "block": pd.array(["sp3"] * len(use), dtype=STRING_DTYPE),
                "faulting_class": pd.array(
                    [b.faulting_class for b in use], dtype=STRING_DTYPE
                ),
                "significant": pd.array(
                    [pd.NA if b.significant is None else b.significant for b in use],
                    dtype="boolean",
                ),
                "seismic_moment_nm": [_opt(b.seismic_moment_nm) for b in use],
                "relative_size": [_opt(b.relative_size) for b in use],
                "p_axis_azimuth": [_opt(b.p_axis_azimuth) for b in use],
                "p_axis_dip": [_opt(b.p_axis_dip) for b in use],
                "t_axis_azimuth": [_opt(b.t_axis_azimuth) for b in use],
                "t_axis_dip": [_opt(b.t_axis_dip) for b in use],
                "block_plane_mismatch_deg": [0.0] * len(use),
            }
        )
    return _finish_frame(data, len(use), extras, plane_tol_deg)


def _finish_frame(
    data: dict[str, Any], n: int, extras: bool, plane_tol_deg: float
) -> pd.DataFrame:
    index = data.pop(INDEX_NAME)
    cols = {k: (v if not np.isscalar(v) else [v] * n) for k, v in data.items()}
    df = pd.DataFrame(cols, index=index)
    df.index.name = INDEX_NAME
    df = coerce_mechanisms(df, source=SOURCE)
    df = with_planes_inconsistent(df, plane_tol_deg=plane_tol_deg)
    keep = list(output_columns()) + (list(EXTRA_COLUMNS) if extras else [])
    return df[keep]


# --------------------------------------------------------------------------
# A day
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DayMechanismsReport:
    """What :func:`read_sil_day_mechanisms` did with one day directory."""

    day_dir: str
    #: ``events.fps`` present (else the day is read from ``events.lib`` + sp3).
    has_fps: bool
    fps_rows: int = 0
    #: fps rows without a solution (``FILE``): no mechanism row.
    no_solution: int = 0
    #: Rows written, per ``method``.
    rows_by_method: Mapping[str, int] = dataclasses.field(default_factory=dict)
    #: fps ``source_file`` values seen on rows with a solution.
    block_kinds: Mapping[str, int] = dataclasses.field(default_factory=dict)
    #: Blocks read and attached.
    blocks_read: int = 0
    #: Ids whose named block file does not exist (row kept, no block figures).
    blocks_missing: tuple[str, ...] = ()
    #: Ids whose block did not reproduce the fps plane a within the tolerance
    #: (row kept from fps, block figures not attached).
    block_mismatch: tuple[str, ...] = ()
    #: Worst block/fps plane-a mismatch among the attached blocks, degrees.
    max_block_mismatch_deg: float = 0.0
    #: Blocks that gave no row on an fps day: those of no-solution rows, and
    #: the other kind next to the one fps used. (Blocks of ids not in the
    #: catalogue are not looked for: the day directory is not listed.)
    unused_blocks: tuple[str, ...] = ()
    #: Blocks with unparsed lines or field warnings.
    block_issues: tuple[Sp3Report, ...] = ()
    #: On a day without fps: reviewed ids without an sp3 block.
    no_block: int = 0

    @property
    def rows(self) -> int:
        return sum(self.rows_by_method.values())

    def summary(self) -> str:
        methods = ", ".join(f"{k}={v}" for k, v in sorted(self.rows_by_method.items()))
        return (
            f"{self.day_dir}: rows={self.rows} ({methods or 'none'}) "
            f"fps={self.fps_rows if self.has_fps else 'absent'} "
            f"no_solution={self.no_solution} blocks_read={self.blocks_read} "
            f"missing={len(self.blocks_missing)} mismatch={len(self.block_mismatch)} "
            f"max_mismatch={self.max_block_mismatch_deg:.2f}° "
            f"unused={len(self.unused_blocks)} block_issues={len(self.block_issues)} "
            f"no_block={self.no_block}"
        )


class DayMechanisms(NamedTuple):
    """:func:`read_sil_day_mechanisms`'s result."""

    frame: pd.DataFrame
    report: DayMechanismsReport


def _read_block(path: Path, issues: list[Sp3Report]) -> Sp3Block | None:
    try:
        parsed = read_sp3(path)
    except FileNotFoundError:
        return None
    if not parsed.report.ok:
        issues.append(parsed.report)
    return parsed.block


def read_sil_day_mechanisms(
    day_dir: str | os.PathLike[str],
    *,
    extras: bool = False,
    plane_tol_deg: float = DEFAULT_PLANE_TOL_DEG,
) -> DayMechanisms:
    """The ``mechanisms`` frame of one SIL day directory.

    With ``events.fps``: one ``sil_fps`` (or ``other`` for ``ap3``) row per
    fps row with a solution, its block ``<id>.<source_file>`` read and
    attached (``quality``, ``misfit``) when it exists and reproduces the fps
    plane a within ``plane_tol_deg`` (:func:`same_plane_deg`). Without
    ``events.fps`` (1990 to late 1991): one ``sil_sp3`` row per
    ``events.lib`` id with an ``.sp3`` block. A day with neither file gives
    an empty frame. ``event_id`` is the SIL file id, as in the events frame
    (T3). Never writes. Rows whose planes are mutually inconsistent are kept
    and marked (:data:`PLANES_INCONSISTENT`); the frame passes
    :func:`validate_marked_mechanisms` (strict).
    """
    day = Path(day_dir)
    fps_path = day / "events.fps"
    issues: list[Sp3Report] = []
    if not fps_path.exists():
        lib_path = day / "events.lib"
        if not lib_path.exists():
            frame = mechanisms_from_blocks(
                _no_events(), {}, extras=extras, plane_tol_deg=plane_tol_deg
            )
            return DayMechanisms(frame, DayMechanismsReport(str(day), has_fps=False))
        lib = read_events_lib(lib_path).frame
        found: dict[str, Sp3Block] = {}
        for i in lib["id"].dropna():
            b = _read_block(block_path(day, str(i), "sp3"), issues)
            if b is not None:
                found[str(i)] = b
        frame = mechanisms_from_blocks(
            lib, found, extras=extras, plane_tol_deg=plane_tol_deg
        )
        report = DayMechanismsReport(
            str(day),
            has_fps=False,
            rows_by_method={"sil_sp3": len(frame)} if len(frame) else {},
            blocks_read=len(found),
            block_issues=tuple(issues),
            no_block=int(lib["id"].notna().sum()) - len(found),
        )
        return DayMechanisms(frame, report)

    fps = read_events_fps(fps_path).frame
    sol = fps.loc[fps["has_solution"].to_numpy(dtype=bool)]
    attached: dict[str, Sp3Block] = {}
    missing: list[str] = []
    mismatch: list[str] = []
    unused: list[str] = []
    worst = 0.0
    for _, row in sol.iterrows():
        eid, kind = str(row["id"]), str(row["source_file"])
        b = _read_block(block_path(day, eid, kind), issues)
        for other in BLOCK_KINDS:
            if other != kind and block_path(day, eid, other).exists():
                unused.append(f"{eid}.{other}")
        if b is None:
            missing.append(eid)
            continue
        if not b.has_planes:
            mismatch.append(eid)
            continue
        (cs, cd, cr), _ = b.planes_aki_richards()
        mis = float(
            same_plane_deg(
                cs, cd, cr, *normalize_plane(*row[["strike_a", "dip_a", "rake_a"]])
            )
        )
        if not mis <= plane_tol_deg:
            mismatch.append(eid)
            continue
        worst = max(worst, mis)
        attached[eid] = b
    for eid in fps.loc[~fps["has_solution"].to_numpy(dtype=bool), "id"].dropna():
        for kind in BLOCK_KINDS:
            if block_path(day, str(eid), kind).exists():
                unused.append(f"{eid}.{kind}")
    frame = mechanisms_from_fps(
        fps, blocks=attached, extras=extras, plane_tol_deg=plane_tol_deg
    )
    by_method = {str(k): int(v) for k, v in frame["method"].value_counts().items() if v}
    kinds = {str(k): int(v) for k, v in sol["source_file"].value_counts().items()}
    report = DayMechanismsReport(
        str(day),
        has_fps=True,
        fps_rows=len(fps),
        no_solution=len(fps) - len(sol),
        rows_by_method=by_method,
        block_kinds=kinds,
        blocks_read=len(attached),
        blocks_missing=tuple(missing),
        block_mismatch=tuple(mismatch),
        max_block_mismatch_deg=worst,
        unused_blocks=tuple(unused),
        block_issues=tuple(issues),
    )
    return DayMechanisms(frame, report)


def _no_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.Series([], dtype=TIME_DTYPE),
            "id": pd.Series([], dtype=STRING_DTYPE),
        }
    )


__all__ = [
    "BLOCK_KINDS",
    "PLANES_INCONSISTENT",
    "DEFAULT_FAULTING_RULE",
    "FAULTING_RULES",
    "EXTRA_COLUMNS",
    "FAULTING_CLASSES",
    "FPS_BLOCK_METHOD",
    "SOURCE",
    "PBTAxes",
    "DayMechanisms",
    "DayMechanismsReport",
    "ParsedSp3",
    "Sp3Block",
    "Sp3Report",
    "block_path",
    "faulting_class",
    "faulting_classes",
    "mechanisms_from_blocks",
    "mechanisms_from_fps",
    "normalize_plane",
    "output_columns",
    "planes_inconsistent",
    "pt_axes",
    "read_sil_day_mechanisms",
    "read_sp3",
    "same_plane_deg",
    "validate_marked_mechanisms",
    "with_planes_inconsistent",
    "sp3_to_aki_richards",
]
