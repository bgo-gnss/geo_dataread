"""Absolute daily positions of a station, as the products see them.

:func:`load_positions` reads the station's ``.pos`` X/Y/Z with its 3×3
covariance and the reviewed junction offsets applied in ECEF (the
coordinate product's read, :mod:`geo_dataread.products.coords`), and marks
which epochs the view keeps — ``cleaned`` (outliers, icing, snow and
network days out), ``declared`` (known blunders out) or ``raw`` — after the
σ screen, matched on the epochs :func:`geo_dataread.products.loader.load`
keeps. Nothing is removed from the positions themselves (no plate, no
seasonal): they are coordinates. ``gpscoord series`` builds its tables
from them with :mod:`geofunc.columns`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from geo_dataread.products.loader import Source

FloatArray = npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class Positions:
    """One station's positions in ``[start, end)``.

    Attributes:
        station: the marker.
        frame: the geocentric frame (``itrf2014`` …).
        t: decimal years (GLOBK convention, the products' time).
        dates: the solution epochs.
        xyz: ECEF [m], (n, 3), junction offsets applied.
        cov: ECEF covariance [m²], (n, 3, 3).
        kept: which epochs the view keeps (after the σ screen).
    """

    station: str
    frame: str
    t: FloatArray
    dates: list[dt.datetime]
    xyz: FloatArray
    cov: FloatArray
    kept: npt.NDArray[np.bool_]


def available(t: FloatArray, epochs: npt.NDArray[np.datetime64]) -> str:
    """``data from A to B (N days)`` — what a station has, for a message."""
    if len(t) == 0:
        return "no data"
    days = epochs.astype("datetime64[D]")
    return f"data from {days[0]} to {days[-1]} ({len(t)} days)"


def read_positions(
    sta: str, src: Source
) -> tuple[FloatArray, npt.NDArray[np.datetime64], FloatArray, FloatArray]:
    """``(t, epochs, xyz (n,3), cov (n,3,3))`` of the station's whole
    ``.pos`` with the reviewed junction offsets, as every product sees
    the coordinates."""
    from geofunc import local

    from geo_dataread import pos

    series = pos.read_pos(pos.pos_path(sta, src.directory(), src.pos_id))
    rules = pos.product_junctions(src.pos_id)
    if rules is None:
        rules = pos._default_junctions(sta)
    xyz = series.xyz
    t = series.yearf
    if rules:
        xyz = local.apply_local_offsets(xyz, pos.junction_dlocal(t, rules))
    return (
        np.asarray(t, float),
        np.asarray(series.epochs),
        np.asarray(xyz, float),
        np.asarray(series.cov_xyz(), float),
    )


def load_positions(
    sta: str,
    start: float,
    end: float,
    src: Source,
    *,
    doc: Mapping[str, Any] | None = None,
) -> Positions:
    """The station's positions in ``[start, end)`` and the view's kept mask.

    Raises:
        FileNotFoundError: the station has no ``.pos`` of the source's product.
        ValueError: no epoch in the period.
    """
    from geo_dataread.products.coords import _match, _view_epochs

    t, epochs, xyz, cov = read_positions(sta, src)
    whole = (float(t[0]) - 1e-6, float(t[-1]) + 1e-6) if t.size else (start, end)
    sel = (t >= start) & (t < end)
    if not sel.any():
        raise ValueError(f"no solution in the period; {available(t, epochs)}")
    t, epochs, xyz, cov = t[sel], epochs[sel], xyz[sel], cov[sel]
    if doc is None:
        doc = src.params_doc()
    # the view over the WHOLE series, whatever the period: the cleaned view's
    # running detector sees its context, so a day is kept or dropped alike in
    # every request (±60 d of padding still flipped two dike days of SENG)
    try:
        t_view, _ = _view_epochs(sta, whole[0], whole[1], src, doc)
    except ValueError:  # e.g. the σ screen left nothing: no day is kept
        t_view = np.array([], dtype=float)
    kept = _match(t, t_view, 0.3 * src.session_years)
    dates = [dt.datetime.fromisoformat(str(e.astype("datetime64[s]"))) for e in epochs]
    return Positions(sta, src.frame_name, t, dates, xyz, cov, kept)
