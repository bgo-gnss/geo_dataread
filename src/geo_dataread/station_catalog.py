"""Per-station catalog lookups (plate, a-priori coordinate) and the plate velocity.

The config-driven half of the old ``geofunc.geofunc.plateVelo`` — which plate
a station is on (``platefile``) and where it is (``coordFile``) — lives here,
in Tier 1, while the maths is ``geofunc.plates`` (Tier 0, numpy only).

:func:`plate_velocity_neu` returns **N, E, U**. ``plateVelo`` returned E, N, U
under an N, E, U docstring and every caller swapped the first two columns
back; the values are the same to ~1e-13 m/yr.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt

from geofunc import plates

__all__ = [
    "plate_table",
    "plate_velocity_neu",
    "station_plate",
    "station_xyz",
]

FloatArray = npt.NDArray[np.float64]


def _postprocess_file(key: str) -> Path:
    from gps_parser import ConfigParser

    return Path(ConfigParser().getPostProcessConfig(key)).expanduser()


def plate_table(path: str | Path | None = None) -> dict[str, str]:
    """``{station: plate code}`` from the ``platefile`` (``STA PLATE`` per line)."""
    p = _postprocess_file("platefile") if path is None else Path(path)
    out: dict[str, str] = {}
    with open(p) as f:
        for line in f:
            tok = line.split()
            if len(tok) >= 2:
                out[tok[0]] = tok[1]
    return out


def station_plate(sta: str, path: str | Path | None = None) -> str | None:
    """The plate code of ``sta`` (None when it is not in the platefile)."""
    return plate_table(path).get(sta)


def station_xyz(sta: str, path: str | Path | None = None) -> FloatArray | None:
    """A-priori ECEF ``(3,)`` of ``sta`` from ``coordFile`` (``X Y Z SITE``).

    A site listed twice resolves to its LAST row, as ``plateVelo`` did
    (``station_coord.xyz`` lists HESA twice).
    """
    p = _postprocess_file("coordFile") if path is None else Path(path)
    found: FloatArray | None = None
    with open(p) as f:
        for line in f:
            tok = line.split()
            if len(tok) >= 4 and tok[3] == sta:
                found = np.array([float(v) for v in tok[:3]])
    return found


def plate_velocity_neu(
    sta: str,
    plate: str | None = None,
    reference: str = "ITRF2008",
    *,
    coord_file: str | Path | None = None,
    plate_file: str | Path | None = None,
) -> FloatArray:
    """Secular plate velocity ``(3,)`` N, E, U [m/yr] at ``sta``'s a-priori position.

    ``plate`` overrides the platefile; ``reference`` picks the plate-motion
    model (``geofunc.plates``: ITRF2008/2014/2020). A station without a
    coordinate gets zeros and a warning, like ``plateVelo``.
    """
    x = station_xyz(sta, coord_file)
    if x is None:
        warnings.warn(f"Cannot determine plate velocity at station {sta}", stacklevel=2)
        return np.zeros(3)
    code = plate if plate else station_plate(sta, plate_file)
    if code is None:
        raise KeyError(f"station {sta} has no plate in the platefile")
    v: FloatArray = plates.plate_velocity_local(code, x, frame=reference).d
    return v
