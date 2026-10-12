"""glorg ``.org`` output: final positions and the stabilization set-up, numpy only.

The daily archive keeps glorg's print file next to each loose SINEX
(``…_SOL.org.gz`` beside ``…_SOL.SNX.gz``). Two things are read from it:

- :func:`read_org_positions` — the final, frame-stabilized X/Y/Z of every
  site with its adjustment and sigma (the ``NNN. SITE_xPS X coordinate (m)``
  rows), the oracle :mod:`geo_dataread.stabilize` reproduces;
- :func:`read_org_setup` — the glorg command file echo: the reference
  (``use_site``) codes and the ``pos_org`` / ``stab_ite`` / ``cnd_hgtv`` /
  ``stab_min`` settings, as a :class:`~geo_dataread.stabilize.StabilizeConfig`.

glorg names a site ``CODE_GPS``/``CODE_nPS``/``CODE_APS`` — the suffix is a
GLOBK rename tag, NOT the SINEX solution number (SINEX ``soln 1`` prints as
``_GPS`` or ``_1PS``), so :func:`match_sinex_rows` pairs org sites with
SINEX rows by 4-char code and the nearest a priori position.
"""

from __future__ import annotations

import dataclasses
import gzip
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from geo_dataread.sinex import SinexSolution
from geo_dataread.stabilize import StabilizeConfig

__all__ = [
    "OrgPosition",
    "OrgSetup",
    "match_sinex_rows",
    "read_org_positions",
    "read_org_setup",
]

FloatArray = npt.NDArray[np.float64]

_ROW = re.compile(
    r"^\s*\d+\.\s+(\S{8})\s+([XYZ]) coordinate\s+\(m\)\s+(\S+)\s+(\S+)\s+(\S+)"
)
_ORG_TYPES = ("xrot", "yrot", "zrot", "xtran", "ytran", "ztran", "scale")
# commands with no effect on position stabilization of a daily SINEX: the
# a priori files are already folded into the SINEX apriori column, the rest
# concern velocities
_BENIGN = {"apr_file", "vel_org", "cnd_hgtv", "stab_min"}
_MODELLED = {"use_site", "pos_org", "stab_ite", "cnd_hgtv", "stab_min"}


class OrgSetup(NamedTuple):
    """The glorg stabilization set-up of an org file."""

    reference: list[str]  # use_site codes, in order
    config: StabilizeConfig
    unsupported: tuple[
        str, ...
    ]  # commands stabilize() does not model (equate, force, …)


@dataclasses.dataclass(frozen=True)
class OrgPosition:
    """One site's final glorg position [m]: value, adjustment to a priori, sigma."""

    name: str
    xyz: FloatArray
    adjust: FloatArray
    sigma: FloatArray

    @property
    def apriori(self) -> FloatArray:
        out: FloatArray = self.xyz - self.adjust
        return out


def _lines(path: str | Path) -> list[str]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", errors="replace") as f:
        return f.read().splitlines()


def read_org_positions(path: str | Path) -> dict[str, OrgPosition]:
    """``{glorg site name: OrgPosition}`` from the parameter-estimate rows."""
    rows: dict[str, dict[str, tuple[float, float, float]]] = {}
    for line in _lines(path):
        m = _ROW.match(line)
        if m:
            name, comp, val, adj, sig = m.groups()
            rows.setdefault(name, {})[comp] = (float(val), float(adj), float(sig))
    out = {}
    for name, r in rows.items():
        if len(r) == 3:
            v = np.array([[*r[c]] for c in "XYZ"])
            out[name] = OrgPosition(name, v[:, 0], v[:, 1], v[:, 2])
    return out


def read_org_setup(path: str | Path) -> OrgSetup:
    """Reference codes and stabilization settings from the GLORG command echo.

    Commands :func:`~geo_dataread.stabilize.stabilize` does not model
    (``equate``, ``force``, ``constrain``, ``first_eqf``, …) are returned in
    ``unsupported`` — a non-empty list means the reproduction may not match
    that glorg run.
    """
    lines = _lines(path)
    try:
        start = next(
            i for i, ln in enumerate(lines) if "Summary of GLORG command file" in ln
        )
    except StopIteration:
        raise ValueError(f"{path}: no GLORG command file summary") from None
    codes: list[str] = []
    cfg: dict[str, object] = {}
    unsupported: list[str] = []
    for line in lines[start + 2 :]:
        if line.startswith("----"):
            break
        tok = line.split("!")[0].split()
        if not tok or tok[0].startswith("#"):
            continue
        key, args = tok[0].lower(), tok[1:]
        if key not in _MODELLED | _BENIGN:
            unsupported.append(key)
        if key == "use_site":
            for a in args:
                if a.lower() == "clear":
                    codes = []
                elif a.startswith("-"):
                    codes = [c for c in codes if c != a[1:5].upper()]
                elif a.lower() != "all":
                    codes.append(a[:4].upper())
        elif key == "pos_org":
            cfg["params"] = tuple(a.lower() for a in args if a.lower() in _ORG_TYPES)
        elif key == "stab_ite" and args:
            cfg["iterations"] = int(args[0])
            if len(args) > 1:
                cfg["rel"] = float(args[1])
            if len(args) > 2:
                cfg["nsig"] = float(args[2])
        elif key == "cnd_hgtv" and args:
            cfg["height_var"] = float(args[0])
            if len(args) > 2:
                cfg["use_ratio"] = float(args[2])
        elif key == "stab_min" and len(args) >= 3:
            cfg["min_rms"], cfg["min_dh"], cfg["min_dne"] = map(float, args[:3])
    return OrgSetup(
        list(dict.fromkeys(codes)),
        StabilizeConfig(**cfg),  # type: ignore[arg-type]
        tuple(dict.fromkeys(unsupported)),
    )


def match_sinex_rows(sol: SinexSolution, org: dict[str, OrgPosition]) -> dict[str, int]:
    """``{org name: SINEX STAX row}`` by code and nearest a priori (≤ 1 m)."""
    stax = np.flatnonzero(sol.ptype == "STAX")
    out: dict[str, int] = {}
    for name, p in org.items():
        rows = [i for i in stax if sol.code[i] == name[:4]]
        if not rows:
            continue
        gap = [np.abs(sol.apriori[i : i + 3] - p.apriori).max() for i in rows]
        k = int(np.argmin(gap))
        if gap[k] <= 1.0:
            out[name] = int(rows[k])
    return out
