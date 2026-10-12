"""GLOBK a priori coordinate (``.apr``) and rename (``eq_file``) files, numpy only.

What a GLOBK run would use as a site's a priori position at an epoch:

1. :func:`read_renames` — the ``rename OLD NEW start end`` lines of one or
   more eq files, in file order. ``OLD`` is a full 8-char name
   (``VILL_GPS``, IMO style; renames chain) or a bare 4-char code
   (``0194``, IGb14 style: any name of that code). A ``NEW`` ending in
   ``XCL`` excludes the site for the window.
2. :func:`site_name` — start from ``CODE_GPS`` (the h-file name) and apply
   every rename whose ``OLD`` matches the current name and whose window
   contains the epoch, in file order.
3. :func:`read_apr` — ``NAME X Y Z Vx Vy Vz EPOCH …`` rows (later files and
   later rows override earlier ones, like successive ``apr_file`` commands)
   plus ``EXTENDED NAME EXP|LOG yyyy mm dd hh mm tau_days dN dE dU``
   post-seismic terms (amplitudes N/E/U [m], rotated to X/Y/Z at the
   a priori position, as ``nonsec_util.f`` does).
4. :meth:`AprSite.at` — ``X + V·Δt + Σ terms`` with ``Δt`` in Julian years of
   365.25 days from the a priori epoch (a decimal year, converted to a date
   within its own calendar year).

:func:`apriori_xyz` strings these together for a list of codes.
"""

from __future__ import annotations

import dataclasses
import datetime
import gzip
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from geofunc import local

__all__ = [
    "AprSite",
    "NonSecular",
    "Rename",
    "apriori_xyz",
    "decimal_year_to_datetime",
    "read_apr",
    "read_renames",
    "site_name",
]

FloatArray = npt.NDArray[np.float64]

_DAY = 86400.0
_JULIAN_YEAR = 365.25


def decimal_year_to_datetime(y: float) -> datetime.datetime:
    """``2005.0`` → 2005-01-01; the fraction is of that calendar year's length."""
    year = int(np.floor(y))
    start = datetime.datetime(year, 1, 1)
    length = (datetime.datetime(year + 1, 1, 1) - start).total_seconds()
    return start + datetime.timedelta(seconds=(y - year) * length)


def _stamp(tok: Sequence[str]) -> datetime.datetime:
    """``yyyy mm dd hh mm`` (hour 24 allowed) → datetime."""
    y, mo, d, h, mi = (int(float(t)) for t in tok[:5])
    return datetime.datetime(y, mo, d) + datetime.timedelta(hours=h, minutes=mi)


@dataclasses.dataclass(frozen=True)
class Rename:
    old: str  # 8-char name, or a 4-char code matching any name of the code
    new: str
    start: datetime.datetime
    end: datetime.datetime

    def applies(self, name: str, when: datetime.datetime) -> bool:
        hit = name == self.old if len(self.old) > 4 else name[:4] == self.old
        return hit and self.start <= when < self.end


@dataclasses.dataclass(frozen=True)
class NonSecular:
    kind: str  # "EXP" | "LOG"
    start: datetime.datetime
    tau_days: float
    neu: FloatArray  # amplitudes N, E, U [m]


@dataclasses.dataclass(frozen=True)
class AprSite:
    name: str
    xyz: FloatArray
    vel: FloatArray  # m/yr
    epoch: datetime.datetime
    terms: tuple[NonSecular, ...] = ()

    def at(self, when: datetime.datetime) -> FloatArray:
        """A priori X/Y/Z [m] at ``when`` (velocity + post-seismic terms)."""
        dt = (when - self.epoch).total_seconds() / _DAY / _JULIAN_YEAR
        out: FloatArray = self.xyz + self.vel * dt
        if self.terms:
            dneu = np.zeros(3)
            for t in self.terms:
                days = (when - t.start).total_seconds() / _DAY
                if days < 0:
                    continue
                if t.kind == "EXP":
                    dneu += (1.0 - np.exp(-days / t.tau_days)) * t.neu
                else:
                    dneu += np.log1p(days / t.tau_days) * t.neu
            g = local.ecef_to_geodetic(self.xyz)
            out = out + local.local_to_ecef(dneu, g.lat, g.lon).d
        return out


def _lines(path: str | Path) -> list[str]:
    p = Path(path).expanduser()
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", errors="replace") as f:
        return f.read().splitlines()


def read_renames(paths: str | Path | Iterable[str | Path]) -> list[Rename]:
    """``rename`` lines of one or more eq files, in order (comments skipped)."""
    files = [paths] if isinstance(paths, (str, Path)) else list(paths)
    out = []
    for p in files:
        for line in _lines(p):
            tok = line.split()
            if (
                len(tok) >= 13
                and tok[0].lower() == "rename"
                and not line.startswith("#")
            ):
                try:
                    out.append(
                        Rename(
                            tok[1].upper(),
                            tok[2].upper(),
                            _stamp(tok[3:8]),
                            _stamp(tok[8:13]),
                        )
                    )
                except ValueError:
                    continue
    return out


def site_name(code: str, when: datetime.datetime, renames: Sequence[Rename]) -> str:
    """The GLOBK site name of ``code`` at ``when`` (``CODE_GPS`` renamed in order)."""
    name = f"{code.upper()[:4]}_GPS"
    for r in renames:
        if r.applies(name, when):
            name = r.new
    return name


def read_apr(paths: str | Path | Iterable[str | Path]) -> dict[str, AprSite]:
    """``{name: AprSite}`` from GLOBK apr files (later entries override)."""
    files = [paths] if isinstance(paths, (str, Path)) else list(paths)
    rows: dict[str, AprSite] = {}
    # (kind, start) keyed: a repeated term replaces the earlier one (decode_nonsut)
    terms: dict[str, dict[tuple[str, datetime.datetime], NonSecular]] = {}
    for p in files:
        for line in _lines(p):
            if not line.startswith(" ") or not line.strip():
                continue
            tok = line.split()
            if tok[0].upper() == "EXTENDED":
                if len(tok) >= 12 and tok[2].upper() in ("EXP", "LOG"):
                    term = NonSecular(
                        tok[2].upper(),
                        _stamp(tok[3:8]),
                        float(tok[8]),
                        np.array([float(v) for v in tok[9:12]]),
                    )
                    terms.setdefault(tok[1].upper(), {})[term.kind, term.start] = term
                continue
            if len(tok) < 8:
                continue
            try:
                vals = [float(v) for v in tok[1:8]]
            except ValueError:
                continue
            name = tok[0].upper()
            rows[name] = AprSite(
                name,
                np.array(vals[:3]),
                np.array(vals[3:6]),
                decimal_year_to_datetime(vals[6]),
            )
    return {
        n: dataclasses.replace(s, terms=tuple(terms.get(n, {}).values()))
        for n, s in rows.items()
    }


def apriori_xyz(
    codes: Iterable[str],
    when: datetime.datetime,
    apr: dict[str, AprSite],
    renames: Sequence[Rename],
) -> tuple[dict[str, FloatArray], dict[str, str]]:
    """A priori X/Y/Z of each code at ``when``, and why the others have none.

    Returns ``(positions, skipped)``: ``skipped[code]`` is ``"excluded"``
    (renamed to ``…XCL``) or ``"no apr <NAME>"``.
    """
    pos: dict[str, FloatArray] = {}
    skipped: dict[str, str] = {}
    for code in codes:
        name = site_name(code, when, renames)
        if name.endswith("XCL"):
            skipped[code] = "excluded"
        elif name not in apr:
            skipped[code] = f"no apr {name}"
        else:
            pos[code] = apr[name].at(when)
    return pos, skipped
