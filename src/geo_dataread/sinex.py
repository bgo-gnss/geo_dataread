"""SINEX solutions: one day's coordinates with the FULL covariance, numpy only.

Layer 1 of the daily-coordinate store is a gzipped SINEX per network
solution, written on okada by ``glbtosnx`` from the loose (pre-glorg) GLOBK
combination (``/D/SNX/<yyyy>/IMO0OPSFIN_<yyyy><doy>0000_01D_01D_SOL.SNX.gz``).
This module reads one into a :class:`SinexSolution`: every estimated
parameter in SINEX order, its a priori value, the full covariance, and the
per-site metadata (receiver, antenna, eccentricity) that came from
station.info at processing time.

It deliberately imports nothing but numpy, like :mod:`geo_dataread.baseline`,
so the API and notebooks can use it; a test pins that.

Parameters are NOT assumed to be ``n_site × 3``: a GLOBK day also carries
the geocentre translation (``XGC``…) and EOP rows, and a station split by an
earthquake/antenna rename appears under its own SINEX solution number
(``Soln``). So everything is addressed by ``(type, code, pt, soln)``.

Pieces, each usable on its own:

- :func:`read_sinex` — path (``.gz`` or plain) → :class:`SinexSolution`.
- :meth:`SinexSolution.xyz_index` / :meth:`SinexSolution.cov_block` — the
  rows of one or more stations' X/Y/Z.
- :func:`neu_rotation` / :func:`xyz_to_llh` — the local frame.
- :func:`baseline_neu` — A − B in N/E/U with its 3×3 covariance
  (``Σ = R (C_AA + C_BB − C_AB − C_BA) Rᵀ``).
"""

from __future__ import annotations

import dataclasses
import datetime
import gzip
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "SinexError",
    "SinexSite",
    "SinexSolution",
    "baseline_neu",
    "neu_rotation",
    "read_sinex",
    "sinex_epoch",
    "xyz_to_llh",
]

FloatArray = npt.NDArray[np.float64]

# GRS80 (GLOBK uses a = 6378137, 1/f = 298.257222101)
_A_E = 6378137.0
_F = 1.0 / 298.257222101
_E2 = _F * (2.0 - _F)

_COORD_TYPES = ("STAX", "STAY", "STAZ")


class SinexError(ValueError):
    """A SINEX file this reader cannot interpret (or a missing station)."""


@dataclasses.dataclass(frozen=True)
class SinexSite:
    """Per-site metadata from the SITE/* blocks (empty strings when absent)."""

    code: str
    pt: str
    domes: str = ""
    description: str = ""
    receiver: str = ""
    antenna: str = ""
    # ARP − benchmark eccentricity (up, north, east) in metres
    ecc_une: tuple[float, float, float] | None = None


@dataclasses.dataclass(frozen=True)
class SinexSolution:
    """One SINEX solution. Arrays are in SINEX parameter order (index − 1)."""

    path: str
    agency: str
    start: datetime.datetime
    end: datetime.datetime
    ptype: npt.NDArray[np.str_]  # "STAX", "XGC", "XPO", …
    code: npt.NDArray[np.str_]  # 4-char site code, "----" for global params
    pt: npt.NDArray[np.str_]
    soln: npt.NDArray[np.str_]
    unit: npt.NDArray[np.str_]
    ref_epoch: npt.NDArray[np.str_]  # SINEX "YY:DOY:SSSSS", kept verbatim
    estimate: FloatArray
    sigma: FloatArray  # the ESTIMATE std-dev column
    apriori: FloatArray  # NaN where SOLUTION/APRIORI has no row
    cov: FloatArray  # full symmetric n×n, units of the parameters
    cov_apriori: FloatArray | None  # constraints, when MATRIX_APRIORI exists
    sites: dict[str, SinexSite]
    statistics: dict[str, float]

    @property
    def n_param(self) -> int:
        return int(self.estimate.size)

    @property
    def mean_epoch(self) -> datetime.datetime:
        return self.start + (self.end - self.start) / 2

    def station_codes(self) -> list[str]:
        """Codes with X/Y/Z rows, in first-appearance order."""
        mask = self.ptype == "STAX"
        return list(dict.fromkeys(self.code[mask].tolist()))

    def xyz_index(self, code: str, soln: str | None = None) -> npt.NDArray[np.intp]:
        """Row indices of ``code``'s STAX/STAY/STAZ.

        A code present under more than one solution number in one file
        needs ``soln``; asking without it is refused rather than guessed.
        """
        rows = []
        for t in _COORD_TYPES:
            hit = (self.ptype == t) & (self.code == code)
            if soln is not None:
                hit &= self.soln == soln
            idx = np.flatnonzero(hit)
            if idx.size == 0:
                raise SinexError(f"{code}: no {t} in {self.path}")
            if idx.size > 1:
                solns = sorted(set(self.soln[idx].tolist()))
                raise SinexError(
                    f"{code}: {t} under several solutions {solns}; pass soln="
                )
            rows.append(int(idx[0]))
        return np.asarray(rows, dtype=np.intp)

    def xyz(self, code: str, soln: str | None = None) -> FloatArray:
        """Estimated X/Y/Z (m) of one station."""
        return self.estimate[self.xyz_index(code, soln)]

    def cov_block(
        self, codes: Sequence[str], solns: Sequence[str | None] | None = None
    ) -> FloatArray:
        """The (3k × 3k) X/Y/Z covariance of ``codes`` in the given order."""
        solns = solns if solns is not None else [None] * len(codes)
        idx = np.concatenate(
            [self.xyz_index(c, s) for c, s in zip(codes, solns, strict=True)]
        )
        block: FloatArray = self.cov[np.ix_(idx, idx)]
        return block


# --------------------------------------------------------------------- epochs


def sinex_epoch(text: str) -> datetime.datetime:
    """``"YY:DOY:SSSSS"`` → UTC-naive datetime (YY < 50 → 20YY)."""
    yy, doy, sec = (int(p) for p in text.strip().split(":"))
    if yy == 0 and doy == 0 and sec == 0:
        raise SinexError("epoch 00:000:00000 (open) has no date")
    year = 2000 + yy if yy < 50 else 1900 + yy
    return datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1, seconds=sec)


# --------------------------------------------------------------------- reading


def _open_text(path: Path) -> list[str]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("ascii", errors="replace").splitlines()


def _blocks(lines: Iterable[str]) -> dict[str, list[str]]:
    """``{"SOLUTION/ESTIMATE": [data lines], ...}`` (comments dropped).

    The block name keeps any matrix qualifier (``"SOLUTION/MATRIX_ESTIMATE
    L COVA"``), normalised to single spaces.
    """
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        if line.startswith("+"):
            current = out.setdefault(" ".join(line[1:].split()), [])
        elif line.startswith("-"):
            current = None
        elif current is not None and line and not line.startswith("*"):
            current.append(line)
    return out


def _find(blocks: dict[str, list[str]], prefix: str) -> tuple[str, list[str]] | None:
    for name, lines in blocks.items():
        if name == prefix or name.startswith(prefix + " "):
            return name, lines
    return None


def _matrix(name: str, lines: list[str], n: int) -> FloatArray:
    """Fill a full symmetric matrix from a SINEX L/U COVA|CORR block."""
    parts = name.split()
    if len(parts) != 3:
        raise SinexError(f"{name}: expected '<block> L|U COVA|CORR'")
    tri, kind = parts[1], parts[2]
    if tri not in ("L", "U") or kind not in ("COVA", "CORR"):
        raise SinexError(f"{name}: only L/U COVA or CORR are supported")
    m = np.zeros((n, n))
    for line in lines:
        f = line.split()
        r, c0 = int(f[0]) - 1, int(f[1]) - 1
        for k, v in enumerate(f[2:]):
            m[r, c0 + k] = float(v)
    full = np.tril(m) + np.tril(m, -1).T if tri == "L" else np.triu(m) + np.triu(m, 1).T
    if kind == "CORR":  # diagonal holds σ, off-diagonal ρ
        s = np.diag(full).copy()
        full = full * np.outer(s, s)
        np.fill_diagonal(full, s**2)
    return full


def _sites(blocks: dict[str, list[str]]) -> dict[str, SinexSite]:
    info: dict[str, dict[str, object]] = {}

    def entry(line: str) -> dict[str, object]:
        code = line[1:5].strip()
        return info.setdefault(code, {"code": code, "pt": line[6:8].strip()})

    for line in blocks.get("SITE/ID", []):
        e = entry(line)
        e["domes"] = line[9:18].strip()
        e["description"] = line[21:43].strip()
    for line in blocks.get("SITE/RECEIVER", []):
        entry(line)["receiver"] = line[42:62].strip()
    for line in blocks.get("SITE/ANTENNA", []):
        entry(line)["antenna"] = line[42:62].strip()
    for line in blocks.get("SITE/ECCENTRICITY", []):
        f = line[42:].split()
        if len(f) >= 4 and f[0] == "UNE":
            entry(line)["ecc_une"] = (float(f[1]), float(f[2]), float(f[3]))
    return {code: SinexSite(**e) for code, e in info.items()}  # type: ignore[arg-type]


def read_sinex(path: str | Path) -> SinexSolution:
    """Read a SINEX solution (plain or gzip) into a :class:`SinexSolution`."""
    path = Path(path)
    lines = _open_text(path)
    if not lines or not lines[0].startswith("%=SNX"):
        raise SinexError(f"{path}: not a SINEX file (no %=SNX header)")
    header = lines[0].split()
    agency = header[2] if len(header) > 2 else ""
    start, end = sinex_epoch(header[5]), sinex_epoch(header[6])
    blocks = _blocks(lines)

    est = blocks.get("SOLUTION/ESTIMATE")
    if not est:
        raise SinexError(f"{path}: no SOLUTION/ESTIMATE block")
    n = len(est)
    cols: dict[str, list[str]] = {k: [] for k in ("t", "c", "p", "s", "e", "u")}
    estimate = np.empty(n)
    sigma = np.empty(n)
    for i, line in enumerate(est):
        f = line.split()
        if int(f[0]) != i + 1:
            raise SinexError(f"{path}: ESTIMATE index {f[0]} out of order")
        for k, v in zip("tcpseu", f[1:7], strict=True):
            cols[k].append(v)
        estimate[i], sigma[i] = float(f[8]), float(f[9])

    apriori = np.full(n, np.nan)
    for line in blocks.get("SOLUTION/APRIORI", []):
        f = line.split()
        apriori[int(f[0]) - 1] = float(f[8])

    found = _find(blocks, "SOLUTION/MATRIX_ESTIMATE")
    if found is None:
        raise SinexError(f"{path}: no SOLUTION/MATRIX_ESTIMATE block")
    cov = _matrix(*found, n)
    found_ap = _find(blocks, "SOLUTION/MATRIX_APRIORI")
    cov_ap = _matrix(*found_ap, n) if found_ap else None

    stats: dict[str, float] = {}
    for line in blocks.get("SOLUTION/STATISTICS", []):
        label, value = line[1:31].strip(), line[31:].strip()
        try:
            stats[label] = float(value)
        except ValueError:
            continue

    def arr(k: str) -> npt.NDArray[np.str_]:
        return np.asarray(cols[k], dtype=str)

    return SinexSolution(
        path=str(path),
        agency=agency,
        start=start,
        end=end,
        ptype=arr("t"),
        code=arr("c"),
        pt=arr("p"),
        soln=arr("s"),
        unit=arr("u"),
        ref_epoch=arr("e"),
        estimate=estimate,
        sigma=sigma,
        apriori=apriori,
        cov=cov,
        cov_apriori=cov_ap,
        sites=_sites(blocks),
        statistics=stats,
    )


# ------------------------------------------------------------ local frame


def xyz_to_llh(xyz: npt.ArrayLike) -> tuple[float, float, float]:
    """Geodetic (lat, lon) in radians and height (m) on GRS80."""
    x, y, z = (float(v) for v in np.asarray(xyz, dtype=float))
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _E2))
    h = 0.0
    for _ in range(10):
        n = _A_E / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - _E2 * n / (n + h)))
    return lat, lon, h


def neu_rotation(lat: float, lon: float) -> FloatArray:
    """R with rows N, E, U, so that ``neu = R @ dxyz`` (geodetic ENU)."""
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([[-sl * co, -sl * so, cl], [-so, co, 0.0], [cl * co, cl * so, sl]])


def baseline_neu(
    sol: SinexSolution,
    a: str,
    b: str,
    *,
    soln_a: str | None = None,
    soln_b: str | None = None,
) -> tuple[FloatArray, FloatArray]:
    """A − B in N/E/U (m) at A's position, with its 3×3 covariance (m²).

    The covariance is the full propagation ``J C Jᵀ`` with ``J = R [I −I]``,
    so the stations' cross-covariance (common-mode errors) is NOT counted
    twice as the quadrature sum would.
    """
    ia, ib = sol.xyz_index(a, soln_a), sol.xyz_index(b, soln_b)
    r = neu_rotation(*xyz_to_llh(sol.estimate[ia])[:2])
    j = np.hstack([r, -r])
    idx = np.concatenate([ia, ib])
    d = r @ (sol.estimate[ia] - sol.estimate[ib])
    return d, j @ sol.cov[np.ix_(idx, idx)] @ j.T
