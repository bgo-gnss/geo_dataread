"""RTK baseline series from the RTKLIB output on rtk.vedur.is.

Each baseline ``ROVER-BASE`` is a directory of daily files
``ROVER-BASE<YYYYMMDD>0000b.pos``: 1 Hz RTKLIB solutions of the rover
relative to the base, in UTC (18 s behind the GPS-time coordinate files
``ROVER<YYYYMMDD>0000b.pos`` beside them, which this module never reads).
Columns, whitespace separated, no header::

    date time  e n u [m]  Q ns  sde sdn sdu sden sdnu sdue [m]  age ratio

ENU order, both for the vector and the sigmas (RTKLIB's "enu" output; the
legacy ``rtk_gps.open_datafile`` labels the sigmas sdn, sde, …, which is
wrong for these files). The vector is rover − base, the same sense as a
GLOBK baseline A − B (checked on SEY1-SEYD against the daily positions:
E 351.157 / N −2582.12 / U 35.02 m vs 351.155 / −2582.119 / 35.13 m).

A day file runs from shortly after 00:00 of its day to a minute or so
past 00:00 of the next, so a UTC day needs its own file and the previous
one, and the overlap holds duplicate epochs (dropped, the later file's
kept). Per interval the median of the kept epochs is taken (``rtk_gps``'s
1-minute product); quality filtering as ``rtk_gps``: Q = 5 (single) out,
each sigma below ``max_sigma``.

The off-diagonal columns are RTKLIB's signed square roots of the
covariances (sden = sign(c_en)·√|c_en|, sdnu for n-u, sdue for u-e). Per
epoch they become correlations ρ = sign·sd²/(σᵢσⱼ), and the interval keeps
the median of each (beside the median σ): :attr:`RtkBaseline.cov` puts the
two back together.

Reading is from a read-only point of view: nothing is ever written under
the data directory. Medians of complete past days can be cached locally
(:func:`default_cache_dir`), keyed on the interval and the filter.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

#: the deployed mount of rtk.vedur.is:/home/gpsops/rtklib-run/data
DEFAULT_RTK_DIR = "/mnt_data/rtgpsdata"

#: a baseline directory: two 4-character codes (skips e.g. ELDV-SNAE_24h)
BASELINE_DIR = re.compile(r"^[A-Z0-9]{4}-[A-Z0-9]{4}$")

#: the file columns, in file order (ENU: see the module docstring)
FILE_COLUMNS = (
    "date", "time", "e", "n", "u", "q", "ns",
    "sde", "sdn", "sdu", "sden", "sdnu", "sdue", "age", "ratio",
)  # fmt: skip

#: N/E/U rows of the returned arrays, from the ENU file columns
NEU = ("n", "e", "u")
SIGMA_NEU = ("sdn", "sde", "sdu")

#: the correlations of the N/E/U pairs (NE, NU, EU) and the off-diagonal
#: column each comes from (sden = e-n, sdnu = n-u, sdue = u-e)
RHO_NEU = ("rne", "rnu", "reu")
RHO_FROM = {"rne": ("sden", "sdn", "sde"), "rnu": ("sdnu", "sdn", "sdu"),
            "reu": ("sdue", "sde", "sdu")}  # fmt: skip
#: (row, column) of each RHO_NEU entry in a N/E/U covariance
RHO_INDEX = ((0, 1), (0, 2), (1, 2))

#: bumped when the cached medians change shape (v2: + the correlations)
CACHE_VERSION = "v2"


def rtk_dir(path: str | os.PathLike[str] | None = None) -> Path:
    """The RTK data directory: ``path``, ``$GPS_RTK_DIR`` or the mount."""
    return Path(path or os.environ.get("GPS_RTK_DIR") or DEFAULT_RTK_DIR)


def default_cache_dir() -> Path:
    """``$GPS_RTK_CACHE``, else ``$XDG_CACHE_HOME/geo_dataread/rtk``."""
    env = os.environ.get("GPS_RTK_CACHE")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(xdg) / "geo_dataread" / "rtk"


def list_baselines(root: str | os.PathLike[str] | None = None) -> list[str]:
    """The baseline directories (``ROVER-BASE``) under the data directory."""
    base = rtk_dir(root)
    return sorted(
        p.name for p in base.iterdir() if p.is_dir() and BASELINE_DIR.match(p.name)
    )


def day_file(root: Path, baseline: str, day: dt.date) -> Path:
    return root / baseline / f"{baseline}{day:%Y%m%d}0000b.pos"


def read_file(path: str | os.PathLike[str]) -> pd.DataFrame:
    """One day file as a frame indexed by UTC epoch; empty if absent/empty."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame(columns=list(FILE_COLUMNS[2:]))
    df = pd.read_csv(
        p,
        header=None,
        sep=r"\s+",
        comment="%",
        names=list(FILE_COLUMNS),
        on_bad_lines="skip",
    )
    time = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce"
    )
    df = df.drop(columns=["date", "time"]).set_index(time)
    df.index.name = "time"
    return df[df.index.notna()]


@dataclasses.dataclass(frozen=True)
class RtkBaseline:
    """An RTK baseline series, N/E/U rows, metres.

    Attributes:
        name: ``ROVER-BASE``.
        time: Interval epochs, UTC, ``datetime64[ns]``, (N,).
        neu: Median rover − base vector per interval, (3, N) [m].
        sigma: Median formal sigma per interval, (3, N) [m].
        count: Epochs behind each median, (N,).
        interval: The median interval (pandas offset, e.g. ``"60s"``).
        rho: Median correlation per interval of the N/E/U pairs
            (:data:`RHO_NEU`: NE, NU, EU), (3, N); None = uncorrelated.
    """

    name: str
    time: npt.NDArray[np.datetime64]
    neu: FloatArray
    sigma: FloatArray
    count: npt.NDArray[np.int64]
    interval: str
    rho: FloatArray | None = None

    @property
    def cov(self) -> FloatArray:
        """N/E/U covariance per interval, (N, 3, 3) [m²], from σ and ρ."""
        rho = np.zeros_like(self.sigma) if self.rho is None else self.rho
        return covariance(self.sigma, rho)


def covariance(sigma: npt.ArrayLike, rho: npt.ArrayLike) -> FloatArray:
    """(N, 3, 3) covariance from σ (3, N) and the pair correlations ρ (3, N).

    The median correlations of an interval need not form a valid
    correlation matrix together: it is projected onto the nearest positive
    semi-definite one (eigenvalues clipped at 0, unit diagonal restored),
    so a rotation of it never gives a negative variance. A NaN ρ is taken
    as 0 (no correlation known); a NaN σ leaves its row and column NaN.
    """
    sd = np.asarray(sigma, dtype=float).T  # (N, 3)
    r = np.nan_to_num(np.asarray(rho, dtype=float).T)  # (N, 3)
    corr = np.broadcast_to(np.eye(3), (sd.shape[0], 3, 3)).copy()
    for k, (i, j) in enumerate(RHO_INDEX):
        corr[:, i, j] = corr[:, j, i] = r[:, k]
    w, v = np.linalg.eigh(corr)
    bad = (w < 0).any(axis=1)
    if bad.any():
        c = np.einsum("nij,nj,nkj->nik", v[bad], np.clip(w[bad], 0.0, None), v[bad])
        d = np.sqrt(np.clip(np.einsum("nii->ni", c), 1e-300, None))
        corr[bad] = c / d[:, :, None] / d[:, None, :]
    # a NaN σ makes its row and column NaN, nothing else
    return np.asarray(corr * sd[:, :, None] * sd[:, None, :], dtype=np.float64)


def _with_rho(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` plus the per-epoch correlations :data:`RHO_NEU`."""
    df = df.copy()
    for name, (off, a, b) in RHO_FROM.items():
        o = df[off].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.sign(o) * o * o / (df[a].astype(float) * df[b].astype(float))
        df[name] = rho.where(np.isfinite(rho)).clip(-1.0, 1.0)
    return df


def _medians(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    cols = list(NEU) + list(SIGMA_NEU) + list(RHO_NEU)
    if df.empty:
        return pd.DataFrame(
            columns=cols + ["count"], index=pd.DatetimeIndex([], name="time")
        )
    g = _with_rho(df)[cols].astype(float).resample(interval)
    out = g.median()
    out["count"] = g.size()
    return out[out["count"] > 0]


def _filtered(
    df: pd.DataFrame, drop_q: Sequence[int], max_sigma: float
) -> pd.DataFrame:
    if df.empty:
        return df
    keep = ~df["q"].isin(list(drop_q))
    if max_sigma > 0:
        for c in SIGMA_NEU:
            keep &= df[c] < max_sigma
    return df[keep]


def _day_raw(root: Path, baseline: str, day: dt.date) -> pd.DataFrame:
    """The raw epochs of UTC ``day``: its file plus the previous file's tail."""
    frames = [
        read_file(day_file(root, baseline, day - dt.timedelta(days=1))),
        read_file(day_file(root, baseline, day)),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=list(FILE_COLUMNS[2:]))
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    lo = pd.Timestamp(day)
    return df[(df.index >= lo) & (df.index < lo + pd.Timedelta(days=1))]


def _cache_key(interval: str, drop_q: Sequence[int], max_sigma: float) -> str:
    raw = f"{CACHE_VERSION}|{interval}|{sorted(drop_q)}|{max_sigma!r}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


#: medians of days still being written, kept for this process: one run of
#: several spans (``gpsplot rtk-run``) reads today's 1 Hz file once per
#: interval, not once per plot. Keyed on the two files' size and mtime, so
#: a file that grew is read again.
_OPEN_DAYS: dict[tuple[object, ...], pd.DataFrame] = {}
_OPEN_DAYS_MAX = 1024


def _stat(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_size, st.st_mtime_ns


def _day_medians(
    root: Path,
    baseline: str,
    day: dt.date,
    *,
    interval: str,
    drop_q: Sequence[int],
    max_sigma: float,
    cache: Path | None,
) -> pd.DataFrame:
    """Medians of one UTC day, from the local cache when the day is complete.

    A day is complete (cacheable) once the NEXT day's file exists: until
    then its own file may still be growing.
    """
    path = None
    complete = day_file(root, baseline, day + dt.timedelta(days=1)).is_file()
    if cache is not None and complete:
        key = _cache_key(interval, drop_q, max_sigma)
        path = cache / baseline / f"{day:%Y%m%d}-{key}.pkl"
        if path.is_file():
            return pd.read_pickle(path)
    open_key: tuple[object, ...] | None = None
    if not complete:
        open_key = (
            str(root), baseline, day, interval, tuple(sorted(drop_q)), max_sigma,
            _stat(day_file(root, baseline, day - dt.timedelta(days=1))),
            _stat(day_file(root, baseline, day)),
        )  # fmt: skip
        if open_key in _OPEN_DAYS:
            return _OPEN_DAYS[open_key].copy()
    out = _medians(
        _filtered(_day_raw(root, baseline, day), drop_q, max_sigma), interval
    )
    if open_key is not None:
        if len(_OPEN_DAYS) >= _OPEN_DAYS_MAX:
            _OPEN_DAYS.clear()
        _OPEN_DAYS[open_key] = out
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        out.to_pickle(tmp)
        tmp.replace(path)
    return out


def read_rtk_baseline(
    baseline: str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    root: str | os.PathLike[str] | None = None,
    interval: str = "60s",
    drop_q: Sequence[int] = (5,),
    max_sigma: float = 0.1,
    cache_dir: str | os.PathLike[str] | None = None,
    use_cache: bool = True,
) -> RtkBaseline:
    """Interval medians of an RTK baseline over ``[start, end]`` (UTC).

    Args:
        baseline: ``ROVER-BASE``, a directory under the data directory.
        start, end: The span, naive UTC datetimes.
        root: Data directory (default :func:`rtk_dir`).
        interval: Median interval, a pandas offset (``"60s"``, ``"10min"``).
        drop_q: RTKLIB quality flags to drop (5 = single).
        max_sigma: Each of sdn/sde/sdu must be below this [m]; 0 = no screen.
        cache_dir: Local cache of complete days' medians (default
            :func:`default_cache_dir`).
        use_cache: False reads every day from the files.

    Raises:
        FileNotFoundError: no such baseline directory.
    """
    base = rtk_dir(root)
    if not (base / baseline).is_dir():
        raise FileNotFoundError(f"no RTK baseline {baseline} under {base}")
    cache = (
        (Path(cache_dir) if cache_dir else default_cache_dir()) if use_cache else None
    )
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    parts = [
        _day_medians(
            base,
            baseline,
            d.date(),
            interval=interval,
            drop_q=drop_q,
            max_sigma=max_sigma,
            cache=cache,
        )
        for d in days
    ]
    parts = [p for p in parts if not p.empty]
    df = pd.concat(parts) if parts else _medians(pd.DataFrame(), interval)
    df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return RtkBaseline(
        name=baseline,
        time=df.index.to_numpy(dtype="datetime64[ns]"),
        neu=df[list(NEU)].to_numpy(dtype=np.float64).T.copy(),
        sigma=df[list(SIGMA_NEU)].to_numpy(dtype=np.float64).T.copy(),
        count=df["count"].to_numpy(dtype=np.int64),
        interval=interval,
        rho=df[list(RHO_NEU)].to_numpy(dtype=np.float64).T.copy(),
    )
