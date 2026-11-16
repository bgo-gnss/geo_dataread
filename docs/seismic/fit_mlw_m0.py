"""Fit the SIL Mlw → log10(M0) relation from the archive (T5b, PLAN-seismicity R1).

Reads the reviewed SIL events of ``[--start, --end)`` year by year through
:func:`geo_dataread.seismic.cache.read_sil_catalog_cached` (``automatic=True``
because ``automatic=False`` bypasses the cache; the manual rows are then kept
here; ``write=False``, so the cache is never written), keeps rows with a
finite ``mag_mlw`` and a positive ``moment_nm`` (the measured ``events.fps``
moment), and prints:

* counts and magnitude resolution per era (0.01 from lib.mag/events.lib, or
  0.1 from events.fps);
* per-era intercept below the knee and the slope of the upper segment;
* per-bin medians of ``log10(M0) − (Mlw + 10)`` and how well each candidate
  functional form fits them (fits are on equal-weight 0.1-wide bin medians,
  so the ~10⁵ small events do not drown the tail);
* residuals of the adopted relation (``filters.moment_from_mlw``) by bin and
  by era, and the Hanks–Kanamori error census (events whose HK moment is off
  by ≥3× / ≥5× either way, and the moment-weighted totals);
* the reviewed Mlw events with **no** measured moment (the rows that
  ``with_derived_moment`` will fill), per era.

Usage (from the repo root, in the ``gpslibrary`` env)::

    PYTHONNOUSERSITE=1 PYTHONPATH=src python docs/seismic/fit_mlw_m0.py \\
        --base /mnt_data/sildata --frame /tmp/.../mlw_m0.parquet \\
        --png docs/seismic/mlw_m0_relation.png

``--frame`` caches the extracted columns as parquet (re-used if present).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from geo_dataread.seismic.cache import read_sil_catalog_cached
from geo_dataread.seismic.filters import (
    MLW_M0_FIT,
    moment_from_magnitude,
    moment_from_mlw,
)

ERAS: tuple[tuple[int, int], ...] = (
    (1990, 1995),
    (1995, 2000),
    (2000, 2005),
    (2005, 2010),
    (2010, 2015),
    (2015, 2020),
    (2020, 2024),
    (2024, 2027),
)
BIN = 0.1
MIN_BIN_N = 5
#: Events at Mlw >= 6 (the 1998/2000 mainshocks) do not follow the SIL
#: definition (their M0 is far below it); the fit stops here.
FIT_MAX_MLW = 6.0
FIT_END = "2026-11-14"
SIGMA_EDGES = (-3.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)


def load(base: str, start: int, end: int, frame: Path | None) -> pd.DataFrame:
    """Reviewed SIL events: ``time, mag_mlw, moment_nm, magnitude_type``."""
    if frame is not None and frame.exists():
        return pd.read_parquet(frame)
    parts = []
    for year in range(start, end):
        df, _ = read_sil_catalog_cached(
            f"{year}-01-01", f"{year + 1}-01-01", base=base, write=False
        )
        df = df[df["evaluation_mode"].astype(object) == "manual"]
        cols = df[["mag_mlw", "moment_nm", "magnitude", "magnitude_type"]].copy()
        cols["magnitude_type"] = cols["magnitude_type"].astype(object)
        cols = cols.reset_index()
        parts.append(cols)
        print(f"{year}: {len(cols)} reviewed", file=sys.stderr, flush=True)
    out = pd.concat(parts, ignore_index=True)
    if frame is not None:
        out.to_parquet(frame)
    return out


def era_of(t: pd.Series) -> pd.Series:
    years = t.dt.year
    labels = pd.Series("", index=t.index, dtype=object)
    for lo, hi in ERAS:
        labels[(years >= lo) & (years < hi)] = f"{lo}–{hi - 1}"
    return labels


def bin_medians(m: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Median of ``y`` in 0.1-wide Mlw bins centred on multiples of 0.1."""
    key = np.round(m / BIN).astype(int)
    g = pd.DataFrame({"k": key, "m": m, "y": y}).groupby("k")
    tab = pd.DataFrame(
        {
            "m": g["m"].median(),
            "y": g["y"].median(),
            "n": g["y"].size(),
            "p16": g["y"].quantile(0.16),
            "p84": g["y"].quantile(0.84),
        }
    )
    return tab[tab["n"] >= MIN_BIN_N]


# -- candidate forms: dev(m) = log10(M0) − (Mlw + 10) -----------------------


def f_knee_linear(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    c, k, s = p
    return np.asarray(c + np.where(m > k, (s - 1.0) * (m - k), 0.0))


def f_hyperbola(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    c, k, d, s = p
    x = m - k
    return np.asarray(c + 0.5 * (s - 1.0) * (x + np.sqrt(x * x + d * d)))


def f_knee_poly(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    c, k, a, b = p
    x = np.clip(m - k, 0.0, None)
    return np.asarray(c + a * x * x + b * x**3)


def f_piecewise2(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    c, k1, k2, s1, s2 = p
    out = c + (s1 - 1.0) * np.clip(m - k1, 0.0, None)
    return np.asarray(out + (s2 - s1) * np.clip(m - k2, 0.0, None))


def f_piecewise3(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    c, k1, k2, k3, s1, s2, s3 = p
    out = f_piecewise2(np.array([c, k1, k2, s1, s2]), m)
    return np.asarray(out + (s3 - s2) * np.clip(m - k3, 0.0, None))


FORMS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], list[float]]] = {
    "hard knee + line (c,k,s)": (f_knee_linear, [0.0, 2.0, 1.25]),
    "hyperbola (c,k,d,s)": (f_hyperbola, [0.0, 2.5, 0.5, 1.3]),
    "C1 knee + a·x²+b·x³ (c,k,a,b)": (f_knee_poly, [0.0, 2.0, 0.1, 0.0]),
    "piecewise linear, 2 hinges (c,k1,k2,s1,s2)": (
        f_piecewise2,
        [0.0, 2.0, 3.0, 1.1, 1.25],
    ),
    "piecewise linear, 3 hinges (c,k1,k2,k3,s1,s2,s3)": (
        f_piecewise3,
        [0.0, 2.0, 3.0, 4.6, 1.1, 1.25, 1.4],
    ),
}


def fit_forms(tab: pd.DataFrame) -> None:
    m = tab["m"].to_numpy()
    y = tab["y"].to_numpy()
    print(f"\n## candidate forms on {len(tab)} bin medians (equal weight)")
    for name, (f, p0) in FORMS.items():
        res = least_squares(lambda p, f=f: f(p, m) - y, p0)
        r = res.fun
        leak = r[(m >= 1.5) & (m < 2.0)]
        print(
            f"{name:34s} p={np.array2string(res.x, precision=4)} "
            f"rms={np.sqrt(np.mean(r * r)):.4f} max={np.abs(r).max():.4f} "
            f"max|r| 1.5–2.0={np.abs(leak).max() if len(leak) else np.nan:.4f}"
        )


def f_piecewise4(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``log10(M0)`` (not the deviation): intercept, 4 knots, 4 upper slopes."""
    c, k1, k2, k3, k4, s1, s2, s3, s4 = p
    out = c + m
    for k, a, b in zip((k1, k2, k3, k4), (1.0, s1, s2, s3), (s1, s2, s3, s4)):
        out = out + (b - a) * np.clip(m - k, 0.0, None)
    return np.asarray(out)


def fit_events_piecewise4(ok: pd.DataFrame) -> None:
    """Free 4-knot fit on the events themselves (robust soft-L1, Mlw 1.5–6)."""
    m = ok["mag_mlw"].to_numpy()
    lg = np.log10(ok["moment_nm"].to_numpy())
    sel = (m >= 1.5) & (m < FIT_MAX_MLW)
    p0 = [10.0, 2.0, 3.0, 4.5, 5.3, 1.1, 1.25, 1.4, 1.9]
    res = least_squares(
        lambda p: f_piecewise4(p, m[sel]) - lg[sel], p0, loss="soft_l1", f_scale=0.01
    )
    c, *rest = res.x
    knots, slopes = rest[:4], rest[4:]
    print(f"\n## free 4-knot piecewise fit on {int(sel.sum())} events, Mlw 1.5–6")
    print(f"intercept {c:.4f}; knots {np.round(knots, 4)}")
    print(f"slopes {np.round(slopes, 4)}; 1/slope {np.round(1 / np.array(slopes), 4)}")


def emit_constants(ok: pd.DataFrame) -> None:
    """The fit record for ``filters.MLW_M0_FIT`` (from the adopted relation)."""
    used = ok[ok["mag_mlw"] < FIT_MAX_MLW]
    r = used["r"].to_numpy()
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    rows = []
    for lo, hi in zip(SIGMA_EDGES, SIGMA_EDGES[1:]):
        s = used.loc[(used["mag_mlw"] >= lo) & (used["mag_mlw"] < hi), "r"]
        if len(s):
            sig = 1.4826 * float(np.median(np.abs(s - s.median())))
            rows.append((lo, hi, len(s), round(sig, 4)))
    print("\n## fit record for filters.MLW_M0_FIT")
    print(f"fit_start={used['time'].min().date()} fit_end(excl)={FIT_END}")
    print(
        f"n_events={len(used)} mlw_range=({used['mag_mlw'].min()}, "
        f"{used['mag_mlw'].max()}) sigma={mad:.4f}"
    )
    print(f"sigma_by_bin={tuple(rows)}")


def era_table(d: pd.DataFrame) -> None:
    print("\n## eras")
    print(
        "era        n_rev   n_mlw  n_M0   res0.01  int(<1.9)  "
        "slope(2.5–4.5)  n>=3  no-M0 Mlw"
    )
    for era, s in d.groupby("era", sort=True):
        ok = s["ok"]
        sm = s[ok]
        m = sm["mag_mlw"].to_numpy()
        lg = np.log10(sm["moment_nm"].to_numpy())
        hund = np.abs(m * 10 - np.round(m * 10)) > 1e-6
        low = m < 1.9
        hi = (m >= 2.5) & (m <= 4.5)
        slope = np.polyfit(m[hi], lg[hi], 1)[0] if hi.sum() >= 10 else np.nan
        nomo = s["mag_mlw"].notna() & ~(s["moment_nm"] > 0)
        print(
            f"{era:10s} {len(s):6d} {s['mag_mlw'].notna().sum():7d} {ok.sum():6d} "
            f"{hund.mean() if len(m) else np.nan:7.2f} "
            f"{np.median(lg[low] - m[low] - 10) if low.any() else np.nan:+9.4f}  "
            f"{slope:14.3f} {(m >= 3).sum():5d} {nomo.sum():9d}"
        )


def residual_tables(d: pd.DataFrame) -> pd.DataFrame:
    ok = d[d["ok"]].copy()
    ok["lg"] = np.log10(ok["moment_nm"])
    ok["r"] = ok["lg"] - np.log10(moment_from_mlw(ok["mag_mlw"].to_numpy()))
    ok["r_hk"] = ok["lg"] - np.log10(moment_from_magnitude(ok["mag_mlw"].to_numpy()))
    edges = [*SIGMA_EDGES, 7.0]
    ok["bin"] = pd.cut(ok["mag_mlw"], edges, right=False)
    print("\n## residual log10(M0_measured / M0_relation) by Mlw bin")
    print("bin           n        median    sigma(MAD)  p2.5     p97.5   HK median")
    for b, s in ok.groupby("bin", observed=True):
        r = s["r"]
        mad = 1.4826 * np.median(np.abs(r - r.median()))
        print(
            f"{b!s:12s} {len(s):7d} {r.median():+9.4f} {mad:9.4f} "
            f"{r.quantile(0.025):+8.4f} {r.quantile(0.975):+8.4f} "
            f"{s['r_hk'].median():+8.3f}"
        )
    print("\n## residual by era: median, robust σ (MAD), n off >= 2x; all and Mlw >= 2")
    lg2 = np.log10(2.0)
    for era, s in ok.groupby("era"):
        out = []
        for part in (s, s[s["mag_mlw"] >= 2]):
            r = part["r"]
            mad = 1.4826 * np.median(np.abs(r - r.median()))
            out.append(
                f"n={len(r):7d} med={r.median():+.4f} σ={mad:.4f} "
                f">=2x={int((r.abs() >= lg2).sum())}"
            )
        print(f"{era:10s} " + " | ".join(out))
    print("\n## Hanks–Kanamori on Mlw: factor errors vs measured M0")
    f = ok["r_hk"]  # log10(measured / HK)
    n = len(f)
    for k in (3, 5, 10):
        lk = np.log10(k)
        under = int((f >= lk).sum())
        over = int((f <= -lk).sum())
        print(
            f">= {k:2d}x: HK too small {under} ({under / n:.1%}), "
            f"HK too large {over} ({over / n:.1%}) of {n}"
        )
    rel = ok["r"]
    for kf in (1.5, 2.0, 3.0):
        lk = np.log10(kf)
        print(f"relation off by >= {kf}x either way: {int((rel.abs() >= lk).sum())}")
    tot = ok["moment_nm"].sum()
    hk = moment_from_magnitude(ok["mag_mlw"].to_numpy()).sum()
    rel_tot = moment_from_mlw(ok["mag_mlw"].to_numpy()).sum()
    print(
        f"total measured M0 {tot:.4e} N·m; HK {hk:.4e} ({hk / tot:.2f}×); "
        f"relation {rel_tot:.4e} ({rel_tot / tot:.3f}×)"
    )
    return ok


def plot(ok: pd.DataFrame, tab: pd.DataFrame, png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    mm = np.linspace(-1.5, 6.0, 300)
    a1.plot(ok["mag_mlw"], ok["lg"] - ok["mag_mlw"] - 10, ",", alpha=0.3, color="0.5")
    a1.plot(tab["m"], tab["y"], "o", ms=3, color="C0", label="bin medians")
    a1.plot(
        mm,
        np.log10(moment_from_mlw(mm)) - mm - 10,
        "-",
        color="C3",
        label="moment_from_mlw",
    )
    a1.plot(
        mm,
        np.log10(moment_from_magnitude(mm)) - mm - 10,
        "--",
        color="C2",
        label="Hanks–Kanamori",
    )
    a1.set_ylabel("log10(M0 / N·m) − (Mlw + 10)")
    a1.set_ylim(-0.6, 1.6)
    a1.legend()
    a1.set_title(f"SIL reviewed events, n = {len(ok)}")
    a2.plot(ok["mag_mlw"], ok["r"], ",", alpha=0.3, color="0.5")
    a2.axhline(0, color="C3")
    a2.set_ylim(-0.3, 0.3)
    a2.set_ylabel("residual log10(M0 / relation)")
    a2.set_xlabel("Mlw")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    print(f"\nwrote {png}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="/mnt_data/sildata")
    ap.add_argument("--start", type=int, default=1990)
    ap.add_argument("--end", type=int, default=2027, help="exclusive year")
    ap.add_argument("--frame", type=Path, default=None)
    ap.add_argument("--png", type=Path, default=None)
    a = ap.parse_args(argv)
    d = load(a.base, a.start, a.end, a.frame)
    d["time"] = pd.to_datetime(d["time"], utc=True)
    d = d[d["time"] < pd.Timestamp(FIT_END, tz="UTC")]
    d["era"] = era_of(d["time"])
    d["ok"] = d["mag_mlw"].notna() & (d["moment_nm"] > 0)
    print(f"reviewed events {len(d)}, with Mlw and M0 {int(d['ok'].sum())}")
    print(f"relation in use: {MLW_M0_FIT}")
    era_table(d)
    ok = d[d["ok"]]
    m = ok["mag_mlw"].to_numpy()
    dev = np.log10(ok["moment_nm"].to_numpy()) - m - 10.0
    tab = bin_medians(m, dev)
    print("\n## bin medians of log10(M0) − (Mlw + 10)")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
    fit_forms(tab)
    fit_events_piecewise4(ok)
    ok2 = residual_tables(d)
    emit_constants(ok2)
    if a.png is not None:
        plot(ok2, tab, a.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
