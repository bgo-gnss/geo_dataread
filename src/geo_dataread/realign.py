"""Re-align the loose daily SINEX archive to a reference frame → per-station ``.pos``.

Each archived day (``<archive>/<yyyy>/IMO0OPSFIN_<yyyy><doy>0000_01D_01D_SOL.SNX.gz``
+ its glorg ``.org.gz``) is stabilized with :func:`geo_dataread.stabilize.stabilize`
using the day's own glorg reference list and settings, and either

- the SINEX a priori (``--frame itrf2008``: glorg's own frame — reproduces
  the archived ``.pos``), or
- a GLOBK apr + eq pair (``--frame igb14``: the IGS realization of ITRF2014
  from the GAMIT tables, with renames and post-seismic terms resolved at the
  day's epoch by :mod:`geo_dataread.globk_apr`).

``--to itrf2020`` then moves every position (and covariance) with the EPSG
time-dependent Helmert ITRF2014→ITRF2020 at its epoch (geofunc.frames; this
is "IGb14 → ITRF2020 via EPSG Helmert", not IGS20). Output: one PBO ``.pos``
per station (:func:`geo_dataread.pos.write_pos`) and ``days.csv`` with one
line per day (reference sites used/dropped, post-fit RMS, parameters).

Writes only under ``--out``; the archive is read-only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from geo_dataread import globk_apr, org, pos, sinex, stabilize

__all__ = ["DayResult", "archive_days", "main", "realign_day"]

FloatArray = npt.NDArray[np.float64]

DEFAULT_ARCHIVE = Path("/mnt_data/okada/D/SNX")
GAMIT_TABLES = Path(os.environ.get("GG_TABLES", Path.home() / "gg" / "tables"))
FRAMES = {
    "itrf2008": None,
    "igb14": ("igb14_comb.apr", "igb14_comb.eq"),
}


@dataclasses.dataclass
class DayResult:
    epoch: datetime.datetime
    sites: dict[str, tuple[FloatArray, FloatArray]]  # code → (xyz, 3×3 cov)
    n_ref: int
    dropped: tuple[str, ...]
    post_rms: float
    theta: FloatArray
    error: str = ""


def archive_days(archive: Path, start: datetime.date, end: datetime.date) -> list[Path]:
    """Loose SINEX files of the archive between two dates (inclusive)."""
    out = []
    for year in range(start.year, end.year + 1):
        for p in sorted((archive / str(year)).glob("IMO0OPSFIN_*_01D_01D_SOL.SNX.gz")):
            day = datetime.datetime.strptime(p.name[11:18], "%Y%j").date()
            if start <= day <= end:
                out.append(p)
    return out


_APR_CACHE: dict[
    tuple[str, str], tuple[dict[str, globk_apr.AprSite], list[globk_apr.Rename]]
] = {}


def _apr(
    frame: str, tables: Path
) -> tuple[dict[str, globk_apr.AprSite], list[globk_apr.Rename]] | None:
    files = FRAMES[frame]
    if files is None:
        return None
    key = (frame, str(tables))
    if key not in _APR_CACHE:
        _APR_CACHE[key] = (
            globk_apr.read_apr(tables / files[0]),
            globk_apr.read_renames(tables / files[1]),
        )
    return _APR_CACHE[key]


def realign_day(
    snx: Path, frame: str = "itrf2008", tables: Path = GAMIT_TABLES
) -> DayResult:
    """Stabilize one archived day in ``frame`` (see the module docstring)."""
    sol = sinex.read_sinex(snx)
    setup = org.read_org_setup(Path(str(snx).replace(".SNX.gz", ".org.gz")))
    when = sol.mean_epoch
    present = set(sol.code.tolist())
    codes = [c for c in setup.reference if c in present]
    dropped: list[str] = []
    ext = _apr(frame, tables)
    apriori = None
    if ext is not None:
        apriori, skipped = globk_apr.apriori_xyz(codes, when, *ext)
        dropped = sorted(skipped)
    st = stabilize.stabilize(sol, codes, setup.config, apriori=apriori)
    last = st.iterations[-1]
    est, cov = st.solution.estimate, st.solution.cov
    sites: dict[str, tuple[FloatArray, FloatArray]] = {}
    for i in np.flatnonzero(sol.ptype == "STAX"):
        c = str(sol.code[i])
        blk = cov[i : i + 3, i : i + 3]
        # a code under two solutions in one day: keep the better-determined
        if c not in sites or np.trace(blk) < np.trace(sites[c][1]):
            sites[c] = (est[i : i + 3].copy(), blk.copy())
    return DayResult(
        when, sites, len(last.sites), tuple(dropped), last.post_rms, st.theta
    )


def _safe(snx: Path, frame: str, tables: Path) -> DayResult:
    try:
        return realign_day(snx, frame, tables)
    except Exception as e:  # noqa: BLE001 - one bad day must not stop a 30-year batch
        day = datetime.datetime.strptime(snx.name[11:18], "%Y%j")
        return DayResult(
            day, {}, 0, (), float("nan"), np.full(6, np.nan), f"{type(e).__name__}: {e}"
        )


def _to_itrf2020(
    series: dict[str, list[tuple[datetime.datetime, FloatArray, FloatArray]]],
) -> None:
    from geofunc import frames, local

    h = frames.helmert_params("itrf2014", "itrf2020")
    for rows in series.values():
        for k, (t, x, c) in enumerate(rows):
            yr = t.year + (t - datetime.datetime(t.year, 1, 1)).total_seconds() / (
                (
                    datetime.datetime(t.year + 1, 1, 1)
                    - datetime.datetime(t.year, 1, 1)
                ).total_seconds()
            )
            out = local.helmert_apply(x, h.at(yr), cov=c)
            assert out.cov is not None
            rows[k] = (t, out.d, out.cov)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gps-realign", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--start", required=True, type=datetime.date.fromisoformat)
    ap.add_argument("--end", required=True, type=datetime.date.fromisoformat)
    ap.add_argument("--frame", choices=sorted(FRAMES), default="igb14")
    ap.add_argument(
        "--to", choices=["itrf2020"], help="post-hoc EPSG Helmert from ITRF2014"
    )
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--tables", type=Path, default=GAMIT_TABLES)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prod-id", help="default imo.daily_<frame>")
    ap.add_argument("--stations", nargs="*", help="write only these codes")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    if a.to and a.frame != "igb14":
        ap.error("--to itrf2020 needs --frame igb14")

    days = archive_days(a.archive, a.start, a.end)
    if not days:
        print("no archived days in range", file=sys.stderr)
        return 1
    label = {"itrf2008": "ITRF2008", "igb14": "IGb14"}[a.frame]
    if a.to:
        label = "ITRF2020 (IGb14 + EPSG Helmert)"
    prod = a.prod_id or f"imo.daily_{a.to or a.frame}"
    keep = {s.upper() for s in a.stations} if a.stations else None

    series: dict[str, list[tuple[datetime.datetime, FloatArray, FloatArray]]] = {}
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "days.csv", "w") as log, cf.ProcessPoolExecutor(a.jobs) as pool:
        log.write(
            "date,n_ref,dropped,post_rms_mm,xrot,yrot,zrot,xtran,ytran,ztran,error\n"
        )
        jobs = [pool.submit(_safe, d, a.frame, a.tables) for d in days]
        for n, fut in enumerate(cf.as_completed(jobs), 1):
            r = fut.result()
            th = ",".join(f"{v:.5f}" for v in r.theta)
            log.write(
                f"{r.epoch:%Y-%m-%d},{r.n_ref},{' '.join(r.dropped)},{r.post_rms * 1e3:.3f},{th},{r.error}\n"
            )
            for c, (x, cv) in r.sites.items():
                if keep is None or c in keep:
                    series.setdefault(c, []).append((r.epoch, x, cv))
            if n % 500 == 0:
                print(f"{n}/{len(days)} days", file=sys.stderr)
    if a.to:
        _to_itrf2020(series)
    for c, rows in sorted(series.items()):
        rows.sort(key=lambda r: r[0])
        pos.write_pos(
            a.out / f"{c}.{prod}.pos",
            c,
            np.array([np.datetime64(r[0], "s") for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]),
            frame=label,
        )
    print(f"{len(days)} days → {len(series)} stations in {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
