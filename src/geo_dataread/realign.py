"""Re-align the loose daily SINEX archive to a reference frame → per-station ``.pos``.

Each archived day (``<archive>/<yyyy>/IMO0OPSFIN_<yyyy><doy>0000_01D_01D_SOL.SNX.gz``
+ its glorg ``.org.gz``) is stabilized with :func:`geo_dataread.stabilize.stabilize`
using the day's own glorg reference list and settings, and either

- the SINEX a priori (``--frame itrf2008``: glorg's own frame — reproduces
  the archived ``.pos``), or
- a GLOBK apr + eq + stab_site set from the GAMIT tables, as MIT's
  ``glorg.cmd`` pairs them, with renames and non-secular terms resolved at
  the day's epoch by :mod:`geo_dataread.globk_apr`: ``--frame igb14``
  (IGb14, the IGS realization of ITRF2014) or ``--frame igb20`` (IGb20, the
  IGS realization of ITRF2020). The reference sites are the day's own glorg
  list (IMO leaves out e.g. REYK/HOFN, which MIT's lists carry) restricted to
  the segments on the frame's ``stab_site`` list.

Frame consistency: the loose h-files carry the antenna model (ANTEX) and
orbits they were processed with (okada: BASELINE mode, IGS orbits held
fixed). Per the SINEX ``SITE/GPS_PHASE_CENTER`` blocks: igs05/igs08 to
2016, mostly igs08 2017-2020, igs14 from 2021; orbits IGb08 to 2016, IGS14 /
IGb14 2017 - week 2238, IGS20/IGb20 from week 2239 (2022-12). Only
ITRF2008 (to 2016) and IGb14 (2021 - 2022-11) are fully consistent; igb20 on
igs14-processed h-files mixes conventions (experimental).

``--to itrf2020`` (with ``--frame igb14``) instead moves every IGb14
position (and covariance) with the EPSG time-dependent Helmert
ITRF2014→ITRF2020 at its epoch (geofunc.frames). Output: one PBO ``.pos``
per station (:func:`geo_dataread.pos.write_pos`) and ``days.<prod-id>.csv``
with one line per day (reference sites used/dropped, post-fit RMS,
parameters). File names carry the product id, so the frames can share one
directory with tssum's ``imo.24hr_itrf08`` / ``imo.24hr_itrf20`` sets
(``update_pos.sh`` leaves other product ids alone): ``--frame igb14`` writes
the ITRF2014 product ``imo.24hr_itrf14``, the other choices write
``imo.realign_<frame>`` unless ``--prod-id`` is given.

A day with fewer than ``--min-ref`` (3) usable reference sites is not
written: one or two sites only fix translations, and on the days without
any global site (the Icelandic network alone) the whole day sits decimetres
off. ``--update`` re-makes only the days in ``--start``…``--end`` and folds
them into the existing files (:func:`geo_dataread.pos.merge_pos`); a re-made
day replaces its old rows, a rejected one removes them.

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
FRAMES: dict[str, tuple[str, str, str] | None] = {
    "itrf2008": None,
    "igb14": ("igb14_comb.apr", "igb14_comb.eq", "igb14_comb.stab_site"),
    "igb20": ("igb20_comb.apr", "igb20_comb.eq", "igb20_comb.stab_site"),
}
LABELS = {
    "itrf2008": "ITRF2008",
    "igb14": "IGb14",
    # igs14-processed h-files on igs20-based coordinates: see the module docstring
    "igb20": "IGb20 (experimental)",
}
MIN_REF = 3
_REJECTED = "rejected: "
CSV_HEAD = "date,n_ref,dropped,post_rms_mm,xrot,yrot,zrot,xtran,ytran,ztran,error\n"


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


_Tables = tuple[dict[str, globk_apr.AprSite], list[globk_apr.Rename], tuple[str, ...]]
_APR_CACHE: dict[tuple[str, str], _Tables] = {}


def _apr(frame: str, tables: Path) -> _Tables | None:
    files = FRAMES[frame]
    if files is None:
        return None
    key = (frame, str(tables))
    if key not in _APR_CACHE:
        _APR_CACHE[key] = (
            globk_apr.read_apr(tables / files[0]),
            globk_apr.read_renames(tables / files[1]),
            globk_apr.read_stab_site(tables / files[2]),
        )
    return _APR_CACHE[key]


def realign_day(
    snx: Path,
    frame: str = "itrf2008",
    tables: Path = GAMIT_TABLES,
    min_ref: int = MIN_REF,
) -> DayResult:
    """Stabilize one archived day in ``frame`` (see the module docstring).

    Fewer than ``min_ref`` reference sites left after the a priori look-up
    (or after the outlier screen) gives a result with no sites and an
    ``error`` starting ``"rejected: "``.
    """
    sol = sinex.read_sinex(snx)
    setup = org.read_org_setup(Path(str(snx).replace(".SNX.gz", ".org.gz")))
    when = sol.mean_epoch
    present = set(sol.code.tolist())
    codes = [c for c in setup.reference if c in present]
    dropped: list[str] = []
    ext = _apr(frame, tables)
    apriori = None
    if ext is not None:
        apr, renames, stab = ext
        apriori, skipped = globk_apr.apriori_xyz(codes, when, apr, renames, stab)
        dropped = sorted(skipped)
    usable = len(codes) - len(dropped)
    if usable < min_ref:
        return _rejected(when, usable, dropped, min_ref)
    st = stabilize.stabilize(sol, codes, setup.config, apriori=apriori)
    last = st.iterations[-1]
    if len(last.sites) < min_ref:
        return _rejected(when, len(last.sites), dropped, min_ref, last.post_rms)
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


def _rejected(
    when: datetime.datetime,
    n: int,
    dropped: Sequence[str],
    min_ref: int,
    rms: float = float("nan"),
) -> DayResult:
    msg = f"{_REJECTED}{n} reference site(s), need {min_ref}"
    return DayResult(when, {}, n, tuple(dropped), rms, np.full(6, np.nan), msg)


def _safe(snx: Path, frame: str, tables: Path, min_ref: int = MIN_REF) -> DayResult:
    try:
        return realign_day(snx, frame, tables, min_ref)
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


def _csv_row(r: DayResult) -> str:
    th = ",".join(f"{v:.5f}" for v in r.theta)
    return f"{r.epoch:%Y-%m-%d},{r.n_ref},{' '.join(r.dropped)},{r.post_rms * 1e3:.3f},{th},{r.error}\n"


def _write_days(path: Path, rows: dict[str, str], update: bool) -> None:
    """``days.<prod>.csv``: one line per day; ``update`` keeps the other days."""
    old: dict[str, str] = {}
    if update and path.exists():
        for ln in path.read_text().splitlines(keepends=True)[1:]:
            old[ln.split(",", 1)[0]] = ln
    old.update(rows)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(CSV_HEAD + "".join(old[d] for d in sorted(old)))
    tmp.replace(path)


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
    ap.add_argument(
        "--prod-id",
        help="default imo.24hr_itrf14 for --frame igb14 (the ITRF2014 .pos "
        "product), else imo.realign_<frame> (experiments, kept apart from the "
        "itrf08/itrf20 products)",
    )
    ap.add_argument("--stations", nargs="*", help="write only these codes")
    ap.add_argument(
        "--min-ref",
        type=int,
        default=MIN_REF,
        help=f"fewest reference sites for a day to be written (default {MIN_REF})",
    )
    ap.add_argument(
        "--update",
        action="store_true",
        help="fold the re-made days into the existing files instead of rewriting them",
    )
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    if a.to and a.frame != "igb14":
        ap.error("--to itrf2020 needs --frame igb14")

    days = archive_days(a.archive, a.start, a.end)
    if not days:
        print("no archived days in range", file=sys.stderr)
        return 1
    label = "ITRF2020 (IGb14 + EPSG Helmert)" if a.to else LABELS[a.frame]
    if a.prod_id:
        prod = a.prod_id
    elif a.frame == "igb14" and not a.to:
        prod = pos.product_id("itrf14")
    else:
        prod = f"imo.realign_{a.to or a.frame}"
    keep = {s.upper() for s in a.stations} if a.stations else None

    series: dict[str, list[tuple[datetime.datetime, FloatArray, FloatArray]]] = {}
    csv: dict[str, str] = {}
    redone: set[str] = set()  # YYYYMMDD of the days re-made or rejected
    a.out.mkdir(parents=True, exist_ok=True)
    with cf.ProcessPoolExecutor(a.jobs) as pool:
        jobs = [pool.submit(_safe, d, a.frame, a.tables, a.min_ref) for d in days]
        for n, fut in enumerate(cf.as_completed(jobs), 1):
            r = fut.result()
            csv[f"{r.epoch:%Y-%m-%d}"] = _csv_row(r)
            if not r.error or r.error.startswith(_REJECTED):
                redone.add(f"{r.epoch:%Y%m%d}")
            for c, (x, cv) in r.sites.items():
                if keep is None or c in keep:
                    series.setdefault(c, []).append((r.epoch, x, cv))
            if n % 500 == 0:
                print(f"{n}/{len(days)} days", file=sys.stderr)
    # named per product so several frames can share one pos_dir
    _write_days(a.out / f"days.{prod}.csv", csv, a.update)
    if a.to:
        _to_itrf2020(series)
    for c, rows in sorted(series.items()):
        rows.sort(key=lambda r: r[0])
        target = a.out / f"{c}.{prod}.pos"
        old = pos.read_pos(target) if a.update and target.exists() else None
        out = pos.write_pos(
            target.with_suffix(".new") if old is not None else target,
            c,
            np.array([np.datetime64(r[0], "s") for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]),
            frame=label,
            ref_xyz=None if old is None else old.ref_xyz,
            soln="24hrs",
        )
        if old is not None:
            pos.merge_pos(target, out, sorted(redone))
    if a.update:  # stations with nothing new may still lose a rejected day
        for f in a.out.glob(f"*.{prod}.pos"):
            if f.name.split(".")[0] not in series:
                pos.merge_pos(f, None, redone)
    what = "updated" if a.update else "written"
    print(
        f"{len(days)} days ({len(redone)} re-made) → {len(series)} stations {what} in {a.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
