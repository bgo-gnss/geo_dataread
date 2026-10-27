"""``gps-velocities``: the velocity product's command line (moved from
geo_dataread.velocity in gpsprod step 3; bodies unchanged)."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from geo_dataread.products.velocity_core import (
    DEFAULT_FRAME,
    DEFAULT_K,
    DEFAULT_UNCERT,
    DEFAULT_VELOCITY_DIR,
    DEFAULT_VIEW,
    K_RESIDUAL_REGIONAL,
    NO_RECORD_POLICIES,
    REFS,
    VIEWS,
    VelocityRecord,
    VelocitySettings,
    _coords,
    build_k_table,
    estimate_velocities,
    k_table_path,
    last_days_start,
    velocity_dir,
    window_bounds,
)
from geo_dataread.products.velocity_files import (
    read_velocity_records,
    set_name,
    write_velocity_set,
)

# ---------------------------------------------------------------------------
# command line: gps-velocities
# ---------------------------------------------------------------------------


def _date(value: str) -> dt.date:
    try:
        return dt.datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: use YYYYMMDD") from None


def _window(value: str) -> tuple[dt.date, dt.date]:
    a, sep, b = value.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"{value!r}: use START:END (YYYYMMDD)")
    return _date(a), _date(b)


def _box(value: str) -> tuple[float, float, float, float]:
    try:
        w, e, s, n = (float(v) for v in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: use W/E/S/N") from None
    return w, e, s, n


def select_stations(
    patterns: Sequence[str], region: tuple[float, float, float, float] | None
) -> list[str]:
    """stations.cfg markers matching ``patterns`` (all when empty) and
    inside ``region`` (W/E/S/N) when given."""
    from gps_parser import ConfigParser

    cfg = ConfigParser()
    names = [str(s) for s in cfg.getStationInfo() if str(s) != "DEFAULTS"]
    pats = [p.upper() for p in patterns] or ["*"]
    picked = [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in pats)]
    if region is None:
        return sorted(picked)
    w, e, s, n = region
    out = []
    for sta in picked:
        try:
            lon, lat = _coords(sta)
        except Exception:  # an unknown station: not in the region
            continue
        if w <= lon <= e and s <= lat <= n:
            out.append(sta)
    return sorted(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gps-velocities",
        description="The velocity product: estimate windowed station velocities "
        "into a versioned set (.gmt/.csv/.jsonl), or show one.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="estimate a set over one window")
    e.add_argument("stations", nargs="*", help="markers or patterns (SEY?, *)")
    e.add_argument("--region", type=_box, default=None, metavar="W/E/S/N")
    e.add_argument("--window", type=_window, default=None, metavar="START:END")
    e.add_argument("--last", type=int, default=None, metavar="DAYS")
    e.add_argument("-e", "--end", type=_date, default=None,
                   help="YYYYMMDD: the last day of --last (default today)")  # fmt: skip
    e.add_argument("--ref", choices=REFS, default="plate-periodic",
                   help="plate-periodic: seasonal removed; background: secular + "
                   "seasonal removed (the transient); plate: nothing more; "
                   "pre-onset: plate-periodic less the --pre-window rate")  # fmt: skip
    e.add_argument("--pre-window", type=_window, default=None, metavar="START:END",
                   help="ref pre-onset: the reference days before the onset, "
                   "YYYYMMDD:YYYYMMDD both included (e.g. 20230806:20231024)")  # fmt: skip
    e.add_argument("--view", choices=VIEWS, default=DEFAULT_VIEW)
    e.add_argument("--frame", default=DEFAULT_FRAME)
    e.add_argument("-u", "--uncert", type=float, default=DEFAULT_UNCERT)
    e.add_argument("--no-record", choices=NO_RECORD_POLICIES, default="plate",
                   help="a station without a detrend record: fall back to ref "
                   "plate (flagged) or refuse")  # fmt: skip
    e.add_argument("--allow-steps", action="store_true",
                   help="keep a window with a declared step inside (flagged)")  # fmt: skip
    e.add_argument("--min-obs", type=int, default=VelocitySettings.min_obs)
    e.add_argument("--min-cover", type=float, default=VelocitySettings.min_cover)
    e.add_argument(
        "--no-short-tier",
        dest="short_tier",
        action="store_false",
        help="refuse every window failing the count/cover gates "
        "(default: keep the parts whose |v|/sigma reaches --short-snr)",
    )
    e.add_argument("--short-snr", type=float, default=VelocitySettings.short_snr)
    e.add_argument("--short-snr-pair", type=float,
                   default=VelocitySettings.short_snr_pair,
                   help="the short tier's |v|/sigma on 2 epochs")  # fmt: skip
    e.add_argument("--default-k", type=float, default=DEFAULT_K)
    e.add_argument(
        "--k-table", default=None,
        help="k(T) table for stations without a quiet span of their own "
        "(default: k_table_<frame>.json in the velocity dir when built; "
        "'none' = default-k)",
    )  # fmt: skip
    e.add_argument("--pos-dir", default=None)
    e.add_argument("--params", default=None, help="detrend parameter document")
    e.add_argument("--steps", default=None, help="steps.yaml")
    e.add_argument("--out", default=None,
                   help="directory (default: velocity_dir of postprocess.cfg, "
                   f"else {DEFAULT_VELOCITY_DIR})")  # fmt: skip
    e.add_argument(
        "--name", default=None, help="set name (default from window/ref/frame)"
    )
    t = sub.add_parser(
        "k-table",
        help="build the k(T) table: the median empirical rate scale over every "
        "station with a detrend record, the fallback of stations without one",
    )
    t.add_argument("stations", nargs="*", help="markers or patterns (default all)")
    t.add_argument("--frame", default=DEFAULT_FRAME)
    t.add_argument("--view", choices=VIEWS, default=DEFAULT_VIEW)
    t.add_argument("-u", "--uncert", type=float, default=DEFAULT_UNCERT)
    t.add_argument("--pos-dir", default=None)
    t.add_argument("--params", default=None)
    t.add_argument("--out", default=None, help="file (default k_table_<frame>.json "
                   "in the velocity dir)")  # fmt: skip
    s = sub.add_parser("show", help="print a set")
    s.add_argument("path")
    s.add_argument("--all", action="store_true", help="refused rows too")
    return p


def _show(records: Sequence[VelocityRecord], show_all: bool) -> None:
    print(f"{'sta':4} {'vn':>9} {'ve':>9} {'vu':>9} {'sn':>7} {'se':>7} {'su':>7} "
          f"{'n':>4} {'σ':>18} ref/flags")  # fmt: skip
    for r in records:
        if not (r.accepted or show_all):
            continue
        if r.accepted:
            vn, ve, vu = r.rate
            sn, se, su = r.sigma
            nums = f"{vn:9.2f} {ve:9.2f} {vu:9.2f} {sn:7.2f} {se:7.2f} {su:7.2f}"
        else:
            nums = f"{'refused':>9} {'':9} {'':9} {'':7} {'':7} {'':7}"
        note = r.refused or "|".join((r.ref, *r.flags))
        print(f"{r.station:4} {nums} {min(r.n_epochs):4d} {r.sigma_method:>18} {note}")


def _k_table_main(args: argparse.Namespace) -> int:
    settings = VelocitySettings(frame=args.frame, view=args.view,
                                uncert=args.uncert, pos_dir=args.pos_dir,
                                params=args.params)  # fmt: skip
    stations = select_stations(args.stations, None)
    table = build_k_table(
        stations, settings, progress=lambda sta: print(sta, file=sys.stderr, end=" ")
    )
    out = Path(args.out).expanduser() if args.out else k_table_path(settings)
    # the `regional` entry is measured by gps-regional-seasonal --k-table, not
    # here: carry it over (marked) rather than silently drop it
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8")).get("residuals") or {}
        if K_RESIDUAL_REGIONAL in old:
            kept = dict(old[K_RESIDUAL_REGIONAL])
            kept["carried_over"] = True
            table["residuals"][K_RESIDUAL_REGIONAL] = kept
            print(f"kept the '{K_RESIDUAL_REGIONAL}' entry of {out}; re-measure it "
                  "with gps-regional-seasonal --k-table", file=sys.stderr)  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{len(table['stations'])} stations -> {out}")
    for residual, entry in table["residuals"].items():
        for d, k, n in zip(entry["days"], entry["k"], entry["n_stations"], strict=True):
            print(f"  {residual:8} {d:4d} d  k N/E/U {k[0]:5.2f} {k[1]:5.2f} {k[2]:5.2f}"
                  f"  ({n} stations)")  # fmt: skip
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "show":
        _show(read_velocity_records(args.path), args.all)
        return 0
    if args.cmd == "k-table":
        return _k_table_main(args)
    if (args.window is None) == (args.last is None):
        print("error: give --window START:END or --last DAYS", file=sys.stderr)
        return 2
    if args.window is not None:
        start, end = args.window
    else:
        end = args.end or dt.date.today()
        start = last_days_start(end, args.last)
    try:
        settings = VelocitySettings(
            frame=args.frame, ref=args.ref, view=args.view, uncert=args.uncert,
            no_record=args.no_record, allow_steps=args.allow_steps,
            min_obs=args.min_obs, min_cover=args.min_cover,
            short_tier=args.short_tier, short_snr=args.short_snr,
            short_snr_pair=args.short_snr_pair,
            default_k=args.default_k, pos_dir=args.pos_dir, params=args.params,
            steps=args.steps, k_table=args.k_table,
            pre_window=window_bounds(*args.pre_window) if args.pre_window else None,
        )  # fmt: skip
        stations = select_stations(args.stations, args.region)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not stations:
        print("error: no station selected", file=sys.stderr)
        return 2

    def report(r: VelocityRecord) -> None:
        state = "ok" if r.accepted else f"refused: {r.refused}"
        print(f"{r.station}: [{state}] {r.ref} {r.sigma_method}", file=sys.stderr)

    records = estimate_velocities(stations, start, end, settings, progress=report)
    name = args.name or set_name(settings, start, end)
    out = Path(args.out).expanduser() if args.out else velocity_dir()
    paths = write_velocity_set(records, out, name)
    n_ok = sum(r.accepted for r in records)
    print(
        f"{n_ok} of {len(records)} stations -> {paths['csv'].with_suffix('')}.{{gmt,csv,jsonl}}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
