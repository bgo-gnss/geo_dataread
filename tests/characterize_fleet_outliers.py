#!/usr/bin/env python
"""Fleet outlier-detection characterization harness (PLAN T0).

Measures what the detector actually does, per station x window x component,
**through the production resolution chain** -- so a candidate
``outlier_overrides.csv`` can be evaluated exactly as deployment would apply
it, rather than through ad-hoc flags.

Not collected by pytest (no ``test_`` prefix); run it by hand, same convention
as ``gps_analysis/tests/verification_outlier_ab.py``.

Why this lives in geo_dataread: the measurement path is
``gps_read.getData(...)`` and the resolution chain is ``gps_views``; the
gps_analysis leaf must not import a downstream package.

REYK note: this harness applies the same finite-epoch mask that
``detect_view_outliers`` does, so stations whose series contain non-finite
epochs are measurable here even though the leaf itself would raise on them.
That leaf-level fragility is a separate, backlogged issue -- not worked around
beyond mirroring production behaviour.

Examples
--------
Baseline over the working set, all three windows::

    python tests/characterize_fleet_outliers.py \\
        --stations RHOF VMEY HOFN SAUD GFUM AKUR REYK \\
        --windows 90d,1yr,full --ref plate --uncert 10 \\
        --save tests/data/fleet_baseline.tsv

Evaluate a candidate per-station config, diffed against that baseline::

    python tests/characterize_fleet_outliers.py \\
        --outlier-overrides ./candidate_overrides.csv \\
        --baseline tests/data/fleet_baseline.tsv

Ad-hoc parameter sweep (fleet-wide, overrides the catalog)::

    python tests/characterize_fleet_outliers.py --stations RHOF \\
        --outlier-param despike=true --outlier-param despike_n_sigma=3.5
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import typing
import warnings

import numpy as np

WORKING_SET = ("RHOF", "VMEY", "HOFN", "SAUD", "GFUM", "AKUR", "REYK", "THEY")
COMPONENTS = ("north", "east", "up")
WINDOWS = {"90d": 91, "1yr": 366, "full": None}

FIELDS = (
    "sta",
    "window",
    "comp",
    "n_epochs",
    "n_candidates",
    "cand_frac",
    "n_flagged",
    "abort",
    "converged",
    "scale_global",
    "n_provisional",
    "params_src",
    "seconds",
)


# --------------------------------------------------------------------------
# parameter resolution -- deliberately the production chain, not a shortcut
# --------------------------------------------------------------------------


def build_outlier_params(assignments):
    """``NAME=VALUE`` strings -> OutlierParams, or None if none given.

    Field names and types come off the dataclass, so this never restates a
    default and cannot drift from the leaf.  Mirrors
    ``gps_plot.plot_gps_timeseries._build_outlier_params``; returning None
    (rather than a default-valued object) is what lets the per-station
    catalog win downstream.
    """
    if not assignments:
        return None
    from gps_analysis import OutlierParams

    hints = typing.get_type_hints(OutlierParams)
    valid = sorted(f.name for f in dataclasses.fields(OutlierParams))
    values = {}
    for item in assignments:
        name, sep, raw = item.partition("=")
        name, raw = name.strip(), raw.strip()
        if not sep:
            raise SystemExit(f"--outlier-param expects NAME=VALUE, got {item!r}")
        if name not in valid:
            raise SystemExit(
                f"--outlier-param: unknown field {name!r}; valid: {', '.join(valid)}"
            )
        typ = hints[name]
        if typ is bool:
            low = raw.lower()
            if low in ("1", "true", "yes", "on"):
                values[name] = True
            elif low in ("0", "false", "no", "off"):
                values[name] = False
            else:
                raise SystemExit(f"--outlier-param {name}: expected bool, got {raw!r}")
        else:
            try:
                values[name] = typ(raw)
            except (ValueError, TypeError):
                raise SystemExit(
                    f"--outlier-param {name}: expected {typ.__name__}, got {raw!r}"
                ) from None
    try:
        return OutlierParams(**values)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"--outlier-param: {exc}") from None


def resolve_for(sta, explicit_params, overrides_path):
    """Station-aware resolution, identical to the plot/write paths."""
    from geo_dataread import gps_views

    steps, _ = gps_views.station_step_epochs(sta)
    windows, _ = gps_views.resolve_protect_windows(sta)
    resolved = gps_views.resolve_outlier_detection(
        sta, outlier_params=explicit_params, outlier_overrides=overrides_path
    )
    src = (
        "explicit"
        if explicit_params is not None
        else (resolved.overrides_source or "spec-default")
    )
    return resolved, steps, windows, src


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def slice_window(yearf, days, end_yearf):
    if days is None:
        return np.ones(yearf.shape, dtype=bool)
    return (yearf >= end_yearf - days / 365.25) & (yearf <= end_yearf)


def measure(sta, args, explicit_params):
    """One station -> a row per (window, component). Never raises."""
    import geo_dataread.gps_read as gpsr
    from gps_analysis import models as ga_models
    from gps_analysis.outliers import detect_outliers

    from geo_dataread import gps_views

    try:
        yearf, data, ddata, _ = gpsr.getData(sta, ref=args.ref, uncert=args.uncert)
    except Exception as exc:
        print(f"# {sta}: read failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return []

    resolved, steps, pwin, src = resolve_for(
        sta, explicit_params, args.outlier_overrides
    )
    end = float(yearf[-1]) if args.end is None else args.end
    rows = []

    for wname in args.windows:
        sel = slice_window(yearf, WINDOWS[wname], end)
        t, y, s = yearf[sel], data[:, sel], ddata[:, sel]
        if t.size < args.min_epochs:
            continue

        # the finite mask production applies (this is what makes REYK work)
        fin = np.isfinite(t) & np.all(np.isfinite(y), axis=0)
        fin &= np.all(np.isfinite(s) & (s > 0.0), axis=0)
        if int(fin.sum()) < args.min_epochs:
            continue

        t0 = time.perf_counter()
        try:
            det = detect_outliers(
                ga_models.lineperiodic,
                t[fin],
                y[:, fin],
                s[:, fin],
                step_epochs=steps if getattr(steps, "size", 0) else None,
                protect_windows=pwin,
                min_outlier=resolved.min_outlier,
                params=resolved.params,
                names=list(COMPONENTS),
            )
        except Exception as exc:
            print(
                f"# {sta}/{wname}: detect failed ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
            continue
        elapsed = time.perf_counter() - t0

        # provisional lives in the geo_dataread wrapper, not the leaf
        nprov = [0, 0, 0]
        if args.provisional:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _f, prov = gps_views.detect_view_outliers(
                    t,
                    y,
                    s,
                    outlier_params=resolved.params,
                    step_epochs=steps if getattr(steps, "size", 0) else None,
                    protect_windows=pwin,
                    min_outlier=resolved.min_outlier,
                )
            pm = np.atleast_2d(np.asarray(prov.get("provisional", False)))
            if pm.shape[0] == 3:
                nprov = pm.sum(axis=1).tolist()

        flags = np.atleast_2d(det.flags)
        cands = np.atleast_2d(det.candidates)
        for c, cname in enumerate(COMPONENTS):
            comp_abort = getattr(det, "component_abort", None)
            aborted = (
                bool(np.atleast_1d(comp_abort)[c])
                if comp_abort is not None
                else bool(det.excess_flag_abort)
            )
            rows.append(
                dict(
                    sta=sta,
                    window=wname,
                    comp=cname,
                    n_epochs=int(fin.sum()),
                    n_candidates=int(cands[c].sum()),
                    cand_frac=float(cands[c].mean()),
                    n_flagged=int(flags[c].sum()),
                    abort=aborted,
                    converged=bool(det.converged),
                    scale_global=float(det.scale_global[c]),
                    n_provisional=int(nprov[c]),
                    params_src=src,
                    seconds=round(elapsed, 3),
                )
            )
    return rows


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def key(row):
    return (row["sta"], row["window"], row["comp"])


def write_tsv(rows, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(FIELDS) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[f]) for f in FIELDS) + "\n")


def read_tsv(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            vals = line.rstrip("\n").split("\t")
            r = dict(zip(head, vals, strict=True))
            out[(r["sta"], r["window"], r["comp"])] = r
    return out


def report(rows, baseline):
    print(
        f"{'sta':6s} {'window':7s} {'comp':6s} {'N':>6s} {'cand':>6s} "
        f"{'frac':>7s} {'flag':>6s} {'abort':>6s} {'conv':>5s} "
        f"{'s_glob':>7s} {'prov':>5s} {'sec':>6s}  params"
    )
    for r in rows:
        line = (
            f"{r['sta']:6s} {r['window']:7s} {r['comp']:6s} {r['n_epochs']:6d} "
            f"{r['n_candidates']:6d} {r['cand_frac']:7.4f} {r['n_flagged']:6d} "
            f"{str(r['abort']):>6s} {str(r['converged']):>5s} "
            f"{r['scale_global']:7.3f} {r['n_provisional']:5d} "
            f"{r['seconds']:6.2f}  {r['params_src']}"
        )
        if baseline:
            b = baseline.get(key(r))
            if b is None:
                line += "   [NEW]"
            else:
                d = r["n_flagged"] - int(b["n_flagged"])
                if d or str(r["abort"]) != b["abort"]:
                    line += f"   [flag {d:+d}"
                    if str(r["abort"]) != b["abort"]:
                        line += f", abort {b['abort']}->{r['abort']}"
                    line += "]"
        print(line)

    tot = sum(r["n_flagged"] for r in rows)
    print(
        f"\nTOTAL flagged: {tot}   rows: {len(rows)}   "
        f"aborted components: {sum(1 for r in rows if r['abort'])}"
    )
    if baseline:
        # Compare ONLY over rows present in both.  Summing the full current
        # run against a baseline with a different row set produces a
        # meaningless percentage -- e.g. adding two stations to the joined
        # dataset once read as "+64.8%" when nothing about the detector had
        # changed.  Rows that exist on only one side are counted, not hidden.
        cur = {key(r): r for r in rows}
        shared = cur.keys() & baseline.keys()
        only_now = sorted(cur.keys() - baseline.keys())
        only_before = sorted(baseline.keys() - cur.keys())
        tot = sum(cur[k]["n_flagged"] for k in shared)
        btot = sum(int(baseline[k]["n_flagged"]) for k in shared)
        print(
            f"comparable rows: {len(shared)} of {len(rows)} current / "
            f"{len(baseline)} baseline"
        )
        if only_now:
            print(
                f"  only in this run ({len(only_now)}): "
                + ", ".join("/".join(k) for k in only_now[:6])
                + (" …" if len(only_now) > 6 else "")
            )
        if only_before:
            print(
                f"  only in baseline ({len(only_before)}): "
                + ", ".join("/".join(k) for k in only_before[:6])
                + (" …" if len(only_before) > 6 else "")
            )
        print(
            f"comparable flagged: now {tot}  baseline {btot}  "
            f"delta {tot - btot:+d} ({100 * (tot - btot) / max(btot, 1):+.1f}%)"
        )


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Fleet outlier-detection characterization (PLAN T0)",
        epilog="Parameters resolve through the production chain: explicit "
        "--outlier-param beats the per-station outlier_overrides.csv "
        "row, which beats the spec defaults.",
    )
    p.add_argument("--stations", nargs="+", default=list(WORKING_SET))
    p.add_argument(
        "--windows",
        default="90d,1yr,full",
        help="comma-separated subset of 90d,1yr,full",
    )
    p.add_argument("--ref", default="plate")
    p.add_argument("--uncert", type=int, default=10)
    p.add_argument(
        "--end",
        type=float,
        default=None,
        help="window anchor [fractional year]; default = each "
        "station's own last epoch (reproducible, and keeps a "
        "station that stopped years ago measurable)",
    )
    p.add_argument("--min-epochs", type=int, default=60)
    p.add_argument(
        "--outlier-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="repeatable; overrides the catalog fleet-wide",
    )
    p.add_argument(
        "--outlier-overrides",
        default=None,
        metavar="PATH",
        help="per-station outlier_overrides.csv to evaluate "
        "instead of the deployed one -- the config-file lever",
    )
    p.add_argument(
        "--no-provisional",
        dest="provisional",
        action="store_false",
        help="skip the provisional mask (halves runtime)",
    )
    p.add_argument("--baseline", default=None, metavar="TSV")
    p.add_argument("--save", default=None, metavar="TSV")
    args = p.parse_args(argv)

    bad = [w for w in args.windows.split(",") if w not in WINDOWS]
    if bad:
        raise SystemExit(f"unknown window(s): {bad}; valid: {list(WINDOWS)}")
    args.windows = args.windows.split(",")

    warnings.filterwarnings("ignore")
    explicit = build_outlier_params(args.outlier_param)

    rows = []
    for sta in args.stations:
        rows.extend(measure(sta, args, explicit))

    baseline = read_tsv(args.baseline) if args.baseline else None
    report(rows, baseline)
    if args.save:
        write_tsv(rows, args.save)
        print(f"\nwrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
