"""``gpsquake``: earthquake catalogues for a region, across the SIL → SeisComP seam.

    gpsquake regions                        named regions (quake_regions.yaml)
    gpsquake fetch REGION -s START [-e END] summary, or -o FILE / --list rows

REGION is a named region, ``W,S,E,N`` (box) or ``LAT,LON,KM`` (radius);
``--polygon`` takes ``lon lat, lon lat, …``. The selection, source, mode and
Q options come from :func:`add_quake_args`, which ``gpsplot`` reuses with
the prefix ``quake-`` (PLAN-ts-quakes Q-T6), so both read quakes the same way
(:func:`geo_dataread.seismic.selection.catalogue_for`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd

TOP_EPILOG = """\
examples:
  gpsquake regions
  gpsquake fetch bardarbunga-caldera -s 20180101                 # summary + largest events
  gpsquake fetch bardarbunga-caldera -s 20260101 --min-q 40      # swarm: raise the Q floor
  gpsquake fetch -18.65,64.18,-16.8,65.16 -s 20260201 --source sil --mode automatic
  gpsquake fetch 63.88,-22.4,15 -s 20231101 -e 20231201 --moment -o skjalftar.csv
"""

FETCH_EPILOG = """\
source (--source):
  auto      SIL files before the seam (2026-02-01), SeisComP (Quakes API) after;
            with --mode automatic: SIL for the whole window (one kind of row)
  sil       SIL files only            seiscomp  Quakes API only
  fdsn      SeisComP via the FDSN event service only

mode (--mode):
  catalog   reviewed events + automatic ones not yet reviewed (default)
  automatic every automatic solution (SIL), or unreviewed events (SeisComP)
  reviewed  reviewed events only

--min-q acts on SIL automatic rows only (default 10; raise it in swarms).
Output: no -o → a summary and the largest events; -o FILE.csv|.parquet writes
the rows and FILE.json (provenance); --list writes CSV to stdout.
"""


# --------------------------------------------------------------------------
# Shared options (gpsquake + gpsplot --quake-*)
# --------------------------------------------------------------------------


def _floats(text: str, n: int, what: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{what}: numbers, got {text!r}") from exc
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f"{what}: {n} comma-separated numbers")
    return vals


def _date(text: str) -> pd.Timestamp:
    t = text.strip()
    if len(t) == 8 and t.isdigit():
        t = f"{t[:4]}-{t[4:6]}-{t[6:]}"
    try:
        ts = pd.Timestamp(t)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad date {text!r}") from exc
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def add_quake_args(parser: argparse.ArgumentParser, prefix: str = "") -> None:
    """Add the selection, source, mode and Q options.

    ``prefix`` (e.g. ``"quake-"``) is put before every long option name, so
    ``gpsplot`` gets ``--quake-mode`` etc.; :func:`quake_kwargs` reads them
    back with the same prefix. The region itself is the caller's argument.
    """
    p = f"--{prefix}"
    g = parser.add_argument_group("quakes")
    g.add_argument(
        f"{p}source",
        choices=("auto", "sil", "seiscomp", "fdsn"),
        default="auto",
        help="catalogue source (default auto: SIL before 2026-02-01, SeisComP after)",
    )
    g.add_argument(
        f"{p}mode",
        choices=("catalog", "automatic", "reviewed"),
        default="catalog",
        help="catalog (default) | automatic | reviewed",
    )
    g.add_argument(
        f"{p}min-q",
        type=float,
        default=10.0,
        metavar="Q",
        help="lowest SIL automatic Q kept (default 10; raise it in swarms; "
        "0 keeps every Q)",
    )
    g.add_argument(
        f"{p}no-duplicates",
        action="store_true",
        help="drop SIL automatic rows flagged possible_duplicate",
    )
    g.add_argument(f"{p}min-mag", type=float, default=None, metavar="M")
    g.add_argument(f"{p}max-mag", type=float, default=None, metavar="M")
    g.add_argument(f"{p}min-depth", type=float, default=None, metavar="KM")
    g.add_argument(f"{p}max-depth", type=float, default=None, metavar="KM")
    g.add_argument(
        f"{p}types",
        default="earthquake",
        metavar="LIST",
        help="event types kept (comma list, or 'all'; default earthquake; "
        "a missing type is kept)",
    )
    g.add_argument(
        f"{p}polygon",
        default=None,
        metavar="'LON LAT, …'",
        help="polygon region instead of REGION",
    )
    g.add_argument(
        f"{p}regions",
        type=Path,
        default=None,
        metavar="FILE",
        help="quake_regions.yaml (default: the deployed one)",
    )
    g.add_argument(
        f"{p}no-cache",
        action="store_true",
        help="read the SIL archive, not the Parquet cache",
    )


def _get(args: argparse.Namespace, prefix: str, name: str) -> Any:
    return getattr(args, (prefix + name).replace("-", "_"))


def quake_selection(
    args: argparse.Namespace, region: str | None, prefix: str = ""
) -> Any:
    """The :class:`~geo_dataread.seismic.selection.QuakeSelection` of the args."""
    from geo_dataread.seismic.selection import (
        QuakeSelection,
        parse_region_spec,
        read_quake_regions,
    )

    poly = _get(args, prefix, "polygon")
    if poly is not None and region is not None:
        raise SystemExit("gpsquake: give REGION or --polygon, not both")
    if poly is not None:
        pts = []
        for pair in poly.split(","):
            bits = pair.split()
            if len(bits) != 2:
                raise SystemExit(f"gpsquake: polygon vertex {pair!r} is not 'LON LAT'")
            pts.append((float(bits[0]), float(bits[1])))
        sel = QuakeSelection(polygon=tuple(pts), name="polygon")
    elif region is not None:
        regions_file = _get(args, prefix, "regions")
        sel = parse_region_spec(region, lambda: read_quake_regions(regions_file))
    else:
        sel = QuakeSelection(name="all")
    types = _get(args, prefix, "types")
    et = None if types.strip() == "all" else tuple(t.strip() for t in types.split(","))
    bounds = {
        "min_mag": _get(args, prefix, "min_mag"),
        "max_mag": _get(args, prefix, "max_mag"),
        "min_depth": _get(args, prefix, "min_depth"),
        "max_depth": _get(args, prefix, "max_depth"),
    }
    # a named region's own bounds stay unless the command line sets them
    changes = {k: v for k, v in bounds.items() if v is not None}
    return sel.replace(event_types=et, **changes)


def quake_kwargs(args: argparse.Namespace, prefix: str = "") -> dict[str, Any]:
    """The :func:`catalogue_for` keyword arguments of the args."""
    q = _get(args, prefix, "min_q")
    return {
        "source": _get(args, prefix, "source"),
        "mode": _get(args, prefix, "mode"),
        "min_quality": None if q is None or q <= 0 else float(q),
        "drop_duplicates": bool(_get(args, prefix, "no_duplicates")),
        "use_cache": not _get(args, prefix, "no_cache"),
    }


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    raw = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(
        prog="gpsquake",
        description="Earthquake catalogues for a region (SIL files + SeisComP).",
        epilog=TOP_EPILOG,
        formatter_class=raw,
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    r = sub.add_parser("regions", help="list the named quake regions")
    r.add_argument("--regions", type=Path, default=None, metavar="FILE")

    f = sub.add_parser(
        "fetch",
        help="the quakes of a region and period",
        epilog=FETCH_EPILOG,
        formatter_class=raw,
    )
    f.add_argument(
        "region",
        nargs="?",
        default=None,
        help="named region, W,S,E,N or LAT,LON,KM (none: no region cut); "
        "a list starting with a minus works as is",
    )
    f.add_argument(
        "-s", "--start", type=_date, required=True, help="YYYYMMDD or ISO (UTC)"
    )
    f.add_argument(
        "-e", "--end", type=_date, default=None, help="exclusive (default: now)"
    )
    add_quake_args(f)
    f.add_argument("--moment", action="store_true", help="add the moment estimate")
    f.add_argument(
        "--top", type=int, default=10, metavar="N", help="largest events shown (10)"
    )
    out = f.add_mutually_exclusive_group()
    out.add_argument("-o", "--output", type=Path, default=None, metavar="FILE")
    out.add_argument("--list", action="store_true", help="CSV rows to stdout")
    return p


def _regions(args: argparse.Namespace, out: TextIO) -> int:
    from geo_dataread.seismic.selection import read_quake_regions

    regions = read_quake_regions(args.regions)
    for r in regions.values():
        s = r.selection
        if s.bbox is not None:
            shape = "bbox " + ",".join(f"{v:g}" for v in s.bbox)
        elif s.radius is not None:
            shape = "radius {:g},{:g},{:g} km".format(*s.radius)
        else:
            shape = f"polygon ({len(s.polygon or ())} vertices)"
        flag = " [default]" if r.default else ""
        area = f"  area={r.area}{flag}" if r.area else ""
        title = f"  {r.title}" if r.title else ""
        print(f"{r.name:28s} {shape}{area}{title}", file=out)
    return 0


#: Columns of --list / -o CSV, in order (the rest follow).
LIST_COLUMNS = (
    "source",
    "event_id",
    "latitude",
    "longitude",
    "depth_km",
    "magnitude",
    "magnitude_type",
    "mag_pgv",
    "evaluation_mode",
    "evaluation_status",
    "event_type",
    "quality",
    "possible_duplicate",
)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    first = [c for c in LIST_COLUMNS if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def _summary(df: pd.DataFrame, report: Any, top: int, out: TextIO) -> None:
    prov = df.attrs.get("catalog", {})
    win = prov.get("window", {})
    print(
        f"{prov.get('selection', {}).get('name') or 'selection'}  "
        f"{win.get('start', '')[:19]} → {win.get('end', '')[:19]}  "
        f"n = {len(df)}",
        file=out,
    )
    for part in report.parts:
        extra = ", ".join(f"{k} {v}" for k, v in part.counts.items())
        print(
            f"  {part.source:16s} {part.start:%Y-%m-%d} → {part.end:%Y-%m-%d}  "
            f"{part.kept} kept of {part.read} read" + (f"  ({extra})" if extra else ""),
            file=out,
        )
    if not len(df):
        return
    by = (
        df.assign(
            mode=df["evaluation_mode"].astype(str),
            mtype=df["magnitude_type"].astype(object).fillna("—"),
        )
        .groupby(["source", "mode", "mtype"], observed=True)
        .size()
    )
    print("  rows by source / mode / magnitude type:", file=out)
    for (src, mode, mtype), n in by.items():
        print(f"    {src:16s} {mode:9s} {mtype:9s} {n}", file=out)
    if "possible_duplicate" in df:
        nd = int(np.asarray(df["possible_duplicate"].fillna(False), dtype=bool).sum())
        if nd:
            print(f"  possible duplicates (kept): {nd}", file=out)
    if "moment_nm" in df and "moment_estimated" in df:
        m0 = pd.to_numeric(df["moment_nm"]).to_numpy(dtype="float64", na_value=np.nan)
        ok = np.isfinite(m0)
        est = np.asarray(df["moment_estimated"], dtype=bool)
        from geo_dataread.seismic.schema import moment_magnitude

        tot = float(np.nansum(m0))
        mw = float(moment_magnitude(tot)) if tot > 0 else float("nan")
        print(
            f"  cumulative moment {tot:.3g} N·m (Mw {mw:.2f}); "
            f"{int(ok.sum())} rows with a moment, {int(est.sum())} estimated",
            file=out,
        )
    if top > 0:
        big = df.assign(_m=pd.to_numeric(df["magnitude"])).nlargest(top, "_m")
        print(f"  largest {len(big)}:", file=out)
        for t, r in big.iterrows():
            pgv = r.get("mag_pgv")
            pgv_s = f"  pgv {pgv:.1f}" if pd.notna(pgv) else ""
            print(
                f"    {pd.Timestamp(t):%Y-%m-%d %H:%M:%S}  M {r['magnitude']:.2f} "
                f"{r['magnitude_type']!s:7s} {r['latitude']:.3f} {r['longitude']:.3f} "
                f"{r['depth_km']:.1f} km  {r['evaluation_mode']}{pgv_s}",
                file=out,
            )


def _fetch(args: argparse.Namespace, out: TextIO) -> int:
    from geo_dataread.seismic.moment import moment_estimate
    from geo_dataread.seismic.selection import catalogue_for

    end = args.end if args.end is not None else pd.Timestamp.now(tz="UTC")
    sel = quake_selection(args, args.region)
    df, report = catalogue_for(sel, args.start, end, **quake_kwargs(args))
    if args.moment:
        df = moment_estimate(df)
    if args.list:
        _ordered(df).to_csv(out, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
        return 0
    if args.output is not None:
        path: Path = args.output
        rows = _ordered(df)
        if path.suffix == ".parquet":
            flat = rows.copy()
            flat.attrs = {}
            flat.to_parquet(path)
        else:
            rows.to_csv(path, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
        side = path.with_suffix(".json")
        side.write_text(
            json.dumps(df.attrs.get("catalog", {}), indent=2, default=str) + "\n"
        )
        print(f"{len(df)} events → {path} (+ {side.name})", file=out)
        return 0
    _summary(df, report, args.top, out)
    return 0


#: A coordinate list (``-18.65,64.18,-16.8,65.16``): argparse would take a
#: leading minus for an option, so :func:`main` passes it as ``--=`` style.
_COORDS = re.compile(r"^-?\d+(\.\d*)?(,\s*-?\d+(\.\d*)?){2,3}$")


def _protect_coords(argv: Sequence[str]) -> list[str]:
    """Put ``--`` before a bare coordinate list that starts with a minus.

    Only the first such token after ``fetch`` (the REGION positional); the
    remaining options are moved in front of it so ``--`` ends nothing else.
    """
    args = list(argv)
    if "fetch" not in args:
        return args
    i0 = args.index("fetch") + 1
    for i in range(i0, len(args)):
        tok = args[i]
        if tok.startswith("-") and _COORDS.match(tok):
            prev = args[i - 1] if i > i0 else ""
            if prev.startswith("-") and not _COORDS.match(prev) and "=" not in prev:
                continue  # it is an option's value (--polygon …)
            return args[:i] + args[i + 1 :] + ["--", tok]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_protect_coords(raw))
    try:
        if args.command == "regions":
            return _regions(args, sys.stdout)
        return _fetch(args, sys.stdout)
    except (FileNotFoundError, ValueError) as exc:
        print(f"gpsquake: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
