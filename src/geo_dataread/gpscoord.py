"""``gpscoord``: coordinates in and out — point tables, nearest stations,
daily station coordinates with the columns you ask for, a position at an epoch.

    gpscoord convert FILE [--from FRAME] [--to FRAME]
    gpscoord nearest FILE [--candidates FILE] [-k N]
    gpscoord series STA… [--columns LIST] [-o DIR/]
    gpscoord at STA… --epoch DATE                 (gpsprod coords)

The maths is geofunc's (:mod:`geofunc.points`, :mod:`geofunc.columns`);
this module picks the stations, reads their ``.pos`` series
(:mod:`geo_dataread.products.positions`) and writes the files.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import numpy as np

TOP_EPILOG = """\
examples:
  gpscoord convert borholur.tsv                      # ISN93 table (header says so) → lon/lat
  gpscoord nearest borholur.tsv -k 2                 # the 2 nearest stations.cfg stations
  gpscoord nearest borholur.tsv --data-before 20231110 --data-after 20261001 --join LISK+LISF
                                                     # nearest with data from before the dike, still running
  gpscoord series SENG                               # daily ISN93 + lat/lon/h to the screen
  gpscoord series --stations-from naesta.csv --add dup=h:first5 -o out/

  gpscoord series --help                             # the column language, with examples
"""

SERIES_EPILOG = """\
columns (--columns / --add): a comma list of [NAME=]KEY[@FRAME][:REL]
  date datetime yearf mjd      time
  x y z                        ECEF of the .pos frame [m]
  lat lon h                    GRS80 at the observation epoch [deg, deg, m]
  east@F north@F               projected frame F: isn93, isn2004, isn2016 [m]
  dn de du                     local N/E/U from --ref-pos [--units]
  sx sy sz sn se su            formal σ [--units]
  kept                         1 = the view keeps the day (with --with-dropped)
  :REL subtracts one value of the column, from the kept days:
       first5 (mean of the first 5; any N) first last mean median
       2024-01-01..2024-01-31 (mean over those days) or a number
  NAME= names a column; a REL column is called d<key> by default.
  presets stand for their columns anywhere: isn93 (default) =
       date,east@isn93,north@isn93,lat,lon,h ; isn2004 ; isn2016 ;
       llh = date,lat,lon,h ; xyz = date,x,y,z,sx,sy,sz ;
       neu = date,yearf,dn,de,du,sn,se,su

east@isn93/north@isn93 are the station's position ON THE DAY, projected
(default, --epoch-shift none): where it is now in ISN93 coordinates. Old
ISN93 coordinates (surveyed points, maps) have drifted with the plate since
1993.6 (~0.5 m by 2026). --epoch-shift plate instead takes each day back to
the frame's epoch (ISN93 1993.6, ISN2004 2004.6, ISN2016 2016.5) on the
station's plate (platefile, the ITRF plate-motion model), to compare with
such old coordinates. The file header says which was done. The shift since
a date is a :REL column: dE=east@isn93:2023-05-01..2023-05-31.

examples:
  gpscoord series SENG                                    # isn93 preset, screen
  gpscoord series SENG SKSH --add dup=h:first5 -o out/    # out/SENG.csv, out/SKSH.csv
  gpscoord series SENG --columns date,yearf,du,su --ref-pos first5 --start 20231101
  gpscoord series SENG --view raw --with-dropped --add dup=h:first5   # every row, kept 0/1
  gpscoord series SENG --add dE=east@isn93:2023-05-01..2023-05-31,dN=north@isn93:2023-05-01..2023-05-31,dU=h:2023-05-01..2023-05-31
                                                          # the shift since May 2023 [mm]
  gpscoord series --stations-from ~/gps-data/data/reykjanes-borholur/borholur-naesta-gnss.csv \\
      --add dup=h:first5 -o ~/gps-data/data/reykjanes-borholur/
"""


def _box(value: str) -> tuple[float, float, float, float]:
    from geo_dataread.products.cli import _box as box

    return box(value)


def build_parser() -> argparse.ArgumentParser:
    from geofunc.coords import frame_names

    from geo_dataread.pos import SESSIONS
    from geo_dataread.products.common import VIEWS

    raw = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(
        prog="gpscoord", description=__doc__.split("\n\n")[0], formatter_class=raw,
        epilog=TOP_EPILOG,
    )  # fmt: skip
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    frames = ", ".join(frame_names())

    cv = sub.add_parser("convert", help="a point table in another frame",
                        epilog=f"frames: {frames}")  # fmt: skip
    _table_args(cv)
    cv.add_argument("--to", dest="dst", default="wgs84", metavar="FRAME",
                    help="output frame (default wgs84: lon, lat)")  # fmt: skip

    nb = sub.add_parser("nearest", help="the stations nearest each point",
                        epilog=f"frames: {frames}")  # fmt: skip
    _table_args(nb)
    nb.add_argument("--candidates", default=None, metavar="FILE",
                    help="the points to search (default: every stations.cfg station)")  # fmt: skip
    nb.add_argument("-k", type=int, default=1, help="how many per point (default 1)")
    nb.add_argument("--max-km", type=float, default=None,
                    help="leave out candidates farther than this")  # fmt: skip
    nb.add_argument("--data-before", metavar="DATE",
                    help="only stations whose .pos series starts before DATE "
                    "(e.g. 20231110: data before the dike)")  # fmt: skip
    nb.add_argument("--data-after", metavar="DATE",
                    help="only stations with .pos data on or after DATE (still running)")  # fmt: skip
    nb.add_argument("--join", action="append", default=[], metavar="A+B",
                    help="one station moved: A then B count as one candidate 'A+B' "
                    "(its data span the union, its place the last one's); repeatable, "
                    "e.g. --join LISK+LISF")  # fmt: skip
    nb.add_argument("--frame", default="itrf14", help="the .pos product of --data-*")
    nb.add_argument("--pos-dir", metavar="DIR", help="the .pos directory")

    se = sub.add_parser(
        "series", help="daily coordinates of stations, the columns you choose",
        formatter_class=raw, epilog=SERIES_EPILOG,
        description="Daily station coordinates from the .pos series, one CSV per "
        "station, with the columns you choose.",
    )  # fmt: skip
    g = se.add_argument_group("stations")
    g.add_argument("stations", nargs="*",
                   help="station markers, 'all', or A+B for a moved station (one "
                   "series, B tied to A)")  # fmt: skip
    g.add_argument("--area", help="the members of a station_areas.yaml area")
    g.add_argument("--region", type=_box, metavar="W/E/S/N",
                   help="every station in the box, or narrows the named ones")  # fmt: skip
    g.add_argument("--stations-from", metavar="FILE[:COLUMN]",
                   help="the stations named in a CSV column (default column 'nearest', "
                   "as gpscoord nearest / gpsplot --nearest-csv write)")  # fmt: skip
    g = se.add_argument_group("period (default: the whole series)")
    g.add_argument("--start", "--since", dest="start", metavar="DATE")
    g.add_argument("-e", "--end", metavar="DATE")
    g.add_argument("--days", "--last", dest="days", type=int, metavar="N")
    g.add_argument("--period", metavar="A:B", help="dates and/or events.yaml names")
    g = se.add_argument_group("series")
    g.add_argument("--frame", default="itrf14",
                   help="the .pos product: itrf14 (default, IGb14), itrf08, itrf20")  # fmt: skip
    g.add_argument("--session", default="24h", choices=tuple(SESSIONS))
    g.add_argument("--view", choices=VIEWS, default="cleaned",
                   help="cleaned (default: outliers, icing, snow, network days out), "
                   "declared (known blunders out) or raw")  # fmt: skip
    g.add_argument("-u", "--uncert", type=float, default=10.0, metavar="MM",
                   help="drop solutions with a formal σ above MM (default 10)")  # fmt: skip
    g.add_argument("--tie-days", type=int, default=5, metavar="N",
                   help="a moved station A+B: tie B to A on N kept days each side (default 5)")  # fmt: skip
    g.add_argument("--with-dropped", action="store_true",
                   help="write every day, with a kept 0/1 column")  # fmt: skip
    g.add_argument("--pos-dir", metavar="DIR", help="the .pos directory")
    g = se.add_argument_group("columns")
    g.add_argument("--columns", default="isn93", metavar="LIST",
                   help="the columns, see below (default isn93)")  # fmt: skip
    g.add_argument("--add", default=None, metavar="LIST",
                   help="more columns after --columns, e.g. dup=h:first5")  # fmt: skip
    g.add_argument("--ref-pos", default="wmean", metavar="HOW",
                   help="reference of dn de du: wmean (default), mean, median, first, "
                   "firstN, or X,Y,Z")  # fmt: skip
    g.add_argument("--epoch-shift", choices=("none", "plate"), default="none",
                   help="ISN columns: the position on the day (none, default) or taken "
                   "back to the frame's epoch on the station's plate (plate)")  # fmt: skip

    g.add_argument("--units", choices=("mm", "m"), default="mm",
                   help="differences and σ in mm (default) or m")  # fmt: skip
    g.add_argument("--decimals", type=int, default=None,
                   help="decimals of every number (default per unit)")  # fmt: skip
    g = se.add_argument_group("output")
    g.add_argument("-o", "--out", metavar="FILE|DIR/|PATTERN",
                   help="FILE (one station), DIR/ (DIR/STA.csv) or a pattern with "
                   "{station} (default: the screen)")  # fmt: skip
    g.add_argument("--no-header", action="store_true", help="no # header lines")
    g.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")

    sub.add_parser(
        "at", help="where stations are at an epoch (gpsprod coords)", add_help=False
    )
    return p


def _table_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="the table (- = stdin)")
    p.add_argument("--from", dest="src", default=None, metavar="FRAME",
                   help="the table's frame (default: its header's, else wgs84)")  # fmt: skip
    p.add_argument("--x", default=None, help="x column (header or 1-based index)")
    p.add_argument("--y", default=None, help="y column (header or 1-based index)")
    p.add_argument("--name", default=None, help="name column")
    p.add_argument("--ncols", type=int, default=None,
                   help="cells per row of a table pasted one cell per line")  # fmt: skip
    p.add_argument("-o", "--output", default=None, help="output file (default stdout)")


def _geofunc_argv(args: argparse.Namespace) -> list[str]:
    out = [args.file]
    for opt in ("src", "x", "y", "name", "ncols", "output"):
        v = getattr(args, opt)
        if v is not None:
            out += [{"src": "--from", "output": "-o"}.get(opt, f"--{opt}"), str(v)]
    return out


def _stations_cfg() -> list[tuple[str, float, float]]:
    from gps_parser import ConfigParser

    cfg = ConfigParser()
    out = []
    for sta in cfg.getStationInfo():
        info = dict(dict(cfg.getStationInfo(sta))["station"])
        try:
            out.append((str(sta), float(info["longitude"]), float(info["latitude"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _nearest(args: argparse.Namespace) -> int:
    from geofunc import cli as gcli

    if args.candidates is not None:
        argv = ["nearest", *_geofunc_argv(args), "--candidates", args.candidates,
                "-k", str(args.k)]  # fmt: skip
        if args.max_km is not None:
            argv += ["--max-km", str(args.max_km)]
        return gcli.main(argv)
    from geofunc import points

    pts = points.to_frame(gcli._read(args), "wgs84")
    cands = _stations_cfg()
    spans = _spans(args, [c[0] for c in cands]) if _needs_spans(args) else {}
    cands = _joined(cands, args.join, spans)
    if args.data_before or args.data_after:
        cands = [c for c in cands if _in_span(spans.get(c[0]), args)]
    max_m = args.max_km * 1000.0 if args.max_km is not None else None
    with gcli._out(args.output) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["name", "lon", "lat", "rank", "nearest", "distance_km",
                    "azimuth_deg", "first_day", "last_day"])  # fmt: skip
        for pt in pts:
            for rank, nb in enumerate(
                points.nearest((pt.x, pt.y), cands, args.k, max_distance=max_m), 1
            ):
                a, b = spans.get(nb.id) or ("", "")
                w.writerow([pt.name, f"{pt.x:.7f}", f"{pt.y:.7f}", rank, nb.id,
                            f"{nb.distance / 1000:.3f}", f"{nb.azimuth:.1f}", a, b])  # fmt: skip
    return 0


def _needs_spans(args: argparse.Namespace) -> bool:
    return bool(args.data_before or args.data_after or args.join)


def pos_span(path: Path) -> tuple[str, str] | None:
    """First and last day (YYYY-MM-DD) of a ``.pos`` file, from its header."""
    first = last = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for _, line in zip(range(12), fh, strict=False):
                if line.startswith("First Epoch"):
                    first = line.split(":", 1)[1].split()[0]
                elif line.startswith("Last Epoch"):
                    last = line.split(":", 1)[1].split()[0]
    except OSError:
        return None
    if not first or not last:
        return None
    return tuple(f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in (first, last))  # type: ignore[return-value]


def _spans(
    args: argparse.Namespace, stations: Sequence[str]
) -> dict[str, tuple[str, str]]:
    """``{station: (first_day, last_day)}`` of the stations' ``.pos`` series."""
    from geo_dataread import pos
    from geo_dataread.products.loader import Source

    src = Source(frame=args.frame, pos_dir=args.pos_dir)
    folder = src.directory()
    out = {}
    for sta in stations:
        try:
            span = pos_span(Path(pos.pos_path(sta, folder, src.pos_id)))
        except FileNotFoundError:  # no series of this product: no span
            continue
        if span is not None:
            out[sta] = span
    return out


def _joined(
    cands: list[tuple[str, float, float]],
    joins: Sequence[str],
    spans: dict[str, tuple[str, str]],
) -> list[tuple[str, float, float]]:
    """Candidates with each ``A+B`` (a moved station) as one: placed at the
    last member that has a place, its data span the union of the members'."""
    by = {c[0]: c for c in cands}
    out = list(cands)
    for spec in joins:
        members = [m.strip().upper() for m in spec.split("+") if m.strip()]
        known = [m for m in members if m in by]
        if len(members) < 2 or not known:
            raise SystemExit(
                f"gpscoord: --join {spec}: give A+B of stations.cfg stations"
            )
        name = "+".join(members)
        last = by[known[-1]]
        out = [c for c in out if c[0] not in members] + [(name, last[1], last[2])]
        member_spans = [spans[m] for m in members if m in spans]
        if member_spans:
            spans[name] = (
                min(a for a, _ in member_spans),
                max(b for _, b in member_spans),
            )
    return out


def _in_span(span: tuple[str, str] | None, args: argparse.Namespace) -> bool:
    if span is None:
        return False

    def iso(d: str) -> str:
        d = d.replace("-", "")
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    if args.data_before and not span[0] < iso(args.data_before):
        return False
    return not (args.data_after and not span[1] >= iso(args.data_after))


def stations_from(spec: str) -> list[str]:
    """The stations in a CSV column (``FILE[:COLUMN]``, default ``nearest``),
    first appearance order, each once; ``A+B`` (a moved station) stays one."""
    path, sep, col = spec.rpartition(":")
    if not sep or not path or Path(spec).expanduser().exists():
        path, col = spec, "nearest"
    rows = [
        ln
        for ln in Path(path).expanduser().read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    reader = csv.DictReader(rows)
    if col not in (reader.fieldnames or ()):
        raise SystemExit(
            f"gpscoord: {path} has no column {col!r} ({reader.fieldnames})"
        )
    out: list[str] = []
    for r in reader:
        # 'A+B' (a moved station, gpscoord nearest --join) stays one name:
        # series writes it as one tied series
        sta = (r[col] or "").strip().upper()
        if sta and sta not in out:
            out.append(sta)
    return out


def _velocity(sta: str, frame: str, xyz: Any) -> tuple[Any, str]:
    """The station's plate velocity (ECEF m/yr) at its mean position, and
    a note; (None, why) without a plate."""
    from geofunc import plates

    from geo_dataread.station_catalog import station_plate

    code = station_plate(sta)
    if code is None:
        return None, f"no plate for {sta} in the platefile: no epoch shift"
    pos = np.nanmean(np.asarray(xyz, float), axis=0)
    v = plates.plate_velocity(code, pos, frame=frame).d
    return np.asarray(v, float), f"plate {code} ({frame} plate-motion model)"


def _target(out: str | None, sta: str) -> Path | None:
    if out is None:
        return None
    if "{station}" in out:
        return Path(out.format(station=sta)).expanduser()
    p = Path(out).expanduser()
    if out.endswith("/") or p.is_dir():
        return p / f"{sta}.csv"
    return p


def _series(args: argparse.Namespace) -> int:
    from geofunc import columns as gcol

    from geo_dataread.products.cli import select
    from geo_dataread.products.common import software
    from geo_dataread.products.loader import Source
    from geo_dataread.products.period import resolve_period
    from geo_dataread.products.positions import load_positions

    try:
        specs = gcol.parse_columns(",".join(x for x in (args.columns, args.add) if x))
        if args.with_dropped and not any(c.key == "kept" for c in specs):
            specs.append(gcol.ColumnSpec("kept"))
    except ValueError as err:
        raise SystemExit(f"gpscoord: {err}") from None
    src = Source(frame=args.frame, session=args.session, view=args.view, ref="plate",
                 uncert=args.uncert, pos_dir=args.pos_dir)  # fmt: skip
    names = list(args.stations)
    if args.stations_from:
        names += stations_from(args.stations_from)
    stations = select(names, area=args.area, region=args.region, source=src)
    if not stations:
        raise SystemExit(
            "gpscoord: no stations (markers, --stations-from, --area, --region)"
        )
    if (
        len(stations) > 1
        and args.out
        and not (
            "{station}" in args.out or args.out.endswith("/") or Path(args.out).is_dir()
        )
    ):
        raise SystemExit(
            "gpscoord: several stations: give -o DIR/ or a {station} pattern"
        )
    if args.period or args.start or args.days:
        try:
            per = resolve_period(
                args.period, start=args.start, end=args.end, days=args.days
            )
        except ValueError as err:
            raise SystemExit(f"gpscoord: {err}") from None
        bounds, label = per.bounds, f"{per.first_day} to {per.last_day}"
    else:
        bounds, label = (1990.0, 2100.0), "the whole series"
    ref_pos: Any = args.ref_pos
    if "," in ref_pos:
        ref_pos = [float(v) for v in ref_pos.split(",")]
    doc = src.params_doc()
    many_screen = args.out is None and len(stations) > 1
    status = 0
    for sta in stations:
        if not args.quiet:
            print(f"  {sta}", file=sys.stderr, end="\r")
        try:
            if "+" in sta:
                pos, tied, marks, ties = load_joined(
                    sta, bounds, src, doc=doc, tie_days=args.tie_days
                )
            else:
                pos = load_positions(sta, bounds[0], bounds[1], src, doc=doc)
                tied, marks, ties = None, None, []
        except (OSError, ValueError) as err:
            print(f"gpscoord: {sta}: {err}", file=sys.stderr)
            status = 1
            continue
        n_kept = int(np.count_nonzero(pos.kept))
        if n_kept == 0 and not args.with_dropped:
            from geo_dataread.products.positions import available

            su = float(np.median(np.sqrt(pos.cov[:, 2, 2]))) * 1000  # rough: Z
            print(f"gpscoord: {sta}: no day passes (view {args.view}, sigma <= "
                  f"{args.uncert:g} mm; median sigma Z {su:.0f} mm); "
                  f"{available(pos.t, np.array(pos.dates, dtype='datetime64[D]'))}",
                  file=sys.stderr)  # fmt: skip
            status = 1
            continue
        velocity, plate_note = (None, "none: projected positions are those of the day")
        if args.epoch_shift == "plate":
            velocity, plate_note = _velocity(sta.split("+")[-1], pos.frame, pos.xyz)
        try:
            table = _table(pos, tied, specs, velocity, ref_pos, args)
        except ValueError as err:
            print(f"gpscoord: {sta}: {err}", file=sys.stderr)
            status = 1
            continue
        if marks is not None:  # which mark each row is from, after the date
            at = (
                1
                if table.columns and table.columns[0].name in ("date", "datetime")
                else 0
            )
            mark = gcol.Column(
                "mark", "", list(marks), None, "the station mark of the row"
            )
            table = gcol.Table(table.columns[:at] + [mark] + table.columns[at:],
                               table.ref_pos, table.notes)  # fmt: skip
        if not args.with_dropped:
            keep = list(pos.kept)
            table = gcol.Table(
                [gcol.Column(c.name, c.unit, _subset(c.values, keep), c.decimals,
                             c.definition) for c in table.columns],
                table.ref_pos, table.notes,
            )  # fmt: skip
        if many_screen:
            table = gcol.Table(
                [gcol.Column("station", "", [sta] * len(table.columns[0].values), None,
                             "station marker"), *table.columns],
                table.ref_pos, table.notes,
            )  # fmt: skip
        head = [] if args.no_header else [
            "gpscoord series", f"station {sta}",
            f"source {src.pos_id} ({pos.frame}), view {args.view}, sigma <= {args.uncert:g} mm",
            f"period {label}; epochs {len(pos.t)}, kept {n_kept}"
            + ("" if args.with_dropped else " (written)"),
            f"epoch shift: {plate_note}",
            *ties,
            "software " + " ".join(f"{k} {v}" for k, v in software().items()),
            f"written {dt.datetime.now().isoformat(timespec='seconds')}",
        ]  # fmt: skip
        if table.ref_pos is not None and not args.no_header:
            head.append(
                "ref_pos ({}) {:.4f} {:.4f} {:.4f}".format(args.ref_pos, *table.ref_pos)
            )
        target = _target(args.out, sta)
        if target is None:
            _write(table, sys.stdout, head, args.no_header)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as fh:
            _write(table, fh, head, args.no_header)
        if not args.quiet:
            print(
                f"gpscoord: {sta}: {n_kept}/{len(pos.t)} days → {target}",
                file=sys.stderr,
            )
    return status


def load_joined(
    name: str,
    bounds: tuple[float, float],
    src: Any,
    *,
    doc: Any = None,
    tie_days: int = 5,
) -> tuple[Any, Any, list[str], list[str]]:
    """A moved station ``A+B…`` as one series: the members' positions in
    time order, each later mark TIED to the one before it (ECEF offset = the
    mean of the earlier mark's last ``tie_days`` kept days less the later
    mark's first ``tie_days``, i.e. no motion between them is assumed).

    Returns ``(positions, tied_xyz, marks, header_lines)``: the positions
    as measured (absolute columns), the tied ECEF (difference columns),
    the mark of each row and the ties for the header.

    Raises:
        ValueError: no member has data in the period, or a member has no
            kept day to tie on.
    """
    import dataclasses

    from geofunc import local

    from geo_dataread.products.positions import load_positions

    parts = []
    for m in name.split("+"):
        try:
            parts.append(load_positions(m, bounds[0], bounds[1], src, doc=doc))
        except ValueError:
            continue  # this mark has nothing in the period
    if not parts:
        raise ValueError(f"none of {name.replace('+', ', ')} has data in the period")
    parts.sort(key=lambda p: float(p.t[0]))
    offset = np.zeros(3)
    tied_parts: list[Any] = []
    lines: list[str] = []
    for k, p in enumerate(parts):
        if k:
            prev, prev_tied = parts[k - 1], tied_parts[k - 1]
            a = prev_tied[prev.kept][-tie_days:]
            b = p.xyz[p.kept][:tie_days]
            if len(a) == 0 or len(b) == 0:
                raise ValueError(f"no kept day to tie {prev.station} and {p.station}")
            offset = a.mean(axis=0) - b.mean(axis=0)
            neu = local.displacements(b.mean(axis=0)[None], a.mean(axis=0)).d[0]
            gap = (p.dates[int(np.argmax(p.kept))]
                   - prev.dates[len(prev.kept) - 1 - int(np.argmax(prev.kept[::-1]))])  # fmt: skip
            lines.append(
                f"tie {prev.station} -> {p.station}: {p.station} moved by the mean of "
                f"{prev.station}'s last {len(a)} kept days less its own first {len(b)} "
                f"({p.station} is N {neu[0]:+.3f} E {neu[1]:+.3f} U {neu[2]:+.3f} m from "
                f"{prev.station}; "
                f"{gap.days} days between the last and first kept day, no motion "
                "assumed across them); difference columns are tied, absolute "
                "columns are each mark's own position"
            )
        tied_parts.append(p.xyz + offset)
    pos = dataclasses.replace(
        parts[-1], station=name,
        t=np.concatenate([p.t for p in parts]),
        dates=[d for p in parts for d in p.dates],
        xyz=np.concatenate([p.xyz for p in parts]),
        cov=np.concatenate([p.cov for p in parts]),
        kept=np.concatenate([p.kept for p in parts]),
    )  # fmt: skip
    marks = [p.station for p in parts for _ in p.t]
    return pos, np.concatenate(tied_parts), marks, lines


def _table(
    pos: Any,
    tied: Any,
    specs: Sequence[Any],
    velocity: Any,
    ref_pos: Any,
    args: argparse.Namespace,
) -> Any:
    """The table of one station; for a tied series the difference columns
    (:REL, dn de du) from the tied positions, the rest as measured."""
    from geofunc import columns as gcol

    def build(xyz: Any) -> Any:
        series = gcol.PositionSeries(pos.t, xyz, frame=pos.frame, cov=pos.cov,
                                     dates=pos.dates, kept=pos.kept, velocity=velocity)  # fmt: skip
        return gcol.build_table(series, specs, ref_pos=ref_pos, units=args.units,
                                decimals=args.decimals, epoch_shift=velocity is not None)  # fmt: skip

    real = build(pos.xyz)
    if tied is None:
        return real
    joined = build(tied)
    diff = [c.rel is not None or c.key in gcol.NEU_KEYS for c in specs]
    cols = [
        t if d else r
        for r, t, d in zip(real.columns, joined.columns, diff, strict=True)
    ]
    notes = list(dict.fromkeys([*real.notes, *joined.notes]))
    return gcol.Table(cols, joined.ref_pos, notes)


def _subset(values: Any, keep: Sequence[bool]) -> Any:
    if isinstance(values, list):
        return [v for v, k in zip(values, keep, strict=True) if k]
    return np.asarray(values)[np.asarray(keep, bool)]


def _write(table: Any, fh: TextIO, head: list[str], no_header: bool) -> None:
    from geofunc import columns as gcol

    if no_header:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([c.name for c in table.columns])
        w.writerows(table.rows())
    else:
        gcol.write_table(table, fh, header=head)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["at"]:
        from geo_dataread.products.cli import main as gpsprod

        return gpsprod(["coords", *argv[1:]])
    args = build_parser().parse_args(argv)
    if args.command == "convert":
        from geofunc.cli import main as gmain

        argv2 = ["points", *_geofunc_argv(args), "--to", args.dst]
        return gmain(argv2)
    if args.command == "nearest":
        return _nearest(args)
    return _series(args)
