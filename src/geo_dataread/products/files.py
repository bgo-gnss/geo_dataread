"""Writers of the gpsprod record sets; the format comes from the file name.

- ``.csv`` — ``#`` provenance header, then one row per record (refused ones
  too, with the reason); fixed decimals, station order, no timestamps, so
  a re-run on the same data writes the same bytes.
- ``.jsonl`` — one record per line, every field (``to_dict``).
- ``-`` — the screen table.

``.gmt``, ``.gbis.txt`` and ``.xlsx`` arrive in step 7 of PLAN-gpsprod.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from geo_dataread.products.common import (
    DisplacementRecord,
    RecordSet,
    provenance_header,
)

DISPLACEMENT_COLUMNS = (
    "station", "lon", "lat", "start", "end",
    "dN", "dE", "dU", "sN", "sE", "sU", "units",
    "n_start", "n_end", "method", "ref", "view", "frame", "session",
    "relative_to", "flags", "refused",
)  # fmt: skip
FORMATS = (".csv", ".jsonl")


def _fmt(v: float, decimals: int) -> str:
    return "" if not math.isfinite(v) else f"{v:.{decimals}f}"


def _ends_n(rec: DisplacementRecord, key: str) -> int | str:
    if not rec.ends:
        return ""
    return min(int(c["n"]) for c in rec.ends[key].values())


def rows(
    records: Sequence[DisplacementRecord],
) -> tuple[tuple[str, ...], list[list[Any]]]:
    """(columns, rows) of a displacement set, values formatted for text."""
    out = []
    for r in records:
        dec = 3 if r.units == "mm" else 6
        out.append(
            [
                r.station,
                _fmt(r.lon, 6),
                _fmt(r.lat, 6),
                r.period_dates[0],
                r.period_dates[1],
                *(_fmt(v, dec) for v in r.displacement),
                *(_fmt(v, dec) for v in r.sigma),
                r.units,
                _ends_n(r, "start"),
                _ends_n(r, "end"),
                r.method,
                r.ref,
                r.view,
                r.frame,
                r.session,
                r.relative_to or "",
                ";".join(r.flags),
                r.refused or "",
            ]  # fmt: skip
        )
    return DISPLACEMENT_COLUMNS, out


def _csv_cell(v: Any) -> str:
    text = str(v)
    return f'"{text}"' if any(c in text for c in ',"\n') else text


def write_csv(rs: RecordSet[Any], path: Path, records: Sequence[Any]) -> None:
    cols, data = rows(records)
    lines = [f"# {line}" for line in provenance_header(rs)]
    lines.append(",".join(cols))
    lines += [",".join(_csv_cell(v) for v in row) for row in data]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Any]) -> None:
    text = "".join(
        json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        for r in records
    )
    path.write_text(text, encoding="utf-8")


def table(records: Sequence[DisplacementRecord]) -> str:
    """The screen table: one aligned line per station."""
    cols, data = rows(records)
    keep = (
        "station",
        "dN",
        "dE",
        "dU",
        "sN",
        "sE",
        "sU",
        "n_start",
        "n_end",
        "flags",
        "refused",
    )
    idx = [cols.index(c) for c in keep]
    grid = [[str(c) for c in keep]] + [[str(row[i]) for i in idx] for row in data]
    widths = [max(len(r[j]) for r in grid) for j in range(len(keep))]
    right = {"dN", "dE", "dU", "sN", "sE", "sU", "n_start", "n_end"}
    lines = []
    for r in grid:
        cells = [
            v.rjust(w) if keep[j] in right else v.ljust(w)
            for j, (v, w) in enumerate(zip(r, widths, strict=True))
        ]
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def write(rs: RecordSet[Any], path: str, *, units: str = "mm") -> str:
    """Write ``rs`` to ``path`` (format by extension); returns the path."""
    records = [r.scaled(units) for r in rs.records]
    target = Path(path).expanduser()
    suffix = target.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(f"{path}: unknown format {suffix!r}; one of {FORMATS}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        write_csv(rs, target, records)
    else:
        write_jsonl(target, records)
    return str(target)
