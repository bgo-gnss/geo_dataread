"""A minimal .xlsx writer (standard library only): sheets of rows.

An .xlsx file is a zip of a few SpreadsheetML parts (ECMA-376). This writes
the smallest set Excel, LibreOffice and pandas open: one worksheet per
``(name, rows)``, numbers as numbers, everything else as inline text, the
first row bold-free (a header row is just the first row). Byte-stable: the
zip entries carry a fixed timestamp.
"""

from __future__ import annotations

import math
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

_FIXED = (2000, 1, 1, 0, 0, 0)
_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
{sheets}
</Types>"""
_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
_WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>{sheets}</sheets>
</workbook>"""
_WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{rels}
</Relationships>"""
_SHEET = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>{rows}</sheetData>
</worksheet>"""


def _col(i: int) -> str:
    """0 → A, 25 → Z, 26 → AA."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(65 + r) + out
    return out


def _number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str) and v.strip():
        try:
            f = float(v)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _cell(ref: str, v: Any) -> str:
    if v is None or v == "":
        return ""
    num = _number(v)
    if num is not None:
        text = repr(int(num)) if num.is_integer() and abs(num) < 1e15 else repr(num)
        return f'<c r="{ref}"><v>{text}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'


def _sheet_xml(rows: Sequence[Sequence[Any]]) -> str:
    out = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(_cell(f"{_col(c)}{r}", v) for c, v in enumerate(row))
        out.append(f'<row r="{r}">{cells}</row>')
    return _SHEET.format(rows="".join(out))


def write_xlsx(
    path: str | Path, sheets: Sequence[tuple[str, Sequence[Sequence[Any]]]]
) -> Path:
    """Write ``sheets`` (``(name, rows)``, names ≤ 31 characters) to ``path``."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    names = [n[:31] for n, _ in sheets]
    if len(set(names)) != len(names):
        raise ValueError(f"sheet names must differ: {names}")
    parts = {
        "[Content_Types].xml": _CONTENT_TYPES.format(sheets="\n".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(sheets) + 1))),
        "_rels/.rels": _ROOT_RELS,
        "xl/workbook.xml": _WORKBOOK.format(sheets="".join(
            f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, n in enumerate(names, start=1))),
        "xl/_rels/workbook.xml.rels": _WORKBOOK_RELS.format(rels="\n".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheets) + 1))),
    }  # fmt: skip
    for i, (_name, rows) in enumerate(sheets, start=1):
        parts[f"xl/worksheets/sheet{i}.xml"] = _sheet_xml(rows)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in parts.items():
            info = zipfile.ZipInfo(name, date_time=_FIXED)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text.encode("utf-8"))
    return target
