"""Recorded API responses parse and are non-trivial (PLAN-seismicity.md T9).

These tests guard the recordings only, so that a broken re-record fails fast.
The readers (T10-T13) test their own mapping onto the schema.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
API_DIRS = ("quakes", "fdsn", "skjalftalisa")
MACHINE_AUTHOR = re.compile(r"^(SIL|[A-Za-z0-9_]+@seiscomp-p01|analyst-\d+)?$")
MAX_FILE_BYTES = 50_000
MAX_TOTAL_BYTES = 400_000


def _manifest(directory: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = json.loads(
        (FIXTURES / directory / "manifest.json").read_text(encoding="utf-8")
    )
    return entries


ENTRIES = [(d, e) for d in API_DIRS for e in _manifest(d)]
IDS = [f"{d}/{e['file']}" for d, e in ENTRIES]


def _text(directory: str, entry: dict[str, Any]) -> str:
    return (FIXTURES / directory / str(entry["file"])).read_text(encoding="utf-8")


def _parse(fmt: str, text: str) -> Any:
    if fmt in ("json", "geojson"):
        return json.loads(text)
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(text)))
    if fmt == "xml":
        return ET.fromstring(text)
    return text


def _events(fmt: str, parsed: Any) -> int:
    if fmt == "geojson":
        return len(parsed["features"] or [])
    if fmt == "csv":
        return len(parsed)
    if fmt == "xml":
        return sum(1 for el in parsed.iter() if el.tag.endswith("}event"))
    lines = [ln for ln in str(parsed).splitlines() if ln.strip()]
    return len(lines) - 1 if lines and lines[0].startswith("#EventID") else 0


@pytest.mark.parametrize("directory", API_DIRS)
def test_manifest_matches_files(directory: str) -> None:
    listed = {e["file"] for e in _manifest(directory)}
    on_disk = {
        p.name
        for p in (FIXTURES / directory).iterdir()
        if p.name not in ("manifest.json", "README.md")
    }
    assert listed == on_disk
    assert (FIXTURES / directory / "README.md").is_file()


def test_sizes_within_budget() -> None:
    sizes = [(FIXTURES / d / e["file"]).stat().st_size for d, e in ENTRIES]
    assert max(sizes) < MAX_FILE_BYTES
    assert sum(sizes) < MAX_TOTAL_BYTES


@pytest.mark.parametrize(("directory", "entry"), ENTRIES, ids=IDS)
def test_fixture_parses(directory: str, entry: dict[str, Any]) -> None:
    text = _text(directory, entry)
    assert len(text.encode("utf-8")) == entry["bytes"]
    parsed = _parse(entry["format"], text)
    kind = entry["kind"]
    if kind == "error":
        assert entry["status"] >= 400
        assert len(text) > 10
        return
    assert entry["status"] == 200
    if kind in ("list", "sample"):
        n = _events(entry["format"], parsed)
        assert n > 0
        assert n == entry["events"]
    elif kind == "empty":
        assert _events(entry["format"], parsed) == 0
    elif kind == "detail":
        assert len(text) > 200
    elif kind == "count":
        value = parsed if isinstance(parsed, int) else parsed["quake_count"]
        assert value > 0


def _authors(fmt: str, text: str) -> set[str]:
    if fmt in ("json", "geojson"):
        return set(re.findall(r'"author"\s*:\s*"([^"]*)"', text))
    if fmt == "xml":
        return set(re.findall(r"<author>([^<]*)</author>", text))
    if fmt == "text" and text.startswith("#EventID"):
        rows = csv.DictReader(io.StringIO(text.lstrip("#")), delimiter="|")
        return {v for r in rows for v in (r["Author"], r["MagAuthor"])}
    return set()


@pytest.mark.parametrize(("directory", "entry"), ENTRIES, ids=IDS)
def test_authors_are_machines_or_redacted(
    directory: str, entry: dict[str, Any]
) -> None:
    authors = _authors(entry["format"], _text(directory, entry))
    assert all(MACHINE_AUTHOR.match(a) for a in authors), authors


def test_legacy_rows_share_event_id() -> None:
    """The legacy list carries reviewed and automatic rows under one id."""
    doc = json.loads(
        (FIXTURES / "skjalftalisa" / "internal_20240822T00.json").read_text()
    )
    by_id: dict[int, set[str]] = {}
    for f in doc["features"]:
        p = f["properties"]
        by_id.setdefault(p["event_id"], set()).add(p["originating_system"])
    assert by_id[1352381] == {"SIL picks", "SIL aut.mag"}
    detail = json.loads(
        (FIXTURES / "skjalftalisa" / "internal_detail_1352381.json").read_text()
    )
    assert detail["slungapath"] == "/2024/aug/22/00:05:00/10:05:430"
    sp3 = json.loads(detail["sp3"])
    assert (sp3["fpa_strike_AR"], sp3["fpa_dip_AR"], sp3["fpa_slip_AR"]) == (
        341.0,
        74.0,
        170.0,
    )
