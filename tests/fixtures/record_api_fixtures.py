"""Re-record the seismicity API fixtures (PLAN-seismicity.md T9).

Run by hand, inside the IMO network, never from the test suite::

    python tests/fixtures/record_api_fixtures.py            # all three APIs
    python tests/fixtures/record_api_fixtures.py fdsn       # one directory

It writes ``quakes/``, ``fdsn/`` and ``skjalftalisa/`` next to this file, each
with a ``manifest.json`` (URL, HTTP status, content type, selected headers,
event count, redaction count). ``tests/test_api_fixtures.py`` reads the
manifests. The READMEs are written by hand: update them after a re-record.

Rules kept here: read-only GETs, at least ``PAUSE_S`` between requests, a
generic User-Agent, no personal identifier in any request. Response bodies are
kept byte-exact except for the redaction of analyst logins (see
:func:`redact`), which replaces each non-machine ``author`` value with
``analyst-N``, numbered in order of first appearance in this run. The mapping
is never written to disk.

Endpoints that list or query users (``/internal/lokimp/*``,
``/internal/picks?user=``) are deliberately never called.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
USER_AGENT = "geo_dataread-fixture-recorder"
PAUSE_S = 1.2
TIMEOUT_S = 60

QUAKES_INT = "https://vi-api.vedur.is/quakes/internal/events"
QUAKES_PUB = "https://api.vedur.is/quakes/events"
SKJ_INT = "https://vi-api.vedur.is/skjalftalisa/internal/quakes"
SKJ_PUB = "https://api.vedur.is/skjalftalisa/quakes"
FDSN = "http://seiscomp-p01.vedur.is:8080/fdsnws/event/1/query"

# Window A: 2024-08-22 00-01 UTC, the T0 SIL fixture day (Sundhnúkur onset).
# Window B: 2026-09-10, the whole day (small), the T0 post-February-2026 day.
A_Z = "start_time=2024-08-22T00:00:00Z&end_time=2024-08-22T01:00:00Z"
B_Z = "start_time=2026-09-10T00:00:00Z&end_time=2026-09-11T00:00:00Z"
EMPTY_Z = "start_time=2024-08-22T00:00:00Z&end_time=2024-08-22T00:02:00Z"
A_F = "starttime=2024-08-22T00:00:00&endtime=2024-08-22T01:00:00"
B_F = "starttime=2026-09-10T00:00:00&endtime=2026-09-11T00:00:00"
LIM = "size_min=-3&size_max=10&depth_min=0&depth_max=50"

# Authors that are programs, not people: kept as they are.
MACHINE_AUTHOR = re.compile(r"^(SIL|[A-Za-z0-9_]+@seiscomp-p01)?$")
PERSON_KEYS = re.compile(r"author|user|analyst|picker|email|login", re.IGNORECASE)


@dataclass(frozen=True)
class Fixture:
    """One recorded request."""

    directory: str
    name: str
    url: str
    kind: str  # list | detail | count | empty | error | sample
    fmt: str  # json | geojson | csv | text | xml


def _q(base: str, *parts: str) -> str:
    return base + "?" + "&".join(p for p in parts if p)


FIXTURES: tuple[Fixture, ...] = (
    # ---------------------------------------------------------------- quakes
    *(
        Fixture(
            "quakes",
            f"{door}_{system}_{tag}.json",
            _q(base, win, LIM, f"system={system}"),
            "list",
            "geojson",
        )
        for door, base in (("internal", QUAKES_INT), ("public", QUAKES_PUB))
        for system in ("seiscomp", "sil")
        for tag, win in (("20240822T00", A_Z), ("20260910", B_Z))
    ),
    *(
        Fixture(
            "quakes",
            f"internal_{system}_20240822T00.csv",
            _q(QUAKES_INT, A_Z, LIM, f"system={system}", "format=csv"),
            "list",
            "csv",
        )
        for system in ("seiscomp", "sil")
    ),
    Fixture(
        "quakes",
        "public_sil_20240822T00.csv",
        _q(QUAKES_PUB, A_Z, LIM, "system=sil", "format=csv"),
        "list",
        "csv",
    ),
    Fixture(
        "quakes",
        "internal_seiscomp_20240822T00_count.json",
        _q(QUAKES_INT + "/count", A_Z, LIM, "system=seiscomp"),
        "count",
        "json",
    ),
    Fixture(
        "quakes",
        "internal_sil_20240822T00_count.json",
        _q(QUAKES_INT + "/count", A_Z, LIM, "system=sil"),
        "count",
        "json",
    ),
    Fixture(
        "quakes",
        "internal_detail_IMO2024qnogqh.json",
        QUAKES_INT + "/IMO2024qnogqh",
        "detail",
        "json",
    ),
    Fixture(
        "quakes",
        "public_detail_IMO2024qnogqh.json",
        QUAKES_PUB + "/IMO2024qnogqh",
        "detail",
        "json",
    ),
    Fixture(
        "quakes",
        "internal_detail_IMO2026rusibb.json",
        QUAKES_INT + "/IMO2026rusibb",
        "detail",
        "json",
    ),
    Fixture(
        "quakes",
        "internal_detail_IMO2026rutiqi.json",
        QUAKES_INT + "/IMO2026rutiqi",
        "detail",
        "json",
    ),
    Fixture(
        "quakes",
        "internal_empty.json",
        _q(QUAKES_INT, EMPTY_Z, LIM, "system=seiscomp"),
        "empty",
        "geojson",
    ),
    Fixture(
        "quakes",
        "internal_empty.csv",
        _q(QUAKES_INT, EMPTY_Z, LIM, "system=seiscomp", "format=csv"),
        "empty",
        "csv",
    ),
    Fixture(
        "quakes",
        "error_bad_system.json",
        _q(QUAKES_INT, A_Z, "system=bogus"),
        "error",
        "json",
    ),
    Fixture(
        "quakes",
        "empty_before_1991.json",
        _q(
            QUAKES_INT,
            "start_time=1990-01-01T00:00:00Z&end_time=1990-01-02T00:00:00Z",
            "system=sil",
        ),
        "empty",
        "json",
    ),
    Fixture(
        "quakes",
        "error_sil_detail_404.json",
        QUAKES_INT + "/SIL1352381",
        "error",
        "json",
    ),
    # ------------------------------------------------------------------ fdsn
    Fixture(
        "fdsn", "text_20240822T00.txt", _q(FDSN, A_F, "format=text"), "list", "text"
    ),
    Fixture("fdsn", "text_20260910.txt", _q(FDSN, B_F, "format=text"), "list", "text"),
    Fixture("fdsn", "quakeml_20240822T00.xml", _q(FDSN, A_F), "list", "xml"),
    Fixture(
        "fdsn",
        "quakeml_IMO2024qnogqh_all.xml",
        _q(
            FDSN,
            "eventid=IMO2024qnogqh",
            "includeallorigins=true",
            "includeallmagnitudes=true",
            "includefocalmechanism=true",
        ),
        "detail",
        "xml",
    ),
    Fixture(
        "fdsn",
        "quakeml_IMO2026rusibb_all.xml",
        _q(
            FDSN,
            "eventid=IMO2026rusibb",
            "includeallorigins=true",
            "includeallmagnitudes=true",
            "includefocalmechanism=true",
        ),
        "detail",
        "xml",
    ),
    Fixture(
        "fdsn",
        "text_20240822T00_limit2_asc.txt",
        _q(FDSN, A_F, "format=text", "limit=2", "orderby=time-asc"),
        "sample",
        "text",
    ),
    Fixture(
        "fdsn",
        "error_nodata404.txt",
        _q(
            FDSN,
            "starttime=2024-08-22T00:00:00&endtime=2024-08-22T00:02:00",
            "format=text",
            "nodata=404",
        ),
        "error",
        "text",
    ),
    Fixture(
        "fdsn",
        "error_bad_param.txt",
        _q(FDSN, A_F, "format=text", "bogus=1"),
        "error",
        "text",
    ),
    Fixture(
        "fdsn",
        "error_updatedafter.txt",
        _q(FDSN, A_F, "format=text", "updatedafter=2026-01-01T00:00:00"),
        "error",
        "text",
    ),
    Fixture(
        "fdsn",
        "error_updateafter.txt",
        _q(FDSN, A_F, "format=text", "updateafter=2026-01-01T00:00:00"),
        "error",
        "text",
    ),
    # ---------------------------------------------------------- skjalftalisa
    *(
        Fixture("skjalftalisa", f"{door}_{tag}.json", _q(base, win), "list", "geojson")
        for door, base in (("internal", SKJ_INT), ("public", SKJ_PUB))
        for tag, win in (("20240822T00", A_Z), ("20260910", B_Z))
    ),
    Fixture(
        "skjalftalisa",
        "internal_20240822T00_count.json",
        _q(SKJ_INT + "/count", A_Z),
        "count",
        "json",
    ),
    Fixture(
        "skjalftalisa",
        "internal_detail_1352381.json",
        SKJ_INT + "/1352381",
        "detail",
        "json",
    ),
    Fixture(
        "skjalftalisa",
        "internal_detail_1352541.json",
        SKJ_INT + "/1352541",
        "detail",
        "json",
    ),
    Fixture(
        "skjalftalisa",
        "internal_detail_1453678.json",
        SKJ_INT + "/1453678",
        "detail",
        "json",
    ),
    Fixture(
        "skjalftalisa",
        "public_detail_1352381.json",
        SKJ_PUB + "/1352381",
        "detail",
        "json",
    ),
    Fixture(
        "skjalftalisa", "internal_empty.json", _q(SKJ_INT, EMPTY_Z), "empty", "geojson"
    ),
    Fixture(
        "skjalftalisa",
        "error_naive_time.json",
        _q(SKJ_INT, "start_time=2024-08-22T00:00:00&end_time=2024-08-22T01:00:00"),
        "error",
        "json",
    ),
    Fixture(
        "skjalftalisa",
        "error_count_out_of_bounds.json",
        _q(
            SKJ_INT + "/count",
            "start_time=1991-01-07T00:00:00Z&end_time=2026-11-13T00:00:00Z",
        ),
        "error",
        "json",
    ),
    Fixture(
        "skjalftalisa", "error_detail_404.json", SKJ_INT + "/999999999", "error", "json"
    ),
)

KEEP_HEADERS = ("content-type", "content-length", "server", "link", "x-total-count")


def fetch(url: str) -> tuple[int, dict[str, str], bytes]:
    """GET ``url``; return status, lower-cased headers and body (errors too)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return (
                resp.status,
                {k.lower(): v for k, v in resp.headers.items()},
                resp.read(),
            )
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read()


class Redactor:
    """Replace analyst logins with ``analyst-N``, consistently over one run."""

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}

    def _token(self, login: str) -> str:
        if login not in self.mapping:
            self.mapping[login] = f"analyst-{len(self.mapping) + 1}"
        return self.mapping[login]

    def logins(self, fmt: str, text: str) -> set[str]:
        """Person-like values found in a body (author fields and the like)."""
        found: set[str] = set()
        if fmt in ("json", "geojson"):
            try:
                found |= _json_person_values(json.loads(text))
            except json.JSONDecodeError:
                pass
        elif fmt == "xml":
            found |= {m for m in re.findall(r"<author>([^<]*)</author>", text)}
        elif fmt == "text" and text.startswith("#EventID"):
            rows = csv.DictReader(io.StringIO(text.lstrip("#")), delimiter="|")
            for row in rows:
                found |= {row.get("Author") or "", row.get("MagAuthor") or ""}
        elif fmt == "csv":
            for row in csv.DictReader(io.StringIO(text)):
                found |= {
                    v for k, v in row.items() if k and PERSON_KEYS.search(k) and v
                }
        return {
            v
            for v in found
            if not MACHINE_AUTHOR.match(v) and not v.startswith("analyst-")
        }

    def redact(self, fmt: str, text: str) -> tuple[str, int]:
        n = 0
        for login in sorted(self.logins(fmt, text), key=len, reverse=True):
            pat = re.compile(
                r"(?<![A-Za-z0-9_.-])" + re.escape(login) + r"(?![A-Za-z0-9_-])"
            )
            text, k = pat.subn(self._token(login), text)
            n += k
        return text, n


def _json_person_values(obj: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and PERSON_KEYS.search(k):
                out.add(v)
            elif isinstance(v, str) and v.startswith("{"):
                try:  # the legacy sp3 field is JSON inside a string
                    out |= _json_person_values(json.loads(v))
                except json.JSONDecodeError:
                    pass
            else:
                out |= _json_person_values(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _json_person_values(v)
    return out


def count_events(fmt: str, text: str) -> int | None:
    """Events (rows) in a list body; ``None`` when the body is not a list."""
    if fmt == "geojson":
        try:
            return len(json.loads(text).get("features", []))
        except (json.JSONDecodeError, AttributeError):
            return None
    if fmt == "csv":
        return max(len([ln for ln in text.splitlines() if ln.strip()]) - 1, 0)
    if fmt == "text" and text.startswith("#EventID"):
        return len([ln for ln in text.splitlines()[1:] if ln.strip()])
    if fmt == "xml":
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return None
        return sum(1 for el in root.iter() if el.tag.endswith("}event"))
    return None


def record(directories: set[str]) -> None:
    redactor = Redactor()
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    manifests: dict[str, list[dict[str, Any]]] = {}
    for i, fx in enumerate(f for f in FIXTURES if f.directory in directories):
        if i:
            time.sleep(PAUSE_S)
        status, headers, body = fetch(fx.url)
        text = body.decode("utf-8")
        text, n_red = redactor.redact(fx.fmt, text)
        out = HERE / fx.directory / fx.name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(text.encode("utf-8"))
        entry = {
            "file": fx.name,
            "url": fx.url,
            "kind": fx.kind,
            "format": fx.fmt,
            "recorded": today,
            "status": status,
            "headers": {k: headers[k] for k in KEEP_HEADERS if k in headers},
            "bytes": len(text.encode("utf-8")),
            "events": count_events(fx.fmt, text),
            "redactions": n_red,
        }
        manifests.setdefault(fx.directory, []).append(entry)
        print(
            f"{status} {entry['bytes']:>7} ev={entry['events']} red={n_red} {fx.directory}/{fx.name}"
        )
    for directory, entries in manifests.items():
        path = HERE / directory / "manifest.json"
        path.write_text(
            json.dumps(entries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"distinct analyst logins redacted: {len(redactor.mapping)}")


if __name__ == "__main__":
    wanted = set(sys.argv[1:]) or {"quakes", "fdsn", "skjalftalisa"}
    record(wanted)
