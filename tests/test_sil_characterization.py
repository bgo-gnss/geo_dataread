"""Characterization of the SIL day files (ticket T0 of PLAN-seismicity.md).

Two groups of tests live here:

1. **Format pins.**  The fixture files under ``tests/fixtures/sil/`` are cut
   byte-exact from the SIL archive (provenance in ``tests/fixtures/sil/README.md``).
   They are parsed here with a deliberately minimal helper, independent of
   ``geo_dataread``, and the row counts, known rows and cross-file relations are
   pinned.  These values are the acceptance targets for the new parsers (T1),
   the day assembly (T3) and the fault-plane reader (T3b).

2. **Broken legacy behaviour** (removed in T6).  ``geo_dataread.sil_read``
   failed on every fixture under pandas 3 / current gtimes (PLAN §2.4, fatal
   errors 1-3); the ``TestLegacySilReadIsBroken`` class that recorded this was
   deleted when T6 replaced the module with a shim over the new reader.  The
   shim is pinned by ``tests/test_sil_read_shim.py``.

No network, no mount, no gps_parser config: only files under ``tests/fixtures/sil``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "sil"

# --------------------------------------------------------------------------
# Minimal, independent format helpers
# --------------------------------------------------------------------------

#: Column specs (0-based, end-exclusive) established on the fixture bytes.
#: events.lib / lib.mag / modern aut.mag agree with the legacy module's specs;
#: the 1995 and 1993 aut.mag specs are new (the legacy one mis-slices them).
EVENTS_LIB: list[tuple[str, int, int]] = [
    ("num", 0, 4),
    ("date", 4, 13),
    ("time", 13, 24),
    ("latitude", 24, 33),
    ("longitude", 33, 43),
    ("depth", 43, 50),
    ("mlw", 50, 55),
    ("event_type", 55, 59),
    ("id", 59, 92),
    ("sh_min", 92, 101),
    ("sh_max", 101, 108),
    ("azim_comp", 108, 115),
    ("num_of_polarities", 115, 118),
]
LIB_MAG: list[tuple[str, int, int]] = [
    ("num", 0, 4),
    ("date", 4, 13),
    ("time", 13, 24),
    ("latitude", 24, 33),
    ("longitude", 33, 43),
    ("depth", 43, 50),
    ("mlw", 50, 56),
    ("ml", 56, 62),
    ("num_of_st_ml", 62, 67),
    ("min_ml", 67, 73),
    ("max_ml", 73, 79),
    ("num_of_st_pha", 79, 82),
    ("num_of_p_pha", 82, 85),
    ("num_of_s_pha", 85, 88),
    ("num_of_polarities", 88, 92),
    ("d_time", 92, 98),
    ("d_latitude", 98, 104),
    ("d_longitude", 104, 110),
    ("d_depth", 110, 116),
    ("w_rms_res", 116, 123),
    ("min_dist_sta", 123, 130),
    ("max_azim_gap", 130, 136),
    ("vel_model", 136, 138),
    ("event_type", 138, 141),
]
AUT_MAG: list[tuple[str, int, int]] = [
    ("num", 0, 4),
    ("date", 4, 13),
    ("time", 13, 22),
    ("latitude", 22, 30),
    ("longitude", 30, 39),
    ("depth", 39, 45),
    ("ml", 45, 51),
    ("q", 51, 57),
    ("id", 57, 89),
]
AUT_MAG_1995: list[tuple[str, int, int]] = [
    ("num", 0, 4),
    ("date", 4, 13),
    ("time", 13, 22),
    ("latitude", 22, 30),
    ("longitude", 30, 40),
    ("depth", 40, 48),
    ("ml", 48, 61),
]
AUT_MAG_1993: list[tuple[str, int, int]] = [
    ("num", 0, 4),
    ("date", 4, 13),
    ("time", 13, 22),
    ("latitude", 22, 30),
    ("longitude", 30, 40),
    ("depth", 40, 48),
    ("c7", 48, 61),
    ("c8", 61, 69),
    ("c9", 69, 81),
    ("c10", 81, 93),
    ("c11", 93, 111),
    ("c12", 111, 117),
]
#: events.fps is whitespace separated, 33 fields (names as in sil_read.py).
EVENTS_FPS: list[str] = [
    "num",
    "date",
    "time",
    "latitude",
    "d_latitude",
    "longitude",
    "d_longitude",
    "depth",
    "d_depth",
    "mlw",
    "id",
    "seismic_moment",
    "ev_mag_ath",
    "event_type",
    "sh_min",
    "sh_max",
    "azim_comp",
    "relative_size",
    "corner_freq",
    "fault_radius",
    "stress_drop",
    "peak_slip",
    "source_file",
    "strike_a",
    "dip_a",
    "rake_a",
    "strike_b",
    "dip_b",
    "rake_b",
    "num_of_polarities",
    "num_of_amplitudes",
    "unknown_pacc",
    "is_significant",
]

#: Expected line width of every row, per layout.
LINE_WIDTH = {"events.lib": 136, "lib.mag": 141, "aut.mag": 89, "cleanaut.mag": 89}


def read_lines(day: str, kind: str) -> list[str]:
    """Raw lines of one fixture file, newline stripped (trailing blanks kept)."""
    return (FIXTURES / day / kind).read_text(encoding="ascii").splitlines()


def slice_fields(line: str, spec: list[tuple[str, int, int]]) -> dict[str, str]:
    """Cut one fixed-width line; each field stripped."""
    return {name: line[a:b].strip() for name, a, b in spec}


def fps_fields(line: str) -> dict[str, str]:
    """Split one events.fps line on whitespace into its 33 named fields."""
    parts = line.split()
    assert len(parts) == len(EVENTS_FPS), line
    return dict(zip(EVENTS_FPS, parts, strict=True))


def aut_spec(day: str) -> list[tuple[str, int, int]]:
    """The aut.mag layout of a fixture day's era."""
    if day.startswith("1993"):
        return AUT_MAG_1993
    if day.startswith("1995"):
        return AUT_MAG_1995
    return AUT_MAG


def parse_time(date: str, time: str) -> datetime:
    """UTC origin time from ``YYYYMMDD`` and ``HHMMSS.sss``.

    The time field is not always zero padded: 1995 events.lib writes
    ``0316 5.701`` for 03:16:05.701, so internal blanks are read as zeros.
    """
    t = time.strip().replace(" ", "0")
    hms, _, frac = t.partition(".")
    hms = hms.zfill(6)
    micro = int(round(float("0." + frac) * 1e6)) if frac else 0
    base = datetime.strptime(date + hms, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return base.replace(microsecond=micro)


def table(day: str, kind: str) -> list[dict[str, str]]:
    """All rows of a fixture file as dicts of strings."""
    lines = read_lines(day, kind)
    if kind == "events.fps":
        return [fps_fields(ln) for ln in lines]
    spec = {"events.lib": EVENTS_LIB, "lib.mag": LIB_MAG}.get(kind) or aut_spec(day)
    return [slice_fields(ln, spec) for ln in lines]


def row(day: str, kind: str, num: int) -> dict[str, str]:
    """The row with the given archive ``num`` (the original numbers are kept)."""
    hits = [r for r in table(day, kind) if int(r["num"]) == num]
    assert len(hits) == 1, (day, kind, num)
    return hits[0]


# --------------------------------------------------------------------------
# Expected counts (acceptance targets for T1/T3)
# --------------------------------------------------------------------------

KINDS = ("events.fps", "events.lib", "lib.mag", "aut.mag", "cleanaut.mag")

#: Rows per file; None = the file is absent from that day directory.
EXPECTED_COUNTS: dict[str, tuple[int | None, ...]] = {
    "1993/jan/29": (0, 0, 0, 5, None),
    "1995/jun/15": (4, 4, 4, 5, None),
    "2022/dec/12": (4, 4, 4, 3, 3),
    "2024/aug/22": (8, 8, 8, 10, 8),
    "2025/may/03": (6, 6, 6, 7, 7),
    "2026/sep/10": (None, 0, None, 6, 6),
    "2026/sep/11": (None, 0, None, 0, 0),  # synthetic empty day
}


def _count_cases() -> Iterator[tuple[str, str, int | None]]:
    for day, counts in EXPECTED_COUNTS.items():
        for kind, n in zip(KINDS, counts, strict=True):
            yield day, kind, n


def test_fixture_tree_has_exactly_the_expected_days() -> None:
    days = sorted(
        str(p.relative_to(FIXTURES))
        for p in FIXTURES.glob("*/*/*")
        if p.is_dir() and re.fullmatch(r"\d\d", p.name)
    )
    assert days == sorted(EXPECTED_COUNTS)


@pytest.mark.parametrize(("day", "kind", "n"), list(_count_cases()))
def test_row_count(day: str, kind: str, n: int | None) -> None:
    path = FIXTURES / day / kind
    if n is None:
        assert not path.exists()
        return
    assert path.exists()
    assert len(read_lines(day, kind)) == n
    if n == 0:
        assert path.stat().st_size == 0  # truly 0-byte, as in the archive


@pytest.mark.parametrize(("day", "kind", "n"), list(_count_cases()))
def test_fixed_width_lines_are_intact(day: str, kind: str, n: int | None) -> None:
    """Trailing blanks are part of the format; a whitespace-stripping hook breaks it."""
    if not n or kind == "events.fps":
        return
    width = LINE_WIDTH[kind]
    if kind == "aut.mag" and day.startswith("1995"):
        width = 61
    elif kind == "aut.mag" and day.startswith("1993"):
        width = 117
    raw = (FIXTURES / day / kind).read_bytes()
    assert raw.endswith(b"\n")
    assert {len(ln) for ln in raw.decode("ascii").splitlines()} == {width}


def test_only_known_file_kinds_are_present() -> None:
    names = {p.name for p in FIXTURES.rglob("*") if p.is_file()}
    day_files = {n for n in names if not n.endswith(".sp3")}
    assert day_files == {*KINDS, "README.md"}


# --------------------------------------------------------------------------
# Known rows (three per file kind, exact values)
# --------------------------------------------------------------------------


def test_events_fps_known_rows() -> None:
    r = row("2024/aug/22", "events.fps", 1)
    assert parse_time(r["date"], r["time"]) == datetime(
        2024, 8, 22, 0, 10, 4, 454000, tzinfo=UTC
    )
    assert (r["latitude"], r["d_latitude"], r["longitude"], r["d_longitude"]) == (
        "63.88752",
        "0.002",
        "-22.37545",
        "0.003",
    )
    assert (r["depth"], r["d_depth"], r["mlw"]) == ("2.111", "0.6", "0.20")
    assert r["id"] == "/2024/aug/22/00:05:00/10:05:430"
    assert float(r["seismic_moment"]) == 1.77e10
    assert (r["ev_mag_ath"], r["event_type"], r["source_file"]) == ("0.19", "qu", "sp3")
    planes = [
        int(r[k])
        for k in ("strike_a", "dip_a", "rake_a", "strike_b", "dip_b", "rake_b")
    ]
    assert planes == [341, 74, 170, 74, 81, 16]
    assert (r["num_of_polarities"], r["num_of_amplitudes"]) == ("0", "24")
    assert (r["unknown_pacc"], r["is_significant"]) == ("100.00", "1")

    # Negative strike on a vertical plane: T3b's validator must accept it.
    r = row("2024/aug/22", "events.fps", 53)
    assert (r["strike_a"], r["dip_a"], r["rake_a"]) == ("-86", "90", "28")
    assert (r["strike_b"], r["dip_b"], r["rake_b"]) == ("184", "62", "180")
    assert (r["mlw"], r["num_of_polarities"], r["num_of_amplitudes"]) == (
        "1.80",
        "10",
        "60",
    )

    # 1995: Ddepth 242.5 is written as 42.5 (leading digit lost, see lib.mag).
    r = row("1995/jun/15", "events.fps", 1)
    assert parse_time(r["date"], r["time"]) == datetime(
        1995, 6, 15, 1, 7, 33, 724000, tzinfo=UTC
    )
    assert (r["d_depth"], r["mlw"], r["unknown_pacc"], r["is_significant"]) == (
        "42.5",
        "-0.40",
        "44.04",
        "0",
    )


def test_events_lib_known_rows() -> None:
    r = row("2024/aug/22", "events.lib", 1)
    assert r == {
        "num": "1",
        "date": "20240822",
        "time": "001004.454",
        "latitude": "63.88752",
        "longitude": "-22.37545",
        "depth": "2.111",
        "mlw": "0.25",
        "event_type": "qu",
        "id": "/2024/aug/22/00:05:00/10:05:430",
        "sh_min": "-0.994",
        "sh_max": "0.905",
        "azim_comp": "27.6",
        "num_of_polarities": "0",
    }
    # 1995: blank seconds padding and a blank event-type column.
    r = row("1995/jun/15", "events.lib", 4)
    assert r["time"] == "0316 5.701"
    assert parse_time(r["date"], r["time"]) == datetime(
        1995, 6, 15, 3, 16, 5, 701000, tzinfo=UTC
    )
    assert (r["mlw"], r["event_type"], r["azim_comp"]) == ("0.40", "", "39.6")
    # depth and a negative magnitude run together: whitespace splitting fails.
    line1 = read_lines("1995/jun/15", "events.lib")[0]
    assert "6.656-0.39" in line1
    assert slice_fields(line1, EVENTS_LIB)["mlw"] == "-0.39"
    # an explosion
    r = row("2022/dec/12", "events.lib", 46)
    assert (r["event_type"], r["mlw"], r["num_of_polarities"]) == ("ex", "1.25", "4")
    assert r["id"] == "/2022/dec/12/15:55:00/57:46:405"


def test_lib_mag_known_rows() -> None:
    r = row("2024/aug/22", "lib.mag", 1)
    assert r == {
        "num": "1",
        "date": "20240822",
        "time": "001004.454",
        "latitude": "63.88752",
        "longitude": "-22.37545",
        "depth": "2.111",
        "mlw": "0.25",
        "ml": "-0.50",
        "num_of_st_ml": "4",
        "min_ml": "-0.74",
        "max_ml": "-0.20",
        "num_of_st_pha": "9",
        "num_of_p_pha": "7",
        "num_of_s_pha": "9",
        "num_of_polarities": "0",
        "d_time": "0.02",
        "d_latitude": "0.002",
        "d_longitude": "0.003",
        "d_depth": "0.6",
        "w_rms_res": "0.06",
        "min_dist_sta": "3.9",
        "max_azim_gap": "102.0",
        "vel_model": "1",
        "event_type": "qu",
    }
    # No ML at all: sentinel -9.99 / 9.99 / -9.99 with zero stations.
    r = row("2024/aug/22", "lib.mag", 42)
    assert (r["ml"], r["num_of_st_ml"], r["min_ml"], r["max_ml"]) == (
        "-9.99",
        "0",
        "9.99",
        "-9.99",
    )
    # The 1995 row whose Ddepth events.fps truncates.
    r = row("1995/jun/15", "lib.mag", 1)
    assert (r["d_depth"], r["max_azim_gap"], r["vel_model"]) == ("242.5", "360.0", "1")
    # a large, distant event with velocity model 4
    r = row("2025/may/03", "lib.mag", 8)
    assert (r["mlw"], r["ml"], r["event_type"], r["vel_model"]) == (
        "3.09",
        "2.32",
        "ql",
        "4",
    )


def test_cleanaut_mag_known_rows() -> None:
    r = row("2024/aug/22", "cleanaut.mag", 7)  # automatic, never reviewed
    assert r == {
        "num": "7",
        "date": "20240822",
        "time": "005029.1",
        "latitude": "63.943",
        "longitude": "-22.402",
        "depth": "5.2",
        "ml": "-1.9",
        "q": "10.00",
        "id": "/2024/aug/22/00:45:00/50:30:280",
    }
    r = row("2025/may/03", "cleanaut.mag", 65)  # 1 ms off the reviewed id
    assert (r["id"], r["q"]) == ("/2025/may/03/08:15:00/20:25:008", "48.73")
    r = row("2026/sep/10", "cleanaut.mag", 6)  # post-Feb-2026, negative zero ML
    assert parse_time(r["date"], r["time"]) == datetime(
        2026, 9, 10, 9, 12, 48, 700000, tzinfo=UTC
    )
    assert (r["ml"], r["q"], r["id"]) == (
        "-0.0",
        "17.12",
        "/2026/sep/10/09:10:00/12:50:312",
    )


def test_aut_mag_known_rows() -> None:
    # modern layout: in aut.mag, not in cleanaut.mag, not reviewed
    r = row("2024/aug/22", "aut.mag", 1)
    assert (r["time"], r["ml"], r["q"], r["id"]) == (
        "000411.7",
        "1.5",
        "10.00",
        "/2024/aug/22/00:00:00/04:14:727",
    )
    # 1995 layout: 61 characters, no Q, no id
    r = row("1995/jun/15", "aut.mag", 10)
    assert r == {
        "num": "10",
        "date": "19950615",
        "time": "102244.4",
        "latitude": "64.156",
        "longitude": "-20.965",
        "depth": "5.2",
        "ml": "0.0",
    }
    # 1993 layout: 117 characters, zero-padded num, 2-decimal coordinates,
    # several unidentified magnitude-like columns, some blank.
    r = row("1993/jan/29", "aut.mag", 3)
    assert r == {
        "num": "0003",
        "date": "19930129",
        "time": "191032.7",
        "latitude": "64.24",
        "longitude": "-20.75",
        "depth": "10.0",
        "c7": "0.94",
        "c8": "1.81",
        "c9": "",
        "c10": "0.32",
        "c11": "0.70",
        "c12": "",
    }


def test_aut_mag_and_cleanaut_disagree_on_one_event() -> None:
    """Same id, different numbers: cleanaut.mag is not a pure subset of aut.mag."""
    a = row("2025/may/03", "aut.mag", 71)
    c = row("2025/may/03", "cleanaut.mag", 64)
    assert a["id"] == c["id"] == "/2025/may/03/08:15:00/18:43:000"
    assert (a["time"], a["q"]) == ("081840.0", "90.05")
    assert (c["time"], c["q"]) == ("081840.1", "90.04")


# --------------------------------------------------------------------------
# Cross-file relations within a day (join targets for T3)
# --------------------------------------------------------------------------

REVIEWED_DAYS = ("1995/jun/15", "2022/dec/12", "2024/aug/22", "2025/may/03")


@pytest.mark.parametrize("day", REVIEWED_DAYS)
def test_reviewed_files_hold_the_same_events_in_the_same_order(day: str) -> None:
    fps = table(day, "events.fps")
    lib = table(day, "events.lib")
    mag = table(day, "lib.mag")
    for f, b, m in zip(fps, lib, mag, strict=True):
        assert f["num"] == b["num"] == m["num"]
        t = parse_time(f["date"], f["time"])
        assert parse_time(b["date"], b["time"]) == t == parse_time(m["date"], m["time"])
        for k in ("latitude", "longitude", "depth"):
            assert f[k] == b[k] == m[k]
        assert f["id"] == b["id"]  # lib.mag has no id: join by row/time
        assert b["mlw"] == m["mlw"]  # both 0.01
        assert b["event_type"] == m["event_type"] or day.startswith("1995")
        assert f["event_type"] == m["event_type"]
        assert f["d_latitude"] == m["d_latitude"]
        assert f["d_longitude"] == m["d_longitude"]
        assert (
            f["num_of_polarities"] == b["num_of_polarities"] == m["num_of_polarities"]
        )
        assert (f["sh_min"], f["sh_max"]) == (b["sh_min"], b["sh_max"])


@pytest.mark.parametrize("day", REVIEWED_DAYS)
def test_fps_derived_fields_lose_precision(day: str) -> None:
    """events.fps rounds Mlw to 0.1, truncates azim_comp, wraps Ddepth mod 100."""
    for f, m, b in zip(
        table(day, "events.fps"),
        table(day, "lib.mag"),
        table(day, "events.lib"),
        strict=True,
    ):
        assert abs(float(f["mlw"]) - float(m["mlw"])) <= 0.05 + 1e-9
        assert float(f["d_depth"]) == pytest.approx(float(m["d_depth"]) % 100)
        az_fps, az_lib = float(f["azim_comp"]), float(b["azim_comp"])
        assert az_fps == int(az_fps)
        assert abs(az_fps) <= abs(az_lib) + 1e-9 and abs(az_lib - az_fps) < 1.0 + 1e-9


def test_events_lib_and_lib_mag_era_differences() -> None:
    assert {r["event_type"] for r in table("1995/jun/15", "events.lib")} == {""}
    assert {r["event_type"] for r in table("1995/jun/15", "lib.mag")} == {"qu"}


def test_event_types_covered() -> None:
    seen = {r["event_type"] for d in REVIEWED_DAYS for r in table(d, "events.fps")}
    # 'un' is not in PLAN §2.1's list (qu, ex, ql, sf, sm, te).
    assert seen == {"qu", "ex", "ql", "sf", "sm", "te", "un"}


def _ids(day: str, kind: str) -> set[str]:
    return {r["id"] for r in table(day, kind)}


@dataclass(frozen=True)
class JoinCase:
    day: str
    both: frozenset[str]  # reviewed and in cleanaut.mag
    reviewed_only: frozenset[str]
    clean_only: frozenset[str]
    aut_only: frozenset[str]  # in aut.mag, not in cleanaut.mag


def _fs(day: str, *tails: str) -> frozenset[str]:
    return frozenset(f"/{day}/{t}" for t in tails)


JOIN_CASES = [
    JoinCase(
        "2024/aug/22",
        both=_fs(
            "2024/aug/22",
            "00:05:00/10:05:430",
            "00:15:00/19:13:327",
            "00:15:00/20:10:992",
            "07:30:00/36:11:616",
            "07:50:00/54:44:219",
            "11:35:00/39:04:992",
        ),
        reviewed_only=_fs("2024/aug/22", "14:40:00/42:55:609", "21:20:00/26:39:991"),
        clean_only=_fs("2024/aug/22", "00:45:00/50:30:280", "03:25:00/31:59:429"),
        aut_only=_fs("2024/aug/22", "00:00:00/04:14:727", "14:40:00/42:55:609"),
    ),
    JoinCase(
        "2025/may/03",
        both=_fs(
            "2025/may/03",
            "00:00:00/01:13:180",
            "00:30:00/38:28:438",
            "08:10:00/15:02:569",
            "08:15:00/18:43:000",
            "09:55:00/57:41:257",
        ),
        reviewed_only=_fs("2025/may/03", "08:15:00/20:25:007"),
        clean_only=_fs("2025/may/03", "07:40:00/43:38:852", "08:15:00/20:25:008"),
        aut_only=frozenset(),
    ),
    JoinCase(
        "2022/dec/12",
        both=_fs("2022/dec/12", "15:55:00/57:46:405", "18:10:00/12:28:312"),
        reviewed_only=_fs("2022/dec/12", "00:25:00/30:22:507", "20:10:00/62:40:000"),
        clean_only=_fs("2022/dec/12", "00:35:00/37:42:031"),
        aut_only=frozenset(),
    ),
]


@pytest.mark.parametrize("case", JOIN_CASES, ids=lambda c: c.day)
def test_reviewed_automatic_id_join(case: JoinCase) -> None:
    rev, clean, aut = (
        _ids(case.day, "events.fps"),
        _ids(case.day, "cleanaut.mag"),
        _ids(case.day, "aut.mag"),
    )
    assert rev & clean == case.both
    assert rev - clean == case.reviewed_only
    assert clean - rev == case.clean_only
    assert aut - clean == case.aut_only


def test_reviewed_id_can_miss_its_automatic_twin_by_one_millisecond() -> None:
    """An id-only join would count this event twice (reviewed + automatic)."""
    r = row("2025/may/03", "events.fps", 64)
    a = row("2025/may/03", "cleanaut.mag", 65)
    assert r["id"][:-1] == a["id"][:-1] and r["id"] != a["id"]
    dt = parse_time(r["date"], r["time"]) - parse_time(a["date"], a["time"])
    assert abs(dt.total_seconds()) < 0.01


def test_event_id_is_opaque_not_a_time() -> None:
    """Manual events carry ids like '.../20:10:00/62:40:000' (minute 62)."""
    r = row("2022/dec/12", "events.fps", 59)
    assert r["id"] == "/2022/dec/12/20:10:00/62:40:000"
    assert parse_time(r["date"], r["time"]) == datetime(
        2022, 12, 12, 21, 13, 24, 756000, tzinfo=UTC
    )


def test_moment_magnitude_from_events_fps() -> None:
    r = row("2025/may/03", "events.fps", 8)
    mw = 2.0 / 3.0 * math.log10(float(r["seismic_moment"])) - 6.03
    assert mw == pytest.approx(2.7817, abs=1e-3)
    assert r["mlw"] == "3.10"


# --------------------------------------------------------------------------
# Mechanisms: events.fps planes and the per-event .sp3 block (targets for T3b)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sp3:
    lib_line: tuple[str, ...]
    moment_nm: float
    plane_a: tuple[float, float, float]
    plane_b: tuple[float, float, float]
    faulting: str
    n_polarities: int
    n_amplitudes: int
    pct_acceptable: float
    significant: bool | None  # None: the block has no significance line


def parse_sp3(path: Path) -> Sp3:
    text = path.read_text(encoding="latin-1")
    # fixed width: a negative Mlw runs into the label ("lib-0.39")
    lib = re.search(r"^\s+lib\s*(-?[\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)", text, re.M)
    mom = re.search(r"seismic moment:\s+(\S+) Nm", text)
    planes = re.search(
        r"plane a\s+(\S+)\.\s+(\S+)\.\s+(\S+)\. degrees\s+plane b\s+(\S+)\.\s+(\S+)\.\s+(\S+)\.",
        text,
    )
    fault = re.search(r"^\s+(.+?) faulting dominates", text, re.M)
    pol = re.search(r"Of\s+(\d+) first motion polarity", text)
    amp = re.search(r"(\d+) amplitudes for P and S", text)
    pct = re.search(r"([\d.]+) % of all mechanisms are acceptable", text)
    assert lib and mom and planes and fault and pol and amp and pct, path
    sig: bool | None = None
    if "is not significant" in text:
        sig = False
    elif "is significant" in text:
        sig = True
    p = [float(x) for x in planes.groups()]
    return Sp3(
        lib_line=lib.groups(),
        moment_nm=float(mom.group(1)),
        plane_a=(p[0], p[1], p[2]),
        plane_b=(p[3], p[4], p[5]),
        faulting=fault.group(1),
        n_polarities=int(pol.group(1)),
        n_amplitudes=int(amp.group(1)),
        pct_acceptable=float(pct.group(1)),
        significant=sig,
    )


def double_couple(strike: float, dip: float, rake: float) -> npt.NDArray[np.float64]:
    """Unit double-couple moment tensor (Aki & Richards, x=N, y=E, z=down)."""
    s, d, r = np.radians([strike, dip, rake])
    n = np.array([-np.sin(d) * np.sin(s), np.sin(d) * np.cos(s), -np.cos(d)])
    u = np.array(
        [
            np.cos(r) * np.cos(s) + np.sin(r) * np.cos(d) * np.sin(s),
            np.cos(r) * np.sin(s) - np.sin(r) * np.cos(d) * np.cos(s),
            -np.sin(r) * np.sin(d),
        ]
    )
    return np.outer(n, u) + np.outer(u, n)


def sp3_to_aki_richards(
    strike: float, dip: float, rake: float
) -> tuple[float, float, float]:
    """Observed sp3 → events.fps rule (left-hand strike, dip may exceed 90)."""
    if dip >= 90:
        return strike, 180 - dip, 180 - rake
    return strike + 180, dip, rake + 180


def fps_rows() -> Iterator[tuple[str, dict[str, str]]]:
    for day in REVIEWED_DAYS:
        for r in table(day, "events.fps"):
            yield day, r


FPS_ROWS = list(fps_rows())


def _planes(
    r: dict[str, str],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a = (float(r["strike_a"]), float(r["dip_a"]), float(r["rake_a"]))
    b = (float(r["strike_b"]), float(r["dip_b"]), float(r["rake_b"]))
    return a, b


@pytest.mark.parametrize(("day", "r"), FPS_ROWS, ids=[r["id"] for _, r in FPS_ROWS])
def test_every_fps_event_has_its_sp3_block(day: str, r: dict[str, str]) -> None:
    assert r["source_file"] == "sp3"
    path = FIXTURES / (r["id"].lstrip("/") + ".sp3")
    assert path.is_file()
    s = parse_sp3(path)
    assert s.moment_nm == pytest.approx(float(r["seismic_moment"]), rel=6e-3)
    assert s.n_polarities == int(r["num_of_polarities"])
    assert s.n_amplitudes == int(r["num_of_amplitudes"])
    if s.significant is not None:
        assert s.significant == (r["is_significant"] == "1")
    lib = row(day, "events.lib", int(r["num"]))
    assert s.lib_line == (lib["mlw"], lib["sh_min"], lib["sh_max"], lib["azim_comp"])


@pytest.mark.parametrize(("day", "r"), FPS_ROWS, ids=[r["id"] for _, r in FPS_ROWS])
def test_fps_planes_are_one_double_couple(day: str, r: dict[str, str]) -> None:
    """Plane b is the auxiliary plane of plane a (integer degrees, ≤ ~1° off)."""
    a, b = _planes(r)
    assert 0 <= a[1] <= 90 and 0 <= b[1] <= 90
    assert -180 <= a[2] <= 180 and -180 <= b[2] <= 180
    assert np.abs(double_couple(*a) - double_couple(*b)).max() < 0.05


@pytest.mark.parametrize(("day", "r"), FPS_ROWS, ids=[r["id"] for _, r in FPS_ROWS])
def test_sp3_planes_map_to_fps_planes(day: str, r: dict[str, str]) -> None:
    a, b = _planes(r)
    s = parse_sp3(FIXTURES / (r["id"].lstrip("/") + ".sp3"))
    for raw, fps in ((s.plane_a, a), (s.plane_b, b)):
        conv = double_couple(*sp3_to_aki_richards(*raw))
        assert np.abs(conv - double_couple(*fps)).max() < 0.05
    # read as plain Aki-Richards the raw sp3 numbers give the wrong mechanism
    # for every non-vertical plane:
    if s.plane_a[1] != 90:
        assert np.abs(double_couple(*s.plane_a) - double_couple(*a)).max() > 0.5


def test_sp3_known_blocks() -> None:
    """Raw sp3 values, exactly as written (before any convention change)."""
    s = parse_sp3(FIXTURES / "2024/aug/22/00:05:00/10:05:430.sp3")
    assert s.plane_a == (-19.0, 106.0, 10.0) and s.plane_b == (74.0, 99.0, 164.0)
    assert s.moment_nm == 0.177e11
    assert s.lib_line == ("0.25", "-0.994", "0.905", "27.6")
    assert (s.faulting, s.significant, s.pct_acceptable) == ("Strike slip", True, 4.63)

    # old layout: 'Hz' tail on the lib line, 'Low frequency limit' line
    p = FIXTURES / "1995/jun/15/01:45:00/47:03:327.sp3"
    s = parse_sp3(p)
    assert s.plane_a == (-40.0, 123.0, 140.0) and s.plane_b == (205.0, 123.0, 40.0)
    assert (s.faulting, s.significant, s.pct_acceptable) == ("Reverse", False, 22.5)
    text = p.read_text(encoding="latin-1")
    assert "lib 1.20   -1.000  0.411   82.8  3.00 Hz" in text
    assert "Low frequency limit" in text

    # vertical plane a; fps writes it with strike -86
    s = parse_sp3(FIXTURES / "2024/aug/22/11:35:00/39:04:992.sp3")
    assert s.plane_a == (-86.0, 90.0, 152.0) and s.plane_b == (184.0, 118.0, 0.0)
    assert (s.moment_nm, s.n_polarities, s.n_amplitudes) == (0.565e12, 10, 60)


def test_sp3_raw_conventions_seen() -> None:
    blocks = [parse_sp3(FIXTURES / (r["id"].lstrip("/") + ".sp3")) for _, r in FPS_ROWS]
    raw = [p for s in blocks for p in (s.plane_a, s.plane_b)]
    assert any(p[1] > 90 for p in raw)
    assert any(p[0] < 0 for p in raw)
    assert any(p[2] > 180 for p in raw)


def test_faulting_classes_covered() -> None:
    classes = {
        parse_sp3(FIXTURES / (r["id"].lstrip("/") + ".sp3")).faulting
        for _, r in FPS_ROWS
    }
    assert classes == {"Strike slip", "Normal", "Reverse", "Dip-slip"}
    by_id = {
        r["id"]: parse_sp3(FIXTURES / (r["id"].lstrip("/") + ".sp3")).faulting
        for _, r in FPS_ROWS
    }
    assert by_id["/2024/aug/22/00:05:00/10:05:430"] == "Strike slip"
    assert by_id["/2024/aug/22/00:15:00/19:13:327"] == "Normal"
    assert by_id["/2024/aug/22/00:15:00/20:10:992"] == "Reverse"
    assert by_id["/2024/aug/22/07:50:00/54:44:219"] == "Dip-slip"


def test_unknown_pacc_meaning_changes_with_era() -> None:
    """1995: equals sp3 '% of all mechanisms are acceptable'; 2022+: always 100.00."""
    for day, r in FPS_ROWS:
        s = parse_sp3(FIXTURES / (r["id"].lstrip("/") + ".sp3"))
        if day.startswith("1995"):
            assert float(r["unknown_pacc"]) == s.pct_acceptable
        else:
            assert r["unknown_pacc"] == "100.00"


def test_fixtures_hold_no_user_ids() -> None:
    for p in FIXTURES.rglob("*"):
        if p.is_file() and p.name != "README.md":
            text = p.read_text(encoding="latin-1").lower()
            assert "user" not in text, p
    assert not list(FIXTURES.rglob("*.meta_lokimp"))
