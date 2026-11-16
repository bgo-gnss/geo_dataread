"""SIL ``events.ref`` (every automatic solution) and ``read_sil_automatic``.

Lines are copied from the archive (2026-06-13, 1992-04-01, 1995-04-01) into
synthetic day trees in ``tmp_path``; no archive needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geo_dataread.seismic import sil_catalog as sc
from geo_dataread.seismic.schema import validate
from geo_dataread.seismic.sil_files import read_events_ref, read_sil_file

# 2026-06-13: three solutions of the M4.8 (one with no magnitude, -8.0, but
# the highest Q), two of another event, and one of the event before midnight
L114 = [
    "    260613 20:14:11.0  64.65 -17.50   90.05    1.1  -8.0  /2026/jun/13/20:10:00        /14:14:522  L16420141     0",
    "    260613 20:14:15.7  64.52 -18.18   62.04    9.2   4.9  /2026/jun/13/20:10:00        /14:18:289  L16420141     0",
    "    260613 20:14:11.2  64.66 -17.51   40.00    1.1   4.1  /2026/jun/13/20:10:00        /14:18:300  L16420141     1",
    "    260613 00:04:11.0  64.37 -18.12    1.81    5.2  -0.2  /2026/jun/13/00:00:00        /04:14:227  L16400041     0",
    "    260613 00:04:11.1  64.38 -18.11    3.50    5.2  -0.1  /2026/jun/13/00:00:00        /04:14:230  L16400041     0",
    "    260612 23:59:58.0  64.60 -17.40   20.00    3.0   0.5  /2026/jun/12/23:55:00        /59:59:999  L16323595     0",
]
L108 = "    920401 06:41: 8.2  64.24 -21.54    1.57    3.7   1.0  /1992/apr/01/06:41:00        /41:13:359  d09206411"
L1995 = "    950401 03:40:00.9  63.92 -21.02   -5.47    4.8  -9.3  /1995/apr/01/03:35:00        /40:03:140  g09103400     0"


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def test_parse_114(tmp_path: Path) -> None:
    p = read_sil_file(_write(tmp_path / "events.ref", L114))
    f, r = p.frame, p.report
    assert r.kind == "events.ref" and r.layout == "events.ref-114" and r.rows == 6
    assert r.sentinels == {"ml": 1} and not r.malformed
    row = f.iloc[0]
    assert str(row["time"]) == "2026-06-13 20:14:11+00:00"
    assert np.isnan(row["ml"]) and row["q"] == pytest.approx(90.05)
    assert row["id"] == "/2026/jun/13/20:10:00/14:14:522"  # the aut.mag id
    assert row["ref_id"] == "L16420141" and row["flag"] == 0
    assert f["num"].isna().all()


def test_parse_108_padded_seconds_and_1995_sentinel(tmp_path: Path) -> None:
    f = read_events_ref(_write(tmp_path / "a" / "events.ref", [L108])).frame
    assert f["layout"].iloc[0] == "events.ref-108"
    assert str(f["time"].iloc[0]) == "1992-04-01 06:41:08.200000+00:00"
    assert f["flag"].isna().all()
    p = read_events_ref(_write(tmp_path / "b" / "events.ref", [L1995]))
    assert np.isnan(p.frame["ml"].iloc[0]) and p.report.sentinels == {"ml": 1}
    assert p.frame["q"].iloc[0] == pytest.approx(-5.47)  # kept as written


def test_bad_width_is_malformed(tmp_path: Path) -> None:
    p = read_events_ref(_write(tmp_path / "events.ref", [L114[0], L114[0][:-3]]))
    assert p.report.rows == 1 and len(p.report.malformed) == 1


def _tree(tmp_path: Path) -> Path:
    _write(tmp_path / "2026" / "jun" / "13" / "events.ref", L114)
    # the next day's file repeats the M4.8's best solution
    _write(tmp_path / "2026" / "jun" / "14" / "events.ref", [L114[0]])
    _write(tmp_path / "2026" / "jun" / "13" / "aut.mag", [
        "   1 20260613 201411.0  64.654  -17.501   1.1   3.5 90.05 /2026/jun/13/20:10:00/14:14:522",
    ])  # fmt: skip
    return tmp_path


def test_read_sil_automatic_best(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    df, rep = sc.read_sil_automatic("2026-06-13", "2026-06-14", base=root)
    validate(df, strict=True)
    assert len(df) == 2  # the event before midnight is outside the window
    big = df[df["event_id"] == "L16420141"].iloc[0]
    assert big["quality"] == pytest.approx(90.05)
    assert big["latitude"] == pytest.approx(64.65)  # the best solution's place
    assert big["magnitude"] == pytest.approx(4.9)  # from the Q 62 solution
    assert big["magnitude_type"] == "events.ref"
    assert big["evaluation_mode"] == "automatic"
    counts = rep.as_dict()["counts"]
    assert counts["magnitude_from_other_solution"] == 1
    small = df[df["event_id"] == "L16400041"].iloc[0]
    assert small["quality"] == pytest.approx(3.5)


def test_read_sil_automatic_all_and_aut_mag(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    df, rep = sc.read_sil_automatic(
        "2026-06-13", "2026-06-14", base=root, solutions="all"
    )
    assert len(df) == 5  # 6 lines, one outside the window; the repeat dropped
    assert rep.as_dict()["counts"]["duplicates:events.ref"] == 1
    am, _ = sc.read_sil_automatic("2026-06-13", "2026-06-14", base=root, file="aut.mag")
    assert len(am) == 1 and am["magnitude_type"].iloc[0] == "aut.mag"
    assert am["event_id"].iloc[0] == "/2026/jun/13/20:10:00/14:14:522"


def test_read_sil_automatic_bad_args(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sc.read_sil_automatic("2026-06-13", "2026-06-14", base=tmp_path, file="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sc.read_sil_automatic("2026-06-14", "2026-06-13", base=tmp_path)
