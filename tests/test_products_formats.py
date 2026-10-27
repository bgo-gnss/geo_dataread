"""gpsprod step 7: .gmt (psvelo), .gbis.txt (GBIS, m) and .xlsx outputs."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

from geo_dataread.products import DisplacementRecord, RecordSet
from geo_dataread.products.files import suffix_of, write
from geo_dataread.products.xlsx import _col, write_xlsx


def _rec(sta: str, d=(10.0, -20.0, 5.0), refused=None) -> DisplacementRecord:
    return DisplacementRecord(
        station=sta, lon=-22.4, lat=63.9, period=(2024.0, 2024.5),
        period_dates=("2024-01-01", "2024-06-30"), frame="itrf2014",
        series="imo.24hr_itrf14", session="24h", ref="plate-periodic",
        ref_requested="plate-periodic", view="declared", method="ends",
        sigma_method="s", displacement=d if refused is None else (float("nan"),) * 3,
        sigma=(1.0, 2.0, 3.0), refused=refused,
    )  # fmt: skip


def _set() -> RecordSet:
    return RecordSet(
        [_rec("AAAA"), _rec("BBBB", refused="no data")],
        "displacement",
        {"frame": "itrf2014"},
    )


def _sheet_cells(path: Path, n: int) -> list[list[str]]:
    with zipfile.ZipFile(path) as z:
        xml = z.read(f"xl/worksheets/sheet{n}.xml").decode()
    rows = []
    for row in re.findall(r"<row [^>]*>(.*?)</row>", xml):
        rows.append(re.findall(r"<v>(.*?)</v>|<t[^>]*>(.*?)</t>", row))
    return [[a or b for a, b in r] for r in rows]


def test_suffixes() -> None:
    assert suffix_of("a/b.gbis.txt") == ".gbis.txt"
    assert suffix_of("x.GMT") == ".gmt" and suffix_of("x.csv") == ".csv"
    assert [_col(i) for i in (0, 25, 26, 27, 701, 702)] == [
        "A",
        "Z",
        "AA",
        "AB",
        "ZZ",
        "AAA",
    ]


def test_gmt_is_psvelo_in_mm(tmp_path: Path) -> None:
    write(_set(), str(tmp_path / "d.gmt"))
    rows = [
        ln.split()
        for ln in (tmp_path / "d.gmt").read_text().splitlines()
        if not ln.startswith("#")
    ]
    assert rows == [
        [
            "-22.400000",
            "63.900000",
            "-20.000",
            "10.000",
            "2.000",
            "1.000",
            "0.000",
            "AAAA",
        ]
    ]


def test_gbis_is_in_metres(tmp_path: Path) -> None:
    write(_set(), str(tmp_path / "d.gbis.txt"))
    rows = [
        ln.split()
        for ln in (tmp_path / "d.gbis.txt").read_text().splitlines()
        if not ln.startswith("%")
    ]
    assert rows == [["-22.400000", "63.900000", "-0.020000", "0.010000", "0.005000",
                     "0.002000", "0.001000", "0.003000", "AAAA"]]  # fmt: skip


def test_gmt_and_gbis_are_for_displacements(tmp_path: Path) -> None:
    coords = RecordSet([], "coords", {})
    with pytest.raises(ValueError, match="displacements only"):
        write(coords, str(tmp_path / "c.gmt"))
    with pytest.raises(ValueError, match="displacements only"):
        write(coords, str(tmp_path / "c.gbis.txt"))


def test_xlsx_sheets(tmp_path: Path) -> None:
    write(_set(), str(tmp_path / "d.xlsx"))
    data = _sheet_cells(tmp_path / "d.xlsx", 1)
    assert data[0][:3] == ["station", "lon", "lat"] and data[1][0] == "AAAA"
    assert (
        data[1][5] == "10" and data[2][-1] == "no data"
    )  # dN as a number; the refusal
    meta = _sheet_cells(tmp_path / "d.xlsx", 2)
    assert meta[0] == ["key", "value"] and ["frame", "itrf2014"] in meta
    # byte-stable: same set, same bytes
    write(_set(), str(tmp_path / "e.xlsx"))
    assert (tmp_path / "d.xlsx").read_bytes() == (tmp_path / "e.xlsx").read_bytes()


def test_xlsx_writer_escapes_and_refuses_twin_sheets(tmp_path: Path) -> None:
    p = write_xlsx(tmp_path / "x.xlsx", [("a", [["<&>", 1.5, None, "nan"]])])
    assert _sheet_cells(p, 1) == [["&lt;&amp;&gt;", "1.5", "nan"]]
    with pytest.raises(ValueError, match="sheet names"):
        write_xlsx(tmp_path / "y.xlsx", [("a", []), ("a", [])])


def test_velocity_and_series_xlsx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from geo_dataread.products import series as ps
    from geo_dataread.products.loader import StationView
    from geo_dataread.products.velocity_core import VelocitySettings
    from geo_dataread.products.velocity_files import write_velocity_xlsx

    import json

    from geo_dataread.products.velocity_core import VelocityRecord

    golden = Path(__file__).parent / "goldenmaster" / "velocity" / "year.jsonl"
    vrec = VelocityRecord.from_dict(json.loads(golden.read_text().splitlines()[0]))
    p = write_velocity_xlsx([vrec], tmp_path / "v.xlsx", VelocitySettings())
    data = _sheet_cells(p, 1)
    assert data[0][:4] == ["station", "lon", "lat", "ve"] and data[1][0] == "AAAA"
    t = 2024.0 + np.arange(3) / 365.25
    view = StationView("AAAA", t, np.zeros((3, 3)), np.ones((3, 3)), "plate")
    monkeypatch.setattr(ps, "load", lambda *a, **k: view)
    rec = ps.station_series("AAAA", None, ps.SeriesSettings(), doc={})
    (path,) = ps.write_series(rec, tmp_path / "s.xlsx", "xlsx")
    assert len(_sheet_cells(path, 1)) == 4
