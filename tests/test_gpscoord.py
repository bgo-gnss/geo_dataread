"""gpscoord: the coordinate tool's CLI (series on synthetic positions)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from geo_dataread import gpscoord
from geo_dataread.products import cli as pcli
from geo_dataread.products import positions

BASE = np.array([2602351.18, -1074404.60, 5704035.79])  # near SENG


def _positions(sta, start, end, src, *, doc=None):
    n = 8
    dates = [dt.datetime(2024, 1, 1, 11, 59) + dt.timedelta(days=k) for k in range(n)]
    t = 2024.0 + (np.arange(n) + 0.5) / 366
    up = BASE / np.linalg.norm(BASE)
    xyz = BASE + np.outer(0.001 * np.arange(n), up)  # rising 1 mm/day
    cov = np.broadcast_to(np.eye(3) * 1e-6, (n, 3, 3)).copy()
    kept = np.ones(n, bool)
    kept[1] = False
    return positions.Positions(sta, "itrf2014", t, dates, xyz, cov, kept)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(positions, "load_positions", _positions)
    monkeypatch.setattr(
        pcli, "select", lambda names, **kw: sorted({n.upper() for n in names})
    )
    monkeypatch.setattr(
        gpscoord,
        "_velocity",
        lambda sta, frame, xyz: (np.array([-0.0195, -0.0015, 0.0086]), "plate NOAM"),
    )
    monkeypatch.setattr(
        "geo_dataread.products.loader.Source.params_doc", lambda self: None
    )


def _rows(text):
    return [ln for ln in text.splitlines() if not ln.startswith("#")]


def test_series_to_dir_dup_and_kept_rows(fake, tmp_path, capsys):
    out = tmp_path / "out"
    assert gpscoord.main(["series", "AAAA", "BBBB", "--add", "dup=h:first5", "--epoch-shift", "plate",
                          "-o", f"{out}/", "-q"]) == 0  # fmt: skip
    text = (out / "AAAA.csv").read_text()
    assert "# epoch shift: plate NOAM" in text
    assert "# note isn93: each position taken back to 1993.6" in text
    rows = _rows(text)
    assert rows[0] == "date,east_isn93,north_isn93,lat,lon,h,dup"
    assert len(rows) == 1 + 7  # the dropped day is not written
    assert [r.split(",")[0] for r in rows[1:3]] == ["2024-01-01", "2024-01-03"]
    # first five kept days: 0,2,3,4,5 mm → mean 2.8
    assert rows[1].split(",")[-1] == "-2.80"
    assert (out / "BBBB.csv").is_file()


def test_with_dropped_and_screen_station_column(fake, capsys):
    assert gpscoord.main(["series", "AAAA", "BBBB", "--columns", "date,du",
                          "--ref-pos", "first", "--with-dropped", "--no-header", "-q"]) == 0  # fmt: skip
    rows = _rows(capsys.readouterr().out)
    assert rows[0] == "station,date,du,kept"
    assert rows[2] == "AAAA,2024-01-02,1.00,0"
    assert sum(r.startswith("BBBB,") for r in rows) == 8


def test_bad_columns_and_several_to_one_file(fake, tmp_path):
    with pytest.raises(SystemExit, match="unknown key"):
        gpscoord.main(["series", "AAAA", "--columns", "foo"])
    with pytest.raises(SystemExit, match="several stations"):
        gpscoord.main(["series", "AAAA", "BBBB", "-o", str(tmp_path / "x.csv")])


def test_stations_from_nearest_column(tmp_path):
    f = tmp_path / "naesta.csv"
    f.write_text(
        "point,lon,lat,nearest,distance_km\nA,1,2,SYRF,1\nB,1,2,syrf,2\nC,1,2,NBIO,3\n"
    )
    assert gpscoord.stations_from(str(f)) == ["SYRF", "NBIO"]
    assert gpscoord.stations_from(f"{f}:point") == ["A", "B", "C"]
    with pytest.raises(SystemExit, match="no column"):
        gpscoord.stations_from(f"{f}:nope")


def test_target_patterns(tmp_path):
    assert gpscoord._target(None, "SENG") is None
    assert gpscoord._target(f"{tmp_path}/", "SENG") == tmp_path / "SENG.csv"
    assert (
        gpscoord._target(f"{tmp_path}/{{station}}-isn93.csv", "SENG")
        == tmp_path / "SENG-isn93.csv"
    )
    assert gpscoord._target(str(tmp_path / "one.csv"), "SENG") == tmp_path / "one.csv"


def test_convert_delegates_to_geofunc(tmp_path, capsys):
    f = tmp_path / "b.tsv"
    f.write_text(
        "Auðkenni\tX (ISN93) [m]\tY (ISN93) [m]\nHSK-06\t329637,4\t383731,4\n",
        encoding="utf-8",
    )
    assert gpscoord.main(["convert", str(f)]) == 0
    assert capsys.readouterr().out.splitlines()[1].startswith("HSK-06,-22.47253")


def test_default_is_the_position_of_the_day(fake, capsys):
    assert gpscoord.main(["series", "AAAA", "--add",
                          "dE=east@isn93:2024-01-01..2024-01-03", "-q"]) == 0  # fmt: skip
    text = capsys.readouterr().out
    assert "# epoch shift: none" in text
    assert "projected with no datum shift" in text
    rows = _rows(text)
    assert rows[0].endswith(",dE")
    assert abs(float(rows[1].split(",")[-1])) < 0.01  # only Up moves


def test_pos_span_join_and_filters(tmp_path):
    p = tmp_path / "X.pos"
    p.write_text("PBO Station Position Time Series.\nFormat Version: 1.1.1\n"
                 "4-character ID: LISK\nStation name  : LISK_GPS\n"
                 "First Epoch   : 20210228 115900\nLast Epoch    : 20241008 115900\n")  # fmt: skip
    assert gpscoord.pos_span(p) == ("2021-02-28", "2024-10-08")
    assert gpscoord.pos_span(tmp_path / "none.pos") is None
    spans = {"LISK": ("2021-02-28", "2024-10-08"), "LISF": ("2024-10-09", "2026-11-02"),
             "SAFH": ("2023-11-11", "2026-11-02")}  # fmt: skip
    cands = [("LISK", -22.37, 63.922), ("LISF", -22.36, 63.923), ("SAFH", -22.6, 63.85)]
    joined = gpscoord._joined(cands, ["LISK+LISF"], spans)
    assert ("LISK+LISF", -22.36, 63.923) in joined and len(joined) == 2
    assert spans["LISK+LISF"] == ("2021-02-28", "2026-11-02")
    args = type("A", (), {"data_before": "20231110", "data_after": "2026-10-01"})()
    assert gpscoord._in_span(spans["LISK+LISF"], args)
    assert not gpscoord._in_span(spans["SAFH"], args)  # starts after the dike
    assert not gpscoord._in_span(spans["LISK"], args)  # stopped
    assert not gpscoord._in_span(None, args)
    with pytest.raises(SystemExit, match="--join"):
        gpscoord._joined(cands, ["NOPE+NADA"], spans)


def test_stations_from_keeps_joined(tmp_path):
    f = tmp_path / "n.csv"
    f.write_text("name,nearest\nA,LISK+LISF\nB,SENG\nC,LISK+LISF\n")
    assert gpscoord.stations_from(str(f)) == ["LISK+LISF", "SENG"]


def _moved(sta, start, end, src, *, doc=None):
    """LISK days 0-7, LISF days 8-15 ~300 m east; the ground rises 1 mm/day."""
    k0 = 0 if sta == "LISK" else 8
    n = 8
    dates = [
        dt.datetime(2024, 1, 1, 11, 59) + dt.timedelta(days=k0 + k) for k in range(n)
    ]
    t = 2024.0 + (k0 + np.arange(n) + 0.5) / 366
    up = BASE / np.linalg.norm(BASE)
    east = np.array([-BASE[1], BASE[0], 0.0]) / np.hypot(BASE[0], BASE[1])
    mark = BASE + (300.0 * east if sta == "LISF" else 0.0)
    xyz = mark + np.outer(0.001 * (k0 + np.arange(n)), up)
    cov = np.broadcast_to(np.eye(3) * 1e-6, (n, 3, 3)).copy()
    return positions.Positions(sta, "itrf2014", t, dates, xyz, cov, np.ones(n, bool))


def test_moved_station_one_tied_series(fake, monkeypatch, capsys):
    monkeypatch.setattr(positions, "load_positions", _moved)
    assert gpscoord.main(["series", "LISK+LISF", "--add", "dup=h:first",
                          "--tie-days", "1", "-q"]) == 0  # fmt: skip
    text = capsys.readouterr().out
    assert "# tie LISK -> LISF:" in text
    assert "E +300.000 U +0.001 m from LISK" in text  # the 1 mm of that day too
    rows = [r.split(",") for r in _rows(text)]
    assert rows[0] == [
        "date",
        "mark",
        "east_isn93",
        "north_isn93",
        "lat",
        "lon",
        "h",
        "dup",
    ]
    assert [r[1] for r in rows[1:]] == ["LISK"] * 8 + ["LISF"] * 8
    dup = [float(r[-1]) for r in rows[1:]]
    # tied on day 7 (LISK) = day 8 (LISF): no motion assumed across the move,
    # so the rise after the move continues from 7 mm, not from 8
    assert dup[7] == 7.0 and dup[8] == 7.0 and dup[15] == 14.0
    # the absolute columns are each mark's own position: ~300 m apart
    assert float(rows[9][2]) - float(rows[8][2]) == pytest.approx(300.0, abs=1.0)
