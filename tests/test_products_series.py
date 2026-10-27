"""gpsprod step 5: the series product (csv, jsonl, .NEU, tot = gps-savetimes)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geo_dataread.products import cli
from geo_dataread.products import series as ps
from geo_dataread.products.loader import (
    REASON_NETWORK,
    REASON_OUTLIER,
    Source,
    StationView,
)

DAY = 1 / 365.25


def _view(sta: str = "AAAA", n: int = 40) -> StationView:
    rng = np.random.default_rng(len(sta))
    t = 2024.0 + (np.arange(n) + 0.5) * DAY
    raw = rng.normal(0, 2.0, (3, n)) + np.array([[10.0], [-5.0], [3.0]])
    y = raw.copy()
    reasons = np.zeros((3, n), np.uint8)
    y[0, 7] = np.nan
    reasons[0, 7] |= REASON_OUTLIER
    reasons[:, 20] |= REASON_NETWORK  # marked, kept
    return StationView(
        sta, t, y, np.full((3, n), 1.5), "plate-periodic", (), {}, raw, reasons
    )


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps, "load", lambda sta, a, b, src, doc=None: _view(sta))
    monkeypatch.setattr(Source, "params_doc", lambda self: None)


def test_kept_epochs_and_with_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    r = ps.station_series("aaaa", None, ps.SeriesSettings())
    assert (
        r.accepted
        and r.t.size == 39
        and r.n_kept == 39
        and r.period_dates == ("all", "all")
    )
    a = ps.station_series("AAAA", None, ps.SeriesSettings(with_dropped=True))
    assert a.t.size == 40 and a.n_kept == 39
    rows = ps.series_rows(a)
    assert rows[7][8] == "0" and rows[7][9] == "outlier"
    assert rows[20][8] == "1" and rows[20][9] == "network"
    assert np.isfinite(a.y[0, 7])  # the raw value is there


def test_refused_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def nothing(sta, a, b, src, doc=None):
        raise ValueError(f"no data for station {sta}")

    monkeypatch.setattr(ps, "load", nothing)
    r = ps.station_series("ZZZZ", None, ps.SeriesSettings(), doc={})
    assert r.refused == "read: no data for station ZZZZ" and r.t.size == 0


def test_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch(monkeypatch)
    r = ps.station_series("AAAA", None, ps.SeriesSettings())
    (csv,) = ps.write_series(r, tmp_path / "a.csv", "csv")
    text = csv.read_text().splitlines()
    assert text[0] == "# gpsprod series schema 1"
    head = text.index(",".join(ps.SERIES_COLUMNS))
    assert len(text) - head - 1 == 39
    again = ps.write_series(r, tmp_path / "b.csv", "csv")[0]
    assert again.read_bytes() == csv.read_bytes()
    (jl,) = ps.write_series(r, tmp_path / "a.jsonl", "jsonl")
    lines = jl.read_text().splitlines()
    assert json.loads(lines[0])["provenance"]["n_kept"] == 39 and len(lines) == 40
    neu, side = ps.write_series(r, tmp_path / "a.neu", "neu")
    body = neu.read_text().splitlines()
    assert body[1].strip().startswith("#yyyy/mm/dd") and len(body) == 2 + 39
    assert json.loads(side.read_text())["view"] == "cleaned"  # the series default
    m = r.scaled("m")
    assert m.units == "m" and m.y[0, 0] == pytest.approx(r.y[0, 0] / 1000)


def test_neu_is_the_savetimes_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same rows through gtoNEU + gamittoFile as gps_read builds them."""
    from geo_dataread import gps_read

    _patch(monkeypatch)
    r = ps.station_series("AAAA", None, ps.SeriesSettings())
    ps.write_neu(r, tmp_path / "x.neu")
    neu = gps_read.gtoNEU(r.t, r.y, r.sigma, dstring=None, rhour=True)
    gps_read.gamittoFile(neu, str(tmp_path / "y.neu"), mm=True, ref=r.ref, dstring=None)
    assert (tmp_path / "x.neu").read_bytes() == (tmp_path / "y.neu").read_bytes()


def test_tot_is_gamittooneuf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from geo_dataread import gps_read

    calls = []
    monkeypatch.setattr(gps_read, "gamittooneuf", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(cli, "select", lambda names, **k: [n.upper() for n in names])
    assert (
        cli.main(
            [
                "series",
                "SENG",
                "--source",
                "tot",
                "--ref",
                "itrf",
                "-o",
                f"{tmp_path}/",
                "-q",
            ]
        )
        == 0
    )
    ((args, kw),) = calls
    assert args == ("SENG", f"{tmp_path}/SENG-itrf2008.NEU")
    assert kw == {"mm": True, "ref": "itrf2008", "dstring": None, "rhour": True}
    with pytest.raises(SystemExit, match="takes --ref"):
        cli.main(
            ["series", "SENG", "--source", "tot", "-q"]
        )  # plate-periodic: no TOT ref


def test_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    _patch(monkeypatch)
    monkeypatch.setattr(
        cli, "select", lambda names, **k: sorted(n.upper() for n in names)
    )
    assert cli.main(["series", "AAAA", "-q"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# gpsprod series schema 1") and out.count("\n2024-") == 39
    assert (
        cli.main(
            ["series", "AAAA", "BBBB", "-o", f"{tmp_path}/d/", "--format", "neu", "-q"]
        )
        == 0
    )
    assert sorted(p.name for p in (tmp_path / "d").iterdir()) == [
        "AAAA-plate-periodic.neu", "AAAA-plate-periodic.neu.prov.json",
        "BBBB-plate-periodic.neu", "BBBB-plate-periodic.neu.prov.json",
    ]  # fmt: skip
    with pytest.raises(SystemExit, match="several stations"):
        cli.main(["series", "AAAA", "BBBB", "-q"])
    with pytest.raises(SystemExit):
        cli.main(["series", "--help"])
    help_text = capsys.readouterr().out
    for opt in (
        "--source",
        "--format",
        "--with-dropped",
        "--time-format",
        "--view",
        "examples:",
        "formats",
        "gps-savetimes",
    ):
        assert opt in help_text, opt  # fmt: skip
    sub = cli.build_parser()._subparsers._group_actions[0].choices["series"]  # type: ignore[union-attr]
    for action in sub._actions:
        if action.dest != "help":
            assert action.help, action.dest
