"""gpsquake CLI (regions, fetch) on the SIL fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from geo_dataread import gpsquake as gq

FIXTURES = Path(__file__).parent / "fixtures" / "sil"
EXAMPLE = Path(__file__).parents[1] / "docs" / "seismic" / "quake_regions.example.yaml"
ICELAND = "-25,63,-13,67"


@pytest.fixture(autouse=True)
def _sil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPS_SIL_DIR", str(FIXTURES))


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    assert gq.main(list(argv)) == 0
    return capsys.readouterr().out


def test_regions_lists_the_example(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(capsys, "regions", "--regions", str(EXAMPLE))
    assert "bardarbunga-caldera " in out and "[default]" in out
    assert "bbox -17.65,64.59,-17.25,64.72" in out


def test_fetch_summary(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(
        capsys,
        "fetch",
        ICELAND,
        "-s",
        "20240822",
        "-e",
        "20240823",
        "--source",
        "sil",
        "--no-cache",
        "--types",
        "all",
        "--moment",
    )
    assert "n = 11" in out
    assert "sil-files" in out and "automatic" in out and "largest" in out
    assert "cumulative moment" in out


def test_fetch_min_q_and_list(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(
        capsys,
        "fetch",
        ICELAND,
        "-s",
        "20240822",
        "-e",
        "20240823",
        "--source",
        "sil",
        "--no-cache",
        "--types",
        "all",
        "--min-q",
        "40",
        "--list",
    )
    lines = out.strip().splitlines()
    assert lines[0].startswith("time,source,event_id")
    assert len(lines) == 1 + 8


def test_fetch_output_and_sidecar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dst = tmp_path / "q.csv"
    run(
        capsys,
        "fetch",
        ICELAND,
        "-s",
        "20240822",
        "-e",
        "20240823",
        "--source",
        "sil",
        "--no-cache",
        "--mode",
        "automatic",
        "-o",
        str(dst),
    )
    df = pd.read_csv(dst)
    assert len(df) == 10 and set(df["evaluation_mode"]) == {"automatic"}
    side = json.loads(dst.with_suffix(".json").read_text())
    assert side["filters"]["mode"] == "automatic" and side["selection"]["bbox"]


def test_named_region_and_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        gq.main(
            [
                "fetch",
                "nowhere",
                "-s",
                "20240822",
                "--regions",
                str(EXAMPLE),
                "--source",
                "sil",
                "--no-cache",
            ]
        )
        == 2
    )
    assert "unknown quake region" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gq.main(["fetch", "-s", "2024-13-45"])


def test_prefixed_args_for_gpsplot() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quakes")
    gq.add_quake_args(p, prefix="quake-")
    a = p.parse_args(
        [
            "--quakes",
            "64.6,-17.4,10",
            "--quake-mode",
            "reviewed",
            "--quake-min-q",
            "60",
            "--quake-no-duplicates",
            "--quake-max-depth",
            "15",
        ]
    )
    kw = gq.quake_kwargs(a, prefix="quake-")
    assert kw == {
        "source": "auto",
        "mode": "reviewed",
        "min_quality": 60.0,
        "drop_duplicates": True,
        "use_cache": True,
    }
    sel = gq.quake_selection(a, a.quakes, prefix="quake-")
    assert sel.radius == (64.6, -17.4, 10.0) and sel.max_depth == 15.0
    assert (
        gq.quake_kwargs(p.parse_args(["--quake-min-q", "0"]), "quake-")["min_quality"]
        is None
    )


def test_polygon_option() -> None:
    p = argparse.ArgumentParser()
    gq.add_quake_args(p)
    a = p.parse_args(["--polygon", "-17.6 64.6, -17.3 64.6, -17.45 64.7"])
    sel = gq.quake_selection(a, None)
    assert sel.polygon == ((-17.6, 64.6), (-17.3, 64.6), (-17.45, 64.7))
