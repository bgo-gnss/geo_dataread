"""catalogue_for, QuakeSelection and quake_regions.yaml (PLAN-ts-quakes Q-T1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from geo_dataread.seismic import selection as qs
from geo_dataread.seismic.schema import coerce_catalog, validate

FIXTURES = Path(__file__).parent / "fixtures" / "sil"
EXAMPLE = Path(__file__).parents[1] / "docs" / "seismic" / "quake_regions.example.yaml"


def frame(rows: list[dict[str, Any]], source: str) -> pd.DataFrame:
    base = {
        "event_id": "x",
        "latitude": 64.65,
        "longitude": -17.45,
        "depth_km": 5.0,
        "magnitude": 1.0,
        "magnitude_type": "ML_SIL",
        "evaluation_mode": "manual",
        "event_type": "earthquake",
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    out = coerce_catalog(df.set_index("time"), source=source)
    out.attrs = {"catalog": {"source": source}}
    return out


class Spy:
    """A reader that records its calls and returns rows inside its window."""

    def __init__(self, source: str, rows: list[dict[str, Any]]) -> None:
        self.df = frame(rows, source)
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp, dict[str, Any]]] = []

    def __call__(self, a: pd.Timestamp, b: pd.Timestamp, **kw: Any) -> pd.DataFrame:
        self.calls.append((a, b, kw))
        t = self.df.index
        out = self.df.loc[(t >= a) & (t < b)].copy()
        out.attrs = dict(self.df.attrs)
        return out


def spies() -> tuple[Spy, Spy]:
    sil = Spy(
        "sil-files",
        [
            {"time": "2026-01-20", "event_id": "s1", "magnitude_type": "Mlw"},
            {"time": "2026-01-31T23:59:59", "event_id": "s2", "magnitude_type": "Mlw"},
            # SIL keeps scattered reviewed events after the seam: never used
            {"time": "2026-02-03", "event_id": "s3", "magnitude_type": "Mlw"},
        ],
    )
    sc = Spy(
        "quakes-seiscomp",
        [
            # SeisComP before the seam is the back-import of SIL: never used
            {"time": "2026-01-20", "event_id": "c0"},
            {"time": "2026-02-01", "event_id": "c1"},
            {"time": "2026-02-05", "event_id": "c2", "evaluation_mode": "automatic"},
            {"time": "2026-02-06", "event_id": "c3", "evaluation_status": "rejected"},
            {"time": "2026-02-07", "event_id": "c4", "event_type": "not existing"},
            {"time": "2026-02-08", "event_id": "c5", "event_type": "other"},
        ],
    )
    return sil, sc


BB = qs.QuakeSelection(bbox=(-17.65, 64.59, -17.25, 64.72), name="bb")


def run(sel: qs.QuakeSelection = BB, **kw: Any) -> tuple[pd.DataFrame, Any, Spy, Spy]:
    sil, sc = spies()
    df, rep = qs.catalogue_for(
        sel,
        kw.pop("start", "2026-01-01"),
        kw.pop("end", "2026-03-01"),
        readers={"sil": sil, "seiscomp": sc, "fdsn": sc},
        **kw,
    )
    return df, rep, sil, sc


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------


def test_auto_cuts_at_the_seam_never_a_union() -> None:
    df, rep, sil, sc = run()
    assert list(df["event_id"]) == ["s1", "s2", "c1", "c2"]
    assert list(df["source"].astype(str)) == ["sil-files"] * 2 + ["quakes-seiscomp"] * 2
    ((a, b, _),) = sil.calls
    ((c, d, kw),) = sc.calls
    assert b == c == qs.SEAM  # half-open on both sides
    assert kw["bbox"] == BB.bbox  # the box goes to the server
    assert [p.source for p in rep.parts] == ["sil-files", "quakes-seiscomp"]
    assert df.attrs["catalog"]["seam"].startswith("2026-02-01")
    assert validate(df).ok


def test_window_on_one_side_reads_one_source() -> None:
    _, rep, sil, sc = run(start="2026-01-01", end="2026-01-25")
    assert len(sil.calls) == 1 and not sc.calls
    _, rep, sil, sc = run(start="2026-02-01", end="2026-02-10")
    assert not sil.calls and len(sc.calls) == 1


def test_auto_automatic_reads_sil_for_the_whole_window() -> None:
    df, _, sil, sc = run(mode="automatic")
    assert not sc.calls
    ((a, b, kw),) = sil.calls
    assert (a, b) == (
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-03-01", tz="UTC"),
    )
    assert kw["mode"] == "automatic"


def test_sil_options_go_to_the_sil_reader_only() -> None:
    df, _, sil, sc = run(min_quality=40.0)
    assert sil.calls[0][2]["min_quality"] == 40.0
    assert "min_quality" not in sc.calls[0][2]
    # SeisComP rows (no Q) survive the floor
    assert {"c1", "c2"} <= set(df["event_id"])


def test_seiscomp_modes_and_rejections() -> None:
    df, rep, _, _ = run(source="seiscomp", start="2026-02-01")
    assert list(df["event_id"]) == [
        "c1",
        "c2",
    ]  # c3 rejected, c4 not existing, c5 other
    assert rep.parts[0].counts["rejected_dropped"] == 2
    df, _, _, _ = run(source="seiscomp", start="2026-02-01", mode="reviewed")
    assert list(df["event_id"]) == ["c1"]
    df, _, _, _ = run(source="seiscomp", start="2026-02-01", mode="automatic")
    assert list(df["event_id"]) == ["c2"]


def test_event_types_all_keeps_other() -> None:
    df, _, _, _ = run(
        BB.replace(event_types=None), source="seiscomp", start="2026-02-01"
    )
    assert "c5" in set(df["event_id"]) and "c4" not in set(df["event_id"])


def test_custom_seam_and_bad_arguments() -> None:
    _, _, sil, sc = run(seam="2026-01-25")
    assert sil.calls[0][1] == sc.calls[0][0] == pd.Timestamp("2026-01-25", tz="UTC")
    with pytest.raises(qs.SelectionError):
        run(source="api")
    with pytest.raises(qs.SelectionError):
        run(mode="both")
    with pytest.raises(ValueError):
        run(start="2026-03-01", end="2026-01-01")


def test_empty_result_is_a_schema_frame() -> None:
    far = qs.QuakeSelection(bbox=(-10.0, 60.0, -9.0, 61.0))
    df, _, _, _ = run(far)
    assert len(df) == 0 and validate(df).ok


# --------------------------------------------------------------------------
# Selection kinds
# --------------------------------------------------------------------------


def pts() -> pd.DataFrame:
    return frame(
        [
            {
                "time": "2026-02-01",
                "event_id": "in",
                "latitude": 64.65,
                "longitude": -17.45,
            },
            {
                "time": "2026-02-02",
                "event_id": "edge",
                "latitude": 64.72,
                "longitude": -17.25,
            },
            {
                "time": "2026-02-03",
                "event_id": "out",
                "latitude": 64.0,
                "longitude": -17.45,
            },
            {
                "time": "2026-02-04",
                "event_id": "deep",
                "latitude": 64.65,
                "longitude": -17.45,
                "depth_km": 25.0,
                "magnitude": 4.0,
            },
        ],
        "quakes-seiscomp",
    )


def ids(df: pd.DataFrame) -> list[str]:
    return list(df["event_id"])


def test_bbox_polygon_radius_and_bounds() -> None:
    df = pts()
    assert ids(BB.apply(df)) == ["in", "edge", "deep"]  # inclusive edge
    poly = qs.QuakeSelection(polygon=((-17.6, 64.6), (-17.5, 64.6), (-17.5, 64.7)))
    assert ids(poly.apply(df)) == []  # the triangle is west of (64.65, -17.45)
    tri = qs.QuakeSelection(polygon=((-17.6, 64.6), (-17.3, 64.6), (-17.45, 64.7)))
    assert ids(tri.apply(df)) == ["in", "deep"]
    rad = qs.QuakeSelection(radius=(64.65, -17.45, 5.0))
    assert ids(rad.apply(df)) == ["in", "deep"]
    assert ids(BB.replace(max_depth=10.0).apply(df)) == ["in", "edge"]
    assert ids(BB.replace(min_mag=3.0).apply(df)) == ["deep"]


def test_server_bbox_encloses_the_region() -> None:
    rad = qs.QuakeSelection(radius=(64.65, -17.45, 10.0))
    w, s, e, n = rad.server_bbox()  # type: ignore[misc]
    assert s < 64.65 - 10 / 111.19 < 64.65 + 10 / 111.19 < n
    assert e - w > 2 * 10 / 111.19  # wider in longitude at 64.65°N
    assert qs.QuakeSelection().server_bbox() is None


def test_selection_validation() -> None:
    with pytest.raises(qs.SelectionError, match="one region"):
        qs.QuakeSelection(bbox=(-18, 64, -17, 65), radius=(64.5, -17.5, 5))
    with pytest.raises(qs.SelectionError, match="inverted"):
        qs.QuakeSelection(bbox=(-17, 64, -18, 65))
    with pytest.raises(qs.SelectionError, match="event types"):
        qs.QuakeSelection(event_types=("quake",))
    with pytest.raises(qs.SelectionError, match="km > 0"):
        qs.QuakeSelection(radius=(64.5, -17.5, 0))


def test_region_spec() -> None:
    regions = qs.read_quake_regions(EXAMPLE)
    assert qs.parse_region_spec("-18,64,-17,65", regions).bbox == (
        -18.0,
        64.0,
        -17.0,
        65.0,
    )
    assert qs.parse_region_spec("64.5,-17.5,12", regions).radius == (64.5, -17.5, 12.0)
    assert (
        qs.parse_region_spec("bardarbunga-caldera", regions).name
        == "bardarbunga-caldera"
    )
    with pytest.raises(qs.SelectionError, match="unknown quake region"):
        qs.parse_region_spec("askja", regions)
    # a coordinate spec never reads the file
    qs.parse_region_spec("1,2,3,4", lambda: (_ for _ in ()).throw(AssertionError()))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# quake_regions.yaml
# --------------------------------------------------------------------------


def test_example_regions_file() -> None:
    regions = qs.read_quake_regions(EXAMPLE)
    bb = qs.region_for_area("bardarbunga", regions)
    assert bb is not None and bb.name == "bardarbunga-caldera"
    assert bb.selection.bbox == (-17.65, 64.59, -17.25, 64.72)  # BGÓ's Q7 box
    assert regions["bardarbunga-caldera-2026proto"].selection.bbox == (
        -17.75,
        64.56,
        -17.30,
        64.71,
    )
    assert qs.region_for_area("askja", regions) is None


def test_regions_file_errors(tmp_path: Path) -> None:
    def bad(doc: Any, match: str) -> None:
        p = tmp_path / "r.yaml"
        p.write_text(yaml.safe_dump(doc))
        with pytest.raises(qs.SelectionError, match=match):
            qs.read_quake_regions(p)

    bad({}, "regions")
    bad({"regions": {"a": {"title": "x"}}}, "needs bbox")
    bad({"regions": {"a": {"bbox": [1, 2, 3, 4], "colour": "red"}}}, "unknown keys")
    bad({"regions": {"a": {"bbox": [1, 2], "area": "x"}}}, "region 'a'")
    bad({"regions": {"a": {"bbox": [1, 2, 3, 4], "default": True}}}, "needs an area")
    two = {"bbox": [1, 2, 3, 4], "area": "x", "default": True}
    bad({"regions": {"a": two, "b": two}}, "two default")
    with pytest.raises(FileNotFoundError, match="example"):
        qs.read_quake_regions(tmp_path / "missing.yaml")


def test_region_bounds_and_types_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "regions": {
                    "r": {
                        "radius": [64.6, -17.4, 8],
                        "max_depth": 12,
                        "event_types": "all",
                    }
                }
            }
        )
    )
    sel = qs.read_quake_regions(p)["r"].selection
    assert sel.radius == (64.6, -17.4, 8.0) and sel.max_depth == 12
    assert sel.event_types is None


# --------------------------------------------------------------------------
# Real SIL fixtures through the default SIL reader
# --------------------------------------------------------------------------


def test_sil_fixture_day_through_catalogue_for() -> None:
    everywhere = qs.QuakeSelection(bbox=(-25.0, 63.0, -13.0, 67.0), event_types=None)
    df, rep = qs.catalogue_for(
        everywhere,
        "2024-08-22",
        "2024-08-23",
        source="sil",
        sil_base=FIXTURES,
        use_cache=False,
    )
    assert len(df) == 11 and rep.parts[0].read == 11
    df40, _ = qs.catalogue_for(
        everywhere,
        "2024-08-22",
        "2024-08-23",
        source="sil",
        sil_base=FIXTURES,
        use_cache=False,
        min_quality=40.0,
    )
    assert len(df40) == 8  # the three Q-10 automatic rows go
    assert np.asarray(df40["evaluation_mode"].astype(str) == "manual").all()
