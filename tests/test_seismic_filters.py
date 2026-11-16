"""Tests for geo_dataread.seismic.filters (T5, PLAN-seismicity.md)."""

from __future__ import annotations

import copy
from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic.filters import (
    DERIVED_MOMENT_FLAG,
    FilterError,
    daily_counts,
    distance_km,
    moment_from_magnitude,
    moment_from_mlw,
    points_in_polygon,
    subset,
    to_utc,
    with_derived_moment,
    within_distance,
)
from geo_dataread.seismic.schema import (
    ATTRS_KEY,
    catalog_provenance,
    coerce_catalog,
    cumulative_moment,
    empty_catalog,
    moment_magnitude,
    validate,
    with_provenance,
)

# -------------------------------------------------------------- fixtures

#: Six events on a 1° grid around Reykjanes; ids e0…e5 in time order.
TIMES = [
    "2024-08-22T00:00:00Z",
    "2024-08-22T06:00:00Z",
    "2024-08-23T00:00:00Z",
    "2024-08-23T12:00:00Z",
    "2024-08-25T00:00:00Z",
    "2024-08-25T23:59:59Z",
]


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": TIMES,
            "event_id": [f"e{i}" for i in range(6)],
            "latitude": [63.0, 63.5, 64.0, 64.5, 65.0, np.nan],
            "longitude": [-23.0, -22.5, -22.0, -21.5, -21.0, -20.0],
            "depth_km": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "magnitude": [0.5, 1.0, 1.5, np.nan, 2.5, 3.0],
            "magnitude_type": ["Mlw", "Mlw", "ML", None, "aut.mag", "Mlw"],
            "moment_nm": [1e10, np.nan, 3e10, np.nan, np.nan, 6e10],
            "evaluation_mode": [
                "manual",
                "manual",
                "manual",
                "automatic",
                "automatic",
                "manual",
            ],
            "evaluation_status": [None, None, None, None, None, "rejected"],
            "event_type": [
                "earthquake",
                "earthquake",
                "explosion",
                None,
                "earthquake",
                "other",
            ],
            "quality": [np.nan, np.nan, np.nan, 80.0, 40.0, np.nan],
        }
    )


@pytest.fixture
def cat() -> pd.DataFrame:
    df = coerce_catalog(_raw(), source="sil-files")
    return with_provenance(
        df,
        catalog_provenance(
            source="sil-files",
            location="/fixture",
            fetched_at=datetime(2026, 11, 13, tzinfo=UTC),
        ),
    )


def ids(df: pd.DataFrame) -> list[str]:
    return list(df["event_id"])


# -------------------------------------------------------------- time


def test_to_utc_naive_is_utc_not_local() -> None:
    assert to_utc("2024-08-22 12:00") == pd.Timestamp("2024-08-22 12:00", tz="UTC")
    assert to_utc(datetime(2024, 8, 22, 12)) == pd.Timestamp(
        "2024-08-22 12:00", tz="UTC"
    )
    assert to_utc(date(2024, 8, 22)) == pd.Timestamp("2024-08-22", tz="UTC")


def test_to_utc_aware_converted() -> None:
    plus2 = datetime(2024, 8, 22, 2, tzinfo=timezone(timedelta(hours=2)))
    assert to_utc(plus2) == pd.Timestamp("2024-08-22 00:00", tz="UTC")
    assert to_utc("2024-08-22T02:00+02:00") == pd.Timestamp("2024-08-22", tz="UTC")
    assert to_utc(pd.Timestamp("2024-08-21 20:00", tz="America/New_York")) == (
        pd.Timestamp("2024-08-22 00:00", tz="UTC")
    )


def test_to_utc_bad() -> None:
    with pytest.raises(FilterError):
        to_utc("not a time")
    with pytest.raises(FilterError):
        to_utc(pd.NaT)


def test_time_bounds_inclusive(cat: pd.DataFrame) -> None:
    out = subset(cat, start="2024-08-22T06:00:00Z", end="2024-08-23T12:00:00Z")
    assert ids(out) == ["e1", "e2", "e3"]


def test_start_alone_and_end_alone(cat: pd.DataFrame) -> None:
    assert ids(subset(cat, start="2024-08-25")) == ["e4", "e5"]
    assert ids(subset(cat, end="2024-08-22")) == ["e0"]


def test_end_date_is_midnight(cat: pd.DataFrame) -> None:
    # documented: end="2024-08-23" is 00:00Z, so e3 (12:00 that day) is out
    assert ids(subset(cat, start="2024-08-23", end="2024-08-23")) == ["e2"]


def test_time_naive_utc_and_non_utc_aware_agree(cat: pd.DataFrame) -> None:
    naive = subset(cat, start="2024-08-22 06:00", end=datetime(2024, 8, 23))
    utc = subset(
        cat,
        start=pd.Timestamp("2024-08-22 06:00", tz="UTC"),
        end=datetime(2024, 8, 23, tzinfo=UTC),
    )
    tz = timezone(timedelta(hours=-1))  # Azores-like offset
    aware = subset(
        cat,
        start=datetime(2024, 8, 22, 5, tzinfo=tz),
        end="2024-08-22T23:00-01:00",
    )
    assert ids(naive) == ids(utc) == ids(aware) == ["e1", "e2"]


def test_start_after_end_raises(cat: pd.DataFrame) -> None:
    with pytest.raises(FilterError, match="after"):
        subset(cat, start="2024-08-25", end="2024-08-22")


def test_naive_index_refused_for_time_filter(cat: pd.DataFrame) -> None:
    naive = cat.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(FilterError, match="tz-naive"):
        subset(naive, start="2024-08-22")
    # a non-time filter does not need the index
    assert len(subset(naive, min_lat=64.0)) == 3


# -------------------------------------------------------------- region


def test_lat_lon_bounds_inclusive(cat: pd.DataFrame) -> None:
    out = subset(cat, min_lat=63.5, max_lat=64.5, min_lon=-22.5, max_lon=-21.5)
    assert ids(out) == ["e1", "e2", "e3"]


def test_max_lon_alone_regression(cat: pd.DataFrame) -> None:
    # legacy seism_subset guarded max_lon by min_lon: max_lon alone was ignored
    assert ids(subset(cat, max_lon=-22.0)) == ["e0", "e1", "e2"]


def test_min_lon_alone_regression(cat: pd.DataFrame) -> None:
    # legacy: min_lon alone compared against max_lon=None
    assert ids(subset(cat, min_lon=-21.0)) == ["e4", "e5"]


def test_each_bound_alone(cat: pd.DataFrame) -> None:
    assert ids(subset(cat, min_lat=64.5)) == ["e3", "e4"]  # e5 NaN lat
    assert ids(subset(cat, max_lat=63.5)) == ["e0", "e1"]
    assert ids(subset(cat, min_depth=4.0)) == ["e3", "e4", "e5"]
    assert ids(subset(cat, max_depth=2.0)) == ["e0", "e1"]
    assert ids(subset(cat, min_mag=2.5)) == ["e4", "e5"]
    assert ids(subset(cat, max_mag=1.0)) == ["e0", "e1"]
    assert ids(subset(cat, min_quality=80)) == ["e3"]
    assert ids(subset(cat, max_quality=40)) == ["e4"]


def test_bbox_equals_separate_bounds(cat: pd.DataFrame) -> None:
    a = subset(cat, bbox=(-22.5, 63.5, -21.5, 64.5))
    b = subset(cat, min_lon=-22.5, min_lat=63.5, max_lon=-21.5, max_lat=64.5)
    assert ids(a) == ids(b) == ["e1", "e2", "e3"]


def test_bbox_conflict_raises(cat: pd.DataFrame) -> None:
    with pytest.raises(FilterError, match="not both"):
        subset(cat, bbox=(-23, 63, -21, 65), min_lat=63.0)


def test_inverted_bounds_raise(cat: pd.DataFrame) -> None:
    with pytest.raises(FilterError, match="greater"):
        subset(cat, min_lon=-21.0, max_lon=-23.0)
    with pytest.raises(FilterError, match="greater"):
        subset(cat, min_mag=3, max_mag=1)


# -------------------------------------------------------------- NaN


def test_nan_fails_only_its_own_filter(cat: pd.DataFrame) -> None:
    # e2 has NaN depth: out under a depth filter, in under a magnitude filter
    assert "e2" not in ids(subset(cat, min_depth=0.0))
    assert "e2" in ids(subset(cat, min_mag=0.0))
    # e3 has NaN magnitude: the reverse
    assert "e3" in ids(subset(cat, min_depth=0.0))
    assert "e3" not in ids(subset(cat, min_mag=0.0))


def test_no_blanket_dropna(cat: pd.DataFrame) -> None:
    assert ids(subset(cat)) == ids(cat)
    assert len(subset(cat, start="2024-08-01")) == 6


def test_quality_filter_drops_reviewed(cat: pd.DataFrame) -> None:
    out = subset(cat, min_quality=0)
    assert set(out["evaluation_mode"]) == {"automatic"}


# -------------------------------------------------------------- categories


def test_categorical_filters(cat: pd.DataFrame) -> None:
    assert ids(subset(cat, magnitude_types=["Mlw"])) == ["e0", "e1", "e5"]
    assert ids(subset(cat, magnitude_types="ML")) == ["e2"]
    assert ids(subset(cat, evaluation_modes=["automatic"])) == ["e3", "e4"]
    assert ids(subset(cat, event_types=["earthquake"])) == ["e0", "e1", "e4"]


def test_unknown_category_value_raises(cat: pd.DataFrame) -> None:
    with pytest.raises(FilterError, match="MLw"):
        subset(cat, magnitude_types=["MLw"])
    with pytest.raises(FilterError):
        subset(cat, evaluation_modes=["reviewed"])
    with pytest.raises(FilterError):
        subset(cat, event_types=["quake"])


def test_reviewed_only_excludes_automatic_and_rejected(cat: pd.DataFrame) -> None:
    assert ids(subset(cat, reviewed_only=True)) == ["e0", "e1", "e2"]


def test_combination(cat: pd.DataFrame) -> None:
    out = subset(
        cat,
        start="2024-08-22",
        end="2024-08-24",
        bbox=(-23.0, 63.0, -22.0, 64.0),
        min_depth=0.5,
        magnitude_types=["Mlw", "ML"],
        reviewed_only=True,
    )
    assert ids(out) == ["e0", "e1"]
    # rows keep their order and values
    pd.testing.assert_frame_equal(out, cat.iloc[[0, 1]], check_like=False)


# -------------------------------------------------------------- columns, empty


def test_missing_needed_column_raises(cat: pd.DataFrame) -> None:
    with pytest.raises(FilterError, match="quality"):
        subset(cat.drop(columns="quality"), min_quality=10)


def test_missing_unrelated_column_ok(cat: pd.DataFrame) -> None:
    out = subset(cat.drop(columns=["quality", "moment_nm"]), min_lat=64.0)
    assert ids(out) == ["e2", "e3", "e4"]


def test_empty_frame_every_filter() -> None:
    df = empty_catalog()
    out = subset(
        df,
        start="2024-01-01",
        end="2025-01-01",
        bbox=(-25, 63, -13, 67),
        polygon=[(-25, 63), (-13, 63), (-13, 67)],
        min_depth=0,
        max_depth=10,
        min_mag=0,
        max_mag=9,
        magnitude_types=["Mlw"],
        evaluation_modes=["manual"],
        event_types=["earthquake"],
        min_quality=0,
        max_quality=100,
        reviewed_only=True,
    )
    assert len(out) == 0
    assert list(out.columns) == list(df.columns)
    assert validate(out).ok
    assert len(within_distance(df, 64.0, -22.0, 10.0)) == 0
    assert len(daily_counts(df)) == 0
    assert len(with_derived_moment(df, magnitude_types=["Mlw"])) == 0
    by = daily_counts(df, by="evaluation_mode")
    assert isinstance(by, pd.DataFrame) and len(by) == 0
    assert list(by.columns) == ["manual", "automatic"]


def test_empty_result_is_valid(cat: pd.DataFrame) -> None:
    out = subset(cat, min_mag=9.0)
    assert len(out) == 0
    assert validate(out).ok


# -------------------------------------------------------------- polygon

#: A U shape (lon, lat): the notch x∈(1,2), y∈(1,3) is outside.
U_SHAPE = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]


def test_polygon_concave_notch() -> None:
    lon = [0.5, 2.5, 1.5, 1.5, 4.0, -0.1]
    lat = [2.0, 2.0, 2.0, 0.5, 1.0, 1.0]
    got = points_in_polygon(lon, lat, U_SHAPE)
    assert got.tolist() == [True, True, False, True, False, False]


def test_polygon_edge_and_vertex_inside() -> None:
    lon = [1.5, 0.0, 3.0, 1.0, 2.0, 1.5]
    lat = [0.0, 1.5, 3.0, 2.0, 1.0, 1.0]  # bottom edge, left edge, vertex,
    # notch walls, notch vertex, notch floor
    assert points_in_polygon(lon, lat, U_SHAPE).all()


def test_polygon_ray_through_vertex_counted_once() -> None:
    diamond = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    # the rays at y=0 pass through vertices (1,0) and (-1,0)
    got = points_in_polygon([0.0, -2.0, 2.0], [0.0, 0.0, 0.0], diamond)
    assert got.tolist() == [True, False, False]


def test_polygon_closed_ring_and_nan() -> None:
    sq = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    got = points_in_polygon([0.5, np.nan, 0.5], [0.5, 0.5, np.nan], sq)
    assert got.tolist() == [True, False, False]


def test_polygon_bad_input() -> None:
    with pytest.raises(FilterError):
        points_in_polygon([0], [0], [(0, 0), (1, 1)])
    with pytest.raises(FilterError):
        points_in_polygon([0], [0], [0, 1, 2])
    with pytest.raises(FilterError):
        points_in_polygon([0, 1], [0], [(0, 0), (1, 0), (1, 1)])


def test_subset_polygon(cat: pd.DataFrame) -> None:
    tri = [(-23.0, 63.0), (-21.0, 63.0), (-21.0, 65.0)]  # e0, e4 vertices
    out = subset(cat, polygon=tri)
    # e0 vertex, e4 vertex; e1..e3 lie on the hypotenuse lat = 63 + (lon+23)
    assert ids(out) == ["e0", "e1", "e2", "e3", "e4"]
    small = [(-22.6, 63.4), (-22.4, 63.4), (-22.4, 63.6), (-22.6, 63.6)]
    assert ids(subset(cat, polygon=small)) == ["e1"]


# -------------------------------------------------------------- distance


def test_distance_known_grs80_values() -> None:
    df = coerce_catalog(
        pd.DataFrame(
            {
                "time": ["2024-01-01T00:00Z", "2024-01-02T00:00Z", "2024-01-03T00:00Z"],
                "event_id": ["lon1", "lat1", "same"],
                "latitude": [0.0, 1.0, 0.0],
                "longitude": [1.0, 0.0, 0.0],
                "evaluation_mode": ["manual"] * 3,
            }
        ),
        source="sil-files",
    )
    d = distance_km(df, 0.0, 0.0).to_numpy()
    # 1° of longitude on the equator, 1° of latitude from the equator (GRS80)
    assert d[0] == pytest.approx(111.3195, abs=1e-3)
    assert d[1] == pytest.approx(110.5743, abs=1e-3)
    assert d[2] == 0.0
    # argument order is lat, lon: swapping it would select "lon1", not "lat1"
    assert ids(within_distance(df, 1.0, 0.0, 1.0)) == ["lat1"]


def test_within_distance_inclusive_and_ring(cat: pd.DataFrame) -> None:
    d = distance_km(cat, 64.0, -22.0)
    assert np.isnan(d.iloc[5])  # NaN latitude
    edge = float(d.iloc[1])
    assert ids(within_distance(cat, 64.0, -22.0, edge)) == ["e1", "e2", "e3"]
    ring = within_distance(cat, 64.0, -22.0, edge, min_km=1.0)
    assert ids(ring) == ["e1", "e3"]
    with pytest.raises(FilterError):
        within_distance(cat, 64.0, -22.0, -1.0)


def test_within_distance_attrs(cat: pd.DataFrame) -> None:
    out = within_distance(cat, 64.0, -22.0, 50.0)
    rec = out.attrs[ATTRS_KEY]["subsets"][-1]
    assert rec["filter"] == "within_distance"
    assert rec["lat"] == 64.0 and rec["lon"] == -22.0 and rec["max_km"] == 50.0
    assert rec["ellps"] == "GRS80"


# -------------------------------------------------------------- attrs


def test_attrs_kept_and_appended(cat: pd.DataFrame) -> None:
    before = copy.deepcopy(cat.attrs)
    out = subset(cat, start="2024-08-22 06:00", max_lon=-22.0, polygon=U_SHAPE)
    assert cat.attrs == before  # input untouched
    catalog = out.attrs[ATTRS_KEY]
    assert catalog["source"] == "sil-files"
    assert catalog["location"] == "/fixture"
    (rec,) = catalog["subsets"]
    assert rec["filter"] == "subset"
    assert rec["start"] == "2024-08-22T06:00:00Z"
    assert rec["max_longitude"] == -22.0
    assert rec["polygon"][0] == [0.0, 0.0]
    assert rec["n_in"] == 6
    assert isinstance(rec["polygon"][0][0], float)


def test_attrs_chain(cat: pd.DataFrame) -> None:
    out = subset(subset(cat, min_lat=63.5), magnitude_types=["Mlw"])
    subs = out.attrs[ATTRS_KEY]["subsets"]
    assert [s.get("min_latitude") for s in subs] == [63.5, None]
    assert subs[1]["magnitude_types"] == ["Mlw"]
    assert subs[0]["n_out"] == subs[1]["n_in"]


def test_attrs_created_when_absent() -> None:
    df = coerce_catalog(_raw(), source="sil-files")
    df.attrs = {}
    out = subset(df, min_mag=1.0)
    assert out.attrs[ATTRS_KEY]["subsets"][0]["min_magnitude"] == 1.0


# -------------------------------------------------------------- daily counts


def test_daily_counts_zero_filled(cat: pd.DataFrame) -> None:
    c = daily_counts(cat)
    assert isinstance(c, pd.Series)
    assert c.name == "count" and c.index.name == "day"
    assert str(c.index.tz) == "UTC"
    assert c.tolist() == [2, 2, 0, 2]
    assert c.index[0] == pd.Timestamp("2024-08-22", tz="UTC")


def test_daily_counts_span(cat: pd.DataFrame) -> None:
    c = daily_counts(cat, start="2024-08-21", end="2024-08-23")
    assert c.tolist() == [0, 2, 2]
    empty = daily_counts(empty_catalog(), start="2024-01-01", end="2024-01-03")
    assert empty.tolist() == [0, 0, 0]


def test_daily_counts_by(cat: pd.DataFrame) -> None:
    t = daily_counts(cat, by="evaluation_mode")
    assert isinstance(t, pd.DataFrame)
    assert list(t.columns) == ["manual", "automatic"]
    assert t["manual"].tolist() == [2, 1, 0, 1]
    assert t["automatic"].tolist() == [0, 1, 0, 1]


def test_daily_counts_by_with_na_adds_up(cat: pd.DataFrame) -> None:
    t = daily_counts(cat, by="magnitude_type")
    assert isinstance(t, pd.DataFrame)
    assert "<NA>" in t.columns
    total = daily_counts(cat)
    assert t.sum(axis=1).tolist() == total.tolist()


def test_daily_counts_missing_by(cat: pd.DataFrame) -> None:
    with pytest.raises(FilterError, match="nope"):
        daily_counts(cat, by="nope")


# -------------------------------------------------------------- moment


def test_moment_from_magnitude_round_trip() -> None:
    m0 = np.array([1e9, 3.2e12, 4.0e17])
    np.testing.assert_allclose(moment_from_magnitude(moment_magnitude(m0)), m0)
    mw = np.array([-1.0, 0.0, 2.5, 6.0])
    np.testing.assert_allclose(moment_magnitude(moment_from_magnitude(mw)), mw)
    # Mw 6.0 ↔ ~1.12e18 N·m (Hanks & Kanamori)
    assert float(moment_from_magnitude(6.0)) == pytest.approx(1.122e18, rel=1e-3)
    assert np.isnan(moment_from_magnitude(np.nan))


def test_with_derived_moment_mlw_only(cat: pd.DataFrame) -> None:
    out = with_derived_moment(cat, magnitude_types=["Mlw"])
    flag = out[DERIVED_MOMENT_FLAG].tolist()
    # only e1 (Mlw, no moment) is filled; e3 (no magnitude), e4 (aut.mag) not
    assert flag == [False, True, False, False, False, False]
    # Mlw is filled by the SIL relation (T5b, R1), not Hanks–Kanamori
    assert out["moment_nm"].iloc[1] == pytest.approx(float(moment_from_mlw(1.0)))
    # existing moments are never overwritten
    assert out["moment_nm"].iloc[0] == 1e10
    assert np.isnan(out["moment_nm"].iloc[4])
    assert out.attrs[ATTRS_KEY]["subsets"][-1]["n_derived"] == 1
    assert DERIVED_MOMENT_FLAG not in cat.columns
    rep = validate(out)
    assert "extra_column" in rep.kinds()
    # the only error is the fixture's deliberate NaN latitude on e5
    assert {(i.kind, i.column) for i in rep.errors} == {("null_values", "latitude")}


def test_with_derived_moment_explicit_types(cat: pd.DataFrame) -> None:
    out = with_derived_moment(cat, magnitude_types=["Mlw", "aut.mag"])
    assert out[DERIVED_MOMENT_FLAG].tolist() == [
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    with pytest.raises(FilterError):
        with_derived_moment(cat, magnitude_types=["Mw"])


def test_cumulative_moment_reuse(cat: pd.DataFrame) -> None:
    cum = cumulative_moment(with_derived_moment(cat, magnitude_types=("Mlw",)))
    m1 = float(moment_from_mlw(1.0))  # T5b: SIL relation for Mlw
    expected = np.cumsum([1e10, m1, 3e10, 0.0, 0.0, 6e10])
    np.testing.assert_allclose(cum.to_numpy(), expected)


def test_with_derived_moment_needs_types(cat: pd.DataFrame) -> None:
    with pytest.raises(TypeError):
        with_derived_moment(cat)  # type: ignore[call-arg]
