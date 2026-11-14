"""Legacy Skjálftalísa API reader (ticket T12 of PLAN-seismicity.md).

Offline: every request goes to :class:`FakeSession`, which serves the T9
recorded responses (``tests/fixtures/skjalftalisa/``) or synthetic bodies and
records the URLs it was asked for. The one live test runs only with
``GPS_LIVE_API=1``.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
from collections.abc import Callable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import skjalftalisa_api as sk
from geo_dataread.seismic.schema import ATTRS_KEY, TIME_DTYPE, column_names, validate

FIXTURES = Path(__file__).parent / "fixtures" / "skjalftalisa"

WIN_A = ("2024-08-22T00:00:00Z", "2024-08-22T01:00:00Z")
DAY_B = ("2026-09-10T00:00:00Z", "2026-09-11T00:00:00Z")


def load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


Route = Callable[[str, dict[str, list[str]]], tuple[int, bytes]]


class FakeSession:
    """A :class:`~sk.Transport` serving bodies by a routing function."""

    def __init__(self, route: Route) -> None:
        self.route = route
        self.urls: list[str] = []
        self.timeouts: list[float] = []

    def get(self, url: str, *, timeout: float) -> sk.HttpResponse:
        self.urls.append(url)
        self.timeouts.append(timeout)
        parts = urllib.parse.urlsplit(url)
        status, body = self.route(
            f"{parts.scheme}://{parts.netloc}{parts.path}",
            urllib.parse.parse_qs(parts.query),
        )
        return sk.HttpResponse(status, body, url)


def serve(body: bytes, status: int = 200) -> FakeSession:
    return FakeSession(lambda path, q: (status, body))


def feature(event_id: int, time: str, system: str, **props: Any) -> dict[str, Any]:
    lon = props.pop("lon", -22.4)
    lat = props.pop("lat", 63.9)
    base: dict[str, Any] = {
        "event_id": event_id,
        "time": time,
        "originating_system": system,
        "depth": props.pop("depth", 5.0),
        "event_type": props.pop("event_type", "qu"),
    }
    if system == sk.SYSTEM_PICKS:
        base.update(m_ml=0.1, m_mlw=0.5, seismic_moment=1e10)
    else:
        base.update(m_autmag=-0.5, quality=50.0)
    base.update(props)
    return {
        "type": "Feature",
        "properties": base,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


def collection(features: list[dict[str, Any]] | None) -> bytes:
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


def read(start: str, end: str, session: FakeSession, **kw: Any) -> Any:
    kw.setdefault("pause_s", 0.0)
    kw.setdefault("pad", timedelta(0))
    return sk.read_skjalftalisa(start, end, session=session, **kw)


# --------------------------------------------------------------------------
# Window A: the dedupe against the count endpoint
# --------------------------------------------------------------------------


@pytest.fixture
def win_a() -> tuple[pd.DataFrame, sk.SkjalftalisaReport, FakeSession]:
    session = serve(load("internal_20240822T00.json"))
    df, rep = read(*WIN_A, session)
    return df, rep, session


def test_window_a_one_row_per_event(win_a: Any) -> None:
    df, rep, _ = win_a
    count = json.loads(load("internal_20240822T00_count.json"))["quake_count"]
    assert len(df) == count == 7
    assert df["event_id"].is_unique
    assert rep.get("rows_fetched") == 13
    assert rep.get("rows:picks") == 6 and rep.get("rows:aut") == 7
    assert rep.get("twins_resolved") == 6
    assert rep.get("aut_only") == 1 and rep.get("picks_only") == 0
    assert rep.get("duplicate_rows") == 0
    assert rep.get("events_reviewed") == 6 and rep.get("events_automatic") == 1
    assert rep.get("requests") == 1


def test_window_a_schema(win_a: Any) -> None:
    df, _, _ = win_a
    report = validate(df)
    assert report.ok, report.summary()
    assert report.kinds() <= {"extra_column"}
    assert list(df.columns[: len(column_names())]) == list(column_names())
    assert df.index.dtype == TIME_DTYPE
    assert df.index.is_monotonic_increasing
    assert set(df["source"]) == {"skjalftalisa"}


def test_window_a_reviewed_row(win_a: Any) -> None:
    df, _, _ = win_a
    r = df.loc[df["event_id"] == "1352381"].iloc[0]
    # the reviewed (picks) time, not the automatic 00:10:04.3
    assert df.index[df["event_id"] == "1352381"][0] == pd.Timestamp(
        "2024-08-22T00:10:04.454Z"
    )
    assert r["evaluation_mode"] == "manual"
    assert pd.isna(r["evaluation_status"])
    assert r["magnitude"] == pytest.approx(0.25) and r["magnitude_type"] == "Mlw"
    assert r["mag_mlw"] == pytest.approx(0.25)
    assert r["mag_ml"] == pytest.approx(-0.5)
    assert r["mag_auto"] == pytest.approx(-1.2)  # from the automatic twin
    assert r["moment_nm"] == pytest.approx(1.77e10)
    assert r["depth_km"] == pytest.approx(2.111)  # picks depth, not aut 0.2
    assert r["latitude"] == pytest.approx(63.88752)
    assert r["longitude"] == pytest.approx(-22.37545)
    assert pd.isna(r["quality"])
    assert pd.isna(r["sil_path"])
    assert r["event_type"] == "earthquake" and r["event_type_code"] == "qu"
    assert r["originating_system"] == sk.SYSTEM_PICKS
    assert r["n_solutions"] == 2
    # raw plane fields kept for T12b (fps plane a)
    assert (r["strike"], r["dip"], r["slip"]) == (341.0, 74.0, 170.0)


def test_window_a_order_independent(win_a: Any) -> None:
    df, _, _ = win_a
    # 1352385 lists picks first, 1352381 aut first: both resolve to picks
    for eid in ("1352385", "1352381"):
        assert df.loc[df["event_id"] == eid, "evaluation_mode"].iloc[0] == "manual"
    feats = json.loads(load("internal_20240822T00.json"))["features"]
    shuffled, _ = read(*WIN_A, serve(collection(feats[::-1])))
    pd.testing.assert_frame_equal(
        df.sort_values("event_id").reset_index(),
        shuffled.sort_values("event_id").reset_index(),
    )


def test_window_a_automatic_only(win_a: Any) -> None:
    df, _, _ = win_a
    r = df.loc[df["event_id"] == "1352541"].iloc[0]
    assert r["evaluation_mode"] == "automatic"
    assert r["magnitude"] == pytest.approx(-1.9) and r["magnitude_type"] == "aut.mag"
    assert r["mag_auto"] == pytest.approx(-1.9)
    assert r["quality"] == pytest.approx(10.0)
    assert pd.isna(r["mag_mlw"]) and pd.isna(r["moment_nm"])
    assert r["n_solutions"] == 1
    assert math.isnan(r["strike"])


def test_window_a_provenance(win_a: Any) -> None:
    df, rep, _ = win_a
    cat = df.attrs[ATTRS_KEY]
    assert cat["source"] == "skjalftalisa"
    assert cat["legacy"] is True
    assert cat["endpoint"] == "internal"
    assert cat["location"] == sk.INTERNAL_URL
    assert cat["window"] == {"start": WIN_A[0], "end": WIN_A[1]}
    assert cat["report"]["counts"]["twins_resolved"] == 6
    assert cat["report"] == rep.as_dict()
    json.dumps(cat)  # JSON-safe


def test_query_url_matches_recorded(win_a: Any) -> None:
    _, _, session = win_a
    manifest = {m["file"]: m for m in json.loads(load("manifest.json"))}
    want = urllib.parse.urlsplit(manifest["internal_20240822T00.json"]["url"])
    got = urllib.parse.urlsplit(session.urls[0])
    assert (got.netloc, got.path) == (want.netloc, want.path)
    assert urllib.parse.parse_qs(got.query) == urllib.parse.parse_qs(want.query)
    assert session.timeouts == [sk.DEFAULT_TIMEOUT_S]


def test_public_window_a() -> None:
    session = serve(load("public_20240822T00.json"))
    df, rep = read(*WIN_A, session, internal=False)
    assert len(df) == 5 and rep.get("rows_fetched") == 10
    assert rep.get("twins_resolved") == 5
    assert session.urls[0].startswith(sk.PUBLIC_URL + "?")
    assert df.attrs[ATTRS_KEY]["endpoint"] == "public"
    # the public list has no fault-plane fields
    assert df["strike"].isna().all()
    assert validate(df).ok


@pytest.mark.parametrize(
    ("name", "n"), [("internal_20260910.json", 19), ("public_20260910.json", 18)]
)
def test_day_b_automatic_only(name: str, n: int) -> None:
    df, rep = read(*DAY_B, serve(load(name)), internal=name.startswith("internal"))
    assert len(df) == n
    assert set(df["evaluation_mode"]) == {"automatic"}
    assert rep.get("twins_resolved") == 0 and rep.get("aut_only") == n
    assert set(df["magnitude_type"].dropna()) == {"aut.mag"}
    assert validate(df).ok


@pytest.mark.parametrize(
    "body",
    [load("internal_empty.json"), b'{"type":"FeatureCollection","features":null}'],
)
def test_empty(body: bytes) -> None:
    df, rep = read(*WIN_A, serve(body))
    assert len(df) == 0
    assert list(df.columns[: len(column_names())]) == list(column_names())
    assert validate(df).ok
    assert rep.get("events") == 0


# --------------------------------------------------------------------------
# Chunks, duplicates, padding, window cut
# --------------------------------------------------------------------------


def routed(by_start: Mapping[str, list[dict[str, Any]]]) -> FakeSession:
    """Serve the features whose time is in the requested [start, end]
    (inclusive at both ends, the worst case), from a pool."""
    pool = [f for feats in by_start.values() for f in feats]

    def route(path: str, q: dict[str, list[str]]) -> tuple[int, bytes]:
        a = pd.Timestamp(q["start_time"][0])
        b = pd.Timestamp(q["end_time"][0])
        hit = [f for f in pool if a <= pd.Timestamp(f["properties"]["time"] + "Z") <= b]
        return 200, collection(hit)

    return FakeSession(route)


def test_duplicate_row_across_chunks() -> None:
    # an event exactly at the chunk edge comes back from both requests
    feats = [
        feature(1, "2024-01-02T00:00:00", sk.SYSTEM_AUT),
        feature(1, "2024-01-02T00:00:00.4", sk.SYSTEM_PICKS),
        feature(2, "2024-01-01T12:00:00", sk.SYSTEM_AUT),
    ]
    session = routed({"x": feats})
    df, rep = read("2024-01-01", "2024-01-03", session, chunk=timedelta(days=1))
    assert rep.get("chunks") == 2 and len(session.urls) == 2
    assert rep.get("rows_fetched") == 4
    assert rep.get("duplicate_rows") == 1
    assert len(df) == 2 and df["event_id"].is_unique
    one = df.loc[df["event_id"] == "1"].iloc[0]
    assert one["evaluation_mode"] == "manual" and one["n_solutions"] == 2


def test_twin_split_across_chunks() -> None:
    # aut row in chunk 1, picks row in chunk 2: still one reviewed event
    feats = [
        feature(7, "2024-01-01T23:59:59.5", sk.SYSTEM_AUT),
        feature(7, "2024-01-02T00:00:01.2", sk.SYSTEM_PICKS),
    ]
    df, rep = read(
        "2024-01-01", "2024-01-03", routed({"x": feats}), chunk=timedelta(days=1)
    )
    assert len(df) == 1
    assert df["evaluation_mode"].iloc[0] == "manual"
    assert rep.get("twins_resolved") == 1 and rep.get("duplicate_rows") == 0


def test_pad_resolves_twin_outside_window() -> None:
    # the picks row lies 40 s before the window, its aut twin just inside
    feats = [
        feature(9, "2024-01-01T00:00:10", sk.SYSTEM_AUT),
        feature(9, "2023-12-31T23:59:30", sk.SYSTEM_PICKS),
    ]
    session = routed({"x": feats})
    no_pad, rep0 = read("2024-01-01", "2024-01-02", routed({"x": feats}))
    assert list(no_pad["evaluation_mode"]) == ["automatic"]  # mislabelled
    df, rep = read("2024-01-01", "2024-01-02", session, pad=timedelta(minutes=5))
    # with the pad the twin resolves to the reviewed row, which is outside
    # [start, end) and so is cut, as read_sil_catalog cuts by origin time
    assert len(df) == 0
    assert rep.get("twins_resolved") == 1 and rep.get("outside_window") == 1
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(session.urls[0]).query)
    assert q["start_time"] == ["2023-12-31T23:55:00Z"]
    assert q["end_time"] == ["2024-01-02T00:05:00Z"]


def test_window_is_half_open() -> None:
    feats = [
        feature(1, "2024-01-01T00:00:00", sk.SYSTEM_AUT),
        feature(2, "2024-01-02T00:00:00", sk.SYSTEM_AUT),
    ]
    df, rep = read("2024-01-01", "2024-01-02", routed({"x": feats}))
    assert list(df["event_id"]) == ["1"]
    assert rep.get("outside_window") == 1


def test_sentinels_and_ranges() -> None:
    feats = [
        feature(1, "2024-01-01T01:00:00", sk.SYSTEM_PICKS, m_mlw=-9.99, m_ml=1.2),
        feature(2, "2024-01-01T02:00:00", sk.SYSTEM_PICKS, m_mlw=-9.99, m_ml=-9.99),
        feature(3, "2024-01-01T03:00:00", sk.SYSTEM_AUT, m_autmag=12.6),
        feature(4, "2024-01-01T04:00:00", sk.SYSTEM_AUT, lat=None, lon=None),
        feature(5, "2024-01-01T05:00:00", "SIL something"),
    ]
    df, rep = read("2024-01-01", "2024-01-02", serve(collection(feats)))
    by = df.set_index("event_id")
    assert by.loc["1", "magnitude"] == pytest.approx(1.2)
    assert by.loc["1", "magnitude_type"] == "ML" and pd.isna(by.loc["1", "mag_mlw"])
    assert pd.isna(by.loc["2", "magnitude"]) and pd.isna(by.loc["2", "magnitude_type"])
    assert pd.isna(by.loc["3", "magnitude"]) and pd.isna(by.loc["3", "magnitude_type"])
    assert "4" not in by.index and "5" not in by.index
    assert rep.get("sentinel:m_mlw") == 2 and rep.get("sentinel:m_ml") == 1
    assert rep.get("out_of_range:magnitude") == 1
    assert rep.get("dropped_bad_position") == 1
    assert rep.get("dropped_unknown_system") == 1
    assert validate(df).ok


def test_unknown_event_code_maps_to_other() -> None:
    feats = [feature(1, "2024-01-01T01:00:00", sk.SYSTEM_PICKS, event_type="sf")]
    df, _ = read("2024-01-01", "2024-01-02", serve(collection(feats)))
    assert df["event_type"].iloc[0] == "other"
    assert df["event_type_code"].iloc[0] == "sf"


# --------------------------------------------------------------------------
# Filters: applied after resolution
# --------------------------------------------------------------------------


def test_depth_filter_acts_on_resolved_event() -> None:
    # 1352381: aut depth 0.2, picks depth 2.111. A per-row server filter
    # depth_max=1 would keep the aut twin and mislabel the event.
    session = serve(load("internal_20240822T00.json"))
    df, rep = read(*WIN_A, session, max_depth=1.0)
    assert "1352381" not in set(df["event_id"])
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(session.urls[0]).query)
    assert set(q) == {"start_time", "end_time"}  # nothing per-row was sent
    assert rep.get("filtered_client") == 7 - len(df)
    subsets = df.attrs[ATTRS_KEY]["subsets"]
    assert subsets[-1]["max_depth_km"] == 1.0


def test_magnitude_and_type_filters() -> None:
    session = serve(load("internal_20240822T00.json"))
    df, _ = read(*WIN_A, session, min_mag=0.5, event_types=["earthquake"])
    assert set(df["event_id"]) == {"1352384", "1352385", "1352386"}


def test_region_sent_buffered_and_cut_exactly() -> None:
    feats = json.loads(load("internal_20240822T00.json"))["features"]
    session = serve(collection(feats))
    bbox = (-22.39, 63.88, -22.37, 63.89)
    df, _ = read(*WIN_A, session, bbox=bbox)
    assert set(df["event_id"]) == {"1352381"}
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(session.urls[0]).query)
    lat = [float(v) for v in q["lat"]]
    lon = [float(v) for v in q["lon"]]
    assert len(lat) == len(lon) == 5  # closed ring, repeated keys
    assert (lat[0], lon[0]) == (lat[-1], lon[-1])
    assert min(lat) < 63.88 - 0.25 and max(lat) > 63.89 + 0.25  # ~30 km buffer
    assert min(lon) < -22.39 - 0.5 and max(lon) > -22.37 + 0.5


def test_polygon_filter() -> None:
    poly = [(-22.39, 63.88), (-22.37, 63.88), (-22.37, 63.89), (-22.39, 63.89)]
    df, _ = read(*WIN_A, serve(load("internal_20240822T00.json")), polygon=poly)
    assert set(df["event_id"]) == {"1352381"}
    assert "polygon" in df.attrs[ATTRS_KEY]["filters"]


# --------------------------------------------------------------------------
# Raw rows and resolution as functions
# --------------------------------------------------------------------------


def test_fetch_rows_keeps_both_solutions() -> None:
    rows, rep = sk.fetch_skjalftalisa_rows(
        *WIN_A, session=serve(load("internal_20240822T00.json")), pause_s=0
    )
    assert len(rows) == 13
    assert list(rows.columns[: len(sk.ROW_COLUMNS)]) == list(sk.ROW_COLUMNS)
    assert str(rows["event_id"].dtype) == "Int64"
    assert str(rows["time"].dtype) == "datetime64[ns, UTC]"
    picks = rows[rows["originating_system"] == sk.SYSTEM_PICKS]
    assert picks["strike"].notna().all()
    assert rep.get("rows_fetched") == 13


def test_resolve_solutions_pure() -> None:
    rows = sk.rows_from_features(
        [
            feature(1, "2024-01-01T00:00:00", sk.SYSTEM_AUT),
            feature(1, "2024-01-01T00:00:00", sk.SYSTEM_AUT),
            feature(1, "2024-01-01T00:00:01", sk.SYSTEM_PICKS),
            feature(2, "2024-01-01T00:01:00", sk.SYSTEM_PICKS),
        ]
    )
    out, counts = sk.resolve_solutions(rows)
    assert sorted(out["event_id"]) == ["1", "2"]
    assert counts["duplicate_rows"] == 1
    assert counts["twins_resolved"] == 1 and counts["picks_only"] == 1
    assert counts["events"] == 2
    assert set(out["evaluation_mode"]) == {"manual"}


# --------------------------------------------------------------------------
# Errors, retries, politeness
# --------------------------------------------------------------------------


def test_422_message_is_shown() -> None:
    with pytest.raises(sk.SkjalftalisaError, match="timezone info") as info:
        read(*WIN_A, serve(load("error_naive_time.json"), 422))
    assert info.value.status == 422


def test_count_out_of_bounds() -> None:
    with pytest.raises(sk.SkjalftalisaError, match="out of bounds"):
        sk.count_skjalftalisa(
            "1991-01-07",
            "2026-11-13",
            session=serve(load("error_count_out_of_bounds.json"), 422),
        )


def test_count() -> None:
    session = serve(load("internal_20240822T00_count.json"))
    assert sk.count_skjalftalisa(*WIN_A, session=session) == 7
    assert urllib.parse.urlsplit(session.urls[0]).path.endswith("/quakes/count")


def test_not_a_collection() -> None:
    with pytest.raises(sk.SkjalftalisaError, match="FeatureCollection"):
        read(*WIN_A, serve(b'{"hello": 1}'))
    with pytest.raises(sk.SkjalftalisaError, match="not JSON"):
        read(*WIN_A, serve(b"<html>"))


class Flaky:
    """Fails ``n`` times (no answer, or a status), then serves ``body``."""

    def __init__(self, n: int, body: bytes, status: int | None = None) -> None:
        self.n, self.body, self.status = n, body, status
        self.calls = 0

    def get(self, url: str, *, timeout: float) -> sk.HttpResponse:
        self.calls += 1
        if self.calls <= self.n:
            if self.status is None:
                raise sk.SkjalftalisaUnreachable("timed out", url=url)
            return sk.HttpResponse(self.status, b"busy", url)
        return sk.HttpResponse(200, self.body, url)


@pytest.mark.parametrize("status", [None, 503])
def test_retries_then_succeeds(status: int | None) -> None:
    session = Flaky(2, load("internal_empty.json"), status)
    df, rep = sk.read_skjalftalisa(
        *WIN_A, session=session, pause_s=0, pad=timedelta(0), retries=2
    )
    assert session.calls == 3 and rep.get("requests") == 3 and len(df) == 0


def test_unreachable_names_the_switch() -> None:
    session = Flaky(10, b"")
    with pytest.raises(sk.SkjalftalisaUnreachable, match="internal=False") as info:
        sk.read_skjalftalisa(*WIN_A, session=session, pause_s=0, retries=1)
    assert session.calls == 2
    assert info.value.url.startswith(sk.INTERNAL_URL)


def test_pause_between_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    session = routed({"x": []})
    sk.read_skjalftalisa(
        "2024-01-01",
        "2024-01-04",
        session=session,
        chunk=timedelta(days=1),
        pause_s=0.5,
        pad=timedelta(0),
    )
    assert len(session.urls) == 3
    assert len(slept) == 2 and all(0 < s <= 0.5 for s in slept)


def test_bad_windows() -> None:
    with pytest.raises(ValueError, match="not after"):
        read("2024-01-02", "2024-01-01", serve(b""))
    with pytest.raises(ValueError, match="first valid time"):
        read("1990-01-01", "1990-02-01", serve(b""))
    with pytest.raises(ValueError, match="chunk"):
        read(*WIN_A, serve(b""), chunk=timedelta(0))


def test_end_clamped_to_tomorrow() -> None:
    session = serve(load("internal_empty.json"))
    now = pd.Timestamp.now(tz="UTC")
    sk.read_skjalftalisa(
        now - pd.Timedelta(hours=1), now + pd.Timedelta(days=5), session=session
    )
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(session.urls[0]).query)
    tomorrow = now.normalize() + pd.Timedelta(days=1)
    assert q["end_time"] == [f"{tomorrow:%Y-%m-%dT%H:%M:%S}Z"]
    with pytest.raises(ValueError, match="last valid time"):
        read(tomorrow + pd.Timedelta(days=1), tomorrow + pd.Timedelta(days=2), session)


def test_start_clamped_to_api_start() -> None:
    session = serve(load("internal_empty.json"))
    sk.read_skjalftalisa("1990-12-01", "1991-01-08", session=session, pause_s=0)
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(session.urls[0]).query)
    assert q["start_time"] == ["1991-01-07T00:00:00Z"]


# --------------------------------------------------------------------------
# Detail
# --------------------------------------------------------------------------


def detail(
    name: str, event_id: int | str, status: int = 200, **kw: Any
) -> sk.SkjalftalisaDetail:
    return sk.read_skjalftalisa_detail(
        event_id, session=serve(load(name), status), **kw
    )


def test_detail_reviewed() -> None:
    d = detail("internal_detail_1352381.json", 1352381)
    assert d.event_id == "1352381"
    assert d.sil_path == "/2024/aug/22/00:05:00/10:05:430"
    assert d.url == f"{sk.INTERNAL_URL}/1352381"
    assert d.touched_aut == 1
    assert dict(d.magnitudes) == {"Mlw": 0.25, "Ml": -0.5, "Autmag": -1.2}
    picks = d.origin(sk.SYSTEM_PICKS)
    aut = d.origin(sk.SYSTEM_AUT)
    assert picks is not None and aut is not None
    assert d.preferred_origin is picks
    assert picks.time == pd.Timestamp("2024-08-22T00:10:04.454Z")
    assert picks.nearest_station_km == pytest.approx(3.9)  # was the string "3.9"
    assert picks.n_stations == 4
    assert picks.depth_err_km == pytest.approx(0.6)
    want_h = max(0.002 * 111.19, 0.003 * 111.19 * math.cos(math.radians(63.88752)))
    assert picks.h_err_km == pytest.approx(want_h)
    assert aut.row_num == 2 and aut.quality == pytest.approx(30.41)
    assert math.isnan(aut.h_err_km)
    # sp3: JSON inside a string, decoded; both Aki-Richards planes for T12b
    assert d.sp3 is not None and isinstance(d.sp3_raw, str)
    assert (d.sp3["fpa_strike_AR"], d.sp3["fpa_dip_AR"], d.sp3["fpa_slip_AR"]) == (
        341.0,
        74.0,
        170.0,
    )
    assert d.sp3["fpb_strike_AR"] == 74.0


def test_detail_to_catalog() -> None:
    row = detail("internal_detail_1352381.json", "SIL1352381").to_catalog()
    assert validate(row, strict=True).ok
    r = row.iloc[0]
    assert r["sil_path"] == "/2024/aug/22/00:05:00/10:05:430"
    assert r["evaluation_mode"] == "manual" and r["magnitude_type"] == "Mlw"
    assert r["mag_auto"] == pytest.approx(-1.2)
    assert r["depth_err_km"] == pytest.approx(0.6)
    assert r["h_err_km"] > 0
    assert r["moment_nm"] == pytest.approx(1.77e10)
    assert row.attrs[ATTRS_KEY]["detail"] is True


@pytest.mark.parametrize(
    ("name", "eid", "path", "row_num"),
    [
        ("internal_detail_1352541.json", 1352541, "/2024/aug/22/00:45:00/50:30:280", 8),
        (
            "internal_detail_1453678.json",
            1453678,
            "/2026/sep/10/09:10:00/12:50:312",
            None,
        ),
    ],
)
def test_detail_automatic(name: str, eid: int, path: str, row_num: int | None) -> None:
    d = detail(name, eid)
    assert d.sil_path == path
    assert d.sp3 is None and d.sp3_raw is None
    o = d.preferred_origin
    assert o is not None and o.evaluation_mode == "automatic"
    if row_num is not None:
        assert o.row_num == row_num
    row = d.to_catalog()
    assert row["evaluation_mode"].iloc[0] == "automatic"
    assert row["magnitude_type"].iloc[0] == "aut.mag"
    assert validate(row).ok


def test_detail_public_reduced() -> None:
    d = detail("public_detail_1352381.json", 1352381, internal=False)
    assert d.url == f"{sk.PUBLIC_URL}/1352381"
    assert d.sil_path == "/2024/aug/22/00:05:00/10:05:430"
    assert [t for t, _ in d.magnitudes] == ["M"]
    assert d.magnitude("M") == pytest.approx(0.3)
    assert d.preferred_origin is not None
    assert d.preferred_origin.originating_system is None
    with pytest.raises(ValueError, match="internal=True"):
        d.to_catalog()


def test_detail_404() -> None:
    with pytest.raises(sk.SkjalftalisaNotFound, match="Quake not found") as info:
        detail("error_detail_404.json", 999999999, 404)
    assert isinstance(info.value, LookupError) and info.value.status == 404


def test_detail_bad_id() -> None:
    with pytest.raises(ValueError, match="legacy integer id"):
        sk.read_skjalftalisa_detail("IMO2024abc", session=serve(b""))


def test_urllib_transport_unreachable() -> None:
    t = sk.UrllibTransport()
    with pytest.raises(sk.SkjalftalisaUnreachable):
        t.get("http://127.0.0.1:9/", timeout=2.0)


# --------------------------------------------------------------------------
# Live (one window, one count, one detail)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("GPS_LIVE_API") != "1", reason="live API test: set GPS_LIVE_API=1"
)
def test_live_window_a() -> None:
    df, rep = sk.read_skjalftalisa(*WIN_A)
    count = sk.count_skjalftalisa(*WIN_A)
    assert validate(df).ok
    assert df["event_id"].is_unique
    assert len(df) == count == 7
    assert rep.get("twins_resolved") >= 6
    r = df.loc[df["event_id"] == "1352381"].iloc[0]
    assert r["evaluation_mode"] == "manual" and r["magnitude"] == pytest.approx(0.25)
    d = sk.read_skjalftalisa_detail(1352381)
    assert d.sil_path == "/2024/aug/22/00:05:00/10:05:430"
    assert np.isfinite(d.preferred_origin.h_err_km)  # type: ignore[union-attr]
