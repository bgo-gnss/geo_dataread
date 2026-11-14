"""Tests for :mod:`geo_dataread.seismic.quakes_api` (ticket T10).

Offline: a fake session serves the recorded responses in
``tests/fixtures/quakes/`` (matched on path + parsed query, so parameter order
does not matter) and synthetic ones for splitting and errors. The default
urllib transport is disabled for the whole module, so no test can touch the
network by accident. One live test runs only with ``GPS_LIVE_API=1``.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from geo_dataread.seismic import quakes_api as qa
from geo_dataread.seismic.schema import SchemaError, validate

FIX = Path(__file__).parent / "fixtures" / "quakes"
NO_PAUSE = qa.FetchPolicy(pause_s=0.0, backoff_s=0.0)

A = ("2024-08-22T00:00:00Z", "2024-08-22T01:00:00Z")
B = ("2026-09-10T00:00:00Z", "2026-09-11T00:00:00Z")


# --------------------------------------------------------------------------
# Fake transport
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeResponse:
    status_code: int
    content: bytes


Key = tuple[str, frozenset[tuple[str, str]]]


def _key(url: str) -> Key:
    parts = urllib.parse.urlsplit(url)
    host_path = parts.netloc + parts.path
    return host_path, frozenset(urllib.parse.parse_qsl(parts.query))


def _manifest() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads((FIX / "manifest.json").read_text())
    return data


def _strip(query: frozenset[tuple[str, str]]) -> frozenset[tuple[str, str]]:
    """Drop what the recorder did not send (format) and normalise times."""
    out = set()
    for k, v in query:
        if k == "format":
            continue
        if k in ("start_time", "end_time"):
            v = pd.Timestamp(v).isoformat()
        out.add((k, v))
    return frozenset(out)


class FixtureSession:
    """Serves the recorded list and count responses; a count of a recorded
    list window is derived from the list when no count was recorded."""

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.headers: list[Mapping[str, str]] = []
        self.table: dict[Key, FakeResponse] = {}
        for e in _manifest():
            if e["format"] != "geojson" and e["kind"] != "count":
                continue
            body = (FIX / e["file"]).read_bytes()
            path, query = _key(e["url"])
            self.table[(path, _strip(query))] = FakeResponse(e["status"], body)
            if e["kind"] in ("list", "empty") and path.endswith("/events"):
                feats = json.loads(body)["features"] or []
                ckey = (path + "/count", _strip(query))
                self.table.setdefault(ckey, FakeResponse(200, str(len(feats)).encode()))

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> FakeResponse:
        self.urls.append(url)
        self.headers.append(headers)
        path, query = _key(url)
        hit = self.table.get((path, _strip(query)))
        if hit is None:
            raise AssertionError(f"no recorded response for {url}")
        return hit


class ScriptSession:
    """Answers through a function of (path, query dict)."""

    def __init__(
        self, answer: Callable[[str, dict[str, list[str]]], FakeResponse]
    ) -> None:
        self.answer = answer
        self.urls: list[str] = []

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> FakeResponse:
        self.urls.append(url)
        parts = urllib.parse.urlsplit(url)
        return self.answer(parts.path, urllib.parse.parse_qs(parts.query))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    monkeypatch.delenv(qa.ENV_ENDPOINT, raising=False)
    if request.node.name.startswith("test_live"):
        return

    def _blocked(self: object, url: str, **_: object) -> None:
        raise AssertionError(f"network access in an offline test: {url}")

    monkeypatch.setattr(qa.UrllibSession, "get", _blocked)


@pytest.fixture()
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(qa, "_sleep", calls.append)
    return calls


def _read(
    window: tuple[str, str],
    *,
    system: qa.System = "seiscomp",
    internal: bool | None = True,
    session: Any = None,
    **kw: Any,
) -> pd.DataFrame:
    return qa.read_quakes(
        window[0],
        window[1],
        system=system,
        internal=internal,
        session=session or FixtureSession(),
        policy=kw.pop("policy", NO_PAUSE),
        **kw,
    )


def _features(name: str) -> list[dict[str, Any]]:
    feats: list[dict[str, Any]] = json.loads((FIX / name).read_text())["features"]
    return feats


# --------------------------------------------------------------------------
# Mapping on the recorded responses
# --------------------------------------------------------------------------


def test_internal_seiscomp_window_a() -> None:
    df = _read(A)
    assert len(df) == 6
    assert validate(df).ok and not validate(df).issues
    first = df.iloc[0]
    assert df.index[0] == pd.Timestamp("2024-08-22T00:10:04.454Z")
    assert first["event_id"] == "IMO2024qnogqh"
    assert first["source"] == "quakes-seiscomp"
    assert first["latitude"] == pytest.approx(63.888)
    assert first["longitude"] == pytest.approx(-22.375)
    assert first["depth_km"] == pytest.approx(2.1)
    assert first["magnitude"] == pytest.approx(-0.58814)
    assert first["magnitude_type"] == "ML_SIL"
    assert first["n_phases"] == 16
    assert first["evaluation_mode"] == "manual"
    assert first["evaluation_status"] == "reviewed"
    assert pd.isna(first["event_type"]) and pd.isna(first["event_type_code"])
    assert first["updated_time"] == pd.Timestamp("2026-02-04T10:09:49.482935Z")
    # ML_SIL is not a SIL scale: no per-scale column is filled
    assert df[["mag_mlw", "mag_ml", "mag_auto"]].isna().all().all()
    for col in ("sil_path", "h_err_km", "depth_err_km", "moment_nm", "quality"):
        assert df[col].isna().all()


def test_day_b_status_and_mode_combine_freely() -> None:
    df = _read(B)
    assert len(df) == 48
    combo = (
        df.groupby(["evaluation_mode", "evaluation_status"], observed=True).size()
    ).to_dict()
    assert combo[("manual", "reviewed")] == 35
    assert combo[("automatic", "rejected")] == 7
    assert combo[("manual", "rejected")] == 4
    assert combo[("automatic", "reviewed")] == 2
    report = df.attrs["catalog"]["report"]["counts"]
    assert report["events_automatic"] == 9 and report["events_manual"] == 39
    assert set(df["event_type"].dropna()) == {
        "earthquake",
        "not locatable",
        "not existing",
    }
    assert (df["event_type"] == df["event_type_code"]).all()


def test_public_seiscomp_lacks_fields_and_rejected() -> None:
    df = _read(B, internal=False)
    assert len(df) == 37
    assert df["magnitude_type"].isna().all()
    assert df["evaluation_status"].isna().all()
    assert df["n_phases"].isna().all()
    assert df["magnitude"].notna().all()
    assert (df["evaluation_mode"] == "automatic").sum() == 2
    cat = df.attrs["catalog"]
    assert cat["filters"]["endpoint"] == "public"
    assert cat["report"]["counts"]["magnitude_type_missing"] == 37
    assert cat["notes"] and "magnitude_type" in cat["notes"][0]
    assert cat["location"] == "https://api.vedur.is/quakes/events"


def test_sil_window_a_scales() -> None:
    df = _read(A, system="sil")
    assert len(df) == 7
    assert (df["source"] == "quakes-sil").all()
    assert validate(df).ok
    first = df.iloc[0]
    assert first["event_id"] == "SIL1352381"
    assert first["magnitude_type"] == "Mlw"
    assert first["mag_mlw"] == pytest.approx(0.25)
    assert first["event_type"] == "earthquake"
    aut = df[df["magnitude_type"] == "aut.mag"]
    assert len(aut) >= 1
    assert (aut["mag_auto"] == aut["magnitude"]).all()
    assert aut["mag_mlw"].isna().all()
    assert (aut["evaluation_mode"] == "automatic").all()
    assert df["evaluation_status"].isna().all()
    assert df["updated_time"].isna().all()


def test_sil_public_equals_internal() -> None:
    a = _read(B, system="sil")
    b = _read(B, system="sil", internal=False)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 19 and (a["evaluation_mode"] == "automatic").all()


def test_empty_window_and_features_null() -> None:
    df = _read(("2024-08-22T00:00:00Z", "2024-08-22T00:02:00Z"))
    assert len(df) == 0 and validate(df).ok
    assert list(df.columns)[:3] == ["source", "event_id", "sil_path"]
    null = qa.features_to_frame([], system="sil")
    assert len(null) == 0
    sess = ScriptSession(
        lambda p, q: FakeResponse(
            200,
            (
                (FIX / "empty_before_1991.json").read_bytes()
                if p.endswith("/events")
                else b"3"
            ),
        )
    )
    df = _read(("1990-01-01", "1990-01-02"), system="sil", session=sess)
    assert len(df) == 0
    assert df.attrs["catalog"]["report"]["counts"]["count_mismatch"] == 1


def test_count_zero_skips_list() -> None:
    sess = FixtureSession()
    _read(("2024-08-22T00:00:00Z", "2024-08-22T00:02:00Z"), session=sess)
    assert len(sess.urls) == 1 and "/events/count?" in sess.urls[0]


def test_extras_keep_region() -> None:
    df = _read(A, extras=True)
    assert df["region"].iloc[0] == "Reykjanes"
    rep = validate(df)
    assert rep.ok and rep.kinds() == {"extra_column"}
    assert "region" not in _read(A).columns


def test_unknown_event_type_maps_to_other() -> None:
    feats = _features("internal_seiscomp_20240822T00.json")[:1]
    feats[0]["properties"]["type"] = "landslide"
    df = qa.features_to_frame(feats, system="seiscomp")
    assert df["event_type"].iloc[0] == "other"
    assert df["event_type_code"].iloc[0] == "landslide"


def test_unknown_magnitude_type_fails_loudly() -> None:
    feats = _features("internal_seiscomp_20240822T00.json")[:1]
    feats[0]["properties"]["magnitude_type"] = "mb"
    body = json.dumps({"type": "FeatureCollection", "features": feats}).encode()
    sess = ScriptSession(
        lambda p, q: FakeResponse(200, body if p.endswith("/events") else b"1")
    )
    with pytest.raises(SchemaError, match="magnitude_type"):
        _read(A, session=sess)


def test_out_of_range_and_bad_position() -> None:
    feats = _features("internal_seiscomp_20240822T00.json")[:3]
    feats[0]["properties"]["depth"] = 900.0
    feats[1]["geometry"]["coordinates"] = [-22.0, 95.0]
    from collections import Counter

    tally: Counter[str] = Counter()
    df = qa.features_to_frame(feats, system="seiscomp", tally=tally)
    assert len(df) == 2
    assert tally["out_of_range:depth_km"] == 1
    assert tally["dropped_bad_position"] == 1
    assert df["depth_km"].isna().sum() == 1


def test_null_magnitude_kept() -> None:
    feats = _features("internal_seiscomp_20240822T00.json")[:1]
    feats[0]["properties"].update(
        magnitude=None, magnitude_type=None, evaluation_mode="automatic", status=None
    )
    df = qa.features_to_frame(feats, system="seiscomp")
    assert validate(df).ok
    assert pd.isna(df["magnitude"].iloc[0]) and pd.isna(df["magnitude_type"].iloc[0])


def test_malformed_body_raises() -> None:
    sess = ScriptSession(
        lambda p, q: FakeResponse(200, b"[1, 2]" if p.endswith("/events") else b"2")
    )
    with pytest.raises(qa.QuakesApiError, match="FeatureCollection"):
        _read(A, session=sess)
    sess = ScriptSession(lambda p, q: FakeResponse(200, b'"six"'))
    with pytest.raises(qa.QuakesApiError, match="count"):
        _read(A, session=sess)


# --------------------------------------------------------------------------
# Requests: parameters, half-open window, provenance
# --------------------------------------------------------------------------


def test_request_parameters() -> None:
    sess = FixtureSession()
    df = _read(A, session=sess)
    count_url, list_url = sess.urls
    assert count_url.startswith(qa.INTERNAL_BASE + "/events/count?")
    assert list_url.startswith(qa.INTERNAL_BASE + "/events?")
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(list_url).query)
    assert q["start_time"] == ["2024-08-22T00:00:00.000Z"]
    assert q["end_time"] == ["2024-08-22T01:00:00.000Z"]
    assert q["size_min"] == ["-3"] and q["size_max"] == ["10"]
    assert q["depth_min"] == ["0"] and q["depth_max"] == ["50"]
    assert q["system"] == ["seiscomp"] and q["format"] == ["json"]
    assert "polygon" not in q
    assert sess.headers[0]["User-Agent"] == qa.USER_AGENT
    cat = df.attrs["catalog"]
    assert cat["source"] == "quakes-seiscomp"
    assert cat["window"] == {
        "start": "2024-08-22T00:00:00Z",
        "end": "2024-08-22T01:00:00Z",
    }
    assert cat["filters"]["depth_km"] == [0.0, 50.0]
    assert cat["report"]["counts"]["requests"] == 2


def test_optional_filters_sent() -> None:
    seen: list[dict[str, list[str]]] = []

    def answer(path: str, q: dict[str, list[str]]) -> FakeResponse:
        seen.append(q)
        return FakeResponse(200, b"0")

    _read(
        A,
        session=ScriptSession(answer),
        bbox=(-23.0, 63.7, -21.5, 64.1),
        evaluation_mode=["manual", "automatic"],
        event_type=["earthquake"],
        depth_min=1,
        depth_max=12.5,
        size_min=0.5,
        size_max=7,
    )
    q = seen[0]
    assert q["polygon"] == [
        "POLYGON((-23.000000 63.700000, -21.500000 63.700000, "
        "-21.500000 64.100000, -23.000000 64.100000, -23.000000 63.700000))"
    ]
    assert q["evaluation_mode"] == ["manual", "automatic"]
    assert q["type"] == ["earthquake"]
    assert q["depth_min"] == ["1"] and q["depth_max"] == ["12.5"]
    assert q["size_min"] == ["0.5"] and q["size_max"] == ["7"]


def test_polygon_forms() -> None:
    ring = [(-22.0, 63.8), (-21.9, 63.8), (-21.9, 63.9)]
    wkt = qa.polygon_wkt(ring)
    assert wkt is not None and wkt.endswith("-22.000000 63.800000))")
    assert qa.polygon_wkt([*ring, ring[0]]) == wkt
    assert (
        qa.polygon_wkt(" POLYGON((0 0, 1 0, 1 1, 0 0)) ")
        == "POLYGON((0 0, 1 0, 1 1, 0 0))"
    )
    assert qa.polygon_wkt() is None
    with pytest.raises(ValueError, match="not both"):
        qa.polygon_wkt(ring, (0, 0, 1, 1))
    with pytest.raises(ValueError, match="3 distinct"):
        qa.polygon_wkt([(0, 0), (1, 1)])
    with pytest.raises(ValueError, match="WKT"):
        qa.polygon_wkt("0 0, 1 1")
    with pytest.raises(ValueError, match="inverted"):
        qa.polygon_wkt(bbox=(1, 0, 0, 1))


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"depth_max": 60}, "depth_max"),
        ({"depth_min": -1}, "depth_min"),
        ({"size_min": -4}, "size_min"),
        ({"size_min": 3, "size_max": 2}, "size_min 3.0 > size_max"),
        ({"system": "bogus"}, "system"),
        ({"evaluation_mode": ["auto"]}, "evaluation_mode"),
        ({"event_type": ["quake"]}, "event_type"),
    ],
)
def test_bad_arguments_raise_before_any_request(kw: dict[str, Any], match: str) -> None:
    sess = ScriptSession(lambda p, q: FakeResponse(500, b""))
    with pytest.raises(ValueError, match=match):
        _read(A, session=sess, **kw)
    assert sess.urls == []


def test_end_before_start() -> None:
    with pytest.raises(ValueError, match="before start"):
        _read((A[1], A[0]))


def test_naive_times_are_utc() -> None:
    sess = FixtureSession()
    df = _read(("2024-08-22 00:00", "2024-08-22 01:00"), session=sess)
    assert len(df) == 6


def test_half_open_cut_and_dedupe() -> None:
    """Events outside [start, end) and repeated ids are dropped client side."""
    feats = _features("internal_seiscomp_20240822T00.json")
    late = json.loads(json.dumps(feats[0]))
    late["properties"]["time"] = "2024-08-22T01:00:00.000Z"
    late["properties"]["event_id"] = "IMOlate"
    body = json.dumps(
        {"type": "FeatureCollection", "features": [*feats, feats[0], late]}
    ).encode()
    sess = ScriptSession(
        lambda p, q: FakeResponse(200, body if p.endswith("/events") else b"8")
    )
    df = _read(A, session=sess)
    counts = df.attrs["catalog"]["report"]["counts"]
    assert len(df) == 6
    assert counts["duplicates_dropped"] == 1 and counts["outside_window"] == 1
    assert df["event_id"].is_unique


def test_end_clamped_to_tomorrow() -> None:
    seen: list[str] = []

    def answer(path: str, q: dict[str, list[str]]) -> FakeResponse:
        seen.append(q["end_time"][0])
        return FakeResponse(200, b"0")

    far = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)
    df = _read(
        (str(pd.Timestamp.now(tz="UTC").date()), str(far)),
        session=ScriptSession(answer),
    )
    assert pd.Timestamp(seen[0]) == pd.Timestamp.now(
        tz="UTC"
    ).normalize() + pd.Timedelta(days=1)
    assert df.attrs["catalog"]["window"]["end"].startswith(str(far.date()))


# --------------------------------------------------------------------------
# Window splitting
# --------------------------------------------------------------------------


def _synthetic_session(
    times: list[pd.Timestamp],
) -> tuple[ScriptSession, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """A server holding events at ``times``; it records each listed window."""
    listed: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def answer(path: str, q: dict[str, list[str]]) -> FakeResponse:
        lo = pd.Timestamp(q["start_time"][0])
        hi = pd.Timestamp(q["end_time"][0])
        inside = [t for t in times if lo <= t < hi]
        if path.endswith("/count"):
            return FakeResponse(200, str(len(inside)).encode())
        listed.append((lo, hi))
        feats = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-22.0, 64.0]},
                "properties": {
                    "event_id": f"E{times.index(t)}",
                    "time": t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "magnitude": 1.0,
                    "magnitude_type": "ML_SIL",
                    "depth": 5.0,
                    "type": "earthquake",
                    "evaluation_mode": "automatic",
                    "status": None,
                    "phases": 7,
                    "updated_time": "2026-01-01T00:00:00Z",
                },
            }
            for t in inside
        ]
        doc = {"type": "FeatureCollection", "features": feats}
        return FakeResponse(200, json.dumps(doc).encode())

    return ScriptSession(answer), listed


def test_bisection_tiles_the_window() -> None:
    lo = pd.Timestamp("2024-01-01T00:00:00Z")
    times = [lo + pd.Timedelta(minutes=7 * i) for i in range(100)]  # ~11.6 h
    sess, listed = _synthetic_session(times)
    policy = qa.FetchPolicy(max_events=10, pause_s=0, backoff_s=0)
    df = _read(("2024-01-01T00:00Z", "2024-01-02T00:00Z"), session=sess, policy=policy)
    assert len(df) == 100 and df["event_id"].is_unique
    spans = sorted(listed)
    for (_, h), (l2, _) in zip(spans, spans[1:], strict=False):
        assert h <= l2  # no overlap between listed windows
    counts = df.attrs["catalog"]["report"]["counts"]
    assert counts["splits"] >= 9
    assert counts["over_limit_windows"] == 0 and counts["count_mismatch"] == 0
    assert counts["list_requests"] == counts["windows_listed"] == len(listed)
    for a, b in listed:  # midpoints on whole milliseconds
        assert a == a.floor("ms") and b == b.floor("ms")


def test_min_window_stops_recursion() -> None:
    lo = pd.Timestamp("2024-01-01T00:00:00Z")
    times = [lo + pd.Timedelta(milliseconds=i) for i in range(50)]
    sess, listed = _synthetic_session(times)
    policy = qa.FetchPolicy(
        max_events=10, min_window=timedelta(minutes=1), pause_s=0, backoff_s=0
    )
    df = _read(("2024-01-01T00:00Z", "2024-01-01T00:04Z"), session=sess, policy=policy)
    assert len(df) == 50
    counts = df.attrs["catalog"]["report"]["counts"]
    assert counts["over_limit_windows"] == 1


def test_pause_between_requests(sleeps: list[float]) -> None:
    policy = qa.FetchPolicy(pause_s=0.7, backoff_s=0)
    _read(A, policy=policy)  # count + list
    assert sleeps == [0.7]


# --------------------------------------------------------------------------
# Errors, retries, endpoint
# --------------------------------------------------------------------------


def test_retry_on_5xx_then_success(sleeps: list[float]) -> None:
    calls = {"n": 0}

    def answer(path: str, q: dict[str, list[str]]) -> FakeResponse:
        calls["n"] += 1
        return (
            FakeResponse(503, b"busy") if calls["n"] <= 2 else FakeResponse(200, b"0")
        )

    policy = qa.FetchPolicy(pause_s=0, backoff_s=1.5, retries=3)
    df = _read(A, session=ScriptSession(answer), policy=policy)
    counts = df.attrs["catalog"]["report"]["counts"]
    assert counts["retries"] == 2 and counts["requests"] == 3
    assert sleeps == [1.5, 3.0]


def test_retries_exhausted() -> None:
    sess = ScriptSession(lambda p, q: FakeResponse(502, b"<html>bad gateway</html>"))
    policy = qa.FetchPolicy(pause_s=0, backoff_s=0, retries=2)
    with pytest.raises(qa.QuakesApiError, match="502.*after 2 retries") as info:
        _read(A, session=sess, policy=policy)
    assert info.value.status == 502
    assert len(sess.urls) == 3


def test_no_retry_on_4xx() -> None:
    body = (FIX / "error_bad_system.json").read_bytes()
    sess = ScriptSession(lambda p, q: FakeResponse(422, body))
    with pytest.raises(
        qa.QuakesApiError, match="Input should be 'sil' or 'seiscomp'"
    ) as info:
        _read(A, session=sess)
    assert info.value.status == 422 and len(sess.urls) == 1


def test_detail_texts() -> None:
    assert (
        qa._detail((FIX / "error_sil_detail_404.json").read_bytes())
        == "Event not found"
    )
    assert qa._detail(b"plain") == "plain"


class _Down:
    def __init__(self, exc: OSError) -> None:
        self.exc = exc
        self.n = 0

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> FakeResponse:
        self.n += 1
        raise self.exc


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError(OSError(-2, "Name or service not known")),
        TimeoutError("timed out"),
        ConnectionRefusedError(111, "refused"),
    ],
)
def test_internal_unreachable_says_how_to_switch(exc: OSError) -> None:
    down = _Down(exc)
    with pytest.raises(qa.QuakesApiUnreachable) as info:
        _read(A, session=down)
    msg = str(info.value)
    assert "vi-api.vedur.is" in msg
    assert "internal=False" in msg and f"{qa.ENV_ENDPOINT}=public" in msg
    assert "does not fall back" in msg
    assert down.n == 1  # no retry, no fallback request


def test_public_unreachable_has_no_switch_hint() -> None:
    with pytest.raises(qa.QuakesApiUnreachable) as info:
        _read(A, internal=False, session=_Down(TimeoutError("t")))
    assert "api.vedur.is" in str(info.value)
    assert "internal=False" not in str(info.value)


def test_endpoint_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    assert qa.resolve_endpoint() == (qa.INTERNAL_BASE, True)
    monkeypatch.setenv(qa.ENV_ENDPOINT, "public")
    assert qa.resolve_endpoint() == (qa.PUBLIC_BASE, False)
    assert qa.resolve_endpoint(True) == (qa.INTERNAL_BASE, True)
    monkeypatch.setenv(qa.ENV_ENDPOINT, " Internal ")
    assert qa.resolve_endpoint() == (qa.INTERNAL_BASE, True)
    assert qa.resolve_endpoint(False) == (qa.PUBLIC_BASE, False)
    monkeypatch.setenv(qa.ENV_ENDPOINT, "https://example.org")
    with pytest.raises(ValueError, match="internal' or 'public"):
        qa.resolve_endpoint()


def test_env_selects_public_in_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(qa.ENV_ENDPOINT, "public")
    df = _read(B, internal=None)
    assert len(df) == 37 and df.attrs["catalog"]["filters"]["endpoint"] == "public"


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        qa.FetchPolicy(max_events=0)
    with pytest.raises(ValueError):
        qa.FetchPolicy(min_window=timedelta(0))
    with pytest.raises(ValueError):
        qa.FetchPolicy(retries=-1)


def test_urllib_session_maps_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default transport returns HTTP errors as responses (not raised)."""
    import io

    monkeypatch.undo()  # restore the real UrllibSession.get

    def fake_urlopen(req: Any, timeout: float) -> Any:
        raise urllib.error.HTTPError(
            req.full_url, 404, "nf", {}, io.BytesIO(b'{"message":"Event not found"}')  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    resp = qa.UrllibSession().get("https://x.invalid/a", timeout=1, headers={})
    assert resp.status_code == 404 and b"not found" in resp.content


# --------------------------------------------------------------------------
# Live (opt-in)
# --------------------------------------------------------------------------


@pytest.fixture()
def live_window() -> Iterator[tuple[str, str]]:
    if os.environ.get("GPS_LIVE_API") != "1":
        pytest.skip("live Quakes API test: set GPS_LIVE_API=1")
    yield A


def test_live_internal_seiscomp(live_window: tuple[str, str]) -> None:
    """One count + one list on the internal endpoint (window A, 1 hour)."""
    df = qa.read_quakes(*live_window, system="seiscomp")
    assert validate(df).ok
    assert len(df) >= 6
    assert "IMO2024qnogqh" in set(df["event_id"])
    counts = df.attrs["catalog"]["report"]["counts"]
    assert counts["requests"] == 2
