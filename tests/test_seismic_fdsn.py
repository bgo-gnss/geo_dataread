"""Tests for :mod:`geo_dataread.seismic.fdsn` (ticket T13).

Offline: a fake fdsnws-event server (:class:`FakeFdsn`) holds the events of a
recorded window (``tests/fixtures/fdsn/``) and answers queries the way the
real service does (probed 2026-11-13/14): inclusive ``starttime``/``endtime``,
newest first unless ``orderby=time-asc``, 1-based ``offset`` (0 is a 400),
``limit``, 204 on no events, 400 ``invalid param`` on unknown parameters.
The default urllib transport is disabled for the module. Live tests run only
with ``GPS_LIVE_API=1``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from geo_dataread.seismic import fdsn
from geo_dataread.seismic import quakes_api as qa
from geo_dataread.seismic.schema import (
    SchemaError,
    auxiliary_plane,
    validate,
    validate_mechanisms,
)

FIX = Path(__file__).parent / "fixtures" / "fdsn"
QFIX = Path(__file__).parent / "fixtures" / "quakes"
NO_PAUSE = fdsn.FdsnPolicy(pause_s=0.0, backoff_s=0.0)

A = ("2024-08-22T00:00:00Z", "2024-08-22T01:00:00Z")
B = ("2026-09-10T00:00:00Z", "2026-09-11T00:00:00Z")

#: Query parameters the fake accepts (the real WADL set the reader may send).
KNOWN = {
    "starttime",
    "endtime",
    "minlatitude",
    "maxlatitude",
    "minlongitude",
    "maxlongitude",
    "latitude",
    "longitude",
    "minradius",
    "maxradius",
    "mindepth",
    "maxdepth",
    "minmagnitude",
    "maxmagnitude",
    "magnitudetype",
    "includeallorigins",
    "includeallmagnitudes",
    "includearrivals",
    "includepicks",
    "includefocalmechanism",
    "eventid",
    "limit",
    "offset",
    "orderby",
    "format",
}


# --------------------------------------------------------------------------
# Fake server
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeResponse:
    status_code: int
    content: bytes


def _error(status: int, title: str, detail: str) -> FakeResponse:
    body = f"Error {status}: {title}\n\n{detail}\n\nUsage details are available\n"
    return FakeResponse(status, body.encode())


@dataclass
class _Ev:
    event_id: str
    time: pd.Timestamp
    chunk: str  # text line or <event>…</event>


def _split_text(body: bytes) -> tuple[str, list[_Ev]]:
    lines = body.decode().splitlines()
    evs = []
    for ln in lines[1:]:
        cells = ln.split("|")
        evs.append(_Ev(cells[0], pd.Timestamp(cells[1], tz="UTC"), ln))
    return lines[0], evs


_EVENT_RE = re.compile(r"<event publicID=.*?</event>", re.S)


def _split_xml(body: bytes) -> tuple[tuple[str, str], list[_Ev]]:
    text = body.decode()
    chunks = _EVENT_RE.findall(text)
    head = text[: text.index("<event publicID=")]
    tail = text[text.rindex("</event>") + len("</event>") :]
    evs = []
    for c in chunks:
        parsed, _ = fdsn.parse_quakeml((head + c + tail).encode())
        assert parsed[0].time is not None
        evs.append(_Ev(parsed[0].event_id, parsed[0].time, c))
    return (head, tail), evs


class FakeFdsn:
    """Serves a window of recorded events, text and/or QuakeML."""

    def __init__(
        self,
        text: bytes | None = None,
        xml: bytes | None = None,
        details: Mapping[str, bytes] | None = None,
    ) -> None:
        self.urls: list[str] = []
        self.headers: list[Mapping[str, str]] = []
        self.text: tuple[str, list[_Ev]] | None = _split_text(text) if text else None
        self.xml = _split_xml(xml) if xml else None
        self.details = dict(details or {})
        self.after_request: Callable[[FakeFdsn], None] | None = None

    def query(self, url: str) -> dict[str, str]:
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> FakeResponse:
        self.urls.append(url)
        self.headers.append(headers)
        try:
            return self._answer(self.query(url))
        finally:
            if self.after_request is not None:
                self.after_request(self)

    def _answer(self, q: dict[str, str]) -> FakeResponse:
        for k in q:
            if k not in KNOWN:
                return _error(400, "Bad Request", f"invalid param: {k}")
        if "eventid" in q:
            body = self.details.get(q["eventid"])
            return FakeResponse(204, b"") if body is None else FakeResponse(200, body)
        if "offset" in q and int(q["offset"]) < 1:
            return _error(
                400,
                "Bad Request",
                "parameter not in domain [1,18446744073709551615]: offset",
            )
        is_text = q.get("format") == "text"
        src = self.text if is_text else self.xml
        assert src is not None, "no recorded window for this format"
        lo = pd.Timestamp(q["starttime"], tz="UTC")
        hi = pd.Timestamp(q["endtime"], tz="UTC")
        evs = [e for e in src[1] if lo <= e.time <= hi]
        evs.sort(key=lambda e: e.time, reverse=q.get("orderby") != "time-asc")
        off = int(q.get("offset", "1")) - 1
        evs = evs[off:]
        if "limit" in q:
            evs = evs[: int(q["limit"])]
        if not evs:
            return FakeResponse(204, b"")
        if is_text:
            head = src[0]
            assert isinstance(head, str)
            return FakeResponse(
                200, ("\n".join([head, *(e.chunk for e in evs)]) + "\n").encode()
            )
        frame = src[0]
        assert not isinstance(frame, str)
        h, t = frame
        return FakeResponse(200, (h + "".join(e.chunk for e in evs) + t).encode())


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    monkeypatch.delenv(fdsn.ENV_BASE, raising=False)
    monkeypatch.delenv(qa.ENV_ENDPOINT, raising=False)
    if request.node.name.startswith("test_live"):
        return

    def _blocked(self: object, url: str, **_: object) -> None:
        raise AssertionError(f"network access in an offline test: {url}")

    monkeypatch.setattr(qa.UrllibSession, "get", _blocked)


@pytest.fixture()
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(fdsn, "_sleep", calls.append)
    return calls


def _b(name: str) -> bytes:
    return (FIX / name).read_bytes()


def _text_a() -> FakeFdsn:
    return FakeFdsn(text=_b("text_20240822T00.txt"))


def _xml_a() -> FakeFdsn:
    return FakeFdsn(xml=_b("quakeml_20240822T00.xml"))


def _quakes_frame(name: str) -> pd.DataFrame:
    doc = json.loads((QFIX / name).read_text())
    df = qa.features_to_frame(doc["features"], system="seiscomp")
    assert validate(df).ok
    return df


def _counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = df.attrs["catalog"]["report"]["counts"]
    return counts


# --------------------------------------------------------------------------
# The fake behaves like the recorded server
# --------------------------------------------------------------------------


def test_fake_matches_recorded_limit2_asc() -> None:
    s = _text_a()
    q = "starttime=2024-08-22T00:00:00&endtime=2024-08-22T01:00:00&format=text"
    got = s.get(
        f"http://x/query?{q}&limit=2&orderby=time-asc", timeout=1, headers={}
    ).content
    assert (
        got.decode().split() == _b("text_20240822T00_limit2_asc.txt").decode().split()
    )
    full = s.get(f"http://x/query?{q}", timeout=1, headers={}).content
    assert full.decode().split() == _b("text_20240822T00.txt").decode().split()


# --------------------------------------------------------------------------
# Text format
# --------------------------------------------------------------------------


def test_text_needs_an_explicit_mode_rule() -> None:
    with pytest.raises(ValueError, match="no evaluation mode"):
        fdsn.read_fdsn(*A, format="text", session=_text_a(), policy=NO_PAUSE)


def test_text_window_a_paged() -> None:
    s = _text_a()
    pol = fdsn.FdsnPolicy(page_size=2, pause_s=0.0)
    df = fdsn.read_fdsn(
        *A, format="text", text_evaluation_mode="author", session=s, policy=pol
    )
    assert validate(df).ok
    assert list(df["event_id"]) == [
        "IMO2024qnogqh",
        "IMO2024qnohcr",
        "IMO2024qnooms",
        "IMO2024qnopim",
        "IMO2024qnoqbg",
        "IMO2024qnpdkq",
    ]
    assert df.index.is_monotonic_increasing
    assert (df["source"] == "fdsn-seiscomp").all()
    assert set(df["magnitude_type"]) == {"ML_SIL"}
    assert df["evaluation_status"].isna().all()
    assert df["event_type"].isna().all()  # back-imported: empty EventType
    row = df[df["event_id"] == "IMO2024qnogqh"].iloc[0]
    assert row["depth_km"] == pytest.approx(2.1)
    assert row["magnitude"] == pytest.approx(-0.5881404930458574)
    c = _counts(df)
    # pages: 2 + 2 + 2 full, then the tie-break page past the end is empty
    assert c["pages"] == 4 and c["empty_pages"] == 1
    assert c["duplicates_dropped"] == 0 and c["mode_inferred"] == 6
    for url in s.urls:
        q = s.query(url)
        assert q["orderby"] == "time-asc" and q["format"] == "text"
        assert q["limit"] == "2"
    assert "offset" not in s.query(s.urls[0])
    assert s.query(s.urls[1])["offset"] == "2"  # 1 seen at the anchor, 1-based
    assert df.attrs["catalog"]["filters"]["text_evaluation_mode"] == "author"


def test_text_author_rule_against_quakes_day_b() -> None:
    """The documented 5/48 error of the author rule (2026-09-10)."""
    s = FakeFdsn(text=_b("text_20260910.txt"))
    df = fdsn.read_fdsn(
        *B, format="text", text_evaluation_mode="author", session=s, policy=NO_PAUSE
    )
    q = _quakes_frame("internal_seiscomp_20260910.json")
    chk = fdsn.compare_with_quakes(df, q)
    assert chk.n_both == 48 and chk.n_fdsn == 48 and chk.n_quakes == 48
    assert chk.fields["evaluation_mode"]["differ"] == 5
    assert chk.fields["evaluation_mode"]["pairs"] == {"manual → automatic": 5}
    # SeisComP's "not locatable" leaves FDSN as QuakeML "other event"
    assert chk.fields["event_type"]["pairs"] == {"other → not locatable": 6}
    for col in ("time", "latitude", "longitude", "depth_km", "magnitude"):
        assert chk.fields[col]["differ"] == 0, col
    other = df[df["event_type"] == "other"]
    assert set(other["event_type_code"]) == {"other event"}


def test_empty_window_204() -> None:
    s = _xml_a()
    df = fdsn.read_fdsn(
        "2024-08-22T00:00:00", "2024-08-22T00:02:00", session=s, policy=NO_PAUSE
    )
    assert len(df) == 0 and validate(df, strict=True).ok
    assert _counts(df)["empty_pages"] == 1
    assert str(df.index.dtype) == "datetime64[ns, UTC]"


def test_half_open_window() -> None:
    """The server is inclusive at endtime; the reader cuts to [start, end)."""
    s = _xml_a()
    end = "2024-08-22T00:19:11.927Z"  # = IMO2024qnooms exactly
    df = fdsn.read_fdsn(A[0], end, session=s, policy=NO_PAUSE)
    assert "IMO2024qnooms" not in set(df["event_id"])
    assert len(df) == 2
    assert _counts(df)["outside_window"] == 1


# --------------------------------------------------------------------------
# QuakeML mapping
# --------------------------------------------------------------------------


def test_xml_window_a_mapping() -> None:
    s = _xml_a()
    df = fdsn.read_fdsn(*A, session=s, policy=NO_PAUSE, extras=True)
    assert len(df) == 6
    assert set(df["evaluation_mode"]) == {"manual"}
    assert set(df["evaluation_status"]) == {"reviewed"}
    r = df[df["event_id"] == "IMO2024qnogqh"].iloc[0]
    assert r["depth_km"] == pytest.approx(2.1)  # 2100 m
    assert r["depth_err_km"] == pytest.approx(0.6)  # 600 m = SIL Ddepth
    # SIL origin: degrees, with cos(lat), = sil_catalog's 0.22238 km
    assert r["h_err_km"] == pytest.approx(0.22238, abs=1e-5)
    assert r["n_phases"] == 16 and r["azimuthal_gap"] == pytest.approx(102)
    assert r["region"] == "Reykjanes" and r["n_origins"] == 1
    assert pd.isna(r["updated_time"]) and pd.isna(r["event_type"])
    assert np.isnan(r["mag_mlw"])  # needs include_all_magnitudes
    assert "format" not in s.query(s.urls[0])
    assert s.headers[0]["User-Agent"] == fdsn.USER_AGENT


def test_xml_matches_quakes_window_a() -> None:
    df = fdsn.read_fdsn(*A, session=_xml_a(), policy=NO_PAUSE)
    chk = fdsn.compare_with_quakes(
        df, _quakes_frame("internal_seiscomp_20240822T00.json")
    )
    assert chk.ok, chk.as_dict()
    assert chk.n_both == 6
    assert chk.fields["magnitude"]["max_abs"] < 1e-5


def test_detail_rusibb_all_origins() -> None:
    s = FakeFdsn(details={"IMO2026rusibb": _b("quakeml_IMO2026rusibb_all.xml")})
    cat = fdsn.read_fdsn_event("IMO2026rusibb", session=s, policy=NO_PAUSE, extras=True)
    q = s.query(s.urls[0])
    assert q["includeallorigins"] == "true" and q["includeallmagnitudes"] == "true"
    ev = cat.events.iloc[0]
    assert ev["evaluation_mode"] == "manual" and ev["evaluation_status"] == "reviewed"
    assert ev["magnitude_type"] == "ML_SIL"
    assert ev["magnitude"] == pytest.approx(1.6556706083052721)
    assert ev["n_phases"] == 50  # usedPhaseCount = Quakes "phases"
    assert ev["mag_auto"] == pytest.approx(1.5)  # Msil_auto on the SIL origin
    assert np.isnan(ev["h_err_km"]) and np.isnan(ev["depth_err_km"])  # LOCSAT
    assert ev["event_type"] == "earthquake"
    assert len(cat.origins) == 6 and int(cat.origins["preferred"].sum()) == 1
    assert len(cat.magnitudes) == 13 and int(cat.magnitudes["preferred"].sum()) == 1
    units = cat.origins.set_index("author")["latlon_unc_unit"]
    assert units["SIL"] == "deg" and set(units.drop("SIL")) == {"km"}
    assert cat.origins["depth_km"].max() < 10  # metres → km
    assert cat.report.counts["h_err_not_sil_model"] == 1
    assert len(cat.arrivals) == 0 and len(cat.mechanisms) == 0


def test_detail_qnogqh_sil_scales() -> None:
    s = FakeFdsn(details={"IMO2024qnogqh": _b("quakeml_IMO2024qnogqh_all.xml")})
    ev = fdsn.read_fdsn_event("IMO2024qnogqh", session=s, policy=NO_PAUSE).events
    r = ev.iloc[0]
    assert r["mag_mlw"] == pytest.approx(0.25)  # Msil_lw = SIL Mlw
    assert r["mag_ml"] == pytest.approx(-0.5)  # Msil_l = SIL ML
    assert r["magnitude_type"] == "ML_SIL"


def test_arrivals_join_picks() -> None:
    s = FakeFdsn(details={"IMO2026rusibb": _b("quakeml_IMO2026rusibb_arrivals.xml")})
    cat = fdsn.read_fdsn_event(
        "IMO2026rusibb",
        session=s,
        policy=NO_PAUSE,
        include_all_origins=False,
        include_all_magnitudes=False,
        include_arrivals=True,
    )
    q = s.query(s.urls[0])
    assert q["includearrivals"] == "true" and q["includepicks"] == "true"
    a = cat.arrivals
    assert len(a) == 5
    assert set(a["station"]) == {"gyg", "ste", "tag"}
    assert set(a["phase"]) == {"P", "S"}
    assert a["pick_time"].notna().all() and str(a["pick_time"].dtype).endswith("UTC]")
    assert (a["time_weight"] == 0).sum() == 1


def _with_mechanism(body: bytes, s1: float, d1: float, r1: float) -> bytes:
    s2, d2, r2 = (float(v) for v in auxiliary_plane(s1, d1, r1))
    fm = (
        '<focalMechanism publicID="smi:test/fm1"><nodalPlanes>'
        f"<nodalPlane1><strike><value>{s1}</value></strike><dip><value>{d1}</value>"
        f"</dip><rake><value>{r1}</value></rake></nodalPlane1>"
        f"<nodalPlane2><strike><value>{s2}</value></strike><dip><value>{d2}</value>"
        f"</dip><rake><value>{r2}</value></rake></nodalPlane2>"
        "</nodalPlanes><stationPolarityCount>12</stationPolarityCount>"
        "<misfit>0.1</misfit></focalMechanism>"
    )
    text = body.decode().replace("<preferredOriginID>", fm + "<preferredOriginID>", 1)
    return text.encode()


def test_focal_mechanism_parsed_if_present() -> None:
    body = _with_mechanism(_b("quakeml_IMO2024qnogqh_all.xml"), 30.0, 60.0, -90.0)
    s = FakeFdsn(details={"IMO2024qnogqh": body})
    cat = fdsn.read_fdsn_event(
        "IMO2024qnogqh", session=s, policy=NO_PAUSE, include_focal_mechanisms=True
    )
    assert s.query(s.urls[0])["includefocalmechanism"] == "true"
    m = cat.mechanisms
    assert len(m) == 1 and validate_mechanisms(m).ok
    r = m.iloc[0]
    assert r["method"] == "other" and r["n_polarities"] == 12
    assert (r["strike1"], r["dip1"], r["rake1"]) == (30.0, 60.0, -90.0)
    assert m.index[0] == cat.events.index[0]
    assert cat.report.counts["focal_mechanisms"] == 1


def test_no_focal_mechanism_in_recorded_details() -> None:
    for name in ("quakeml_IMO2024qnogqh_all.xml", "quakeml_IMO2026rusibb_all.xml"):
        _, _ = fdsn.parse_quakeml(_b(name))
        assert b"focalMechanism" not in _b(name)


def test_unknown_preferred_magnitude_type_fails_loudly() -> None:
    body = _b("quakeml_IMO2024qnogqh_all.xml").decode()
    pref = re.search(r"<preferredMagnitudeID>([^<]*)<", body)
    assert pref is not None
    body = body.replace(
        pref.group(1),
        "smi:local/imo/SILMAG/Origin#20240822001004.454000.123456#netMag.Msil_lw",
    )
    s = FakeFdsn(details={"IMO2024qnogqh": body.encode()})
    with pytest.raises(SchemaError):
        fdsn.read_fdsn_event("IMO2024qnogqh", session=s, policy=NO_PAUSE)


def test_not_quakeml_raises() -> None:
    with pytest.raises(fdsn.FdsnError, match="not XML"):
        fdsn.parse_quakeml(b"<html")
    with pytest.raises(fdsn.FdsnError, match="eventParameters"):
        fdsn.parse_quakeml(b"<root/>")


# --------------------------------------------------------------------------
# Paging
# --------------------------------------------------------------------------


def test_xml_paging_page_size_1() -> None:
    s = _xml_a()
    df = fdsn.read_fdsn(*A, session=s, policy=fdsn.FdsnPolicy(page_size=1, pause_s=0))
    assert len(df) == 6 and df["event_id"].is_unique
    assert _counts(df)["duplicates_dropped"] == 0


def _tied_text(n: int) -> bytes:
    head = "#" + "|".join(fdsn.TEXT_COLUMNS)
    rows = []
    for i in range(n):
        t = "2024-08-22T00:30:00.000000" if 1 <= i <= 4 else f"2024-08-22T00:{i:02d}:10"
        rows.append(
            f"EV{i:02d}|{t}|64.0|-21.0|5.0|SIL||VI|EV{i:02d}|ML_SIL|1.0|SIL|X|earthquake"
        )
    return ("\n".join([head, *rows]) + "\n").encode()


def test_paging_ties_across_pages() -> None:
    """Four events in the same microsecond, page size 2: the offset tie-break
    still returns each exactly once."""
    s = FakeFdsn(text=_tied_text(8))
    df = fdsn.read_fdsn(
        *A,
        format="text",
        text_evaluation_mode="author",
        session=s,
        policy=fdsn.FdsnPolicy(page_size=2, pause_s=0),
    )
    assert sorted(df["event_id"]) == [f"EV{i:02d}" for i in range(8)]


def test_paging_seen_event_leaves_mid_scan_no_skip() -> None:
    """After page 1, an event already served leaves the result set (deleted,
    or its origin time revised). A pure offset scan would then skip the first
    unseen event of the next page; keyset paging re-anchors on time and
    returns every event not yet served."""
    s = _xml_a()
    removed: list[str] = []

    def remove_first(srv: FakeFdsn) -> None:
        if removed or srv.xml is None:
            return
        first = min(srv.xml[1], key=lambda e: e.time)
        srv.xml[1].remove(first)
        removed.append(first.event_id)

    s.after_request = remove_first
    df = fdsn.read_fdsn(*A, session=s, policy=fdsn.FdsnPolicy(page_size=2, pause_s=0))
    assert removed == ["IMO2024qnogqh"]
    assert set(df["event_id"]) == {
        "IMO2024qnogqh",  # served on page 1 before it left
        "IMO2024qnohcr",
        "IMO2024qnooms",
        "IMO2024qnopim",
        "IMO2024qnoqbg",
        "IMO2024qnpdkq",
    }


def test_offset_scan_would_skip() -> None:
    """The control for the test above: the same removal under a plain
    growing offset loses IMO2024qnooms."""
    s = _xml_a()
    assert s.xml is not None
    base = "http://x/query?starttime=2024-08-22T00:00:00&endtime=2024-08-22T01:00:00"
    p1 = s.get(f"{base}&orderby=time-asc&limit=2", timeout=1, headers={})
    s.xml[1].remove(min(s.xml[1], key=lambda e: e.time))
    p2 = s.get(f"{base}&orderby=time-asc&limit=2&offset=3", timeout=1, headers={})
    got = {e.event_id for b in (p1, p2) for e in fdsn.parse_quakeml(b.content)[0]}
    assert "IMO2024qnooms" not in got


def test_max_pages_raises_not_truncates() -> None:
    with pytest.raises(fdsn.FdsnError, match="more than 2 pages"):
        fdsn.read_fdsn(
            *A,
            session=_xml_a(),
            policy=fdsn.FdsnPolicy(page_size=1, max_pages=2, pause_s=0),
        )


def test_pause_between_requests(sleeps: list[float]) -> None:
    fdsn.read_fdsn(*A, session=_xml_a(), policy=fdsn.FdsnPolicy(page_size=4))
    assert sleeps == [1.0]  # two pages → one pause


# --------------------------------------------------------------------------
# Errors and parameters
# --------------------------------------------------------------------------


class ScriptSession:
    def __init__(self, answers: list[FakeResponse | Exception]) -> None:
        self.answers = answers
        self.urls: list[str] = []

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> FakeResponse:
        self.urls.append(url)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


@pytest.mark.parametrize(
    ("name", "detail"),
    [
        ("error_bad_param.txt", "invalid param: bogus"),
        ("error_updatedafter.txt", "invalid param: updatedafter"),
        ("error_updateafter.txt", "filtering based on update time not supported"),
    ],
)
def test_recorded_400_detail(name: str, detail: str) -> None:
    s = ScriptSession([FakeResponse(400, _b(name))])
    with pytest.raises(fdsn.FdsnError, match=detail) as ei:
        fdsn.read_fdsn(*A, session=s, policy=NO_PAUSE)
    assert ei.value.status == 400


def test_recorded_404_nodata_is_empty() -> None:
    s = ScriptSession([FakeResponse(404, _b("error_nodata404.txt"))])
    df = fdsn.read_fdsn(*A, session=s, policy=NO_PAUSE)
    assert len(df) == 0


def test_5xx_retried_then_ok(sleeps: list[float]) -> None:
    ok = _xml_a().get(
        "http://x/query?starttime=2024-08-22T00:00:00&endtime=2024-08-22T01:00:00"
        "&orderby=time-asc",
        timeout=1,
        headers={},
    )
    s = ScriptSession([FakeResponse(503, b"busy"), ok])
    df = fdsn.read_fdsn(*A, session=s, policy=fdsn.FdsnPolicy(pause_s=0))
    assert len(df) == 6 and _counts(df)["retries"] == 1
    assert sleeps == [2.0]


def test_5xx_gives_up() -> None:
    s = ScriptSession([FakeResponse(500, b"x")] * 3)
    with pytest.raises(fdsn.FdsnError, match="after 2 retries"):
        fdsn.read_fdsn(
            *A, session=s, policy=fdsn.FdsnPolicy(pause_s=0, retries=2, backoff_s=0)
        )


def test_unreachable_no_fallback() -> None:
    s = ScriptSession([OSError("Name or service not known")])
    with pytest.raises(fdsn.FdsnUnreachable, match=fdsn.ENV_BASE):
        fdsn.read_fdsn(*A, session=s, policy=NO_PAUSE)
    assert len(s.urls) == 1


def test_filter_params_sent() -> None:
    s = _xml_a()
    fdsn.read_fdsn(
        *A,
        session=s,
        policy=NO_PAUSE,
        bbox=(-23.0, 63.5, -21.0, 64.5),
        min_depth_km=-2,
        max_depth_km=60,
        min_magnitude=-1,
        magnitude_type="ML_SIL",
        include_all_magnitudes=True,
    )
    q = s.query(s.urls[0])
    assert q["minlatitude"] == "63.5" and q["maxlongitude"] == "-21"
    assert q["mindepth"] == "-2" and q["maxdepth"] == "60"
    assert q["minmagnitude"] == "-1" and "maxmagnitude" not in q
    assert q["magnitudetype"] == "ML_SIL" and q["includeallmagnitudes"] == "true"
    s2 = _xml_a()
    fdsn.read_fdsn(*A, session=s2, policy=NO_PAUSE, radius=(63.9, -22.4, 0.2))
    q2 = s2.query(s2.urls[0])
    assert (q2["latitude"], q2["longitude"], q2["maxradius"]) == (
        "63.9",
        "-22.4",
        "0.2",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bbox": (0, 0, 1, 1), "radius": (0, 0, 1)},
        {"bbox": (1, 0, 0, 1)},
        {"radius": (0, 0)},
        {"min_depth_km": 10, "max_depth_km": 5},
        {"min_magnitude": float("nan")},
        {"format": "csv"},
        {
            "format": "text",
            "text_evaluation_mode": "author",
            "include_all_origins": True,
        },
    ],
)
def test_bad_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        fdsn.read_fdsn(*A, session=_xml_a(), policy=NO_PAUSE, **kwargs)


def test_end_before_start() -> None:
    with pytest.raises(ValueError, match="before start"):
        fdsn.read_fdsn(A[1], A[0], session=_xml_a(), policy=NO_PAUSE)


def test_resolve_base(monkeypatch: pytest.MonkeyPatch) -> None:
    assert fdsn.resolve_base() == fdsn.DEFAULT_BASE
    monkeypatch.setenv(fdsn.ENV_BASE, "http://other:8080/fdsnws/event/1/query")
    assert fdsn.resolve_base() == "http://other:8080/fdsnws/event/1"
    assert fdsn.resolve_base("https://a/b/") == "https://a/b"
    with pytest.raises(ValueError):
        fdsn.resolve_base("ftp://a")


def test_provenance() -> None:
    df = fdsn.read_fdsn(*A, session=_xml_a(), policy=NO_PAUSE)
    prov = df.attrs["catalog"]
    assert prov["source"] == "fdsn-seiscomp" and prov["schema_version"] == 3
    assert prov["window"] == {
        "start": "2024-08-22T00:00:00Z",
        "end": "2024-08-22T01:00:00Z",
    }
    assert prov["location"].endswith("/fdsnws/event/1/query")
    assert prov["report"]["counts"]["events"] == 6


# --------------------------------------------------------------------------
# Cross-check
# --------------------------------------------------------------------------


def test_compare_reports_unmatched_and_reasons() -> None:
    f = fdsn.read_fdsn(*A, session=_xml_a(), policy=NO_PAUSE)
    q = _quakes_frame("internal_seiscomp_20240822T00.json")
    f2 = f.copy()
    f2.loc[f2["event_id"] == "IMO2024qnogqh", "depth_km"] = 60.0
    f2.loc[f2["event_id"] == "IMO2024qnohcr", "magnitude"] = np.nan
    q2 = q[~q["event_id"].isin(["IMO2024qnogqh", "IMO2024qnohcr", "IMO2024qnooms"])]
    chk = fdsn.compare_with_quakes(f2, q2)
    d = chk.as_dict()
    assert d["only_fdsn"]["by_reason"] == {
        "depth outside Quakes bounds": 1,
        "no magnitude": 1,
        "unexplained": 1,
    }
    assert not chk.ok and chk.n_both == 3
    json.dumps(d)  # JSON-safe


def test_compare_field_diff_beyond_tolerance() -> None:
    f = fdsn.read_fdsn(*A, session=_xml_a(), policy=NO_PAUSE)
    q = _quakes_frame("internal_seiscomp_20240822T00.json")
    q2 = q.copy()
    q2.loc[q2["event_id"] == "IMO2024qnogqh", "magnitude"] += 0.01
    chk = fdsn.compare_with_quakes(f, q2)
    assert chk.fields["magnitude"]["differ"] == 1
    assert list(chk.diffs["field"]) == ["magnitude"]
    assert chk.diffs["delta"].iloc[0] == pytest.approx(-0.01, abs=1e-6)


def test_cross_check_quakes_offline() -> None:
    feats = json.loads((QFIX / "internal_seiscomp_20240822T00.json").read_text())

    def quakes_answer(url: str) -> FakeResponse:
        path = urllib.parse.urlsplit(url).path
        if path.endswith("/count"):
            return FakeResponse(200, str(len(feats["features"])).encode())
        return FakeResponse(200, json.dumps(feats).encode())

    class QS:
        def get(
            self, url: str, *, timeout: float, headers: Mapping[str, str]
        ) -> FakeResponse:
            return quakes_answer(url)

    chk = fdsn.cross_check_quakes(
        *A, fdsn_session=_xml_a(), quakes_session=QS(), policy=NO_PAUSE
    )
    assert chk.ok and chk.n_both == 6


# --------------------------------------------------------------------------
# Live (GPS_LIVE_API=1)
# --------------------------------------------------------------------------


@pytest.fixture()
def live() -> Iterator[None]:
    if os.environ.get("GPS_LIVE_API") != "1":
        pytest.skip("live FDSN test: set GPS_LIVE_API=1")
    yield


def test_live_fdsn_window_a(live: None) -> None:
    """One QuakeML page of window A (1 hour, 6 events)."""
    df = fdsn.read_fdsn(*A)
    assert validate(df).ok
    assert {"IMO2024qnogqh", "IMO2024qnohcr"} <= set(df["event_id"])
    assert _counts(df)["pages"] == 1


def test_dangling_preferred_magnitude_is_na_not_replaced() -> None:
    """As IMO2026wjdjzm (2026-11-13): preferredMagnitudeID names a magnitude
    the service does not export; the reader leaves NA and counts it."""
    body = _b("quakeml_IMO2024qnogqh_all.xml").decode()
    pref = re.search(r"<preferredMagnitudeID>([^<]*)<", body)
    assert pref is not None
    body = body.replace(
        f"<preferredMagnitudeID>{pref.group(1)}<",
        "<preferredMagnitudeID>smi:local/imo/SILCLEAN/Origin#x/netMag/ML_SIL<",
    )
    s = FakeFdsn(details={"IMO2024qnogqh": body.encode()})
    cat = fdsn.read_fdsn_event("IMO2024qnogqh", session=s, policy=NO_PAUSE)
    r = cat.events.iloc[0]
    assert np.isnan(r["magnitude"]) and pd.isna(r["magnitude_type"])
    assert cat.report.counts["preferred_magnitude_dangling"] == 1
    assert cat.report.counts["magnitude_missing"] == 1


def test_unknown_event_id_raises() -> None:
    with pytest.raises(fdsn.FdsnError, match="not found"):
        fdsn.read_fdsn_event("IMO2024nosuch", session=FakeFdsn(), policy=NO_PAUSE)


def test_read_fdsn_catalog_window() -> None:
    s = _xml_a()
    cat = fdsn.read_fdsn_catalog(
        A[0], "2024-08-22T00:19:11.927Z", session=s, policy=NO_PAUSE
    )
    q = s.query(s.urls[0])
    assert q["includeallorigins"] == "true" and q["includeallmagnitudes"] == "true"
    assert len(cat.events) == 2  # qnooms at the end is cut (half-open)
    kept = set(cat.events["event_id"])
    assert set(cat.origins["event_id"]) == kept and len(cat.origins) == 2
    assert set(cat.magnitudes["event_id"]) == kept
    assert cat.origins["preferred"].all()
    assert cat.events.attrs["catalog"]["filters"]["include_all_origins"] is True
