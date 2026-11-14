"""alertMap sources, association, ``read_alert_pgv`` and the sensor report (T7c).

Fixtures: ``tests/fixtures/alert`` (README there). No network and no mount:
the HTTP source runs on a stub session, the directory source on the fixture
tree or a ``tmp_path`` copy. Live tests run only with ``GPS_LIVE_API=1``
(and, for the mount, ``$GPS_ALERTMAP_DIR``).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from geo_dataread.seismic import alert_sensors as asen
from geo_dataread.seismic import alert_source as src
from geo_dataread.seismic import quakes_api as qa
from geo_dataread.seismic import sil_alert as sa

ALERT = Path(__file__).parent / "fixtures" / "alert"
SIL = ALERT / "sil"
GOLDEN_ID = "/2026/nov/12/21:50:00/53:43:797"
GOLDEN_DAY = date(2026, 11, 12)
#: The cleanaut.mag origin the Mpgv page used.
CLEANAUT = sa.Origin(pd.Timestamp("2026-11-12 21:53:40.5", tz="UTC"), 64.653, -17.481)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.name.startswith("test_live"):
        return  # live tests read the real configuration
    for var in (
        src.ENV_ALERTMAP_DIR,
        src.ENV_ALERTMAP_URL,
        src.ENV_ALERTMAP_CACHE,
        src.ENV_ALERTMAP_FALLBACK,
        asen.ENV_STATION_URL,
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(src, "_config_alertmap_dir", lambda: None)
    monkeypatch.setattr(src, "_sleep", lambda s: None)

    def _blocked(self: object, url: str, **_: object) -> None:
        raise AssertionError(f"network access in an offline test: {url}")

    monkeypatch.setattr(qa.UrllibSession, "get", _blocked)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@dataclass
class FakeResponse:
    status_code: int
    content: bytes


class FakeSession:
    """Serves files of a local tree as ``<base>/<relative path>``."""

    def __init__(
        self,
        root: Path,
        base: str = src.DEFAULT_ALERTMAP_URL,
        *,
        fail: Callable[[str], FakeResponse | None] | None = None,
    ) -> None:
        self.root, self.base, self.fail = root, base, fail
        self.urls: list[str] = []

    def get(
        self, url: str, *, timeout: float, headers: Mapping[str, str]
    ) -> FakeResponse:
        self.urls.append(url)
        assert headers["User-Agent"] == src.USER_AGENT
        if self.fail is not None:
            r = self.fail(url)
            if r is not None:
                return r
        assert url.startswith(self.base)
        p = self.root / url[len(self.base) :]
        if not p.is_file():
            return FakeResponse(404, b"not found")
        return FakeResponse(200, p.read_bytes())


def _event_text(header_time: str, alert_id: int = 76) -> str:
    """The golden 076.event with another number and header time."""
    lines = (ALERT / "2026/nov/12/076.event").read_text(encoding="latin-1").splitlines()
    tok = lines[0].split()
    t = pd.Timestamp(header_time)
    tok[0] = f"{alert_id:03d}"
    # Fixed English names (strftime is locale dependent; the weekday is unused).
    month = sa._MONTHS[t.month - 1].capitalize()
    tok[4:9] = ["Mon", month, str(t.day), f"{t:%H:%M:%S}", str(t.year)]
    return "\n".join([" ".join(tok), *lines[1:]]) + "\n"


def _write(root: Path, day: date, alert_id: int, text: str) -> None:
    d = root / src.alert_day_path(day)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{alert_id:03d}.event").write_text(text, encoding="latin-1")


def _index_html(day: date, rows: list[tuple[int, str]]) -> str:
    body = "".join(
        f'<tr><td>[<a href="{i:03d}.html">{t}</a>] \n</td><td> 66</td><td> 3307</td></tr>\n'
        for i, t in sorted(rows, reverse=True)
    )
    title = f"Alert Map {day.year} {sa._MONTHS[day.month - 1].capitalize()} {day.day}"
    return f"<html><title>{title}</title><body><table >\n{body}</table></body></html>"


# --------------------------------------------------------------------------
# Paths, ids and configuration
# --------------------------------------------------------------------------


def test_day_path_and_sil_ids(tmp_path: Path) -> None:
    assert src.alert_day_path(date(2016, 6, 5)) == "2016/jun/05"
    assert src.sil_event_day(GOLDEN_ID) == GOLDEN_DAY
    assert src.sil_event_day(GOLDEN_ID.lstrip("/")) == GOLDEN_DAY
    assert src.evb_path(GOLDEN_ID, tmp_path) == (
        tmp_path / "2026/nov/12/21:50:00/53:43:797.evb"
    )
    for bad in ("/2026/xyz/12/21:50:00/53:43:797", "2026-11-12", ""):
        with pytest.raises(ValueError, match="not a SIL event id"):
            src.sil_event_day(bad)


def test_alertmap_dirs_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    assert src.alertmap_dirs() == ()
    monkeypatch.setattr(src, "_config_alertmap_dir", lambda: str(b))
    assert src.alertmap_dirs() == (b,)
    monkeypatch.setenv(src.ENV_ALERTMAP_DIR, f"{a}{os.pathsep}{b}{os.pathsep}")
    assert src.alertmap_dirs() == (a, b)
    assert src.alertmap_dirs(str(b)) == (b,)
    assert src.alertmap_dirs([b, a]) == (b, a)


def test_url_and_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert src.alertmap_url() == src.DEFAULT_ALERTMAP_URL
    assert src.alertmap_url("http://x/alert") == "http://x/alert/"
    monkeypatch.setenv(src.ENV_ALERTMAP_URL, "http://hraun.vedur.is/ja/alert/")
    assert src.alertmap_url() == "http://hraun.vedur.is/ja/alert/"
    monkeypatch.setenv(src.ENV_ALERTMAP_CACHE, str(tmp_path / "c"))
    assert src.alertmap_cache_dir() == tmp_path / "c" / "hraun.vedur.is_ja_alert"
    assert src.alertmap_cache_dir(tmp_path / "x") == tmp_path / "x"
    monkeypatch.delenv(src.ENV_ALERTMAP_CACHE)
    monkeypatch.setenv("GPS_SIL_CACHE", str(tmp_path / "s"))
    assert src.alertmap_cache_dir(url="http://sil.vedur.is/alert/") == (
        tmp_path / "s" / "alertmap" / "sil.vedur.is_alert"
    )


def test_resolve_source_choice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert isinstance(src.resolve_alert_source(), src.HttpAlertSource)
    assert isinstance(src.resolve_alert_source("http"), src.HttpAlertSource)
    with pytest.raises(src.AlertSourceUnavailable, match="no alertMap root"):
        src.resolve_alert_source("directory")
    with pytest.raises(ValueError, match="expected 'auto'"):
        src.resolve_alert_source("ftp")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        src.resolve_alert_source(42)  # type: ignore[arg-type]
    monkeypatch.setenv(src.ENV_ALERTMAP_DIR, str(ALERT))
    d = src.resolve_alert_source()
    assert isinstance(d, src.DirectoryAlertSource) and d.fallback is None
    assert src.resolve_alert_source(d) is d
    monkeypatch.setenv(src.ENV_ALERTMAP_FALLBACK, "http")
    d2 = src.resolve_alert_source()
    assert isinstance(d2, src.DirectoryAlertSource)
    assert isinstance(d2.fallback, src.HttpAlertSource)
    monkeypatch.setenv(src.ENV_ALERTMAP_FALLBACK, "maybe")
    with pytest.raises(ValueError, match="GPS_ALERTMAP_FALLBACK"):
        src.resolve_alert_source()


def test_configured_missing_mount_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B2: a configured but absent mount raises; it never falls back to HTTP."""
    monkeypatch.setenv(src.ENV_ALERTMAP_DIR, str(tmp_path / "not-mounted"))
    with pytest.raises(src.AlertSourceUnavailable) as exc:
        src.resolve_alert_source()
    msg = str(exc.value)
    assert "not-mounted" in msg and "source='http'" in msg and "GPS_ALERTMAP_DIR" in msg
    monkeypatch.setenv(src.ENV_ALERTMAP_FALLBACK, "http")
    with pytest.raises(src.AlertSourceUnavailable):
        src.resolve_alert_source()


# --------------------------------------------------------------------------
# Directory source
# --------------------------------------------------------------------------


def test_directory_lists_from_files_and_reads() -> None:
    d = src.DirectoryAlertSource(ALERT)
    assert d.list_day(GOLDEN_DAY) == (1, 76)  # index.html / alert.mag ignored
    ev = d.read_event(GOLDEN_DAY, 76)
    assert ev.header.alert_id == 76 and len(ev.frame) == 66
    assert d.local_day_dir(GOLDEN_DAY) == ALERT / "2026/nov/12"
    with pytest.raises(src.AlertEventNotFound):
        d.read_event(GOLDEN_DAY, 2)
    with pytest.raises(src.AlertDayNotFound, match="2026/nov/13"):
        d.list_day(date(2026, 11, 13))
    assert isinstance(d, src.AlertSource)


def _two_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Primary with a real 2026 and a dangling 2016 symlink; archive with 2016."""
    primary, archive = tmp_path / "alert", tmp_path / "alert_archive"
    shutil.copytree(ALERT / "2026", primary / "2026")
    (primary / "2016").symlink_to(tmp_path / "unreachable" / "2016")
    (primary / "2020").symlink_to(tmp_path / "unreachable" / "2020")
    shutil.copytree(ALERT / "2016", archive / "2016")
    return primary, archive


def test_directory_years_across_roots(tmp_path: Path) -> None:
    """The first root where the year resolves wins; a dangling symlink never does."""
    primary, archive = _two_roots(tmp_path)
    d = src.DirectoryAlertSource(f"{primary}{os.pathsep}{archive}")
    assert d.year_dir(2026) == primary / "2026"
    assert d.year_dir(2016) == archive / "2016"
    assert d.list_day(date(2016, 6, 15)) == (57,)
    assert d.read_event(date(2016, 6, 15), 57).header.n_stations == 14
    with pytest.raises(src.AlertYearUnavailable) as exc:
        d.list_day(date(2020, 6, 25))
    msg = str(exc.value)
    assert "symlink target unreachable" in msg
    assert str(archive / "2020") + " (missing)" in msg
    assert "GPS_ALERTMAP_FALLBACK=http" in msg and "GPS_ALERTMAP_DIR" in msg


def test_directory_year_unavailable_without_archive(tmp_path: Path) -> None:
    primary, _ = _two_roots(tmp_path)
    d = src.DirectoryAlertSource(primary)
    with pytest.raises(src.AlertYearUnavailable, match="unreachable"):
        d.list_day(date(2016, 6, 15))
    # Not an empty listing: the error is a FileNotFoundError, not a LookupError.
    assert issubclass(src.AlertYearUnavailable, FileNotFoundError)
    assert not issubclass(src.AlertYearUnavailable, LookupError)


def test_directory_http_fallback_per_year(tmp_path: Path) -> None:
    primary, _ = _two_roots(tmp_path)
    sess = FakeSession(ALERT)
    http = src.HttpAlertSource(session=sess, use_cache=False)
    d = src.DirectoryAlertSource(primary, fallback=http)
    assert d.list_day(date(2016, 6, 15))[-1] == 66  # from the 2016 index.html
    assert d.read_event(date(2016, 6, 15), 57).header.alert_id == 57
    assert d.fallback_years == {2016}
    assert d.list_day(GOLDEN_DAY) == (1, 76)  # 2026 still from the mount
    assert all("/2016/" in u for u in sess.urls)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")
def test_directory_permission_denied_is_an_error(tmp_path: Path) -> None:
    """Like archive 2011/may: an unreadable month is an error, not an empty day."""
    root = tmp_path / "alert"
    _write(root, date(2011, 5, 10), 1, _event_text("2011-05-10 01:00:00", 1))
    month = root / "2011" / "may"
    month.chmod(0)
    try:
        d = src.DirectoryAlertSource(root)
        with pytest.raises(src.AlertAccessDenied, match="permission") as exc:
            d.list_day(date(2011, 5, 10))
        assert isinstance(exc.value, PermissionError)
        with pytest.raises(src.AlertAccessDenied):
            src.read_alert_pgv(
                sa.Origin(pd.Timestamp("2011-05-10 00:59:55", tz="UTC"), 64.0, -19.0),
                source=d,
            )
    finally:
        month.chmod(0o755)


def test_directory_source_rejects_bad_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no root"):
        src.DirectoryAlertSource([])
    with pytest.raises(src.AlertSourceUnavailable, match="not found"):
        src.DirectoryAlertSource([ALERT, tmp_path / "gone"])
    src.DirectoryAlertSource(tmp_path / "gone", check=False)


# --------------------------------------------------------------------------
# HTTP source
# --------------------------------------------------------------------------


def _http_tree(tmp_path: Path) -> Path:
    """A served tree: 2016-06-15 (real index, one real file) + 2026-11-12."""
    served = tmp_path / "served"
    shutil.copytree(ALERT / "2016", served / "2016")
    d = served / "2026/nov/12"
    d.mkdir(parents=True)
    shutil.copy(ALERT / "2026/nov/12/076.event", d / "076.event")
    shutil.copy(ALERT / "2026/nov/12/001.event", d / "001.event")
    (d / "index.html").write_text(
        _index_html(GOLDEN_DAY, [(1, "00:00:57"), (76, "21:53:45")]), encoding="latin-1"
    )
    return served


def test_http_lists_from_index_never_probes(tmp_path: Path) -> None:
    served = _http_tree(tmp_path)
    sess = FakeSession(served)
    h = src.HttpAlertSource(
        session=sess, cache_dir=tmp_path / "cache", today=date(2026, 11, 14)
    )
    ids = h.list_day(date(2016, 6, 15))
    assert ids == tuple(range(1, 67))
    assert sess.urls == ["http://sil.vedur.is/alert/2016/jun/15/index.html"]
    ev = h.read_event(date(2016, 6, 15), 57)
    assert ev.header.n_stations == 14
    with pytest.raises(src.AlertEventNotFound, match="058.event"):
        h.read_event(date(2016, 6, 15), 58)  # listed, not served
    with pytest.raises(src.AlertDayNotFound, match="2016/jun/16/index.html"):
        h.list_day(date(2016, 6, 16))
    assert isinstance(h, src.AlertSource)


def test_http_cache_finished_days_only(tmp_path: Path) -> None:
    served = _http_tree(tmp_path)
    cache = tmp_path / "cache"
    sess = FakeSession(served)
    h = src.HttpAlertSource(session=sess, cache_dir=cache, today=date(2026, 11, 13))
    h.list_day(date(2016, 6, 15))
    h.read_event(date(2016, 6, 15), 57)
    assert (cache / "2016/jun/15/index.html").read_bytes() == (
        served / "2016/jun/15/index.html"
    ).read_bytes()
    assert (cache / "2016/jun/15/057.event").is_file()
    # 2026-11-12 is younger than recent_days=2: memory only, never on disk.
    assert h.list_day(GOLDEN_DAY) == (1, 76)
    h.read_event(GOLDEN_DAY, 76)
    assert not (cache / "2026").exists()
    n = len(sess.urls)
    h.read_event(GOLDEN_DAY, 76)
    assert len(sess.urls) == n  # in-memory hit
    # A new instance reads the finished day from the cache, no request.
    sess2 = FakeSession(served)
    h2 = src.HttpAlertSource(session=sess2, cache_dir=cache, today=date(2026, 11, 13))
    assert h2.read_event(date(2016, 6, 15), 57).header.alert_id == 57
    assert sess2.urls == []
    assert not list(cache.rglob("*.tmp"))
    assert (cache / src.CACHE_MARKER).is_file()


def test_http_cache_off_and_recent_days(tmp_path: Path) -> None:
    served = _http_tree(tmp_path)
    with pytest.raises(ValueError, match="recent_days"):
        src.HttpAlertSource(recent_days=0)
    h = src.HttpAlertSource(session=FakeSession(served), use_cache=False)
    assert h.cache_dir is None
    h.read_event(date(2016, 6, 15), 57)
    assert not (tmp_path / "cache").exists()


def test_http_errors(tmp_path: Path) -> None:
    served = _http_tree(tmp_path)

    def boom(url: str) -> FakeResponse | None:
        raise OSError("Name or service not known")

    h = src.HttpAlertSource(session=FakeSession(served, fail=boom), use_cache=False)
    with pytest.raises(src.AlertSourceUnavailable, match="IMO network"):
        h.list_day(GOLDEN_DAY)

    def five(url: str) -> FakeResponse | None:
        return FakeResponse(503, b"busy")

    h = src.HttpAlertSource(session=FakeSession(served, fail=five), use_cache=False)
    with pytest.raises(src.AlertSourceError, match="HTTP 503"):
        h.list_day(GOLDEN_DAY)


def test_http_local_day_dir(tmp_path: Path) -> None:
    served = _http_tree(tmp_path)
    cache = tmp_path / "cache"
    h = src.HttpAlertSource(
        session=FakeSession(served), cache_dir=cache, today=date(2026, 11, 13)
    )
    recent = h.local_day_dir(GOLDEN_DAY)
    assert recent == cache / "_recent/2026/nov/12"
    assert sorted(p.name for p in recent.glob("*.event")) == ["001.event", "076.event"]
    h2 = src.HttpAlertSource(
        session=FakeSession(served), cache_dir=cache, today=date(2027, 1, 1)
    )
    done = h2.local_day_dir(GOLDEN_DAY)
    assert done == cache / "2026/nov/12"
    assert (done / "076.event").read_bytes() == (
        served / "2026/nov/12/076.event"
    ).read_bytes()


def test_http_pause_between_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    served = _http_tree(tmp_path)
    sleeps: list[float] = []
    monkeypatch.setattr(src, "_sleep", sleeps.append)
    h = src.HttpAlertSource(session=FakeSession(served), use_cache=False, pause_s=0.2)
    h.list_day(GOLDEN_DAY)
    h.read_event(GOLDEN_DAY, 76)
    assert sleeps == [0.2] and h.requests == 2


# --------------------------------------------------------------------------
# Methods and association (pure)
# --------------------------------------------------------------------------


def test_methods() -> None:
    assert sa.resolve_method("page") is sa.PAGE_METHOD
    assert sa.PAGE_METHOD.settings == sa.DEFAULT_PGV and sa.PAGE_METHOD.require_fit
    leg = sa.resolve_method("legacy")
    assert leg.window_s == 30.0 and not leg.require_fit
    assert leg.settings.distance == "evb" and leg.settings.inverse == "farfield"
    assert (
        leg.settings.exclude_class_codes == () and leg.settings.exclude_stations == ()
    )
    m = sa.resolve_method("page", window_s=10, require_fit=False)
    assert m.window_s == 10.0 and not m.require_fit and m.name == "page"
    with pytest.raises(ValueError, match="unknown PGV method"):
        sa.resolve_method("hraun")
    with pytest.raises(ValueError, match="window_s"):
        sa.AlertMethod("x", sa.DEFAULT_PGV, window_s=-1, require_fit=False)


def test_association_window_from_evb_and_origin() -> None:
    evb = sa.read_evb(SIL / "2026/nov/12/21:50:00/53:43:797.evb")
    lo, hi = sa.association_window(evb.origin, evb, 30)
    arr = evb.phases["arrival_time"]
    assert lo == arr.min() - pd.Timedelta(seconds=30)
    assert hi == arr.max() + pd.Timedelta(seconds=30)
    lo, hi = sa.association_window(CLEANAUT, None, 150)
    assert (hi - lo) == pd.Timedelta(seconds=300) and lo + pd.Timedelta(
        seconds=150
    ) == CLEANAUT.time


def _ev(text: str) -> sa.AlertEvent:
    return sa.parse_alert_event(text, strict=False)


def test_associate_chooses_closest_not_first() -> None:
    late = _ev(_event_text("2026-11-12 21:54:00", 70))
    right = _ev(_event_text("2026-11-12 21:53:45", 76))
    window = sa.association_window(CLEANAUT, None, 150)
    a = sa.associate_alert(
        CLEANAUT,
        [(GOLDEN_DAY, late), (GOLDEN_DAY, right)],
        window=window,
        require_fit=False,
    )
    assert a.found and a.chosen is right and a.chosen_day == GOLDEN_DAY
    c = a.candidates
    assert list(c.columns) == list(sa.CANDIDATE_COLUMNS)
    assert c.loc[c["chosen"], "alert_id"].tolist() == [76]
    assert c["in_window"].all()
    # With the fit required, the shifted copy fails its travel times.
    a2 = sa.associate_alert(
        CLEANAUT, [(GOLDEN_DAY, late), (GOLDEN_DAY, right)], window=window
    )
    assert a2.chosen is right and a2.fit is not None and a2.fit.label == "61/66 (92%)"
    assert not bool(a2.candidates.set_index("alert_id").at[70, "accepted"])


def test_associate_tie_is_deterministic() -> None:
    a1 = _ev(_event_text("2026-11-12 21:53:45", 5))
    a2 = _ev(_event_text("2026-11-12 21:53:45", 3))
    window = sa.association_window(CLEANAUT, None, 30)
    for order in ([a1, a2], [a2, a1]):
        a = sa.associate_alert(
            CLEANAUT, [(GOLDEN_DAY, e) for e in order], window=window, require_fit=False
        )
        assert a.chosen is not None and a.chosen.header.alert_id == 3


def test_associate_outside_window_and_rejected() -> None:
    far = _ev(_event_text("2026-11-12 22:30:00", 80))
    small = sa.read_alert_event(ALERT / "2026/nov/12/001.event")
    window = sa.association_window(CLEANAUT, None, 150)
    a = sa.associate_alert(
        CLEANAUT, [(GOLDEN_DAY, far), (GOLDEN_DAY, small)], window=window
    )
    assert not a.found and a.fit is None
    assert not a.candidates["in_window"].any()
    assert a.candidates["n_fit"].isna().all()
    with pytest.raises(ValueError, match="before start"):
        sa.associate_alert(CLEANAUT, [], window=(window[1], window[0]))


def test_associate_across_midnight(tmp_path: Path) -> None:
    """The legacy time-of-day window broke here; full datetimes do not."""
    root = tmp_path / "alert"
    origin = sa.Origin(pd.Timestamp("2026-11-12 23:59:59.5", tz="UTC"), 64.653, -17.481)
    _write(root, GOLDEN_DAY, 90, _event_text("2026-11-12 23:40:00", 90))  # decoy
    _write(root, date(2026, 11, 13), 1, _event_text("2026-11-13 00:00:04", 1))
    r = src.read_alert_pgv(origin, source=src.DirectoryAlertSource(root))
    assert r.found and r.association.chosen_day == date(2026, 11, 13)
    assert r.result is not None and r.result.alert_id == 1
    assert r.result.n_kept == 54 and round(r.magnitude, 2) == 4.15
    assert set(r.association.candidates["alert_day"]) == {"2026-11-12", "2026-11-13"}


# --------------------------------------------------------------------------
# read_alert_pgv
# --------------------------------------------------------------------------


@pytest.mark.parametrize("origin", [None, CLEANAUT], ids=["evb-origin", "cleanaut"])
def test_read_alert_pgv_page_golden(origin: sa.Origin | None) -> None:
    r = src.read_alert_pgv(
        GOLDEN_ID, source=src.DirectoryAlertSource(ALERT), sil_path=SIL, origin=origin
    )
    assert r.found and r.method.name == "page" and r.event_id == GOLDEN_ID
    assert r.result is not None
    s = r.result.summary()
    assert (s["n_alert"], s["n_kept"], s["alertfit"]) == (66, 54, "61/66 (92%)")
    assert round(s["magnitude"], 2) == 4.15 and round(s["robust_mean"], 2) == 4.17
    assert s["magnitude_type"] == "Mpgv_alert"
    assert r.association.missing_days == () and r.association.errors == ()
    st = r.stations
    assert list(st.columns) == list(sa.PGV_STATION_COLUMNS)
    assert set(st.loc[~st["kept"], "drop_reason"]) >= {
        "excluded_class_code",
        "excluded_station",
    }
    assert (st["event_id"] == GOLDEN_ID).all()


def test_read_alert_pgv_legacy_golden() -> None:
    r = src.read_alert_pgv(
        GOLDEN_ID, method="legacy", source=src.DirectoryAlertSource(ALERT), sil_path=SIL
    )
    assert r.result is not None and r.result.distance == "evb"
    evb_sta = set(r.evb.station_distances().index) if r.evb else set()
    kept = set(r.stations.loc[r.stations["kept"], "station"])
    assert kept == evb_sta & set(r.stations["station"])
    assert r.result.n_kept == 29
    assert r.magnitude == pytest.approx(4.134124, abs=1e-6)


def test_read_alert_pgv_not_found_and_missing_day(tmp_path: Path) -> None:
    root = tmp_path / "alert"
    _write(root, GOLDEN_DAY, 1, (ALERT / "2026/nov/12/001.event").read_text())
    r = src.read_alert_pgv(
        GOLDEN_ID, source=src.DirectoryAlertSource(root), sil_path=SIL
    )
    assert not r.found and r.result is None and r.stations.empty
    assert r.magnitude != r.magnitude  # NaN
    o = sa.Origin(pd.Timestamp("2026-11-20 12:00", tz="UTC"), 64.0, -19.0)
    r2 = src.read_alert_pgv(o, source=src.DirectoryAlertSource(root))
    assert r2.association.missing_days == ("2026-11-20",)


def test_read_alert_pgv_reports_bad_files(tmp_path: Path) -> None:
    root = tmp_path / "alert"
    shutil.copytree(ALERT / "2026/nov/12", root / "2026/nov/12")
    (root / "2026/nov/12/002.event").write_text("garbage\n")
    r = src.read_alert_pgv(
        GOLDEN_ID, source=src.DirectoryAlertSource(root), sil_path=SIL
    )
    assert r.found
    assert len(r.association.errors) == 1 and "#002" in r.association.errors[0]


def test_read_alert_pgv_argument_errors() -> None:
    d = src.DirectoryAlertSource(ALERT)
    with pytest.raises(ValueError, match="distances from the evb"):
        src.read_alert_pgv(CLEANAUT, method="legacy", source=d)
    with pytest.raises(TypeError):
        src.read_alert_pgv(42, source=d)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        src.read_alert_pgv("/2026/nov/12/21:50:00/00:00:000", source=d, sil_path=SIL)


def test_read_alert_pgv_over_http(tmp_path: Path) -> None:
    served = _http_tree(tmp_path)
    sess = FakeSession(served)
    h = src.HttpAlertSource(
        session=sess, cache_dir=tmp_path / "c", today=date(2026, 11, 13)
    )
    r = src.read_alert_pgv(GOLDEN_ID, source=h, sil_path=SIL)
    assert r.result is not None and r.result.n_kept == 54
    assert r.source == src.DEFAULT_ALERTMAP_URL
    assert sorted(u.rsplit("/", 1)[1] for u in sess.urls) == [
        "001.event",
        "076.event",
        "index.html",
    ]


# --------------------------------------------------------------------------
# FDSN sensor cross-check (B4)
# --------------------------------------------------------------------------

CHANNELS = ALERT / "fdsn" / "station_channels_Z.txt"


def test_parse_fdsn_channels() -> None:
    ch = asen.parse_fdsn_channels(CHANNELS.read_text(encoding="utf-8"))
    assert list(ch.columns) == list(asen.FDSN_CHANNEL_COLUMNS)
    hei = ch[ch["station"] == "hei"]
    assert len(hei) == 3 and hei["end"].isna().sum() == 1
    assert ch["sensor"].isna().any()  # empty descriptions stay NA
    with pytest.raises(ValueError, match="#Network"):
        asen.parse_fdsn_channels("<xml/>")


def test_sensors_at_epochs() -> None:
    ch = asen.parse_fdsn_channels(CHANNELS.read_text(encoding="utf-8"))
    now = asen.sensors_at(ch, pd.Timestamp("2026-11-12 21:53:45", tz="UTC"))
    assert now.at["gha", "sensor"] == "Guralp 40T" and not now.at["gha", "one_hz"]
    assert now.at["hei", "sensor"] == "LE-3D.1.400" and now.at["hei", "one_hz"]
    assert now.at["ada", "one_hz"] is pd.NA
    old = asen.sensors_at(ch, pd.Timestamp("2020-01-01"))
    assert old.at["gha", "sensor"] == "LE-3D5s.400" and not old.at["gha", "one_hz"]


def test_sensor_code_report_golden_and_1110() -> None:
    ch = asen.parse_fdsn_channels(CHANNELS.read_text(encoding="utf-8"))
    evs = [
        (
            date(2026, 11, d),
            sa.read_alert_event(ALERT / f"2026/nov/{d:02d}/{n}.event", strict=False),
        )
        for d, n in ((12, "076"), (10, "050"))
    ]
    rep = asen.sensor_code_report(evs, ch)
    assert list(rep.columns) == list(asen.SENSOR_REPORT_COLUMNS)
    got = {(r.alert_day, r.station): r.issue for r in rep.itertuples()}
    assert got == {
        ("2026-11-12", "hei"): "code22_one_hz",
        ("2026-11-12", "gha"): "code22_other_sensor",
        ("2026-11-12", "ada"): "code22_unknown_sensor",
        ("2026-11-10", "hei"): "one_hz_without_code22",
    }
    assert asen.sensor_code_report([], ch).empty


def test_read_fdsn_channels_stub() -> None:
    body = CHANNELS.read_bytes()

    class S:
        url = ""

        def get(
            self, url: str, *, timeout: float, headers: Mapping[str, str]
        ) -> FakeResponse:
            S.url = url
            return FakeResponse(200, body)

    ch = asen.read_fdsn_channels(["hei", "gha"], session=S())
    assert "station=gha%2Chei" in S.url and "level=channel" in S.url
    assert len(ch) > 0

    class Down:
        def get(
            self, url: str, *, timeout: float, headers: Mapping[str, str]
        ) -> FakeResponse:
            raise OSError("no route")

    with pytest.raises(asen.SensorServiceError, match="IMO network"):
        asen.read_fdsn_channels(session=Down())

    class Bad:
        def get(
            self, url: str, *, timeout: float, headers: Mapping[str, str]
        ) -> FakeResponse:
            return FakeResponse(400, b"bad")

    with pytest.raises(asen.SensorServiceError, match="400"):
        asen.read_fdsn_channels(session=Bad())

    class Empty:
        def get(
            self, url: str, *, timeout: float, headers: Mapping[str, str]
        ) -> FakeResponse:
            return FakeResponse(204, b"")

    empty = asen.read_fdsn_channels(["zzz"], session=Empty())
    assert empty.empty and list(empty.columns) == list(asen.FDSN_CHANNEL_COLUMNS)


# --------------------------------------------------------------------------
# Live (GPS_LIVE_API=1)
# --------------------------------------------------------------------------


@pytest.fixture()
def live() -> None:
    if os.environ.get("GPS_LIVE_API") != "1":
        pytest.skip("live alertMap test: set GPS_LIVE_API=1")


def test_live_http_2016_day(live: None, tmp_path: Path) -> None:
    h = src.HttpAlertSource(cache_dir=tmp_path / "c")
    day = date(2016, 6, 15)
    ids = h.list_day(day)
    assert 57 in ids
    assert h.read_event(day, 57).header.n_stations == 14


def test_live_fdsn_sensors(live: None) -> None:
    ch = asen.read_fdsn_channels(["hei", "gha"])
    now = asen.sensors_at(ch, pd.Timestamp("2026-11-12 21:53:45", tz="UTC"))
    assert now.at["hei", "one_hz"]


def test_live_mount_golden(live: None) -> None:
    if not os.environ.get(src.ENV_ALERTMAP_DIR) or not os.environ.get("GPS_SIL_DIR"):
        pytest.skip("set $GPS_ALERTMAP_DIR and $GPS_SIL_DIR")
    r = src.read_alert_pgv(GOLDEN_ID, origin=CLEANAUT)
    assert r.result is not None and r.result.alert_id == 76
