"""rtk_read: RTKLIB baseline day files -> interval medians (synthetic fixture)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from geo_dataread import rtk_read

BASELINE = "ROVR-BASE"


def _line(t: dt.datetime, e: float, n: float, u: float, q: int = 1,
          sde: float = 0.004, sdn: float = 0.007, sdu: float = 0.015) -> str:  # fmt: skip
    return (
        f"{t:%Y/%m/%d %H:%M:%S}.000 {e:14.4f} {n:14.4f} {u:14.4f} {q:3d} 9 "
        f"{sde:8.4f} {sdn:8.4f} {sdu:8.4f} -0.0030 0.0080 -0.0050 1.00 3.6\n"
    )


def _write_day(root: Path, day: dt.date, rows: list[str]) -> None:
    d = root / BASELINE
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{BASELINE}{day:%Y%m%d}0000b.pos").write_text("".join(rows))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Two day files, 1 Hz over a few minutes each, overlapping at midnight.

    E = 351 + seconds-of-minute/1000, N = −2582, U = 35 + minute/100; the
    10/12 file runs to 10/13 00:01:59 and the 10/13 file starts 00:01:00,
    so 00:01:00–00:01:59 is in both (the later file's values differ by +1
    in E there, to see which one is kept).
    """
    r = tmp_path / "rtk"
    d12, d13 = dt.date(2026, 10, 12), dt.date(2026, 10, 13)
    rows12 = []
    for k in range(120):  # 10/12 23:58:00 .. 23:59:59
        t = dt.datetime(2026, 10, 12, 23, 58) + dt.timedelta(seconds=k)
        rows12.append(_line(t, 351 + t.second / 1000, -2582.0, 35 + t.minute / 100))
    for k in range(120):  # tail into 10/13 00:00:00 .. 00:01:59
        t = dt.datetime(2026, 10, 13) + dt.timedelta(seconds=k)
        rows12.append(_line(t, 351 + t.second / 1000, -2582.0, 35 + t.minute / 100))
    rows13 = []
    for k in range(60, 300):  # 10/13 00:01:00 .. 00:04:59
        t = dt.datetime(2026, 10, 13) + dt.timedelta(seconds=k)
        e = 351 + t.second / 1000 + (1.0 if t.minute == 1 else 0.0)
        q = 5 if t.minute == 3 and t.second < 40 else 1  # 40 single epochs
        sdn = 0.5 if t.minute == 4 and t.second < 10 else 0.007  # 10 noisy
        rows13.append(_line(t, e, -2582.0, 35 + t.minute / 100, q=q, sdn=sdn))
    _write_day(r, d12, rows12)
    _write_day(r, d13, rows13)
    return r


def test_columns_are_enu_and_rows_come_back_neu(root: Path) -> None:
    df = rtk_read.read_file(root / BASELINE / f"{BASELINE}202610130000b.pos")
    assert df["e"].iloc[0] == pytest.approx(352.0)  # 00:01:00, the +1 file
    assert df["sde"].iloc[0] == pytest.approx(0.004)
    assert df["sdn"].iloc[0] == pytest.approx(0.007)
    s = rtk_read.read_rtk_baseline(
        BASELINE, dt.datetime(2026, 10, 13, 0, 2), dt.datetime(2026, 10, 13, 0, 2, 59),
        root=root, use_cache=False,
    )  # fmt: skip
    assert s.neu.shape == (3, 1)
    np.testing.assert_allclose(s.neu[:, 0], [-2582.0, 351.0295, 35.02])
    np.testing.assert_allclose(s.sigma[:, 0], [0.007, 0.004, 0.015])


def test_day_overlap_keeps_one_epoch_each_from_the_later_file(root: Path) -> None:
    s = rtk_read.read_rtk_baseline(
        BASELINE, dt.datetime(2026, 10, 13, 0, 0), dt.datetime(2026, 10, 13, 0, 1, 59),
        root=root, use_cache=False,
    )  # fmt: skip
    assert list(s.count) == [60, 60]
    assert s.neu[1, 0] == pytest.approx(351.0295)  # 00:00 only in the 10/12 file
    assert s.neu[1, 1] == pytest.approx(352.0295)  # 00:01 from the later file


def test_spans_two_utc_days_in_time_order(root: Path) -> None:
    s = rtk_read.read_rtk_baseline(
        BASELINE, dt.datetime(2026, 10, 12, 23, 0), dt.datetime(2026, 10, 13, 1, 0),
        root=root, use_cache=False,
    )  # fmt: skip
    t = s.time.astype("datetime64[m]").astype(str).tolist()
    assert t[0] == "2026-10-12T23:58" and t[-1] == "2026-10-13T00:04"
    assert np.all(np.diff(s.time) > np.timedelta64(0))


def test_quality_and_sigma_screens(root: Path) -> None:
    s = rtk_read.read_rtk_baseline(
        BASELINE, dt.datetime(2026, 10, 13, 0, 3), dt.datetime(2026, 10, 13, 0, 4, 59),
        root=root, use_cache=False,
    )  # fmt: skip
    assert list(s.count) == [20, 50]  # 40 Q=5 out, 10 with sdn 0.5 out
    loose = rtk_read.read_rtk_baseline(
        BASELINE, dt.datetime(2026, 10, 13, 0, 3), dt.datetime(2026, 10, 13, 0, 4, 59),
        root=root, use_cache=False, drop_q=(), max_sigma=0,
    )  # fmt: skip
    assert list(loose.count) == [60, 60]


def test_cache_only_complete_days_and_reuses_them(root: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    kw = dict(root=root, cache_dir=cache)
    span = (dt.datetime(2026, 10, 12), dt.datetime(2026, 10, 13, 23))
    a = rtk_read.read_rtk_baseline(BASELINE, *span, **kw)
    files = sorted(p.name for p in (cache / BASELINE).iterdir())
    # 10/12 is complete (10/13's file exists); 10/13 is not (no 10/14 file)
    assert len(files) == 1 and files[0].startswith("20261012-")
    b = rtk_read.read_rtk_baseline(BASELINE, *span, **kw)
    np.testing.assert_array_equal(a.neu, b.neu)
    other = rtk_read.read_rtk_baseline(BASELINE, *span, interval="120s", **kw)
    assert len(list((cache / BASELINE).iterdir())) == 2  # keyed on the interval
    assert other.interval == "120s"


def test_nothing_is_written_under_the_data_directory(
    root: Path, tmp_path: Path
) -> None:
    before = sorted(p for p in root.rglob("*"))
    rtk_read.read_rtk_baseline(
        BASELINE, dt.datetime(2026, 10, 12), dt.datetime(2026, 10, 14),
        root=root, cache_dir=tmp_path / "c",
    )  # fmt: skip
    assert sorted(p for p in root.rglob("*")) == before


def test_list_baselines_skips_non_pairs(root: Path) -> None:
    (root / "ELDV-SNAE_24h").mkdir()
    (root / "0").mkdir()
    assert rtk_read.list_baselines(root) == [BASELINE]


def test_missing_baseline_is_an_error(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no RTK baseline"):
        rtk_read.read_rtk_baseline(
            "XXXX-YYYY", dt.datetime(2026, 1, 1), dt.datetime(2026, 1, 2),
            root=root, use_cache=False,
        )  # fmt: skip


def test_empty_span_gives_empty_arrays(root: Path) -> None:
    s = rtk_read.read_rtk_baseline(
        BASELINE, dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2),
        root=root, use_cache=False,
    )  # fmt: skip
    assert s.neu.shape == (3, 0) and s.time.shape == (0,)


def test_an_open_day_is_read_once_until_its_file_grows(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rtk_read._OPEN_DAYS.clear()
    reads: list[dt.date] = []
    real = rtk_read._day_raw

    def counting(r: Path, baseline: str, day: dt.date) -> object:
        reads.append(day)
        return real(r, baseline, day)

    monkeypatch.setattr(rtk_read, "_day_raw", counting)
    span = (dt.datetime(2026, 10, 13), dt.datetime(2026, 10, 13, 23))
    kw = dict(root=root, use_cache=False)
    a = rtk_read.read_rtk_baseline(BASELINE, *span, **kw)
    b = rtk_read.read_rtk_baseline(BASELINE, *span, **kw)
    assert reads == [dt.date(2026, 10, 13)]  # 10/13 is open: memo, one read
    np.testing.assert_array_equal(a.neu, b.neu)
    rtk_read.read_rtk_baseline(BASELINE, *span, interval="120s", **kw)
    assert len(reads) == 2  # keyed on the interval
    # the file grows: read again, the new epochs are in
    f = root / BASELINE / f"{BASELINE}202610130000b.pos"
    t = dt.datetime(2026, 10, 13, 0, 10)
    with open(f, "a") as fh:
        fh.write(_line(t, 351.0, -2582.0, 35.1))
    c = rtk_read.read_rtk_baseline(BASELINE, *span, **kw)
    assert len(reads) == 3 and c.time.size == a.time.size + 1
    rtk_read._OPEN_DAYS.clear()
