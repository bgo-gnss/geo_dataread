"""Tests for the cleaned ``.NEU`` writer (geo_dataread.gps_write).

Pinned here:

- the cleaned file is byte-format identical to the raw product (same
  header, surviving data lines byte-identical), differing ONLY by the
  dropped rows,
- the drop set is the row-wise UNION of the per-component outlier flags,
- the provenance sidecar: counts add up, params echo + stable params_hash,
  version stamps,
- graceful degrade (detector error AND excess-flag abort): the FULL raw
  series is written to a `.DEGRADED.` filename, the sidecar is marked
  degraded, a UserWarning fires,
- declared steps (steps.csv) are read (per-station flat union of epochs)
  and fed to detect_outliers so a stepped series that aborts BARE no
  longer aborts; a missing/unreadable catalog degrades gracefully,
- declared protect windows (protect_windows.csv) are read (per-station
  sorted intervals) and fed to detect_outliers as the active-unrest lever,
  so an unrest station that aborts→DEGRADED bare CLEANS→plain _cleaned.NEU
  with the window declared; missing/unreadable catalog degrades gracefully,
- the gps-savetimes CLI: --clean ALSO writes the cleaned .NEU (+ sidecar)
  and leaves the raw output byte-identical; --clean without --file refused.

Uses the goldenmaster fixtures (frozen GLOBK series + hermetic gpsconfig).
"""

import json
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "goldenmaster"))

from cases import TOT, build_config_dir  # noqa: E402

import geo_dataread.gps_read as gpsr  # noqa: E402
import geo_dataread.gps_savetimes as gps_savetimes  # noqa: E402
from geo_dataread import gps_views, gps_write  # noqa: E402
from gps_analysis import OutlierParams  # noqa: E402

STA = "SENG"
REF = "plate"
SAVETIMES_KW = dict(mm=True, ref=REF, dstring=None, rhour=True)
HEADER_LINES = 2  # gamittoFile: "printing header and file \n" + header line


# ---------------------------------------------------------------------------
# environment / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def write_env(tmp_path_factory):
    config_dir = build_config_dir(tmp_path_factory.mktemp("gpsconfig-write"))
    # The CLI reads its INPUT series from the config totDir (only --Dir, the
    # OUTPUT dir, is a flag). The installed gps_parser 0.3.0 resolves totDir
    # from a [PATHS] section, while the shared fixture config uses [Configs]
    # (every golden test passes Dir= explicitly). Add a [PATHS] totDir so the
    # CLI raw read resolves here — local to this suite's config dir only.
    cfg = config_dir / "postprocess.cfg"
    cfg.write_text(cfg.read_text() + f"\n[PATHS]\ntotDir = {TOT}\n")
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(config_dir)
    yield {"config_dir": config_dir}
    if old is None:
        os.environ.pop("GPS_CONFIG_PATH", None)
    else:
        os.environ["GPS_CONFIG_PATH"] = old


@pytest.fixture(scope="module")
def raw_neu(write_env, tmp_path_factory):
    """The exact production raw product (gps-savetimes profile)."""
    out = tmp_path_factory.mktemp("raw") / f"{STA}-{REF}.NEU"
    gpsr.gamittooneuf(STA, str(out), Dir=TOT, **SAVETIMES_KW)
    return out


def _lines(path: Path) -> tuple[list[str], list[str]]:
    """(header lines, data lines) of a written .NEU file."""
    lines = path.read_text().splitlines()
    return lines[:HEADER_LINES], lines[HEADER_LINES:]


def _fake_detection(monkeypatch, flag_idx=((0, 1, 2), (3, 4), (4, 5))):
    """Deterministic detector double: flag fixed finite-subset indices."""

    def fake(model, t, y, sigma=None, **kwargs):
        flags = np.zeros(np.atleast_2d(y).shape, dtype=np.bool_)
        for c, idx in enumerate(flag_idx):
            flags[c, list(idx)] = True
        return SimpleNamespace(flags=flags, excess_flag_abort=False)

    monkeypatch.setattr(gps_views, "detect_outliers", fake)
    return flag_idx


def _numeric_arrays():
    """Numeric twin of the savetimes read (what the writer detects on)."""
    neu = gpsr.gamittoNEU(STA, mm=True, ref=REF, dstring="yearf", Dir=TOT, rhour=True)
    t = np.asarray(neu["yearf"], dtype=np.float64)
    y = np.vstack([np.asarray(neu[f"data[{c}]"]) for c in range(3)])
    sigma = np.vstack([np.asarray(neu[f"Ddata[{c}]"]) for c in range(3)])
    return t, y, sigma


# ---------------------------------------------------------------------------
# cleaned file — format identity + union drop set
# ---------------------------------------------------------------------------


def test_cleaned_file_drops_union_rows_byte_identically(
    write_env, raw_neu, tmp_path, monkeypatch
):
    _fake_detection(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    result = gps_write.write_cleaned_neu(STA, out, Dir=TOT, **SAVETIMES_KW)

    # the union mask the writer must have used (same fake, same read)
    t, y, sigma = _numeric_arrays()
    flags, _ = gps_views.detect_view_outliers(t, y, sigma)
    union = flags.any(axis=0)

    raw_header, raw_data = _lines(raw_neu)
    cln_header, cln_data = _lines(out)

    assert cln_header == raw_header, "header must be byte-identical to raw"
    assert len(raw_data) == result["n_total"]
    assert len(cln_data) == result["n_kept"] < len(raw_data)
    assert result["n_removed"] == int(union.sum()) > 0

    expected_kept = [ln for ln, drop in zip(raw_data, union) if not drop]
    assert cln_data == expected_kept, (
        "surviving rows must be byte-identical to the raw rows; dropped "
        "rows must be exactly the union-flagged epochs"
    )

    # raw file untouched
    assert raw_neu.read_text() == raw_neu.read_text()
    assert result["degraded"] is False
    assert result["excess_flag_abort"] is False


def test_cleaned_yearf_format_single_read(write_env, tmp_path, monkeypatch):
    """dstring='yearf' path: numeric read doubles as the output rows."""
    _fake_detection(monkeypatch)
    raw = tmp_path / f"{STA}-{REF}.NEU"
    gpsr.gamittooneuf(STA, str(raw), mm=True, ref=REF, dstring="yearf", Dir=TOT)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    result = gps_write.write_cleaned_neu(
        STA, out, mm=True, ref=REF, dstring="yearf", Dir=TOT
    )
    raw_header, raw_data = _lines(raw)
    cln_header, cln_data = _lines(out)
    assert cln_header == raw_header
    assert len(cln_data) == result["n_kept"] == result["n_total"] - result["n_removed"]
    assert set(cln_data) <= set(raw_data)


# ---------------------------------------------------------------------------
# provenance sidecar
# ---------------------------------------------------------------------------


def test_sidecar_contents_and_counts(write_env, tmp_path, monkeypatch):
    flag_idx = _fake_detection(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    params = OutlierParams(global_n_sigma=6.0)
    result = gps_write.write_cleaned_neu(
        STA, out, Dir=TOT, outlier_params=params, **SAVETIMES_KW
    )

    sidecar = Path(result["sidecar"])
    assert sidecar == Path(str(out) + ".prov.json")
    prov = json.loads(sidecar.read_text())

    assert prov["schema_version"] == gps_write.PROV_SCHEMA_VERSION
    assert prov["kind"] == "cleaned_neu"
    assert prov["station"] == STA
    assert prov["ref"] == REF
    assert prov["reference_frame"] == "ITRF2008"
    assert prov["unit"] == "mm"
    assert prov["cleaned_file"] == out.name

    # counts add up
    assert prov["n_total"] == prov["n_kept"] + prov["n_removed"]
    union = set()
    for c, name in enumerate(("north", "east", "up")):
        assert prov["n_flagged_by_component"][name] == len(flag_idx[c])
        union.update(flag_idx[c])
    assert prov["n_removed"] == len(union)
    _, cln_data = _lines(out)
    assert len(cln_data) == prov["n_kept"]

    # detector echo + stable hash
    det = prov["detector"]
    assert det["function"] == "gps_analysis.detect_outliers"
    assert det["model"] == "lineperiodic"
    assert det["detection_unit"] == "mm"
    assert det["row_policy"] == "union"
    assert det["params"] == asdict(params)
    assert det["params_hash"] == gps_write.outlier_params_hash(params)

    assert prov["degraded"] is False
    assert prov["excess_flag_abort"] is False
    assert prov["degrade_reason"] is None
    for dist in ("geo_dataread", "gps_analysis"):
        assert prov["versions"][dist] not in ("", None)

    # returned record mirrors the sidecar (plus the paths)
    on_disk = dict(prov)
    in_memory = {k: v for k, v in result.items() if k not in ("outfile", "sidecar")}
    assert in_memory == on_disk


def test_params_hash_is_stable_and_sensitive():
    a = gps_write.outlier_params_hash(OutlierParams())
    b = gps_write.outlier_params_hash(OutlierParams())
    c = gps_write.outlier_params_hash(OutlierParams(global_n_sigma=6.0))
    assert a == b
    assert a != c
    assert a.startswith("sha256:")


# ---------------------------------------------------------------------------
# graceful degrade — full raw series, marked, loudly
# ---------------------------------------------------------------------------


def test_detector_error_degrades_to_full_marked_file(
    write_env, raw_neu, tmp_path, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic detector failure")

    monkeypatch.setattr(gps_views, "detect_outliers", boom)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with pytest.warns(UserWarning, match="cleaned .NEU degraded"):
        result = gps_write.write_cleaned_neu(STA, out, Dir=TOT, **SAVETIMES_KW)

    # degrade -> structurally-marked name; the plain name must NOT appear
    actual = Path(result["outfile"])
    assert actual.name == f"{STA}-{REF}_cleaned.DEGRADED.NEU"
    assert not out.exists()
    # the FULL raw series, byte-identical
    assert actual.read_text() == raw_neu.read_text()
    prov = json.loads(Path(result["sidecar"]).read_text())
    assert prov["degraded"] is True
    assert prov["excess_flag_abort"] is False
    assert "synthetic detector failure" in prov["degrade_reason"]
    assert prov["n_removed"] == 0
    assert prov["n_kept"] == prov["n_total"]


def test_excess_flag_abort_degrades_to_full_marked_file(
    write_env, raw_neu, tmp_path, monkeypatch
):
    def abort(*args, **kwargs):
        return SimpleNamespace(flags=None, excess_flag_abort=True)

    monkeypatch.setattr(gps_views, "detect_outliers", abort)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with pytest.warns(UserWarning, match="degraded"):
        result = gps_write.write_cleaned_neu(STA, out, Dir=TOT, **SAVETIMES_KW)

    actual = Path(result["outfile"])
    assert actual.name == f"{STA}-{REF}_cleaned.DEGRADED.NEU"
    assert not out.exists()
    assert actual.read_text() == raw_neu.read_text()
    prov = json.loads(Path(result["sidecar"]).read_text())
    assert prov["degraded"] is True
    assert prov["excess_flag_abort"] is True
    assert prov["n_removed"] == 0


# ---------------------------------------------------------------------------
# real detection — end-to-end invariants (no doubles)
# ---------------------------------------------------------------------------


def test_real_detection_end_to_end(write_env, raw_neu, tmp_path):
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # full SENG spans unrest; abort allowed
        result = gps_write.write_cleaned_neu(STA, out, Dir=TOT, **SAVETIMES_KW)

    raw_header, raw_data = _lines(raw_neu)
    cln_header, cln_data = _lines(Path(result["outfile"]))  # may be .DEGRADED
    assert cln_header == raw_header
    assert result["n_total"] == len(raw_data)
    assert result["n_kept"] == len(cln_data)
    assert result["n_removed"] == len(raw_data) - len(cln_data)
    # surviving rows are an in-order byte-identical subsequence of raw
    it = iter(raw_data)
    assert all(ln in it for ln in cln_data)
    if result["degraded"]:
        assert result["n_removed"] == 0


# ---------------------------------------------------------------------------
# CLI — gps-savetimes --clean
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["gps-savetimes", *argv])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gps_savetimes.main()


def test_cli_without_clean_is_unchanged(write_env, tmp_path, monkeypatch):
    _run_cli(monkeypatch, [STA, "--file", "--Dir", str(tmp_path)])
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [f"{STA}-{REF}.NEU"], "no --clean -> raw file only"


def test_cli_clean_also_writes_cleaned_and_sidecar(
    write_env, raw_neu, tmp_path, monkeypatch
):
    _run_cli(monkeypatch, [STA, "--file", "--clean", "--Dir", str(tmp_path)])
    raw = tmp_path / f"{STA}-{REF}.NEU"
    # full-span SENG aborts on defaults -> structurally-marked .DEGRADED name
    cleaned_glob = sorted(tmp_path.glob(f"{STA}-{REF}_cleaned*.NEU"))
    assert raw.is_file()
    assert len(cleaned_glob) == 1
    cleaned = cleaned_glob[0]
    sidecar = Path(str(cleaned) + ".prov.json")
    assert sidecar.is_file()

    # raw output byte-identical to a run without --clean
    assert raw.read_text() == raw_neu.read_text()

    prov = json.loads(sidecar.read_text())
    _, cln_data = _lines(cleaned)
    assert prov["n_kept"] == len(cln_data)
    assert prov["n_total"] == prov["n_kept"] + prov["n_removed"]
    assert prov["station"] == STA and prov["ref"] == REF
    # cleaned file name carries the degrade state structurally
    if prov["degraded"]:
        assert cleaned.name == f"{STA}-{REF}_cleaned.DEGRADED.NEU"
    else:
        assert cleaned.name == f"{STA}-{REF}_cleaned.NEU"


def test_cli_clean_requires_file(write_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gps-savetimes", STA, "--clean"])
    with pytest.raises(SystemExit):
        gps_savetimes.main()
    assert "--clean requires --file" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# declared steps (steps.csv) -> step_epochs into detection
# ---------------------------------------------------------------------------


def _end_stepped_series(seed=5):
    """Synthetic series with a step in the last ~7% of epochs.

    A step near the end makes a STEPLESS model over-flag the offset tail
    (fraction > max_flag_fraction -> excess-candidate abort); declaring the
    step lets the model absorb it, so detection no longer aborts.
    """
    rng = np.random.default_rng(seed)
    t = 2015.0 + np.arange(1461) / 365.25
    step_t = float(t[int(0.93 * t.size)])
    base = (
        20.0 * (t - 2015.0) + 3.0 * np.cos(2 * np.pi * t) + 1.5 * np.sin(2 * np.pi * t)
    )
    jump = np.where(t >= step_t, 40.0, 0.0)
    y = np.vstack([base + jump + rng.normal(0.0, 1.0, t.size) for _ in range(3)])
    sigma = np.full(y.shape, 1.0)
    return t, y, sigma, step_t


def test_declared_step_prevents_abort():
    t, y, sigma, step_t = _end_stepped_series()
    with pytest.warns(UserWarning, match="aborted"):
        bare_flags, bare = gps_views.detect_view_outliers(t, y, sigma)
    assert bare["outlier_abort"] is True and bare["degraded"] is True

    stepped_flags, stepped = gps_views.detect_view_outliers(
        t, y, sigma, step_epochs=np.array([step_t])
    )
    assert stepped["outlier_abort"] is False and stepped["degraded"] is False


def _write_steps_csv(path, rows):
    header = "sta,epoch_yearf,component,kind,source,comment\n"
    body = "".join(
        f"{sta},{epoch},{comp},{kind},{src},{cmt}\n"
        for sta, epoch, comp, kind, src, cmt in rows
    )
    path.write_text(header + body)


def test_read_step_catalog_flat_union(tmp_path):
    p = tmp_path / "steps.csv"
    _write_steps_csv(
        p,
        [
            ("SENG", 2020.5, "ALL", "equip", "tos", "antenna"),
            ("SENG", 2021.0, "N", "coseismic", "sil", "eq"),
            ("SENG", 2020.5, "E", "dup", "x", "same epoch, other comp"),
            ("ELDC", 2019.25, "U", "equip", "tos", "receiver"),
        ],
    )
    catalog = gps_views.read_step_catalog(p)
    # per-station flat union, sorted + de-duplicated across components
    assert catalog["SENG"] == (2020.5, 2021.0)
    assert catalog["ELDC"] == (2019.25,)


def test_read_step_catalog_rejects_bad_rows(tmp_path):
    bad_comp = tmp_path / "badcomp.csv"
    _write_steps_csv(bad_comp, [("SENG", 2020.5, "X", "k", "s", "c")])
    with pytest.raises(ValueError, match="component"):
        gps_views.read_step_catalog(bad_comp)

    bad_epoch = tmp_path / "badepoch.csv"
    bad_epoch.write_text(
        "sta,epoch_yearf,component,kind,source,comment\nSENG,notayear,N,k,s,c\n"
    )
    with pytest.raises(ValueError, match="fractional year"):
        gps_views.read_step_catalog(bad_epoch)

    with pytest.raises(FileNotFoundError):
        gps_views.read_step_catalog(tmp_path / "missing.csv")


def test_station_step_epochs_graceful_missing(tmp_path):
    with pytest.warns(UserWarning, match="no step catalog"):
        epochs, source = gps_views.station_step_epochs(STA, steps=tmp_path / "nope.csv")
    assert epochs.size == 0 and source is None


def test_station_step_epochs_graceful_corrupt(tmp_path):
    p = tmp_path / "steps.csv"
    _write_steps_csv(p, [("SENG", 2020.5, "BOGUS", "k", "s", "c")])
    with pytest.warns(UserWarning, match="unreadable"):
        epochs, source = gps_views.station_step_epochs(STA, steps=p)
    assert epochs.size == 0 and source is None


def test_writer_feeds_step_epochs_and_records_source(write_env, tmp_path, monkeypatch):
    steps_csv = tmp_path / "steps.csv"
    _write_steps_csv(
        steps_csv,
        [
            ("SENG", 2020.5, "ALL", "equip", "tos", "antenna"),
            ("SENG", 2021.0, "N", "coseismic", "sil", "eq"),
        ],
    )
    captured = {}

    def capture(model, t, y, sigma=None, **kwargs):
        captured["step_epochs"] = kwargs.get("step_epochs")
        flags = np.zeros(np.atleast_2d(y).shape, dtype=np.bool_)
        return SimpleNamespace(flags=flags, excess_flag_abort=False)

    monkeypatch.setattr(gps_views, "detect_outliers", capture)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    result = gps_write.write_cleaned_neu(
        STA, out, Dir=TOT, steps=steps_csv, **SAVETIMES_KW
    )

    np.testing.assert_array_equal(
        np.asarray(captured["step_epochs"], dtype=float), np.array([2020.5, 2021.0])
    )
    det = result["detector"]
    assert det["step_epochs_applied"] == 2
    assert det["steps_source"] == str(steps_csv)
    # sidecar echoes the same
    prov = json.loads(Path(result["sidecar"]).read_text())
    assert prov["detector"]["step_epochs_applied"] == 2
    assert prov["detector"]["steps_source"] == str(steps_csv)


def test_writer_missing_steps_records_zero(write_env, tmp_path, monkeypatch):
    _fake_detection(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with pytest.warns(UserWarning, match="no step catalog"):
        result = gps_write.write_cleaned_neu(
            STA, out, Dir=TOT, steps=tmp_path / "nope.csv", **SAVETIMES_KW
        )
    assert result["detector"]["step_epochs_applied"] == 0
    assert result["detector"]["steps_source"] is None


# ---------------------------------------------------------------------------
# .DEGRADED. filename (structural cleanliness signal)
# ---------------------------------------------------------------------------


def test_plain_name_when_clean(write_env, tmp_path, monkeypatch):
    _fake_detection(monkeypatch)  # no abort
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    result = gps_write.write_cleaned_neu(STA, out, Dir=TOT, **SAVETIMES_KW)
    assert result["degraded"] is False
    assert Path(result["outfile"]).name == f"{STA}-{REF}_cleaned.NEU"
    assert out.is_file()
    assert not (tmp_path / f"{STA}-{REF}_cleaned.DEGRADED.NEU").exists()


def test_degraded_name_on_abort(write_env, raw_neu, tmp_path, monkeypatch):
    def abort(*args, **kwargs):
        return SimpleNamespace(flags=None, excess_flag_abort=True)

    monkeypatch.setattr(gps_views, "detect_outliers", abort)
    requested = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with pytest.warns(UserWarning, match="DEGRADED"):
        result = gps_write.write_cleaned_neu(STA, requested, Dir=TOT, **SAVETIMES_KW)

    degraded = tmp_path / f"{STA}-{REF}_cleaned.DEGRADED.NEU"
    assert Path(result["outfile"]) == degraded
    assert degraded.is_file()
    assert not requested.exists(), "the plain _cleaned.NEU name must NOT appear"
    assert Path(result["sidecar"]) == Path(str(degraded) + ".prov.json")
    # degraded file is the FULL raw series
    assert degraded.read_text() == raw_neu.read_text()
    assert result["degraded"] is True and result["n_removed"] == 0


# ---------------------------------------------------------------------------
# declared protect windows (protect_windows.csv) — active-unrest lever
# ---------------------------------------------------------------------------


def _write_protect_windows_csv(path, rows):
    header = "sta,start_yearf,end_yearf,comment\n"
    body = "".join(f"{sta},{start},{end},{cmt}\n" for sta, start, end, cmt in rows)
    path.write_text(header + body)


def _unrest_series(seed=5):
    """Synthetic series with continuous rapid unrest in the last ~10%.

    A lineperiodic model cannot fit the unrest segment, so a BARE detection
    over-flags it (fraction > max_flag_fraction -> excess-candidate abort).
    A protect window over the segment excludes it from the fit, the
    identifiers AND the abort, so detection cleans instead of degrading.
    """
    rng = np.random.default_rng(seed)
    t = 2015.0 + np.arange(1461) / 365.25
    onset = float(t[int(0.90 * t.size)])
    unrest = np.where(t >= onset, 60.0 * (t - onset) + 400.0 * (t - onset) ** 2, 0.0)
    base = (
        20.0 * (t - 2015.0) + 3.0 * np.cos(2 * np.pi * t) + 1.5 * np.sin(2 * np.pi * t)
    )
    y = np.vstack([base + unrest + rng.normal(0.0, 1.0, t.size) for _ in range(3)])
    sigma = np.full(y.shape, 1.0)
    return t, y, sigma, onset


def test_protect_window_prevents_abort():
    """Headline: an unrest series aborts BARE, cleans with a protect window."""
    t, y, sigma, onset = _unrest_series()
    with pytest.warns(UserWarning, match="aborted"):
        _bare_flags, bare = gps_views.detect_view_outliers(t, y, sigma)
    assert bare["outlier_abort"] is True and bare["degraded"] is True

    _flags, protected = gps_views.detect_view_outliers(
        t, y, sigma, protect_windows=((onset, float(t[-1])),)
    )
    assert protected["outlier_abort"] is False and protected["degraded"] is False


def test_read_protect_windows_parses_intervals(tmp_path):
    p = tmp_path / "protect_windows.csv"
    _write_protect_windows_csv(
        p,
        [
            ("SENG", 2023.9, 2025.0, "svartsengi unrest"),
            ("SENG", 2020.1, 2020.8, "reykjanes onset"),  # out of order on purpose
            ("ELDC", 2021.0, 2021.5, "swarm"),
        ],
    )
    catalog = gps_views.read_protect_windows(p)
    # per-station intervals, sorted by start
    assert catalog["SENG"] == ((2020.1, 2020.8), (2023.9, 2025.0))
    assert catalog["ELDC"] == ((2021.0, 2021.5),)


def test_read_protect_windows_rejects_bad_rows(tmp_path):
    end_lt_start = tmp_path / "endlt.csv"
    _write_protect_windows_csv(end_lt_start, [("SENG", 2025.0, 2023.0, "reversed")])
    with pytest.raises(ValueError, match="end.*<.*start|end .* start"):
        gps_views.read_protect_windows(end_lt_start)

    bad_num = tmp_path / "badnum.csv"
    bad_num.write_text("sta,start_yearf,end_yearf,comment\nSENG,notayear,2025.0,c\n")
    with pytest.raises(ValueError, match="fractional year"):
        gps_views.read_protect_windows(bad_num)

    with pytest.raises(FileNotFoundError):
        gps_views.read_protect_windows(tmp_path / "missing.csv")


def test_station_protect_windows_graceful_missing(tmp_path):
    with pytest.warns(UserWarning, match="no protect-window catalog"):
        windows, source = gps_views.station_protect_windows(
            STA, catalog=tmp_path / "nope.csv"
        )
    assert windows == () and source is None


def test_station_protect_windows_graceful_corrupt(tmp_path):
    p = tmp_path / "protect_windows.csv"
    _write_protect_windows_csv(p, [("SENG", 2025.0, 2023.0, "reversed")])
    with pytest.warns(UserWarning, match="unreadable"):
        windows, source = gps_views.station_protect_windows(STA, catalog=p)
    assert windows == () and source is None


def test_writer_feeds_protect_windows_and_records_source(
    write_env, tmp_path, monkeypatch
):
    pw_csv = tmp_path / "protect_windows.csv"
    _write_protect_windows_csv(
        pw_csv,
        [
            ("SENG", 2020.1, 2020.8, "onset"),
            ("SENG", 2023.9, 2025.0, "unrest"),
        ],
    )
    captured = {}

    def capture(model, t, y, sigma=None, **kwargs):
        captured["protect_windows"] = kwargs.get("protect_windows")
        flags = np.zeros(np.atleast_2d(y).shape, dtype=np.bool_)
        return SimpleNamespace(flags=flags, excess_flag_abort=False)

    monkeypatch.setattr(gps_views, "detect_outliers", capture)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no steps.csv deployed
        result = gps_write.write_cleaned_neu(
            STA, out, Dir=TOT, protect_windows=pw_csv, **SAVETIMES_KW
        )

    assert tuple(captured["protect_windows"]) == ((2020.1, 2020.8), (2023.9, 2025.0))
    det = result["detector"]
    assert det["protect_windows_applied"] == 2
    assert det["protect_windows_source"] == str(pw_csv)
    prov = json.loads(Path(result["sidecar"]).read_text())
    assert prov["detector"]["protect_windows_applied"] == 2
    assert prov["detector"]["protect_windows_source"] == str(pw_csv)


def test_writer_protect_windows_compose_with_steps(write_env, tmp_path, monkeypatch):
    steps_csv = tmp_path / "steps.csv"
    _write_steps_csv(steps_csv, [("SENG", 2020.5, "ALL", "equip", "tos", "antenna")])
    pw_csv = tmp_path / "protect_windows.csv"
    _write_protect_windows_csv(pw_csv, [("SENG", 2023.9, 2025.0, "unrest")])
    captured = {}

    def capture(model, t, y, sigma=None, **kwargs):
        captured["step_epochs"] = kwargs.get("step_epochs")
        captured["protect_windows"] = kwargs.get("protect_windows")
        flags = np.zeros(np.atleast_2d(y).shape, dtype=np.bool_)
        return SimpleNamespace(flags=flags, excess_flag_abort=False)

    monkeypatch.setattr(gps_views, "detect_outliers", capture)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    result = gps_write.write_cleaned_neu(
        STA, out, Dir=TOT, steps=steps_csv, protect_windows=pw_csv, **SAVETIMES_KW
    )
    # BOTH levers compose in the same detection call
    np.testing.assert_array_equal(
        np.asarray(captured["step_epochs"], dtype=float), np.array([2020.5])
    )
    assert tuple(captured["protect_windows"]) == ((2023.9, 2025.0),)
    det = result["detector"]
    assert det["step_epochs_applied"] == 1
    assert det["protect_windows_applied"] == 1


def test_writer_missing_protect_windows_records_zero(write_env, tmp_path, monkeypatch):
    _fake_detection(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gps_write.write_cleaned_neu(
            STA,
            out,
            Dir=TOT,
            protect_windows=tmp_path / "nope.csv",
            **SAVETIMES_KW,
        )
    assert result["detector"]["protect_windows_applied"] == 0
    assert result["detector"]["protect_windows_source"] is None


def test_real_abort_to_clean_composition(write_env, raw_neu, tmp_path):
    """File-level headline: real SENG aborts→DEGRADED bare, cleans→plain with a
    protect window over the unrest (which the bare stepless model over-flags)."""
    # bare: no window -> abort -> structurally-marked .DEGRADED name
    bare_out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bare = gps_write.write_cleaned_neu(STA, bare_out, Dir=TOT, **SAVETIMES_KW)
    assert bare["degraded"] is True and bare["excess_flag_abort"] is True
    assert Path(bare["outfile"]).name == f"{STA}-{REF}_cleaned.DEGRADED.NEU"
    assert not bare_out.exists()

    # protected: a window over the unrest span clears the abort -> plain name
    prot_out = tmp_path / f"P_{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prot = gps_write.write_cleaned_neu(
            STA, prot_out, Dir=TOT, protect_windows=((2016.5, 2025.2),), **SAVETIMES_KW
        )
    assert prot["degraded"] is False and prot["excess_flag_abort"] is False
    assert Path(prot["outfile"]).name == f"P_{STA}-{REF}_cleaned.NEU"
    assert prot_out.is_file()
    assert prot["detector"]["protect_windows_applied"] == 1
    assert prot["detector"]["protect_windows_source"] == "explicit"
    # a real clean drops the outlier rows -> fewer than raw, header unchanged
    raw_header, raw_data = _lines(raw_neu)
    prot_header, prot_data = _lines(prot_out)
    assert prot_header == raw_header
    assert len(prot_data) == prot["n_kept"] <= prot["n_total"]


def test_resolve_protect_windows_override_modes():
    """The shared resolver: explicit sequence used directly (source explicit)."""
    windows, source = gps_views.resolve_protect_windows(STA, ((2023.9, 2025.0),))
    assert windows == ((2023.9, 2025.0),) and source == "explicit"
    empty, empty_src = gps_views.resolve_protect_windows(STA, ())
    assert empty == () and empty_src == "explicit"


def test_cleaned_view_records_protect_windows(write_env):
    """read_gps_view(view="cleaned") records protect-window provenance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # full SENG spans unrest; abort allowed
        df = gps_views.read_gps_view(
            STA,
            view="cleaned",
            protect_windows=((2016.5, 2025.2),),
            Dir=TOT,
        )
    attrs = df.attrs["gps_view"]
    assert attrs["protect_windows_applied"] == 1
    assert attrs["protect_windows_source"] == "explicit"
    # the wide window clears the abort -> not degraded, real flags present
    assert attrs["degraded"] is False and attrs["outlier_abort"] is False


# ---------------------------------------------------------------------------
# per-station outlier-parameter overrides (outlier_overrides.csv)
# ---------------------------------------------------------------------------


def _write_overrides_csv(path, header, rows):
    path.write_text(header + "".join(rows))


def _capture_params(monkeypatch):
    """Detector double that records the resolved OutlierParams it received."""
    captured = {}

    def capture(model, t, y, sigma=None, **kwargs):
        captured["params"] = kwargs.get("params")
        captured["step_epochs"] = kwargs.get("step_epochs")
        captured["protect_windows"] = kwargs.get("protect_windows")
        captured["min_outlier"] = kwargs.get("min_outlier")
        flags = np.zeros(np.atleast_2d(y).shape, dtype=np.bool_)
        return SimpleNamespace(flags=flags, excess_flag_abort=False)

    monkeypatch.setattr(gps_views, "detect_outliers", capture)
    return captured


def test_read_outlier_overrides_parses_only_provided_fields(tmp_path):
    p = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(
        p,
        "sta,despike,window_order,window_robust_iterations,epoch_policy,"
        "despike_n_sigma,min_outlier_n,min_outlier_e,min_outlier_u,comment\n",
        [
            "SENG,true,1,2,union,4.5,5,5,5,unrest\n",
            "ELDC,,2,,,,,,,quad only\n",  # only window_order provided
        ],
    )
    catalog = gps_views.read_outlier_overrides(p)
    seng = catalog["SENG"]
    assert seng.params_fields == {
        "despike": True,
        "window_order": 1,
        "window_robust_iterations": 2,
        "epoch_policy": "union",
        "despike_n_sigma": 4.5,
    }
    # min_outlier_{n,e,u} route to the per-component floor, NOT params_fields
    assert seng.min_outlier == (5.0, 5.0, 5.0)
    assert catalog["ELDC"].params_fields == {"window_order": 2}
    assert catalog["ELDC"].min_outlier is None  # blanks left at default


def test_read_outlier_overrides_rejects_bad(tmp_path):
    cases = {
        "sta,window_order\nSENG,3\n": "window_order",
        "sta,epoch_policy\nSENG,bogus\n": "epoch_policy",
        "sta,despike\nSENG,maybe\n": "boolean",
        "sta,window_robust_iterations\nSENG,-1\n": ">= 0",
        "sta,frobnicate\nSENG,7\n": "unknown column",
        "sta,window_order\nSENG,1\nSENG,2\n": "duplicate",
        "sta,min_outlier_n\nSENG,notanum\n": "not a number",
        "sta,min_outlier_u\nSENG,-3\n": "finite and >= 0",
    }
    for content, needle in cases.items():
        p = tmp_path / "bad.csv"
        p.write_text(content)
        with pytest.raises(ValueError, match=needle):
            gps_views.read_outlier_overrides(p)
    with pytest.raises(FileNotFoundError):
        gps_views.read_outlier_overrides(tmp_path / "missing.csv")


def test_read_outlier_overrides_min_outlier_per_component(tmp_path):
    p = tmp_path / "outlier_overrides.csv"
    # differing per-component floors are now VALID (the whole point)
    _write_overrides_csv(
        p,
        "sta,min_outlier_n,min_outlier_e,min_outlier_u\n",
        ["SENG,5,6,7\n", "ELDC,,,10\n"],  # ELDC: partial -> missing fill 0.0
    )
    catalog = gps_views.read_outlier_overrides(p)
    assert catalog["SENG"].min_outlier == (5.0, 6.0, 7.0)
    assert catalog["ELDC"].min_outlier == (0.0, 0.0, 10.0)  # partial fills 0.0
    assert catalog["ELDC"].params_fields == {}


def test_station_outlier_params_applies_and_absent(tmp_path):
    p = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(
        p, "sta,despike,window_order,epoch_policy\n", ["SENG,true,1,union\n"]
    )
    params, floor, source = gps_views.station_outlier_params(STA, catalog=p)
    assert params.despike is True
    assert params.window_order == 1
    assert params.epoch_policy == "union"
    assert floor is None  # no min_outlier columns in this catalog
    assert source == str(p)
    assert gps_views.outlier_override_delta(params) == {
        "despike": True,
        "window_order": 1,
        "epoch_policy": "union",
    }
    # a station with no row -> base unchanged, source still known
    absent, absent_floor, absent_src = gps_views.station_outlier_params(
        "ZZZZ", catalog=p
    )
    assert absent == OutlierParams() and absent_floor is None and absent_src == str(p)
    # an explicit base is honoured
    based, _, _ = gps_views.station_outlier_params(
        STA, base=OutlierParams(global_n_sigma=9.0), catalog=p
    )
    assert based.global_n_sigma == 9.0 and based.window_order == 1


def test_station_outlier_params_returns_floor(tmp_path):
    p = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(
        p,
        "sta,window_order,min_outlier_n,min_outlier_e,min_outlier_u\n",
        ["SENG,1,5,5,10\n"],
    )
    params, floor, source = gps_views.station_outlier_params(STA, catalog=p)
    assert params.window_order == 1
    assert floor == (5.0, 5.0, 10.0)
    assert source == str(p)


def test_station_outlier_params_graceful_missing(tmp_path):
    with pytest.warns(UserWarning, match="no outlier-override catalog"):
        params, floor, source = gps_views.station_outlier_params(
            STA, catalog=tmp_path / "nope.csv"
        )
    assert params == OutlierParams() and floor is None and source is None


def test_station_outlier_params_graceful_corrupt(tmp_path):
    p = tmp_path / "outlier_overrides.csv"
    p.write_text("sta,window_order\nSENG,7\n")
    with pytest.warns(UserWarning, match="unreadable"):
        params, floor, source = gps_views.station_outlier_params(STA, catalog=p)
    assert params == OutlierParams() and floor is None and source is None


def test_override_changes_detection_behaviorally():
    """The resolved params actually change detection (order-1 vs order-0)."""
    rng = np.random.default_rng(11)
    t = 2015.0 + np.arange(1200) / 365.25
    trend = 50.0 * np.sin(2 * np.pi * (t - 2015.0) / 1.5)  # sharp local curvature
    y = np.vstack([trend + rng.normal(0.0, 1.0, t.size) for _ in range(3)])
    sigma = np.full(y.shape, 1.0)
    base = OutlierParams()
    override = OutlierParams(window_order=1, despike=True, epoch_policy="union")
    flags_base, _ = gps_views.detect_view_outliers(t, y, sigma, outlier_params=base)
    flags_over, _ = gps_views.detect_view_outliers(t, y, sigma, outlier_params=override)
    assert int(flags_base.sum()) != int(flags_over.sum())
    # and the params_hash separates the two configurations
    assert gps_write.outlier_params_hash(base) != gps_write.outlier_params_hash(
        override
    )


def test_writer_catalog_override_provenance(write_env, tmp_path, monkeypatch):
    ov_csv = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(
        ov_csv, "sta,despike,window_order,epoch_policy\n", ["SENG,true,1,union\n"]
    )
    captured = _capture_params(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no steps/protect catalogs deployed
        result = gps_write.write_cleaned_neu(
            STA, out, Dir=TOT, outlier_overrides=ov_csv, **SAVETIMES_KW
        )
    # the resolved params (with overrides) reached the detector
    used = captured["params"]
    assert used.despike is True and used.window_order == 1
    assert used.epoch_policy == "union"
    det = result["detector"]
    assert det["outlier_overrides_applied"] == {
        "despike": True,
        "window_order": 1,
        "epoch_policy": "union",
    }
    assert det["outlier_overrides_source"] == str(ov_csv)
    # params/params_hash echo the RESOLVED params actually used
    assert det["params"]["despike"] is True and det["params"]["window_order"] == 1
    assert det["params_hash"] == gps_write.outlier_params_hash(
        gps_views.station_outlier_params(STA, catalog=ov_csv)[0]
    )


def test_writer_precedence_explicit_over_catalog_over_default(
    write_env, tmp_path, monkeypatch
):
    ov_csv = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(ov_csv, "sta,window_order\n", ["SENG,1\n"])

    # (1) explicit outlier_params WINS — catalog ignored entirely
    captured = _capture_params(monkeypatch)
    out1 = tmp_path / "explicit.NEU"
    explicit = OutlierParams(global_n_sigma=9.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = gps_write.write_cleaned_neu(
            STA,
            out1,
            Dir=TOT,
            outlier_params=explicit,
            outlier_overrides=ov_csv,
            **SAVETIMES_KW,
        )
    assert captured["params"].global_n_sigma == 9.0
    assert captured["params"].window_order == 0  # catalog NOT applied
    assert r1["detector"]["outlier_overrides_applied"] == {}
    assert r1["detector"]["outlier_overrides_source"] is None

    # (2) no explicit arg -> catalog override applies
    captured2 = _capture_params(monkeypatch)
    out2 = tmp_path / "catalog.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = gps_write.write_cleaned_neu(
            STA, out2, Dir=TOT, outlier_overrides=ov_csv, **SAVETIMES_KW
        )
    assert captured2["params"].window_order == 1
    assert r2["detector"]["outlier_overrides_applied"] == {"window_order": 1}

    # (3) no explicit, no catalog -> base default (byte/hash unchanged from today)
    captured3 = _capture_params(monkeypatch)
    out3 = tmp_path / "default.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r3 = gps_write.write_cleaned_neu(
            STA, out3, Dir=TOT, outlier_overrides=tmp_path / "nope.csv", **SAVETIMES_KW
        )
    assert captured3["params"] == OutlierParams()
    assert r3["detector"]["outlier_overrides_applied"] == {}
    assert r3["detector"]["outlier_overrides_source"] is None
    assert r3["detector"]["params_hash"] == gps_write.outlier_params_hash(
        OutlierParams()
    )


def test_overrides_compose_with_steps_and_protect_windows(
    write_env, tmp_path, monkeypatch
):
    steps_csv = tmp_path / "steps.csv"
    _write_steps_csv(steps_csv, [("SENG", 2020.5, "ALL", "equip", "tos", "a")])
    pw_csv = tmp_path / "protect_windows.csv"
    _write_protect_windows_csv(pw_csv, [("SENG", 2023.9, 2025.0, "unrest")])
    ov_csv = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(ov_csv, "sta,window_order,despike\n", ["SENG,1,true\n"])

    captured = _capture_params(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    result = gps_write.write_cleaned_neu(
        STA,
        out,
        Dir=TOT,
        steps=steps_csv,
        protect_windows=pw_csv,
        outlier_overrides=ov_csv,
        **SAVETIMES_KW,
    )
    # all three levers arrive in the SAME detection call
    np.testing.assert_array_equal(
        np.asarray(captured["step_epochs"], dtype=float), np.array([2020.5])
    )
    assert tuple(captured["protect_windows"]) == ((2023.9, 2025.0),)
    assert captured["params"].window_order == 1 and captured["params"].despike is True
    det = result["detector"]
    assert det["step_epochs_applied"] == 1
    assert det["protect_windows_applied"] == 1
    assert det["outlier_overrides_applied"] == {"window_order": 1, "despike": True}


def test_cleaned_view_records_outlier_overrides(write_env, tmp_path):
    ov_csv = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(ov_csv, "sta,window_order,despike\n", ["SENG,1,true\n"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = gps_views.read_gps_view(
            STA,
            view="cleaned",
            outlier_overrides=ov_csv,
            protect_windows=((2016.5, 2025.2),),  # keep it from aborting
            Dir=TOT,
        )
    attrs = df.attrs["gps_view"]
    assert attrs["outlier_overrides_source"] == str(ov_csv)
    assert attrs["outlier_overrides_applied"] == {"window_order": 1, "despike": True}


# ---------------------------------------------------------------------------
# per-component min_outlier floor (detect_outliers kwarg)
# ---------------------------------------------------------------------------


def _spiked_series(seed=4):
    """Clean periodic series with an 8 mm (8-sigma) spike in N and in U."""
    rng = np.random.default_rng(seed)
    t = 2015.0 + np.arange(800) / 365.25
    base = 3.0 * np.cos(2 * np.pi * t) + 1.5 * np.sin(2 * np.pi * t)
    y = np.vstack([base + rng.normal(0.0, 1.0, t.size) for _ in range(3)])
    sigma = np.full(y.shape, 1.0)
    i_n, i_u = 200, 500
    y[0, i_n] += 8.0  # N: statistical outlier, magnitude 8 mm
    y[2, i_u] += 8.0  # U: statistical outlier, magnitude 8 mm
    return t, y, sigma, i_n, i_u


def test_per_component_floor_gates_by_magnitude():
    """[5,5,10]: the 8 mm N spike (>5) is flagged, the 8 mm U spike (<10) is not."""
    t, y, sigma, i_n, i_u = _spiked_series()
    bare, _ = gps_views.detect_view_outliers(t, y, sigma)
    assert bare[0, i_n] and bare[2, i_u]  # both are real statistical outliers

    floored, _ = gps_views.detect_view_outliers(t, y, sigma, min_outlier=[5, 5, 10])
    assert floored[0, i_n], "N spike (8 mm) is above its 5 mm floor -> flagged"
    assert not floored[2, i_u], "U spike (8 mm) is below its 10 mm floor -> NOT flagged"


def test_min_outlier_scalar_broadcasts():
    triple = gps_views._normalize_min_outlier(7.0)
    assert triple == (7.0, 7.0, 7.0)
    assert gps_views._normalize_min_outlier([5, 5, 10]) == (5.0, 5.0, 10.0)
    with pytest.raises(ValueError, match="scalar or length-3"):
        gps_views._normalize_min_outlier([5, 5])
    with pytest.raises(ValueError, match="finite and >= 0"):
        gps_views._normalize_min_outlier([5, -1, 10])


def test_resolve_outlier_detection_precedence(tmp_path):
    ov_csv = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(
        ov_csv,
        "sta,window_order,min_outlier_n,min_outlier_e,min_outlier_u\n",
        ["SENG,1,5,5,10\n"],
    )
    # explicit min_outlier arg WINS over the catalog floor
    r_arg = gps_views.resolve_outlier_detection(
        STA, min_outlier=[1, 2, 3], outlier_overrides=ov_csv
    )
    assert r_arg.min_outlier == (1.0, 2.0, 3.0)
    assert r_arg.min_outlier_source == "explicit"
    assert r_arg.params.window_order == 1  # catalog params still apply

    # no explicit arg -> catalog floor applies
    r_cat = gps_views.resolve_outlier_detection(STA, outlier_overrides=ov_csv)
    assert r_cat.min_outlier == (5.0, 5.0, 10.0)
    assert r_cat.min_outlier_source == str(ov_csv)

    # neither -> None (leaf default)
    with pytest.warns(UserWarning, match="no outlier-override catalog"):
        r_none = gps_views.resolve_outlier_detection(
            STA, outlier_overrides=tmp_path / "nope.csv"
        )
    assert r_none.min_outlier is None and r_none.min_outlier_source is None


def test_explicit_params_still_lets_catalog_floor_apply(tmp_path):
    """explicit outlier_params bypasses catalog PARAMS but not the floor."""
    ov_csv = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(
        ov_csv,
        "sta,window_order,min_outlier_n,min_outlier_e,min_outlier_u\n",
        ["SENG,1,5,5,10\n"],
    )
    r = gps_views.resolve_outlier_detection(
        STA, outlier_params=OutlierParams(global_n_sigma=9.0), outlier_overrides=ov_csv
    )
    # params from the explicit arg (catalog window_order IGNORED)
    assert r.params.global_n_sigma == 9.0
    assert r.params.window_order == 0
    assert r.overrides_applied == {}
    # but the catalog's per-component floor STILL applies
    assert r.min_outlier == (5.0, 5.0, 10.0)
    assert r.min_outlier_source == str(ov_csv)


def test_both_explicit_skips_catalog(tmp_path):
    """explicit params + explicit floor -> catalog not consulted (no warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any catalog read would warn -> fail
        r = gps_views.resolve_outlier_detection(
            STA,
            outlier_params=OutlierParams(global_n_sigma=9.0),
            min_outlier=[5, 5, 10],
            outlier_overrides=tmp_path / "nope.csv",
        )
    assert r.params.global_n_sigma == 9.0
    assert r.min_outlier == (5.0, 5.0, 10.0)
    assert r.min_outlier_source == "explicit"
    assert r.overrides_source is None


def test_writer_records_min_outlier_provenance(write_env, tmp_path, monkeypatch):
    captured = _capture_params(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gps_write.write_cleaned_neu(
            STA, out, Dir=TOT, min_outlier=[5, 5, 10], **SAVETIMES_KW
        )
    # the floor reached the detector
    np.testing.assert_array_equal(
        np.asarray(captured["min_outlier"], dtype=float), np.array([5.0, 5.0, 10.0])
    )
    det = result["detector"]
    assert det["min_outlier"] == [5.0, 5.0, 10.0]
    assert det["min_outlier_source"] == "explicit"
    # params_hash folds the floor in
    assert det["params_hash"] == gps_write.outlier_params_hash(
        OutlierParams(), [5.0, 5.0, 10.0]
    )
    assert det["params_hash"] != gps_write.outlier_params_hash(OutlierParams())


def test_writer_catalog_min_outlier_provenance(write_env, tmp_path, monkeypatch):
    ov_csv = tmp_path / "outlier_overrides.csv"
    _write_overrides_csv(
        ov_csv,
        "sta,min_outlier_n,min_outlier_e,min_outlier_u\n",
        ["SENG,5,5,10\n"],
    )
    captured = _capture_params(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gps_write.write_cleaned_neu(
            STA, out, Dir=TOT, outlier_overrides=ov_csv, **SAVETIMES_KW
        )
    np.testing.assert_array_equal(
        np.asarray(captured["min_outlier"], dtype=float), np.array([5.0, 5.0, 10.0])
    )
    det = result["detector"]
    assert det["min_outlier"] == [5.0, 5.0, 10.0]
    assert det["min_outlier_source"] == str(ov_csv)


def test_writer_default_no_floor_hash_unchanged(write_env, tmp_path, monkeypatch):
    """Zero regression: no catalog + no arg -> min_outlier None, hash == today."""
    captured = _capture_params(monkeypatch)
    out = tmp_path / f"{STA}-{REF}_cleaned.NEU"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gps_write.write_cleaned_neu(
            STA, out, Dir=TOT, outlier_overrides=tmp_path / "nope.csv", **SAVETIMES_KW
        )
    assert captured["min_outlier"] is None  # leaf default, not (0,0,0)
    det = result["detector"]
    assert det["min_outlier"] is None
    assert det["min_outlier_source"] is None
    # the None branch reproduces the pre-floor payload byte-for-byte
    assert det["params_hash"] == gps_write.outlier_params_hash(OutlierParams())


def test_cleaned_view_records_min_outlier(write_env):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = gps_views.read_gps_view(
            STA,
            view="cleaned",
            min_outlier=[5, 5, 10],
            protect_windows=((2016.5, 2025.2),),  # keep it from aborting
            Dir=TOT,
        )
    attrs = df.attrs["gps_view"]
    assert attrs["min_outlier"] == [5.0, 5.0, 10.0]
    assert attrs["min_outlier_source"] == "explicit"
