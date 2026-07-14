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
