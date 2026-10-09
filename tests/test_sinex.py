"""geo_dataread.sinex: daily network solutions with the full covariance."""

from __future__ import annotations

import datetime
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geo_dataread import sinex

# Two Icelandic-ish stations ~125 m apart plus one global (geocentre) row.
XYZ = {
    "AAAA": (2_587_000.000, -1_043_000.000, 5_716_000.000),
    "BBBB": (2_587_080.000, -1_043_050.000, 5_715_970.000),
}


def _cov(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) * 1e-3
    return a @ a.T + np.eye(n) * 1e-6  # SPD, mm-level, correlated


def _write(
    path: Path,
    cov: np.ndarray,
    *,
    kind: str = "COVA",
    tri: str = "L",
    dup_soln: bool = False,
    gz: bool = False,
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []  # (type, code, soln)
    for code in XYZ:
        rows += [(t, code, "1") for t in ("STAX", "STAY", "STAZ")]
    if dup_soln:
        rows += [(t, "BBBB", "2") for t in ("STAX", "STAY", "STAZ")]
    rows.append(("XGC", "----", "1"))
    n = len(rows)
    assert cov.shape == (n, n)
    vals = []
    for t, code, _ in rows:
        vals.append(0.0 if code == "----" else XYZ[code]["XYZ".index(t[-1])])
    lines = [
        f"%=SNX 2.01 IMO 26:282:02100 IMO 26:244:00000 26:244:86370 P {n:05d} 2 S E",
        "+SITE/ID",
        "*Code Pt __Domes__ T _Station Description__ _Longitude_ _Latitude__ _Height",
    ]
    for code in XYZ:
        lines.append(
            f" {code}  A 10202M001 P {code + '_GPS':<22} 337 48 00.0  64 08 00.0    50.0"
        )
    lines += ["-SITE/ID", "+SITE/ANTENNA"]
    for code in XYZ:
        lines.append(
            f" {code}  A    1 P 26:244:00000 26:244:86370 {'TRM59800.00     NONE':<20} 12345    0"
        )
    lines += ["-SITE/ANTENNA", "+SITE/RECEIVER"]
    for code in XYZ:
        lines.append(
            f" {code}  A    1 P 26:244:00000 26:244:86370 {'SEPT POLARX5':<20} 4711  5.5"
        )
    lines += ["-SITE/RECEIVER", "+SITE/ECCENTRICITY"]
    for code in XYZ:
        lines.append(
            f" {code}  A    1 P 26:244:00000 26:244:86370 UNE   0.2000   0.0000   0.0000"
        )
    lines += [
        "-SITE/ECCENTRICITY",
        "+SOLUTION/STATISTICS",
        f" {'VARIANCE FACTOR':<30} 0.5",
        "-SOLUTION/STATISTICS",
    ]
    for block, sd in (("ESTIMATE", None), ("APRIORI", 100.0)):
        lines.append(f"+SOLUTION/{block}")
        lines.append(
            "*Index _Type_ Code Pt Soln _Ref_Epoch__ Unit S __Value____ _Std_Dev___"
        )
        for i, ((t, code, soln), v) in enumerate(zip(rows, vals, strict=True)):
            s = sd if sd is not None else float(np.sqrt(cov[i, i]))
            lines.append(
                f"{i + 1:6d} {t:<6} {code:>4}  A {soln:>4} 26:244:43185 m    2 "
                f"{v + (0.001 if sd is None else 0.0): .15E} {s:.6E}"
            )
        lines.append(f"-SOLUTION/{block}")
    shown = cov.copy()
    if kind == "CORR":
        s = np.sqrt(np.diag(cov))
        shown = cov / np.outer(s, s)
        np.fill_diagonal(shown, s)
    lines.append(f"+SOLUTION/MATRIX_ESTIMATE {tri} {kind}")
    for r in range(n):
        cols = range(r + 1) if tri == "L" else range(r, n)
        cols = list(cols)
        for k in range(0, len(cols), 3):
            chunk = cols[k : k + 3]
            lines.append(
                f"{r + 1:6d}{chunk[0] + 1:6d}"
                + "".join(f" {shown[r, c]: .14E}" for c in chunk)
            )
    lines.append(f"-SOLUTION/MATRIX_ESTIMATE {tri} {kind}")
    lines.append("%ENDSNX")
    text = "\n".join(lines) + "\n"
    if gz:
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return rows


def test_imports_nothing_heavy() -> None:
    code = (
        "import sys, geo_dataread.sinex; "
        "bad = [m for m in ('matplotlib', 'gps_analysis', 'gps_parser', 'pandas') "
        "if m in sys.modules]; print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


@pytest.mark.parametrize(
    ("kind", "tri", "gz"),
    [("COVA", "L", False), ("COVA", "U", True), ("CORR", "L", False)],
)
def test_reads_estimates_and_full_covariance(tmp_path, kind, tri, gz) -> None:
    cov = _cov(7)
    f = tmp_path / ("x.snx.gz" if gz else "x.snx")
    _write(f, cov, kind=kind, tri=tri, gz=gz)
    sol = sinex.read_sinex(f)
    assert sol.n_param == 7
    assert sol.station_codes() == ["AAAA", "BBBB"]
    np.testing.assert_allclose(sol.cov, cov, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(sol.xyz("BBBB"), np.array(XYZ["BBBB"]) + 0.001)
    np.testing.assert_allclose(sol.apriori[:6], np.concatenate(list(XYZ.values())))
    assert sol.start == datetime.datetime(2026, 9, 1)
    assert sol.statistics["VARIANCE FACTOR"] == 0.5
    assert sol.ptype[-1] == "XGC"  # non-station rows are kept, not reshaped away


def test_site_metadata(tmp_path) -> None:
    _write(tmp_path / "x.snx", _cov(7))
    site = sinex.read_sinex(tmp_path / "x.snx").sites["AAAA"]
    assert site.domes == "10202M001"
    assert site.description == "AAAA_GPS"
    assert site.antenna == "TRM59800.00     NONE"
    assert site.receiver == "SEPT POLARX5"
    assert site.ecc_une == (0.2, 0.0, 0.0)


def test_baseline_counts_the_cross_covariance_once(tmp_path) -> None:
    cov = _cov(7)
    _write(tmp_path / "x.snx", cov)
    sol = sinex.read_sinex(tmp_path / "x.snx")
    d, c = sinex.baseline_neu(sol, "AAAA", "BBBB")
    r = sinex.neu_rotation(*sinex.xyz_to_llh(sol.xyz("AAAA"))[:2])
    np.testing.assert_allclose(d, r @ (sol.xyz("AAAA") - sol.xyz("BBBB")))
    caa, cbb, cab = cov[:3, :3], cov[3:6, 3:6], cov[:3, 3:6]
    expect = r @ (caa + cbb - cab - cab.T) @ r.T
    np.testing.assert_allclose(c, expect, rtol=1e-12)
    quad = r @ (caa + cbb) @ r.T
    assert not np.allclose(np.diag(c), np.diag(quad))


def test_identical_errors_cancel_in_a_baseline(tmp_path) -> None:
    """Perfectly common-mode error → zero baseline variance."""
    block = _cov(3)
    cov = np.zeros((7, 7))
    for i in (0, 3):
        for j in (0, 3):
            cov[i : i + 3, j : j + 3] = block
    cov[6, 6] = 1e-4
    _write(tmp_path / "x.snx", cov)
    _, c = sinex.baseline_neu(sinex.read_sinex(tmp_path / "x.snx"), "AAAA", "BBBB")
    np.testing.assert_allclose(c, 0.0, atol=1e-18)


def test_a_station_under_two_solutions_must_be_named(tmp_path) -> None:
    _write(tmp_path / "x.snx", _cov(10), dup_soln=True)
    sol = sinex.read_sinex(tmp_path / "x.snx")
    with pytest.raises(sinex.SinexError, match="several solutions"):
        sol.xyz_index("BBBB")
    assert sol.xyz_index("BBBB", soln="2").tolist() == [6, 7, 8]
    assert sol.cov_block(["AAAA", "BBBB"], ["1", "2"]).shape == (6, 6)


def test_refusals(tmp_path) -> None:
    (tmp_path / "bad.snx").write_text("not a sinex\n")
    with pytest.raises(sinex.SinexError, match="not a SINEX"):
        sinex.read_sinex(tmp_path / "bad.snx")
    _write(tmp_path / "x.snx", _cov(7))
    with pytest.raises(sinex.SinexError, match="no STAX"):
        sinex.read_sinex(tmp_path / "x.snx").xyz_index("ZZZZ")


def test_llh_round_trip() -> None:
    lat, lon, h = sinex.xyz_to_llh(XYZ["AAAA"])
    n = 6378137.0 / np.sqrt(1 - sinex._E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    z = (n * (1 - sinex._E2) + h) * np.sin(lat)
    np.testing.assert_allclose([x, z], [XYZ["AAAA"][0], XYZ["AAAA"][2]], atol=1e-6)
    r = sinex.neu_rotation(lat, lon)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-15)
