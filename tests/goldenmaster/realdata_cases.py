"""Real-data golden-master cases for gps_read (data->plot baseline, scope A).

Extends the Phase-0 synthetic suite (cases.py, commit 21bf3ef) with fixtures
frozen from REAL GLOBK solutions, pinning the CURRENT behavior of the
production-critical read path on production-shaped data ahead of the S10.1
refactor. See DATA_TO_PLOT_FINDINGS.md and .interrogate-data-to-plot-baseline.md.

Fixture provenance (frozen 2026-07-11, self-contained — tests never read the
production mount):

- fixtures/realdata/TOT/mb_{SENG,HOFN,REYK}_TOT.dat{1,2,3} — the 3 real GLOBK
  header lines (GGVer line, "<STA>_GPS to <comp> Solution ... m" reference
  line, blank line) + the LAST 500 epochs of /mnt_data/gpsdata (original GLOBK
  solutions, snapshot ending 2026.52192). Unlike the synthetic fixtures (which
  are headerless, so skiprows=3 eats 3 epochs), these have real headers —
  skiprows=3 lands exactly on the first data row. Slices per station:
  SENG 2025.13561-2026.52192 (Sundhnukur unrest), HOFN 2025.14931-2026.52192,
  REYK 2025.13287-2026.52192.
- fixtures/realdata/gpsconfig/station_coord.xyz — the SENG/HOFN/REYK rows of
  the deployed ~/.config/gpsconfig/station_coord.xyz (ECEF XYZ, the format
  geofunc.plateVelo actually consumes: rotpole cross-product + itrf08towgs84
  need cartesian meters). NOTE: the synthetic suite's station_coord.llh feeds
  lon/lat/h into the same code path — pinned there as-is; here we pin the
  production-real XYZ behavior.
- fixtures/realdata/gpsconfig/station-plate — the SENG/HOFN/REYK rows of the
  deployed station-plate (HOFN EURA, REYK NOAM, SENG NOAM).

Argument profiles mirror the live call sites (same survey as cases.py):
openGlobkTimes raw read, getData for gps_plot (ref itrf2008 AND plate),
gamittoNEU/gamittooneuf for the .NEU product path (gps-savetimes profile:
mm=True, ref="plate", dstring=None, rhour=True).

Config isolation: every runner swaps GPS_CONFIG_PATH to the rendered realdata
config dir for the duration of the call (gps_parser.ConfigParser re-reads the
env var on every instantiation, and gps_read/geofunc construct a fresh parser
per call), then restores it — so these cases coexist with the synthetic
suite's session-wide GPS_CONFIG_PATH in any test order, and nothing ever reads
~/.config/gpsconfig.
"""

import contextlib
import os
from pathlib import Path

import geo_dataread.gps_read as gpsr

from cases import _structured_to_dict, _tuple_to_dict

HERE = Path(__file__).resolve().parent
REAL_FIXTURES = HERE / "fixtures" / "realdata"
REAL_TOT = str(REAL_FIXTURES / "TOT") + os.sep
REAL_EXPECTED = HERE / "expected" / "realdata"

REAL_STATIONS = ("SENG", "HOFN", "REYK")

REAL_POSTPROCESS_TEMPLATE = """\
[PATHS]
totDir = {totdir}

[FILES]
coordFile = {config_dir}/station_coord.xyz
platefile = {config_dir}/station-plate
"""


def build_real_config_dir(dest: Path) -> Path:
    """Render the hermetic realdata gpsconfig dir (for GPS_CONFIG_PATH)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("station_coord.xyz", "station-plate"):
        dest.joinpath(name).write_bytes(
            (REAL_FIXTURES / "gpsconfig" / name).read_bytes()
        )
    dest.joinpath("postprocess.cfg").write_text(
        REAL_POSTPROCESS_TEMPLATE.format(totdir=REAL_TOT, config_dir=dest)
    )
    return dest


@contextlib.contextmanager
def _real_config(env):
    """Point GPS_CONFIG_PATH at the realdata config dir, restore on exit."""
    old = os.environ.get("GPS_CONFIG_PATH")
    os.environ["GPS_CONFIG_PATH"] = str(env["config_dir"])
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GPS_CONFIG_PATH", None)
        else:
            os.environ["GPS_CONFIG_PATH"] = old


# --- case runners (same shape as cases.py: env -> dict[str, np.ndarray]) ----


def _real_ogt(sta):
    def run(env):
        with _real_config(env):
            yearf, data, Ddata = gpsr.openGlobkTimes(sta, Dir=REAL_TOT)
        return _tuple_to_dict(yearf, data, Ddata)

    return run


def _real_getdata(sta, **kwargs):
    def run(env):
        with _real_config(env):
            yearf, data, Ddata, offset = gpsr.getData(sta, Dir=REAL_TOT, **kwargs)
        return _tuple_to_dict(yearf, data, Ddata, offset)

    return run


def _real_gamittoneu(sta, **kwargs):
    def run(env):
        with _real_config(env):
            neu = gpsr.gamittoNEU(sta, Dir=REAL_TOT, **kwargs)
        return _structured_to_dict(neu)

    return run


REAL_CASES = {}
for _sta in REAL_STATIONS:
    REAL_CASES[f"real_ogt_{_sta}"] = _real_ogt(_sta)
    REAL_CASES[f"real_getdata_{_sta}_itrf2008"] = _real_getdata(_sta)
    REAL_CASES[f"real_getdata_{_sta}_plate"] = _real_getdata(_sta, ref="plate")
    # the LIVE gps-savetimes inner profile (feeds .NEU -> cdn.vedur.is)
    REAL_CASES[f"real_neu_{_sta}_savetimes"] = _real_gamittoneu(
        _sta, mm=True, ref="plate", dstring=None, rhour=True
    )

# .NEU product files: the exact production gps-savetimes call. Golden = file.
REAL_NEU_FILE_CASES = {f"real_neufile_{_sta}_plate": (_sta, "plate") for _sta in REAL_STATIONS}


def run_real_neu_file_case(case_id, env, out_dir: Path) -> Path:
    sta, ref = REAL_NEU_FILE_CASES[case_id]
    out = out_dir / f"{sta}-{ref}.NEU"
    with _real_config(env):
        gpsr.gamittooneuf(
            sta, str(out), mm=True, ref=ref, dstring=None, rhour=True, Dir=REAL_TOT
        )
    return out
