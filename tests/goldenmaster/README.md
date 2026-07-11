# Golden-master suite for `gps_read` (plan §15.6)

Pins the **current** behavior of `openGlobkTimes`, `getData`, `gamittoNEU`,
`gamittooneuf` (.NEU product files) and `read_gps_data` on frozen fixture
data, so the Phase 1 geo_dataread → gps_analysis refactor can prove it
changed nothing. See `PLAN-postprocessing-revamp.md` §10.1/§13 (risks R1/R5).

```bash
uv run pytest tests/goldenmaster/            # compare against goldens
uv run python tests/goldenmaster/capture_goldens.py   # REGENERATE goldens
```

**Never recapture casually.** A red golden test during the refactor means
"behavior moved" — that is the test doing its job. Recapture only when a
change is intentional and reviewed, and say so in the commit message.
The capture script runs every case twice and refuses to write on any
run-to-run difference (nondeterminism guard).

## Layout

- `cases.py` — single case registry shared by capture and tests (no drift).
  Argument profiles mirror the 2026-07-08 production call-site survey.
- `fixtures/TOT/` — frozen GAMIT `mb_*.dat{1,2,3}` series (SENG, ELDC, SKSH,
  OLAC, DYNG, ENTC; snapshot ending 2024.98, spans the Reykjanes unrest).
  `mb_SENG_08h.*` are TOT copies that existed to reach the (since-removed)
  broken 08h branch; kept frozen, no longer reached — openGlobkTimes now
  rejects `tType="08h"` before file I/O. ENTC is there because it is the only station with sigmas in
  (1.1 m, 20 m) — it pins `gamittoNEU`'s hardcoded `vshift(uncert=1.1)`.
- `fixtures/gpsconfig/` — hermetic config (stations.cfg subset, postprocess
  template, detrend CSV, station-plate, station_coord.llh). Tests set
  `GPS_CONFIG_PATH` to a rendered copy; nothing reads `~/.config/gpsconfig`.
- `expected/` — the goldens: `.npz` per case + literal `.NEU` product files.

## Real-data cases (added 2026-07-11, data→plot baseline scope A)

`realdata_cases.py` / `test_realdata.py` / `capture_realdata_goldens.py`
extend the suite with fixtures frozen from **real GLOBK solutions** —
the safety net the §10.1 refactor runs against production-shaped data.
See `DATA_TO_PLOT_FINDINGS.md` at the repo root.

- `fixtures/realdata/TOT/` — SENG, HOFN, REYK: the 3 real header lines +
  the **last 500 epochs** of `/mnt_data/gpsdata/mb_*_TOT.dat{1,2,3}`
  (snapshot 2026-07-11, all slices end 2026.52192; SENG spans the
  Sundhnúkur unrest). Self-contained — tests never read the mount.
  Unlike the synthetic fixtures these have real headers, so `skiprows=3`
  lands exactly on the first data row (no epoch loss).
- `fixtures/realdata/gpsconfig/` — SENG/HOFN/REYK subset of the deployed
  `station_coord.xyz` (**ECEF XYZ** — the format `geofunc.plateVelo`
  actually consumes; the synthetic suite's `.llh` file is pinned as-is,
  separately) + `station-plate` (HOFN EURA, REYK/SENG NOAM). Rendered to a
  temp config dir; runners swap `GPS_CONFIG_PATH` per call and restore it,
  so the real cases coexist with the synthetic session env in any order.
- `expected/realdata/` — the real goldens: `.npz` per case + `.NEU` files.
  Recapture (only when intentional + reviewed):
  `uv run python tests/goldenmaster/capture_realdata_goldens.py`
- Pinned per station: `openGlobkTimes` raw read; `getData` with
  `ref="itrf2008"` and `ref="plate"` (gps_plot profile — the real SENG
  slice exercises the 15 mm `iprep` threshold: 4 epochs filtered);
  `gamittoNEU` + `gamittooneuf` savetimes profile (`mm=True, ref="plate",
  dstring=None, rhour=True` — the `.NEU → cdn.vedur.is` path).
- Sensitivity (checked 2026-07-11): swapping SENG's plate NOAM→EURA moves
  `real_getdata_SENG_plate` by up to 22 mm; plate vs itrf2008 goldens
  differ by up to 26 mm — the plate path is genuinely pinned.

## Verified sensitivity (mutation-tested 2026-07-08)

- Swapping plate-subtraction/mm-conversion order in `gamittoNEU` → 5 tests fail.
- Loosening `vshift(uncert=1.1)` to 20.0 → `neu_ENTC_savetimes` fails.
- `getData`'s 15/20 mm thresholds are discriminated by SENG epochs with
  sigmas in (15, 20) mm.

## Dead options pinned as CLEAN ERRORS (updated in refactor-B slice 4, D4)

Both used to be pinned as ACCIDENTAL crashes; slice 4 removed the dead
branches and replaced them with explicit `ValueError`s — the only deliberate
golden change of that slice (pins updated `KeyError`/`NameError` → `ValueError`):

- `read_gps_data(..., useSTA=..., useFIT="periodic")` → `ValueError`
  (was `KeyError('Fit')`: getDetrFit wrote lowercase `fit`/`useSTA` columns
  while read_gps_data read `Fit` — the pygmt-era borrowing branch was dead).
- `openGlobkTimes(..., tType="08h")` → `ValueError` (was `NameError` on
  undefined `shiftime`/`timetoyearf`/`todatetime`). gps_plot's `tType="JOIN"`
  sub-daily path stays dead until the slice-6 JOIN revival.

## Other quirks deliberately preserved in the goldens

- `openGlobkTimes` skiprows=3 on headerless files → first 3 epochs dropped.
- `gamittoNEU` subtracts plate velocity in **meters before** mm conversion;
  `getData` subtracts `plateVelo*1000` on mm data after `iprep`.
- With the pinned gps_parser 0.3.0, `detrendFile` resolves relative to the
  **current working directory** (tests chdir accordingly; the missing-CSV
  fallback branch is pinned by `rgd_SENG_no_csv`).

---
*Created 2026-07-08 (Phase 0 exit item §15.6). Fixture source:
`~/work/projects/gps_data_analyses/chiara_bardabunga/TOT` snapshot.*
