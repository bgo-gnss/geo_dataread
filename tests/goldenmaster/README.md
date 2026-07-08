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
  `mb_SENG_08h.*` are TOT copies that exist only to reach the broken 08h
  branch. ENTC is there because it is the only station with sigmas in
  (1.1 m, 20 m) — it pins `gamittoNEU`'s hardcoded `vshift(uncert=1.1)`.
- `fixtures/gpsconfig/` — hermetic config (stations.cfg subset, postprocess
  template, detrend CSV, station-plate, station_coord.llh). Tests set
  `GPS_CONFIG_PATH` to a rendered copy; nothing reads `~/.config/gpsconfig`.
- `expected/` — the goldens: `.npz` per case + literal `.NEU` product files.

## Verified sensitivity (mutation-tested 2026-07-08)

- Swapping plate-subtraction/mm-conversion order in `gamittoNEU` → 5 tests fail.
- Loosening `vshift(uncert=1.1)` to 20.0 → `neu_ENTC_savetimes` fails.
- `getData`'s 15/20 mm thresholds are discriminated by SENG epochs with
  sigmas in (15, 20) mm.

## Behavior pinned as CRASHES (known-broken, do not "fix" silently)

- `read_gps_data(..., useSTA=..., useFIT="periodic")` → `KeyError('Fit')`
  (getDetrFit writes lowercase `fit`/`useSTA` columns; read_gps_data reads
  `Fit`/`useSTA` — the pygmt-era borrowing branch is dead).
- `openGlobkTimes(..., tType="08h")` → `NameError` (undefined `shiftime`/
  `timetoyearf`/`todatetime`), so gps_plot's `tType="JOIN"` path is dead.

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
