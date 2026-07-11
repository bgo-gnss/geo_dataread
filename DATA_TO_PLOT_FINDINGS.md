# Data → plot baseline — findings & fix/defer ledger

> From the real-data verification (2026-07-11) driving `/mnt_data/gpsdata`
> (original GLOBK solutions) → `getData` → the modernized `gps_plot`, per
> `.interrogate-data-to-plot-baseline.md`. **A = verify + characterize;** the
> golden masters + `.NEU` are the tripwire against B.

## Verified working
- Core read (`getData`) on real data: **SENG 4834 epochs**, itrf2008 and plate.
- Full **data → plot end-to-end** on real stations (SENG, HOFN, REYK, AKUR),
  `ref="plate"`, native **PNG/PDF/EPS**, correct semantic green header, plate name
  resolved ("Svartsengi … North American plate"). By-eye: correct (SENG shows the
  real Sundhnúkur 2023–2026 unrest). Samples: `verify_data_to_plot/`.

## Drift characterization (the §10.1 R2 concern) — DOWNGRADED
`getData` (plot) = `openGlobkTimes → dPeriod → iprep(uncert=15) → subtract
plateVel·1000·Δt`. `gamittoNEU` (.NEU) = `openGlobkTimes → vshift(uncert=1.1,
Period=5) → subtract plateVel·Δt → ×1000`. Both use the **same** `plateVel` with the
**same axis-swap** (N←`[0,1]`, E←`[0,0]`) and **same** `(yearf−yearf[0])`.

**Numerical result (SENG, 2768 common epochs):** after removing a constant, the two
agree to **std 0.0000 mm / max 0.0000 mm** in N, E, U. The plate/unit *math is exactly
equivalent* — `(d−v·Δt)·1000 ≡ d·1000 − v·1000·Δt` (order is cosmetic, not numerical).
The only differences are **preprocessing**: outlier threshold (`uncert` 1.1 vs 15 →
different epoch subsets) and reference/offset (`iprep` returns an ~8.3 m offset the plot
zero-references with; the `.NEU` keeps it). Neither is a units/plate correctness bug.
*(Verified on SENG; re-confirm on 1–2 more stations before B leans on it.)*

## Fix / defer ledger
| # | Finding | Class | Disposition |
|---|---|---|---|
| 1 | `station-plate` not deployed to `~/.config/gpsconfig` | deploy gap | **fixed locally** (copied from gps-config-data) |
| 2 | `coordFile` (capital-F) = stale `chiara_bardabunga/*.llh`; case-collides with `coordfile` (`/mnt/gpsconfig/*.xyz`, absent locally); `geofunc.plateVelo` reads `coordFile` | config drift (§10.4) | **fixed LOCAL dev cfg** (both keys → deployed `station_coord.xyz`); **source fix touches `.NEU` (gamittoNEU uses plateVelo) → B + sign-off** |
| 3 | `skiprows=3` "silent epoch drop" | false alarm | none — 3 real header lines (GGVer, `…m` ref, blank) |
| 4 | `plotTime(save=None)` launches WebAgg + hangs headless | wart (gps_plot) | fix-in-A candidate (off `.NEU`/masters); workaround: `MPLBACKEND=Agg` / pass `save=` |
| 5 | plate/unit "order drift" (`gamittoNEU` vs `getData`) | characterized | **not a value bug** (identical to float eps); real diff = preprocessing (see above) → B refactor de-risked |
| 6 | `08h`/`JOIN` dead branch (undefined names) | dead code | leave crash-pinned; **BGÓ uses JOIN, revive later** (not first priority) |
| 7 | **Stale venv shebangs** after the 2026-07-10 repo move: `.venv/bin/*` still point at old `gps/gpslibrary_new/...`; `uv run` silently falls back to **system python** → misleading ImportError / the `black`/`mypy` "Failed to spawn" seen earlier | env wart (cross-repo) | fixed pytest in geo_dataread; **all moved repos need `uv sync --reinstall`** |
| 8 | synthetic goldens feed an **`.llh`** coord file into `plateVelo`, which actually needs **ECEF XYZ** (`np.cross`+geocentric) → the synthetic plate velocities are stable-but-physically-meaningless; the new real-data goldens pin production-correct XYZ (~1–2 cm/yr) | characterization | note for B config reconciliation; synthetic goldens left untouched |

## Golden fixtures (done 2026-07-11)
`tests/goldenmaster/` gains **15 real-data cases** (SENG/HOFN/REYK, last 500 epochs frozen,
self-contained; coord/plate config bundled per-fixture, CI needs no `~/.config/gpsconfig`):
`openGlobkTimes`, `getData` (itrf2008 + plate), `gamittoNEU` (savetimes profile),
`gamittooneuf` (.NEU files). **`uv run pytest tests/goldenmaster/` → 38 passed** (23 existing +
15 new). Existing synthetic goldens untouched. Sensitivity sane (NOAM→EURA = 22 mm; plate vs
itrf2008 = 26 mm). On real headers `skiprows=3` lands exactly on row 1 of data — **no epoch loss**
(confirms ledger #3). No crashes/NaNs/mismatches vs the characterization on any station.

## Local env changes made (dev only — production gps-config-data UNTOUCHED)
- Deployed `station-plate`, `station_coord.xyz` → `~/.config/gpsconfig/`.
- `~/.config/gpsconfig/postprocess.cfg`: both coord keys → the deployed `station_coord.xyz`.
- `geo_dataread` installed editable into `gps_plot`'s env (so `plotTime → getData` runs).

## Remaining A deliverables
- **Real-data golden fixtures** — 2–3 real stations (SENG/HOFN/REYK) frozen into
  `tests/goldenmaster/` (pin current behavior; do NOT recapture existing goldens).
- WebAgg wart fix (#4) — BGÓ's call (gps_plot scope).
- Source config fix (#2) + any plate/unit reconciliation → **B + sign-off** (touches `.NEU`).

## `/mnt_data/gpsdata` data-hygiene (2026-07-11, from B slice-6 `read_join` checks)
`read_join(sta, schemes=…)` is **string-driven / semantics-agnostic** — each scheme is just
the `mb_{STA}_{scheme}.dat{1,2,3}` filename token; it globs + concatenates whatever matches.
Verified against real overlapping data (**DYNG 2019**: 454 `TOT` + 269 `08h` correctly
interleaved by time). Data-hygiene notes on the dir itself (not code):
- **Scheme inventory:** 427 `TOT`, **475 `08h`**, **2 `8hr`** (only `DYNG`, `MOHA` — a legacy
  spelling of the standard `08h`), 4 `GPS`. The `08h`/`8hr` split is a naming inconsistency;
  `read_join` handles either (pass the right token), but worth normalizing someday.
- **8-hour coverage is 2016–2021 only** (earliest epoch anywhere `2016.00136`, latest
  `2021.98767`). **No 2014 8-hour data is present** in this dir — e.g. DYNG's remembered
  Aug–Dec 2014 8h solution isn't here (its `08h`=2019, `8hr`=2021 files are later
  reprocessings). If needed, the 2014 8h lives in an archive elsewhere.
- Minor: `TOT` can carry two globk epochs that `convGlobktopandas`'s 1 h index-rounding
  collapses to one timestamp (visible as a doubled row) — legacy behavior, golden-pinned.

---
*Created 2026-07-11 from the real-data verification; §"data-hygiene" added from the B/read_join checks.*
