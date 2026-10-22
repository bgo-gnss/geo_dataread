# geo_dataread

Geospatial data I/O — GPS displacement and coordinate time series, plus auxiliary readers
(GAS strain, SIL seismic, hydrology). **Tier 1** — processing layer; depends on Tier 0.

> **Note**: Modern import is `geo_dataread`. Older docs and the project README sometimes
> refer to it as `dread.*` — same package, legacy alias.

## Status

Stable, ~3948 LOC. Used by research workflows and `gps_plot`.

> **Refactor guard (Phase 0, plan §15.6):** `tests/goldenmaster/` pins current
> behavior of `openGlobkTimes`/`getData`/`gamittoNEU`/`gamittooneuf`/
> `read_gps_data` on frozen fixtures — run `uv run pytest tests/goldenmaster/`
> before AND after touching `gps_read.py`; never recapture goldens casually
> (see that dir's README). Production reality per the 2026-07-08 call-site
> survey: aflogun does NOT call this package anymore — it consumes the .NEU
> files that `gps-savetimes` (→ `gamittooneuf`) publishes to cdn.vedur.is.
> Live callers: `gps-savetimes`, `gps-displacemnts` (`simpleDisp`), and
> `gps_plot.timesmatplt` (`getData`). `read_gps_data` is research-only.
> Dead option `useFIT="periodic"` (was an accidental-crash pin): removed in
> refactor-B slice 4 — now raises a clean `ValueError`. `tType="08h"` got the
> same treatment in slice 4 but was REVIVED in slice 6 (JOIN, D4):
> `openGlobkTimes` reads any scheme with `mb_STA_<scheme>.dat{1,2,3}` files
> (missing scheme → clear `FileNotFoundError`), and the new `read_join(sta,
> schemes=("TOT","08h"))` holds multiple processing schemes together in one
> scheme-labeled, time-sorted DataFrame. `getData(tType="JOIN")` stays a
> legacy alias for TOT (array return, back-compat).
> Slice 7 (`refactor-B-pure-io`): the model evaluators (`line`, `periodic`,
> `xf`, `expxf`, `expf`, `dexpf`, `secondorder`) are bitwise-identical shims
> over `gps_analysis.models`, pinned by `tests/test_model_shims.py`;
> `lineperiodic` deliberately keeps its local single-expression evaluation
> (leaf association differs by ≤1 ulp; it feeds curve_fit on golden-pinned
> paths). Remaining-work ledger: `PLAN-geo_dataread-refactor-B.md`.

## Layout

```
geo_dataread/
├── src/geo_dataread/
│   ├── __init__.py
│   ├── gps_read.py       # ~1660 LOC — main GPS time-series reader (slice-4 purge + slice-6 read_join)
│   ├── gps_views.py      # apply-on-read views: raw|cleaned|detrended toggle (cleaned: OutlierParams.method running → gps_analysis.detect_outliers_running, prior-free, steps not used); window_datum_shift = the constant between a windowed getData read and the full-span one a record is fitted to (typed, mypy-strict)
│   ├── baseline.py       # baselines A−B: split_baseline, match_epochs (by day), baseline() → Baseline incl. both ends referenced alike, check_same_view. NUMPY ONLY (a test pins it) so gps_api can reuse it; feed it read_gps_view output
│   ├── baseline_noise.py # BaselineNoiseRecord (ρ(d) law + station σ scale k + median k + window/provenance) → $GPS_API_STORE/noise/baseline_noise.json (atomic write; NOT the config tree); fit_baseline_noise, own_shared_fraction, pair_inputs (own ρ ≥60 consecutive days, else ρ(d)); gps-baseline-noise refits from TOT (stations.cfg lat/lon → geofunc.local)
│   ├── sinex.py          # numpy-only SINEX reader (full covariance); geometry via geofunc.local
│   ├── stabilize.py      # glorg frame stabilization re-implemented on the LOOSE daily SINEX (pmu/translation partials, geocentric-NEU 75/25 weights, height/NE screens, n-sigma deletion, 4 iterations from loose, applied via geofunc.local.condition over the FULL covariance) → Stabilized(solution, theta±, iteration reports); reproduces org positions+σ of every site to ≤0.011 mm (numpy only)
│   ├── globk_apr.py      # GLOBK .apr (pos+vel+epoch, EXTENDED EXP/LOG post-seismic in NEU) + eq rename (8-char chains, 4-char code, _XCL) readers → apriori_xyz(codes, epoch) (numpy only)
│   ├── realign.py        # gps-realign: loose archive → stabilize per day in itrf2008 (glorg frame, reproduces tssum .pos) or igb14 (GAMIT tables apr/eq), optional --to itrf2020 (EPSG Helmert w/ cov) → per-station .pos + days.csv
│   ├── org.py            # glorg .org reader: read_org_positions (final XYZ/adj/σ), read_org_setup (use_site codes + pos_org/stab_ite/cnd_hgtv → StabilizeConfig), match_sinex_rows (code + nearest apriori; _GPS/_nPS suffix ≠ SINEX soln) (numpy only)
│   ├── pos.py            # numpy-only PBO .pos reader (tssum output, okada /D/SNX/pos): PosSeries (XYZ/NEU + σ + correlations, GLOBK yearf); openGlobkTimes(tType="POS") reads it in place of mb_ files; POS_PRODUCTS = imo.daily_itr08 (okada) / imo.daily_igb14 (realign) / imo.final_igb20 (gamit-p01 finals, alias igb20 = itrf2020)
│   ├── station_catalog.py # platefile/coordFile lookups + plate_velocity_neu (N,E,U via geofunc.plates; replaces gf.plateVelo's E,N,U + the swap in _remove_plate_velocity) (typed)
│   ├── gps_write.py      # cleaned .NEU writer: gamittoNEU→gamittoFile, union-drop + .prov.json sidecar; steps.csv→step_epochs + protect_windows.csv (unrest lever) + outlier_overrides.csv (per-station levers incl. per-component min_outlier floor [N,E,U]); degrade → _cleaned.DEGRADED.NEU (typed, mypy-strict)
│   ├── gps_displ.py      # displacements / station-relative motion
│   ├── gps_savetimes.py  # serialise time series to disk (gps-savetimes; --clean also-writes cleaned .NEU)
│   ├── globk_join.py     # GLOBK multibase segment read + datum-consistent join (10 m de-wrap, min-σ dedup) + mb_STA_TOT.dat writer (format_tot_file/write_joined_series, 3-line-header contract) (typed, mypy-strict)
│   ├── globk_tot.py      # gps-globk-tot CLI: batch pre/rap → local TOT dir; per-station segment-exclusion catalog (segment_exclusions.csv) + declared non-quantum junction offsets (junction_offsets.csv, e.g. GUSK U +1.093 m for a stale station.info height), both deployed via gps_parser outlier_catalogs; --exclusions / --junction-offsets dev overrides; reason mandatory (typed, mypy-strict)
│   ├── detrend_estimate.py # gps-estimate-detrend CLI: batch estimate stored-detrend params from local TOT (ref=plate) → detrend_params.json (schema v1); per-station fit catalog fit_windows.csv (window + gate overrides, e.g. DYNG max_gap 1.0) via gps_parser resolver; unstamped = byte-reproducible, --stamp for timestamps; --uncert is the read-time sigma screen (default 15 = getData's own, workbench uses 10) and rides into refs, because it changes WHICH epochs were fitted without touching any fitted quantity; station_estimate_from_arrays returns record + the inlier mask lifted to the caller's index space (station_record_from_arrays wraps it) — the workbench's grey overlay, which no re-run detector could match (typed, mypy-strict)
│   ├── velocity.py       # THE velocity product (BGÓ 2026-10-16): estimate_station_velocity → VelocityRecord (window rate, ref plate-periodic|background|plate, σ = formal × empirical k(T): the station's own on its quiet span (residual by ref: plate keeps the seasonal), else the stored k(T) table (gps-velocities k-table → velocity_dir/k_table_<frame>.json, median of 113 stations), else 5 — never a run median, so a record depends on its station alone; k is in the fingerprint; gates n_obs ≥ 5 + cover ≥ 0.8, provenance + fingerprint); write_velocity_set → .gmt/.csv/.jsonl (one frame, byte-stable); gps-velocities estimate|k-table|show (typed)
│   ├── events.py         # events.yaml (Reykjanes volcano-tectonic catalog, BGÓ 2026-10-17) → Event(id, kind, onset, end, region); station_events by station_areas.yaml region + kind (dike,eruption default); a region-less or unknown-region event concerns no station; corrupt catalog refused (typed)
│   ├── autcln_stats.py   # icing index: gps-autcln-harvest reads okada WEEKS autcln.post.sum (ro) → <SITE>.autcln.24hr.csv + network.autcln.24hr.csv in pos_dir (indicators, incremental); icing_days = azimuth mean / network median vs own rolling q25, start 2.0×, stay 1.5×, bridge ≤5 d; gps_views.resolve_icing_epochs → cleaned view (taken out before detection, then flagged)
│   ├── snow_burial.py    # snow-burial index (2026-10-20): antenna UNDER snow ≠ icing — Up bump above its lower envelope (±60 d), horizontals quiet, A/B (atelv) raised vs own q25; episode kept if A/B stays raised, no horizontal jump at its end (eruption sawtooth), bump above the flank chord + peak above the after-level; live end = provisional; resolve_snow_epochs reads the FULL .pos (window-independent); view flags Up only; estimator drops the epochs (refs n_snow_dropped only when >0; --no-snow)
│   ├── network_days.py   # network days (2026-10-21): gps-network-days scans every station's 24 h .pos (itrf14; icing/snow days out) with the running detector → gps_analysis.network → network_days.<frame>.csv in pos_dir (regenerated whole: date,status decided|partial,share,counts per component); gps_views.resolve_network_epochs → cleaned view / writer / estimator take the decided days out before detection and remove them in every component (network_mode); second pass = the daily NETWORK COMMON MODE (median midpoint residual vs the one-sided running lines, stations in the Iceland box outside volcanic_areas, ≥15) → network_cm.<frame>.csv; gps_views.resolve_common_mode/subtract_common_mode (applied in gpsplot by default; writer/estimator/velocity NOT yet — their k(T)/ρ/records were measured unfiltered)
│   ├── rtk_read.py       # RTK baselines (RTKLIB ROVER-BASE day files on rtk.vedur.is, mounted /mnt_data/rtgpsdata, $GPS_RTK_DIR): ENU columns (not rtk_gps's sdn/sde labels), UTC, day files overlap midnight (dedup), Q=5 + sigma screen, interval medians; complete days cached under $GPS_RTK_CACHE / XDG cache; never writes under the data dir (typed)
│   ├── gas_read.py       # GAS (strainmeter) data
│   ├── sil_read.py       # SIL seismic data
│   └── hytro_read.py     # hydrology data
├── tests/test-gps_read.py      # legacy smoke script (not pytest-collected)
├── tests/test_gps_views.py     # view-toggle suite (raw parity, degrade, borrowing, provenance)
├── tests/test_cleaned_neu.py   # cleaned .NEU writer: byte-format identity, union drop, sidecar, steps.csv→step_epochs, protect_windows.csv (abort→clean), outlier_overrides.csv (per-station levers, precedence), .DEGRADED naming, --clean CLI
├── tests/test_globk_join.py    # segment read/join acceptance gates (SENG wrap, REYK dedup, HOFN) + TOT-writer format contract & openGlobkTimes round-trip
├── tests/test_globk_tot.py     # gps-globk-tot CLI + exclusion rules (seed mirrors gps-config-data/analysis-lane/segment_exclusions.csv; deployed-catalog resolution)
├── tests/test_detrend_estimate.py # fit-catalog reader/resolver, estimate→record→read_detrend_params→apply round-trip (flat detrended), gap-gate unlock (DYNG), CLI + determinism
├── tests/goldenmaster/         # behavior pins for the Phase 1 refactor 📄 README
└── pyproject.toml
```

## View toggle (internal delivery, DESIGN_live_detrending §0)

`gps_views.read_gps_view(sta, view="raw"|"cleaned"|"detrended", ...)` is the
first-class raw↔cleaned↔detrended switch of the internal (direct-read) path:
raw columns are ALWAYS present and bit-identical to the legacy read; views
only ADD columns (`*_outlier`/`*_cleaned` masks from
`gps_analysis.detect_outliers`; `*_detrended` = raw − stored trajectory via
`gps_analysis.apply_detrend` — pure apply of the deployed
`detrend_params.json` record, NO re-fit on read). Provenance in
`df.attrs["gps_view"]` (`detrend_method`, frame, record version, `fitted_at`,
borrowed/UseSTA source, degrade state). Graceful degrade: any clean/detrend
failure warns (`UserWarning` + log) and serves raw — only a frame mismatch
hard-fails. Array paths revived: `getData(..., ref="detrend")` and
`gamittoNEU(..., ref="detrend")` (so `gps-savetimes --ref detrend` now
produces stored-params detrended .NEU, plate-first). The legacy
`getDetrFit`/`convconst`/`save_detrend_const` + `detrend_itrf2008.csv`
mechanism is SUPERSEDED (kept as shims for `read_gps_data` until design §8
step 5). Lint/type scope: ruff excludes + mypy per-module ignores cover the
legacy aux readers only — new code must pass `mypy --strict` (pyproject).

**Provisional epochs (2026-07-27).** `detect_view_outliers` provenance also
carries `provisional` (mask shaped like `flags`) + `n_provisional`: recent
candidates the detector could NOT rule on because the step evidence is
indeterminate (`D` NaN — no usable post-flank), so a blunder and the onset of
real deformation are indistinguishable until data follows. Disjoint from
`flags` and purely DIAGNOSTIC — the series is unchanged, a caller ignoring the
key behaves exactly as before, and a reduced detection object degrades to an
empty mask rather than raising (§0.4). The `provisional_days` bound (default
`PROVISIONAL_DAYS = 14`) is load-bearing: indeterminate clusters also occur at
mid-series gaps wider than `step_flank_max_reach_days` and would otherwise
dominate the mask (RHOF: 3 clusters, at 323 / 3400 / 3400 days from the end).
These verdicts are UNSTABLE by nature — they resolve as epochs arrive.
`gps_plot` renders them gold; the epoch stays IN the series.

## Dependencies

- **In** (declared in `pyproject.toml`):
  - `geofunc` (git+https) — coordinate transforms
  - `gps_parser` (git+https; **local editable via `[tool.uv.sources]`, 0.4.x**)
    — station configuration + the shared `outlier_catalogs` resolver (steps /
    protect_windows / outlier_overrides — the single source gps_api also reads).
    **Migrated 0.3.0 → 0.4.x 2026-07-14**: `detrendFile` now resolves
    config-dir-relative (was CWD-relative). Golden `read_gps_data` cases select
    CSV presence via `GPS_CONFIG_PATH` (config dir with/without the CSV), not
    chdir; pinned VALUES unchanged. **Deploy note:** CI/prod without the local
    source override pull git HEAD — bump the git dep to a 0.4.x commit carrying
    `outlier_catalogs` before that path is exercised (mirrors gps_api).
  - `gtimes`, `numpy`, `pandas`, `pyproj`, `scipy` — external (gtimes/numpy/
    pandas added 2026-07-08; the package could not import from a clean install
    without them)
- **Out** (used by): `gps_plot` (loads time series for plotting); research scripts.

## Console Scripts

```bash
geo-dataread          # entry: geo_dataread:main
gps-savetimes ...     # entry: geo_dataread.gps_savetimes:main
gps-displacemnts ...  # entry: geo_dataread.gps_displ:main   (sic — typo preserved verbatim from pyproject.toml)
gps-globk-tot ...     # entry: geo_dataread.globk_tot:main — batch GLOBK pre/rap segments → local mb_STA_TOT.dat{1,2,3} (deployed segment_exclusions.csv + junction_offsets.csv; --exclusions / --junction-offsets dev overrides)
gps-realign --start D --end D --frame igb14|itrf2008 [--to itrf2020] --out DIR [-j N] # re-align the loose SINEX archive → per-station .pos
gps-baseline-noise [--dir TOT] [--window-years 2] # refit the empirical baseline-σ record (cron-able, ~4 s)
gps-velocities estimate --region W/E/S/N --window A:B [--ref plate-periodic|background|plate] # the velocity product → velocity_dir (default ~/gps-data/velocities)
gps-network-days [-j N] [--dry-run] [--no-common-mode] # whole-network bad solution days → <pos_dir>/network_days.itrf14.csv + network_cm.itrf14.csv (daily common mode) (run after the .pos update, before plots; ~5 min at -j 12)
gps-estimate-detrend ... # entry: geo_dataread.detrend_estimate:main — batch detrend-parameter estimation over local TOT → detrend_params.json (fit_windows.csv per-station windows/gates)
                         #   --analysis-yaml: per-station STAGE PLANS (detrend.estimation.stage_plans), what gps-detrend-workbench --commit writes
                         #   --donor-params:  document a 'donor:' hold borrows from (default: the DEPLOYED one, never this run's --out)
```

## Cross-References

- `../CLAUDE.md` — ecosystem overview + dependency graph
- `../geofunc/CLAUDE.md` — coordinate-math primitives this package consumes
- `../gps_parser/CLAUDE.md` — station-config provider
- `docs/velocity-product.md` — the velocity product: decisions, gates, k(T) table, schema, phase 2 questions
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-10-21 (network_days: whole-network bad solution days, in the cleaned view, the writer and the estimator). Previous 2026-10-20 (snow_burial: the snow-burial index beside icing, in the cleaned view, the writer and the estimator). Previous 2026-10-19 (events: the event catalog reader; cleaned view dispatches the running-window detector; autcln_stats: the GAMIT icing index). Previous 2026-10-16 (velocity: the velocity product, phase 1 windowed). Previous 2026-10-14 (rtk_read: RTK baselines for gpsplot rtk; gps_views.window_datum_shift, for gpsplot ts --model). Previous 2026-10-13 (pos: products = itr08/igb14/final_igb20; itrf2020 now names gamit-p01's IGb20). Previous 2026-10-12 (stabilize/org: glorg stabilization on the loose SINEX; station_catalog: plate velocity via geofunc.plates). Previous 2026-10-10 (pos.py: .pos reader + tType="POS"; write_pos 2026-10-12). Previous 2026-10-09 (baseline_noise + sinex modules; baseline() opt-in rho=/sigma_scale=). Previous 2026-08-26 (re-anchoring now covers BOTH hold kinds:
`donor:` (a finished record in detrend_params.json) carried the donor's datum
exactly as `store:` did — measured, THOB holding SENG via donor: sat
+75.3/-34.3/-124.6 mm off its own data and ignored --anchor-window entirely.
One pass handles both; provenance still spells which store was read
(`donor:STA@… anchored […]` vs `store:…`). A legacy record with no
param_names falls back to the ORDER (GROUP_ORDER puts the polynomial first,
`velocity._RATE_INDEX = 1`), scoped to `secular` — never to taking the level
verbatim. Consequence: the donor rate propagates on a re-estimate (pointer
semantics kept) but the donor OFFSET no longer crosses; two batch tests
asserting the old behaviour were updated. Also: secular_store: `SecularEntry.kind`
(`station`/`derived`) + `frame` — a background that belongs to no station
(a cluster mean, a regional-model value) must not read as a measured one,
and must say which plate model it is referenced to. `secular_lookup` now
REFUSES a borrow that crosses plate frames: the per-station assignment is
mixed around Svartsengi and EURA−NOAM = N +2.26, E −15.97 mm/yr (measured on
SENG's series through both models) — intrusion-sized, so a silent cross would
read as signal. Default `kind` is unserialised, so the 16 deployed station
entries round-trip byte-identically. Previous entry (stage_plan: cross-station `store:STA` borrows are
RE-ANCHORED at resolution time — the donor's `offset` is the donor's datum
(measured: SENG holding SKSH's s(t) was off (−2.9, −30.1, +41.5) mm N/E/U,
pure level error). `resolve_stage_plan` replaces the zeroth poly coefficient
BY NAME with a 1/σ²-weighted local anchor over `anchor_window` (default: full
fit span; donor seasonal subtracted only when borrowed too); provenance
`store:STA@… anchored [a,b]`; needs the borrower's series (threaded from
`_restage`) — refused without it; `store:self` byte-identical. Apply-only
plans (`apply:` + holds) now legal end to end — a fully-borrowed record gets
`borrowed={from,terms:all,…}`; `check_stage_plan_sources` = data-free
pre-flight. Fixed: `_spell` wrote StoreRef as `donor:…` (silent hold-kind
flip); `SecularEntry.fit` documented inert. Batch main still wires no
lookup_secular: store: holds + `--anchor-window` are workbench-only.
Previous 2026-08-16: batch reads its stored stage plans. Older: `git log`.)*
