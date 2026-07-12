---
slug: geo_dataread-refactor-B
created: 2026-07-11
mode: coding
repo: geo_dataread
builds_on: A baseline (DATA_TO_PLOT_FINDINGS.md), master plan §10.1
execution: autonomous Fable, golden-master tripwire, .NEU sign-off gates
---

# Scope: B — geo_dataread → pure I/O refactor (§10.1)

## Goal
Turn `gps_read.py` (1997 LOC, 46 funcs) into **thin I/O**: readers do file→array, the
**math moves to `gps_analysis`** (which now exists, with `fitting`/`models`/`baseline`),
dead code is deleted, and **every move is gated by the golden masters** A built (synthetic
+ real-data). Public reader names stay (deprecation shims) so consumers don't break. R2
(plate/unit drift) is already de-risked by A — this is now mechanical + safety-netted, not
high-risk.

## In scope
- **Golden-master gate (hard):** run `tests/goldenmaster/` before AND after every move;
  a move that shifts a golden = a behavior change → stop.
- **Move math → `gps_analysis`:** `detrend`/`fittimes`/`getDetrFit`/`errfunc` (fit machinery),
  and the filters `iprep`/`vshift`/`dPeriod` (outlier/reference prep). Readers keep only I/O.
- **Thin readers + shims:** `getData`, `gamittoNEU`/`gamittooneuf`, `openGlobkTimes`,
  `toDateTime`, `convGlobktopandas` become thin; freeze their public signatures + keep
  deprecation shims (blast radius: `gps_plot.getData`, internal `gps_savetimes`/`gps_displ`).
- **Delete dead code:** `compGlobkTimes`, `TieTimes`, `savedisp`, `gpsvelo*`, the dead
  `leastsq` block, `useFIT="periodic"`, the `08h` branch. (Golden masters crash-pin the dead ones.)
- **Hygiene:** no `basicConfig`/logging side-effects inside functions; no in-place input mutation.

## Out of scope
- Any change to **`.NEU` output** (`gamittoNEU`/`gamittooneuf`) → **B sub-gate: BGÓ sign-off**
  (it feeds cdn.vedur.is; golden masters pin it — a moved `.NEU` golden needs explicit approval).
- The **`gps-config-data` source config fix** (`coordFile` case-collision / chiara path, §10.4) —
  separate small sign-off item; touches `.NEU` plate velocities.
- `read_gps_data` deep refactor — research-only, no live consumer → **freeze + shim, don't invest.**
- `gps_analysis` leaf purity: the moved math must stay pure (no geo_dataread back-import).

## Key decisions (need BGÓ)
| # | Decision | My recommendation |
|---|---|---|
| D1 | Preprocessing divergence: `vshift`(uncert 1.1) vs `iprep`(uncert 15) + offset handling — unify or keep two profiles? | **Extract both to `gps_analysis` as named, tested functions; keep the plot-profile vs .NEU-profile explicit (config-selected), don't silently unify** — they serve different consumers; pin both with goldens |
| D2 | Slice order | **`getData` (plot) first** (A-verified, low `.NEU` risk) → then `gamittooneuf` (`.NEU`, sign-off) |
| D3 | `DetrendStore` (replace `getDetrFit`/`convconst`/`save_detrend_const`) | **Defer** — detrend persistence is research-only now; note it, build when the detrend path is productized |
| D4 | JOIN / 08h | **Revive JOIN** (BGÓ uses it) as its own slice; **delete 08h** (truly dead) |

## Execution
Autonomous **Fable**, one reader/slice at a time, golden-master tripwire around each move;
`.NEU`-touching slices pause for BGÓ sign-off. This is the big autonomous job the A baseline
was built to make safe.

## Tracer-bullet slices (each independently mergeable, goldens green)
1. **Move the fit/detrend math** (`detrend`/`fittimes`/`errfunc`/`getDetrFit`) → `gps_analysis`;
   `geo_dataread` calls it via shim. *(research path; no `.NEU` risk)*
2. **Move the filters** (`iprep`/`vshift`/`dPeriod`) → `gps_analysis` per D1; `getData` uses them.
3. **Thin `getData`** (plot path) to pure I/O + the moved math. *(A-verified; gps_plot is the consumer)*
4. **Delete dead code** (D4's 08h + `compGlobkTimes`/`TieTimes`/`savedisp`/`gpsvelo`/`leastsq`).
5. **Thin `gamittoNEU`/`gamittooneuf`** (`.NEU` path). *(⚠️ sign-off gate)*
6. **Revive JOIN** (D4) as a clean, tested mode.
7. **Model evaluators → leaf shims** *(landed on `refactor-B-pure-io`)*:
   `line`/`periodic`/`xf`/`expxf`/`expf`/`dexpf`/`secondorder` delegate to
   `gps_analysis.models` (`linear`/`periodic`/`exp_linear`/`exp_linear_rate`/`poly2`) —
   bitwise-identical delegation, pinned by `tests/test_model_shims.py`.
   **`lineperiodic` deliberately stays local**: the leaf's `linear + periodic`
   association differs from the legacy single expression by ≤1 ulp (measured
   300/300 trials), and it feeds `curve_fit` on golden-pinned paths — consolidate
   only at a deliberate golden re-baseline (or if the leaf grows a
   legacy-association evaluator).

## Remaining (next slices)
- `read_gps_data` derived columns (`hlength`/`hangle`/`Dhlength`): leaf has
  `velocity.horizontal_magnitude/azimuth`, but the azimuth CONVENTION differs
  (leaf = geodetic CW-from-north; legacy `hangle` = math CCW-from-east) — not a
  drop-in; research-only path, frozen per "freeze + shim, don't invest".
- `pvel`/`printvelocity`: trivial rate/σ extraction + GMT-style printing —
  presentation, no leaf target; leave.
- D3 `DetrendStore` (replace `getDetrFit`/`convconst`/`save_detrend_const`) —
  deferred by decision until the detrend path is productized.
- Hygiene: `logging.basicConfig` inside `getDetrFit`/`read_gps_data`; dead
  `fromord` (`Timeto` undefined) + bare `except` in `__converter` (pre-existing
  ruff findings) — safe, behavior-invisible cleanup slice.
- `gps_displ.py`: `xyzDict`/`llh` still import the long-gone `cparser` module
  (broken since the gps_parser rename — `gps-displacemnts` CLI cannot run in a
  clean env); `fitDisp` is unused polyfit research code. Needs its own
  characterize-then-fix slice.

## Feeds from A
Golden masters (synthetic + real SENG/HOFN/REYK) = the safety net; R2 de-risked (plate/unit
identical); config drift + `.NEU` boundary documented; `skiprows=3` correct; stale-venv fixed.
