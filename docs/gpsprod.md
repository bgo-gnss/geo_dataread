# gpsprod — station motion products

One place, one estimator for every number about station motion
(`../PLAN-gpsprod.md`, destination `../.interrogate-gpsprod.md`, review
`../PLAN-gpsprod-review.md`). Library: `geo_dataread.products`; maths:
`gps_analysis.displacement`. Built in steps; this page grows with them.

## `--common-mode` — the network's shared daily motion (2026-10-29, BGÓ)

On some days the whole solution moves a few mm at every station together
(that day's orbits and frame realisation), which is not ground motion.
`gps-network-days` measures it from quiet stations outside the volcanic
areas (`network_cm.itrf14.csv` in `pos_dir`); gpsplot subtracts it, and
`loader.load` now can (`Source.common_mode`, `VelocitySettings.common_mode`).

| command | default |
|---|---|
| `displacement` (ends, points, step) | subtract |
| `displacement --method rate`, `velocity` | off (= the velocity product, whose k(T) table was measured without it) |
| `transient` | subtract |
| `series` | off |

Flag `common_mode` when subtracted, `common_mode:none` where the series has
none (itrf20, 8 h, no file). Svartsengi 90-day ends: every station shifted
by about N +1.4, E −1.5, U −2.3 mm, East σ 8 → 1.4 mm.

## Step 1 — `gpsprod displacement` (method `ends`)

```bash
gpsprod displacement SENG THOB --start 20231112           # screen table
gpsprod displacement --area reykjanes --period 20240901:20241031 -o d.csv
gpsprod displacement SENG --relative-to SKSH --days 30
gpsprod displacement all --days 7 --ends 1                # one day per end
```

- **Level at each end** = inverse-variance mean of the `--ends N` whole days
  (default 5) inside the period: `[P0, P0+N d)` and `[P1−N d, P1)`.
  The end window closes at the station's last epoch (a period to today is
  not shortened by the processing lag; `end_early:Xd` past 1.5 days).
- **σ of a level** = max(k_day·σ_formal, σ_formal·√(χ²/ν), s_day/√n) — never a
  product. k_day and s_day are measured on the station's own series around
  the period (±60 d), at the session's step (24 h or 8 h), on the
  differenced series with `--relative-to`. Real motion inside an end window
  (days after a dike) inflates σ through χ²/ν — by design.
- **N ≤ 2**: each end is tested against the line through the next
  `--neighbour-days` (5) days; > 4σ → `end_disputed:<start|end>:<N|E|U>`,
  too few → `end_unchecked:…`. The number is kept; the flag says doubt.
- **Steps**: a declared step inside an end window refuses the station
  (`step_in_end_window@date` — e.g. `--since 20231110` puts the dike day in
  the first window); one inside the period is flagged `steps_in_period`.
- **Defaults**: `--frame itrf14` (`imo.24hr_itrf14` .pos), `--ref
  plate-periodic` (plate and the stored seasonal removed; `--keep-seasonal`
  = `--ref plate`, `--ref background|itrf`), `--view declared`
  (`cleaned` drops real motion as outliers in unrest windows — BGÓ
  2026-10-26), `--session 24h`. A station without a detrend record falls
  back to plate (`fallback:plate`).
- **Output**: screen, or `-o FILE.csv|.jsonl` (`--units mm|m`). CSV has a
  `#` provenance header and one row per station, refused ones with the reason;
  byte-stable. Every record carries frame, series, session, ref, view,
  method, period, σ method, input fingerprint and software versions.
- `gps-displacemnts` (TOT, ITRF2008, end day − start day; its CLI was broken)
  now translates to `gpsprod displacement` and says so.

## Step 2 — `step`, `rate`, conversions, event names

```bash
gpsprod displacement --area reykjanes --method step --event 2024-03-02-dike
gpsprod displacement THOB --method step --at 20231110 --step-window 20
gpsprod displacement THOB --method rate --period 20250101:20250601
gpsprod displacement THOB --period 2023-11-10-dike:2023-12-eruption --as-velocity
gpsprod displacement --area reykjanes --period after:2023-11-10-dike -o d.csv
```

- **Event names in periods** (`events.yaml` id, or a part naming one event):
  as a start = the day after the event's last day; as an end = the day before
  its onset. `DIKE:ERUPTION` = the quiet between; `after:EVENT` or `EVENT:` =
  from it to today; `EVENT` alone = its own days. Names go into the records'
  `period_names`.
- **`step`**: lines through `--step-window` (10, BGÓ 2026-10-26: 20–30 d over-extrapolated THOB E by 4–5 cm after the dike) days before the onset and
  after the last day (event days in neither), each evaluated at the event's
  edge: Δ = f_after(last day + 1) − f_before(onset); a multi-day eruption's
  displacement includes the motion during it. Each side passes the velocity
  product's gates (≥ 5 epochs, ≥ 80 % cover, no declared step) or the station
  is refused (`step: after cover=0.43<0.8`). σ: each line's level σ ×
  max(k_day, √χ²/ν), its rate σ × k(T) — the station's own k(T), else the
  k(T) table, else 5 (`--relative-to` and 8 h: table or 5, flagged
  `k:baseline_unmeasured` / `k:session_unmeasured`). A curved side (F of a
  quadratic > the product's threshold) is flagged `step_nonlinear:<side>:<c>`
  — post-dike relaxation. The event: `--event`, `--at DATE` (one day), or
  the station's one dike/eruption in the period (none or several: refused).
  With `--event`/`--at` and no period, the period is the event ± the window.
- **`rate`**: the velocity product's record over the period × its length
  (velocity → displacement; σ the product's σ × T; its flags kept). Valid
  for steady motion only: an event inside the period inflates σ (χ²). Not
  yet for `--relative-to` or 8 h (step 3), nor `--ref itrf`.
- **Conversions**: `products.to_displacement(VelocityRecord, period)` (d = v·T)
  and `products.to_velocity(DisplacementRecord)` (v = d/T, T between the
  end-window centres for `ends`, the period otherwise) — exact inverses;
  `--as-velocity` prints/writes the velocity product's table/files.

### `points` — end point minus start point (2026-10-26, BGÓ)

```bash
gpsprod displacement SENG THOB SKSH --method points --period 20231109:20231112
```

The last epoch of the period minus the first (all three components
present), no averaging. In fast motion a mean over days moves the
effective date: SENG East over the dike is −815 mm with single days
(9 → 12 Nov) and −929 mm with 5-day ends over 1–20 Nov. Missing end days:
the nearest whole epochs inside the period, named in `start_at`/`end_at`.
`--ends 1` gives the same number when both end days exist, and refuses
otherwise.

The end check (`--ends ≤ 2` and `points`) now fits its line through the
days OUTSIDE the period (before the start, after the end), the inside only
when outside is too thin, never across a declared step — a line through the
inside days crossed the very motion being measured. A flag that remains is
real: SENG on 9 Nov 2023 sits 18 mm East of the line through 4–8 Nov (the
motion began the day before the dike); for one day the check cannot tell a
blunder from the onset of motion, so it flags and keeps the number.

### Periods by options (2026-10-26, BGÓ)

`--start A --end B` (both days in), `--start A --days N` (N days from A),
`--days N [--end B]` (the N days ending B, default today), or `--period`
alone (dates and event names). `--since`/`--last` remain as the old names
of `--start`/`--days`.

### The σ screen (`-u`, 2026-10-26, BGÓ)

`gpsprod displacement` drops solutions with a formal σ above 100 mm by
default (10 for `--method rate`, the velocity product's screen). The day of
an event is a noisy solution (SENG 10 Nov 2023: σ 9/15/36 mm N/E/U) but it
is exactly the day asked for; its σ is carried into the displacement. The
10 mm screen made `--period 20231110:20231111 --ends 1` refuse SENG and THOB.
Broken solutions (SENG Jan 2024, σ ≈ 3 m) stay out.

## Step 3 — the velocity product moves in (`gpsprod velocity`)

```bash
gpsprod velocity --area reykjanes --start 20200101 --end 20221231
gpsprod velocity SENG THOB --period 2023-11-10-dike:2023-12-eruption
gpsprod velocity all --days 365 --save                   # into velocity_dir
gpsprod velocity SENG --relative-to SKSH --start 20240901 --end 20241031
gpsprod k-table …  /  gpsprod show set.jsonl              # = gps-velocities k-table / show
```

- Code: `geo_dataread/products/velocity_core.py` (estimator, k(T); moved with
  `git mv`), `velocity_files.py` (.gmt/.csv/.jsonl), `velocity_cli.py`
  (`gps-velocities`), function bodies unchanged. `geo_dataread.velocity` stays
  as the same module (a `sys.modules` alias of the core), so every import and
  test monkeypatch keeps working; the moved names load on first use.
- **Byte identity, proved twice**: `tests/test_velocity_golden.py` (5 synthetic
  runs × 3 files, frozen before the move) and a real-data run (6 sets, 10
  stations: long, short, short tier, background, plate, pre-onset) are
  byte-identical before and after; `gpsprod velocity` writes the same bytes as
  `gps-velocities estimate` for the same settings.
- **New in the product**: `session` (8 h series) and `relative_to` (a
  baseline). Both read through gpsprod's `loader.load`; at their defaults they
  are left out of the fingerprint and the JSON, so earlier sets are unchanged.
  Their k(T) is not measured yet: a baseline takes the table (single-station,
  so conservative), an 8 h series `--default-k`, flagged
  `k:baseline_unmeasured` / `k:session_unmeasured`. `displacement --method
  rate` now works for both.
- `gpsprod velocity` uses gpsprod's stations (`--area`, `all`), periods
  (`--start/--end/--days`, `--period` with event names) and output (screen by
  default, `-o DIR/NAME`, `--save` = velocity_dir as gps-velocities); σ screen
  10 mm, view declared — the product's own defaults.

## Step 4 — `gpsprod coords` (coordinates at an epoch)

```bash
gpsprod coords SENG THOB --epoch 20240101
gpsprod coords --area reykjanes --epoch 20231109 -o coords.csv
gpsprod coords all --epoch 20250601 --window-days 0          # single days
gpsprod coords THOB --relative-to SKSH --epoch 20240601      # baseline vector
gpsprod coords SENG --epoch 20240115 --method model          # in SENG's gap
gpsprod coords HOFN --epoch 20240101 --method apr            # GLOBK apriori
```

- **observed** (default): the GLS mean (full 3×3 covariance,
  `gps_analysis.gls_mean`) of the `.pos` X/Y/Z within `--window-days` (5)
  either side of the epoch (a day's solution = its noon), after the reviewed
  junction offsets and the view (`cleaned` by default, BGÓ 2026-10-26), σ
  screen 100 mm. σ = max(k_day, √χ²/ν) × the mean's formal σ.
  `--window-days 0` = that day's solution (HOFN 2024-01-01 reproduces the
  `.pos` row exactly).
- **Events cut the window.** A declared step (steps.csv) or the start/end of
  a dike/eruption of the station's area (events.yaml) inside the window keeps
  only the solutions on the epoch's side (`window_cut_at_step:D`): THOB on
  2024-06-01 without the cut had χ²/ν 981 and σU 76 mm (the 29 May eruption
  inside the window); with it, 9 solutions and σU 10 mm. A window with large
  scatter left (χ²/ν > 25) is flagged `fast_motion` — use `--window-days 0/1`.
- **No solution near the epoch** (a gap, before the station, the future):
  extrapolated from the nearest window by the detrend record + the plate
  motion (horizontals, as the series' plate removal) — `extrapolated:Nd`,
  `steps_between` when an event lies in between (the model does not know it),
  `model:plate_only` without a record, `sigma:observed_only`. `--method model`
  always does this (a coordinate without the daily noise).
- **apr**: the GAMIT tables' `igb14_comb.apr/.eq` (itrf14) or igb20 (itrf20);
  stations not in them are refused with the reason. No σ.
- **--relative-to**: the baseline vector X/Y/Z(A) − X/Y/Z(B), σ in quadrature,
  its N/E/U at A (`dN dE dU`).
- Output in metres: X Y Z, sX sY sZ, lat lon h, sN sE sU (+ dN dE dU), n_used,
  t_obs; screen, `-o .csv/.jsonl`.

## Step 5 — `gpsprod series` (time series to file)

```bash
gpsprod series OLAC                                     # csv to the screen
gpsprod series OLAC --start 20240101 -o olac.csv
gpsprod series --area reykjanes --period after:2023-11-10-dike -o out/ --format neu
gpsprod series THOB --with-dropped --days 90 -o thob_all.csv
gpsprod series SENG --source tot --ref plate -o tot/     # = gps-savetimes --file
```

- The series every product reads (`loader.load`): `.pos` of the frame, ref
  `plate-periodic`, view `cleaned` (BGÓ: cleaned for series/coords), σ screen
  10 mm, the whole series unless a period is given. Kept epochs only;
  `--with-dropped` writes every epoch with its raw value, `kept` 0/1 and the
  reason (outlier, declared, icing, snow, network — the loader now keeps the
  raw values and per-epoch reason bits, `StationView.y_raw`/`reasons`).
- Formats: CSV (`#` provenance, date, yearf, dN dE dU, sN sE sU, kept, flags),
  JSONL (provenance line + one epoch per line), `.NEU` in gps-savetimes'
  layout (`gtoNEU` + `gamittoFile`, its legacy first line included) with a
  `.prov.json`. One station → `-o FILE`; several → `-o DIR/` (`STA-<ref>.<fmt>`).
- `--source tot`: the legacy ITRF2008 TOT series through `gamittooneuf` —
  byte-identical to `gps-savetimes` (checked: SENG plate and itrf2008, OLAC).
  **`gps-savetimes` itself is unchanged** (it feeds the production .NEU on the
  CDN; not aliased, unlike the plan).

## Step 6 — gpsplot uses the products (gps_api: pending BGÓ)

- `gpsplot map deformation` = `gpsprod displacement` (method ends, ref
  background by default, declared view): the level of 5 days at each end with
  the max-of-three σ, instead of gps_plot's own 1/σ-weighted 5-sample mean.
- `gpsplot map velocity` without a window = the stored records' secular rate,
  now `geo_dataread.products.secular_rate` (moved from gps_plot).
- gps_plot's `sources/velocity.py` (window_velocity, window_displacement,
  record_velocity) is deleted.
- One estimator, test-pinned (`gps_plot/tests/test_one_estimator.py`):
  `gpsprod velocity`, `gpsplot map velocity --window` and `gpsplot ts --rate`
  build identical velocity settings — frame by its full name, session from the
  product — so the same fingerprint and numbers, 24 h and 8 h. (Map/ts velocity
  sets written before this carry the short frame name in their fingerprint.)
- A station with only a regional seasonal record asked for `--ref background`
  now falls back to plate-periodic (`seasonal:regional;fallback:plate-periodic`)
  in every product, instead of being refused.

## Step 7 — more formats (`.gmt`, `.gbis.txt`, `.xlsx`)

- `gpsprod displacement -o d.gmt` — GMT psvelo rows (`lon lat dE dN sE sN corr
  station`, mm; refused stations left out) for `gmt velo -Se` / pygmt.
- `-o d.gbis.txt` — `lon lat dE dN dU sE sN sU station` in metres, `%`
  comments: the text GBIS's GPS .mat (`ll`, `displacements`, `covariance`) is
  made from (BGÓ to confirm the column order against his GBIS setup).
- `-o x.xlsx` — displacement, coords, velocity (`gpsprod velocity -o
  NAME.xlsx`: the velocity CSV columns exactly) and series (`--format xlsx`):
  a `data` sheet and a `meta` sheet. Written with the standard library
  (`products/xlsx.py`, no openpyxl), byte-stable; checked to open in
  LibreOffice.
- `.gmt`/`.gbis.txt` are for displacements only (a velocity set already
  writes its own `.gmt`).

## `gpsprod transient` — the velocity on a date from a fitted curve (2026-10-28, BGÓ)

After an eruption the Svartsengi stations move on a curve: fast at first, then
slowing to a steady rate. A straight line over a year misfits it badly; the
3 + 3 month windows track it once the curve is nearly flat. This product fits
the curve instead, after the method of `gps_data_analyses/detrend-bb` (line +
exponential, then its derivative):

    x(t) = offset + v·t + Φ·(1 − e^(−(t − t₀)/τ))      dx/dt = v + (Φ/τ)·e^(−(t − t₀)/τ)

- **One τ for the run** (one source): estimated jointly from all usable
  stations (`gps_analysis.profile_shared_tau`, a summed VARPRO profile), or
  `--half-life DAYS`, or `--decay FILE` from an earlier run. Written to
  `NAME.decay.json` (τ, σ_τ, its interval, t₀, the stations, the settings).
- **Each station with τ fixed** (`gps_analysis.transient_rate`); σ_τ is carried
  into the rate. Outliers (> 4 robust σ off the curve) are dropped from the
  series as read, at the final τ — so a station's numbers depend on its data
  and the decay only (test-pinned), the k(T)-table rule of the velocity
  product.
- **The record** is a `VelocityRecord` (`kind` "transient", `model`
  "exp_linear") with two new optional fields, omitted from JSON when unset (the
  goldens are unchanged): `at` — the date the rate is FOR — and `transient`
  (t₀, τ, half-life, per component `v`, `phi`, `A = −phi`, steps, rejections).
  `gpsplot map velocity --velocities` draws it, labelled "rate on DATE".
- **σ is formal × √(χ²/ν) × k** (`sigma_method` lag1_residual, flag
  `sigma:lag1`, k per component in `k`): k = √((1+ρ)/(1−ρ)), ρ the
  day-to-day correlation of the station's residuals from the curve
  (`gps_analysis.lag1_scale`); σ_τ added after. `sigma_formal` is the σ
  without k. A lower bound: Svartsengi 2025-08..2026-10 gives k 1.7–2.3, the
  velocity product's k(T) at that length 5–7 (flicker noise, which one lag
  does not see).
- **The network common mode is subtracted** by default (`--common-mode off`
  keeps it; flag `common_mode`, or `common_mode:none` where a series has
  none: only okada's itrf14/itrf08 24 h solutions do).
- t₀ is the period's first day: `--period after:EVENT` starts the day after an
  eruption's last day. `--at` defaults to the newest day with data (not the
  calendar: the processing lag would make every rate an extrapolation).

Svartsengi, 26 stations, t₀ 2025-08-06 → 2026-10-28: half-life 60.3 d
(τ 0.238 ± 0.002 yr), χ²/ν 1.76; the rate on 2026-10-25 matches the
experiment in `gps_data_analyses/svartsengi-explin` (NBIO Up 116.7 vs 117.4,
SENG 169.2 vs 169.4 mm/yr). `tau_differs` flags AUSV, FEFC, HRAG, KEIC, THOB
(own half-lives 30–80 d).

Open: a σ scale that sees the slower noise (the lag-1 k is a lower bound);
a ts overlay of the fitted curve.

```bash
gpsprod transient --area reykjanes --period after:sundhnukur-2025-07-eruption
gpsprod transient --region=-22.72/-22.17/63.76/63.945 --start 20250806 \
                  --at 20260101 --at 20261025 -o ~/tmp/svartsengi
gpsprod transient SENG --start 20250806 --decay ~/tmp/svartsengi.decay.json
```
