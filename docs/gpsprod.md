# gpsprod — station motion products

One place, one estimator for every number about station motion
(`../PLAN-gpsprod.md`, destination `../.interrogate-gpsprod.md`, review
`../PLAN-gpsprod-review.md`). Library: `geo_dataread.products`; maths:
`gps_analysis.displacement`. Built in steps; this page grows with them.

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
- **`step`**: lines through `--step-window` (30) days before the onset and
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
