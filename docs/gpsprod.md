# gpsprod — station motion products

One place, one estimator for every number about station motion
(`../PLAN-gpsprod.md`, destination `../.interrogate-gpsprod.md`, review
`../PLAN-gpsprod-review.md`). Library: `geo_dataread.products`; maths:
`gps_analysis.displacement`. Built in steps; this page grows with them.

## Step 1 — `gpsprod displacement` (method `ends`)

```bash
gpsprod displacement SENG THOB --since 20231112           # screen table
gpsprod displacement --area reykjanes --period 20240901:20241031 -o d.csv
gpsprod displacement SENG --relative-to SKSH --last 30
gpsprod displacement all --last 7 --ends 1                # one day per end
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
