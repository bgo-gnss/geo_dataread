# The velocity product

Velocity estimation is its own process, and its output a data product
(BGÓ, 2026-10-16). The map's vectors, the rates on time-series plots and
the velocity tables fed to deformation models all read the **same numbers**:
`geo_dataread.velocity.estimate_station_velocity`, or the files it writes.

Code: `geo_dataread/src/geo_dataread/velocity.py` (orchestration, files,
CLI), `gps_analysis/src/gps_analysis/rate_quality.py` (leaf math). Consumers:
`gpsplot map velocity` (`--window`, `--velocities`, `--velocities-out`),
`gpsplot ts --rate` / `--rate-window`.

## Why (measured 2026-10-16, Fable review)

Before, three paths gave three answers for the same station and window:

| Path | Series | Model | σ |
|---|---|---|---|
| `gpsplot map --window` | itrf2014 `.pos`, plate removed, raw | line, WLS | formal |
| `gpsplot map` (secular) | the detrend record | lineperiodic, WLS | formal (NYLA ±0.008 mm/yr) |
| `gps_api` velocities | `.NEU` (ITRF2008 plate), stamped ITRF2014 | lineperiodic, last 2 yr | formal |

- Formal WLS σ is 4–17× too small on daily GNSS: the noise is correlated.
- Unrest windows were published with no refusal.
- `lineperiodic` on a short window is unidentifiable: 58 epochs gave
  −11220 ± 38770 mm/yr, with no warning.
- None of the paths carried the frame, the epoch count, the model or the
  E–N correlation.

## Decisions (BGÓ 2026-10-16)

1. **Plate removed, always.** Two reference modes, both available:
   - `plate-periodic`: the stored seasonal removed, then the rate;
   - `background`: the stored secular + seasonal removed, so the rate is the
     transient alone.

   `plate` (nothing more removed) remains for stations without a record.
2. **σ = formal × empirical k(T).** MLE comes later, once it handles gaps.
3. **A station without a detrend record** gets windowed velocities only,
   flagged `fallback:plate` / `background:none`. No MIDAS secular stand-in:
   MIDAS on unrest data is a false background.
4. **8-h series deferred.** When 8-h resolution matters, the unrest rate is
   far above the background, which can then be neglected.
5. **Legacy `.vel` readers** are migrated to this format, not emulated.

## Phase 1: windowed velocities (done)

**Estimator.**
- A line, per component, over a window of the plate-removed `.pos` series
  in ONE frame (itrf2014 by default).
- The ref's stored terms are removed first; the `declared` view by default
  (below).
- The model is always a line. A seasonal cannot be estimated inside a window
  shorter than ~2.5 yr: it is removed from the record or not at all.
- The window is whole days for dates: from the start date's midnight to the
  midnight after the end date, half-open (an epoch at that midnight is out).
  `--last N` is N whole days ending with `-e` (default today), in
  `gps-velocities` and the map alike.
- `gpsplot ts --rate` without `--rate-window` takes the same whole days, from
  the plot's `-s`/`-e` (else its first/last epoch): the same T, k(T) and σ
  as `map --window` over the same dates.

**Gates: two tiers** (BGÓ 2026-10-16: slow motion must be robust, fast
motion is well determined from few epochs, even two).

Refused outright, with the reason:
- a declared step strictly inside (`--allow-steps` turns that into a flag);
- a failed background removal, no `.pos`, fewer than 2 epochs;
- no record when one is required (`--no-record refuse`).

*Standard tier*: at least 5 epochs and 80 % of the window covered. Cover
counts each daily epoch's own day, so five daily epochs cover a five-day
window, and is measured against the station's last epoch when the data stop
inside the window (the processing lag; flag `truncated@<yr>`, `data_end`).
A recent window's record therefore changes once the processing catches up
(data end, gate, k and fingerprint): by design, but not reproducible until
then.

*Short tier* (`tier = short`): a window failing only the count/cover gates
keeps each part, horizontal pair or up, whose |v|/σ reaches 5 (8 on two
epochs); a part that does not stand is NaN, with its reason in
`parts_refused`; no part standing refuses the window. σ here is a-priori ×
√max(1, χ²/ν) × k: χ² on 2–4 epochs is too noisy to shrink σ, but a blunder
must still inflate it. `--no-short-tier`, `--short-snr`, `--short-snr-pair`.

Thresholds, measured 2026-10-16 (raw series, background removed, 3-yr quiet
spans of 20 stations, every 1–8 day window, ~20 000 per epoch count):

| | false accepts per 10⁴, \|v_h\|/σ_h ≥ 5 | ≥ 8 | ≥ 10 |
|---|---|---|---|
| 2 epochs | 23 | ~10 | 2 |
| 3–4 epochs | 1.6–2 | 0 | 0 |

Up never passed 5 on quiet data. The fast motion passes: the Svartsengi dike
~200 on two epochs (but it is a step, declared, so refused); post-dike
2023-11-12..14 SENG/SKSH/ELDC 6–13 on three epochs, while the two-epoch
11-12..13 window gives SENG 9.1, SKSH 10.0, THOB 8.9. The pair threshold is
8 (BGÓ 2026-10-17), which takes them at ~10 false per 10⁴ quiet two-epoch
windows; `--short-snr-pair 10` is the stricter choice (2 per 10⁴). The 2024
inflation reaches 5 only over ~10–14 days.

*Nonlinear flag* (`nonlinear:NEU`): the F of a quadratic against the line
≥ 50 in a component (`gps_analysis.NONLINEAR_F`), in windows of 61 days or
less (`NONLINEAR_MAX_DAYS`: calibrated only that far) and on 7 epochs or more
in the component (`NONLINEAR_MIN_OBS`: F(1, n−3) is heavy tailed on few
epochs, white noise reaches 50 in 9 % of 4-epoch, 2 % of 5-epoch windows);
an exact line gives no F (round-off). Quiet p99 is 23–42 at
7–61 d, 80 at 91 d; post-dike 14–30 d windows reach 30–310. A flag, not a
refusal: the rate of a decaying pulse is a mean, not a rate. χ²/ν was not
used: it depends on the component and on `-u` (quiet p99 N 10, E 16, U 2.5).

An earlier draft used a floor of 20, which refused every 7–14 day window.
The CLI defaults are the
`VelocitySettings` defaults (test-pinned): the map and `gps-velocities` give
one number.

A station with no `.pos` or no stations.cfg entry is refused (`read: …`,
`no_coordinates`), never a crash of the run. A `k` passed by the caller is
`sigma_method = given_k`.

**σ: k(T), the empirical rate scale.**
- On a quiet series with its background removed, slide windows of length T,
  fit a line in each, and take `k = max(1, 1.4826·MAD(rates) / median σ_formal)`.
- k is measured on the residual that matches the ref (`K_RESIDUAL`): `plate`
  keeps the seasonal, so a plate rate's σ includes the seasonal slope.
- Schema 2 (this branch) changed every `input_fingerprint` (the short-tier
  settings and `data_end` joined it); schema-1 sets read, as schema 2 with
  the standard-tier defaults, but do not reproduce byte for byte.
- The order:
  1. the station's own k, on its detrend record window (`empirical_k`);
  2. the stored k(T) table (`empirical_k_table`);
  3. `default_k = 5` (`default_k`).
- **Never a median of the run.** An earlier draft used one, which made a
  station's σ depend on which other stations were drawn. A record now
  depends on its station, the settings and the k table named in
  `k_source`; k, `sigma_method` and `k_source` are part of
  `input_fingerprint`.
- **The table is a file.** A host without it (rek, until it is deployed)
  falls back to `default_k = 5` for the stations without a record of their
  own, which `sigma_method` says. A table built on another view or uncert is
  used but flagged (`k_table_settings_differ`).

**The k(T) table**: `gps-velocities k-table` writes
`~/gps-data/velocities/k_table_itrf14.json`. It is the median over 113
stations with records, built in 2 min.

| T | k N/E/U, seasonal removed | k N/E/U, seasonal kept | stations |
|---|---|---|---|
| 7 d | 1.49 1.46 1.56 | 1.49 1.48 1.56 | 113 |
| 30 d | 2.01 2.10 2.05 | 2.08 2.12 2.14 | 113 |
| 61 d | 2.56 2.52 2.50 | 2.76 2.81 3.08 | 113 |
| 91 d | 2.95 2.63 2.66 | 3.45 3.35 4.05 | 113 |
| 182 d | 3.73 3.39 3.59 | 5.65 4.75 6.36 | 107 |
| 365 d | 5.38 4.55 6.09 | 5.98 5.66 7.21 | 87 |
| 730 d | 6.02 6.91 11.01 | 5.53 7.24 8.74 | 35 |

k grows with T, as correlated noise predicts. Where the seasonal is kept,
k is higher at 2–6 months: that excess is the seasonal aliasing.

**Record** (`VelocityRecord`, schema 3, one line of `.jsonl`; schemas 1
and 2 read, schema 1 with the standard-tier defaults). Every set written
since schema 3 is schema 3, whatever its ref: a reader still at
`READ_SCHEMAS = (1, 2)` (gps_plot `--velocities`, gps_api on an older
geo_dataread) refuses it, so deploy geo_dataread before sharing new sets:
- station, lon/lat, frame, plate, ref (used and requested), series, view;
- uncert, window, span used, n epochs per component, largest gap;
- method `wls`, model `linear`;
- rate, σ, formal σ, `sigma_method`, k, `k_source`, `corr_en` (0: the
  components are fitted on their own);
- a record summary, steps in the window, outlier abort, flags, refusal;
- `tier`, `snr` (horizontal, up), `chi2nu`, `f_quadratic`, `parts_refused`,
  `data_end`;
- `pre_onset` (ref `pre-onset` only: what was subtracted);
- software versions, `input_fingerprint`.

**Files** (`write_velocity_set`): sorted by station, fixed decimals, no
timestamps, so they are byte-stable (verified). One frame per set.
- `.gmt`: `lon lat ve vn sve svn corr_en station`, GMT `velo -Se`, accepted
  rows with a horizontal rate only;
- `.csv`: every row, with the reason a row was refused;
- `.jsonl`: the full record per line.

**Measured:**
- 48 Reykjanes stations, 2026-08-15 to 10-15: 4.4 s, 38 accepted.
- `gpsplot map velocity --velocities FILE` is pixel-identical to the map
  that wrote the file.

## The `pre-onset` reference (BGÓ 2026-10-17)

`--ref pre-onset --pre-window START:END`: the window's `plate-periodic` rate
less the station's own `plate-periodic` rate over a fixed reference window
before the unrest's onset. The rates are then relative to the preceding
state, in the frame of the activity; the plate (NOAM/EURA) drops out with
it. For the 2023 Svartsengi unrest: `--pre-window 20230806:20231024`, the
day after Litli-Hrútur ended (2023-08-05) to the day before the uplift
(onset 2023-10-27 in VÍ's timeline and in the GNSS, the daily solutions
show it from the 27th; the series used 25 Oct as a safe end).

- Two estimates of the ordinary estimator: the window (its gates, short
  tier, k) and the pre-window (standard tier only, no step inside). rate =
  v_win − v_pre; σ = √(σ_win² + σ_pre²) per component. The short tier's
  part decisions are the window's own (what its false-accept rates were
  measured on); `snr` is the relative rate's. A kept part whose relative
  `snr` falls under the short-tier threshold is flagged
  `short:pre_snr<5:horizontal` (or `…:up`): the window stood on its own,
  the difference does not.
- The pre-window must start before the window (else `ValueError`: a
  swapped `--window`/`--pre-window`).
- A refused pre-window refuses the record (`pre_onset: <reason>`): no data
  before the onset (the stations installed in November 2023), too little
  cover. The window's short-tier state (`tier`, `parts_refused`,
  `chi2nu`, `f_quadratic`, `short:`/`nonlinear` flags) is cleared. A neighbour's pre-rate is never borrowed (SENG and SKSH differ
  5.7 mm/yr east in their secular rate).
- `seasonal:none`: a station without a stored record has no seasonal
  removed, so its pre-window rate aliases the seasonal; its k(L) (seasonal
  kept, `K_RESIDUAL["plate"]`) says how much (~3.3–4 at 80–90 days). A
  regional seasonal (open item 2) is the fix.
- `pre_overlap`: the window starts before the pre-window ends; the two
  rates share epochs and σ is conservative.
- The record keeps what was subtracted in `pre_onset` (window, rate, σ,
  k, `k_source`, n, flags, fingerprint), schema 3.
- Measured on the 2023 reference (itrf20, 48 Reykjanes stations): 25
  have one; 2 are refused on cover, 21 have no data in it (installed later,
  or none). The reference is not quiet near Fagradalsfjall (FEFC U +135,
  GONH U +164, VOGC N +111 mm/yr in Aug–Oct 2023; VÍ: deep inflation there
  from August 2023): "relative to the preceding state", not to a
  background. BGÓ 2026-10-17: wanted, that signal has another source and
  must not leak into the Sundhnúkur signal.
- A caller's `k` is the window's; the pre-window takes its own.
- `gpsplot ts --rate` does not take this ref yet; `gpsplot map --velocities`
  takes the reference days from the set and refuses a `--pre-window`.

The events catalog (33 Reykjanes events since 2020, Veðurstofa news as
the source, joined to `steps.yaml` by `event_id`) is in `gps-config-data`
`analysis-lane/events.yaml` (47fe763, with the 2023-11-10 dike as a
declared step at 25 stations). Nothing reads it yet: the reference window
is given by hand.

## Phase 2: secular velocities (not started: decisions needed)

The detrend record already holds a curated secular rate: its window, steps
and `lineperiodic` fit. A fresh fit reproduces it to 0.01 mm/yr. Its σ is
formal. Open questions for BGÓ:

1. **σ over multi-year spans.** The table ends at 2 yr: k = 6–11 there,
   from 35 stations. A 10-yr record window extrapolates k, and the
   flicker-noise rate σ keeps changing with T. The options:
   - extend the table to 5–10 yr, where few stations have enough quiet span;
   - use the MIDAS σ, which is calibrated for long spans, as a floor;
   - wait for MLE with gap handling.

   Fable's advice was formal × k with MIDAS as the cross-check and refusal
   signal: refuse when MIDAS trims more than 20 % of pairs, flag when
   |WLS − MIDAS| > 2σ.
2. **The secular map today** still draws the record rate with formal σ (the
   ±0.008 ellipses). It should move to the product with honest σ.
3. **Borrowed secular records** stay refused, as `record_velocity` does now.
4. **The Reykjanes stations without records** need item 3 of the Reykjanes
   plan (a regional secular model) first.

## The view: `declared` by default (BGÓ 2026-10-16)

`declared` is the raw series less the operator-declared blunder epochs
(`excluded_epochs.csv`, matched as the cleaned view matches them), with no
detection. `cleaned` stays available; its limitation follows. `gpsplot ts
--rate` uses the product's view, declared, whatever the plot's `--view`, so
its number is the map's (the raw-view rate is `gps-velocities --view raw`).
A host without the catalog falls back to raw and says so: flag
`no_excluded_catalog`, the catalog source in `input_fingerprint`. The k table is built on `declared`
(`gps-velocities k-table --view declared`; within 0.1 of the cleaned one).

### The `cleaned` view (Fable code review 2026-10-16)

`view=cleaned` runs outlier detection on the window alone, fitting a
`lineperiodic` model to the window's epochs (`read_gps_view(clean=True)` on
[t0, t1]). In unrest that flags real signal: on the Nov-2023 dike a 61-d
window aborts on SENG east (flagged `outlier_abort`), and drops 2 epochs on
SKSH and 1 on THOB **silently**. The cleaned set also depends on the window
length, so it is not the set `gpsplot ts --view cleaned` shows. Options
were the product default to `raw` + the declared `excluded_epochs.csv` (taken:
`declared`), or detect once on the record window + the window and slice.

## Also open

- **Which plate for model input.** `plate-periodic` and every
  `fallback:plate` station remove each station's own plate model. Around
  Svartsengi the assignment mixes NOAM and EURA, about 16 mm/yr apart in
  east. `background` mode cancels it, because the station's own secular is
  in its own plate frame. Per-station plate, or NOAM throughout for the
  Reykjanes model input? (BGÓ)
- **`--last N` during unrest.** The data end a few days before today, so
  `--last 7` holds ~4 epochs and is refused (the map example `--last 7
  --vertical` now draws nothing). Options (BGÓ): count `--last` back from
  each station's last epoch, use `-e` set to the last data day, or change
  the example.

- `gps_api` velocities: becomes a projection of this product. The `.NEU`
  `_velocity_feature` path retires; it stamps ITRF2014 on ITRF2008-plate data.
- Model-input export (`gps-velocities export --format gbis|…`): the `.gmt` file
  already is GMT velo input; GBIS needs its own column order.
- `corr_en` is 0 because the components are fitted separately; a joint fit
  with the `.pos` Rne correlations would give a real value.
