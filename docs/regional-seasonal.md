# Regional seasonal (BGÓ 2026-10-17) — design

**Status:** implemented on branch `feat-regional-seasonal` (gps_analysis,
geo_dataread, gps_plot, gps_api), with the `regional` k(T) entry
(`--k-table`). Not deployed.

## Why

A station without a detrend record of its own has no seasonal removed, so a
`plate-periodic` or `pre-onset` rate over weeks–months aliases the seasonal
(`seasonal:none`; k ≈ 3–4 at 60–120 days against ≈ 2.5–3 with a seasonal
removed). On Reykjanes that is 43 of the 48 stations: every station installed
in the unrest has no quiet span to estimate a seasonal on. BGÓ's curation
rule for them was already a *seasonal-only borrow* with no donor chosen
(detrend campaign, 2026-09); this is the donor.

## Decisions (BGÓ 2026-10-17)

1. **Method A**: interpolate the stored seasonal terms of nearby stations; not
   a daily common-mode stack (that may come later, on top).
2. **All three components** (N, E, U).
3. **Frame itrf2014**; the file is the one the detrend estimator already
   writes, `detrend_params.itrf14.json`.
4. **Every record says how each term group was derived**: `estimated` (fitted
   at the station) or `regional` (interpolated from nearby stations). The
   existing records get an explicit `estimated`.

## What was measured (scratch `seasonal/loo.py`, 2026-10-17)

- Annual-term σ is small (median 0.06 mm N/E, 0.2 mm U), so the station-to-
  station differences (≈ 1 mm horizontal at every distance, 0.3–0.5 of the
  amplitude in U within 100 km) are real, not estimation noise.
- Leave-one-out at the 12 own-seasonal stations within 80 km of Reykjanes
  (itrf14), predicting each from its neighbours (Gaussian weights, L = 50 km,
  donors within 150 km), then k(T) on the station's quiet span. Fraction of
  the gap between nothing removed and its own seasonal closed:

  | | 61 d | 91 d | 122 d |
  |---|---|---|---|
  | U | 85 % | 80 % | 71 % |
  | N | 17 % | 24 % | 16 % |
  | E | 36 % | 49 % | 73 % |

  itrf20 (9 stations) agrees. L = 25 and 100 km were both worse; the nearest
  station alone is unstable.

## The model

For a target station at x, donors i within R of x with an `estimated`
periodic group (4 coefficients c_i per component):

    w_i = exp(-(d_i / L)^2),   c(x) = Σ w_i c_i / Σ w_i

- L = 50 km, R = 150 km (the validated values; stored in the record).
- **Refuse** (no record written, reason logged) when fewer than 2 donors lie
  within 2L, or Σ w_i < 0.1: no silent fall-back to a distant mean.
- Donors are records whose periodic group is `estimated` only — never a
  `regional` or a borrowed one — so the result cannot chain and does not
  depend on processing order.
- Prediction σ per component: the leave-one-out RMS of (predicted − own)
  annual coefficients over the validation stations, stored in the record.
- First version: targets inside the Reykjanes area only (where it was
  validated).

Math lives in the leaf (`gps_analysis`): a pure function from donor
positions + coefficients + target position to (coefficients, weights,
refusal). geo_dataread gathers donors from the detrend document.

## The record

A regional record is a **seasonal-only** record:

- `model: "periodic"` (already in the model registry), `param_names` = the
  4 periodic terms, one `components[]` entry per N/E/U with those params; no
  covariance from a fit (`cov_upper` from the prediction σ, diagonal).
- `detrend_method: "regional"`.
- `groups.periodic`: `{"provenance": "regional:gauss L=50km R=150km",
  "donors": [{"station", "weight", "distance_km", "fitted_at", "digest"}],
  "sigma_pred": [n, e, u], "L_km": 50, "R_km": 150}`.
- No secular group. Nothing may read an offset or a rate from it.

Every other record gains `groups.secular.provenance` and
`groups.periodic.provenance` = `estimated` (a record without a `groups`
block reads as `estimated`; the existing spelling `self` is read as
`estimated` too).

**Writer:** `gps-regional-seasonal --params <detrend doc> --out <merged
doc>` (`geo_dataread.regional_seasonal`). It needs no time series, only the
other records, so it is its own step after `gps-estimate-detrend` rather
than a stage-plan hold: it reads the document, adds a regional record for
each target in `--region` (default Reykjanes) without one of its own
(re-making an old regional one), marks every fitted record `estimated`, and
writes the merged document; the input is never written. A
`periodic=regional` stage-plan hold (fitting a secular while holding the
regional seasonal) is the later step for the 39 curation stations.

First run on the deployed itrf14 document (2026-10-17): 43 regional records,
none refused, 21–38 donors each (top weights SENG, SKSH, KRIV, NYLA, HVSK,
REYK); leave-one-out σ of the annual amplitude 0.77 / 0.57 / 1.93 mm N/E/U
over the 20 own-seasonal stations within 100 km of a target.

`--k-table` measures the k(T) `regional` entry on the same 20 stations: each
station's quiet span with its own secular removed, less the seasonal its
neighbours predict. itrf14 (median N/E/U):

| T | all (own seasonal) | regional | secular (none removed) |
|---|---|---|---|
| 61 d | 2.5 / 2.4 / 2.5 | 2.7 / 2.4 / 2.5 | 2.7 / 2.6 / 3.0 |
| 91 d | 2.8 / 2.6 / 2.6 | 3.3 / 2.6 / 2.7 | 3.3 / 3.2 / 4.1 |
| 182 d | 3.7 / 3.2 / 3.5 | 4.3 / 3.8 / 3.9 | 5.2 / 4.6 / 6.3 |

U and E gain most; N hardly (as the leave-one-out said).

## Fail closed (from the consumer audit, 2026-10-17)

A `model: "periodic"` record is legal today and several paths treat its
missing secular as zero. These must refuse instead:

1. **Leaf** (`gps_analysis.detrend`): `select_terms` / `select_record_groups`
   / `_selected_fits` raise `ValueError` when a requested group (`secular`,
   or `all` on a record without a secular group) has no parameters;
   `record_group_mask` (staged) the same.
2. `gps_views.detrend_arrays`: `all`/`secular` on a seasonal-only record →
   degrade (raw, labelled), never "applied".
3. `velocity`: `background` refuses; `plate-periodic`/`pre-onset` use the
   regional seasonal and flag `seasonal:regional`; `build_k_table` and
   `station_rate_scales` skip seasonal-only records; k for these stations
   from a `regional` k(T) entry measured by leave-one-out (`K_RESIDUAL`).
4. gps_plot: `timesmatplt.trajectory_views`, `sources/gps.model_values`,
   `trajectory.py` refuse a secular/full line; the picker `_summary` reads
   the rate by name (it reads `params[1]` today).
5. Workbench: `borrow_record` must not copy a regional record as a donor;
   `group_contribution` and the done-count must know seasonal-only.
6. gps_api precompute: `_pinned_record` refuses for secular products.
7. `secular_store.secular_from_record`: refuses a regional record (it would
   re-label it `store:STA`).
8. `stage_plan`: `_spell`, `resolve_stage_plan` and
   `check_stage_plan_sources` fall through to the donor branch for an
   unknown hold kind — make them exhaustive before adding `regional`.

Each gets a test with a seasonal-only fixture record, for `terms=all`,
`secular`, `periodic` and velocity `background`.

## Not in this version

- A daily common-mode stack (method B).
- Targets outside Reykjanes.
- Seasonal borrowing for the detrend fits of the 39 Reykjanes curation
  stations (their secular is still unknown); the regional record is what
  such a fit would hold later.
