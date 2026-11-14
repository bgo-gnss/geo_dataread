# T5b: the SIL Mlw → M0 relation

Ticket T5b of `PLAN-seismicity.md` (§7, decision R1 in §9c). Fitted
2026-11-14 from the full reviewed SIL archive. Code:
`geo_dataread.seismic.filters` (`MLW_M0_FIT`, `moment_from_mlw`,
`mlw_from_moment`, `with_derived_moment`). Script:
`docs/seismic/fit_mlw_m0.py`. Figure: `mlw_m0_relation.png`.

## 1. Result

SIL's Mlw is a function of the measured seismic moment. The fit recovers that
function exactly: a continuous piecewise-linear curve in log10(M0).

| Mlw range | log10(M0 / N·m) | Mlw per decade of M0 |
|-----------|-----------------|----------------------|
| Mlw ≤ 2 | Mlw + 10 | 1.0 |
| 2 – 3 | 12 + (Mlw − 2)/0.9 | 0.9 |
| 3 – 4.6 | 13.111 + (Mlw − 3)/0.8 | 0.8 |
| 4.6 – 5.4 | 15.111 + (Mlw − 4.6)/0.7 | 0.7 |
| > 5.4 | 16.254 + (Mlw − 5.4)/0.5 | 0.5 |

A free fit (9 parameters, robust soft-L1, 153 701 events with Mlw 1.5–6) gives
intercept 10.0000, knees at Mlw 2.0002 / 2.9993 / 4.6 / 5.4027 and
1/slope 0.9 / 0.8 / 0.6998 / 0.4974. The code uses the rounded values. A
third knee at Mlw 4.55 or 4.65 instead of 4.6 widens the residual σ of the
Mlw 4.3–5 events from 0.0047 to 0.0057–0.0066.

- **Fit data.** 797 559 reviewed events with Mlw < 6 and a positive
  `events.fps` moment, 1991-01-17 to 2026-11-13. They were read through
  `read_sil_catalog_cached(..., write=False)`, year by year, with the manual
  rows kept.
- **Why `automatic=True`.** `automatic=False` bypasses the cache and reads
  the archive directly. A check on 2023 showed that the manual rows of the
  cached path equal a direct read: 42 648 rows, with identical `mag_mlw` and
  `moment_nm`.
- **Residual.** log10(M0_measured / M0_relation) has median 0.0000 and a
  robust σ (MAD) of 0.0039. That is the 0.01 rounding of Mlw, and the
  relation adds nothing to it.

Plan R1 expected "slope ≈ 1 bending towards HK above ~M2". The bend is real,
but it does **not** follow Hanks–Kanamori:

- The slope is 1.11 from Mlw 2 to 3, then 1.25 up to 4.6, then 1.43 up to
  5.4. HK's slope is 1.5.
- Above Mlw 5.4 the slope is 2.0, steeper than HK.
- HK never matches the relation except where the two lines cross, near
  Mlw 1.9.

## 2. Residuals

### By Mlw bin

Robust σ is 1.4826·MAD.

| Mlw | n | median | σ | p2.5 | p97.5 | HK median |
|-----|---|--------|---|------|-------|-----------|
| < 0 | 51 366 | +0.0001 | 0.0039 | −0.0050 | +0.0108 | +1.023 |
| 0–1 | 432 364 | +0.0000 | 0.0038 | −0.0051 | +0.0051 | +0.684 |
| 1–1.5 | 160 128 | +0.0000 | 0.0040 | −0.0055 | +0.0053 | +0.343 |
| 1.5–2 | 88 114 | −0.0001 | 0.0037 | −0.0049 | +0.0049 | +0.098 |
| 2–2.5 | 41 878 | +0.0001 | 0.0041 | −0.0061 | +0.0058 | −0.121 |
| 2.5–3 | 14 212 | −0.0001 | 0.0041 | −0.0056 | +0.0054 | −0.315 |
| 3–3.5 | 5 812 | +0.0001 | 0.0047 | −0.0063 | +0.0064 | −0.486 |
| 3.5–4 | 2 349 | +0.0002 | 0.0048 | −0.0064 | +0.0062 | −0.613 |
| 4–4.5 | 888 | −0.0002 | 0.0045 | −0.0061 | +0.0061 | −0.734 |
| 4.5–5 | 305 | +0.0006 | 0.0049 | −0.0070 | +0.0068 | −0.843 |
| 5–5.5 | 114 | +0.0006 | 0.0065 | −0.0077 | +0.0076 | −0.881 |
| 5.5–6 | 29 | −0.0008 | 0.0079 | −0.0088 | +0.0143 | −0.771 |

"HK median" is log10(M0_measured / M0_HK) for the same events. It is positive
where HK's moment is too small and negative where it is too large.

### By era

| era | n | median | σ | off ≥ 2× | n (Mlw ≥ 2) | σ (Mlw ≥ 2) | off ≥ 2× (Mlw ≥ 2) |
|-----|---|--------|---|----------|-------------|-------------|--------------------|
| 1990–94 | 24 313 | +0.0000 | 0.0039 | 2 | 1 588 | 0.0046 | 0 |
| 1995–99 | 124 951 | +0.0000 | 0.0038 | 41 | 7 078 | 0.0042 | 2 |
| 2000–04 | 76 246 | −0.0000 | 0.0046 | 654 | 7 352 | 0.0045 | 43 |
| 2005–09 | 91 448 | +0.0000 | 0.0039 | 185 | 11 075 | 0.0042 | 14 |
| 2010–14 | 118 983 | +0.0000 | 0.0038 | 69 | 13 721 | 0.0043 | 37 |
| 2015–19 | 123 175 | +0.0000 | 0.0038 | 3 | 7 240 | 0.0042 | 0 |
| 2020–23 | 172 080 | +0.0000 | 0.0038 | 1 | 14 178 | 0.0042 | 1 |
| 2024–26 | 66 365 | +0.0000 | 0.0038 | 0 | 3 357 | 0.0041 | 0 |

- **Stable over time.** The definition has not changed since 1991: in every
  era the intercept below Mlw 1.9 is +0.0000 to +0.0001, and σ is the same.
- **Support for the upper knees.** The knees at Mlw 4.6 and 5.4 rest on fewer
  events: 337 events at Mlw 4.6–6 and 49 at 5.4–6. The Mlw ≥ 5.3 events come
  from 1992, 1998, 2000, 2002, 2003, 2012–2014, 2018 and 2025, and all of
  them lie on the same segments (σ 0.0065–0.0079).
- **Resolution.** `mag_mlw` comes at 0.01 resolution in every era (from
  lib.mag or events.lib). About 10 % of the values are multiples of 0.1, which
  is what 0.01 resolution gives by chance.

### Events off the relation

- **955 events (0.12 %) are off by ≥ 2×; 400 by ≥ 3×.**
  - Mostly 2000–2002 (624 of the 955), the years of the South Iceland
    Seismic Zone sequences.
  - Mostly small: median Mlw 0.77.
  - 13 % have a 0.1-step Mlw.
  - They are probably events whose Mlw or moment was edited by hand after
    the moment was written. They do not bias the fit (median and MAD are
    robust to them).
- **Mlw ≥ 6 is outside the relation.** There are two such reviewed events:
  - 1998-06-04 21:36:53, Mlw 6.10, M0 5.0e15 N·m;
  - 2000-06-21 00:51:47, Mlw 6.57, M0 1.2e17 N·m.

  Their measured moments are 50–100× below the relation, and both are also
  far below the true size of those earthquakes: SIL's spectral M0 saturates
  for the largest events. `moment_from_mlw` above Mlw 5.95 (the largest
  fitted value) extrapolates the 0.5-per-decade segment.
- **The relation reproduces SIL's moment, not the true moment.** The
  saturation already starts below Mlw 6. The 2000-06-17 15:40:41 South
  Iceland mainshock (Mw ≈ 6.5, true M0 ≈ 6e18 N·m) is in the archive as
  Mlw 5.46, M0 2.4e16 N·m, and it sits on the relation. So a cumulative-moment
  curve across the 2000 or 2014 sequences is a lower bound, whether it uses
  measured or derived moments.

## 3. Hanks–Kanamori applied to Mlw (the T5 behaviour)

Counts are out of the 797 561 events that have both Mlw and a measured M0.

| factor | HK moment too small | HK moment too large |
|--------|--------------------|---------------------|
| ≥ 3× | 463 182 (58.1 %) | 7 103 (0.9 %) |
| ≥ 5× | 251 894 (31.6 %) | 1 261 (0.2 %) |
| ≥ 10× | 33 345 (4.2 %) | 18 |

Total moment of the events with Mlw < 6:

| source | total M0 (N·m) | relative to measured |
|--------|----------------|----------------------|
| measured | 5.39e18 | 1 |
| `moment_from_mlw` | 5.22e18 | 0.97× |
| HK on Mlw | 3.02e19 | 5.6× |

HK errs in both directions:

- **Too small** for the many small events. An Mlw 1 event gets 2.8× too
  little moment, an Mlw 0 event 8.9× too little.
- **Too large** for the few big ones that dominate the sum. An Mlw 3 event
  gets 2.8× too much, an Mlw 4 event 4.9× too much and an Mlw 5 event 7.1×
  too much.

So a cumulative-moment curve built with HK would be wrong by a factor of
about 5, and its steps would be in the wrong proportion.

## 4. Which reviewed events need a derived moment

10 009 reviewed events have an Mlw but no `events.fps` moment. These are the
rows `with_derived_moment(df, magnitude_types=["Mlw"])` fills.

- **1990–1991: 7 268 events.**
  - In 1990 no event has a moment, so the relation is not verified there;
    it is the same SIL definition and is assumed to hold.
  - The 1991–1994 events that do have a moment fit with σ 0.004.
- **The rest are in busy sequences:**
  - 2000: 734
  - 2009: 135
  - 2014: 616 (Bárðarbunga)
  - 2023: 873 (November dike)

These events have median Mlw 0.46, 99th percentile 2.6 and a maximum of 5.39.
They lie inside the fitted range.

## 5. API

- **`MLW_M0_FIT: MlwMomentRelation`** (frozen dataclass). It holds the curve
  as `intercept`, `knots` and `slopes`, and the fit record as `fit_start`,
  `fit_end`, `n_events`, `mlw_range`, `sigma`, `sigma_by_bin` and
  `reference`. Its methods are `.log10_moment(mlw)`, the exact inverse
  `.mlw(log10_m0)`, and `.as_dict()`.
- **`moment_from_mlw(mlw, *, relation=MLW_M0_FIT)`** gives N·m.
- **`mlw_from_moment(m0, *, relation=MLW_M0_FIT)`** is the exact inverse.
  A moment that is not positive, or NaN, gives NaN.
- **`with_derived_moment(df, *, magnitude_types, mlw_relation=MLW_M0_FIT)`**
  fills a missing moment by magnitude type:
  - `Mlw` rows use the SIL relation;
  - every other listed type uses HK, because listing it declares that type
    to be Mw;
  - `ML_SIL` and `M` raise `FilterError` (no relation until T18);
  - `mlw_relation=None` brings back the old HK-for-Mlw behaviour, for
    comparison only.

  The subset record now carries `relations` (the relation used per type) and
  `mlw_relation` (the fit record).

## 6. Reproduce

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src mamba run -n gpslibrary \
    python docs/seismic/fit_mlw_m0.py --base /mnt_data/sildata \
    --frame <scratch>/mlw_m0.parquet --png docs/seismic/mlw_m0_relation.png
```

The first run reads all 37 years from the SIL cache, so `build_sil_cache`
must have filled it first. `--frame` keeps the extracted columns, so later
runs take seconds.
