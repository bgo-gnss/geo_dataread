# SeisComP magnitudes → seismic moment (PLAN-ts-quakes Q4)

`geo_dataread.seismic.moment_estimate` fills `moment_nm` per row by the
row's `magnitude_type`. After 2026-02-01 the reviewed catalogue is
SeisComP's, which has no measured moment; these relations give the
cumulative-moment curves an estimate across the seam. BGÓ allowed them for
plotting on 2026-11-15 (Q4), labelled as estimates. This overrides decision R1
of PLAN-seismicity §9c for this use only; `with_derived_moment` still refuses
`ML_SIL`. T18 replaces both relations with a documented fit that includes
residuals.

| Type | Relation (log10 M0, N·m) | Range | Fit set | n | rms |
|------|--------------------------|-------|---------|---|-----|
| measured | `events.fps` moment as read | — | — | — | — |
| `Mlw` | SIL archive relation `MLW_M0_FIT` (`mlw_m0_relation.md`) | all | SIL events.fps 1991–2026 | see there | see there |
| `ML_SIL` | 10.41 + 1.045 M | all | SeisComP back-import ML_SIL of SIL events with a measured moment, matched 2023-01 … 2026-01 | 2622 | 0.29 |
| `Mpgv_w` | 8.95 + 1.347 M | M ≥ 3 | SIL Mlw ≥ 4 events with a measured moment; Mpgv_w taken as Mlw (5.38 vs 5.27, 2026-01-31) | 24 | not recorded |

Not estimated (counted in `attrs["catalog"]["subsets"]`):
- `Mpgv_w` < 3. At small M, Mpgv_w runs 0.7–1.2 above ML_SIL (SIL/SeisComP
  report 2026), so neither relation fits it.
- SIL automatic magnitudes (`aut.mag`, `events.ref`).
- SIL `ML`, `Msil_auto`, the Quakes API's unlabelled `M`, and `Mpgv_alert`.

Known limits:
- ML_SIL saturates near 4 (4.03 for Mpgv_w 5.24), so large events must come
  through Mpgv_w.
- Our Mw from these relations is about 0.75 below GEOFON Mw, consistently.
  BGÓ: keep our scale.
- The two fits were made interactively on 2026-11-15, and no script was kept.
  T18 must re-derive them reproducibly, as `fit_mlw_m0.py` does for Mlw.
- Mpgv_w treats the moment as Mlw-like. SIL's alert2sil PGV magnitude
  (`mag_pgv`) is not used.
