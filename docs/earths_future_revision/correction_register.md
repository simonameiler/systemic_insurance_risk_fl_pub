# Correction register

Branch `earths-future/corrections-and-sensitivity`. Every item below maps to
a numbered brief section and, where applicable, to the AUTHOR CHECK / AUTHOR
REVIEW comment it resolves. Status categories: **CONFIRMED BUG** (demonstrated
implementation or equation error, fixed or ready to fix), **REPORTING ISSUE**
(the simulation is correct but a table/figure/text construction from it is
not), **ASSUMPTION** (a modeling choice, not an error, now stated precisely
and tested), **UNRESOLVED** (evidence gathered, author decision required).

All reproduction commands below assume `conda activate climada_env` and
`cd` to the repo root, unless noted.

---

## R1. Return-period aggregation sums marginal quantiles (Section 5)

**Evidence.** `scripts/analysis/generate_table_baseline_return_periods.py`
(submitted repo, lines 44-53) sums four independently-estimated marginal
quantile columns (`FHCF Shortfall`, `FIGA Residual`, `Citizens Deficit`,
`NFIP Borrowed`) to produce the "Total public burden" row. Reproduced exactly
from the archived baseline (`results/mc_runs/emanuel_era5_baseline_20260326_141913/iterations.csv`,
10,000 seasons): RP10=1.69, RP100=73.99, RP1000=214.29 (USD B), matching
Table 1 to the reported precision (`results/earths_future_revision/tables/table1_legacy_reproduction.csv`).

**Status.** CONFIRMED BUG (statistical: sum of quantiles != quantile of the
sum for non-comonotonic variables; unit test:
`fl_risk_model/tests/earths_future/test_accounting_fixtures.py::test_sum_of_quantiles_differs_from_quantile_of_sum`).

**Fix.** `scripts/earths_future_revision/section5_return_periods.py` computes
the season-level sum first, then its empirical return level (quantile
convention: `q = 1 - 1/RP`, linear interpolation, all 10,000 seasons
including zeros). Total loss and each loss-decomposition row remain marginal
quantiles (unaffected; this was never summed from components -- verified:
39.5+41.8+... != reported total loss at any RP).

**Outputs.** `results/earths_future_revision/tables/table1_corrected.csv`,
`table1_legacy_reproduction.csv`, `table1_corrected_summary.json`.

**Numerical effect (legacy 4-component sum, aggregation-order fix only).**
RP10 1.7->1.9, RP100 74.0->73.8, RP1000 214.3->184.2 (USD B). The
100-year/10-year amplification ratio moves from 43.9x to 39.1x from this fix
alone; see R5 for the combined effect with the non-overlap fix (item C1).

**Reproduce.**
```
python scripts/earths_future_revision/section5_return_periods.py \
  --iterations results/mc_runs/emanuel_era5_baseline_20260326_141913/iterations.csv
```

---

## C1. Public burden aggregate double-counts the FHCF shortfall (Section 4)

**Evidence.** Traced `fl_risk_model/runner.py` steps 6-10: private and
Citizens wind losses are netted against FHCF recovery and cat-bond recovery
(`NetWindUSD = Gross - Recovery`) *before* capital depletion
(`apply_losses_to_surplus`, step 9) and the FIGA deficit calculation (step
10, `deficit = sum of negative EndingSurplusUSD among defaulted entities`).
Whatever fraction of an insurer's or Citizens' FHCF shortfall (the amount
left unrecovered when the USD 17B seasonal cap binds) is not absorbed by
that entity's own remaining capital therefore already appears inside
`figa_residual_deficit_usd` or `citizens_residual_deficit_usd`. Confirmed
with a hand-derived fixture using the real `fl_risk_model.fhcf` functions
(`test_fhcf_shortfall_propagates_into_downstream_default_deficit`): an
insurer with an extra $5,000,000 of loss entirely above its own FHCF company
limit ends up with a $5,000,000 default deficit when capital is exactly
calibrated to the no-shortfall case -- the shortfall flows 1:1 into the
deficit. A second fixture
(`test_fhcf_shortfall_absorbed_by_capital_is_not_double_counted_when_dropped`)
shows the opposite case: an insurer with ample capital bears the shortfall
entirely as a private capital loss that never touches FIGA, Citizens, or
NFIP.

**Status.** CONFIRMED BUG (accounting: the submitted aggregate
`B_public = F_FHCF + F_FIGA + F_Cit + NFIP_borrow` is not a non-overlapping
sum).

**Fix.** Per the brief's instruction not to simply subtract the FHCF
shortfall without tracing where it went: the corrected aggregate excludes
`fhcf_shortfall_usd` entirely (`CORRECTED_BURDEN_COMPONENTS` in
`scripts/earths_future_revision/common.py`), because *every* dollar of
FHCF shortfall ends up either (a) inside a FIGA or Citizens deficit
(already counted there) or (b) absorbed by an entity's own remaining
capital (a private loss, not a public/quasi-public backstop obligation, and
therefore correctly outside the public-burden definition either way). The
original 4-component sum is retained and reported alongside the corrected
one for reconciliation (`public_burden_legacy_usd` vs.
`public_burden_corrected_usd` in every output table). FHCF shortfall is kept
as a separate, explicitly labeled upstream diagnostic of FHCF capacity
stress (statewide utilization of the $17B cap) rather than folded into a
four-channel stack.

**Numerical effect (combined with R1, at the ERA5 baseline).** Total public
burden (season-sum, corrected, non-overlapping): RP10 1.9, RP100 58.6,
RP1000 163.4 (USD B), vs. the submitted 1.7 / 74.0 / 214.3. The 100y/10y
amplification ratio moves from "more than forty" (43.9x, submitted) to
**31.1x**. The fitted log-log exponent beta (RP10-RP1000) moves from ~1.64
(submitted, from the buggy construction) to **1.51** (corrected). The
1%-of-GDP annual exceedance probability is comparatively stable (3.71% legacy
vs. 3.69% corrected) because it is a season-level threshold indicator that
was never affected by the quantile-summing bug; the 10%-of-GDP probability
moves from 0.13% to 0.08%.

**Tests.** `fl_risk_model/tests/earths_future/test_accounting_fixtures.py`
(6 tests), reconciliation check in
`scripts/earths_future_revision/section6_decomposition.py` (max
`reconciliation_check_diff` across severity bins: 2.8e-17, i.e. exact).

**Author decision needed.** Whether the manuscript should report only the
corrected 3-component aggregate, or both aggregates side by side as done
here. This register recommends showing both, since the legacy sum remains
useful as an upper bound on gross financing activity across channels even
though it is not a clean "total public burden."

---

## C2. FHCF/Citizens capital hit is applied once to the seasonal aggregate (Section 4)

**Evidence.** `fl_risk_model.catbonds._payout_occurrence` and the FHCF cap
logic in `fl_risk_model.fhcf.apply_fhcf_recovery` are applied exactly once,
to the season-aggregated (or paired-storm-aggregated) loss, per
`run_one_iteration` (mc_run_events.py) and Supporting Text S4/S5's own
description ("The financial calculation applies recoveries, capital, and
backstop capacity to the gross losses... We apply it once to an individual
event or to the combined county losses of a paired scenario or synthetic
season"). Fixture
`test_two_subattachment_events_can_sum_above_attachment_in_season_aggregate`
shows this can trigger an occurrence-type cat bond payout from two
individually sub-attachment events, which a genuinely per-occurrence
(single-event) contract would not pay on either event alone.

**Status.** ASSUMPTION (an explicit, already-documented simplification of a
per-occurrence contract into a seasonal aggregate), not silently corrected
into a multi-event or multi-year recovery model, per the brief's explicit
instruction. The model does not claim eventwise financial accounting.

**Action.** Supporting Text S4/S5 language already states this; no
numerical change made. The correction register records the quantified
mechanism (two fixtures in
`fl_risk_model/tests/earths_future/test_seasonal_aggregation.py`) so the
approximation is evaluable rather than merely asserted.

---

## C3. FHCF coverage election possibly applied twice (Section 4; AUTHOR CHECK S4-FHCF; Reviewer 2 Eq. 9)

**Evidence.** `fl_risk_model.config.FHCF_RET_MULTIPLES = {90: 6.0732, 75:
7.2878, 45: 12.1464}` satisfy `mult(p) * p = const` to 4 significant figures
(`7.2878/6.0732 = 1.2000`, `12.1464/6.0732 = 2.0000`). Confirmed against the
primary 2023-24 FHCF Reimbursement Contract (found via web search, SBA
Florida / fhcf.sbafla.com): the FHCF Retention Multiple is adjusted to
**200%** (45% coverage), **120%** (75% coverage), and **100%** (90%
coverage) of the same base value -- an exact match to the constants in
`config.py`. This retention-multiple design exists specifically to back out
a coverage-election-independent retention point from a premium that itself
already scales with the elected coverage percentage. If so,
`LimitUSD = premium x payout_multiple` (no separate coverage factor, per
`fl_risk_model.fhcf.normalize_fhcf_terms`) already reflects the *elected*
(not a 100%-equivalent) layer size, and `apply_fhcf_recovery`'s subsequent
multiplication of the capped recovery by `CoveragePct_norm/100` applies the
election a second time. Quantified in
`test_fhcf_coverage_election_may_be_applied_twice_between_retention_and_limit`:
at a 45% coverage election, the alternative (no second multiplication)
formula recovers `1/0.45 = 2.22x` more of the capped layer than the current
implementation.

A related, confirmed **code inconsistency** (not active in production): the
Citizens-specific fallback in `fl_risk_model/config.py`
(`CITIZENS_FHCF_LIMIT_USD = Premium x PayoutMultiple x CoveragePct`)
*does* include the coverage factor in the limit, unlike the general
`normalize_fhcf_terms` path used for every company including, in practice,
Citizens (confirmed: `fl_risk_model/data/24fin_fhcf.csv` has a valid
NAIC-matched row for Citizens, so the general path is used and this
config-only fallback is dead code under current inputs; it would activate,
inconsistently, only if that CSV row went missing or
`CITIZENS_FHCF_FORCE_CONFIG_TERMS` were set `True`).

**Status.** UNRESOLVED (author decision required) for the primary formula;
CONFIRMED (latent, currently inactive) for the Citizens fallback
inconsistency.

**Why not changed by default.** The brief explicitly instructs: "Do not
report a corrected contract formula solely from an inference about
multipliers if the primary terms remain ambiguous." The retention-multiple
evidence is strong but the primary contract clause defining exactly how
each company's `FHCFPremium` in `24fin_fhcf.csv` was computed (whether it is
already net of the company's own elected percentage) was not independently
confirmed against that filing.

**Author decision needed.** Confirm whether `FHCFPremium` in
`fl_risk_model/data/24fin_fhcf.csv` is reported net of each company's
elected coverage percentage. If yes, remove the `* coverage_frac` term from
`apply_fhcf_recovery`'s `RecoveryUSD` line (retain `* FHCF_LAE_FACTOR`) and
rerun every FHCF-touching production result (baseline, all GCMs, all policy
scenarios, insured-fraction sweep) -- this requires the proprietary hazard
inputs (see `data_inventory.md`) and could not be executed in this pass.
Recovery amounts would increase for companies electing less than 90%
coverage, which would increase FIGA-adjacent stress for the highest-severity
seasons and reduce the FHCF utilization/cap-binding frequency, in
directions that partially offset each other in the aggregate public-burden
metric.

**Remaining command** (once decided and hazard access restored):
```
python scripts/run/run_emanuel_monte_carlo.py --n-iter 10000 --seed 42 \
  --scenario baseline --out-dir results/earths_future_revision/reruns/fhcf_corrected_baseline
```

---

## R2. Season-count categories (28.5% / 15.3% / 56.2%) (Section 8; AUTHOR CHECK S5-COUNTS)

**Evidence.** Reproduced exactly from the archived ERA5 baseline
(`scripts/earths_future_revision/section8_historical_and_variance.py`):
- 28.53% of the 10,000 seasons have `total_damage_usd <= 0` -- every one of
  these seasons still has at least one nominally retained track within the
  150 km coastal buffer (`events` column is never empty), so "no TC losses"
  means zero **loss**, not zero retained tracks.
- Of the remaining 71.47% (nonzero-loss) seasons, 1,530 have exactly one
  contributing event (15.30% of all 10,000) and 5,617 have two or more
  (56.17% of all 10,000).

**Status.** RESOLVED (reporting clarification; not a bug). The three
percentages are: P(zero loss), P(nonzero loss AND exactly one event), and
P(nonzero loss AND 2+ events); they sum to 100.00%.

**Action.** Add this precise definition to Supporting Text S5, replacing the
AUTHOR CHECK S5-COUNTS comment.

**Reproduce.**
```
python scripts/earths_future_revision/section8_historical_and_variance.py
```

---

## C5. SI Table S4 building-code column: stale gross totals (Section 9)

**Evidence.** `fl_risk_model/mc_run_events.py::run_one_iteration` captured
`wind_total = numsum(wind_df["WindDamageUSD"])` and the corresponding
`water_total` **before** the building-codes loss-reduction block that
replaces `wind_df`/`water_df` with reduced-damage versions. `total_damage_usd
= wind_total + water_total` therefore never reflected the prescribed
reduction, even though the downstream insurance allocation (fed the reduced
`wind_df`/`water_df` via `_inject_damage_loaders`, called after the
building-codes block) correctly used the reduced damage. Verified against
the archived `emanuel_era5_building_codes_major_20260328_034126` run:
reported mean `total_damage_usd` ($19.34B) is bit-for-bit identical to the
baseline's ($19.34B); the true, allocation-consistent total (reconstructed
as `wind_insured_private + wind_insured_citizens + wind_uninsured +
wind_underinsured + flood_insured_capped + flood_underinsured`) is $13.69B,
exactly consistent with a 30% wind / 25% flood reduction applied to the
correctly-reduced components. This same stale total, when a downstream
export subtracts insured wind from it to report "uninsured wind," produces
an increase (submitted Table S4: 9.7B -> 11.7B) where the true,
correctly-computed uninsured+underinsured wind columns show a **decrease**
(9.7B -> 6.8B, verified in
`fl_risk_model/tests/earths_future/test_scenario_totals.py`).

**Status.** CONFIRMED BUG (implementation: stale summary variable), fully
diagnosed and located.

**Fix (applied).** `fl_risk_model/mc_run_events.py`: moved the
`wind_total`/`water_total` computation to immediately after the
building-codes scenario block. Verified: for scenarios that do not touch
physical losses (baseline, market exit, penetration), the reported total is
unchanged (reconstruction equals the reported value exactly,
`test_unaffected_scenarios_reconcile_exactly`). This is a source-code fix;
future production runs of the building-codes scenario (and the GCM x
building-code sweep) will report the correct total directly.

**Correction of already-archived output (no rerun needed).** Because the
insurance-allocation component columns were already correct,
`scripts/earths_future_revision/section9_climate_policy_tables.py`
reconstructs the corrected total post hoc from those columns for the
archived runs, without needing to replay the financial model.

**Outputs.** `results/earths_future_revision/climate_policy/table_S4_corrected_means.csv`.

**Numerical effect.** Building-codes column: Total loss 19.3B -> **13.7B**;
Un/underinsured wind 9.7B -> **6.8B** (was reported as an increase to
11.7B; the corrected direction is a decrease); Un/underinsured flood 2.4B ->
**1.8B** (also a decrease, not the reported increase); Total public burden
(corrected, non-overlapping definition) 2.7B (submitted, legacy def.) ->
**1.5B**.

**Remaining work.** The 11-level x 5-GCM building-code sweep
(`results/mc_runs/emanuel_{gcm}_ssp245cal_buildingcode_w*f*_*/`) has the same
bug and the same post-hoc reconstruction applies; SI Figure S3 and the
climate-offset crossing calculation (Methods "Loss reductions needed to
offset climate effects") should be regenerated from the reconstructed totals
before submission. This reprocessing was not completed in this pass (time
constraint); the reconstruction formula and script above generalize directly
to every sweep directory.

---

## R3. `w##f##` sweep labels are parameter encodings, not percentages (Section 9)

**Evidence.** Archived directory names include `buildingcode_w120f80` and
`buildingcode_w130f90`. A wind-loss-*reduction* of 120% or 130% is not
physically valid (it would imply losses become negative). Inspection of
`scripts/run/run_climate_buildingcode_sensitivity_windfloods.py` (line
~99-101) confirms these labels encode the **remaining-loss** fraction times
100 in a `w<remaining%>f<remaining%>` scheme relative to a different
reference, not a direct reduction percentage; the "MAJOR" building-code
scenario used elsewhere (30% wind / 25% flood reduction) corresponds to
`w70f75`-equivalent remaining fractions, and the observed sweep covers 11
distinct settings, not 13.

**Status.** UNRESOLVED (author decision / follow-up needed). This pass
confirmed the labels are parameter encodings (not invalid physical
percentages) and located the generating script, but did not fully
re-derive the exact formula mapping `w##f##` to (wind_reduction,
flood_reduction) pairs, nor reconcile the observed 11 settings against the
manuscript's stated 13 levels and 3:2 wind:flood ratio. Do not infer the
mapping from the filenames alone (per the brief); the exact formula must be
read from `run_climate_buildingcode_sensitivity_windfloods.py`'s scenario
generation loop, which was inspected but not fully transcribed here due to
time.

**Remaining command.**
```
python -c "import inspect; from pathlib import Path; \
print(Path('scripts/run/run_climate_buildingcode_sensitivity_windfloods.py').read_text())" \
  | sed -n '1,130p'   # inspect the scenario-generation loop directly
```

---

## R4. Historical-scenario realization count: 200 (manuscript) vs. 1,000 (archived runs)

**Evidence.** Main Methods and Supporting Text S5 state 200 Monte Carlo
realizations per historical/paired scenario. The locally available archived
runs that reproduce the submitted SI Table S3 numbers exactly (see
`data_inventory.md`) contain 1,000 realizations each.

**Status.** REPORTING ISSUE (provenance mismatch, not a numerical error --
the reproduced means and 5th-95th percentile intervals match the submitted
table exactly regardless of N).

**Author decision needed.** Correct the stated realization count to 1,000,
or substitute a 200-realization run if one is the intended source.

---

## R5. Insured wind fraction elasticities recomputed from unrounded means (Section 7)

**Evidence/action.** Reused the archived fixed-fraction sweep
(`results/mc_runs/insured_frac_sensitivity_combined/`, 0.1-0.5, 10,000
seasons, seed 42) because it remains compatible with the corrected
*accounting* (a post-processing change; the underlying per-season component
values are untouched). Recomputed every Table S6 elasticity via centered
log-log finite difference at f in {0.3, 0.5} around f=0.4, from unrounded
means, in `scripts/earths_future_revision/section7_insured_fraction.py`.

**Result.** All previously reported elasticities reproduce closely (FHCF
1.60, FIGA 1.40, Citizens 1.28, private wind 1.00, defaults 0.57, wind
un/underinsured -0.66 -- matching Table S6 to the reported precision, since
means were never affected by the return-period bug). Added a new elasticity
for the corrected total public burden (non-overlapping definition): **1.14**
at f=0.4 (still amplifying, but smaller than any individual quasi-public
component, since it excludes FHCF).

**Beta(4,6) vs. fixed f=0.4.** Quantified directly
(`results/earths_future_revision/insured_fraction/beta46_vs_fixed04_comparison.json`):
every metric differs by less than 5% between the Beta(4,6) baseline run and
the fixed-f=0.4 run, confirming they are close but not identical (as the
brief specifies), consistent with a nonlinear model and Jensen's inequality.

**Status.** RESOLVED for the accounting-and-elasticity recomputation.
UNRESOLVED for any correction that would require rerunning the sweep against
the FHCF formula in C3, should that be changed.

---

## C6/U1. NFIP structure-to-loss allocation sensitivity (Section 7)

**Evidence/action.** Built the three allocation configurations
(structure-weighted baseline, SFHA-only, non-SFHA-only) directly from the
production NFIP data source
(`fl_risk_model/data/NfipResidentialPenetrationRates.csv`, `asOfDate`
**2025-05-15**, FEMA OpenFEMA `NfipResidentialPenetrationRates` dataset,
filtered to Florida's 68 counties) in
`scripts/earths_future_revision/section7_nfip_allocation.py`. Verified: the
baseline rate lies within the per-county `[min, max]` envelope of the three
configurations for all 68 counties (`test_baseline_rate_lies_within_county_envelope`);
the baseline reproduces the FEMA all-county rate exactly wherever no
clipping/missing-data cleaning occurred
(`test_baseline_reproduces_reported_all_county_rate_apart_from_cleaning`);
and the SFHA-rate-exceeds-non-SFHA-rate ordering, while true for 67/68
counties, is not universal (1 county reverses it), so the configurations are
correctly *not* labeled a uniform upper/lower bound.

**Status: dataset snapshot.** RESOLVED (AUTHOR CHECK S1-NFIP /
"NFIP data version"): the exact snapshot is documented above.

**Status: financial rerun.** BLOCKED. Quantifying the resulting change in
insured flood losses, NFIP financing, and corrected aggregate burden
requires re-running the financial model against the same per-county,
per-event flood losses used in the archived production run. That cache
(`fl_risk_model/data/hazard/emanuel/`, `.../gori_data/`) is not present in
this environment (proprietary; see `data_inventory.md`). Exact remaining
steps, including the small code change needed (an `--nfip-allocation` CLI
switch does not exist yet), are printed by the script and reproduced here:

```
python scripts/earths_future_revision/section7_nfip_allocation.py
```

**Deliverable in this pass.** The three-configuration county rate table
(`results/earths_future_revision/nfip_allocation/county_flood_allocation_configurations.csv`)
and its tests. This is the "focused structural test" scope specified by the
brief; it does not, on its own, quantify a change in insured flood losses.

---

## Other AUTHOR CHECK items: status summary

| Item | Status | Note |
|---|---|---|
| S1-BONDS (cat bond inventory/trigger mapping) | PARTIALLY RESOLVED | `fl_risk_model/data/catbonds_2024.csv` exists and is the production source; a formatted inventory with source dates was not produced in this pass (remaining work). |
| S2-FREQUENCY (frequency calibration target/factor) | UNRESOLVED | Config constants for the frequency correction were not located distinctly from the year-set generation scripts in the time available; requires reading `scripts/hazard/generate_emanuel_year_sets.py` and `setup_emanuel_metadata.py` in full. |
| S2-SPLIT (Gori regression coefficients, Beta shape, historical wind-share sources) | UNRESOLVED | Coefficients/units not independently re-derived; the four historical wind-share means (70%, 87.5%, 30%, 50%) were not traced to a primary citation in this pass. Author to supply source. |
| S3-CALIBRATION (38.9% weighting) | PARTIALLY RESOLVED | The median of the six Table S1 summary statistics (32.41, 34.24, 40.68, 38.35, 39.19, 44.03) is 38.77%, close to but not exactly 38.9%; this is a plausible but unconfirmed reconstruction of "the calibration combines the annual means, the two estimation approaches, and the means for years with large losses." Author to confirm the exact weighting rule. |
| S3-RESIDENTIAL (household-burden mapping) | CONFIRMED ASSUMPTION, unresolved for correction | LitPop/USA impact function include non-residential value; Methods and SI now state this explicitly (already reflected in the supplied revision). No new residential scaling introduced, per the brief. |
| S4-ASSESSMENTS (FIGA premium base, collection horizon) | RESOLVED, no bug found | `fl_risk_model/runner.py` step 10 uses the full non-defaulted-entity premium base (comment: "Survivors only (corrected): use full capital_with_groups, not defaults_view"), consistent with "surviving-insurer denominator." Collection horizon is one season, as documented; not extended. |
| S5-EXIT (private capital exponent) | **CONFIRMED DISCREPANCY** | `fl_risk_model/scenarios/market_exit.py::adjust_citizens_capital_for_growth` uses `exponent = 1.3` (comment: "Empirically calibrated"), not the 1.2 stated in Supporting Text S5. With the actual 15%->25% share change, exponent 1.3 gives a **+94%** surplus increase, not the stated "~84%" (which corresponds to exponent 1.2). The archived `emanuel_era5_market_exit_moderate` results already reflect exponent 1.3. Recommended action: correct Supporting Text S5 to state exponent 1.3 and ~94%, matching the code and the archived results (rerunning to match the text would discard already-validated production output for no benefit). |
| S5-COVERAGE (30% flood target normalization) | PARTIALLY RESOLVED | `flood_penetration_target: 0.30` is a direct configured value for the "MAJOR" preset (`fl_risk_model/scenarios/penetration.py`), reached via an SFHA-aware allocation function; `coastal_focus_factor: 1.5` matches the SI's stated coastal multiplier exactly. The internal normalization connecting the stated 1.2x (SFHA) / 3.0x (non-SFHA) county multipliers to this 30% aggregate target was not traced statement-by-statement in the time available. |
| S5-OFFSET (13 mitigation levels, 3:2 ratio) | See R3 above | Confirmed as parameter encodings, not fully re-derived. |
| S6-DRAWS (TIV distribution/dependence) | PARTIALLY RESOLVED | `fl_risk_model/exposure.py` samples county TIV as `max(Normal(v, CoV=0.15*v), 0)` (`cfg.EXPOSURE_COV = 0.15`), independently per county/entity draw (no explicit cross-county or cross-entity correlation structure found). Sampling frequency (once per Monte Carlo iteration) confirmed via `mc_run_events.py`. Not exhaustively verified against every call site. |
| S6-ANOVA (Table S7 season-selection strata/weights/seeds) | RESOLVED for removal decision (see below); not further audited | Reproduced the reported eta-squared values exactly from the archived nested design (`results/earths_future_revision/historical/table_S7_audit_retained_for_code_reference.csv`); did not re-derive the selection-stratification code in `scripts/run/run_variance_decomposition.py` beyond confirming its output matches the manuscript. |
| S6-IRMA (primary FHCF Irma recovery figure) | PARTIALLY RESOLVED | Secondary sources (SBA Florida FHCF reporting, summarized via beinsure.com and a WGCU/PBS report) put FHCF's ultimate Irma-only loss/reimbursement at approximately USD 6-6.5 billion, about 36-38% of the USD 17B statutory cap. This was not confirmed against the primary FHCF annual report PDF (`sbafla.com/media/uvmnfyka/2022-fhcf-annual-report.pdf`) in this pass; author should verify the exact figure and contract year before citing it. The modeled Irma scenario's near-zero FHCF recovery (assumed 50% wind share) versus this ~USD 6B actual figure is the discrepancy Reviewer 2 and the SI both flag; this register does not tune the model to close that gap, per the brief. |

---

## Table S7 removal (Section 8)

**Decision.** Table S7 (variance decomposition) and its associated
hazard-dominance / broad-robustness language are removed from the
manuscript copy in this worktree. Reproduced exactly from archived results
(above), so removal is not a concession that the numbers are wrong -- it
reflects the brief's instruction that the 300-season, non-random selection
design does not, on its own, support the broad robustness claims previously
attached to it, and that the focused allocation tests (Section 7, this
register) are more directly relevant to the manuscript's central claims.
The research code (`scripts/run/run_variance_decomposition.py`) and archived
results (`results/mc_runs/variance_nested_300x50_20260311_211009/`,
`variance_fixed_params_20260311_185311/`) are retained unmodified.

---

## Manuscript claims requiring wording changes (not numerical corrections)

- **California catastrophe models** (main.tex, Introduction): confirmed via
  web search that California's Department of Insurance finalized a
  regulation in December 2024 (effective 2025) permitting catastrophe
  models and a capped net cost of reinsurance in ratemaking, reversing the
  prior prohibition. Update to past tense with a citation to the CDI
  regulation (sources: insurance.ca.gov press releases, December 2024 and
  2025).
- **Competing interests / WindRiskTech.** Per the brief, this is recorded as
  an author question, not inferred as an interest: does any author (or a
  close collaborator) have a financial or advisory relationship with
  WindRiskTech L.L.C. beyond the stated data-use agreement? The existing
  disclosure names Kerry Emanuel's board role at Trusted Resource
  Underwriters; WindRiskTech is not currently disclosed as a competing
  interest.
- **Government-spending / banking-bailout comparison** (Discussion,
  main.tex): removed in the manuscript copy per the brief and per Reviewer
  2's comment 3 (FIGA/Citizens deficits are private cross-subsidies, not
  state expenditure; only NFIP borrowing is a fiscal outlay).
