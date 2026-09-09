# FHCF patch, validation, and local pilot report

Branch `earths-future/corrections-and-sensitivity`. At handoff for this task
the branch was clean at `e584cef`; it remained clean (no other agent or user
commits landed) throughout this work. This report is the primary deliverable
requested for this task; it supersedes only the "smallest proposed patch"
(still-to-apply) and "dependency map" (all-provisional) framing of
`docs/earths_future_revision/fhcf_contract_verification.md`, which is
otherwise unchanged and remains the source-citation record.

## 1. What changed

**Patched:** `fl_risk_model/fhcf.py::apply_fhcf_recovery`, and nothing else.
No other production file was modified. The function's inputs and outputs
(column names, required columns, calling convention) are unchanged, so
every existing caller (`fl_risk_model/runner.py`,
`fl_risk_model/branches/citizens.py`) required no changes.

Both confirmed defects are fixed inside this one function:

1. **Formula order** (Article IV(1) of the FHCF 2023-2024 Reimbursement
   Contract): the Company's Limit now caps the fully coverage-and-expense-
   scaled reimbursement, `min((1+a) * p * E, K)`, instead of capping the
   raw excess before scaling, `(1+a) * p * min(E, K)`.
2. **Aggregation level** (Article V(26)/(28)): `GrossWindLossUSD` is now
   summed to one row per Company before Retention, Excess, and the Limit
   are applied, regardless of whether the input is a single aggregated row
   or split across county rows. The resulting single company-level recovery
   is allocated back to the original rows in proportion to each row's share
   of the company's gross loss (0/0 defined as zero), so `NetWindUSD` is
   still reported at the original row granularity.

Two diagnostic columns changed meaning and were renamed rather than
silently repurposed: the old row-level `ExcessUSD` and `RecoverableUSD` are
replaced by company-level (broadcast) `CompanyGrossWindLossUSD`,
`CompanyExcessUSD`, and `CompanyRecoveryUSD`. No caller in this repository
read the old columns after the function returned, so nothing downstream
depended on their old names or meaning.

**Not changed:** `fl_risk_model/runner.py`'s call sites, the statewide
$17B cap logic (`_apply_industry_season_cap`, verified correct against
Article IV(3) and left untouched), Citizens' representation as one combined
entity, the seasonal-aggregation approximation, the insured wind fraction,
NFIP allocation, portfolio geography, capital/assessment rules, or any
policy-scenario configuration. `fl_risk_model/branches/citizens.py`'s
unreachable `citizens_fhcf_terms_from_cfg_or_csv` dead code is untouched
(documented, not patched, per the brief).

## 2. Verification record corrections

Applied to `docs/earths_future_revision/fhcf_contract_verification.md` and
`docs/earths_future_revision/correction_register.md` (item C3):

- The retention multiples 6.0732 / 7.2878 / 12.1464 are confirmed directly
  on printed page 7 (PDF page 11) of the **2023** Ratemaking Formula Report.
- The payout multiple 11.2368 is confirmed directly on PDF page 3 of the
  **2024** Ratemaking Formula Report with Supplemental Information, in the
  "2023 Contract Year Actual as of 10/24/2023 for Ratemaking" column,
  footnoted "As of 12/31/2023 ... Projected Payout Multiple was 11.2368."
  The 2023 report's own original projection for the same contract year was
  **11.7254** (same table, column "2023 Contract Year Modeled"). Both
  figures pertain to the 2023-2024 Contract Year; `config.py` was not, and
  is not, changed to 11.7254 -- 11.2368 is the better, later, actualized
  figure for the same year, and it is what every archived production run
  was already computed with.
- `fl_risk_model/tests/earths_future/test_fhcf_contract_verification.py::test_company_total_vs_county_split_would_agree_under_aggregate_first_fix`
  (the prior pass's aggregation-invariance test) called the OLD,
  cap-before-scaling formula after manual pre-aggregation. It validated
  aggregation invariance in isolation, not the two fixes together or their
  integration with the statewide cap. It has been superseded by the new
  file's Sections 2-3 (below), which run both fixes together through the
  real production wrapper (`attach_fhcf_terms_for_losses` +
  `apply_fhcf_recovery`) and through `_apply_industry_season_cap`.
- Every place in the prior verification report that stated or implied a
  direction or magnitude for the effect on published headline numbers has
  been left as "not yet known, pending replay" in that document. This
  report's Section 5 (below) reports what the *pilot* (not the ERA5
  baseline) actually showed, scoped explicitly to the pilot.

## 3. Test results

`fl_risk_model/tests/earths_future/`: **46 of 46 pass**
(`python3 -m pytest fl_risk_model/tests/earths_future/ -v`).

- `test_accounting_fixtures.py` (8 tests): one fixture
  (`test_fhcf_shortfall_propagates_into_downstream_default_deficit`) was
  updated because its old loss level (`retention + limit`, i.e. exactly
  `E = K`) no longer demonstrates "loss above saturation flows 1:1 to net
  loss" under the corrected formula (saturation now occurs at
  `E = K/((1+a)*p) > K`, not at `E = K`). The fixture now uses a loss
  comfortably past the true saturation point and asserts the same
  downstream-double-count logic, which is unaffected by which formula
  computes the saturation point. `test_fhcf_coverage_election_may_be_applied_twice...`
  is retained unmodified: it compares current code against a *different,
  not-adopted* hypothesis (dropping the coverage factor entirely) at
  `E = K` exactly, where old and new code agree, so it is unaffected by the
  patch either way and remains a correct record of that earlier,
  not-adopted alternative.
- `test_fhcf_contract_verification.py` (23 tests, substantially rewritten
  from the prior pass): every test that used to assert the *old, defective*
  behavior (e.g. "current code plateaus at 49.5% of Limit at 45% coverage")
  now asserts the *verified* behavior instead (e.g. "current code now
  reaches the full Limit"), per the brief's instruction not to leave
  defect-confirming assertions in place once the defect is fixed. New
  tests added, all against independently hand-derived expected values, not
  copied from a run of the code under test:
  - Formula verified at all three coverage elections across six loss
    regions: below retention, at retention, within the covered layer, at
    `E=K`, at the true saturation point `E=K/((1+a)*p)`, and well past
    saturation.
  - Aggregation verified through the real production wrapper: unequal
    county splits, a case where every county is individually below
    retention but the company sum is not, a Limit-exhausting case with a
    zero-loss row included, row order/count preservation, and exact
    reconciliation (`gross == net + recovery`, no NaNs, no negative values,
    company recovery broadcasts identically across a company's rows).
  - Statewide cap **integration** test: recoveries for two private insurers
    (45% and 90% coverage) and Citizens are computed with the real,
    patched `apply_fhcf_recovery` (not supplied as dummy pre-cap numbers),
    then passed through `_apply_industry_season_cap` below, at, and above a
    cap sized from the fixture's own pre-cap total. Verifies the common
    scaling factor, that scaling never lets any company exceed what
    `apply_fhcf_recovery` had already capped it at, and full reconciliation
    after scaling.
  - Citizens: live fallback path verified equal to the general path;
    multi-county aggregation verified through Citizens' own live path; the
    unreachable dead-code inconsistency (`citizens_fhcf_terms_from_cfg_or_csv`)
    is documented, not exercised via any reachable call path, and left
    unpatched.
- `test_nfip_allocation.py` (4 tests) and `test_scenario_totals.py`
  (3 tests, including the building-code total-loss regression): unaffected
  by the FHCF patch (NFIP and the building-code total-loss fix are
  independent of `apply_fhcf_recovery`); all still pass.
- `test_seasonal_aggregation.py` (2 tests): documents the seasonal/
  occurrence approximation, unaffected by this patch; still pass.

**Passing does not by itself validate the patch against the contract** --
each assertion's expected value is independently computed
(`_verified_recovery`, hand-derived company totals, or a fixture-derived
pre-cap total), not taken from the code under test, which is what makes
these regression tests rather than tautologies.

## 4. Participant/premium reconciliation (statewide cap upper bound)

From `fl_risk_model/data/24fin_fhcf.csv` (the active production input,
loaded via `fl_risk_model.fhcf.normalize_fhcf_terms`):

| Quantity | Value |
|---|---|
| Rows (participants, including Citizens once) | 138 |
| Sum of `FHCFPremium` | $1,435,205,092 |
| Sum of `RetentionUSD` (industry aggregate retention implied by this snapshot) | $9,030,692,592 |
| Sum of `LimitUSD` = Premium x 11.2368 (aggregate nominal company limits) | $16,127,112,577.79 |
| Citizens' own row (NAIC 10064, 90% coverage) | 1 row, Limit $4,568,240,047.68 |
| Modeled statewide cap (`FHCF_SEASON_CAP`) | $17,000,000,000 |

**16,127,112,577.79 < 17,000,000,000.** Once the aggregation fix applies
each company's Limit exactly once (rather than once per county row), the
sum of every participant's own maximum possible recovery is **mathematically
incapable of exceeding the $17B statewide cap under this premium snapshot,
regardless of loss severity.** This is not an estimate from a small sample;
it follows directly from the data (every term is nonnegative, so the
company-level recoveries entering `_apply_industry_season_cap` can never
sum above $16.127B). Consequently, **`fhcf_shortfall_usd` and
`fhcf_cap_binding` are structurally zero after the aggregation fix**, for
any input that draws its terms from this file. This was demonstrated, not
merely asserted: the local pilot (Section 5) shows exactly 0.00 for both
diagnostics across all 2,800 pilot iterations (7 scenarios x 4 code
variants x 100 iterations), including scenarios where the pre-patch code
showed the cap binding in up to 93% of iterations (Double Great Miami).

This does **not** mean company-level FHCF capacity is unlimited, and it
does not mean FHCF recoveries stop affecting downstream finances -- company
Limits still bind constantly (that is exactly what the pilot's large swings
in FHCF recovery, defaults, and residual financing requirement come from).
It means the *industry-wide, cross-company* $17B backstop specifically does
not bind under the current 138-participant premium snapshot once recoveries
are correctly capped per company. This bound would change if the
participant set, premiums, or the $17B figure changed; it was not retuned
to produce this result, and no configuration was changed to reproduce or
avoid the old nonzero shortfall.

**Figure/table plan implication (not implemented in this pass):** any
manuscript figure or table that previously showed FHCF shortfall as a
nonzero, tail-sensitive quantity should show FHCF total recovery
(`fhcf_total_precap_usd`, equivalently `fhcf_total_postcap_usd` once
`fhcf_shortfall_usd` is zero) or per-company Limit exhaustion instead, once
the full rerun (Section 6) is complete. `fhcf_cap_binding`/
`fhcf_shortfall_usd` remain valid, correctly-defined columns; they are just
expected to read zero given this premium snapshot, which is a finding, not
a broken diagnostic.

## 5. Local pilot (not a cluster run)

**Cluster access.** TCP connectivity to `login.sherlock.stanford.edu:22` is
reachable from this environment (confirmed with a bare TCP check; no
authentication was attempted). Stanford Sherlock requires interactive Duo
two-factor approval on the account holder's device, which this session
cannot perform. No SSH login, no job submission, and no cluster path was
invented. The ERA5 baseline (10,000 synthetic seasons) requires exactly the
proprietary WindRiskTech/Emanuel hazard cache this environment does not
have (`fl_risk_model/data/hazard/emanuel/` and `.../gori_data/` are
`.gitkeep` placeholders here, same as in the original checkout) and is
**not included** in this pilot.

**What was run instead.** The eight historical/sequential scenarios use
real, git-tracked, non-proprietary county-level wind and flood damage
inputs (`fl_risk_model/data/hazard/historical_events/`), which let the
actual production pipeline
(`fl_risk_model.mc_run_events.run_one_iteration` ->
`fl_risk_model.runner.run_one_scenario` -> `apply_fhcf_recovery`) run
end to end with real inputs, locally, in seconds per configuration. Two
additional proprietary-but-locally-present inputs
(`fl_risk_model/data/FL HO Market Share Report_6.10.25.xlsx` and
`.../20250805 FL Surplus Capital, Group v Entity.xlsx`) were required and
were present, gitignored, in the original checkout
(`systemic_insurance_risk_fl_pub`); they were copied read-only into this
worktree (same treatment as the historical-scenario archives copied in the
prior pass) and are not committed to git.

**Design.** `scripts/earths_future_revision/fhcf_pilot_isolate_fixes.py`
runs, for each of 7 scenarios (Great Miami, Andrew, Lake Okeechobee, Irma,
Great-Miami-then-Andrew, Double Great Miami, Double Irma -- all 8 historical
scenarios except Andrew-then-Great-Miami, which is not part of the
manuscript's stress-test set), 4 monkeypatched variants of
`apply_fhcf_recovery`, each through the identical, unmodified
`run_scenario_mc` entry point, seed 42, 100 iterations each (2,800 total):

- `old_both_bugs`: the exact pre-patch production formula.
- `formula_fix_only`: cap-after-scaling, still row-wise (isolates defect 1).
- `aggregation_fix_only`: company-aggregated first, cap-before-scaling
  (isolates defect 2).
- `both_fixed`: the current, patched `fl_risk_model.fhcf.apply_fhcf_recovery`.

**Matching draws confirmed, not assumed.** `total_damage_usd`,
`wind_total_usd`, `water_total_usd`, and the per-season sampled `wind_shares`
were compared row-for-row between the `old_both_bugs` and `both_fixed`
outputs for Great Miami and found identical (string-exact), confirming the
same seed produces the same upstream hazard/exposure/wind-share draws
regardless of which FHCF variant is monkeypatched in -- the four variants
differ only in the FHCF calculation. `nfip_borrowed_usd` was likewise
identical, confirming the flood/NFIP side is untouched by this patch.
Outputs, the full comparison table, and a manifest with a git revision tag
and SHA-256 (16-hex-char) checksums of every hazard and premium/surplus
input are saved under `results/earths_future_revision/fhcf_pilot/`.

**Results** (means across 100 iterations; USD unless noted; "RFR" =
residual financing requirement = FIGA residual + Citizens residual + NFIP
borrowing):

| Scenario | Variant | FHCF precap | FHCF shortfall | Cap binds | Private recov. | Citizens recov. | Defaults | RFR |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Great Miami | old_both_bugs | 15.66B | 1.93B | 49% | 10.19B | 3.55B | 18.16 | 23.93B |
| Great Miami | formula_fix_only | 15.97B | 2.16B | 49% | 10.29B | 3.51B | 18.18 | 23.99B |
| Great Miami | aggregation_fix_only | 14.27B | **0.00** | **0%** | 9.98B | 4.29B | 17.63 | 22.75B |
| Great Miami | **both_fixed** | 14.68B | **0.00** | **0%** | 10.36B | 4.32B | 17.60 | 22.67B |
| Double Great Miami | old_both_bugs | 24.89B | 8.18B | 91% | 12.15B | 4.57B | 22.15 | 57.48B |
| Double Great Miami | **both_fixed** | 15.17B | **0.00** | **0%** | 10.61B | 4.56B | 22.12 | 58.16B |
| Great Miami then Andrew | old_both_bugs | 22.22B | 5.74B | 83% | 12.09B | 4.39B | 21.35 | 45.30B |
| Great Miami then Andrew | **both_fixed** | 15.13B | **0.00** | **0%** | 10.59B | 4.54B | 21.45 | 45.78B |
| Andrew | old_both_bugs | 12.85B | 0.0016B | 1% | 8.85B | 3.99B | 15.83 | 10.85B |
| Andrew | **both_fixed** | 13.91B | **0.00** | **0%** | 9.65B | 4.26B | 14.93 | 10.22B |
| Lake Okeechobee | old_both_bugs | 0.330B | 0.00 | 0% | 0.327B | 0.0028B | 13.78 | 18.37B |
| Lake Okeechobee | **both_fixed** | 7.34B | 0.00 | 0% | 6.63B | 0.705B | 9.99 | 15.10B |
| Irma | old_both_bugs | 0.0055B | 0.00 | 0% | 0.0055B | 0.00 | 5.19 | 0.145B |
| Irma | **both_fixed** | 0.846B | 0.00 | 0% | 0.846B | 0.00 | 2.97 | 0.0338B |
| Double Irma | old_both_bugs | 0.0683B | 0.00 | 0% | 0.0683B | 0.00 | 11.81 | 3.08B |
| Double Irma | **both_fixed** | 4.60B | 0.00 | 0% | 4.60B | 0.00 | 7.39 | 1.76B |

Full table (all 4 variants x 7 scenarios) in
`results/earths_future_revision/fhcf_pilot/pilot_comparison_summary.csv`.

**Both defects contribute, in different directions, depending on how a
scenario's losses are distributed across counties and companies:**

- **Great Miami, Great-Miami-then-Andrew, Double Great Miami** (wind-
  concentrated, high-severity): pre-patch, the statewide cap bound in
  49-91% of iterations and FHCF shortfall was large (1.9-8.2B). Applying
  *only* the formula-order fix (without the aggregation fix) made the
  precap demand and the shortfall **larger**, not smaller (e.g. Great Miami
  shortfall 1.93B -> 2.16B), because it let companies reach higher
  (correctly-scaled) recoveries on individual, already-over-counted county
  rows. Only once the aggregation fix is also applied does each company's
  demand correctly cap at its own Limit, precap demand drops below the
  $17B aggregate ceiling, and the shortfall (and cap-binding rate) collapse
  to exactly zero. FIGA/Citizens residual deficits and defaults fall
  correspondingly once true FHCF recovery is available to companies.
- **Lake Okeechobee, Irma, Double Irma** (flood-dominated / diffuse wind,
  lower severity per company): the OPPOSITE failure mode dominates.
  Pre-patch, most companies' *individual county* losses never exceeded
  Retention even though their *summed* loss did, so FHCF recovery was
  drastically **under-stated** (e.g. Lake Okeechobee private recovery
  0.33B pre-patch vs. 6.63B patched -- a 20x increase; Irma 5.5M vs. 846M,
  a roughly 150x increase). This directly reduced defaults (13.78 -> 9.99
  for Lake Okeechobee; 5.19 -> 2.97 for Irma) and RFR, because companies
  that were wrongly denied FHCF recovery pre-patch are correctly recovering
  it now.
- **Andrew** sits in between: a modest increase in FHCF recovery, a modest
  decrease in defaults and RFR.

**Gross losses and NFIP financing are unchanged in every scenario**
(confirmed identical, not merely close, across all four variants), exactly
as expected for a patch confined to `apply_fhcf_recovery`.

**Scope of this finding.** This is a pilot on 7 historical/sequential
scenarios with real inputs, not the ERA5 probabilistic baseline. It shows
the patch behaves exactly as the contract-verified formula and the
company-limit reconciliation (Section 4) predict, in both directions
(over- and under-recovery pre-patch), and that it does not disturb
gross losses or NFIP. **It is not evidence, by itself, about the direction
or magnitude of any headline manuscript number** (which depends on the
10,000-season ERA5 distribution, dominated by moderate seasons very
different from these severity-selected historical footprints). No such
extrapolation is made here.

## 6. Full rerun manifest

Every output below was generated with the pre-patch `apply_fhcf_recovery`.
Reconciled against the actual repository inventory as of this commit (not
against any prior report's stated counts, several of which were imprecise
-- see corrections below).

**Corrections to prior counts:** the GCM x building-code sweep is **13**
levels x 5 GCMs = **65** directories (matching the manuscript's stated 13
levels), not "11-level" as stated in the prior pass's `report.md` and
`data_inventory.md`. The w##f## labels observed are: w00f00, w20f13,
w30f20, w40f27, w50f33, w60f40, w70f47, w80f53, w90f60, w100f67, w110f70,
w120f80, w130f90 -- 13 distinct settings, confirmed by directory listing,
not inferred from the manuscript's prose count.

| Output group | Count | Path pattern | FHCF patch affects it? | Rerun scope |
|---|--:|---|---|---|
| ERA5 baseline | 1 | `results/mc_runs/emanuel_era5_baseline_*` | Yes | Full replay |
| ERA5 policy scenarios | 3 | `results/mc_runs/emanuel_era5_{market_exit_moderate,penetration_major,building_codes_major}_*` | Yes (FHCF/FIGA/Citizens columns); building-code total-loss fix (register item C5) is independent and already correct | Full replay |
| GCM baseline (5 GCM x 5 period: 20thcal, ssp245cal, ssp245_2cal, ssp585cal, ssp585_2cal) | 25 | `results/mc_runs/emanuel_{canesm,cnrm6,ecearth6,ipsl6,miroc6}_{period}_baseline_*` | Yes | Full replay, all 25 |
| GCM x building-code sweep (13 levels x 5 GCM) | 65 | `results/mc_runs/emanuel_{gcm}_ssp245cal_buildingcode_w##f##_*` | Yes | Full replay, all 65 |
| Historical/sequential scenarios | 8 | `{great_miami,andrew,andrew_then_gm,gm_then_andrew,double_gm,lake_okeechobee,irma,double_irma}_*` (gitignored; copied read-only into this worktree from the original checkout) | Yes | Full replay, all 8 (7 already piloted above; add andrew_then_gm) |
| Insured-fraction sweep (f=0.1-0.5) | 5 | `results/mc_runs/insured_frac_sensitivity_combined/iterations_frac_*.csv` (gitignored; copied) | Yes | Full replay, all 5 fractions, same seed(s) as archived |
| NFIP allocation county-rate table | -- | `results/earths_future_revision/nfip_allocation/` | No (independent of FHCF) | No rerun needed |
| Variance decomposition (Table S7, removed from manuscript) | 2 | `results/mc_runs/variance_{nested_300x50,fixed_params}_*` (gitignored; copied) | Yes, but table already removed from the manuscript per the prior pass | **Not rerun** -- explicitly out of scope; do not restore Table S7 or expand this design |

**Post-processing-only vs. replay-required**, restated: `scripts/earths_future_revision/section5_return_periods.py`,
`section6_decomposition.py`, `section7_insured_fraction.py`,
`section8_historical_and_variance.py`, and `section9_climate_policy_tables.py`
do not need code changes for the FHCF patch -- they operate on whatever
`iterations.csv` they are pointed at. Every table and figure they currently
produce (Table 1 and its bootstrap, the severity-bin decomposition, SI
Table S6 elasticities, SI Table S3, SI Table S7's audit, SI Table S4) was
computed from **pre-patch** archives and must be regenerated by re-running
these unchanged scripts against the post-patch `iterations.csv` files once
the full rerun (below) completes. **No manuscript numbers currently in the
repository (from either this task or the prior pass) should be treated as
final.**

**Not previously planned, and not launched:** a new sensitivity sweep
isolating the FHCF patch's effect across the full climate/policy grid, or
any expansion of the participant/premium reconciliation into a full
company-level audit. Per the brief, these are noted as possible future
work, not started.

**Input paths to prevent old/corrected mixing.** All post-patch reruns
should write to a new path segment, e.g.
`results/mc_runs_fhcf_patched/<same_directory_name>/`, never overwriting
the existing `results/mc_runs/<name>/` directories (which remain the
exact-reproduction record for the submitted manuscript and for the prior
pass's R1/C1/C5 corrections). `scripts/earths_future_revision/*.py` accept
an explicit `--iterations`/`--frac-dir`/input path argument in every case
except `section9_climate_policy_tables.py`'s hardcoded `SCENARIO_DIRS` and
`section8_historical_and_variance.py`'s hardcoded `HISTORICAL_DIRS`, which
point at the March-2026-dated archived run directories by name; these two
dicts will need updating to the new post-patch directory names before
re-running (this is exactly the "some post-processing scripts still point
to archived March run directories" issue flagged in the brief -- confirmed
by inspection, not assumed).

**SSP5-8.5.** Both pathways (SSP2-4.5 and SSP5-8.5) are already included in
the GCM baseline count above (`ssp245cal`/`ssp245_2cal` and
`ssp585cal`/`ssp585_2cal` for each of the 5 GCMs); no separate SSP5-8.5
rerun scope is needed beyond what is already listed. Per the prior pass's
editorial decision, SSP5-8.5 stays in the SI, not the main-text climate
figure.

## 7. Runtime estimate

The local pilot (100 iterations, 1 scenario, 1 variant) completed in a few
seconds of wall time on this machine; the full 2,800-iteration, 7-scenario,
4-variant pilot completed in approximately 10 minutes. This is not
representative of the ERA5 baseline or GCM runs, which sample from a
10,000-year synthetic catalog with substantially larger per-iteration
hazard-processing cost (county-level wind field and impact computation
across the full Florida exposure book, not just a fixed historical
footprint). Per `results/README.md`, "a full baseline run of 10,000 years
takes roughly 4-8 hours on a single compute node" under the existing
production pipeline; the FHCF patch adds a groupby/merge over companies
per iteration, which is computationally minor relative to hazard
processing and is not expected to materially change that per-run estimate.
At that rate: 1 ERA5 baseline + 3 ERA5 policy scenarios + 25 GCM baseline
runs + 65 GCM building-code runs + 8 historical scenarios (each far
cheaper, as this pilot showed) + 5 insured-fraction sweep points is
approximately **94 ten-thousand-season-scale runs**, i.e. on the order of
**400-750 compute-node-hours** if run serially, before post-processing.
This is an order-of-magnitude planning estimate carried over from the
existing README figure, not a new benchmark measured in this pass (no
production-scale run was executed here, per the brief).

## 8. Is the patch ready for the full production replay?

**Yes, with one explicit caveat.** The formula-order and aggregation fixes
are both source-verified (Section 2 and
`fhcf_contract_verification.md`), both are covered by regression tests
against independently derived expected values (Section 3, 46/46 passing),
their combined production path (real company terms, real multi-county
splits, the real statewide-cap integration) is validated, and the
participant/premium reconciliation (Section 4) explains and confirms the
resulting structural zero for `fhcf_shortfall_usd`/`fhcf_cap_binding` from
first principles, not just from the pilot's output. The paired local pilot
(Section 5) shows the patch changing gross FHCF recovery, defaults, and
residual financing requirement in both directions depending on scenario
structure, exactly as the contract text and the reconciliation predict, and
leaves gross losses and NFIP untouched as required.

**The caveat:** the exact vintage/date of `FHCFPremium` in
`fl_risk_model/data/24fin_fhcf.csv` (as distinct from the industry-wide
retention/payout multipliers, which are dated and confirmed) is still not
independently confirmed against a dated source filing -- see
`fhcf_contract_verification.md` Section 6, unresolved question 1. This does
not block applying the patch (the *formula* is correct regardless of the
premium vintage), but it means the *company-level* Retention/Limit dollar
values used throughout are provisional in the same way they already were
before this task. Tracing this fully would require either the data
preparer's notes or a matching FHCF exposure/premium bulletin, which this
session does not have access to; it is recorded as an open factual question,
not asked of the author as a choice.

**Recommended next action:** freeze this patch, then execute Section 6's
full rerun manifest exactly as scoped (no new sensitivities, no Table S7
restoration, no hazard regeneration), starting with the small cluster pilot
already specified in `fhcf_contract_verification.md` Section 7 (ERA5
baseline + Great Miami, identical inputs/seeds) once cluster access with
working Duo authentication is available to whoever executes it. That step
was not launched here.
