# FHCF contract verification

Scope: verify the FHCF reimbursement formula, input definitions, Citizens
fallback, and statewide cap against primary FHCF/SBA documents. No
production financial rules were changed and no production analyses were
rerun in this task. Branch `earths-future/corrections-and-sensitivity`,
worktree `systemic_insurance_risk_fl_earths_future`, base commit
`d4a4fbb` (verified clean at the start of this task; no local changes
existed to preserve beyond that commit).

This brief supersedes the earlier broad handoff's correction-register item
C3 wherever they conflict. C3's own text already flagged the coverage
double-application question as **unresolved, author decision needed**, and
explicitly declined to change production code; nothing here reopens an
accepted modeling choice. What follows replaces C3's tentative reading with
a source-backed one and adds a second, previously undocumented defect (the
per-county aggregation issue) that C3 did not raise.

## 1. Direct conclusion

**The current formula does not match the applicable contract, and the
mismatch is demonstrable, not merely suspected.** Two independent defects
were found:

1. **Cap placement (formula order).** Article IV(1) of the FHCF 2023-2024
   Reimbursement Contract defines reimbursement as `(E x p) + a x (E x p)`,
   i.e. `(1+a) x p x E`, **"the total of which shall not exceed the
   Company's Limit."** The cap therefore applies to the fully scaled
   reimbursement amount. The current implementation
   (`fl_risk_model/fhcf.py::apply_fhcf_recovery`) instead computes
   `RecoverableUSD = min(ExcessUSD, LimitUSD)` and only then multiplies by
   `p` and `(1+a)`, i.e. it caps `E` before scaling, not the scaled result.
   **Confidence: high.** The contract text is unambiguous on this point
   (quoted below verbatim), and the two formulas provably agree everywhere
   they were likely to be checked by inspection or by low-severity tests
   (`E <= K`, which is roughly the bulk of the loss distribution for most
   companies) and diverge only once a company's excess-over-retention
   exceeds its own Limit -- exactly the tail region a manual spot check is
   least likely to land in.

2. **Aggregation level.** Article V(26) and V(28) define Retention and
   Ultimate Net Loss once per Covered Event for the Company's entire book
   of Covered Policies. The active call path
   (`fl_risk_model/runner.py` -> `attach_fhcf_terms_for_losses` ->
   `apply_fhcf_recovery`) passes company x county loss rows into
   `apply_fhcf_recovery`, which merges one company-level Retention/Limit
   onto every county row and then computes excess and the capped
   recoverable amount **per row**, not once for the company's summed loss.
   **Confidence: high.** This follows directly from reading the merge and
   is confirmed by a same-total, different-granularity test using the
   actual production wrapper (Section 4 below); it does not depend on any
   contract interpretation.

**What is not being reported as an error:** the coverage percentage `p`
itself, applied to losses above retention, is squarely required by Article
IV(1) and must not be removed, dropped, or replaced by `min(E,K)*(1+a)`
without the multiplication by `p`. The correction register's earlier
inverse-retention-multiple argument for suspecting a double application of
`p` is a red herring on its own (nothing in the retained multiplier
structure implies `p` should be removed); the actual, evidenced defect is
solely about *where* `K` is applied.

Neither defect is triggered under the same conditions, and neither has a
knowable net effect on any previously reported headline number without a
rerun (Section 7).

## 2. Source table

| Model term / field | Meaning in code | Contract definition | Year / source | Match? |
|---|---|---|---|---|
| `FHCFPremium` (`24fin_fhcf.csv`) | Per-company premium used to derive Retention and Limit | "Reimbursement Premium or Premium": amount paid by the Company, determined by multiplying each $1,000 of reported insured value by the Premium Formula rate | Art. V(24), p.7, FHCF-2023K | Consistent with usage; the exact reported-insured-value date (June 30 of the Contract Year, Art. IX(2), p.14) was not cross-checked against the CSV's own vintage (see Section 6, unresolved question 1) |
| `CoveragePct` / `CoveragePct_norm` | Company's elected coverage level, snapped to {45,75,90} | "Coverage Level": 90%, 75%, or 45%, elected under Art. XXI or deemed under Art. III(3) | Art. V(11), p.6 | Matches |
| `RetentionUSD` = `FHCFPremium x FHCF_RET_MULTIPLES[cov]` | Company retention in USD | "Retention" = Retention Multiple x Company's Reimbursement Premium (Art. V(26)(c), p.8-9); "Retention Multiple" = a base value adjusted to 100%/120%/200% of the 90%-level amount for the 90%/75%/45% coverage elections (Art. V(27), p.8) | Art. V(26)-(27), p.8-9, FHCF-2023K | Matches. `FHCF_RET_MULTIPLES = {90: 6.0732, 75: 7.2878, 45: 12.1464}` reproduce the 100%/120%/200% ratios exactly (7.2878/6.0732 = 1.2000; 12.1464/6.0732 = 2.0000) |
| `LimitUSD` = `FHCFPremium x FHCF_PAYOUT_MULTIPLE` | Company's maximum recoverable amount | "Limit" = maximum amount a Company may recover, calculated by multiplying the Company's Reimbursement Premium by the Payout Multiple | Art. V(17), p.6 | Matches: Limit is *not* coverage-level-scaled in its own definition, confirming `p` is applied elsewhere (Art. IV(1)), not folded into `K` |
| `FHCF_PAYOUT_MULTIPLE = 11.2368` | Single payout multiple, same for all coverage levels | "Payout Multiple" = single-season industry Claims-Paying Capacity / total aggregate industry Reimbursement Premium for the Contract Year, one value for all companies | Art. V(21), p.7 | Matches; a single statewide multiple, not company- or coverage-level-specific |
| `FHCF_LAE_FACTOR = 1.10` (i.e. `(1+a)`, `a=0.10`) | Multiplicative loss-adjustment-expense factor | "Loss Adjustment Expense Allowance" = 10% of reimbursed Losses under Art. IV, **included in, and not in addition to, the Limit** | Art. V(19)(a)-(b), p.7 | The 10% rate matches. "Included in ... the Limit" confirms LAE is part of what `K` caps, i.e. it belongs inside the `min(...,K)`, consistent with the verified formula and inconsistent with adding LAE on top of an already-capped `min(E,K)` |
| Formula: `recovery = (1+a) x p x min(E,K)` (current code) | -- | Not supported: Art. IV(1) caps the *total* of the coverage-scaled excess plus LAE at the Limit, not the raw excess | Art. IV(1), p.3 | **Mismatch (Section 3)** |
| Formula: `recovery = min((1+a) x p x E, K)` (verified) | -- | Supported | Art. IV(1), p.3 | Verified |
| Aggregation: per-county application of company Retention/Limit (current code) | -- | Not supported: Retention and Ultimate Net Loss apply once, per Covered Event, to the Company's full book | Art. V(26)(a)-(b), V(28)(a), p.8-9 | **Mismatch (Section 4)** |
| `FHCF_SEASON_CAP = $17,000,000,000` | Modeled statewide seasonal cap | "$17.000 billion" FHCF limit level assumed in developing 2023 Contract Year rates | 2023 Ratemaking Formula Report, cover letter, Mar. 17, 2023, p.1 (of that report) | Matches |
| Statewide cap: pro-rata scaling of every company (and Citizens) when precap recoveries exceed the cap | `_apply_industry_season_cap` in `runner.py` | "the SBA shall reduce the projected payout factors or multiples for determining each participating insurer's projected payout **uniformly among all insurers**" | Art. IV(3), p.4 | Matches -- no defect found here |
| Statewide cap applied once to combined Private + Citizens | Single call combining `private_precap` and `citizens_precap` | Citizens' coastal account and personal-lines/commercial-lines account are each treated "as if it were... a separate participating insurer with its own... Retention, and Ultimate Net Loss," i.e. part of the same statewide pool, not a separate cap | Art. V(8), p.5 | Matches -- no defect found here |
| Citizens general/live-fallback path: `Limit = Premium x Payout Multiple` (no coverage factor) | `_citizens_terms_fallback_row` + `normalize_fhcf_terms`, `runner.py` lines ~768-817 | Same as Art. V(17) | Art. V(17), p.6 | Matches |
| Citizens dead-code path: `CITIZENS_FHCF_LIMIT_USD = Premium x PayoutMultiplier x CoveragePct` | `config.py` line 138, read only by `citizens_fhcf_terms_from_cfg_or_csv` in `branches/citizens.py` | Not supported (same reasoning as the general Limit mismatch above, in the opposite direction: this formula pre-applies a coverage factor Art. V(17) does not include) | Art. V(17), p.6 | **Inconsistent, but unreachable (Section 4)** |
| Contract year of the inputs | `24fin_fhcf.csv`, `config.py` comment "2023-24 contract" | -- | FHCF-2023K, "Coverage Effective: June 1, 2023" (Contract Year June 1 2023 - May 31 2024, Art. III(1)) | The numeric multipliers in `config.py` (6.0732, 7.2878, 12.1464, 11.2368) match the **2023 Ratemaking Formula Report** (presented to the SBA March 23, 2023, for the FHCF 2023 Contract Year), **not** a 2024-2025 vintage despite the `24fin_fhcf.csv` filename. See Section 6. |

Primary documents used:
- 2023-2024 Reimbursement Contract ("FHCF-2023K," Rule 19-8.010 F.A.C.), fetched from
  https://fhcf.sbafla.com/media/sr4invnv/2023-reimbursement-contract-final-7-18-2022.pdf
  (36 pages; read in full).
- Florida Hurricane Catastrophe Fund 2023 Ratemaking Formula Report, Paragon
  Strategic Solutions Inc., presented to the SBA March 23, 2023, cover
  letter dated March 17, 2023, fetched from
  https://fhcf.sbafla.com/media/pfihme0c/20230323_2023ratemakingformulareport.pdf
  (cover letter and title page read; the numeric retention-multiple table
  itself, cited via a secondary web search result quoting this report, was
  not independently re-opened page-by-page inside the 100+ page report --
  see Section 6, unresolved question 2).

The template contract (FHCF-2023K) itself does not state the final dollar
values of the Retention Multiple or Payout Multiple -- these are calculated
annually by the SBA's actuary and published in the Ratemaking Formula
Report; the contract only states the *method* (Art. V(21), V(27)).

## 3. Verified formula and difference from current implementation

Let `E = max(UNL - Retention, 0)` (loss in excess of retention for the
Covered Event), `p` the coverage election, `a = 0.10`, `K` the Company's
Limit.

**Current code** (`fl_risk_model/fhcf.py::apply_fhcf_recovery`):
```
ExcessUSD      = max(Gross - Retention, 0)
RecoverableUSD = min(ExcessUSD, K)
RecoveryUSD    = RecoverableUSD * p * (1 + a)
             =  min(E, K) * p * (1 + a)
```

**Verified formula** (Article IV(1)):
```
RecoveryUSD = min( E * p * (1 + a),  K )
```

**Plain-language difference.** Both formulas start by scaling the excess
loss by the coverage percentage and adding the 10% expense allowance. They
differ only in when the Limit is applied. The contract applies the Limit to
the *final, scaled* dollar amount. The current code applies it to the *raw
excess loss* before scaling. Because `p <= 0.90` and `a = 0.10`, the factor
`p*(1+a)` is at most `0.99`, so the two formulas produce identical results
whenever `E <= K` (including exactly `E == K`). They diverge only once a
Company's excess-over-retention loss exceeds its own Limit -- at that point
the contract says the Company should still receive up to its full Limit
`K`, while the current code permanently plateaus at `p*(1+a)*K`, which is
strictly less than `K`:

| Coverage election | Current code's effective ceiling, once `E` is very large | Contract's ceiling |
|---|---|---|
| 90% | `0.99 K` (1% short) | `K` |
| 75% | `0.825 K` (17.5% short) | `K` |
| 45% | `0.495 K` (50.5% short) | `K` |

This is a **permanent, coverage-election-dependent under-recovery** in the
current code for any company whose losses reach deep enough into the
covered layer, worst at low coverage elections. It is not a rounding
artifact: at 45% coverage and large losses, the current code recovers
essentially half of what the contract entitles the company to.

## 4. Numerical examples and executed verification-test results

All tests below use `Premium = $10,000,000` (held fixed across coverage
levels to isolate the formula's behavior, matching the existing test
suite's convention), `K = Premium x 11.2368 = $112,368,000` (same for all
coverage levels, per Art. V(17)/V(21)).

New file:
`fl_risk_model/tests/earths_future/test_fhcf_contract_verification.py` (23
tests, all currently passing -- they assert what the current code
*actually does*, both where it is correct and where it is wrong; passing
is not validation of correctness where the assertions target the confirmed
defect).

| Test | Loss region | Current code vs. verified formula | Result |
|---|---|---|---|
| `test_below_retention_current_code_matches_contract` | `Gross < Retention` | Both give 0 | Agree (pass) |
| `test_at_retention_current_code_matches_contract` | `Gross == Retention` | Both give 0 | Agree (pass) |
| `test_in_covered_layer_current_code_matches_contract` | `E = $30M < K` | Both give `p*(1+a)*E` | Agree (pass) |
| `test_at_e_equals_limit_boundary_still_agrees` | `E == K` exactly | Both give `p*(1+a)*K` | Agree (pass) -- last point of agreement |
| `test_far_above_limit_current_code_UNDER_recovers` | `E = 3K` | Contract: `K`. Current code: `p*(1+a)*K` | **Diverge** (pass -- confirms the defect exists as specified) |
| `test_shortfall_is_largest_at_the_45_percent_election` | `E = 3K`, 45% vs 90% | Current code recovers 49.5% of `K` at 45% coverage vs. 99% at 90% | Confirms coverage-dependence of the shortfall |
| `test_company_total_vs_county_split_diverge_under_current_code` | Same total loss, 1 row vs. 3 county rows, real `attach_fhcf_terms_for_losses` + `apply_fhcf_recovery` wrapper | One row: correctly plateaus at `p*(1+a)*K`. Three-county split of the *identical total*: recovers `3 x p*(1+a)*K`, i.e. 3x the Company's actual Limit | **Diverge** (pass -- confirms the aggregation-level defect) |
| `test_company_total_vs_county_split_would_agree_under_aggregate_first_fix` | Same fixture, but `GrossWindLossUSD` summed to company level before calling `apply_fhcf_recovery` | Agree exactly | Confirms the proposed fix (Section 5) resolves the discrepancy |
| `test_statewide_cap_below_capacity_no_scaling`, `..._exactly_at_capacity`, `..._above_capacity_prorates_private_and_citizens_together` | 2 insurers + Citizens, below/at/above $17B | Pro-rata scaling applied identically to Private and Citizens as one pool; reconciliation (`Net + Recovery == Gross`) holds row-by-row | No defect found; matches Art. IV(3) |
| `test_live_citizens_fallback_matches_general_path_formula` | -- | Live runner.py Citizens fallback (`_citizens_terms_fallback_row`) gives the same Limit as the general company path | No defect in the reachable Citizens code |
| `test_dead_cfg_helper_would_apply_coverage_factor_twice_if_ever_called` | -- | The separate, unreachable `citizens_fhcf_terms_from_cfg_or_csv` function *would* multiply Limit by CoveragePct if it were ever called | Confirms the inconsistency is real but confined to dead code |

**Existing tests that are insufficient or misleading, given the above:**

- `fl_risk_model/tests/earths_future/test_accounting_fixtures.py::test_fhcf_coverage_election_may_be_applied_twice_between_retention_and_limit`
  is **retained unmodified** as a record of the prior, evidence-lighter
  suspicion (it only showed the two formulas *can* diverge and by how much
  at one loss level; it did not establish which formula the contract
  requires). It should not be read as validating or refuting the current
  implementation on its own -- the new file above supersedes it for that
  purpose.
- No existing test in the repository exercises the statewide cap with more
  than one private insurer, so none of them could have caught or ruled out
  a cross-company allocation defect in `_apply_industry_season_cap`. The
  new `test_statewide_cap_above_capacity_prorates_private_and_citizens_together`
  fills this gap and finds no defect.
- No existing test compares company-total vs. company x county
  representations of the same loss, so the aggregation-level defect
  (Section 1, item 2) was previously untested in either direction.

## 5. Smallest proposed production patch

Both changes are confined to `fl_risk_model/fhcf.py::apply_fhcf_recovery`.
Neither changes any accepted modeling choice (seasonal aggregation across
events/years, the insured-fraction assumption, NFIP allocation, capital
support rules, or any policy-scenario configuration remain untouched). This
is a specification for review, **not yet applied**.

**Patch 1 -- formula order** (resolves Section 1, item 1):
```python
# Current:
df["RecoverableUSD"] = df[["ExcessUSD", "LimitUSD"]].min(axis=1).fillna(0.0)
coverage_frac = (df["CoveragePct_norm"].fillna(0.0) / 100.0).clip(lower=0.0, upper=1.0)
df["RecoveryUSD"] = (df["RecoverableUSD"] * coverage_frac * float(FHCF_LAE_FACTOR)).astype(float)

# Proposed:
coverage_frac = (df["CoveragePct_norm"].fillna(0.0) / 100.0).clip(lower=0.0, upper=1.0)
df["ScaledExcessUSD"] = df["ExcessUSD"] * coverage_frac * float(FHCF_LAE_FACTOR)
df["RecoveryUSD"] = df[["ScaledExcessUSD", "LimitUSD"]].min(axis=1).fillna(0.0)
```
`RecoverableUSD` becomes redundant under this ordering (or can be kept as
`ExcessUSD` unscaled, renamed, for diagnostic continuity -- reviewer's
choice). No other column, argument, or caller needs to change.

**Patch 2 -- aggregation level** (resolves Section 1, item 2): sum
`GrossWindLossUSD` to one row per `Company` before calling
`apply_fhcf_recovery`, then re-join the company-level `RecoveryUSD` back
onto the original county rows (e.g. pro-rated by each county's share of the
company's total gross loss, so downstream code that expects
company x county granularity for `NetWindUSD` continues to work
unchanged). Concretely, in `fl_risk_model/runner.py` immediately before the
existing calls at lines ~724 and ~814:
```python
priv_company_totals = priv_gross.groupby("Company", as_index=False)["GrossWindLossUSD"].sum()
private_company_precap = apply_fhcf_recovery(priv_company_totals, terms_for_private)
# re-attach the company-level RecoveryUSD/NetWindUSD ratio to each county row
# of priv_gross, e.g. via a per-company recovery-rate merge, before continuing
# to _apply_industry_season_cap unchanged.
```
The exact re-allocation-to-county mechanics are an implementation detail
for the reviewer (proportional to county gross loss is the natural choice
and preserves the existing company x county output granularity used
downstream); the load-bearing requirement is that `ExcessUSD` and
`RecoverableUSD`/`RecoveryUSD` be computed once per `Company` per season,
not once per county row. `attach_fhcf_terms_for_losses` does not need to
change (it already returns one row per Company).

Both patches together are what `test_company_total_vs_county_split_would_agree_under_aggregate_first_fix`
already demonstrates is sufficient and internally consistent.

**Not proposed, and out of scope:** event-level retentions, reinstatement
logic, the two-largest-events-at-full-retention / other-events-at-one-third
rule (Art. V(26)(b)), or any change to the seasonal-aggregation
approximation itself (Section 6 below explains why these remain
appropriately out of scope for this task).

## 6. Unresolved source questions

1. **`FHCFPremium` reporting date in `24fin_fhcf.csv`.** Article IX(2)
   ties the Reimbursement Premium to insured values reported as of June 30
   of the Contract Year. The exact as-of date and Contract Year of the
   values in `fl_risk_model/data/24fin_fhcf.csv` and
   `fhcf_terms_keyed.csv` were not independently confirmed against a
   dated source filing (the CSVs themselves carry no date field). This is
   a factual question about a specific data snapshot that only the data's
   preparer or a matching FHCF exposure/premium bulletin can answer; it is
   not a question this verification can resolve by re-reading the contract
   template.
2. **Exact page/table location of the numeric Retention Multiple values
   inside the 2023 Ratemaking Formula Report.** The report's cover letter
   (read in full) confirms the $9.067 billion aggregate industry retention
   and $17.000 billion limit level for the 2023 Contract Year, consistent
   with `config.py`'s constants and with Article V(27)'s formula structure,
   and a general web search independently attributed the specific values
   6.0732/7.2878/12.1464 to this same report. The full ~100-page report was
   not paginated through in this pass to locate and quote that exact
   exhibit table directly. This does not change the conclusion (the values
   are corroborated from two directions -- the cover letter's aggregate
   dollar figures combine with Article V(27)'s stated ratios to reproduce
   them, and the web search independently names the same report) but a
   direct page citation is still open.
3. **Whether the model represents Citizens as one combined entity or as
   its two statutory accounts.** Article V(8) treats Citizens' coastal
   account and its personal-lines/commercial-lines account as two separate
   FHCF participants, each with its own Retention, Premium, and Ultimate
   Net Loss. The model represents Citizens as a single combined entity
   (`fl_risk_model/config.py::CITIZENS_FHCF_PREMIUM_USD` is one number).
   This is a retained aggregation choice, not a contract-compliance defect
   (the brief instructs keeping Citizens at "the currently adopted model
   resolution"), but it means the model's single Citizens Retention and
   Limit are an aggregate approximation of two separate contractual
   Retentions/Limits. This is not something the primary contract can
   resolve on its own -- it is a modeling-resolution question the author
   has already decided to leave as is; it is recorded here only so the
   approximation is documented, not to reopen it.

No question is being posed here that the contract text already answers;
per the brief, the formula-order and aggregation-level questions are
treated as resolved above, not asked of the author.

## 7. Dependency map for the next rerun

Every output below was generated using the current, now-confirmed-incorrect
`apply_fhcf_recovery`, applied at company x county granularity. Because the
FHCF recovery amount feeds directly into `NetWindUSD`, which drives capital
depletion, defaults, FIGA and Citizens deficits, and (via the earlier
correction pass) the corrected public-burden aggregate, **any output that
depends on FHCF recovery for a private insurer or Citizens must be treated
as provisional until the financial model is replayed with the patch in
Section 5.** This supersedes the "corrected" numbers delivered under the
prior handoff's correction register items R1, C1, and C5: those items'
*methodology* (season-level-sum-before-quantile, non-overlapping aggregate
definition, corrected building-code total reconstruction) remains valid,
but the *underlying archived `iterations.csv` values they operated on* used
the pre-patch FHCF formula and are not yet re-derived. Do not read those
earlier deliverables' numeric tables as final.

| Output | Depends on FHCF recovery? | Rerun needed? |
|---|---|---|
| ERA5 baseline (`emanuel_era5_baseline_20260326_141913`), Table 1, Fig. 3, all corrected-burden post-processing from the prior pass | Yes (every season with nonzero wind loss computes FHCF recovery for private insurers and/or Citizens) | **Yes -- financial-model replay required.** Post-processing scripts (`scripts/earths_future_revision/section5_return_periods.py`, `section6_decomposition.py`) do not need to change; they must be re-run against the new `iterations.csv`. |
| Historical/sequential scenarios (Great Miami, Andrew, Lake Okeechobee, Irma, and the three paired scenarios), SI Table S3 | Yes | **Yes.** Great Miami and the two Great-Miami-containing sequential scenarios have the largest FHCF shortfalls in the archived data and are the most likely to be materially affected by both defects. |
| Insured-fraction sensitivity (`insured_frac_sensitivity_combined`, f = 0.1-0.5), SI Table S6 | Yes (FHCF/FIGA/Citizens columns); wind un/underinsured and private/Citizens gross insured wind columns are allocation outputs upstream of FHCF and are unaffected in themselves | **Yes**, for every FHCF-, FIGA-, Citizens-, and burden-related column and elasticity. The wind-allocation-only elasticities (e.g. "Wind insured, private," "Wind un/underinsured") do not depend on FHCF recovery and would be unchanged by this patch alone. |
| NFIP allocation county-rate table (`section7_nfip_allocation.py`) and its tests | No (NFIP flood recovery does not touch FHCF) | **No.** This output and its blocked status (missing proprietary per-event hazard cache) are unrelated to the FHCF patch. |
| Policy scenarios: market exit, insurance penetration, building codes (ERA5) | Yes | **Yes**, for FHCF/FIGA/Citizens/burden columns in all three. The building-code total-loss and uninsured-residual fix already applied in `mc_run_events.py` (register item C5) is independent of the FHCF formula and remains correct on its own, but the *insured* wind, FHCF shortfall, FIGA, and Citizens columns for the building-codes scenario still need the FHCF patch and a rerun. |
| Five-GCM baseline comparisons (20 directories, `emanuel_{gcm}_{period}_baseline_*`) and the building-code x GCM sweep (55 directories, `emanuel_{gcm}_ssp245cal_buildingcode_w*f*_*`) | Yes | **Yes**, all of them. This is the largest share of the compute footprint of any planned rerun. |
| Variance decomposition (Table S7, already removed from the manuscript; research code and archived results retained) | Yes, but the table is not used in the manuscript | Not needed for the manuscript. Retained as research code; would need a rerun only if the author later reverses the Table S7 removal decision. |
| Newly identified but not yet run: a targeted before/after comparison isolating each of the two FHCF defects separately (formula-order only, aggregation-only, both together) on the ERA5 baseline | -- | **Not launched.** This is a natural small addition to the eventual rerun plan (it would show which defect dominates the change in headline numbers) but was not requested by this brief and is not started here. |

### Next-stage plan (specified, not launched)

1. Implement and code-review the Section 5 patch.
2. Re-run the existing test suites (`fl_risk_model/tests/earths_future/`,
   including the 23 new tests here) against the patched code to confirm the
   confirmed-defect tests now pass under the *verified* formula (they
   currently assert the *current* code's behavior, including where it is
   wrong; several assertions -- e.g. in
   `test_far_above_limit_current_code_UNDER_recovers` -- will need to be
   flipped to assert `actual == expected` once the patch lands, since their
   present purpose is to document the discrepancy, not to lock in the bug).
3. Small cluster pilot: replay the ERA5 baseline (and, if feasible, one
   historical scenario with a known-large FHCF shortfall, e.g. Great Miami)
   using identical hazard inputs, county-loss caches, seeds, and parameter
   draws as the archived production runs, to produce a paired before/after
   comparison attributable solely to the FHCF patch.
4. Freeze the verified code and rerun all affected financial analyses
   listed in the table above: the ERA5 baseline, all eight historical/
   sequential scenarios, the insured-fraction sweep, all three policy
   scenarios, all five GCM baseline sets, and the full building-code x GCM
   sweep.
5. Re-run `scripts/earths_future_revision/section5_return_periods.py`,
   `section6_decomposition.py`, `section7_insured_fraction.py`,
   `section8_historical_and_variance.py`, and
   `section9_climate_policy_tables.py` against the new outputs, and update
   the manuscript's numeric tables and the correction register accordingly.

This plan reuses existing hazard and county-loss inputs and generates no
new hazard catalogs or climate scenarios, consistent with the brief. It is
not launched in this task.

## Correction register update

`docs/earths_future_revision/correction_register.md` item **C3** ("FHCF
coverage election possibly applied twice") is superseded by this document.
C3's own status was already "UNRESOLVED, AUTHOR DECISION NEEDED," so this
does not overturn a completed correction; it replaces an inference-only
hypothesis with a source-verified one and adds the previously undocumented
aggregation-level defect. The register's Section 4 items (C1, the
FHCF-shortfall non-overlap fix) and its dependency on FHCF recovery values
are unaffected in their *logic* -- an insurer's FHCF shortfall still
overlaps with its own downstream deficit under either formula -- but the
*numeric* shortfall amounts feeding that logic will change once this patch
is applied and the model is rerun. A corresponding note has been added to
`correction_register.md` (item C3) pointing to this document; the original
C3 text is preserved below it for the record rather than deleted, per the
brief's instruction to keep the history of what was suspected versus
resolved.
