# Code–paper fixes (branch `fix/remaining-code-paper-discrepancies`)

This branch starts from `earths-future/corrections-and-sensitivity` at
`b189a89` and fixes the remaining places where the code does not do what the
preprint's Methods and Supporting Text S4 describe. Together with the fixes
already on that branch, it is the corrected model for re-running all analyses
on Sherlock. It is also the benchmark (golden master) for the general engine
in `systemic_insurance_model` (architecture §8.2a).

## 1. Status of every known discrepancy

| # | Discrepancy | Effect in submitted runs | Status |
|---|---|---|---|
| 1 | FHCF retention and limit applied per insurer × county row, not per insurer (S11) | Private FHCF recovery inflated; statewide $17B cap reached in 3.0% of ERA5 baseline seasons (paper: cannot bind) | Fixed earlier: `ef77065` |
| 2 | FHCF formula order (cap applied before coverage share and LAE) | Recovery capped at K·p·1.1 instead of K | Fixed earlier: `ef77065` |
| 3 | Cat bonds: inert issue-date filter, industry-index bonds paid pro rata to all insurers, beneficiary mapping | $9.46B limit in force vs the paper's eight indemnity bonds ($2.54B) | Fixed earlier: `373994d`, `b189a89` |
| 4 | Return-period "total public burden" summed marginal quantiles; FHCF shortfall double counted; stale building-code totals | Table 1 / S4 reporting | Fixed earlier: register items R1, C1, C5 |
| 5 | **Citizens Tier 2 = 10% of Citizens' own premium.** `runner.py` summed a `PremiumUSD` column that `market_share_df` does not carry, so the private base was 0 | Tier-2 capacity $0.36B in every season instead of 10% of the statewide property base ($1.93B) | **Fixed here** |
| 6 | **29 of the top-100 private insurers have no surplus row.** Losses were joined onto the surplus table by name, so their losses never depleted capital, never caused a default and never reached FIGA. They were also missing from FIGA's survivor premium base. | 22% of private premium outside the solvency calculation | **Fixed here** (code); **data re-pull needed** (§2) |
| 7 | **Penetration scenario: Citizens share** divided scenario-scaled Citizens TIV by the *unscaled* FHCF county TIV | Citizens' effective share about 12–18% instead of the 8% target | **Fixed here** |
| 8 | USD 0.01 roundoff rule (Text S4) not implemented | Defaults counted on roundoff deficits | **Fixed here** |
| 9 | Zero-event seasons written as stub rows with 9 of ~60 columns | Means over other columns silently excluded them | **Fixed here** |
| 10 | Silent fallback to zero when the FIGA / Tier-2 premium base fails to load | FIGA capacity silently zero | **Fixed here**: raises unless `cfg.STRICT_INPUTS = False` (demo only) |
| 11 | Variance decomposition "fixed parameter" run did not pin wind-share concentration (module global bound at import) or the insured fraction | Run B could not test "hazard dominates" | **Fixed here** (Table S7 is removed from the manuscript; fix matters only if it returns) |
| 12 | Demo failed every iteration on the corrections branch. Its synthetic roster lacked Citizens, so the cat-bond check rejected Citizens' Everglades Re bond. It also read the licensed workbook for FIGA when present. | Peer-review demo unusable | **Fixed here**: Citizens added to the demo roster with zero private share; the demo uses a synthetic premium base (demo shares × USD 16B). `demo_output/expected_demo_summary.csv` was regenerated with no licensed files present. |

**Deliberately not changed** (modeling choices for the authors, not bugs):
- Group-support triage cures the smallest deficits first, which maximizes the
  number of rescued entities. The paper doesn't state this (S14); add it to
  Text S4.
- `GROUP_CONTRIBUTION_RANGE` is drawn but not used by the support rule. The
  draw is kept so RNG streams stay comparable.
- Market exit defines its 25% Citizens target against the pre-exit total, so
  its split denominator is correct and unchanged.
- Statutory refinements found in the statute check are model extensions, not
  bugs:
  - FIGA and Citizens emergency assessments are multi-year streams;
  - assessment bases are broader than homeowners premium;
  - FIGA claim caps;
  - FHCF per-event retention (full, full, then one-third).

  See `systemic_insurance_model/docs/research/fl_de_ch_institutions.md`.

## 2. Surplus data gap (item 6): what to re-pull from S&P Capital IQ

The 2024 surplus extract (`20250805 FL Surplus Capital, Group v Entity.xlsx`)
has 113 entities. Of the top-100 homeowners writers in the market-share
workbook, 29 are absent by both name and S&P Statutory Entity Key. One more
(Hartford Ins Co. of the Midwest) differs only in spelling and now matches by
key. The extract instead contains 43 entities that aren't in the market-share
list.

**Interim handling.** `cfg.MISSING_SURPLUS_POLICY = "impute_premium_ratio"`:
- surplus = median(surplus / FL HO premium) of matched entities × the
  entity's premium;
- no group support for imputed entities;
- a one-time warning lists the imputed entities.

The other policies:
- `"error"`: refuse to run with missing surplus;
- `"legacy_drop"`: reproduce the submitted behavior.

**Fix:** re-pull entity and group surplus (2024, Surplus as Regards
Policyholders, with NAIC group code) for these 29 entities and add them to
the extract. The imputation then switches off automatically, since it only
applies to entities without a row.

| # | Entity (market-share workbook) | S&P Statutory Entity Key |
|---|---|---|
| 1 | Slide Insurance Co. | C9834 |
| 2 | Amer Integrity Ins Co FL | C3986 |
| 3 | Olympus Insurance Co. | C6248 |
| 4 | Monarch National Insurance Co. | C8792 |
| 5 | Security First Insurance Co. | C4528 |
| 6 | Southern Oak Insurance Co. | C4548 |
| 7 | People's Trust Insurance Co. | C6513 |
| 8 | American Traditions Ins Co. | C3993 |
| 9 | Loggerhead Recpl Interinsuranc | C9910 |
| 10 | Vyrd Insurance Co. | C9805 |
| 11 | Accredited Specialty Ins Co. | C9700 |
| 12 | Manatee Insurance Exchange | C10040 |
| 13 | Safepoint Insurance Co. | C8520 |
| 14 | Orange Insurance Exchange | C10014 |
| 15 | SureChoice Underwriters Recpl | C9757 |
| 16 | Trusted Resource Underwriters | C9699 |
| 17 | Ovation Home Ins Exchange | C10067 |
| 18 | Topa Insurance Co. | C1803 |
| 19 | Centauri Specialty Ins Co. | C4297 |
| 20 | American Modern Home Ins Co. | C144 |
| 21 | MS Transverse Spclty Ins Co. | C1622 |
| 22 | KW Specialty Insurance Co. | C9175 |
| 23 | Victor Insurance Exchange | C9979 |
| 24 | Sutton National Insurance Co. | C1866 |
| 25 | Sutton Specialty Insurance Co. | C9661 |
| 26 | Beazley Excess & Surplus Ins I | C9994 |
| 27 | Lemonade Insurance Co. | C8979 |
| 28 | Kinsale Insurance Co. | C681 |
| 29 | Covington Specialty Ins Co. | C6404 |

## 3. Effect on the historical stress tests

**Setup.**
- 200 realizations per scenario with seed 42 and the same draws in each
  version, run locally on 2026-10-22 with the licensed 2024 inputs.
- "Before" is `earths-future/corrections-and-sensitivity` at `b189a89`, which
  already includes fixes 1–4. "After" is this branch.
- Script: `scripts/code_paper_fixes_compare.py`. Means for all three variants,
  including the intermediate one without the surplus fix:
  `results/code_paper_fixes/historical_comparison_means.csv`.
- The probabilistic runs (10,000 seasons, ERA5 and GCMs) need the Emanuel
  event caches and run on Sherlock (§4).

| Scenario | Residual financing (USD B) | FIGA residual | Citizens residual | Private defaults |
|---|---|---|---|---|
| Great Miami | 22.18 → 20.62 | 7.00 → 6.78 | 4.17 → 2.83 | 17.8 → 18.8 |
| Andrew | 10.14 → 8.62 | 3.95 → 3.71 | 4.05 → 2.76 | 15.5 → 15.7 |
| Lake Okeechobee | 15.07 → 14.54 | 1.17 → 0.96 | 0.33 → 0.00 | 11.0 → 11.0 |
| Irma | 0.14 → 0.06 | 0.10 → 0.01 | 0.00 → 0.00 | 5.5 → 5.5 |
| Great Miami then Andrew | 45.44 → 45.17 | 16.97 → 18.24 | 12.73 → 11.19 | 21.5 → 36.0 |
| Double Great Miami | 57.83 → 58.59 | 20.30 → 22.61 | 13.22 → 11.66 | 22.2 → 41.1 |
| Double Irma | 2.26 → 1.97 | 0.40 → 0.17 | 0.06 → 0.00 | 9.6 → 9.6 |
| Great Miami, market exit | 25.56 → 23.79 | 4.85 → 4.61 | 9.70 → 8.17 | 16.5 → 16.7 |
| Great Miami, penetration | 58.41 → 52.82 | 13.02 → 14.73 | 9.90 → 2.59 | 18.3 → 18.9 |
| Great Miami, building codes | 11.99 → 10.81 | 2.96 → 2.72 | 1.62 → 0.68 | 15.0 → 15.0 |

NFIP financing and the largest entity deficit are unchanged in every
scenario.

**What drives the changes**
- **Tier 2 (item 5):** capacity rises from $0.36B to $1.93B ($2.82B under
  penetration). This lowers the Citizens residual by about $1.3–1.5B in every
  scenario where Citizens is in deficit.
- **Penetration split (item 7):**
  - Citizens' share of insured wind in the Great Miami penetration scenario
    falls from 22.6% to 15.7%. The baseline is 21.7%, and the scenario
    depopulates Citizens toward 8% statewide, but Miami-Dade stays above the
    state share.
  - The Citizens residual drops from $9.9B to $2.6B, while the FIGA residual
    rises from $13.0B to $14.7B.
  - The paper's finding that penetration raises Citizens capacity exceedance
    (+2.3 pp, Table S5) is likely to shrink or reverse. Check it in the
    re-run.
- **Surplus coverage (item 6):**
  - The 29 previously untested insurers now deplete capital and can default.
    Defaults nearly double in the two-storm sequences (21.5 → 36.0 and
    22.2 → 41.1), and the FIGA residual rises there.
  - In single storms the FIGA residual falls slightly, because the added
    survivors also enlarge FIGA's assessment base.
  - **These numbers are provisional.** The imputation ratio (median
    surplus / FL HO premium = 3.4, interquartile range 0.7–16.7) is dominated
    by national carriers whose surplus is large relative to their Florida
    book. It probably overstates the capital of Florida-focused writers such
    as Slide, Olympus or Security First, so the true effect on defaults and
    FIGA is likely larger.
  - **Re-pull the surplus data before the production re-run**, or treat the
    re-run's default and FIGA numbers as provisional.

## 4. Re-running on Sherlock

The cluster workflow from the corrections branch
(`docs/earths_future_revision/cluster_runbook.md`,
`scripts/cluster/earths_future.sh`) applies unchanged. `check` requires the
FHCF patch commit `ef77065`, which this branch contains.

1. Push this branch and check it out on Sherlock in place of
   `earths-future/corrections-and-sensitivity`.
2. If the re-pulled S&P surplus extract is ready, copy it to
   `fl_risk_model/data/` and point `cfg.SURPLUS_FILE` at it. The `check`
   command records input-file hashes, so the manifest shows which extract a
   run used. Without it, runs use the interim imputation (§2) and log a
   warning.
3. `scripts/cluster/earths_future.sh check` → `pilot` → `pilot-report` →
   `production`. Production refuses to resubmit only for the *same* code
   revision and inputs, so a new commit or a new surplus file starts a fresh
   inventory.
4. The production outputs become the benchmark for the general engine. Record
   the commit hash and the manifest path in
   `systemic_insurance_model/docs/architecture.md` §8.2a.

## 5. Tests

`fl_risk_model/tests/earths_future/test_remaining_discrepancies.py` (11
tests) covers items 5–9. It runs in the `check` suite. Full suite: 130 passed.
