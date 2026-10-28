# Code–paper fixes (branch `rerun/code-paper-fixes-20261028`)

These commits fix the remaining places where the code does not do what the
preprint's Methods and Supporting Text S4 describe. They were written on the
local branch `fix/remaining-code-paper-discrepancies` (from
`earths-future/corrections-and-sensitivity` at `b189a89`; commit `957bd67` is
the golden master for the general engine in `systemic_insurance_model`,
architecture §8.2a). This branch carries them on top of
`rerun/default-boundary-20260923` (`014ed02`), whose model code is that of
release v1.1.0, so a re-run isolates these fixes against the v1.1.0 results.
The two USD 0.01 balance rules (item 8 here, sub-cent normalization in
`014ed02`) are the same rule and are merged; the model code otherwise equals
`957bd67`. Per-company figures from the licensed market-share workbook are
left out of this document.

## 1. Status of every known discrepancy

| # | Discrepancy | Effect in submitted runs | Status |
|---|---|---|---|
| 1 | FHCF retention and limit applied per insurer × county row, not per insurer (S11) | Private FHCF recovery inflated; statewide $17B cap reached in 3.0% of ERA5 baseline seasons (paper: cannot bind) | Fixed earlier: `ef77065` |
| 2 | FHCF formula order (cap applied before coverage share and LAE) | Recovery capped at K·p·1.1 instead of K | Fixed earlier: `ef77065` |
| 3 | Cat bonds: inert issue-date filter, industry-index bonds paid pro rata to all insurers, beneficiary mapping | $9.46B limit in force vs the paper's eight indemnity bonds ($2.54B) | Fixed earlier: `373994d`, `b189a89` |
| 4 | Return-period "total public burden" summed marginal quantiles; FHCF shortfall double counted; stale building-code totals | Table 1 / S4 reporting | Fixed earlier: register items R1, C1, C5 |
| 5 | **Citizens Tier 2 = 10% of Citizens' own premium.** `runner.py` summed a `PremiumUSD` column that `market_share_df` does not carry, so the private base was 0 | Tier-2 capacity $0.36B in every season instead of 10% of the statewide property base ($1.93B) | **Fixed here** |
| 6 | **29 of the top-100 private insurers have no surplus row.** Losses were joined onto the surplus table by name, so their losses never depleted capital, never caused a default and never reached FIGA. They were also missing from FIGA's survivor premium base. | 22% of modeled premium (26% of private) outside the solvency calculation | **Fixed here**: surplus from the market-share workbook (§2) |
| 7 | **Penetration scenario: Citizens share** divided scenario-scaled Citizens TIV by the *unscaled* FHCF county TIV | Citizens' effective share about 12–18% instead of the 8% target | **Fixed here** |
| 8 | USD 0.01 roundoff rule (Text S4) not implemented | Defaults counted on roundoff deficits | **Fixed here** |
| 9 | Zero-event seasons written as stub rows with 9 of ~60 columns | Means over other columns silently excluded them | **Fixed here** |
| 10 | Silent fallback to zero when the FIGA / Tier-2 premium base fails to load | FIGA capacity silently zero | **Fixed here**: raises unless `cfg.STRICT_INPUTS = False` (demo only) |
| 11 | Variance decomposition "fixed parameter" run did not pin wind-share concentration (module global bound at import) or the insured fraction | Run B could not test "hazard dominates" | **Fixed here** (Table S7 is removed from the manuscript; fix matters only if it returns) |
| 13 | **Citizens' starting surplus read as $0.** `load_citizens_capital_row_from_csv` split each line at the first comma and passed the rest of the row ("$4'183'356'808.00 , $6'284'900'000 , …") to a parser that cannot read `$` or apostrophe separators. It returned 0. Same on `master`, so every archived run used it. | Citizens' deficits overstated by up to $4.18B per season (2024 surplus, `citizens_capital_pml.csv`); the paper says claims first draw on Citizens' statutory surplus | **Fixed here**: parse the capital column only; unparseable values raise instead of returning 0 |
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

## 2. Surplus data gap (item 6)

The 2024 surplus extract (`20250805 FL Surplus Capital, Group v Entity.xlsx`)
has 113 entities. Of the top-100 homeowners writers in the market-share
workbook, 29 are absent by both name and S&P Statutory Entity Key. Together they
hold 22% of modeled premium, 26% of private premium. One more (Hartford Ins Co.
of the Midwest) differs only in spelling and now matches by key. The extract
instead contains 42 small writers that aren't in the market-share list.

The extract seems to keep only writers with three years of Florida homeowners
premium (its column A is 3 for every row). That explains 10 of the 29 (new
entrants such as Manatee, Ovation, Orange, Loggerhead), but not the other 19,
which had premium in all three years.

**Source used.** The market-share workbook (`FL HO Market Share
Report_6.10.25.xlsx`, sheet "FL - Individual Co", column "Capital & Surplus
($000)") gives 2024 company-level capital for all 100 writers. For the 70
private writers in both files it equals the extract's Surplus as Regards
Policyholders within 0.1%. `load_surplus_data_with_groups` therefore adds the 29
from it (`cfg.SURPLUS_FILL_FROM_MARKET_SHARE = True`), before any scenario
adjustment. Group fields:

| Source | Writers | Rule |
|---|---|---|
| Unaffiliated | 16 | Standalone; no parent support |
| Sibling in the extract with the same S&P group | 3 (Ovation, American Modern, Covington) | Extract's group number and group surplus |
| Market-share workbook, sheet "FL - Group" | 10, in 9 groups (Slide, Accredited, Iron Family, Topa, HJJ, MS&AD, Sutton, Beazley, Lemonade) | That sheet's group Capital & Surplus. It agrees with the extract's group surplus within 1% for only 41 of 69 grouped writers, so these values are flagged in `GroupSurplusSource` |

The earlier premium-ratio estimate (median surplus / premium = 3.4) put Slide's surplus at about 17 times its actual
value. It remains only as a fallback
(`cfg.MISSING_SURPLUS_POLICY`) that warns. `"legacy_drop"` with the fill switched
off reproduces the submitted runs. Optional follow-up: group surplus on the
extract's definition for the 9 groups above.

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
  version, run locally (2026-10-22 and 2026-10-27) with the licensed 2024 inputs.
- "Before" is `earths-future/corrections-and-sensitivity` at `b189a89`, which
  already includes fixes 1–4. "After" is this branch.
- Script: `scripts/code_paper_fixes_compare.py`. Means for before, all fixes
  with the earlier premium-ratio surplus estimate, and all fixes:
  `results/code_paper_fixes/historical_comparison_means.csv`.
- The probabilistic runs (10,000 seasons, ERA5 and GCMs) need the Emanuel
  event caches and run on Sherlock (§4).

| Scenario | Residual financing (USD B) | FIGA residual | Citizens residual | Private defaults |
|---|---|---|---|---|
| Great Miami | 22.18 → 23.61 | 7.00 → 11.86 | 4.17 → 0.75 | 17.8 → 33.4 |
| Andrew | 10.14 → 9.89 | 3.95 → 6.90 | 4.05 → 0.85 | 15.5 → 29.9 |
| Lake Okeechobee | 15.07 → 15.65 | 1.17 → 2.08 | 0.33 → 0.00 | 11.0 → 21.7 |
| Irma | 0.14 → 0.15 | 0.10 → 0.11 | 0.00 → 0.00 | 5.5 → 10.7 |
| Great Miami then Andrew | 45.44 → 50.65 | 16.97 → 27.65 | 12.73 → 7.27 | 21.5 → 39.5 |
| Double Great Miami | 57.83 → 64.87 | 20.30 → 32.91 | 13.22 → 7.65 | 22.2 → 40.7 |
| Double Irma | 2.26 → 2.30 | 0.40 → 0.49 | 0.06 → 0.00 | 9.6 → 18.1 |
| Great Miami, market exit | 25.56 → 23.90 | 4.85 → 8.41 | 9.70 → 4.48 | 16.5 → 31.5 |
| Great Miami, penetration | 58.41 → 60.64 | 13.02 → 24.82 | 9.90 → 0.32 | 18.3 → 35.1 |
| Great Miami, building codes | 11.99 → 12.79 | 2.96 → 5.33 | 1.62 → 0.05 | 15.0 → 29.4 |

NFIP financing is unchanged in every scenario. The largest entity deficit
changes by less than 2%.

**What drives the changes**
- **Citizens' surplus (item 13):** Citizens now absorbs up to $4.18B before
  any assessment. This is the largest single correction for Citizens: in Great
  Miami its residual falls from $2.83B (all other fixes) to $0.75B. In the
  two-storm sequences it falls from about $11.2–11.7B to $7.3–7.7B.
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
- **Surplus coverage (item 6):** the largest change overall.
  - The 29 previously untested writers now deplete capital and can fail. With
    their actual surplus, defaults roughly double in every scenario (Great
    Miami 17.8 → 33.4), and the FIGA residual rises by about 60–90% (Great
    Miami $7.0B → $11.9B).
  - With the earlier premium-ratio estimate the same fix had shown almost no
    change in single storms, because the estimate gave these mostly Florida-
    focused writers far too much capital.
  - The residual financing requirement rises in most scenarios despite the
    lower Citizens residual. For example, Great Miami goes from $22.2B to
    $23.6B, and double Great Miami from $57.8B to $64.9B.

## 4. Re-running on Sherlock

The workflow is the one used for the default-boundary re-run
(`docs/default_boundary_rerun.md`, `scripts/cluster/catbond_revision.sh`). It
reruns the 106 financial analyses from the existing loss caches; no hazard
preprocessing. No new S&P data is needed: the 29 writers come from the
market-share workbook (§2), which `MARKET_SHARE_XLSX` already points to.
`check` records input-file hashes, so the manifest shows which inputs each run
used.

On Sherlock, from the existing checkout, with no local code edits pending:

```bash
cd ~/repos/systemic_insurance_risk_fl_earths_future
git status --short        # must print nothing; otherwise preserve the changes first
git fetch origin
git switch --track origin/rerun/code-paper-fixes-20261028
conda activate climada_env
python -m pytest fl_risk_model/tests/earths_future -q
bash scripts/cluster/catbond_revision.sh check
bash scripts/cluster/catbond_revision.sh pilot
```

When the pilot job has finished:

```bash
bash scripts/cluster/catbond_revision.sh pilot-report
bash scripts/cluster/catbond_revision.sh production
bash scripts/cluster/catbond_revision.sh status        # until all 106 are done
bash scripts/cluster/catbond_revision.sh postprocess
```

Production is gated on a passing pilot for the same code and inputs and
refuses a duplicate submission; a new commit starts a fresh inventory. Keep
the checkout at this commit until the jobs finish. Copy the results back as in
`docs/default_boundary_rerun.md` (no `--delete`).

The production outputs are the distribution-level benchmark for the general
engine (architecture §8.2a(ii)). Record the commit hash and the manifest path
in `systemic_insurance_model/docs/architecture.md` §8.2a.

## 5. Tests

`fl_risk_model/tests/earths_future/test_remaining_discrepancies.py` (13
tests) covers items 5–9 and 13. It runs in the `check` suite. Full suite on
this branch: 135 passed, 4 skipped.
