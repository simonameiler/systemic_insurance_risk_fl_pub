# Earth's Future corrections and sensitivity: summary report

Branch `earths-future/corrections-and-sensitivity`, worktree
`/Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl_earths_future`,
base commit `9fad8e4babf9fcdf8ccb24280cf8156225dd03f6`. Full detail is in
`correction_register.md`, `before_after_table.md`, and `data_inventory.md`
in this directory.

## What was actually completed

1. **Reproduced the submitted headline numbers exactly** from archived
   Monte Carlo output before changing anything (Table 1, SI Table S3, SI
   Table S7's eta-squared values, SI Table S4's building-code total loss and
   uninsured residual all match to the reported precision). This confirms
   the archived `emanuel_era5_baseline_20260326_141913`, historical-scenario,
   and building-code runs are the actual sources for the submitted tables.

2. **Confirmed and fixed a real code bug** in `fl_risk_model/mc_run_events.py`:
   the building-code scenario's reported gross loss totals were captured
   before the loss-reduction step, so the submitted SI Table S4 showed an
   essentially unchanged total loss and an *increase* in uninsured wind and
   flood losses under building codes, when the correctly-allocated
   components show a genuine ~29% loss reduction and a *decrease* in
   uninsured losses. Fixed at the source; reconstructed corrected values for
   the already-archived run without needing a rerun.

3. **Confirmed and corrected a real accounting bug**: the submitted "total
   public burden" summed the FHCF shortfall on top of FIGA and Citizens
   deficits that, whenever the shortfall was not absorbed by an insurer's
   own capital, already contained it. Traced through the actual model code,
   demonstrated with deterministic fixtures using the real
   `fl_risk_model.fhcf`/`capital` functions, and corrected by excluding FHCF
   shortfall from the summed aggregate (kept as a separate diagnostic).

4. **Confirmed and corrected the return-period aggregation bug** flagged in
   the handoff brief: Table 1's public-burden row summed four separately
   estimated marginal quantiles instead of taking the quantile of the
   season-level sum. Recomputed correctly from the archived 10,000-season
   baseline, with a 1,000-resample bootstrap.

5. **Combined effect on the headline claim**: "public burden increases more
   than fortyfold" (10y to 100y) becomes **about thirtyfold** (43.9x to
   31.1x); the fitted power-law exponent moves from 1.64 to **1.51**. The
   qualitative claim (supralinear amplification, concentrated in FIGA and
   Citizens rather than FHCF) is unchanged. The 1%-of-GDP annual exceedance
   probability is essentially unchanged (3.7%); the 10%-of-GDP probability
   moves from 0.13% to 0.08% (both round to "about 0.1%").

6. **Executed the common-season burden decomposition** (Section 6) on the
   corrected baseline: severity bins from zero-loss through beyond the
   100-250 year range, ratio-of-means shares, 1,000-resample bootstrap, and
   an exact reconciliation check (max discrepancy 3e-17). Output figure and
   machine-readable table delivered.

7. **Reprocessed the insured-fraction sensitivity** (Section 7) using the
   existing 0.1-0.5 archived sweep with corrected accounting: recomputed
   every Table S6 elasticity from unrounded means via centered log-log
   finite differences, added a new elasticity for the corrected aggregate
   (1.14), and quantified the Beta(4,6)-vs-fixed-0.4 gap directly (all
   metrics differ by <5%, confirming they are close but not identical).

8. **Built and tested the NFIP allocation sensitivity's data layer**
   (Section 7): the three county rate configurations (structure-weighted
   baseline, SFHA-only, non-SFHA-only) from the actual production FEMA
   dataset, verified the baseline lies within the per-county envelope for
   all 68 counties, and confirmed the rate ordering is not uniform (so
   neither configuration is a blanket upper or lower bound). Scoring these
   against corrected burden/insured-flood-loss outcomes is blocked by
   missing proprietary hazard inputs (below); the remaining steps and code
   change needed are documented exactly.

9. **Verified the submitted 28.5% / 15.3% / 56.2% season-count claim**
   exactly and gave it a precise, previously-undocumented definition
   (zero-loss / nonzero-loss-one-event / nonzero-loss-multi-event).

10. **Audited and removed SI Table S7** (variance decomposition):
    reproduced every eta-squared value exactly, then removed the table and
    its associated robustness language from the manuscript copy per the
    revision brief, since the 300 non-randomly-selected seasons do not
    license the broad claims previously attached to them. Research code and
    archived results are untouched. All SI table numbering, cross-references,
    and the `\SITab*` macros in both the main and SI LaTeX projects were
    updated accordingly.

11. **Found and documented additional, previously unflagged issues** while
    tracing the code: (a) a confirmed, currently-inactive inconsistency in
    the Citizens-specific FHCF fallback formula; (b) a confirmed discrepancy
    between the SI's stated Citizens surplus-scaling exponent (1.2) and the
    actual code (1.3, which the archived results already reflect) --
    corrected the SI text to match the code; (c) via targeted web search,
    confirmed California's December 2024 catastrophe-model/reinsurance-cost
    regulation supersedes the manuscript's "prohibition" claim, and located
    a secondary-source figure (~USD 6B, ~37% of the FHCF cap) for the
    Hurricane Irma FHCF-reimbursement discrepancy Reviewer 2 requested.

12. **Wrote 17 passing tests** covering: the general statistical fact behind
    the return-period bug; the FHCF-shortfall double-count and its
    resolution, using the real production FHCF/capital functions; the
    seasonal-aggregation approximation for cat bonds and paired storms;
    bounded NFIP allocation with edge cases (no SFHA stock in a county); and
    a regression lock on the building-codes total-loss fix.

## What changed the paper's conclusions and what did not

See `before_after_table.md` for the full list. The two claims that change in
kind, not just in decimal places: (1) the "more than fortyfold" public-burden
amplification becomes "more than thirtyfold," and (2) SI Table S4's
building-code column, which appeared to show building codes barely reducing
losses and even *increasing* uninsured exposure, is corrected to show the
intended ~29% loss reduction and a genuine decrease in uninsured exposure.
Every other qualitative finding I checked -- FIGA as the dominant and most
frequent channel, Citizens second, FHCF most resilient; the historical/
sequential-event ranking; insurance-penetration expansion raising NFIP/
Citizens stress even as private-market stress falls -- survives the
corrections.

## What remains blocked, and why

Full production reruns (a corrected FHCF equation, new insured-fraction
hazard draws, the three NFIP allocation configurations scored against
insured losses and financing, most of SI Table S5's exceedance-probability
grid recomputed under corrected accounting for the climate/policy columns)
require replaying the financial model against per-event, per-county hazard
data. The proprietary WindRiskTech synthetic tropical-cyclone event sets and
the Gori et al. wind/flood attribution outputs that this requires are not
distributed in this repository (`fl_risk_model/data/hazard/emanuel/` and
`.../gori_data/` contain only `.gitkeep` placeholders), and S&P Capital IQ
company surplus data is licensed and also absent. This was true in the
original checkout as well as this worktree; it is not something this pass
could work around. `data_inventory.md` and the correction register record
the exact commands and code changes needed once those inputs are available.
No compute cluster access was used or assumed for anything above; every
completed analysis ran locally against archived or already-tracked data.

A working LaTeX toolchain (`pdflatex`/`latexmk`) is not available in this
environment, so the manuscript and SI LaTeX sources were edited and
structurally checked (`\begin`/`\end` balance for every modified
environment) but not compiled to PDF. This should be done before submission.

Several AUTHOR CHECK items in the correction register remain genuinely
unresolved and need an author decision or a primary-source lookup this pass
could not complete in the time available: the FHCF coverage-election
formula (strong evidence, not certain), the exact 38.9% insured-fraction
calibration weighting, the primary FHCF Irma reimbursement figure, the exact
13-level mitigation-sweep encoding, and several TIV-sampling and
frequency-calibration documentation gaps. These are listed individually,
with what was and was not checked, in `correction_register.md`.

## Reproduction

```
conda activate climada_env
cd scripts/earths_future_revision
python run_all.py
```

Runs the full pytest suite (17 tests) and every reprocessing stage in
Sections 5-9 in about 30 seconds total. Individual-stage commands are in
each script's module docstring and in `correction_register.md`.
