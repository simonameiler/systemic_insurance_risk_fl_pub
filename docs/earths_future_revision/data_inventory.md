# Data inventory: this worktree vs. the original checkout

Branch `earths-future/corrections-and-sensitivity`, worktree
`/Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl_earths_future`,
base commit `9fad8e4babf9fcdf8ccb24280cf8156225dd03f6` (`master`, "updates figures").

`git worktree add` only checks out git-tracked files at the base commit. It
does **not** copy ignored files, so several locally generated (but
gitignored) result directories present in the original checkout
(`/Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl_pub`)
were absent here until copied explicitly. This file records exactly what was
copied, from where, and what remains unavailable.

## Copied read-only from the original checkout (untracked, gitignored, ~958 MB)

All copied `into` this worktree's `results/mc_runs/`, unmodified, and remain
gitignored here (not committed):

| Directory | Contents | Used for |
|---|---|---|
| `great_miami_20260326_190906/`, `andrew_20260326_192708/`, `lake_okeechobee_20260326_212501/`, `irma_20260326_213211/`, `gm_then_andrew_20260326_211013/`, `double_gm_20260326_211737/`, `double_irma_20260326_213909/` | 1,000-realization historical/paired-storm Monte Carlo runs (`iterations.csv`) | Section 8 corrected SI Table S3 reconstruction |
| `insured_frac_sensitivity_combined/` | `iterations_frac_{0.1..0.5}.csv`, 10,000 seasons each, fixed insured wind fraction, seed 42 | Section 7 insured-fraction sensitivity |
| `variance_nested_300x50_20260311_211009/` | `variance_decomposition_nested.csv` (300 seasons x 50 draws) | Section 8 SI Table S7 audit |
| `variance_fixed_params_20260311_185311/` | Comparison run for the variance design | Section 8 audit, reference only |

Note: `andrew_then_gm_20260326_210244/` was also copied but is not currently
used by any script (kept for completeness; the manuscript's "sequential"
scenarios are Great Miami -> Andrew, double Great Miami, and double Irma).

**Added in the FHCF corrections and pilot task**: two more gitignored,
licensed inputs, present in the original checkout but absent from this
worktree, were required to run any local financial-model pilot at all
(`fl_risk_model.loader.load_market_share` and the surplus loader raise
`FileNotFoundError` without them) and were copied read-only, unmodified,
not committed:

| File | Contents | Used for |
|---|---|---|
| `fl_risk_model/data/FL HO Market Share Report_6.10.25.xlsx` | FLOIR statewide residential market-share-by-company report | Company premium/exposure allocation in every `run_one_scenario` call |
| `fl_risk_model/data/20250805 FL Surplus Capital, Group v Entity.xlsx` | S&P Capital IQ-derived entity/group statutory surplus | Capital depletion in every `run_one_scenario` call |

SHA-256 (16-hex-char) checksums of both files, alongside every other input
the local FHCF pilot depends on, are recorded in
`results/earths_future_revision/fhcf_pilot/pilot_manifest.json`.

**Discrepancy found while reconciling this inventory**: the copied historical
run directories contain **1,000** realizations per scenario, and reproduce
the submitted SI Table S3 numbers to the reported precision. Main Methods
and Supporting Text S5 state **200** Monte Carlo realizations per historical
scenario. This is recorded as correction-register item R4 (reporting
discrepancy, not a numerical error): either the stated realization count
should be corrected to 1,000, or the run that used exactly 200 realizations
should be located and substituted.

## Already git-tracked in this worktree (unchanged)

`results/mc_runs/emanuel_era5_baseline_20260326_141913/`,
`emanuel_era5_market_exit_moderate_*/`, `emanuel_era5_penetration_major_*/`,
`emanuel_era5_building_codes_major_*/`, all 5-GCM x 5-period baseline runs,
and the 13-level x 5-GCM building-code sweep (`*_buildingcode_w*f*_*/`, corrected count -- see correction register item R3) are
tracked in git and were present immediately after `git worktree add`. These
are the inputs for Sections 5, 6, and 9.

## Not available in this checkout (blocks full production reruns)

`fl_risk_model/data/hazard/emanuel/` and `fl_risk_model/data/hazard/gori_data/`
contain only `.gitkeep` placeholders in both the original checkout and this
worktree. Per `results/README.md` and the paper's Data Availability
statement, these are:

- Proprietary WindRiskTech L.L.C. synthetic tropical-cyclone event sets
  (Emanuel statistical-dynamical model output), and
- The Gori et al. (2025) multi-hazard wind/flood attribution regression
  outputs used to split gridded CLIMADA losses into wind and flood shares
  by county and event.

Both require a data-sharing agreement with the respective owners and are not
redistributed in this repository. Their absence blocks any correction that
requires **replaying the financial model against per-event, per-county
losses** with a changed rule (a new NFIP allocation configuration, a
corrected FHCF equation, a new insured-fraction hazard draw). It does not
block corrections that are purely **post-processing of already-simulated
season-level totals** (Sections 4's aggregate definition, Section 5's
return-period estimator, Section 6's decomposition, Section 9's building-code
total reconstruction), because the archived `iterations.csv` files already
contain the correctly-computed underlying component columns.

S&P Capital IQ company-level surplus data (licensed, not redistributable)
is likewise absent; it affects only fresh production reruns, not
post-processing of archived outputs.

## Original checkout: unaffected

`/Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl_pub`
was not reset, stashed, or cleaned. Its pre-existing uncommitted change to
`notebooks/probabilistic_risk_analysis_pub.ipynb` (a two-line `execution_count`
metadata diff with no code or output changes -- confirmed via `git diff`)
remains exactly as it was found and was not committed by this work.
