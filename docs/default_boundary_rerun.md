# Financial rerun before the next release

The final insurer balance is now set to zero when its absolute value is less than USD 0.01, before default classification and FIGA assessment membership are determined. A deficit of exactly USD 0.01 remains a default. The company output retains `SurplusRoundingAdjustmentUSD` so the normalization can be reconciled with capital transfers. This prevents floating-point cancellation in group support from changing the default count and FIGA's premium assessment base.

The existing production inventory reruns 106 financial analyses using the existing loss caches. No hazard preprocessing is required. All outputs go to fresh timestamped folders; keep the previous production results for comparison. Do not publish a new release until the results and publication outputs have been checked.

## On Sherlock

From the existing checkout, with no local code edits pending:

```bash
cd ~/repos/systemic_insurance_risk_fl_earths_future
git fetch origin
git switch --track origin/rerun/default-boundary-20260923
conda activate climada_env
python -m pytest fl_risk_model/tests/earths_future/test_default_boundary.py -q
bash scripts/cluster/catbond_revision.sh check
bash scripts/cluster/catbond_revision.sh pilot
```

If the branch already exists locally, use `git switch rerun/default-boundary-20260923` followed by `git pull --ff-only`. If Git reports local changes, stop and preserve them before switching; do not reset the checkout.

When the pilot job has finished:

```bash
bash scripts/cluster/catbond_revision.sh pilot-report
bash scripts/cluster/catbond_revision.sh production
```

The pilot retains the established paired catastrophe-bond checks, with the new balance convention applied to both sides. The dedicated tests above check the balance boundary. Production is gated on a successful pilot for the same code and inputs. Keep this checkout at the same commit until the jobs finish.

```bash
bash scripts/cluster/catbond_revision.sh status
bash scripts/cluster/catbond_revision.sh postprocess
```

The wrapper retains its existing `catbond` directory names. The new timestamp distinguishes this financial rerun from the completed September 18 campaign.

## Copy results back

On the local computer, use your Sherlock login:

```bash
rsync -av --progress smeiler@login.sherlock.stanford.edu:~/repos/systemic_insurance_risk_fl_earths_future/results/mc_runs_catbond_patched/ /Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl_earths_future/results/mc_runs_catbond_patched/
rsync -av --progress smeiler@login.sherlock.stanford.edu:~/repos/systemic_insurance_risk_fl_earths_future/results/earths_future_revision/catbond_cluster/ /Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl_earths_future/results/earths_future_revision/catbond_cluster/
rsync -av --progress 'smeiler@login.sherlock.stanford.edu:~/repos/systemic_insurance_risk_fl_earths_future/logs/ef_production_*' /Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl_earths_future/logs/
```

No `--delete` is used. Transfer the new pilot logs too if the pilot needs diagnosis.

## Publication refresh

The publication scripts normalize FIGA residual deficits smaller than USD 0.01 before deriving probabilities and financing requirements. They preserve the original iteration files. The insurer-balance correction occurs upstream in the financial model and cannot be recovered from the old iteration summaries.

Use `scripts/earths_future_revision/refresh_publication.py --help` to regenerate figures, table fragments, and numerical reference values from the newly validated production manifest. Supply the local ERA5 `all_events.csv` directory with `--physical-data`. The optional `--manuscript-dir` installs figures and tables; manuscript prose still needs review against the generated `text_values.json`. Keep the main branch and Zenodo release unchanged until that review is complete.
