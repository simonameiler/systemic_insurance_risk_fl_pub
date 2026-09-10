# Cluster execution runbook: `scripts/cluster/earths_future.sh`

Status as of this commit: **prepared and locally checked only**. No SSH
session to Sherlock was opened, no job was submitted, and the branch has
not been pushed, in this task. Everything below was validated by (a)
`bash -n` / `python -c "ast.parse(...)"` syntax checks, (b) running the
existing + new test suite locally (74 tests, see below), and (c) a full
dry run of all six commands with mocked `sbatch`/`squeue`/`sacct` and
synthetic iterations.csv fixtures standing in for real cluster output --
never a real Slurm submission. Simona still needs to push the branch, log
into Sherlock, and run the six commands for real.

`scripts/cluster/earths_future.sh` is a thin wrapper: it builds and
submits `sbatch` scripts around the existing drivers in `scripts/run/` and
calls the existing `scripts/earths_future_revision/section*.py`
post-processors. `scripts/cluster/earths_future_lib.py` is a small helper
for the JSON-manifest / hashing / CSV-validation / squeue-sacct bookkeeping
a shell script does awkwardly; it never itself calls `sbatch`.

## Prerequisites (once, on Sherlock)

```bash
git clone/pull the earths-future/corrections-and-sensitivity branch to
  /home/users/smeiler/repos/systemic_insurance_risk_fl_pub  (or wherever;
  the wrapper derives PROJECT_DIR from its own location, see below)
conda activate climada_env   # for running `check` interactively; jobs
                              # themselves activate it independently
```

## The six commands

Run all of these from a Sherlock **login or dev node** (`sdev`), never
inside a batch job. `pilot` and `production` submit real Slurm jobs, they
do not run the science themselves.

### 1. `check`
```bash
scripts/cluster/earths_future.sh check
```
Read-only. Reports, without changing anything:
- whether the current commit contains the reviewed FHCF patch
  (`ef7706513e9437a47c33dc6b4972297f8c235975`) and whether tracked source
  files (`fl_risk_model/`, `scripts/`) are dirty (untracked/ignored output
  under `results/` does not count as dirty),
- whether `climada_env` is active,
- whether all 26 required event-set caches
  (`FL_era5_reanalcal` + 5 GCMs x 5 periods) have both
  `year_sets_N10000_seed42.csv` and `event_metadata.csv` under
  `--impact-root` (default `/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts`,
  override with `EF_IMPACT_ROOT`) -- lists exactly which are missing; never
  regenerates hazard data,
- SHA-256 hashes of all 17 active financial inputs and 8 historical hazard
  files,
- the existing `fl_risk_model/tests/earths_future` suite.

Exits 0 only if every check passes. Full JSON report under
`results/earths_future_revision/fhcf_cluster/reports/check_<timestamp>.json`.
**Do not run `pilot` until this passes** (the wrapper does not currently
hard-block `pilot` on `check` -- Simona should treat a failing `check` as
a stop sign; `production` IS hard-gated, via `pilot-report`).

### 2. `pilot`
```bash
scripts/cluster/earths_future.sh pilot
```
Submits ONE Slurm job (12h, 32G) running
`scripts/earths_future_revision/fhcf_pilot_era5.py` against
`FL_era5_reanalcal`, first 200 year IDs, seed 42, both the
`old_both_bugs` and `both_fixed` FHCF variants (sequentially, same
process, same draws). Writes a fresh, timestamped manifest under
`results/earths_future_revision/fhcf_cluster/manifests/pilot_manifest_<ts>.json`
(symlinked as `pilot_manifest_latest.json`) recording the Slurm job id,
code revision (commit + dirty-tracked-source flag), and hashes of every
active financial/historical-hazard input. Output goes to a fresh
`results/mc_runs_fhcf_patched/pilot_era5_<ts>/`.

### 3. `pilot-report`
```bash
scripts/cluster/earths_future.sh pilot-report   # or: --manifest PATH
```
Checks the pilot job's Slurm state via `squeue`/`sacct`. **Fails (exit 1)
while the job is pending, running, or failed** -- it does not infer
anything from the output directory existing. Once Slurm reports it
completed, compares the `old_both_bugs` and `both_fixed` output
directories: row counts match expectations, no `scenario=='error'` rows,
no duplicate `year_id`s, and the upstream columns (`total_damage_usd`,
`wind_total_usd`, `water_total_usd`, `nfip_borrowed_usd`,
`nfip_claims_paid_usd`) are byte-identical between variants (as they must
be -- only downstream FHCF columns should differ). Full JSON under
`results/earths_future_revision/fhcf_cluster/reports/pilot_report_<ts>.json`.
Note: `status`'s generic per-job `output_valid` field will show `false`
for the pilot job specifically, because it nests two variant
subdirectories rather than one run directory -- that is expected;
`pilot-report`'s own check (above) is the authoritative pass/fail for the
pilot, not `status`.

### 4. `production`
```bash
scripts/cluster/earths_future.sh production [--force] [--concurrency N]
```
Gated: refuses to submit unless `pilot-report` (against the current
`pilot_manifest_latest.json`) passes right now. Also refuses (exit 1) if
`production_manifest_latest.json` already has a submitted/running/completed
job at the *same* code revision and input hashes -- pass `--force` to
resubmit anyway (e.g. to retry failures under the same code+inputs).

Enumerates the fixed, required 106-run inventory (see table below),
verifies the count is exactly 106, then submits it as ONE Slurm array job
(`--array=0-105%N`, default concurrency `N=20`, override with
`--concurrency` or `EF_CONCURRENCY`). Each array task looks up its own
line in a generated task list, `cd`s into the checkout, prints the code
revision it sees **at execution time** (compare against the manifest's
submission-time revision if you suspect the checkout changed while jobs
were queued), activates `climada_env` itself, and runs one driver
invocation into its own dedicated, otherwise-empty output directory under
a fresh `results/mc_runs_fhcf_patched/production_<ts>/`. Writes
`production_manifest_<ts>.json` (symlinked as `production_manifest_latest.json`)
with all 106 jobs recorded (`slurm_job_id` = `<array_job_id>_<task_index>`).

No `git pull` or branch switch happens anywhere in this path. **Do not
edit the checkout while a production array is running** -- the
execution-time revision check exists precisely to catch it if you do.

| Family | Driver | Count | Seasons each |
|---|---|---:|---:|
| ERA5 baseline | `run_emanuel_monte_carlo.py` | 1 | 10,000 |
| ERA5 policies (`market_exit_moderate`, `penetration_major`, `building_codes_major`) | same | 3 | 10,000 |
| GCM baselines (5 GCMs x 5 periods) | same | 25 | 10,000 |
| Building-code sweep (13 levels x 5 GCMs, `ssp245cal`) | `run_climate_buildingcode_sensitivity_windfloods.py` | 65 | 10,000 |
| Historical scenarios (7, excludes `andrew_then_gm`) | `run_historical_scenarios_mc.py` | 7 | 1,000 |
| Insured-fraction sweep (0.1-0.5) | `run_insured_fraction_sensitivity.py` | 5 | 10,000 |
| **Total** | | **106** | |

### 5. `status`
```bash
scripts/cluster/earths_future.sh status              # both manifests
scripts/cluster/earths_future.sh status --manifest PATH
```
Queries `squeue` then `sacct` for every job in the manifest(s) and prints
each job's queue state (`running`, `completed`, `failed`, `pending`,
`unknown (not in squeue or sacct)`, ...) plus, for jobs Slurm reports
completed, whether their output actually validates. Never reports success
from directory existence alone.

### 6. `postprocess`
```bash
scripts/cluster/earths_future.sh postprocess   # or: --manifest PATH
```
Resolves every production job against `production_manifest_latest.json`
(completed in Slurm **and** passing `validate-run`); anything else is
listed as missing, never guessed at or silently substituted. With
whatever is resolved, it runs, into a fresh
`results/earths_future_revision/fhcf_cluster/postprocess_<ts>/` (never the
already-reviewed local-pilot tables elsewhere under
`results/earths_future_revision/`):

- `section5_return_periods.py` and `section6_decomposition.py` (need
  `era5_baseline`),
- `section7_nfip_allocation.py` (independent of MC output),
- `section7_insured_fraction.py` (needs `era5_baseline` + all 5 fractions;
  consolidates each fraction's `iterations_frac_<f>.csv` from its own
  dedicated run directory first, since the driver was run once per
  fraction rather than as one combined sweep),
- `section8_historical_and_variance.py` via `--scenario-map` (as many of
  the 7 historical scenarios as have resolved; reports the actual count),
- `section9_climate_policy_tables.py` via `--scenario-map`, including the
  exceedance-probability table (as many of the 4 ERA5 baseline/policy runs
  as have resolved).

It always prints two fixed reminders, regardless of production coverage,
because nothing currently automates them:
- **building-code curves / offset estimates** (13 levels x 5 GCMs): no
  script under `scripts/analysis/` takes a fresh output-directory
  override; the nearest candidate hardcodes an unrelated path and has no
  CLI. This needs a new or rewritten script before it can be added here.
- **main/SI figures**: driven by `notebooks/probabilistic_risk_analysis_pub.ipynb`,
  which this wrapper does not execute.

Exit code reflects whether every section that had sufficient resolved
input ran cleanly (0) or something is missing/partial/failed (2) --
**not** whether the two reminders above are done. Read the printed
"Ran" / "NOT run" summary; do not treat a `postprocess` exit 0 as "the
publication update is complete."

## Environment overrides

All optional; `check` validates them, it does not assume they are
correct.

| Variable | Default | Meaning |
|---|---|---|
| `EF_PROJECT_DIR` | derived from the wrapper's own location (`../..`) | Sherlock checkout root |
| `EF_IMPACT_ROOT` | `/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts` | impact-cache root |
| `EF_PARTITION` | `serc` | Slurm partition |
| `EF_CONCURRENCY` | `20` | max concurrent `production` array tasks |
| `EF_PYTHON` | `python3` | interpreter for `earths_future_lib.py` itself (submitted jobs always activate `climada_env` independently) |

## What was and was not validated locally

Validated: shell syntax (`bash -n`) and Python syntax; every CLI flag
against the actual driver scripts' argparse definitions; path propagation
from `list-jobs` through the generated array script to each driver's
`--out`/`--out_dir`; manifest read/write/symlink routing; the full
`check -> pilot -> pilot-report -> production -> production` (duplicate
refusal) `-> production --force -> status -> postprocess` sequence
end-to-end with `sbatch`/`squeue`/`sacct` replaced by mocks that never
execute anything and synthetic iterations.csv fixtures standing in for
real driver output; that a pending/running/failed job, or a "completed"
job with a wrong row count or `scenario=='error'` rows, can never be
reported as resolved or passing (`fl_risk_model/tests/earths_future/test_cluster_wrapper.py`,
25 tests); the full existing suite (74 tests total, all passing).

Not validated (cannot be, without cluster access): that the generated
`sbatch` scripts are accepted by the real Sherlock scheduler; that the
real drivers succeed against the real impact caches; wall-clock time and
memory headroom for the `--time`/`--mem` values carried over from the
existing submission scripts; the two `postprocess` reminders above.
