# Catastrophe-bond correction and Sherlock rerun

This campaign retains the corrected FHCF calculation and the principal-based
catastrophe-bond payout approximation. It fixes bond eligibility and beneficiary
allocation. The old code shared industry-triggered payouts across the market,
missed Citizens because its company names did not match, and admitted late or
non-Florida issues. The reviewed inventory maps beneficiaries explicitly and
excludes bonds whose allocation cannot be supported with the available inputs.

Eight selected 2024 issues, totaling USD 2.54 billion, are represented. This is
not the full stock of outstanding bonds. All included issues are indemnity bonds.
For each included bond, the assumed attachment equals principal P; the assumed
payout is min(max(N-P, 0), P), where N is the beneficiary's retained wind loss after
FHCF. These are assumed layer terms, not reconstructed transaction terms. The
layer is applied once to aggregate event or seasonal loss, with no reinstatement.
Credited recoveries cannot exceed retained claims. Mapping errors now stop the
financial calculation instead of silently removing the layer.

## Update the existing Sherlock checkout

From the existing repository directory, with no concurrent jobs using that
checkout, run:

```bash
git fetch origin
git checkout earths-future/corrections-and-sensitivity
git pull --ff-only origin earths-future/corrections-and-sensitivity
conda activate climada_env
```

Use `catbond_revision.sh` below. Calling `earths_future.sh` directly retains the
previous FHCF pilot by default. Existing manuscript edits and old results need
not be deleted. Keep this checkout and its financial inputs unchanged while jobs
are queued or running.

## Check and run the paired pilot

```bash
bash scripts/cluster/catbond_revision.sh check
bash scripts/cluster/catbond_revision.sh pilot
bash scripts/cluster/catbond_revision.sh status
```

Proceed with `pilot` only after `check` passes. The check verifies the environment,
26 existing event-set catalogs, financial input files, and tests. It does not
regenerate hurricane hazards or county losses. The default cache root is
`/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts`; override it with
`export EF_IMPACT_ROOT=/path/to/existing/impacts` if necessary.

The pilot is one Slurm job using the first 200 ERA5 year IDs and seed 42. It
compares the original and reviewed cat-bond implementations through the actual
stochastic production driver, keeping FHCF corrected in both variants. It retains
zero-event years. After Slurm reports completion:

```bash
bash scripts/cluster/catbond_revision.sh pilot-report
```

The report checks row counts, error rows, paired year IDs, and unchanged physical
losses, initial insurance allocation, FHCF reimbursements, and NFIP outcomes.
It also verifies that the current code revision and input hashes match the pilot.
Changes in bond payouts, defaults, FIGA, and Citizens are expected; a passing
report does not mean the numerical results are unchanged.

## Run the full financial campaign

After the paired report passes:

```bash
bash scripts/cluster/catbond_revision.sh production
bash scripts/cluster/catbond_revision.sh status
```

This submits the existing 106-run inventory with at most 20 simultaneous tasks
(default partition `serc`). Use `production --concurrency 10` to lower that limit.
The inventory comprises ERA5 baseline and three policy scenarios; 25 GCM/period
baselines; 65 mid-century SSP2-4.5 physical-loss-reduction runs; seven historical
and sequential scenarios; and five insured-fraction sensitivities. Historical
scenarios use 1,000 realizations each; stochastic runs use the existing 10,000
season catalogs. No new hazard simulations are needed.

New outputs and manifests are separate from the prior FHCF campaign:

- `results/mc_runs_catbond_patched/production_<timestamp>/`
- `results/earths_future_revision/catbond_cluster/manifests/`
- `results/earths_future_revision/catbond_cluster/reports/`

Production is gated on a passing cat-bond pilot at the same code revision and
with unchanged inputs. Accidental duplicate submissions are refused.

## After completion

```bash
bash scripts/cluster/catbond_revision.sh postprocess
```

This validates completed outputs and produces the existing diagnostic tables.
It reports missing or failed runs; partial output is not publication-ready.
Keep the production manifest, reports, and per-run `iterations*.csv` and
`run_config.json` files for the subsequent figure/table and manuscript refresh.
The full publication refresh is a separate step. Existing manuscript values
should remain marked as awaiting replacement until that step is complete.

## Validation and remaining interpretation

The local paired historical comparison used 100 realizations per scenario and
identical upstream draws. The correction lowered mean residual financing needs
in the larger storms but increased them for Irma; default counts also changed.
Those pilot values must not replace the full 1,000-realization published values.

A separate comparison with older archived outputs found occasional differences
of one default when group support leaves an insurer's balance exactly at zero.
The strict default threshold and group-support calculation are unchanged in this
correction. Use the same environment for paired comparisons and review such
boundary cases before interpreting very small changes in default probabilities.
