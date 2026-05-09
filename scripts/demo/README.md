# Demo Script — Expected Output and Usage Guide

## Overview

`run_demo.py` runs a self-contained demonstration of the Florida systemic
insurance risk model.  It uses only publicly redistributable data and synthetic
insurer financial inputs; no licensed S&P Capital IQ or WindRiskTech data are
required.

**Scenario**: Great Miami Hurricane (1926) — the primary illustrative event in
the manuscript (Fig. 2).

**Companies**: All 99 real Florida private homeowners insurers, with synthetic
market shares and surplus (see `demo_data/README.md`).

---

## Quick Start

```bash
# From the repository root:
pip install -e .
python scripts/demo/run_demo.py
```

Optional arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--n_iter` | 100 | Number of Monte Carlo iterations |
| `--seed` | 42 | Integer random seed for reproducibility |
| `--out` | `demo_output` | Output directory (relative to repo root) |

Example:

```bash
python scripts/demo/run_demo.py --n_iter 50 --seed 7 --out my_demo_output
```

---

## What the Demo Does

1. Loads **all 99 real Florida private HO insurers** from `demo_data/` with
   synthetic Pareto-distributed market shares and plausible surplus values.
   Company names and statutory entity keys are from the public regulatory record;
   only the financial figures are synthetic.
2. Uses the **real Great Miami Hurricane (1926) county-level impact footprint**
   (`fl_risk_model/data/hazard/historical_events/1926255N15314.csv`) derived
   from IBTrACS track data via CLIMADA (see
   `scripts/hazard/simulate_historical_event_losses.py` for full reproduction).
3. Runs the full insurance waterfall `n_iter` times (default: 100) with a
   Monte Carlo wind/water split sampled from the event's Beta prior
   (mean = 0.70 wind share, concentration = 10).
4. Applies the FHCF, Citizens, NFIP, and FIGA institutional layers using
   publicly available terms and balance-sheet data.
5. Saves per-iteration results to `demo_output/demo_summary.csv`.

---

## Approximate Runtime

Measured on Apple M-series MacBook: **~28 s for 100 iterations** (~0.28 s/iteration).
Runtime scales approximately linearly with `--n_iter`.

---

## Expected Output

With default settings (`--n_iter 100 --seed 42`) the console prints:

```
============================================================
  Florida Systemic Insurance Risk Model — Demo Run
============================================================
  Scenario     : Great Miami Hurricane (1926)
  IBTrACS ID   : 1926255N15314
  Companies    : 99 FL private homeowners insurers
  Financials   : SYNTHETIC (illustrative only — see demo_data/)
  Iterations   : 100
  Seed         : 42

Loading demo data...
  Loaded 99 private insurers
  Total synthetic entity surplus: $18.2B
Building model inputs (FHCF, Citizens, NFIP, county xwalk)...
Running 100 iterations...
  Completed  25/100 iterations  (6.5s elapsed)
  Completed  50/100 iterations  (13.4s elapsed)
  Completed  75/100 iterations  (20.2s elapsed)
  Completed 100/100 iterations  (27.5s elapsed)

  All 100 iterations completed in 27.5s

  Results saved to: demo_output/demo_summary.csv

============================================================
  Demo Results Summary
============================================================
  ...
  Systemic stress (% of iterations):
    Any private default  : 100.0% of iterations
    Defaults > 10 firms  : 100.0% of iterations
    FHCF shortfall > 0   : 0.0% of iterations
    Citizens deficit > 0 : 100.0% of iterations
    NFIP borrowing > 0   : 96.0% of iterations

  IMPORTANT: These outputs use SYNTHETIC insurer financial data.
  Company names are real (public regulatory record); market shares
  and surplus values are illustrative only.  See demo_data/README.md.
  Results do NOT match the manuscript figures or tables.
============================================================
```

### Key Qualitative Expectations

With default settings (`--seed 42`) the measured results are:

| Metric | Mean (100 iters) |
|--------|------------------|
| Total gross damage | $170.4 B |
| Wind damage | $123.6 B |
| Water/flood damage | $46.8 B |
| Private wind insured | $38.9 B |
| Citizens wind insured | $10.4 B |
| NFIP flood paid | $13.8 B |
| Any private default | 100% of iterations |
| Defaults > 10 firms | 100% of iterations |
| FHCF shortfall | $0 (capacity not exhausted) |
| Citizens deficit | $5.3 B |
| NFIP borrowing | $10.4 B (96% of iterations) |
| FIGA residual deficit | $15.1 B |

The FHCF pays out a mean of **$8.4 B** total (private + Citizens layers) but
stays within its statutory capacity — so shortfall is zero.  Private defaults
are driven by net losses after FHCF recovery (~$34.6 B) exceeding the $18.2 B
total synthetic entity surplus.  These patterns are consistent with the
manuscript's core finding.  Exact numbers differ because the demo uses synthetic
financial data.

### Exact Numeric Values

Because the demo uses **synthetic** insurer data, the exact numbers differ from
the manuscript.  Two replications with the **same seed** on the same platform
will produce **identical** results.  Minor floating-point differences across
different OS / NumPy versions are possible.

The first run saves `demo_output/expected_demo_summary.csv` as a reference
snapshot.  Delete this file to regenerate it.

---

## Hazard Pipeline (Optional)

`scripts/hazard/simulate_historical_event_losses.py` is the actual script used
to generate the pre-computed county impact footprints from IBTrACS tracks via
CLIMADA (Holland 2008 wind model, RMSF-calibrated impact functions):

```bash
conda activate climada_env
python scripts/hazard/simulate_historical_event_losses.py
```

This script requires CLIMADA ≥ 4.0 and internet access (downloads IBTrACS
and LitPop tiles on first run, ~500 MB).  It is **not required** for the core
demo — the pre-computed footprint is already in the repository.

---

## Output File Schema

`demo_output/demo_summary.csv` contains one row per iteration with columns
including (not limited to):

| Column | Description |
|--------|-------------|
| `iteration` | 0-based iteration index |
| `scenario` | Always `"great_miami"` in the demo |
| `total_damage_usd` | Total gross damage across all perils (USD) |
| `wind_total_usd` | Total wind damage (USD) |
| `water_total_usd` | Total water/flood damage (USD) |
| `wind_insured_private_usd` | Wind losses absorbed by private insurers (USD) |
| `wind_insured_citizens_usd` | Wind losses absorbed by Citizens (USD) |
| `flood_insured_capped_usd` | Flood losses absorbed by NFIP (USD) |
| `wind_underinsured_usd` | Uninsured/underinsured wind losses (USD) |
| `flood_underinsured_usd` | Uninsured/underinsured flood losses (USD) |
| `defaults_pre` | Private insurer defaults before intragroup support |
| `defaults_post` | Private insurer defaults after intragroup support |
| `fhcf_shortfall_usd` | FHCF reinsurance shortfall (USD) |
| `nfip_borrowed_usd` | NFIP federal borrowing required (USD) |
| `citizens_residual_deficit_usd` | Citizens residual deficit after assessments (USD) |
| `figa_residual_deficit_usd` | FIGA residual deficit (USD) |

Additional diagnostic columns may be present depending on the model version.

---

## Relationship to Manuscript

The demo is **illustrative only**.  To reproduce the manuscript figures exactly,
use the precomputed results in `results/` with the notebooks described in
`notebooks/` — see `results/README.md` for full instructions.

---

## Troubleshooting

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: No module named 'fl_risk_model'` | Package not installed | `pip install -e .` from repo root |
| `Missing required files: fhcf_exposure` | FHCF xlsx absent | Ensure full repo clone |
| `Missing required files: great_miami_event` | Hazard data absent | Ensure full repo clone |
| `KeyError: 'Share'` | Wrong CSV column names in demo_data/ | Check `demo_data/demo_market_share.csv` has `MarketShare2024` column |
| Very slow runtime | Flood enabled, slow NFIP load | Normal; runtime scales ~linearly with iterations |
