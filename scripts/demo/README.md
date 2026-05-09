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
   from IBTrACS track data via CLIMADA (see `run_climada_hazard_pipeline.py`
   for full reproduction).
3. Runs the full insurance waterfall `n_iter` times (default: 100) with a
   Monte Carlo wind/water split sampled from the event's Beta prior
   (mean = 0.70 wind share, concentration = 10).
4. Applies the FHCF, Citizens, NFIP, and FIGA institutional layers using
   publicly available terms and balance-sheet data.
5. Saves per-iteration results to `demo_output/demo_summary.csv`.

---

## Approximate Runtime

| Hardware | 100 iterations | 1 000 iterations |
|----------|---------------|-----------------|
| Modern laptop (Apple M-series or Intel i7) | ~2–5 min | ~20–40 min |
| Older laptop / CI machine | ~5–10 min | ~60–90 min |

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
  Completed  25/100 iterations  (Xs elapsed)
  Completed  50/100 iterations  (Xs elapsed)
  Completed  75/100 iterations  (Xs elapsed)
  Completed 100/100 iterations  (Xs elapsed)

  All 100 iterations completed in Xs

  Results saved to: demo_output/demo_summary.csv

============================================================
  Demo Results Summary
============================================================
  ...
  Systemic stress (% of iterations):
    Any private default  : ~95–100% of iterations
    Defaults > 10 firms  : ~70–90% of iterations
    FHCF shortfall > 0   : ~80–100% of iterations
    ...

  IMPORTANT: These outputs use SYNTHETIC insurer financial data.
  Company names are real (public regulatory record); market shares
  and surplus values are illustrative only.  See demo_data/README.md.
  Results do NOT match the manuscript figures or tables.
============================================================
```

### Key Qualitative Expectations

The 1926 Great Miami Hurricane is the paper's primary "extreme stress" scenario.
At 2024 Florida exposure levels, this event is expected to generate:

- **Total gross damage**: $150–200 B USD (mean across 100 iterations).
- **Private wind insured losses**: $35–45 B USD (mean) — substantially larger
  than the $18.2 B total synthetic entity surplus.
- **Private insurer defaults**: most or all iterations will show defaults, because
  mean private wind losses exceed total sector surplus.  Expect 15–25 defaults
  per iteration on average.
- **FHCF near-exhaustion**: common in most iterations.
- **Citizens residual deficit**: frequent given the scale of the event.
- **NFIP borrowing**: common but smaller magnitude than wind losses.

These qualitative patterns are consistent with the manuscript's core finding
that a Great Miami–class event would exhaust private-sector capacity.  Exact
numbers differ because the demo uses synthetic financial data.

### Exact Numeric Values

Because the demo uses **synthetic** insurer data, the exact numbers differ from
the manuscript.  Two replications with the **same seed** on the same platform
will produce **identical** results.  Minor floating-point differences across
different OS / NumPy versions are possible.

The first run saves `demo_output/expected_demo_summary.csv` as a reference
snapshot.  Delete this file to regenerate it.

---

## Hazard Pipeline (Optional)

`run_climada_hazard_pipeline.py` shows how the Great Miami impact footprint
was computed from the raw IBTrACS track using CLIMADA:

```bash
conda activate climada_env
python scripts/demo/run_climada_hazard_pipeline.py
```

This script requires CLIMADA ≥ 6.1.0 and internet access (downloads IBTrACS
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
