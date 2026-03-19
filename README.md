# Florida Systemic Insurance Risk Model

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Stress Testing Financial Stability of Florida's Insurance Market Under Climate Change**

This repository contains the code and data accompanying the paper:

> Meiler, S., et al. (2026). *[Paper Title]*. [Journal]. DOI: [PLACEHOLDER]

---

## Overview

This model assesses whether major tropical cyclone catastrophes in Florida could generate systemic financial risks across insurers, homeowners, mortgage lenders, capital markets, and public finance institutions. 

The model provides:

1. **County-level wind/water damage attribution** — Physics-based split of tropical cyclone losses for Florida's 67 counties
2. **Climate change projections** — Future damage scaling under SSP2-4.5 (2070–2100)
3. **Insurance market stress testing** — Monte Carlo simulation of insurer defaults, NFIP claims, Citizens exposure, and FHCF capacity
4. **Policy scenario analysis** — Evaluation of building codes, market exit, and penetration changes

### Key Features

- **Kerry Emanuel TC event sets** — 10,000-year probabilistic hazard based on ERA5 and CMIP6 GCMs
- **Sequential event modeling** — Multi-event season impacts on building stock
- **Multi-branch loss allocation** — Private wind insurers, NFIP (flood), Citizens, FHCF, cat bonds, uninsured
- **Empirically calibrated** — Uses Florida OIR financial data, NFIP claims, and county-level exposure

---

## Repository Structure

```
systemic_insurance_risk_fl/
├── README.md                     # This file
├── LICENSE                       # Apache 2.0 license
├── CITATION.cff                  # Citation metadata
├── pyproject.toml                # Package configuration
│
├── fl_risk_model/                # Core Python package
│   ├── config.py                 # Configuration and paths
│   ├── runner.py                 # Main simulation runner
│   ├── mc_run_events.py          # Monte Carlo engine
│   ├── capital.py                # Insurer capital modeling
│   ├── fhcf.py                   # Florida Hurricane Catastrophe Fund
│   ├── nfip.py                   # National Flood Insurance Program
│   ├── catbonds.py               # Catastrophe bonds
│   ├── branches/                 # Loss allocation modules
│   ├── scenarios/                # Policy scenarios
│   ├── loss_calc_utils/          # Impact function utilities
│   └── data/                     # Input data files
│
├── scripts/
│   ├── run/                      # Main simulation scripts
│   ├── analysis/                 # Results analysis
│   ├── hazard/                   # Hazard preprocessing
│   └── cluster/                  # HPC job scripts
│
├── notebooks/                    # Analysis & visualization notebooks
│   ├── emanuel_tc_policy_analysis.ipynb
│   ├── climate_buildingcode_windfloods_analysis.ipynb
│   ├── policy_scenario_analysis.ipynb
│   └── hurricane_loss_maps.ipynb
│
└── docs/                         # Documentation
```

---

## Installation

### Requirements
- Python ≥ 3.9
- CLIMADA (for hazard modeling)

### Setup

```bash
# Clone repository
git clone https://github.com/simonameiler/systemic_insurance_risk_fl_pub.git
cd systemic_insurance_risk_fl_pub

# Create conda environment
conda create -n fl_risk python=3.11
conda activate fl_risk

# Install CLIMADA
conda install -c conda-forge climada

# Install this package
pip install -e .
```

---

## Quick Start

### Run Monte Carlo Simulation

```bash
# Run with ERA5 baseline climate (1000 years)
python scripts/run/run_emanuel_monte_carlo.py \
    --event_set FL_era5_reanalcal \
    --n_years 1000

# Run with climate change scenario
python scripts/run/run_emanuel_monte_carlo.py \
    --event_set FL_canesm5_ssp245_2081-2100

# Run with policy scenario
python scripts/run/run_emanuel_monte_carlo.py \
    --event_set FL_era5_reanalcal \
    --policy building_codes_major
```

### Run Policy Comparison Suite

```bash
python scripts/run/run_emanuel_policy_suite.py \
    --event_set FL_era5_reanalcal \
    --policies baseline building_codes_major market_exit_moderate
```

### Sensitivity Analyses

```bash
# Insured wind fraction sensitivity
python scripts/run/run_insured_fraction_sensitivity.py

# Variance decomposition (ANOVA)
python scripts/run/run_variance_decomposition.py --mode nested --M 300 --K 50
```

---

## Data Availability

### Included Data
- FHCF terms and reimbursement structure
- Citizens Property Insurance county-level data
- NFIP penetration rates and claims statistics
- County geographic reference files
- Impact function parameters
- Pre-computed per-event impacts (historical climate)

### Large Data Files (Not Included)
Due to GitHub file size limits (~166 MB):
- `fl_risk_model/data/fl_per_event_impacts_future_ssp245.csv`

Generate with: `python scripts/hazard/build_per_event_impacts.py --climate ssp245`

### External Data Requirements
The following must be obtained separately:

**Kerry Emanuel TC Tracks** (contact authors):
- `.mat` files with synthetic TC tracks for ERA5 and CMIP6 GCMs
- Required for windfield generation

**CLIMADA Exposure** (auto-downloaded):
- LitPop exposure for Florida (downloaded by CLIMADA on first run)

---

## Complete Workflow

The full pipeline from TC tracks to final analysis:

```
TC Tracks (.mat) → Windfields → Per-Event Impacts → Year-Sets → Monte Carlo → Analysis
```

### Step 1: Generate Windfields (requires CLIMADA + Emanuel tracks)
Place Emanuel track files in `$CLIMADA_DATA/tracks/Kerry/Florida/`, then:
```bash
python scripts/hazard/compute_windfields_emanuel.py Simona_FLA_AL_era5_reanalcal.mat
```
Output: `$CLIMADA_DATA/hazard/Florida/FL_era5_reanalcal.hdf5`

### Step 2: Compute Per-Event County Impacts
```bash
python scripts/hazard/precompute_emanuel_tc_impacts.py FL_era5_reanalcal
```

### Step 3: Generate Year-Sets
```bash
python scripts/hazard/generate_emanuel_year_sets.py \
    --event_set FL_era5_reanalcal --n_years 10000 --seed 42
```

### Step 4: Run Monte Carlo
```bash
python scripts/run/run_emanuel_monte_carlo.py \
    --event_set FL_era5_reanalcal --n_years 10000
```

### Step 5: Analysis
Open notebooks in `notebooks/` to reproduce paper figures.

---

## Reproducing Paper Results

The main analysis notebooks reproduce all figures and tables in the paper:

| Figure/Table | Notebook |
|--------------|----------|
| Fig. 1: Loss return periods | `emanuel_tc_policy_analysis.ipynb` |
| Fig. 2: Institutional stress | `policy_scenario_analysis.ipynb` |
| Fig. 3: Policy divergence | `emanuel_tc_policy_analysis.ipynb` |
| Fig. 4: Climate scenarios | `emanuel_tc_policy_analysis.ipynb` |
| Fig. 5: Building code sensitivity | `climate_buildingcode_windfloods_analysis.ipynb` |
| Table 1: Baseline metrics | `emanuel_tc_policy_analysis.ipynb` |
| Table 2: Policy comparison | `emanuel_tc_policy_analysis.ipynb` |
| SI: Variance decomposition | `scripts/run/run_variance_decomposition.py` |
| SI: Insured fraction sensitivity | `scripts/run/run_insured_fraction_sensitivity.py` |

---

## Citation

If you use this code, please cite:

```bibtex
@article{meiler2026florida,
  title = {{Stress Testing Financial Stability of Florida's Insurance Market Under Climate Change}},
  author = {Meiler, Simona and [co-authors]},
  journal = {[Journal]},
  year = {2026},
  doi = {PLACEHOLDER}
}
```

See also [CITATION.cff](CITATION.cff) for citation metadata.

---

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work was developed under the U.S. Federal Reserve's IPAC Climate Working Group. We thank Kerry Emanuel for providing the tropical cyclone event sets, and the Florida Office of Insurance Regulation for data access.

---

## Contact

Simona Meiler — [email] — [institution]
