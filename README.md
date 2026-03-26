# Florida Systemic Insurance Risk Model

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

These scripts reproduce the main results of the paper:

**Simona Meiler**(1), Steven I. Jackson (2), Kerry Emanuel (3), Noah S. Diffenbaugh (4), Jack W.
Baker (1): *[Stress testing insurance market stability under climate-driven disaster risk]*

Publication status: [under review]

(1) Civil and Environmental Engineering, Stanford University, CA, USA
(2) American Academy of Actuaries, Washington, DC, USA
(3) Lorenz Center, Massachusetts Institute of Technology, Cambridge, Massachusetts, USA
(4) Earth System Science, Stanford University, CA, USA

---

## Repository structure

```
systemic_insurance_risk_fl_pub/
│
├── README.md                          ← you are here
├── LICENSE                            ← GPL-3.0
├── pyproject.toml                     ← package metadata & dependencies
│
├── fl_risk_model/                     ← core model package
│   ├── config.py                      ← central configuration, file paths, constants
│   ├── loader.py                      ← data loading for exposure, premium, Citizens, NFIP
│   ├── exposure.py                    ← wind exposure matrix construction (company × county)
│   ├── runner.py                      ← single-event scenario runner (exposure → loss → capital)
│   ├── mc_run_events.py               ← Monte Carlo event simulator (systemic risk propagation)
│   ├── capital.py                     ← capital depletion, group support, surplus sampling
│   ├── catbonds.py                    ← catastrophe bond pricing and recovery
│   ├── fhcf.py                        ← Florida Hurricane Catastrophe Fund recoveries
│   ├── nfip.py                        ← NFIP payouts and Empirical-Bayes payout rates
│   ├── utils.py                       ← county name normalization helper
│   │
│   ├── branches/                      ← risk waterfall implementation
│   │   ├── wind.py                    ← private wind losses by company × county + FHCF
│   │   ├── citizens.py                ← Citizens Property Insurance wind losses + FHCF
│   │   ├── flood.py                   ← NFIP flood losses and recoveries
│   │   └── uninsured.py               ← uninsured / underinsured loss accounting
│   │
│   ├── scenarios/                     ← policy & adaptation scenario transforms
│   │   ├── market_exit.py             ← private insurer withdrawal to Citizens
│   │   ├── penetration.py             ← insurance take-up / NFIP expansion
│   │   └── building_codes.py          ← wind/flood loss reduction from stricter codes
│   │
│   └── data/                          ← input datasets (see Data Availability)
│       ├── *.csv / *.xlsx             ← FHCF terms, Citizens, NFIP, county mappings, etc.
│       ├── hazard/                    ← per-event impacts and historical scenarios
│       │   ├── gori_data/             ← Gori et al. 2025 hazard matrices (Wind, Rain, Surge, EAD)
│       │   ├── emanuel/               ← Kerry Emanuel TC event sets (ERA5 baseline)
│       │   └── historical_events/     ← individual historical hurricane impact CSVs
│       └── US_counties/               ← Florida county shapefiles
│
├── scripts/
│   ├── run/                           ← Monte Carlo run scripts
│   │   ├── run_emanuel_monte_carlo.py             ← baseline MC with Emanuel TC event sets
│   │   ├── run_emanuel_policy_suite.py            ← side-by-side policy scenario comparisons
│   │   ├── run_historical_scenarios_mc.py         ← 8 historical scenarios (200 MC iterations)
│   │   ├── run_climate_buildingcode_sensitivity_windfloods.py  ← building code × climate sensitivity
│   │   ├── run_penetration_capital_sensitivity.py ← capital multiplier sweep under penetration increase
│   │   ├── run_insured_fraction_sensitivity.py    ← insured fraction sweep (0.1–0.5)
│   │   └── run_variance_decomposition.py          ← hazard vs. parameter variance decomposition
│   │
│   ├── analysis/                      ← post-processing & table/figure generation
│   │   ├── analyze_emanuel_comprehensive.py       ← loss composition, institutional stress analysis
│   │   ├── analyze_penetration_capital_sensitivity.py ← optimal capital multiplier identification
│   │   ├── build_scenario_report_with_uncertainty.py  ← Excel reports with uncertainty bands
│   │   ├── combine_probabilistic_tables.py        ← unify probabilistic loss tables
│   │   ├── combine_systemic_risk_tables.py        ← unify systemic risk comparison tables
│   │   ├── compute_climate_deltas.py              ← GCM ensemble climate change deltas
│   │   └── generate_si_table_insured_frac.py      ← SI table for insured fraction sensitivity
│   │
│   ├── hazard/                        ← hazard data preprocessing
│   │   ├── build_per_event_impacts.py             ← log-linear damage model from Gori hazard matrices
│   │   ├── calculate_empirical_hazard_attribution.py ← wind/water attribution (P95 weighted)
│   │   ├── generate_log_contribution_from_mat_files.py ← wind/water attribution from raw .mat files
│   │   ├── compute_sequential_events.py           ← multi-hurricane scenario builder
│   │   ├── compute_windfields_emanuel.py          ← CLIMADA windfields from Emanuel TC tracks
│   │   ├── precompute_emanuel_tc_impacts.py       ← county-level impact precomputation
│   │   ├── generate_emanuel_year_sets.py          ← stochastic year-set generation
│   │   └── setup_emanuel_metadata.py              ← event metadata for year-set generation
│   │
│   └── cluster/                       ← SLURM job submission scripts (Stanford Sherlock)
│
├── notebooks/                         ← reproduce all publication figures and tables
│   ├── historical_scenario_analysis.ipynb         ← Fig. X, SI Table X, SI Fig. X
│   └── probabilistic_risk_analysis.ipynb          ← Fig. X–X, Table X, SI Tables X–X, SI Figs. X–X
│
└── results/                           ← output directories (MC runs, figures, tables)
    ├── figures/
    ├── tables/
    ├── mc_runs/
    └── emanuel_mc_runs/
```

## Content

### `fl_risk_model/`

Core Python package implementing a probabilistic catastrophe model for stress-testing Florida's property insurance system.
The model simulates hurricane events through the full insurance waterfall: wind/flood exposure allocation, FHCF and NFIP recoveries, private insurer capital depletion, Citizens Property Insurance backstop, and catastrophe bond losses.
Policy scenarios (market exit, penetration increase, building code improvements) modify the system state before simulation.

### `scripts/run/`

Monte Carlo simulation scripts. Each script configures and launches `mc_run_events.run_stochastic_tc_monte_carlo()` with different parameter sweeps or event sets.
Computationally demanding runs are designed for HPC execution via the corresponding SLURM scripts in `scripts/cluster/`.

### `scripts/analysis/`

Post-processing scripts that read Monte Carlo output and produce summary tables, comparison reports, and figures for the publication.

### `scripts/hazard/`

Preprocessing scripts that convert raw hazard data (Gori et al. matrices, Kerry Emanuel TC tracks) into per-event county-level impact tables consumed by the risk model.
These scripts require [CLIMADA](https://github.com/CLIMADA-project/climada_python) and, for windfield computation, an HPC cluster.

### `scripts/cluster/`

SLURM job submission scripts for the Stanford Sherlock HPC cluster - not included in this repository.

### `notebooks/`

Jupyter notebooks that reproduce all figures and tables in the publication from pre-computed Monte Carlo results.

### `fl_risk_model/data/`

Input data files. See **Data Availability** below for details on public vs. proprietary data.

---

## Installation

```bash
git clone https://github.com/simonameiler/systemic_insurance_risk_fl_pub.git
cd systemic_insurance_risk_fl_pub
pip install -e .
```

The model is designed to run within a [CLIMADA](https://github.com/CLIMADA-project/climada_python) conda environment (v6.1.0+), which provides all required dependencies (numpy, pandas, scipy, matplotlib, h5py, etc.).
CLIMADA itself is only needed for hazard preprocessing (`scripts/hazard/`); we used v6.1.0, develop; the core risk model and Monte Carlo runs do not depend on it.

---

## Reproducing paper results

**Figures and tables** are generated by the two notebooks in `notebooks/`:

| Notebook | Content |
|----------|---------|
| `historical_scenario_analysis.ipynb` | Fig. X, SI Table X, SI Fig. X |
| `probabilistic_risk_analysis.ipynb` | Fig. X–X, Table X, SI Tables X–X, SI Figs. X–X |

**Monte Carlo runs** are launched via `scripts/run/` (locally or on a cluster via `scripts/cluster/`).
Pre-computed results used in the notebooks are stored in `results/`.

---

## Data availability

**Included in the repository**: FHCF contract terms, Citizens Property Insurance county data, NFIP claims and penetration rates, county mappings, per-event impacts, catastrophe bond terms, and historical hurricane scenario impacts.

**Proprietary** (contact authors): Florida company-level wind exposure data, statutory surplus capital data. See `fl_risk_model/config.py` for details.

**External**: The synthetic TC data from the MIT model are proprietary and owned by WindRiskTech L.L.C., a company that provides hurricane risk assessments to clients worldwide. Due to proprietary restrictions, these datasets are not publicly archived. However, researchers interested in accessing the data for scientific purposes can contact WindRiskTech L.L.C. at info@windrisktech.com, subject to a non-redistribution agreement.

---

## Requirements

- Python 3.11+ (recommended: use the CLIMADA conda environment)
- [CLIMADA](https://github.com/CLIMADA-project/climada_python) v6.1.0+ (only for hazard preprocessing)
- HPC cluster

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
