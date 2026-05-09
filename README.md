# Florida Systemic Insurance Risk Model

[![DOI](https://zenodo.org/badge/1185729011.svg)](https://doi.org/10.5281/zenodo.19361127)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

These scripts reproduce the main results of the paper:

**Simona Meiler**(1), Steven I. Jackson (2), Kerry Emanuel (3), Noah S. Diffenbaugh (4), Jack W.
Baker (1): *Stress testing insurance market stability under climate risk*

A preprint is available here: [https://doi.org/10.31223/X59X9X](https://doi.org/10.31223/X59X9X)

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
│   ├── branches/                      ← risk propagation implementation
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
│       ├── *.csv                      ← FHCF terms, Citizens, NFIP, county mappings, wind/water attribution
│       ├── FHCF_2024_Exposure_byCounty.xlsx  ← FHCF county exposure (public regulatory filing)
│       ├── hazard/                    ← per-event impacts and historical scenarios
│       │   ├── gori_data/             ← placeholder; raw Gori et al. 2025 .mat files not redistributed (see Data Availability)
│       │   ├── emanuel/               ← placeholder; precomputed Emanuel TC impacts available on request (see Data Availability)
│       │   └── historical_events/     ← county-level wind damage for 4 model events + 4 composite scenarios
│       └── US_counties/               ← Florida county shapefiles
│
├── demo_data/                         ← synthetic insurer data for the demo (see demo_data/README.md)
├── demo_output/                       ← reference output for the demo (demo_summary.csv)
│
├── scripts/
│   ├── demo/                          ← self-contained model demo
│   │   └── run_demo.py                ← Great Miami Hurricane, 99 synthetic insurers (~30 s)
│   │
│   ├── run/                           ← Monte Carlo run scripts
│   │   ├── run_emanuel_monte_carlo.py             ← baseline MC with Emanuel TC event sets
│   │   ├── run_emanuel_policy_suite.py            ← side-by-side policy scenario comparisons
│   │   ├── run_historical_scenarios_mc.py         ← 8 historical scenarios
│   │   ├── run_climate_buildingcode_sensitivity_windfloods.py  ← building code × climate sensitivity
│   │   ├── run_insured_fraction_sensitivity.py    ← insured fraction sweep (0.1–0.5), SI Table 6
│   │   └── run_variance_decomposition.py          ← hazard vs. parameter variance decomposition, SI Table 7
│   │
│   ├── analysis/                      ← post-processing & table/figure generation
│   │   ├── analyze_emanuel_comprehensive.py       ← loss composition, institutional stress analysis
│   │   ├── build_scenario_report_with_uncertainty.py  ← Excel reports with uncertainty bands
│   │   ├── combine_probabilistic_tables.py        ← unify probabilistic loss tables
│   │   ├── combine_systemic_risk_tables.py        ← unify systemic risk comparison tables
│   │   ├── compute_climate_deltas.py              ← GCM ensemble climate change deltas
│   │   ├── generate_si_table_insured_frac.py      ← SI table for insured fraction sensitivity
│   │   ├── generate_si_table_variance_decomp.py   ← SI table for variance decomposition
│   │   └── generate_table_baseline_return_periods.py ← baseline return period table
│   │
│   ├── hazard/                        ← hazard data preprocessing
│   │   ├── simulate_historical_event_losses.py    ← IBTrACS→CLIMADA pipeline for the 4 historical events
│   │   ├── generate_log_contribution_from_mat_files.py ← wind/water attribution (Gori log-contribution, P95)
│   │   ├── compute_sequential_events.py           ← multi-hurricane scenario builder
│   │   ├── compute_windfields_emanuel.py          ← CLIMADA windfields from Emanuel TC tracks
│   │   ├── precompute_emanuel_tc_impacts.py       ← county-level impact precomputation
│   │   ├── generate_emanuel_year_sets.py          ← stochastic year-set generation
│   │   └── setup_emanuel_metadata.py              ← event metadata for year-set generation
│   │
│   └── cluster/                       ← SLURM job submission scripts (Stanford Sherlock)
│
├── notebooks/                         ← reproduce all publication figures and tables
│   ├── historical_scenario_analysis.ipynb         ← Fig. 2, SI Fig. 2
│   └── probabilistic_risk_analysis_pub.ipynb      ← Fig. 3-4, Table 1, SI Tables 4-5, SI Fig. 1
│
└── results/                           ← pre-computed outputs for notebook reproduction
    ├── figures/                       ← publication figures (PDF + PNG)
    ├── tables/                        ← publication tables (CSV + LaTeX)
    ├── climate_deltas/                ← GCM ensemble climate change delta CSVs
    └── mc_runs/                       ← archived MC run directories
        ├── emanuel_era5_baseline_*/                ← ERA5 baseline (10,000 yr)
        ├── emanuel_era5_market_exit_*/             ← market-exit policy scenario
        ├── emanuel_era5_penetration_*/             ← flood penetration scenario
        ├── emanuel_era5_building_codes_*/          ← ERA5 building code scenario
        ├── emanuel_{canesm,cnrm6,ecearth6,ipsl6,miroc6}_{20thcal,ssp245cal,ssp245_2cal,ssp585cal,ssp585_2cal}_baseline_*/  ← 5 GCMs × 5 period/pathway combinations
        ├── emanuel_{gcm}_ssp245cal_buildingcode_{params}_*/  ← 11-level building code sweep per GCM
        └── {great_miami,andrew,irma,lake_okeechobee,...}_*/  ← 8 historical scenarios (gitignored; run locally)
```

## Content

### `fl_risk_model/`

Core Python package implementing a probabilistic risk propagation model for stress-testing Florida's property insurance system.
The model simulates hurricane events through the full insurance market: wind/flood exposure allocation, FHCF and NFIP recoveries, private insurer capital depletion, Citizens Property Insurance backstop, and catastrophe bond losses.
Stylized policy scenarios (market exit, penetration increase, building code improvements) modify the system state before simulation.

### `scripts/run/`

Monte Carlo simulation scripts. Each script configures and launches `mc_run_events.run_stochastic_tc_monte_carlo()` with different parameter sweeps or event sets.
Computationally demanding runs are designed for HPC execution via the corresponding SLURM scripts in `scripts/cluster/`.
Two scripts produce the sensitivity analyses reported in the Supplementary Information tables: `run_insured_fraction_sensitivity.py` (insured wind fraction sweep) and `run_variance_decomposition.py` (hazard vs. parameter variance decomposition).

### `scripts/analysis/`

Post-processing scripts that read Monte Carlo output and produce summary tables, comparison reports, and figures for the publication.

### `scripts/hazard/`

Preprocessing scripts that convert raw hazard data (IBTrACS, Kerry Emanuel TC tracks) into per-event county-level impact tables consumed by the risk model.
These scripts require [CLIMADA](https://github.com/CLIMADA-project/climada_python) and, for windfield computation, an HPC cluster.

### `scripts/cluster/`

SLURM job submission scripts for the Stanford Sherlock HPC cluster - not included in this repository.

### `notebooks/`

Jupyter notebooks that reproduce all figures and some tables in the publication from pre-computed Monte Carlo results.

### `fl_risk_model/data/`

Input data files. See **Data Availability** below for details.

---

## System requirements

### Operating system

macOS 12+ or Linux (Ubuntu 20.04+). Windows is not tested.

### Python version

Python 3.11 or 3.12 (recommended). Python 3.10 may work but is untested.

### Required Python packages

The following packages are installed automatically via `pip install -e .`:

| Package | Purpose |
|---------|---------|
| `numpy ≥ 1.26` | Numerical arrays and random sampling |
| `pandas ≥ 2.0` | Tabular data loading and manipulation |
| `scipy ≥ 1.12` | Statistical distributions (Beta, log-normal) |
| `matplotlib ≥ 3.8` | Figure generation in notebooks |
| `openpyxl ≥ 3.1` | Reading FHCF and insurer Excel files |
| `tqdm` | Progress bars in long MC runs |
| `jupyter` | Running the analysis notebooks |

See `pyproject.toml` for the full pinned dependency list.

### CLIMADA (hazard preprocessing only)

[CLIMADA](https://github.com/CLIMADA-project/climada_python) v6.1.0+ is required **only** to re-run the hazard preprocessing pipeline (`scripts/hazard/`).  CLIMADA is **not** required to run the demo, the notebooks, or the Monte Carlo risk model.  We used CLIMADA 6.1.0-develop.

### HPC cluster (full upstream computation only)

Monte Carlo runs with the complete synthetic TC event sets (Emanuel model, ~200 000 storm years) were executed on the Stanford Sherlock HPC cluster using SLURM job arrays.  A standard laptop or desktop is sufficient for the demo and notebook reproduction steps.

### Non-standard hardware

No GPUs or other specialised hardware are required for any step in this repository.

---

## Installation

```bash
git clone https://github.com/simonameiler/systemic_insurance_risk_fl_pub.git
cd systemic_insurance_risk_fl_pub
pip install -e .
```

Typical installation time: 2–10 minutes depending on network speed and whether
numpy/scipy need to be compiled.

The model is designed to also run inside a [CLIMADA](https://github.com/CLIMADA-project/climada_python) conda environment, which provides all required dependencies.

---

## Peer review reproducibility

There are **three levels of reproducibility**, depending on available data:

### (a) Reproducing manuscript figures and tables from archived results — no proprietary data needed

All pre-computed Monte Carlo outputs needed to reproduce the paper's figures and
tables are archived in `results/`.  No proprietary data, HPC access, or long
computation is required.

```bash
pip install -e .
jupyter lab notebooks/historical_scenario_analysis.ipynb      # Fig. 2, SI Fig. 2
jupyter lab notebooks/probabilistic_risk_analysis_pub.ipynb   # Fig. 3-4, Table 1, SI Tables 4-5, SI Fig. 1
```

Run all cells from top to bottom.  See `results/README.md` for a detailed
mapping of notebooks to figures and an explanation of the archived directory
structure.

> **Note**: The notebooks load results from `results/mc_runs/` via relative
> paths.  They do **not** require the proprietary S&P Capital IQ or
> MIT TC track data files to generate figures.

### (b) Lightweight model demo — no proprietary data needed

A self-contained demo runs the full risk propagation on the **Great Miami Hurricane (1926)** using synthetic insurer data. Expected runtime: ~30 seconds on a laptop.

```bash
pip install -e .
python scripts/demo/run_demo.py
```

See [`scripts/demo/README.md`](scripts/demo/README.md) for expected output,
output schema, and troubleshooting.  See [`demo_data/README.md`](demo_data/README.md)
for a description of the synthetic insurer datasets and the data restrictions
that make them necessary.

### (c) Full upstream Monte Carlo computation — requires licensed data

Reproducing the Monte Carlo outputs from scratch requires:
1. Licensed S&P Capital IQ surplus and market share data (see *Data availability and restrictions* below).
2. MIT model TC tracks (WindRiskTech L.L.C.) (see below).
3. An HPC cluster for runs with large event sets (see `scripts/cluster/`).

Steps:
```bash
pip install -e .                            # install package
# Place proprietary data in fl_risk_model/data/ (see config.py for expected filenames)
python scripts/run/run_historical_scenarios_mc.py   # historical scenario MC
python scripts/run/run_emanuel_monte_carlo.py        # full probabilistic MC
```

---

## Instructions for use

To adapt the model to a different event, region, or policy scenario:

1. **Provide a new event impact file**: a CSV with columns `countyfp, county_name, value`
   (where `value` is the fraction of county TIV affected) placed in
   `fl_risk_model/data/hazard/historical_events/`.
2. **Override configuration** in `fl_risk_model/config.py` or via `cfg` attributes
   before running — e.g., `cfg.FIXED_YEAR`, `cfg.DO_FLOOD`, `cfg.FHCF_LAYER`.
3. **Call the runner directly**:
   ```python
   from fl_risk_model.mc_run_events import run_one_iteration, _prepare_common_inputs
   common = _prepare_common_inputs()
   result = run_one_iteration("my_event", ["my_stem"], rng, common)
   ```
4. **Add a policy scenario**: subclass or configure one of the transforms in
   `fl_risk_model/scenarios/` and pass `policy_scenario_config` to `run_one_iteration`.

---

## Data availability and restrictions

### Data included in the repository (publicly redistributable)

| File / Directory | Source | Notes |
|-----------------|--------|-------|
| `fl_risk_model/data/*.csv` | Public regulatory filings, FEMA | FHCF terms, Citizens capital, NFIP premium/penetration, county FIPS crosswalk, wind/water attribution |
| `fl_risk_model/data/FHCF_2024_Exposure_byCounty.xlsx` | Florida Hurricane Catastrophe Fund (public) | County-level residential exposure used for FHCF layer sizing |
| `fl_risk_model/data/hazard/historical_events/*.csv` | Derived from IBTrACS via CLIMADA | County-level wind damage (USD) for the 4 model events (Great Miami, Lake Okeechobee, Andrew, Irma) plus 4 composite multi-storm scenarios; see `scripts/hazard/simulate_historical_event_losses.py` |
| `fl_risk_model/data/hazard/fl_per_event_impacts*.csv` | Derived from Gori et al. (2025) | Log-linear damage model outputs; see attribution below |
| `fl_risk_model/data/catbonds_2024.csv` | Public cat bond prospectuses | Catastrophe bond terms and attachment points |
| `demo_data/` | **Synthetic / fictitious** | Illustrative insurer data for demo only; not based on real companies |

### Data requiring a commercial license (S&P Capital IQ)

Two input files are **not included** in the repository because they are sourced
from S&P Capital IQ under a commercial data license:

- **Florida homeowners market share**: company-level direct premiums written
  (used in `fl_risk_model/config.py` as `MARKET_SHARE_XLSX`).
- **Florida statutory surplus and capital**: entity-level and group-level
  surplus (used as `SURPLUS_FILE`).

Researchers with access to S&P Capital IQ can retrieve these datasets from the
Capital IQ platform and place them in `fl_risk_model/data/` with the filenames
specified in `fl_risk_model/config.py`.  The demo (`scripts/demo/run_demo.py`)
and the figure-reproduction notebooks do **not** require these files.

### Data requiring a non-redistribution agreement (WindRiskTech L.L.C.)

The synthetic tropical cyclone event sets from the MIT model are proprietary and owned by WindRiskTech L.L.C.  Due to proprietary restrictions, these data are not publicly archived.  Researchers interested in accessing the data for scientific purposes may contact WindRiskTech L.L.C. at info@windrisktech.com, subject to a non-redistribution agreement.

Precomputed per-event county-level impacts derived from these event sets (`fl_risk_model/data/hazard/emanuel/`) are not included in the repository due to their volume and the underlying data restrictions.  They are available from the corresponding author upon reasonable request.

### External public data (Gori et al. 2025)

The synthetic TC hazard and damage simulations from:

> Gori, A. (2025). "Tropical Cyclone Synthetic Hazard and Damage Simulations",
> in *Sensitivity of TC risk to storm climatology change and socioeconomic growth*.
> DesignSafe-CI. https://doi.org/10.17603/ds2-0jkm-h487

were used to derive the county-wide wind/flood loss attribution tables
(`fl_risk_model/data/florida_log_contribution_p95_present.csv`).  The raw `.mat` files are publicly available at the DOI above; they are not redistributed here (the `fl_risk_model/data/hazard/gori_data/` directory is an empty placeholder).  The script `scripts/hazard/generate_log_contribution_from_mat_files.py` reproduces the preprocessing steps.

---

## Requirements

- Python 3.11+ (recommended: use the CLIMADA conda environment)
- [CLIMADA](https://github.com/CLIMADA-project/climada_python) v6.1.0+ (only for hazard preprocessing)
- HPC cluster (only for full MC runs with large event sets)

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
