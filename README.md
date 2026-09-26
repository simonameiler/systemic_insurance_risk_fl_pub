# Florida Systemic Insurance Risk Model

[![DOI](https://zenodo.org/badge/1185729011.svg)](https://doi.org/10.5281/zenodo.19361127)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Model and analysis scripts for:

**Simona Meiler** (1), Steven I. Jackson (2), Kerry Emanuel (3), Noah S. Diffenbaugh (4), and Jack W. Baker (1): *Stress testing insurance market stability under climate risk*.

[Preprint](https://doi.org/10.31223/X59X9X)

(1) Civil and Environmental Engineering, Stanford University, CA, USA

(2) American Academy of Actuaries, Washington, DC, USA

(3) Lorenz Center, Massachusetts Institute of Technology, Cambridge, MA, USA

(4) Earth System Science, Stanford University, CA, USA

The model links hurricane losses to insurance coverage, insurer capital, risk transfer, and institutional financing rules. It tracks uninsured losses, private insurer defaults, and residual financing requirements across Florida's insurance system. Historical footprints, simulated seasons, climate scenarios, market changes, and prescribed physical loss reductions can be assessed with the same framework.

## Repository structure

```text
fl_risk_model/                  Financial model, configuration, and public inputs
  branches/                    Private wind, Citizens, flood, and uninsured losses
  scenarios/                   Market exit, expanded coverage, and loss reduction
  tests/                       Accounting and implementation checks
scripts/
  demo/                        Example using synthetic insurer financial data
  run/                         Monte Carlo drivers and sensitivity analyses
  hazard/                      Hazard and county-loss preprocessing
  cluster/                     Slurm submission and run validation
  analysis/publication/        Publication figures, tables, and archive validation
  analysis/                    Additional analysis utilities
notebooks/                     Historical and probabilistic reproduction notebooks
results/                       Publication outputs, manifest, and archive instructions
CITATION.cff                   Software citation metadata
```

## Content

### Financial model

`fl_risk_model/runner.py` allocates losses and applies FHCF reimbursements, stylized catastrophe-bond recoveries, insurer capital and group support, FIGA and Citizens assessments, and NFIP financing. `mc_run_events.py` aggregates the outcomes over simulated seasons. Exposure, capital, coverage assumptions, and institutional parameters are configured in `fl_risk_model/config.py`.

Residual financing requirement is the season-level sum of FIGA residual deficit, Citizens residual deficit, and NFIP financing requirement. FHCF shortfall is reported separately. Final insurer balances with magnitude below USD 0.01 are treated as zero before default classification; publication processing applies the same convention to FIGA residuals.

### Analyses and figures

The publication pipeline in `scripts/analysis/publication/reproduce.py` validates the result archive and regenerates eight financial-analysis figures, six numerical table fragments, and numerical reference values. Figure S1 is included as a published reference; its event curve requires the restricted event-loss catalog to recompute. It retains zero-loss seasons, calculates aggregate return levels from season-level sums, and adds median within-GCM future-minus-historical changes to the ERA5 baseline.

The manuscript uses five main figures and four SI figures. Generated SI tables cover historical scenarios (S3), climate and policy means (S4), threshold probabilities (S5), insured-fraction sensitivity (S6), and common-season decomposition (S8). The input, metric-definition, and parameter tables (S1, S2, S7) are described in the manuscript and SI rather than generated from simulations.

### Hazard preprocessing and cluster runs

`scripts/hazard/` prepares historical footprints, synthetic event impacts, and year sets. These steps require CLIMADA and the relevant hazard inputs. `scripts/run/` applies the financial model to the prepared losses; large campaigns can use the Slurm tools in `scripts/cluster/`.

## System requirements

- Python 3.11 or newer; macOS and Linux are used for development. Windows is untested.
- NumPy, pandas, SciPy, Matplotlib, OpenPyXL, and tqdm are installed with the package. Bounds are specified in `pyproject.toml`.
- CLIMADA is needed for hazard preprocessing, not for the financial model or the included demo. The study uses the CLIMADA 6.1 development series.
- Jupyter is optional, for the reproduction notebooks: `pip install -e ".[notebooks]"`.
- A cluster is useful for the full synthetic event campaign. No GPU is required.

## Installation

```bash
git clone --depth 1 https://github.com/simonameiler/systemic_insurance_risk_fl_pub.git
cd systemic_insurance_risk_fl_pub
python -m pip install -e .
```

Installation usually takes a few minutes when binary dependencies are available. In an existing environment that already provides the dependencies, use `python -m pip install -e . --no-deps`.

## Reproducibility

### (a) Rebuild figures and tables from completed runs

Download [the v1.1.0 results archive](https://github.com/simonameiler/systemic_insurance_risk_fl_pub/releases/download/v1.1.0/systemic_insurance_risk_fl_v1.1.0_results.tar.gz), verify its [SHA-256 checksum](results/SHA256SUMS), and extract it into `results/`:

```bash
shasum -a 256 -c results/SHA256SUMS
tar -xzf systemic_insurance_risk_fl_v1.1.0_results.tar.gz -C results
python scripts/analysis/publication/reproduce.py --archive results/campaign
```

The command validates all 106 distinct analyses, file checksums, season/realization IDs, finite amounts, source execution, and wind/flood reconciliation before processing. It then writes PDF and PNG figures, CSV summaries, LaTeX table fragments, and numerical reference values to `results/publication/`. The committed summaries and figures are ready to inspect without downloading the full archive. See [`results/README.md`](results/README.md) for the archive inventory and figure/table mapping.

The financial figures and tables require no licensed inputs. Figure S1 compares event and seasonal return periods: its published image is included, while recomputing the event curve requires authorized access to the ERA5 event-loss catalog. Supply `--physical-data /path/to/era5_event_catalog` to recompute it. Restricted event-level records are not in the public archive.

The two notebooks provide the same workflow with figure displays:

```bash
jupyter lab notebooks/historical_scenario_analysis.ipynb
jupyter lab notebooks/probabilistic_risk_analysis_pub.ipynb
```

### (b) Run the model demo

The demo uses the Great Miami Hurricane footprint, public institutional inputs, and synthetic insurer market shares and capital. Company names and identifiers are public; the financial values are illustrative.

```bash
python scripts/demo/run_demo.py --n_iter 100 --seed 42
```

Results are written to `demo_output/`. They illustrate the software and are not the manuscript results. See [`scripts/demo/README.md`](scripts/demo/README.md) and [`demo_data/README.md`](demo_data/README.md).

### (c) Recompute the financial analyses

This requires the licensed insurer inputs and access to the synthetic hurricane-loss caches described below. Set input paths in `fl_risk_model/config.py` and provide the cached event set explicitly.

```bash
python scripts/run/run_historical_scenarios_mc.py --help
python scripts/run/run_emanuel_monte_carlo.py --help
python scripts/run/run_climate_buildingcode_sensitivity_windfloods.py --help
python scripts/run/run_insured_fraction_sensitivity.py --help
```

A complete campaign consists of 106 analyses: four ERA5 baseline/policy runs, 25 GCM runs, 65 climate–loss-reduction runs, seven historical/sequential scenarios, and five insured-fraction runs. Synthetic analyses use 10,000 seasons each; historical scenarios use 1,000 realizations. The five GCMs are CanESM, CNRM6, EC-Earth6, IPSL6, and MIROC6. Climate comparisons cover mid- and end-century SSP2–4.5 and SSP5–8.5.

`scripts/cluster/campaign_inventory.py` exports this exact inventory as commands with explicit input and output paths:

```bash
python scripts/cluster/campaign_inventory.py \
  --impact-root /path/to/impacts --out-root results/new_campaign \
  --output results/campaign_jobs.json
```

The existing Slurm launchers are examples for individual analyses and must be adapted to the local cluster. For the manuscript's full 106-run design, use the exported inventory. Every analysis must finish with the expected number of seasons before its results are used.

## Instructions for use

To apply the model to another hazard footprint, provide county-level loss inputs and locally appropriate exposure, coverage, insurer portfolios, and institutional parameters. The included run scripts show how to configure an analysis and pass a market or damage-reduction scenario. Inspect their `--help` output before starting a run.

The loss-reduction scenarios prescribe avoided wind and flood damage. They do not estimate the effectiveness or cost of a particular building standard. Insurer portfolios, reinsurance arrangements, public backstops, and financing rules must be adapted when applying the framework elsewhere.

## Data availability and restrictions

### Public inputs included

| Inputs | Source and purpose |
|---|---|
| FHCF exposure workbook and contract terms | Public FHCF filings; county exposure and reimbursement calculations |
| Citizens county exposure and capital tables | Citizens Property Insurance reports |
| NFIP participation, coverage, claims, and premium tables | FEMA/OpenFEMA inputs |
| Company identifiers and county crosswalks | Public regulatory and geographic identifiers |
| `catbonds_2024.csv`, `catbonds_2024_reviewed.csv` | Artemis inventory and explicit eligibility/beneficiary mapping; attachment and payout assumptions are modeled rather than observed contract terms |
| Historical-event county losses | Derived from public IBTrACS tracks through CLIMADA |
| Wind/flood attribution tables | Derived from the public Gori et al. hazard and damage simulations |
| `demo_data/` | Synthetic market shares and capital for the demonstration |

### Licensed insurer inputs

Two S&P Capital IQ files are excluded from the repository:

- `FL HO Market Share Report_6.10.25.xlsx`, used for company market shares.
- `20250805 FL Surplus Capital, Group v Entity.xlsx`, used for entity and group capital.

Researchers need their own access to S&P Capital IQ to obtain these inputs. Configuration names are `MARKET_SHARE_XLSX` and `SURPLUS_FILE`.

### Synthetic hurricane inputs

MIT tropical cyclone event sets are owned by WindRiskTech L.L.C. They and the derived event-level impact caches are not redistributed here. Scientific access must be arranged with WindRiskTech at info@windrisktech.com under its applicable terms. The authors cannot independently redistribute restricted inputs.

### External public hazard data

Gori, A. (2025), *Tropical Cyclone Synthetic Hazard and Damage Simulations*, DesignSafe-CI, [doi:10.17603/ds2-0jkm-h487](https://doi.org/10.17603/ds2-0jkm-h487), supplies the data used to estimate county wind/flood attribution. The raw MATLAB files are obtained from that archive; preprocessing is in `scripts/hazard/generate_log_contribution_from_mat_files.py`.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest fl_risk_model/tests -q
```

The tests cover institutional accounting, FHCF reimbursement limits, catastrophe-bond beneficiaries, seasonal aggregation, default boundaries, and result-archive validation. Two integration tests require licensed insurer inputs and skip when those files are unavailable.

## License

GNU General Public License v3.0 or later. See [LICENSE](LICENSE).
