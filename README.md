# Florida Systemic Insurance Risk Model

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Code and data accompanying the paper:

> Meiler, S., et al. (2026). *[Paper Title]*. [Journal]. DOI: [PLACEHOLDER]

## Installation

```bash
git clone https://github.com/simonameiler/systemic_insurance_risk_fl_pub.git
cd systemic_insurance_risk_fl_pub
conda create -n fl_risk python=3.11
conda activate fl_risk
conda install -c conda-forge climada
pip install -e .
```

## Reproducing Paper Results

Two notebooks in `notebooks/` reproduce all figures and tables:

| Notebook | Content |
|----------|---------|
| `historical_scenario_analysis.ipynb` | Fig. 2, SI Table 2, SI Fig. 3 |
| `probabilistic_risk_analysis.ipynb` | Fig. 3–6, Table 1, SI Tables 3–4, SI Figs. 1–2 |

## Data Availability

**Included**: FHCF terms, Citizens/NFIP data, county shapefiles, per-event impacts.

**Proprietary** (contact authors): Florida OIR company-level exposure and capital data. See `fl_risk_model/config.py` for details.

**External**: Kerry Emanuel TC tracks (contact Prof. Emanuel).

## Citation

```bibtex
@article{meiler2026florida,
  title = {{[Paper Title]}},
  author = {Meiler, Simona and [co-authors]},
  journal = {[Journal]},
  year = {2026},
  doi = {PLACEHOLDER}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
