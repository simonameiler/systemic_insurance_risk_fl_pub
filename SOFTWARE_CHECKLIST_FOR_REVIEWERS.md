# Software Checklist for Reviewers

This document maps the Nature Research Software Submission Checklist items to
the corresponding locations in this repository.

---

## Checklist

| Checklist Item | Status | Location / Notes |
|----------------|--------|-----------------|
| **Source code** | ✅ Included | `fl_risk_model/` (core model package); `scripts/` (run, analysis, hazard preprocessing); `notebooks/` (figure generation) |
| **Demo dataset** | ✅ Included | `demo_data/demo_market_share.csv`, `demo_data/demo_surplus.csv`. Synthetic insurer data replacing proprietary S&P inputs. See `demo_data/README.md`. |
| **README** | ✅ Included | `README.md` — covers system requirements, installation, peer-review reproducibility (three tiers), instructions for use, and data restrictions |
| **System requirements: OS** | ✅ Documented | `README.md` § *System requirements* — macOS 12+ or Linux (Ubuntu 20.04+) |
| **System requirements: tested versions** | ✅ Documented | `README.md` § *System requirements* — Python 3.11/3.12; see `pyproject.toml` for package versions |
| **Non-standard hardware** | ✅ Documented | `README.md` § *System requirements* — none required for demo or notebooks |
| **Installation guide** | ✅ Documented | `README.md` § *Installation* — `git clone` + `pip install -e .` |
| **Typical install time** | ✅ Documented | `README.md` § *Installation* — 2–10 minutes |
| **Demo instructions** | ✅ Documented | `README.md` § *Peer review reproducibility (b)* and `scripts/demo/README.md` |
| **Expected output of demo** | ✅ Documented | `scripts/demo/README.md` § *Expected Output*; `demo_output/expected_demo_summary.csv` generated on first run |
| **Expected runtime of demo** | ✅ Documented | `scripts/demo/README.md` § *Approximate Runtime* — ~2–5 min on a modern laptop |
| **Instructions for use** | ✅ Documented | `README.md` § *Instructions for use* |
| **Reproducing manuscript results** | ✅ Documented | `README.md` § *Peer review reproducibility (a)* — notebooks + archived results; `results/README.md` — figure-to-notebook mapping table |
| **Data restrictions** | ✅ Documented | `README.md` § *Data availability and restrictions*; `demo_data/README.md` — distinguishes public, proprietary (S&P Capital IQ license required), and non-redistribution (WindRiskTech L.L.C.) data |

---

## Three-tier reproducibility summary

| Tier | What it reproduces | Data needed | Runtime |
|------|-------------------|-------------|---------|
| **(a) Figures from archived results** | All manuscript figures and tables | None beyond the repository | ~5 min per notebook |
| **(b) Lightweight model demo** | Full risk waterfall on Hurricane Irma, synthetic insurers | None beyond the repository | 2–5 min |
| **(c) Full upstream MC computation** | Raw Monte Carlo outputs | S&P Capital IQ (commercial), WindRiskTech (non-redistribution), HPC cluster | Days on HPC |

---

## Key file locations

| Component | File |
|-----------|------|
| Core model package | `fl_risk_model/` |
| Central configuration | `fl_risk_model/config.py` |
| Risk waterfall entry point | `fl_risk_model/runner.py` |
| Monte Carlo simulator | `fl_risk_model/mc_run_events.py` |
| Demo script | `scripts/demo/run_demo.py` |
| Demo documentation | `scripts/demo/README.md` |
| Demo data | `demo_data/` |
| Historical figure notebook | `notebooks/historical_scenario_analysis.ipynb` |
| Probabilistic figure notebook | `notebooks/probabilistic_risk_analysis_pub.ipynb` |
| Archived MC results | `results/mc_runs/` |
| Results documentation | `results/README.md` |
| Package metadata | `pyproject.toml` |
| License | `LICENSE` (GPL-3.0) |

---

## What is and is not reproduced

The manuscript uses two proprietary datasets (S&P Capital IQ market share and
surplus) and one dataset subject to non-redistribution restrictions (WindRiskTech
L.L.C. synthetic TC event sets).  These data are **not included** in the
repository.

However:
- All **manuscript figures and tables** can be reproduced by running the
  two Jupyter notebooks against the pre-computed archived results in `results/`.
  The notebooks do **not** load proprietary data.
- The **risk model code** can be exercised end-to-end using the demo, which
  substitutes synthetic insurer data for the proprietary inputs.
- Researchers who obtain the proprietary datasets independently can reproduce
  the full Monte Carlo computation from scratch using `scripts/run/`.

The authors do not claim that the precise numerical outputs of the full MC
computation can be independently reproduced without the proprietary data.
