# Publication results — v1.1.0

The final campaign contains **106 analyses and 997,000 seasons/realizations**:

| Analysis | Runs | Rows per run |
|---|---:|---:|
| ERA5 baseline and three policies | 4 | 10,000 |
| Five GCMs × five climate periods/pathways | 25 | 10,000 |
| Five GCMs × thirteen wind/flood reduction pairs | 65 | 10,000 |
| Historical and sequential scenarios | 7 | 1,000 |
| Fixed insured fractions 0.1–0.5 | 5 | 10,000 |

## Download and reproduce

Download the [compressed results archive](https://github.com/simonameiler/systemic_insurance_risk_fl_pub/releases/download/v1.1.0/systemic_insurance_risk_fl_v1.1.0_results.tar.gz) and check it against [SHA256SUMS](SHA256SUMS). From the repository root:

```bash
shasum -a 256 -c results/SHA256SUMS
tar -xzf systemic_insurance_risk_fl_v1.1.0_results.tar.gz -C results
python scripts/analysis/publication/reproduce.py --archive results/campaign
```

The archive expands to `campaign/manifest.json` and `campaign/outputs/<analysis>/iterations.csv`. The [manifest](manifest.json) records checksums, row counts, and execution provenance. Each retained numeric field and its row ordering are unchanged from the validated source CSV. Restricted event identifiers, event-level wind shares, and company default identities are omitted. No licensed insurer inputs, synthetic storm tracks, or event-level loss caches are distributed.

The simulations used model commit `014ed02c1b3cad1334411b82ce440a01f5b24f1b` and seed 42. The release preserves these financial calculations. The original execution logs were checked for that revision and successful completion; their checksums are retained in the manifest without publishing local paths or cluster logs.

## Figures and tables

Committed outputs in `publication/` can be inspected directly. Regeneration writes into the same directory unless `--out-dir` is supplied.

| Manuscript display | Output |
|---|---|
| Figure 1: model overview | `figures/fig1_systemic_risk_overview_florida.*` |
| Figure 2: historical scenarios | `figures/fig_loss_institutional_stress.*` |
| Figure 3: return-level scaling | `figures/fig_public_burden_scaling_linear.*` |
| Figure 4: climate and policy stress | `figures/fig_combined_climate_policy_systemic_risk.*` |
| Figure 5: financing requirement and loss reduction | `figures/fig_climate_buildingcode_sensitivity_public_burden.*` |
| Figure S1: event and seasonal losses | `figures/fig_loss_return_period.*` |
| Figure S2: loss-reduction sensitivity | `figures/fig_climate_buildingcode_sensitivity.*` |
| Figure S3: SSP5–8.5 stress | `figures/fig_combined_climate_policy_systemic_risk_ssp585.*` |
| Figure S4: common-season severity groups | `figures/fig_residual_financing_severity_decomposition.*` |
| Table 1 | `tables/table1_return_levels.tex` and corresponding CSV |
| Tables S3–S6 | Historical, climate/policy, probability, and insured-fraction files in `tables/` |
| Table S8 | `tables/tableS_decomposition.tex`, `tables/severity_bin_decomposition.csv` |

Figure S1 is copied from its published reference unless `--physical-data` is supplied. Recomputing its event curve requires authorized access to `all_events.csv`; see the [data-access terms](../README.md#data-availability-and-restrictions). The financial figures and all simulation-derived tables are regenerated from the public archive. The conceptual overview retains the original map artwork.

Residual financing requirement sums FIGA residual deficit, Citizens residual deficit, and NFIP financing within each season before quantiles. FHCF shortfall is a separate upstream diagnostic. Values below USD 0.01 in FIGA residuals are treated as numerical zero during publication processing. Source CSVs are preserved. Return-level intervals resample whole seasons; climate intervals represent variation across five GCM changes, not sampling confidence intervals.

The original preprint release and subsequent published snapshots remain available in Git history. Use the results matching the code version; do not mix campaigns.
