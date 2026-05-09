# Demo Data — `demo_data/`

This directory contains synthetic financial datasets for all 99 real Florida
private homeowners insurers, designed to run the `fl_risk_model` risk-propagation
workflow on a normal desktop or laptop, without any restricted or proprietary data.

---

## What the demo data represent

| File | Rows | Description |
|---|---|---|
| `demo_market_share.csv` | 99 | **Synthetic** Pareto-distributed market shares for all 99 real Florida private HO insurers. Company names and statutory entity keys (`StatEntityKey`) come from the public regulatory record (`fl_risk_model/data/company_keys.csv`). Market share percentages are **synthetic** (power-law draw, α = 0.8, seed = 2024). |
| `demo_surplus.csv` | 99 | **Synthetic** statutory surplus (entity- and group-level) for the same 99 real companies. Calibrated so that total entity surplus ≈ $18.2 B and total group surplus ≈ $29.5 B — order-of-magnitude plausible relative to FL HO premium volumes. Individual values are not derived from any real filing or S&P Capital IQ record. |

**Important**: Company names and `StatEntityKey` values are taken from Florida's
public regulatory record (FHCF exposure disclosure, Citizens filings, NFIP data).
The financial values — market shares and surplus — are entirely synthetic.
This means demo outputs show plausible systemic risk dynamics (company defaults,
FHCF stress, FIGA activation) but **do not represent the financial condition of
any real insurer**.

### Market share distribution

Shares are drawn from a Pareto power-law distribution (α = 0.8) and normalised
to sum to 1.0 across all 99 private companies (Citizens, C6949, is handled
separately as a residual market carrier and is excluded from this dataset).
The largest synthetic company holds approximately 12.3% of the private market.

### Surplus calibration

Entity surplus is set proportional to market share × ($18 B total) × a uniform
random factor in [0.60, 1.40].  Group surplus multiplies entity surplus by a
size-dependent ratio: 1.8–2.6× for large carriers (share > 5%), 1.4–2.2×
for mid-sized, and 1.0–1.4× for small.  Total entity surplus ≈ $18.2 B is
consistent with a stressed but plausible market capitalisation given ~$15 B in
Florida HO premium volume.

---

## Public data used by the demo (already in `fl_risk_model/data/`)

- `FHCF_2024_Exposure_byCounty.xlsx` — FL Hurricane Catastrophe Fund county-level TIV (public)
- `citizens_county_data_all_harmonized.csv` — Citizens Property Insurance county exposure (public)
- `citizens_capital_pml.csv` — Citizens capital (public filings)
- `nfip_FL_coverage_premium_by_year.csv` — NFIP policy counts and premiums (public)
- `fl_county_fips.csv` — Florida county FIPS crosswalk (public)
- `fhcf_terms_keyed.csv` — FHCF reimbursement contract terms (public)
- `catbonds_2024.csv` — Catastrophe bond terms (public)
- `hazard/historical_events/1926255N15314.csv` — Great Miami Hurricane (1926) per-county
  wind damage fractions derived from IBTrACS + CLIMADA (public; see
  `scripts/demo/run_climada_hazard_pipeline.py` for full reproduction)

---

## What the demo does

Running `python scripts/demo/run_demo.py` executes **100 Monte Carlo iterations**
of the **Great Miami Hurricane (1926)** scenario — the primary illustrative event
in the manuscript (Fig. 2) — through the full risk-propagation waterfall:

1. Allocate county-level wind and flood damage to insured / un(der)insured categories.
2. Route private wind losses through the FHCF reinsurance layer.
3. Apply Citizens Property Insurance backstop.
4. Compute NFIP flood payouts.
5. Deplete individual company surplus; flag defaults; apply intragroup capital support.
6. Compute FIGA assessment, Citizens Tier-1/Tier-2 assessment, and NFIP borrowing.
7. Summarize losses, defaults, and public institutional stress.

### Expected systemic stress

With synthetic total entity surplus ≈ $18.2 B and Great Miami generating
≈ $35–45 B in mean private wind losses at 2024 exposure, the demo is calibrated
to show substantial systemic stress: the majority of private carriers will
default in most iterations, FHCF exhaustion is common, and Citizens and FIGA
face significant residual deficits.  This illustrates the core finding that a
major Florida hurricane would exceed the capacity of the private insurance sector.

---

## Important caveats — outputs are illustrative only

- **Company names are real; financial values are synthetic.**  The demo uses real
  statutory entity keys so that the full waterfall mechanics (group capital
  support, FIGA assessment, Citizens backstop) work correctly, but no real insurer
  financial data was used.
- Demo outputs **will not match** any manuscript figure or table value.
- Because a random seed is fixed (`seed = 42`), outputs are **fully reproducible**
  across software versions; slight floating-point differences may occur on
  different hardware.
- Do not draw any conclusions about the financial condition of individual companies
  from demo outputs.

---

## Data restrictions that apply to the full manuscript analysis

| Data source | Status |
|---|---|
| FHCF county exposure (public) | Included in `fl_risk_model/data/` |
| Citizens county data (public) | Included in `fl_risk_model/data/` |
| NFIP claims and premiums (public) | Included in `fl_risk_model/data/` |
| Historical hurricane event impacts (derived from public IBTrACS) | Included in `fl_risk_model/data/hazard/historical_events/` |
| Company wind exposure & surplus — S&P Capital IQ | **Not included** — commercial license required |
| Synthetic TC event sets — WindRiskTech L.L.C. / MIT model | **Not included** — proprietary, available to researchers on request |
| Gori et al. 2025 hazard matrices (public, DesignSafe-CI) | Not included; download separately for full hazard preprocessing |
