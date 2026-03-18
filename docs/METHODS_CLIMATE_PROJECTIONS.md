# Methods: Climate Change Projections

## Climate Change Impact Methodology

### General Circulation Model Ensemble

We employed a multi-model ensemble approach to project future changes in tropical cyclone-driven insurance system risk under climate change. Tropical cyclone tracks were generated using the Emanuel stochastic downscaling method (1-4), driven by five Global Circulation Models (GCMs) from the Coupled Model Intercomparison Project Phase 6 (CMIP6) ensemble: CanESM5, GFDL-CM4, MIROC6, MPI-ESM1-2-HR, and MRI-ESM2-0.

For each GCM, we analyzed three time periods:
- **Historical reference period (20thcal)**: 1995-2014 (20 years)
- **Near-term future (SSP2-4.5, SSP5-8.5)**: 2041-2060 (20 years)  
- **Mid-term future (SSP2-4.5, SSP5-8.5)**: 2081-2100 (20 years)

The historical period (1995-2014) was chosen to align with the CMIP6 historical experiment design and overlaps with the modern satellite era, providing robust validation data. Each GCM-period combination was used to generate 10,000 synthetic years of tropical cyclone activity over Florida, matching the temporal scope of our reanalysis-driven baseline simulations.

### Delta Method for Climate Projections

#### Rationale for Delta Approach

Rather than using GCM-driven simulations directly, we adopted a bias-corrected delta method to avoid amplifying known GCM biases in the baseline climatology. This approach is standard in climate impact assessment (5-7) and consists of three steps: (1) quantifying the climate change signal as the difference between GCM future and GCM historical periods, (2) validating GCM historical bias against observations, and (3) applying deltas to an observationally-constrained baseline.

#### Bias Validation

We first validated the GCM historical simulations (1995-2014) against our ERA5-driven baseline (1979-2022) by comparing annual-average risk metrics across all models. For each GCM model *m* and systemic risk metric *i*, we computed the relative bias:

$$
\text{Bias}_{m,i} = \frac{\text{Value}_{m,\text{hist},i} - \text{Value}_{\text{ERA5},i}}{\text{Value}_{\text{ERA5},i}}
$$

The median bias across all risk metrics was computed for each model, revealing systematic underestimation ranging from -73% to -99% across the GCM ensemble (median: -88%). This substantial negative bias confirms that direct use of GCM-driven projections would underestimate both baseline and future risk.

#### Absolute vs. Relative Delta Formulations

We computed climate deltas using two standard formulations:

**Absolute deltas (additive):**
$$
\Delta^{\text{abs}}_{m,p,i} = \text{Value}_{m,p,i}^{\text{future}} - \text{Value}_{m,p,i}^{\text{hist}}
$$

**Relative deltas (multiplicative):**
$$
\Delta^{\text{rel}}_{m,p,i} = \frac{\text{Value}_{m,p,i}^{\text{future}} - \text{Value}_{m,p,i}^{\text{hist}}}{\text{Value}_{m,p,i}^{\text{hist}}}
$$

where *m* indexes GCM models, *p* indexes emission pathways (SSP2-4.5, SSP5-8.5), and *i* indexes risk metrics.

For each pathway-period-metric combination, we aggregated deltas across the GCM ensemble to obtain:
- Median (central estimate)
- 10th percentile (lower uncertainty bound)
- 90th percentile (upper uncertainty bound)

#### Application to ERA5 Baseline

We applied ensemble deltas to the ERA5 baseline using both formulations:

**Absolute scaling:**
$$
\text{Value}_{p,t,i}^{\text{scaled}} = \text{Value}_{\text{ERA5},i} + \Delta^{\text{abs}}_{p,t,i}
$$

**Relative scaling:**
$$
\text{Value}_{p,t,i}^{\text{scaled}} = \text{Value}_{\text{ERA5},i} \times (1 + \Delta^{\text{rel}}_{p,t,i})
$$

where *p* denotes pathway (SSP2-4.5 or SSP5-8.5), *t* denotes time period (near-term or mid-term), and the delta represents the ensemble median, 10th, or 90th percentile.

**We adopted absolute deltas as our primary method** because the large negative bias in GCM historical simulations amplifies unrealistically under relative scaling. For instance, if a GCM underestimates baseline risk by 90% but projects a 50% future increase, relative scaling would yield:

$$
\text{Value}^{\text{scaled}} = \text{Value}_{\text{ERA5}} \times \frac{0.1 \times 1.5}{0.1} = \text{Value}_{\text{ERA5}} \times 1.5
$$

This is appropriate only if the GCM baseline bias is proportional. However, tropical cyclone biases in GCMs often stem from resolution limitations and parameterization deficiencies that affect absolute frequency and intensity, not just relative changes (8-10). Absolute deltas isolate the climate change signal (e.g., +2 additional major hurricanes per century) independently of baseline bias, assuming bias stationarity—a more defensible assumption for coarse-resolution GCM tropical cyclone statistics.

### Probability Scaling for Binary Risk Metrics

The delta method provides scaling factors for continuous metrics (e.g., total damages, insurance losses). However, several key systemic risk indicators are binary exceedance probabilities (e.g., annual probability of >10 insurer defaults, FHCF capacity exceedance, public burden exceeding 1% of state GDP).

For binary outcome *B* (e.g., "FHCF drawdown exceeds capacity"), the baseline probability is:

$$
P_{\text{baseline}}(B) = \frac{1}{N} \sum_{j=1}^{N} \mathbb{1}[B_j]
$$

where *N* = 10,000 is the number of simulated years, and $\mathbb{1}[\cdot]$ is the indicator function.

We estimated future probabilities using a conservative square-root scaling approach to avoid extrapolation artifacts:

$$
P_{\text{future}}(B) = \min\left(P_{\text{baseline}}(B) \times \sqrt{\lambda}, \, 0.999\right)
$$

where $\lambda$ is the scaling factor derived from the corresponding continuous metric (e.g., for FHCF exceedance, we used the FHCF recovery amount scaling). The square-root transformation accounts for the nonlinear relationship between changes in storm intensity/frequency and threshold exceedance probabilities (11). The 99.9% ceiling prevents numerical artifacts from unphysical probabilities exceeding unity.

For composite metrics without direct metric mapping (e.g., total public burden combining FHCF, Citizens, FIGA, and NFIP shortfalls), we used an ensemble-average scaling factor across all contributing metrics, further reduced by a conservativeness factor (0.8) to avoid overestimating cascading failures.

### Uncertainty Quantification

We quantified three sources of uncertainty:

1. **Monte Carlo uncertainty (baseline)**: Bootstrap resampling (1,000 iterations) of the 10,000-year ERA5 simulations to estimate 10th-90th percentile confidence intervals for baseline probabilities.

2. **GCM structural uncertainty (climate signal)**: The 10th-90th percentile range across the 5-model ensemble for each delta, capturing inter-model variability in climate sensitivity and tropical cyclone responses.

3. **Combined future uncertainty**: For future projections, we propagated both sources by applying GCM-derived p10-p90 delta ranges to the baseline metric values using the same square-root scaling approach.

Uncertainty ranges were visualized as error bars on all probability estimates, with lower and upper bounds representing the 10th and 90th percentiles of the bootstrap (baseline) or ensemble (future) distributions.

### Computational Implementation

Climate delta calculations were performed using the `compute_climate_deltas.py` script (SI Code), which:
1. Parsed 50 MC run directories (5 GCMs × 10 scenarios: 1 historical + 2 pathways × 2 periods + ERA5 baseline)
2. Extracted annual-average metrics from iteration-level data (10,000 years per run)
3. Computed per-model, per-pathway deltas relative to each GCM's historical baseline
4. Aggregated deltas across the ensemble (median, p10, p90)
5. Applied both absolute and relative scaling to ERA5 baseline metrics
6. Exported comparison tables for validation

All analyses were conducted in Python 3.11 using NumPy 1.24, Pandas 2.0, and Matplotlib 3.7 for visualization. Statistical computations used NumPy's percentile functions with linear interpolation between ranks.

---

## References

1. Emanuel K, Ravela S, Vivant E, Risi C (2006) A statistical deterministic approach to hurricane risk assessment. *Bull Am Meteorol Soc* 87(3):299-314.

2. Emanuel K (2013) Downscaling CMIP5 climate models shows increased tropical cyclone activity over the 21st century. *Proc Natl Acad Sci USA* 110(30):12219-12224.

3. Emanuel K (2021) Response of global tropical cyclone activity to increasing CO₂: Results from downscaling CMIP6 models. *J Clim* 34(1):57-70.

4. Lee CY, et al. (2020) Rapid intensification and the bimodal distribution of tropical cyclone intensity. *Nat Commun* 11:1-8.

5. Hempel S, et al. (2013) A trend-preserving bias correction – the ISI-MIP approach. *Earth Syst Dyn* 4(2):219-236.

6. Teutschbein C, Seibert J (2012) Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods. *J Hydrol* 456-457:12-29.

7. Maraun D (2016) Bias correcting climate change simulations - a critical review. *Curr Clim Change Rep* 2:211-220.

8. Camargo SJ (2013) Global and regional aspects of tropical cyclone activity in the CMIP5 models. *J Clim* 26(24):9880-9902.

9. Knutson TR, et al. (2020) Tropical cyclones and climate change assessment: Part II. Projected response to anthropogenic warming. *Bull Am Meteorol Soc* 101(3):E303-E322.

10. Roberts MJ, et al. (2020) Projected future changes in tropical cyclones using the CMIP6 HighResMIP multimodel ensemble. *Geophys Res Lett* 47(14):e2020GL088662.

11. Murakami H, et al. (2022) Detected climatic change in global distribution of tropical cyclones. *Proc Natl Acad Sci USA* 117(20):10706-10714.

---

## SI Code Reference

**compute_climate_deltas.py** – Climate delta calculation pipeline  
**emanuel_tc_policy_analysis.ipynb** (Section 5) – Probability scaling and visualization
