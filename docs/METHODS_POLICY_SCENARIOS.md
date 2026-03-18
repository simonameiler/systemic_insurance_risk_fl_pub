# Methods: Policy Intervention Scenarios

## Policy Scenario Framework

We evaluated three distinct policy intervention pathways representing realistic adaptation strategies available to Florida policymakers. Each scenario modifies specific components of the insurance system while holding all other model parameters constant, allowing isolation of intervention effects on systemic risk metrics.

All policy scenarios were implemented at the "major" intensity level, representing ambitious but achievable interventions with 10-20 year implementation timelines (2024-2040). Parameter values were calibrated to empirical evidence from insurance market data, regulatory filings, and peer-reviewed building science literature.

---

## Scenario 1: Market Exit (Moderate)

### Policy Context

This scenario models continued contraction of the private insurance market driven by catastrophic loss experience, regulatory constraints, and reinsurance market stress—trends observed in Florida from 2022-2024 when Citizens Property Insurance Corporation's policy count grew by 57% (1). The scenario represents a plausible continuation of this trajectory through 2030.

### Implementation Parameters

**Target Citizens market share**: 25% (increase from 15% baseline; +10 percentage points)

**Market exit mechanism**: Uniform reduction across all private insurers, proportional to current market share

**Citizens absorption rate**: 85% of exited exposure transfers to Citizens, 15% becomes uninsured

**Timeline**: 2024-2030 (6 years)

### Computational Method

The market exit scenario modifies the pre-event exposure allocation through the following sequence:

1. **Calculate required private market contraction**:
   $$
   \Delta_{\text{private}} = \frac{T \times s_{\text{target}} - E_{\text{Citizens}}^{\text{baseline}}}{(1 - \alpha)}
   $$
   where $T$ is total insurable value (TIV), $s_{\text{target}}$ is target Citizens share (0.25), $E_{\text{Citizens}}^{\text{baseline}}$ is baseline Citizens exposure, and $\alpha$ is absorption rate (0.85).

2. **Apply geographic bias** (coastal counties experience higher exit rates):
   $$
   w_c = \begin{cases}
   1.0 & \text{if county } c \text{ is inland} \\
   1.0 & \text{if county } c \text{ is coastal (uniform baseline)}
   \end{cases}
   $$
   For this moderate scenario, no coastal bias was applied (baseline implementation uses $w_c = 1.0$ for all counties).

3. **Reduce private exposure proportionally**:
   $$
   E_{i,c}^{\text{new}} = E_{i,c}^{\text{baseline}} \times \left(1 - \frac{\Delta_{\text{private}} \times w_c}{\sum_{i',c'} E_{i',c'}^{\text{baseline}} \times w_{c'}}\right)
   $$
   where $E_{i,c}$ is exposure for insurer $i$ in county $c$.

4. **Transfer absorbed portion to Citizens**:
   $$
   E_{\text{Citizens},c}^{\text{new}} = E_{\text{Citizens},c}^{\text{baseline}} + \alpha \times \sum_i \left(E_{i,c}^{\text{baseline}} - E_{i,c}^{\text{new}}\right)
   $$

5. **Adjust insurer capital** to reflect reduced exposure:
   
   For each insurer $i$, group-level surplus is reduced using a "middle-ground" method that balances fixed-cost retention with exposure-proportional scaling:
   $$
   S_i^{\text{new}} = S_i^{\text{baseline}} \times \left[0.3 + 0.7 \times \frac{E_i^{\text{new}}}{E_i^{\text{baseline}}}\right]
   $$
   This assumes 30% of capital represents fixed corporate infrastructure that remains even with reduced exposure, while 70% scales proportionally.

6. **Adjust Citizens capital** for increased exposure using adverse selection assumption:
   
   Citizens capital requirements scale superlinearly with share because exited exposure tends to be higher-risk:
   $$
   S_{\text{Citizens}}^{\text{new}} = S_{\text{Citizens}}^{\text{baseline}} \times \left(\frac{s_{\text{target}}}{s_{\text{baseline}}}\right)^{1.2}
   $$
   The exponent 1.2 reflects empirical evidence that Citizens exposures have 15-25% higher loss costs than private market averages (2,3).

### Model Integration

Market exit modifications are applied **before** tropical cyclone event simulation during the exposure initialization phase. The modified exposure distributions ($E_{i,c}^{\text{new}}$) and capital levels ($S_i^{\text{new}}$) are then held constant across all 10,000 simulated years, representing the steady-state market structure after exit is complete.

Crucially, gross hurricane losses (wind and flood damage) are **not** modified by market exit—the scenario only reallocates who bears the financial burden of those losses. This reallocation affects systemic outcomes through three mechanisms:

1. **Concentration risk**: Losses concentrate in Citizens, increasing probability of capacity exhaustion
2. **Capital deterioration**: Exiting private insurers reduce capital, weakening remaining companies' balance sheets
3. **Uninsured exposure growth**: 15% of exited policies become uninsured, shifting losses to households and government

---

## Scenario 2: Penetration Increase (Major)

### Policy Context

This scenario models a coordinated state-federal effort to increase insurance take-up through affordability programs, mandatory purchase requirements, and building-tied insurance (4,5). It represents the "opposite" of market exit: expanding coverage breadth while reducing Citizens' role via depopulation to the private market.

### Implementation Parameters

**Wind insurance penetration**: 60% (increase from 40% baseline; +20 percentage points / 50% relative increase)

**Flood insurance penetration**: 30% (increase from 11% baseline; +19 percentage points / 173% relative increase)

**Citizens market share**: 8% (decrease from 15% baseline; -7 percentage points via depopulation to private market)

**Geographic targeting**: Coastal counties experience 1.5× higher penetration increases than inland counties

**SFHA-aware flood scaling**: Non-SFHA zones prioritized (3× growth) vs. SFHA zones (1.2× growth), reflecting that SFHA zones already have ~35% penetration due to mandatory purchase requirements while non-SFHA zones average only ~5%

**Surplus adjustment method**: Proportional (capital scales linearly with exposure growth)

**Timeline**: 2024-2040 (16 years)

### Computational Method

The penetration scenario modifies both exposure allocation and insurer capital through three sequential interventions:

#### 2.1 Citizens Depopulation

Transfer exposure from Citizens to private insurers to reduce state liability:

1. **Calculate required transfer**:
   $$
   \Delta_{\text{depop}} = E_{\text{Citizens}}^{\text{baseline}} - (T \times s_{\text{target}})
   $$
   where $s_{\text{target}} = 0.08$ is the target Citizens share.

2. **Allocate to private insurers** proportional to current market share:
   $$
   E_{i,c}^{\text{after depop}} = E_{i,c}^{\text{baseline}} + \Delta_{\text{depop}} \times \frac{E_{i,c}^{\text{baseline}}}{\sum_{i',c'} E_{i',c'}^{\text{baseline}}}
   $$

#### 2.2 Wind Penetration Increase

Scale private and Citizens wind exposure to reflect higher insurance take-up:

1. **Calculate penetration scaling factor**:
   $$
   \lambda_{\text{wind}} = \frac{p_{\text{target}}}{p_{\text{baseline}}} = \frac{0.60}{0.40} = 1.50
   $$

2. **Apply geographic bias** (coastal counties prioritized):
   $$
   \lambda_{c}^{\text{wind}} = \begin{cases}
   1.50 \times 1.5 = 2.25 & \text{if county } c \text{ is coastal} \\
   1.50 \times 1.0 = 1.50 & \text{if county } c \text{ is inland}
   \end{cases}
   $$

3. **Scale exposure**:
   $$
   E_{i,c}^{\text{after wind}} = E_{i,c}^{\text{after depop}} \times \lambda_c^{\text{wind}}
   $$

#### 2.3 NFIP Flood Penetration Increase (SFHA-Aware)

Use county-specific FEMA flood zone data to target non-SFHA areas where penetration is lowest:

1. **For each county** $c$, retrieve SFHA share $\phi_c$ from FEMA data (e.g., Miami-Dade: $\phi_c = 0.18$).

2. **Calculate baseline penetration decomposition**:
   $$
   p_{\text{baseline}} = \phi_c \times p_{\text{SFHA}} + (1 - \phi_c) \times p_{\text{non-SFHA}}
   $$
   where $p_{\text{SFHA}} \approx 0.35$ (mandatory purchase) and $p_{\text{non-SFHA}} \approx 0.05$ (voluntary).

3. **Apply differential growth** to non-SFHA zones (3× scaling) vs. SFHA zones (1.2× scaling):
   $$
   p_{\text{SFHA}}^{\text{new}} = p_{\text{SFHA}} \times 1.2
   $$
   $$
   p_{\text{non-SFHA}}^{\text{new}} = p_{\text{non-SFHA}} \times 3.0
   $$

4. **Recompose county-level penetration**:
   $$
   p_c^{\text{new}} = \phi_c \times p_{\text{SFHA}}^{\text{new}} + (1 - \phi_c) \times p_{\text{non-SFHA}}^{\text{new}}
   $$

5. **Scale NFIP exposure** (policy count, coverage, premiums) by county:
   $$
   N_{c}^{\text{NFIP, new}} = N_{c}^{\text{NFIP, baseline}} \times \frac{p_c^{\text{new}}}{p_c^{\text{baseline}}}
   $$

#### 2.4 Capital Scaling (Proportional Method)

Adjust insurer surplus to maintain baseline stress ratios:

$$
S_i^{\text{new}} = S_i^{\text{baseline}} \times \left(\frac{E_i^{\text{new}}}{E_i^{\text{baseline}}}\right)
$$

This assumes insurers have sufficient time (16-year timeline) to raise capital proportionally through retained earnings, new equity issuance, and premium rate increases. For our "major" scenario, private market exposure grows by approximately 56% on average, requiring commensurate capital growth.

For rapid penetration increases (e.g., 2-3 year mandates), alternative scaling methods are available:

- **No adjustment** (`method="none"`): Capital remains fixed, stress ratios worsen, insolvency risk increases (conservative)
- **Square-root scaling** (`method="sqrt"`): $S_i^{\text{new}} = S_i^{\text{baseline}} \times \sqrt{E_i^{\text{new}}/E_i^{\text{baseline}}}$, reflecting diversification benefits (optimistic)

### Model Integration

Penetration modifications are applied **before** event simulation but use a two-stage process:

1. **Pre-event exposure scaling** (as described above) modifies TIV distributions
2. **Post-impact penetration rates** modify the insured/underinsured/uninsured allocation

For each simulated hurricane event, gross wind and flood damages are calculated first, then carved out into three categories:

- **Insured losses**: Covered by private insurers, Citizens, or NFIP
- **Underinsured losses**: Losses exceeding policy limits (25-30% of household exposure under baseline)
- **Uninsured losses**: Borne entirely by households with no coverage

Higher penetration rates increase the insured fraction while reducing uninsured fractions. Critically, total gross damage is **unchanged**—only the distribution of who pays differs. This affects systemic outcomes by:

1. **Risk pooling**: More policyholders → better loss diversification for insurers
2. **Premium base expansion**: Higher exposure → more premium revenue to cover losses and build reserves
3. **Reduced government ad-hoc assistance**: Fewer uninsured households → less need for post-disaster aid

---

## Scenario 3: Building Codes (Major)

### Policy Context

This scenario models loss reduction from improved building standards through a combination of new construction subject to enhanced codes and retrofitting of existing vulnerable structures. Parameter values are calibrated to mid-range estimates from empirical building science studies (6-8).

### Implementation Parameters

**Wind loss reduction**: 30% (applied to all wind losses after impact calculation)

**Flood loss reduction**: 25% (applied to all flood losses after impact calculation)

**Retrofit rate**: 50% of pre-2002 building stock retrofitted by 2040

**New construction rate**: 32% of 2024 stock built post-2024 with enhanced standards (≈2% annual turnover × 16 years)

**Evidence base**: 
- Wind: IBHS FORTIFIED program midpoint estimate (7), FLASH study modern codes (8)
- Flood: FEMA elevation guidelines (9), NIST combined mitigation measures (10)

**Timeline**: 2024-2040 (16 years)

### Computational Method

Building code improvements directly reduce gross hurricane losses **after** impact function calculation but **before** the insured/underinsured/uninsured carve-out:

1. **Calculate baseline gross losses** from CLIMADA impact functions:
   $$
   L_c^{\text{gross, wind}} = f_{\text{wind}}(v_{\text{max},c}, \text{TIV}_c, \text{vulnerability}_c)
   $$
   $$
   L_c^{\text{gross, flood}} = f_{\text{flood}}(d_{\text{max},c}, \text{TIV}_c, \text{vulnerability}_c)
   $$
   where $v_{\text{max},c}$ is maximum sustained wind speed in county $c$, $d_{\text{max},c}$ is maximum flood depth, and $f(\cdot)$ are non-linear impact functions from CLIMADA (11).

2. **Apply loss reduction** multiplicatively:
   $$
   L_c^{\text{mitigated, wind}} = L_c^{\text{gross, wind}} \times (1 - r_{\text{wind}})
   $$
   $$
   L_c^{\text{mitigated, flood}} = L_c^{\text{gross, flood}} \times (1 - r_{\text{flood}})
   $$
   where $r_{\text{wind}} = 0.30$ and $r_{\text{flood}} = 0.25$ are loss reduction fractions.

3. **Propagate through exposure carve-out**:
   
   The mitigated losses are then allocated to insured/underinsured/uninsured categories using baseline penetration rates. Critically, **building codes do not modify exposure (TIV) or capital**—only loss magnitudes. This reflects the empirical finding that stronger buildings reduce damage per event without changing total replacement cost (12).

### Interpretation of Loss Reduction Factors

The 30% wind reduction represents a **stock-weighted average** across the Florida building inventory in 2040:

- **68% of stock**: Pre-2002 construction with 50% retrofit rate
  - 50% retrofitted: 40% loss reduction (FORTIFIED-level improvements)
  - 50% un-retrofitted: 0% loss reduction (baseline vulnerability)
  - Average: $(0.5 \times 0.40) + (0.5 \times 0.0) = 0.20$ (20% reduction)

- **32% of stock**: Post-2024 construction with enhanced codes
  - New builds: 50% loss reduction (modern FL Building Code + enforcement)

- **Overall average**: $(0.68 \times 0.20) + (0.32 \times 0.50) = 0.136 + 0.160 = 0.296 \approx 0.30$ (30% reduction)

This calculation assumes uniform geographic application. In practice, coastal counties may achieve higher reductions (35-40%) due to stricter enforcement and higher retrofit participation, while inland counties lag (25-30%). The model can accommodate county-specific reduction factors via the `apply_by_county=True` parameter, but we used spatially uniform factors for the main analysis.

### Model Integration

Building code modifications are applied **during** each hurricane event simulation at a specific point in the damage propagation chain:

1. CLIMADA calculates gross wind and flood damages using baseline vulnerability curves
2. **Building code loss reduction is applied** (this scenario)
3. Climate scaling is applied if enabled (separate scenario)
4. Damages are split into insured/underinsured/uninsured using penetration rates
5. Insured losses flow through the insurance waterfall (deductibles → company retention → reinsurance → FHCF → FIGA → Citizens → bonding)

This sequencing ensures building codes reduce the initial damage "pie" that is then divided among insurers and households. The effect cascades through the financial system:

1. **Primary insurers**: Lower losses → fewer defaults
2. **Reinsurers & FHCF**: Lower recovery claims → less capacity consumed
3. **Backstops (FIGA/Citizens)**: Lower residual deficits → reduced state liability
4. **Households**: Lower uninsured losses → less need for post-disaster borrowing

Unlike market exit or penetration scenarios, building codes provide a "true" risk reduction rather than merely reallocating losses. This makes them particularly valuable for long-term resilience planning.

---

## Scenario Comparison & Baseline

All three policy scenarios were compared against a **baseline scenario** representing the 2024 status quo:

- Wind penetration: 40% (Beta(4,6) distribution mean)
- Flood penetration: 11% (FEMA NFIP 2024-25 Florida average)
- Citizens market share: 15% (Q4 2024 regulatory filings)
- Building codes: Current 2020 FL Building Code with ~10% of stock retrofitted
- Market structure: 155 active insurers with 2024 surplus levels

Each simulation consisted of 10,000 stochastic tropical cyclone years drawn from the Emanuel downscaling method (13) applied to ERA5 reanalysis (1979-2022 climatology). Monte Carlo sampling uncertainty was quantified via bootstrap resampling (1,000 iterations) to compute 10th-90th percentile confidence intervals for all probability estimates.

---

## Building Code-Climate Change Offsetting Analysis

### Research Question

Beyond evaluating individual policy scenarios, we conducted a sensitivity analysis to answer: **What level of building code improvement is required to offset mid-century climate change impacts and maintain current-climate risk levels?**

This analysis provides policymakers with quantitative targets for adaptation planning by identifying the minimum building performance standards needed to neutralize projected increases in tropical cyclone risk under SSP2-4.5 (moderate emissions scenario, 2041-2060).

### Experimental Design

#### GCM Ensemble and Climate Scenarios

We utilized the same 5-model GCM ensemble as the climate projection analysis (CanESM5, CNRM-CM6, EC-Earth3, IPSL-CM6A, MIROC6), each downscaled using the Emanuel physics-based tropical cyclone model (13,14). For each GCM:

- **Historical baseline** (20thcal): 1995-2014, 10,000 synthetic years
- **Future scenario** (ssp245cal): 2041-2060, 10,000 synthetic years × 13 building code levels

#### Building Code Gradient

We tested 13 building code loss reduction levels, applied simultaneously to both wind and flood damages using a 3:2 wind-to-flood ratio reflecting Florida's tropical cyclone damage composition (84.6% wind, 15.4% flood based on historical FLOIR/NFIP data):

| Label | Wind Reduction | Flood Reduction | Combined Average |
|-------|----------------|-----------------|------------------|
| w00f00 | 0% | 0% | 0% |
| w20f13 | 20% | 13% | 16.5% |
| w30f20 | 30% | 20% | 25% |
| w40f27 | 40% | 27% | 33.5% |
| w50f33 | 50% | 33% | 41.5% |
| w60f40 | 60% | 40% | 50% |
| w70f47 | 70% | 47% | 58.5% |
| w80f53 | 80% | 53% | 66.5% |
| w90f60 | 90% | 60% | 75% |
| w100f67 | 100% | 67% | 83.5% |
| w110f70 | 100% | 70% | 85% |
| w120f80 | 100% | 80% | 90% |
| w130f90 | 100% | 90% | 95% |

The 100% wind reduction represents theoretical complete elimination of wind damage (not physically realistic but provides an upper bound). Flood reductions were capped at 90% reflecting practical limits of elevation and floodproofing.

**Total simulations**: 5 GCMs × (1 historical + 13 future × building codes) = 70 runs × 10,000 years = 700,000 simulation-years

### Climate Delta Methodology with Building Code Integration

Traditional climate delta approaches separate climate change from adaptation: first compute the climate signal, then layer on building codes. Our approach integrates them within the same simulation framework to capture nonlinear interactions between hurricane characteristics and building performance.

For each GCM model $m$ and building code level $b$:

1. **Compute climate delta with building codes applied**:
   $$
   \Delta_{m,b}(M) = \mathbb{E}[\text{Metric}_m^{\text{SSP245}, b}] - \mathbb{E}[\text{Metric}_m^{\text{Historical}, b=0}]
   $$
   where $M$ is a systemic risk metric (e.g., annual defaults, public burden) and $\mathbb{E}[\cdot]$ denotes expected value across 10,000 simulated years.

   Critically, building code $b$ is applied **only** to the future (SSP245) scenario, not to the historical baseline. This isolates the net effect of climate change after accounting for adaptation.

2. **Aggregate across GCM ensemble**:
   $$
   \Delta_b^{\text{median}}(M) = \text{median}\{\Delta_{1,b}(M), \Delta_{2,b}(M), \ldots, \Delta_{5,b}(M)\}
   $$
   $$
   \Delta_b^{\text{p10}}(M) = \text{p10}\{\Delta_{1,b}(M), \ldots\}, \quad \Delta_b^{\text{p90}}(M) = \text{p90}\{\Delta_{1,b}(M), \ldots\}
   $$

3. **Apply ensemble delta to ERA5 baseline**:
   $$
   M_{\text{ERA5}}^{\text{future}, b} = M_{\text{ERA5}}^{\text{baseline}} + \Delta_b^{\text{median}}(M)
   $$

   This gives climate-adjusted ERA5 values for each building code level, representing what systemic risk would be under mid-century climate with that level of building code implementation.

4. **Identify offsetting building code level** via linear interpolation:
   $$
   b^*(M) = f^{-1}\left(M_{\text{ERA5}}^{\text{baseline}}\right)
   $$
   where $f(b) = M_{\text{ERA5}}^{\text{future}, b}$ is the continuous interpolation function across the 13 discrete building code levels.

   This identifies the building code strength required to return metric $M$ to its current-climate baseline level.

### Wind-Flood Attribution

All simulations (GCM historical, GCM future, ERA5 baseline) used consistent wind-flood damage attribution:

- **Statewide baseline**: 84.6% wind, 15.4% flood (Gori et al. 2022 empirical analysis)
- **County-specific adjustment**: Spatial heterogeneity from FLOIR/NFIP damage records
- **No climate-induced shifts**: Attribution ratios held constant across all climate scenarios

This approach assumes that the spatial distribution of wind vs. flood damage remains stable under climate change—a conservative assumption given emerging evidence that precipitation intensification may shift the ratio toward flood (15). Sensitivity tests with climate-varying attribution could be explored in future work.

### Key Metrics Evaluated

We evaluated building code requirements across three systemic risk dimensions:

1. **Total economic burden**: Sum of insured, underinsured, and uninsured losses (comprehensive societal cost)
2. **Insurance system defaults**: Number of private companies failing (market stability)
3. **Total public burden**: Sum of FHCF shortfalls, FIGA deficits, Citizens deficits, and NFIP borrowing (taxpayer exposure)

For each metric, we reported:
- Median estimate across 5-model GCM ensemble
- 10th-90th percentile uncertainty range (inter-model spread)
- Required building code level to offset climate change (with uncertainty bounds)

### Results Interpretation

The offsetting building code level represents the **minimum adaptation necessary** to prevent climate change from worsening systemic risk. Values below this threshold result in net risk increase; values above provide a climate risk reduction buffer.

For example, if economic burden requires 45% combined wind-flood reduction to offset climate change:
- 30% codes → Risk increases despite adaptation
- 45% codes → Risk maintains current level (climate neutrality)
- 60% codes → Risk decreases below current level (net improvement)

Uncertainty ranges reflect both GCM structural uncertainty (inter-model spread in tropical cyclone projections) and Monte Carlo sampling uncertainty. The 10th-90th percentile bounds span outcomes under optimistic vs. pessimistic GCM climate sensitivities, providing decision-makers with a range of plausible adaptation requirements.

---

## References

1. Florida Office of Insurance Regulation (2024) "Citizens Property Insurance Corporation Market Share Analysis, Q4 2024."

2. Florida TaxWatch (2022) "Citizens Property Insurance: Measuring the Risks."

3. Kousky C, Kunreuther H (2014) "Addressing affordability in the National Flood Insurance Program." *Journal of Extreme Events* 1(1):1450001.

4. Dixon L, et al. (2017) "The Cost and Affordability of Flood Insurance in New York City." *RAND Corporation* RR-1776.

5. Kunreuther H, Michel-Kerjan E (2011) *At War with the Weather: Managing Large-Scale Risks in a New Era of Catastrophes*. MIT Press.

6. IBHS (2021) "FORTIFIED Home Hurricane Standards." Insurance Institute for Business & Home Safety Technical Report.

7. Brown University IBHS (2020) "Wind Loss Mitigation: Empirical Analysis of Insurance Claims Data." *Structural Safety* 85:101951.

8. Johns Hopkins APL (2019) "Florida Loss Assessment of Hurricane Scenario (FLASH) Project: Final Report."

9. FEMA (2022) "Homeowner's Guide to Retrofitting: Six Ways to Protect Your Home from Flooding." FEMA P-312.

10. NIST (2020) "Community Resilience Planning Guide for Buildings and Infrastructure Systems." NIST Special Publication 1190.

11. Aznar-Siguan G, Bresch DN (2019) "CLIMADA v1: A global weather and climate risk assessment platform." *Geosci Model Dev* 12:3085-3097.

12. Multihazard Mitigation Council (2019) "Natural Hazard Mitigation Saves: 2019 Report." National Institute of Building Sciences.

13. Emanuel K (2021) "Response of global tropical cyclone activity to increasing CO₂: Results from downscaling CMIP6 models." *J Clim* 34(1):57-70.

14. Lee CY, et al. (2020) "Rapid intensification and the bimodal distribution of tropical cyclone intensity." *Nat Commun* 11:1-8.

15. Gori A, et al. (2022) "Tropical cyclone compound flood hazard assessment: from investigating drivers to quantifying extreme water levels." *Earth Syst Dyn* 13:1817-1843.

---

## SI Code References

**fl_risk_model/scenarios/market_exit.py** – Market exit scenario implementation  
**fl_risk_model/scenarios/penetration.py** – Penetration increase scenario implementation  
**fl_risk_model/scenarios/building_codes.py** – Building code scenario implementation  
**fl_risk_model/runner.py** (lines 450-600) – Scenario integration in exposure initialization  
**fl_risk_model/mc_run_events.py** (lines 990-1050) – Building code application in damage propagation  
**fl_risk_model/config.py** (lines 365-460) – Scenario parameter definitions  
**notebooks/climate_buildingcode_windfloods_analysis.ipynb** – Building code sensitivity analysis and climate offsetting calculations  
**scripts/cluster/submit_climate_buildingcode_gcm_sensitivity.sh** – Cluster deployment script for 5 GCMs × 13 building code levels
