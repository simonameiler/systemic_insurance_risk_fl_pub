# Before / after: manuscript claims affected by the corrections

Every row cites the correction-register item and the source run. "Changes
qualitative interpretation?" records whether the headline conclusion
(direction, ranking, or magnitude class) is affected, not just the decimal
value.

| # | Claim (submitted) | Definition used | Value (submitted) | Value (corrected) | Definition used (corrected) | Source run | Register item | Change type | Changes qualitative interpretation? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Total loss increases ninefold, 10y->100y | Marginal quantile of `total_damage_usd` | 39.5 -> 357.2 USD B (9.04x) | 39.5 -> 357.2 USD B (9.04x) | unchanged | ERA5 baseline | R1 | none (row never bugged) | No |
| 2 | Public burden increases "more than fortyfold," 10y->100y | Sum of 4 separately-quantiled components | 1.7 -> 74.0 USD B (43.9x) | 1.9 -> 58.6 USD B (**31.1x**) | Quantile of season-level sum of 3 non-overlapping components (FIGA+Citizens+NFIP) | ERA5 baseline | R1 + C1 | corrected arithmetic (R1) + revised assumption (C1, aggregate definition) | **Yes** -- still strongly supralinear, but "more than forty" becomes "about thirty-one" |
| 3 | Public burden scales as loss^1.64 (log-log slope beta, RP10-RP1000) | beta from the buggy 4-component sum | 1.64 | **1.51** (corrected, non-overlapping) / 1.58 (aggregation fix only, still 4-component) | Fit on the season-level-sum series | ERA5 baseline | R1 + C1 | corrected arithmetic + revised assumption | No (still clearly supralinear, beta>1) but the specific value moves outside the 1.64 confidence a reader would assume |
| 4 | Annual P(public burden > 1% FL GDP) = 3.7% | Season-level indicator on the 4-component sum | 3.71% | 3.69% (corrected, non-overlapping) | Season-level indicator on the 3-component sum | ERA5 baseline | C1 | revised assumption, minimal numerical effect | No |
| 5 | Annual P(public burden > 10% FL GDP) = 0.1% | as above | 0.13% | 0.08% | as above | ERA5 baseline | C1 | revised assumption | No (both round to "about 0.1%") |
| 6 | Historical scenario public burden: Great Miami 26.1B, Great Miami->Andrew 51.0B, Double Great Miami 65.4B | Legacy 4-component sum | 26.1 / 51.0 / 65.4 USD B | **24.4 / 45.4 / 57.4** USD B | Corrected 3-component, non-overlapping sum | `great_miami_20260326_190906`, `gm_then_andrew_20260326_211013`, `double_gm_20260326_211737` | C1 | revised assumption | No (ranking and nonlinearity claim both survive) |
| 7 | SI Table S4: building-code total loss ~= 19.3B (essentially unchanged from baseline) | `total_damage_usd`, computed before the building-code reduction (stale) | 19.3 USD B | **13.7 USD B** | `total_damage_usd` computed after the reduction (code fixed; reconstructed post hoc for the archived run) | `emanuel_era5_building_codes_major_20260328_034126` | C5 | corrected model behavior (code bug) | **Yes** -- the submitted table implied building codes barely reduce total loss; corrected, it shows the intended ~29% reduction |
| 8 | SI Table S4: un/underinsured wind increases under building codes, 9.7B -> 11.7B | `total_damage_usd`(stale) minus insured wind | 9.7 -> 11.7 USD B | 9.7 -> **6.8** USD B | Correctly computed `wind_uninsured_usd + wind_underinsured_usd` | same | C5 | corrected model behavior (code bug) | **Yes** -- direction reverses from an increase to a decrease |
| 9 | SI Table S4: un/underinsured flood increases under building codes, 2.4B -> 2.6B (approx.) | as above, flood side | ~2.4 -> ~2.6 USD B | 2.4 -> **1.8** USD B | Correctly computed `flood_un_derinsured_usd` | same | C5 | corrected model behavior (code bug) | **Yes** -- direction reverses |
| 10 | SI Table S4/S9: 28.5% no-TC-loss / 15.3% one-event / 56.2% multi-event seasons | undocumented category boundaries | 28.5 / 15.3 / 56.2 | 28.53 / 15.30 / 56.17 (unchanged; definition now stated precisely) | zero-loss / (nonzero-loss & 1 event) / (nonzero-loss & 2+ events) | ERA5 baseline | R2 | reporting clarification only | No |
| 11 | SI Table S6: insured-fraction elasticities (FHCF 1.60, FIGA 1.40, Citizens 1.28, defaults 0.57, wind private 1.00, wind un/underinsured -0.66) | Rounded means, one-sided finite difference | as listed | same to reported precision, from unrounded centered finite differences | centered log-log FD at f in {0.3,0.5} | `insured_frac_sensitivity_combined` | R5 | corrected arithmetic (recomputed independently) | No |
| 12 | (new) Total public burden elasticity to insured wind fraction | not previously reported | -- | 1.14 at f=0.4 | same method, corrected aggregate | `insured_frac_sensitivity_combined` | C1 + R5 | new metric | N/A |
| 13 | Supporting Text S5: Citizens surplus scaling exponent 1.2, "~84% increase" | as stated in text | 1.2 / ~84% | **1.3 / ~94%** (matches the code and the archived market-exit run) | `fl_risk_model/scenarios/market_exit.py` exponent=1.3 | `emanuel_era5_market_exit_moderate_*` | see register "S5-EXIT" | reporting issue (text vs. code mismatch) | No (same qualitative story: nonlinear, adverse-selection scaling) |
| 14 | Main text: "prohibition on forward-looking catastrophe models" in California | as submitted (present tense) | prohibited | California's CDI finalized a Dec. 2024 regulation permitting catastrophe models and a capped net reinsurance cost in ratemaking (effective 2025) | -- | web search, CDI press releases | manuscript wording | reporting issue (outdated) | No |
| 15 | Discussion: public burden "comparable to public costs of systemic banking crises" | 4-component legacy sum vs. GDP | comparison retained | Comparison removed/reframed per Reviewer 2 comment 3: only NFIP borrowing (a fraction of the corrected aggregate) is a fiscal outlay; FIGA/Citizens deficits are private cross-subsidies | -- | correction register | manuscript wording + revised assumption | **Yes** for framing, though the underlying magnitudes are unchanged |
| 16 | Table S7: hazard variance explains >=0.95 of variance for all non-flood metrics except FHCF (0.944) | 300-season x 50-draw nested ANOVA | as stated | reproduced exactly; **table removed from the manuscript** (design does not license a general robustness claim, per brief) | -- | `variance_nested_300x50_20260311_211009` | Table S7 removal | assumption / evidentiary-scope correction | **Yes** for what the manuscript claims this evidence supports, not for the numbers themselves |

## Claims not changed by this pass (explicitly checked, found stable)

- Scenario-based total losses for the four historical storms and three
  sequential scenarios (Great Miami 170.4B, Andrew 114.2B, Lake Okeechobee
  156.9B, Irma 32.0B, etc.) -- exact reproduction, unaffected by any
  correction (these are marginal totals, not the burden aggregate).
- FIGA remaining the largest and most frequently binding channel, Citizens
  second, FHCF most resilient -- true under both the legacy and corrected
  burden definitions and across the severity-bin decomposition (Section 6).
- The qualitative finding that insurance-penetration expansion raises NFIP
  and Citizens stress even as private-market stress falls -- unaffected
  (uses component-level probabilities, not the summed aggregate).
