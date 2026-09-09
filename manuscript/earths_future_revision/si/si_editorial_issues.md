# Notes on the revised Supporting Information

The revised SI reorganizes the submitted material into six supporting texts. Text S1 documents the inputs. Text S2 explains the physical loss calculation and peril allocation. Text S3 describes insurance allocation and retains the original insured fraction reconstruction. Text S4 provides the financial equations. Text S5 describes simulations and experiments. Text S6 separates metric estimation, uncertainty, and evaluation.

All nine submitted tables and all three submitted figures remain in their original order. Numerical result cells are unchanged. The FHCF recovery equation and the definition of public burden are unchanged. Table S9 now reports the full precision of the FHCF multipliers already provided in the main manuscript. It also removes a duplicated fragment in the original table source. Supporting equations use S numbering. The files compile independently of the main manuscript.

The revision does not resolve the scientific issues below. LaTeX comments mark their locations without placing editorial instructions in the paper.

## Issues affecting interpretation

1. **The aggregate public burden needs an accounting check.** FHCF reimbursement shortfalls remain in insurers' retained losses and can contribute to FIGA or Citizens deficits. Adding the FHCF shortfall to those downstream deficits may count some financial stress more than once. The draft keeps the submitted sum and describes it as a composite institutional stress indicator. It does not describe that sum as a demonstrated total government cost.

2. **The main return-period comparison sums component return levels.** Read-only inspection of `scripts/analysis/generate_table_baseline_return_periods.py` confirms that the submitted total burden column is the sum of separately calculated component return levels. This differs from the return level of the sum across components in each season. Text S6 now states this distinction. The numerical comparison has not been recalculated. Annual exceedance indicators use the modeled season aggregate and should remain conceptually separate from this return-period comparison.

3. **The wind insured fraction changes denominator.** The calibration divides insured wind losses by reconstructed total economic loss, then applies the resulting fraction to wind loss. Text S3 now states this assumption. Its justification or recalibration remains an author decision. The calculation that gives 38.9% also needs an explicit weighting description.

4. **Aggregate economic losses do not identify household losses.** LitPop and the USA impact function cover more than residential property. The SI retains the submitted notation and output labels but states the assumption required to assign the whole residual to households. No new residential scaling has been introduced.

5. **The FHCF recovery limit requires verification.** The original equation applies the company cap before the reimbursement percentage and expense factor. The appropriate placement of these terms should be checked against the actual contract and the implementation. The equation is preserved pending that check.

6. **The financial model treats each season as one aggregate.** Inspection of the model shows that county losses are combined before one financial calculation. Paired historical storms first account for physical exposure depletion. The financial model then receives their combined losses. Each synthetic season starts from baseline capital. The revised SI follows this implementation and does not imply that it tracks financial transactions, recoveries, or reinstatements between individual storms.

## Details needed for reproducibility

- **Hazard frequencies and sampling**. Document the frequency calibration target, correction factor, annual event count distribution, and whether the reported 28.5%, 15.3%, and 56.2% categories count retained tracks or positive-loss events.
- **Wind and flood allocation**. Add the exact regression coefficients and units, the Beta shape parameters, the sources of the four historical means, and the treatment of county fractions that could exceed the unit interval when rescaled.
- **NFIP data version**. The structure denominator and SFHA weighting are now verified. FEMA defines penetration using insured residential structures and a residential structure estimate from the 2022 National Structure Inventory. The public model reconstructs non-SFHA rates from county and SFHA rates, clips them, and weights by structure counts. The exact snapshot used in the submitted simulations still needs to be recorded. The method assumes structure shares also represent loss shares within each county.
- **Exposure perturbation and nested simulations**. Specify the TIV distribution, dependence among counties and entities, sampling frequency, random seeds, season-selection procedure, and any weighting in the 300-season experiment.
- **Climate uncertainty**. State exactly how bootstrap baseline uncertainty and the five-GCM range were combined. Document any bounds imposed when additive changes produce negative outcomes or probabilities outside zero through one.
- **Intervention normalization**. Explain how the market-exit target of 15% to 25% Citizens share is reconciled with the 85% transfer and 15% uninsured split. Likewise, specify the normalization that gives the 30% flood target from the SFHA and non-SFHA multipliers and coastal adjustment. The private capital scaling also needs its full formula.
- **Mitigation settings**. List all 13 reduction levels and define how the stated 3 to 2 ratio and 95% endpoint apply to wind and flood reductions.
- **Institutional inputs**. Provide the catastrophe bond inventory and trigger mappings. Verify assessment premium bases and collection horizons. The revised prose explicitly states that NFIP reinsurance, a borrowing ceiling, and competing claims from other states are not represented.

## Evidence-backed prose corrections

The former claim that every year with economic loss above USD 25 billion had an insured share above 40% conflicts with the 2024 row in Table S1. The text now reports the actual range and the reported subset means.

The former claim that every non-flood metric has a hazard variance share of at least 0.95 conflicts with the FHCF value of 0.944 in Table S7. The revised prose and caption state the exception. They also distinguish the selected 300-season design from a random sample of annual losses and avoid attributing each parameter's separate contribution without such a decomposition.

Table S8 no longer assigns blanket upper-bound or minimal-bias labels to untested assumptions. It describes the relevant limitation and whether its magnitude has been evaluated. The figure caption for county wind shares now correctly says events at or above the 95th percentile, rather than the upper 95th percentile. Table and figure cross-references are local and resolved.

## Bibliography treatment

The SI bibliography retains the original SI version of every existing entry. Entries needed from the main bibliography were added with no duplicate keys. Some shared keys differ between the two supplied bibliographies, including Artemis, Citizens county exposure, the NFIP fund balance, and several general sources. These were not silently overwritten. A final source-date harmonization across both manuscripts would be appropriate before submission.

