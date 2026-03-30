#!/usr/bin/env python3
"""
combine_systemic_risk_tables.py - Combine baseline, climate, and policy systemic risk tables
"""
import pandas as pd
from pathlib import Path


def format_value_with_range(median, p10, p90):
    """Format value as: median (p10-p90)"""
    return f"{median:.1f} ({p10:.1f}-{p90:.1f})"


def combine_systemic_risk_tables(baseline_csv, climate_csv, policy_csv, output_csv):
    """
    Combine three systemic risk comparison tables into one comprehensive table.
    
    Parameters
    ----------
    baseline_csv : str
        Path to era5_baseline_probabilities.csv
    climate_csv : str
        Path to era5_climate_comparison.csv
    policy_csv : str
        Path to era5_policy_comparison.csv
    output_csv : str
        Path to output combined CSV file
    """
    # Read the three CSV files
    df_baseline = pd.read_csv(baseline_csv)
    df_climate = pd.read_csv(climate_csv)
    df_policy = pd.read_csv(policy_csv)
    
    # Start with the Metric column from baseline
    result_df = df_baseline[['Metric']].copy()
    
    # Add Baseline column (from baseline file) - formatted as value (p10-p90)
    # The baseline file has: Metric, Annual Probability (%), P10, P90
    result_df['Baseline'] = df_baseline.apply(
        lambda row: format_value_with_range(
            row['Annual Probability (%)'], row['P10'], row['P90']
        ), axis=1
    )
    
    # Add Climate scenario columns (formatted as value (p10-p90))
    # Climate scenarios: 2050 SSP2-4.5, 2050 SSP5-8.5, 2100 SSP2-4.5, 2100 SSP5-8.5
    climate_scenarios = [
        '2050 SSP2-4.5',
        '2050 SSP5-8.5',
        '2100 SSP2-4.5',
        '2100 SSP5-8.5'
    ]
    
    for scenario in climate_scenarios:
        # Merge the metric column to ensure alignment
        if scenario in df_climate.columns:
            temp_df = df_climate[['Metric', scenario, f'{scenario}_p10', f'{scenario}_p90']].copy()
            temp_df[scenario + '_formatted'] = temp_df.apply(
                lambda row: format_value_with_range(
                    row[scenario], row[f'{scenario}_p10'], row[f'{scenario}_p90']
                ), axis=1
            )
            result_df = result_df.merge(
                temp_df[['Metric', scenario + '_formatted']], 
                on='Metric', 
                how='left'
            )
            result_df.rename(columns={scenario + '_formatted': scenario}, inplace=True)
    
    # Add Policy scenario columns (formatted as value (p10-p90))
    # Policy scenarios: Market Exit, Penetration, Building Codes
    policy_scenarios = ['Market Exit', 'Penetration', 'Building Codes']
    
    for scenario in policy_scenarios:
        if scenario in df_policy.columns:
            temp_df = df_policy[['Metric', scenario, f'{scenario}_P10', f'{scenario}_P90']].copy()
            temp_df[scenario + '_formatted'] = temp_df.apply(
                lambda row: format_value_with_range(
                    row[scenario], row[f'{scenario}_P10'], row[f'{scenario}_P90']
                ), axis=1
            )
            result_df = result_df.merge(
                temp_df[['Metric', scenario + '_formatted']], 
                on='Metric', 
                how='left'
            )
            result_df.rename(columns={scenario + '_formatted': scenario}, inplace=True)
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Combined systemic risk table saved to: {output_csv}")
    
    # Print summary
    print(f"\nCombined table contains:")
    print(f"  - {len(result_df)} metrics (rows)")
    print(f"  - {len(result_df.columns)} columns")
    print(f"\nColumn order:")
    for i, col in enumerate(result_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return result_df


def build_latex_table(df_baseline, df_climate, df_policy):
    """Build LaTeX table for systemic risk probabilities."""

    scenarios_climate = [
        '2050 SSP2-4.5', '2050 SSP5-8.5',
        '2100 SSP2-4.5', '2100 SSP5-8.5',
    ]
    scenarios_policy = ['Market Exit', 'Penetration', 'Building Codes']

    # Metrics in publication order, with LaTeX labels and separator positions
    metrics = [
        ("Defaults > 10",                 r"Defaults > 10",                   False),
        ("Single Deficit > $1B",          r"Single Deficit > \$1B",           True),
        ("FHCF > 100% Cap",              r"FHCF > 100\% Cap",               False),
        ("FIGA > 100% Capacity",          r"FIGA > 100\% Capacity",           False),
        ("Citizens > 100% Capacity",      r"Citizens > 100\% Capacity",       False),
        ("NFIP > 200% Annual Premium",    r"NFIP > 200\% Annual Premium",     True),
        ("Public Burden > 1% FL GDP",     r"Public Burden > 1\% FL GDP",      False),
        ("Public Burden > 10% FL GDP",    r"Public Burden > 10\% FL GDP",     False),
    ]

    def _fmt(mean, p10, p90):
        return f"{mean:.1f} ({p10:.1f}-{p90:.1f})"

    def _row(metric_key, metric_label):
        # Baseline
        brow = df_baseline[df_baseline['Metric'] == metric_key]
        if brow.empty:
            return f"{metric_label} & N/A & " + " & ".join(["N/A"] * 7) + r" \\"
        baseline = _fmt(brow['Annual Probability (%)'].values[0],
                        brow['P10'].values[0], brow['P90'].values[0])

        # Climate
        climate_vals = []
        for s in scenarios_climate:
            crow = df_climate[df_climate['Metric'] == metric_key]
            if crow.empty or s not in df_climate.columns:
                climate_vals.append("N/A")
            else:
                climate_vals.append(_fmt(
                    crow[s].values[0],
                    crow[f'{s}_p10'].values[0],
                    crow[f'{s}_p90'].values[0],
                ))

        # Policy
        policy_vals = []
        for s in scenarios_policy:
            prow = df_policy[df_policy['Metric'] == metric_key]
            if prow.empty or s not in df_policy.columns:
                policy_vals.append("N/A")
            else:
                policy_vals.append(_fmt(
                    prow[s].values[0],
                    prow[f'{s}_P10'].values[0],
                    prow[f'{s}_P90'].values[0],
                ))

        all_vals = [baseline] + climate_vals + policy_vals
        return f"{metric_label} &\n  " + " &\n  ".join(all_vals) + r" \\"

    lines = [
        r"\begin{table}[htb!]",
        r"\centering",
        r"\caption{\textbf{Probabilistic exceedance of systemic insurance stress thresholds across climate and policy scenarios.}",
        r"Annual exceedance probabilities (percent) of systemic stress thresholds for private insurers, public and quasi-public institutions, and aggregate fiscal impact, estimated from \num{10000} simulated tropical cyclone seasons. Results are shown for present-day conditions, mid-century (2050) and end-of-century (2100) climate scenarios under SSP2-4.5 and SSP5-8.5, as well as illustrative market and policy interventions evaluated under present-day climate forcing. Values denote mean probabilities, with uncertainty ranges (10th--90th percentile) in parentheses.}",
        r"\label{TabSystemicRiskProb}",
        r"",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"",
        r"\begin{tabular}{@{}l|l|llll|lll@{}}",
        r"\toprule",
        r"\textbf{Metric (\%)} &",
        r"  \textbf{Baseline} &",
        r"  \textbf{2050 SSP2-4.5} &",
        r"  \textbf{2050 SSP5-8.5} &",
        r"  \textbf{2100 SSP2-4.5} &",
        r"  \textbf{2100 SSP5-8.5} &",
        r"  \textbf{Market Exit} &",
        r"  \textbf{Insurance Penetration} &",
        r"  \textbf{Building Codes} \\ \midrule",
        r"",
    ]

    for metric_key, metric_label, add_midrule in metrics:
        lines.append(_row(metric_key, metric_label))
        lines.append("")
        if add_midrule:
            lines.append(r"\midrule")
            lines.append("")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\end{landscape}",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Define paths
    base_path = Path(__file__).parent.parent.parent
    tables_path = base_path / "results" / "tables"
    
    baseline_csv = tables_path / "era5_baseline_probabilities.csv"
    climate_csv = tables_path / "era5_climate_comparison.csv"
    policy_csv = tables_path / "era5_policy_comparison.csv"
    output_csv = tables_path / "systemic_risk_all_scenarios.csv"
    
    # Check if all input files exist
    missing_files = []
    for file_path in [baseline_csv, climate_csv, policy_csv]:
        if not file_path.exists():
            missing_files.append(file_path.name)
    
    if missing_files:
        print(f"[WARNING] Missing input files: {', '.join(missing_files)}")
        print("    Run the notebook cells to generate these files first.")
    else:
        # Combine tables
        df = combine_systemic_risk_tables(baseline_csv, climate_csv, policy_csv, output_csv)
        
        # Display preview (first 3 metrics, first 8 columns)
        print("\n" + "="*120)
        print("PREVIEW OF COMBINED TABLE (first 3 metrics, truncated columns)")
        print("="*120)
        preview_cols = df.columns[:8].tolist() if len(df.columns) > 8 else df.columns.tolist()
        print(df[preview_cols].head(3).to_string(index=False))
        print("...")

        # Build and write LaTeX table
        df_baseline = pd.read_csv(baseline_csv)
        df_climate = pd.read_csv(climate_csv)
        df_policy = pd.read_csv(policy_csv)
        latex = build_latex_table(df_baseline, df_climate, df_policy)
        tex_path = tables_path / "systemic_risk_all_scenarios.tex"
        tex_path.write_text(latex)
        print(f"\nWrote LaTeX table -> {tex_path}")