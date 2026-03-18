"""
Combine baseline, climate, and policy systemic risk probability tables.
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
        print(f"⚠️  Missing input files: {', '.join(missing_files)}")
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