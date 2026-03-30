#!/usr/bin/env python3
"""
combine_probabilistic_tables.py - Combine probabilistic loss and institutional data tables
"""
import pandas as pd
from pathlib import Path

def format_value_with_range(median, p10, p90):
    """Format value as: median (p10-p90)"""
    return f"{median:.1f} ({p10:.1f}-{p90:.1f})"

def combine_probabilistic_tables(loss_csv, institutional_csv, output_csv):
    """
    Combine and reorganize probabilistic loss and institutional data.
    
    Parameters
    ----------
    loss_csv : str
        Path to probabilistic_loss_data.csv
    institutional_csv : str
        Path to probabilistic_institutional_data.csv
    output_csv : str
        Path to output combined CSV file
    """
    # Read both CSV files
    loss_df = pd.read_csv(loss_csv)
    inst_df = pd.read_csv(institutional_csv)
    
    # Define the scenario mapping (from CSV to desired column names)
    scenario_map = {
        "Baseline\n(ERA5)": "Baseline",
        "2050\nSSP2-4.5": "2050 ssp245",
        "2050\nSSP5-8.5": "2050 ssp585",
        "2100\nSSP2-4.5": "2100 ssp245",
        "2100\nSSP5-8.5": "2100 ssp585",
        "market_exit": "Market Exit",
        "penetration": "Ins. Penetration",
        "building_codes": "Building Codes"
    }
    
    # Define the metrics to extract with their display names
    metrics = [
        ("insured_wind", "Wind Insured (Private)"),
        ("citizens", "Wind Insured (Citizens)"),
        ("insured_flood", "Flood Insured (NFIP)"),
        ("wind_under", "Wind Un/Underinsured"),
        ("flood_under", "Flood Un/Underinsured"),
        ("total_loss", "Total Losses"),
        ("fhcf_shortfall", "FHCF Shortfall"),
        ("figa_residual", "FIGA Residual"),
        ("citizens_deficit", "Citizens Deficit"),
        ("nfip_borrowed", "NFIP Borrowed"),
    ]
    
    # Create the combined data structure
    result_data = []
    
    for metric_col, metric_name in metrics:
        row = {"Metric": metric_name}
        
        for scenario_original, scenario_display in scenario_map.items():
            # Determine which dataframe to use
            if metric_col in loss_df.columns:
                df = loss_df
            elif metric_col in inst_df.columns:
                df = inst_df
            else:
                continue
            
            # Find the row for this scenario
            scenario_data = df[df['Scenario'] == scenario_original]
            
            if not scenario_data.empty:
                median = scenario_data[metric_col].values[0]
                p10 = scenario_data[f"{metric_col}_p10"].values[0]
                p90 = scenario_data[f"{metric_col}_p90"].values[0]
                
                row[scenario_display] = format_value_with_range(median, p10, p90)
            else:
                row[scenario_display] = "N/A"
        
        result_data.append(row)
    
    # Add Total public burden row
    public_burden_row = {"Metric": "Total public burden"}
    
    for scenario_original, scenario_display in scenario_map.items():
        # Sum up: FHCF Shortfall, FIGA Residual, Citizens Deficit, NFIP Borrowed
        inst_data = inst_df[inst_df['Scenario'] == scenario_original]
        
        if not inst_data.empty:
            total_median = (
                inst_data['fhcf_shortfall'].values[0] +
                inst_data['figa_residual'].values[0] +
                inst_data['citizens_deficit'].values[0] +
                inst_data['nfip_borrowed'].values[0]
            )
            total_p10 = (
                inst_data['fhcf_shortfall_p10'].values[0] +
                inst_data['figa_residual_p10'].values[0] +
                inst_data['citizens_deficit_p10'].values[0] +
                inst_data['nfip_borrowed_p10'].values[0]
            )
            total_p90 = (
                inst_data['fhcf_shortfall_p90'].values[0] +
                inst_data['figa_residual_p90'].values[0] +
                inst_data['citizens_deficit_p90'].values[0] +
                inst_data['nfip_borrowed_p90'].values[0]
            )
            
            public_burden_row[scenario_display] = format_value_with_range(
                total_median, total_p10, total_p90
            )
        else:
            public_burden_row[scenario_display] = "N/A"
    
    result_data.append(public_burden_row)
    
    # Create final dataframe
    result_df = pd.DataFrame(result_data)
    
    # Ensure column order
    columns = ["Metric", "Baseline", "2050 ssp245", "2050 ssp585", 
               "2100 ssp245", "2100 ssp585", "Market Exit", 
               "Ins. Penetration", "Building Codes"]
    result_df = result_df[columns]
    
    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Combined table saved to: {output_csv}")
    
    return result_df, loss_df, inst_df


def format_latex_val(mean, p10, p90):
    """Format a value as 'X.X (X.X--X.X)' for LaTeX."""
    return f"{mean:.1f} ({p10:.1f}--{p90:.1f})"


def build_latex_table(loss_df, inst_df):
    """Build LaTeX table matching the publication format."""

    scenario_map = {
        "Baseline\n(ERA5)": "Baseline",
        "2050\nSSP2-4.5": "2050 SSP2-4.5",
        "2050\nSSP5-8.5": "2050 SSP5-8.5",
        "2100\nSSP2-4.5": "2100 SSP2-4.5",
        "2100\nSSP5-8.5": "2100 SSP5-8.5",
        "market_exit": "Market Exit",
        "penetration": "Penetration",
        "building_codes": "Building Codes",
    }

    # Loss decomposition metrics (order matches publication table)
    loss_metrics = [
        ("total_loss",    "Total loss"),
        ("insured_wind",  "Insured wind -- private"),
        ("citizens",      "Citizens wind"),
        ("insured_flood", "Insured flood -- NFIP"),
        ("wind_under",    "Un/underinsured wind"),
        ("flood_under",   "Un/underinsured flood"),
    ]

    # Institutional stress metrics
    inst_metrics = [
        ("fhcf_shortfall",    "FHCF shortfall"),
        ("figa_residual",     "FIGA residual"),
        ("citizens_deficit",  "Citizens deficit"),
        ("nfip_borrowed",     "NFIP Treasury borrowing"),
    ]

    scenarios = list(scenario_map.keys())

    def _get_val(df, scenario, col):
        row = df[df['Scenario'] == scenario]
        if row.empty:
            return "N/A"
        return format_latex_val(
            row[col].values[0],
            row[f"{col}_p10"].values[0],
            row[f"{col}_p90"].values[0],
        )

    def _row(label, df, col):
        vals = [_get_val(df, s, col) for s in scenarios]
        cells = " & ".join(vals)
        return f"{label:<34s} & {cells} \\\\"

    lines = [
        r"\begin{landscape}",
        r"\begin{table}[htb!]",
        r"\centering",
        r"\caption{\textbf{Loss decomposition and public institutional burden across climate and policy scenarios.}",
        r"Values show expected annual losses and institutional burdens under present-day baseline conditions, future climate scenarios (2050 and 2100 under SSP2-4.5 and SSP5-8.5), and illustrative market and policy interventions evaluated under present-day climate forcing. Values are reported as means in billion USD with uncertainty ranges (10th--90th percentile) in parentheses.}",
        r"\label{TabClimatePolicyLosses}",
        r"",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"",
        r"\begin{tabular}{@{}l|l|llll|lll@{}}",
        r"\toprule",
        r"\textbf{Metric (USD B)} &",
        r"\textbf{Baseline} &",
        r"\textbf{2050 SSP2-4.5} &",
        r"\textbf{2050 SSP5-8.5} &",
        r"\textbf{2100 SSP2-4.5} &",
        r"\textbf{2100 SSP5-8.5} &",
        r"\textbf{Market Exit} &",
        r"\textbf{Insurance Penetration} &",
        r"\textbf{Building Codes} \\",
        r"\midrule",
        r"",
        r"\multicolumn{9}{l}{\textit{Loss decomposition}} \\",
    ]

    for col, label in loss_metrics:
        lines.append(_row(label, loss_df, col))

    lines += [
        r"",
        r"\midrule",
        r"\multicolumn{9}{l}{\textit{Institutional stress}} \\",
    ]

    # Total public burden row (summed)
    burden_cols = ["fhcf_shortfall", "figa_residual", "citizens_deficit", "nfip_borrowed"]
    burden_vals = []
    for s in scenarios:
        row = inst_df[inst_df['Scenario'] == s]
        if row.empty:
            burden_vals.append("N/A")
        else:
            m = sum(row[c].values[0] for c in burden_cols)
            p10 = sum(row[f"{c}_p10"].values[0] for c in burden_cols)
            p90 = sum(row[f"{c}_p90"].values[0] for c in burden_cols)
            burden_vals.append(format_latex_val(m, p10, p90))
    lines.append(f"{'Total public burden':<34s} & {' & '.join(burden_vals)} \\\\")

    for col, label in inst_metrics:
        lines.append(_row(label, inst_df, col))

    lines += [
        r"",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Define paths
    base_path = Path(__file__).parent.parent.parent
    tables_path = base_path / "results" / "tables"
    
    loss_csv = tables_path / "probabilistic_loss_data.csv"
    institutional_csv = tables_path / "probabilistic_institutional_data.csv"
    output_csv = tables_path / "probabilistic_combined_table.csv"
    
    # Combine tables
    df, loss_df, inst_df = combine_probabilistic_tables(loss_csv, institutional_csv, output_csv)
    
    # Display preview
    print("\nPreview of combined table:")
    print(df.to_string(index=False))

    # Build and write LaTeX table
    latex = build_latex_table(loss_df, inst_df)
    tex_path = tables_path / "probabilistic_combined_table.tex"
    tex_path.write_text(latex)
    print(f"\nWrote LaTeX table -> {tex_path}")
