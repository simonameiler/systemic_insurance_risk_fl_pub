#!/usr/bin/env python
"""
Analyze Penetration Capital Sensitivity Results
===============================================

Analyzes Monte Carlo results across different capital scaling multipliers
to find the optimal capital requirement that stabilizes defaults.

This script:
1. Loads all sensitivity run results
2. Compares to baseline (no penetration increase)
3. Identifies the capital multiplier where defaults stabilize
4. Generates publication-ready figures and tables

Key Metrics:
- Mean company defaults
- Probability of systemic stress (10+ defaults)
- FIGA deficit levels
- Cost-benefit analysis of capital requirements

Usage:
    python analyze_penetration_capital_sensitivity.py
    
    # With custom baseline
    python analyze_penetration_capital_sensitivity.py --baseline results/mc_runs/baseline_20250120/

Output:
    - results/analysis/penetration_capital_sensitivity/
        - summary_table.csv
        - optimal_multiplier.json
        - figures/
            - defaults_vs_multiplier.png
            - stress_probability_curve.png
            - figa_vs_multiplier.png

Author: Simona Meiler
Date: 2026-01-23
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# =============================================================================
# Helper Functions
# =============================================================================

def load_sensitivity_results(results_dir: Path = Path("results/mc_runs")) -> pd.DataFrame:
    """
    Load all penetration capital sensitivity results.
    
    Returns
    -------
    pd.DataFrame
        Combined results with columns:
        - capital_multiplier
        - All MC iteration metrics
    """
    sensitivity_dirs = sorted(results_dir.glob("penetration_capital_m*"))
    
    if not sensitivity_dirs:
        raise FileNotFoundError(
            f"No sensitivity results found in {results_dir}.\n"
            "Run: python run_penetration_capital_sensitivity.py --all"
        )
    
    all_results = []
    
    for run_dir in sensitivity_dirs:
        # Load iterations
        iterations_file = run_dir / "iterations.csv"
        if not iterations_file.exists():
            print(f"Warning: Skipping {run_dir.name} (no iterations.csv)")
            continue
        
        df = pd.read_csv(iterations_file)
        
        # Load config to get multiplier
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                multiplier = config.get("capital_multiplier")
        else:
            # Try to parse from directory name
            try:
                multiplier = float(run_dir.name.split("_m")[1].split("_")[0])
            except:
                print(f"Warning: Cannot parse multiplier from {run_dir.name}")
                continue
        
        df["capital_multiplier"] = multiplier
        df["run_dir"] = str(run_dir)
        
        all_results.append(df)
    
    if not all_results:
        raise ValueError("No valid sensitivity results found")
    
    combined = pd.concat(all_results, ignore_index=True)
    
    print(f"✓ Loaded {len(sensitivity_dirs)} sensitivity runs")
    print(f"  Multipliers: {sorted(combined['capital_multiplier'].unique())}")
    print(f"  Total iterations: {len(combined):,}")
    
    return combined


def load_baseline_results(baseline_path: Path = None) -> pd.DataFrame:
    """
    Load baseline results (no penetration increase).
    
    If baseline_path not provided, searches for most recent baseline run.
    """
    if baseline_path is None:
        # Search for baseline runs
        results_dir = Path("results/mc_runs")
        baseline_dirs = sorted(results_dir.glob("*baseline*"))
        
        if not baseline_dirs:
            raise FileNotFoundError(
                "No baseline results found. Run baseline scenario first:\n"
                "python -m fl_risk_model.mc_run_events --mode single --policy baseline --n_iter 200"
            )
        
        # Use most recent
        baseline_path = baseline_dirs[-1]
        print(f"Using baseline: {baseline_path.name}")
    
    iterations_file = baseline_path / "iterations.csv"
    
    if not iterations_file.exists():
        raise FileNotFoundError(f"No iterations.csv in {baseline_path}")
    
    df = pd.read_csv(iterations_file)
    df["capital_multiplier"] = 0.0  # Mark as baseline (no multiplier)
    
    print(f"✓ Loaded baseline: {len(df)} iterations")
    
    return df


def compute_summary_statistics(df: pd.DataFrame, group_by: str = "capital_multiplier") -> pd.DataFrame:
    """
    Compute summary statistics for each capital multiplier.
    
    Returns
    -------
    pd.DataFrame
        Summary with one row per multiplier
    """
    summary = df.groupby(group_by).agg({
        "defaults_post": ["mean", "std", "max"],
        "wind_insured_private_usd": "mean",
        "total_damage_usd": "mean",
        "figa_residual_deficit_usd": "mean",
        "citizens_residual_deficit_usd": "mean",
        "nfip_borrowed_usd": "mean",
    }).round(2)
    
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    
    # Add probability metrics
    for multiplier in df[group_by].unique():
        subset = df[df[group_by] == multiplier]
        
        summary.loc[multiplier, "prob_any_default"] = (subset["defaults_post"] > 0).mean()
        summary.loc[multiplier, "prob_10plus_defaults"] = (subset["defaults_post"] >= 10).mean()
        summary.loc[multiplier, "prob_20plus_defaults"] = (subset["defaults_post"] >= 20).mean()
        summary.loc[multiplier, "n_iterations"] = len(subset)
    
    summary = summary.reset_index()
    
    return summary


def find_optimal_multiplier(
    summary_df: pd.DataFrame,
    baseline_defaults: float,
    baseline_prob_10plus: float,
    tolerance: float = 0.05,
) -> dict:
    """
    Find the capital multiplier that stabilizes defaults to baseline levels.
    
    Uses interpolation to find the exact multiplier where defaults match baseline.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics per multiplier
    baseline_defaults : float
        Target mean defaults (from baseline)
    baseline_prob_10plus : float
        Target probability of 10+ defaults
    tolerance : float
        Acceptable tolerance for matching baseline (default: 5%)
        
    Returns
    -------
    dict
        Optimal multiplier and diagnostics
    """
    # Sort by multiplier
    summary_sorted = summary_df.sort_values("capital_multiplier")
    
    multipliers = summary_sorted["capital_multiplier"].values
    mean_defaults = summary_sorted["defaults_post_mean"].values
    prob_10plus = summary_sorted["prob_10plus_defaults"].values
    
    # Interpolate to find exact crossing point
    # For mean defaults
    if mean_defaults.min() <= baseline_defaults <= mean_defaults.max():
        f_defaults = interpolate.interp1d(mean_defaults, multipliers, kind="linear")
        optimal_mean = float(f_defaults(baseline_defaults))
    else:
        # Extrapolate if needed
        if baseline_defaults < mean_defaults.min():
            optimal_mean = multipliers[0]
        else:
            optimal_mean = multipliers[-1]
    
    # For probability of 10+ defaults
    if prob_10plus.min() <= baseline_prob_10plus <= prob_10plus.max():
        f_prob = interpolate.interp1d(prob_10plus, multipliers, kind="linear")
        optimal_prob = float(f_prob(baseline_prob_10plus))
    else:
        if baseline_prob_10plus < prob_10plus.min():
            optimal_prob = multipliers[0]
        else:
            optimal_prob = multipliers[-1]
    
    # Use average of both metrics
    optimal_combined = (optimal_mean + optimal_prob) / 2
    
    # Find closest tested multiplier
    closest_idx = np.argmin(np.abs(multipliers - optimal_combined))
    closest_multiplier = multipliers[closest_idx]
    closest_defaults = mean_defaults[closest_idx]
    closest_prob = prob_10plus[closest_idx]
    
    return {
        "optimal_multiplier_mean": optimal_mean,
        "optimal_multiplier_prob": optimal_prob,
        "optimal_multiplier_combined": optimal_combined,
        "closest_tested_multiplier": closest_multiplier,
        "closest_mean_defaults": closest_defaults,
        "closest_prob_10plus": closest_prob,
        "baseline_mean_defaults": baseline_defaults,
        "baseline_prob_10plus": baseline_prob_10plus,
        "difference_mean_defaults_pct": ((closest_defaults - baseline_defaults) / baseline_defaults) * 100,
        "difference_prob_10plus_pct": ((closest_prob - baseline_prob_10plus) / baseline_prob_10plus) * 100 if baseline_prob_10plus > 0 else 0,
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_defaults_vs_multiplier(
    summary_df: pd.DataFrame,
    baseline_defaults: float,
    output_dir: Path,
):
    """Plot mean defaults vs capital multiplier with baseline reference."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    multipliers = summary_df["capital_multiplier"].values
    mean_defaults = summary_df["defaults_post_mean"].values
    std_defaults = summary_df["defaults_post_std"].values
    
    # Plot sensitivity curve
    ax.plot(multipliers, mean_defaults, "o-", linewidth=2, markersize=8, label="Penetration MAJOR")
    ax.fill_between(
        multipliers,
        mean_defaults - std_defaults,
        mean_defaults + std_defaults,
        alpha=0.2,
    )
    
    # Baseline reference
    ax.axhline(baseline_defaults, color="red", linestyle="--", linewidth=2, label="Baseline (no penetration)")
    
    ax.set_xlabel("Capital Scaling Multiplier", fontsize=12)
    ax.set_ylabel("Mean Company Defaults", fontsize=12)
    ax.set_title("Capital Requirements to Stabilize Defaults\nunder Increased Penetration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "defaults_vs_multiplier.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: defaults_vs_multiplier.png")


def plot_stress_probability_curve(
    summary_df: pd.DataFrame,
    baseline_prob_10plus: float,
    output_dir: Path,
):
    """Plot probability of systemic stress vs capital multiplier."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    multipliers = summary_df["capital_multiplier"].values
    prob_10plus = summary_df["prob_10plus_defaults"].values * 100  # Convert to percentage
    
    ax.plot(multipliers, prob_10plus, "o-", linewidth=2, markersize=8, color="darkred", label="P(10+ defaults)")
    
    # Baseline reference
    ax.axhline(baseline_prob_10plus * 100, color="red", linestyle="--", linewidth=2, label="Baseline")
    
    ax.set_xlabel("Capital Scaling Multiplier", fontsize=12)
    ax.set_ylabel("Probability of Systemic Stress (%)", fontsize=12)
    ax.set_title("Systemic Risk Mitigation through Capital Requirements", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "stress_probability_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: stress_probability_curve.png")


def plot_figa_vs_multiplier(
    summary_df: pd.DataFrame,
    baseline_figa: float,
    output_dir: Path,
):
    """Plot FIGA deficit vs capital multiplier."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    multipliers = summary_df["capital_multiplier"].values
    figa_deficit = summary_df["figa_residual_deficit_usd_mean"].values / 1e9  # Convert to billions
    
    ax.plot(multipliers, figa_deficit, "o-", linewidth=2, markersize=8, color="navy", label="FIGA Deficit")
    
    # Baseline reference
    ax.axhline(baseline_figa / 1e9, color="red", linestyle="--", linewidth=2, label="Baseline")
    
    ax.set_xlabel("Capital Scaling Multiplier", fontsize=12)
    ax.set_ylabel("Mean FIGA Deficit ($B)", fontsize=12)
    ax.set_title("Public Sector Burden vs Private Capital Requirements", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figa_vs_multiplier.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: figa_vs_multiplier.png")


def plot_combined_dashboard(
    summary_df: pd.DataFrame,
    baseline_stats: dict,
    optimal_info: dict,
    output_dir: Path,
):
    """Create combined dashboard with all key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    multipliers = summary_df["capital_multiplier"].values
    
    # 1. Mean defaults
    ax = axes[0, 0]
    mean_defaults = summary_df["defaults_post_mean"].values
    ax.plot(multipliers, mean_defaults, "o-", linewidth=2, markersize=8)
    ax.axhline(baseline_stats["mean_defaults"], color="red", linestyle="--", linewidth=2, label="Baseline")
    ax.axvline(optimal_info["optimal_multiplier_combined"], color="green", linestyle=":", linewidth=2, alpha=0.7, label="Optimal")
    ax.set_xlabel("Capital Multiplier")
    ax.set_ylabel("Mean Defaults")
    ax.set_title("Company Defaults")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Probability of stress
    ax = axes[0, 1]
    prob_10plus = summary_df["prob_10plus_defaults"].values * 100
    ax.plot(multipliers, prob_10plus, "o-", linewidth=2, markersize=8, color="darkred")
    ax.axhline(baseline_stats["prob_10plus"] * 100, color="red", linestyle="--", linewidth=2, label="Baseline")
    ax.axvline(optimal_info["optimal_multiplier_combined"], color="green", linestyle=":", linewidth=2, alpha=0.7, label="Optimal")
    ax.set_xlabel("Capital Multiplier")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Systemic Stress (10+ Defaults)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. FIGA deficit
    ax = axes[1, 0]
    figa_deficit = summary_df["figa_residual_deficit_usd_mean"].values / 1e9
    ax.plot(multipliers, figa_deficit, "o-", linewidth=2, markersize=8, color="navy")
    ax.axhline(baseline_stats["figa_deficit"] / 1e9, color="red", linestyle="--", linewidth=2, label="Baseline")
    ax.axvline(optimal_info["optimal_multiplier_combined"], color="green", linestyle=":", linewidth=2, alpha=0.7, label="Optimal")
    ax.set_xlabel("Capital Multiplier")
    ax.set_ylabel("FIGA Deficit ($B)")
    ax.set_title("Public Sector Burden")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Private insurer losses
    ax = axes[1, 1]
    private_loss = summary_df["wind_insured_private_usd_mean"].values / 1e9
    ax.plot(multipliers, private_loss, "o-", linewidth=2, markersize=8, color="darkgreen")
    ax.axhline(baseline_stats["private_loss"] / 1e9, color="red", linestyle="--", linewidth=2, label="Baseline")
    ax.axvline(optimal_info["optimal_multiplier_combined"], color="green", linestyle=":", linewidth=2, alpha=0.7, label="Optimal")
    ax.set_xlabel("Capital Multiplier")
    ax.set_ylabel("Private Loss ($B)")
    ax.set_title("Private Insurer Payouts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Penetration Capital Sensitivity Dashboard", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Saved: sensitivity_dashboard.png")


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze penetration capital sensitivity results")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline results directory (default: auto-detect)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/mc_runs"),
        help="Directory containing sensitivity results",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("PENETRATION CAPITAL SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path("results/analysis/penetration_capital_sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Load data
    print("[1/6] Loading sensitivity results...")
    sensitivity_df = load_sensitivity_results(args.results_dir)
    
    print("[2/6] Loading baseline results...")
    baseline_df = load_baseline_results(args.baseline)
    
    # Compute summary statistics
    print("[3/6] Computing summary statistics...")
    
    sensitivity_summary = compute_summary_statistics(sensitivity_df)
    baseline_summary = compute_summary_statistics(baseline_df)
    
    # Baseline stats
    baseline_stats = {
        "mean_defaults": baseline_summary["defaults_post_mean"].iloc[0],
        "std_defaults": baseline_summary["defaults_post_std"].iloc[0],
        "prob_any": baseline_summary["prob_any_default"].iloc[0],
        "prob_10plus": baseline_summary["prob_10plus_defaults"].iloc[0],
        "prob_20plus": baseline_summary["prob_20plus_defaults"].iloc[0],
        "figa_deficit": baseline_summary["figa_residual_deficit_usd_mean"].iloc[0],
        "citizens_deficit": baseline_summary["citizens_residual_deficit_usd_mean"].iloc[0],
        "private_loss": baseline_summary["wind_insured_private_usd_mean"].iloc[0],
    }
    
    print(f"\nBaseline Statistics:")
    print(f"  Mean defaults: {baseline_stats['mean_defaults']:.2f}")
    print(f"  P(10+ defaults): {baseline_stats['prob_10plus']*100:.1f}%")
    print(f"  FIGA deficit: ${baseline_stats['figa_deficit']/1e9:.2f}B")
    print()
    
    # Find optimal multiplier
    print("[4/6] Finding optimal capital multiplier...")
    optimal_info = find_optimal_multiplier(
        sensitivity_summary,
        baseline_stats["mean_defaults"],
        baseline_stats["prob_10plus"],
    )
    
    print(f"\nOptimal Capital Multiplier: {optimal_info['optimal_multiplier_combined']:.2f}x")
    print(f"  (Mean defaults metric: {optimal_info['optimal_multiplier_mean']:.2f}x)")
    print(f"  (Systemic stress metric: {optimal_info['optimal_multiplier_prob']:.2f}x)")
    print(f"\nClosest tested multiplier: {optimal_info['closest_tested_multiplier']:.2f}x")
    print(f"  Mean defaults: {optimal_info['closest_mean_defaults']:.2f} (baseline: {baseline_stats['mean_defaults']:.2f})")
    print(f"  P(10+ defaults): {optimal_info['closest_prob_10plus']*100:.1f}% (baseline: {baseline_stats['prob_10plus']*100:.1f}%)")
    print()
    
    # Save results
    print("[5/6] Saving results...")
    
    sensitivity_summary.to_csv(output_dir / "summary_table.csv", index=False)
    print(f"  ✓ Saved: summary_table.csv")
    
    with open(output_dir / "baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"  ✓ Saved: baseline_stats.json")
    
    with open(output_dir / "optimal_multiplier.json", "w") as f:
        json.dump(optimal_info, f, indent=2)
    print(f"  ✓ Saved: optimal_multiplier.json")
    
    # Generate plots
    print("[6/6] Generating figures...")
    
    plot_defaults_vs_multiplier(sensitivity_summary, baseline_stats["mean_defaults"], figures_dir)
    plot_stress_probability_curve(sensitivity_summary, baseline_stats["prob_10plus"], figures_dir)
    plot_figa_vs_multiplier(sensitivity_summary, baseline_stats["figa_deficit"], figures_dir)
    plot_combined_dashboard(sensitivity_summary, baseline_stats, optimal_info, figures_dir)
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📊 KEY FINDING:")
    print(f"   To maintain baseline default rates under MAJOR penetration increase,")
    print(f"   insurers need capital reserves to scale {optimal_info['optimal_multiplier_combined']:.2f}x faster than exposure.")
    print(f"\n   This means:")
    print(f"   • If exposure increases by 20%, capital must increase by {optimal_info['optimal_multiplier_combined']*20:.0f}%")
    print(f"   • Linear scaling (1.0x) is INSUFFICIENT")
    print(f"   • Required capital scaling is {optimal_info['optimal_multiplier_combined']:.1f}x proportional growth")
    print(f"\n📁 Results saved to: {output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
