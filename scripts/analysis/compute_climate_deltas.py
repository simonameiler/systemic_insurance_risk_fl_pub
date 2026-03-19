#!/usr/bin/env python
"""
Compute climate change deltas from GCM ensemble and apply to ERA5 baseline.

This script:
1. Loads return_period_summary.csv from all MC runs (ERA5 + GCMs)
2. Parses GCM model and time period from directory names
3. Computes deltas for each GCM: Future - Historical (20thcal reference)
4. Aggregates deltas across GCM ensemble: median, p10, p90
5. Applies deltas to ERA5 using BOTH absolute and relative approaches
6. Outputs comparison CSVs for validation

Usage:
    python fl_risk_model/analysis/compute_climate_deltas.py \
        --mc_root results/mc_runs \
        --out results/climate_deltas

Output files:
    - climate_deltas_by_gcm.csv: Raw deltas per model/period
    - climate_deltas_ensemble_absolute.csv: Aggregated absolute deltas
    - climate_deltas_ensemble_relative.csv: Aggregated relative deltas
    - era5_climate_scaled_absolute.csv: ERA5 + absolute deltas
    - era5_climate_scaled_relative.csv: ERA5 × (1 + relative deltas)
    - comparison_absolute_vs_relative.csv: Side-by-side comparison
    - gcm_era5_alignment.csv: Validation of GCM historical vs ERA5
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_event_set_info(dirname):
    """
    Parse GCM model and period from directory name.
    
    Examples:
        emanuel_FL_canesm_20thcal_TIMESTAMP -> (canesm, 20thcal)
        emanuel_canesm_20thcal_baseline_TIMESTAMP -> (canesm, 20thcal)
        emanuel_canesm_ssp245cal_baseline_TIMESTAMP -> (canesm, ssp245cal)
        emanuel_era5_baseline_TIMESTAMP -> (era5, era5)
    
    Returns:
        (model, period) or (None, None) if not a GCM/ERA5 run
    """
    # Check for ERA5 first
    if 'era5' in dirname and 'baseline' in dirname:
        return ('era5', 'era5')
    
    # Pattern 1: emanuel_FL_{model}_{period}_TIMESTAMP
    match = re.match(r'emanuel_FL_(\w+)_(\w+)_\d+', dirname)
    if match:
        model = match.group(1)
        period = match.group(2)
        return (model, period)
    
    # Pattern 2: emanuel_{model}_{period}_baseline_TIMESTAMP (actual Sherlock format)
    # Need to handle: emanuel_canesm_20thcal_baseline or emanuel_canesm_ssp245_2cal_baseline
    # Use non-greedy match for model name to avoid capturing period prefix
    match = re.match(r'emanuel_([a-z0-9]+)_(20thcal|ssp245cal|ssp245_2cal|ssp585cal|ssp585_2cal)_baseline_\d+', dirname)
    if match:
        model = match.group(1)
        period = match.group(2)
        return (model, period)
    
    return (None, None)


def get_period_category(period):
    """
    Categorize time period into historical or future.
    
    Returns:
        'historical', 'near', 'mid', or None
    """
    if period == '20thcal':
        return 'historical'  # 1995-2014
    elif period in ['cal', 'ssp245cal', 'ssp585cal']:
        return 'near'  # 2041-2060 typically
    elif period in ['2cal', 'ssp245_2cal', 'ssp585_2cal']:
        return 'mid'  # 2081-2100 typically
    else:
        return None


def get_pathway(period):
    """
    Extract emission pathway from period name.
    
    Returns:
        'ssp245', 'ssp585', or None
    """
    if 'ssp245' in period or period in ['cal', '2cal']:
        return 'ssp245'
    elif 'ssp585' in period:
        return 'ssp585'
    else:
        return None


def compute_annual_average_metrics(df_iterations):
    """
    Compute annual average (mean) values from iteration-level data.
    This matches the approach in build_emanuel_summary.py.
    
    Returns:
        DataFrame with columns: metric, value (annual mean)
    """
    # Filter out error years
    df_valid = df_iterations[df_iterations['scenario'] != 'error'].copy()
    
    if len(df_valid) == 0:
        return None
    
    # Identify numeric columns (metrics)
    numeric_cols = df_valid.select_dtypes(include=[np.number]).columns
    exclude_cols = ['year', 'iteration', 'scenario_id']
    metric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    results = []
    
    for metric in metric_cols:
        try:
            # Compute mean across all years (annual expected value)
            mean_value = df_valid[metric].mean()
            results.append({
                'metric': metric,
                'value': mean_value
            })
        except:
            continue
    
    return pd.DataFrame(results) if results else None


def load_mc_results(mc_root):
    """
    Load iterations.csv from all MC run directories and compute annual average metrics.
    
    Returns:
        DataFrame with columns: model, period, time_category, metric, value (annual mean)
    """
    mc_root = Path(mc_root)
    
    results = []
    
    for run_dir in sorted(mc_root.glob('emanuel_*')):
        if not run_dir.is_dir():
            continue
        
        iterations_file = run_dir / 'iterations.csv'
        if not iterations_file.exists():
            print(f"⚠️  Skipping {run_dir.name}: no iterations.csv")
            continue
        
        # Parse model and period
        model, period = parse_event_set_info(run_dir.name)
        if model is None:
            print(f"⚠️  Skipping {run_dir.name}: could not parse model/period")
            continue
        
        time_category = get_period_category(period) if model != 'era5' else 'reanalysis'
        
        # Load iterations
        try:
            df_iterations = pd.read_csv(iterations_file)
        except Exception as e:
            print(f"⚠️  Error loading {iterations_file}: {e}")
            continue
        
        # Compute annual average metrics
        df_metrics = compute_annual_average_metrics(df_iterations)
        if df_metrics is None:
            print(f"⚠️  Skipping {run_dir.name}: could not compute metrics")
            continue
        
        # Add metadata
        df_metrics['model'] = model
        df_metrics['period'] = period
        df_metrics['time_category'] = time_category
        df_metrics['run_dir'] = run_dir.name
        
        results.append(df_metrics)
        print(f"✓ Loaded {run_dir.name}: {model}/{period} ({time_category}, {len(df_metrics)} metrics)")
    
    if not results:
        raise ValueError(f"No valid MC results found in {mc_root}")
    
    # Combine all results
    all_results = pd.concat(results, ignore_index=True)
    
    print(f"\n📊 Loaded {len(results)} MC runs:")
    print(f"   Models: {sorted(all_results['model'].unique())}")
    print(f"   Periods: {sorted(all_results['period'].unique())}")
    print(f"   Time categories: {sorted(all_results['time_category'].unique())}")
    
    return all_results


def compute_deltas_by_gcm(df):
    """
    Compute climate deltas for each GCM model.
    
    For each model:
        - Reference: 20thcal (historical, 1995-2014)
        - Future: cal, 2cal, ssp245cal, ssp245_2cal
        - Delta_abs = Future - Historical
        - Delta_rel = (Future - Historical) / Historical
    
    Returns:
        DataFrame with deltas for each model/period/metric combination
    """
    # Get list of GCM models (exclude era5)
    gcm_models = [m for m in df['model'].unique() if m != 'era5']
    
    print(f"\n🔬 Computing deltas for {len(gcm_models)} GCM models...")
    
    deltas = []
    
    for model in gcm_models:
        model_data = df[df['model'] == model]
        
        # Find historical reference (20thcal)
        hist_data = model_data[model_data['period'] == '20thcal']
        
        if hist_data.empty:
            print(f"⚠️  Model {model}: No historical (20thcal) data, skipping")
            continue
        
        # Get all future periods for this model
        future_periods = model_data[
            (model_data['time_category'].isin(['near', 'mid'])) &
            (model_data['period'] != '20thcal')
        ]['period'].unique()
        
        if len(future_periods) == 0:
            print(f"⚠️  Model {model}: No future periods found, skipping")
            continue
        
        print(f"   {model}: historical=20thcal, future={list(future_periods)}")
        
        # Compute deltas for each future period
        for future_period in future_periods:
            future_data = model_data[model_data['period'] == future_period]
            
            # For each metric, compute deltas (no return period grouping)
            # Merge on metric name only
            merged = hist_data[['metric', 'value']].merge(
                future_data[['metric', 'value']],
                on='metric',
                suffixes=('_hist', '_future')
            )
            
            if merged.empty:
                continue
            
            # Compute deltas for each metric
            for _, row in merged.iterrows():
                hist_val = row['value_hist']
                future_val = row['value_future']
                
                # Absolute delta
                delta_abs = future_val - hist_val
                
                # Relative delta (handle division by zero)
                if hist_val != 0:
                    delta_rel = (future_val - hist_val) / hist_val
                else:
                    delta_rel = np.nan
                
                deltas.append({
                    'model': model,
                    'period': future_period,
                    'time_category': get_period_category(future_period),
                    'pathway': get_pathway(future_period),
                    'metric': row['metric'],
                    'historical_value': hist_val,
                    'future_value': future_val,
                    'delta_absolute': delta_abs,
                    'delta_relative': delta_rel,
                })
    
    deltas_df = pd.DataFrame(deltas)
    
    print(f"\n✓ Computed deltas: {len(deltas_df)} rows")
    print(f"   Models: {deltas_df['model'].nunique()}")
    print(f"   Metrics: {deltas_df['metric'].nunique()}")
    
    return deltas_df


def aggregate_deltas_ensemble(deltas_df, method='absolute'):
    """
    Aggregate deltas across GCM ensemble.
    
    For each time_category/metric:
        - Median (central estimate)
        - 10th percentile (lower bound)
        - 90th percentile (upper bound)
        - Mean (alternative central estimate)
        - Std dev (spread)
    
    Parameters:
        method: 'absolute' or 'relative'
    
    Returns:
        DataFrame with ensemble statistics
    """
    delta_col = 'delta_absolute' if method == 'absolute' else 'delta_relative'
    
    print(f"\n📊 Aggregating {method} deltas across ensemble...")
    
    # Group by pathway, time_category, and metric
    grouped = deltas_df.groupby(['pathway', 'time_category', 'metric'])
    
    ensemble = []
    
    for (pathway, time_cat, metric), group in grouped:
        deltas = group[delta_col].dropna()
        
        if len(deltas) == 0:
            continue
        
        ensemble.append({
            'pathway': pathway,
            'time_category': time_cat,
            'metric': metric,
            'n_models': len(deltas),
            'delta_median': deltas.median(),
            'delta_mean': deltas.mean(),
            'delta_std': deltas.std(),
            'delta_p10': deltas.quantile(0.1),
            'delta_p90': deltas.quantile(0.9),
            'delta_min': deltas.min(),
            'delta_max': deltas.max(),
        })
    
    ensemble_df = pd.DataFrame(ensemble)
    
    print(f"✓ Ensemble statistics computed: {len(ensemble_df)} rows")
    print(f"   Pathways: {sorted(ensemble_df['pathway'].unique())}")
    print(f"   Time categories: {sorted(ensemble_df['time_category'].unique())}")
    
    return ensemble_df


def apply_deltas_to_era5(era5_data, ensemble_deltas, method='absolute'):
    """
    Apply ensemble deltas to ERA5 baseline.
    
    Parameters:
        era5_data: DataFrame with ERA5 annual average metrics
        ensemble_deltas: DataFrame with aggregated deltas
        method: 'absolute' or 'relative'
    
    Returns:
        DataFrame with scaled ERA5 results (median, p10, p90)
    """
    print(f"\n🔧 Applying {method} deltas to ERA5...")
    
    scaled = []
    
    for pathway in ensemble_deltas['pathway'].unique():
        pathway_deltas = ensemble_deltas[ensemble_deltas['pathway'] == pathway]
        
        for time_cat in pathway_deltas['time_category'].unique():
            time_deltas = pathway_deltas[pathway_deltas['time_category'] == time_cat]
            
            for _, delta_row in time_deltas.iterrows():
                metric = delta_row['metric']
                
                # Find corresponding ERA5 value (in long format, no return_period)
                era5_row = era5_data[era5_data['metric'] == metric]
                
                if era5_row.empty:
                    continue
                
                era5_value = era5_row['value'].iloc[0]
                
                if pd.isna(era5_value):
                    continue
                
                # Apply deltas
                if method == 'absolute':
                    scaled_median = era5_value + delta_row['delta_median']
                    scaled_p10 = era5_value + delta_row['delta_p10']
                    scaled_p90 = era5_value + delta_row['delta_p90']
                else:  # relative
                    scaled_median = era5_value * (1 + delta_row['delta_median'])
                    scaled_p10 = era5_value * (1 + delta_row['delta_p10'])
                    scaled_p90 = era5_value * (1 + delta_row['delta_p90'])
                
                scaled.append({
                    'pathway': pathway,
                    'time_category': time_cat,
                    'metric': metric,
                    'era5_baseline': era5_value,
                    'delta_median': delta_row['delta_median'],
                    'delta_p10': delta_row['delta_p10'],
                    'delta_p90': delta_row['delta_p90'],
                    'scaled_median': scaled_median,
                    'scaled_p10': scaled_p10,
                    'scaled_p90': scaled_p90,
                    'n_models': delta_row['n_models'],
                })
    
    scaled_df = pd.DataFrame(scaled)
    
    print(f"✓ Applied deltas: {len(scaled_df)} rows")
    
    return scaled_df


def validate_gcm_era5_alignment(all_results):
    """
    Validate how well GCM historical (20thcal) aligns with ERA5.
    
    Computes bias: (GCM_20thcal - ERA5) / ERA5 for each model.
    
    Returns:
        DataFrame with alignment metrics
    """
    print(f"\n🔍 Validating GCM historical vs ERA5 alignment...")
    
    # Get ERA5 data
    era5_data = all_results[all_results['model'] == 'era5']
    
    if era5_data.empty:
        print("⚠️  No ERA5 data found, skipping alignment validation")
        return pd.DataFrame()
    
    # Get GCM historical data (20thcal)
    gcm_hist = all_results[all_results['period'] == '20thcal']
    
    if gcm_hist.empty:
        print("⚠️  No GCM historical (20thcal) data found, skipping alignment validation")
        return pd.DataFrame()
    
    alignment = []
    
    for model in gcm_hist['model'].unique():
        model_hist = gcm_hist[gcm_hist['model'] == model]
        
        # For each metric, compare GCM vs ERA5 (no return period)
        for _, gcm_row in model_hist.iterrows():
            metric = gcm_row['metric']
            gcm_val = gcm_row['value']
            
            # Find matching ERA5 value
            era5_row = era5_data[era5_data['metric'] == metric]
            
            if era5_row.empty:
                continue
            
            era5_val = era5_row['value'].iloc[0]
            
            if pd.isna(era5_val) or pd.isna(gcm_val):
                continue
            
            # Compute bias
            bias_abs = gcm_val - era5_val
            bias_rel = (gcm_val - era5_val) / era5_val if era5_val != 0 else np.nan
            
            alignment.append({
                'model': model,
                'metric': metric,
                'era5_value': era5_val,
                'gcm_20thcal_value': gcm_val,
                'bias_absolute': bias_abs,
                'bias_relative': bias_rel,
            })
    
    alignment_df = pd.DataFrame(alignment)
    
    if not alignment_df.empty:
        print(f"✓ Alignment validation: {len(alignment_df)} comparisons")
        
        # Summary statistics
        print(f"\n📈 Bias Summary (median across all metrics):")
        for model in alignment_df['model'].unique():
            model_data = alignment_df[alignment_df['model'] == model]
            median_bias_rel = model_data['bias_relative'].median() * 100
            print(f"   {model}: {median_bias_rel:+.1f}%")
    
    return alignment_df


def compare_absolute_vs_relative(scaled_abs, scaled_rel):
    """
    Create side-by-side comparison of absolute vs relative scaling.
    
    Returns:
        DataFrame comparing both approaches
    """
    print(f"\n🔀 Comparing absolute vs relative scaling approaches...")
    
    # Merge on time_category, metric (no return_period)
    comparison = scaled_abs.merge(
        scaled_rel,
        on=['time_category', 'metric'],
        suffixes=('_abs', '_rel')
    )
    
    # Compute difference between approaches
    comparison['diff_median'] = comparison['scaled_median_abs'] - comparison['scaled_median_rel']
    comparison['diff_median_pct'] = (
        (comparison['scaled_median_abs'] - comparison['scaled_median_rel']) / 
        comparison['scaled_median_rel'] * 100
    )
    
    print(f"✓ Comparison created: {len(comparison)} rows")
    
    return comparison


def main():
    ap = argparse.ArgumentParser(
        description="Compute climate change deltas from GCM ensemble and apply to ERA5"
    )
    ap.add_argument("--mc_root", required=True,
                    help="Root directory containing all MC run directories")
    ap.add_argument("--out", required=True,
                    help="Output directory for delta files")
    ap.add_argument("--min_models", type=int, default=3,
                    help="Minimum number of models required for ensemble stats (default: 3)")
    args = ap.parse_args()
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CLIMATE DELTA ANALYSIS")
    print("=" * 80)
    
    # 1. Load all MC results
    all_results = load_mc_results(args.mc_root)
    
    # 2. Validate GCM-ERA5 alignment
    alignment = validate_gcm_era5_alignment(all_results)
    if not alignment.empty:
        alignment_file = out_dir / 'gcm_era5_alignment.csv'
        alignment.to_csv(alignment_file, index=False)
        print(f"\n💾 Saved: {alignment_file}")
    
    # 3. Compute deltas by GCM
    deltas = compute_deltas_by_gcm(all_results)
    
    if deltas.empty:
        print("\n❌ No deltas computed. Check that you have both historical and future GCM runs.")
        return
    
    deltas_file = out_dir / 'climate_deltas_by_gcm.csv'
    deltas.to_csv(deltas_file, index=False)
    print(f"\n💾 Saved: {deltas_file}")
    
    # 4. Aggregate deltas - ABSOLUTE
    ensemble_abs = aggregate_deltas_ensemble(deltas, method='absolute')
    ensemble_abs_file = out_dir / 'climate_deltas_ensemble_absolute.csv'
    ensemble_abs.to_csv(ensemble_abs_file, index=False)
    print(f"💾 Saved: {ensemble_abs_file}")
    
    # 5. Aggregate deltas - RELATIVE
    ensemble_rel = aggregate_deltas_ensemble(deltas, method='relative')
    ensemble_rel_file = out_dir / 'climate_deltas_ensemble_relative.csv'
    ensemble_rel.to_csv(ensemble_rel_file, index=False)
    print(f"💾 Saved: {ensemble_rel_file}")
    
    # 6. Get ERA5 baseline data
    era5_data = all_results[all_results['model'] == 'era5']
    
    if era5_data.empty:
        print("\n⚠️  No ERA5 data found. Cannot apply deltas to baseline.")
        return
    
    # 7. Apply deltas to ERA5 - ABSOLUTE
    scaled_abs = apply_deltas_to_era5(era5_data, ensemble_abs, method='absolute')
    scaled_abs_file = out_dir / 'era5_climate_scaled_absolute.csv'
    scaled_abs.to_csv(scaled_abs_file, index=False)
    print(f"💾 Saved: {scaled_abs_file}")
    
    # 8. Apply deltas to ERA5 - RELATIVE
    scaled_rel = apply_deltas_to_era5(era5_data, ensemble_rel, method='relative')
    scaled_rel_file = out_dir / 'era5_climate_scaled_relative.csv'
    scaled_rel.to_csv(scaled_rel_file, index=False)
    print(f"💾 Saved: {scaled_rel_file}")
    
    # 9. Compare absolute vs relative
    comparison = compare_absolute_vs_relative(scaled_abs, scaled_rel)
    comparison_file = out_dir / 'comparison_absolute_vs_relative.csv'
    comparison.to_csv(comparison_file, index=False)
    print(f"💾 Saved: {comparison_file}")
    
    print("\n" + "=" * 80)
    print("✅ CLIMATE DELTA ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {out_dir}")
    print(f"\nKey files:")
    print(f"  1. {alignment_file.name} - GCM vs ERA5 validation")
    print(f"  2. {deltas_file.name} - Raw deltas per model")
    print(f"  3. {ensemble_abs_file.name} - Ensemble absolute deltas")
    print(f"  4. {ensemble_rel_file.name} - Ensemble relative deltas")
    print(f"  5. {scaled_abs_file.name} - ERA5 scaled (absolute)")
    print(f"  6. {scaled_rel_file.name} - ERA5 scaled (relative)")
    print(f"  7. {comparison_file.name} - Method comparison")
    print("\n📊 Next steps:")
    print("  - Review gcm_era5_alignment.csv to check model bias")
    print("  - Compare absolute vs relative scaling in comparison file")
    print("  - Visualize results using scaled ERA5 files")


if __name__ == "__main__":
    main()
