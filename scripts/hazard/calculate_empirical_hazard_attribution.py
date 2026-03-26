"""
Empirical Wind/Water Attribution Using Direct Hazard Ratios

This script generates the wind/water attribution file used by the historical
scenario analysis notebook:
- florida_empirical_hazard_attribution_p95.csv

METHODOLOGY (Gori-weighted P95):
1. Normalizes hazards by county-specific climatology (95th percentile)
2. Identifies extreme compound hazard events (top 5%)
   - Uses Gori et al. (2022) regression coefficients (beta_W, beta_R, beta_S) to weight hazards
   - Creates compound hazard metric: beta_W*wind_norm + beta_R*rain_norm + beta_S*surge_norm
   - Applies 95th percentile threshold to this weighted metric
3. Calculates empirical wind/rain/surge composition for those extreme events
   - Averages the weighted normalized hazards for extreme events only
   - Converts to percentage shares
4. Avoids assuming damage functions without exposure/vulnerability data
   - Betas used only to define "severity" and weight contributions
   - Final attribution is empirical from observed hazard patterns

Author: Simona Meiler
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path("fl_risk_model/data")
GORI_PRESENT = DATA_DIR / "fl_per_event_impacts.csv"
COUNTY_REGION = DATA_DIR / "county_region.csv"
COASTAL_CLASSIFICATION = DATA_DIR / "florida_coastal_counties.csv"

OUTPUT_PRESENT = DATA_DIR / "florida_empirical_hazard_attribution_p95.csv"

# Configuration
PERCENTILE = 95  # Focus on top 5% most extreme events

# Gori regression coefficients from Table S1 (Gori et al. 2022)
# Florida is in Southeast region - use SE coefficients directly
GORI_WEIGHTS_COASTAL = {
    'wind': 5.21,   # SE Coast beta_W
    'rain': 0.52,   # SE Coast beta_R
    'surge': 0.62   # SE Coast beta_S
}

GORI_WEIGHTS_INLAND = {
    'wind': 4.38,   # SE Inland beta_W
    'rain': 0.45,   # SE Inland beta_R
    'surge': 0.0    # No surge term for inland counties
}


def load_coastal_classification():
    """Load coastal/inland classification for Florida counties."""
    print("\nLoading coastal/inland classification...")
    coastal_df = pd.read_csv(COASTAL_CLASSIFICATION)
    coastal_map = dict(zip(coastal_df['county_name'], coastal_df['is_coastal']))

    n_coastal = sum(coastal_df['is_coastal'])
    n_inland = len(coastal_df) - n_coastal
    print(f"  Coastal counties: {n_coastal}")
    print(f"  Inland counties: {n_inland}")

    return coastal_map


def load_gori_data(filepath):
    """Load Gori TC data and filter to Florida counties."""
    print(f"\nLoading data from: {filepath}")
    df = pd.read_csv(filepath)

    # Rename columns to standard format
    if 'W_ms' in df.columns:
        df = df.rename(columns={'W_ms': 'Wind', 'R_mm': 'Rain', 'S_m': 'Surge',
                                'event_id': 'event_name'})

        # Handle county names
        if 'county_name' in df.columns:
            df = df.rename(columns={'county_name': 'admin2_name'})

    print(f"  Loaded {len(df):,} county-event observations")
    print(f"  Counties: {df['admin2_name'].nunique()}")
    print(f"  Events: {df['event_name'].nunique()}")

    return df


def calculate_county_climatology(df):
    """Calculate 95th percentile hazard values for each county (climatology)."""
    print(f"\nCalculating county-specific {PERCENTILE}th percentile climatology...")

    climatology = df.groupby('admin2_name').agg({
        'Wind': lambda x: np.nanpercentile(x, PERCENTILE),
        'Rain': lambda x: np.nanpercentile(x, PERCENTILE),
        'Surge': lambda x: np.nanpercentile(x, PERCENTILE)
    }).rename(columns={
        'Wind': 'wind_p95',
        'Rain': 'rain_p95',
        'Surge': 'surge_p95'
    })

    print(f"  Wind p95 range: {climatology['wind_p95'].min():.1f} - {climatology['wind_p95'].max():.1f} m/s")
    print(f"  Rain p95 range: {climatology['rain_p95'].min():.1f} - {climatology['rain_p95'].max():.1f} mm")
    print(f"  Surge p95 range: {climatology['surge_p95'].min():.2f} - {climatology['surge_p95'].max():.2f} m")

    return climatology


def normalize_hazards(df, climatology):
    """Normalize hazards by county climatology (0-1+ scale)."""
    print("\nNormalizing hazards by county climatology...")

    df = df.merge(climatology, left_on='admin2_name', right_index=True, how='left')

    # Normalize (values >1 indicate above-climatology events)
    df['wind_norm'] = df['Wind'] / df['wind_p95']
    df['rain_norm'] = df['Rain'] / df['rain_p95']
    df['surge_norm'] = df['Surge'] / df['surge_p95']

    # Handle division by zero (counties with no hazard in climatology)
    df['wind_norm'] = df['wind_norm'].fillna(0)
    df['rain_norm'] = df['rain_norm'].fillna(0)
    df['surge_norm'] = df['surge_norm'].fillna(0)

    # Cap at reasonable values (some events may be extreme outliers)
    df['wind_norm'] = df['wind_norm'].clip(upper=5.0)
    df['rain_norm'] = df['rain_norm'].clip(upper=5.0)
    df['surge_norm'] = df['surge_norm'].clip(upper=5.0)

    print(f"  Wind normalized range: {df['wind_norm'].min():.2f} - {df['wind_norm'].max():.2f}")
    print(f"  Rain normalized range: {df['rain_norm'].min():.2f} - {df['rain_norm'].max():.2f}")
    print(f"  Surge normalized range: {df['surge_norm'].min():.2f} - {df['surge_norm'].max():.2f}")

    return df


def calculate_compound_hazard(df, coastal_map):
    """Calculate compound hazard metric with coastal/inland distinction."""
    print(f"\nCalculating compound hazard metric (Gori-weighted, coastal-aware)...")

    # Map county to coastal status
    df['is_coastal'] = df['admin2_name'].map(coastal_map)

    # Apply appropriate weights based on coastal status
    def apply_weights(row):
        if row['is_coastal']:
            return (
                GORI_WEIGHTS_COASTAL['wind'] * row['wind_norm'] +
                GORI_WEIGHTS_COASTAL['rain'] * row['rain_norm'] +
                GORI_WEIGHTS_COASTAL['surge'] * row['surge_norm']
            )
        else:
            # Inland counties: no surge term
            return (
                GORI_WEIGHTS_INLAND['wind'] * row['wind_norm'] +
                GORI_WEIGHTS_INLAND['rain'] * row['rain_norm']
            )

    df['compound_hazard'] = df.apply(apply_weights, axis=1)

    n_coastal = df['is_coastal'].sum()
    n_inland = (~df['is_coastal']).sum()
    print(f"  Using Gori SE region coefficients:")
    print(f"    Coastal ({n_coastal:,} obs): wind={GORI_WEIGHTS_COASTAL['wind']:.2f}, "
          f"rain={GORI_WEIGHTS_COASTAL['rain']:.2f}, surge={GORI_WEIGHTS_COASTAL['surge']:.2f}")
    print(f"    Inland  ({n_inland:,} obs): wind={GORI_WEIGHTS_INLAND['wind']:.2f}, "
          f"rain={GORI_WEIGHTS_INLAND['rain']:.2f}, surge={GORI_WEIGHTS_INLAND['surge']:.2f}")

    print(f"  Compound hazard range: {df['compound_hazard'].min():.2f} - {df['compound_hazard'].max():.2f}")
    print(f"  Mean: {df['compound_hazard'].mean():.2f}, Median: {df['compound_hazard'].median():.2f}")

    return df


def identify_extreme_events(df):
    """Identify top 5% most severe compound hazard events per county."""
    print(f"\nIdentifying top {100-PERCENTILE}% extreme events by county...")

    # Calculate county-specific thresholds
    county_thresholds = df.groupby('admin2_name')['compound_hazard'].quantile(PERCENTILE/100)

    # Merge thresholds and flag extreme events
    df = df.merge(county_thresholds.rename('threshold'), left_on='admin2_name', right_index=True)
    df['is_extreme'] = df['compound_hazard'] >= df['threshold']

    n_extreme = df['is_extreme'].sum()
    pct_extreme = 100 * n_extreme / len(df)

    print(f"  Extreme events: {n_extreme:,} / {len(df):,} ({pct_extreme:.1f}%)")
    print(f"  Counties with extreme events: {df[df['is_extreme']]['admin2_name'].nunique()}")

    return df


def calculate_empirical_attribution(df, coastal_map):
    """Calculate empirical hazard attribution for extreme events by county."""
    print(f"\nCalculating empirical attribution for extreme events (P{PERCENTILE})...")

    # Add coastal classification to dataframe
    df = df.copy()
    df['is_coastal'] = df['admin2_name'].map(coastal_map)

    # Filter to extreme events only
    extreme = df[df['is_extreme']].copy()

    # Weighted attribution (by Gori coefficients, coastal/inland specific)
    def calc_weighted_attrs(x):
        is_coastal = x['is_coastal'].iloc[0]

        if is_coastal:
            wind_w = (GORI_WEIGHTS_COASTAL['wind'] * x['wind_norm']).mean()
            rain_w = (GORI_WEIGHTS_COASTAL['rain'] * x['rain_norm']).mean()
            surge_w = (GORI_WEIGHTS_COASTAL['surge'] * x['surge_norm']).mean()
        else:
            wind_w = (GORI_WEIGHTS_INLAND['wind'] * x['wind_norm']).mean()
            rain_w = (GORI_WEIGHTS_INLAND['rain'] * x['rain_norm']).mean()
            surge_w = 0.0  # No surge term for inland

        return pd.Series({
            'wind_weighted': wind_w,
            'rain_weighted': rain_w,
            'surge_weighted': surge_w,
            'is_coastal': is_coastal,
            'n_extreme_events': len(x),
            'mean_compound_hazard': x['compound_hazard'].mean()
        })

    results = extreme.groupby('admin2_name').apply(calc_weighted_attrs)

    # Calculate shares
    total = results['wind_weighted'] + results['rain_weighted'] + results['surge_weighted']
    results['wind_share'] = 100 * results['wind_weighted'] / total
    results['rain_share'] = 100 * results['rain_weighted'] / total
    results['surge_share'] = 100 * results['surge_weighted'] / total

    # Summary statistics
    print(f"\n  Florida-wide average (P{PERCENTILE} extreme events):")
    print(f"    Wind share:  {results['wind_share'].mean():.1f}% (range: {results['wind_share'].min():.1f}% - {results['wind_share'].max():.1f}%)")
    print(f"    Rain share:  {results['rain_share'].mean():.1f}% (range: {results['rain_share'].min():.1f}% - {results['rain_share'].max():.1f}%)")
    print(f"    Surge share: {results['surge_share'].mean():.1f}% (range: {results['surge_share'].min():.1f}% - {results['surge_share'].max():.1f}%)")

    return results


def compare_to_baseline(results, baseline_wind_share=70.0):
    """Compare empirical attribution to baseline and calculate scaling factors."""
    print(f"\nComparing to baseline ({baseline_wind_share:.1f}% wind)...")

    baseline_water_share = 100 - baseline_wind_share

    # Calculate deviations from baseline
    results['wind_deviation_pp'] = results['wind_share'] - baseline_wind_share
    results['water_deviation_pp'] = (results['rain_share'] + results['surge_share']) - baseline_water_share

    # Calculate county-specific scaling factors
    results['wind_scaling'] = 1.0 + (results['wind_deviation_pp'] / baseline_wind_share)
    results['water_scaling'] = 1.0 + (results['water_deviation_pp'] / baseline_water_share)

    print(f"\n  Wind deviation: {results['wind_deviation_pp'].mean():.1f}pp "
          f"(range: {results['wind_deviation_pp'].min():.1f}pp - {results['wind_deviation_pp'].max():.1f}pp)")
    print(f"  Water deviation: {results['water_deviation_pp'].mean():.1f}pp "
          f"(range: {results['water_deviation_pp'].min():.1f}pp - {results['water_deviation_pp'].max():.1f}pp)")

    return results


def main():
    """Main execution function."""
    print("="*80)
    print("EMPIRICAL HAZARD ATTRIBUTION ANALYSIS (P95)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Percentile threshold: {PERCENTILE}th (top {100-PERCENTILE}% events)")
    print(f"  Gori weights:")
    print(f"    Coastal - Wind: {GORI_WEIGHTS_COASTAL['wind']:.2f}, Rain: {GORI_WEIGHTS_COASTAL['rain']:.2f}, Surge: {GORI_WEIGHTS_COASTAL['surge']:.2f}")
    print(f"    Inland  - Wind: {GORI_WEIGHTS_INLAND['wind']:.2f}, Rain: {GORI_WEIGHTS_INLAND['rain']:.2f}, Surge: {GORI_WEIGHTS_INLAND['surge']:.2f}")

    # Load coastal classification
    coastal_map = load_coastal_classification()

    # Process present climate
    print("\n" + "="*80)
    print("PRESENT CLIMATE (1980-2019)")
    print("="*80)

    df_present = load_gori_data(GORI_PRESENT)
    climatology_present = calculate_county_climatology(df_present)
    df_present = normalize_hazards(df_present, climatology_present)
    df_present = calculate_compound_hazard(df_present, coastal_map=coastal_map)
    df_present = identify_extreme_events(df_present)
    results_present = calculate_empirical_attribution(df_present, coastal_map=coastal_map)
    results_present = compare_to_baseline(results_present, baseline_wind_share=70.0)

    # Save results
    results_present.to_csv(OUTPUT_PRESENT)
    print(f"\nResults saved to: {OUTPUT_PRESENT}")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
