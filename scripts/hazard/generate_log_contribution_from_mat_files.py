"""
Generate Wind/Water Attribution from Raw Gori .mat Files

This script generates wind/water attribution CSV files using:
1. Raw Gori .mat files (not pre-processed CSVs)
2. Coastal/inland coefficient differentiation
3. Logarithmic contribution method (exact same as build_per_event_impacts.py)

METHODOLOGY:
The Gori model: ln(Damage) = β_W×ln(W) + β_R×ln(R+1) + β_S×S
Wind/water shares from log-space contributions:
  - c_wind = β_W × ln(W)
  - c_rain = β_R × ln(R+1)
  - c_surge = β_S × S
  - wind_share = c_wind / (c_wind + c_rain + c_surge)

GORI COEFFICIENTS (Southeast region, Gori et al. 2022, Table S1):
  Coastal: β_W=5.21, β_R=0.52, β_S=0.62
  Inland:  β_W=4.38, β_R=0.45, β_S=0.0 (no surge term)

DATA SOURCES:
  - fl_risk_model/data/hazard/gori_data/Wind/maxwindmat_ncep_reanal.mat (present)
  - fl_risk_model/data/hazard/gori_data/Wind/maxwindmat_*_ssp245cal.mat (future, 5 GCMs)
  - fl_risk_model/data/hazard/gori_data/Rain/ptot_rain_county_ncep_reanal.mat
  - fl_risk_model/data/hazard/gori_data/Surge/maxelev_coastcounty_ncep_reanal.mat

OUTPUT FILES:
1. florida_log_contribution_p95_present.csv - Present climate, P95 extremes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat

# Paths
DATA_DIR = Path("fl_risk_model/data")
GORI_DATA_DIR = DATA_DIR / "hazard" / "gori_data"
COASTAL_FILE = DATA_DIR / "florida_coastal_counties.csv"
COUNTY_REGION_FILE = DATA_DIR / "county_region.csv"

# Output files
OUTPUT_P95_PRESENT = DATA_DIR / "florida_log_contribution_p95_present.csv"

# Gori coefficients (Southeast region)
GORI_COASTAL = {'wind': 5.21, 'rain': 0.52, 'surge': 0.62}
GORI_INLAND = {'wind': 4.38, 'rain': 0.45, 'surge': 0.0}

PERCENTILE = 95
FLORIDA_STCODE = 12


def load_florida_counties():
    """Load Florida county indices and names."""
    print("Loading Florida county information...")
    
    # Load county region mapping
    county_region = pd.read_csv(COUNTY_REGION_FILE)
    fl_counties = county_region[county_region['stcode'] == FLORIDA_STCODE].copy()
    
    # county_index is 0-based index into the .mat arrays
    fl_indices = fl_counties['county_index'].astype(int).values
    county_fps = fl_counties['fips'].astype(int).values  # Use full FIPS codes
    
    print(f"  Found {len(fl_indices)} Florida counties")
    
    # Create FIPS to index mapping
    fips_to_idx = dict(zip(county_fps, range(len(county_fps))))
    idx_to_fips = dict(zip(range(len(county_fps)), county_fps))
    
    return fl_indices, county_fps, fips_to_idx, idx_to_fips


def load_coastal_classification(county_fps):
    """Load and align coastal/inland classification."""
    print("Loading coastal/inland classification...")
    
    coastal_df = pd.read_csv(COASTAL_FILE)
    
    # Create mapping from FIPS to coastal status
    coastal_map = dict(zip(coastal_df['fips'], coastal_df['is_coastal']))
    
    # Create boolean array aligned with county_fps
    # Need to match full FIPS codes (e.g., 12001 vs 1)
    is_coastal = np.array([coastal_map.get(int(fips), False) for fips in county_fps])
    
    n_coastal = is_coastal.sum()
    n_inland = len(is_coastal) - n_coastal
    print(f"  Coastal counties: {n_coastal}")
    print(f"  Inland counties: {n_inland}")
    
    return is_coastal


def load_mat_hazard(file_path, fl_indices, var_name, transpose=False):
    """Load hazard matrix and extract Florida counties."""
    mat_data = loadmat(file_path)
    hazard_matrix = mat_data[var_name]
    
    # Transpose if needed (wind matrices are stored as events x counties)
    if transpose:
        hazard_matrix = hazard_matrix.T  # Now (counties, events)
    
    # Extract Florida counties (rows)
    fl_hazard = hazard_matrix[fl_indices, :]
    
    return fl_hazard


def calculate_log_contributions(wind, rain, surge, is_coastal):
    """
    Calculate logarithmic contributions using Gori coefficients.
    
    Args:
        wind: Wind hazard array (counties x events), m/s
        rain: Rain hazard array (counties x events), mm
        surge: Surge hazard array (counties x events), m
        is_coastal: Boolean array (counties,) indicating coastal status
    
    Returns:
        wind_share: Wind contribution share (counties x events), 0-1
    """
    n_counties, n_events = wind.shape
    
    # Initialize contribution arrays
    c_wind = np.zeros((n_counties, n_events))
    c_rain = np.zeros((n_counties, n_events))
    c_surge = np.zeros((n_counties, n_events))
    
    # Apply coastal coefficients
    coastal_mask = is_coastal.reshape(-1, 1)  # Shape: (counties, 1) for broadcasting
    
    # Coastal counties
    W_coastal = np.where(coastal_mask, wind, 0)
    R_coastal = np.where(coastal_mask, rain, 0)
    S_coastal = np.where(coastal_mask, surge, 0)
    
    # Avoid log(0) - use small epsilon for zero winds
    W_coastal_safe = np.maximum(W_coastal, 1e-9)
    
    c_wind += np.where(coastal_mask, GORI_COASTAL['wind'] * np.log(W_coastal_safe), 0)
    c_rain += np.where(coastal_mask, GORI_COASTAL['rain'] * np.log(R_coastal + 1.0), 0)
    c_surge += np.where(coastal_mask, GORI_COASTAL['surge'] * S_coastal, 0)
    
    # Inland counties
    inland_mask = ~is_coastal.reshape(-1, 1)
    
    W_inland = np.where(inland_mask, wind, 0)
    R_inland = np.where(inland_mask, rain, 0)
    
    W_inland_safe = np.maximum(W_inland, 1e-9)
    
    c_wind += np.where(inland_mask, GORI_INLAND['wind'] * np.log(W_inland_safe), 0)
    c_rain += np.where(inland_mask, GORI_INLAND['rain'] * np.log(R_inland + 1.0), 0)
    # No surge term for inland
    
    # Calculate shares
    c_total = c_wind + c_rain + c_surge
    
    # Avoid division by zero - where total=0, set shares to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        wind_share = np.where(c_total > 0, c_wind / c_total, 0.0)
    
    wind_share = np.clip(wind_share, 0.0, 1.0)
    
    return wind_share


def identify_extreme_events(wind_share, compound_hazard, percentile=95):
    """
    Identify P95 extreme events for each county.
    
    Args:
        wind_share: Wind share array (counties x events)
        compound_hazard: Compound hazard metric (counties x events)
        percentile: Percentile threshold (default 95)
    
    Returns:
        extreme_mask: Boolean array (counties x events)
    """
    n_counties, n_events = wind_share.shape
    extreme_mask = np.zeros((n_counties, n_events), dtype=bool)
    
    # For each county, find P95 threshold
    for i in range(n_counties):
        threshold = np.percentile(compound_hazard[i, :], percentile)
        extreme_mask[i, :] = compound_hazard[i, :] >= threshold
    
    return extreme_mask


def generate_county_attribution(wind_share, extreme_mask, county_fps, idx_to_fips):
    """
    Generate county-level average wind shares.
    
    Args:
        wind_share: Wind share array (counties x events)
        extreme_mask: Boolean mask for extreme events (counties x events) or None for full dist
        county_fps: Array of county FIPS codes
        idx_to_fips: Dict mapping index to FIPS
    
    Returns:
        DataFrame with columns: county_name, fips, wind_share
    """
    n_counties = wind_share.shape[0]
    
    results = []
    for i in range(n_counties):
        fips = idx_to_fips[i]
        
        if extreme_mask is not None:
            # Use only extreme events
            shares = wind_share[i, extreme_mask[i, :]]
        else:
            # Use all events
            shares = wind_share[i, :]
        
        if len(shares) > 0:
            avg_share = shares.mean()
        else:
            avg_share = 0.0
        
        results.append({
            'fips': fips,
            'wind_share': avg_share
        })
    
    df = pd.DataFrame(results)
    
    # Add county names - get from county_region
    county_region = pd.read_csv(COUNTY_REGION_FILE)
    fl_region = county_region[county_region['stcode'] == FLORIDA_STCODE].copy()
    
    # Try to get county names if column exists
    if 'county_name' in fl_region.columns:
        fips_to_name = dict(zip(fl_region['fips'], fl_region['county_name']))
    else:
        # Use FIPS as county name if no name column
        fips_to_name = {fips: f"County_{fips}" for fips in df['fips']}
    
    df['county_name'] = df['fips'].map(fips_to_name)
    
    # Reorder columns
    df = df[['county_name', 'fips', 'wind_share']]
    
    return df


def process_present_climate():
    """Process present climate (NCEP reanalysis)."""
    print("\n" + "="*70)
    print("PROCESSING PRESENT CLIMATE")
    print("="*70)
    
    # Load Florida counties
    fl_indices, county_fps, fips_to_idx, idx_to_fips = load_florida_counties()
    is_coastal = load_coastal_classification(county_fps)
    
    # Load hazard .mat files
    print("\nLoading hazard data from .mat files...")
    wind_file = GORI_DATA_DIR / "Wind" / "maxwindmat_ncep_reanal.mat"
    rain_file = GORI_DATA_DIR / "Rain" / "ptot_rain_county_ncep_reanal.mat"
    surge_file = GORI_DATA_DIR / "Surge" / "maxelev_coastcounty_ncep_reanal.mat"
    
    wind = load_mat_hazard(wind_file, fl_indices, 'maxwindmat', transpose=True)
    rain = load_mat_hazard(rain_file, fl_indices, 'ptot_mat')
    
    # Surge file uses different variable name
    surge_mat = loadmat(surge_file)
    surge_var = 'scounty_mhhw' if 'scounty_mhhw' in surge_mat else 'scounty'
    surge = load_mat_hazard(surge_file, fl_indices, surge_var)
    
    n_counties, n_events = wind.shape
    print(f"  Loaded {n_counties} FL counties × {n_events} events")
    
    # Calculate logarithmic contributions
    print("\nCalculating logarithmic contributions...")
    wind_share = calculate_log_contributions(wind, rain, surge, is_coastal)
    
    # Calculate compound hazard for P95 selection
    # Use same weighted approach as contributions
    compound = np.zeros_like(wind)
    for i in range(n_counties):
        if is_coastal[i]:
            compound[i, :] = (
                GORI_COASTAL['wind'] * np.log(np.maximum(wind[i, :], 1e-9)) +
                GORI_COASTAL['rain'] * np.log(rain[i, :] + 1.0) +
                GORI_COASTAL['surge'] * surge[i, :]
            )
        else:
            compound[i, :] = (
                GORI_INLAND['wind'] * np.log(np.maximum(wind[i, :], 1e-9)) +
                GORI_INLAND['rain'] * np.log(rain[i, :] + 1.0)
            )
    
    # Generate P95 extreme events file
    print(f"\nGenerating P95 extreme events attribution...")
    extreme_mask = identify_extreme_events(wind_share, compound, percentile=PERCENTILE)
    n_extreme_total = extreme_mask.sum()
    print(f"  Total extreme events: {n_extreme_total:,} / {wind.size:,}")
    
    df_p95 = generate_county_attribution(wind_share, extreme_mask, county_fps, idx_to_fips)
    avg_wind_p95 = df_p95['wind_share'].mean()
    print(f"  Average wind share (P95): {100*avg_wind_p95:.2f}%")
    
    df_p95.to_csv(OUTPUT_P95_PRESENT, index=False)
    print(f"  Saved: {OUTPUT_P95_PRESENT}")
    
    return avg_wind_p95


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("WIND/WATER ATTRIBUTION - LOGARITHMIC CONTRIBUTION METHOD")
    print("="*70)
    print("\nMETHODOLOGY:")
    print("  - Load raw Gori .mat files")
    print("  - Apply coastal/inland Gori coefficients")
    print(f"    * Coastal: β_W={GORI_COASTAL['wind']}, β_R={GORI_COASTAL['rain']}, β_S={GORI_COASTAL['surge']}")
    print(f"    * Inland:  β_W={GORI_INLAND['wind']}, β_R={GORI_INLAND['rain']}, β_S={GORI_INLAND['surge']}")
    print("  - Calculate: c_wind = β_W×ln(W), c_rain = β_R×ln(R+1), c_surge = β_S×S")
    print("  - Share: wind_share = c_wind / (c_wind + c_rain + c_surge)")
    print(f"  - P{PERCENTILE} extreme events per county")
    
    # Process present climate
    avg_present_p95 = process_present_climate()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nPresent Climate (NCEP):")
    print(f"  P95 extremes:      {100*avg_present_p95:.2f}% wind")
    
    print("\n" + "="*70)
    print("FILES GENERATED:")
    print(f"  1. {OUTPUT_P95_PRESENT}")
    print("="*70)


if __name__ == "__main__":
    main()
