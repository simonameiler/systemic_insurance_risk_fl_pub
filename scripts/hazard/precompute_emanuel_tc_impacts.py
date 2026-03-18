"""
Precompute County-Level Impacts for Kerry Emanuel TC Event Sets

This script:
1. Loads a specified Emanuel TC hazard set (from Sherlock cluster)
2. Loads Florida exposure (LitPop at 120 arcsec)
3. Computes county-aggregated damage for each event using RMSF impact functions
4. Saves non-zero events as individual CSV files
5. Creates metadata file with event characteristics and annual frequencies

This enables fast Monte Carlo runs using year-set generation based on observed
annual frequency distributions from the Emanuel simulations.

Usage:
    python scripts/hazard/precompute_emanuel_tc_impacts.py <event_set_name>

Example:
    python scripts/hazard/precompute_emanuel_tc_impacts.py FL_era5_reanalcal

Outputs (in OUTPUT_BASE/<event_set>/):
    - <event_id>.csv (ALL events with county-level damage)
    - event_metadata.csv (event characteristics)
    - all_events.csv (aggregated events summary)
    - annual_frequencies.csv (frequency by year)

Author: Simona Meiler
Date: December 2025
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.io as sio
from tqdm import tqdm

# CLIMADA imports
from climada.hazard import Hazard
from climada.entity import Exposures, LitPop
from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone
from climada.engine import ImpactCalc
from climada.util.constants import SYSTEM_DIR

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fl_risk_model import config as cfg

# ============================================================================
# CONFIGURATION
# ============================================================================

# Detect if running on cluster (Sherlock) vs local
IS_CLUSTER = 'SLURM_JOB_ID' in os.environ or 'SHERLOCK' in os.environ.get('HOSTNAME', '')

if IS_CLUSTER:
    # Sherlock cluster paths (Stanford HPC)
    HAZARD_DIR = Path("/home/groups/bakerjw/smeiler/climada_data/data/hazard/Florida")
    TRACKS_DIR = Path("/home/groups/bakerjw/smeiler/climada_data/data/tracks/Kerry/Florida")
    OUTPUT_BASE = Path("/home/groups/bakerjw/smeiler/climada_data/data/impact")
    DATA_DIR = Path(cfg.DATA_DIR)
else:
    # Local paths - MODIFY THESE FOR YOUR SETUP
    # Point to directory containing Emanuel TC hazard NetCDF files from CLIMADA
    DATA_DIR = Path(cfg.DATA_DIR)
    HAZARD_DIR = Path.home() / "climada" / "data" / "hazard" / "Florida"
    TRACKS_DIR = Path.home() / "climada" / "data" / "tracks" / "Kerry" / "Florida"
    OUTPUT_BASE = DATA_DIR / "hazard" / "emanuel"

# Thresholds
DAMAGE_THRESHOLD_USD = 0  # Keep ALL events (including zero-damage) to preserve true distribution

# Exposure parameters
EXPOSURE_ARCSEC = 120  # 2 arcminutes (~3.7 km at Florida latitude)
REFERENCE_YEAR = 2024

# Impact function
IMPACT_FUNC_CALIBRATION = "RMSF"  # Same as used for historical events

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_track_filename(event_set_name):
    """
    Convert event set name to original track filename.
    
    Example:
        FL_era5_reanalcal -> Simona_FLA_AL_era5_reanalcal.mat
        FL_canesm_20thcal -> Simona_FLA_AL_canesm_20thcal.mat
    """
    # Remove FL_ prefix
    if event_set_name.startswith("FL_"):
        base_name = event_set_name[3:]
    else:
        base_name = event_set_name
    
    # Add original prefix and extension
    track_filename = f"Simona_FLA_AL_{base_name}.mat"
    return track_filename


def load_annual_frequencies(track_file):
    """
    Load annual frequency data (freq and freqyear) from track .mat file.
    
    Args:
        track_file: Path to track .mat file
    
    Returns:
        dict with 'freq' (scalar), 'freqyear' (array), 'n_years' (int)
    """
    print(f"\nLoading annual frequencies from track file...")
    print(f"  {track_file.name}")
    
    if not track_file.exists():
        raise FileNotFoundError(f"Track file not found: {track_file}")
    
    # Load .mat file
    mat_data = sio.loadmat(str(track_file))
    
    # Extract frequency variables
    freq = float(mat_data['freq'].flatten()[0])
    freqyear = mat_data['freqyear'].flatten()
    
    print(f"  Mean annual frequency: {freq:.3f} events/year")
    print(f"  Years in dataset: {len(freqyear)}")
    print(f"  Frequency range: [{freqyear.min():.3f}, {freqyear.max():.3f}]")
    print(f"  Frequency std dev: {freqyear.std():.3f}")
    
    return {
        'freq': freq,
        'freqyear': freqyear,
        'n_years': len(freqyear)
    }


def load_florida_exposure(cache_path=None):
    """
    Load or create Florida exposure data (LitPop).
    
    Args:
        cache_path: Path to cached exposure HDF5 (None = create fresh)
    
    Returns:
        Exposures object with Florida data
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached exposure from {cache_path}...")
        exp = Exposures.from_hdf5(cache_path)
        print(f"  Loaded {len(exp.gdf):,} exposure points")
        return exp
    
    print("Creating Florida exposure from LitPop...")
    print(f"  Resolution: {EXPOSURE_ARCSEC} arcsec")
    print(f"  Reference year: {REFERENCE_YEAR}")
    
    # Load USA exposure
    exp_usa = LitPop.from_countries(
        "USA",
        fin_mode="pc",
        res_arcsec=EXPOSURE_ARCSEC,
        exponents=(1, 1),
        admin1_calc=True,
        reference_year=REFERENCE_YEAR
    )
    
    # Filter to Florida
    exp = Exposures()
    exp.set_gdf(exp_usa.gdf[exp_usa.gdf.admin1 == "Florida"])
    
    # Set impact function ID (region 2 = USA for RMSF calibration)
    exp.gdf['impf_TC'] = 2
    
    print(f"  Florida exposure: {len(exp.gdf):,} points")
    print(f"  Total value: ${exp.gdf['value'].sum()/1e9:.1f}B")
    
    # Cache for future use
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        exp.write_hdf5(cache_path)
        print(f"  Cached to {cache_path}")
    
    return exp


def load_florida_counties():
    """Load Florida county boundaries for spatial aggregation."""
    print("Loading Florida county boundaries...")
    county_fp = DATA_DIR / "US_counties"
    
    if not county_fp.exists():
        raise FileNotFoundError(
            f"County shapefile not found: {county_fp}\n"
            "Download from: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html"
        )
    
    counties = gpd.read_file(county_fp)
    fl_counties = counties[counties["STATEFP"] == "12"].copy()
    
    # Clean county names (remove " County" suffix)
    fl_counties["county_name"] = (
        fl_counties["NAME"].str.replace(r"\s+County$", "", regex=True).str.strip()
    )
    
    # Fix geometries
    fl_counties['geometry'] = fl_counties['geometry'].buffer(0)
    
    print(f"  Loaded {len(fl_counties)} Florida counties")
    
    return fl_counties


def assign_exposure_to_counties(exp, fl_counties):
    """
    Spatially join exposure points to counties.
    
    Args:
        exp: CLIMADA Exposures object
        fl_counties: GeoDataFrame with county boundaries
    
    Returns:
        GeoDataFrame with county assignments
    """
    print("Assigning exposure to counties...")
    
    # Ensure consistent CRS
    if exp.gdf.crs is None:
        exp.gdf = exp.set_crs(crs='EPSG:4326')
    else:
        exp.set_crs(crs='EPSG:4326')
    
    if fl_counties.crs is None:
        fl_counties = fl_counties.set_crs(epsg=4326)
    else:
        fl_counties = fl_counties.to_crs('EPSG:4326')
    
    # Spatial join
    exp_with_county = gpd.sjoin(
        exp.gdf,
        fl_counties[['COUNTYFP', 'county_name', 'geometry']],
        how='left',
        predicate='within'
    )
    
    # Clean up
    exp_with_county = exp_with_county.drop(columns=['index_right'], errors='ignore')
    exp_with_county = exp_with_county.rename(columns={'COUNTYFP': 'countyfp'})
    
    # Check for unassigned points
    unassigned = exp_with_county['countyfp'].isna().sum()
    if unassigned > 0:
        print(f"  Warning: {unassigned} exposure points not assigned to counties")
    
    print(f"  Successfully assigned {(~exp_with_county['countyfp'].isna()).sum():,} points")
    
    return exp_with_county


def compute_event_impacts(hazard, exposure_gdf, impact_func_set):
    """
    Compute county-aggregated impacts for all events in hazard set.
    
    Args:
        hazard: CLIMADA Hazard object
        exposure_gdf: GeoDataFrame with county assignments
        impact_func_set: CLIMADA impact function set
    
    Returns:
        DataFrame with columns: event_id, countyfp, county_name, value (damage USD)
    """
    print(f"\nComputing impacts for {hazard.size} events...")
    
    # Create exposures object from GeoDataFrame
    exp = Exposures(exposure_gdf)
    
    # Assign centroids
    exp.assign_centroids(hazard)
    
    # Compute impacts (with sparse impact matrix)
    imp_calc = ImpactCalc(exp, impact_func_set, hazard)
    imp = imp_calc.impact(save_mat=True)
    
    print(f"  Impact computation complete")
    print(f"  Events with damage >$0: {(imp.at_event > 0).sum()} / {hazard.size}")
    print(f"  Total damage: ${imp.at_event.sum()/1e9:.2f}B")
    
    # Extract per-event impacts
    print(f"  Extracting county-level impacts...")
    
    all_events = []
    
    for i in tqdm(range(imp.at_event.size), desc="Processing events"):
        event_total = imp.at_event[i]
        
        # KEEP zero-damage events to preserve true frequency-intensity distribution
        # Don't skip: if event_total == 0: continue
        
        if event_total > 0:
            # Extract event-specific impacts
            event_impacts = imp.imp_mat[i, :].toarray().ravel()
            
            # Create temporary GeoDataFrame
            temp_gdf = exposure_gdf.copy()
            temp_gdf['event_impact'] = event_impacts
            
            # Aggregate by county
            county_damage = (
                temp_gdf.groupby(['countyfp', 'county_name'], dropna=False)['event_impact']
                .sum()
                .reset_index()
                .rename(columns={'event_impact': 'value'})
            )
            
            # Add event metadata
            county_damage['event_id'] = imp.event_name[i]
            county_damage['event_index'] = i
        else:
            # Zero-damage event: create single-row record with zero total damage
            # This ensures the event appears in metadata even with no county-level damage
            county_damage = pd.DataFrame({
                'event_id': [imp.event_name[i]],
                'event_index': [i],
                'countyfp': [None],
                'county_name': [None],
                'value': [0.0]
            })
        
        all_events.append(county_damage)
    
    # Combine all events
    if not all_events:
        raise ValueError("No events found in hazard set!")
    
    impacts_df = pd.concat(all_events, ignore_index=True)
    
    print(f"\n  Total events processed: {len(all_events):,} (including zero-damage)")
    print(f"  Total event-county observations: {len(impacts_df):,}")
    
    return impacts_df


def save_event_csvs(impacts_df, output_dir, threshold_usd=1e6):
    """
    Save individual CSV files for each viable event.
    
    Args:
        impacts_df: DataFrame with event_id, countyfp, county_name, value
        output_dir: Directory to save CSV files
        threshold_usd: Minimum total damage to save event
    
    Returns:
        DataFrame with event metadata
    """
    print(f"\nSaving event CSV files...")
    print(f"  Output directory: {output_dir}")
    print(f"  Damage threshold: ${threshold_usd:,.0f}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by event
    event_groups = impacts_df.groupby('event_id')
    
    viable_events = []
    
    for event_id, group in tqdm(event_groups, desc="Writing CSVs"):
        # Get event index and total damage
        event_idx = group['event_index'].iloc[0]
        
        # Filter out placeholder rows (countyfp=None for zero-damage events)
        damage_rows = group[group['countyfp'].notna()].copy()
        total_damage = damage_rows['value'].sum() if len(damage_rows) > 0 else 0.0
        
        if total_damage < threshold_usd:
            continue
        
        # Format county data
        if len(damage_rows) > 0:
            # Save actual damage data
            event_csv = damage_rows[['countyfp', 'county_name', 'value']].copy()
            # Ensure countyfp is 3-digit string
            event_csv['countyfp'] = event_csv['countyfp'].astype(str).str.zfill(3)
        else:
            # Zero-damage event: create empty CSV
            event_csv = pd.DataFrame({
                'countyfp': [],
                'county_name': [],
                'value': []
            })
        
        # Save to CSV
        csv_path = output_dir / f"{event_id}.csv"
        event_csv.to_csv(csv_path, index=False)
        
        viable_events.append({
            'event_id': event_id,
            'event_index': event_idx,
            'total_damage_usd': total_damage,
            'n_counties_affected': len(event_csv),
            'csv_path': str(csv_path)
        })
    
    metadata_df = pd.DataFrame(viable_events)
    
    print(f"\n  Saved {len(metadata_df)} events (including zero-damage events)")
    if threshold_usd > 0:
        print(f"  Excluded {len(event_groups) - len(metadata_df)} events below threshold")
    
    if len(metadata_df) > 0:
        n_nonzero = (metadata_df['total_damage_usd'] > 0).sum()
        print(f"  Events with damage >$0: {n_nonzero} / {len(metadata_df)}")
        if n_nonzero > 0:
            print(f"  Damage range: ${metadata_df[metadata_df['total_damage_usd'] > 0]['total_damage_usd'].min()/1e9:.3f}B - ${metadata_df['total_damage_usd'].max()/1e9:.1f}B")
    
    return metadata_df


def extract_event_characteristics(hazard, metadata_df):
    """
    Extract hazard characteristics for viable events.
    
    Args:
        hazard: CLIMADA Hazard object
        metadata_df: DataFrame with event_id and event_index columns
    
    Returns:
        DataFrame with event_id, max_wind_ms
    """
    print("\nExtracting event characteristics...")
    
    characteristics = []
    
    for _, row in metadata_df.iterrows():
        event_id = row['event_id']
        event_idx = int(row['event_index'])
        
        # Extract max wind speed for this event using the stored index
        max_wind = hazard.intensity[event_idx, :].max()
        
        characteristics.append({
            'event_id': event_id,
            'max_wind_ms': max_wind
        })
    
    chars_df = pd.DataFrame(characteristics)
    
    print(f"  Extracted characteristics for {len(chars_df)} events")
    if len(chars_df) > 0:
        print(f"  Max wind range: {chars_df['max_wind_ms'].min():.1f} - {chars_df['max_wind_ms'].max():.1f} m/s")
    
    return chars_df


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_event_set(event_set_name):
    """
    Main processing function for a single Emanuel TC event set.
    
    Args:
        event_set_name: Name of event set (e.g., "FL_era5_reanalcal")
    """
    print("="*80)
    print(f"PRECOMPUTING EMANUEL TC IMPACTS: {event_set_name}")
    print("="*80)
    
    # Set up paths
    hazard_file = HAZARD_DIR / f"{event_set_name}.hdf5"
    track_filename = get_track_filename(event_set_name)
    track_file = TRACKS_DIR / track_filename
    output_dir = OUTPUT_BASE / event_set_name
    
    print(f"\nPaths:")
    print(f"  Hazard: {hazard_file}")
    print(f"  Tracks: {track_file}")
    print(f"  Output: {output_dir}")
    
    # Check if hazard file exists
    if not hazard_file.exists():
        raise FileNotFoundError(f"Hazard file not found: {hazard_file}")
    
    # Load annual frequencies from track file
    freq_data = load_annual_frequencies(track_file)
    
    # Load exposure
    cache_path = DATA_DIR / "FL_exposure_120as.hdf5"
    exposure = load_florida_exposure(cache_path)
    
    # Load counties
    fl_counties = load_florida_counties()
    
    # Assign exposure to counties
    exp_with_county = assign_exposure_to_counties(exposure, fl_counties)
    
    # Load hazard
    print(f"\nLoading hazard set...")
    hazard = Hazard.from_hdf5(hazard_file)
    print(f"  Events: {hazard.size}")
    print(f"  Centroids: {hazard.centroids.size}")
    
    # Load impact functions
    print(f"\nLoading impact functions ({IMPACT_FUNC_CALIBRATION})...")
    impact_func_set = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet(
        calibration_approach=IMPACT_FUNC_CALIBRATION
    )
    
    # Compute impacts
    impacts_df = compute_event_impacts(hazard, exp_with_county, impact_func_set)
    
    # Save event CSVs
    metadata_df = save_event_csvs(impacts_df, output_dir, DAMAGE_THRESHOLD_USD)
    
    # Extract event characteristics
    chars_df = extract_event_characteristics(hazard, metadata_df)
    
    # Merge metadata with characteristics
    full_metadata = metadata_df.merge(chars_df, on='event_id', how='left')
    
    # Save metadata
    metadata_path = output_dir / "event_metadata.csv"
    full_metadata.to_csv(metadata_path, index=False)
    print(f"\nSaved event metadata: {metadata_path}")
    
    # Save all events list (including zero-damage to preserve true distribution)
    all_events_path = output_dir / "all_events.csv"
    full_metadata[['event_id', 'total_damage_usd']].to_csv(all_events_path, index=False)
    print(f"Saved all events list: {all_events_path}")
    
    # Save annual frequencies
    freq_df = pd.DataFrame({
        'year': range(len(freq_data['freqyear'])),
        'frequency': freq_data['freqyear']
    })
    freq_df['mean_frequency'] = freq_data['freq']
    
    freq_path = output_dir / "annual_frequencies.csv"
    freq_df.to_csv(freq_path, index=False)
    print(f"Saved annual frequencies: {freq_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Event set: {event_set_name}")
    print(f"Total events in hazard: {hazard.size}")
    print(f"Viable events (damage >${DAMAGE_THRESHOLD_USD/1e6:.0f}M): {len(metadata_df)}")
    print(f"Mean annual frequency: {freq_data['freq']:.3f} events/year")
    print(f"Annual frequency range: [{freq_data['freqyear'].min():.3f}, {freq_data['freqyear'].max():.3f}]")
    print(f"Total damage (all viable events): ${full_metadata['total_damage_usd'].sum()/1e9:.2f}B")
    print(f"Output directory: {output_dir}")
    print("="*80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python precompute_emanuel_tc_impacts.py <event_set_name>")
        print("\nExample:")
        print("  python precompute_emanuel_tc_impacts.py FL_era5_reanalcal")
        print("\nAvailable event sets:")
        print("  FL_era5_reanalcal")
        print("  FL_canesm_20thcal")
        print("  FL_canesm_ssp245cal")
        print("  FL_canesm_ssp585cal")
        print("  ... (see submit script for full list)")
        sys.exit(1)
    
    event_set_name = sys.argv[1]
    
    try:
        process_event_set(event_set_name)
        print("\n✓ SUCCESS\n")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
