#!/usr/bin/env python3
"""
Sequential Event Impact Calculator - BUILDING-LEVEL VERSION

This version uses individual building exposure (florida_exposure.hdf5)
instead of LitPop aggregate cells. Each exposure point represents a
single building with its replacement cost.

Key differences from LitPop version:
- Higher spatial resolution (6.6M buildings vs 12K grid cells)
- Physical building replacement costs (not GDP proxy)
- More accurate sequential impact (destroyed building can't be damaged twice)

Author: GitHub Copilot
Date: November 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

# Add repository root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from climada.hazard import TropCyclone, TCTracks, Centroids
    from climada.entity.exposures import Exposures
    from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone
    from climada.engine import ImpactCalc
except ImportError as e:
    print(f"\n❌ CLIMADA not found: {e}")
    print("\nInstall CLIMADA in your conda environment:")
    print("  conda install -c conda-forge climada")
    sys.exit(1)

try:
    from fl_risk_model import config as cfg
except ImportError as e:
    print(f"\n❌ fl_risk_model not found: {e}")
    print(f"Repository root: {REPO_ROOT}")
    sys.exit(1)

# Import CAPRA impact functions
try:
    from fl_risk_model.loss_calc_utils.impfunc_utils import IMPF_SET_TC_CAPRA, DICT_PAGER_TCIMPF_CAPRA
except ImportError as e:
    print(f"\n❌ impfunc_utils not found: {e}")
    print("Make sure fl_risk_model.loss_calc_utils.impfunc_utils is accessible")
    sys.exit(1)


def _get_event_index(hazard, event_id: str) -> int:
    """Find event index in hazard.event_name array."""
    for i, name in enumerate(hazard.event_name):
        if name == event_id:
            return i
    return None


def compute_sequential_impact(
    exp: Exposures,
    event1_id: str,
    event2_id: str,
    hazard,
    imp_fun_set,
    degradation_mode: str = "direct",
    verbose: bool = True,
):
    """
    Compute realistic total damage from two sequential events on BUILDINGS.
    
    This version is optimized for building-level exposure where each point
    represents a physical structure with replacement cost.
    
    Parameters
    ----------
    exp : Exposures
        CLIMADA exposure with building-level data (one row per building)
    event1_id : str
        IBTrACS storm ID for first event
    event2_id : str
        IBTrACS storm ID for second event
    hazard : Hazard
        CLIMADA hazard object containing both events
    imp_fun_set : ImpfSetTropCyclone
        Calibrated impact function set
    degradation_mode : str
        - "direct": subtract absolute damage (recommended for buildings)
        - "ratio": apply damage ratio (for aggregate cells)
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    composite_impact : np.ndarray
        Point-level composite impact (length = n_buildings)
    diagnostics : dict
        Summary statistics
    """
    
    # 1) Find event indices
    event1_idx = _get_event_index(hazard, event1_id)
    event2_idx = _get_event_index(hazard, event2_id)
    
    if event1_idx is None:
        raise ValueError(f"Event '{event1_id}' not found in hazard.event_name")
    if event2_idx is None:
        raise ValueError(f"Event '{event2_id}' not found in hazard.event_name")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Computing Sequential Impact (BUILDING-LEVEL)")
        print(f"  Event 1: {event1_id} (index {event1_idx})")
        print(f"  Event 2: {event2_id} (index {event2_idx})")
        print(f"  Number of buildings: {len(exp.gdf):,}")
        print(f"  Degradation mode: {degradation_mode}")
        print('='*70)
    
    # 2) Run Event 1 on original exposure
    if verbose:
        print(f"\n[1/3] Computing Event 1 impact on original building stock...")
    
    imp1 = ImpactCalc(exp, imp_fun_set, hazard).impact(save_mat=True)
    event1_impact = imp1.imp_mat[event1_idx, :].toarray().ravel()
    event1_total = float(event1_impact.sum())
    
    if verbose:
        print(f"      Event 1 total loss: ${event1_total:,.0f}")
        # Count buildings with damage
        buildings_damaged = (event1_impact > 0).sum()
        buildings_destroyed = (event1_impact >= exp.gdf['value'].values * 0.95).sum()
        print(f"      Buildings damaged: {buildings_damaged:,} ({100*buildings_damaged/len(exp.gdf):.2f}%)")
        print(f"      Buildings destroyed (>95% loss): {buildings_destroyed:,}")
    
    # 3) Degrade exposure for Event 2
    if verbose:
        print(f"\n[2/3] Degrading building stock based on Event 1 damage...")
    
    exp_degraded = exp.copy()
    original_values = exp_degraded.gdf['value'].values.copy()
    original_total = float(original_values.sum())
    
    if degradation_mode == "direct":
        # Subtract damage, clipped at zero
        exp_degraded.gdf['value'] = np.maximum(
            original_values - event1_impact,
            0.0
        )
    elif degradation_mode == "ratio":
        # Apply damage ratio
        damage_ratios = np.where(
            original_values > 0,
            np.minimum(event1_impact / original_values, 1.0),
            0.0
        )
        exp_degraded.gdf['value'] = original_values * (1.0 - damage_ratios)
    else:
        raise ValueError(f"Unknown degradation_mode: '{degradation_mode}'")
    
    remaining_total = float(exp_degraded.gdf['value'].sum())
    pct_destroyed = 100.0 * (1.0 - remaining_total / original_total) if original_total > 0 else 0.0
    
    if verbose:
        print(f"      Original building stock: ${original_total:,.0f}")
        print(f"      Remaining after Event 1: ${remaining_total:,.0f}")
        print(f"      Value destroyed:         {pct_destroyed:.2f}%")
        
        # Show distribution of damage severity
        damage_pct = np.where(original_values > 0, event1_impact / original_values, 0) * 100
        bins = [0, 5, 25, 50, 75, 95, 100]
        for i in range(len(bins)-1):
            count = ((damage_pct >= bins[i]) & (damage_pct < bins[i+1])).sum()
            if count > 0:
                print(f"      Buildings with {bins[i]}-{bins[i+1]}% damage: {count:,}")
    
    # 4) Run Event 2 on degraded exposure
    if verbose:
        print(f"\n[3/3] Computing Event 2 impact on degraded building stock...")
    
    # Re-assign centroids if needed
    if not hasattr(exp_degraded.gdf, 'centr_TC') or exp_degraded.gdf['centr_TC'].isna().all():
        exp_degraded.assign_centroids(hazard)
    
    imp2 = ImpactCalc(exp_degraded, imp_fun_set, hazard).impact(save_mat=True)
    event2_impact = imp2.imp_mat[event2_idx, :].toarray().ravel()
    event2_total = float(event2_impact.sum())
    
    if verbose:
        print(f"      Event 2 total loss: ${event2_total:,.0f} (on degraded stock)")
        buildings_damaged_ev2 = (event2_impact > 0).sum()
        print(f"      Buildings damaged by Event 2: {buildings_damaged_ev2:,}")
        
        # Verification
        degraded_values = exp_degraded.gdf['value'].values
        violations = event2_impact > degraded_values
        n_violations = violations.sum()
        
        if n_violations > 0:
            print(f"      ⚠️  WARNING: {n_violations} buildings where Event2 > Remaining!")
        else:
            print(f"      ✓ All Event 2 damages ≤ remaining value (physically valid)")
    
    # 5) Composite impact
    composite_impact = event1_impact + event2_impact
    composite_total = float(composite_impact.sum())
    
    # Naive sum (both events on original stock)
    naive_event1 = event1_total
    naive_event2 = float(imp1.at_event[event2_idx])
    naive_sum = naive_event1 + naive_event2
    savings = naive_sum - composite_total
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results:")
        print(f"  Composite total (realistic):  ${composite_total:,.0f}")
        print(f"  Naive sum (wrong):            ${naive_sum:,.0f}")
        print(f"  Savings vs. naive:            ${savings:,.0f} ({100*savings/naive_sum:.1f}%)")
        print('='*70)
    
    # 6) Diagnostics
    hit_by_event1 = event1_impact > 0
    hit_by_event2 = event2_impact > 0
    n_hit_by_both = int((hit_by_event1 & hit_by_event2).sum())
    n_hit_by_either = int((hit_by_event1 | hit_by_event2).sum())
    
    diagnostics = {
        'event1_id': event1_id,
        'event2_id': event2_id,
        'event1_total': event1_total,
        'event2_total': event2_total,
        'composite_total': composite_total,
        'naive_sum': naive_sum,
        'savings_vs_naive': savings,
        'savings_pct': 100.0 * savings / naive_sum if naive_sum > 0 else 0.0,
        'original_exposure': original_total,
        'remaining_exposure_after_event1': remaining_total,
        'pct_exposure_destroyed_event1': pct_destroyed,
        'degradation_mode': degradation_mode,
        'n_buildings_hit_by_both': n_hit_by_both,
        'n_buildings_hit_by_either': n_hit_by_either,
        'overlap_rate': n_hit_by_both / n_hit_by_either if n_hit_by_either > 0 else 0.0,
        'n_buildings_total': len(exp.gdf),
    }
    
    return composite_impact, diagnostics


def aggregate_to_county(
    exp_gdf: gpd.GeoDataFrame,
    composite_impact: np.ndarray,
    county_fips_col: str = 'countyfp',
    county_name_col: str = 'county_name',
) -> pd.DataFrame:
    """Aggregate building-level composite impact to county totals."""
    
    df = exp_gdf.copy()
    df['composite_impact'] = composite_impact
    
    county_totals = (
        df.groupby([county_fips_col, county_name_col], dropna=False)['composite_impact']
        .sum(min_count=1)
        .rename('value')
        .reset_index()
    )
    
    # Clean county FIPS
    county_totals[county_fips_col] = (
        county_totals[county_fips_col]
        .astype(str)
        .str.replace(r'\.0$', '', regex=True)
        .str.zfill(3)
    )
    
    return county_totals


def generate_composite_scenarios(
    exp_with_county: gpd.GeoDataFrame,
    hazard,
    imp_fun_set,
    scenarios: list,
    output_dir: Path,
    verbose: bool = True,
):
    """
    Generate multiple composite event scenarios and save county-aggregated CSVs.
    
    Parameters
    ----------
    scenarios : list of tuples
        [(event1_id, event2_id, scenario_name), ...]
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Generating {len(scenarios)} Composite Scenarios (BUILDING-LEVEL)")
    print(f"Output directory: {output_dir}")
    print('='*70)
    
    results = []
    
    for i, (event1_id, event2_id, scenario_name) in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario_name}")
        print(f"       Event 1: {event1_id}")
        print(f"       Event 2: {event2_id}")
        
        # Create Exposures object from GeoDataFrame
        exp_climada = Exposures(exp_with_county)
        
        # Compute sequential impact
        composite_impact, diag = compute_sequential_impact(
            exp_climada, event1_id, event2_id, hazard, imp_fun_set,
            degradation_mode="direct", verbose=verbose
        )
        
        # Aggregate to counties
        county_totals = aggregate_to_county(exp_with_county, composite_impact)
        
        # Save CSV
        csv_path = output_dir / f"{scenario_name}.csv"
        county_totals.to_csv(csv_path, index=False)
        print(f"       ✓ Wrote: {csv_path.name}")
        
        # Store diagnostics
        diag['scenario_name'] = scenario_name
        results.append(diag)
    
    # Save diagnostics
    diag_df = pd.DataFrame(results)
    diag_path = output_dir / "sequential_scenarios_diagnostics_capra.csv"
    diag_df.to_csv(diag_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Generated {len(results)}/{len(scenarios)} scenarios")
    print(f"✓ Diagnostics: {diag_path}")
    print('='*70)
    
    print("\nSummary:")
    print(diag_df[['scenario_name', 'composite_total', 'savings_vs_naive', 'savings_pct']].to_string(index=False))
    
    return results


def main():
    """Main execution function."""
    
    DATA_DIR = Path(cfg.DATA_DIR) if hasattr(cfg, 'DATA_DIR') else Path('../data')
    HAZARD_OUTPUT_DIR = DATA_DIR / "hazard"
    
    print("\n" + "="*70)
    print("Sequential Event Composite Generator - BUILDING-LEVEL VERSION")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1) Load BUILDING-LEVEL exposure
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading building-level exposure...")
    
    exp_path = DATA_DIR / "florida_exposure.hdf5"
    if not exp_path.exists():
        print(f"❌ Building exposure not found: {exp_path}")
        sys.exit(1)
    
    exp = Exposures.from_hdf5(str(exp_path))
    
    # Set value from ReplacementCost
    if 'ReplacementCost' in exp.gdf.columns:
        exp.gdf['value'] = exp.gdf['ReplacementCost']
        print(f"      ✓ Set value = ReplacementCost")
    
    # Assign structure-type-specific impact function IDs using CAPRA mapping
    if 'StructureType' in exp.gdf.columns:
        exp.gdf['impf_TC'] = exp.gdf.apply(
            lambda row: DICT_PAGER_TCIMPF_CAPRA.get(row.StructureType, 73),  # Default to URM if unknown
            axis=1
        )
        print(f"      ✓ Assigned CAPRA impact function IDs by StructureType")
        # Show distribution of structure types
        type_counts = exp.gdf['StructureType'].value_counts().head(5)
        print(f"      Top 5 structure types:")
        for stype, count in type_counts.items():
            print(f"        {stype}: {count:,} buildings ({100*count/len(exp.gdf):.1f}%)")
    else:
        exp.gdf['impf_TC'] = 73  # Default to URM if no StructureType column
        print(f"      ⚠️  No StructureType column, using default impact function (URM)")
    
    print(f"      Loaded {len(exp.gdf):,} buildings")
    print(f"      Total replacement cost: ${exp.gdf['value'].sum():,.0f}")
    
    # -------------------------------------------------------------------------
    # 2) Load county boundaries and spatial join
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading county boundaries and joining buildings...")
    
    county_fp = DATA_DIR / "US_counties"
    if not county_fp.exists():
        raise FileNotFoundError(f"County shapefile not found: {county_fp}")
    
    counties = gpd.read_file(county_fp)
    fl_counties = counties[counties["STATEFP"] == "12"].copy()
    fl_counties['geometry'] = fl_counties['geometry'].buffer(0)
    
    if fl_counties.crs is None:
        fl_counties = fl_counties.set_crs('EPSG:4269')
    
    # Spatial join
    exp_gdf = exp.gdf.copy()
    if exp_gdf.crs is None:
        exp_gdf = exp_gdf.set_crs('EPSG:4326')
    
    fl_counties = fl_counties.to_crs(exp_gdf.crs)
    
    print(f"      Spatially joining {len(exp_gdf):,} buildings with counties...")
    exp_with_county = gpd.sjoin(
        exp_gdf,
        fl_counties[['COUNTYFP', 'NAME', 'geometry']],
        how='left',
        predicate='within',
    )
    exp_with_county = exp_with_county.drop(columns=['index_right'], errors='ignore')
    exp_with_county = exp_with_county.rename(columns={'COUNTYFP': 'countyfp', 'NAME': 'county_name'})
    
    print(f"      ✓ Joined {len(exp_with_county):,} buildings")
    
    # -------------------------------------------------------------------------
    # 3) Load hazard
    # -------------------------------------------------------------------------
    print("\n[3/5] Loading hazard for sequential events...")
    
    storm_ids = [
        "1926255N15314",  # Great Miami
        "1992230N11325",  # Andrew
        "2017242N16333",  # Irma
    ]
    
    hurricanes = TCTracks.from_ibtracs_netcdf(storm_id=storm_ids)
    hurricanes.equal_timestep(time_step_h=0.5)
    
    cent = Centroids.from_pnt_bounds(
        (-90.0, 24.0, -79.0, 31.5),
        res=120/3600
    )
    
    hazard = TropCyclone.from_tracks(hurricanes, centroids=cent)
    print(f"      Loaded hazard with {len(hazard.event_name)} events")
    
    # -------------------------------------------------------------------------
    # 4) Load impact function set
    # -------------------------------------------------------------------------
    print("\n[4/5] Loading CAPRA impact function set...")
    
    # Use CAPRA impact functions (structure-type specific)
    imp_fun_set = IMPF_SET_TC_CAPRA
    print(f"      ✓ Loaded {imp_fun_set.size()} structure-type-specific impact functions")
    
    # Assign centroids
    exp_climada = Exposures(exp_with_county)
    exp_climada.assign_centroids(hazard)
    
    # -------------------------------------------------------------------------
    # 5) Generate composite scenarios
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating composite scenarios...")
    
    scenarios = [
        ("1926255N15314", "1992230N11325", "great_miami_then_andrew_capra"),
        ("1992230N11325", "1926255N15314", "andrew_then_great_miami_capra"),
        ("1926255N15314", "1926255N15314", "great_miami_twice_capra"),
        ("2017242N16333", "2017242N16333", "irma_twice_capra"),
    ]
    
    results = generate_composite_scenarios(
        exp_with_county=exp_with_county,
        hazard=hazard,
        imp_fun_set=imp_fun_set,
        scenarios=scenarios,
        output_dir=HAZARD_OUTPUT_DIR,
        verbose=True,
    )
    
    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("✓ Done!")
    print("="*70)
    
    print(f"\nGenerated 4 composite event CSVs (CAPRA building-level) in:")
    print(f"  {HAZARD_OUTPUT_DIR}/")
    print(f"\nFiles created:")
    print(f"  - great_miami_then_andrew_capra.csv")
    print(f"  - andrew_then_great_miami_capra.csv")
    print(f"  - great_miami_twice_capra.csv")
    print(f"  - irma_twice_capra.csv")
    print(f"  - sequential_scenarios_diagnostics_capra.csv")
    print(f"\nCompare with:")
    print(f"  - LitPop results (no suffix)")
    print(f"  - Previous building results (if any, with '_buildings' suffix)")
    print(f"\nKey difference: Using CAPRA structure-type-specific impact functions")


if __name__ == '__main__':
    main()
