#!/usr/bin/env python3
"""
compute_sequential_events.py - Compute damage from sequential hurricane events

Computes physically realistic total damage from sequential hurricane events by:
  1. Running Event 1 on original exposure
  2. Degrading exposure state (subtract destroyed value)
  3. Running Event 2 on degraded exposure
  4. Aggregating to county-level composite losses

This approach prevents double-counting of destroyed buildings when combining
multiple events in the same season.

Usage
-----
Run as script to generate composite event CSVs:
    python compute_sequential_events.py

Or import functions for custom scenarios:
    from scripts.hazard.compute_sequential_events import compute_sequential_impact
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import warnings

# Add parent directory to path for fl_risk_model imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Check for required packages
try:
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install numpy pandas geopandas matplotlib")
    sys.exit(1)

# CLIMADA imports
try:
    from climada.hazard import Centroids, Hazard, TropCyclone, TCTracks
    from climada.entity.exposures import Exposures
    from climada.entity import LitPop
    from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone
    from climada.engine import ImpactCalc
except ImportError as e:
    print(f"ERROR: CLIMADA not installed or not accessible: {e}")
    print("\nPlease install CLIMADA:")
    print("  pip install climada")
    print("\nOr activate the correct conda environment:")
    print("  conda activate climada_env")
    sys.exit(1)

from fl_risk_model import config as cfg

# Suppress CLIMADA warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='climada')

# ============================================================================
# Configuration
# ============================================================================

def P(x):
    """Coerce string paths to Path objects."""
    return x if isinstance(x, Path) else Path(x)

DATA_DIR = P(getattr(cfg, "DATA_DIR", "data"))
HAZARD_OUTPUT_DIR = DATA_DIR / "hazard"
FIGURES_DIR = Path(getattr(cfg, "FIGURES_DIR", DATA_DIR.parent / "results" / "figures"))

# Ensure output directories exist
HAZARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Core Functions: Sequential Event Impact
# ============================================================================

def compute_sequential_impact(
    exp: Exposures,
    event1_id: str,
    event2_id: str,
    hazard: Hazard,
    imp_fun_set: ImpfSetTropCyclone,
    degradation_mode: str = "direct",
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Compute realistic total damage from two sequential events.
    
    Workflow:
      1. Run Event 1 -> get point-level impact (building losses)
      2. Degrade exposure: value[i] -= impact1[i] for each location i
      3. Run Event 2 on degraded exposure -> get capped impact
      4. Return: composite_impact = event1_impact + event2_impact
    
    Parameters
    ----------
    exp : Exposures
        CLIMADA exposure object with GeoDataFrame of building values
    event1_id : str
        IBTrACS storm ID for first event (e.g., "1926255N15314")
    event2_id : str
        IBTrACS storm ID for second event (can be same as event1_id)
    hazard : Hazard
        CLIMADA hazard object containing both events
    imp_fun_set : ImpfSetTropCyclone
        Calibrated impact function set for tropical cyclones
    degradation_mode : str, default "direct"
        - "direct": subtract absolute damage from value (recommended)
        - "ratio": apply damage ratio to remaining value (more conservative)
    verbose : bool, default True
        Print progress and diagnostics
    
    Returns
    -------
    composite_impact : np.ndarray
        Point-level composite impact (length = n_exposure_points)
    diagnostics : dict
        Summary statistics:
          - event1_total, event2_total, composite_total (USD)
          - pct_exposure_destroyed_event1 (%)
          - remaining_exposure_after_event1 (USD)
          - savings_vs_naive (USD, how much double-counting avoided)
          - n_locations_hit_by_both, n_locations_hit_by_either
    
    Raises
    ------
    ValueError
        If event IDs not found in hazard.event_name
    
    Examples
    --------
    >>> haz = TropCyclone.from_tracks(tracks, centroids)
    >>> exp = Exposures.from_hdf5("FL_exposure.hdf5")
    >>> imp_set = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet()
    >>> composite, diag = compute_sequential_impact(
    ...     exp, "1926255N15314", "1992230N11325", haz, imp_set
    ... )
    >>> print(f"Composite total: ${diag['composite_total']:,.0f}")
    """
    
    # 1) Find event indices in hazard
    event1_idx = _get_event_index(hazard, event1_id)
    event2_idx = _get_event_index(hazard, event2_id)
    
    if event1_idx is None:
        raise ValueError(f"Event '{event1_id}' not found in hazard.event_name")
    if event2_idx is None:
        raise ValueError(f"Event '{event2_id}' not found in hazard.event_name")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Computing Sequential Impact")
        print(f"  Event 1: {event1_id} (index {event1_idx})")
        print(f"  Event 2: {event2_id} (index {event2_idx})")
        print(f"  Degradation mode: {degradation_mode}")
        print('='*70)
    
    # 2) Run Event 1 on original exposure
    if verbose:
        print(f"\n[1/3] Computing Event 1 impact on original exposure...")
    
    imp1 = ImpactCalc(exp, imp_fun_set, hazard).impact(save_mat=True)
    event1_impact = imp1.imp_mat[event1_idx, :].toarray().ravel()
    event1_total = float(event1_impact.sum())
    
    if verbose:
        print(f"      Event 1 total loss: ${event1_total:,.0f}")
    
    # 3) Degrade exposure for Event 2
    if verbose:
        print(f"\n[2/3] Degrading exposure based on Event 1 damage...")
    
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
        # Apply damage ratio (more conservative for partial damage)
        damage_ratios = np.where(
            original_values > 0,
            np.minimum(event1_impact / original_values, 1.0),
            0.0
        )
        exp_degraded.gdf['value'] = original_values * (1.0 - damage_ratios)
    else:
        raise ValueError(f"Unknown degradation_mode: '{degradation_mode}'. Use 'direct' or 'ratio'.")
    
    remaining_total = float(exp_degraded.gdf['value'].sum())
    pct_destroyed = 100.0 * (1.0 - remaining_total / original_total) if original_total > 0 else 0.0
    
    if verbose:
        print(f"      Original exposure:   ${original_total:,.0f}")
        print(f"      Remaining exposure:  ${remaining_total:,.0f}")
        print(f"      Destroyed:           {pct_destroyed:.2f}%")
        
        # Diagnostic: Show top 10 most-damaged points
        print(f"\n      Top 10 most-damaged points by Event 1:")
        print(f"      {'Idx':<6} {'Original $':<15} {'Event1 Dmg':<15} {'Remaining $':<15} {'% Lost':<8}")
        print(f"      {'-'*70}")
        top_damage_idx = np.argsort(event1_impact)[-10:][::-1]
        for i, idx in enumerate(top_damage_idx, 1):
            orig = original_values[idx]
            dmg = event1_impact[idx]
            remain = exp_degraded.gdf['value'].values[idx]
            pct = 100 * dmg / orig if orig > 0 else 0
            print(f"      {idx:<6} ${orig:>13,.0f} ${dmg:>13,.0f} ${remain:>13,.0f} {pct:>6.1f}%")
    
    # 4) Run Event 2 on degraded exposure
    if verbose:
        print(f"\n[3/3] Computing Event 2 impact on degraded exposure...")
    
    # Re-assign centroids if needed (usually preserved in copy)
    if not hasattr(exp_degraded.gdf, 'centr_TC') or exp_degraded.gdf['centr_TC'].isna().all():
        exp_degraded.assign_centroids(hazard)
    
    imp2 = ImpactCalc(exp_degraded, imp_fun_set, hazard).impact(save_mat=True)
    event2_impact = imp2.imp_mat[event2_idx, :].toarray().ravel()
    event2_total = float(event2_impact.sum())
    
    if verbose:
        print(f"      Event 2 total loss: ${event2_total:,.0f} (on degraded exposure)")
        
        # Diagnostic: Verify Event 2 didn't exceed remaining exposure at any point
        print(f"\n      Verification - Event 2 vs Remaining Exposure:")
        degraded_values = exp_degraded.gdf['value'].values
        violations = event2_impact > degraded_values
        n_violations = violations.sum()
        
        if n_violations > 0:
            print(f"      [WARNING] WARNING: {n_violations} points where Event2 > Remaining!")
            print(f"      {'Idx':<6} {'Remaining $':<15} {'Event2 Dmg':<15} {'Excess $':<15}")
            print(f"      {'-'*60}")
            violation_idx = np.where(violations)[0][:5]  # Show first 5
            for idx in violation_idx:
                remain = degraded_values[idx]
                dmg2 = event2_impact[idx]
                excess = dmg2 - remain
                print(f"      {idx:<6} ${remain:>13,.0f} ${dmg2:>13,.0f} ${excess:>13,.0f}")
        else:
            print(f"      [OK] All Event 2 damages ≤ remaining exposure (physically valid)")
        
        # Show top 10 Event 2 damages and compare to original Event 2 standalone
        print(f"\n      Top 10 Event 2 impacts (on degraded vs original exposure):")
        print(f"      {'Idx':<6} {'Event2 Actual':<15} {'Event2 Naive':<15} {'Difference':<15} {'% Reduced':<10}")
        print(f"      {'-'*75}")
        top_ev2_idx = np.argsort(event2_impact)[-10:][::-1]
        event2_naive_impact = imp1.imp_mat[event2_idx, :].toarray().ravel()
        for idx in top_ev2_idx:
            actual = event2_impact[idx]
            naive = event2_naive_impact[idx]
            diff = naive - actual
            pct_red = 100 * diff / naive if naive > 0 else 0
            print(f"      {idx:<6} ${actual:>13,.0f} ${naive:>13,.0f} ${diff:>13,.0f} {pct_red:>8.1f}%")
    
    # 5) Composite impact
    composite_impact = event1_impact + event2_impact
    composite_total = float(composite_impact.sum())
    
    # What would naive sum give? (wrong approach - both events on ORIGINAL exposure)
    # Note: We already have imp1 (on original), need to ensure imp2 is also on original for comparison
    # Actually, we should recalculate both on original for the naive case
    naive_event1 = event1_total  # Already calculated on original
    naive_event2 = float(imp1.at_event[event2_idx])  # Event 2 also on original
    naive_sum = naive_event1 + naive_event2
    savings = naive_sum - composite_total
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Results:")
        print(f"  Composite total (realistic):  ${composite_total:,.0f}")
        print(f"  Naive sum (wrong):            ${naive_sum:,.0f}")
        print(f"  Savings vs. naive:            ${savings:,.0f} ({100*savings/naive_sum:.1f}%)")
        print('='*70)
    
    # 6) Spatial overlap diagnostics
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
        'n_locations_hit_by_both': n_hit_by_both,
        'n_locations_hit_by_either': n_hit_by_either,
        'overlap_rate': n_hit_by_both / n_hit_by_either if n_hit_by_either > 0 else 0.0,
    }
    
    return composite_impact, diagnostics


def aggregate_to_county(
    exp_gdf: gpd.GeoDataFrame,
    composite_impact: np.ndarray,
    county_fips_col: str = 'countyfp',
    county_name_col: str = 'county_name',
) -> pd.DataFrame:
    """
    Aggregate point-level composite impact to county totals.
    
    Parameters
    ----------
    exp_gdf : GeoDataFrame
        Exposure GeoDataFrame with county fields (from spatial join)
    composite_impact : np.ndarray
        Point-level impact values (length = len(exp_gdf))
    county_fips_col : str, default 'countyfp'
        Column name for county FIPS code
    county_name_col : str, default 'county_name'
        Column name for county name
    
    Returns
    -------
    pd.DataFrame
        County-level totals: [countyfp, county_name, value]
        
    Raises
    ------
    ValueError
        If county columns not found in exp_gdf
    """
    
    if county_fips_col not in exp_gdf.columns:
        raise ValueError(f"Column '{county_fips_col}' not found in exposure GDF. "
                        "Run spatial join with county boundaries first.")
    if county_name_col not in exp_gdf.columns:
        raise ValueError(f"Column '{county_name_col}' not found in exposure GDF.")
    
    if len(composite_impact) != len(exp_gdf):
        raise ValueError(f"Impact array length ({len(composite_impact)}) != "
                        f"exposure GDF length ({len(exp_gdf)})")
    
    # Build temporary DataFrame
    temp = exp_gdf[[county_fips_col, county_name_col]].copy()
    temp['composite_impact'] = composite_impact
    
    # Aggregate to county
    county_df = (
        temp.groupby([county_fips_col, county_name_col], dropna=False)['composite_impact']
        .sum()
        .rename('value')
        .reset_index()
    )
    
    return county_df


def _get_event_index(hazard: Hazard, event_id: str) -> Optional[int]:
    """Find index of event_id in hazard.event_name list."""
    try:
        return list(hazard.event_name).index(event_id)
    except (ValueError, AttributeError):
        return None


# ============================================================================
# Batch Processing: Generate Multiple Scenarios
# ============================================================================

def generate_composite_scenarios(
    exp_with_county: gpd.GeoDataFrame,
    hazard: Hazard,
    imp_fun_set: ImpfSetTropCyclone,
    scenarios: List[Tuple[str, str, str]],
    output_dir: Path = HAZARD_OUTPUT_DIR,
    verbose: bool = True,
) -> dict:
    """
    Generate composite event CSVs for multiple sequential event pairs.
    
    Parameters
    ----------
    exp_with_county : GeoDataFrame
        Exposure with county spatial join already completed
        (must have 'countyfp' and 'county_name' columns)
    hazard : Hazard
        CLIMADA hazard containing all events
    imp_fun_set : ImpfSetTropCyclone
        Calibrated impact function set
    scenarios : list of (event1_id, event2_id, output_name)
        Each tuple defines one scenario to generate
    output_dir : Path, default DATA_DIR/hazard
        Directory to write composite event CSVs
    verbose : bool, default True
        Print progress
    
    Returns
    -------
    dict
        {scenario_name: {'county_df': pd.DataFrame, 'diagnostics': dict}}
        
    Side Effects
    ------------
    - Writes CSV files to output_dir: {scenario_name}.csv
    - Writes diagnostics summary: sequential_scenarios_diagnostics.csv
    
    Examples
    --------
    >>> scenarios = [
    ...     ("1926255N15314", "1992230N11325", "great_miami_then_andrew"),
    ...     ("1926255N15314", "1926255N15314", "great_miami_twice"),
    ... ]
    >>> results = generate_composite_scenarios(exp_gdf, hazard, imp_set, scenarios)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert GeoDataFrame to Exposures object (CLIMADA requirement)
    exp = Exposures(exp_with_county)
    
    results = {}
    all_diagnostics = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Generating {len(scenarios)} Composite Scenarios")
        print(f"Output directory: {output_dir}")
        print('='*70)
    
    for i, (event1_id, event2_id, scenario_name) in enumerate(scenarios, 1):
        if verbose:
            print(f"\n[{i}/{len(scenarios)}] {scenario_name}")
            print(f"       Event 1: {event1_id}")
            print(f"       Event 2: {event2_id}")
        
        try:
            # Compute sequential impact
            composite_impact, diag = compute_sequential_impact(
                exp=exp,
                event1_id=event1_id,
                event2_id=event2_id,
                hazard=hazard,
                imp_fun_set=imp_fun_set,
                degradation_mode="direct",
                verbose=verbose,
            )
            
            # Aggregate to county
            county_df = aggregate_to_county(
                exp_gdf=exp_with_county,
                composite_impact=composite_impact,
            )
            
            # Write CSV
            output_path = output_dir / f"{scenario_name}.csv"
            county_df.to_csv(output_path, index=False)
            
            if verbose:
                print(f"       [OK] Wrote: {output_path.name}")
            
            # Store results
            results[scenario_name] = {
                'county_df': county_df,
                'diagnostics': diag,
            }
            
            # Collect diagnostics for summary
            all_diagnostics.append({
                'scenario_name': scenario_name,
                **diag
            })
            
        except Exception as e:
            print(f"       [FAIL] Failed: {e}")
            continue
    
    # Write diagnostics summary
    if all_diagnostics:
        diag_df = pd.DataFrame(all_diagnostics)
        diag_path = output_dir / "sequential_scenarios_diagnostics.csv"
        diag_df.to_csv(diag_path, index=False)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"[OK] Generated {len(results)}/{len(scenarios)} scenarios")
            print(f"[OK] Diagnostics: {diag_path}")
            print('='*70)
            print("\nSummary:")
            print(diag_df[['scenario_name', 'composite_total', 'savings_vs_naive', 'savings_pct']]
                  .to_string(index=False))
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_sequential_comparison(
    exp_with_county: gpd.GeoDataFrame,
    fl_counties: gpd.GeoDataFrame,
    event1_id: str,
    event2_id: str,
    composite_name: str,
    composite_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    vmin: float = 1e5,
    vmax: float = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create 3-panel comparison map:
      [Event 1 standalone] | [Event 2 standalone] | [Sequential composite]
    
    Parameters
    ----------
    exp_with_county : GeoDataFrame
        Exposure with event impact columns (e.g., event_id = column)
    fl_counties : GeoDataFrame
        Florida county boundaries
    event1_id : str
        Column name for Event 1 (IBTrACS ID)
    event2_id : str
        Column name for Event 2
    composite_name : str
        Label for composite scenario
    composite_df : pd.DataFrame
        County-level composite totals from aggregate_to_county()
    output_dir : Path
        Where to save figure
    vmin, vmax : float, optional
        Color scale limits (USD)
    
    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Helper to aggregate point-level to county
    def _county_totals(gdf, event_col):
        return (
            gdf.groupby(['countyfp', 'county_name'], dropna=False)[event_col]
            .sum()
            .reset_index()
            .rename(columns={event_col: 'value'})
        )
    
    # Panel 1: Event 1 standalone
    if event1_id in exp_with_county.columns:
        e1_counties = fl_counties.merge(
            _county_totals(exp_with_county, event1_id),
            left_on='COUNTYFP', right_on='countyfp', how='left'
        )
        
        if vmax is None:
            vmax = e1_counties['value'].max()
        
        e1_counties.plot(
            column='value',
            ax=axes[0],
            legend=True,
            cmap='YlOrRd',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            missing_kwds={'color': 'lightgrey', 'label': 'No data'},
        )
        axes[0].set_title(f"Event 1: {event1_id}\n(standalone)", fontsize=11)
    else:
        axes[0].text(0.5, 0.5, f"Event {event1_id}\nnot in exposure",
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("Event 1")
    
    axes[0].axis('off')
    
    # Panel 2: Event 2 standalone
    if event2_id in exp_with_county.columns:
        e2_counties = fl_counties.merge(
            _county_totals(exp_with_county, event2_id),
            left_on='COUNTYFP', right_on='countyfp', how='left'
        )
        
        e2_counties.plot(
            column='value',
            ax=axes[1],
            legend=True,
            cmap='YlOrRd',
            norm=LogNorm(vmin=vmin, vmax=vmax),
            missing_kwds={'color': 'lightgrey'},
        )
        axes[1].set_title(f"Event 2: {event2_id}\n(standalone)", fontsize=11)
    else:
        axes[1].text(0.5, 0.5, f"Event {event2_id}\nnot in exposure",
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Event 2")
    
    axes[1].axis('off')
    
    # Panel 3: Sequential composite
    comp_counties = fl_counties.merge(
        composite_df,
        left_on='COUNTYFP', right_on='countyfp', how='left'
    )
    
    comp_counties.plot(
        column='value',
        ax=axes[2],
        legend=True,
        cmap='YlOrRd',
        norm=LogNorm(vmin=vmin, vmax=vmax),
        missing_kwds={'color': 'lightgrey'},
    )
    axes[2].set_title(f"Sequential Composite:\n{composite_name}", fontsize=11)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f"{composite_name}_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {fig_path}")
    
    return fig, axes


# ============================================================================
# Main Script: Generate Default Scenarios
# ============================================================================

def main():
    """
    Generate the four requested composite scenarios:
      a) Great Miami -> Andrew
      b) Andrew -> Great Miami
      c) Great Miami -> Great Miami
      d) Irma -> Irma
    """
    
    print("\n" + "="*70)
    print("Sequential Event Composite Generator")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # 1) Load exposure
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading exposure...")
    
    exp_path = DATA_DIR / 'FL_exposure_120as.hdf5'
    if not exp_path.exists():
        print(f"\nERROR: Exposure file not found: {exp_path}")
        print("\nYou need to generate the exposure file first.")
        print("Options:")
        print("\n  Option 1: Run simulate_event_losses.py cells up to exposure export")
        print("  Open: scripts/hazard/simulate_event_losses.py")
        print("  Run cells until you see: FL.write_hdf5(...)")
        print("\n  Option 2: Quick exposure generation:")
        print("  Run this code in Python/Jupyter:")
        print("  ```python")
        print("  from climada.entity import LitPop")
        print("  from climada.entity.exposures import Exposures")
        print("  import os")
        print("  exp = LitPop.from_countries('USA', fin_mode='pc', res_arcsec=120,")
        print("                               exponents=(1,1), admin1_calc=True, reference_year=2024)")
        print("  FL = Exposures()")
        print("  FL.set_gdf(exp.gdf[exp.gdf.admin1=='Florida'])")
        print(f"  FL.write_hdf5('{exp_path}')")
        print("  ```")
        print(f"\nExpected location: {exp_path}")
        sys.exit(1)
    
    exp = Exposures.from_hdf5(str(exp_path))
    exp.gdf['impf_TC'] = 2  # Assign USA impact function (region 2)
    print(f"      Loaded {len(exp.gdf):,} exposure points")
    
    # -------------------------------------------------------------------------
    # 2) Load county boundaries and spatial join
    # -------------------------------------------------------------------------
    print("\n[2/5] Loading county boundaries...")
    
    county_fp = DATA_DIR / "US_counties"
    if not county_fp.exists():
        raise FileNotFoundError(f"County shapefile not found: {county_fp}")
    
    counties = gpd.read_file(county_fp)
    fl_counties = counties[counties["STATEFP"] == "12"].copy()
    fl_counties['geometry'] = fl_counties['geometry'].buffer(0)  # fix invalid geoms
    
    # Ensure counties have CRS (usually EPSG:4269 NAD83, convert to WGS84)
    if fl_counties.crs is None:
        print("      Warning: Counties shapefile has no CRS, assuming EPSG:4269 (NAD83)")
        fl_counties = fl_counties.set_crs('EPSG:4269')
    
    # Spatial join - work with the GeoDataFrame directly
    exp_gdf = exp.gdf.copy()
    if exp_gdf.crs is None:
        exp_gdf = exp_gdf.set_crs('EPSG:4326')
    
    fl_counties = fl_counties.to_crs(exp_gdf.crs)
    
    exp_with_county = gpd.sjoin(
        exp_gdf,
        fl_counties[['COUNTYFP', 'NAME', 'geometry']],
        how='left',
        predicate='within',
    )
    exp_with_county = exp_with_county.drop(columns=['index_right'], errors='ignore')
    exp_with_county = exp_with_county.rename(columns={'COUNTYFP': 'countyfp', 'NAME': 'county_name'})
    
    print(f"      Joined {len(exp_with_county):,} points with counties")
    
    # -------------------------------------------------------------------------
    # 3) Load hazard for sequential events
    # -------------------------------------------------------------------------
    print("\n[3/5] Loading hazard for sequential events...")
    
    storm_ids = [
        "1926255N15314",  # Great Miami
        "1928250N14343",  # Lake Okeechobee
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
    print("\n[4/5] Loading impact function set...")
    
    imp_fun_set = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet(
        calibration_approach='RMSF'
    )
    
    # Assign centroids
    exp_climada = Exposures(exp_with_county)
    exp_climada.assign_centroids(hazard)
    
    # -------------------------------------------------------------------------
    # 5) Generate composite scenarios
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating composite scenarios...")
    
    scenarios = [
        ("1926255N15314", "1992230N11325", "great_miami_then_andrew"),
        ("1992230N11325", "1926255N15314", "andrew_then_great_miami"),
        ("1926255N15314", "1926255N15314", "great_miami_twice"),
        ("2017242N16333", "2017242N16333", "irma_twice"),
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
    # Optional: Generate comparison plots
    # -------------------------------------------------------------------------
    print("\n[Optional] Generating comparison plots...")
    
    # Need to load standalone impacts first (if not already in exp_with_county)
    # For now, skip plotting if events aren't in the GDF
    # You can uncomment and adapt if needed:
    
    # for scenario_name, result in results.items():
    #     try:
    #         event1_id = result['diagnostics']['event1_id']
    #         event2_id = result['diagnostics']['event2_id']
            
    #         plot_sequential_comparison(
    #             exp_with_county=exp_with_county,
    #             fl_counties=fl_counties,
    #             event1_id=event1_id,
    #             event2_id=event2_id,
    #             composite_name=scenario_name,
    #             composite_df=result['county_df'],
    #             output_dir=FIGURES_DIR,
    #         )
    #     except Exception as e:
    #         print(f"  Skipping plot for {scenario_name}: {e}")
    
    print("\n" + "="*70)
    print("[OK] Done!")
    print("="*70)
    print(f"\nGenerated {len(results)} composite event CSVs in:")
    print(f"  {HAZARD_OUTPUT_DIR}/")
    print("\nNext steps:")
    print("  1. Check diagnostics CSV for validation")
    print("  2. Update mc_run_events.py SCENARIOS dict:")
    print("       SCENARIOS = {")
    for scenario_name, _, _ in scenarios:
        print(f"           '{scenario_name}': ['{scenario_name}'],")
    print("       }")
    print("  3. Run Monte Carlo with new composite events")


if __name__ == "__main__":
    main()
