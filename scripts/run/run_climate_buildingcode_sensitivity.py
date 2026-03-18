#!/usr/bin/env python3
"""
Climate Change + Building Codes Sensitivity Analysis

Research Question: How much building code improvement is needed to offset 
SSP245 2050 climate change impacts on systemic insurance risk?

Experimental Design:
- Fixed: SSP245 2050 climate change scenario 
  * Uses county-specific damage scaling factors from Gori et al. damage functions
  * Applied to CLIMADA TC climate projections (GCM ensemble median)
  * File: florida_linear_damage_scaling_factors.csv
  * Mean statewide scaling: ~1.67x damage increase (67% increase)
- Varying: Building code wind loss reduction [0%, 5%, 10%, ..., 50%]
- Target: Find loss_reduction that brings metrics back to current climate baseline

Climate Scaling Methodology:
The climate_scaling=True parameter triggers county-specific damage multipliers
derived from Gori et al. (2022) TC damage functions applied to CLIMADA's GCM
ensemble projections for SSP2-4.5 2050. This represents the expected damage
increase from more intense hurricanes under climate change.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fl_risk_model.mc_run_events import run_stochastic_tc_monte_carlo
from fl_risk_model.scenarios import building_codes
import json
import argparse


def run_climate_buildingcode_sweep(loss_reduction_pct: float, seed: int = 42):
    """
    Run Monte Carlo simulation with SSP245 2050 climate + building codes.
    
    Parameters
    ----------
    loss_reduction_pct : float
        Wind loss reduction percentage (0-50)
    seed : int
        Random seed for reproducibility
    """
    
    # Create custom building code scenario with specified loss reduction
    scenario_name = f"CUSTOM_{int(loss_reduction_pct):02d}PCT"
    building_codes.BUILDING_CODE_PRESETS[scenario_name] = {
        "wind_loss_reduction": loss_reduction_pct / 100.0,
        "flood_loss_reduction": 0.0,  # Focus on wind
        "retrofit_rate": 0.0,
        "description": f"Custom {loss_reduction_pct}% wind loss reduction",
        "references": "Climate adaptation sensitivity analysis"
    }
    
    # Policy scenario: Building codes + climate change
    policy_scenario_config = {
        "type": "building_codes",
        "params": {
            "scenario": scenario_name,
            "hazard_type": "wind",
            "apply_by_county": False,
        }
    }
    
    # Custom run label for identification
    run_label = f"climate_buildingcode_{int(loss_reduction_pct):02d}pct"
    
    print("="*80)
    print("CLIMATE + BUILDING CODES SENSITIVITY RUN")
    print("="*80)
    print(f"Climate Scenario: SSP245 2050 (damage deltas applied)")
    print(f"Building Code Wind Loss Reduction: {loss_reduction_pct}%")
    print(f"Run Label: {run_label}")
    print(f"Random Seed: {seed}")
    print("="*80)
    
    # Run Monte Carlo simulation with climate scaling enabled
    run_stochastic_tc_monte_carlo(
        n_years=10000,
        policy_scenario_config=policy_scenario_config,
        climate_scaling=True,  # Enable SSP245 2050 climate deltas
        run_label=run_label,
        seed=seed
    )
    
    # Save metadata for analysis
    output_dir = Path(f"results/mc_runs/{run_label}_*").resolve().parent
    metadata = {
        "wind_loss_reduction_pct": loss_reduction_pct,
        "flood_loss_reduction_pct": 0.0,
        "climate_scenario": "ssp245_2050",
        "policy_scenario": "building_codes",
        "seed": seed,
        "n_years": 10000,
    }
    
    # Find the actual output directory (has timestamp)
    import glob
    matching_dirs = sorted(glob.glob(str(output_dir / f"../{run_label}_*")))
    if matching_dirs:
        actual_dir = Path(matching_dirs[-1])
        metadata_file = actual_dir / 'climate_buildingcode_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✓ Saved metadata: {metadata_file}")
    
    print(f"\n✓ Climate + building codes run complete: {loss_reduction_pct}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run climate change + building codes sensitivity analysis"
    )
    parser.add_argument(
        "--loss_reduction",
        type=float,
        required=True,
        help="Wind loss reduction percentage (0-50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate loss reduction range
    if not (0 <= args.loss_reduction <= 50):
        raise ValueError("loss_reduction must be between 0 and 50")
    
    run_climate_buildingcode_sweep(
        loss_reduction_pct=args.loss_reduction,
        seed=args.seed
    )
