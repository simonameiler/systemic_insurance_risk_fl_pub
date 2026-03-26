#!/usr/bin/env python3
"""
run_penetration_capital_sensitivity.py - Monte Carlo runner for penetration capital sensitivity
=============================================================

Sensitivity analysis to find the optimal capital scaling factor needed
to stabilize company defaults under increased insurance penetration.

Research Question:
    How much additional capital (beyond linear scaling) do insurers need
    to maintain baseline default rates when penetration increases?

Experimental Design:
    - Fixed: Penetration increase to MAJOR level (40->60% wind, 11->30% flood)
    - Varying: Capital scaling multiplier [1.0x, 1.2x, 1.4x, 1.6x, 1.8x, 2.0x, 2.5x, 3.0x]
    - Target: Find multiplier where defaults match baseline
    
    Capital scaling multiplier = additional factor beyond proportional scaling
    Example: 1.5x means capital scales 1.5× faster than exposure increase

Output:
    - MC results for each capital multiplier
    - Comparison to baseline (no penetration increase)
    
Usage:
    # Run all multipliers
    python run_penetration_capital_sensitivity.py --all
    
    # Run specific multiplier (for cluster array jobs)
    python run_penetration_capital_sensitivity.py --multiplier 1.5
    
    # Test with fewer iterations
    python run_penetration_capital_sensitivity.py --all --n_iter 50

Author: Simona Meiler
Date: 2026-01-23
"""

import sys
import argparse
from pathlib import Path
import time
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fl_risk_model.mc_run_events import run_stochastic_tc_monte_carlo
from fl_risk_model.loader import (
    load_fhcf_county_exposure,
    load_market_share,
    load_nfip_policy_coverage,
)
from fl_risk_model.capital import load_surplus_data_with_groups
from fl_risk_model.exposure import build_wind_exposures
from fl_risk_model.utils import make_xwalk_from_tiger
from fl_risk_model.scenarios.penetration import apply_penetration_increase_scenario
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

# Capital scaling multipliers to test
DEFAULT_MULTIPLIERS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]


def run_mc_with_capital_multiplier(
    capital_multiplier: float,
    n_years: int = 10000,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """
    Run stochastic Monte Carlo simulation with specific capital multiplier.
    
    Parameters
    ----------
    capital_multiplier : float
        Capital scaling multiplier (1.0 = proportional)
    Uses the existing run_stochastic_tc_monte_carlo() framework but with
    custom penetration scenario and capital adjustments.
    
    Parameters
    ----------
    capital_multiplier : float
        Capital scaling multiplier (1.0 = proportional)
    n_years : int
        Number of years for stochastic simulation
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    Path
        Results directory
    """
    if verbose:
        print("=" * 80)
        print(f"PENETRATION CAPITAL SENSITIVITY: Multiplier = {capital_multiplier:.2f}x")
        print("=" * 80)
        print()
    
    # Create custom policy scenario configuration with capital multiplier
    policy_scenario_config = {
        "type": "penetration",
        "params": {
            "scenario": "MAJOR",
            "surplus_adjustment": "proportional",
            "capital_multiplier": capital_multiplier,  # Pass capital multiplier directly
        }
    }
    
    # Run stochastic MC with penetration scenario
    if verbose:
        print(f"Running {n_years}-year stochastic simulation...")
        print(f"Policy scenario: Penetration MAJOR")
        print(f"Capital multiplier: {capital_multiplier:.2f}x")
        print()
    
    # Create unique run label with capital multiplier
    # Format: penetration_capital_m{multiplier}
    run_label = f"penetration_capital_m{capital_multiplier:.2f}"
    
    result_dir = run_stochastic_tc_monte_carlo(
        year_sets_csv=None,  # Use config default
        n_years=n_years,
        seed=seed,
        policy_scenario_config=policy_scenario_config,
        out_dir=Path("results/mc_runs"),
        run_label=run_label,
    )
    
    # Add our metadata to the results
    metadata_file = result_dir / "capital_sensitivity_metadata.json"
    metadata = {
        "analysis": "penetration_capital_sensitivity",
        "capital_multiplier": capital_multiplier,
        "n_years": n_years,
        "seed": seed,
        "policy_scenario": "penetration_major",
    }
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print()
        print("=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {result_dir}")
        print("=" * 80)
    
    return result_dir
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Penetration capital sensitivity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--multiplier",
        type=float,
        help="Single capital multiplier to run (for cluster array jobs)",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all multipliers in sequence",
    )
    
    parser.add_argument(
        "--n_years",
        type=int,
        default=10000,
        help="Number of years for stochastic simulation (default: 10000)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=DEFAULT_MULTIPLIERS,
        help=f"Custom list of multipliers (default: {DEFAULT_MULTIPLIERS})",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    if args.multiplier:
        # Single multiplier mode (for cluster array jobs)
        run_mc_with_capital_multiplier(
            capital_multiplier=args.multiplier,
            n_years=args.n_years,
            seed=args.seed,
            verbose=True,
        )
    elif args.all:
        # Run all multipliers in sequence
        print("=" * 80)
        print("PENETRATION CAPITAL SENSITIVITY: FULL SWEEP")
        print("=" * 80)
        print(f"Multipliers: {args.multipliers}")
        print(f"Years per multiplier: {args.n_years:,}")
        print(f"Total simulation years: {len(args.multipliers) * args.n_years:,}")
        print("=" * 80)
        print()
        
        for multiplier in args.multipliers:
            run_mc_with_capital_multiplier(
                capital_multiplier=multiplier,
                n_years=args.n_years,
                seed=args.seed,
                verbose=True,
            )
            print()
    else:
        print("Error: Must specify either --multiplier <value> or --all")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
