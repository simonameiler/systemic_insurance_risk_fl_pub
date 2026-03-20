#!/usr/bin/env python3
"""
run_emanuel_policy_suite.py - Run multiple policy scenarios with Emanuel TC event sets

This script runs Monte Carlo analysis across multiple policy scenarios
using Emanuel TC events. Useful for policy comparison studies.

Usage:
    # Run 3 key policies with ERA5 baseline
    python run_emanuel_policy_suite.py --event_set FL_era5_reanalcal
    
    # Run all policies
    python run_emanuel_policy_suite.py --event_set FL_era5_reanalcal --all
    
    # Custom policy list
    python run_emanuel_policy_suite.py --event_set FL_era5_reanalcal \
        --policies baseline market_exit_moderate building_codes_major
    
    # Run on cluster with explicit impact directory
    python run_emanuel_policy_suite.py --event_set FL_era5_reanalcal \
        --impact_dir /path/to/impact/FL_era5_reanalcal \
        --out results/mc_runs \
        --all
"""

import sys
from pathlib import Path
import argparse
import time
import socket

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from fl_risk_model.mc_run_events import run_stochastic_tc_monte_carlo
from fl_risk_model import config as cfg


def main():
    ap = argparse.ArgumentParser(
        description="Run multiple policy scenarios with Emanuel TC events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ERA5 with 3 key policies (baseline, market_exit, building_codes)
  python run_emanuel_policy_suite.py --event_set FL_era5_reanalcal
  
  # ERA5 with all policies
  python run_emanuel_policy_suite.py --event_set FL_era5_reanalcal --all
  
  # Custom policy list
  python run_emanuel_policy_suite.py --event_set FL_era5_reanalcal \\
      --policies baseline market_exit_moderate penetration_major building_codes_major
"""
    )
    
    # Event set configuration
    ap.add_argument("--event_set", required=True,
                    help="Emanuel event set name (e.g., FL_era5_reanalcal)")
    ap.add_argument("--impact_dir", type=str, default=None,
                    help="Directory with Emanuel impacts (default: DATA_DIR/hazard/emanuel_impacts/<event_set>)")
    
    # Policy configuration
    ap.add_argument("--policies", nargs="+", default=None,
                    help="List of policy scenarios (default: baseline, market_exit_moderate, building_codes_major)")
    ap.add_argument("--all", action="store_true",
                    help="Run all available policy scenarios")
    
    # Simulation parameters
    ap.add_argument("--n_years", type=int, default=None,
                    help="Number of years to simulate (default: all 10,000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for wind/water split")
    
    # Output configuration
    ap.add_argument("--out", type=str, default="results/emanuel_policy_comparison",
                    help="Output directory root")
    
    args = ap.parse_args()
    
    # Configure paths
    if args.impact_dir:
        impact_dir = Path(args.impact_dir)
    else:
        # Detect if we're on Sherlock
        hostname = socket.gethostname()
        is_sherlock = 'sherlock' in hostname.lower() or hostname.startswith('sh')
        
        if is_sherlock:
            # On Sherlock - use scratch directory
            impact_dir = Path("/scratch/groups/bakerjw/smeiler/impacts") / args.event_set
        else:
            # Local - use DATA_DIR structure
            impact_dir = cfg.DATA_DIR / "hazard" / "emanuel_impacts" / args.event_set
    
    # Validate prerequisites
    year_sets_csv = impact_dir / "year_sets_N10000_seed42.csv"
    metadata_csv = impact_dir / "event_metadata.csv"
    
    if not year_sets_csv.exists():
        print(f"[ERROR] ERROR: Year-sets file not found: {year_sets_csv}")
        sys.exit(1)
    
    if not metadata_csv.exists():
        print(f"[ERROR] ERROR: Event metadata not found: {metadata_csv}")
        sys.exit(1)
    
    # Determine policy list
    if args.all:
        policy_list = list(cfg.POLICY_SCENARIOS.keys())
    elif args.policies:
        policy_list = args.policies
    else:
        # Default: 3 key policies for comparison
        policy_list = ["baseline", "market_exit_moderate", "building_codes_major"]
    
    # Validate policies
    for policy in policy_list:
        if policy != "baseline" and policy not in cfg.POLICY_SCENARIOS:
            print(f"[ERROR] ERROR: Unknown policy scenario '{policy}'")
            print(f"   Available: {list(cfg.POLICY_SCENARIOS.keys())}")
            sys.exit(1)
    
    # Configure paths for this event set
    original_event_dir = cfg.SYNTHETIC_EVENT_DIR
    original_metadata_csv = cfg.SYNTHETIC_EVENT_METADATA_CSV
    
    cfg.SYNTHETIC_EVENT_DIR = impact_dir
    cfg.SYNTHETIC_EVENT_METADATA_CSV = metadata_csv
    
    # Print configuration
    print("="*80)
    print("EMANUEL TC POLICY SUITE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Event set: {args.event_set}")
    print(f"  Impact directory: {impact_dir}")
    print(f"  Policies: {policy_list}")
    print(f"  N years: {args.n_years or 'all (10,000)'}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output: {args.out}")
    print()
    
    # Run each policy
    results = {}
    start_time = time.time()
    
    for i, policy in enumerate(policy_list, 1):
        print(f"\n{'='*80}")
        print(f"POLICY {i}/{len(policy_list)}: {policy}")
        print(f"{'='*80}\n")
        
        # Extract short name from event set
        event_short = args.event_set.replace("FL_", "").replace("_reanalcal", "")
        run_label = f"emanuel_{event_short}_{policy}"
        
        # Get policy config
        policy_config = None if policy == "baseline" else cfg.POLICY_SCENARIOS[policy]
        
        try:
            out_dir = run_stochastic_tc_monte_carlo(
                year_sets_csv=year_sets_csv,
                n_years=args.n_years,
                seed=args.seed,
                out_dir=Path(args.out),
                run_label=run_label,
                climate_scaling=False,
                policy_scenario_config=policy_config,
            )
            results[policy] = out_dir
            print(f"[OK] {policy} complete -> {out_dir}")
            
        except Exception as e:
            print(f"[ERROR] {policy} FAILED: {e}")
            results[policy] = None
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("POLICY SUITE COMPLETE")
    print("="*80)
    print(f"\nTotal time: {elapsed/3600:.1f} hours")
    print(f"\nResults:")
    for policy, path in results.items():
        if path:
            print(f"  [OK] {policy}: {path}")
        else:
            print(f"  [ERROR] {policy}: FAILED")
    print()
    
    # Restore original config
    cfg.SYNTHETIC_EVENT_DIR = original_event_dir
    cfg.SYNTHETIC_EVENT_METADATA_CSV = original_metadata_csv


if __name__ == "__main__":
    main()
