#!/usr/bin/env python
"""
Run Monte Carlo analysis with Emanuel TC event sets.

This script runs the insurance risk model using precomputed Emanuel TC impacts
and generated year-sets. It follows the same stochastic TC workflow as 
run_stochastic_policies.py but uses Emanuel events instead of synthetic events.

Usage:
    # Run ERA5 baseline (historical climate)
    python run_emanuel_monte_carlo.py --event_set FL_era5_reanalcal
    
    # Run with climate change scenario
    python run_emanuel_monte_carlo.py --event_set FL_canesm5_ssp585_2081-2100
    
    # Run subset of years for testing
    python run_emanuel_monte_carlo.py --event_set FL_era5_reanalcal --n_years 1000
    
    # Run with specific policy scenario
    python run_emanuel_monte_carlo.py --event_set FL_era5_reanalcal --policy building_codes_major
    
    # Run on cluster with explicit impact directory
    python run_emanuel_monte_carlo.py --event_set FL_era5_reanalcal \
        --impact_dir /path/to/impact/FL_era5_reanalcal
"""

import sys
from pathlib import Path
import argparse
import socket

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from fl_risk_model.mc_run_events import run_stochastic_tc_monte_carlo
from fl_risk_model import config as cfg


def main():
    ap = argparse.ArgumentParser(
        description="Run Monte Carlo with Emanuel TC events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ERA5 baseline
  python run_emanuel_monte_carlo.py --event_set FL_era5_reanalcal
  
  # CanESM5 future climate
  python run_emanuel_monte_carlo.py --event_set FL_canesm5_ssp585_2081-2100
  
  # ERA5 with building codes policy
  python run_emanuel_monte_carlo.py --event_set FL_era5_reanalcal \\
      --policy building_codes_major
"""
    )
    
    # Event set configuration
    ap.add_argument("--event_set", required=True,
                    help="Emanuel event set name (e.g., FL_era5_reanalcal, FL_canesm5_ssp585_2081-2100)")
    ap.add_argument("--impact_dir", type=str, default=None,
                    help="Directory with Emanuel impacts (default: DATA_DIR/hazard/emanuel_impacts/<event_set>)")
    ap.add_argument("--year_sets_file", type=str, default=None,
                    help="Year-sets filename (default: year_sets_N10000_seed42.csv, e.g., year_sets_N10000_seed42_19952014.csv)")
    
    # Simulation parameters
    ap.add_argument("--n_years", type=int, default=None,
                    help="Number of years to simulate (default: all 10,000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for wind/water split")
    
    # Policy scenario
    ap.add_argument("--policy", type=str, default=None,
                    help="Policy scenario to apply (default: None = baseline)")
    ap.add_argument("--wind_loss_reduction", type=float, default=None,
                    help="Override wind loss reduction for building codes (0.0-1.0, e.g., 0.15 = 15%%)")
    ap.add_argument("--flood_loss_reduction", type=float, default=None,
                    help="Override flood loss reduction for building codes (0.0-1.0, e.g., 0.10 = 10%%)")
    
    # Climate scaling and wind/water attribution
    ap.add_argument("--climate_scaling", action="store_true",
                    help="Apply climate change damage scaling (OBSOLETE - not used for Emanuel runs)")
    ap.add_argument("--future_wind_share", action="store_true",
                    help="Use future climate wind/water attribution (77%% wind vs 84.6%% present, Gori et al. SSP245 2081-2100)")
    
    # Output configuration
    ap.add_argument("--out", type=str, default="results/emanuel_mc_runs",
                    help="Output directory root")
    ap.add_argument("--run_label", type=str, default=None,
                    help="Custom label for output directory")
    
    args = ap.parse_args()
    
    # Configure paths
    if args.impact_dir:
        impact_dir = Path(args.impact_dir)
    else:
        # Detect if we're on cluster (Stanford Sherlock HPC)
        hostname = socket.gethostname()
        is_sherlock = 'sherlock' in hostname.lower() or hostname.startswith('sh')
        
        if is_sherlock:
            # On Sherlock - use research group storage
            # MODIFY THIS PATH for your cluster setup
            impact_dir = Path("/home/groups/bakerjw/smeiler/climada_data/data/impact") / args.event_set
        else:
            # Local - use DATA_DIR structure
            impact_dir = cfg.DATA_DIR / "hazard" / "emanuel_impacts" / args.event_set
    
    # Check for year-sets file
    if args.year_sets_file:
        year_sets_csv = impact_dir / args.year_sets_file
    else:
        year_sets_csv = impact_dir / "year_sets_N10000_seed42.csv"
    
    if not year_sets_csv.exists():
        print(f"❌ ERROR: Year-sets file not found: {year_sets_csv}")
        print(f"   Generate year-sets first:")
        if args.year_sets_file:
            print(f"   python scripts/hazard/generate_emanuel_year_sets_subset.py --event_set {args.event_set} --year_start 1995 --year_end 2014")
        else:
            print(f"   python scripts/hazard/generate_emanuel_year_sets.py --event_set {args.event_set}")
        sys.exit(1)
    
    # Check for event metadata
    metadata_csv = impact_dir / "event_metadata.csv"
    if not metadata_csv.exists():
        print(f"❌ ERROR: Event metadata not found: {metadata_csv}")
        print(f"   Precompute impacts first:")
        print(f"   python scripts/hazard/precompute_emanuel_tc_impacts.py --event_set {args.event_set}")
        sys.exit(1)
    
    # Configure SYNTHETIC_EVENT_DIR and METADATA for this event set
    # The mc_run_events code will read from these locations
    original_event_dir = cfg.SYNTHETIC_EVENT_DIR
    original_metadata_csv = cfg.SYNTHETIC_EVENT_METADATA_CSV
    
    cfg.SYNTHETIC_EVENT_DIR = impact_dir
    cfg.SYNTHETIC_EVENT_METADATA_CSV = metadata_csv
    
    # Get policy scenario config
    policy_scenario_config = None
    if args.policy:
        if args.policy not in cfg.POLICY_SCENARIOS:
            print(f"❌ ERROR: Unknown policy scenario '{args.policy}'")
            print(f"   Available: {list(cfg.POLICY_SCENARIOS.keys())}")
            sys.exit(1)
        policy_scenario_config = cfg.POLICY_SCENARIOS[args.policy]
        print(f"🎯 Applying policy scenario: {args.policy}")
        
        # Override loss reduction if specified
        if args.wind_loss_reduction is not None or args.flood_loss_reduction is not None:
            if policy_scenario_config.get("type") == "building_codes":
                policy_scenario_config = dict(policy_scenario_config)  # Make a copy
                policy_scenario_config["params"] = dict(policy_scenario_config["params"])
                if args.wind_loss_reduction is not None:
                    policy_scenario_config["params"]["wind_loss_reduction"] = args.wind_loss_reduction
                    print(f"   Overriding wind loss reduction: {args.wind_loss_reduction:.1%}")
                if args.flood_loss_reduction is not None:
                    policy_scenario_config["params"]["flood_loss_reduction"] = args.flood_loss_reduction
                    print(f"   Overriding flood loss reduction: {args.flood_loss_reduction:.1%}")
            else:
                print(f"⚠️  WARNING: --wind/flood_loss_reduction only applies to building_codes policies (ignored)")
    else:
        print(f"🎯 Running baseline (no policy changes)")
    
    # Create run label
    if args.run_label:
        run_label = args.run_label
    else:
        # Extract short name from event set
        # FL_era5_reanalcal -> era5
        # FL_canesm5_ssp585_2081-2100 -> canesm5_ssp585_2081-2100
        event_short = args.event_set.replace("FL_", "").replace("_reanalcal", "")
        
        if args.policy:
            run_label = f"emanuel_{event_short}_{args.policy}"
        else:
            run_label = f"emanuel_{event_short}_baseline"
        
        # Add year range suffix if using subset year sets
        if args.year_sets_file and "19952014" in args.year_sets_file:
            run_label += "_19952014"
        
        if args.climate_scaling:
            run_label += "_climate2050"
        if args.future_wind_share:
            run_label += "_future_wind"
    
    # Print configuration
    print("="*80)
    print("EMANUEL TC MONTE CARLO")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Event set: {args.event_set}")
    print(f"  Impact directory: {impact_dir}")
    print(f"  Year-sets: {year_sets_csv}")
    print(f"  N years: {args.n_years or 'all (10,000)'}")
    print(f"  Policy scenario: {args.policy or 'baseline'}")
    print(f"  Climate scaling: {args.climate_scaling}")
    print(f"  Future wind/water: {args.future_wind_share}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output: {args.out}/{run_label}_TIMESTAMP")
    print()
    
    # Run Monte Carlo
    try:
        out_dir = run_stochastic_tc_monte_carlo(
            year_sets_csv=year_sets_csv,
            n_years=args.n_years,
            seed=args.seed,
            out_dir=Path(args.out),
            run_label=run_label,
            climate_scaling=args.climate_scaling,
            future_wind_share=args.future_wind_share,
            policy_scenario_config=policy_scenario_config,
        )
        
        print(f"\n✅ Monte Carlo complete!")
        print(f"   Results: {out_dir}")
        print()
        
    finally:
        # Restore original config
        cfg.SYNTHETIC_EVENT_DIR = original_event_dir
        cfg.SYNTHETIC_EVENT_METADATA_CSV = original_metadata_csv


if __name__ == "__main__":
    main()
