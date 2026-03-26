#!/usr/bin/env python3
"""
run_climate_buildingcode_sensitivity_windfloods.py - Climate change and building code sensitivity

Research Question: How do building codes interact with climate change across
different wind and flood reduction levels?

Scientific Hypothesis:
H0: Building code effectiveness (% loss reduction) remains constant regardless 
    of climate change intensity
H1: Building code effectiveness degrades as climate intensifies (because codes 
    designed for current climate become less effective)

Experimental Design:
- Climate scenarios: Baseline (ERA5), Near-term (2041-2060), Mid-term (2081-2100)
- SSP pathway: SSP2-4.5 (moderate emissions)
- Building code levels (13):
    Wind:   0%,  20%,  30%,  40%,  50%,  60%,  70%,  80%,  90%, 100%, 100%, 100%, 100%
    Flood:  0%,  13%,  20%,  27%,  33%,  40%,  47%,  53%,  60%,  67%,  70%,  80%,  90%
    Ratio:  -    3:2   3:2   3:2   3:2   3:2   3:2   3:2   3:2   3:2    -     -     -

Key anchor points:
- 0/0: Pure climate effect (no codes)
- 30/20: Current strong codes (close to FL building codes)
- 60/40: Aggressive future policy horizon
- 100/67: Maximum wind reduction at 3:2 ratio
- 100/90: Maximum feasible reductions for both wind and flood

Total runs for full analysis: 13 levels × 3 periods = 39 runs (SSP2-4.5)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fl_risk_model.mc_run_events import run_stochastic_tc_monte_carlo
from fl_risk_model.scenarios import building_codes
import json
import argparse


# Building code sensitivity levels (3:2 ratio up to 100% wind, then flood only)
BUILDING_CODE_LEVELS = [
    {"wind": 0.0,  "flood": 0.0,  "label": "00"},   # No codes
    {"wind": 0.20, "flood": 0.13, "label": "20"},  # Low codes
    {"wind": 0.30, "flood": 0.20, "label": "30"},  # Current strong (MAJOR)
    {"wind": 0.40, "flood": 0.27, "label": "40"},  # Enhanced codes
    {"wind": 0.50, "flood": 0.33, "label": "50"},  # Very strong codes
    {"wind": 0.60, "flood": 0.40, "label": "60"},  # Aggressive codes
    {"wind": 0.70, "flood": 0.47, "label": "70"},  # Very aggressive codes
    {"wind": 0.80, "flood": 0.53, "label": "80"},  # Near-maximum codes
    {"wind": 0.90, "flood": 0.60, "label": "90"},  # Near-maximum codes
    {"wind": 1.00, "flood": 0.67, "label": "100"}, # Maximum wind reduction (3:2 ratio)
    {"wind": 1.00, "flood": 0.70, "label": "110"}, # Max wind + higher flood
    {"wind": 1.00, "flood": 0.80, "label": "120"}, # Max wind + very high flood
    {"wind": 1.00, "flood": 0.90, "label": "130"}, # Max wind + maximum flood
]


def run_climate_buildingcode_sensitivity(
    code_level_idx: int,
    event_set: str = None,
    impact_dir: str = None,
    climate_scenario: str = "era5",
    seed: int = 42
):
    """
    Run Monte Carlo simulation with climate change + building codes.
    
    Parameters
    ----------
    code_level_idx : int
        Index into BUILDING_CODE_LEVELS (0-12)
    event_set : str, optional
        GCM event set name (e.g., 'FL_canesm_ssp245cal'). If provided, uses 
        direct GCM impacts instead of climate scaling.
    impact_dir : str, optional
        Path to impact directory for GCM event set. Required if event_set is provided.
    climate_scenario : str, optional
        Climate scenario for damage scaling approach: 'era5', 'ssp245_2050', 'ssp245_2100'.
        Ignored if event_set is provided.
    seed : int
        Random seed for reproducibility
    """
    
    if not (0 <= code_level_idx < len(BUILDING_CODE_LEVELS)):
        raise ValueError(f"code_level_idx must be 0-{len(BUILDING_CODE_LEVELS)-1}")
    
    # Get building code parameters
    code_level = BUILDING_CODE_LEVELS[code_level_idx]
    wind_reduction = code_level["wind"]
    flood_reduction = code_level["flood"]
    label = code_level["label"]
    
    # Create custom building code scenario
    scenario_name = f"SENS_W{int(wind_reduction*100):02d}_F{int(flood_reduction*100):02d}"
    building_codes.BUILDING_CODE_PRESETS[scenario_name] = {
        "wind_loss_reduction": wind_reduction,
        "flood_loss_reduction": flood_reduction,
        "retrofit_rate": 0.5,  # 50% retrofit rate (consistent with MAJOR)
        "timeline_years": 16,  # 16-year timeline (consistent with MAJOR)
        "description": f"Sensitivity: {int(wind_reduction*100)}% wind, {int(flood_reduction*100)}% flood",
        "references": "Climate-building code sensitivity analysis"
    }
    
    # Policy scenario configuration
    policy_scenario_config = {
        "type": "building_codes",
        "params": {
            "scenario": scenario_name,
        }
    }
    
    # Determine approach: GCM event sets vs climate scaling
    if event_set:
        # GCM event set approach
        from fl_risk_model import config as cfg
        
        # Extract GCM name from event_set (e.g., FL_canesm_ssp245cal -> canesm)
        gcm_name = event_set.split("_")[1]
        run_label = f"emanuel_{gcm_name}_ssp245cal_buildingcode_w{label}f{int(flood_reduction*100):02d}"
        
        # Setup paths for GCM event set
        year_sets_csv = Path(impact_dir) / f"year_sets_N10000_seed{seed}.csv"
        
        # Temporarily override config to use GCM event set
        original_event_dir = cfg.SYNTHETIC_EVENT_DIR
        original_metadata_csv = cfg.SYNTHETIC_EVENT_METADATA_CSV
        
        cfg.SYNTHETIC_EVENT_DIR = Path(impact_dir)
        cfg.SYNTHETIC_EVENT_METADATA_CSV = Path(impact_dir) / "event_metadata.csv"
        
        print("="*80)
        print("CLIMATE + BUILDING CODES SENSITIVITY (GCM EVENT SET)")
        print("="*80)
        print(f"GCM Event Set: {event_set}")
        print(f"Impact Directory: {impact_dir}")
        print(f"Building Code Wind Reduction: {int(wind_reduction*100)}%")
        print(f"Building Code Flood Reduction: {int(flood_reduction*100)}%")
        ratio_str = f"{wind_reduction/flood_reduction:.2f}" if flood_reduction > 0 else "N/A"
        print(f"Ratio: {ratio_str}")
        print(f"Run Label: {run_label}")
        print(f"Random Seed: {seed}")
        print("="*80)
        
        try:
            # Run Monte Carlo with GCM event set
            run_stochastic_tc_monte_carlo(
                year_sets_csv=year_sets_csv,
                n_years=10000,
                policy_scenario_config=policy_scenario_config,
                run_label=run_label,
                seed=seed
            )
        finally:
            # Restore original config
            cfg.SYNTHETIC_EVENT_DIR = original_event_dir
            cfg.SYNTHETIC_EVENT_METADATA_CSV = original_metadata_csv
        
        climate_info = {"event_set": event_set, "gcm": gcm_name, "ssp": "ssp245cal"}
        
    else:
        # Climate scaling approach
        
        # Parse climate scenario for run label
        if climate_scenario == "era5":
            run_label = f"era5_buildingcode_sens_w{label}f{int(flood_reduction*100):02d}"
            climate_ssp = None
            climate_period = None
        else:
            parts = climate_scenario.split("_")
            climate_ssp = parts[0]  # ssp245 or ssp585
            climate_period = parts[1]  # 2050 or 2100
            run_label = f"{climate_ssp}_{climate_period}_buildingcode_sens_w{label}f{int(flood_reduction*100):02d}"
        
        print("="*80)
        print("CLIMATE + BUILDING CODES SENSITIVITY (DAMAGE SCALING)")
        print("="*80)
        print(f"Climate Scenario: {climate_scenario}")
        print(f"Building Code Wind Reduction: {int(wind_reduction*100)}%")
        print(f"Building Code Flood Reduction: {int(flood_reduction*100)}%")
        ratio_str = f"{wind_reduction/flood_reduction:.2f}" if flood_reduction > 0 else "N/A"
        print(f"Ratio: {ratio_str}")
        print(f"Run Label: {run_label}")
        print(f"Random Seed: {seed}")
        print("="*80)
        
        # Run Monte Carlo with climate scaling
        run_stochastic_tc_monte_carlo(
            n_years=10000,
            policy_scenario_config=policy_scenario_config,
            run_label=run_label,
            seed=seed
        )
        
        climate_info = {"scenario": climate_scenario, "ssp": climate_ssp, "period": climate_period}
    
    # Save metadata for analysis
    import glob
    matching_dirs = sorted(glob.glob(f"results/mc_runs/{run_label}_*"))
    if matching_dirs:
        actual_dir = Path(matching_dirs[-1])
        metadata = {
            "wind_loss_reduction_pct": int(wind_reduction * 100),
            "flood_loss_reduction_pct": int(flood_reduction * 100),
            "wind_flood_ratio": wind_reduction / flood_reduction if flood_reduction > 0 else None,
            "policy_scenario": "building_codes",
            "code_level_index": code_level_idx,
            "seed": seed,
            "n_years": 10000,
            **climate_info  # Add climate-specific info
        }
        metadata_file = actual_dir / 'sensitivity_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n[OK] Saved metadata: {metadata_file}")
    
    print(f"\n[OK] Sensitivity run complete: Wind {int(wind_reduction*100)}% / Flood {int(flood_reduction*100)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run climate change + building codes sensitivity (wind/flood split)"
    )
    parser.add_argument(
        "--code_level",
        type=int,
        required=True,
        choices=range(len(BUILDING_CODE_LEVELS)),
        help="Building code level index (0-5). "
             "L0: 0%/0%, L1: 20%/13%, L2: 30%/20%, L3: 40%/27%, L4: 50%/33%, L5: 60%/40%"
    )
    parser.add_argument(
        "--event_set",
        type=str,
        default=None,
        help="GCM event set name (e.g., 'FL_canesm_ssp245cal'). If provided, uses GCM impacts directly."
    )
    parser.add_argument(
        "--impact_dir",
        type=str,
        default=None,
        help="Path to impact directory for GCM event set. Required if --event_set is provided."
    )
    parser.add_argument(
        "--climate",
        type=str,
        default="era5",
        choices=["era5", "ssp245_2050", "ssp245_2100"],
        help="Climate scenario for damage scaling approach (default: era5). Ignored if --event_set is provided."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate event_set and impact_dir combination
    if args.event_set and not args.impact_dir:
        parser.error("--impact_dir is required when --event_set is provided")
    
    run_climate_buildingcode_sensitivity(
        code_level_idx=args.code_level,
        event_set=args.event_set,
        impact_dir=args.impact_dir,
        climate_scenario=args.climate,
        seed=args.seed
    )
