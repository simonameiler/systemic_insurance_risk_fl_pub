#!/usr/bin/env python3
"""
run_historical_scenarios_mc.py - Run Monte Carlo simulations for historical hurricanes

This script runs 200 MC iterations for each of the 8 historical scenarios:
- great_miami
- andrew
- andrew_then_gm
- gm_then_andrew
- double_gm
- lake_okeechobee
- irma
- double_irma

After completing all MC runs, it builds an Excel report with uncertainty 
quantification (mean, std dev, 5th/95th percentiles for all metrics).

Usage:
    python scripts/run/run_historical_scenarios_mc.py
    python scripts/run/run_historical_scenarios_mc.py --n_iter 500 --seed 99
    python scripts/run/run_historical_scenarios_mc.py --scenario great_miami  # Run just one
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fl_risk_model import config as cfg
from fl_risk_model.mc_run_events import (
    run_one_iteration,
    _prepare_common_inputs,
    _preflight_events,
    _compute_summary,
    SCENARIOS,
    DO_FLOOD,
    YEAR,
)
import numpy as np
import pandas as pd
import traceback


def run_scenario_mc(scenario_name, n_iter=200, seed=42, out_dir=None):
    """
    Run Monte Carlo for a single historical scenario.
    
    Parameters:
    -----------
    scenario_name : str
        Scenario name (must be in SCENARIOS dict)
    n_iter : int
        Number of iterations
    seed : int
        Random seed
    out_dir : Path
        Output directory root
        
    Returns:
    --------
    Path : Directory containing iterations.csv
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")
    
    if out_dir is None:
        out_dir = Path(cfg.MC_OUT_DIR) if hasattr(cfg, "MC_OUT_DIR") else Path("results/mc_runs")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create scenario-specific directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"{scenario_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Running Monte Carlo: {scenario_name}")
    print(f"  Iterations: {n_iter}")
    print(f"  Seed: {seed}")
    print(f"  Output: {run_dir}")
    print(f"{'='*80}\n")
    
    # Setup
    rng = np.random.default_rng(seed)
    common_inputs = _prepare_common_inputs()
    _preflight_events()
    
    event_stems = SCENARIOS[scenario_name]
    
    rows = []
    errors = []
    
    for i in range(1, n_iter + 1):
        try:
            rec = run_one_iteration(
                scenario_name=scenario_name,
                event_stems=event_stems,
                rng=rng,
                common_inputs=common_inputs,
                do_flood=DO_FLOOD,
                surplus_year=YEAR,
            )
            rec["iter"] = i
            rows.append(rec)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[iter {i}] ERROR:\n{tb}")
            rows.append({
                "iter": i,
                "scenario": scenario_name,
                "events": ",".join(event_stems),
                "error": str(e)
            })
            errors.append((i, tb))
        
        if i % 25 == 0:
            print(f"  ... {i}/{n_iter} iterations complete")
    
    # Save results
    df = pd.DataFrame(rows)
    iterations_file = run_dir / "iterations.csv"
    df.to_csv(iterations_file, index=False)
    
    if errors:
        with open(run_dir / "errors_summary.txt", "w") as fh:
            for i_, tb in errors:
                fh.write(f"iter={i_}\n{tb}\n---\n")
    
    # Compute summary (excluding errors)
    df_clean = df.copy()
    if "error" in df_clean.columns:
        n_err = int(df_clean["error"].notna().sum())
        if n_err > 0:
            print(f"  Skipping {n_err} errored iteration(s)")
            df_clean = df_clean[df_clean["error"].isna()].copy()
    
    summary = _compute_summary(df_clean)
    summary.to_csv(run_dir / "summary.csv", index=False)
    
    print(f"\n[OK] Completed: {scenario_name} -> {run_dir}")
    print(f"   Iterations: {iterations_file}")
    print(f"   Summary: {run_dir / 'summary.csv'}")
    
    return run_dir


def build_uncertainty_report(mc_dirs, out_file=None):
    """
    Build Excel report from multiple MC run directories.
    
    Parameters:
    -----------
    mc_dirs : list of Path
        Directories containing iterations.csv files
    out_file : Path, optional
        Output Excel file path. If None, derives from first directory.
    """
    # Import the report builder
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
    from build_scenario_report_with_uncertainty import build_report_with_uncertainty
    
    # Combine all iterations into one CSV
    all_iterations = []
    for mc_dir in mc_dirs:
        iterations_csv = mc_dir / "iterations.csv"
        if not iterations_csv.exists():
            print(f"[WARNING] Warning: {iterations_csv} not found, skipping")
            continue
        df = pd.read_csv(iterations_csv)
        all_iterations.append(df)
    
    if not all_iterations:
        print("[ERROR] No iterations found!")
        return None
    
    combined_df = pd.concat(all_iterations, ignore_index=True)
    
    # Save combined iterations
    if out_file is None:
        parent_dir = mc_dirs[0].parent
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = parent_dir / f"scenario_report_with_uncertainty_{ts}.xlsx"
    else:
        out_file = Path(out_file)
    
    combined_csv = out_file.parent / f"{out_file.stem}_iterations.csv"
    combined_df.to_csv(combined_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"Building uncertainty report from {len(combined_df)} total iterations")
    print(f"  Combined iterations: {combined_csv}")
    print(f"  Output Excel: {out_file}")
    print(f"{'='*80}\n")
    
    # Build the report
    fhcf_cap = getattr(cfg, "FHCF_SEASON_CAP", 17e9)
    nfip_premium = getattr(cfg, "NFIP_FL_PREMIUM_BASE", 930_000_000)
    
    build_report_with_uncertainty(
        iterations_csv=combined_csv,
        out_xlsx=out_file,
        fhcf_cap=fhcf_cap,
        nfip_fl_premium_base=nfip_premium,
    )
    
    print(f"\n[OK] Report complete: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo for historical hurricane scenarios"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run only this scenario (default: run all 8 scenarios)"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=200,
        help="Number of MC iterations per scenario (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: results/mc_runs)"
    )
    parser.add_argument(
        "--skip_report",
        action="store_true",
        help="Skip building Excel report after MC runs"
    )
    
    args = parser.parse_args()
    
    # Determine which scenarios to run
    if args.scenario:
        scenarios_to_run = [args.scenario]
    else:
        scenarios_to_run = [
            "great_miami",
            "andrew",
            "andrew_then_gm",
            "gm_then_andrew",
            "double_gm",
            "lake_okeechobee",
            "irma",
            "double_irma",
        ]
    
    # Validate all scenarios exist
    for scen in scenarios_to_run:
        if scen not in SCENARIOS:
            print(f"[ERROR] Unknown scenario: {scen}")
            print(f"   Available: {list(SCENARIOS.keys())}")
            return 1
    
    # Run MC for each scenario
    mc_dirs = []
    start_time = time.time()
    
    for scen in scenarios_to_run:
        try:
            run_dir = run_scenario_mc(
                scenario_name=scen,
                n_iter=args.n_iter,
                seed=args.seed,
                out_dir=args.out,
            )
            mc_dirs.append(run_dir)
        except Exception as e:
            print(f"\n[ERROR] Failed to run {scen}:")
            print(f"   {e}")
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"All scenarios complete in {elapsed/60:.1f} minutes")
    print(f"Results directories:")
    for mc_dir in mc_dirs:
        print(f"  - {mc_dir}")
    print(f"{'='*80}\n")
    
    # Build combined uncertainty report
    if not args.skip_report and mc_dirs:
        try:
            report_file = build_uncertainty_report(mc_dirs)
        except Exception as e:
            print(f"\n[ERROR] Failed to build report:")
            print(f"   {e}")
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
