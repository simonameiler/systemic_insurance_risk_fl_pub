#!/usr/bin/env python3
"""
run_insured_fraction_sensitivity.py - Sensitivity analysis for insured fraction

Sweeps the insured fraction from 0.1 to 0.5 (default: 0.40 from Beta(4,6))
and replays all 10,000 synthetic seasons for each value.  This tests how
sensitive systemic-risk metrics are to the assumed share of wind damage
that falls on insured vs. uninsured/underinsured households.

The underinsured share of the household portion is held at 0.30 (mean of
Beta(3,7)) throughout, so only the insured <-> uninsured split changes.

Usage
-----
    # Run all fractions (0.1-0.5 in steps of 0.1)
    python run_insured_fraction_sensitivity.py

    # Run specific fractions
    python run_insured_fraction_sensitivity.py --fractions 0.1 0.3 0.5

    # Limit to fewer seasons for a local test
    python run_insured_fraction_sensitivity.py --n_years 500 --fractions 0.3 0.4

    # Analyze previously saved results
    python run_insured_fraction_sensitivity.py --mode analyze \
        --results_dir results/mc_runs/insured_frac_sensitivity_*
"""

import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from fl_risk_model import config as cfg
from fl_risk_model.mc_run_events import (
    run_one_iteration,
    _prepare_common_inputs,
    DO_FLOOD,
    RNG_SEED,
)

YEAR = int(getattr(cfg, "MARKET_SHARE_YEAR", 2024))

METRICS = [
    "total_damage_usd",
    "wind_total_usd",
    "water_total_usd",
    "defaults_post",
    "figa_residual_deficit_usd",
    "citizens_residual_deficit_usd",
    "nfip_borrowed_usd",
    "catbond_payout_usd",
    "group_contrib_total_usd",
]

DEFAULT_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]


def _load_year_sets(path=None):
    if path is None:
        path = Path(getattr(cfg, "SYNTHETIC_YEAR_SETS_CSV"))
    return pd.read_csv(path)


def _run_year(year_id, event_stems, rng, common_inputs, group_threshold=10.0):
    try:
        rec = run_one_iteration(
            scenario_name=f"year_{year_id}",
            event_stems=event_stems,
            rng=rng,
            common_inputs=common_inputs,
            do_flood=DO_FLOOD,
            surplus_year=YEAR,
            policy_scenario_config=None,
            group_support_eligibility_threshold=group_threshold,
        )
        rec["year_id"] = year_id
        return rec
    except Exception as e:
        print(f"  [ERROR] year {year_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
def run_sweep(fractions: list[float],
              seed: int = RNG_SEED,
              out_dir: Path = Path("results/mc_runs"),
              n_years: int | None = None) -> Path:
    """Run 10K seasons for each insured-fraction value."""
    import fl_risk_model.mc_run_events as _mce
    _mce._USE_STOCHASTIC_EVENTS = True

    year_sets = _load_year_sets()
    total_years = year_sets["year_id"].nunique()
    if n_years is not None:
        total_years = min(total_years, n_years)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"insured_frac_sensitivity_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "run_config.json", "w") as f:
        json.dump({
            "mode": "insured_fraction_sensitivity",
            "fractions": fractions,
            "n_years": total_years,
            "seed": seed,
        }, f, indent=2)

    common_inputs = _prepare_common_inputs()

    for frac in fractions:
        print(f"\n{'='*60}")
        print(f"  Insured fraction = {frac:.2f}")
        print(f"{'='*60}")

        # Set the global override
        cfg.FIXED_INSURED_FRAC = frac

        rng = np.random.default_rng(seed)  # same seed per fraction for fair comparison
        rows = []

        for year_id in range(1, total_years + 1):
            ydf = year_sets[year_sets["year_id"] == year_id]
            if ydf["event_id"].isna().all():
                rows.append({
                    "year_id": year_id,
                    "insured_frac": frac,
                    "scenario": "zero_events",
                    **{m: 0.0 for m in METRICS},
                })
                continue

            event_stems = ydf["event_id"].dropna().tolist()
            rec = _run_year(year_id, event_stems, rng, common_inputs)
            if rec is not None:
                rec["insured_frac"] = frac
                rows.append(rec)

            if year_id % 1000 == 0:
                print(f"  [{frac:.2f}] {year_id:,}/{total_years:,}")

        df_frac = pd.DataFrame(rows)
        frac_csv = run_dir / f"iterations_frac_{frac:.2f}.csv"
        df_frac.to_csv(frac_csv, index=False)
        print(f"  Saved -> {frac_csv}  ({len(df_frac)} rows)")

    # Reset
    cfg.FIXED_INSURED_FRAC = None
    _mce._USE_STOCHASTIC_EVENTS = False

    print(f"\n  All results -> {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(results_dir: Path):
    """Compare metrics across insured-fraction values."""
    with open(results_dir / "run_config.json") as f:
        rc = json.load(f)
    fractions = rc["fractions"]

    print("=" * 80)
    print("INSURED FRACTION SENSITIVITY - RESULTS")
    print("=" * 80)
    print(f"  Source: {results_dir}")
    print(f"  Fractions: {fractions}")
    print()

    # Load all fraction CSVs
    dfs = {}
    for frac in fractions:
        csv_path = results_dir / f"iterations_frac_{frac:.2f}.csv"
        if csv_path.exists():
            dfs[frac] = pd.read_csv(csv_path)
        else:
            print(f"  [WARN] Missing: {csv_path}")

    if not dfs:
        print("  No data found!")
        return

    # ── Summary statistics ──
    ref_frac = 0.4 if 0.4 in dfs else fractions[0]
    print(f"  Reference fraction: {ref_frac:.2f}")
    print()

    header = f"  {'Metric':<40s}"
    for frac in sorted(dfs.keys()):
        header += f"  frac={frac:.1f}"
    print(header)
    print("  " + "-" * (40 + 12 * len(dfs)))

    summary_rows = []
    for m in METRICS:
        row_str = f"  {m:<40s}"
        row_data = {"metric": m}
        for frac in sorted(dfs.keys()):
            val = dfs[frac][m].mean()
            row_data[f"mean_{frac:.2f}"] = val
            row_data[f"std_{frac:.2f}"] = dfs[frac][m].std()
            if val >= 1e9:
                row_str += f"  {val/1e9:>8.2f}B"
            elif val >= 1e6:
                row_str += f"  {val/1e6:>8.2f}M"
            elif val >= 1:
                row_str += f"  {val:>9.1f}"
            else:
                row_str += f"  {val:>9.4f}"
        summary_rows.append(row_data)
        print(row_str)

    # ── Relative changes vs reference ──
    if ref_frac in dfs:
        print(f"\n  Relative change vs frac={ref_frac:.2f} (mean):")
        print(f"  {'Metric':<40s}", end="")
        for frac in sorted(dfs.keys()):
            if frac != ref_frac:
                print(f"  frac={frac:.1f}", end="")
        print()
        print("  " + "-" * (40 + 12 * (len(dfs) - 1)))

        for m in METRICS:
            ref_mean = dfs[ref_frac][m].mean()
            row_str = f"  {m:<40s}"
            for frac in sorted(dfs.keys()):
                if frac == ref_frac:
                    continue
                alt_mean = dfs[frac][m].mean()
                if ref_mean != 0:
                    pct = (alt_mean - ref_mean) / abs(ref_mean) * 100
                    row_str += f"  {pct:>+8.1f}%"
                else:
                    row_str += f"  {'n/a':>9s}"
            print(row_str)

    # ── Variance comparison ──
    if ref_frac in dfs:
        print(f"\n  Variance ratio vs frac={ref_frac:.2f}:")
        print(f"  {'Metric':<40s}", end="")
        for frac in sorted(dfs.keys()):
            if frac != ref_frac:
                print(f"  frac={frac:.1f}", end="")
        print()
        print("  " + "-" * (40 + 12 * (len(dfs) - 1)))

        for m in METRICS:
            ref_var = dfs[ref_frac][m].var()
            row_str = f"  {m:<40s}"
            for frac in sorted(dfs.keys()):
                if frac == ref_frac:
                    continue
                alt_var = dfs[frac][m].var()
                if ref_var > 0:
                    ratio = alt_var / ref_var
                    row_str += f"  {ratio:>9.4f}"
                else:
                    row_str += f"  {'n/a':>9s}"
            print(row_str)

    # Save combined summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary -> {summary_path}")

    # ── Tail comparison (P99) ──
    print(f"\n  99th percentile comparison:")
    print(f"  {'Metric':<40s}", end="")
    for frac in sorted(dfs.keys()):
        print(f"  frac={frac:.1f}", end="")
    print()
    print("  " + "-" * (40 + 12 * len(dfs)))

    for m in METRICS:
        row_str = f"  {m:<40s}"
        for frac in sorted(dfs.keys()):
            val = dfs[frac][m].quantile(0.99)
            if val >= 1e9:
                row_str += f"  {val/1e9:>8.2f}B"
            elif val >= 1e6:
                row_str += f"  {val/1e6:>8.2f}M"
            elif val >= 1:
                row_str += f"  {val:>9.1f}"
            else:
                row_str += f"  {val:>9.4f}"
        print(row_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Insured fraction sensitivity analysis")
    parser.add_argument("--mode", default="run",
                        choices=["run", "analyze"],
                        help="'run' to sweep fractions, 'analyze' to compare saved results")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=DEFAULT_FRACTIONS,
                        help="Insured fractions to test (default: 0.1 0.2 0.3 0.4 0.5)")
    parser.add_argument("--n_years", type=int, default=None,
                        help="Limit number of seasons (default: all 10,000)")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results dir (for --mode analyze)")

    args = parser.parse_args()

    if args.mode == "analyze":
        if args.results_dir is None:
            parser.error("--results_dir required for --mode analyze")
        analyze(Path(args.results_dir))
    else:
        run_dir = run_sweep(
            fractions=args.fractions,
            seed=args.seed,
            n_years=args.n_years,
        )
        print(f"\n  To analyze:\n    python {__file__} --mode analyze --results_dir {run_dir}")


if __name__ == "__main__":
    main()
