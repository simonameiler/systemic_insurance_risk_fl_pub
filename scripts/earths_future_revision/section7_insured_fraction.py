#!/usr/bin/env python3
"""Section 7: insured wind fraction sensitivity, reprocessed with corrected
accounting.

Reuses the existing archived insured-fraction sweep
(results/mc_runs/insured_frac_sensitivity_combined/iterations_frac_*.csv,
fixed fractions 0.1-0.5, 10,000 seasons each, seed=42) because it remains
compatible with the corrected *accounting* (a post-processing correction of
already-simulated season totals -- see docs/earths_future_revision/
correction_register.md item C1). It is NOT rerun here because the underlying
per-season FHCF/FIGA/Citizens/NFIP component values were not themselves
changed by that correction, only how they are summed into "total public
burden" and how return levels are estimated from them (item R1, Section 5).

This script does NOT apply the still-undecided FHCF coverage-election
correction (item C4); if that formula changes, this sweep will need to be
rerun against the corrected fl_risk_model.fhcf.apply_fhcf_recovery, which is
not possible in this environment (see data_inventory.md).

For each fraction f in {0.1, 0.2, 0.3, 0.4, 0.5} reports mean values, the
corrected aggregate return-period table, the corrected amplification ratio,
and the corrected annual burden-exceedance probability. Elasticities are
computed by centered finite difference in log-log space around f=0.4 using
UNROUNDED means (f=0.3 and f=0.5), reported alongside the one-sided finite
differences already used for Table S6's "vs. f=0.4" deltas.

f=0.4 (this fixed-value run) is kept distinct from the baseline Beta(4,6)
run: the model is nonlinear in f, so E[y(f)] under a fixed f=0.4 need not
equal E[y(F)] for F ~ Beta(4,6) with mean 0.4 (Jensen's inequality). Both are
reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    OUT_DIR,
    RETURN_PERIODS,
    load_iterations,
    empirical_return_level,
    to_billion,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAC_DIR = REPO_ROOT / "results" / "mc_runs" / "insured_frac_sensitivity_combined"
DEFAULT_BASELINE = REPO_ROOT / "results" / "mc_runs" / "emanuel_era5_baseline_20260326_141913" / "iterations.csv"

METRICS = {
    "wind_insured_private_usd": "Wind insured, private",
    "wind_insured_citizens_usd": "Wind insured, Citizens",
    "wind_un_underinsured_usd": "Wind un/underinsured",
    "fhcf_shortfall_usd": "FHCF shortfall (diagnostic)",
    "figa_residual_deficit_usd": "FIGA residual deficit",
    "citizens_residual_deficit_usd": "Citizens residual deficit",
    "nfip_borrowed_usd": "NFIP Treasury borrowing",
    "public_burden_corrected_usd": "Residual financing requirement",
    "defaults_post": "Insurer defaults (count)",
    "largest_entity_deficit_usd": "Largest entity deficit",
}


def load_fraction_runs(frac_dir: Path, fractions) -> dict[float, pd.DataFrame]:
    out = {}
    for f in fractions:
        path = frac_dir / f"iterations_frac_{f:.2f}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        out[f] = load_iterations(path)
    return out


def means_table(runs: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for col, label in METRICS.items():
        row = {"Metric": label}
        for f, df in sorted(runs.items()):
            row[f"f={f}"] = float(df[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def elasticities(runs: dict[float, pd.DataFrame], f0: float = 0.4) -> pd.DataFrame:
    fs = sorted(runs.keys())
    if f0 not in fs:
        raise ValueError(f"reference fraction {f0} not in sweep {fs}")
    rows = []
    for col, label in METRICS.items():
        means = {f: float(runs[f][col].mean()) for f in fs}
        row = {"Metric": label}
        for f in fs:
            if f == f0:
                continue
            m0, mf = means[f0], means[f]
            row[f"delta_vs_f0.4_at_f={f}"] = (
                (mf - m0) / m0 if m0 != 0 else float("nan")
            )
        # Centered log-log finite difference using f=0.3 and f=0.5 around 0.4,
        # on UNROUNDED means (Section 7 requirement).
        if 0.3 in means and 0.5 in means and means[0.3] > 0 and means[0.5] > 0:
            eps_central = (
                (np.log(means[0.5]) - np.log(means[0.3])) / (np.log(0.5) - np.log(0.3))
            )
        else:
            eps_central = float("nan")
        row["elasticity_centered_loglog_0.3_0.5"] = eps_central
        rows.append(row)
    return pd.DataFrame(rows)


def return_period_by_fraction(runs: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for f, df in sorted(runs.items()):
        rl_loss = empirical_return_level(df["total_damage_usd"].to_numpy(dtype=float))
        rl_burden = empirical_return_level(df["public_burden_corrected_usd"].to_numpy(dtype=float))
        row = {
            "f": f,
            "loss_RP10_B": to_billion(rl_loss["RP10"]),
            "loss_RP100_B": to_billion(rl_loss["RP100"]),
            "loss_amp_100_over_10": rl_loss["RP100"] / rl_loss["RP10"] if rl_loss["RP10"] > 0 else np.nan,
            "burden_RP10_B": to_billion(rl_burden["RP10"]),
            "burden_RP100_B": to_billion(rl_burden["RP100"]),
            "burden_amp_100_over_10": rl_burden["RP100"] / rl_burden["RP10"] if rl_burden["RP10"] > 0 else np.nan,
            "P(burden>1%GDP)": float((df["public_burden_corrected_usd"] > 0.01 * 1.7e12).mean()),
            "mean_public_burden_corrected_B": to_billion(df["public_burden_corrected_usd"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def beta_4_6_vs_fixed_04(baseline_path: Path, runs: dict[float, pd.DataFrame]) -> dict:
    baseline = load_iterations(baseline_path)
    fixed04 = runs.get(0.4)
    out = {}
    for col, label in METRICS.items():
        if fixed04 is None:
            continue
        beta_mean = float(baseline[col].mean())
        fixed_mean = float(fixed04[col].mean())
        out[label] = {
            "Beta(4,6)_mean_run": beta_mean,
            "fixed_f=0.4_mean_run": fixed_mean,
            "relative_diff": (fixed_mean - beta_mean) / beta_mean if beta_mean != 0 else float("nan"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frac-dir", type=Path, default=DEFAULT_FRAC_DIR)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "insured_fraction")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_fraction_runs(args.frac_dir, args.fractions)

    means_table(runs).to_csv(args.out_dir / "means_by_fraction.csv", index=False)
    elasticities(runs).to_csv(args.out_dir / "elasticities.csv", index=False)
    return_period_by_fraction(runs).to_csv(args.out_dir / "return_periods_by_fraction.csv", index=False)

    comparison = beta_4_6_vs_fixed_04(args.baseline, runs)
    with open(args.out_dir / "beta46_vs_fixed04_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(means_table(runs).to_string(index=False))
    print()
    print(elasticities(runs).to_string(index=False))
    print()
    print(return_period_by_fraction(runs).to_string(index=False))
    print()
    print(json.dumps(comparison, indent=2))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
