#!/usr/bin/env python3
"""Section 5: corrected return-period table and amplification comparison.

Reproduces the submitted (buggy) Table 1 construction for provenance, then
recomputes it correctly: the total public burden return level at each return
period is the empirical quantile of the *season-level sum* of its components,
not the sum of the separately estimated component quantiles. Total loss and
each loss-decomposition row remain marginal quantiles, as in the submission
(this was not the bug -- see docs/earths_future_revision/correction_register.md
item R1).

Also recomputes:
  - the 10-year / 100-year amplification ratios for total loss and public
    burden (the "ninefold" / "fortyfold" claims),
  - the fitted power-law exponent (beta) of public burden vs. total loss
    across return periods,
  - the annual probability that public burden exceeds 1% / 10% of Florida GDP,
  - a 1,000-resample bootstrap (10th-90th percentile) for every reported
    return level, resampling whole season rows.

Usage
-----
    python scripts/earths_future_revision/section5_return_periods.py \\
        --iterations results/mc_runs/emanuel_era5_baseline_20260326_141913/iterations.csv \\
        --gdp 1.7e12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    LOSS_COLUMNS,
    INST_COLUMNS,
    LEGACY_BURDEN_COMPONENTS,
    CORRECTED_BURDEN_COMPONENTS,
    RETURN_PERIODS,
    OUT_DIR,
    load_iterations,
    empirical_return_level,
    bootstrap_return_levels,
    to_billion,
)


def legacy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the submitted construction: marginal quantile per column,
    then SUM the marginal quantiles of the four burden components. This is
    provided only to document exact reproduction of the submitted numbers.
    """
    rows = {}
    for label, col in LOSS_COLUMNS.items():
        if col is None:
            col = "wind_un_underinsured_usd"
        rl = empirical_return_level(df[col].to_numpy(dtype=float))
        rows[label] = {f"RP{rp}": to_billion(rl[f"RP{rp}"]) for rp in RETURN_PERIODS}

    comp_quantiles = {}
    for label, col in INST_COLUMNS.items():
        rl = empirical_return_level(df[col].to_numpy(dtype=float))
        comp_quantiles[label] = {f"RP{rp}": rl[f"RP{rp}"] for rp in RETURN_PERIODS}
        rows[label] = {f"RP{rp}": to_billion(rl[f"RP{rp}"]) for rp in RETURN_PERIODS}

    burden_sum = {
        f"RP{rp}": sum(comp_quantiles[label][f"RP{rp}"] for label in INST_COLUMNS)
        for rp in RETURN_PERIODS
    }
    rows["Total public burden (legacy: sum of marginal quantiles)"] = {
        f"RP{rp}": to_billion(burden_sum[f"RP{rp}"]) for rp in RETURN_PERIODS
    }
    out = pd.DataFrame(rows).T
    out.index.name = "Metric"
    return out.reset_index()


def corrected_table(df: pd.DataFrame) -> pd.DataFrame:
    """Season-level-first construction. Total loss and loss-decomposition
    rows are unchanged marginal quantiles (this is the correct convention
    for those rows, and matches the submission). Total public burden is the
    quantile of the season-level sum, computed for both the legacy
    (4-component, overlapping) and corrected (3-component, non-overlapping)
    definitions.
    """
    rows = {}
    for label, col in LOSS_COLUMNS.items():
        if col is None:
            col = "wind_un_underinsured_usd"
        rl = empirical_return_level(df[col].to_numpy(dtype=float))
        rows[label] = {f"RP{rp}": to_billion(rl[f"RP{rp}"]) for rp in RETURN_PERIODS}

    for label, col in INST_COLUMNS.items():
        rl = empirical_return_level(df[col].to_numpy(dtype=float))
        rows[label] = {f"RP{rp}": to_billion(rl[f"RP{rp}"]) for rp in RETURN_PERIODS}

    rl_legacy_season = empirical_return_level(df["public_burden_legacy_usd"].to_numpy(dtype=float))
    rows["Residual financing requirement (season-sum, legacy 4-component def., for reconciliation)"] = {
        f"RP{rp}": to_billion(rl_legacy_season[f"RP{rp}"]) for rp in RETURN_PERIODS
    }

    rl_corr_season = empirical_return_level(df["public_burden_corrected_usd"].to_numpy(dtype=float))
    rows["Residual financing requirement (season-sum, FIGA+Citizens+NFIP)"] = {
        f"RP{rp}": to_billion(rl_corr_season[f"RP{rp}"]) for rp in RETURN_PERIODS
    }

    out = pd.DataFrame(rows).T
    out.index.name = "Metric"
    return out.reset_index()


def amplification_summary(df: pd.DataFrame) -> dict:
    total_loss = df["total_damage_usd"].to_numpy(dtype=float)
    legacy = df["public_burden_legacy_usd"].to_numpy(dtype=float)
    corrected = df["public_burden_corrected_usd"].to_numpy(dtype=float)

    def ratios(x, label):
        rl = empirical_return_level(x)
        rp10, rp100 = rl["RP10"], rl["RP100"]
        ratio = (rp100 / rp10) if rp10 > 0 else float("inf")
        return {
            "label": label,
            "RP10_usd": rp10,
            "RP100_usd": rp100,
            "RP100_over_RP10": ratio,
        }

    res = {
        "total_loss": ratios(total_loss, "Total loss"),
        "public_burden_legacy": ratios(legacy, "Public burden (legacy def.)"),
        "public_burden_corrected": ratios(corrected, "Public burden (corrected def.)"),
    }

    # Power-law exponent beta: log(burden) ~ beta * log(total_loss) + const,
    # fit across the reported return-period grid (RP10..RP1000), matching the
    # manuscript's stated fitting range. Reported for both burden definitions.
    rl_loss = empirical_return_level(total_loss)
    for key, arr in [("public_burden_legacy", legacy), ("public_burden_corrected", corrected)]:
        rl_burden = empirical_return_level(arr)
        xs, ys = [], []
        for rp in RETURN_PERIODS:
            lx, ly = rl_loss[f"RP{rp}"], rl_burden[f"RP{rp}"]
            if lx > 0 and ly > 0:
                xs.append(np.log(lx))
                ys.append(np.log(ly))
        if len(xs) >= 2:
            beta, intercept = np.polyfit(xs, ys, 1)
        else:
            beta, intercept = float("nan"), float("nan")
        res[key]["beta_loglog_RP10_RP1000"] = float(beta)

    return res


def gdp_exceedance(df: pd.DataFrame, gdp_usd: float) -> dict:
    legacy = df["public_burden_legacy_usd"].to_numpy(dtype=float)
    corrected = df["public_burden_corrected_usd"].to_numpy(dtype=float)
    return {
        "gdp_usd": gdp_usd,
        "legacy": {
            "P(burden > 1% GDP)": float((legacy > 0.01 * gdp_usd).mean()),
            "P(burden > 10% GDP)": float((legacy > 0.10 * gdp_usd).mean()),
        },
        "corrected": {
            "P(burden > 1% GDP)": float((corrected > 0.01 * gdp_usd).mean()),
            "P(burden > 10% GDP)": float((corrected > 0.10 * gdp_usd).mean()),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=Path, required=True)
    ap.add_argument("--gdp", type=float, default=1.7e12, help="Florida GDP in USD (2024 value used in submission)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "tables")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_iterations(args.iterations)

    legacy_table(df).to_csv(args.out_dir / "table1_legacy_reproduction.csv", index=False)
    corrected = corrected_table(df)
    corrected.to_csv(args.out_dir / "table1_corrected.csv", index=False)

    amp = amplification_summary(df)
    gdp = gdp_exceedance(df, args.gdp)

    boot = bootstrap_return_levels(
        df,
        ["total_damage_usd", "public_burden_legacy_usd", "public_burden_corrected_usd"],
        n_boot=args.n_boot,
    )
    boot_billion = {
        c: {rp: {k: to_billion(v) if k != "point" else to_billion(v) for k, v in d.items()}
            for rp, d in rps.items()}
        for c, rps in boot.items()
    }

    summary = {
        "source_iterations": str(args.iterations),
        "n_seasons": int(len(df)),
        "amplification": amp,
        "gdp_exceedance": gdp,
        "bootstrap_1000_resamples_10_90pct_billionUSD": boot_billion,
    }
    with open(args.out_dir / "table1_corrected_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {args.out_dir / 'table1_legacy_reproduction.csv'}")
    print(f"Wrote: {args.out_dir / 'table1_corrected.csv'}")
    print(f"Wrote: {args.out_dir / 'table1_corrected_summary.json'}")


if __name__ == "__main__":
    main()
