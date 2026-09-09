#!/usr/bin/env python3
"""Section 6: common-season burden decomposition by loss-severity bin.

Uses the corrected baseline seasons (same iterations.csv as Section 5). For
each severity bin we report, using the SAME seasons for every component (so
the decomposition is internally consistent within a season):

  - mean economic loss (total_damage_usd)
  - mean of each final burden component (FHCF shortfall, FIGA residual,
    Citizens deficit, NFIP borrowing)
  - each component's additive burden share = mean(component) / mean(economic
    loss) -- a ratio of means, not a mean of ratios, and not a mixture of the
    two.

Bins are defined on total_damage_usd using approximate return-period anchors
so that "moderate to severe" severity (roughly the 10- to 100-year range) is
resolved with more than one bin, while very sparse tail seasons are merged
into a single top bin (documented rule: any bin with fewer than
`--min-count` seasons is merged with the adjacent bin). Zero-loss seasons
are reported as their own bin and excluded from the ratio calculation only
where the denominator is exactly zero (this bin is reported as burden share
= 0 by definition, not divided).

Uncertainty: each bin's shares are bootstrapped by resampling season rows
*within the bin* 1,000 times (seasons are never moved between bins across
resamples), reporting the 10th-90th percentile interval.

FHCF shortfall is shown as a separate, explicitly labeled upstream diagnostic
(it overlaps with FIGA/Citizens; see common.py and the correction register)
rather than stacked as a fifth non-overlapping channel. The three
non-overlapping components (FIGA, Citizens, NFIP) reconcile exactly with the
corrected total public burden by construction.

Outputs
-------
  results/earths_future_revision/decomposition/severity_bin_decomposition.csv
  results/earths_future_revision/figures/fig_decomposition.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT_DIR, load_iterations, to_billion

BIN_EDGES_USD = [0.0, 1.0, 39.5e9, 147e9, 246e9, 357e9, 564e9, np.inf]
BIN_LABELS = [
    "Zero loss",
    "> 0 -- 10yr (39.5B)",
    "10yr -- 25yr (39.5-147B)",
    "25yr -- 50yr (147-246B)",
    "50yr -- 100yr (246-357B)",
    "100yr -- 250yr (357-564B)",
    "> 250yr (> 564B)",
]

COMPONENTS = {
    "fhcf_shortfall_usd": "FHCF shortfall (upstream diagnostic, overlaps FIGA/Citizens)",
    "figa_residual_deficit_usd": "FIGA residual",
    "citizens_residual_deficit_usd": "Citizens deficit",
    "nfip_borrowed_usd": "NFIP Treasury borrowing",
}


def assign_bins(df: pd.DataFrame, min_count: int) -> pd.Series:
    bins = pd.cut(df["total_damage_usd"], BIN_EDGES_USD, labels=BIN_LABELS,
                  right=True, include_lowest=True)
    # Zero-loss seasons: cut with include_lowest puts damage==0 into the first
    # bin already since edges start at 0.0; make it explicit.
    bins = bins.astype(object)
    bins[df["total_damage_usd"] <= 0.0] = "Zero loss"

    # Merge sparse bins (documented rule: merge a bin with < min_count seasons
    # into the next-lower-severity bin that has adequate counts).
    counts = bins.value_counts()
    order = BIN_LABELS
    merged = bins.copy()
    for i, label in enumerate(order):
        if label == "Zero loss":
            continue
        if counts.get(label, 0) < min_count and i > 1:
            prev_label = order[i - 1]
            merged[merged == label] = prev_label
    return merged


def bin_summary(df: pd.DataFrame, bins: pd.Series, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label in bins.unique():
        sub = df.loc[bins == label]
        n = len(sub)
        mean_loss = float(sub["total_damage_usd"].mean())
        row = {"bin": label, "n_seasons": n, "mean_total_loss_usd": mean_loss}

        boot_shares = {c: np.empty(n_boot) for c in COMPONENTS}
        for c in COMPONENTS:
            mean_c = float(sub[c].mean())
            row[f"mean_{c}"] = mean_c
            row[f"share_{c}"] = (mean_c / mean_loss) if mean_loss > 0 else 0.0

        if n >= 2:
            arr_loss = sub["total_damage_usd"].to_numpy(dtype=float)
            arr_comp = {c: sub[c].to_numpy(dtype=float) for c in COMPONENTS}
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                ml = arr_loss[idx].mean()
                for c in COMPONENTS:
                    mc = arr_comp[c][idx].mean()
                    boot_shares[c][b] = (mc / ml) if ml > 0 else 0.0
            for c in COMPONENTS:
                lo, hi = np.percentile(boot_shares[c], [10, 90])
                row[f"share_{c}_lo90"] = lo
                row[f"share_{c}_hi90"] = hi
        else:
            for c in COMPONENTS:
                row[f"share_{c}_lo90"] = np.nan
                row[f"share_{c}_hi90"] = np.nan

        # Reconciliation: corrected total (FIGA+Citizens+NFIP) share vs. mean
        # public_burden_corrected_usd / mean_loss (must match by construction).
        nonoverlap = [c for c in COMPONENTS if c != "fhcf_shortfall_usd"]
        total_share = sum(row[f"share_{c}"] for c in nonoverlap)
        recon_share = (float(sub["public_burden_corrected_usd"].mean()) / mean_loss) if mean_loss > 0 else 0.0
        row["total_nonoverlapping_share"] = total_share
        row["reconciliation_check_diff"] = total_share - recon_share
        rows.append(row)

    out = pd.DataFrame(rows)
    order_map = {label: i for i, label in enumerate(BIN_LABELS)}
    out["_order"] = out["bin"].map(order_map)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def make_figure(summary: pd.DataFrame, out_dir: Path):
    plot_df = summary[summary["n_seasons"] >= 5].copy()
    if plot_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(plot_df))
    width = 0.25
    colors = {"figa_residual_deficit_usd": "#4C72B0",
              "citizens_residual_deficit_usd": "#DD8452",
              "nfip_borrowed_usd": "#55A868"}
    bottom = np.zeros(len(plot_df))
    for c, color in colors.items():
        vals = plot_df[f"mean_{c}"].to_numpy(dtype=float) / 1e9
        axes[0].bar(x, vals, bottom=bottom, width=0.6, label=c.replace("_usd", ""), color=color)
        bottom += vals
    fhcf = plot_df["mean_fhcf_shortfall_usd"].to_numpy(dtype=float) / 1e9
    axes[0].plot(x, fhcf, "k--", marker="o", label="FHCF shortfall (upstream diagnostic, overlapping)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["bin"], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Mean component amount (USD B)")
    axes[0].set_title("a) Component amounts by severity bin")
    axes[0].legend(fontsize=7)

    bottom = np.zeros(len(plot_df))
    for c, color in colors.items():
        vals = plot_df[f"share_{c}"].to_numpy(dtype=float)
        axes[1].bar(x, vals, bottom=bottom, width=0.6, color=color)
        bottom += vals
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["bin"], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Additive burden share (component mean / loss mean)")
    axes[1].set_title("b) Non-overlapping burden shares by severity bin")

    fig.suptitle("Accounting decomposition of modeled public burden by loss severity"
                  " (ERA5 baseline, corrected non-overlapping definition)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_decomposition.pdf")
    fig.savefig(out_dir / "fig_decomposition.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=Path, required=True)
    ap.add_argument("--min-count", type=int, default=50,
                     help="Minimum seasons per bin before merging with the adjacent bin.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "decomposition")
    ap.add_argument("--fig-dir", type=Path, default=OUT_DIR / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_iterations(args.iterations)
    bins = assign_bins(df, args.min_count)
    summary = bin_summary(df, bins, args.n_boot, seed=42)

    csv_path = args.out_dir / "severity_bin_decomposition.csv"
    summary.to_csv(csv_path, index=False)
    print(summary.to_string(index=False))
    max_recon_diff = summary["reconciliation_check_diff"].abs().max()
    print(f"\nMax reconciliation check |diff| across bins: {max_recon_diff:.2e} (should be ~0)")

    make_figure(summary, args.fig_dir)
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {args.fig_dir / 'fig_decomposition.pdf'} / .png")


if __name__ == "__main__":
    main()
