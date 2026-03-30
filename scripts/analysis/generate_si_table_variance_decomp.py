#!/usr/bin/env python3
"""
generate_si_table_variance_decomp.py - Generate SI LaTeX table for variance decomposition

Reads the nested MC variance decomposition CSV and produces a LaTeX table
showing eta-squared (fraction of variance from hazard vs. parameters).

Usage:
    python scripts/analysis/generate_si_table_variance_decomp.py \
        [--csv results/mc_runs/variance_nested_300x50_20260311_211009/variance_decomposition_nested.csv]
"""

import argparse
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Metrics to include, grouped into panels
# (csv_column, display_label, panel)
# ---------------------------------------------------------------------------
METRICS = [
    ("wind_insured_private_usd",      "Private insurer wind losses",   "Loss decomposition"),
    ("wind_insured_citizens_usd",     "Citizens wind losses",          "Loss decomposition"),
    ("flood_insured_capped_usd",      "NFIP flood (insured)",          "Loss decomposition"),
    ("wind_uninsured_usd",            "Un/underinsured wind",          "Loss decomposition"),
    ("flood_un_derinsured_usd",       "Un/underinsured flood",         "Loss decomposition"),
    ("fhcf_shortfall_usd",            "FHCF shortfall",                "Institutional stress"),
    ("figa_residual_deficit_usd",     "FIGA residual deficit",         "Institutional stress"),
    ("citizens_residual_deficit_usd", "Citizens residual deficit",     "Institutional stress"),
    ("nfip_borrowed_usd",             "NFIP Treasury borrowing",       "Institutional stress"),
    ("defaults_post",                 "Insurer defaults (count)",      "Defaults"),
    ("largest_entity_deficit_usd",    "Largest entity deficit",        "Defaults"),
]


def build_table(df: pd.DataFrame) -> str:
    # Index by metric name for easy lookup
    df = df.set_index("metric")

    lines = []

    # ── preamble ────────────────────────────────────────────────────────────
    lines += [
        r"\begin{table}[htb!]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Relative contributions of hazard variability and parameter uncertainty to model output variance.} Values of $\eta^2$ from a one-way ANOVA applied to a nested Monte Carlo design (300 seasons $\times$ 50 independent parameter draws per season), measuring the fraction of total variance attributable to between-season hazard differences versus within-season parameter uncertainty. $\eta^2 \geq 0.95$ for all non-flood metrics indicates that hazard realization dominates; the lower values for flood-linked metrics reflect the wind/water share Beta prior, which controls how much total damage is routed to NFIP versus private wind coverage.}",
        r"\label{tab:si_variance_decomp}",
    ]

    # ── column spec ─────────────────────────────────────────────────────────
    lines += [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & $\eta^2$ (hazard) & $1 - \eta^2$ (parameters) \\",
        r"\midrule",
    ]

    # ── rows ────────────────────────────────────────────────────────────────
    current_panel = None

    for col, label, panel in METRICS:
        if panel != current_panel:
            if current_panel is not None:
                lines.append(r"\addlinespace")
            lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{panel}}}}} \\")
            current_panel = panel

        if col not in df.index:
            print(f"  WARNING: metric '{col}' not found in CSV, skipping")
            continue

        eta2 = df.loc[col, "eta_squared"]
        one_minus = 1 - eta2

        lines.append(
            f"{label:<34s} & {eta2:.3f} & {one_minus:.3f} \\\\"
        )

    # ── footer ──────────────────────────────────────────────────────────────
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="results/mc_runs/variance_nested_300x50_20260311_211009/variance_decomposition_nested.csv",
        help="Path to variance_decomposition_nested.csv",
    )
    parser.add_argument(
        "--out",
        default="results/tables/si_table_variance_decomp.tex",
        help="Write LaTeX to this file",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} metrics from {args.csv}")

    table = build_table(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"Wrote LaTeX table -> {out_path}")

    # Also print to stdout for verification
    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    print(table)


if __name__ == "__main__":
    main()
