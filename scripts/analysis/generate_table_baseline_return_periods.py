#!/usr/bin/env python3
"""Generate LaTeX table for baseline loss decomposition and institutional
stress across return periods.

Reads ``results/tables/baseline_metrics_return_periods.csv`` (produced by the
probabilistic_risk_analysis notebook) and writes a formatted LaTeX table
to ``results/tables/baseline_return_periods.tex``.
"""

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

# Metric ordering and display names
LOSS_METRICS = [
    ("Total Losses",              "Total loss"),
    ("Wind Insured (Private)",    "Insured wind -- private"),
    ("Wind Insured (Citizens)",   "Citizens wind"),
    ("Flood Insured (NFIP)",      "Insured flood -- NFIP"),
    ("Wind Un/Underinsured",      "Un/underinsured wind"),
    ("Flood Un/Underinsured",     "Un/underinsured flood"),
]

INST_METRICS = [
    ("_total_public_burden",      "Total public burden"),
    ("FHCF Shortfall",            "FHCF shortfall"),
    ("FIGA Residual",             "FIGA residual"),
    ("Citizens Deficit",          "Citizens deficit"),
    ("NFIP Borrowed",             "NFIP Treasury borrowing"),
]

RP_COLS = ["RP10", "RP25", "RP50", "RP100", "RP250", "RP500", "RP1000"]


def _fmt(val: float) -> str:
    """Format a value to one decimal place."""
    return f"{val:.1f}"


def build_latex_table(df: pd.DataFrame) -> str:
    """Return the full LaTeX table string."""
    # Index by metric name for easy lookup
    lookup = df.set_index("Metric")

    # Derive total public burden as sum of institutional components
    burden_keys = ["FHCF Shortfall", "FIGA Residual",
                   "Citizens Deficit", "NFIP Borrowed"]
    burden = lookup.loc[burden_keys, RP_COLS].sum()

    lines = []

    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Loss decomposition and public institutional burden across return periods."
    )
    lines.append(
        r"Return period (RP) estimates are shown for total losses, their decomposition "
        r"across insured and uninsured components, and the resulting public and quasi-public "
        r"institutional burden. Public burden reflects residual financial obligations borne "
        r"by public backstops after exhaustion of private insurer capital and risk-transfer "
        r"mechanisms.}"
    )
    lines.append(r"\label{TabProbRPvalues}")
    lines.append("")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append("")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{7}{c}{\textbf{Return period [years]}} \\")
    lines.append(r"\cmidrule(l){2-8}")
    lines.append(
        r"\textbf{Metric (USD B)} &"
    )
    lines.append(
        r"\textbf{10} &"
    )
    lines.append(
        r"\textbf{25} &"
    )
    lines.append(
        r"\textbf{50} &"
    )
    lines.append(
        r"\textbf{100} &"
    )
    lines.append(
        r"\textbf{250} &"
    )
    lines.append(
        r"\textbf{500} &"
    )
    lines.append(
        r"\textbf{1000} \\")
    lines.append(r"\midrule")

    # --- Loss decomposition section ---
    lines.append(r"\multicolumn{8}{l}{\textit{Loss decomposition}} \\")
    for csv_name, display_name in LOSS_METRICS:
        vals = [_fmt(lookup.loc[csv_name, c]) for c in RP_COLS]
        padded = f"{display_name:<30s}"
        lines.append(f"{padded} & " + " & ".join(vals) + r" \\")

    # --- Institutional stress section ---
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Institutional stress}} \\")
    for csv_name, display_name in INST_METRICS:
        if csv_name == "_total_public_burden":
            vals = [_fmt(burden[c]) for c in RP_COLS]
        else:
            vals = [_fmt(lookup.loc[csv_name, c]) for c in RP_COLS]
        padded = f"{display_name:<30s}"
        lines.append(f"{padded} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "baseline_metrics_return_periods.csv",
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "tables" / "baseline_return_periods.tex",
        help="Path for the output LaTeX file.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    tex = build_latex_table(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"✅ Saved LaTeX table: {args.out}")


if __name__ == "__main__":
    main()
