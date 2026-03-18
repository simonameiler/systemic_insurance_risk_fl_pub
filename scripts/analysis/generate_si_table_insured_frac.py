#!/usr/bin/env python3
"""
Generate SI LaTeX table for insured fraction sensitivity analysis.

Outputs a table showing mean values (in $B) and relative change vs. the
reference fraction (f=0.4, Beta(4,6) mean) for all 15 output metrics,
plus the approximate log-log elasticity (ε).

Usage:
    python scripts/analysis/generate_si_table_insured_frac.py \
        [--dir results/mc_runs/insured_frac_sensitivity_combined]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Metrics — trimmed to SI-table subset (matches variance-decomposition groups)
# ---------------------------------------------------------------------------
METRICS = [
    # (column_name, label, group)
    ("wind_insured_private_usd",     "Wind insured --- private",         "Loss decomposition"),
    ("wind_insured_citizens_usd",    "Wind insured --- Citizens",        "Loss decomposition"),
    ("wind_uninsured_usd",           "Wind un/underinsured",             "Loss decomposition"),
    ("fhcf_shortfall_usd",           "FHCF shortfall",                   "Institutional stress"),
    ("figa_residual_deficit_usd",    "FIGA residual deficit",            "Institutional stress"),
    ("citizens_residual_deficit_usd","Citizens residual deficit",        "Institutional stress"),
    ("defaults_post",                "Insurer defaults (count)",         "Capital / defaults"),
    ("largest_entity_deficit_usd",   "Largest entity deficit",           "Capital / defaults"),
]

# Metrics that are counts rather than USD amounts — formatted differently (not /1e9)
COUNT_METRICS = {"defaults_post"}

FRACS = [0.1, 0.2, 0.3, 0.4, 0.5]
REF   = 0.4


def load_data(data_dir: Path) -> dict[float, pd.DataFrame]:
    dfs = {}
    for f in FRACS:
        path = data_dir / f"iterations_frac_{f:.2f}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        dfs[f] = pd.read_csv(path)
    return dfs


def fmt_val(col: str, v: float) -> str:
    """Format a metric value: counts as raw float (4 dp), USD as $B (2 dp)."""
    if col in COUNT_METRICS:
        return f"{v:.4f}"
    return f"{v / 1e9:.2f}"


def fmt_pct(pct: float) -> str:
    """Format a relative change as a signed percentage."""
    if not np.isfinite(pct):
        return "n/a"
    return f"{pct:+.0f}\\%"


def elasticity(dfs: dict, col: str) -> float:
    """Log-log elasticity centred at f=0.4 (using f=0.3 and f=0.5)."""
    m3 = dfs[0.3][col].mean()
    m5 = dfs[0.5][col].mean()
    if m3 > 0 and m5 > 0:
        return (np.log(m5) - np.log(m3)) / (np.log(0.5) - np.log(0.3))
    return np.nan


def build_table(dfs: dict) -> str:
    lines = []

    # ── preamble ────────────────────────────────────────────────────────────
    lines += [
        r"\begin{table}[htb!]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Insured wind fraction sensitivity.}",
        r"  Mean values (in \$B, except insurer defaults which are a count",
        r"  per 10{,}000-season simulation) for each output metric across five",
        r"  fixed insured wind fraction values, along with relative change",
        r"  versus the reference ($f = 0.4$, the mean of the baseline",
        r"  Beta(4,\,6) prior) and the approximate log--log elasticity",
        r"  $\varepsilon$ centred at $f = 0.4$.",
        r"  Metrics with $\varepsilon \approx 0$ are invariant to the insured",
        r"  fraction by construction; $\varepsilon = 1$ indicates proportional",
        r"  scaling; $\varepsilon > 1$ indicates amplification through the",
        r"  insurance market.}",
        r"\label{tab:si_insured_frac}",
    ]

    # ── column spec ─────────────────────────────────────────────────────────
    # Metric | f=0.1 | f=0.2 | f=0.3 | f=0.4* | f=0.5 | Δ 0.1 | Δ 0.2 | Δ 0.3 | Δ 0.5 | ε
    lines += [
        r"\begin{tabular}{lrrrrr|rrrrr}",
        r"\toprule",
        (r"& \multicolumn{5}{c|}{Mean value (\$B)} "
         r"& \multicolumn{4}{c}{Change vs.\ $f{=}0.4$} "
         r"& \\"),
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}",
        (r"Metric "
         r"& $f{=}0.1$ & $f{=}0.2$ & $f{=}0.3$ & $f{=}0.4$ & $f{=}0.5$ "
         r"& $\Delta_{0.1}$ & $\Delta_{0.2}$ & $\Delta_{0.3}$ & $\Delta_{0.5}$ "
         r"& $\varepsilon$ \\"),
        r"\midrule",
    ]

    # ── rows ────────────────────────────────────────────────────────────────
    current_group = None
    ref_means = {col: dfs[REF][col].mean() for col, *_ in METRICS}

    for col, label, group in METRICS:
        # group header
        if group != current_group:
            if current_group is not None:
                lines.append(r"\addlinespace")
            lines.append(
                rf"\multicolumn{{11}}{{l}}{{\textit{{{group}}}}} \\\\"
            )
            current_group = group

        # absolute means
        vals = [fmt_val(col, dfs[f][col].mean()) for f in FRACS]

        # relative changes (vs f=0.4)
        ref_m = ref_means[col]
        deltas = []
        for f in [0.1, 0.2, 0.3, 0.5]:
            alt_m = dfs[f][col].mean()
            if abs(ref_m) > 0:
                pct = (alt_m - ref_m) / abs(ref_m) * 100
            else:
                pct = np.nan
            deltas.append(fmt_pct(pct))

        # elasticity
        eps = elasticity(dfs, col)
        eps_str = f"{eps:+.2f}" if np.isfinite(eps) else "n/a"

        row = (
            f"{label} "
            + "& " + " & ".join(vals)
            + " & " + " & ".join(deltas)
            + f" & {eps_str}"
            + r" \\"
        )
        lines.append(row)

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
        "--dir",
        default="results/mc_runs/insured_frac_sensitivity_combined",
        help="Directory containing iterations_frac_*.csv files",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write LaTeX to this file (default: print to stdout)",
    )
    args = parser.parse_args()

    data_dir = Path(args.dir)
    dfs = load_data(data_dir)

    print(f"Loaded data for fracs: {FRACS}")
    for f in FRACS:
        print(f"  f={f:.1f}: {len(dfs[f]):,} iterations")

    table = build_table(dfs)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table)
        print(f"\nWrote LaTeX table → {out_path}")
    else:
        print("\n" + "=" * 70)
        print("LATEX TABLE")
        print("=" * 70)
        print(table)

    # ── also print a quick plain-text summary for verification ──────────────
    print("\n" + "=" * 70)
    print("PLAIN-TEXT SUMMARY (verification)")
    print("=" * 70)
    header = f"{'Metric':<38s}" + "".join(f"{'f='+str(f):>10s}" for f in FRACS) + f"{'ε':>8s}"
    print(header)
    print("-" * 100)
    for col, label, group in METRICS:
        if col in COUNT_METRICS:
            vals_str = "".join(f"{dfs[f][col].mean():>10.4f}" for f in FRACS)
        else:
            vals_str = "".join(f"{dfs[f][col].mean()/1e9:>10.3f}" for f in FRACS)
        eps = elasticity(dfs, col)
        eps_str = f"{eps:>+8.2f}" if np.isfinite(eps) else f"{'n/a':>8s}"
        print(f"{label:<38s}{vals_str}{eps_str}")


if __name__ == "__main__":
    main()
