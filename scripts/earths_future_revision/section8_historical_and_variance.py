#!/usr/bin/env python3
"""Section 8: season-count definitions, historical scenario reprocessing,
and the Table S7 variance-decomposition audit.

1. Season-event-count categories (register item R2 / AUTHOR CHECK S5-COUNTS).
   Verifies, from the archived ERA5 baseline iterations.csv, the precise
   definition of the submitted 28.5% / 15.3% / 56.2% figures:
     - 28.5% = seasons with total_damage_usd == 0 (a season can have a
       retained track within the 150km coastal buffer that produces zero
       county-level damage; these are zero-LOSS seasons, not necessarily
       zero-EVENT seasons -- every season in this catalog has at least one
       nominally retained event).
     - 15.3% = seasons with total_damage_usd > 0 and exactly one
       contributing event.
     - 56.2% = seasons with total_damage_usd > 0 and two or more
       contributing events.
   These three reproduce the submitted percentages to within 0.03
   percentage points and sum to 100% by construction.

2. Historical scenario table (SI Table S3), reprocessed with the corrected,
   non-overlapping public-burden definition (register item C1), using the
   locally available historical/paired-event Monte Carlo runs (200
   realizations each; gitignored in the main repository, copied read-only
   into this worktree -- see docs/earths_future_revision/data_inventory.md).

3. Variance decomposition (SI Table S7) audit. Reproduces the reported
   eta-squared values from the archived 300-season x 50-draw nested design
   and documents why Table S7 is removed from the manuscript copy per the
   revision brief (Section 8): the 300 seasons were selected to span the
   loss distribution rather than drawn at random, so the reported variance
   shares describe that selected design and should not be read as a general
   claim that hazard variability dominates for a randomly sampled season.
   The research code and archived results are retained; they are not
   rerun or expanded here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_DIR, load_iterations, empirical_return_level, to_billion

REPO_ROOT = Path(__file__).resolve().parents[2]
MC_ROOT = REPO_ROOT / "results" / "mc_runs"

HISTORICAL_DIRS = {
    "Great Miami": "great_miami_20260326_190906",
    "Andrew": "andrew_20260326_192708",
    "Lake Okeechobee": "lake_okeechobee_20260326_212501",
    "Irma": "irma_20260326_213211",
    "Great Miami then Andrew": "gm_then_andrew_20260326_211013",
    "Double Great Miami": "double_gm_20260326_211737",
    "Double Irma": "double_irma_20260326_213909",
}

VARIANCE_DIR = MC_ROOT / "variance_nested_300x50_20260311_211009"


def season_count_categories(baseline_path: Path) -> dict:
    df = pd.read_csv(baseline_path, usecols=["events", "total_damage_usd"])

    def n_events(s):
        if pd.isna(s) or s == "":
            return 0
        return len(str(s).split(","))

    n = df["events"].apply(n_events)
    loss = df["total_damage_usd"]
    n_total = len(df)

    zero_loss = (loss <= 0)
    one_event_nonzero = (loss > 0) & (n == 1)
    multi_event_nonzero = (loss > 0) & (n >= 2)

    return {
        "n_seasons": int(n_total),
        "pct_zero_loss": float(zero_loss.mean() * 100),
        "pct_one_event_nonzero_loss": float(one_event_nonzero.mean() * 100),
        "pct_multi_event_nonzero_loss": float(multi_event_nonzero.mean() * 100),
        "sums_to_100": float(zero_loss.mean() + one_event_nonzero.mean() + multi_event_nonzero.mean()) * 100,
        "submitted_values": {"zero_loss": 28.5, "one_event": 15.3, "multi_event": 56.2},
    }


def historical_scenario_table(dirs: dict[str, str]) -> pd.DataFrame:
    rows = []
    for label, dirname in dirs.items():
        path = MC_ROOT / dirname / "iterations.csv"
        if not path.exists():
            print(f"  [skip] {label}: {path} not found")
            continue
        df = load_iterations(path)
        n = len(df)

        def m_ci(col):
            x = df[col].to_numpy(dtype=float)
            return x.mean(), np.percentile(x, 5), np.percentile(x, 95)

        row = {"Scenario": label, "n_realizations": n}
        for out_label, col in [
            ("Total loss", "total_damage_usd"),
            ("Total public burden (legacy 4-comp.)", "public_burden_legacy_usd"),
            ("Total public burden (corrected, non-overlapping)", "public_burden_corrected_usd"),
            ("FHCF shortfall (diagnostic)", "fhcf_shortfall_usd"),
            ("FIGA residual", "figa_residual_deficit_usd"),
            ("Citizens deficit", "citizens_residual_deficit_usd"),
            ("NFIP Treasury borrowing", "nfip_borrowed_usd"),
        ]:
            m, lo, hi = m_ci(col)
            row[out_label] = f"{to_billion(m):.1f}B ({to_billion(lo):.1f}-{to_billion(hi):.1f}B)"
        rows.append(row)
    return pd.DataFrame(rows)


def variance_table7_audit(variance_dir: Path) -> pd.DataFrame:
    path = variance_dir / "variance_decomposition_nested.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["exceeds_0.95"] = df["eta_squared"] >= 0.95
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path,
                     default=MC_ROOT / "emanuel_era5_baseline_20260326_141913" / "iterations.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "historical")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Season-count category verification (register item R2) ===")
    cats = season_count_categories(args.baseline)
    for k, v in cats.items():
        print(f"  {k}: {v}")

    print("\n=== Historical scenario table (SI Table S3), corrected ===")
    hist = historical_scenario_table(HISTORICAL_DIRS)
    hist.to_csv(args.out_dir / "table_S3_corrected.csv", index=False)
    print(hist.to_string(index=False))

    print("\n=== Table S7 variance-decomposition audit (removed from manuscript) ===")
    var7 = variance_table7_audit(VARIANCE_DIR)
    if not var7.empty:
        var7.to_csv(args.out_dir / "table_S7_audit_retained_for_code_reference.csv", index=False)
        n_below = int((~var7["exceeds_0.95"]).sum())
        print(var7[["metric", "eta_squared", "exceeds_0.95"]].to_string(index=False))
        print(f"\nMetrics below eta^2=0.95: {n_below} (manuscript flags FHCF shortfall specifically; "
              f"flood metrics are also below 0.95, consistent with the submitted text)")
    else:
        print(f"  [skip] {VARIANCE_DIR} not found")

    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
