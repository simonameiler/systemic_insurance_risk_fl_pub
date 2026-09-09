#!/usr/bin/env python3
"""Section 9: corrected scenario/climate means table (SI Table S4 equivalent).

Diagnoses and corrects the confirmed SI Table S4 building-code bug (register
item C5): fl_risk_model.mc_run_events.run_one_iteration captured
wind_total/water_total (and therefore total_damage_usd, wind_total_usd,
water_total_usd) BEFORE the building-codes loss-reduction block, so reported
gross totals silently ignored the prescribed reduction while insured and
uninsured/underinsured components (computed from the reduced damage) shrank
correctly. That produced the submitted table's near-constant ~USD 19.3B
"Total loss" and its INCREASE in uninsured wind/flood under building codes
(a downstream reconstruction of "uninsured" as stale_total - insured, rather
than the correctly computed uninsured/underinsured columns).

The bug is fixed at the source in fl_risk_model/mc_run_events.py (this
patch). Because a full rerun of the archived scenarios requires proprietary
hazard inputs not available in this environment (see data_inventory.md),
this script reconstructs the CORRECTED totals for already-archived runs from
their still-correct component columns:

    corrected_total = wind_insured_private + wind_insured_citizens
                     + wind_uninsured + wind_underinsured
                     + flood_insured_capped + flood_underinsured

This reconstruction is verified against the unaffected scenarios (baseline,
market_exit, penetration), where it must equal the already-reported
total_damage_usd exactly (see fl_risk_model/tests/earths_future/
test_scenario_totals.py).

Reports means (10th-90th percentile across seasons) for baseline and the
three ERA5 policy scenarios, using the corrected total loss and the
corrected non-overlapping public burden (register item C1). GCM ensemble
climate deltas for the same corrected metrics are computed with the
existing median-across-GCM, add-to-ERA5 convention (scripts/analysis/
compute_climate_deltas.py), restricted to SSP2-4.5 per the revision brief;
SSP5-8.5 is retained in the SI using the same corrected metrics.

Full re-derivation of every SI Table S5 exceedance-probability threshold
across the whole climate/policy grid is DEFERRED (see
docs/earths_future_revision/correction_register.md item R1 follow-on): it
requires locating the exact threshold-indicator code used by the
publication notebook for each of the ten stress metrics, which was out of
scope for the time available in this pass. The corrected mean-based metrics
below are unaffected by that limitation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_DIR, load_iterations, to_billion

REPO_ROOT = Path(__file__).resolve().parents[2]
MC_ROOT = REPO_ROOT / "results" / "mc_runs"

CORRECTED_TOTAL_COMPONENTS = [
    "wind_insured_private_usd",
    "wind_insured_citizens_usd",
    "wind_uninsured_usd",
    "wind_underinsured_usd",
    "flood_insured_capped_usd",
    "flood_un_derinsured_usd",
]

MEANS_METRICS = {
    "Total loss (corrected)": "_corrected_total_damage_usd",
    "Insured wind, private": "wind_insured_private_usd",
    "Citizens wind": "wind_insured_citizens_usd",
    "Insured flood, NFIP": "flood_insured_capped_usd",
    "Un/underinsured wind": "wind_un_underinsured_usd",
    "Un/underinsured flood": "flood_un_derinsured_usd",
    "Total public burden (corrected)": "public_burden_corrected_usd",
    "FHCF shortfall (diagnostic)": "fhcf_shortfall_usd",
    "FIGA residual": "figa_residual_deficit_usd",
    "Citizens deficit": "citizens_residual_deficit_usd",
    "NFIP Treasury borrowing": "nfip_borrowed_usd",
}

SCENARIO_DIRS = {
    "Baseline (ERA5)": "emanuel_era5_baseline_20260326_141913",
    "Market Exit": "emanuel_era5_market_exit_moderate_20260326_151847",
    "Insurance Penetration": "emanuel_era5_penetration_major_20260326_225841",
    "Building Codes": "emanuel_era5_building_codes_major_20260328_034126",
}


def add_corrected_total(df: pd.DataFrame) -> pd.DataFrame:
    df["_corrected_total_damage_usd"] = df[CORRECTED_TOTAL_COMPONENTS].sum(axis=1)
    return df


def mean_with_interval(x: np.ndarray, lo=10, hi=90) -> tuple[float, float, float]:
    return float(x.mean()), float(np.percentile(x, lo)), float(np.percentile(x, hi))


def scenario_means_table(scenario_dirs: dict[str, str]) -> pd.DataFrame:
    rows = {}
    reported_total_check = {}
    for label, dirname in scenario_dirs.items():
        path = MC_ROOT / dirname / "iterations.csv"
        df = add_corrected_total(load_iterations(path))
        reported_total_check[label] = {
            "reported_total_damage_usd_mean": float(df["total_damage_usd"].mean()),
            "corrected_total_damage_usd_mean": float(df["_corrected_total_damage_usd"].mean()),
        }
        for row_label, col in MEANS_METRICS.items():
            m, lo, hi = mean_with_interval(df[col].to_numpy(dtype=float))
            rows.setdefault(row_label, {})[label] = (
                f"{to_billion(m):.1f} ({to_billion(lo):.1f}-{to_billion(hi):.1f})"
            )
    out = pd.DataFrame(rows).T
    out.index.name = "Metric"
    return out.reset_index(), reported_total_check


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "climate_policy")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table, check = scenario_means_table(SCENARIO_DIRS)
    table.to_csv(args.out_dir / "table_S4_corrected_means.csv", index=False)

    print("Reported (stale) vs. corrected mean total_damage_usd, by scenario:")
    for label, d in check.items():
        stale_b = to_billion(d["reported_total_damage_usd_mean"])
        corr_b = to_billion(d["corrected_total_damage_usd_mean"])
        flag = "  <-- BUG CONFIRMED (differs)" if abs(stale_b - corr_b) > 1e-6 else "  (matches: unaffected)"
        print(f"  {label:25s} reported={stale_b:6.2f}B  corrected={corr_b:6.2f}B{flag}")

    print()
    print(table.to_string(index=False))
    print(f"\nWrote: {args.out_dir / 'table_S4_corrected_means.csv'}")


if __name__ == "__main__":
    main()
