#!/usr/bin/env python3
"""Local FHCF pilot: paired old-vs-corrected comparison, with each of the
two confirmed defects isolated separately, on real (non-synthetic)
historical hazard inputs.

This is NOT a cluster run. Cluster access to the Stanford Sherlock HPC
cluster (where the proprietary WindRiskTech/Emanuel synthetic TC event sets
live) requires interactive Duo two-factor authentication that this session
cannot perform; TCP connectivity to login.sherlock.stanford.edu:22 was
confirmed reachable, but no login was attempted and no cluster job was
submitted or invented. Consequently, the ERA5 baseline (10,000 synthetic
seasons) is NOT included in this pilot -- it requires exactly the hazard
cache this environment does not have (see docs/earths_future_revision/
data_inventory.md). What IS included are the eight historical/sequential
scenarios, which use real, non-proprietary, git-tracked county-level wind
and flood damage inputs (fl_risk_model/data/hazard/historical_events/) and
therefore let the actual production pipeline
(fl_risk_model.mc_run_events.run_one_iteration ->
fl_risk_model.runner.run_one_scenario -> fl_risk_model.fhcf.apply_fhcf_recovery)
run end to end with real inputs. Do not describe this as an ERA5 baseline
replay.

Four variants of apply_fhcf_recovery are monkeypatched in turn, so the same
production call path (runner.py, mc_run_events.py, branches/citizens.py) is
exercised unchanged for every variant, isolating exactly the FHCF formula:

  old_both_bugs         : the original, pre-patch production code (row-wise
                           granularity, cap applied BEFORE coverage/LAE scaling)
  formula_fix_only       : cap applied AFTER scaling, but still row-wise
                           (no company aggregation) -- isolates defect 1 alone
  aggregation_fix_only   : company-aggregated first, but cap still applied
                           BEFORE scaling -- isolates defect 2 alone
  both_fixed             : the current fl_risk_model.fhcf.apply_fhcf_recovery
                           (company-aggregated, cap after scaling) -- what
                           is now in production on this branch

All four variants are run with the SAME seed, iteration count, and scenario,
through the SAME entry point (run_historical_scenarios_mc.run_scenario_mc),
so any difference is attributable solely to the FHCF calculation.

Usage
-----
    conda activate climada_env
    python scripts/earths_future_revision/fhcf_pilot_isolate_fixes.py \\
        --scenarios great_miami andrew lake_okeechobee irma \\
                    gm_then_andrew double_gm double_irma \\
        --n-iter 100 --seed 42 \\
        --out-root results/earths_future_revision/fhcf_pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import fl_risk_model.fhcf as fhcf_mod
import fl_risk_model.runner as runner_mod
import fl_risk_model.branches.citizens as citizens_mod
from fl_risk_model.config import FHCF_LAE_FACTOR


def _require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns {missing}")


def _prep(loss_df, terms_df):
    df = loss_df.copy()
    df["GrossWindLossUSD"] = pd.to_numeric(df["GrossWindLossUSD"], errors="coerce").fillna(0.0)
    t = terms_df[["Company", "CoveragePct_norm", "RetentionUSD", "LimitUSD"]].copy()
    for c in ["CoveragePct_norm", "RetentionUSD", "LimitUSD"]:
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(0.0)
    return df, t


def apply_fhcf_recovery_old_both_bugs(loss_df: pd.DataFrame, terms_df: pd.DataFrame) -> pd.DataFrame:
    """Original production formula: row-wise granularity (no company
    aggregation), cap applied to the excess BEFORE coverage/LAE scaling."""
    _require_cols(terms_df, ["Company", "CoveragePct_norm", "RetentionUSD", "LimitUSD"])
    _require_cols(loss_df, ["Company", "GrossWindLossUSD"])
    df, t = _prep(loss_df, terms_df)
    df = df.merge(t, on="Company", how="left")
    df["ExcessUSD"] = (df["GrossWindLossUSD"] - df["RetentionUSD"].fillna(0.0)).clip(lower=0.0)
    df["RecoverableUSD"] = df[["ExcessUSD", "LimitUSD"]].min(axis=1).fillna(0.0)
    coverage_frac = (df["CoveragePct_norm"].fillna(0.0) / 100.0).clip(lower=0.0, upper=1.0)
    df["RecoveryUSD"] = (df["RecoverableUSD"] * coverage_frac * float(FHCF_LAE_FACTOR)).astype(float)
    df["NetWindUSD"] = df["GrossWindLossUSD"] - df["RecoveryUSD"]
    return df


def apply_fhcf_recovery_formula_fix_only(loss_df: pd.DataFrame, terms_df: pd.DataFrame) -> pd.DataFrame:
    """Defect 1 fixed alone: cap applied AFTER scaling, but still row-wise
    (no company aggregation) -- isolates the formula-order effect from the
    aggregation effect."""
    _require_cols(terms_df, ["Company", "CoveragePct_norm", "RetentionUSD", "LimitUSD"])
    _require_cols(loss_df, ["Company", "GrossWindLossUSD"])
    df, t = _prep(loss_df, terms_df)
    df = df.merge(t, on="Company", how="left")
    df["ExcessUSD"] = (df["GrossWindLossUSD"] - df["RetentionUSD"].fillna(0.0)).clip(lower=0.0)
    coverage_frac = (df["CoveragePct_norm"].fillna(0.0) / 100.0).clip(lower=0.0, upper=1.0)
    scaled = df["ExcessUSD"] * coverage_frac * float(FHCF_LAE_FACTOR)
    df["RecoveryUSD"] = pd.concat([scaled, df["LimitUSD"]], axis=1).min(axis=1).fillna(0.0)
    df["NetWindUSD"] = df["GrossWindLossUSD"] - df["RecoveryUSD"]
    return df


def apply_fhcf_recovery_aggregation_fix_only(loss_df: pd.DataFrame, terms_df: pd.DataFrame) -> pd.DataFrame:
    """Defect 2 fixed alone: company-aggregated first, but the cap is still
    applied BEFORE scaling (old order) -- isolates the aggregation effect
    from the formula-order effect."""
    _require_cols(terms_df, ["Company", "CoveragePct_norm", "RetentionUSD", "LimitUSD"])
    _require_cols(loss_df, ["Company", "GrossWindLossUSD"])
    df, t = _prep(loss_df, terms_df)
    company_totals = (
        df.groupby("Company", as_index=False)["GrossWindLossUSD"]
        .sum().rename(columns={"GrossWindLossUSD": "CompanyGrossWindLossUSD"})
    )
    company_totals = company_totals.merge(t, on="Company", how="left")
    company_totals["RetentionUSD"] = company_totals["RetentionUSD"].fillna(0.0)
    company_totals["LimitUSD"] = company_totals["LimitUSD"].fillna(0.0)
    coverage_frac = (company_totals["CoveragePct_norm"].fillna(0.0) / 100.0).clip(lower=0.0, upper=1.0)
    company_totals["CompanyExcessUSD"] = (
        company_totals["CompanyGrossWindLossUSD"] - company_totals["RetentionUSD"]
    ).clip(lower=0.0)
    company_totals["CompanyRecoverableUSD"] = (
        pd.concat([company_totals["CompanyExcessUSD"], company_totals["LimitUSD"]], axis=1)
        .min(axis=1).fillna(0.0)
    )
    company_totals["CompanyRecoveryUSD"] = (
        company_totals["CompanyRecoverableUSD"] * coverage_frac * float(FHCF_LAE_FACTOR)
    )
    df = df.merge(
        company_totals[["Company", "CompanyGrossWindLossUSD", "CompanyRecoveryUSD"]],
        on="Company", how="left",
    )
    df["CompanyGrossWindLossUSD"] = df["CompanyGrossWindLossUSD"].fillna(0.0)
    df["CompanyRecoveryUSD"] = df["CompanyRecoveryUSD"].fillna(0.0)
    row_share = np.where(df["CompanyGrossWindLossUSD"] > 0,
                          df["GrossWindLossUSD"] / df["CompanyGrossWindLossUSD"], 0.0)
    df["RecoveryUSD"] = (df["CompanyRecoveryUSD"] * row_share).astype(float)
    df["NetWindUSD"] = df["GrossWindLossUSD"] - df["RecoveryUSD"]
    return df


VARIANTS = {
    "old_both_bugs": apply_fhcf_recovery_old_both_bugs,
    "formula_fix_only": apply_fhcf_recovery_formula_fix_only,
    "aggregation_fix_only": apply_fhcf_recovery_aggregation_fix_only,
    "both_fixed": fhcf_mod.apply_fhcf_recovery,  # current production code, unmodified
}


def _set_variant(fn):
    fhcf_mod.apply_fhcf_recovery = fn
    runner_mod.apply_fhcf_recovery = fn
    citizens_mod.apply_fhcf_recovery = fn


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _input_manifest() -> dict:
    """Checksums of every input the paired comparison depends on, so a
    reader can confirm old and new runs used identical inputs."""
    manifest = {}
    hazard_dir = REPO_ROOT / "fl_risk_model" / "data" / "hazard" / "historical_events"
    for p in sorted(hazard_dir.glob("*.csv")):
        manifest[f"hazard/{p.name}"] = _file_sha256(p)
    for fname in [
        "24fin_fhcf.csv", "fhcf_terms_keyed.csv", "company_keys.csv",
        "FL HO Market Share Report_6.10.25.xlsx",
        "20250805 FL Surplus Capital, Group v Entity.xlsx",
        "FHCF_2024_Exposure_byCounty.xlsx",
    ]:
        p = REPO_ROOT / "fl_risk_model" / "data" / fname
        if p.exists():
            manifest[f"data/{fname}"] = _file_sha256(p)
        else:
            manifest[f"data/{fname}"] = "MISSING"
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["great_miami"])
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path,
                     default=REPO_ROOT / "results" / "earths_future_revision" / "fhcf_pilot")
    args = ap.parse_args()

    from scripts.run.run_historical_scenarios_mc import run_scenario_mc  # local import after sys.path insert

    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "code_revision_git_describe": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "describe", "--always", "--dirty"],
            capture_output=True, text=True
        ).stdout.strip(),
        "seed": args.seed,
        "n_iter": args.n_iter,
        "scenarios": args.scenarios,
        "input_checksums_sha256_16": _input_manifest(),
    }
    with open(args.out_root / "pilot_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

    summary_rows = []
    for scenario in args.scenarios:
        for variant_name, variant_fn in VARIANTS.items():
            _set_variant(variant_fn)
            out_dir = args.out_root / variant_name
            print(f"\n=== scenario={scenario} variant={variant_name} ===")
            run_dir = run_scenario_mc(scenario, n_iter=args.n_iter, seed=args.seed, out_dir=out_dir)
            if run_dir is None:
                print(f"[SKIP] {scenario}/{variant_name} failed; see traceback above")
                continue
            it = pd.read_csv(Path(run_dir) / "iterations.csv")
            row = {"scenario": scenario, "variant": variant_name, "n": len(it)}
            for col in [
                "total_damage_usd", "wind_total_usd", "water_total_usd",
                "fhcf_total_precap_usd", "fhcf_total_postcap_usd", "fhcf_shortfall_usd",
                "fhcf_cap_binding", "fhcf_recovery_private_usd", "fhcf_recovery_citizens_usd",
                "catbond_payout_usd", "defaults_post",
                "figa_residual_deficit_usd", "citizens_residual_deficit_usd",
                "nfip_borrowed_usd", "largest_entity_deficit_usd",
            ]:
                if col in it.columns:
                    row[f"{col}_mean"] = float(pd.to_numeric(it[col], errors="coerce").mean())
            row["residual_financing_requirement_mean"] = (
                row.get("figa_residual_deficit_usd_mean", 0.0)
                + row.get("citizens_residual_deficit_usd_mean", 0.0)
                + row.get("nfip_borrowed_usd_mean", 0.0)
            )
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_root / "pilot_comparison_summary.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n\n=== PILOT COMPARISON SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nWrote: {args.out_root / 'pilot_comparison_summary.csv'}")
    print(f"Wrote: {args.out_root / 'pilot_manifest.json'}")


if __name__ == "__main__":
    main()
