#!/usr/bin/env python3
"""Section 7: NFIP structure-to-loss allocation sensitivity (county rates).

Builds the three county-level flood-loss allocation configurations requested
by the revision brief, using the same FEMA NFIP Residential Penetration
Rates dataset already used by the production model
(fl_risk_model/data/NfipResidentialPenetrationRates.csv, asOfDate
2025-05-15, filtered to Florida):

  1. baseline           -- structure-weighted effective rate tau_c, as used
                            in production (fl_risk_model.nfip.load_nfip_penetration).
  2. sfha_only          -- county flood loss allocated entirely inside SFHAs,
                            using the county's SFHA participation rate.
  3. non_sfha_only      -- county flood loss allocated entirely outside SFHAs,
                            using the county's inferred non-SFHA rate.
  4. envelope           -- per-county min/max of the three valid rates above,
                            restricted to zones that exist in the county
                            (s_sfha in (0,1)); NOT automatically labeled an
                            upper/lower bound, since the ordering of
                            r_sfha vs. r_non differs across counties.

This script verifies that the baseline rate lies within the per-county
envelope (as it must by construction, tau_c = s*r_sfha + (1-s)*r_non is a
convex combination of r_sfha and r_non) and reports the county rate table.

Quantifying the resulting change in insured flood losses, NFIP financing
requirements, and corrected aggregate burden requires re-running the
financial model on the same per-county event losses used in the archived
production run. That per-county, per-event flood loss cache
(fl_risk_model/data/hazard/emanuel/ and fl_risk_model/data/hazard/gori_data/)
is not distributed in this repository (see
docs/earths_future_revision/data_inventory.md) and could not be reproduced
in this environment. This script documents the exact remaining command
(--rerun-financial-model) that applies once that cache is available; without
it, the three allocation configurations cannot be scored against the
probabilistic catalog and are reported as county-level rate tables only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
PENETRATION_CSV = REPO_ROOT / "fl_risk_model" / "data" / "NfipResidentialPenetrationRates.csv"


def build_county_rate_table(path: Path = PENETRATION_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["state"] == "Florida"].copy()

    n_all = pd.to_numeric(df["totalResStructures"], errors="coerce")
    n_sfha = pd.to_numeric(df["totalResStructuresSfha"], errors="coerce")
    r_all = pd.to_numeric(df["resPenetrationRate"], errors="coerce").clip(0, 1)
    r_sfha = pd.to_numeric(df["resPenetrationRateSfha"], errors="coerce").clip(0, 1)

    s_sfha = (n_sfha / n_all).clip(lower=0, upper=1)
    s_sfha = s_sfha.fillna(0.0)

    denom = (1 - s_sfha).replace(0, np.nan)
    r_non = ((r_all - s_sfha * r_sfha) / denom).clip(lower=0, upper=1)
    r_non = r_non.fillna(0.0)

    tau_baseline = (s_sfha * r_sfha.fillna(0.0) + (1 - s_sfha) * r_non).clip(0, 1)

    out = pd.DataFrame({
        "county": df["county"].values,
        "county_fips": df["county_fips"].values,
        "n_res_structures_all": n_all.values,
        "n_res_structures_sfha": n_sfha.values,
        "s_sfha_share_of_stock": s_sfha.values,
        "rate_all_county_reported": r_all.values,
        "rate_sfha_reported": r_sfha.values,
        "rate_non_sfha_inferred": r_non.values,
        "rate_baseline_structure_weighted": tau_baseline.values,
    })

    # sfha_only / non_sfha_only allocation configurations, restricted to
    # zones that exist in the county (SFHA share strictly between 0 and 1;
    # a county with s_sfha == 0 has no SFHA zone, so "sfha_only" is undefined
    # there, and a county with s_sfha == 1 has no non-SFHA zone).
    out["rate_sfha_only_config"] = np.where(out["s_sfha_share_of_stock"] > 0.0,
                                             out["rate_sfha_reported"], np.nan)
    out["rate_non_sfha_only_config"] = np.where(out["s_sfha_share_of_stock"] < 1.0,
                                                 out["rate_non_sfha_inferred"], np.nan)

    valid_rates = out[["rate_sfha_only_config", "rate_non_sfha_only_config",
                        "rate_baseline_structure_weighted"]]
    out["envelope_min"] = valid_rates.min(axis=1, skipna=True)
    out["envelope_max"] = valid_rates.max(axis=1, skipna=True)
    out["baseline_within_envelope"] = (
        (out["rate_baseline_structure_weighted"] >= out["envelope_min"] - 1e-9)
        & (out["rate_baseline_structure_weighted"] <= out["envelope_max"] + 1e-9)
    )
    # Which config is larger varies by county -- explicitly do not label
    # either "upper" or "lower" bound.
    out["sfha_rate_exceeds_non_sfha_rate"] = (
        out["rate_sfha_only_config"] > out["rate_non_sfha_only_config"]
    )

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--penetration-csv", type=Path, default=PENETRATION_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "nfip_allocation")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = build_county_rate_table(args.penetration_csv)
    out_path = args.out_dir / "county_flood_allocation_configurations.csv"
    table.to_csv(out_path, index=False)

    n_bad = int((~table["baseline_within_envelope"]).sum())
    n_ordering_mixed = table["sfha_rate_exceeds_non_sfha_rate"].nunique()

    print(f"Florida counties: {len(table)}")
    print(f"Counties where baseline rate falls OUTSIDE the [min,max] envelope: {n_bad} (expected 0)")
    print(f"Rate ordering (SFHA rate > non-SFHA rate) varies across counties: "
          f"{n_ordering_mixed > 1} "
          f"({int(table['sfha_rate_exceeds_non_sfha_rate'].sum())} of {len(table)} counties have SFHA > non-SFHA)")
    print(f"\nWrote: {out_path}")
    print(
        "\nBLOCKED for financial-model rerun: scoring these three allocation\n"
        "configurations against corrected aggregate burden, insured flood\n"
        "losses, and NFIP financing requires the per-county, per-event flood\n"
        "loss cache used by fl_risk_model.runner (fl_risk_model/data/hazard/\n"
        "emanuel/ and .../gori_data/), which contains only .gitkeep\n"
        "placeholders in this checkout (proprietary WindRiskTech / Gori et al.\n"
        "inputs, see docs/earths_future_revision/data_inventory.md).\n\n"
        "Remaining work once that cache is restored (no CLI flag for this\n"
        "exists yet -- fl_risk_model.runner.run_one_scenario currently reads a\n"
        "single county rate column, 'NFIP_r_eff', from fl_risk_model.nfip."
        "load_nfip_penetration):\n"
        "  1. Add an --nfip-allocation {baseline,sfha_only,non_sfha_only}\n"
        "     argument to scripts/run/run_stochastic_tc_monte_carlo.py that\n"
        "     selects which column of county_flood_allocation_configurations.csv\n"
        "     (rate_baseline_structure_weighted / rate_sfha_only_config /\n"
        "     rate_non_sfha_only_config) is substituted for NFIP_r_eff before\n"
        "     calling fl_risk_model.nfip.carveout_flood_from_penetration.\n"
        "  2. Re-run scripts/run/run_stochastic_tc_monte_carlo.py three times\n"
        "     (same 10,000-season ERA5 event draws, same seed=42, same\n"
        "     capital/coverage caps) with each allocation, writing to\n"
        "     results/earths_future_revision/nfip_allocation/iterations_<config>.csv.\n"
        "  3. Re-run scripts/earths_future_revision/section5_return_periods.py\n"
        "     and section6_decomposition.py on each output to obtain corrected\n"
        "     insured flood losses, NFIP financing, and burden metrics.\n"
        "  For zero-count zones and missing rates, carry the current model's\n"
        "  treatment unchanged (missing/undefined rates set to 0, per\n"
        "  Supporting Text S3): a county with no SFHA stock (s_sfha==0) has an\n"
        "  undefined 'sfha_only' rate and is assigned 0 insured flood loss\n"
        "  under that configuration; symmetrically for s_sfha==1 under\n"
        "  'non_sfha_only'."
    )


if __name__ == "__main__":
    main()
