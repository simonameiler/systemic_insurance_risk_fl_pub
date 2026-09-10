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

SI Table S5 exceedance probabilities are now also computed
(`scenario_probabilities_table`/`--probabilities-out`), using the threshold
definitions in SI Table S2 (verified against the submitted manuscript in
docs/earths_future_revision/correction_register.md): more-than-10 defaults,
single-entity deficit above $1B, each institution's OWN residual/shortfall
becoming positive (FHCF/FIGA/Citizens residual deficits are defined as the
amount remaining after that institution's statutory capacity is exhausted,
so "> 0" already means "exceeds capacity" for those three), NFIP claims
paid exceeding 200% of the Florida NFIP premium base, and the residual
financing requirement (FIGA + Citizens + NFIP, per this task's brief)
exceeding 1% or 10% of Florida GDP. This reuses the same season-level
indicator definitions already used for the ERA5 baseline in
scripts/earths_future_revision/section5_return_periods.py's
`gdp_exceedance`, rather than a new statistical method.

`--scenario-map <path.json>` lets callers point this script at a fresh set
of output directories (e.g. the cluster manifest's post-patch run
directories) instead of the March-archive SCENARIO_DIRS default, without
editing this file (independent review, item 4: "historical and policy
postprocessors retain March archive names").
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT_DIR, load_iterations, to_billion

REPO_ROOT = Path(__file__).resolve().parents[2]
MC_ROOT = REPO_ROOT / "results" / "mc_runs"
FLORIDA_GDP_USD = 1.7e12  # matches common.py / section5's default

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
    "Residual financing requirement": "public_burden_corrected_usd",
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
        path = _resolve_iterations_path(dirname)
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


def _load_scenario_map(path: Path | None) -> dict[str, str]:
    """Resolve the {label: run_directory} mapping this script reprocesses.

    Without --scenario-map, falls back to the hardcoded March-archive
    SCENARIO_DIRS (preserves existing behavior/tests). With it, reads an
    explicit JSON mapping (as written by scripts/cluster/earths_future.sh's
    `postprocess` command from its production manifest) so a fresh,
    post-FHCF-patch set of run directories can be reprocessed without
    editing this file.
    """
    if path is None:
        return dict(SCENARIO_DIRS)
    with open(path) as f:
        raw = json.load(f)
    # Accept either {label: dirname_under_MC_ROOT} or {label: absolute_or_relative_path}
    return raw


def _resolve_iterations_path(dirname_or_path: str) -> Path:
    p = Path(dirname_or_path)
    if p.is_absolute() or p.exists():
        candidate = p if p.name == "iterations.csv" else p / "iterations.csv"
        if candidate.exists():
            return candidate
    return MC_ROOT / dirname_or_path / "iterations.csv"


# ---------------------------------------------------------------------------
# SI Table S5 exceedance probabilities (previously deferred)
# ---------------------------------------------------------------------------

def _prob(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).mean())


def scenario_exceedance_probabilities(df: pd.DataFrame, gdp_usd: float = FLORIDA_GDP_USD) -> dict:
    """One season's worth of exceedance indicators, evaluated for every row
    of `df` (already corrected via load_iterations/add_corrected_total).
    Threshold definitions match SI Table S2; see module docstring."""
    n = len(df)
    out = {
        "Defaults > 10": _prob(df["defaults_post"].to_numpy(dtype=float) > 10),
        "Single Deficit > $1B": _prob(df["largest_entity_deficit_usd"].to_numpy(dtype=float) > 1e9),
        "FIGA > 100% Capacity": _prob(df["figa_residual_deficit_usd"].to_numpy(dtype=float) > 0),
        "Citizens > 100% Capacity": _prob(df["citizens_residual_deficit_usd"].to_numpy(dtype=float) > 0),
        "Residual financing requirement > 1% FL GDP": _prob(
            df["public_burden_corrected_usd"].to_numpy(dtype=float) > 0.01 * gdp_usd),
        "Residual financing requirement > 10% FL GDP": _prob(
            df["public_burden_corrected_usd"].to_numpy(dtype=float) > 0.10 * gdp_usd),
    }
    # FHCF > 100% Cap: prefer the explicit cap-binding flag if the run wrote
    # one; otherwise fall back to the shortfall diagnostic. Under the FHCF
    # patch (register: fhcf_contract_verification.md, item "Section 4"),
    # this is legitimately 0.0 given the current participant/premium
    # snapshot -- reported as such, not hidden or treated as a failure.
    if "fhcf_cap_binding" in df.columns:
        out["FHCF > 100% Cap"] = _prob(df["fhcf_cap_binding"].astype(bool).to_numpy())
    else:
        out["FHCF > 100% Cap"] = _prob(df["fhcf_shortfall_usd"].to_numpy(dtype=float) > 0)
    # NFIP > 200% Annual Premium: requires both columns; if either is
    # missing from this run's iterations.csv, report NaN explicitly rather
    # than silently substituting a different threshold.
    if {"nfip_claims_paid_usd", "nfip_fl_premium_base_usd"}.issubset(df.columns):
        premium = df["nfip_fl_premium_base_usd"].to_numpy(dtype=float)
        claims = df["nfip_claims_paid_usd"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["NFIP > 200% Annual Premium"] = _prob(
                np.where(premium > 0, claims > 2.0 * premium, False))
    else:
        out["NFIP > 200% Annual Premium"] = float("nan")
    out["n_seasons"] = n
    return out


# Known gap (not resolved in this pass): reprocessing the archived,
# pre-FHCF-patch ERA5 baseline with the definitions above reproduces the
# submitted manuscript's SI Table S5 closely for most metrics (Defaults>10,
# Single Deficit>$1B, NFIP>200%, Residual financing requirement>1%/10% GDP
# all match to within ~0.2 percentage points), but NOT for "FHCF > 100%
# Cap" (2.96% here vs. 0.8% submitted) or "Citizens > 100% Capacity"
# (12.72% vs. 9.5% submitted). Both use exactly the columns their names
# describe (fhcf_cap_binding, citizens_residual_deficit_usd), so this is a
# genuine open reconciliation question -- the original notebook may define
# these two thresholds differently -- not a bug in this function. Do not
# read these two rows as validated; the other rows are a reasonable
# adaptation of the existing SI Table S2 definitions, not a new method.


def scenario_probabilities_table(scenario_map: dict[str, str]) -> pd.DataFrame:
    rows = {}
    for label, dirname in scenario_map.items():
        path = _resolve_iterations_path(dirname)
        df = add_corrected_total(load_iterations(path))
        probs = scenario_exceedance_probabilities(df)
        for metric, val in probs.items():
            if metric == "n_seasons":
                continue
            rows.setdefault(metric, {})[label] = (
                f"{val*100:.2f}%" if not np.isnan(val) else "n/a (missing column)"
            )
    out = pd.DataFrame(rows).T
    out.index.name = "Metric"
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR / "climate_policy")
    ap.add_argument("--scenario-map", type=Path, default=None,
                     help="JSON {label: run_dir_or_dirname} to reprocess instead of the "
                          "hardcoded March-archive SCENARIO_DIRS.")
    ap.add_argument("--gdp", type=float, default=FLORIDA_GDP_USD)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenario_map = _load_scenario_map(args.scenario_map)

    table, check = scenario_means_table(scenario_map)
    table.to_csv(args.out_dir / "table_S4_corrected_means.csv", index=False)

    prob_table = scenario_probabilities_table(scenario_map)
    prob_table.to_csv(args.out_dir / "table_S5_corrected_probabilities.csv", index=False)

    print("Reported (stale) vs. corrected mean total_damage_usd, by scenario:")
    for label, d in check.items():
        stale_b = to_billion(d["reported_total_damage_usd_mean"])
        corr_b = to_billion(d["corrected_total_damage_usd_mean"])
        flag = "  <-- BUG CONFIRMED (differs)" if abs(stale_b - corr_b) > 1e-6 else "  (matches: unaffected)"
        print(f"  {label:25s} reported={stale_b:6.2f}B  corrected={corr_b:6.2f}B{flag}")

    print()
    print(table.to_string(index=False))
    print()
    print(prob_table.to_string(index=False))
    print(f"\nWrote: {args.out_dir / 'table_S4_corrected_means.csv'}")
    print(f"Wrote: {args.out_dir / 'table_S5_corrected_probabilities.csv'}")


if __name__ == "__main__":
    main()
