from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

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
