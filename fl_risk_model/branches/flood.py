"""
flood.py — NFIP (flood) branch
------------------------------

Implements the flood side of the risk propagation model, aligned with the
Methods description:

1) Load per-county **water damage** from a scenario CSV.
2) Join with NFIP **coverage in force** (FloodTIV) to form **gross flood losses**,
   with a capped variant (min[damage, coverage]).
3) Apply **NFIP recoveries** using county payout rates derived from historical
   claims (provided in `rates_df`), capping insured flood by coverage first.

Design notes
------------
- All monetary values are USD.
- County labels are normalized via `norm_county_name` for joins (except where
  the original behavior only strips whitespace).
- Function names, parameters, defaults, and core logic are preserved.

Public API
----------
- load_water_damage_scenario
- build_gross_flood_losses
- apply_nfip_recovery_from_rates
"""

from __future__ import annotations
from typing import Optional
import os
import io
import re
from pathlib import Path
import numpy as np
import pandas as pd

from fl_risk_model.config import DATA_DIR
from fl_risk_model.utils import norm_county_name

from fl_risk_model import config as cfg

__all__ = [
    "load_water_damage_scenario",
    "build_gross_flood_losses",
    "apply_nfip_recovery_from_rates",
]


# -----------------------------------------------------------------------------
# 1) Load scenario water damage
# -----------------------------------------------------------------------------

def _read_hazard_total_loss(storm_name: str) -> pd.DataFrame:
    hazard_path = Path(cfg.EVENT_REPORTS_DIR) / f"{storm_name}.csv"
    if not hazard_path.exists():
        raise FileNotFoundError(f"Total-loss hazard file not found: {hazard_path}")
    raw = pd.read_csv(hazard_path)
    low = {c.lower(): c for c in raw.columns}
    county_col = low.get("county_name") or low.get("county") or low.get("name")
    value_col  = low.get("value") or low.get("totallossusd") or low.get("loss_usd") or low.get("loss")
    if county_col is None or value_col is None:
        raise ValueError(f"{hazard_path.name}: need county + total loss columns; got {list(raw.columns)}")
    df = raw[[county_col, value_col]].rename(columns={county_col: "County", value_col: "TotalLossUSD"}).copy()
    df["County"] = df["County"].map(norm_county_name)
    df["TotalLossUSD"] = pd.to_numeric(df["TotalLossUSD"], errors="coerce").fillna(0.0)
    return df

def load_water_damage_scenario(
    storm_name: str,
    *,
    data_dir: str | Path | None = None,
    county_col: str = "County",
    water_col: str = "water_damage_usd",
) -> pd.DataFrame:
    """
    Preferred: derive water losses from total-loss hazard CSV using water_share = 1 - wind_share.
    Fallback: existing split file 'hurricane_{storm_name}_damage_split.csv'.
    """
    # 1) Prefer total-loss CSV and split by (1 - wind_share)
    try:
        df = _read_hazard_total_loss(storm_name)
        wind_share = cfg.RUNTIME_WIND_SHARE_OVERRIDES.get(storm_name)
        if wind_share is None:
            a, b = cfg.EVENT_WIND_SHARE_BOUNDS.get(storm_name, cfg.DEFAULT_WIND_SHARE_BOUNDS)
            wind_share = 0.5 * (float(a) + float(b))
        water_share = 1.0 - float(wind_share)
        df["WaterDamageUSD"] = df["TotalLossUSD"] * water_share
        return df[["County", "WaterDamageUSD"]]
    except FileNotFoundError:
        pass  # try fallback below

    # 2) Fallback to legacy split file if present
    data_dir = Path(data_dir or cfg.DATA_DIR)
    path = data_dir / f"hurricane_{storm_name}_damage_split.csv"
    if not path.exists():
        raise FileNotFoundError(f"Neither hazard total-loss nor split water file found for storm {storm_name}")
    s = pd.read_csv(path)
    # Flexible column match
    lower = {c.lower(): c for c in s.columns}
    ccol = lower.get(county_col.lower(), county_col)
    wcol = lower.get(water_col.lower(), water_col)
    s = s[[ccol, wcol]].rename(columns={ccol: "County", wcol: "WaterDamageUSD"}).copy()
    s["County"] = s["County"].map(norm_county_name)
    s["WaterDamageUSD"] = pd.to_numeric(s["WaterDamageUSD"], errors="coerce").fillna(0.0)
    return s[["County", "WaterDamageUSD"]]


# -----------------------------------------------------------------------------
# 2) Build gross flood losses
# -----------------------------------------------------------------------------
def build_gross_flood_losses(
    nfip_exposure_df: pd.DataFrame,
    water_damage_df: pd.DataFrame,
    *,
    cap_by_exposure: bool = False,
) -> pd.DataFrame:
    """
    Merge NFIP county coverage with scenario water damage to produce **gross flood losses**.
    - If 'FloodTIV_sampled' exists, use it for capping; else use 'FloodTIV'.
    """
    # Validate schema
    req1 = {"County", "FloodTIV"}            # FloodTIV_sampled is optional
    req2 = {"County", "WaterDamageUSD"}
    if not req1.issubset(nfip_exposure_df.columns):
        missing = sorted(req1 - set(nfip_exposure_df.columns))
        raise ValueError(f"nfip_exposure_df missing columns: {missing}")
    if not req2.issubset(water_damage_df.columns):
        missing = sorted(req2 - set(water_damage_df.columns))
        raise ValueError(f"water_damage_df missing columns: {missing}")

    # Clean & harmonize
    exp = nfip_exposure_df.copy()
    exp["County"] = exp["County"].map(norm_county_name)
    for col in ("FloodTIV", "FloodTIV_sampled"):
        if col in exp.columns:
            exp[col] = pd.to_numeric(exp[col], errors="coerce").fillna(0.0)

    water = water_damage_df.copy()
    water["County"] = water["County"].map(norm_county_name)
    water["WaterDamageUSD"] = pd.to_numeric(water["WaterDamageUSD"], errors="coerce").fillna(0.0)

    df = exp.merge(water, on="County", how="left")
    df["WaterDamageUSD"] = df["WaterDamageUSD"].fillna(0.0)

    # Base gross flood loss = scenario water damage
    df["FloodLossUSD"] = df["WaterDamageUSD"]

    # Capped variant (policy ceiling proxy): use sampled cap if available
    cap_col = "FloodTIV_sampled" if "FloodTIV_sampled" in df.columns else "FloodTIV"
    df["FloodLossUSD_capped"] = np.minimum(df["FloodLossUSD"], df[cap_col])

    if cap_by_exposure:
        df["FloodLossUSD"] = df["FloodLossUSD_capped"]
    
    # add before return:
    df["FloodTIV_used_for_cap"] = df[cap_col]
    cols = ["County", "FloodTIV", "FloodTIV_used_for_cap", "WaterDamageUSD", "FloodLossUSD", "FloodLossUSD_capped"]
    return df[cols]

# -----------------------------------------------------------------------------
# 3) NFIP recovery
# -----------------------------------------------------------------------------
def apply_nfip_recovery_from_rates(
    carved_flood_df: pd.DataFrame,
    flood_tiv_df: pd.DataFrame,
    rates_df: pd.DataFrame,
    county_xwalk: pd.DataFrame,
    fallback_rate: float = 0.25,
) -> pd.DataFrame:
    """
    Apply NFIP payout rates to **insured flood** amounts (post uninsured carve-out).

    Parameters
    ----------
    carved_flood_df : pandas.DataFrame
        Must include:
          - 'County' : str
          - 'InsuredFloodUSD' : float
    flood_tiv_df : pandas.DataFrame
        Must include:
          - 'County' : str
          - 'FloodTIV' : float
        May include:
          - 'FloodTIV_sampled' : float
    rates_df : pandas.DataFrame
        Must include:
          - 'county_fips' : str or int
          - 'nfip_payout_rate' : float in [0,1]
    county_xwalk : pandas.DataFrame
        Crosswalk with:
          - 'County' : str
          - 'county_fips' : str or int
    fallback_rate : float, default 0.25
        Rate to apply where a county has no rate; clipped to [0,1].

    Returns
    -------
    pandas.DataFrame
        Input `carved_flood_df` with added columns:
          - 'county_fips' : str (from crosswalk)
          - 'FloodTIV' : float (from `flood_tiv_df`)
          - 'FloodTIV_sampled' : float (if present in `flood_tiv_df`)
          - 'nfip_payout_rate' : float in [0,1] (with fallback applied)
          - 'NFIPRecoveryUSD' : float

    Notes
    -----
    - The recovery calculation caps **insured flood** by available NFIP coverage:
        cap = FloodTIV_sampled if present else FloodTIV
        base = min(InsuredFloodUSD, cap)
        NFIPRecoveryUSD = base × rate
    - Consistent with Methods: payout rates are exogenous inputs already processed
      from historical claims (possibly using recency weighting & EB shrinkage).
    """
    # Copies
    L = carved_flood_df.copy()
    E = flood_tiv_df.copy()
    X = county_xwalk.copy()
    R = rates_df.copy()

    # Normalize county names (use the project-wide helper if available)
    L["County"] = L["County"].astype(str).map(norm_county_name)
    E["County"] = E["County"].astype(str).map(norm_county_name)
    X["County"] = X["County"].astype(str).map(norm_county_name)

    # Ensure numeric
    L["InsuredFloodUSD"] = pd.to_numeric(L["InsuredFloodUSD"], errors="coerce").fillna(0.0)
    for col in ("FloodTIV", "FloodTIV_sampled"):
        if col in E.columns:
            E[col] = pd.to_numeric(E[col], errors="coerce").fillna(0.0)

    # Attach FIPS and FloodTIV (+ sampled if present)
    L = L.merge(X[["County", "county_fips"]], on="County", how="left")
    tiv_cols = ["County", "FloodTIV"] + (["FloodTIV_sampled"] if "FloodTIV_sampled" in E.columns else [])
    L = L.merge(E[tiv_cols], on="County", how="left")

    # Coerce FIPS to string on both sides for a clean join
    L["county_fips"] = L["county_fips"].astype(str)
    R["county_fips"] = R["county_fips"].astype(str)

    # Attach rate with fallback and clip into [0,1]
    L = L.merge(R[["county_fips", "nfip_payout_rate"]], on="county_fips", how="left")
    L["nfip_payout_rate"] = (
        pd.to_numeric(L["nfip_payout_rate"], errors="coerce")
        .fillna(float(fallback_rate))
        .clip(0, 1)
    )

    # Cap insured flood by coverage (prefer sampled cap if available), then apply rate
    cap_col = "FloodTIV_sampled" if "FloodTIV_sampled" in L.columns else "FloodTIV"
    L[cap_col] = pd.to_numeric(L[cap_col], errors="coerce").fillna(0.0)
    base = np.minimum(L["InsuredFloodUSD"].astype(float), L[cap_col].astype(float))
    L["NFIPRecoveryUSD"] = (base * L["nfip_payout_rate"]).astype(float)

    return L
