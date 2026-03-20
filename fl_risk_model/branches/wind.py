"""
wind.py - Private wind branch: gross losses, insured allocation, FHCF recovery
-----------------------------------------------------------------------------

Implements the wind side of the risk propagation model for private carriers.
This module is intentionally thin and delegates policy math to shared FHCF
utilities. It aligns with the Methods description:

- Load county-level wind damage for a given storm.
- Allocate county wind damage to companies by exposure share (TIV_sampled
  if present, else TIV).
- Apply FHCF recoveries using normalized per-company terms (retention,
  limit, coverage %, LAE) via the shared recovery engine.
- Return gross, recovery, and net wind losses by ['Company','County'].

Design notes
------------
- Input exposure frames are assumed to be the private-company × county slices
  from the exposure assembly, containing 'TIV' and optionally 'TIV_sampled'.
- County names are normalized using the same utility as elsewhere to ensure
  joins behave deterministically.

Public API
----------
- load_wind_damage
- gross_wind_loss
- allocate_insured_wind_to_private
- recover_wind_loss
- net_wind_loss
"""

from __future__ import annotations

import os
import pandas as pd
from pathlib import Path

from fl_risk_model.config import FHCF_TERMS_CSV, DATA_DIR
from fl_risk_model.fhcf import normalize_fhcf_terms, apply_fhcf_recovery
from fl_risk_model.utils import norm_county_name

from fl_risk_model import config as cfg

__all__ = [
    "load_wind_damage",
    "gross_wind_loss",
    "allocate_insured_wind_to_private",
    "recover_wind_loss",
    "net_wind_loss",
]

# =============================================================================
# Load county wind damage
# =============================================================================

def _read_hazard_total_loss(storm_name: str) -> pd.DataFrame:
    """Load total-loss hazard CSV: columns {county_name, value} (or flexible variants)."""
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

def load_wind_damage(storm_name: str, *, data_dir: str = None) -> pd.DataFrame:
    """
    Preferred: derive wind losses from total-loss hazard CSV using wind_share.
    Fallback: existing split file 'hurricane_{storm_name}_damage_split.csv'.
    """
    # 1) Prefer total-loss CSV and split by wind_share
    try:
        df = _read_hazard_total_loss(storm_name)
        wind_share = cfg.RUNTIME_WIND_SHARE_OVERRIDES.get(storm_name)
        if wind_share is None:
            a, b = cfg.EVENT_WIND_SHARE_BOUNDS.get(storm_name, cfg.DEFAULT_WIND_SHARE_BOUNDS)
            wind_share = 0.5 * (float(a) + float(b))  # midpoint if not overridden this iteration
        df["WindDamageUSD"] = df["TotalLossUSD"] * float(wind_share)
        return df[["County", "WindDamageUSD"]]
    except FileNotFoundError:
        pass  # try fallback below

    # 2) Fallback to legacy split file if present
    data_dir = data_dir or str(cfg.DATA_DIR)
    split_path = os.path.join(data_dir, f"hurricane_{storm_name}_damage_split.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Neither hazard total-loss nor split wind file found for storm {storm_name}")
    s = pd.read_csv(split_path, usecols=["County", "wind_damage_usd"])
    s["County"] = s["County"].map(norm_county_name)
    s["WindDamageUSD"] = pd.to_numeric(s["wind_damage_usd"], errors="coerce").fillna(0.0)
    return s[["County", "WindDamageUSD"]]
# =============================================================================
# Gross losses & insured allocation
# =============================================================================

def gross_wind_loss(exp_df: pd.DataFrame, storm_name: str) -> pd.DataFrame:
    """
    Allocate county wind damage to companies by exposure share to get gross loss.

    Parameters
    ----------
    exp_df : pandas.DataFrame
        Exposure for private companies by county. Must contain:
        ['Company','County'] and either 'TIV_sampled' or 'TIV'.
    storm_name : str
        Storm identifier passed to `load_wind_damage`.

    Returns
    -------
    pandas.DataFrame
        Columns:
          - Company : str
          - County : str
          - GrossWindLossUSD : float

    Raises
    ------
    ValueError
        If neither 'TIV_sampled' nor 'TIV' is present.

    Notes
    -----
    Allocation is proportional within each county:
        share_i = weight_i / sum_j weight_j,
    where weight = TIV_sampled if available else TIV.
    """
    wind_df = load_wind_damage(storm_name)
    df = exp_df.merge(wind_df, on="County", how="left")

    weight_col = (
        "TIV_sampled" if "TIV_sampled" in df.columns
        else ("TIV" if "TIV" in df.columns else None)
    )
    if weight_col is None:
        raise ValueError("exp_df must contain 'TIV' or 'TIV_sampled'.")

    county_totals = df.groupby("County")[weight_col].transform("sum")
    share = (df[weight_col] / county_totals).where(county_totals > 0, other=0.0)
    df["GrossWindLossUSD"] = share * df["WindDamageUSD"].fillna(0.0)

    return df[["Company", "County", "GrossWindLossUSD"]]

def allocate_insured_wind_to_private(
    exp_df: pd.DataFrame,
    insured_wind_by_county: pd.DataFrame
) -> pd.DataFrame:
    """
    Allocate county insured wind amounts to private companies by exposure share.

    Parameters
    ----------
    exp_df : pandas.DataFrame
        Private company exposure by county with ['Company','County'] and
        'TIV_sampled' or 'TIV'.
    insured_wind_by_county : pandas.DataFrame
        County totals after uninsured carve-out. Must contain:
        ['County','InsuredWindUSD'].

    Returns
    -------
    pandas.DataFrame
        Columns:
          - Company : str
          - County : str
          - InsuredWindUSD_alloc : float

    Raises
    ------
    ValueError
        If neither 'TIV_sampled' nor 'TIV' is present in `exp_df`.
    """
    df = exp_df.merge(insured_wind_by_county, on="County", how="left").copy()
    df["InsuredWindUSD"] = df["InsuredWindUSD"].fillna(0.0)

    weight_col = (
        "TIV_sampled" if "TIV_sampled" in df.columns
        else ("TIV" if "TIV" in df.columns else None)
    )
    if weight_col is None:
        raise ValueError("exp_df must contain 'TIV' or 'TIV_sampled'.")

    county_totals = df.groupby("County")[weight_col].transform("sum")
    share = (df[weight_col] / county_totals).where(county_totals > 0, other=0.0)
    df["InsuredWindUSD_alloc"] = share * df["InsuredWindUSD"]

    return df[["Company", "County", "InsuredWindUSD_alloc"]]

# =============================================================================
# FHCF recovery & net losses
# =============================================================================

def recover_wind_loss(
    gross_df: pd.DataFrame,
    fhcf_terms_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply FHCF recoveries to private carriers' gross wind losses.

    Parameters
    ----------
    gross_df : pandas.DataFrame
        Must contain ['Company','County','GrossWindLossUSD'].
    fhcf_terms_df : pandas.DataFrame or None, default None
        Optional per-company FHCF terms. If None, read from FHCF_TERMS_CSV.
        Expected at minimum: ['Company','CoveragePct'].
        If premium/retention/limit inputs are missing, `normalize_fhcf_terms`
        should zero-fill those, effectively reducing to coverage-% only.

    Returns
    -------
    pandas.DataFrame
        Columns:
          - Company : str
          - County : str
          - GrossWindLossUSD : float
          - RecoveryUSD : float
          - NetWindUSD : float

    Raises
    ------
    ValueError
        If `gross_df` lacks required columns.

    Notes
    -----
    Uses `normalize_fhcf_terms` and `apply_fhcf_recovery` from the shared FHCF
    utilities; statewide seasonal cap and LAE treatment are handled there.
    """
    required = {"Company", "County", "GrossWindLossUSD"}
    missing = required - set(gross_df.columns)
    if missing:
        raise ValueError(f"gross_df missing columns: {sorted(missing)}")

    terms = pd.read_csv(FHCF_TERMS_CSV) if fhcf_terms_df is None else fhcf_terms_df.copy()
    terms = normalize_fhcf_terms(terms)

    out = apply_fhcf_recovery(gross_df.copy(), terms)
    out["RecoveryUSD"] = out["RecoveryUSD"].fillna(0.0).clip(lower=0.0)
    out["NetWindUSD"] = out["NetWindUSD"].fillna(out["GrossWindLossUSD"]).clip(lower=0.0)

    return out[["Company", "County", "GrossWindLossUSD", "RecoveryUSD", "NetWindUSD"]]

def net_wind_loss(gross_df: pd.DataFrame, recover_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: subtract recoveries from gross losses to get net, by key.

    Parameters
    ----------
    gross_df : pandas.DataFrame
        ['Company','County','GrossWindLossUSD'].
    recover_df : pandas.DataFrame
        ['Company','County','RecoveryUSD'].

    Returns
    -------
    pandas.DataFrame
        Columns:
          - Company : str
          - County : str
          - NetWindLossUSD : float
    """
    net = gross_df.merge(
        recover_df[["Company", "County", "RecoveryUSD"]],
        on=["Company", "County"],
        how="left",
    )
    net["NetWindLossUSD"] = net["GrossWindLossUSD"] - net["RecoveryUSD"].fillna(0.0)
    return net[["Company", "County", "NetWindLossUSD"]]
