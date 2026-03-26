"""
exposure.py - Exposure matrix construction for Florida insurance entities
-----------------------------------------------------------------------

This module assembles exposure inputs for the risk propagation model:

- **FHCF industry county TIV** (wind, industry-wide)
- **Private wind exposure by company × county**
  (FHCF county TIV minus Citizens' county TIV, allocated by DPW shares)
- **Citizens wind exposure by county** (from Citizens County View snapshot)
- **NFIP flood exposure by county** (coverage-in-force)

Design principles
-----------------
- This is an *assembly* layer: it delegates I/O to loader functions and
  keeps the allocation math minimal and explicit.
- Contracts are documented in each function; units are USD unless noted.
- Mass-balance QA ensures that Citizens + private allocations reconcile
  to FHCF county totals within configured tolerances.

Public API
----------
- build_wind_exposures
- build_exposure_matrix
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional
from pathlib import Path


import numpy as np
import pandas as pd

from fl_risk_model.loader import (
    load_fhcf_county_exposure,
    load_market_share,
    load_nfip_county_exposure,
    load_citizens_county,
    load_nfip_policy_coverage,
)
from fl_risk_model import config as cfg

__all__ = ["build_wind_exposures", "build_exposure_matrix"]

# =============================================================================
# Internal helpers
# =============================================================================

# --- light, local normalizers (only used if your utils isn't available) ---
try:
    from fl_risk_model.utils import norm_county_name
except Exception:
    import re
    def norm_county_name(s: str) -> str:
        s = re.sub(r"\s+", " ", str(s)).strip()
        s = re.sub(r"\s+(County|Parish|Borough|City)$", "", s, flags=re.I)
        return s.title()

def _citizens_county_tiv(
    citizens_csv_path: Optional[str] = None,
    as_of: Optional[str] = None,
    include_products: Optional[list[str]] = None,
    county_xwalk: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aggregate Citizens' county-level TIV (USD) from the County View snapshot.

    Parameters
    ----------
    citizens_csv_path : str, optional
        Path to the Citizens county-level CSV snapshot; if None, uses cfg.CITIZENS_COUNTY_CSV.
    as_of : str, optional
        Snapshot date (e.g., "2024-12-31"); if None, uses cfg.CITIZENS_AS_OF.
    include_products : list of str, optional
        Optional filter for 'product_line' values; if None, uses cfg.CITIZENS_PRODUCTS.
    county_xwalk : pandas.DataFrame, optional
        Optional county crosswalk (e.g., to attach/align FIPS).

    Returns
    -------
    pandas.DataFrame
        Columns:
          - County : str
          - CitizensTIV : float (USD)
    """
    path = citizens_csv_path or cfg.CITIZENS_COUNTY_CSV
    snap = as_of or cfg.CITIZENS_AS_OF
    prods = include_products or cfg.CITIZENS_PRODUCTS
    
    # read mode and window from config with fallbacks
    mode = getattr(cfg, "SAMPLING_MODE_EXPOSURE", "FIXED_YEAR")
    lookback_years = int(getattr(cfg, "EWA_WINDOW_YEARS", 5))
    half_life = float(getattr(cfg, "EWA_HALF_LIFE_YEARS", 2.0))

    cdf = load_citizens_county(
        path_csv=path,
        as_of=snap,
        include_products=prods,
        county_xwalk=county_xwalk,
        mode=mode,                        
        lookback_years=lookback_years,    
        half_life=half_life,              
    )
    return (
        cdf.groupby("county", as_index=False)["tiv_usd"]
        .sum()
        .rename(columns={"county": "County", "tiv_usd": "CitizensTIV"})
    )


def _prep_market_share(ms: pd.DataFrame, exclude_citizens: bool = True) -> pd.DataFrame:
    """
    Normalize market share input to a single 'Share' column in [0,1].

    Parameters
    ----------
    ms : pandas.DataFrame
        Market share table containing either 'Share' or a single 'MarketShare*' column,
        plus 'Company'.
    exclude_citizens : bool, default True
        If True, removes any row that corresponds to Citizens before normalization.

    Returns
    -------
    pandas.DataFrame
        Columns:
          - Company : str
          - Share : float in [0,1]
    """
    out = ms.copy()
    out["Company"] = out["Company"].astype(str).str.strip()

    if "Share" not in out.columns:
        share_cols = [c for c in out.columns if c.lower().startswith("marketshare")]
        if not share_cols:
            raise ValueError("No column named 'Share' or starting with 'MarketShare'.")
        out["Share"] = out[share_cols[0]]

    out["Share"] = pd.to_numeric(out["Share"], errors="coerce").fillna(0.0)

    if exclude_citizens:
        cname = getattr(cfg, "CITIZENS_COMPANY_NAME", "Citizens")
        mask = out["Company"].str.contains("citizens", case=False, na=False) | (out["Company"] == cname)
        out = out.loc[~mask]

    total = float(out["Share"].sum())
    out["Share"] = 0.0 if total <= 0 else out["Share"] / total
    return out[["Company", "Share"]]

# =============================================================================
# Public API
# =============================================================================

def build_wind_exposures(
    fhcf_county_df: pd.DataFrame,
    market_share_df: pd.DataFrame,
    citizens_csv_path: Optional[str] = None,
    citizens_as_of: Optional[str] = None,
    citizens_products: Optional[list[str]] = None,
    county_xwalk: Optional[pd.DataFrame] = None,
    sample: Optional[bool] = None,
    cov: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Allocate county-level wind exposure to private insurers and Citizens.

    Method
    ------
    1) Start from FHCF county TIV (industry-wide).
    2) Subtract Citizens' observed county TIV (from County View) to form
       **PrivateCountyTIV** per county.
    3) Allocate each county's **PrivateCountyTIV** to private insurers by
       their Direct Premiums Written market shares.
    4) (Optional) Sample company×county TIV via Normal(mean=TIV, sd=cov*TIV), floored at 0.
    5) QA: verify county mass-balance (Citizens + private = FHCF county TIV)
       within absolute/relative tolerances from config.

    Parameters
    ----------
    fhcf_county_df : pandas.DataFrame
        Columns: ['County','CountyTIV'] (USD). Additional columns ignored.
    market_share_df : pandas.DataFrame
        Company-level shares. Must contain 'Company' and either 'Share'
        or a single 'MarketShare*' column.
    citizens_csv_path : str, optional
        Path to Citizens' county view CSV.
    citizens_as_of : str, optional
        Snapshot date for Citizens (e.g., "2024-12-31").
    citizens_products : list of str, optional
        Optional product-line filter for Citizens snapshot.
    county_xwalk : pandas.DataFrame, optional
        County crosswalk for harmonization (optional).
    sample : bool, optional
        If True (or cfg.SAMPLE_EXPOSURE), draws TIV_sampled ~ Normal(TIV, cov*TIV) clipped at 0.
    cov : float, optional
        Coefficient of variation used when sampling (default cfg.EXPOSURE_COV).
    rng : numpy.random.Generator, optional
        RNG used for sampling (defaults to np.random.default_rng()).

    Returns
    -------
    private_exposure : pandas.DataFrame
        Columns:
          - Company : str
          - County : str
          - TIV : float (USD)
          - TIV_sampled : float (USD; ==TIV if sampling disabled)
    citizens_exposure : pandas.DataFrame
        Columns:
          - Company : str  (Citizens name per cfg.CITIZENS_COMPANY_NAME)
          - County : str
          - TIV : float (USD)
          - TIV_sampled : float (USD; ==TIV if sampling disabled)

    Raises
    ------
    AssertionError
        If county mass-balance fails tolerance checks.
    """
    # Defaults from config
    sample = cfg.SAMPLE_EXPOSURE if sample is None else bool(sample)
    cov = float(cfg.EXPOSURE_COV if cov is None else cov)
    rng = rng or np.random.default_rng()

    # Prepare inputs
    ms = _prep_market_share(market_share_df, exclude_citizens=True)
    try:
        # Newer helper that expects args
        cit = _citizens_county_tiv(citizens_csv_path, citizens_as_of, citizens_products, county_xwalk)
    except TypeError:
        # Backward-compatible: older stub with no args
        cit = _citizens_county_tiv()

    base = fhcf_county_df.merge(cit, on="County", how="left")
    base["CitizensTIV"] = base["CitizensTIV"].fillna(0.0)
    base["PrivateCountyTIV"] = (base["CountyTIV"] - base["CitizensTIV"]).clip(lower=0.0)

    # Cartesian expand to allocate private shares per county
    priv = base.assign(key=1).merge(ms.assign(key=1), on="key").drop(columns="key")
    priv["TIV"] = priv["PrivateCountyTIV"] * priv["Share"]

    # Citizens exposure as a single 'Company'
    citizens_name = getattr(cfg, "CITIZENS_COMPANY_NAME", "Citizens Property Insurance Corporation")
    citizens_exposure = (
        base.loc[base["CitizensTIV"] > 0, ["County", "CitizensTIV"]]
        .rename(columns={"CitizensTIV": "TIV"})
        .assign(Company=citizens_name)
    ) if (base["CitizensTIV"] > 0).any() else pd.DataFrame(columns=["Company", "County", "TIV"])

    # Optional sampling (non-negative)
    def _sample_nonneg(v: pd.Series) -> pd.Series:
        if not sample:
            return v
        sd = v * cov
        return pd.Series(np.maximum(rng.normal(v, sd), 0.0), index=v.index)

    private_exposure = priv[["Company", "County", "TIV"]].copy()
    private_exposure["TIV_sampled"] = _sample_nonneg(private_exposure["TIV"])
    citizens_exposure["TIV_sampled"] = _sample_nonneg(citizens_exposure["TIV"])

    # Mass-balance QA: (private + citizens) ~= FHCF county totals
    recon = (
        pd.concat([private_exposure, citizens_exposure], ignore_index=True)
        .groupby("County", as_index=False)["TIV"]
        .sum()
        .rename(columns={"TIV": "reconstructed"})
    )
    merged = fhcf_county_df.merge(recon, on="County", how="left").fillna({"reconstructed": 0.0})

    abs_floor = float(cfg.MASSBAL_ABS_FLOOR)
    rel_tol = float(cfg.MASSBAL_REL_TOL)
    delta = merged["CountyTIV"].astype(float) - merged["reconstructed"].astype(float)
    tol = np.maximum(abs_floor, rel_tol * merged["CountyTIV"].abs())

    bad = merged.loc[delta.abs() > tol, ["County", "CountyTIV", "reconstructed"]]
    if not bad.empty:
        # Keep message concise but informative
        raise AssertionError(f"County TIV mass-balance failed for {len(bad)} counties; first rows:\n{bad.head(20)}")

    return private_exposure, citizens_exposure


def build_exposure_matrix(
    fhcf_path: Optional[str] = None,
    market_share_path: Optional[str] = None,
    citizens_csv_path: Optional[str] = None,
    citizens_as_of: Optional[str] = None,
    citizens_products: Optional[list[str]] = None,
    nfip_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build exposure matrices for wind (private + Citizens) and flood.

    Parameters
    ----------
    fhcf_path : str, optional
        FHCF Excel file path. Defaults to f"{cfg.DATA_DIR}/FHCF_2024_Exposure_byCounty.xlsx".
    market_share_path : str, optional
        Market-share workbook path. Defaults to f"{cfg.DATA_DIR}/FL HO Market Share Report_6.10.25.xlsx".
    citizens_csv_path : str, optional
        Citizens County View CSV path. If None, _citizens_county_tiv() uses cfg.
    citizens_as_of : str, optional
        Citizens snapshot date; if None, uses cfg.
    citizens_products : list of str, optional
        Product-line filter for Citizens snapshot; if None, uses cfg.
    nfip_path : str, optional
        Fallback FEMA CSV (full county coverage) if minimal policy file is unavailable.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
          - "county_industry" : FHCF industry-wide county TIV (columns ['County','CountyTIV'])
          - "company_county_private" : private insurer allocations (company×county)
          - "citizens_county" : Citizens exposure by county
          - "nfip" : NFIP county coverage (coverage-in-force) if available; else None

    Notes
    -----
    - Prefers the minimal NFIP policy coverage file (cfg.NFIP_POLICIES_MIN, filtered to cfg.NFIP_POLICY_YEAR).
      Falls back to full NFIP county CSV if minimal file is missing or errors.
    """
    fhcf_path = fhcf_path or f"{cfg.DATA_DIR}/FHCF_2024_Exposure_byCounty.xlsx"
    market_share_path = market_share_path or f"{cfg.DATA_DIR}/FL HO Market Share Report_6.10.25.xlsx"
    nfip_min_path = getattr(cfg, "NFIP_POLICIES_MIN", None)

    # FHCF + market shares
    county_df = load_fhcf_county_exposure(fhcf_path)[["County", "CountyTIV"]]
    ms_df = load_market_share(market_share_path)

    # Wind (private & Citizens)
    priv_x_county, cit_county = build_wind_exposures(
        fhcf_county_df=county_df,
        market_share_df=ms_df,
        citizens_csv_path=citizens_csv_path,
        citizens_as_of=citizens_as_of,
        citizens_products=citizens_products,
    )

    # --- NFIP (prefer minimal policies by year; else full county coverage) ---
    nfip_df = pd.DataFrame()
    if nfip_path:
        try:
            nfip_df = load_nfip_policy_coverage(
                path=nfip_path,
                mode=getattr(cfg, "SAMPLING_MODE_NFIP_POLICIES", "FIXED_YEAR"),
                year=int(getattr(cfg, "NFIP_POLICY_YEAR", getattr(cfg, "FIXED_YEAR", 2024))),
                lookback_years=int(getattr(cfg, "EWA_WINDOW_YEARS", 5)),
                half_life=float(getattr(cfg, "EWA_HALF_LIFE_YEARS", 2.0)),
                county_xwalk=None,  # join-to-county not required here
            )

            if nfip_df is None:
                nfip_df = pd.DataFrame()

            # Normalize coverage column -> FloodTIV
            if "FloodTIV" not in nfip_df.columns:
                cov_col = next(
                    (c for c in nfip_df.columns
                     if str(c).lower() in (
                         "floodtiv", "flood_tiv",
                         "coverage_in_force", "nfip_coverage_usd",
                         "coverage_usd", "tiv_usd", "tiv"
                     )),
                    None
                )
                if cov_col:
                    nfip_df = nfip_df.rename(columns={cov_col: "FloodTIV"})
                else:
                    # no usable coverage column -> empty
                    nfip_df = pd.DataFrame()

            # Keep a usable key: prefer FIPS if present; else County
            if not nfip_df.empty:
                if "county_fips" in nfip_df.columns:
                    nfip_df["county_fips"] = (
                        nfip_df["county_fips"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
                    )
                    nfip_df = nfip_df[["county_fips", "FloodTIV"]].copy()
                elif "County" in nfip_df.columns:
                    nfip_df["County"] = nfip_df["County"].astype(str)
                    nfip_df = nfip_df[["County", "FloodTIV"]].copy()
                else:
                    # Still no join key -> empty
                    nfip_df = pd.DataFrame(columns=["county_fips", "FloodTIV"])

        except Exception as e:
            # NFIP load failure - fall back to empty
            nfip_df = pd.DataFrame(columns=["county_fips", "FloodTIV"])

    return {
        "county_industry": county_df,
        "company_county_private": priv_x_county,
        "citizens_county": cit_county,
        "nfip": nfip_df,
    }

