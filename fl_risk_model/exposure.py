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

def _coerce_currency(s: pd.Series) -> pd.Series:
    # keep digits, dot, minus; treat empties as zero; coerce to float
    cleaned = (
        s.astype(str)
         .str.replace(r"[^\d\.\-eE]", "", regex=True)
         .str.replace(r"^\s*$", "0", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0).astype(float)

def _norm_company_name(s: pd.Series) -> pd.Series:
    """UPPER + keep alnum/space + collapse spaces."""
    return (
        s.astype(str)
         .str.upper()
         .str.replace(r"[^A-Z0-9 ]", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

def _load_company_key_map_ms(path_csv: str) -> pd.DataFrame:
    """
    Build a mapping keyed by normalized Company_MS (preferred) with an optional
    backup via normalized Company_FHCF. Returns:
      - MS_norm   (normalized Company_MS)
      - FHCF_norm (normalized Company_FHCF)
      - StatEntityKey (string)
    """
    df = pd.read_csv(path_csv)
    cols = {c.lower(): c for c in df.columns}
    ms_col   = cols.get("company_ms")
    fhcf_col = cols.get("company_fhcf")
    stat_col = cols.get("statentitykey") or cols.get("naic")

    if ms_col is None:
        raise KeyError("company_keys.csv must include 'Company_MS'.")
    if stat_col is None:
        raise KeyError("company_keys.csv must include 'StatEntityKey' or 'NAIC'.")

    out = df[[ms_col, stat_col]].copy().rename(columns={ms_col: "Company_MS", stat_col: "StatEntityKey"})
    out["MS_norm"] = _norm_company_name(out["Company_MS"])
    out["StatEntityKey"] = out["StatEntityKey"].astype(str).str.strip()

    if fhcf_col:
        out_fhcf = df[[fhcf_col, stat_col]].copy().rename(columns={fhcf_col: "Company_FHCF", stat_col: "StatEntityKey"})
        out_fhcf["FHCF_norm"] = _norm_company_name(out_fhcf["Company_FHCF"])
        out = out.merge(out_fhcf[["FHCF_norm","StatEntityKey"]].dropna(), on="StatEntityKey", how="outer")
    else:
        out["FHCF_norm"] = pd.NA

    # Deduplicate on keys
    out = out.drop_duplicates(subset=["StatEntityKey","MS_norm","FHCF_norm"])
    return out[["MS_norm","FHCF_norm","StatEntityKey"]]

def _read_company_county_workbook(path_xlsx: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path_xlsx)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet_name=sheet)
        lower = {c.lower(): c for c in df.columns}

        # Company column = 'NAME'
        if "name" not in lower:
            continue
        name_col = lower["name"]

        # Exposure column (exact header or anything containing 'exposure')
        vcol = None
        for cand in ("total $ value of exposure for policies in force",
                     "total value of exposure for policies in force",
                     "exposure", "tiv", "sum insured", "coverage_in_force"):
            if cand in lower:
                vcol = lower[cand]; break
        if vcol is None:
            ex_cols = [c for c in df.columns if "exposure" in c.lower()]
            vcol = ex_cols[0] if ex_cols else None

        tmp = df[[name_col] + ([vcol] if vcol else [])].copy()
        tmp = tmp.rename(columns={name_col: "Company", (vcol or name_col): "TIV"})
        # Coerce currency (if TIV not present, keep as 0)
        if vcol:
            tmp["TIV"] = (tmp["TIV"].astype(str)
                          .str.replace(r"[^\d\.\-eE]", "", regex=True)
                          .replace({"": "0"}).astype(float))
        else:
            tmp["TIV"] = 0.0

        tmp["Company"] = tmp["Company"].astype(str).str.strip()
        tmp["County"] = sheet
        frames.append(tmp[["County","Company","TIV"]])

    if not frames:
        return pd.DataFrame(columns=["County","Company","TIV"])

    out = pd.concat(frames, ignore_index=True)
    out["MS_norm"] = _norm_company_name(out["Company"])
    # aggregate duplicates company×county
    return out.groupby(["County","Company","MS_norm"], as_index=False)["TIV"].sum()

def _drop_citizens_rows(df: pd.DataFrame, label: str, cit_keys: set[str], citizens_name: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    # by name (case-insensitive), catches “Citizens Property Insurance Corporation”, etc.
    m_name = df["Company"].astype(str).str.contains(r"(?i)\bcitizen", na=False)
    if citizens_name:
        m_name = m_name | df["Company"].astype(str).str.fullmatch(citizens_name, case=False, na=False)
    # by StatEntityKey (if present)
    m_key = df["StatEntityKey"].astype(str).isin(cit_keys) if "StatEntityKey" in df.columns else False
    m = m_name | m_key
    if bool(m.any()):
        df = df.loc[~m].copy()
    return df

def _build_citizens_keyset(keymap: pd.DataFrame) -> set[str]:
    keys = set()
    if "MS_norm" in keymap.columns:
        keys |= set(
            keymap.loc[keymap["MS_norm"].str.contains("CITIZEN", case=False, na=False), "StatEntityKey"]
                .astype(str)
        )
    if "FHCF_norm" in keymap.columns:
        keys |= set(
            keymap.loc[keymap["FHCF_norm"].str.contains("CITIZEN", case=False, na=False), "StatEntityKey"]
                .astype(str)
        )
    # seed from config (optional but robust)
    for hint in (getattr(cfg, "CITIZENS_STATKEY", None), getattr(cfg, "CITIZENS_NAIC", None)):
        if hint:
            keys.add(str(hint).strip())
    return {k for k in keys if k and k.lower() != "nan"}

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

def build_wind_exposures_alt_company_county(
    *,
    fhcf_path: Optional[str] = None,
    market_share_path: Optional[str] = None,
    citizens_csv_path: Optional[str] = None,
    citizens_as_of: Optional[str] = None,
    citizens_products: Optional[list[str]] = None,
    company_county_workbook: str,
    company_key_csv: str,
    county_xwalk: Optional[pd.DataFrame] = None,
    sample: Optional[bool] = None,
    cov: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Alternative wind exposure builder:
      - For counties present in the workbook, set private TIV directly by company×county.
      - For the remaining counties, fall back to the existing FHCF-minus-Citizens allocated by market share.

    Returns
    -------
    private_exposure : DataFrame
        ['Company','StatEntityKey','County','TIV','TIV_sampled']
    citizens_exposure : DataFrame
        ['Company','County','TIV','TIV_sampled']
    """
    # Config defaults
    fhcf_path = fhcf_path or f"{cfg.DATA_DIR}/FHCF_2024_Exposure_byCounty.xlsx"
    market_share_path = market_share_path or f"{cfg.DATA_DIR}/FL HO Market Share Report_6.10.25.xlsx"
    sample = cfg.SAMPLE_EXPOSURE if sample is None else bool(sample)
    cov = float(cfg.EXPOSURE_COV if cov is None else cov)
    rng = rng or np.random.default_rng(getattr(cfg, "RNG_SEED", None))

    # 1) FHCF county totals (industry)
    fhcf = load_fhcf_county_exposure(fhcf_path)[["County","CountyTIV"]].copy()
    fhcf["County"] = fhcf["County"].map(norm_county_name)

    # 2) Citizens county TIV (unchanged)
    cit = _citizens_county_tiv(citizens_csv_path, citizens_as_of, citizens_products, county_xwalk)

    # 3) Company×County private TIV from workbook
    alt_priv = _read_company_county_workbook(company_county_workbook)
    if alt_priv.empty:
        raise ValueError("No usable sheets found in the company×county workbook.")
    
    keymap = _load_company_key_map_ms(company_key_csv)  # MS_norm / FHCF_norm / StatEntityKey

    # First pass: match workbook NAME -> Company_MS
    alt_priv = alt_priv.merge(keymap[["MS_norm","StatEntityKey"]], on="MS_norm", how="left")

    # Optional fallback: try matching NAME -> Company_FHCF for any still-NA (helps coverage)
    if alt_priv["StatEntityKey"].isna().any():
        alt_priv = alt_priv.merge(
            keymap[["FHCF_norm","StatEntityKey"]].rename(columns={"FHCF_norm":"MS_norm", "StatEntityKey":"StatEntityKey_fhcf"}),
            on="MS_norm", how="left"
        )
        alt_priv["StatEntityKey"] = alt_priv["StatEntityKey"].fillna(alt_priv["StatEntityKey_fhcf"])
        alt_priv = alt_priv.drop(columns=["StatEntityKey_fhcf"])

    priv_cov = alt_priv.copy()
    if "StatEntityKey" not in priv_cov.columns:
        priv_cov["StatEntityKey"] = pd.NA

    # --- build once, after you've created `keymap` with MS_norm/FHCF_norm/StatEntityKey ---
    # Citizens statutory keys from the mapping (match on either MS_norm or FHCF_norm)
    cit_mask = keymap["MS_norm"].str.contains("CITIZEN", case=False, na=False)
    if "FHCF_norm" in keymap.columns:
        cit_mask = cit_mask | keymap["FHCF_norm"].str.contains("CITIZEN", case=False, na=False)

    cit_keys = _build_citizens_keyset(keymap)
    citizens_name = getattr(cfg, "CITIZENS_COMPANY_NAME", None)

    # --- apply to workbook-driven block (after merging keys into `alt_priv`) ---
    alt_priv = _drop_citizens_rows(alt_priv, "workbook", cit_keys, citizens_name)
    priv_cov = alt_priv.copy()

    # 2) Drop any workbook rows that map to Citizens, or look like Citizens by name
    is_cit_key = alt_priv["StatEntityKey"].astype(str).isin(cit_keys)
    is_cit_name = alt_priv["Company"].str.upper().str.contains("CITIZENS", na=False)
    alt_priv = alt_priv.loc[~(is_cit_key | is_cit_name)].copy()

    # 5) Identify covered vs uncovered counties (by any private TIV in workbook)
    covered_counties = set(alt_priv["County"].unique().tolist())
    fhcf["is_covered"] = fhcf["County"].isin(covered_counties)

    # 6) For covered counties -> we will trust alt_priv directly
    priv_cov = alt_priv.copy()

    # initialize so it's always defined
    fallback_priv = pd.DataFrame(columns=["County","Company","TIV","StatEntityKey"])

    # 7) For uncovered counties -> fallback
    uncovered = fhcf.loc[~fhcf["is_covered"], ["County","CountyTIV"]].copy()
    if not uncovered.empty:
        # attach Citizens
        tmp = uncovered.merge(cit, on="County", how="left")
        tmp["CitizensTIV"] = tmp["CitizensTIV"].fillna(0.0)
        tmp["PrivateCountyTIV"] = (tmp["CountyTIV"] - tmp["CitizensTIV"]).clip(lower=0.0)

        # market shares
        ms = load_market_share(market_share_path)
        if "Share" not in ms.columns:
            share_cols = [c for c in ms.columns if c.lower().startswith("marketshare")]
            if not share_cols:
                raise KeyError("Market share file missing a 'Share' or 'MarketShare*' column.")
            ms = ms.rename(columns={share_cols[0]: "Share"})

        ms["Company"] = ms["Company"].astype(str).str.strip()
        ms["Share"]   = pd.to_numeric(ms["Share"], errors="coerce").fillna(0.0)

        # --- NEW: drop Citizens from the share table BEFORE allocation, then renormalize ---
        # (match by name; and also by StatEntityKey via keymap if available)
        ms["Company_norm"] = _norm_company_name(ms["Company"])
        ms = ms.merge(
            keymap[["MS_norm","StatEntityKey"]].rename(columns={"MS_norm":"Company_norm"}),
            on="Company_norm", how="left"
        )

        ms_cit_name = ms["Company"].str.contains(r"(?i)\bcitizen", na=False)
        ms_cit_key  = ms["StatEntityKey"].astype(str).isin(cit_keys)
        ms = ms.loc[~(ms_cit_name | ms_cit_key)].copy()

        # renormalize shares AFTER removing Citizens
        tot = float(ms["Share"].sum())
        ms["Share"] = 0.0 if tot <= 0 else ms["Share"] / tot

        # clean helper cols (optional)
        ms = ms.drop(columns=["Company_norm","StatEntityKey"], errors="ignore")

        # Cartesian allocate private county TIV (now shares exclude Citizens and sum to 1)
        fallback_priv = tmp.assign(key=1).merge(ms.assign(key=1), on="key").drop(columns="key")
        fallback_priv["TIV"] = fallback_priv["PrivateCountyTIV"] * fallback_priv["Share"]
        fallback_priv = fallback_priv[["County","Company","TIV"]].copy()

        # attach keys (MS_norm first, optional FHCF fallback)
        fallback_priv["MS_norm"] = _norm_company_name(fallback_priv["Company"])
        fallback_priv = fallback_priv.merge(keymap[["MS_norm","StatEntityKey"]], on="MS_norm", how="left")
        if fallback_priv["StatEntityKey"].isna().any():
            fallback_priv = fallback_priv.merge(
                keymap[["FHCF_norm","StatEntityKey"]]
                    .rename(columns={"FHCF_norm":"MS_norm","StatEntityKey":"StatEntityKey_fhcf"}),
                on="MS_norm", how="left"
            )
            fallback_priv["StatEntityKey"] = fallback_priv["StatEntityKey"].fillna(fallback_priv["StatEntityKey_fhcf"])
            fallback_priv = fallback_priv.drop(columns=["StatEntityKey_fhcf"])
        fallback_priv = fallback_priv.drop(columns=["MS_norm"])

    # Defensive: ensure column exists in both dataframes
    for _df in (priv_cov, fallback_priv):
        if "StatEntityKey" not in _df.columns:
            _df["StatEntityKey"] = pd.NA

    # 8) Concatenate alt + fallback
    private_exposure = pd.concat(
        [priv_cov[["County","Company","TIV","StatEntityKey"]],
         fallback_priv[["County","Company","TIV","StatEntityKey"]]],
        ignore_index=True
    )

    # 9) Sampling (non-negative normal with CoV)
    def _sample_nonneg(v: pd.Series) -> pd.Series:
        if not sample:
            return v
        sd = v.astype(float) * cov
        return pd.Series(np.maximum(rng.normal(v.astype(float), sd), 0.0), index=v.index)

    private_exposure["TIV_sampled"] = _sample_nonneg(private_exposure["TIV"])

    # 10) Citizens exposure dataframe (as your runner expects)
    citizens_name = getattr(cfg, "CITIZENS_COMPANY_NAME", "Citizens Property Insurance Corporation")
    citizens_exposure = (
        cit.loc[cit["CitizensTIV"] > 0, ["County","CitizensTIV"]]
           .rename(columns={"CitizensTIV": "TIV"})
           .assign(Company=citizens_name)
           .reset_index(drop=True)
    )
    citizens_exposure["TIV_sampled"] = _sample_nonneg(citizens_exposure["TIV"])

    # 11) Mass-balance QA on all counties (DIAGNOSTIC ONLY - NO MUTATION)

    def _recon_private(_priv_cov: pd.DataFrame, _fallback_priv: pd.DataFrame) -> pd.DataFrame:
        """Sum private TIV by county from workbook-covered + fallback blocks."""
        return (
            pd.concat([_priv_cov[["County","TIV"]], _fallback_priv[["County","TIV"]]], ignore_index=True)
            .groupby("County", as_index=False)["TIV"].sum()
            .rename(columns={"TIV": "Private"})
        )

    def _recon_total(_recon_priv: pd.DataFrame, _cit: pd.DataFrame) -> pd.DataFrame:
        """Add Citizens to Private -> reconstructed county total."""
        return (
            _recon_priv.merge(_cit[["County","CitizensTIV"]], on="County", how="left")
                    .fillna({"CitizensTIV": 0.0})
                    .assign(reconstructed=lambda d: d["Private"] + d["CitizensTIV"])
                    [["County","reconstructed"]]
        )

    # Build snapshot
    recon_priv = _recon_private(priv_cov, fallback_priv)
    snap = (
        fhcf[["County","CountyTIV"]]
        .merge(cit[["County","CitizensTIV"]], on="County", how="left")
        .merge(recon_priv, on="County", how="left")
        .fillna({"CitizensTIV": 0.0, "Private": 0.0})
    )
    snap["Reconstructed"] = snap["Private"] + snap["CitizensTIV"]
    snap["Delta"] = snap["CountyTIV"] - snap["Reconstructed"]
    snap["RelGap_%"] = 100 * snap["Delta"].abs() / snap["CountyTIV"].clip(lower=1)

    # Final strict check: compare (Private + Citizens) to FHCF
    abs_floor = float(getattr(cfg, "MASSBAL_ABS_FLOOR", 1.0))
    rel_tol  = float(getattr(cfg, "MASSBAL_REL_TOL", 1e-9))
    tol = np.maximum(abs_floor, rel_tol * snap["CountyTIV"].abs())

    bad = snap.loc[snap["Delta"].abs() > tol, ["County","CountyTIV","CitizensTIV","Private","Reconstructed","Delta"]]
    if not bad.empty:
        raise AssertionError(f"[Alt Exposure] Mass-balance failed for {len(bad)} counties; e.g.\n{bad.head(12)}")

    # 12) Final column order
    private_exposure = private_exposure[["Company","StatEntityKey","County","TIV","TIV_sampled"]].copy()
    citizens_exposure = citizens_exposure[["Company","County","TIV","TIV_sampled"]].copy()

    return private_exposure, citizens_exposure