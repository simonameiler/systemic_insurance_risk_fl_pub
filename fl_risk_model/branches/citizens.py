"""
citizens.py — Citizens (wind) branch
------------------------------------

Implements the Citizens side of the wind branch, aligned with the Methods:

1) Allocate **insured** county wind to Citizens by county share:
     Citizens share (county) = Citizens TIV / Industry TIV  (by county)
     Allocated insured wind = InsuredWindUSD * share
   (Robust to joining by FIPS first, then by normalized county name.)

2) Convert allocated insured wind to a Citizens gross loss table suitable for FHCF.

3) Apply FHCF recovery to Citizens (full-contract via shared FHCF utilities
   when available; otherwise coverage-only fallback).

4) Apply the post-FHCF **capital hit** against Citizens’ starting surplus.

Design notes
------------
- All money values are USD.
- County names are normalized consistently (via `norm_county_name`).
- Joins prefer FIPS; fall back to county names where FIPS is missing.
- Behavior is unchanged from the original version; only documentation and
  organization were improved.

Public API
----------
- allocate_insured_wind_to_citizens
- prepare_citizens_gross_wind
- recover_citizens_wind
- apply_citizens_capital_hit
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd

from fl_risk_model.fhcf import normalize_fhcf_terms, apply_fhcf_recovery
from fl_risk_model.utils import norm_county_name
from fl_risk_model.capital import load_citizens_capital_row_from_csv

CITIZENS_NAME = "Citizens Property Insurance Corporation"
_FHCF_AVAILABLE = True  # flag used to gate the shared FHCF helpers in fallback logic

__all__ = [
    "allocate_insured_wind_to_citizens",
    "prepare_citizens_gross_wind",
    "recover_citizens_wind",
    "apply_citizens_capital_hit",
]

# =============================================================================
# Helpers (joins & normalization)
# =============================================================================

def _to_fips_str(series: pd.Series) -> pd.Series:
    """
    Coerce values into zero-padded 5-character FIPS strings, preserving NA.

    Parameters
    ----------
    series : pandas.Series
        Values may be strings, ints, floats, or NA.

    Returns
    -------
    pandas.Series
        dtype 'string', with 5-char, zero-padded FIPS where parseable; NA otherwise.
    """
    s = pd.to_numeric(series, errors="coerce")
    try:
        s = s.astype("Int64")
    except Exception:
        pass

    def _fmt(x):
        if pd.isna(x):
            return np.nan
        try:
            xi = int(x)
        except Exception:
            xs = str(x)
            return xs.zfill(5) if xs.isdigit() else np.nan
        return str(xi).zfill(5)

    return s.map(_fmt).astype("string")

# =============================================================================
# Stage 1: Allocate insured wind to Citizens by county exposure share
# =============================================================================

def allocate_insured_wind_to_citizens(
    insured_wind_by_county: pd.DataFrame,
    citizens_exposure: pd.DataFrame,
    industry_county_tiv: pd.DataFrame,
    *,
    county_col_input: str = "county",          # accepts "county" or "County"
    county_fips_col_input: str = "county_fips",
    county_col_exposure: str = "county",
    county_fips_col_exposure: str = "county_fips",
    county_col_industry: str = "County",
    county_tiv_col_industry: str = "CountyTIV",
) -> pd.DataFrame:
    """
    Allocate the **insured** wind (already carved at gross) to Citizens by county share:

        share_county = CitizensTIV_county / IndustryTIV_county
        alloc = InsuredWindUSD * share_county

    Inputs
    ------
    insured_wind_by_county : pandas.DataFrame
        Must include:
          - county_col_input : str (county name; default 'county')
          - county_fips_col_input : str (FIPS; default 'county_fips', optional)
          - 'InsuredWindUSD' : float (industry-wide insured wind by county)
    citizens_exposure : pandas.DataFrame
        Citizens County View snapshot with:
          - county_col_exposure : str (default 'county')
          - county_fips_col_exposure : str (default 'county_fips')
          - 'tiv_usd' : float
    industry_county_tiv : pandas.DataFrame
        FHCF industry totals with:
          - county_col_industry : str (default 'County')
          - county_tiv_col_industry : str (default 'CountyTIV')

    Returns
    -------
    pandas.DataFrame
        Input rows with added columns:
          - 'citizens_share_of_county' : float in [0, 1]
          - 'citizens_allocated_insured_wind_usd' : float
        and diagnostics in `.attrs['join_stats']`.

    Raises
    ------
    ValueError
        If 'InsuredWindUSD' is not present in `insured_wind_by_county`.

    Notes
    -----
    - Join preference: FIPS first; if missing or NA, fallback to normalized county name.
    - Industry TIV comes from FHCF totals; Citizens TIV from County View snapshot.
    """
    df = insured_wind_by_county.copy()

    # --- validations
    if "InsuredWindUSD" not in df.columns:
        raise ValueError("Citizens allocator expects 'InsuredWindUSD' (carved at gross).")

    # Ensure join-key columns exist
    if county_col_input not in df.columns:
        df[county_col_input] = pd.NA
    if county_fips_col_input not in df.columns:
        df[county_fips_col_input] = pd.NA
    if county_col_exposure not in citizens_exposure.columns:
        citizens_exposure = citizens_exposure.assign(**{county_col_exposure: pd.NA})
    if county_fips_col_exposure not in citizens_exposure.columns:
        citizens_exposure = citizens_exposure.assign(**{county_fips_col_exposure: pd.NA})

    # --- normalize keys (input & exposure)
    df[county_col_input] = df[county_col_input].map(norm_county_name)
    df[county_fips_col_input] = _to_fips_str(df[county_fips_col_input])

    exp = citizens_exposure.copy()
    exp[county_col_exposure] = exp[county_col_exposure].map(norm_county_name)
    exp[county_fips_col_exposure] = _to_fips_str(exp[county_fips_col_exposure])

    # --- collapse Citizens exposure by (name, fips)
    exp_sum = (
        exp.groupby([county_col_exposure, county_fips_col_exposure], dropna=False, as_index=False)["tiv_usd"]
        .sum()
    )

    # --- choose join strategy: name-only if both sides have all-NA FIPS
    use_name_only = (
        df[county_fips_col_input].isna().all()
        and exp[county_fips_col_exposure].isna().all()
    )

    if use_name_only:
        # NAME-ONLY merge for Citizens numerator
        m = df.merge(
            exp_sum[[county_col_exposure, "tiv_usd"]],
            left_on=county_col_input,
            right_on=county_col_exposure,
            how="left",
        )
        # keep the LEFT 'county', drop the RIGHT key if it exists
        cx = f"{county_col_input}_x"
        cy = f"{county_col_input}_y"
        if cx in m.columns:
            m.rename(columns={cx: county_col_input}, inplace=True)
            if cy in m.columns:
                m.drop(columns=[cy], inplace=True, errors="ignore")
        else:
            if county_col_exposure in m.columns and county_col_exposure != county_col_input:
                m.drop(columns=[county_col_exposure], inplace=True, errors="ignore")
    else:
        # FIPS-FIRST merge, then fill by name for misses
        m = df.merge(
            exp_sum[[county_fips_col_exposure, "tiv_usd"]],
            left_on=county_fips_col_input,
            right_on=county_fips_col_exposure,
            how="left",
        )
        m.drop(columns=[county_fips_col_exposure], inplace=True, errors="ignore")

        miss = m["tiv_usd"].isna()
        if miss.any():
            name_fill = df.loc[miss, [county_col_input]].merge(
                exp_sum[[county_col_exposure, "tiv_usd"]],
                left_on=county_col_input,
                right_on=county_col_exposure,
                how="left",
            )
            m.loc[miss, "tiv_usd"] = name_fill["tiv_usd"].to_numpy()

    m["tiv_usd"] = m["tiv_usd"].fillna(0.0)

    # --- Industry county normalization and group
    ind = industry_county_tiv.copy()
    if county_col_industry not in ind.columns:
        alt = "county" if county_col_industry == "County" else "County"
        if alt in ind.columns:
            county_col_industry = alt
    ind[county_col_industry] = ind[county_col_industry].map(norm_county_name)

    ind_g = ind.groupby(county_col_industry, as_index=False)[county_tiv_col_industry].sum()

    # --- merge industry TIV by county name
    m = m.merge(
        ind_g[[county_col_industry, county_tiv_col_industry]],
        left_on=county_col_input,
        right_on=county_col_industry,
        how="left",
    )
    if county_col_industry in m.columns and county_col_industry != county_col_input:
        m.drop(columns=[county_col_industry], inplace=True, errors="ignore")

    # --- compute share + allocation
    denom = pd.to_numeric(m[county_tiv_col_industry], errors="coerce").fillna(0.0)
    num   = pd.to_numeric(m["tiv_usd"],              errors="coerce").fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(denom > 0, np.clip(num / denom, 0.0, 1.0), 0.0)

    m["citizens_share_of_county"] = share
    m["citizens_allocated_insured_wind_usd"] = m["InsuredWindUSD"].astype(float) * share

    # (optional) diagnostics
    m.attrs["join_stats"] = {
        "rows": int(len(m)),
        "cit_exposure_rows": int(len(exp_sum)),
        "sum_citizens_tiv": float(num.sum()),
        "matched_nonzero": int((num > 0).sum()),
        "industry_rows": int(len(ind)),
        "unmatched_industry_counties": int(m[county_tiv_col_industry].isna().sum()),
    }

    return m

# =============================================================================
# Stage 2: Prepare Citizens gross for FHCF
# =============================================================================

def prepare_citizens_gross_wind(alloc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the allocation output into a Citizens gross-loss table for FHCF.

    Accepts either:
      - 'InsuredWindUSD' (preferred; already the allocated Citizens-insured total), or
      - 'citizens_allocated_insured_wind_usd' (legacy/internal name).

    Returns ['Company','County','GrossWindLossUSD'] with totals preserved.
    """
    required = "citizens_allocated_insured_wind_usd"
    if required not in alloc_df.columns:
        raise ValueError("alloc_df must include 'citizens_allocated_insured_wind_usd'")

    out = alloc_df[["county", required]].copy()
    out.rename(columns={
        "county": "County",
        required: "GrossWindLossUSD",
    }, inplace=True)
    out["Company"] = CITIZENS_NAME
    out = out[out["GrossWindLossUSD"].fillna(0) > 0].reset_index(drop=True)
    return out[["Company", "County", "GrossWindLossUSD"]]

# =============================================================================
# Stage 3: FHCF recovery for Citizens
# =============================================================================

def recover_citizens_wind(
    gross_citizens_df: pd.DataFrame,
    fhcf_terms_df: pd.DataFrame,
    mode: str = "auto"
) -> pd.DataFrame:
    """
    Apply FHCF recovery to Citizens' gross insured wind.

    Parameters
    ----------
    gross_citizens_df : pandas.DataFrame
        ['Company','County','GrossWindLossUSD'] with Company == Citizens.
    fhcf_terms_df : pandas.DataFrame
        FHCF terms with at least ['Company','CoveragePct'].
        If it also includes ['FHCFPremium'], the full-contract path is available.
    mode : {'auto','contract','coverage_only'}, default 'auto'
        - 'contract': use the shared FHCF engine (requires normalize/apply helpers).
        - 'coverage_only': apply CoveragePct × LAE to Gross (no retention/limit).
        - 'auto': choose 'contract' if FHCFPremium present and helpers available; else 'coverage_only'.

    Returns
    -------
    pandas.DataFrame
        ['Company','County','GrossWindLossUSD','RecoveryUSD','NetWindUSD']

    Notes
    -----
    LAE factor is 1.10 in the coverage-only fallback to mirror your config.
    """
    # Filter to Citizens' terms if a multi-company terms sheet is passed
    terms = fhcf_terms_df.copy()
    if "Company" in terms.columns:
        mask = terms["Company"].astype(str).str.contains("Citizens", case=False, na=False)
        terms = terms.loc[mask].copy()

    if terms.empty:
        # No terms: zero recovery
        out = gross_citizens_df.copy()
        out["RecoveryUSD"] = 0.0
        out["NetWindUSD"] = out["GrossWindLossUSD"]
        return out[["Company", "County", "GrossWindLossUSD", "RecoveryUSD", "NetWindUSD"]]

    has_premium = any(col.lower() == "fhcfpremium" for col in terms.columns)
    use_contract = (mode == "contract") or (mode == "auto" and has_premium and _FHCF_AVAILABLE)

    if use_contract:
        # Full-contract via shared FHCF engine (retention, limit, coverage%, LAE, cap)
        terms_norm = normalize_fhcf_terms(terms) if _FHCF_AVAILABLE else terms.copy()
        out = apply_fhcf_recovery(gross_citizens_df.copy(), terms_norm) if _FHCF_AVAILABLE else gross_citizens_df.copy()
        if not _FHCF_AVAILABLE:
            out["RecoveryUSD"] = 0.0
            out["NetWindUSD"] = out["GrossWindLossUSD"]
        return out[["Company", "County", "GrossWindLossUSD", "RecoveryUSD", "NetWindUSD"]]

    # Coverage-only fallback: Recovery = CoveragePct × LAE × Gross (no retention/limit)
    cov_col = "CoveragePct" if "CoveragePct" in terms.columns else next(
        (c for c in terms.columns if "coverage" in str(c).lower()), None
    )
    if cov_col is None:
        out = gross_citizens_df.copy()
        out["RecoveryUSD"] = 0.0
        out["NetWindUSD"] = out["GrossWindLossUSD"]
        return out[["Company", "County", "GrossWindLossUSD", "RecoveryUSD", "NetWindUSD"]]

    cov_val = float(terms.iloc[0][cov_col])
    if cov_val > 1.0:
        cov_val /= 100.0

    LAE = 1.10
    out = gross_citizens_df.copy()
    out["RecoveryUSD"] = (out["GrossWindLossUSD"] * cov_val * LAE).astype(float)
    out["NetWindUSD"] = out["GrossWindLossUSD"] - out["RecoveryUSD"]
    return out[["Company", "County", "GrossWindLossUSD", "RecoveryUSD", "NetWindUSD"]]

# =============================================================================
# Stage 4: Capital hit
# =============================================================================

def apply_citizens_capital_hit(
    citizens_net_df: pd.DataFrame,
    citizens_capital_row: dict,
    surplus_field: str = "projected_year_end_surplus_usd",
) -> dict:
    """
    Subtract Citizens' net wind after FHCF from starting surplus.

    Parameters
    ----------
    citizens_net_df : pandas.DataFrame
        Must include 'NetWindUSD' by county (or pre-aggregated). The function sums it.
    citizens_capital_row : dict
        Row containing starting surplus (USD) and optional metadata (e.g., year, notes).
    surplus_field : str, default "projected_year_end_surplus_usd"
        Key in `citizens_capital_row` holding the starting surplus.

    Returns
    -------
    dict
        {
          'citizens_net_wind_after_fhcf_usd': float,
          'citizens_starting_surplus_usd'   : float,
          'citizens_ending_surplus_usd'     : float,
          'citizens_ruined'                 : bool,
          'year'                            : (if present),
          'source_note' / 'source_page'     : (if present)
        }
    """
    total_net = float(citizens_net_df["NetWindUSD"].sum())
    starting = float(citizens_capital_row.get(surplus_field, np.nan))
    ending = starting - total_net

    return dict(
        citizens_net_wind_after_fhcf_usd=total_net,
        citizens_starting_surplus_usd=starting,
        citizens_ending_surplus_usd=ending,
        citizens_ruined=bool(ending < 0),
        year=citizens_capital_row.get("year"),
        source_note=citizens_capital_row.get("source_note"),
        source_page=citizens_capital_row.get("source_page"),
    )

def citizens_fhcf_terms_from_cfg_or_csv(
    terms_norm: pd.DataFrame,
    company_keys: pd.DataFrame,
    cfg
) -> pd.DataFrame:
    """
    Return a single-row FHCF terms frame for Citizens:
      columns: ['Company','CoveragePct_norm','RetentionUSD','LimitUSD'].
    Resolution order:
      - if CITIZENS_FHCF_MODE == 'config': build from config
      - else ('auto'): try FHCF CSV by NAIC/StatEntityKey; patch zeros from config; if still zero limit, leave as zero (runner will fallback to coverage-only)
    """
    name = getattr(cfg, "CITIZENS_COMPANY_NAME", "Citizens Property Insurance Corporation")
    naic = str(getattr(cfg, "CITIZENS_NAIC", "10064")).strip()
    stat = str(getattr(cfg, "CITIZENS_STATKEY", "C6949")).strip()

    cov_cfg = float(getattr(cfg, "CITIZENS_FHCF_COVERAGE_PCT", 0.90))
    ret_cfg = float(getattr(cfg, "CITIZENS_FHCF_RETENTION_USD", 0.0))
    lim_cfg = float(getattr(cfg, "CITIZENS_FHCF_LIMIT_USD", 0.0))

    mode = str(getattr(cfg, "CITIZENS_FHCF_MODE", "auto")).lower()

    def _mk(cov, ret, lim):
        return pd.DataFrame([{
            "Company": name,
            "CoveragePct_norm": float(cov),
            "RetentionUSD": float(ret),
            "LimitUSD": float(lim),
        }])

    if mode == "config":
        # Build entirely from config
        return _mk(cov_cfg, ret_cfg, lim_cfg)

    # --- 'auto' path: resolve from FHCF CSV by NAIC/StatEntityKey ---
    cand = pd.DataFrame()
    if "NAIC" in terms_norm.columns:
        cand = terms_norm[terms_norm["NAIC"].astype(str).str.strip() == naic]
    if cand.empty and "StatEntityKey" in terms_norm.columns:
        cand = terms_norm[terms_norm["StatEntityKey"].astype(str).str.strip() == stat]
    # join via company_keys if necessary
    if cand.empty and {"NAIC"}.issubset(terms_norm.columns) and "NAIC" in company_keys.columns:
        ck = company_keys[company_keys["NAIC"].astype(str).str.strip() == naic]
        if not ck.empty:
            cand = terms_norm.merge(ck[["NAIC"]].drop_duplicates(), on="NAIC", how="inner")
    if cand.empty and {"StatEntityKey"}.issubset(terms_norm.columns) and "StatEntityKey" in company_keys.columns:
        ck = company_keys[company_keys["StatEntityKey"].astype(str).str.strip() == stat]
        if not ck.empty:
            cand = terms_norm.merge(ck[["StatEntityKey"]].drop_duplicates(), on="StatEntityKey", how="inner")

    # If still empty, fall back to config-specified row (runner may still choose coverage-only if lim=0)
    if cand.empty:
        return _mk(cov_cfg, ret_cfg, lim_cfg)

    # Normalize required columns
    keep = ["CoveragePct_norm","RetentionUSD","LimitUSD"]
    out = cand.iloc[[0]][[c for c in keep if c in cand.columns]].copy()
    for col in keep:
        if col not in out.columns:
            out[col] = 0.0

    # Patch zeros from config (guardrails)
    for col, cfg_val in [("CoveragePct_norm", cov_cfg), ("RetentionUSD", ret_cfg), ("LimitUSD", lim_cfg)]:
        val = float(pd.to_numeric(out[col], errors="coerce").fillna(0.0).iloc[0])
        if val == 0.0 and cfg_val > 0.0:
            out[col] = cfg_val

    out.insert(0, "Company", name)
    return out[["Company","CoveragePct_norm","RetentionUSD","LimitUSD"]]

