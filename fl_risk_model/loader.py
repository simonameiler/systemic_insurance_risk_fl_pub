"""
loader.py - Data loading utilities for exposure, premium base, Citizens, and NFIP
----------------------------------------------------------------------------------

Provides functions to load and parse data files required by the risk model.

Public API
----------
- load_fhcf_county_exposure
- load_market_share
- load_private_premium_base_from_market_share_xlsx
- load_citizens_premium_base
- load_nfip_county_exposure
- load_nfip_policy_coverage
- load_citizens_county
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import re
from typing import Optional, Tuple


from fl_risk_model.utils import norm_county_name

# Use your normalizer if available; else a safe fallback
try:
    from .utils import norm_county_name as _norm_base
except Exception:
    def _norm_base(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip().lower()

def _strip_suffix(s: str) -> str:
    return re.sub(r"\s+(County|Parish|Borough|City)$", "", str(s), flags=re.I).strip()

def _norm_county_series(series: pd.Series) -> pd.Series:
    return series.astype(str).map(_norm_base).map(_strip_suffix)

def _norm_fips_series(series: pd.Series) -> pd.Series:
    """Robust: accepts strings or floats (12061.0) -> 5-digit FIPS ('12061')."""
    s = pd.to_numeric(series, errors="coerce")
    return s.apply(lambda x: pd.NA if pd.isna(x) else str(int(x)).zfill(5)).astype("string")

def _pick_cov_col(df: pd.DataFrame) -> str:
    """Find a plausible coverage column."""
    for c in ["coverage_in_force", "tiv_usd", "FloodTIV", "coverage", "sum_insured"]:
        if c in df.columns:
            return c
    raise KeyError(
        "NFIP policies CSV missing a coverage column. "
        "Expected one of: coverage_in_force, tiv_usd, FloodTIV, coverage, sum_insured."
    )

def _coerce_currency_or_number(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
         .str.replace(r"[^\d\.\-eE]", "", regex=True)
         .str.replace(r"^\s*$", "0", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _exp_weights(end_year: int, lookback_years: int, half_life: float) -> pd.DataFrame:
    """Exponentially weighted annual weights that sum to 1 over [end-lookback+1..end]."""
    years = list(range(end_year - int(lookback_years) + 1, int(end_year) + 1))
    if not years:
        raise ValueError("Invalid window for EWA weights.")
    decay = np.log(2.0) / float(half_life)
    w = np.array([np.exp(-decay * (end_year - y)) for y in years], dtype=float)
    w = w / w.sum()
    return pd.DataFrame({"year": years, "weight": w})

def _fl_xwalk(county_xwalk: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a crosswalk and return ONLY Florida counties (exact FIPS set).
    Requires columns that map to County + county_fips (case/variant tolerant).
    """
    if county_xwalk is None or not isinstance(county_xwalk, pd.DataFrame):
        raise KeyError("county_xwalk is required to align exposure to the 67 Florida counties.")

    cols = {c.lower(): c for c in county_xwalk.columns}
    county_key = cols.get("county") or cols.get("countyname") or cols.get("name")
    fips_key   = cols.get("county_fips") or cols.get("fips") or cols.get("countyfips")
    if not county_key or not fips_key:
        raise KeyError(
            f"county_xwalk must include County + county_fips columns; got {list(county_xwalk.columns)}"
        )

    xw = county_xwalk[[county_key, fips_key]].drop_duplicates().rename(
        columns={county_key: "County", fips_key: "county_fips"}
    ).copy()
    xw["County"] = _norm_county_series(xw["County"])
    xw["county_fips"] = _norm_fips_series(xw["county_fips"])

    # Keep only FL (12***), drop known bogus like '12000'
    xw = xw[xw["county_fips"].str.startswith("12")].copy()
    xw = xw[xw["county_fips"] != "12000"]

    # Dedup and sanity check ~67 counties
    xw = xw.drop_duplicates(subset=["county_fips"]).sort_values("county_fips")
    if len(xw) != 67:
        # We don't error hard; we proceed but this is a signal your xwalk isn't the canonical 67.
        # You can raise here if you prefer strictness.
        pass
    return xw[["County", "county_fips"]]

# =============================================================================
# FHCF - County exposure
# =============================================================================

def load_fhcf_county_exposure(path, sheet_name=0, header_row=4):
    """
    Load FHCF exposures by county from Excel.

    Parameters
    ----------
    path : str
        Path to the FHCF workbook.
    sheet_name : int or str, default 0
        Worksheet to read.
    header_row : int, default 4
        Zero-based Excel row index containing column headers (row 5 in UI).

    Returns
    -------
    pandas.DataFrame
        Columns:
          - County : str
          - CountyTIV : float
              County total insured value (TIV) in USD units as provided.
          - CountyPct : float
              Share of statewide TIV (% as provided in the sheet).
          - CountyRisks : float
              Number of risks/policies.

    Notes
    -----
    - Drops summary/footer rows like 'Total' and 'TIV in US $ Billions'.
    - Leaves County names exactly as in the source (no normalization here).
    """
    # 1) Read using the correct header row
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)

    # 2) Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 3) Subset & rename to stable schema
    df = df[['County', 'TIV', '% Total TIV', 'Risks']].copy()
    df = df.rename(columns={
        'TIV':          'CountyTIV',
        '% Total TIV':  'CountyPct',
        'Risks':        'CountyRisks'
    })

    # 4) Drop any rows without a valid county
    df = df.dropna(subset=['County'])

    # 5) Exclude summary rows commonly present
    df = df[~df['County'].isin(['Total', 'TIV in US $ Billions'])].reset_index(drop=True)

    return df[['County', 'CountyTIV', 'CountyPct', 'CountyRisks']]

# =============================================================================
# Market share / Premium base (private insurers)
# =============================================================================

def _flatten_two_row_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a 2-row header to single-level columns.

    For columns like ('Direct Premiums Written ($000)', '2024'), this produces
    'Direct Premiums Written ($000) 2024'. Unnamed level entries are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame read from Excel with header=[rowA, rowB].

    Returns
    -------
    pandas.DataFrame
        Copy of df with single-level string columns.
    """
    def flatten(col):
        lvl0, lvl1 = col
        base = str(lvl0).strip()
        if pd.isna(lvl1):
            return base
        s1 = str(lvl1).strip()
        if s1.lower().startswith("unnamed"):
            return base
        try:
            # Interpret numeric-like second level (e.g., '2024') as year suffix.
            y = int(float(s1))
            return f"{base} {y}"
        except ValueError:
            return base
    df = df.copy()
    df.columns = [flatten(c) for c in df.columns.to_list()]
    return df

def _read_market_share_premiums_usd(
    path,
    sheet_name="FL - Individual Co",
    header_rows=(8, 11),
    company_col="Entity *",
    statkey_col="Stat Entity Key",
    premium_metric="Direct Premiums Written ($000)",
    year=2024,
) -> pd.DataFrame:
    """
    Internal helper: read company premiums (USD) from a 2-row header workbook.

    Parameters
    ----------
    path : str
        XLSX path.
    sheet_name : int or str, default "FL - Individual Co"
        Worksheet with company-level rows.
    header_rows : tuple[int, int], default (8, 11)
        Two header rows (0-based) to pass to pandas read_excel.
    company_col : str, default "Entity *"
        Column name for the company display name.
    statkey_col : str, default "Stat Entity Key"
        Column name for the S&P statutory entity key.
    premium_metric : str, default "Direct Premiums Written ($000)"
        Top-level header text for premium metric (in $000).
    year : int, default 2024
        Year column within the metric band to extract.

    Returns
    -------
    pandas.DataFrame
        Columns:
          - Company : str
          - StatEntityKey : str
          - DirectPremiumUSD : float
              Premium in USD (converted from $000).

    Notes
    -----
    - Collapses duplicate keys by summing DirectPremiumUSD and keeping first
      company display name.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=list(header_rows))
    df  = _flatten_two_row_header(raw)

    col_name = f"{premium_metric} {year}"
    missing = [c for c in (company_col, statkey_col, col_name) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in market-share sheet: {missing}")

    out = (
        df[[company_col, statkey_col, col_name]]
        .dropna(subset=[company_col])
        .rename(columns={
            company_col: "Company",
            statkey_col: "StatEntityKey",
            col_name:    "DirectPremium_thousands",
        })
        .copy()
    )
    # $000 -> USD
    out["DirectPremiumUSD"] = (
        pd.to_numeric(out["DirectPremium_thousands"], errors="coerce")
          .fillna(0.0).astype(float) * 1_000.0
    )
    out["StatEntityKey"] = out["StatEntityKey"].astype(str).str.strip()

    # Collapse potential duplicates by key; keep a display name
    out = (
        out.groupby("StatEntityKey", as_index=False)
           .agg({"Company": "first", "DirectPremiumUSD": "sum"})
    )
    return out[["Company", "StatEntityKey", "DirectPremiumUSD"]]

def load_private_premium_base_from_market_share_xlsx(
    path,
    sheet_name="FL - Individual Co",
    header_rows=(8, 11),
    company_col="Entity *",
    statkey_col="Stat Entity Key",
    premium_metric="Direct Premiums Written ($000)",
    year=2024,
) -> pd.DataFrame:
    """
    Extract FIGA-ready premium base with statutory keys.

    Parameters
    ----------
    path, sheet_name, header_rows, company_col, statkey_col, premium_metric, year
        Same semantics as `_read_market_share_premiums_usd`.

    Returns
    -------
    pandas.DataFrame
        Columns:
          - Company : str
          - StatEntityKey : str
          - PremiumUSD : float
    """
    core = _read_market_share_premiums_usd(
        path=path,
        sheet_name=sheet_name,
        header_rows=header_rows,
        company_col=company_col,
        statkey_col=statkey_col,
        premium_metric=premium_metric,
        year=year,
    )
    return core.rename(columns={"DirectPremiumUSD": "PremiumUSD"})

def load_market_share(
    path,
    sheet_name="FL - Individual Co",
    header_rows=(8, 11),
    company_col="Entity *",
    statkey_col="Stat Entity Key",
    premium_metric="Direct Premiums Written ($000)",
    year=2024,
    include_key: bool = False,
):
    """
    Compute market shares from the same premium base.

    Parameters
    ----------
    path, sheet_name, header_rows, company_col, statkey_col, premium_metric, year
        As in `_read_market_share_premiums_usd`.
    include_key : bool, default False
        If True, include 'StatEntityKey' in the result.

    Returns
    -------
    pandas.DataFrame
        If include_key=False:
          - Company, MarketShare{year}
        If include_key=True:
          - Company, StatEntityKey, MarketShare{year}
    """
    core = _read_market_share_premiums_usd(
        path=path,
        sheet_name=sheet_name,
        header_rows=header_rows,
        company_col=company_col,
        statkey_col=statkey_col,
        premium_metric=premium_metric,
        year=year,
    )
    total = core["DirectPremiumUSD"].sum()
    core[f"MarketShare{year}"] = 0.0 if total <= 0 else core["DirectPremiumUSD"] / total

    cols = ["Company", f"MarketShare{year}"]
    if include_key:
        cols.insert(1, "StatEntityKey")
    return core[cols]

# =============================================================================
# Citizens - Premium base
# =============================================================================

def load_citizens_premium_base(
    path_csv: str,
    as_of: Optional[str] = None,
    include_products: Optional[list[str]] = None,
) -> float:
    """
    Sum Citizens premium base for a snapshot (optionally filtered by product).

    Parameters
    ----------
    path_csv : str
        CSV with at least 'as_of' and the premium column.
    as_of : str, optional
        If provided, filter to this exact snapshot date.
    include_products : list[str], optional
        If provided, filter 'product_line' in this list.

    Returns
    -------
    float
        Total premium in USD for the selected slice.

    Notes
    -----
    Accepts 'total_premium' (preferred) or 'premium_usd' column names.
    """
    df = pd.read_csv(path_csv)
    if as_of and "as_of" in df.columns:
        df = df[df["as_of"] == as_of]
    if include_products and "product_line" in df.columns:
        df = df[df["product_line"].isin(include_products)]

    prem_col = "total_premium"
    if prem_col not in df.columns:
        if "premium_usd" in df.columns:
            prem_col = "premium_usd"
        else:
            raise ValueError("Citizens CSV must include 'total_premium' or 'premium_usd'.")

    vals = (
        df[prem_col].astype(str).str.strip()
          .str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (1,234) -> -1,234
          .str.replace(r"[^\d\.\-]", "", regex=True)        # drop $ , spaces, etc.
    )
    citizens_premium_base = float(pd.to_numeric(vals, errors="coerce").fillna(0.0).sum())
    return citizens_premium_base

# =============================================================================
# NFIP - Exposure (coverage) & policies
# =============================================================================

def load_nfip_county_exposure(
    path,
    state: str = "FL",
    county_col: str = "County",
    coverage_col: str = "Total Coverage",
    state_col: str = "State"
):
    """
    Load NFIP 'Total Coverage in Force' by county from FEMA CSV.

    Parameters
    ----------
    path : str
        CSV path.
    state : str, default "FL"
        Two-letter state filter if state column present.
    county_col : str, default "County"
        County name column in the CSV.
    coverage_col : str, default "Total Coverage"
        Coverage-in-force column in the CSV.
    state_col : str, default "State"
        State column name (if present).

    Returns
    -------
    pandas.DataFrame
        Columns:
          - County : str (normalized to title-case, common aliases unified)
          - FloodTIV : float
    """
    df = pd.read_csv(path)

    # Filter to state if present
    if state_col in df.columns:
        df = df[df[state_col].astype(str).str.upper().eq(state.upper())].copy()

    out = (df.groupby(county_col, as_index=False)[coverage_col]
             .sum()
             .rename(columns={county_col: "County", coverage_col: "FloodTIV"}))

    out["County"] = out["County"].map(norm_county_name)
    out["FloodTIV"] = pd.to_numeric(out["FloodTIV"], errors="coerce").fillna(0.0).astype(float)
    return out[["County", "FloodTIV"]]


def load_nfip_policy_coverage(
    path: str,
    mode: str = "FIXED_YEAR",           # "FIXED_YEAR" | "EWA_5Y"
    year: int = 2024,                   # anchor year
    lookback_years: int = 5,
    half_life: float = 2.0,
    county_xwalk: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load and aggregate NFIP coverage, aligned to Florida's 67 counties by 5-digit FIPS.

    Input CSV may contain either:
      - 'county_fips', 'year', and coverage column (preferred), or
      - 'County', 'year', and coverage column (we'll attach FIPS via county_xwalk)

    Returns (always):
      DataFrame with EXACTLY the Florida FIPS set from county_xwalk:
        ['county_fips', 'FloodTIV']   (zeros where exposure is missing)
    """
    if county_xwalk is None:
        raise KeyError("county_xwalk is required so we can align to the 67 Florida counties.")

    # --- Read + basic columns ---
    df = pd.read_csv(path)
    cov_col = _pick_cov_col(df)
    df[cov_col] = _coerce_currency_or_number(df[cov_col]).fillna(0.0)

    if "county_fips" in df.columns:
        df["county_fips"] = _norm_fips_series(df["county_fips"])

    # Florida only (NA-safe)
    fl_mask = df["county_fips"].str.startswith("12", na=False)
    df = df.loc[fl_mask].copy()

    # Drop the statewide aggregate ('12000'), NA-safe
    df = df.loc[df["county_fips"].notna() & df["county_fips"].ne("12000")].copy()

    # Year handling
    if "year" not in df.columns:
        if mode.upper() == "FIXED_YEAR":
            df["year"] = int(year)  # assume single-year file
        else:
            raise KeyError("EWA_5Y mode requires a 'year' column in the NFIP policies CSV.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # county_fips handling (attach via County if needed)
    if "county_fips" not in df.columns:
        if "County" not in df.columns:
            raise KeyError("NFIP policies CSV must have either 'county_fips' or 'County' to attach FIPS.")
        df["County"] = _norm_county_series(df["County"])
        xw = _fl_xwalk(county_xwalk)
        df = df.merge(xw, on="County", how="left")
    else:
        df["county_fips"] = _norm_fips_series(df["county_fips"])

    # Normalize keys and Florida-only filter now that FIPS is present
    df["county_fips"] = _norm_fips_series(df["county_fips"])
    df = df[df["county_fips"].str.startswith("12")]            # Florida only
    df = df[df["county_fips"] != "12000"]                      # drop bogus

    # --- Aggregate coverage by mode ---
    m = mode.upper()
    if m == "FIXED_YEAR":
        use = df[df["year"].eq(int(year))].copy()
        if use.empty:
            latest = pd.to_numeric(df["year"], errors="coerce").dropna().max()
            if pd.isna(latest):
                raise ValueError("No valid 'year' values in NFIP policies CSV.")
            use = df[df["year"].eq(int(latest))].copy()
        agg = use.groupby("county_fips", as_index=False)[cov_col].sum()

    elif m == "EWA_5Y":
        weights = _exp_weights(end_year=int(year), lookback_years=int(lookback_years), half_life=float(half_life))
        win = df.merge(weights, on="year", how="inner")
        if win.empty:
            avail = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
            if avail.empty:
                raise ValueError("No years available in NFIP policies CSV for EWA.")
            end = int(min(int(year), int(avail.max())))
            start = int(max(int(avail.min()), end - int(lookback_years) + 1))
            weights = _exp_weights(end_year=end, lookback_years=end - start + 1, half_life=float(half_life))
            win = df.merge(weights, on="year", how="inner")
        win["_wcov"] = win[cov_col] * win["weight"]
        agg = (win.groupby("county_fips", as_index=False)["_wcov"].sum()
                  .rename(columns={"_wcov": cov_col}))
    else:
        raise ValueError("mode must be one of {'FIXED_YEAR','EWA_5Y'}")

    agg = agg.rename(columns={cov_col: "FloodTIV"})
    agg["FloodTIV"] = pd.to_numeric(agg["FloodTIV"], errors="coerce").fillna(0.0)

    # --- ALIGN to the exact 67 Florida counties from the crosswalk ---
    xw_fl = _fl_xwalk(county_xwalk)  # ['County','county_fips'] for FL only
    out = xw_fl.merge(agg[["county_fips", "FloodTIV"]], on="county_fips", how="left")
    out["FloodTIV"] = pd.to_numeric(out["FloodTIV"], errors="coerce").fillna(0.0)

    # Return stable schema for runner: FIPS + FloodTIV only
    return out[["county_fips", "FloodTIV"]]

def load_nfip_fl_premium_base(
    path: str,
    year: int | None = None,
    *,
    year_col_candidates: tuple[str, ...] = ("year", "policy_year", "calendar_year"),
    premium_col_candidates: tuple[str, ...] = (
        "premium_usd", "written_premium_usd", "written_premium",
        "premium", "total_premium_usd", "total_premium",
    ),
    state_col_candidates: tuple[str, ...] = ("state", "state_code", "policy_state", "st"),
) -> float:
    """
    Sum Florida NFIP written premium for the selected year (if present).
    Accepts county-level or already-aggregated state data.
    """
    import pandas as pd

    df = pd.read_csv(path)

    # Pick year column (optional)
    low = {c.lower(): c for c in df.columns}
    ycol = next((low[c] for c in year_col_candidates if c in low), None)
    if year is not None and ycol is not None:
        df = df[df[ycol].astype(float).astype(int).eq(int(year))]

    # Pick premium column
    pcol = next((low[c] for c in premium_col_candidates if c in low), None)
    if pcol is None:
        # broader fallback: first column that contains 'premium'
        cand = [c for c in df.columns if "premium" in c.lower()]
        if not cand:
            raise KeyError("NFIP premium-base CSV is missing a premium column.")
        pcol = cand[0]

    # If state column exists, filter to Florida
    scol = next((low[c] for c in state_col_candidates if c in low), None)
    if scol is not None:
        s = df[scol].astype(str).str.upper().str.strip()
        df = df[s.isin(["FL", "FLORIDA", "12"])].copy()

    # If county_fips exists, filter to Florida by FIPS and drop 12000
    if "county_fips" in df.columns:
        df["county_fips"] = _norm_fips_series(df["county_fips"])
        df = df[df["county_fips"].str.startswith("12")]
        df = df[df["county_fips"] != "12000"]

    prem = _coerce_currency_or_number(df[pcol]).fillna(0.0)
    return float(prem.sum())


# =============================================================================
# Citizens - County view snapshot (exposure & premiums)
# =============================================================================

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Best-effort numeric parser: strips commas, currency, and apostrophes.

    Parameters
    ----------
    series : pandas.Series

    Returns
    -------
    pandas.Series
        Float dtype with NaNs where coercion fails.
    """
    s = series.astype(str).str.replace(r"[,$']", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def _maybe_to_fips(df: pd.DataFrame, county_col: str, xwalk: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Optionally attach county_fips using a crosswalk; zero-pads to 5 digits.

    Parameters
    ----------
    df : pandas.DataFrame
    county_col : str
        Column in df containing county names.
    xwalk : pandas.DataFrame or None
        Crosswalk with ['County','county_fips'].

    Returns
    -------
    pandas.DataFrame
        df with 'county_fips' column attached (or preserved if already present).
    """
    if xwalk is None or county_col not in df.columns:
        return df.assign(county_fips=pd.NA) if "county_fips" not in df.columns else df
    xx = xwalk.copy()
    xx["county_name_norm"] = xx["County"].map(norm_county_name)
    out = df.copy()
    out["county_name_norm"] = out[county_col].map(norm_county_name)
    out = out.merge(xx[["county_name_norm","county_fips"]], on="county_name_norm", how="left")
    out.drop(columns=["county_name_norm"], inplace=True)
    # zero-pad 5
    out["county_fips"] = out["county_fips"].astype(str).str.replace(r"\D","",regex=True).str.zfill(5)
    return out

def load_citizens_county(
    path_csv: str,
    as_of: Optional[str] = None,
    include_products: Optional[list[str]] = None,
    county_xwalk: Optional[pd.DataFrame] = None,
    *,
    mode: str = "FIXED_YEAR",          # NEW: "FIXED_YEAR" | "EWA_5Y"
    lookback_years: int = 5,           # NEW
    half_life: float = 2.0,            # NEW
) -> pd.DataFrame:
    """
    Load harmonized Citizens "County View" CSV and aggregate to a single snapshot
    (FIXED_YEAR) or to a 5y exponentially weighted average (EWA_5Y).

    Returns columns:
      {'as_of','product_line','county','county_fips','tiv_usd','premium_usd','pif','building_count'}
    """
    df = pd.read_csv(path_csv)

    # --- normalize column names we care about (robust to variants)
    # required base cols
    for req in ("as_of", "county"):
        if req not in df.columns:
            raise ValueError(f"Citizens CSV must include '{req}'")

    # optional/variant numeric columns
    tiv_col_candidates = ["total_exposure", "tiv_usd", "tiv", "exposure_usd"]
    prem_col_candidates = ["total_premium", "premium_usd", "prem_usd"]
    pif_col_candidates = ["policies_in_force", "pif"]
    bldg_col_candidates = ["building_count", "buildings"]

    def first_present(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    tiv_col   = first_present(tiv_col_candidates)
    prem_col  = first_present(prem_col_candidates)
    pif_col   = first_present(pif_col_candidates)
    bldg_col  = first_present(bldg_col_candidates)

    # parse & clean
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df["county"] = df["county"].map(norm_county_name)

    if "product_line" not in df.columns:
        df["product_line"] = "ALL"

    if tiv_col is None:
        raise ValueError("Citizens CSV must include a TIV column like 'total_exposure' or 'tiv_usd'.")

    df["tiv_usd"] = _coerce_numeric(df[tiv_col]).fillna(0.0)
    if prem_col is not None:
        df["premium_usd"] = _coerce_numeric(df[prem_col]).fillna(0.0)
    else:
        df["premium_usd"] = 0.0
    if pif_col is not None:
        df["pif"] = pd.to_numeric(df[pif_col], errors="coerce").fillna(0.0)
    else:
        df["pif"] = 0.0
    if bldg_col is not None:
        df["building_count"] = pd.to_numeric(df[bldg_col], errors="coerce").fillna(0.0)
    else:
        df["building_count"] = 0.0

    # optional filter
    if include_products:
        df = df[df["product_line"].isin(include_products)].copy()

    # attach FIPS if crosswalk provided (keeps NA if not resolvable)
    df = _maybe_to_fips(df, county_col="county", xwalk=county_xwalk)

    mode_u = str(mode).upper()

    if mode_u == "FIXED_YEAR":
        # pick a single snapshot, robust to exact-date misses
        as_of_ts = pd.to_datetime(as_of) if as_of else None

        if as_of_ts is not None:
            snap = df[df["as_of"] == as_of_ts].copy()
            if snap.empty:
                # fallback to any row in the same year
                yr = as_of_ts.year
                snap = df[df["as_of"].dt.year.eq(yr)].copy()
            if snap.empty:
                # final fallback: latest date in the filtered dataset
                snap = df[df["as_of"] == df["as_of"].max()].copy()
        else:
            # no as_of provided -> use latest date
            snap = df[df["as_of"] == df["as_of"].max()].copy()

        # Ensure 'county_fips' column exists even when no crosswalk is provided
        if "county_fips" not in snap.columns:
            snap["county_fips"] = pd.NA

        # aggregate that snapshot by county
        out = (
            snap.groupby(["county", "county_fips", "product_line"], as_index=False, dropna=False)[
                ["tiv_usd", "premium_usd", "pif", "building_count"]
            ].sum()
        )

        # Stamp the actual snapshot date used
        as_of_val = pd.to_datetime(snap["as_of"].max())
        out.insert(0, "as_of", as_of_val.strftime("%Y-%m-%d"))

        return out[["as_of","product_line","county","county_fips","tiv_usd","premium_usd","pif","building_count"]]

    elif mode_u == "EWA_5Y":
        # anchor year
        anchor_year = (
            pd.to_datetime(as_of).year if as_of is not None
            else int(df["as_of"].dt.year.max())
        )
        df["year"] = df["as_of"].dt.year

        # restrict to window
        lo = anchor_year - int(lookback_years) + 1
        win = df[(df["year"] >= lo) & (df["year"] <= anchor_year)].copy()
        if win.empty:
            anchor_year = int(df["year"].max())
            lo = anchor_year - int(lookback_years) + 1
            win = df[(df["year"] >= lo) & (df["year"] <= anchor_year)].copy()

        # year weights (half-life in years)
        weights = (
            pd.DataFrame({"year": np.arange(lo, anchor_year + 1, dtype=int)})
            .assign(weight=lambda x: (0.5 ** ((anchor_year - x["year"]) / float(half_life))).astype(float))
        )

        # aggregate by county×year (sum across product lines inside the year)
        grp = (
            win.groupby(["county", "county_fips", "year"], as_index=False, dropna=False)[
                ["tiv_usd", "premium_usd", "pif", "building_count"]
            ].sum()
        )

        # attach weights by year
        grp = grp.merge(weights, on="year", how="left")

        # apply weights and collapse the window
        for col in ["tiv_usd", "premium_usd", "pif", "building_count"]:
            grp[f"w_{col}"] = grp[col] * grp["weight"]

        agg = grp.groupby(["county", "county_fips"], as_index=False, dropna=False)[
            ["w_tiv_usd", "w_premium_usd", "w_pif", "w_building_count", "weight"]
            ].sum().rename(columns={"weight": "w_sum"})

        # weighted mean; guard div-by-zero
        def _safe_div(n, d):
            n = pd.to_numeric(n, errors="coerce").fillna(0.0)
            d = pd.to_numeric(d, errors="coerce").fillna(0.0)
            out = np.zeros(len(n), dtype=float)
            mask = d > 0
            out[mask] = (n[mask] / d[mask]).astype(float)
            return out

        out = pd.DataFrame({
            "as_of": f"{anchor_year}-12-31",
            "product_line": "ALL (EWA)",
            "county": agg["county"],
            "county_fips": agg["county_fips"],
            "tiv_usd": _safe_div(agg["w_tiv_usd"], agg["w_sum"]),
            "premium_usd": _safe_div(agg["w_premium_usd"], agg["w_sum"]),
            "pif": _safe_div(agg["w_pif"], agg["w_sum"]),
            "building_count": _safe_div(agg["w_building_count"], agg["w_sum"]),
        })

        for c in ["tiv_usd", "premium_usd", "pif", "building_count"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

        return out[
            ["as_of", "product_line", "county", "county_fips", "tiv_usd", "premium_usd", "pif", "building_count"]
        ]

    else:
        raise ValueError("mode must be 'FIXED_YEAR' or 'EWA_5Y'")
