# fl_risk_model/nfip.py
"""
nfip.py — NFIP payouts and payout-rate helpers
----------------------------------------------

Implements utilities to:
1) load NFIP penetration rates by county,
2) carve out insured flood losses from total flood losses using NFIP penetration rates.
3) Load county–year NFIP claims paid totals from a normalized CSV.
4) Extract the latest year present in a claims table.
5) Build **county payout rates** using recent-history weighting, winsorization,
   capping, and Empirical-Bayes shrinkage toward the statewide rate.

This aligns with the Methods: a five-year, recency-weighted analysis with
winsorization and EB shrinkage; rates are applied to *insured* flood and are
capped at 100%.

Public API
----------
- load_nfip_penetration(path, xwalk)
- carveout_flood_from_penetration(water_df, pen_df, xwalk)
- load_nfip_claims_county_year(path)
- aggregate_nfip_claims
- latest_year_in(df)
- make_nfip_payout_rates(claims_cy, flood_tiv, county_xwalk, ...)

Notes
-----
- Money is USD. Rates are unitless in [0,1].
- Coverage base is **FloodTIV** (coverage in force).
- We prefer **county_fips** for joins; fall back to County name → FIPS via `county_xwalk`.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

import re
from fl_risk_model.utils import norm_county_name

def _strip_suffix(s: str) -> str:
    if s is None:
        return None
    return re.sub(r"\s+(County|Parish|Borough|City)$", "", str(s), flags=re.I).strip()

def _norm_county_series(s):
    return s.astype(str).map(norm_county_name).map(_strip_suffix)

def load_nfip_penetration(path, xwalk):
    """
    Load NFIP penetration with flexible schemas.

    Acceptable columns:
      - FIPS: any of ['fipsCode','county_fips','FIPS','fips','CountyFIPS']
      - County name (fallback): any of ['County','CountyName','county'] + xwalk required
      - Rates: ['resPenetrationRate','resPenetrationRateSfha']
      - Stock: ['totalResStructures','totalResStructuresSfha']
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(path, dtype=str)

    # --- FIPS detection / construction ---
    fips_col_candidates = ["fipsCode", "county_fips", "FIPS", "fips", "CountyFIPS"]
    fips_col = next((c for c in fips_col_candidates if c in df.columns), None)

    if fips_col is not None:
        df["county_fips"] = (
            df[fips_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
        )
    else:
        # Try to construct from County name via xwalk
        county_col_candidates = ["County", "CountyName", "county"]
        county_col = next((c for c in county_col_candidates if c in df.columns), None)
        if county_col is None:
            raise KeyError(
                "NFIP penetration CSV must include a FIPS column "
                f"({fips_col_candidates}) or a County column ({county_col_candidates})."
            )
        # Normalize and map to FIPS via xwalk
        from .utils import norm_county_name
        df = df.rename(columns={county_col: "County"})
        df["County"] = df["County"].astype(str).map(norm_county_name)
        xw = xwalk[["County", "county_fips"]].drop_duplicates()
        df = df.merge(xw, on="County", how="left")

    if "county_fips" not in df.columns:
        raise KeyError("Could not construct 'county_fips' for NFIP penetration.")

    df["county_fips"] = (
        df["county_fips"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    )

    # --- numeric columns (support slight header drift) ---
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    r_all_col  = pick("resPenetrationRate", "penetrationRate", "res_rate")
    r_sfha_col = pick("resPenetrationRateSfha", "penetrationRateSfha", "res_rate_sfha")
    n_all_col  = pick("totalResStructures", "residentialStructures", "res_stock")
    n_sfha_col = pick("totalResStructuresSfha", "residentialStructuresSfha", "res_stock_sfha")

    for need, opts in [
        ("resPenetrationRate", (r_all_col, ["resPenetrationRate"])),
        ("resPenetrationRateSfha", (r_sfha_col, ["resPenetrationRateSfha"])),
        ("totalResStructures", (n_all_col, ["totalResStructures"])),
        ("totalResStructuresSfha", (n_sfha_col, ["totalResStructuresSfha"])),
    ]:
        if opts[0] is None:
            raise KeyError(f"NFIP penetration CSV missing required column like {opts[1][0]!r}.")
    # Coerce
    df["resPenetrationRate"]      = pd.to_numeric(df[r_all_col], errors="coerce").clip(0, 1)
    df["resPenetrationRateSfha"]  = pd.to_numeric(df[r_sfha_col], errors="coerce").clip(0, 1)
    df["totalResStructures"]      = pd.to_numeric(df[n_all_col], errors="coerce")
    df["totalResStructuresSfha"]  = pd.to_numeric(df[n_sfha_col], errors="coerce")

    # --- derive SFHA share and effective county take-up ---
    s_sfha = (df["totalResStructuresSfha"] / df["totalResStructures"]).clip(lower=0, upper=1).fillna(0.0)
    r_all  = df["resPenetrationRate"].fillna(0.0)
    r_sfha = df["resPenetrationRateSfha"].fillna(0.0)

    denom = (1 - s_sfha).replace(0, pd.NA)
    r_non = ((r_all - s_sfha * r_sfha) / denom).clip(lower=0, upper=1).fillna(0.0)
    r_eff = (s_sfha * r_sfha + (1 - s_sfha) * r_non).clip(lower=0, upper=1)

    return df[["county_fips"]].assign(
        NFIP_r_eff=r_eff, NFIP_r_sfha=r_sfha, NFIP_r_non=r_non, NFIP_s_sfha=s_sfha
    )

# def carveout_flood_from_penetration(water_df, pen_df, xwalk):
#     # water_df: ['County','GrossFloodLossUSD']
#     w = water_df.merge(xwalk[["County","county_fips"]].drop_duplicates(), on="County", how="left")
#     w["county_fips"] = w["county_fips"].astype(str).str.replace(r"\D","",regex=True).str.zfill(5)
#     w["GrossFloodLossUSD"] = pd.to_numeric(w["GrossFloodLossUSD"], errors="coerce").fillna(0.0)

#     p = pen_df[["county_fips","NFIP_r_eff"]].copy()
#     out = w.merge(p, on="county_fips", how="left").fillna({"NFIP_r_eff": 0.0})

#     insured_cand = out["NFIP_r_eff"] * out["GrossFloodLossUSD"]
#     out["InsuredFloodUSD"]     = insured_cand
#     out["UninsuredFloodUSD"]   = out["GrossFloodLossUSD"] - insured_cand
#     out["UnderinsuredFloodUSD"] = 0.0  # filled after capping by FloodTIV
#     return out[["County","county_fips","GrossFloodLossUSD","InsuredFloodUSD","UninsuredFloodUSD","UnderinsuredFloodUSD"]]

def carveout_flood_from_penetration(water_df, pen_df, xwalk=None,
                                     county_col="County",
                                     loss_col=None):
    """
    Split flood losses using NFIP penetration.

    water_df columns: must include either `loss_col` or one of:
        - 'GrossFloodLossUSD' or 'WaterDamageUSD'
        and either:
        - 'county_fips' OR a county name column (default 'County') plus xwalk.

    pen_df columns: 'county_fips','NFIP_r_eff'
    xwalk columns (flexible): something like ('County' or 'CountyName' or 'county') and
                              ('county_fips' or 'FIPS' or 'fips')
    """
    w = water_df.copy()

    # Normalize the loss column
    if loss_col is None:
        if "GrossFloodLossUSD" in w.columns:
            loss_col = "GrossFloodLossUSD"
        elif "WaterDamageUSD" in w.columns:
            w = w.rename(columns={"WaterDamageUSD": "GrossFloodLossUSD"})
            loss_col = "GrossFloodLossUSD"
        else:
            raise KeyError("water_df must include 'GrossFloodLossUSD' or 'WaterDamageUSD'.")
    elif loss_col != "GrossFloodLossUSD":
        w = w.rename(columns={loss_col: "GrossFloodLossUSD"})

    # Clean county
    if county_col != "County" and county_col in w.columns:
        w = w.rename(columns={county_col: "County"})

    # 1) Normalize the incoming water county names
    if "County" in w.columns:
        w = w[w["County"].notna() & (w["County"].astype(str).str.lower() != "none")]
        w["County"] = _norm_county_series(w["County"])

    # 2) If county_fips missing, build it from xwalk (normalize xwalk too)
    if "county_fips" not in w.columns:
        if xwalk is None:
            raise KeyError("county_fips missing in water_df and no xwalk provided.")
        xl = {c.lower(): c for c in xwalk.columns}
        county_key = xl.get("county") or xl.get("countyname") or xl.get("name")
        fips_key   = xl.get("county_fips") or xl.get("fips") or xl.get("countyfips")
        if not county_key or not fips_key:
            raise KeyError(f"xwalk must have county and fips columns; got {list(xwalk.columns)}")

        xw = xwalk[[county_key, fips_key]].drop_duplicates().rename(
            columns={county_key: "County", fips_key: "county_fips"}
        ).copy()
        xw["County"] = _norm_county_series(xw["County"])

        w = w.merge(xw, on="County", how="left")

    # 3) Normalize FIPS and losses
    w["county_fips"] = (
        w["county_fips"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    )
    w["GrossFloodLossUSD"] = pd.to_numeric(w["GrossFloodLossUSD"], errors="coerce").fillna(0.0)

    # 4) Fail fast if any 00000 remain
    bad = (w["county_fips"] == "00000") | (w["county_fips"].isna())
    if bad.any():
        examples = sorted(w.loc[bad, "County"].dropna().unique().tolist()[:10])
        raise RuntimeError(f"Could not map County→FIPS for {bad.sum()} rows. Examples: {examples}")

    # 5) Proceed with penetration join...
    p = pen_df[["county_fips","NFIP_r_eff"]].copy()
    out = w.merge(p, on="county_fips", how="left").fillna({"NFIP_r_eff": 0.0})
    insured_cand = out["NFIP_r_eff"] * out["GrossFloodLossUSD"]
    out["InsuredFloodUSD"]      = insured_cand
    out["UninsuredFloodUSD"]    = out["GrossFloodLossUSD"] - insured_cand
    out["UnderinsuredFloodUSD"] = 0.0
    return out[["County","county_fips","GrossFloodLossUSD","InsuredFloodUSD","UninsuredFloodUSD","UnderinsuredFloodUSD"]]
    # if "County" in w.columns:
    #     w = w[w["County"].notna() & (w["County"].astype(str).str.lower() != "none")]

    # # Attach county_fips if missing
    # if "county_fips" not in w.columns:
    #     if xwalk is None:
    #         raise KeyError("county_fips missing in water_df and no xwalk provided.")

    #     # Flexible xwalk normalization
    #     xl = {c.lower(): c for c in xwalk.columns}
    #     county_key = xl.get("county") or xl.get("countyname") or xl.get("name")
    #     fips_key   = xl.get("county_fips") or xl.get("fips") or xl.get("countyfips")

    #     if not county_key or not fips_key:
    #         raise KeyError(f"xwalk must have county and fips columns; got {list(xwalk.columns)}")

    #     xw = xwalk[[county_key, fips_key]].drop_duplicates().rename(
    #         columns={county_key: "County", fips_key: "county_fips"}
    #     )
    #     w = w.merge(xw, on="County", how="left")

    # # Normalize FIPS and numeric loss
    # w["county_fips"] = (
    #     w["county_fips"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    # )
    # w["GrossFloodLossUSD"] = pd.to_numeric(w["GrossFloodLossUSD"], errors="coerce").fillna(0.0)

    # # Join NFIP effective take-up
    # p = pen_df[["county_fips", "NFIP_r_eff"]].copy()
    # out = w.merge(p, on="county_fips", how="left").fillna({"NFIP_r_eff": 0.0})

    # # Compute buckets
    # insured_cand = out["NFIP_r_eff"] * out["GrossFloodLossUSD"]
    # out["InsuredFloodUSD"]       = insured_cand
    # out["UninsuredFloodUSD"]     = out["GrossFloodLossUSD"] - insured_cand
    # out["UnderinsuredFloodUSD"]  = 0.0  # filled later after capping by FloodTIV

    # return out[[
    #     "County", "county_fips", "GrossFloodLossUSD",
    #     "InsuredFloodUSD", "UninsuredFloodUSD", "UnderinsuredFloodUSD"
    # ]]



def load_nfip_claims_county_year(path: str) -> pd.DataFrame:
    """
    Load a county–year NFIP claims CSV and normalize schema.

    Parameters
    ----------
    path : str
        CSV path. Must contain:
          - 'county_fips' : str/int
          - 'year'        : int
          - one of {'nfip_paid_total_usd','claims_gross_usd','claims_net_usd'}

    Returns
    -------
    pandas.DataFrame
        Columns:
          - county_fips : str (zero-padded 5)
          - year        : pandas Int64
          - nfip_paid_total_usd : float

    Raises
    ------
    ValueError
        If no paid column is found.
    """
    df = pd.read_csv(path)

    # Normalize FIPS and year
    df["county_fips"] = (
        df["county_fips"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(5)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Accept either harmonized names or legacy names
    paid_cols = [c for c in ["nfip_paid_total_usd", "claims_gross_usd", "claims_net_usd"] if c in df.columns]
    if not paid_cols:
        raise ValueError(
            "NFIP claims CSV must include one of: "
            "'nfip_paid_total_usd', 'claims_gross_usd', or 'claims_net_usd'."
        )

    df["nfip_paid_total_usd"] = pd.to_numeric(df[paid_cols[0]], errors="coerce").fillna(0.0)
    return df[["county_fips", "year", "nfip_paid_total_usd"]].dropna(subset=["county_fips"])

def aggregate_nfip_claims(
    claims_cy,
    mode: str = "FIXED_YEAR",
    year: int = 2024,
    lookback_years: int = 5,
    half_life: float = 2.0,
) -> pd.DataFrame:
    """
    Return county-level NFIP paid claims (USD) using FIXED_YEAR or 5y EWA.
    Output columns: ['county_fips', 'nfip_paid_total_usd']  ← matches tests
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd

    # coerce to DataFrame
    if isinstance(claims_cy, (str, Path)):
        df = pd.read_csv(claims_cy)
    else:
        df = claims_cy.copy()

    # normalize keys
    county_col = next((c for c in ["county_fips","CountyFIPS","County","county"] if c in df.columns), None)
    if county_col is None:
        raise ValueError("NFIP claims must include a county FIPS column (e.g., 'county_fips').")
    df["county_fips"] = df[county_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)

    if "year" not in df.columns:
        raise ValueError("NFIP claims must include 'year' column.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    paid_col = next((c for c in [
        "nfip_paid_total_usd","nfip_paid","paid_total_usd","paid_usd","total_paid_usd"
    ] if c in df.columns), None)
    if paid_col is None:
        raise ValueError("NFIP claims must include a paid column like 'nfip_paid_total_usd'.")
    df["paid_val"] = pd.to_numeric(df[paid_col], errors="coerce").fillna(0.0)

    if mode.upper() == "FIXED_YEAR":
        y0 = int(year)
        out = (
            df.loc[df["year"].eq(y0), ["county_fips","paid_val"]]
              .groupby("county_fips", as_index=False)["paid_val"].sum()
              .rename(columns={"paid_val": "nfip_paid_total_usd"})
        )

    elif mode.upper() == "EWA_5Y":
        y0 = int(year)
        lo = y0 - int(lookback_years) + 1
        win = df[df["year"].between(lo, y0)].copy()
        win["age"] = (y0 - win["year"]).astype(float)
        win["w"] = (0.5 ** (win["age"] / float(half_life))).astype(float)
        win["w_paid"] = win["paid_val"] * win["w"]

        agg = win.groupby("county_fips", as_index=False)[["w_paid","w"]].sum()
        agg["nfip_paid_total_usd"] = np.where(agg["w"] > 0, agg["w_paid"] / agg["w"], 0.0)
        out = agg[["county_fips","nfip_paid_total_usd"]].copy()
        out["nfip_paid_total_usd"] = pd.to_numeric(out["nfip_paid_total_usd"], errors="coerce").fillna(0.0)

    else:
        raise ValueError("mode must be 'FIXED_YEAR' or 'EWA_5Y'")

    return out


def latest_year_in(df: pd.DataFrame) -> int:
    """
    Return the latest (max) year present in a 'year' column, or current year if none.

    Parameters
    ----------
    df : pandas.DataFrame
        Must include a 'year' column.

    Returns
    -------
    int
        Latest year present, or today's year if empty.
    """
    y = df["year"].dropna()
    return int(y.max()) if not y.empty else pd.Timestamp.today().year


def make_nfip_payout_rates(
    claims_cy: pd.DataFrame,       # from load_nfip_claims_county_year
    flood_tiv: pd.DataFrame,       # ['County','FloodTIV'] or may include 'county_fips'
    county_xwalk: pd.DataFrame,    # ['County','county_fips'] for names→FIPS
    window_years: int = 5,
    end_year: int | None = None,
    weighting: str = "exp",        # {'uniform','exp'}
    half_life_years: float = 2.0,  # only used when weighting == 'exp'
    cap: float = 1.0,              # hard cap on rate
    winsor_q: float = 0.99,        # winsorization quantile
    shrink_tau: float = 5e7,       # EB shrink prior "coverage scale" (USD)
) -> pd.DataFrame:
    """
    Build **county payout rates** with recency weighting, winsorization, cap, and EB shrinkage.

    Rate definition
    ---------------
    raw_rate(county) = (Σ_y paid_y * weight_y) / FloodTIV_county

    Post-processing
    ---------------
    - winsorize raw_rate at `winsor_q`
    - clip to [0, cap]
    - compute statewide_rate = (Σ paid_w) / (Σ FloodTIV)
    - EB shrink: r_hat = (n/(n+τ)) * raw + (τ/(n+τ)) * statewide_rate, where n=FloodTIV

    Parameters
    ----------
    claims_cy : pandas.DataFrame
        Output of `load_nfip_claims_county_year`.
    flood_tiv : pandas.DataFrame
        NFIP exposure by county:
          - must have 'FloodTIV'
          - if missing 'county_fips', we join via `county_xwalk`
    county_xwalk : pandas.DataFrame
        Name→FIPS crosswalk with ['County','county_fips'].
    window_years : int, default 5
        Number of trailing years to include.
    end_year : int or None, default None
        Last year to include; if None, uses current year.
    weighting : {'uniform','exp'}, default 'exp'
        'exp' applies exponential recency weights with half-life `half_life_years`.
    half_life_years : float, default 2.0
        Half-life for exponential weights.
    cap : float, default 1.0
        Upper cap for rates.
    winsor_q : float, default 0.99
        Winsorization quantile for raw rates before capping.
    shrink_tau : float, default 5e7
        Empirical-Bayes shrinkage scale in USD (acts like prior “exposure” strength).

    Returns
    -------
    pandas.DataFrame
        Columns:
          - county_fips : str
          - nfip_payout_rate : float in [0, 1]
    """
    claims = claims_cy.copy()
    if end_year is None:
        end_year = int(pd.to_datetime("today").year)
    start_year = end_year - window_years + 1
    claims = claims[(claims["year"] >= start_year) & (claims["year"] <= end_year)].copy()

    # Year weights
    if weighting == "exp":
        years = np.arange(start_year, end_year + 1)
        k = np.log(2) / max(half_life_years, 0.1)  # avoid divide-by-zero
        w_map = {y: float(np.exp(-k * (end_year - y))) for y in years}
    else:  # 'uniform' or anything else → treat as uniform
        w_map = {int(y): 1.0 for y in claims["year"].dropna().unique()}

    claims["w"] = claims["year"].map(w_map).fillna(0.0)
    claims["paid_w"] = claims["nfip_paid_total_usd"] * claims["w"]

    # Normalize County→FIPS and attach current FloodTIV
    xw = county_xwalk.copy()
    xw["County"] = xw["County"].astype(str).str.strip()
    xw["county_fips"] = xw["county_fips"].astype(str).str.zfill(5)

    tiv = flood_tiv.copy()
    if "county_fips" not in tiv.columns:
        tiv = tiv.merge(xw, on="County", how="left")
    tiv["FloodTIV"] = pd.to_numeric(tiv["FloodTIV"], errors="coerce").fillna(0.0)

    paid_w = (
        claims.groupby("county_fips", dropna=False)["paid_w"]
        .sum(min_count=1)
        .rename("paid_w_sum")
        .reset_index()
    )
    rates = tiv.merge(paid_w, on="county_fips", how="left").fillna({"paid_w_sum": 0.0})
    rates["raw_rate"] = (rates["paid_w_sum"] / rates["FloodTIV"]).replace([np.inf, -np.inf], 0.0)

    # Winsorize then cap
    hi = rates["raw_rate"].quantile(winsor_q)
    rates["raw_rate"] = rates["raw_rate"].clip(lower=0.0, upper=min(float(hi), cap))

    # Statewide baseline for EB shrinkage
    state_paid = float(paid_w["paid_w_sum"].sum())
    state_tiv = float(rates["FloodTIV"].sum())
    statewide_rate = 0.0 if state_tiv <= 0 else min(state_paid / state_tiv, cap)

    n = rates["FloodTIV"].values  # exposure strength proxy
    r = rates["raw_rate"].values
    rates["nfip_payout_rate"] = (n / (n + shrink_tau)) * r + (shrink_tau / (n + shrink_tau)) * statewide_rate
    rates["nfip_payout_rate"] = rates["nfip_payout_rate"].clip(0.0, cap)

    return rates[["county_fips", "nfip_payout_rate"]]
