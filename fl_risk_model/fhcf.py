"""
fhcf.py - Florida Hurricane Catastrophe Fund helpers
----------------------------------------------------

This module standardizes FHCF contract terms and applies recoveries to
company wind loss tables.

Public API
----------
- attach_fhcf_terms_for_losses(loss_df, terms_df, market_share_df, ...)
- normalize_fhcf_terms(terms)
- apply_fhcf_recovery(loss_df, terms_df)

Conventions
-----------
- Monetary amounts are in USD (floats).
- CoveragePct can be expressed either as a fraction (e.g., 0.90) or a whole
  percentage (e.g., 90). We normalize to an **integer 0..100** in the
  'CoveragePct_norm' column.
- Contract fields required for recovery:
    RetentionUSD, LimitUSD, CoveragePct_norm
  These are produced by `normalize_fhcf_terms`.

Config
------
Relies on the following constants from fl_risk_model.config:
- FHCF_RET_MULTIPLES : dict[int, float]   # mapping coverage % -> retention multiple
- FHCF_PAYOUT_MULTIPLE : float            # payout multiple × FHCFPremium -> LimitUSD
- FHCF_LAE_FACTOR : float                 # applied multiplicatively to recoveries
"""

from __future__ import annotations

import re
import pandas as pd
import numpy as np
from typing import Iterable, Optional
from difflib import SequenceMatcher

from fl_risk_model.config import (
    FHCF_RET_MULTIPLES,
    FHCF_PAYOUT_MULTIPLE,
    FHCF_LAE_FACTOR,
)

from fl_risk_model import config as cfg

__all__ = ["attach_fhcf_terms_for_losses", "normalize_fhcf_terms", "apply_fhcf_recovery"]


# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #

def _require(df: pd.DataFrame, cols: Iterable[str], ctx: str) -> None:
    """Raise a descriptive error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns {missing}")


def _to_float(series: pd.Series) -> pd.Series:
    """Coerce to float, replacing non-parsable values with 0.0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _norm_pct_to_int_0_100(x) -> int:
    """
    Normalize coverage input (fraction or percent) to an INT in [0, 100].

    Examples
    --------
    0.9 -> 90
    90  -> 90
    '75' -> 75
    """
    try:
        val = float(x)
    except Exception:
        return 0
    # If value looks like a fraction, treat it as such
    if 0.0 <= val <= 1.0:
        return int(round(val * 100))
    # Otherwise assume whole-percent already
    return int(round(val))

def _norm_company_name_for_join(s: pd.Series) -> pd.Series:
    def _one(x: str) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)): return ""
        y = str(x).lower()
        y = y.replace("&", " and ")
        y = re.sub(r"\bfl\b", " florida ", y)
        y = re.sub(r"\bp\s*&\s*c\b|\bp\s*c\b", " property casualty ", y)
        y = re.sub(r"\bsvc?s\b", " services ", y)
        y = re.sub(r"\bspcl?ty\b", " specialty ", y)
        y = re.sub(r"\bassn\b", " association ", y)
        y = re.sub(r"\bnatl\b", " national ", y)
        y = re.sub(r"\bintl\b", " international ", y)
        y = re.sub(r"[\W_]+", " ", y)
        y = re.sub(r"\b(insurance|ins|insur|company|co|corp|corporation|inc|llc|mutual|exchange|group|the)\b", " ", y)
        return re.sub(r"\s+", " ", y).strip()
    return s.map(_one)

def _find_naic_col(cols):
    for c in cols:
        if "NAIC" in str(c):
            return c
    return None

def _ensure_contract_cols(terms: pd.DataFrame) -> pd.DataFrame:
    """Ensure terms has CoveragePct_norm, RetentionUSD, LimitUSD. Create them if missing."""
    t = terms.copy()

    # Normalize headers (strip spaces; collapse weird unicode)
    t.columns = [str(c).replace("\xa0", " ").strip() for c in t.columns]

    # 1) CoveragePct_norm
    if "CoveragePct_norm" not in t.columns:
        if "CoveragePct" not in t.columns:
            raise KeyError("[FHCF] Terms missing 'CoveragePct' and 'CoveragePct_norm'.")
        cov = pd.to_numeric(t["CoveragePct"], errors="coerce")
        cov = np.where(cov <= 1.0, cov * 100.0, cov)  # 0.90 -> 90
        t["CoveragePct_norm"] = pd.Series(cov).round().astype("Int64")

    # 2) FHCFPremium numeric
    prem_col = "FHCFPremium"
    if prem_col not in t.columns:
        raise KeyError("[FHCF] Terms missing 'FHCFPremium'.")
    t[prem_col] = pd.to_numeric(t[prem_col], errors="coerce").fillna(0.0).astype(float)

    # 3) RetentionUSD, LimitUSD
    if "RetentionUSD" not in t.columns:
        def _retention(pct, prem):
            pcti = int(pct) if pd.notna(pct) else 0
            mult = float(FHCF_RET_MULTIPLES.get(pcti, 0.0))
            return float(prem) * mult
        t["RetentionUSD"] = [ _retention(p, pr) for p, pr in zip(t["CoveragePct_norm"], t[prem_col]) ]

    if "LimitUSD" not in t.columns:
        t["LimitUSD"] = t[prem_col] * float(FHCF_PAYOUT_MULTIPLE)

    return t

def attach_fhcf_terms_for_losses(
    loss_df: pd.DataFrame,
    terms_df: pd.DataFrame,
    market_share_df: pd.DataFrame,
    *,
    company_crosswalk_df: pd.DataFrame | None = None,
    terms_statkey_col: str | None = None,  # ignored in crosswalk-only mode
    allow_name_fallback: bool = True,      # accepted but ignored when crosswalk provided
    qa_strict: bool = False,               # if True, fail when *participants* lack NAIC
) -> pd.DataFrame:
    """
    Crosswalk-only attachment of FHCF terms for loss companies.

    - Maps loss_df['Company'] -> StatEntityKey via market_share_df
    - Joins to company_crosswalk_df on StatEntityKey to get [NAIC, fhcf_participant]
    - Joins FHCF terms by NAIC to fetch [CoveragePct_norm, RetentionUSD, LimitUSD]
    - For non-participants or missing NAIC: assigns zero terms (no FHCF)
    - Returns a company-level table: ['Company','CoveragePct_norm','RetentionUSD','LimitUSD']

    NOTE: Name fallback/aliases/fuzzy are DISABLED when a crosswalk is provided.
    """
    if company_crosswalk_df is None or company_crosswalk_df.empty:
        raise ValueError("[FHCF] company_crosswalk_df is required for crosswalk-only mode.")

    # 1) Normalize inputs
    L = loss_df.copy()
    L.columns = [str(c).replace("\xa0"," ").strip() for c in L.columns]
    if "Company" not in L.columns:
        raise KeyError("[FHCF] loss_df must contain 'Company'")
    # ensure single row per company for terms-join
    Lc = L[["Company"]].drop_duplicates().copy()

    MS = market_share_df.copy()
    MS.columns = [str(c).replace("\xa0"," ").strip() for c in MS.columns]
    if "StatEntityKey" not in MS.columns and "Stat Entity Key" in MS.columns:
        MS = MS.rename(columns={"Stat Entity Key": "StatEntityKey"})
    if "Company" not in MS.columns or "StatEntityKey" not in MS.columns:
        raise KeyError("[FHCF] market_share_df must contain ['Company','StatEntityKey']")
    MS["StatEntityKey"] = MS["StatEntityKey"].astype(str)
    MS = MS[["Company","StatEntityKey"]].drop_duplicates()

    CW = company_crosswalk_df.copy()
    CW.columns = [str(c).replace("\xa0"," ").strip() for c in CW.columns]
    need = {"StatEntityKey","NAIC","fhcf_participant"}
    if not need.issubset(CW.columns):
        raise KeyError(f"[FHCF] company_crosswalk_df missing {need - set(CW.columns)}")
    CW["StatEntityKey"] = CW["StatEntityKey"].astype(str)
    CW["NAIC"] = CW["NAIC"].astype(str)
    CW.loc[CW["NAIC"].isin(["", "nan", "None", "NaN"]), "NAIC"] = np.nan
    CW["fhcf_participant"] = CW["fhcf_participant"].astype(bool)

    T = terms_df.copy()
    T.columns = [str(c).replace("\xa0"," ").strip() for c in T.columns]
    naic_col = _find_naic_col(T.columns)
    if naic_col is None:
        raise KeyError("[FHCF] terms_df must include an NAIC column")
    if naic_col != "NAIC":
        T = T.rename(columns={naic_col: "NAIC"})
    T["NAIC"] = T["NAIC"].astype(str)
    # Required contract fields
    req_contract = {"CoveragePct_norm","RetentionUSD","LimitUSD"}
    missing = req_contract - set(T.columns)
    if missing:
        raise KeyError(f"[FHCF] terms_df missing {missing} (did you run normalize_fhcf_terms?)")

    # 2) Company -> StatEntityKey
    Lc = Lc.merge(MS, on="Company", how="left")

    # 3) StatEntityKey -> [NAIC, participant]
    Lc = Lc.merge(CW[["StatEntityKey","NAIC","fhcf_participant"]], on="StatEntityKey", how="left")

    # 4) Strict QA (participants must have NAIC)
    if qa_strict:
        # Loss-weighted QA only across participants
        loss_by_co = (L.groupby("Company", as_index=False)["GrossWindLossUSD"]
                        .sum().rename(columns={"GrossWindLossUSD":"_loss"}))
        QA = Lc.merge(loss_by_co, on="Company", how="left").fillna({"_loss": 0.0})
        part_mask = QA["fhcf_participant"] == True
        loss_part = float(QA.loc[part_mask, "_loss"].sum())
        loss_part_resolved = float(QA.loc[part_mask & QA["NAIC"].notna(), "_loss"].sum())
        if loss_part > 0 and (loss_part_resolved / loss_part) < 0.98:
            offenders = (QA.loc[part_mask & QA["NAIC"].isna(), ["Company","StatEntityKey","_loss"]]
                           .sort_values("_loss", ascending=False).head(12))
            raise ValueError("[FHCF] Crosswalk missing NAIC for participants; "
                             f"resolved {loss_part_resolved/loss_part:.1%}. "
                             f"Top offenders:\n{offenders.to_string(index=False)}")

    # 5) Join FHCF contract terms by NAIC; zero-terms for non-participants / missing NAIC
    terms = Lc.merge(T[["NAIC","CoveragePct_norm","RetentionUSD","LimitUSD"]],
                     on="NAIC", how="left")

    # zero-out non-participants or missing NAIC
    zero_mask = (terms["fhcf_participant"] == False) | (terms["NAIC"].isna())
    for c in ["CoveragePct_norm","RetentionUSD","LimitUSD"]:
        terms.loc[zero_mask, c] = 0.0

    # 6) Output one row per Company with contract terms
    out = (terms[["Company","CoveragePct_norm","RetentionUSD","LimitUSD"]]
                 .drop_duplicates()
                 .copy())
    # sanitize numeric
    out["CoveragePct_norm"] = pd.to_numeric(out["CoveragePct_norm"], errors="coerce").fillna(0.0)
    out["RetentionUSD"]     = pd.to_numeric(out["RetentionUSD"], errors="coerce").fillna(0.0)
    out["LimitUSD"]         = pd.to_numeric(out["LimitUSD"], errors="coerce").fillna(0.0)

    return out


def normalize_fhcf_terms(terms: pd.DataFrame) -> pd.DataFrame:
    t = terms.copy()

    # --- Harmonize columns (keeps extras intact) ---
    rename_map = {
        "FHCF_PremiumUSD": "FHCFPremium",
        "ReimbPremiumUSD": "FHCFPremium",
        "FHCF Premium":    "FHCFPremium",
        "Coverage_%":      "CoveragePct",
        "Coverage":        "CoveragePct",
        "Company_FHCF":    "Company",
        "Company_Name":    "Company",
    }
    for k, v in rename_map.items():
        if k in t.columns and v not in t.columns:
            t = t.rename(columns={k: v})

    req = ["Company", "FHCFPremium", "CoveragePct"]
    missing = [c for c in req if c not in t.columns]
    if missing:
        raise KeyError(f"[normalize_fhcf_terms] Missing required columns: {missing}")

    prem = pd.to_numeric(t["FHCFPremium"], errors="coerce").fillna(0.0)

    # Normalize coverage: fractions -> percent; then snap to {45,75,90}
    cov_raw = pd.to_numeric(t["CoveragePct"], errors="coerce")
    cov_pct = np.where(cov_raw.notna() & (cov_raw <= 1.0), cov_raw * 100.0, cov_raw)
    cov_pct = pd.Series(cov_pct, index=t.index).fillna(0.0).clip(0, 100)

    def _snap(x):
        for target in (45, 75, 90):
            if abs(x - target) <= 1.0:
                return target
        return int(round(x))

    t["CoveragePct_norm"] = cov_pct.apply(_snap).clip(0, 100)

    # Retention: premium × coverage-specific multiple (fallback to 90% multiple if missing)
    mults = getattr(cfg, "FHCF_RET_MULTIPLES", {})
    default_mult = float(mults.get(90, 6.0732))
    t["RetentionUSD"] = prem * t["CoveragePct_norm"].map(mults).fillna(default_mult)

    # Limit: premium × payout multiple (no extra coverage factor)
    t["LimitUSD"] = prem * float(getattr(cfg, "FHCF_PAYOUT_MULTIPLE", 11.2368))

    # Hygiene
    t["RetentionUSD"] = pd.to_numeric(t["RetentionUSD"], errors="coerce").fillna(0.0).clip(lower=0.0)
    t["LimitUSD"]     = pd.to_numeric(t["LimitUSD"],     errors="coerce").fillna(0.0).clip(lower=0.0)

    return t

# ----------------------------------------------------------------------------- #
# Recovery application
# ----------------------------------------------------------------------------- #

def apply_fhcf_recovery(loss_df: pd.DataFrame, terms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FHCF recoveries to gross wind losses.

    Parameters
    ----------
    loss_df : pd.DataFrame
        Required columns:
          - 'Company' : str
          - 'GrossWindLossUSD' : float
        Optional:
          - 'County' : str (carried through unmodified)

        Losses may be at Company×County granularity or company-aggregated.

    terms_df : pd.DataFrame
        Terms **already normalized** by `normalize_fhcf_terms`, must contain:
          - 'Company', 'CoveragePct_norm', 'RetentionUSD', 'LimitUSD'

    Returns
    -------
    pd.DataFrame
        `loss_df` with added columns:
          - 'ExcessUSD'       : max(Gross - Retention, 0)
          - 'RecoverableUSD'  : min(ExcessUSD, LimitUSD)
          - 'RecoveryUSD'     : RecoverableUSD × (CoveragePct_norm/100) × FHCF_LAE_FACTOR
          - 'NetWindUSD'      : GrossWindLossUSD - RecoveryUSD

    Notes
    -----
    - If a company is missing from terms, it receives zero recovery by design.
    - All computations are non-negative; NaNs are treated as zero.
    """
    ctx_terms = "apply_fhcf_recovery: terms_df (normalized)"
    _require(terms_df, ["Company", "CoveragePct_norm", "RetentionUSD", "LimitUSD"], ctx_terms)

    ctx_loss = "apply_fhcf_recovery: loss_df"
    _require(loss_df, ["Company", "GrossWindLossUSD"], ctx_loss)

    # Work on copies; never modify inputs in place.
    df = loss_df.copy()
    df["GrossWindLossUSD"] = _to_float(df["GrossWindLossUSD"])

    # Keep only the contract columns we need from normalized terms
    t = terms_df[["Company", "CoveragePct_norm", "RetentionUSD", "LimitUSD"]].copy()
    t["CoveragePct_norm"] = _to_float(t["CoveragePct_norm"])
    t["RetentionUSD"] = _to_float(t["RetentionUSD"])
    t["LimitUSD"] = _to_float(t["LimitUSD"])

    # Left-join: companies without terms -> zero recovery
    df = df.merge(t, on="Company", how="left")

    # Excess over retention (floor at zero)
    df["ExcessUSD"] = (df["GrossWindLossUSD"] - df["RetentionUSD"].fillna(0.0)).clip(lower=0.0)

    # Cap by company limit
    df["RecoverableUSD"] = df[["ExcessUSD", "LimitUSD"]].min(axis=1).fillna(0.0)

    # Apply coverage percent and LAE
    coverage_frac = (df["CoveragePct_norm"].fillna(0.0) / 100.0).clip(lower=0.0, upper=1.0)
    df["RecoveryUSD"] = (df["RecoverableUSD"] * coverage_frac * float(FHCF_LAE_FACTOR)).astype(float)

    # Net
    df["NetWindUSD"] = df["GrossWindLossUSD"] - df["RecoveryUSD"]

    return df

def build_company_crosswalk(
    terms_df: pd.DataFrame,
    market_share_df: pd.DataFrame,
    alias_df: pd.DataFrame | None = None,
    min_similarity: float = 0.88,
) -> pd.DataFrame:
    """
    Returns a curated crosswalk DataFrame:
      [StatEntityKey, Company_MS, NAIC, Company_FHCF, name_norm_ms, name_norm_fhcf, match_method, similarity, fhcf_participant]
    """
    ms = market_share_df.copy()
    ms.columns = [str(c).replace("\xa0"," ").strip() for c in ms.columns]
    if "StatEntityKey" not in ms.columns and "Stat Entity Key" in ms.columns:
        ms = ms.rename(columns={"Stat Entity Key": "StatEntityKey"})
    ms = ms[["Company","StatEntityKey"]].dropna().drop_duplicates()
    ms["StatEntityKey"] = ms["StatEntityKey"].astype(str)
    ms = ms.rename(columns={"Company":"Company_MS"})
    ms["name_norm_ms"] = _norm_company_name_for_join(ms["Company_MS"])

    T = terms_df.copy()
    T.columns = [str(c).replace("\xa0"," ").strip() for c in T.columns]
    naic_col = _find_naic_col(T.columns)
    if naic_col is None:
        raise KeyError("FHCF terms missing NAIC column")
    T = T.rename(columns={naic_col: "NAIC", "Company": "Company_FHCF"})
    T["name_norm_fhcf"] = _norm_company_name_for_join(T["Company_FHCF"])
    T["fhcf_participant"] = True
    T["NAIC"] = T["NAIC"].astype(str)

    # Optional alias rewrite on normalized names
    if alias_df is not None and not alias_df.empty:
        alias = alias_df.copy()
        alias["from_norm"] = _norm_company_name_for_join(alias["from"])
        alias["to_norm"]   = _norm_company_name_for_join(alias["to"])
        ms = ms.merge(alias[["from_norm","to_norm"]].rename(columns={"from_norm":"name_norm_ms"}), on="name_norm_ms", how="left")
        ms["name_norm_ms"] = ms["to_norm"].fillna(ms["name_norm_ms"])
        ms.drop(columns=["to_norm"], inplace=True)

        T = T.merge(alias[["from_norm","to_norm"]].rename(columns={"from_norm":"name_norm_fhcf"}), on="name_norm_fhcf", how="left")
        T["name_norm_fhcf"] = T["to_norm"].fillna(T["name_norm_fhcf"])
        T.drop(columns=["to_norm"], inplace=True)

    # 1) Exact normalized name matches
    exact = ms.merge(T[["name_norm_fhcf","NAIC","Company_FHCF"]].rename(columns={"name_norm_fhcf":"name_norm_ms"}),
                     on="name_norm_ms", how="left")
    exact["match_method"] = np.where(exact["NAIC"].notna(), "name_exact", None)
    exact["similarity"] = np.where(exact["NAIC"].notna(), 1.0, np.nan)

    # 2) Fuzzy fallback for rows still without NAIC
    need = exact["NAIC"].isna()
    if need.any():
        # Build a small lookup for fuzzy
        terms_names = T[["name_norm_fhcf","Company_FHCF"]].drop_duplicates().values.tolist()
        def _best(cname: str):
            best_to, best_norm, best_s = "", "", 0.0
            for norm, raw in terms_names:
                s = SequenceMatcher(None, cname, norm).ratio()
                if s > best_s:
                    best_to, best_norm, best_s = raw, norm, s
            if best_s >= min_similarity:
                naic = T.loc[T["name_norm_fhcf"] == best_norm, "NAIC"].iloc[0]
                return pd.Series([naic, best_to, "fuzzy", best_s])
            return pd.Series([np.nan, np.nan, None, np.nan])

        fuzzy = exact.loc[need, ["name_norm_ms"]].drop_duplicates().assign(tmp=1)
        fuzzy[["NAIC","Company_FHCF","match_method","similarity"]] = (
            fuzzy["name_norm_ms"].apply(_best)
        )
        exact = exact.merge(fuzzy.drop(columns=["tmp"]), on="name_norm_ms", how="left", suffixes=("","_f"))
        # prefer exact if present; else use fuzzy
        for c in ["NAIC","Company_FHCF","match_method","similarity"]:
            exact[c] = exact[c].where(exact[c].notna(), exact[f"{c}_f"])
        exact = exact.drop(columns=[c for c in exact.columns if c.endswith("_f")])

    # 3) Join fhcf_participant flag and emit
    out = exact.merge(T[["NAIC","fhcf_participant","name_norm_fhcf"]].drop_duplicates(), on="NAIC", how="left")
    out["fhcf_participant"] = out["fhcf_participant"].fillna(False)

    cols = ["StatEntityKey","Company_MS","NAIC","Company_FHCF","name_norm_ms","name_norm_fhcf","match_method","similarity","fhcf_participant"]
    return out[cols].drop_duplicates()


def save_company_crosswalk(df: pd.DataFrame, path) -> str:
    df.to_csv(path, index=False)
    return str(path)

def load_company_crosswalk(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"StatEntityKey": str, "NAIC": str})
    return df