"""
capital.py - Capital depletion, group support, and Citizens capital parsing
----------------------------------------------------------------------------

Loads statutory surplus for private insurers (with optional group metadata),
applies scenario losses to deplete surplus, optionally samples surplus for MC,
computes intragroup capital contributions, and parses Citizens' surplus from
a simple CSV.

Mirrors the Methods specification:
- Private insurer capital from NAIC/S&P filings (Surplus as Regards Policyholders)
- Group support: pooled group surplus - sum(entity) with a global contribution rate
- Ruin/default: negative post-loss surplus (and optionally RBC threshold)
- Citizens capital: latest-year surplus from annual statements

Public API
----------
- load_surplus_data
- load_surplus_data_with_groups
- apply_losses_to_surplus
- apply_group_capital_contributions
- load_citizens_capital_row_from_csv
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from fl_risk_model import config as cfg

__all__ = [
    "load_surplus_data",
    "load_surplus_data_with_groups",
    "apply_losses_to_surplus",
    "apply_group_capital_contributions",
    "load_citizens_capital_row_from_csv",
]

# =============================================================================
# Helpers (header parsing & normalization)
# =============================================================================

def _norm(x) -> str:
    """Normalize a header cell (lowercase, strip punctuation/whitespace)."""
    s = "" if pd.isna(x) else str(x)
    s = s.replace("\n", " ")
    s = re.sub(r"[^\w\s$()/%-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _year_in(s, year: int) -> bool:
    """Return True if a year string like '2024' appears in s (robust to spacing)."""
    s = _norm(s)
    return str(year) in s

def _ffill_header(cols: pd.MultiIndex) -> list[tuple]:
    """
    Forward-fill each header level across columns to resolve Excel-merged NaNs.

    Parameters
    ----------
    cols : pd.MultiIndex
        Multi-level header.

    Returns
    -------
    list of tuple
        Forward-filled header tuples.
    """
    arr = pd.DataFrame(list(cols)).T  # shape: (n_levels, n_cols)
    arr = arr.ffill(axis=1)
    return [tuple(row) for row in arr.T.values.tolist()]

def _match_col(
    cols: list[tuple],
    want0=None,
    want1=None,
    want2=None,
    year: Optional[int] = None
):
    """
    Find a 3-level header column that matches desired predicates/tokens.

    Returns
    -------
    tuple | None
        Matching header tuple or None.
    """
    def _ok(level, want):
        if want is None:
            return True
        val = _norm(level)
        if callable(want):
            return want(val)
        return want in val

    for c in cols:
        l0, l1, l2 = tuple(list(c) + [None, None, None])[:3]
        if not _ok(l0, want0):
            continue
        if not _ok(l1, want1):
            continue
        if not _ok(l2, want2):
            continue
        if year is not None and not (_year_in(l2, year) or _year_in(l1, year) or _year_in(l0, year)):
            continue
        return c
    return None

def _match_anylevel(cols: list[tuple], token: str):
    """Find first header with token present in any level."""
    tok = _norm(token)
    for c in cols:
        l0, l1, l2 = tuple(list(c) + [None, None, None])[:3]
        if any(tok in _norm(x) for x in (l0, l1, l2)):
            return c
    return None

def _ewa_weights(end_year: int, lookback_years: int, half_life: float) -> pd.DataFrame:
    years = list(range(end_year - lookback_years + 1, end_year + 1))
    w = [0.5 ** ((end_year - y) / max(half_life, 1e-6)) for y in years]
    s = sum(w) or 1.0
    return pd.DataFrame({"year": years, "weight": [wi / s for wi in w]})

# =============================================================================
# Surplus loaders
# =============================================================================

def load_surplus_data(
    path: str,
    sheet_name: int | str = 0,
    header_rows: Tuple[int, int] = (2, 4),
    company_col_fragment: str = "Entity Name",
    surplus_col_fragment: str = "Surplus as Regards Policyholders",
    statkey_col_fragment: str = "S&P Statutory Entity Key",
    year: int = 2024,
) -> pd.DataFrame:
    """
    Load entity-level surplus from a 2-row header Excel sheet.

    Parameters
    ----------
    path : str
        Excel file path.
    sheet_name : int or str, default 0
        Worksheet name or index.
    header_rows : tuple[int, int], default (2, 4)
        Header rows (0-based).
    company_col_fragment : str, default "Entity Name"
        Token to detect the company column.
    surplus_col_fragment : str, default "Surplus as Regards Policyholders"
        Token to detect the surplus column (year-matched if present).
    statkey_col_fragment : str, default "S&P Statutory Entity Key"
        Token for S&P statutory key (optional).
    year : int, default 2024
        Preferred year to select in the header if multiple exist.

    Returns
    -------
    pd.DataFrame
        ['Company','StatEntityKey','SurplusUSD'] in dollars.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_rows)

    flat_cols = []
    for lvl0, lvl1 in df.columns:
        parts = []
        if pd.notna(lvl0):
            parts.append(str(lvl0).strip())
        if pd.notna(lvl1):
            parts.append(str(lvl1).strip())
        flat_cols.append(" ".join(parts).replace("\n", " "))
    df.columns = flat_cols

    comp_candidates = [c for c in flat_cols if company_col_fragment in c]
    if not comp_candidates:
        raise KeyError(f"No column containing '{company_col_fragment}'")
    company_col = comp_candidates[0]

    surplus_candidates = [c for c in flat_cols if surplus_col_fragment in c and str(year) in c]
    if not surplus_candidates:
        surplus_candidates = [c for c in flat_cols if surplus_col_fragment in c]
    if not surplus_candidates:
        raise KeyError(f"No column matching '{surplus_col_fragment}' (year {year})")
    surplus_col = surplus_candidates[0]

    sk_candidates = [c for c in flat_cols if statkey_col_fragment.lower() in c.lower()]
    sk_col = sk_candidates[0] if sk_candidates else None

    keep = [company_col, surplus_col] + ([sk_col] if sk_col else [])
    out = df[keep].dropna(subset=[company_col]).copy()
    out = out.rename(
        columns={
            company_col: "Company",
            surplus_col: "SurplusUSD_thousands",
            **({sk_col: "StatEntityKey"} if sk_col else {}),
        }
    )
    out["SurplusUSD"] = (
        pd.to_numeric(out["SurplusUSD_thousands"], errors="coerce").fillna(0.0).astype(float) * 1_000.0
    )
    out = out.drop(columns=["SurplusUSD_thousands"])

    if "StatEntityKey" not in out.columns:
        out["StatEntityKey"] = ""  # fallback if missing
    else:
        out["StatEntityKey"] = out["StatEntityKey"].astype(str).str.strip()

    return out[["Company", "StatEntityKey", "SurplusUSD"]]


def load_surplus_data_with_groups(
    path: Path,
    sheet_name: int | str = 0,
    year: int = 2024,
    header_rows: Tuple[int, ...] = (0, 2, 4),
) -> pd.DataFrame:
    """
    Load entity and group surplus from a 3-level header Excel sheet.

    Notes
    -----
    We forward-fill merged header cells to ensure band labels are present
    on every column and locate:
      - 'Entity Name'
      - 'Surplus as Regards Policyholders ($000)' (entity & group)
      - 'Group surplus / entity surplus' (ratio, optional)
      - 'NAIC group name' / 'NAIC group number' (optional)

    Parameters
    ----------
    path : Path
        Excel path.
    sheet_name : int or str, default 0
        Worksheet name or index.
    year : int, default 2024
        Year to match in the header.
    header_rows : tuple[int, ...], default (0, 2, 4)
        Header rows (0-based) for a 3-level header.

    Returns
    -------
    pd.DataFrame
        Columns: ['Company','StatEntityKey','SurplusUSD',
                  'GroupSurplusUSD','GroupToEntityRatio',
                  ('NAICGroupName','NAICGroupNumber' if present)]
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=list(header_rows))
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            f"Expected a MultiIndex header from rows {header_rows}, got {type(df.columns)}"
        )

    cols_ff = _ffill_header(df.columns)

    company_col = _match_col(cols_ff, want1="entity name") or _match_anylevel(cols_ff, "entity name")
    if company_col is None:
        raise KeyError("Could not find 'Entity Name' column in any header level.")

    df = df[~pd.isna(df[company_col])].copy()

    statkey_col = (
        _match_col(cols_ff, want1=lambda s: "s&p statutory entity key" in s or "s p statutory entity key" in s)
        or _match_anylevel(cols_ff, "s&p statutory entity key")
    )

    def _is_surplus_000(s: str) -> bool:
        s = _norm(s)
        return (
            ("surplus" in s)
            and (("policyholders" in s) or ("as regards policyholders" in s))
            and ("$000" in s)
        )

    entity_surplus_col = (
        _match_col(cols_ff, want0=lambda s: "entity" in s and "surplus" in s, want1=_is_surplus_000, year=year)
        or _match_col(
            cols_ff, want0=lambda s: "entity" in s and "surplus" in s, want1=lambda s: ("surplus" in s and "$000" in s), year=year
        )
    )
    if entity_surplus_col is None:
        raise KeyError(
            f"Could not find ENTITY SURPLUS -> 'Surplus as Regards Policyholders ($000)' for year {year}."
        )

    group_surplus_col = (
        _match_col(cols_ff, want0=lambda s: "group" in s and "surplus" in s, want1=_is_surplus_000, year=year)
        or _match_col(
            cols_ff, want0=lambda s: "group" in s and "surplus" in s, want1=lambda s: ("surplus" in s and "$000" in s), year=year
        )
    )

    ratio_col = _match_col(cols_ff, want1="group surplus / entity surplus") or _match_anylevel(
        cols_ff, "group surplus / entity surplus"
    )

    out = pd.DataFrame(
        {
            "Company": df[company_col],
            "StatEntityKey": (df[statkey_col].astype(str).str.strip() if statkey_col is not None else ""),
        }
    )

    ent = pd.to_numeric(df[entity_surplus_col], errors="coerce").fillna(0.0).astype(float) * 1_000.0
    out["SurplusUSD"] = ent

    if group_surplus_col is not None:
        grp = pd.to_numeric(df[group_surplus_col], errors="coerce").fillna(0.0).astype(float) * 1_000.0
    else:
        grp = pd.Series(pd.NA, index=df.index, dtype="float")
    out["GroupSurplusUSD"] = grp

    if ratio_col is not None:
        ratio = pd.to_numeric(df[ratio_col], errors="coerce").astype(float)
    else:
        with pd.option_context("mode.use_inf_as_na", True):
            ratio = (grp / ent).astype(float)
    out["GroupToEntityRatio"] = ratio

    naic_group_name_col = _match_anylevel(cols_ff, "naic group name")
    naic_group_num_col = _match_anylevel(cols_ff, "naic group number")
    if naic_group_name_col is not None:
        out["NAICGroupName"] = df[naic_group_name_col].astype(str).str.strip()
    if naic_group_num_col is not None:
        out["NAICGroupNumber"] = df[naic_group_num_col]

    cols_order = ["Company", "StatEntityKey", "SurplusUSD", "GroupSurplusUSD", "GroupToEntityRatio"]
    for opt in ["NAICGroupName", "NAICGroupNumber"]:
        if opt in out.columns:
            cols_order.append(opt)
    return out[cols_order]

# =============================================================================
# Core operations
# =============================================================================

def apply_losses_to_surplus(
    surplus_df: pd.DataFrame,
    losses_df: pd.DataFrame,
    rbc_df: pd.DataFrame | None = None,
    *,
    rbc_affects_ruinflag: bool = True,
    sample: bool | None = None,
    cov: float | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Merge surplus and losses, optionally sample surplus, deplete, and flag ruin.

    Parameters
    ----------
    surplus_df : pd.DataFrame
        Expected columns include ['Company','SurplusUSD'] and (optionally) ['StatEntityKey'].
    losses_df : pd.DataFrame
        Columns ['Company','TotalLossUSD'] - scenario total loss (USD), net of prior recoveries.
    rbc_df : pd.DataFrame, optional
        Columns should include ['StatEntityKey','RBCReq'] or ['Company','RBCReq'].
    rbc_affects_ruinflag : bool, default True
        If True, RuinFlag is set when EndingSurplusUSD < 0 **or** < RBCReq (when provided).
        If False, RuinFlag only checks EndingSurplusUSD < 0.
    sample : bool or None
        If True, sample SurplusUSD ~ Normal(SurplusUSD, SurplusUSD*cov), floored at 0.
        If None, uses cfg.SAMPLE_SURPLUS (default False).
    cov : float or None
        Coefficient of variation when sampling. If None, uses cfg.SURPLUS_COV.
    rng : np.random.Generator, optional
        RNG instance; defaults to np.random.default_rng().

    Returns
    -------
    pd.DataFrame
        ['Company','StatEntityKey?' ,'SurplusUSD','SurplusSampledUSD','TotalLossUSD',
         'EndingSurplusUSD','RuinFlag', 'RBCReq?']
    """
    # Keep StatEntityKey if present in surplus_df
    merge_cols = ["Company", "SurplusUSD"] + (["StatEntityKey"] if "StatEntityKey" in surplus_df.columns else [])
    df = surplus_df[merge_cols].merge(
        losses_df[["Company", "TotalLossUSD"]], on="Company", how="left"
    )

    # Coerce numerics, fill NaNs
    df["SurplusUSD"]    = pd.to_numeric(df.get("SurplusUSD", 0.0), errors="coerce").fillna(0.0)
    df["TotalLossUSD"]  = pd.to_numeric(df.get("TotalLossUSD", 0.0), errors="coerce").fillna(0.0)

    # Sampling
    if sample is None:
        sample = bool(getattr(cfg, "SAMPLE_SURPLUS", False))
    if cov is None:
        cov = float(getattr(cfg, "SURPLUS_COV", 0.0))
    rng = rng or np.random.default_rng()

    if sample and cov > 0:
        sd = df["SurplusUSD"] * cov
        draw = rng.normal(loc=df["SurplusUSD"].to_numpy(float), scale=sd.to_numpy(float))
        df["SurplusSampledUSD"] = np.maximum(draw, 0.0)
    else:
        df["SurplusSampledUSD"] = df["SurplusUSD"]

    # Deplete and base ruin flag
    df["EndingSurplusUSD"] = df["SurplusSampledUSD"] - df["TotalLossUSD"]
    df["EndingSurplus"]    = df["EndingSurplusUSD"]  # alias if other code expects it
    df["RuinFlag"]         = (df["EndingSurplusUSD"] < 0.0).astype(bool)

    # RBC gate (prefer StatEntityKey, fallback to Company)
    if rbc_df is not None:
        if ("StatEntityKey" in rbc_df.columns) and ("StatEntityKey" in df.columns):
            df = df.merge(rbc_df[["StatEntityKey", "RBCReq"]], on="StatEntityKey", how="left")
        else:
            df = df.merge(rbc_df[["Company", "RBCReq"]], on="Company", how="left")

        df["RBCReq"] = pd.to_numeric(df["RBCReq"], errors="coerce")  # keep NaN if missing
        df["EndingSurplusUSD"] = pd.to_numeric(df["EndingSurplusUSD"], errors="coerce")

        if rbc_affects_ruinflag:
            rbc_breach = df["RBCReq"].notna() & (df["EndingSurplusUSD"] < df["RBCReq"])
            df["RuinFlag"] = ( (df["EndingSurplusUSD"] < 0.0) | rbc_breach ).astype(bool)

    # Order columns
    base_cols = ["Company"] + (["StatEntityKey"] if "StatEntityKey" in df.columns else [])
    base_cols += ["SurplusUSD", "SurplusSampledUSD", "TotalLossUSD", "EndingSurplusUSD", "RuinFlag"]
    if "RBCReq" in df.columns:
        base_cols.append("RBCReq")

    return df[base_cols]

def apply_group_capital_contributions(
    capital_post: pd.DataFrame,
    surplus_df: pd.DataFrame,
    contribution_rate_range: tuple[float, float] = (0.0, 0.20),
    rng: np.random.Generator | None = None,
    eligibility_threshold: float = 10.0,  # NEW: fixed gate (>10)
) -> pd.DataFrame:
    """
    Intragroup capital support with conservation of group capital.

    - Draw ONE scenario-wide rate r ~ Uniform(low, high).
    - Eligibility: EndingSurplusUSD < 0 and GroupToEntityRatio > 10.0 (configurable via eligibility_threshold).
    - For each group g:
        need_g = sum(deficits of eligible members)
        extra_parent_pool_g = max(GroupSurplusUSD_g - sum(entity SurplusUSD), 0)
        donor_capacity_g = sum(EndingSurplusUSD_i for i in g if > 0)
        available_pool_g = extra_parent_pool_g + donor_capacity_g
        budget_g = need_g  # always rescue eligible members up to zero, funded parent-first then donors
        Allocate budget_g to eligible members by TRIAGE (smallest deficit first)
        so some firms cross zero. Fund remainder beyond extra_parent_pool_g from donors
        proportionally (donors never go below zero).
    """
    rng = rng or np.random.default_rng()
    res = capital_post.copy()

    # Ensure EndingSurplusUSD present
    if "EndingSurplusUSD" not in res.columns:
        if "EndingSurplus" in res.columns:
            res["EndingSurplusUSD"] = pd.to_numeric(res["EndingSurplus"], errors="coerce")
        else:
            raise KeyError("apply_group_capital_contributions expects 'EndingSurplusUSD'.")

    # Group metadata
    gcols = ["Company", "SurplusUSD", "GroupSurplusUSD", "NAICGroupNumber", "NAICGroupName", "GroupToEntityRatio"]
    gmeta = surplus_df[[c for c in gcols if c in surplus_df.columns]].copy()

    # GroupID preference
    if "NAICGroupNumber" in gmeta.columns and gmeta["NAICGroupNumber"].notna().any():
        gmeta["GroupID"] = gmeta["NAICGroupNumber"].astype(str)
    elif "NAICGroupName" in gmeta.columns:
        gmeta["GroupID"] = gmeta["NAICGroupName"].astype(str)
    else:
        gmeta["GroupID"] = gmeta["Company"]

    # Extra parent pool
    if "GroupSurplusUSD" in gmeta.columns:
        grp_surplus = gmeta.groupby("GroupID")["GroupSurplusUSD"].max().fillna(0.0)
    else:
        grp_surplus = gmeta.groupby("GroupID")["Company"].size().mul(0.0)
    if "SurplusUSD" in gmeta.columns:
        sum_entity_surplus = gmeta.groupby("GroupID")["SurplusUSD"].sum().fillna(0.0)
    else:
        sum_entity_surplus = gmeta.groupby("GroupID")["Company"].size().mul(0.0)
    extra_parent_pool = (grp_surplus - sum_entity_surplus).clip(lower=0.0)

    # Join group info
    res = res.merge(
        gmeta[["Company", "GroupID", "GroupSurplusUSD", "NAICGroupNumber", "NAICGroupName"]].drop_duplicates("Company"),
        on="Company", how="left"
    )
    if "GroupToEntityRatio" in gmeta.columns:
        res = res.merge(gmeta[["Company", "GroupToEntityRatio"]].drop_duplicates("Company"), on="Company", how="left")
    res["GroupToEntityRatio"] = pd.to_numeric(res.get("GroupToEntityRatio"), errors="coerce").fillna(1.0)

    # Deficits & eligibility
    res["EndingSurplusUSD"] = pd.to_numeric(res["EndingSurplusUSD"], errors="coerce").fillna(0.0)
    res["Deficit"] = (-res["EndingSurplusUSD"]).clip(lower=0.0)
    res["EligibleFlag"] = (res["EndingSurplusUSD"] < 0) & (res["GroupToEntityRatio"] > float(eligibility_threshold))
    res["EligibilityThresholdUsed"] = float(eligibility_threshold)

    # Scenario rate (kept for diagnostics/back-compat; not used to cap)
    lo, hi = contribution_rate_range
    scenario_rate = float(rng.uniform(lo, hi))
    res["ScenarioContributionRate"] = scenario_rate
    res["GroupRateApplied"] = 0.0
    res["GroupContributionUSD"] = 0.0
    res["AvailableGroupPoolUSD"] = 0.0

    # Optional diagnostics for "save all eligibles" feasibility
    for col in [
        "DiagRequiredTopUpUSD", "DiagExtraParentPoolUSD", "DiagDonorCapacityUSD",
        "DiagAvailablePoolUSD", "DiagParentDrawIfFullRescueUSD",
        "DiagDonorDrawIfFullRescueUSD", "DiagGroupShortfallUSD", "DiagGroupFullyFundedFlag"
    ]:
        res[col] = np.nan

    # ---------- ALLOCATION PER GROUP (always rescue if eligible) ----------
    for gid, idx in res.groupby("GroupID").groups.items():
        idx = list(idx)

        # Eligible deficits and need
        elig_mask = res.loc[idx, "EligibleFlag"].to_numpy(bool)
        deficits = res.loc[idx, "Deficit"].to_numpy(float)
        need = float(deficits[elig_mask].sum())
        if need <= 0:
            continue

        # Donor capacity = sum of positive EndingSurplusUSD within the group
        end_vals = res.loc[idx, "EndingSurplusUSD"].to_numpy(float)
        donors_mask = end_vals > 0
        donor_capacity = float(end_vals[donors_mask].sum())

        extra_pool = float(extra_parent_pool.get(gid, 0.0))
        available_pool = extra_pool + donor_capacity
        res.loc[idx, "AvailableGroupPoolUSD"] = available_pool

        # Diagnostics: feasibility of rescuing all eligibles
        required_topup = need
        group_shortfall = max(required_topup - available_pool, 0.0)
        parent_draw = min(required_topup, extra_pool)
        donor_draw_needed = max(required_topup - parent_draw, 0.0)
        donor_draw = min(donor_draw_needed, donor_capacity)

        res.loc[idx, "DiagRequiredTopUpUSD"] = required_topup
        res.loc[idx, "DiagExtraParentPoolUSD"] = extra_pool
        res.loc[idx, "DiagDonorCapacityUSD"] = donor_capacity
        res.loc[idx, "DiagAvailablePoolUSD"] = available_pool
        res.loc[idx, "DiagParentDrawIfFullRescueUSD"] = parent_draw
        res.loc[idx, "DiagDonorDrawIfFullRescueUSD"] = donor_draw
        res.loc[idx, "DiagGroupShortfallUSD"] = group_shortfall
        res.loc[idx, "DiagGroupFullyFundedFlag"] = (group_shortfall == 0.0)

        if available_pool <= 0:
            continue

        # Budget = need (aim to fully cure deficits of all eligible members)
        budget = need
        if budget <= 0:
            continue

        # TRIAGE: allocate budget to smallest eligible deficits first
        contrib = np.zeros_like(deficits)
        elig_indices = np.where(elig_mask)[0]
        order = elig_indices[np.argsort(deficits[elig_indices])]  # ascending deficits
        remaining = min(budget, available_pool)  # can't allocate beyond available pool
        for j in order:
            if remaining <= 0:
                break
            d = deficits[j]
            take = min(d, remaining)
            contrib[j] = take
            remaining -= take

        total_contrib = float(contrib.sum())
        if total_contrib <= 0:
            continue

        # Record inflows to recipients
        res.loc[idx, "GroupContributionUSD"] = (
            pd.to_numeric(res.loc[idx, "GroupContributionUSD"], errors="coerce").fillna(0.0).to_numpy(float) + contrib
        )
        # Rate actually applied relative to total need (1.0 means fully rescued)
        res.loc[idx, "GroupRateApplied"] = float(min(1.0, total_contrib / need))

        # Fund contributions: first from extra pool, then donors proportionally
        need_from_donors = max(total_contrib - extra_pool, 0.0)
        if need_from_donors > 0 and donor_capacity > 0:
            donor_idx = np.where(donors_mask)[0]
            donor_vals = end_vals[donor_idx]
            w = donor_vals / donor_vals.sum()
            donor_outflows = need_from_donors * w
            # debit donors (negative contribution)
            curr = pd.to_numeric(
                res.loc[[idx[k] for k in donor_idx], "GroupContributionUSD"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            res.loc[[idx[k] for k in donor_idx], "GroupContributionUSD"] = curr - donor_outflows

    # Final surplus & default flag (no RBC gate)
    res["AdjustedSurplusUSD"] = res["EndingSurplusUSD"] + pd.to_numeric(
        res["GroupContributionUSD"], errors="coerce"
    ).fillna(0.0)
    res["DefaultFlag"] = res["AdjustedSurplusUSD"].lt(0).astype(bool)
    return res


def _parse_money_to_float(x: object) -> float:
    s = str(x).strip()
    if not s:
        return 0.0
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "").replace(",", "")
    val = pd.to_numeric(s, errors="coerce")
    if pd.isna(val):
        return 0.0
    v = float(val)
    return -v if neg else v


def load_citizens_capital_row_from_csv(
    path: str | None = None,
    year: int | None = None,
    *,
    surplus_field_out: str = "projected_year_end_surplus_usd",
    capital_is_thousands: bool = False,
    mode: str | None = None,                 # NEW: "FIXED_YEAR" | "EWA_5Y"
    lookback_years: int | None = None,       # NEW: window length (yrs)
    half_life: float | None = None,          # NEW: EWA half-life (yrs)
) -> dict:
    """
    Parse Citizens’ surplus from a CSV that may contain thousands-separators and
    parentheses. Returns a dict like:
        {surplus_field_out: <float>, "year": <int>}

    If mode == "EWA_5Y", returns the exponentially-weighted average (anchored at
    `year` or config fallback), using `lookback_years` and `half_life`. The
    returned "year" will be the anchor year used for the EWA.
    """
    import os
    import pandas as pd
    from . import config as cfg

    csv_path = path or str(getattr(cfg, "CITIZENS_CAPITAL_CSV"))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Citizens capital CSV not found: {csv_path}")

    rows: list[tuple[int, float]] = []

    with open(csv_path, "r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            raise ValueError("Citizens capital CSV is empty.")

        # Very tolerant parsing: first col ~ year, everything after first comma is capital token
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            y_str, cap_str = parts[0].strip(), parts[1].strip()
            y = pd.to_numeric(y_str, errors="coerce")
            if pd.isna(y):
                continue
            cap = _parse_money_to_float(cap_str)
            if capital_is_thousands:
                cap *= 1_000.0
            rows.append((int(y), float(cap)))

    if not rows:
        raise ValueError("No valid rows parsed from Citizens capital CSV.")

    df = pd.DataFrame(rows, columns=["year", "capital_num"]).dropna()
    df["year"] = df["year"].astype(int)
    df["capital_num"] = pd.to_numeric(df["capital_num"], errors="coerce").fillna(0.0)

    # --- Defaults / knobs
    mode = (mode or getattr(cfg, "SAMPLING_MODE_CAPITAL", "FIXED_YEAR")).upper()
    anchor_year = int(year if year is not None else getattr(cfg, "CAPITAL_YEAR", getattr(cfg, "FIXED_YEAR", 2024)))
    lookback_years = int(lookback_years or getattr(cfg, "EWA_WINDOW_YEARS", 5))
    half_life = float(half_life or getattr(cfg, "EWA_HALF_LIFE_YEARS", 2.0))

    def _ewa_weights(end_year: int, window: int, hl: float) -> pd.DataFrame:
        # end_year inclusive; window>=1
        ys = list(range(end_year - window + 1, end_year + 1))
        raw = [0.5 ** ((end_year - y) / max(hl, 1e-9)) for y in ys]
        s = sum(raw) or 1.0
        return pd.DataFrame({"year": ys, "weight": [w / s for w in raw]})

    if mode == "FIXED_YEAR":
        sel = df.loc[df["year"] == anchor_year]
        if sel.empty:
            # fallback = latest available year
            latest = int(df["year"].max())
            sel = df.loc[df["year"] == latest]
            anchor_year = latest
        cap_val = float(sel["capital_num"].sum())

    elif mode == "EWA_5Y":
        # Limit window to available range if needed
        min_avail, max_avail = int(df["year"].min()), int(df["year"].max())
        if anchor_year < min_avail:
            anchor_year = max_avail  # push to latest available if anchor too early
        start = max(min_avail, anchor_year - lookback_years + 1)
        window = anchor_year - start + 1
        weights = _ewa_weights(anchor_year, window, half_life)
        win = df.merge(weights, on="year", how="inner")
        if win.empty:
            cap_val = 0.0
        else:
            cap_val = float((win["capital_num"] * win["weight"]).sum())
    else:
        raise ValueError("mode must be one of {'FIXED_YEAR','EWA_5Y'}")

    return {surplus_field_out: cap_val, "year": int(anchor_year)}


def _flatten_multilevel_columns(cols) -> list[str]:
    flat = []
    for parts in zip(*[cols.get_level_values(i) for i in range(cols.nlevels)]):
        tokens = [str(p).strip() for p in parts if pd.notna(p) and str(p).strip() != ""]
        flat.append(" | ".join(tokens).replace("\n", " "))
    return flat

def _pick_col(flat_cols: list[str], *must_include: str, year: Optional[int] = None) -> Optional[str]:
    """
    Return the first column name that contains all must_include tokens (case-insensitive),
    optionally also containing the given year (as string). Falls back to year-agnostic match.
    """
    tokens = [t.lower() for t in must_include if t]
    year_str = str(year) if year is not None else None

    def matches(c: str, require_year: bool) -> bool:
        s = c.lower()
        if not all(tok in s for tok in tokens):
            return False
        if require_year and year_str is not None and year_str not in s:
            return False
        return True

    # Prefer year-specific
    for c in flat_cols:
        if matches(c, require_year=True):
            return c
    # Fallback without year
    for c in flat_cols:
        if matches(c, require_year=False):
            return c
    return None

def load_reserves_data(
    path: str,
    sheet_name: int | str = 0,
    header_rows: Tuple[int, int] | Tuple[int, int, int] | Tuple[int, int, int, int] = (2, 5),
    company_col_fragment: str = "Entity Name",
    statkey_col_fragment: str = "S&P Statutory Entity Key",
    year: int = 2024,
) -> pd.DataFrame:
    """
    Load entity-level FL homeowners reserves and FL DPW from a multi-row header Excel.

    Returns
    -------
    pd.DataFrame
        Columns: ['Company','StatEntityKey','HOReservesUSD','DPW_FL_USD']
        - HOReservesUSD and DPW_FL_USD are in USD (not thousands).
        - StatEntityKey is the preferred join key for downstream merges.
    """
    # Pandas expects a list of header rows for MultiIndex columns
    if isinstance(header_rows, tuple):
        header = list(range(header_rows[0], header_rows[-1] + 1))
    else:
        header = header_rows

    df = pd.read_excel(path, sheet_name=sheet_name, header=header)

    # Flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = _flatten_multilevel_columns(df.columns)
    else:
        flat_cols = [str(c).strip() for c in df.columns]
    df.columns = flat_cols

    # --- Identify key columns ---
    comp_candidates = [c for c in flat_cols if company_col_fragment.lower() in c.lower()]
    if not comp_candidates:
        raise KeyError(f"No column containing '{company_col_fragment}'")
    company_col = comp_candidates[0]

    sk_candidates = [c for c in flat_cols if statkey_col_fragment.lower() in c.lower()]
    sk_col = sk_candidates[0] if sk_candidates else None

    # Reserves (prefer implied FL; fallback to generic HO/Farmowners)
    reserves_col = _pick_col(
        flat_cols,
        "IMPLIED FLORIDA RESERVES FOR HOMEONWER",  # keep exact header text
        year=year
    )
    if reserves_col is None:
        reserves_col = _pick_col(
            flat_cols,
            "Homeowner, Farmowner: Loss & Loss Adj Exp Reserves ($000)",
            year=year
        )
    if reserves_col is None:
        raise KeyError("Could not locate a reserves column for the requested year.")

    # Florida DPW preferred; fallback to broader DPW if needed
    dpw_fl_col = _pick_col(
        flat_cols,
        "Direct Premiums Written ($000)",
        "DPW_LOB_GEO_PC",
        "AR: Homeowners MP|NAIC Rptd:FL",
        year=year
    )
    if dpw_fl_col is None:
        dpw_fl_col = _pick_col(
            flat_cols,
            "Direct Premiums Written ($000)",
            "PREMS_WRITTEN_FROM_DIRECT_BUSINESS_U_AND_I",
            "AR: Homeowners MP",
            year=year
        )

    keep_cols = [company_col, reserves_col]
    if sk_col:
        keep_cols.append(sk_col)
    if dpw_fl_col:
        keep_cols.append(dpw_fl_col)

    out = df[keep_cols].dropna(subset=[company_col]).copy()

    rename_map = {
        company_col: "Company",
        reserves_col: "Reserves_thousands",
    }
    if sk_col:
        rename_map[sk_col] = "StatEntityKey"
    if dpw_fl_col:
        rename_map[dpw_fl_col] = "DPW_FL_thousands"

    out = out.rename(columns=rename_map)

    # Convert $000 to $
    out["HOReservesUSD"] = pd.to_numeric(out["Reserves_thousands"], errors="coerce").fillna(0.0) * 1_000.0
    out = out.drop(columns=["Reserves_thousands"])

    if "DPW_FL_thousands" in out.columns:
        out["DPW_FL_USD"] = pd.to_numeric(out["DPW_FL_thousands"], errors="coerce").fillna(0.0) * 1_000.0
        out = out.drop(columns=["DPW_FL_thousands"])
    else:
        out["DPW_FL_USD"] = 0.0

    # Clean strings
    out["Company"] = out["Company"].astype(str).str.strip()
    if "StatEntityKey" in out.columns:
        out["StatEntityKey"] = out["StatEntityKey"].astype(str).str.strip()
    else:
        # If the sheet somehow lacks the key, keep the column and let downstream raise/log
        out["StatEntityKey"] = ""

    return out[["Company", "StatEntityKey", "HOReservesUSD", "DPW_FL_USD"]]


def build_rbc_df(
    reserves_df: pd.DataFrame,
    k_res: float = 0.35,
    k_wp: Optional[float] = 0.05
) -> pd.DataFrame:
    """
    Build an RBC requirement table from reserves_df.

    RBCReq = max(k_res * HOReservesUSD, k_wp * DPW_FL_USD) if k_wp is not None,
             else k_res * HOReservesUSD.

    Returns
    -------
    pd.DataFrame
        Columns: ['Company','StatEntityKey','RBCReq']
        (StatEntityKey is provided for reliable merges in runner/capital)
    """
    df = reserves_df.copy()
    df["HOReservesUSD"] = pd.to_numeric(df["HOReservesUSD"], errors="coerce").fillna(0.0)
    if "DPW_FL_USD" not in df.columns:
        df["DPW_FL_USD"] = 0.0
    else:
        df["DPW_FL_USD"] = pd.to_numeric(df["DPW_FL_USD"], errors="coerce").fillna(0.0)

    r_from_res = k_res * df["HOReservesUSD"]
    if k_wp is not None:
        r_from_prem = k_wp * df["DPW_FL_USD"]
        df["RBCReq"] = pd.concat([r_from_res, r_from_prem], axis=1).max(axis=1)
    else:
        df["RBCReq"] = r_from_res

    # Ensure keys exist
    if "StatEntityKey" not in df.columns:
        df["StatEntityKey"] = ""

    return df[["Company", "StatEntityKey", "RBCReq"]]

# --- NFIP capital check -------------------------------------------------------
def compute_nfip_capital_depletion(
    paid_total: float,
    premium_base: float,
    *,
    mode: str = "UNLIMITED",
    pool_usd: float = 0.0,
    surcharge_max_rate: float = 0.0,
    borrow_enabled: bool = False,
) -> dict:
    """
    Policy-agnostic NFIP depletion summary.
    Returns keys: mode, paid_total, pool_capacity, pool_used, surcharge_capacity,
                  surcharge_rate, surcharge_used, borrowed, residual_unfunded
    """
    paid_total = float(paid_total or 0.0)
    premium_base = float(premium_base or 0.0)
    pool_usd = float(pool_usd or 0.0)
    surcharge_max_rate = max(float(surcharge_max_rate or 0.0), 0.0)

    if mode.upper() == "UNLIMITED":
        return {
            "mode": "UNLIMITED",
            "paid_total": paid_total,
            "pool_capacity": float("inf"),
            "pool_used": 0.0,
            "surcharge_capacity": float("inf"),
            "surcharge_rate": 0.0,
            "surcharge_used": 0.0,
            "borrowed": 0.0,
            "residual_unfunded": 0.0,
        }

    # CAPPED or BORROW
    pool_used = min(paid_total, pool_usd)
    remaining = paid_total - pool_used

    surcharge_cap_amt = premium_base * surcharge_max_rate if premium_base > 0 else 0.0
    surcharge_used = min(remaining, surcharge_cap_amt)
    surcharge_rate = (surcharge_used / premium_base) if premium_base > 0 else 0.0
    remaining_after = remaining - surcharge_used

    if mode.upper() == "BORROW" and borrow_enabled:
        borrowed = remaining_after
        residual_unfunded = 0.0
    else:
        borrowed = 0.0
        residual_unfunded = remaining_after

    return {
        "mode": mode.upper(),
        "paid_total": paid_total,
        "pool_capacity": float(pool_usd),
        "pool_used": float(pool_used),
        "surcharge_capacity": float(surcharge_cap_amt),
        "surcharge_rate": float(surcharge_rate),
        "surcharge_used": float(surcharge_used),
        "borrowed": float(borrowed),
        "residual_unfunded": float(residual_unfunded),
    }