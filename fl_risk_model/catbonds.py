"""
catbonds.py - Stylized catastrophe bond recoveries
------------------------------------------------------------------

Uses a reviewed inventory and explicit beneficiary keys. Attachment and
limits are stylized principal-based layers, applied once to aggregate losses.
This does not reconstruct transaction-specific contractual payouts.

Public API
----------
- CatBond (dataclass)
- parse_cat_bond_terms
- compute_cat_bond_recovery
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import pandas as pd
import numpy as np

def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"\b(company|co\.?|corporation|corp\.?|inc\.?|limited|ltd\.?|llc|group|mutual|insurance|ins\.?|property|casualty|exchange|holdings|services|service|plc|p\.?l\.?c)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _is_fl_relevant(perils: str) -> bool:
    if not isinstance(perils, str):
        return False
    p = perils.lower()
    if "excl. florida" in p or "exclude florida" in p:
        return False
    if "florida" in p:
        return True
    # A US heading followed by an explicit state list is not nationwide cover.
    if ":" in p:
        return False
    if "us named storm" in p or "u.s. named storm" in p or "u.s named storm" in p:
        return True
    if "tropical cyclone" in p and ("us" in p or "u.s" in p):
        return True
    # “Florida multi-peril” or “US hurricane” variants will pass via 'florida'/'us named storm' above
    return "us multi-peril" in p or "southeast us named storm" in p

def _in_force_for_season(issue_date_str: str, season_year: int = 2024) -> bool:
    """
    Pragmatic rule:
      - Jan..Sep of season_year => in force for that season
      - Oct..Dec of season_year => for next season (exclude)
    Unparseable dates are excluded. This is a September 2024 snapshot,
    not a reconstruction of coverage on each historical event date.
    """
    try:
        value = str(issue_date_str).strip()
        if re.fullmatch(r"[A-Za-z]{3}\.\d{2}", value):
            dt = pd.to_datetime(value, format="%b.%y", errors="raise")
        else:
            dt = pd.to_datetime(value, errors="raise")
        if pd.isna(dt):
            return False
        if dt.year != season_year:
            # If you later include 2023+, you can add a tenor heuristic here
            return dt.year < season_year  # include prior issues by default
        return dt.month <= 9
    except Exception:
        return False

def load_catbond_table(path: str | pd.DataFrame,
                       season_year: int = 2024,
                       attach_mult: float = 1.0,
                       exhaust_mult: float = 2.0) -> pd.DataFrame:
    """
    Load your catbonds CSV and add:
      - FL_relevant (bool)
      - InForce (bool)
      - LimitUSD, AttachUSD, ExhaustUSD (defaults when not provided)
      - TriggerClass ('indemnity' or 'industry')
      - BondID (Issuer:Series)
    Expected columns (from your catbonds_2024.csv):
      ['Bond_Name','Series','Issuer','Cedent_Sponsor','Size_Million_USD',
       'Trigger_Type','Risks_Perils','Issue_Date', ...]
    """
    df = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    required = {"ModelInclude", "BeneficiaryStatEntityKeys", "EligibilityReason"}
    if not required.issubset(df.columns):
        raise ValueError("Use the reviewed bond inventory with explicit eligibility and beneficiary keys.")
    # Clean columns we care about
    if "Size_Million_USD" in df.columns:
        limit = pd.to_numeric(df["Size_Million_USD"], errors="coerce") * 1_000_000.0
    elif "SizeUSD" in df.columns:
        limit = pd.to_numeric(df["SizeUSD"], errors="coerce")
    else:
        limit = pd.Series(0.0, index=df.index)

    issues = df.get("Issue_Date_Sort", df.get("Issue_Date", df.get("IssueDate", "")))

    out = pd.DataFrame({
        "Bond_Name": df.get("Bond_Name", df.get("BondName", "")),
        "Series": df.get("Series", ""),
        "Issuer": df.get("Issuer", df.get("IssuerSPV","")),
        "Cedent_Sponsor": df.get("Cedent_Sponsor", df.get("Cedent","")),
        "Risks_Perils": df.get("Risks_Perils", df.get("Perils","")),
        "Trigger_Type": df.get("Trigger_Type", df.get("TriggerType","")),
        "Issue_Date": issues,
        "LimitUSD": limit.fillna(0.0),
        "ModelInclude": df["ModelInclude"].astype(str).str.lower().isin(["1", "true"]),
        "BeneficiaryStatEntityKeys": df["BeneficiaryStatEntityKeys"].fillna("").astype(str),
        "EligibilityReason": df["EligibilityReason"],
    })

    out["BondID"] = (out["Issuer"].astype(str).str.strip() + ":" +
                     out["Series"].astype(str).str.strip()).str.strip(":")

    out["FL_relevant"] = out["Risks_Perils"].apply(_is_fl_relevant)
    out["InForce"]     = out["Issue_Date"].apply(lambda s: _in_force_for_season(s, season_year))

    # Trigger normalization
    trig = out["Trigger_Type"].astype(str).str.lower()
    out["TriggerClass"] = np.select(
        [trig.str.contains("industry|pcs|index"), trig.str.fullmatch("indemnity")],
        ["industry", "indemnity"], default="unsupported")

    # Defaults for layer points
    if "AttachmentUSD" in df.columns and "ExhaustionUSD" in df.columns:
        out["AttachmentUSD"] = pd.to_numeric(df["AttachmentUSD"], errors="coerce")
        out["ExhaustionUSD"] = pd.to_numeric(df["ExhaustionUSD"], errors="coerce")
        # Fill sensible defaults if blank
        needs = out["AttachmentUSD"].isna() | out["ExhaustionUSD"].isna()
        out.loc[needs, "AttachmentUSD"] = attach_mult * out.loc[needs, "LimitUSD"]
        out.loc[needs, "ExhaustionUSD"] = (attach_mult + (exhaust_mult - attach_mult)) * out.loc[needs, "LimitUSD"]
    else:
        out["AttachmentUSD"] = attach_mult * out["LimitUSD"]
        out["ExhaustionUSD"] = exhaust_mult * out["LimitUSD"]

    # Clip Exhaust >= Attach + tiny eps
    out["ExhaustionUSD"] = np.maximum(out["ExhaustionUSD"], out["AttachmentUSD"] + 1.0)

    # Keep only what can matter this season in FL
    out = out[out["ModelInclude"] & out["FL_relevant"] & out["InForce"]
              & (out["LimitUSD"] > 0) & out["TriggerClass"].ne("unsupported")].copy()
    if out["BeneficiaryStatEntityKeys"].str.strip().eq("").any():
        raise ValueError("An included bond lacks explicit beneficiary keys.")
    return out

def _build_cedent_to_keys(crosswalk: pd.DataFrame) -> pd.DataFrame:
    cw = crosswalk.copy()
    # Expected columns: StatEntityKey, Company_MS, NAIC, Company_FHCF, fhcf_participant
    cw["norm_MS"] = cw["Company_MS"].map(_norm_name)
    cw["norm_FHCF"] = cw["Company_FHCF"].map(_norm_name)
    return cw

def _lookup_keys_for_cedent(cedent: str, cw: pd.DataFrame) -> pd.DataFrame:
    n = _norm_name(cedent)
    hit = cw[(cw["norm_MS"] == n) | (cw["norm_FHCF"] == n)]
    return hit[["StatEntityKey","NAIC","fhcf_participant","Company_MS","Company_FHCF"]].drop_duplicates()

def _payout_occurrence(driver: float, attach: float, limit: float) -> float:
    base = max(driver - attach, 0.0)
    return float(min(base, limit))

def apply_catbond_recovery(
    private_after_fhcf: pd.DataFrame,
    citizens_after_fhcf: pd.DataFrame,
    catbonds: pd.DataFrame,
    market_share_df: pd.DataFrame,
    company_keys_df: pd.DataFrame,
    *,
    industry_insured_wind_pre_fhcf_usd: Optional[float] = None,
    issue_year: Optional[int] = None,   # <- NEW
) -> Tuple[pd.DataFrame, dict]:
    """
    Returns (payout_by_company_county_df, diag)
      - payout_by_company_county_df has ['Company','County','CatBondRecoveryUSD']
      - diag contains bond-level and sponsor totals + capacity diagnostics
    """

    # --- normalize sizes (no year logic needed) ---
    cb = catbonds.copy()

    # Coerce any numeric-ish columns early
    for col in ["AttachmentUSD", "LimitUSD", "IssueSizeUSD", "SizeUSD", "Size_Million_USD"]:
        if col in cb.columns:
            cb[col] = pd.to_numeric(cb[col], errors="coerce")

    # Build IssueSizeUSD if it's missing or zeros
    if "IssueSizeUSD" not in cb.columns or float(cb["IssueSizeUSD"].fillna(0).sum()) == 0.0:
        if "Size_Million_USD" in cb.columns and cb["Size_Million_USD"].notna().any():
            cb["IssueSizeUSD"] = cb["Size_Million_USD"].fillna(0.0) * 1_000_000.0
        elif "SizeUSD" in cb.columns:
            cb["IssueSizeUSD"] = cb["SizeUSD"].fillna(0.0)
        else:
            # As a last resort, treat limit as issue size for capacity diagnostics
            cb["IssueSizeUSD"] = pd.to_numeric(cb.get("LimitUSD", 0.0), errors="coerce").fillna(0.0)

    # If LimitUSD is missing, mirror IssueSizeUSD so we can still report "limit in force"
    if "LimitUSD" not in cb.columns:
        cb["LimitUSD"] = cb["IssueSizeUSD"]
    else:
        cb["LimitUSD"] = pd.to_numeric(cb["LimitUSD"], errors="coerce").fillna(0.0)

    # Unify trigger class naming
    if "TriggerClass" not in cb.columns:
        # Try to infer: if 'TriggerType' exists and equals 'Indemnity', treat as indemnity
        if "TriggerType" in cb.columns:
            cb["TriggerClass"] = np.where(cb["TriggerType"].str.lower().str.contains("industry", na=False),
                                          "industry", "indemnity")
        else:
            cb["TriggerClass"] = "indemnity"

    # Robust year extraction for capacity diagnostics
    year_series = None
    for col in ("InceptionYear", "IssueYear", "InceptionDate", "IssueDate"):
        if col in cb.columns:
            if col.endswith("Date"):
                year_series = pd.to_datetime(cb[col], errors="coerce").dt.year
            else:
                year_series = pd.to_numeric(cb[col], errors="coerce")
            cb["_Year"] = year_series
            break

    # ---------- Existing mapping & recovery logic (unchanged except iterate over cb) ----------
    ms = market_share_df.copy()
    if "StatEntityKey" not in ms.columns and "Stat Entity Key" in ms.columns:
        ms = ms.rename(columns={"Stat Entity Key": "StatEntityKey"})
    ms["StatEntityKey"] = ms["StatEntityKey"].astype(str)

    cols_needed = {"Company", "County", "NetWindUSD"}
    for df in (private_after_fhcf, citizens_after_fhcf):
        if not set(df.columns).issuperset(cols_needed):
            raise KeyError(f"[catbond] missing columns {cols_needed} in one of the FHCF outputs")

    company_net = (
        pd.concat(
            [
                private_after_fhcf[["Company", "County", "NetWindUSD"]],
                citizens_after_fhcf[["Company", "County", "NetWindUSD"]],
            ],
            ignore_index=True,
        )
        .groupby("Company", as_index=False)["NetWindUSD"].sum()
        .merge(ms[["Company", "StatEntityKey"]].drop_duplicates(), on="Company", how="left")
    )

    cw = _build_cedent_to_keys(company_keys_df)
    # Citizens uses its full legal name in the loss table and an abbreviated
    # display name in market shares. Resolve both names through the crosswalk.
    aliases = pd.concat([
        cw[[name, "StatEntityKey"]].rename(columns={name: "Company"})
        for name in ["Company_MS", "Company_FHCF"]
    ], ignore_index=True).dropna().drop_duplicates()
    if aliases.groupby("Company")["StatEntityKey"].nunique().gt(1).any():
        raise ValueError("Company crosswalk maps one name to multiple legal entities")
    aliases = aliases.drop_duplicates("Company").set_index("Company")["StatEntityKey"]
    company_net["StatEntityKey"] = company_net["StatEntityKey"].fillna(company_net["Company"].map(aliases))

    if industry_insured_wind_pre_fhcf_usd is None:
        industry_driver = float(company_net["NetWindUSD"].sum())
    else:
        industry_driver = float(industry_insured_wind_pre_fhcf_usd)

    payouts: List[Dict[str, Any]] = []
    bond_diag: List[Dict[str, Any]] = []

    # Eligibility is a property of the modeled roster, not this season's losses.
    # The upstream FHCF branch returns no private rows when wind losses are zero.
    modeled_keys = set(ms["StatEntityKey"].dropna().astype(str))
    known_keys = set(cw["StatEntityKey"].dropna().astype(str))

    for _, b in cb.iterrows():
        ced = str(b["Cedent_Sponsor"])
        keys = [k.strip() for k in str(b.get("BeneficiaryStatEntityKeys", "")).split(";") if k.strip()]
        if not keys or not set(keys).issubset(known_keys):
            raise ValueError(f"Missing or unknown beneficiary keys for {b.get('BondID', ced)}: {keys}")
        if not set(keys).issubset(modeled_keys):
            raise ValueError(f"Beneficiary outside modeled market for {b.get('BondID', ced)}: {keys}")
        sponsor_keys = keys
        if b["TriggerClass"] == "industry":
            driver = industry_driver
        elif b["TriggerClass"] == "indemnity":
            driver = float(company_net.loc[company_net["StatEntityKey"].isin(keys), "NetWindUSD"].sum())
        else:
            raise ValueError(f"Unsupported trigger for {b.get('BondID', ced)}")

        attach = float(b.get("AttachmentUSD", 0.0))
        limit  = float(b.get("LimitUSD", 0.0))
        payout = _payout_occurrence(driver, attach, limit)

        if payout <= 0:
            bond_diag.append({
                "BondID": b.get("BondID", ""),
                "Cedent": ced,
                "TriggerClass": b["TriggerClass"],
                "DriverUSD": driver,
                "AttachmentUSD": attach,
                "LimitUSD": limit,
                "PayoutUSD": 0.0,
            })
            continue

        # The trigger determines the payout; only named beneficiaries receive it.
        sponsor_rows = company_net.loc[company_net["StatEntityKey"].isin(sponsor_keys)].copy()
        tot = float(sponsor_rows["NetWindUSD"].sum())
        if tot <= 0:
            continue
        sponsor_rows["alloc_share"] = sponsor_rows["NetWindUSD"] / tot
        sponsor_rows["alloc_usd"]   = payout * sponsor_rows["alloc_share"]

        base_rows = pd.concat(
            [
                private_after_fhcf[["Company", "County", "NetWindUSD"]],
                citizens_after_fhcf[["Company", "County", "NetWindUSD"]],
            ],
            ignore_index=True,
        )
        base_rows = base_rows.merge(
            sponsor_rows[["Company", "NetWindUSD", "alloc_usd"]],
            on="Company",
            how="inner",
            suffixes=("", "_comp"),
        )
        comp_tot = base_rows.groupby("Company")["NetWindUSD"].transform("sum")
        base_rows["county_share"] = np.where(
            base_rows["NetWindUSD"] > 0, base_rows["NetWindUSD"] / comp_tot, 0.0
        )
        base_rows["CatBondRecoveryUSD"] = base_rows["alloc_usd"] * base_rows["county_share"]
        payouts.append(base_rows[["Company", "County", "CatBondRecoveryUSD"]])

        bond_diag.append({
            "BondID": b.get("BondID", ""),
            "Cedent": ced,
            "TriggerClass": b["TriggerClass"],
            "DriverUSD": driver,
            "AttachmentUSD": attach,
            "LimitUSD": limit,
            "PayoutUSD": payout,
        })

    if payouts:
        recov = pd.concat(payouts, ignore_index=True)
        recov = recov.groupby(["Company", "County"], as_index=False)["CatBondRecoveryUSD"].sum()
        # Several layers must not credit more than the retained claim.
        available = pd.concat([private_after_fhcf, citizens_after_fhcf], ignore_index=True)
        available = available.groupby(["Company", "County"], as_index=False)["NetWindUSD"].sum()
        recov = recov.merge(available, on=["Company", "County"], validate="one_to_one")
        recov["CatBondRecoveryUSD"] = np.minimum(recov["CatBondRecoveryUSD"], recov["NetWindUSD"].clip(lower=0))
        recov = recov.drop(columns="NetWindUSD")
        total_payout = float(recov["CatBondRecoveryUSD"].sum())
    else:
        recov = pd.DataFrame(columns=["Company", "County", "CatBondRecoveryUSD"])
        total_payout = 0.0

    # ---------- Capacity diagnostics ----------
    # ---------- Capacity diagnostics (single-year file) ----------
    limit_in_force = float(cb["LimitUSD"].sum())
    issue_vol_year = float(cb["IssueSizeUSD"].sum())
    # limit_in_force = float(pd.to_numeric(cb.get("LimitUSD", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    # if issue_year is not None and "IssueSizeUSD" in cb.columns and "_Year" in cb.columns:
    #     issue_vol_year = float(
    #         pd.to_numeric(cb.loc[(cb["_Year"] == issue_year), "IssueSizeUSD"], errors="coerce").fillna(0).sum()
    #     )
    # else:
    #     issue_vol_year = 0.0

    diag = {
        "bond_diag": pd.DataFrame(bond_diag),
        "catbond_payout_total": total_payout,
        "catbond_triggered_payout_total": float(sum(d.get("PayoutUSD", 0.0) for d in bond_diag)),
        "catbond_attachment_hits": int(sum(d.get("PayoutUSD", 0.0) > 0 for d in bond_diag)),
        "catbond_limit_in_force_usd": limit_in_force,
        "catbond_issue_volume_year_usd": issue_vol_year,
    }
    return recov, diag
