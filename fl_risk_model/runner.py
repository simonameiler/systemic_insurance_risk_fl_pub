"""
runner.py - Scenario runner for the Florida risk-flow model
-------------------------------------------------------------------------

Flow (aligned with Methods)
---------------------------
1) Build exposures (private + Citizens), optionally sampled (cfg.SAMPLE_EXPOSURE / cfg.EXPOSURE_COV).
   - Private exposure is used to allocate insured wind among private carriers.
   - Citizens exposure is loaded via loader.load_citizens_county (County View + FIPS).
   NOTE: market_share_df may include 'StatEntityKey'; ignored for allocation but kept for metadata.
2) Load county wind damages (USD).
3) CARVE-OUT at county level (insured/underinsured/uninsured).
4) PRE-SPLIT insured wind by Citizens/Private share of county TIV.
5) PRIVATE: allocate private share by exposure -> FHCF (using keyed terms) -> apply seasonal industry cap (pro-rata).
6) CITIZENS: allocate Citizens share (County View) -> FHCF (using keyed terms) -> Citizens capital hit.
7) Flood -> NFIP recovery.
8) Aggregate company losses; load surplus; apply capital depletion and group support.
9) Assessments: FIGA (private, keyed by Statutory Entity Key) and Citizens Tier-1/Tier-2.

Notes
-----
- Applies a **single** industry seasonal cap across Private + Citizens (pro-rata), once.
- Removes duplicate helpers and dead code; keeps public signatures intact.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
import datetime as _dt

from fl_risk_model import config as cfg
from fl_risk_model.utils import make_xwalk_from_tiger, norm_county_name

from fl_risk_model.loader import (
    load_citizens_county,
    load_private_premium_base_from_market_share_xlsx,
    load_nfip_policy_coverage,
    load_citizens_premium_base,
    load_nfip_county_exposure,
    load_nfip_fl_premium_base,
    load_fhcf_county_exposure,
)
from fl_risk_model.exposure import build_wind_exposures, build_wind_exposures_alt_company_county
from fl_risk_model.branches.wind import load_wind_damage, allocate_insured_wind_to_private
from fl_risk_model.branches.flood import load_water_damage_scenario 
from fl_risk_model.branches.citizens import (
    allocate_insured_wind_to_citizens,
    prepare_citizens_gross_wind,
    recover_citizens_wind,
    CITIZENS_NAME,
    apply_citizens_capital_hit,
    citizens_fhcf_terms_from_cfg_or_csv
)
from fl_risk_model.fhcf import (
    apply_fhcf_recovery, 
    attach_fhcf_terms_for_losses,
    normalize_fhcf_terms,
    )
from fl_risk_model.nfip import (
    load_nfip_claims_county_year, 
    aggregate_nfip_claims, 
    make_nfip_payout_rates, 
    carveout_flood_from_penetration,
    load_nfip_penetration,
)
from fl_risk_model.capital import (
    apply_losses_to_surplus,
    load_surplus_data_with_groups,
    apply_group_capital_contributions,
    load_citizens_capital_row_from_csv,
)
from fl_risk_model.catbonds import (
    load_catbond_table,
    apply_catbond_recovery,
)
from fl_risk_model.branches.uninsured import apply_gross_carveout_wind

# Scenario analysis modules
from fl_risk_model import scenarios

# =============================================================================
# RNG + debug helpers
# =============================================================================

def rng_gen(seed: int | None = None) -> np.random.Generator:
    """Create a numpy Generator seeded from arg or cfg.RNG_SEED."""
    if seed is None:
        seed = getattr(cfg, "RNG_SEED", None)
    return np.random.default_rng(seed)

DEBUG_PRINTS = bool(getattr(cfg, "DEBUG_PRINTS", False))
# def dbg(*args, **kwargs):
#     if DEBUG_PRINTS:
#         print(*args, **kwargs)
def dbg(*args, **kwargs):
    if bool(getattr(cfg, "DEBUG_PRINTS", False)):
        print(*args, **kwargs)

def make_output_dir(storm_tag: str = "adhoc") -> Path:
    """
    Create a stable debug folder under <repo>/results/debug/nfip_<storm_tag>.
    Falls back to CWD if PROJECT_ROOT is missing.
    """
    base = getattr(cfg, "PROJECT_ROOT", Path.cwd())
    out = Path(base) / "results" / "debug" / f"nfip_{storm_tag}"
    out.mkdir(parents=True, exist_ok=True)
    return out

def numseries(x, default=0.0):
    if x is None:
        return pd.Series([default], dtype="float64")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return pd.Series([x], dtype="float64").fillna(default)
    try:
        s = pd.Series(x)
        return pd.to_numeric(s, errors="coerce").fillna(default)
    except Exception:
        return pd.Series([default], dtype="float64")

def numsum(x) -> float:
    return float(numseries(x).sum())

def numval(x, default=0.0) -> float:
    s = numseries(x, default=default)
    return float(s.iloc[0] if len(s) else default)

    
def _sum_col(df, col) -> float:
    if col in df.columns:
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    return 0.0

# Use whatever identifier you have for the run; fallback is fine.
storm_tag = locals().get("storm_id") or locals().get("storm") or "adhoc"
output_dir = make_output_dir(storm_tag)

def _apply_industry_season_cap(
    private_precap: pd.DataFrame,
    citizens_precap: pd.DataFrame,
    cap_usd: float,
    *, gross_col: str = "GrossWindLossUSD"
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Pro-rata scales FHCF recoveries across Private + Citizens once per scenario
    to respect an industry-wide seasonal cap.

    Returns (private_post, citizens_post, diag)
    """
    p = private_precap.copy()
    c = citizens_precap.copy()

    p["FHCF_RecoveryPreCapUSD"] = numseries(p["FHCF_RecoveryPreCapUSD"])
    c["FHCF_RecoveryPreCapUSD"] = numseries(c["FHCF_RecoveryPreCapUSD"])

    pre_total = float(p["FHCF_RecoveryPreCapUSD"].sum() + c["FHCF_RecoveryPreCapUSD"].sum())
    if cap_usd is None or cap_usd <= 0 or pre_total <= cap_usd:
        scale = 1.0
    else:
        scale = float(cap_usd) / pre_total

    for df in (p, c):
        df["FHCF_RecoveryUSD"] = df["FHCF_RecoveryPreCapUSD"] * scale
        # compute NetWindUSD deterministically from gross - post-cap recovery
        df[gross_col] = numseries(df[gross_col])
        df["NetWindUSD"] = df[gross_col] - df["FHCF_RecoveryUSD"]

    diag = {
        "fhcf_total_precap_usd": pre_total,
        "fhcf_total_postcap_usd": pre_total * scale,
        "fhcf_shortfall_usd": max(pre_total - pre_total * scale, 0.0),
        "fhcf_scaling_factor": scale,
        "fhcf_cap_binding": bool(scale < 0.999999 and pre_total > 0),
        "fhcf_cap_usd": float(cap_usd or 0.0),
    }
    return p, c, diag


# --- Canonical County<->FIPS crosswalk -> ['County','county_fips'] ----------------

def _strip_suffix(s: str) -> str:
    return re.sub(r"\s+(County|Parish|Borough|City)$", "", str(s), flags=re.I).strip()

def _norm_display(name: str) -> str:
    return _strip_suffix(norm_county_name(name))

def _build_xwalk(county_xwalk, fhcf_county_df=None, statefp_default="12"):
    xw = None

    # 1) Prefer FHCF table if it already has both columns
    if fhcf_county_df is not None and {"County","county_fips"}.issubset(fhcf_county_df.columns):
        xw = fhcf_county_df[["County","county_fips"]].copy()

    # 2) Else: use provided crosswalk and detect columns robustly
    if xw is None:
        xraw = county_xwalk.copy()
        cols = {c.lower(): c for c in xraw.columns}

        # FIPS detection: county_fips | fips | geoid | statefp+countyfp
        if "county_fips" in cols:
            fips_series = xraw[cols["county_fips"]].astype(str)
        elif "fips" in cols:
            fips_series = xraw[cols["fips"]].astype(str)
        elif "geoid" in cols:
            fips_series = xraw[cols["geoid"]].astype(str)
        elif "statefp" in cols and "countyfp" in cols:
            sf = xraw[cols["statefp"]].astype(str).str.replace(r"\\D","",regex=True).str.zfill(2)
            cf = xraw[cols["countyfp"]].astype(str).str.replace(r"\\D","",regex=True).str.zfill(3)
            fips_series = sf + cf
        else:
            raise KeyError(
                f"county_xwalk needs FIPS-like columns: one of ['county_fips','fips','GEOID'] or the pair ['STATEFP','COUNTYFP'].\\n"
                f"Available: {list(xraw.columns)}"
            )

        county_fips = fips_series.str.replace(r"\\D","",regex=True).str.zfill(5)

        # County name detection: County | county | NAME | NAMELSAD | COUNTYNAME
        name_col = None
        for cand in ["county", "County", "NAME", "NAMELSAD", "CountyName", "COUNTYNAME"]:
            if cand in xraw.columns:
                name_col = cand
                break
        if name_col is None:
            raise KeyError(
                "county_xwalk needs a county name column (e.g., 'County', 'NAME', or 'NAMELSAD'). "
                f"Available: {list(xraw.columns)}"
            )
        county_name = (
            xraw[name_col].astype(str)
                .str.replace(r"\\s+County$", "", regex=True)
                .str.strip()
        )

        xw = pd.DataFrame({"County": county_name, "county_fips": county_fips})

    # Final clean-up
    xw["county_fips"] = xw["county_fips"].astype(str).str.replace(r"\\D","",regex=True).str.zfill(5)
    xw["County"] = xw["County"].astype(str).str.replace(r"\\s+County$","",regex=True).str.strip()
    xw = xw.dropna().drop_duplicates()[["County","county_fips"]]
    if xw.empty or not {"County","county_fips"}.issubset(xw.columns):
        raise ValueError("Failed to construct County<->FIPS crosswalk.")
    return xw

def _print_exposure_diagnostics(private_exp: pd.DataFrame,
                                    citizens_exp: pd.DataFrame,
                                    fhcf_county_df: pd.DataFrame,
                                    mode_label: str,
                                    verbose: bool = False,
                                    show_mb: bool = False) -> None:
        """Lightweight, optional console diagnostics."""
        if not verbose or private_exp.empty:
            return

        # Small, readable sample
        cols = [c for c in ["County","Company","TIV","StatEntityKey"] if c in private_exp.columns]
        print(f"[runner][{mode_label}] sample:", private_exp[cols].head(5).to_dict(orient="records"))

        # StatEntityKey coverage (only if present)
        if "StatEntityKey" in private_exp.columns:
            total = len(private_exp)
            mapped = private_exp["StatEntityKey"].notna().sum()
            pct = (mapped/total*100.0) if total else 0.0
            print(f"[check] StatEntityKey coverage: {mapped:,}/{total:,} ({pct:.1f}%)")

            unmapped = (private_exp[private_exp["StatEntityKey"].isna()]
                        .groupby("Company").size().sort_values(ascending=False).head(10))
            if not unmapped.empty:
                print("[check] Unmapped examples:")
                print(unmapped)

        # Sanity: ensure no Citizens shows up in private block
        if "Company" in private_exp.columns:
            n_cit = private_exp["Company"].astype(str).str.contains(r"(?i)\bcitizen", na=False).sum()
            if n_cit:
                print(f"[WARN] Found {n_cit} Citizens rows in private_exp (should be 0).")

        # Optional mass-balance snapshot: (Private + Citizens) vs FHCF CountyTIV
        if show_mb:
            priv = (private_exp.groupby("County", as_index=False)["TIV"]
                    .sum().rename(columns={"TIV":"Private"}))
            cit  = (citizens_exp.groupby("County", as_index=False)["TIV"]
                    .sum().rename(columns={"TIV":"Citizens"}))
            fhcf = fhcf_county_df[["County","CountyTIV"]]
            cmp = (fhcf.merge(priv, on="County", how="left")
                    .merge(cit, on="County", how="left")
                    .fillna(0))
            cmp["reconstructed"] = cmp["Private"] + cmp["Citizens"]
            cmp["delta"] = cmp["CountyTIV"] - cmp["reconstructed"]
            cmp["rel_gap_%"] = 100.0 * cmp["delta"].abs() / cmp["CountyTIV"].clip(lower=1)
            print("[check] Largest rel. gaps (top 5):")
            print(cmp.sort_values("rel_gap_%", ascending=False).head(5)[
                ["County","CountyTIV","reconstructed","delta","rel_gap_%"]
            ].to_string(index=False))

# =============================================================================
# One-scenario orchestration
# =============================================================================

def run_one_scenario(
    storm_name: str,
    fhcf_county_df: pd.DataFrame,            # ['County','CountyTIV']
    market_share_df: pd.DataFrame,           # ['Company','Share'] (may include 'StatEntityKey')
    county_xwalk: pd.DataFrame,              # TIGER-style or already ['County','county_fips']
    citizens_csv_path: str | None = None,
    citizens_as_of: str | None = None,
    citizens_products: list[str] | None = None,
    fhcf_terms_csv: str | None = None,       # kept for API compatibility; ignored (we use TERMS_KEYED)
    fhcf_terms_fallback: str = "error",      # kept for API compatibility
    nfip_claims_df: pd.DataFrame | None = None,
    nfip_exposure_df: pd.DataFrame | None = None,
    seed: int | None = None,
    do_flood: bool | None = None,
    surplus_year: int | None = None,
    citizens_capital_row: dict | None = None,
    scenario_config: dict | None = None,     # NEW: optional scenario configuration
    group_support_eligibility_threshold: float = 10.0,  # Group-to-Entity ratio threshold for support eligibility
) -> dict[str, pd.DataFrame | dict]:
    """
    Run a single storm scenario through wind (private & Citizens), optional flood, capital & assessments.
    
    NEW: scenario_config parameter enables adaptation scenario analysis while preserving baseline behavior.
    When scenario_config=None (default), runs the baseline model unchanged.
    
    scenario_config structure:
        {
            'type': 'market_exit' | 'penetration' | 'building_codes',
            'params': {<scenario-specific parameters>}
        }
    
    Parameters
    ----------
    group_support_eligibility_threshold : float, default=10.0
        Group-to-Entity surplus ratio threshold for intragroup capital support eligibility.
        Distressed insurers are eligible for group support only if their group's surplus
        is at least this many times larger than the entity's own baseline surplus.
        - Lower values (e.g., 5.0): More permissive -> more companies eligible for support
        - Higher values (e.g., 15.0): More restrictive -> fewer companies eligible
        Default of 10.0 represents a conservative threshold requiring substantial group strength.
    
    See fl_risk_model/scenarios/ modules for detailed parameter documentation.
    """
    rng = rng_gen(seed)
    # Default do_flood from cfg if not explicitly passed
    nfip_paid_total = 0.0
    nfip_pool_used = 0.0
    nfip_borrow = 0.0
    nfip_residual_unfunded = 0.0
    nfip_surcharge_rate = 0.0

    if do_flood is None:
        do_flood = bool(getattr(cfg, "DO_FLOOD", False))

    #xwalk = _build_xwalk(county_xwalk, fhcf_county_df)
    # Build xwalk with normalized 'County'
    xwalk = _build_xwalk(county_xwalk, fhcf_county_df=None)   # force normalization path
    xwalk["County"] = xwalk["County"].map(_norm_display)
    xwalk["county_fips"] = xwalk["county_fips"].astype(str).str.replace(r"\D","",regex=True)

    # --- 1) PRIVATE & Citizens exposures (optionally sampled via cfg)
    # Normalize market-share column to "Share"
    ms = market_share_df.copy()
    if "Share" not in ms.columns:
        yr = getattr(cfg, "MARKET_SHARE_YEAR", 2024)
        exact = f"MarketShare{yr}"
        if exact in ms.columns:
            ms = ms.rename(columns={exact: "Share"})
        else:
            cands = [c for c in ms.columns if str(c).lower().startswith("marketshare")]
            if cands:
                ms = ms.rename(columns={cands[0]: "Share"})

    # Choose build path
    mode = str(getattr(cfg, "EXPOSURE_METHOD", "topdown")).lower()
    if mode == "topdown":
        private_exp, citizens_exp = build_wind_exposures(
            fhcf_county_df=fhcf_county_df,
            market_share_df=ms,
            citizens_csv_path=citizens_csv_path or cfg.CITIZENS_COUNTY_CSV,
            citizens_as_of=citizens_as_of or cfg.CITIZENS_AS_OF,
            citizens_products=citizens_products or cfg.CITIZENS_PRODUCTS,
            county_xwalk=xwalk,
            rng=rng,
        )

        # Strict (but tolerant) mass-balance: FHCF ~ Private + Citizens
        priv = private_exp.groupby("County", as_index=False)["TIV"].sum().rename(columns={"TIV":"Private"})
        cit  = citizens_exp.groupby("County", as_index=False)["TIV"].sum().rename(columns={"TIV":"Citizens"})
        chk = (fhcf_county_df[["County","CountyTIV"]]
            .merge(priv, on="County", how="left")
            .merge(cit, on="County", how="left")
            .fillna(0))
        tol = 1e-3
        miss = (chk["CountyTIV"] - (chk["Private"] + chk["Citizens"])).abs()
        if (miss > tol).any():
            offenders = chk.loc[miss > tol, ["County","CountyTIV","Private","Citizens"]]
            raise AssertionError(f"Top-down mass balance miss for {len(offenders)} counties;\n{offenders.head(8)}")

    else:  # "bottomup"
        private_exp, citizens_exp = build_wind_exposures_alt_company_county(
            fhcf_path=str(cfg.EXPOSURE_FILE),
            market_share_path=str(cfg.MARKET_SHARE_XLSX),
            citizens_csv_path=str(cfg.CITIZENS_COUNTY_CSV),
            citizens_as_of=cfg.CITIZENS_AS_OF,
            citizens_products=cfg.CITIZENS_PRODUCTS,
            company_county_workbook=str(cfg.EXPOSURE_BY_COUNTY_XLSX),
            company_key_csv=str(cfg.COMPANY_KEYS),
            county_xwalk=xwalk,     # keep if you want FIPS enforcement in Citizens
            sample=None,            # defers to cfg.SAMPLE_EXPOSURE inside builder
            cov=None,               # defers to cfg.EXPOSURE_COV inside builder
        )

    # Optional diagnostics (silenced by default via config)
    _print_exposure_diagnostics(
        private_exp,
        citizens_exp,
        fhcf_county_df,
        mode_label=("TOPDOWN" if mode == "topdown" else "BOTTOMUP"),
        verbose=getattr(cfg, "VERBOSE_EXPOSURE", False),
        show_mb=getattr(cfg, "PRINT_MASSBALANCE_TOP5", False),
    )

    # =============================================================================
    # SCENARIO APPLICATION: Pre-event scenarios (modify exposures)
    # =============================================================================
    scenario_diagnostics = {}
    surplus_df_for_scenario = None  # Will be set by scenarios that adjust surplus (e.g., penetration)
    
    if scenario_config is not None and scenario_config.get("type") in ["market_exit", "penetration"]:
        scenario_type = scenario_config["type"]
        scenario_params = scenario_config.get("params", {})
        
        if scenario_type == "market_exit":
            # Apply market exit scenario with capital adjustments
            dbg(f"[SCENARIO] Applying market_exit scenario: {scenario_params.get('scenario', 'BASELINE')}")
            
            # Load surplus for capital adjustments
            surplus_df_for_scenario = load_surplus_data_with_groups(
                path=getattr(cfg, "SURPLUS_FILE"),
                year=int(surplus_year) if surplus_year is not None else int(getattr(cfg, "CITIZENS_CAPITAL_YEAR", 2024)),
            )
            
            # Load Citizens capital for growth adjustments
            citizens_cap_for_scenario = citizens_capital_row
            if citizens_cap_for_scenario is None:
                try:
                    citizens_cap_for_scenario = load_citizens_capital_row_from_csv(
                        path=getattr(cfg, "CITIZENS_CAPITAL_CSV", None),
                        year=getattr(cfg, "CITIZENS_CAPITAL_YEAR", None),
                        mode=getattr(cfg, "SAMPLING_MODE_CAPITAL", "FIXED_YEAR"),
                        lookback_years=int(getattr(cfg, "EWA_WINDOW_YEARS", 5)),
                        half_life=float(getattr(cfg, "EWA_HALF_LIFE_YEARS", 2.0)),
                        surplus_field_out="projected_year_end_surplus_usd",
                        capital_is_thousands=bool(getattr(cfg, "CITIZENS_CAPITAL_IS_THOUSANDS", False)),
                    )
                except Exception:
                    citizens_cap_for_scenario = None
            
            # Separate capital adjustment params from scenario params
            capital_params = {}
            scenario_only_params = {}
            for key, value in scenario_params.items():
                if key in ["group_capital_method", "citizens_capital_method"]:
                    capital_params[key] = value
                else:
                    scenario_only_params[key] = value
            
            # Apply market exit (without capital params)
            private_exp, citizens_exp, market_exit_diag = scenarios.apply_market_exit_scenario(
                private_exp=private_exp,
                citizens_exp=citizens_exp,
                surplus_df=surplus_df_for_scenario,
                **scenario_only_params
            )
            
            # Apply capital adjustments if requested
            if capital_params:
                group_method = capital_params.get("group_capital_method")
                citizens_method = capital_params.get("citizens_capital_method")
                
                if group_method and "companies_exited" in market_exit_diag:
                    # Adjust group capital for exited companies
                    exited_companies = market_exit_diag.get("exited_companies", [])
                    if exited_companies:
                        group_adj = scenarios.adjust_group_capital_for_exits(
                            surplus_df=surplus_df_for_scenario,
                            exited_companies=exited_companies,
                            method=group_method,
                        )
                        market_exit_diag["group_capital_adjustment_usd"] = group_adj.get("total_adjustment_usd", 0.0)
                
                if citizens_method and citizens_cap_for_scenario is not None:
                    # Adjust Citizens capital for increased exposure
                    exposure_increase = market_exit_diag.get("exposure_absorbed_usd", 0.0)
                    if exposure_increase > 0:
                        citizens_adj = scenarios.adjust_citizens_capital_for_growth(
                            citizens_capital_row=citizens_cap_for_scenario,
                            exposure_increase_usd=exposure_increase,
                            method=citizens_method,
                        )
                        market_exit_diag["citizens_capital_adjustment_usd"] = citizens_adj.get("capital_increase_usd", 0.0)
            
            scenario_diagnostics["market_exit"] = market_exit_diag
            
        elif scenario_type == "penetration":
            # Apply penetration increase scenario
            dbg(f"[SCENARIO] Applying penetration scenario: {scenario_params.get('scenario', 'BASELINE')}")
            
            # Separate surplus adjustment param from scenario params
            surplus_adjustment_method = scenario_params.get("surplus_adjustment", None)
            capital_multiplier = scenario_params.get("capital_multiplier", 1.0)  # Extract capital multiplier
            
            # Extract coastal_counties from params (don't pass it twice)
            coastal_counties = scenario_params.get("coastal_counties", None)
            if coastal_counties is None:
                try:
                    coastal_df = pd.read_csv(cfg.DATA_DIR / "florida_coastal_counties.csv")
                    coastal_counties = coastal_df["County"].str.strip().tolist()
                except Exception:
                    coastal_counties = None
            
            # Load NFIP exposure for penetration scenario
            if nfip_exposure_df is None and do_flood:
                nfip_exposure_df = load_nfip_policy_coverage(
                    path=str(getattr(cfg, "NFIP_POLICIES_CSV")),
                    mode=getattr(cfg, "SAMPLING_MODE_NFIP_POLICIES", "FIXED_YEAR"),
                    year=int(getattr(cfg, "NFIP_POLICY_YEAR", getattr(cfg, "FIXED_YEAR", 2024))),
                    lookback_years=int(getattr(cfg, "EWA_WINDOW_YEARS", 5)),
                    half_life=float(getattr(cfg, "EWA_HALF_LIFE_YEARS", 2.0)),
                    county_xwalk=xwalk,
                )
            
            # Load NFIP penetration data for SFHA-aware scaling
            try:
                penetration_df = load_nfip_penetration(getattr(cfg, "NFIP_PENETRATION_CSV"), xwalk)
            except Exception as e:
                dbg(f"[SCENARIO] Warning: Could not load NFIP penetration data: {e}")
                penetration_df = None
            
            # Filter out capital params before passing to penetration function
            # (surplus_adjustment and capital_multiplier are handled separately below)
            penetration_only_params = {k: v for k, v in scenario_params.items() 
                                      if k not in ['surplus_adjustment', 'capital_multiplier']}
            
            # Apply penetration increase
            private_exp, citizens_exp, nfip_exposure_df, penetration_diag = scenarios.apply_penetration_increase_scenario(
                private_exp=private_exp,
                citizens_exp=citizens_exp,
                nfip_df=nfip_exposure_df,
                coastal_counties=coastal_counties,
                penetration_df=penetration_df,
                **penetration_only_params
            )
            
            # Apply surplus adjustment if requested
            if surplus_adjustment_method:
                # Load surplus data
                surplus_df_for_pen = load_surplus_data_with_groups(
                    path=getattr(cfg, "SURPLUS_FILE"),
                    year=int(surplus_year) if surplus_year is not None else int(getattr(cfg, "CITIZENS_CAPITAL_YEAR", 2024)),
                )
                
                # Calculate exposure change
                exposure_change_pct = penetration_diag.get("private_exposure_change_pct", 0.0)
                dbg(f"[SURPLUS ADJ] Exposure change: {exposure_change_pct:.2f}%, Method: {surplus_adjustment_method}, Multiplier: {capital_multiplier}x")
                
                if abs(exposure_change_pct) > 0.001:  # Only if meaningful change
                    surplus_adj = scenarios.adjust_surplus_for_penetration(
                        surplus_df=surplus_df_for_pen,
                        exposure_change_pct=exposure_change_pct,
                        method=surplus_adjustment_method,
                        capital_multiplier=capital_multiplier,  # Pass capital multiplier
                    )
                    penetration_diag["surplus_adjustment_total_usd"] = surplus_adj.get("total_adjustment_usd", 0.0)
                    penetration_diag["surplus_adjustment_method"] = surplus_adjustment_method
                    dbg(f"[SURPLUS ADJ] Total adjustment: ${surplus_adj.get('total_adjustment_usd', 0.0)/1e9:.2f}B")
                    
                    # Use the adjusted surplus for capital calculations
                    surplus_df_for_scenario = surplus_adj.get("adjusted_surplus_df")
                else:
                    # No meaningful change, use original
                    surplus_df_for_scenario = surplus_df_for_pen
                    dbg(f"[SURPLUS ADJ] No adjustment (exposure change < 0.001%)")
            
            scenario_diagnostics["penetration"] = penetration_diag
# --- OLD CODE BLOCK (to be deleted) ---
    # ms = market_share_df.copy()
    # if "Share" not in ms.columns:
    #     yr = getattr(cfg, "MARKET_SHARE_YEAR", 2024)
    #     exact = f"MarketShare{yr}"
    #     if exact in ms.columns:
    #         ms = ms.rename(columns={exact: "Share"})
    #     else:
    #         cands = [c for c in ms.columns if str(c).lower().startswith("marketshare")]
    #         if cands:
    #             ms = ms.rename(columns={cands[0]: "Share"})

    # if cfg.EXPOSURE_METHOD == "topdown":
    #     private_exp, citizens_exp = build_wind_exposures(
    #         fhcf_county_df=fhcf_county_df,
    #         market_share_df=ms,
    #         citizens_csv_path=citizens_csv_path or cfg.CITIZENS_COUNTY_CSV,
    #         citizens_as_of=citizens_as_of or cfg.CITIZENS_AS_OF,
    #         citizens_products=citizens_products or cfg.CITIZENS_PRODUCTS,
    #         county_xwalk=xwalk,
    #         rng=rng,
    #     )
    
    #     # after private_exp, citizens_exp are built
    #     priv = private_exp.groupby("County", as_index=False)["TIV"].sum().rename(columns={"TIV":"Private"})
    #     cit  = citizens_exp.groupby("County", as_index=False)["TIV"].sum().rename(columns={"TIV":"Citizens"})
    #     fhcf = fhcf_county_df[["County","CountyTIV"]]
    #     chk = fhcf.merge(priv, on="County", how="left").merge(cit, on="County", how="left").fillna(0)
    #     chk["delta"] = chk["CountyTIV"] - (chk["Private"] + chk["Citizens"])
    #     assert (chk["delta"].abs() <= 1e-3).all(), f"Top-down mass balance miss:\n{chk.loc[chk['delta'].abs()>1e-3].head()}"


    # else: # "bottomup"
    #     private_exp, citizens_exp = build_wind_exposures_alt_company_county(
    #             fhcf_path=str(cfg.EXPOSURE_FILE),                  
    #             market_share_path=str(cfg.MARKET_SHARE_XLSX),         
    #             citizens_csv_path=str(cfg.CITIZENS_COUNTY_CSV),       
    #             citizens_as_of=cfg.CITIZENS_AS_OF,
    #             citizens_products=cfg.CITIZENS_PRODUCTS,
    #             company_county_workbook=str(cfg.EXPOSURE_BY_COUNTY_XLSX),
    #             company_key_csv=str(cfg.COMPANY_KEYS),
    #             county_xwalk=xwalk,  # pass if you want to enforce FIPS attachment in Citizens
    #             sample=None,        # uses cfg.SAMPLE_EXPOSURE
    #             cov=None,           # uses cfg.EXPOSURE_COV
    #     )
    #     print("[runner] ALT private rows:", f"{len(private_exp):,}", "| citizens rows:", f"{len(citizens_exp):,}")

    # if not private_exp.empty:
    #     # Small, readable sample
    #     cols = [c for c in ["County","Company","TIV","StatEntityKey"] if c in private_exp.columns]
    #     print("[runner] ALT sample:", private_exp[cols].head(5).to_dict(orient="records"))

    #     # Mapping coverage (guarded)
    #     total = len(private_exp)
    #     mapped = private_exp["StatEntityKey"].notna().sum() if "StatEntityKey" in private_exp.columns else 0
    #     pct = (mapped/total*100) if total else 0.0
    #     print(f"[check] StatEntityKey coverage: {mapped:,}/{total:,} ({pct:.1f}%)")

    #     # Unmapped examples (guarded)
    #     if "StatEntityKey" in private_exp.columns:
    #         unmapped = (private_exp[private_exp["StatEntityKey"].isna()]
    #                     .groupby("Company").size().sort_values(ascending=False).head(10))
    #         if not unmapped.empty:
    #             print("[check] Unmapped examples:")
    #             print(unmapped)

    #     # Sanity: no Citizens in private_exp
    #     n_cit = private_exp["Company"].astype(str).str.contains(r"(?i)\bcitizen", na=False).sum()
    #     if n_cit:
    #         print(f"[WARN] Found {n_cit} Citizens rows in private_exp (should be 0).")

    #     # Mass-balance snapshot: (Private + Citizens) vs FHCF CountyTIV
    #     fhcf = load_fhcf_county_exposure(str(cfg.EXPOSURE_FILE))[["County","CountyTIV"]]
    #     recon = (pd.concat([private_exp[["County","TIV"]], citizens_exp[["County","TIV"]]], ignore_index=True)
    #             .groupby("County", as_index=False)["TIV"].sum()
    #             .rename(columns={"TIV":"reconstructed"}))
    #     cmp = fhcf.merge(recon, on="County", how="left").fillna({"reconstructed":0.0})
    #     cmp["delta"] = cmp["CountyTIV"] - cmp["reconstructed"]
    #     cmp["rel_gap_%"] = 100 * cmp["delta"].abs() / cmp["CountyTIV"].clip(lower=1)
    #     print("[check] Largest rel. gaps (top 5):")
    #     print(cmp.sort_values("rel_gap_%", ascending=False).head(5).to_string(index=False))
    # else:
    #     print("[runner] ALT private is empty - check workbook path and filters.")

    # --- 2) County-level wind damages
    county_wind = load_wind_damage(storm_name)  # ['County','WindDamageUSD']
    
    # =============================================================================
    # NOTE: Building codes is now applied EARLIER in mc_run_events.py
    # =============================================================================
    # Building codes must be applied BEFORE carveout (which happens in mc_run_events.py)
    # so that underinsured/uninsured calculations use the reduced wind damage.
    # This was moved from here to mc_run_events.py._run_one_mc_iter() line ~920

    # --- 3) County-level CARVE-OUT (insured vs under/uninsured)
    # If penetration scenario is active, calculate actual insured fraction from modified TIV
    carveout_rates = None

    # Global override for sensitivity analysis
    _fixed_ins = getattr(cfg, "FIXED_INSURED_FRAC", None)
    if _fixed_ins is not None:
        _hh = 1.0 - _fixed_ins
        carveout_rates = {
            "insured": _fixed_ins,
            "underinsured": _hh * 0.30,
            "uninsured": _hh * 0.70,
        }
    elif scenario_config is not None and scenario_config.get("type") == "penetration":
        # Calculate total TIV from modified exposure
        total_tiv = private_exp["TIV"].sum() + citizens_exp["TIV"].sum()
        
        # Estimate insured fraction from penetration scenario diagnostics
        if "penetration" in scenario_diagnostics:
            wind_penetration_after = scenario_diagnostics["penetration"].get("wind_penetration_after", 0.85)
        else:
            # Fallback: estimate from TIV ratio vs baseline
            # Baseline wind penetration is ~85%, scale proportionally
            baseline_tiv = fhcf_county_df["CountyTIV"].sum()
            if baseline_tiv > 0:
                tiv_ratio = total_tiv / baseline_tiv
                wind_penetration_after = min(0.85 * tiv_ratio, 0.98)  # Cap at 98%
            else:
                wind_penetration_after = 0.85
        
        # Use penetration rate to set insured fraction
        # Underinsured is 30% of household portion (Beta(3,7) mean)
        # Uninsured is 70% of household portion
        insured_frac = wind_penetration_after
        household_portion = 1.0 - insured_frac
        under_share_of_hh = 0.30  # Beta(3,7) mean
        underinsured_frac = household_portion * under_share_of_hh
        uninsured_frac = household_portion * (1.0 - under_share_of_hh)
        
        carveout_rates = {
            "insured": insured_frac,
            "underinsured": underinsured_frac,
            "uninsured": max(0.0, uninsured_frac),
        }
        dbg(f"[SCENARIO] Using penetration-adjusted carveout rates: insured={insured_frac:.1%}, underinsured={underinsured_frac:.1%}, uninsured={uninsured_frac:.1%}")
    
    carved = apply_gross_carveout_wind(
        county_wind.rename(columns={"WindDamageUSD": "GrossWindLossUSD"}),
        county_col="County",
        loss_col="GrossWindLossUSD",
        rates=carveout_rates,
        rng=rng,
    )
    # carved: ['County','GrossWindLossUSD','InsuredWindUSD','UnderinsuredWindUSD','UninsuredWindUSD']

    # --- 4) PRE-SPLIT insured wind between Citizens and Private (use sampled Citizens exposure)
    _cit_col = "TIV_sampled" if "TIV_sampled" in citizens_exp.columns else "TIV"
    cit_tiv = (
        citizens_exp.groupby("County", as_index=False)[_cit_col]
        .sum()
        .rename(columns={_cit_col: "CitizensTIV"})
    )

    tiv_join = (
        fhcf_county_df[["County", "CountyTIV"]]
        .merge(cit_tiv, on="County", how="left")
        .fillna({"CitizensTIV": 0.0})
    )

    tiv_join["CitizensFrac"] = 0.0
    pos = tiv_join["CountyTIV"] > 0
    tiv_join.loc[pos, "CitizensFrac"] = tiv_join.loc[pos, "CitizensTIV"] / tiv_join.loc[pos, "CountyTIV"]
    tiv_join["CitizensFrac"] = tiv_join["CitizensFrac"].clip(0.0, 1.0)
    tiv_join["PrivateFrac"]  = 1.0 - tiv_join["CitizensFrac"]

    insured_split = (
        carved[["County", "InsuredWindUSD"]]
        .merge(tiv_join[["County", "CitizensFrac", "PrivateFrac"]], on="County", how="left")
        .fillna({"CitizensFrac": 0.0, "PrivateFrac": 1.0})
    )
    insured_split["CitizensInsuredUSD"] = insured_split["InsuredWindUSD"] * insured_split["CitizensFrac"]
    insured_split["PrivateInsuredUSD"]  = insured_split["InsuredWindUSD"] * insured_split["PrivateFrac"]

    citizens_insured_by_county = (
        insured_split[["County", "CitizensInsuredUSD"]]
        .rename(columns={"CitizensInsuredUSD": "InsuredWindUSD"})
    )
    private_insured_by_county = (
        insured_split[["County", "PrivateInsuredUSD"]]
        .rename(columns={"PrivateInsuredUSD": "InsuredWindUSD"})
    )

    # --- 5) PRIVATE + CITIZENS FHCF (attach terms -> recover -> seasonal cap) ---
    company_keys = pd.read_csv(cfg.DATA_DIR / "company_keys.csv",
                           dtype={"StatEntityKey": str, "NAIC": str})
    # Optional: sanity
    assert set(["StatEntityKey","Company_MS","NAIC","Company_FHCF","fhcf_participant"]).issubset(company_keys.columns)

    # 5.1 PRIVATE: company×county gross insured wind (safe empty default)
    private_alloc = allocate_insured_wind_to_private(private_exp, private_insured_by_county)
    if private_alloc is not None and not private_alloc.empty:
        priv_gross = (
            private_alloc.rename(columns={"InsuredWindUSD_alloc": "GrossWindLossUSD"})
                        [["Company", "County", "GrossWindLossUSD"]]
                        .copy()
        )
    else:
        priv_gross = pd.DataFrame(columns=["Company", "County", "GrossWindLossUSD"])

    # Optional QA (not used in FHCF math)
    insured_private_wind_pre_usd  = _sum_col(priv_gross, "GrossWindLossUSD")

    # 5.2 Load FHCF terms and normalize (creates/ensures CoveragePct_norm, RetentionUSD, LimitUSD)
    raw_terms_path = getattr(cfg, "FHCF_TERMS_CSV", cfg.DATA_DIR / "24fin_fhcf.csv")
    terms_norm = normalize_fhcf_terms(pd.read_csv(raw_terms_path))

    cit = terms_norm[terms_norm["Company"].str.contains("Citizens", case=False, na=False)]
    # Debug: dbg(cit[["Company","CoveragePct_norm","FHCFPremium","RetentionUSD","LimitUSD"]])

    # Ensure market_share_df exposes StatEntityKey for primary key join
    if "StatEntityKey" not in market_share_df.columns and "Stat Entity Key" in market_share_df.columns:
        market_share_df = market_share_df.rename(columns={"Stat Entity Key": "StatEntityKey"})
    if "StatEntityKey" in market_share_df.columns:
        market_share_df["StatEntityKey"] = market_share_df["StatEntityKey"].astype(str)

    # 5.3 PRIVATE: attach terms & compute pre-cap recovery (only if there’s loss to recover)
    loss_total = numsum(priv_gross.get("GrossWindLossUSD"))

    if loss_total > 0:
        terms_for_private = attach_fhcf_terms_for_losses(
            loss_df=priv_gross,
            terms_df=terms_norm,
            market_share_df=market_share_df,
            company_crosswalk_df=company_keys,   # deterministic mapping
            allow_name_fallback=False,           # no guessing
            qa_strict=True                       # fail only if a participant can’t be resolved
        )

        req = {"Company","CoveragePct_norm","RetentionUSD","LimitUSD"}
        missing = req - set(terms_for_private.columns)
        if missing:
            raise KeyError(f"[FHCF] Terms for private missing {missing} after attach_fhcf_terms_for_losses")

        private_precap = apply_fhcf_recovery(priv_gross, terms_for_private)
        private_precap = (private_precap[["Company","County","GrossWindLossUSD","RecoveryUSD"]]
                        .rename(columns={"RecoveryUSD":"FHCF_RecoveryPreCapUSD"}))
    else:
        private_precap = pd.DataFrame(
            columns=["Company","County","GrossWindLossUSD","FHCF_RecoveryPreCapUSD"]
        )

    # --- 5.4 CITIZENS: gross insured wind -> FHCF terms -> pre-cap recovery ---
    # CIT_NAME = getattr(cfg, "CITIZENS_COMPANY_NAME", getattr(cfg, "CITIZENS_NAME", "Citizens Property Insurance Corporation"))

    # citizens_gross = (
    #     citizens_insured_by_county[["County", "InsuredWindUSD"]]
    #     .rename(columns={"InsuredWindUSD": "GrossWindLossUSD"})
    #     .assign(Company=CIT_NAME)[["Company", "County", "GrossWindLossUSD"]]
    # )

    # # Scalar insured wind (pre-FHCF) for diagnostics/industry driver
    # insured_citizens_wind_pre_usd = float(pd.to_numeric(citizens_gross["GrossWindLossUSD"], errors="coerce").sum())
    # if DEBUG_PRINTS:
    #     dbg(f"[CIT SPLIT] insured_citizens_wind_pre_usd={insured_citizens_wind_pre_usd:,.0f}")

    # if insured_citizens_wind_pre_usd > 0:
    #     cit_terms = citizens_fhcf_terms_from_cfg_or_csv(terms_norm, company_keys, cfg)
    #     # If limit is zero after guardrails, we will do coverage-only fallback
    #     lim_val = float(pd.to_numeric(cit_terms["LimitUSD"], errors="coerce").fillna(0.0).iloc[0])

    #     if lim_val == 0.0:
    #         if DEBUG_PRINTS:
    #             dbg("[FHCF][Citizens] contract limit is zero; using coverage-only payout")
    #         cov = float(getattr(cfg, "CITIZENS_FHCF_COVERAGE_PCT", 0.90))
    #         lae = float(getattr(cfg, "FHCF_LAE_FACTOR", 0.10))
    #         citizens_precap = citizens_gross.copy()
    #         citizens_precap["FHCF_RecoveryPreCapUSD"] = (
    #             pd.to_numeric(citizens_precap["GrossWindLossUSD"], errors="coerce").fillna(0.0) * cov * (1.0 + lae)
    #         )
    #     else:
    #         citizens_precap = apply_fhcf_recovery(
    #             loss_df=citizens_gross, terms_df=cit_terms
    #         )[["Company","County","GrossWindLossUSD","RecoveryUSD"]].rename(
    #             columns={"RecoveryUSD":"FHCF_RecoveryPreCapUSD"}
    #         )
    # else:
    #     citizens_precap = pd.DataFrame(columns=["Company","County","GrossWindLossUSD","FHCF_RecoveryPreCapUSD"])
    def _citizens_terms_fallback_row(cfg) -> pd.DataFrame:
        """One-row fallback for Citizens -> will be normalized to get Retention/Limit."""
        return pd.DataFrame([{
            "Company": getattr(cfg, "CITIZENS_COMPANY_NAME", "Citizens Property Insurance Corporation"),
            "FHCFPremium": float(getattr(cfg, "CITIZENS_FHCF_PREMIUM_USD", 0.0)),
            "CoveragePct": float(getattr(cfg, "CITIZENS_FHCF_COVERAGE_PCT", 90)),  # 45/75/90 or fraction
        }])

    # --- Citizens: gross -> FHCF terms -> pre-cap recovery -------------------------
    CITIZENS_NAME = getattr(cfg, "CITIZENS_COMPANY_NAME", "Citizens Property Insurance Corporation")
    citizens_gross = (
        citizens_insured_by_county[["County", "InsuredWindUSD"]]
        .rename(columns={"InsuredWindUSD": "GrossWindLossUSD"})
        .assign(Company=CITIZENS_NAME)[["Company", "County", "GrossWindLossUSD"]]
    )

    insured_citizens_wind_pre_usd = float(pd.to_numeric(citizens_gross["GrossWindLossUSD"], errors="coerce").sum())

    # 1) Try to attach terms from normalized 24fin_fhcf using your existing helper
    terms_for_citizens = None
    force_cfg = bool(getattr(cfg, "CITIZENS_FHCF_FORCE_CONFIG_TERMS", False))
    try:
        tmp = attach_fhcf_terms_for_losses(
            loss_df=citizens_gross,
            terms_df=terms_norm,                # <- already passed through normalize_fhcf_terms
            market_share_df=market_share_df,
            company_crosswalk_df=company_keys,
            allow_name_fallback=True,
            qa_strict=False,
        )
        # sanity: nonzero retention & limit?
        lim_ok = float(pd.to_numeric(tmp.get("LimitUSD", 0.0), errors="coerce").fillna(0.0).sum()) > 0.0
        ret_ok = float(pd.to_numeric(tmp.get("RetentionUSD", 0.0), errors="coerce").fillna(0.0).sum()) > 0.0
        if not force_cfg and lim_ok and ret_ok:
            terms_for_citizens = tmp[["Company","CoveragePct_norm","RetentionUSD","LimitUSD"]].copy()
    except Exception:
        pass

    # 2) Fallback to config -> normalize to get Retention/Limit
    if terms_for_citizens is None or force_cfg:
        fb = _citizens_terms_fallback_row(cfg)
        fb_norm = normalize_fhcf_terms(fb)
        terms_for_citizens = fb_norm[["Company","CoveragePct_norm","RetentionUSD","LimitUSD"]].copy()

    # 3) Pre-cap recovery (Citizens)
    citizens_precap = (
        apply_fhcf_recovery(loss_df=citizens_gross, terms_df=terms_for_citizens)
        [["Company","County","GrossWindLossUSD","RecoveryUSD"]]
        .rename(columns={"RecoveryUSD":"FHCF_RecoveryPreCapUSD"})
    )

    # Apply the single industry seasonal cap (unchanged)
    cap_limit = float(getattr(cfg, "FHCF_SEASONAL_INDUSTRY_CAP_USD",
                            getattr(cfg, "FHCF_SEASON_CAP", 17_000_000_000.0)))
    private_after_fhcf, citizens_wind_after_fhcf, fhcf_diag = _apply_industry_season_cap(
        private_precap, citizens_precap, cap_limit
    )

    # --- FHCF split totals (scalars) ---
    fhcf_recovery_private_usd  = numsum(private_after_fhcf.get("FHCF_RecoveryUSD"))
    fhcf_recovery_citizens_usd = numsum(citizens_wind_after_fhcf.get("FHCF_RecoveryUSD"))

    if DEBUG_PRINTS:
        dbg(f"[FHCF] pre_total={fhcf_diag['fhcf_total_precap_usd']:,.0f} "
            f"cap={fhcf_diag['fhcf_cap_usd']:,.0f} "
            f"scale={fhcf_diag['fhcf_scaling_factor']:.3f} "
            f"bind={fhcf_diag['fhcf_cap_binding']}")
        dbg(f"[CIT FHCF] pre-cap={numsum(citizens_precap.get('FHCF_RecoveryPreCapUSD')):,.0f} "
            f"post-cap={fhcf_recovery_citizens_usd:,.0f}")

   
        # --- CAT BOND RECOVERIES (after FHCF, before capital/defaults) ---
    try:
        # Load season catbonds & apply heuristics
        cat_csv = getattr(cfg, "CATBONDS_CSV", cfg.DATA_DIR / "catbonds_2024.csv")
        attach_mult = float(getattr(cfg, "CATBOND_DEFAULT_ATTACH_MULT", 1.0))
        exhaust_mult = float(getattr(cfg, "CATBOND_DEFAULT_EXH_MULT", 2.0))
        catbonds = load_catbond_table(cat_csv, season_year=2024,
                                    attach_mult=attach_mult, exhaust_mult=exhaust_mult)

        # Industry driver proxy: pre-FHCF insured wind (you already compute these)
        industry_pre = float(
            (locals().get("insured_private_wind_pre_usd", 0.0) or 0.0) +
            (locals().get("insured_citizens_wind_pre_usd", 0.0) or 0.0)
        )

        # Company keys for cedent mapping
        company_keys = pd.read_csv(cfg.DATA_DIR / "company_keys.csv", dtype={"StatEntityKey": str, "NAIC": str})

        cat_recov, cat_diag = apply_catbond_recovery(
            private_after_fhcf=private_after_fhcf,
            citizens_after_fhcf=citizens_wind_after_fhcf,
            catbonds=catbonds,
            market_share_df=market_share_df,
            company_keys_df=company_keys,
            industry_insured_wind_pre_fhcf_usd=industry_pre,
            issue_year=surplus_year,   # <- add this
        )

        # Subtract cat-bond recovery from NetWindUSD (company×county)
        if not cat_recov.empty:
            # Merge back into private/citizens frames
            def _apply_cb(df):
                out = df.merge(cat_recov, on=["Company","County"], how="left")
                out["CatBondRecoveryUSD"] = numseries(out["CatBondRecoveryUSD"])
                out["NetWindUSD"] = out["NetWindUSD"] - out["CatBondRecoveryUSD"]
                out["NetWindUSD"] = out["NetWindUSD"].clip(lower=0.0)
                return out.drop(columns=["CatBondRecoveryUSD"])

            private_after_fhcf = _apply_cb(private_after_fhcf)
            citizens_wind_after_fhcf = _apply_cb(citizens_wind_after_fhcf)
        else:
            cat_diag["catbond_payout_total"] = 0.0

        # Optionally print a one-line summary
        if DEBUG_PRINTS:
            dbg(f"[CATBOND] series_in_force={len(catbonds)} payout_total=${cat_diag['catbond_payout_total']:,.0f} hits={cat_diag['catbond_attachment_hits']}")

    except Exception as e:
        if DEBUG_PRINTS:
            dbg(f"[CATBOND] skipped due to error: {e}")
        cat_diag = {"bond_diag": pd.DataFrame(), "catbond_payout_total": 0.0, "catbond_attachment_hits": 0}

            
    # --- 7) Flood -> NFIP recovery -----------------------------------
    flood_nfip = pd.DataFrame()
    flood_nfip_view = pd.DataFrame()
    nfip_capital_summary = {"paid_total": 0.0, "pool_cap": 0.0, "pool_used": 0.0, "borrowed": 0.0}
    insured_flood_pre_usd = 0.0

    if do_flood:
        dbg("[NFIP] Flood branch enabled")

        # 7a) NFIP exposure (FIPS × FloodTIV). The loader should already align to FL and return FIPS + FloodTIV.
        if nfip_exposure_df is None:
            nfip_exposure_df = load_nfip_policy_coverage(
                path=str(getattr(cfg, "NFIP_POLICIES_CSV")),
                mode=getattr(cfg, "SAMPLING_MODE_NFIP_POLICIES", "FIXED_YEAR"),
                year=int(getattr(cfg, "NFIP_POLICY_YEAR", getattr(cfg, "FIXED_YEAR", 2024))),
                lookback_years=int(getattr(cfg, "EWA_WINDOW_YEARS", 5)),
                half_life=float(getattr(cfg, "EWA_HALF_LIFE_YEARS", 2.0)),
                county_xwalk=county_xwalk,
            )
        else:
            dbg(f"[NFIP] exposure provided by caller rows: {len(nfip_exposure_df)}")

        # Normalize schema just in case
        nfip_exposure_df = nfip_exposure_df.copy()
        if "FloodTIV" not in nfip_exposure_df.columns and "tiv_usd" in nfip_exposure_df.columns:
            nfip_exposure_df = nfip_exposure_df.rename(columns={"tiv_usd": "FloodTIV"})
        nfip_exposure_df["county_fips"] = (
            nfip_exposure_df["county_fips"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
        )
        nfip_exposure_df["FloodTIV"] = numseries(nfip_exposure_df["FloodTIV"])
        nfip_exposure_df = nfip_exposure_df.groupby("county_fips", as_index=False)["FloodTIV"].sum()

        # 7b) Water losses & insured-vs-uninsured split via penetration
        water = load_water_damage_scenario(storm_name)  # expects ['County','WaterDamageUSD']
        dbg("[NFIP] water damage rows:", len(water),
            "sum WaterDamageUSD:", numsum(water.get("WaterDamageUSD", 0.0)))

        pen_df = load_nfip_penetration(getattr(cfg, "NFIP_PENETRATION_CSV"), xwalk)
        
        # If penetration scenario is active, adjust NFIP penetration rates
        if scenario_config is not None and scenario_config.get("type") == "penetration":
            if "penetration" in scenario_diagnostics:
                flood_penetration_after = scenario_diagnostics["penetration"].get("flood_penetration_after", 0.30)
                flood_penetration_before = scenario_diagnostics["penetration"].get("flood_penetration_before", 0.30)
                
                if flood_penetration_before > 0:
                    # Scale penetration rates proportionally
                    scaling_factor = flood_penetration_after / flood_penetration_before
                    pen_df["NFIP_r_eff"] = (pen_df["NFIP_r_eff"] * scaling_factor).clip(upper=0.98)
                    dbg(f"[SCENARIO] Scaled NFIP penetration rates by {scaling_factor:.2f}x (before={flood_penetration_before:.1%}, after={flood_penetration_after:.1%})")

        carved_flood = carveout_flood_from_penetration(
            water.rename(columns={"WaterDamageUSD": "GrossFloodLossUSD"}),
            pen_df,
            xwalk,
        )
        # Ensure county_fips exists and is normalized for the join
        if "county_fips" not in carved_flood.columns:
            carved_flood = carved_flood.merge(
                xwalk[["County", "county_fips"]].drop_duplicates(), on="County", how="left"
            )
        carved_flood["county_fips"] = (
            carved_flood["county_fips"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
        )

        # 7c) Join on FIPS and cap insured flood by FloodTIV
        flood_join = carved_flood.merge(nfip_exposure_df[["county_fips", "FloodTIV"]], on="county_fips", how="left")
        flood_join["FloodTIV"] = numseries(flood_join.get("FloodTIV", 0.0))

        insured_raw = numseries(flood_join.get("InsuredFloodUSD", 0.0))
        gross_flood = numseries(flood_join.get("GrossFloodLossUSD", 0.0))

        flood_join["InsuredFloodCappedUSD"] = np.minimum(insured_raw, flood_join["FloodTIV"])
        flood_join["UnderinsuredFloodUSD"] = (insured_raw - flood_join["InsuredFloodCappedUSD"]).clip(lower=0.0)
        if "UninsuredFloodUSD" not in flood_join.columns:
            flood_join["UninsuredFloodUSD"] = (gross_flood - insured_raw).clip(lower=0.0)

        def _sum(x): 
            return float(numsum(x))

        dbg("[NFIP] before cap totals:",
            "FloodTIV_sum=", _sum(flood_join["FloodTIV"]),
            "InsuredFloodUSD_sum=", _sum(flood_join["InsuredFloodUSD"]),
            "GrossFloodLossUSD_sum=", _sum(flood_join["GrossFloodLossUSD"]))

        # 7d) Payout rate & payout dollars
        flood_nfip = flood_join.copy()
        payout_mode = str(getattr(cfg, "NFIP_PAYOUT_MODE", "unity")).lower()
        if payout_mode == "fixed":
            rate = float(getattr(cfg, "NFIP_PAYOUT_FIXED_RATE", 0.90))
            flood_nfip["NFIP_PayoutRate"] = max(0.0, min(1.0, rate))
        else:
            # 'unity' or any unknown mode defaults to full pay on insured/capped losses
            flood_nfip["NFIP_PayoutRate"] = 1.0

        flood_nfip["NFIP_PayoutUSD"] = (
            numseries(flood_nfip["InsuredFloodCappedUSD"])
            * numseries(flood_nfip["NFIP_PayoutRate"])
        )
        flood_nfip["PolicyholderShortfallUSD"] = (
            numseries(flood_nfip["InsuredFloodCappedUSD"])
            - numseries(flood_nfip["NFIP_PayoutUSD"])
        ).clip(lower=0.0)

        # 7e) Capital accounting: pool first, then borrowing
        nfip_paid_total = numsum(flood_nfip["NFIP_PayoutUSD"])
        nfip_pool_cap = float(
            getattr(cfg, "NFIP_NATIONAL_POOL_USD", getattr(cfg, "NFIP_CAPITAL_POOL_USD", 0.0))
        )
        nfip_pool_used_total = min(nfip_paid_total, nfip_pool_cap)
        nfip_borrowed_total = max(nfip_paid_total - nfip_pool_cap, 0.0)

        nfip_capital_summary = {
            "paid_total": nfip_paid_total,
            "pool_cap": nfip_pool_cap,
            "pool_used": nfip_pool_used_total,
            "borrowed": nfip_borrowed_total,
        }

        # Slim view (for reporting / export)
        _cols = [
            "County", "county_fips", "GrossFloodLossUSD",
            "InsuredFloodUSD", "InsuredFloodCappedUSD",
            "UnderinsuredFloodUSD", "UninsuredFloodUSD",
            "PolicyholderShortfallUSD", "NFIP_PayoutRate", "NFIP_PayoutUSD", "FloodTIV",
        ]
        flood_nfip_view = flood_nfip[[c for c in _cols if c in flood_nfip.columns]].copy()

        # Pre-payout insured flood scalar (useful upstream)
        insured_flood_pre_usd = numsum(carved_flood.get("InsuredFloodUSD"))

        # 7f) Florida premium-base perspective (depletion relative to FL contributions)
        try:
            fl_prem_year = int(getattr(cfg, "NFIP_PREMIUM_BASE_YEAR", getattr(cfg, "FIXED_YEAR", 2024)))
            nfip_fl_premium_base_usd = float(
                load_nfip_fl_premium_base(path=getattr(cfg, "NFIP_PREMIUM_BASE_CSV"), year=fl_prem_year)
            )
        except Exception as e:
            dbg(f"[NFIP] WARN: could not load FL premium base: {e}")
            nfip_fl_premium_base_usd = 0.0

        nfip_fl_premium_depletion_usd = float(min(nfip_paid_total, nfip_fl_premium_base_usd))


    # --- 8) Aggregate per company (sum modeled branches) ------------------------
    wind_company_net = (
        pd.concat(
            [
                private_after_fhcf[["Company", "NetWindUSD"]],
                citizens_wind_after_fhcf[["Company", "NetWindUSD"]],
            ],
            ignore_index=True,
        )
        .groupby("Company", as_index=False)
        .sum()
        .rename(columns={"NetWindUSD": "WindLossUSD"})
    )
    company_total_losses = wind_company_net.rename(columns={"WindLossUSD": "TotalLossUSD"}).copy()

    # --- 9) Load surplus & apply capital depletion with group contributions -----
    # Use scenario-adjusted surplus if available (e.g., from penetration scenario)
    if surplus_df_for_scenario is not None:
        surplus_df = surplus_df_for_scenario
        dbg(f"[SURPLUS] Using scenario-adjusted surplus (Total: ${surplus_df['SurplusUSD'].sum()/1e9:.2f}B)")
    else:
        surplus_df = load_surplus_data_with_groups(
            path=getattr(cfg, "SURPLUS_FILE"),
            year=int(surplus_year) if surplus_year is not None else int(getattr(cfg, "CITIZENS_CAPITAL_YEAR", 2024)),
        )
        dbg(f"[SURPLUS] Using baseline surplus (Total: ${surplus_df['SurplusUSD'].sum()/1e9:.2f}B)")

    capital_post = apply_losses_to_surplus(
        surplus_df=surplus_df[['Company', 'StatEntityKey', 'SurplusUSD']],
        losses_df=company_total_losses[["Company", "TotalLossUSD"]],
        rbc_df=None,
        sample=None,
        cov=None,
        rng=rng,
        rbc_affects_ruinflag=False,
    )

    capital_with_groups = apply_group_capital_contributions(
        capital_post=capital_post,
        surplus_df=surplus_df,
        contribution_rate_range=getattr(cfg, "GROUP_CONTRIBUTION_RANGE", (0.0, 0.20)),
        rng=rng,
        eligibility_threshold=group_support_eligibility_threshold,
    )

    # pick the column for "post" safely
    end_candidates = ["AdjustedSurplusUSD", "EndingSurplusUSD", "EndingSurplus"]
    endcol = next((c for c in end_candidates if c in capital_with_groups.columns), None)

    # defaults BEFORE group support (count rows with negative ending surplus)
    s_pre = numseries(capital_post["EndingSurplusUSD"]) if "EndingSurplusUSD" in capital_post.columns else numseries([])
    defaults_pre = int((s_pre < 0).sum())

    # defaults AFTER group support
    if endcol:
        s_post = numseries(capital_with_groups[endcol])
        defaults_post = int((s_post < 0).sum())
    else:
        defaults_post = 0


    if DEBUG_PRINTS and "GroupContributionUSD" in capital_with_groups.columns:
        donors = int((capital_with_groups["GroupContributionUSD"] < 0).sum())
        recips = int((capital_with_groups["GroupContributionUSD"] > 0).sum())
        dbg(f"[group] contrib_rate~{capital_with_groups.get('GroupRateApplied', pd.Series([0])).max():.2f} "
            f"groups={capital_with_groups.get('GroupID', pd.Series([])).nunique()} recips={recips} donors={donors} "
            f"total={capital_with_groups['GroupContributionUSD'].sum():,.0f}")

    # Ensure post-group table carries StatEntityKey
    if "StatEntityKey" not in capital_with_groups.columns:
        capital_with_groups = capital_with_groups.merge(
            surplus_df[["Company", "StatEntityKey"]], on="Company", how="left"
        )

    # Defaults view (only defaulters), carry StatEntityKey
    defaults_view = capital_with_groups[capital_with_groups["DefaultFlag"]].merge(
        surplus_df[["Company", "StatEntityKey"]], on="Company", how="left"
    )

    # --- Diagnostics: group shortfalls (scenario-level) ---
    if isinstance(capital_with_groups, pd.DataFrame) and not capital_with_groups.empty:
        # one value per group (same value on each member row), so use max() by group
        grp_short = (capital_with_groups
                    .groupby("GroupID", dropna=False)["DiagGroupShortfallUSD"]
                    .max()
                    .fillna(0.0))
        diag_group_shortfall_total_usd = float(grp_short.sum())
        diag_groups_with_shortfall = int((grp_short > 0).sum())
        diag_groups_fully_funded_share = float(((grp_short == 0).mean()) if len(grp_short) else 0.0)
    else:
        diag_group_shortfall_total_usd = 0.0
        diag_groups_with_shortfall = 0
        diag_groups_fully_funded_share = 0.0


    # --- 10) FIGA assessments (keyed by StatEntityKey; exclude Citizens) -------
    try:
        premium_base_df = load_private_premium_base_from_market_share_xlsx(
            path=getattr(cfg, "MARKET_SHARE_XLSX"),
            year=getattr(cfg, "MARKET_SHARE_YEAR", 2024),
        )
    except Exception:
        premium_base_df = None

    if premium_base_df is not None and not premium_base_df.empty:
        base = premium_base_df.copy()
        base["StatEntityKey"] = base["StatEntityKey"].astype(str).str.strip()
        base["Company"] = base["Company"].astype(str).str.strip()

        # Exclude Citizens by explicit key if configured; else by display name
        citizens_key = getattr(cfg, "CITIZENS_STATKEY", None)
        if citizens_key:
            base = base[base["StatEntityKey"] != str(citizens_key)].copy()
        else:
            base = base[~base["Company"].str.contains("citizens", case=False, na=False)].copy()

        # Survivors only (corrected): use full capital_with_groups, not defaults_view
        survivors_only = bool(getattr(cfg, "FIGA_SURVIVORS_ONLY", True))
        if survivors_only:
            surv_keys = set(
                capital_with_groups.loc[~capital_with_groups["DefaultFlag"].astype(bool), "StatEntityKey"]
                                   .dropna().astype(str)
            )
            if surv_keys:
                base = base[base["StatEntityKey"].astype(str).isin(surv_keys)].copy()

        # Map key -> display name (first seen), then group base by key
        name_by_key = (
            base.groupby("StatEntityKey", as_index=False)
                .agg({"Company": "first"})
                .set_index("StatEntityKey")["Company"]
                .to_dict()
        )
        base_group = base.groupby("StatEntityKey", as_index=False)["PremiumUSD"].sum()

        # Apply penetration scenario scaling to FIGA premium base
        if "penetration" in scenario_diagnostics:
            private_scale = scenario_diagnostics["penetration"].get("private_premium_scale_factor", 1.0)
            if abs(private_scale - 1.0) > 0.001:
                dbg(f"[FIGA PREMIUM BASE] Scaling by {private_scale:.3f}x for penetration scenario")
                base_group["PremiumUSD"] *= private_scale
                # Also update total_base_usd calculation below
        
        # Compute capped rates
        # rows where entity defaulted
        mask = capital_with_groups["DefaultFlag"].astype(bool)

        # work on a Series so clip/abs/sum are valid, then cast to float
        s = numseries(capital_with_groups.loc[mask, endcol])
        deficit = float(s.clip(upper=0).abs().sum())


        if deficit <= 0 or base_group.empty:
            figa_result = {"deficit": deficit, "normal_rate": 0.0, "emergency_rate": 0.0,
                           "assessments": pd.DataFrame(), "residual_deficit": max(deficit, 0.0)}
        else:
            total_base = float(base_group["PremiumUSD"].sum())
            cap_normal    = float(getattr(cfg, "FIGA_CAP_NORMAL", 0.02))
            cap_emergency = float(getattr(cfg, "FIGA_CAP_EMERGENCY", 0.04))
            rate_needed   = deficit / total_base if total_base > 0 else np.inf
            normal_rate   = min(rate_needed, cap_normal)
            remaining     = max(rate_needed - normal_rate, 0.0)
            emergency_rate= min(remaining, cap_emergency)
            total_rate    = normal_rate + emergency_rate

            assess = base_group.rename(columns={"StatEntityKey": "Company"}).copy()
            assess["FIGA_AssessmentRate"] = total_rate
            assess["FIGA_AssessmentUSD"]  = assess["PremiumUSD"] * total_rate
            assess["StatEntityKey"] = assess["Company"]
            assess["Company"] = assess["StatEntityKey"].map(name_by_key).fillna(assess["StatEntityKey"])
            assess = assess[["Company","StatEntityKey","PremiumUSD","FIGA_AssessmentRate","FIGA_AssessmentUSD"]]

            figa_result = {
                "deficit": deficit,
                "normal_rate": normal_rate,
                "emergency_rate": emergency_rate,
                "assessments": assess,
                "residual_deficit": max(deficit - float(assess["FIGA_AssessmentUSD"].sum()), 0.0)
            }
    else:
        figa_result = {"deficit": 0.0, "normal_rate": 0.0, "emergency_rate": 0.0,
                       "assessments": pd.DataFrame(), "residual_deficit": 0.0}
        
    # --- Metrics: defaulted premium base % & largest single-entity deficit ----------
    # Ensure StatEntityKey is string for joins
    if "StatEntityKey" in capital_with_groups.columns:
        cap_keys = capital_with_groups["StatEntityKey"].astype(str)
    else:
        # Map from surplus_df if needed (you already did this earlier)
        capital_with_groups = capital_with_groups.merge(
            surplus_df[["Company", "StatEntityKey"]], on="Company", how="left"
        )
        cap_keys = capital_with_groups["StatEntityKey"].astype(str)

    # Pick ending surplus column (same logic you use above)
    endcol = next(
        (c for c in ["AdjustedSurplusUSD", "EndingSurplusUSD", "EndingSurplus"] if c in capital_with_groups.columns),
        None
    )

    # Defaulted set (prefer DefaultFlag; else use endcol < 0 fallback)
    if "DefaultFlag" in capital_with_groups.columns:
        default_mask = capital_with_groups["DefaultFlag"].astype(bool)
    elif endcol is not None:
        default_mask = (numseries(capital_with_groups[endcol]) < 0)
    else:
        default_mask = pd.Series(False, index=capital_with_groups.index)

    defaulting_keys = set(cap_keys[default_mask].dropna().astype(str))

    # Statewide private premium base (exclude Citizens by construction of loader)
    total_base_usd = numsum(premium_base_df["PremiumUSD"])

    # Premium base held by defaulting entities
    if "StatEntityKey" in premium_base_df.columns and defaulting_keys:
        def_base_usd = float(
            numsum(
                premium_base_df.loc[premium_base_df["StatEntityKey"].astype(str).isin(defaulting_keys), "PremiumUSD"],
            )
        )
    else:
        def_base_usd = 0.0

    defaulted_premium_base_pct = (100.0 * def_base_usd / total_base_usd) if total_base_usd > 0.0 else 0.0

    # Largest single-entity deficit (absolute USD)
    if endcol is not None:
        ending = numseries(capital_with_groups[endcol])
        largest_entity_deficit_usd = float((-ending).clip(lower=0.0).max())
    else:
        largest_entity_deficit_usd = 0.0


    # --- 11) Citizens capital hit + assessments --------------------------------
    net_wind = numsum(citizens_wind_after_fhcf.get("NetWindUSD"))

    if citizens_capital_row is None:
        try:
            citizens_capital_row = load_citizens_capital_row_from_csv(
                path=getattr(cfg, "CITIZENS_CAPITAL_CSV", None),
                year=getattr(cfg, "CITIZENS_CAPITAL_YEAR", None),
                mode=getattr(cfg, "SAMPLING_MODE_CAPITAL", "FIXED_YEAR"),
                lookback_years=int(getattr(cfg, "EWA_WINDOW_YEARS", 5)),
                half_life=float(getattr(cfg, "EWA_HALF_LIFE_YEARS", 2.0)),
                surplus_field_out="projected_year_end_surplus_usd",
                capital_is_thousands=bool(getattr(cfg, "CITIZENS_CAPITAL_IS_THOUSANDS", False)),
            )
        except Exception:
            citizens_capital_row = {
                "projected_year_end_surplus_usd": np.nan,
                "year": getattr(cfg, "CITIZENS_CAPITAL_YEAR", None),
            }

    citizens_capital_summary = apply_citizens_capital_hit(
        citizens_net_df=citizens_wind_after_fhcf,     # must include 'NetWindUSD'
        citizens_capital_row=citizens_capital_row,
        surplus_field="projected_year_end_surplus_usd",
    )

    # --- Citizens premium bases -------------------------------------------------
    # Citizens' own premium base (Tier 1 denominator)
    try:
        citizens_premium_base = float(
            load_citizens_premium_base(
                path_csv=citizens_csv_path or cfg.CITIZENS_COUNTY_CSV,
                as_of=citizens_as_of or cfg.CITIZENS_AS_OF,
                include_products=citizens_products or cfg.CITIZENS_PRODUCTS,
            )
        )
    except Exception:
        citizens_premium_base = float(getattr(cfg, "CITIZENS_PREMIUM_BASE_DEFAULT", 0.0))

    # Private premium base (from market share workbook)
    private_premium_base = float(
        numsum(market_share_df.get("PremiumUSD", 0.0))
    )
    
    # Apply penetration scenario premium base scaling if available
    if "penetration" in scenario_diagnostics:
        citizens_scale = scenario_diagnostics["penetration"].get("citizens_premium_scale_factor", 1.0)
        private_scale = scenario_diagnostics["penetration"].get("private_premium_scale_factor", 1.0)
        
        if abs(citizens_scale - 1.0) > 0.001 or abs(private_scale - 1.0) > 0.001:
            dbg(f"[PREMIUM BASE] Scaling Citizens: {citizens_scale:.3f}x, Private: {private_scale:.3f}x")
            citizens_premium_base *= citizens_scale
            private_premium_base *= private_scale

    # Statewide property base (Tier 2 denominator)
    statewide_property_premium_base = citizens_premium_base + private_premium_base

    # --- Deficit to be funded by assessments (pre-assessment) -------------------
    ending_surplus = float(citizens_capital_summary.get("citizens_ending_surplus_usd", np.nan))
    if np.isfinite(ending_surplus):
        citizens_deficit = max(-ending_surplus, 0.0)
    else:
        # fallback if ending surplus missing: use starting surplus if available, else net_wind
        start_surplus = float(citizens_capital_summary.get("citizens_start_surplus_usd", 0.0))
        citizens_deficit = max(net_wind - start_surplus, 0.0) if start_surplus > 0 else max(net_wind, 0.0)

    # --- Rates, capacities, amounts --------------------------------------------
    tier1_cap_rate = float(getattr(cfg, "CITIZENS_TIER1_RATE_MAX", getattr(cfg, "CIT_TIER1_CAP", 0.15)))  # 15%
    tier2_cap_rate = float(getattr(cfg, "CITIZENS_TIER2_RATE_MAX", getattr(cfg, "CIT_TIER2_CAP", 0.10)))  # 10%

    tier1_capacity_usd = tier1_cap_rate * citizens_premium_base
    tier2_capacity_usd = tier2_cap_rate * statewide_property_premium_base

    tier1_amount = min(citizens_deficit, tier1_capacity_usd)
    remaining    = max(citizens_deficit - tier1_amount, 0.0)
    tier2_amount = min(remaining, tier2_capacity_usd)

    residual_deficit = max(citizens_deficit - tier1_amount - tier2_amount, 0.0)

    tier1_rate_applied = (tier1_amount / citizens_premium_base) if citizens_premium_base > 0 else 0.0
    tier2_rate_applied = (tier2_amount / statewide_property_premium_base) if statewide_property_premium_base > 0 else 0.0

    citizens_assessments = {
        "tier1_capacity": tier1_capacity_usd,
        "tier1_rate": tier1_rate_applied,
        "tier1_amount": tier1_amount,
        "tier2_capacity": tier2_capacity_usd,
        "tier2_rate": tier2_rate_applied,
        "tier2_amount": tier2_amount,
        "tier1_collected": tier1_amount,   # amounts are already capped, so collected == amount
        "tier2_collected": tier2_amount,
        "residual_deficit": residual_deficit,

        # helpful QA fields
        "citizens_premium_base_usd": citizens_premium_base,
        "statewide_property_premium_base_usd": statewide_property_premium_base,
    }

    if DEBUG_PRINTS:
        dbg(f"[CIT Assess] deficit={citizens_deficit:,.0f}  "
            f"T1(base={citizens_premium_base:,.0f}, cap={tier1_capacity_usd:,.0f}, rate={tier1_rate_applied:.2%})  "
            f"T2(base={statewide_property_premium_base:,.0f}, cap={tier2_capacity_usd:,.0f}, rate={tier2_rate_applied:.2%})  "
            f"residual={residual_deficit:,.0f}")


    dbg('[NFIP] returning flood_nfip_recovery columns:', list(flood_nfip.columns))

    return {
        "insured_private_wind_pre_usd":  float(insured_private_wind_pre_usd),
        "insured_citizens_wind_pre_usd": float(insured_citizens_wind_pre_usd),
        "insured_flood_pre_usd":         float(insured_flood_pre_usd),

        # Wind (pre -> post FHCF)
        "private_allocated_wind":  private_alloc,
        "private_wind_after_fhcf": private_after_fhcf,
        "citizens_wind_after_fhcf": citizens_wind_after_fhcf,

        # Flood / NFIP
        "flood_nfip_recovery":   flood_nfip_view,
        "nfip_capital_summary":  nfip_capital_summary,
        "nfip_claims_paid_total": nfip_capital_summary["paid_total"],
        "nfip_pool_used_total":   nfip_capital_summary["pool_used"],
        "nfip_borrowed_total":    nfip_capital_summary["borrowed"],
        "nfip_fl_premium_base_usd":      nfip_fl_premium_base_usd,
        "nfip_fl_premium_depletion_usd": nfip_fl_premium_depletion_usd,

        # Company capital / defaults / assessments (unchanged)
        "company_total_losses": company_total_losses,
        "capital_post":         capital_post,
        "capital_with_groups":  capital_with_groups,
        "defaults_pre":         defaults_pre,
        "defaults_post":        defaults_post,
        "diag_group_shortfall_total_usd": diag_group_shortfall_total_usd,
        "diag_groups_with_shortfall": diag_groups_with_shortfall,
        "diag_groups_fully_funded_share": diag_groups_fully_funded_share,
        "carveout_wind_county": carved,
        "defaults_view":        defaults_view,
        "figa_result":          figa_result,
        "citizens_capital_summary": citizens_capital_summary,
        "citizens_assessments":     citizens_assessments,
        "defaulted_premium_base_pct": defaulted_premium_base_pct,
        "largest_entity_deficit_usd": largest_entity_deficit_usd,

        # FHCF diagnostics (used by mc_run_events.py)
        **fhcf_diag,
        "fhcf_recovery_private_usd":  fhcf_recovery_private_usd,
        "fhcf_recovery_citizens_usd": fhcf_recovery_citizens_usd,

        # Cat bond diagnostics
        "catbond_diag": cat_diag,
        "catbond_payout_total": cat_diag.get("catbond_payout_total", 0.0),
        "catbond_hits": cat_diag.get("catbond_attachment_hits", 0),
        "catbond_limit_in_force_usd":    float(cat_diag.get("catbond_limit_in_force_usd", 0.0)),
        "catbond_issue_volume_year_usd": float(cat_diag.get("catbond_issue_volume_year_usd", 0.0)),
        
        # Scenario diagnostics (empty dict if no scenario applied)
        "scenario_diagnostics": scenario_diagnostics,
    }


# =============================================================================
# Monte Carlo convenience wrapper
# =============================================================================

def run_monte_carlo(n_iter: int, **kwargs) -> pd.DataFrame:
    """Run repeated scenarios, return per-company ruin probabilities."""
    records = []
    for i in range(n_iter):
        res = run_one_scenario(**kwargs)
        rec = res["capital_post"][["Company", "RuinFlag"]].assign(iter=i)
        records.append(rec)
    df = pd.concat(records, ignore_index=True)
    pr = df.groupby("Company", as_index=False)["RuinFlag"].mean().rename(columns={"RuinFlag": "RuinProb"})
    return pr


if __name__ == "__main__":
    pass
