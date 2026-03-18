"""
penetration.py — Insurance penetration increase scenarios
---------------------------------------------------------

Models improved insurance take-up through policy interventions:
- NFIP expansion (flood penetration increase)
- Citizens depopulation (private market growth)
- Private market expansion (new policies)

Design Principles:
1. **Positive adaptation**: More coverage → better risk pooling
2. **Program-specific**: Model actual policy levers (NFIP, Citizens, private)
3. **Geographic targeting**: Focus increases on high-risk coastal areas
4. **Capital scaling**: Adjust surplus for larger/smaller books

Public API:
- apply_penetration_increase_scenario: Transform exposure and NFIP data
- adjust_surplus_for_penetration: Scale capital for exposure changes
- PENETRATION_INCREASE_PRESETS: Pre-configured scenarios (MODERATE, MAJOR, EXTREME)
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd

# =============================================================================
# Preset Scenario Definitions
# =============================================================================

PENETRATION_INCREASE_PRESETS = {
    "BASELINE": {
        "wind_penetration_target": 0.40,  # Current ~40% (Beta(4,6) mean from config)
        "flood_penetration_target": 0.11,  # Current ~11% (FEMA NFIP data 2025)
        "citizens_share_target": 0.15,  # Current ~15%
        "description": "No penetration increase (2024 baseline)",
    },
    "MODERATE": {
        "wind_penetration_target": 0.50,  # +10pp
        "flood_penetration_target": 0.20,  # +9pp (non-SFHA focus: ~3× growth)
        "citizens_share_target": 0.12,  # -3pp (depopulation)
        "geographic_targeting": True,
        "coastal_focus_factor": 1.3,  # Coastal counties +30% vs. inland
        "sfha_scaling_mode": "aggressive_nonsfha",  # Focus on non-SFHA areas (currently ~5%)
        "surplus_adjustment": "proportional",  # Scale capital with exposure growth
        "description": "Incremental improvements: targeted mandatory coverage + NFIP affordability (2030)",
    },
    "MAJOR": {
        "wind_penetration_target": 0.60,  # +20pp
        "flood_penetration_target": 0.30,  # +19pp (balanced growth)
        "citizens_share_target": 0.08,  # -7pp
        "geographic_targeting": True,
        "coastal_focus_factor": 1.5,  # Coastal +50% vs. inland
        "sfha_scaling_mode": "aggressive_nonsfha",  # Continue non-SFHA focus
        "surplus_adjustment": "proportional",  # Scale capital with exposure growth
        "description": "Aggressive policy push: statewide mandatory coverage + building code requirements + subsidies (2040)",
    },
    "EXTREME": {
        "wind_penetration_target": 0.70,  # +30pp
        "flood_penetration_target": 0.50,  # +39pp (near-universal)
        "citizens_share_target": 0.05,  # -10pp
        "geographic_targeting": True,
        "coastal_focus_factor": 2.0,  # Coastal +100% vs. inland
        "sfha_scaling_mode": "balanced",  # Balanced growth across all zones
        "surplus_adjustment": "proportional",  # Scale capital with exposure growth
        "description": "Near-universal coverage: mandatory purchase + deep subsidies + enforcement (2050)",
    },
}


# =============================================================================
# Main Scenario Application
# =============================================================================

def apply_penetration_increase_scenario(
    private_exp: pd.DataFrame,
    citizens_exp: pd.DataFrame,
    nfip_df: Optional[pd.DataFrame] = None,
    scenario: str = "MODERATE",
    coastal_counties: Optional[set[str]] = None,
    penetration_df: Optional[pd.DataFrame] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict]:
    """
    Apply penetration increase scenario to exposure DataFrames.
    
    Models three policy interventions:
    1. NFIP expansion (flood penetration increase with SFHA-aware targeting)
    2. Citizens depopulation (transfer to private market)
    3. Private market expansion (new wind policies)
    
    Parameters
    ----------
    private_exp : pd.DataFrame
        Private insurer exposure with columns ['Company','County','TIV','TIV_sampled']
    citizens_exp : pd.DataFrame
        Citizens exposure with columns ['Company','County','TIV','TIV_sampled']
    nfip_df : pd.DataFrame, optional
        NFIP policy data with columns ['County', 'Policies', 'Coverage', 'county_fips']
    scenario : str, default "MODERATE"
        One of: "BASELINE", "MODERATE", "MAJOR", "EXTREME"
    coastal_counties : set of str, optional
        Set of coastal county names for geographic targeting
    penetration_df : pd.DataFrame, optional
        FEMA NFIP penetration data with columns ['county_fips', 'NFIP_r_sfha', 'NFIP_r_non', 'NFIP_s_sfha']
        from load_nfip_penetration(). Used for SFHA-aware flood scaling.
    rng : np.random.Generator, optional
        Random number generator for stochastic allocation
        
    Returns
    -------
    private_exp_new : pd.DataFrame
        Modified private exposure (increased)
    citizens_exp_new : pd.DataFrame
        Modified Citizens exposure (decreased via depopulation)
    nfip_df_new : pd.DataFrame or None
        Modified NFIP data (increased flood penetration)
    diagnostics : dict
        Scenario metrics: exposure changes, policy counts, penetration rates
    """
    if scenario not in PENETRATION_INCREASE_PRESETS:
        raise ValueError(f"Unknown scenario '{scenario}'. Must be one of {list(PENETRATION_INCREASE_PRESETS.keys())}")
    
    config = PENETRATION_INCREASE_PRESETS[scenario]
    
    if scenario == "BASELINE":
        # No changes
        return private_exp.copy(), citizens_exp.copy(), nfip_df.copy() if nfip_df is not None else None, {
            "scenario": scenario,
            "wind_penetration_before": 0.40,
            "wind_penetration_after": 0.40,
            "flood_penetration_before": 0.11,
            "flood_penetration_after": 0.11,
            "citizens_share_before": 0.15,
            "citizens_share_after": 0.15,
        }
    
    rng = rng or np.random.default_rng()
    coastal_counties = coastal_counties or set()
    
    # Calculate baseline state
    total_tiv_before = private_exp["TIV"].sum() + citizens_exp["TIV"].sum()
    citizens_tiv_before = citizens_exp["TIV"].sum()
    citizens_share_before = citizens_tiv_before / total_tiv_before if total_tiv_before > 0 else 0.15
    
    # Baseline penetration rates (from config Beta(4,6) and FEMA NFIP data)
    baseline_wind_penetration = 0.40  # Beta(4,6) mean = 40% insured
    baseline_flood_penetration = 0.11  # FEMA NFIP 2025 FL average: 10.6%
    
    # --- 1. Citizens Depopulation (Citizens → Private) ---
    target_citizens_share = config["citizens_share_target"]
    private_new, citizens_new, depopulation_amt = _apply_citizens_depopulation(
        private_exp=private_exp,
        citizens_exp=citizens_exp,
        target_citizens_share=target_citizens_share,
        coastal_counties=coastal_counties,
        rng=rng,
    )
    
    # --- 2. Wind Penetration Increase (New Private Policies) ---
    wind_penetration_target = config["wind_penetration_target"]
    wind_scaling_factor = wind_penetration_target / baseline_wind_penetration
    
    if config.get("geographic_targeting", False):
        # Apply differential scaling: coastal gets more increase than inland
        # coastal_focus_factor determines the ratio between coastal and inland increases
        # Both are scaled to average to the target penetration
        coastal_bias = config.get("coastal_focus_factor", 1.3)
        
        # Coastal gets coastal_bias × wind_scaling_factor
        # Inland gets 1.0 × wind_scaling_factor (baseline increase)
        coastal_multiplier = wind_scaling_factor * coastal_bias
        inland_multiplier = wind_scaling_factor
        
        private_new = _apply_geographic_scaling(
            private_new,
            coastal_counties=coastal_counties,
            coastal_factor=coastal_multiplier,
            inland_factor=inland_multiplier,
        )
        citizens_new = _apply_geographic_scaling(
            citizens_new,
            coastal_counties=coastal_counties,
            coastal_factor=coastal_multiplier,
            inland_factor=inland_multiplier,
        )
    else:
        # Uniform scaling
        private_new["TIV"] *= wind_scaling_factor
        citizens_new["TIV"] *= wind_scaling_factor
        if "TIV_sampled" in private_new.columns:
            private_new["TIV_sampled"] *= wind_scaling_factor
        if "TIV_sampled" in citizens_new.columns:
            citizens_new["TIV_sampled"] *= wind_scaling_factor
    
    # --- 3. NFIP Flood Penetration Increase (SFHA-Aware) ---
    nfip_new = None
    nfip_increase_amt = 0.0
    if nfip_df is not None:
        flood_penetration_target = config["flood_penetration_target"]
        
        # Use SFHA-aware scaling if penetration data available
        if penetration_df is not None and "county_fips" in nfip_df.columns:
            nfip_new = _apply_sfha_aware_flood_scaling(
                nfip_df=nfip_df,
                penetration_df=penetration_df,
                baseline_overall=baseline_flood_penetration,
                target_overall=flood_penetration_target,
                config=config,
            )
        else:
            # Fallback to coastal/inland targeting if SFHA data unavailable
            flood_scaling_factor = flood_penetration_target / baseline_flood_penetration
            nfip_new = nfip_df.copy()
            
            if config.get("geographic_targeting", False) and "County" in nfip_new.columns:
                # Higher increase in coastal flood-prone areas
                coastal_factor = config.get("coastal_focus_factor", 1.3)
                inland_factor = 1.0 + (flood_scaling_factor - 1.0) * 0.5
                
                for idx, row in nfip_new.iterrows():
                    county = row.get("County", "")
                    if county in coastal_counties:
                        factor = coastal_factor * (flood_scaling_factor - 1.0) + 1.0
                    else:
                        factor = inland_factor
                    
                    for col in ["Policies", "Coverage", "Premium"]:
                        if col in nfip_new.columns and pd.notna(nfip_new.at[idx, col]):
                            nfip_new.at[idx, col] *= factor
            else:
                # Uniform scaling
                for col in ["Policies", "Coverage", "Premium"]:
                    if col in nfip_new.columns:
                        nfip_new[col] *= flood_scaling_factor
        
        nfip_increase_amt = (
            nfip_new["Coverage"].sum() - nfip_df["Coverage"].sum()
            if "Coverage" in nfip_new.columns
            else 0.0
        )
    
    # Calculate final state
    total_tiv_after = private_new["TIV"].sum() + citizens_new["TIV"].sum()
    citizens_tiv_after = citizens_new["TIV"].sum()
    citizens_share_after = citizens_tiv_after / total_tiv_after if total_tiv_after > 0 else 0.0
    
    wind_penetration_after = wind_penetration_target
    flood_penetration_after = config["flood_penetration_target"]
    
    # Calculate total new coverage
    wind_increase_amt = total_tiv_after - total_tiv_before
    
    # Calculate private exposure change percentage for surplus adjustment
    private_tiv_before = private_exp["TIV"].sum()
    private_tiv_after = private_new["TIV"].sum()
    private_exposure_change_pct = ((private_tiv_after / private_tiv_before) - 1.0) * 100.0 if private_tiv_before > 0 else 0.0
    
    # Calculate Citizens exposure change for premium base scaling
    citizens_tiv_before = citizens_exp["TIV"].sum()
    citizens_tiv_after_val = citizens_new["TIV"].sum()
    citizens_exposure_change_pct = ((citizens_tiv_after_val / citizens_tiv_before) - 1.0) * 100.0 if citizens_tiv_before > 0 else 0.0
    
    # Calculate overall wind market change for statewide premium base
    total_exposure_change_pct = ((total_tiv_after / total_tiv_before) - 1.0) * 100.0 if total_tiv_before > 0 else 0.0
    
    diagnostics = {
        "scenario": scenario,
        "description": config["description"],
        
        # Wind penetration
        "wind_penetration_before": baseline_wind_penetration,
        "wind_penetration_after": wind_penetration_after,
        "wind_penetration_increase_pp": (wind_penetration_after - baseline_wind_penetration) * 100,
        
        # Flood penetration
        "flood_penetration_before": baseline_flood_penetration,
        "flood_penetration_after": flood_penetration_after,
        "flood_penetration_increase_pp": (flood_penetration_after - baseline_flood_penetration) * 100,
        
        # Citizens share
        "citizens_share_before": citizens_share_before,
        "citizens_share_after": citizens_share_after,
        "citizens_share_change_pp": (citizens_share_after - citizens_share_before) * 100,
        
        # TIV changes
        "total_wind_tiv_before_usd": total_tiv_before,
        "total_wind_tiv_after_usd": total_tiv_after,
        "wind_tiv_increase_usd": wind_increase_amt,
        "citizens_depopulation_usd": depopulation_amt,
        
        # Private exposure change (for surplus adjustment)
        "private_tiv_before_usd": private_tiv_before,
        "private_tiv_after_usd": private_tiv_after,
        "private_exposure_change_pct": private_exposure_change_pct,
        
        # Citizens exposure change (for premium base scaling)
        "citizens_tiv_before_usd": citizens_tiv_before,
        "citizens_tiv_after_usd": citizens_tiv_after_val,
        "citizens_exposure_change_pct": citizens_exposure_change_pct,
        
        # Overall market change (for statewide premium base scaling)
        "total_exposure_change_pct": total_exposure_change_pct,
        
        # Premium base scaling factors (multiplicative)
        "private_premium_scale_factor": 1.0 + (private_exposure_change_pct / 100.0),
        "citizens_premium_scale_factor": 1.0 + (citizens_exposure_change_pct / 100.0),
        "total_premium_scale_factor": 1.0 + (total_exposure_change_pct / 100.0),
        
        # NFIP changes
        "nfip_coverage_increase_usd": nfip_increase_amt,
        
        # Policy count estimates (rough)
        "estimated_new_wind_policies": int(wind_increase_amt / 250_000),  # Assume $250K avg TIV
        "estimated_new_flood_policies": int(nfip_increase_amt / 250_000) if nfip_increase_amt > 0 else 0,
    }
    
    return private_new, citizens_new, nfip_new, diagnostics


# =============================================================================
# Helper Functions
# =============================================================================

def _apply_sfha_aware_flood_scaling(
    nfip_df: pd.DataFrame,
    penetration_df: pd.DataFrame,
    baseline_overall: float,
    target_overall: float,
    config: dict,
) -> pd.DataFrame:
    """
    Apply SFHA-aware flood penetration scaling using FEMA flood zone data.
    
    Strategy:
    1. Focus increases on non-SFHA areas (low current penetration, high growth potential)
    2. Moderate increases in SFHA (already ~35% due to mandatory purchase)
    3. Use actual county-specific SFHA shares from FEMA data
    
    Parameters
    ----------
    nfip_df : pd.DataFrame
        NFIP policy data with columns ['County', 'Policies', 'Coverage', 'Premium']
    penetration_df : pd.DataFrame
        FEMA penetration data with columns ['county_fips', 'NFIP_r_sfha', 'NFIP_r_non', 'NFIP_s_sfha']
    baseline_overall : float
        Current statewide average penetration (~0.11)
    target_overall : float
        Target statewide average penetration (e.g., 0.20, 0.30)
    config : dict
        Scenario configuration with optional 'sfha_scaling_mode' parameter
        
    Returns
    -------
    nfip_new : pd.DataFrame
        Modified NFIP data with scaled policies/coverage/premium
        
    Notes
    -----
    SFHA scaling modes:
    - "aggressive_nonsfha" (default): 3× increase in non-SFHA, 1.2× in SFHA
    - "balanced": 2× increase in non-SFHA, 1.5× in SFHA  
    - "uniform": Same scaling factor everywhere (simple multiplier)
    """
    nfip_new = nfip_df.copy()
    
    # Try to merge penetration data if available
    if "county_fips" not in nfip_new.columns:
        # Fall back to simple uniform scaling
        scaling_factor = target_overall / baseline_overall
        for col in ["Policies", "Coverage", "Premium"]:
            if col in nfip_new.columns:
                nfip_new[col] *= scaling_factor
        return nfip_new
    
    # Merge with FEMA penetration data to get SFHA info
    nfip_new = nfip_new.merge(
        penetration_df[["county_fips", "NFIP_r_sfha", "NFIP_r_non", "NFIP_s_sfha"]],
        on="county_fips",
        how="left"
    )
    
    # Get scaling mode
    mode = config.get("sfha_scaling_mode", "aggressive_nonsfha")
    overall_scaling = target_overall / baseline_overall
    
    for idx, row in nfip_new.iterrows():
        s_sfha = row.get("NFIP_s_sfha", 0.0)
        
        if pd.isna(s_sfha) or mode == "uniform":
            # Uniform scaling
            factor = overall_scaling
        else:
            # SFHA-aware scaling
            if mode == "aggressive_nonsfha":
                # Focus on non-SFHA (low current penetration = high potential)
                sfha_mult = 1.2  # Modest increase in SFHA (already ~35%)
                non_sfha_mult = min(overall_scaling * 3, 5.0)  # Aggressive in non-SFHA (currently ~5%)
            elif mode == "balanced":
                # Balanced growth
                sfha_mult = 1.5
                non_sfha_mult = overall_scaling * 2
            else:
                # Fallback to uniform
                factor = overall_scaling
                sfha_mult = non_sfha_mult = None
            
            if sfha_mult is not None:
                # Weighted average: policies are split between SFHA and non-SFHA
                factor = s_sfha * sfha_mult + (1 - s_sfha) * non_sfha_mult
        
        # Apply scaling
        for col in ["Policies", "Coverage", "Premium"]:
            if col in nfip_new.columns and pd.notna(nfip_new.at[idx, col]):
                nfip_new.at[idx, col] *= factor
    
    # Clean up temporary columns
    nfip_new = nfip_new.drop(columns=["NFIP_r_sfha", "NFIP_r_non", "NFIP_s_sfha"], errors="ignore")
    
    return nfip_new


def _apply_citizens_depopulation(
    private_exp: pd.DataFrame,
    citizens_exp: pd.DataFrame,
    target_citizens_share: float,
    coastal_counties: set[str],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Transfer exposure from Citizens to private market (reverse of market exit).
    
    Returns
    -------
    private_exp_new : pd.DataFrame
    citizens_exp_new : pd.DataFrame
    transfer_amt : float
        Amount of TIV transferred from Citizens to Private
    """
    private_new = private_exp.copy()
    citizens_new = citizens_exp.copy()
    
    total_tiv = private_exp["TIV"].sum() + citizens_exp["TIV"].sum()
    current_citizens_share = citizens_exp["TIV"].sum() / total_tiv if total_tiv > 0 else 0.15
    
    if target_citizens_share >= current_citizens_share:
        # No depopulation needed
        return private_new, citizens_new, 0.0
    
    # Calculate how much to transfer
    target_citizens_tiv = total_tiv * target_citizens_share
    current_citizens_tiv = citizens_exp["TIV"].sum()
    transfer_amt = current_citizens_tiv - target_citizens_tiv
    
    if transfer_amt <= 0:
        return private_new, citizens_new, 0.0
    
    # Reduce Citizens proportionally across counties
    reduction_factor = 1.0 - (transfer_amt / current_citizens_tiv)
    citizens_new["TIV"] *= reduction_factor
    if "TIV_sampled" in citizens_new.columns:
        citizens_new["TIV_sampled"] *= reduction_factor
    
    # Add to private market proportionally
    # Distribute to existing private companies by market share
    if not private_new.empty:
        company_shares = private_new.groupby("Company")["TIV"].sum()
        company_shares = company_shares / company_shares.sum()
        
        for company, share in company_shares.items():
            company_transfer = transfer_amt * share
            
            # Distribute company's new exposure across counties proportionally
            company_mask = private_new["Company"] == company
            company_tiv_by_county = private_new[company_mask].groupby("County")["TIV"].sum()
            
            if company_tiv_by_county.sum() > 0:
                for county, county_tiv in company_tiv_by_county.items():
                    county_share = county_tiv / company_tiv_by_county.sum()
                    county_transfer = company_transfer * county_share
                    
                    mask = (private_new["Company"] == company) & (private_new["County"] == county)
                    private_new.loc[mask, "TIV"] += county_transfer
                    if "TIV_sampled" in private_new.columns:
                        private_new.loc[mask, "TIV_sampled"] += county_transfer
    
    return private_new, citizens_new, transfer_amt


def _apply_geographic_scaling(
    exp_df: pd.DataFrame,
    coastal_counties: set[str],
    coastal_factor: float,
    inland_factor: float,
) -> pd.DataFrame:
    """
    Apply different TIV scaling factors to coastal vs. inland counties.
    
    Parameters
    ----------
    exp_df : pd.DataFrame
        Exposure DataFrame with 'County' column
    coastal_counties : set of str
        Set of coastal county names
    coastal_factor : float
        Scaling factor for coastal counties (e.g., 1.3 = +30%)
    inland_factor : float
        Scaling factor for inland counties (e.g., 1.1 = +10%)
        
    Returns
    -------
    pd.DataFrame
        Exposure with scaled TIV values
    """
    result = exp_df.copy()
    
    for idx, row in result.iterrows():
        county = row.get("County", "")
        factor = coastal_factor if county in coastal_counties else inland_factor
        
        result.at[idx, "TIV"] *= factor
        if "TIV_sampled" in result.columns:
            result.at[idx, "TIV_sampled"] *= factor
    
    return result


# =============================================================================
# Capital Adjustment
# =============================================================================

def adjust_surplus_for_penetration(
    surplus_df: pd.DataFrame,
    exposure_change_pct: float,
    method: str = "proportional",
    capital_multiplier: float = 1.0,
) -> dict:
    """
    Adjust insurer surplus to reflect changes in exposure from penetration increase.
    
    Simplified version that scales surplus based on aggregate exposure change.
    
    Parameters
    ----------
    surplus_df : pd.DataFrame
        Surplus data with columns ['Company', 'SurplusUSD']
    exposure_change_pct : float
        Percentage change in private exposure (e.g., 11.8 for 11.8% increase)
    method : str, default "proportional"
        How to scale surplus:
        - "proportional": Scale surplus linearly with TIV change (maintains stress ratio)
        - "none": No adjustment (stress ratio worsens - conservative)
        - "sqrt": Sublinear scaling (diversification benefit - optimistic)
    capital_multiplier : float, default 1.0
        Multiplier for capital growth beyond proportional scaling.
        1.0 = proportional (capital grows at same rate as exposure)
        1.5 = capital grows 50% faster than exposure
        Example: if exposure grows 56%, capital at 1.5x grows by 84%
        
    Returns
    -------
    dict
        Adjustment diagnostics with keys:
        - total_adjustment_usd: Total surplus increase across all companies
        - method: Adjustment method used
        - exposure_change_pct: Input exposure change
        
    Notes
    -----
    IMPORTANT: Without surplus adjustment (method="none"), increasing penetration
    can lead to MORE defaults because:
    - Exposure increases (more policies written)
    - Surplus stays constant
    - Stress ratio (TIV/Surplus) worsens
    - More companies fail under catastrophic loss
    
    This is realistic for SHORT-TERM forced penetration increases where insurers
    cannot raise capital quickly enough. For LONG-TERM scenarios (10+ years),
    use "proportional" or "sqrt" to reflect capital raising.
    
    Example:
        Baseline: TIV=$100B, Surplus=$10B → Stress ratio = 10:1
        After 11.8% exposure increase with no surplus adjustment:
            TIV=$111.8B, Surplus=$10B → Stress ratio = 11.18:1 (worse)
        After 11.8% exposure increase with proportional adjustment:
            TIV=$111.8B, Surplus=$11.18B → Stress ratio = 10:1 (maintained)
    """
    if method == "none" or abs(exposure_change_pct) < 0.001:
        return {
            "total_adjustment_usd": 0.0,
            "method": method,
            "exposure_change_pct": exposure_change_pct,
        }
    
    surplus_adjusted = surplus_df.copy()
    total_adjustment = 0.0
    
    # Convert percentage to multiplier
    exposure_multiplier = 1.0 + (exposure_change_pct / 100.0)
    
    # Apply capital multiplier to get actual capital growth
    # capital_multiplier = 1.0 → proportional growth
    # capital_multiplier > 1.0 → capital grows faster than exposure
    capital_growth_factor = 1.0 + (exposure_multiplier - 1.0) * capital_multiplier
    
    for idx, row in surplus_adjusted.iterrows():
        old_surplus = row["SurplusUSD"]
        
        if method == "proportional":
            # Linear scaling with capital multiplier applied
            new_surplus = old_surplus * capital_growth_factor
        elif method == "sqrt":
            # Sublinear scaling: reflects diversification benefit
            new_surplus = old_surplus * np.sqrt(exposure_multiplier) * capital_multiplier
        else:
            raise ValueError(f"Unknown surplus adjustment method: {method}")
        
        adjustment = new_surplus - old_surplus
        surplus_adjusted.at[idx, "SurplusUSD"] = new_surplus
        
        # Also adjust GroupSurplusUSD if present
        if "GroupSurplusUSD" in surplus_adjusted.columns and pd.notna(row.get("GroupSurplusUSD")):
            old_group_surplus = row["GroupSurplusUSD"]
            if method == "proportional":
                new_group_surplus = old_group_surplus * capital_growth_factor
            elif method == "sqrt":
                new_group_surplus = old_group_surplus * np.sqrt(exposure_multiplier) * capital_multiplier
            surplus_adjusted.at[idx, "GroupSurplusUSD"] = new_group_surplus
        
        total_adjustment += adjustment
    
    return {
        "total_adjustment_usd": total_adjustment,
        "adjusted_surplus_df": surplus_adjusted,
        "method": method,
        "exposure_change_pct": exposure_change_pct,
    }
