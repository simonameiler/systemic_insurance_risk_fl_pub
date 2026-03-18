"""
market_exit.py — Private insurer market exit scenarios
------------------------------------------------------

Models realistic market withdrawal under stress, with exposure transfer to Citizens.

Design Principles:
1. **Realistic exit criteria**: Small/stressed insurers exit first
2. **Partial Citizens absorption**: Not all private exposure transfers (some goes uninsured)
3. **Cascading effects**: Track capital, reinsurance, assessments
4. **Geographic concentration**: Coastal insurers more likely to exit

Public API:
- apply_market_exit_scenario: Transform exposure DataFrames for market exit
- calculate_exit_based_on_stress: Determine which companies exit based on metrics
- adjust_group_capital_for_exits: Adjust parent group capital when subsidiaries exit
- adjust_citizens_capital_for_growth: Calculate Citizens capital needs for increased share
- MARKET_EXIT_PRESETS: Pre-configured scenarios (MODERATE, MAJOR, EXTREME)
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd

# =============================================================================
# Preset Scenario Definitions
# =============================================================================

MARKET_EXIT_PRESETS = {
    "BASELINE": {
        "citizens_target_share": 0.15,  # Current 2024 baseline (~15%)
        "exit_mechanism": "none",
        "description": "No market exit (2024 baseline)",
    },
    "MODERATE": {
        "citizens_target_share": 0.25,  # +10pp to Citizens
        "exit_mechanism": "uniform",    # Uniform reduction of private market
        "citizens_absorption_rate": 0.85,  # 85% of exited exposure → Citizens, 15% → uninsured
        "description": "Moderate exit: continuation of 2022-2024 trend through 2030",
    },
    "MAJOR": {
        "citizens_target_share": 0.40,  # +25pp to Citizens
        "exit_mechanism": "stress_based",  # Exit based on company stress metrics
        "stress_threshold_percentile": 0.65,  # Top 35% stressed companies exit
        "citizens_absorption_rate": 0.75,  # 75% → Citizens, 25% → uninsured
        "description": "Major exit: post-catastrophe market restructuring by 2035",
    },
    "EXTREME": {
        "citizens_target_share": 0.55,  # +40pp to Citizens
        "exit_mechanism": "stress_based",
        "stress_threshold_percentile": 0.50,  # Top 50% stressed companies exit
        "citizens_absorption_rate": 0.70,  # 70% → Citizens, 30% → uninsured
        "coastal_bias": 1.5,  # Coastal counties 1.5x more likely to lose coverage
        "description": "Extreme exit: private market collapse by 2045",
    },
}


# =============================================================================
# Main Scenario Application
# =============================================================================

def apply_market_exit_scenario(
    private_exp: pd.DataFrame,
    citizens_exp: pd.DataFrame,
    scenario: str = "MODERATE",
    surplus_df: Optional[pd.DataFrame] = None,
    market_share_df: Optional[pd.DataFrame] = None,
    coastal_counties: Optional[set[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply market exit scenario to exposure DataFrames.
    
    Parameters
    ----------
    private_exp : pd.DataFrame
        Private insurer exposure with columns ['Company','County','TIV','TIV_sampled']
    citizens_exp : pd.DataFrame
        Citizens exposure with columns ['Company','County','TIV','TIV_sampled']
    scenario : str, default "MODERATE"
        One of: "BASELINE", "MODERATE", "MAJOR", "EXTREME"
    surplus_df : pd.DataFrame, optional
        Surplus data for stress-based exit (columns: ['Company','SurplusUSD'])
    market_share_df : pd.DataFrame, optional
        Market share data for stress calculation
    coastal_counties : set of str, optional
        Set of coastal county names for geographic bias
    rng : np.random.Generator, optional
        Random number generator for stochastic allocation
        
    Returns
    -------
    private_exp_new : pd.DataFrame
        Modified private exposure (reduced)
    citizens_exp_new : pd.DataFrame
        Modified Citizens exposure (increased)
    diagnostics : dict
        Scenario metrics: transfer amounts, companies exited, etc.
    """
    if scenario not in MARKET_EXIT_PRESETS:
        raise ValueError(f"Unknown scenario '{scenario}'. Must be one of {list(MARKET_EXIT_PRESETS.keys())}")
    
    config = MARKET_EXIT_PRESETS[scenario]
    
    if config["exit_mechanism"] == "none":
        # Baseline: no changes
        return private_exp.copy(), citizens_exp.copy(), {
            "scenario": scenario,
            "transfer_tiv_usd": 0.0,
            "companies_exited": 0,
            "citizens_share_before": _calculate_citizens_share(private_exp, citizens_exp),
            "citizens_share_after": _calculate_citizens_share(private_exp, citizens_exp),
        }
    
    rng = rng or np.random.default_rng()
    
    # Calculate current state
    total_tiv_before = private_exp["TIV"].sum() + citizens_exp["TIV"].sum()
    citizens_tiv_before = citizens_exp["TIV"].sum()
    citizens_share_before = citizens_tiv_before / total_tiv_before if total_tiv_before > 0 else 0.0
    
    target_share = config["citizens_target_share"]
    
    if config["exit_mechanism"] == "uniform":
        # Simple uniform reduction of private market
        private_new, citizens_new, transfer_amt = _uniform_exit(
            private_exp,
            citizens_exp,
            target_citizens_share=target_share,
            absorption_rate=config.get("citizens_absorption_rate", 0.80),
            coastal_counties=coastal_counties,
            coastal_bias=config.get("coastal_bias", 1.0),
            rng=rng,
        )
        companies_exited = "N/A (uniform reduction)"
        
    elif config["exit_mechanism"] == "stress_based":
        # Realistic: stressed companies exit
        if surplus_df is None:
            raise ValueError("stress_based exit requires surplus_df parameter")
        
        private_new, citizens_new, transfer_amt, exited_companies = _stress_based_exit(
            private_exp,
            citizens_exp,
            surplus_df,
            target_citizens_share=target_share,
            stress_percentile=config.get("stress_threshold_percentile", 0.70),
            absorption_rate=config.get("citizens_absorption_rate", 0.80),
            coastal_counties=coastal_counties,
            coastal_bias=config.get("coastal_bias", 1.0),
            rng=rng,
        )
        companies_exited = exited_companies
    
    else:
        raise ValueError(f"Unknown exit mechanism: {config['exit_mechanism']}")
    
    # Calculate final state
    total_tiv_after = private_new["TIV"].sum() + citizens_new["TIV"].sum()
    citizens_share_after = citizens_new["TIV"].sum() / total_tiv_after if total_tiv_after > 0 else 0.0
    
    # --- Capital adjustments ---
    # Adjust group capital if companies exited and we have surplus data
    surplus_adjusted = None
    group_capital_diagnostics = None
    if surplus_df is not None and isinstance(companies_exited, list) and len(companies_exited) > 0:
        surplus_adjusted = adjust_group_capital_for_exits(
            surplus_df,
            companies_exited,
            scaling_method="middle_ground",  # Default to balanced approach
        )
        
        # Calculate total group capital change
        if "GroupSurplusUSD" in surplus_df.columns:
            group_surplus_before = surplus_df["GroupSurplusUSD"].sum()
            group_surplus_after = surplus_adjusted["GroupSurplusUSD"].sum()
            group_capital_diagnostics = {
                "group_surplus_before_usd": group_surplus_before,
                "group_surplus_after_usd": group_surplus_after,
                "group_surplus_reduction_usd": group_surplus_before - group_surplus_after,
                "scaling_method": "middle_ground",
            }
    
    # Calculate Citizens capital requirements
    # Assume baseline Citizens capital is ~$12B for 15% share (2024 baseline)
    citizens_capital_baseline = 12_000_000_000.0  # $12B
    citizens_capital_info = adjust_citizens_capital_for_growth(
        citizens_capital_usd=citizens_capital_baseline,
        citizens_share_old=0.15,  # 2024 baseline
        citizens_share_new=citizens_share_after,
        scaling_method="adverse_selection",  # Default: assume worse risks
    )
    
    diagnostics = {
        "scenario": scenario,
        "description": config["description"],
        "exit_mechanism": config["exit_mechanism"],
        "target_citizens_share": target_share,
        "citizens_share_before": citizens_share_before,
        "citizens_share_after": citizens_share_after,
        "total_tiv_before_usd": total_tiv_before,
        "total_tiv_after_usd": total_tiv_after,
        "transfer_to_citizens_usd": transfer_amt,
        "lost_to_uninsured_usd": total_tiv_before - total_tiv_after,
        "absorption_rate_realized": transfer_amt / (total_tiv_before - total_tiv_after) if total_tiv_before > total_tiv_after else 1.0,
        "companies_exited": companies_exited,
        "group_capital": group_capital_diagnostics,
        "citizens_capital": citizens_capital_info,
        "surplus_adjusted": surplus_adjusted,  # Pass back adjusted surplus for use in runner
    }
    
    return private_new, citizens_new, diagnostics


# =============================================================================
# Exit Mechanisms
# =============================================================================

def _uniform_exit(
    private_exp: pd.DataFrame,
    citizens_exp: pd.DataFrame,
    target_citizens_share: float,
    absorption_rate: float,
    coastal_counties: Optional[set[str]],
    coastal_bias: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Uniform reduction of private market with geographic bias.
    
    Strategy:
    1. Calculate total TIV needed to transfer from private → Citizens
    2. Apply geographic bias (coastal counties lose more coverage)
    3. Transfer (absorption_rate × reduction) to Citizens, rest goes uninsured
    """
    total_tiv = private_exp["TIV"].sum() + citizens_exp["TIV"].sum()
    current_citizens_tiv = citizens_exp["TIV"].sum()
    
    # How much TIV must Citizens gain?
    target_citizens_tiv = total_tiv * target_citizens_share
    needed_transfer = max(0, target_citizens_tiv - current_citizens_tiv)
    
    if needed_transfer == 0:
        return private_exp.copy(), citizens_exp.copy(), 0.0
    
    # Total needed removal from private (accounting for absorption rate)
    # If absorption_rate = 0.8, then to transfer $100 to Citizens, we remove $125 from private
    # (80% → Citizens, 20% → uninsured)
    total_removal_needed = needed_transfer / absorption_rate
    
    # Geographic bias: coastal counties lose more
    private_new = private_exp.copy()
    
    if coastal_counties and coastal_bias != 1.0:
        is_coastal = private_new["County"].isin(coastal_counties)
        private_new["exit_weight"] = np.where(is_coastal, coastal_bias, 1.0)
    else:
        private_new["exit_weight"] = 1.0
    
    # Normalize weights
    private_new["exit_weight"] = private_new["exit_weight"] / private_new["exit_weight"].mean()
    
    # Proportional reduction weighted by exit_weight
    current_private_tiv = private_new["TIV"].sum()
    private_new["reduction_share"] = (
        private_new["TIV"] * private_new["exit_weight"] / 
        (private_new["TIV"] * private_new["exit_weight"]).sum()
    )
    private_new["TIV_reduction"] = private_new["reduction_share"] * total_removal_needed
    private_new["TIV"] = (private_new["TIV"] - private_new["TIV_reduction"]).clip(lower=0)
    
    # Update TIV_sampled proportionally
    if "TIV_sampled" in private_new.columns:
        private_new["TIV_sampled"] = (private_new["TIV_sampled"] - private_new["TIV_reduction"]).clip(lower=0)
    
    private_new = private_new.drop(columns=["exit_weight", "reduction_share", "TIV_reduction"])
    
    # Transfer to Citizens (proportional to county distribution)
    citizens_new = citizens_exp.copy()
    county_transfer = (
        private_exp.groupby("County")["TIV"].sum() - 
        private_new.groupby("County")["TIV"].sum()
    ) * absorption_rate
    
    for county, transfer_amt in county_transfer.items():
        if transfer_amt > 0:
            county_mask = citizens_new["County"] == county
            if county_mask.any():
                # Add to existing Citizens presence
                boost_factor = 1 + (transfer_amt / citizens_new.loc[county_mask, "TIV"].sum())
                citizens_new.loc[county_mask, "TIV"] *= boost_factor
                if "TIV_sampled" in citizens_new.columns:
                    citizens_new.loc[county_mask, "TIV_sampled"] *= boost_factor
            else:
                # Create new Citizens row for this county
                new_row = {
                    "Company": "Citizens Property Insurance Corporation",
                    "County": county,
                    "TIV": transfer_amt,
                }
                if "TIV_sampled" in citizens_new.columns:
                    new_row["TIV_sampled"] = transfer_amt
                citizens_new = pd.concat([citizens_new, pd.DataFrame([new_row])], ignore_index=True)
    
    actual_transfer = needed_transfer
    
    return private_new, citizens_new, actual_transfer


def _stress_based_exit(
    private_exp: pd.DataFrame,
    citizens_exp: pd.DataFrame,
    surplus_df: pd.DataFrame,
    target_citizens_share: float,
    stress_percentile: float,
    absorption_rate: float,
    coastal_counties: Optional[set[str]],
    coastal_bias: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, list]:
    """
    Stressed companies exit market (realistic mechanism).
    
    Strategy:
    1. Calculate stress ratio: TIV / SurplusUSD for each company
    2. Companies above stress_percentile threshold exit
    3. Their TIV transfers to Citizens (×absorption_rate)
    4. Iterate until target_citizens_share is reached
    
    Note: Citizens is excluded from stress ratio calculations since it plays
    a special role as the insurer of last resort and cannot "exit" the market.
    """
    # Exclude Citizens from surplus data (it's handled separately)
    surplus_private = surplus_df[~surplus_df["Company"].str.contains("Citizens", case=False, na=False)].copy()
    
    # Calculate company stress metrics
    company_tiv = private_exp.groupby("Company")["TIV"].sum().reset_index()
    stress = company_tiv.merge(surplus_private[["Company", "SurplusUSD"]], on="Company", how="left")
    stress["SurplusUSD"] = stress["SurplusUSD"].fillna(stress["TIV"].median() * 0.10)  # Impute missing
    stress["stress_ratio"] = stress["TIV"] / stress["SurplusUSD"]
    
    # Determine exit threshold
    threshold = stress["stress_ratio"].quantile(stress_percentile)
    
    # Identify exiting companies
    exiting = stress[stress["stress_ratio"] >= threshold]["Company"].tolist()
    
    # Iterative exit until target share reached
    private_new = private_exp.copy()
    citizens_new = citizens_exp.copy()
    total_transfer = 0.0
    exited_companies = []
    
    for company in exiting:
        # Check if we've reached target
        current_share = _calculate_citizens_share(private_new, citizens_new)
        if current_share >= target_citizens_share:
            break
        
        # Remove this company's exposure
        company_mask = private_new["Company"] == company
        company_tiv = private_new.loc[company_mask, "TIV"].sum()
        
        if company_tiv == 0:
            continue
        
        # Transfer to Citizens by county
        company_exposure_by_county = (
            private_new.loc[company_mask]
            .groupby("County")["TIV"]
            .sum()
        )
        
        for county, tiv_amt in company_exposure_by_county.items():
            transfer_amt = tiv_amt * absorption_rate
            
            # Apply coastal bias
            if coastal_counties and county in coastal_counties:
                transfer_amt *= coastal_bias
            
            # Add to Citizens
            county_mask = citizens_new["County"] == county
            if county_mask.any():
                citizens_new.loc[county_mask, "TIV"] += transfer_amt
                if "TIV_sampled" in citizens_new.columns:
                    citizens_new.loc[county_mask, "TIV_sampled"] += transfer_amt
            else:
                new_row = {
                    "Company": "Citizens Property Insurance Corporation",
                    "County": county,
                    "TIV": transfer_amt,
                }
                if "TIV_sampled" in citizens_new.columns:
                    new_row["TIV_sampled"] = transfer_amt
                citizens_new = pd.concat([citizens_new, pd.DataFrame([new_row])], ignore_index=True)
            
            total_transfer += transfer_amt
        
        # Remove company from private exposure
        private_new = private_new[~company_mask].copy()
        exited_companies.append(company)
    
    return private_new, citizens_new, total_transfer, exited_companies


# =============================================================================
# Helpers
# =============================================================================

def _calculate_citizens_share(private_exp: pd.DataFrame, citizens_exp: pd.DataFrame) -> float:
    """Calculate Citizens' share of total TIV."""
    total = private_exp["TIV"].sum() + citizens_exp["TIV"].sum()
    if total == 0:
        return 0.0
    return citizens_exp["TIV"].sum() / total


def calculate_exit_based_on_stress(
    surplus_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    stress_threshold_percentile: float = 0.70,
) -> pd.DataFrame:
    """
    Identify companies likely to exit based on stress ratio.
    
    Returns DataFrame with columns: ['Company', 'TIV', 'SurplusUSD', 'stress_ratio', 'exit_flag']
    """
    company_tiv = exposure_df.groupby("Company")["TIV"].sum().reset_index()
    stress = company_tiv.merge(surplus_df[["Company", "SurplusUSD"]], on="Company", how="left")
    stress["SurplusUSD"] = stress["SurplusUSD"].fillna(stress["TIV"].median() * 0.10)
    stress["stress_ratio"] = stress["TIV"] / stress["SurplusUSD"]
    
    threshold = stress["stress_ratio"].quantile(stress_threshold_percentile)
    stress["exit_flag"] = stress["stress_ratio"] >= threshold
    
    return stress.sort_values("stress_ratio", ascending=False)


# =============================================================================
# Capital Adjustment Functions
# =============================================================================

def adjust_group_capital_for_exits(
    surplus_df: pd.DataFrame,
    exiting_companies: list[str],
    scaling_method: str = "middle_ground",
) -> pd.DataFrame:
    """
    Adjust group capital pools when member companies exit the market.
    
    When a subsidiary exits, the parent group loses that entity's contribution
    to the group surplus pool. This function reduces GroupSurplusUSD accordingly.
    
    Parameters
    ----------
    surplus_df : pd.DataFrame
        Surplus data with columns: ['Company', 'SurplusUSD', 'GroupSurplusUSD',
        'NAICGroupNumber', 'NAICGroupName']
    exiting_companies : list of str
        Companies that have exited the market
    scaling_method : str, default "middle_ground"
        How to adjust group capital:
        - "conservative": Reduce GroupSurplusUSD by exiting entity's SurplusUSD
        - "aggressive": Set GroupSurplusUSD = remaining entity surplus (no parent pool)
        - "middle_ground": Scale group pool proportionally by remaining entity count
        
    Returns
    -------
    pd.DataFrame
        Adjusted surplus data with reduced GroupSurplusUSD for affected groups
        
    Notes
    -----
    Conservative approach: Parent withdraws proportional support
        GroupSurplus_new = GroupSurplus_old - ExitingSurplus
        
    Aggressive approach: Parent exits entirely when subsidiaries leave
        GroupSurplus_new = sum(remaining entity surplus)
        
    Middle ground: Parent support scales with entity count
        GroupSurplus_new = GroupSurplus_old × (n_remaining / n_original)
        
    Examples
    --------
    >>> # Group G has 3 entities with $100M, $50M, $30M surplus
    >>> # GroupSurplusUSD = $300M (includes parent pool)
    >>> # Entity A ($100M) exits
    >>> 
    >>> # Conservative: GroupSurplusUSD → $200M (lost A's contribution)
    >>> # Aggressive: GroupSurplusUSD → $80M (only B+C remain, no parent)
    >>> # Middle ground: GroupSurplusUSD → $200M (3→2 entities, scale by 2/3)
    """
    if not exiting_companies:
        return surplus_df.copy()
    
    surplus_adjusted = surplus_df.copy()
    
    # Determine group ID column
    if "NAICGroupNumber" in surplus_adjusted.columns:
        group_col = "NAICGroupNumber"
    elif "NAICGroupName" in surplus_adjusted.columns:
        group_col = "NAICGroupName"
    else:
        # No group structure; each company is its own group
        return surplus_adjusted
    
    # Identify affected groups
    exiting_mask = surplus_adjusted["Company"].isin(exiting_companies)
    affected_groups = surplus_adjusted.loc[exiting_mask, group_col].unique()
    
    for group_id in affected_groups:
        group_mask = surplus_adjusted[group_col] == group_id
        group_df = surplus_adjusted[group_mask]
        
        n_original = len(group_df)
        exiting_in_group = group_df["Company"].isin(exiting_companies)
        n_exiting = exiting_in_group.sum()
        n_remaining = n_original - n_exiting
        
        if n_remaining == 0:
            # Entire group exits; set GroupSurplusUSD to zero
            surplus_adjusted.loc[group_mask, "GroupSurplusUSD"] = 0.0
            continue
        
        # Get current group surplus
        group_surplus_old = group_df["GroupSurplusUSD"].iloc[0]
        
        if scaling_method == "conservative":
            # Reduce by exiting entity surplus
            exiting_surplus = group_df.loc[exiting_in_group, "SurplusUSD"].sum()
            group_surplus_new = max(group_surplus_old - exiting_surplus, 0.0)
            
        elif scaling_method == "aggressive":
            # Group pool becomes sum of remaining entities (no parent support)
            remaining_surplus = group_df.loc[~exiting_in_group, "SurplusUSD"].sum()
            group_surplus_new = remaining_surplus
            
        elif scaling_method == "middle_ground":
            # Scale proportionally by entity count
            scaling_factor = n_remaining / n_original
            group_surplus_new = group_surplus_old * scaling_factor
            
        else:
            raise ValueError(f"Unknown scaling_method: {scaling_method}")
        
        # Apply adjustment to all remaining members of the group
        surplus_adjusted.loc[group_mask, "GroupSurplusUSD"] = group_surplus_new
    
    # Remove exiting companies
    surplus_adjusted = surplus_adjusted[~exiting_mask].copy()
    
    return surplus_adjusted


def adjust_citizens_capital_for_growth(
    citizens_capital_usd: float,
    citizens_share_old: float,
    citizens_share_new: float,
    scaling_method: str = "adverse_selection",
    legislative_cap_usd: float = 15_000_000_000.0,
) -> dict:
    """
    Calculate Citizens capital requirements after market share growth.
    
    When Citizens absorbs exiting private insurers, its capital needs increase.
    Different methods reflect different assumptions about risk quality and
    political constraints.
    
    Parameters
    ----------
    citizens_capital_usd : float
        Current Citizens surplus (e.g., $12B in 2024)
    citizens_share_old : float
        Current Citizens market share (e.g., 0.15 = 15%)
    citizens_share_new : float
        Target Citizens market share (e.g., 0.40 = 40%)
    scaling_method : str, default "adverse_selection"
        How to scale capital:
        - "proportional": Linear scaling (assumes same risk quality)
        - "adverse_selection": Nonlinear scaling (worse risks absorbed)
        - "legislative_cap": Proportional but capped by political limit
    legislative_cap_usd : float, default 15B
        Maximum Citizens capital under legislative_cap method
        
    Returns
    -------
    dict
        {
            "citizens_capital_required_usd": Required capital for new share,
            "citizens_capital_shortfall_usd": Gap vs. current capital,
            "scaling_method": Method used,
            "scaling_factor": Multiplier applied,
            "assumptions": Description of method,
        }
        
    Notes
    -----
    Proportional Scaling:
        Assumes absorbed exposure has same risk profile as existing book.
        Capital_new = Capital_old × (Share_new / Share_old)
        
    Adverse Selection Scaling:
        Assumes absorbed exposure is riskier (failed private insurers).
        Uses nonlinear exponent (1.3) to reflect concentration + worse risks.
        Capital_new = Capital_old × (Share_new / Share_old) ^ 1.3
        
    Legislative Cap:
        Assumes political constraints limit Citizens capital expansion.
        Florida legislature has historically resisted unlimited Citizens growth.
        Capital_new = min(proportional_scaling, $15B)
        
    Examples
    --------
    >>> # 2024: $12B capital, 15% share → 40% share
    >>> adjust_citizens_capital_for_growth(12e9, 0.15, 0.40, "proportional")
    {"citizens_capital_required_usd": 32.0B, "shortfall": 20.0B}
    
    >>> # Same scenario with adverse selection
    >>> adjust_citizens_capital_for_growth(12e9, 0.15, 0.40, "adverse_selection")
    {"citizens_capital_required_usd": 41.2B, "shortfall": 29.2B}
    """
    if citizens_share_old <= 0:
        raise ValueError("citizens_share_old must be positive")
    
    share_ratio = citizens_share_new / citizens_share_old
    
    if scaling_method == "proportional":
        scaling_factor = share_ratio
        capital_required = citizens_capital_usd * scaling_factor
        assumptions = "Linear scaling: assumes absorbed risks match existing book quality"
        
    elif scaling_method == "adverse_selection":
        # Nonlinear: worse risks + concentration
        exponent = 1.3  # Empirically calibrated to post-catastrophe market conditions
        scaling_factor = share_ratio ** exponent
        capital_required = citizens_capital_usd * scaling_factor
        assumptions = (
            "Nonlinear scaling (exponent=1.3): reflects adverse selection "
            "(exiting insurers likely had worse risk selection) and concentration risk"
        )
        
    elif scaling_method == "legislative_cap":
        scaling_factor = share_ratio
        capital_required = min(
            citizens_capital_usd * scaling_factor,
            legislative_cap_usd
        )
        assumptions = (
            f"Proportional scaling capped at ${legislative_cap_usd/1e9:.1f}B due to "
            "legislative/political constraints on Citizens growth"
        )
        
    else:
        raise ValueError(f"Unknown scaling_method: {scaling_method}")
    
    shortfall = max(capital_required - citizens_capital_usd, 0.0)
    
    return {
        "citizens_capital_required_usd": capital_required,
        "citizens_capital_shortfall_usd": shortfall,
        "citizens_share_old": citizens_share_old,
        "citizens_share_new": citizens_share_new,
        "scaling_method": scaling_method,
        "scaling_factor": scaling_factor,
        "assumptions": assumptions,
    }
