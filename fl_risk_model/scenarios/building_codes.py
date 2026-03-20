"""
building_codes.py - Building code improvement scenarios
-------------------------------------------------------

Models loss reduction from improved building standards:
- Stricter wind resistance requirements
- Flood elevation requirements  
- Retrofitting of existing stock
- Enhanced code enforcement

Design Principles:
1. **Positive adaptation**: Stronger buildings -> lower losses
2. **Evidence-based**: Loss reduction factors from empirical studies (IBHS, FLASH, FEMA)
3. **Wind-focused**: Great Miami is primarily wind event
4. **Conservative calibration**: Use lower bound of literature estimates

Public API:
- apply_building_code_scenario: Apply loss reduction to damage calculations
- calculate_avoided_losses: Quantify mitigation benefit
- BUILDING_CODE_PRESETS: Pre-configured scenarios (MODERATE, MAJOR, EXTREME)
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd

# =============================================================================
# Preset Scenario Definitions
# =============================================================================

BUILDING_CODE_PRESETS = {
    "BASELINE": {
        "wind_loss_reduction": 0.00,
        "flood_loss_reduction": 0.00,
        "retrofit_rate": 0.00,
        "description": "Current 2024 building stock (no improvements)",
        "references": "Current FL Building Code (2020)",
    },
    
    "MINOR": {
        "wind_loss_reduction": 0.10,  # 10% loss reduction
        "flood_loss_reduction": 0.05,  # 5% loss reduction
        "retrofit_rate": 0.15,  # 15% of pre-2002 stock retrofitted
        "new_construction_rate": 0.06,  # ~2% annual × 3 years
        "timeline_years": 3,  # 2024 -> 2027
        "description": "Minimal improvements + code enforcement (2027 projection)",
        "references": "Enhanced FL Building Code enforcement, limited voluntary retrofits",
    },
    
    "MODERATE": {
        "wind_loss_reduction": 0.15,  # 15% loss reduction
        "flood_loss_reduction": 0.10,  # 10% loss reduction
        "retrofit_rate": 0.25,  # 25% of pre-2002 stock retrofitted
        "new_construction_rate": 0.12,  # ~2% annual × 6 years
        "timeline_years": 6,  # 2024 -> 2030
        "description": "Partial retrofit + new construction (2030 projection)",
        "references": "IBHS FORTIFIED (lower bound), FL Building Code enforcement improvements",
    },
    
    "MAJOR": {
        "wind_loss_reduction": 0.30,  # 30% loss reduction
        "flood_loss_reduction": 0.25,  # 25% loss reduction
        "retrofit_rate": 0.50,  # 50% retrofitted
        "new_construction_rate": 0.32,  # ~2% annual × 16 years
        "timeline_years": 16,  # 2024 -> 2040
        "description": "Aggressive retrofit program + stricter codes (2040 projection)",
        "references": "IBHS FORTIFIED (midpoint), FLASH study modern codes",
    },
    
    "EXTREME": {
        "wind_loss_reduction": 0.50,  # 50% loss reduction
        "flood_loss_reduction": 0.40,  # 40% loss reduction
        "retrofit_rate": 0.75,  # 75% retrofitted
        "new_construction_rate": 0.52,  # ~2% annual × 26 years
        "timeline_years": 26,  # 2024 -> 2050
        "description": "Full stock turnover + enhanced standards (2050 projection)",
        "references": "IBHS FORTIFIED + FLASH (upper bound), widespread elevation",
    },
}


# Empirical loss reduction estimates from literature
LOSS_REDUCTION_EVIDENCE = {
    "IBHS_FORTIFIED_roof": {
        "wind_loss_reduction": 0.35,  # 30-40%
        "source": "Insurance Institute for Business & Home Safety",
        "year": 2021,
    },
    "IBHS_FORTIFIED_windows": {
        "wind_loss_reduction": 0.25,  # 20-30%
        "source": "IBHS Impact-Resistant Windows Study",
        "year": 2020,
    },
    "FLASH_modern_codes": {
        "wind_loss_reduction": 0.50,  # 40-60%
        "source": "Johns Hopkins FLASH Project (2002+ FL Building Code)",
        "year": 2019,
    },
    "FEMA_elevation": {
        "flood_loss_reduction": 0.65,  # 50-80%
        "source": "FEMA Elevation Guidelines (1-2 ft above BFE)",
        "year": 2022,
    },
    "NIST_combined": {
        "wind_loss_reduction": 0.50,  # 45-55%
        "source": "NIST Combined Wind Mitigation Measures",
        "year": 2020,
    },
}


# =============================================================================
# Main Scenario Application
# =============================================================================

def apply_building_code_scenario(
    loss_df: pd.DataFrame,
    scenario: str = "MODERATE",
    hazard_type: str = "wind",
    apply_by_county: bool = False,
    coastal_counties: Optional[set[str]] = None,
    wind_loss_reduction: Optional[float] = None,
    flood_loss_reduction: Optional[float] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply loss reduction from improved building codes.
    
    This function modifies loss estimates to reflect improved building stock.
    Apply AFTER impact function calculation but BEFORE financial propagation.
    
    Parameters
    ----------
    loss_df : pd.DataFrame
        Loss estimates with columns like ['Company', 'County', 'loss_usd']
        Common column names: 'loss_usd', 'LossUSD', 'NetLossAfterFHCFUSD'
    scenario : str, default "MODERATE"
        One of: "BASELINE", "MODERATE", "MAJOR", "EXTREME"
        Ignored if wind_loss_reduction/flood_loss_reduction are provided directly
    hazard_type : str, default "wind"
        Type of hazard: "wind" | "flood" | "both"
    apply_by_county : bool, default False
        If True, apply different reduction factors to coastal vs. inland counties
        (coastal counties have better code enforcement)
    coastal_counties : set of str, optional
        Set of coastal county names (higher reduction if apply_by_county=True)
    wind_loss_reduction : float, optional
        Wind loss reduction factor (0.0-1.0, e.g., 0.15 = 15% reduction)
        If provided, overrides the preset scenario value
    flood_loss_reduction : float, optional
        Flood loss reduction factor (0.0-1.0)
        If provided, overrides the preset scenario value
        
    Returns
    -------
    loss_df_mitigated : pd.DataFrame
        Loss DataFrame with reduced losses
    diagnostics : dict
        Metrics: avoided losses, reduction rates, etc.
        
    Notes
    -----
    Loss reduction is applied multiplicatively:
        loss_after = loss_before × (1 - reduction_factor)
    
    For wind losses, we use the 'wind_loss_reduction' parameter.
    For flood losses, we use the 'flood_loss_reduction' parameter.
    
    You can either:
    1. Use a preset scenario: scenario="MAJOR"
    2. Provide custom reduction values: wind_loss_reduction=0.35, scenario is ignored
    
    Example with MODERATE scenario (15% reduction):
        Original loss: $10B
        Mitigated loss: $10B × (1 - 0.15) = $8.5B
        Avoided loss: $1.5B
    """
    if scenario not in BUILDING_CODE_PRESETS:
        raise ValueError(f"Unknown scenario '{scenario}'. Must be one of {list(BUILDING_CODE_PRESETS.keys())}")
    
    # Start with preset config
    config = BUILDING_CODE_PRESETS[scenario]
    
    # Override with custom values if provided
    if wind_loss_reduction is not None or flood_loss_reduction is not None:
        config = dict(config)  # Make a copy
        if wind_loss_reduction is not None:
            config["wind_loss_reduction"] = wind_loss_reduction
        if flood_loss_reduction is not None:
            config["flood_loss_reduction"] = flood_loss_reduction
        config["description"] = f"Custom (wind={config.get('wind_loss_reduction', 0):.1%}, flood={config.get('flood_loss_reduction', 0):.1%})"
    
    if scenario == "BASELINE" and wind_loss_reduction is None and flood_loss_reduction is None:
        # No mitigation
        return loss_df.copy(), {
            "scenario": scenario,
            "wind_loss_reduction": 0.0,
            "flood_loss_reduction": 0.0,
            "total_loss_before_usd": _get_total_loss(loss_df),
            "total_loss_after_usd": _get_total_loss(loss_df),
            "avoided_loss_usd": 0.0,
        }
    
    result = loss_df.copy()
    coastal_counties = coastal_counties or set()
    
    # Get reduction factor based on hazard type
    if hazard_type == "wind":
        base_reduction = config["wind_loss_reduction"]
    elif hazard_type == "flood":
        base_reduction = config["flood_loss_reduction"]
    elif hazard_type == "both":
        # Weighted average (assume 80% wind, 20% flood for Florida)
        base_reduction = 0.8 * config["wind_loss_reduction"] + 0.2 * config["flood_loss_reduction"]
    else:
        raise ValueError(f"Unknown hazard_type: {hazard_type}")
    
    # Identify loss column(s)
    loss_cols = _identify_loss_columns(result)
    
    if not loss_cols:
        raise ValueError("Could not identify loss columns in DataFrame. Expected columns like 'loss_usd', 'LossUSD', etc.")
    
    # Track total losses before mitigation
    total_loss_before = _get_total_loss(result, loss_cols)
    
    # Apply loss reduction
    if apply_by_county and "County" in result.columns:
        # Different reduction for coastal vs. inland
        # Coastal counties: 20% higher reduction (better enforcement)
        for idx, row in result.iterrows():
            county = row.get("County", "")
            
            if county in coastal_counties:
                reduction_factor = base_reduction * 1.2  # 20% more effective
            else:
                reduction_factor = base_reduction * 0.9  # 10% less effective
            
            reduction_factor = min(reduction_factor, 0.80)  # Cap at 80% reduction
            
            for col in loss_cols:
                if col in result.columns:
                    original_loss = result.at[idx, col]
                    result.at[idx, col] = original_loss * (1 - reduction_factor)
    else:
        # Uniform reduction across all counties
        for col in loss_cols:
            if col in result.columns:
                result[col] = result[col] * (1 - base_reduction)
    
    # Track total losses after mitigation
    total_loss_after = _get_total_loss(result, loss_cols)
    avoided_loss = total_loss_before - total_loss_after
    
    diagnostics = {
        "scenario": scenario,
        "description": config["description"],
        "hazard_type": hazard_type,
        "wind_loss_reduction": config["wind_loss_reduction"],
        "flood_loss_reduction": config["flood_loss_reduction"],
        "base_reduction_applied": base_reduction,
        "retrofit_rate": config.get("retrofit_rate", 0.0),
        "timeline_years": config.get("timeline_years", 0),
        "total_loss_before_usd": total_loss_before,
        "total_loss_after_usd": total_loss_after,
        "avoided_loss_usd": avoided_loss,
        "avoided_loss_pct": (avoided_loss / total_loss_before * 100) if total_loss_before > 0 else 0.0,
        "references": config.get("references", ""),
    }
    
    return result, diagnostics


# =============================================================================
# Helper Functions
# =============================================================================

def _identify_loss_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify which columns contain loss values.
    
    Common patterns:
    - loss_usd, LossUSD, loss
    - NetLoss, NetLossAfterFHCF, NetLossUSD
    - TotalLoss, GroundUpLoss
    - Wind/Water damage columns: InsuredWindUSD, UninsuredWindUSD, etc.
    """
    loss_patterns = [
        "loss_usd",
        "LossUSD",
        "loss",
        "NetLoss",
        "NetLossAfterFHCF",
        "NetLossUSD",
        "TotalLoss",
        "GroundUpLoss",
        "ClaimUSD",
        "WindUSD",  # Catches InsuredWindUSD, UninsuredWindUSD, etc.
        "WaterUSD",  # For flood losses
        "DamageUSD",  # Catches GrossWindDamageUSD, WindDamageUSD, etc.
    ]
    
    found_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern.lower() in col_lower for pattern in loss_patterns):
            found_cols.append(col)
    
    return found_cols


def _get_total_loss(df: pd.DataFrame, loss_cols: Optional[list[str]] = None) -> float:
    """
    Calculate total loss from DataFrame.
    """
    if loss_cols is None:
        loss_cols = _identify_loss_columns(df)
    
    if not loss_cols:
        return 0.0
    
    # Sum across all loss columns (typically just one)
    total = 0.0
    for col in loss_cols:
        if col in df.columns:
            total += df[col].sum()
    
    return total


# =============================================================================
# Analysis Functions
# =============================================================================

def calculate_avoided_losses(
    baseline_losses: pd.DataFrame,
    scenario: str = "MODERATE",
    hazard_type: str = "wind",
) -> dict:
    """
    Calculate avoided losses from building code improvements.
    
    Parameters
    ----------
    baseline_losses : pd.DataFrame
        Loss estimates without mitigation
    scenario : str
        Building code scenario
    hazard_type : str
        "wind" | "flood" | "both"
        
    Returns
    -------
    dict
        Summary of avoided losses by company, county, etc.
    """
    mitigated_losses, diag = apply_building_code_scenario(
        baseline_losses,
        scenario=scenario,
        hazard_type=hazard_type,
    )
    
    # Calculate by company
    loss_cols = _identify_loss_columns(baseline_losses)
    
    if "Company" in baseline_losses.columns and loss_cols:
        col = loss_cols[0]
        by_company = pd.DataFrame({
            "Company": baseline_losses["Company"].unique(),
        })
        
        baseline_by_co = baseline_losses.groupby("Company")[col].sum()
        mitigated_by_co = mitigated_losses.groupby("Company")[col].sum()
        
        by_company = by_company.set_index("Company")
        by_company["baseline_loss_usd"] = baseline_by_co
        by_company["mitigated_loss_usd"] = mitigated_by_co
        by_company["avoided_loss_usd"] = baseline_by_co - mitigated_by_co
        by_company["avoided_pct"] = (
            (by_company["avoided_loss_usd"] / by_company["baseline_loss_usd"] * 100)
            .fillna(0)
        )
        
        diag["by_company"] = by_company.reset_index()
    
    # Calculate by county
    if "County" in baseline_losses.columns and loss_cols:
        col = loss_cols[0]
        baseline_by_county = baseline_losses.groupby("County")[col].sum()
        mitigated_by_county = mitigated_losses.groupby("County")[col].sum()
        
        by_county = pd.DataFrame({
            "County": baseline_by_county.index,
            "baseline_loss_usd": baseline_by_county.values,
            "mitigated_loss_usd": mitigated_by_county.values,
        })
        by_county["avoided_loss_usd"] = by_county["baseline_loss_usd"] - by_county["mitigated_loss_usd"]
        by_county["avoided_pct"] = (
            (by_county["avoided_loss_usd"] / by_county["baseline_loss_usd"] * 100)
            .fillna(0)
        )
        
        diag["by_county"] = by_county
    
    return diag


def compare_scenarios(
    baseline_losses: pd.DataFrame,
    scenarios: list[str] = None,
    hazard_type: str = "wind",
) -> pd.DataFrame:
    """
    Compare building code scenarios side-by-side.
    
    Parameters
    ----------
    baseline_losses : pd.DataFrame
        Loss estimates without mitigation
    scenarios : list of str, optional
        Scenarios to compare. Default: all scenarios
    hazard_type : str
        "wind" | "flood" | "both"
        
    Returns
    -------
    pd.DataFrame
        Comparison table with avoided losses for each scenario
    """
    if scenarios is None:
        scenarios = ["BASELINE", "MODERATE", "MAJOR", "EXTREME"]
    
    results = []
    
    for scenario in scenarios:
        _, diag = apply_building_code_scenario(
            baseline_losses,
            scenario=scenario,
            hazard_type=hazard_type,
        )
        
        results.append({
            "Scenario": scenario,
            "Description": BUILDING_CODE_PRESETS[scenario]["description"],
            "Loss Reduction (%)": BUILDING_CODE_PRESETS[scenario].get(f"{hazard_type}_loss_reduction", 0) * 100,
            "Baseline Loss ($B)": diag["total_loss_before_usd"] / 1e9,
            "Mitigated Loss ($B)": diag["total_loss_after_usd"] / 1e9,
            "Avoided Loss ($B)": diag["avoided_loss_usd"] / 1e9,
            "Avoided (%)": diag["avoided_loss_pct"],
            "Timeline (years)": BUILDING_CODE_PRESETS[scenario].get("timeline_years", 0),
            "Retrofit Rate (%)": BUILDING_CODE_PRESETS[scenario].get("retrofit_rate", 0) * 100,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_loss_reduction_factors(scenario: str = "MODERATE") -> dict:
    """
    Validate that scenario loss reduction factors are within literature bounds.
    
    Returns dictionary showing how scenario compares to empirical evidence.
    """
    config = BUILDING_CODE_PRESETS[scenario]
    
    wind_reduction = config.get("wind_loss_reduction", 0)
    flood_reduction = config.get("flood_loss_reduction", 0)
    
    validation = {
        "scenario": scenario,
        "wind_loss_reduction_used": wind_reduction,
        "flood_loss_reduction_used": flood_reduction,
        "wind_evidence": {},
        "flood_evidence": {},
        "validation_status": "PASS",
        "warnings": [],
    }
    
    # Compare to wind evidence
    wind_evidence = {
        "IBHS_FORTIFIED_roof": 0.35,
        "IBHS_FORTIFIED_windows": 0.25,
        "FLASH_modern_codes": 0.50,
        "NIST_combined": 0.50,
    }
    
    validation["wind_evidence"] = wind_evidence
    
    wind_max = max(wind_evidence.values())
    if wind_reduction > wind_max:
        validation["validation_status"] = "WARNING"
        validation["warnings"].append(
            f"Wind reduction {wind_reduction:.0%} exceeds literature maximum {wind_max:.0%}"
        )
    
    # Compare to flood evidence
    flood_max = LOSS_REDUCTION_EVIDENCE["FEMA_elevation"]["flood_loss_reduction"]
    validation["flood_evidence"] = {"FEMA_elevation": flood_max}
    
    if flood_reduction > flood_max:
        validation["validation_status"] = "WARNING"
        validation["warnings"].append(
            f"Flood reduction {flood_reduction:.0%} exceeds literature maximum {flood_max:.0%}"
        )
    
    if not validation["warnings"]:
        validation["warnings"].append("All reduction factors within literature bounds")
    
    return validation
