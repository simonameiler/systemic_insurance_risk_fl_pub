"""
scenarios - Scenario configuration and transformation utilities
---------------------------------------------------------------

This module provides scenario-specific transformations for:
1. Market exit (private -> Citizens reallocation)
2. Penetration increase (coverage expansion)
3. Building code mitigation (loss reduction)

Each scenario modifies baseline inputs before or during risk propagation.
"""

from .market_exit import (
    apply_market_exit_scenario,
    calculate_exit_based_on_stress,
    adjust_group_capital_for_exits,
    adjust_citizens_capital_for_growth,
    MARKET_EXIT_PRESETS,
)

from .penetration import (
    apply_penetration_increase_scenario,
    adjust_surplus_for_penetration,
    PENETRATION_INCREASE_PRESETS,
)

from .building_codes import (
    apply_building_code_scenario,
    calculate_avoided_losses,
    compare_scenarios as compare_building_code_scenarios,
    validate_loss_reduction_factors,
    BUILDING_CODE_PRESETS,
)

__all__ = [
    # Scenario 1: Market Exit
    "apply_market_exit_scenario",
    "calculate_exit_based_on_stress",
    "adjust_group_capital_for_exits",
    "adjust_citizens_capital_for_growth",
    "MARKET_EXIT_PRESETS",
    
    # Scenario 2: Penetration Increase
    "apply_penetration_increase_scenario",
    "adjust_surplus_for_penetration",
    "PENETRATION_INCREASE_PRESETS",
    
    # Scenario 3: Building Codes
    "apply_building_code_scenario",
    "calculate_avoided_losses",
    "compare_building_code_scenarios",
    "validate_loss_reduction_factors",
    "BUILDING_CODE_PRESETS",
]
