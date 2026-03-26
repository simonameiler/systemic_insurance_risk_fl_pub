"""
Configuration parameters for Florida three-branch risk-flow model.

DATA FILE REQUIREMENTS
======================
The model requires several data files. Some are included in this repository;
others must be obtained from the cited sources:

PUBLIC DATA (included in fl_risk_model/data/):
- FHCF_2024_Exposure_byCounty.xlsx - Florida Hurricane Catastrophe Fund
- company_keys.csv - Company identifier mappings
- citizens_*.csv - Citizens Property Insurance data
- nfip_*.csv - National Flood Insurance Program data
- catbonds_2024.csv - Catastrophe bond data
- 24fin_fhcf.csv - FHCF contract terms
- florida_*.csv - County-level attribution factors

PROPRIETARY DATA (available on request):
The following files contain proprietary Florida OIR data and must be
obtained separately. Contact the authors for access instructions:
- FL Surplus Capital, Group v Entity.xlsx - Insurer capital data
- FL HO Market Share Report.xlsx - Market share data
"""
import os
from pathlib import Path

# Project roots
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (adjust if needed)

# Use pathlib throughout
DATA_DIR = PROJECT_ROOT / "fl_risk_model" / "data"

# --- Global fixed year knob (used when SAMPLING_MODE_* == "FIXED_YEAR") ---
FIXED_YEAR = 2024

# --- Input data files ---
# Public data (included in repository)
EXPOSURE_FILE     = DATA_DIR / "FHCF_2024_Exposure_byCounty.xlsx"

# Proprietary data (must be obtained separately - contact authors or Florida OIR)
SURPLUS_FILE      = DATA_DIR / "20250805 FL Surplus Capital, Group v Entity.xlsx"
MARKET_SHARE_XLSX = DATA_DIR / "FL HO Market Share Report_6.10.25.xlsx"

MARKET_SHARE_YEAR = FIXED_YEAR
VERBOSE_EXPOSURE = False      # gate noisy per-county/company prints
PRINT_MASSBALANCE_TOP5 = False  # gate the “Largest rel. gaps” table

COMPANY_KEYS   = DATA_DIR / "company_keys.csv"   # mapping NAIC/statkey to company name

# Citizens & FHCF constants

# Identity (used for joining keys, QA, and reporting)
CITIZENS_COMPANY_NAME = "Citizens Property Insurance Corporation"
CITIZENS_NAIC         = "10064"
CITIZENS_STATKEY      = "C6949"

# Citizens exposure inputs (you already have these)
CITIZENS_AS_OF        = "2024-09-30"
CITIZENS_PRODUCTS     = ['PLA PR-M','PLA PR-W']
CITIZENS_COUNTY_CSV   = DATA_DIR / "citizens_county_data_all_harmonized.csv"

# Citizens capital inputs
CITIZENS_CAPITAL_YEAR = FIXED_YEAR
CITIZENS_CAPITAL_CSV  = DATA_DIR / "citizens_capital_pml.csv"

# Flood (NFIP)
NFIP_PENETRATION_CSV   = DATA_DIR / "NfipResidentialPenetrationRates.csv"
NFIP_POLICIES_CSV      = DATA_DIR / "nfip_FL_coverage_premium_by_year.csv"
NFIP_POLICY_YEAR       = 2024

# Payout logic
NFIP_PAYOUT_MODE       = "unity"   # "unity" | "fixed" | "historical"
NFIP_PAYOUT_FIXED_RATE = 0.90      # only used if mode == "fixed"

# Financing
NFIP_CAPITAL_POOL_USD  = 3_441_000_000.0

# Randomness / reproducibility
RNG_SEED = 42

# --- Sampling strategy options ---
# Options: "FIXED_YEAR" | "EWA_5Y"
SAMPLING_MODE_EXPOSURE      = "FIXED_YEAR"   # private wind + Citizens exposures
SAMPLING_MODE_CAPITAL       = "FIXED_YEAR"   # private + Citizens surplus
SAMPLING_MODE_NFIP_POLICIES = "FIXED_YEAR"   # flood coverage
SAMPLING_MODE_NFIP_CLAIMS   = "EWA_5Y"       # flood claims

# --- Parameters for EWA (exponentially weighted average) ---
EWA_HALF_LIFE_YEARS = 2  # recency weighting (2-year half-life)
EWA_WINDOW_YEARS    = 5  # number of years to include

# Sampling switches (wind exposure)
SAMPLE_EXPOSURE = True        # if True, sample company×county TIV
SAMPLE_SURPLUS = False

# Monte Carlo parameters
EXPOSURE_COV = 0.15       # Coefficient of variation for TIV sampling
SURPLUS_COV = 0.15        # CoV for surplus sampling

# Insured fraction ~ Beta(4,6) -> mean 0.40
INSURED_ALPHA = 4
INSURED_BETA  = 6

# Override: set to a float (e.g. 0.3) to bypass Beta sampling and use a
# fixed insured fraction for sensitivity analysis.  None -> use Beta(α, β).
FIXED_INSURED_FRAC = None

# Underinsured share of the household fraction ~ Beta(3,7) -> mean 0.30
UNDER_HH_ALPHA = 3
UNDER_HH_BETA  = 7

# NOTE: RBC proxy factors removed (DO_RBC, RBC_FROM_RESERVES_FACTOR, RBC_FROM_PREMIUMS_FACTOR)
# These were planned features that were never implemented.

# FHCF contract‐year multipliers (2023-24 contract)
FHCF_RET_MULTIPLES    = {90: 6.0732, 75: 7.2878, 45: 12.1464}
FHCF_PAYOUT_MULTIPLE  = 11.2368
FHCF_LAE_FACTOR       = 1.10
FHCF_SEASON_CAP          = 17_000_000_000.0
FHCF_PAYOUT_MULTIPLIER    = 11.2368  # season-wide payout multiple (aka limit multiple)

# Path to your per‐insurer FHCF terms CSV
FHCF_TERMS_CSV       = DATA_DIR / "24fin_fhcf.csv"

# --- FIGA assessment caps (as % of premium base) ---
FIGA_CAP_NORMAL    = 0.02   # 2%
FIGA_CAP_EMERGENCY = 0.04   # +4% emergency (total possible 6%)

# ---- Citizens-specific FHCF anchors (from filings/budget) ----
# We treat the Citizens FHCF reimbursement premium as *known* from the 2024 filings.
CITIZENS_FHCF_PREMIUM_USD   = 406_542_771
CITIZENS_FHCF_COVERAGE_PCT  = 0.90       # Citizens’ FHCF coverage election (90%)

# Derived FHCF terms for Citizens (computed from the premium + season multipliers)
# Retention = Premium × RetentionMultiple
# Max Reimbursement (limit) = Premium × PayoutMultiple × CoveragePct
CITIZENS_FHCF_RETENTION_USD = CITIZENS_FHCF_PREMIUM_USD * FHCF_RET_MULTIPLES[90] # using 90% retention multiple
CITIZENS_FHCF_LIMIT_USD     = CITIZENS_FHCF_PREMIUM_USD * FHCF_PAYOUT_MULTIPLIER * CITIZENS_FHCF_COVERAGE_PCT

# Controls: when True, force the runner to use these config terms for Citizens
# if the 24fin_fhcf.csv row is missing/zero for retention/limit.
CITIZENS_FHCF_FORCE_CONFIG_TERMS = False

# --- Citizens assessments caps (as % of premium base) ---
CIT_TIER1_CAP = 0.15  # ≤ 15%
CIT_TIER2_CAP = 0.10  # up to 10%

# Fallback if runner can't find real premium base
CITIZENS_PREMIUM_BASE_DEFAULT = 0.0

# --- NFIP capital / assessment ---
DO_FLOOD = True

# Modes: "UNLIMITED" (status quo), "CAPPED" (finite pool + assessment), "BORROW" (cap then borrow)
NFIP_CAPITAL_MODE = "BORROW"       # or "CAPPED" | "BORROW"

# Size of the modeled NFIP capital pool (USD) used when NFIP_CAPITAL_MODE != "UNLIMITED"
#NFIP_CAPITAL_POOL_USD = 0

# Post-event surcharge capacity on NFIP premium base (share of annual written premium)
# applied only when pool is insufficient (CAPPED or BORROW modes)
NFIP_SURCHARGE_MAX_RATE = 0.00        # e.g., up to +15% of the annual premium base

# If mode="BORROW", residual after pool+assessment is covered by borrowing (no default).
# If mode="CAPPED", residual after pool+assessment is deemed "unfunded".
NFIP_BORROW_ENABLED = True            # respected only if NFIP_CAPITAL_MODE == "BORROW"

# Input for assessment capacity
NFIP_PREMIUM_BASE_CSV = DATA_DIR / "nfip_FL_coverage_premium_by_year.csv"   # or roll up from policies
NFIP_PREMIUM_BASE_YEAR = FIXED_YEAR

# Cat bonds
CATBONDS_CSV = DATA_DIR / "catbonds_2024.csv"
CATBOND_DEFAULT_ATTACH_MULT = 1.0
CATBOND_DEFAULT_EXH_MULT   = 2.0

# for QA: county TIV mass balance
MASSBAL_ABS_FLOOR = 1.0     # $1 absolute tolerance
MASSBAL_REL_TOL   = 1e-9    # relative to CountyTIV

# Group capital contribution settings
GROUP_CONTRIBUTION_RANGE = (0.6, 0.9)  # Min and max contribution rates

# === Event hazard inputs & scenarios ===
# Directory holding per-event county loss CSVs (one file per event)
EVENT_REPORTS_DIR = DATA_DIR / "hazard" / "historical_events"  # e.g., fl_risk_model/data/hazard/historical_events

# Stochastic TC event set (synthetic storms)
# Detect if running on cluster (Sherlock) vs local
import os as _os
import socket as _socket
_hostname = _socket.gethostname().lower()
_IS_CLUSTER = (
    'SLURM_JOB_ID' in _os.environ or 
    'sh-' in _hostname or 'sherlock' in _hostname or
    str(Path.home()).startswith('/home/users/')
)

if _IS_CLUSTER:
    # Sherlock cluster paths - ERA5 baseline year-sets
    # NOTE: FL_present_stochastic was OLD approach with lambda1.5 scaling - DELETED
    # Correct path: FL_era5_reanalcal (no frequency scaling)
    SYNTHETIC_EVENT_DIR = Path("/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts/FL_era5_reanalcal")
    SYNTHETIC_YEAR_SETS_CSV = Path("/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts/FL_era5_reanalcal/year_sets_N10000_seed42.csv")
else:
    # Local paths
    SYNTHETIC_EVENT_DIR = DATA_DIR / "hazard" / "synthetic"
    SYNTHETIC_YEAR_SETS_CSV = SYNTHETIC_EVENT_DIR / "year_sets_N10000_seed42.csv"  # Local should NOT have lambda1.5
SYNTHETIC_EVENT_METADATA_CSV = SYNTHETIC_EVENT_DIR / "event_metadata.csv"

# Scenario -> list of event file stems (without .csv) to combine in ONE season
SCENARIOS = {
    # a) Great Miami Hurricane (1926)
    "great_miami": ["1926255N15314"],
    # b) Hurricane Andrew (1992)
    "andrew": ["1992230N11325"],
    # c) Andrew then Great Miami (uses composite with sequential degradation)
    "andrew_then_gm": ["COMPOSITE"],
    # d) Great Miami then Andrew (uses composite with sequential degradation)
    "gm_then_andrew": ["COMPOSITE"],
    # e) Great Miami twice (uses composite with sequential degradation)
    "double_gm": ["COMPOSITE"],
    # f) Lake Okeechobee Hurricane (1928)
    "lake_okeechobee": ["1928250N14343"],
    # g) Hurricane Irma (2017)
    "irma": ["2017242N16333"],
    # h) Irma twice (uses composite with sequential degradation)
    "double_irma": ["COMPOSITE"],
}

# Composite scenarios mapping: scenario_name -> (csv_filename, legacy_wind_share)
# Note: Wind shares now come from COMPOSITE_WIND_SHARE_PARAMS (Beta distribution)
# The second tuple element is kept for backward compatibility but not used when mode="beta"
COMPOSITE_SCENARIOS = {
    # Andrew hits first (destroys 2.4% exposure), then Great Miami (on 97.6% remaining)
    "andrew_then_gm": ("andrew_then_great_miami.csv", 0.77),  # mean=0.77 in PARAMS
    
    # Great Miami hits first (destroys 3.7% exposure), then Andrew (on 96.3% remaining)
    "gm_then_andrew": ("great_miami_then_andrew.csv", 0.76),  # mean=0.76 in PARAMS
    
    # Great Miami hitting twice
    "double_gm": ("great_miami_twice.csv", 0.70),  # mean=0.70 in PARAMS
    
    # Irma hitting twice
    "double_irma": ("irma_twice.csv", 0.50),  # mean=0.50 in PARAMS
}

# Event-specific wind-share priors using Beta distributions
# Beta distribution parameters: mean and concentration
# - mean: best estimate of wind share (0 to 1)
# - concentration: controls spread (higher = tighter around mean)
#   * Low (5-10): Wide uncertainty, epistemic ignorance
#   * Medium (15-25): Moderate certainty
#   * High (30+): Strong evidence, tight around mean
#
# For single events: concentration ~10 reflects large epistemic uncertainty
# For composites: concentration ~25-30 reflects reduced uncertainty from averaging
EVENT_WIND_SHARE_PARAMS = {
    "1926255N15314": {"mean": 0.70, "concentration": 10},   # Great Miami: centered at 70% wind
    "1992230N11325": {"mean": 0.875, "concentration": 10},  # Andrew: centered at 87.5% wind
    "1928250N14343": {"mean": 0.30, "concentration": 8},    # Lake Okeechobee: centered at 30% wind
    "2017242N16333": {"mean": 0.50, "concentration": 10},   # Irma: centered at 50% wind
}

# Default for events without specific parameters
# Empirical county mean (Gori-weighted P95): 84.6% wind (range: 72.4-92.5%, n=67 counties)
DEFAULT_WIND_SHARE_MEAN = 0.846  # Present: 84.6% wind (P95 extremes, Gori-weighted method)
DEFAULT_WIND_SHARE_CONCENTRATION = 10

# Wind share bounds used by runner.py branch fallback (wind.py, flood.py)
EVENT_WIND_SHARE_BOUNDS = {
    "1926255N15314": (0.60, 0.80),  # Great Miami
    "1992230N11325": (0.80, 0.95),  # Andrew
    "1928250N14343": (0.20, 0.40),  # Lake Okeechobee
    "2017242N16333": (0.40, 0.60),  # Irma
}
DEFAULT_WIND_SHARE_BOUNDS = (0.65, 0.85)

# MC will populate this per iteration, e.g. {"1926255N15314": 0.72, "1992230N11325": 0.85}
RUNTIME_WIND_SHARE_OVERRIDES: dict[str, float] = {}

# Composite wind share parameters (Beta distribution)
# Higher concentration than single events reflects reduced uncertainty from averaging
COMPOSITE_WIND_SHARE_PARAMS = {
    # Andrew hits first, then Great Miami
    # Weighted: (109*0.875 + 157*0.70)/266 ~ 0.77
    "andrew_then_gm": {"mean": 0.77, "concentration": 25},
    
    # Great Miami hits first, then Andrew
    # Weighted: (170*0.70 + 96*0.875)/266 ~ 0.76
    "gm_then_andrew": {"mean": 0.76, "concentration": 25},
    
    # Great Miami hitting twice (same event, tightest uncertainty)
    "double_gm": {"mean": 0.70, "concentration": 30},
    
    # Irma hitting twice (same event, tightest uncertainty)
    "double_irma": {"mean": 0.50, "concentration": 30},
}

# --------------------------------------------------------------------------------------
# County-specific wind/water redistribution
# --------------------------------------------------------------------------------------
# Toggle to enable spatial heterogeneity in wind/water splits while preserving overall share
# When enabled:
#   1. Sample overall wind share from Beta distribution (e.g., 70% for Great Miami)
#   2. Load county-specific deviations from EMPIRICAL climatology (historical TC events)
#   3. Miami-Dade: 70% + (87.8% - 70.0% empirical mean) = 87.8% wind (example)
#   4. Escambia: 70% + (78.9% - 70.0% empirical mean) = 78.9% wind (example)
#   5. Weighted by losses, still averages to exactly 70%
#
# Data source: florida_empirical_hazard_attribution_p95.csv (for extreme events)
#   Based on actual FLOIR/NFIP damage observations from historical TC events
#   P95 represents extreme events (major hurricanes)
#   Use for stress testing / extreme event scenarios
#
# For stochastic TC analysis (full event distribution), use mean attribution instead
USE_COUNTY_REDISTRIBUTION = True  # Set to True to enable
COUNTY_ATTRIBUTION_MODE = "p95"   # "p95" for extreme events, "mean" for stochastic full distribution

# --------------------------------------------------------------------------------------
# Policy Scenario Configurations for Monte Carlo Analysis
# --------------------------------------------------------------------------------------
# Define policy interventions to test in MC runs alongside hazard scenarios
# Each configuration is passed to runner.py's scenario_config parameter
#
# Available scenario types:
#   - "market_exit": Private insurers exit high-risk counties, Citizens absorbs exposure
#   - "penetration": Increase private wind and NFIP flood penetration rates
#   - "building_codes": Apply loss reduction from improved construction standards

POLICY_SCENARIOS = {
    "baseline": None,  # No intervention, current market structure
    
    # === MARKET EXIT scenarios (all severity levels) ===
    "market_exit_baseline": {
        "type": "market_exit",
        "params": {"scenario": "BASELINE"}
    },
    "market_exit_moderate": {
        "type": "market_exit",
        "params": {"scenario": "MODERATE"}
    },
    "market_exit_major": {
        "type": "market_exit",
        "params": {"scenario": "MAJOR"}
    },
    "market_exit_extreme": {
        "type": "market_exit",
        "params": {"scenario": "EXTREME"}
    },
    
    # === PENETRATION scenarios (all severity levels with surplus adjustment) ===
    "penetration_baseline": {
        "type": "penetration",
        "params": {
            "scenario": "BASELINE",
            "surplus_adjustment": "proportional",
        }
    },
    "penetration_moderate": {
        "type": "penetration",
        "params": {
            "scenario": "MODERATE",
            "surplus_adjustment": "proportional",
        }
    },
    "penetration_major": {
        "type": "penetration",
        "params": {
            "scenario": "MAJOR",
            "surplus_adjustment": "proportional",
            # INTERPRETATION:
            #   - "proportional": Assumes gradual penetration increase (16-year timeline to 2040)
            #                     with corresponding insurer capital raising. Surplus scales linearly
            #                     with TIV to maintain constant stress ratios.
            #   - "none": Represents forced rapid penetration increase (e.g., 2-3 years) where
            #             insurers cannot raise capital quickly enough. Stress ratios worsen,
            #             leading to higher default rates. This is realistic for policy-mandated
            #             rapid market changes.
            #   - "sqrt": Assumes diversification benefits from larger books of business.
            #             Sublinear scaling between "none" and "proportional".
        }
    },
    "penetration_extreme": {
        "type": "penetration",
        "params": {
            "scenario": "EXTREME",
            "surplus_adjustment": "proportional",
        }
    },
    
    # === BUILDING CODES scenarios (all severity levels) ===
    "building_codes_baseline": {
        "type": "building_codes",
        "params": {"scenario": "BASELINE"}
    },
    "building_codes_minor": {
        "type": "building_codes",
        "params": {"scenario": "MINOR"}
    },
    "building_codes_moderate": {
        "type": "building_codes",
        "params": {"scenario": "MODERATE"}
    },
    "building_codes_major": {
        "type": "building_codes",
        "params": {"scenario": "MAJOR"}
    },
    "building_codes_extreme": {
        "type": "building_codes",
        "params": {"scenario": "EXTREME"}
    },
}

# For Great Miami focused analysis, specify which scenarios to run
# This allows running 4 scenarios × N iterations efficiently
GREAT_MIAMI_POLICY_RUN = {
    "hazard_scenario": "great_miami",  # Which hazard event from SCENARIOS
    "policy_scenarios": ["baseline", "market_exit_moderate", "penetration_major", "building_codes_major"],
    "n_iter": 200,  # Monte Carlo iterations per policy scenario
}

# MC controls and output
MC_N_ITER = 500         # Default for full hazard scenario mix
MC_OUT_DIR = Path("results/mc_runs")  # IMPORTANT: must be a Path, not a string


from pathlib import Path as _P
def _p(x): return x if isinstance(x, _P) else _P(x)
for _name in ("DATA_DIR","EXPOSURE_FILE","SURPLUS_FILE",
              "MARKET_SHARE_XLSX","FHCF_TERMS_CSV","CATBONDS_CSV",
              "CITIZENS_COUNTY_CSV","CITIZENS_CAPITAL_CSV","NFIP_PENETRATION_CSV",
              "NFIP_POLICIES_CSV","NFIP_PREMIUM_BASE_CSV","COMPANY_KEYS"):
    if _name in globals():
        globals()[_name] = _p(globals()[_name])