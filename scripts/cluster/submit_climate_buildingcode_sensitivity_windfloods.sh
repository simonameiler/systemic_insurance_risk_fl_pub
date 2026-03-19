#!/bin/bash
#SBATCH --job-name=climate_bc_sens
#SBATCH --output=/home/users/smeiler/repos/systemic_insurance_risk_fl/logs/climate_bc_sens_%A_%a.out
#SBATCH --error=/home/users/smeiler/repos/systemic_insurance_risk_fl/logs/climate_bc_sens_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-64

################################################################################
# Climate + Building Codes Sensitivity Analysis - GCM × Building Codes
#
# Research Question: How effective are building codes at reducing insurance
# system stress under SSP2-4.5 2050 climate change across different GCMs?
#
# Approach:
#   - Use 5 GCM TC event sets for SSP2-4.5 mid-century (2041-2060)
#   - Test 13 building code levels with wind/flood split (0% to 100%/90%)
#   - Compare insurance system metrics across all combinations
#   - Total jobs: 5 GCMs × 13 building code levels = 65 array tasks
#
# Usage:
#   sbatch scripts/cluster/submit_climate_buildingcode_sensitivity_windfloods.sh
#
# Output:
#   results/mc_runs/emanuel_{gcm}_ssp245cal_building_codes_wf_{wind}w_{flood}f_*
#
# Notes:
#   - Wind/flood split maintains 3:2 ratio (e.g., 30% wind / 20% flood)
#   - Uses physics-based GCM TC catalogs (not damage scaling)
#   - Building codes applied during simulation (not post-processing)
#   - SSP2-4.5 2050 represents mid-century climate conditions
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "======================================================================"
echo "CLIMATE + BUILDING CODES SENSITIVITY (GCM × BUILDING CODES)"
echo "======================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "======================================================================"
echo ""

# Set up paths
PROJECT_DIR="/home/users/smeiler/repos/systemic_insurance_risk_fl"
cd ${PROJECT_DIR}

# Create logs directory
mkdir -p logs

# Activate conda environment
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate climada_env
else
  echo "ERROR: Could not find conda. Please ensure conda is in PATH."
  exit 1
fi

# GCMs for SSP2-4.5 mid-century (2041-2060)
GCMS=(
    "FL_canesm_ssp245cal"
    "FL_cnrm6_ssp245cal"
    "FL_ecearth6_ssp245cal"
    "FL_ipsl6_ssp245cal"
    "FL_miroc6_ssp245cal"
)

# Building code levels (indexed 0-12)
# Level 0: 0% wind, 0% flood     | Level 1: 20% wind, 13% flood
# Level 2: 30% wind, 20% flood   | Level 3: 40% wind, 27% flood
# Level 4: 50% wind, 33% flood   | Level 5: 60% wind, 40% flood
# Level 6: 70% wind, 47% flood   | Level 7: 80% wind, 53% flood
# Level 8: 90% wind, 60% flood   | Level 9: 100% wind, 67% flood
# Level 10: 100% wind, 70% flood | Level 11: 100% wind, 80% flood
# Level 12: 100% wind, 90% flood
CODE_LEVELS=(0 1 2 3 4 5 6 7 8 9 10 11 12)

# Map array task ID to GCM × code_level combination
N_GCMS=${#GCMS[@]}
N_CODE_LEVELS=${#CODE_LEVELS[@]}

GCM_IDX=$((SLURM_ARRAY_TASK_ID / N_CODE_LEVELS))
CODE_IDX=$((SLURM_ARRAY_TASK_ID % N_CODE_LEVELS))

EVENT_SET="${GCMS[$GCM_IDX]}"
CODE_LEVEL="${CODE_LEVELS[$CODE_IDX]}"

# Paths
IMPACT_ROOT="/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts"
OUTPUT_ROOT="${PROJECT_DIR}/results/mc_runs"
IMPACT_DIR="${IMPACT_ROOT}/${EVENT_SET}"

# Validate event set exists
if [ ! -d "$IMPACT_DIR" ]; then
    echo "ERROR: Impact directory not found: $IMPACT_DIR"
    exit 1
fi

# Check for required files
YEAR_SETS_CSV="${IMPACT_DIR}/year_sets_N10000_seed42.csv"
if [ ! -f "$YEAR_SETS_CSV" ]; then
    echo "ERROR: Year-sets file not found: $YEAR_SETS_CSV"
    exit 1
fi

METADATA_CSV="${IMPACT_DIR}/event_metadata.csv"
if [ ! -f "$METADATA_CSV" ]; then
    echo "ERROR: Event metadata not found: $METADATA_CSV"
    exit 1
fi

# Print configuration
echo "==========================================================================="
echo "CONFIGURATION"
echo "==========================================================================="
echo "GCM Event Set:          $EVENT_SET"
echo "Building Code Level:    $CODE_LEVEL"
echo "Impact Directory:       $IMPACT_DIR"
echo "Output Directory:       $OUTPUT_ROOT"
echo "==========================================================================="
echo ""

# Run Monte Carlo simulation with run_climate_buildingcode_sensitivity_windfloods.py
# This script handles the wind/flood split mapping for each code level
CMD="python scripts/run/run_climate_buildingcode_sensitivity_windfloods.py"
CMD="$CMD --event_set $EVENT_SET"
CMD="$CMD --impact_dir $IMPACT_DIR"
CMD="$CMD --code_level $CODE_LEVEL"
CMD="$CMD --seed 42"

echo "Command: $CMD"
echo ""
$CMD

EXIT_CODE=$?

echo ""
echo "==========================================================================="
echo "COMPLETED: $(date)"
echo "Exit code: $EXIT_CODE"
echo "==========================================================================="

exit $EXIT_CODE
