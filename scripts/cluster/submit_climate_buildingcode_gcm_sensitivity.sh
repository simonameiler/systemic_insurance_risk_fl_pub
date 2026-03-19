#!/bin/bash
#SBATCH --job-name=climate_bc_gcm
#SBATCH --output=logs/climate_bc_gcm_%A_%a.out
#SBATCH --error=logs/climate_bc_gcm_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-49

################################################################################
# Climate + Building Codes Sensitivity Analysis - GCM Event Sets
#
# Research Question: How much building code improvement is needed to offset
# SSP2-4.5 2050 climate change impacts on Florida's insurance system?
#
# Approach:
#   - Use 5 GCM TC event sets for SSP2-4.5 mid-century (2041-2060)
#   - Test 10 building code wind loss reduction levels (10%, 20%, ..., 100%)
#   - Compare to ERA5 baseline (current climate, no building codes)
#   - Total jobs: 5 GCMs × 10 building code levels = 50 array tasks
#
# Usage:
#   sbatch scripts/cluster/submit_climate_buildingcode_gcm_sensitivity.sh
#
# Output:
#   results/mc_runs/emanuel_{gcm}_ssp245cal_building_codes_bc{loss_reduction}pct_*
#
# Notes:
#   - Uses physics-based GCM TC catalogs (not damage scaling)
#   - Mid-century SSP2-4.5 represents ~2050 conditions
#   - Building codes applied during simulation (not post-processing)
#   - Directory names include bc{X}pct to prevent timestamp collisions
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "======================================================================"
echo "CLIMATE + BUILDING CODES GCM SENSITIVITY"
echo "======================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "======================================================================"
echo ""

# Set up paths
PROJECT_DIR="${HOME}/repos/systemic_insurance_risk_fl"
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

# Building code wind loss reduction levels (10% to 100% in 10% increments)
LOSS_REDUCTIONS=(10 20 30 40 50 60 70 80 90 100)

# Map array task ID to GCM × loss_reduction combination
N_GCMS=${#GCMS[@]}
N_LOSS_REDUCTIONS=${#LOSS_REDUCTIONS[@]}

GCM_IDX=$((SLURM_ARRAY_TASK_ID / N_LOSS_REDUCTIONS))
LOSS_IDX=$((SLURM_ARRAY_TASK_ID % N_LOSS_REDUCTIONS))

EVENT_SET="${GCMS[$GCM_IDX]}"
LOSS_REDUCTION="${LOSS_REDUCTIONS[$LOSS_IDX]}"

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
echo "Building Code Reduction: ${LOSS_REDUCTION}%"
echo "Impact Directory:       $IMPACT_DIR"
echo "Output Directory:       $OUTPUT_ROOT"
echo "==========================================================================="
echo ""

# Map loss reduction to building code scenario
# Use building_codes_baseline (0% baseline) and override with custom loss reduction
# Apply SAME reduction to both wind and flood losses
POLICY_SCENARIO="building_codes_baseline"
LOSS_REDUCTION_FRACTION=$(python -c "print($LOSS_REDUCTION / 100.0)")

echo "Policy Scenario: $POLICY_SCENARIO (baseline preset)"
echo "Loss Reduction: $LOSS_REDUCTION_FRACTION (${LOSS_REDUCTION}%) applied to BOTH wind and flood"
echo ""

# Create unique run label with building code percentage to prevent timestamp collisions
GCM_SHORT=$(echo $EVENT_SET | cut -d'_' -f2)  # Extract gcm name (canesm, cnrm6, etc)
RUN_LABEL="emanuel_${GCM_SHORT}_ssp245cal_building_codes_bc${LOSS_REDUCTION}pct"

echo "Run Label: $RUN_LABEL"
echo ""

# Run Monte Carlo simulation
CMD="python scripts/run/run_emanuel_monte_carlo.py"
CMD="$CMD --event_set $EVENT_SET"
CMD="$CMD --impact_dir $IMPACT_DIR"
CMD="$CMD --out $OUTPUT_ROOT"
CMD="$CMD --fast_threshold 0"
CMD="$CMD --policy $POLICY_SCENARIO"
CMD="$CMD --wind_loss_reduction $LOSS_REDUCTION_FRACTION"  # Wind loss reduction
CMD="$CMD --flood_loss_reduction $LOSS_REDUCTION_FRACTION"  # Flood loss reduction (same as wind)
CMD="$CMD --run_label $RUN_LABEL"  # Unique directory name

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
