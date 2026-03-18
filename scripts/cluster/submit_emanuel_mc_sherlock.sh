#!/bin/bash
#SBATCH --job-name=emanuel_mc
#SBATCH --output=logs/emanuel_mc_%A_%a.out
#SBATCH --error=logs/emanuel_mc_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal

################################################################################
# Submit Emanuel TC Monte Carlo analysis on Sherlock
#
# Usage:
#   # Run ERA5 baseline (historical climate)
#   sbatch submit_emanuel_mc_sherlock.sh FL_era5_reanalcal
#
#   # Run ERA5 with policy scenario
#   sbatch submit_emanuel_mc_sherlock.sh FL_era5_reanalcal building_codes_major
#
#   # Run future climate scenario
#   sbatch submit_emanuel_mc_sherlock.sh FL_canesm5_ssp585_2081-2100
#
#   # Run multiple event sets in parallel using array jobs
#   sbatch --array=0-25 submit_emanuel_mc_sherlock.sh
#
# Notes:
#   - First argument: event set name (required unless using array jobs)
#   - Second argument: policy scenario (optional, default: baseline)
#   - Reads impact data from /scratch/groups/bakerjw/smeiler/impacts/
#   - Writes results to /scratch/groups/bakerjw/smeiler/results/emanuel_mc_runs/
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "======================================================================"
echo "SLURM Job Info"
echo "======================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
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

# Paths
IMPACT_ROOT="/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts"
OUTPUT_ROOT="${PROJECT_DIR}/results/mc_runs"

# Event set determination (command-line OR array job)
if [ -n "${1:-}" ]; then
    # Command-line argument takes precedence
    EVENT_SET="$1"
    POLICY="${2:-}"  # Optional policy scenario
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Array job mode - map task ID to event set
    EVENT_SETS=(
        "FL_era5_reanalcal"
        "FL_canesm_20thcal"
        "FL_canesm_ssp245cal"
        "FL_canesm_ssp245_2cal"
        "FL_canesm_ssp585cal"
        "FL_canesm_ssp585_2cal"
        "FL_cnrm6_20thcal"
        "FL_cnrm6_ssp245cal"
        "FL_cnrm6_ssp245_2cal"
        "FL_cnrm6_ssp585cal"
        "FL_cnrm6_ssp585_2cal"
        "FL_ecearth6_20thcal"
        "FL_ecearth6_ssp245cal"
        "FL_ecearth6_ssp245_2cal"
        "FL_ecearth6_ssp585cal"
        "FL_ecearth6_ssp585_2cal"
        "FL_ipsl6_20thcal"
        "FL_ipsl6_ssp245cal"
        "FL_ipsl6_ssp245_2cal"
        "FL_ipsl6_ssp585cal"
        "FL_ipsl6_ssp585_2cal"
        "FL_miroc6_20thcal"
        "FL_miroc6_ssp245cal"
        "FL_miroc6_ssp245_2cal"
        "FL_miroc6_ssp585cal"
        "FL_miroc6_ssp585_2cal"
    )
    EVENT_SET="${EVENT_SETS[$SLURM_ARRAY_TASK_ID]}"
    POLICY=""
else
    echo "ERROR: No event set specified"
    echo "Usage: sbatch submit_emanuel_mc_sherlock.sh <event_set> [policy_scenario]"
    echo "   OR: sbatch --array=0-25 submit_emanuel_mc_sherlock.sh"
    exit 1
fi

# Validate event set exists
IMPACT_DIR="${IMPACT_ROOT}/${EVENT_SET}"
if [ ! -d "$IMPACT_DIR" ]; then
    echo "ERROR: Impact directory not found: $IMPACT_DIR"
    echo "Run precompute_emanuel_tc_impacts.py first!"
    exit 1
fi

# Check for required files
YEAR_SETS_CSV="${IMPACT_DIR}/year_sets_N10000_seed42.csv"
if [ ! -f "$YEAR_SETS_CSV" ]; then
    echo "ERROR: Year-sets file not found: $YEAR_SETS_CSV"
    echo "Run generate_emanuel_year_sets.py first!"
    exit 1
fi

METADATA_CSV="${IMPACT_DIR}/event_metadata.csv"
if [ ! -f "$METADATA_CSV" ]; then
    echo "ERROR: Event metadata not found: $METADATA_CSV"
    echo "Run precompute_emanuel_tc_impacts.py first!"
    exit 1
fi

# Print configuration
echo "==========================================================================="
echo "EMANUEL TC MONTE CARLO - SHERLOCK JOB"
echo "==========================================================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Event set:     $EVENT_SET"
echo "Policy:        ${POLICY:-baseline}"
echo "Impact dir:    $IMPACT_DIR"
echo "Output dir:    $OUTPUT_ROOT"
echo "Host:          $(hostname)"
echo "Started:       $(date)"
echo ""

# Build command
CMD="python run_emanuel_monte_carlo.py"
CMD="$CMD --event_set $EVENT_SET"
CMD="$CMD --impact_dir $IMPACT_DIR"
CMD="$CMD --out $OUTPUT_ROOT"
CMD="$CMD --fast_threshold 0"  # Full modeling for all years

# Add policy if specified
if [ -n "${POLICY:-}" ]; then
    CMD="$CMD --policy $POLICY"
fi

# Run Monte Carlo
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
