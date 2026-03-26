#!/bin/bash
#SBATCH --job-name=emanuel_policy_suite
#SBATCH --output=logs/emanuel_policy_%A.out
#SBATCH --error=logs/emanuel_policy_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

################################################################################
# Submit Emanuel TC Policy Suite on Sherlock
#
# Runs multiple policy scenarios with a single Emanuel event set for comparison.
#
# Usage:
#   # Run 3 key policies (baseline, market_exit, building_codes) with ERA5
#   sbatch submit_emanuel_policy_suite_sherlock.sh FL_era5_reanalcal
#
#   # Run all policies
#   sbatch submit_emanuel_policy_suite_sherlock.sh FL_era5_reanalcal --all
#
#   # Custom policy list
#   sbatch submit_emanuel_policy_suite_sherlock.sh FL_era5_reanalcal \
#       --policies baseline market_exit_moderate penetration_major building_codes_major
#
# Notes:
#   - Runs policies sequentially (48h time limit - normal partition max)
#   - For parallel policy runs, use submit_policy_suite_parallel.sh
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Set up paths
PROJECT_DIR="${HOME}/repos/systemic_insurance_risk_fl_pub"
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

# Event set (required)
if [ -z "$1" ]; then
    echo "ERROR: Event set required"
    echo "Usage: sbatch submit_emanuel_policy_suite_sherlock.sh <event_set> [options]"
    exit 1
fi

EVENT_SET="$1"
shift  # Remove first argument, rest are passed to Python

# Validate event set exists
IMPACT_DIR="${IMPACT_ROOT}/${EVENT_SET}"
if [ ! -d "$IMPACT_DIR" ]; then
    echo "ERROR: Impact directory not found: $IMPACT_DIR"
    exit 1
fi

# Print configuration
echo "==========================================================================="
echo "EMANUEL TC POLICY SUITE - SHERLOCK JOB"
echo "==========================================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Event set:  $EVENT_SET"
echo "Impact dir: $IMPACT_DIR"
echo "Output dir: $OUTPUT_ROOT"
echo "Host:       $(hostname)"
echo "Started:    $(date)"
echo ""

# Build command
CMD="python scripts/run/run_emanuel_policy_suite.py"
CMD="$CMD --event_set $EVENT_SET"
CMD="$CMD --impact_dir $IMPACT_DIR"
CMD="$CMD --out $OUTPUT_ROOT"

# Add any additional arguments passed to this script
CMD="$CMD $@"

# Run policy suite
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
