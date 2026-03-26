#!/bin/bash
#SBATCH --job-name=var_decomp
#SBATCH --output=logs/var_decomp_%j.out
#SBATCH --error=logs/var_decomp_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc

################################################################################
# Variance Decomposition: hazard variability vs. parameter uncertainty
#
# Usage:
#   # Run both fixed-param (10K) and nested MC (300×50)
#   sbatch submit_variance_decomposition.sh
#
#   # Run only nested MC with custom M and K
#   sbatch submit_variance_decomposition.sh nested 300 50
#
#   # Run only fixed-param sweep
#   sbatch submit_variance_decomposition.sh fixed
#
# Arguments (all optional):
#   $1  Mode: "fixed", "nested", or "both" (default: both)
#   $2  M: number of seasons for nested MC (default: 300)
#   $3  K: parameter draws per season  (default: 50)
################################################################################

set -e
set -u

echo "======================================================================"
echo "SLURM Job Info"
echo "======================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "======================================================================"
echo ""

# Parse arguments
MODE="${1:-both}"
M="${2:-300}"
K="${3:-50}"

# Set up paths
PROJECT_DIR="${HOME}/repos/systemic_insurance_risk_fl_pub"
cd "${PROJECT_DIR}"
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

echo "Python: $(which python)"
echo "Mode: ${MODE}"
echo "M (seasons): ${M}"
echo "K (draws):   ${K}"
echo ""

# Year-sets CSV (cluster path)
YEAR_SETS="/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts/FL_era5_reanalcal/year_sets_N10000_seed42.csv"
if [ ! -f "${YEAR_SETS}" ]; then
    echo "ERROR: Year-sets CSV not found: ${YEAR_SETS}"
    exit 1
fi

# Output directory
OUT_DIR="${PROJECT_DIR}/results/mc_runs"
mkdir -p "${OUT_DIR}"

echo "Year-sets: ${YEAR_SETS}"
echo "Output:    ${OUT_DIR}"
echo ""

# Run variance decomposition
python scripts/run/run_variance_decomposition.py \
    --mode "${MODE}" \
    --M "${M}" \
    --K "${K}" \
    --seed 42 \
    --out_dir "${OUT_DIR}" \
    --year_sets_csv "${YEAR_SETS}"

echo ""
echo "======================================================================"
echo "Finished at $(date)"
echo "======================================================================"
