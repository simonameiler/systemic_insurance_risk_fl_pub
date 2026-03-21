#!/bin/bash
#SBATCH --job-name=test_emanuel_quick
#SBATCH --partition=serc
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=test_emanuel_quick_%j.out
#SBATCH --error=test_emanuel_quick_%j.err

# Quick test of Emanuel MC on Sherlock cluster (100 years)
# Usage:
#   Interactive: bash scripts/cluster/test_emanuel_mc_quick.sh
#   Batch:       sbatch scripts/cluster/test_emanuel_mc_quick.sh

set -e

echo "=========================================="
echo "EMANUEL MC QUICK TEST (100 years)"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo ""

# Navigate to repo
REPO_DIR="${HOME}/repos/systemic_insurance_risk_fl_pub"
cd "${REPO_DIR}"

# Pull latest changes
echo "Pulling latest code..."
git pull

# Activate conda environment (if not already active)
if [[ "$CONDA_DEFAULT_ENV" != "climada_env" ]]; then
    echo "Activating conda environment..."
    source activate climada_env || conda activate climada_env
else
    echo "climada_env already active"
fi

# Verify paths exist
IMPACT_DIR="/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts/FL_era5_reanalcal"
echo ""
echo "Checking cluster paths..."
echo "  Impact dir: ${IMPACT_DIR}"

if [ -d "${IMPACT_DIR}" ]; then
    echo "  [OK] Impact directory exists"
    echo "  Event files: $(ls ${IMPACT_DIR}/*.csv 2>/dev/null | wc -l)"
else
    echo "  [ERROR] Impact directory not found!"
    exit 1
fi

if [ -f "${IMPACT_DIR}/year_sets_N10000_seed42.csv" ]; then
    echo "  [OK] Year-sets file exists"
    echo "  Year-sets rows: $(wc -l < ${IMPACT_DIR}/year_sets_N10000_seed42.csv)"
else
    echo "  [ERROR] Year-sets file not found!"
    exit 1
fi

if [ -f "${IMPACT_DIR}/event_metadata.csv" ]; then
    echo "  [OK] Event metadata exists"
else
    echo "  [ERROR] Event metadata not found!"
    exit 1
fi

# Run quick test
echo ""
echo "=========================================="
echo "Running 100-year test..."
echo "=========================================="

python scripts/run/run_emanuel_monte_carlo.py \
    --event_set FL_era5_reanalcal \
    --n_years 100 \
    --seed 42 \
    --out results/test_runs \
    --run_label cluster_test

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo "Check results in: results/test_runs/"
ls -la results/test_runs/cluster_test_*/
