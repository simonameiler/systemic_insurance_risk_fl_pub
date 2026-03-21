#!/bin/bash
#SBATCH --job-name=emanuel_era5_baseline
#SBATCH --partition=serc
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=emanuel_era5_baseline_%j.out
#SBATCH --error=emanuel_era5_baseline_%j.err

# Full Emanuel ERA5 baseline Monte Carlo (10,000 years)
# Usage: sbatch scripts/cluster/submit_emanuel_era5_baseline.sh

set -e

echo "=========================================="
echo "EMANUEL ERA5 BASELINE - 10,000 YEARS"
echo "=========================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo ""

# Navigate to repo
REPO_DIR="${HOME}/repos/systemic_insurance_risk_fl_pub"
cd "${REPO_DIR}"

# Activate conda environment
echo "Activating conda environment..."
module load anaconda3/2024.06
source activate climada_env || conda activate climada_env

# Verify Python and key packages
echo "Python: $(which python)"
python -c "import fl_risk_model; print(f'fl_risk_model loaded from: {fl_risk_model.__file__}')"

# Configuration
EVENT_SET="FL_era5_reanalcal"
N_YEARS=10000
SEED=42
OUT_DIR="results/emanue_mc_pub"
RUN_LABEL="era5_baseline"

echo ""
echo "Configuration:"
echo "  Event set: ${EVENT_SET}"
echo "  N years: ${N_YEARS}"
echo "  Seed: ${SEED}"
echo "  Output: ${OUT_DIR}/${RUN_LABEL}_*"
echo ""

# Run Monte Carlo
echo "Starting Monte Carlo simulation..."
echo "=========================================="

python scripts/run/run_emanuel_monte_carlo.py \
    --event_set ${EVENT_SET} \
    --n_years ${N_YEARS} \
    --seed ${SEED} \
    --out ${OUT_DIR} \
    --run_label ${RUN_LABEL}

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End time: $(date)"

# Show output
RUN_DIR=$(ls -td ${OUT_DIR}/${RUN_LABEL}_* | head -1)
echo "Results: ${RUN_DIR}"
echo ""
echo "Summary:"
cat "${RUN_DIR}/return_period_summary.csv" | head -3
