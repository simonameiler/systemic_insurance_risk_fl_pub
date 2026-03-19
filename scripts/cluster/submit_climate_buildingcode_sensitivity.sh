#!/bin/bash
#SBATCH --job-name=climate_buildingcode_sens
#SBATCH --output=logs/climate_buildingcode_sensitivity_%A_%a.out
#SBATCH --error=logs/climate_buildingcode_sensitivity_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --array=0-10

# Climate + Building Codes Sensitivity Analysis
# Sweep building code wind loss reduction from 0% to 50% in 5% increments
# Under SSP245 2050 climate change scenario

echo "=========================================="
echo "Climate + Building Codes Sensitivity Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Building code loss reduction percentages (0%, 5%, 10%, ..., 50%)
LOSS_REDUCTIONS=(0 5 10 15 20 25 30 35 40 45 50)

# Get loss reduction for this array task
LOSS_REDUCTION=${LOSS_REDUCTIONS[$SLURM_ARRAY_TASK_ID]}

echo "Building Code Wind Loss Reduction: ${LOSS_REDUCTION}%"
echo "Climate Scenario: SSP245 2050"

# Navigate to project directory
cd $HOME/repos/systemic_insurance_risk_fl || exit 1

# Activate conda environment
source ~/.bashrc
conda activate climada_env

# Verify environment
echo "Python: $(which python)"
echo "Python Version: $(python --version)"

# Check hazard file paths
echo ""
echo "Checking hazard file paths..."
python scripts/run/check_hazard_paths.py

# Run sensitivity analysis
echo ""
echo "Starting simulation..."
python scripts/run/run_climate_buildingcode_sensitivity.py \
    --loss_reduction ${LOSS_REDUCTION} \
    --seed 42

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Simulation completed successfully"
else
    echo ""
    echo "✗ Simulation failed with exit code $?"
    exit 1
fi

echo ""
echo "End Time: $(date)"
echo "=========================================="
