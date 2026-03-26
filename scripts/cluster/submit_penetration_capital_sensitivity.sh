#!/bin/bash
#SBATCH --job-name=pen_cap_sens
#SBATCH --output=logs/penetration_capital_sensitivity_%A_%a.out
#SBATCH --error=logs/penetration_capital_sensitivity_%A_%a.err
#SBATCH --account=bakerjw
#SBATCH --partition=serc
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-7

#==============================================================================
# Penetration Capital Sensitivity Analysis - Cluster Submission
#==============================================================================
# 
# Runs Monte Carlo simulations across different capital scaling multipliers
# to find the optimal capital requirement that stabilizes defaults under
# increased insurance penetration.
#
# Array indices map to capital multipliers:
#   0 → 1.0x (proportional, baseline)
#   1 → 1.2x
#   2 → 1.4x
#   3 → 1.6x
#   4 → 1.8x
#   5 → 2.0x
#   6 → 2.5x
#   7 → 3.0x
#
# Usage:
#   # Submit all multipliers as array job
#   sbatch submit_penetration_capital_sensitivity.sh
#
#   # Submit single multiplier (e.g., 1.5x)
#   sbatch --array=3 submit_penetration_capital_sensitivity.sh
#
# Runtime: ~8-12 hours per multiplier (10,000 years)
# Total array job runtime: Parallel execution across 8 nodes
#==============================================================================

# Create logs directory
mkdir -p logs

echo "======================================================================"
echo "SLURM Job Info"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"
echo "======================================================================"
echo ""

#==============================================================================
# Environment Setup
#==============================================================================

# Activate conda environment
source ~/.bashrc
conda activate climada_env

# Verify Python
echo "Python: $(which python)"
python --version
echo ""

# Navigate to repo
cd ~/repos/systemic_insurance_risk_fl_pub || exit 1

#==============================================================================
# Capital Multiplier Configuration
#==============================================================================

# Map array index to capital multiplier
MULTIPLIERS=(1.0 1.2 1.4 1.6 1.8 2.0 2.5 3.0)
CAPITAL_MULTIPLIER=${MULTIPLIERS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "Capital Sensitivity Configuration"
echo "======================================================================"
echo "Mode: Stochastic (full Emanuel TC event set)"
echo "Capital multiplier: ${CAPITAL_MULTIPLIER}x"
echo "Penetration scenario: MAJOR (40→60% wind, 11→30% flood)"
echo "Years: 10,000"
echo "Random seed: 42"
echo "Output: penetration_capital_m${CAPITAL_MULTIPLIER}_baseline_<timestamp>"
echo "======================================================================"
echo ""

#==============================================================================
# Run Monte Carlo Simulation
#==============================================================================

echo "Starting Monte Carlo simulation with multiplier ${CAPITAL_MULTIPLIER}x..."
echo ""

# Create run marker
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_MARKER="logs/running_pen_cap_m${CAPITAL_MULTIPLIER}_${TIMESTAMP}.marker"
touch ${RUN_MARKER}

# Run the simulation
python scripts/run/run_penetration_capital_sensitivity.py \
    --multiplier ${CAPITAL_MULTIPLIER} \
    --n_years 10000 \
    --seed 42

EXIT_CODE=$?

# Remove marker
rm -f ${RUN_MARKER}

# Report status
echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Capital multiplier ${CAPITAL_MULTIPLIER}x completed"
    echo "Exit code: 0"
else
    echo "FAILED: Capital multiplier ${CAPITAL_MULTIPLIER}x failed"
    echo "Exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "======================================================================"

exit $EXIT_CODE
