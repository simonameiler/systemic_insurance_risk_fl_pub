#!/bin/bash
#SBATCH --job-name=emanuel_impacts
#SBATCH --output=logs/emanuel_impacts_%A_%a.out
#SBATCH --error=logs/emanuel_impacts_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# Parallel batch script to precompute impacts from Kerry Emanuel TC event sets
# Each array job processes one event set in parallel

echo "=========================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules
module purge
module load python/3.9.0

# Activate conda environment
source activate climada_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Navigate to project directory
cd $HOME/systemic_insurance_risk_fl

# Define all event sets (corresponding to processed windfields)
EVENT_SETS=(
    "FL_canesm_20thcal"
    "FL_canesm_ssp245_2cal"
    "FL_canesm_ssp245cal"
    "FL_canesm_ssp585_2cal"
    "FL_canesm_ssp585cal"
    "FL_cnrm6_20thcal"
    "FL_cnrm6_ssp245_2cal"
    "FL_cnrm6_ssp245cal"
    "FL_cnrm6_ssp585_2cal"
    "FL_cnrm6_ssp585cal"
    "FL_ecearth6_20thcal"
    "FL_ecearth6_ssp245_2cal"
    "FL_ecearth6_ssp245cal"
    "FL_ecearth6_ssp585_2cal"
    "FL_ecearth6_ssp585cal"
    "FL_era5_reanalcal"
    "FL_ipsl6_20thcal"
    "FL_ipsl6_ssp245_2cal"
    "FL_ipsl6_ssp245cal"
    "FL_ipsl6_ssp585_2cal"
    "FL_ipsl6_ssp585cal"
    "FL_miroc6_20thcal"
    "FL_miroc6_ssp245_2cal"
    "FL_miroc6_ssp245cal"
    "FL_miroc6_ssp585_2cal"
    "FL_miroc6_ssp585cal"
)

# Get the event set for this array task
EVENT_SET="${EVENT_SETS[$SLURM_ARRAY_TASK_ID]}"

echo "Processing event set: $EVENT_SET"

# Run the impact computation script
python scripts/hazard/precompute_emanuel_tc_impacts.py "$EVENT_SET"

exit_code=$?

echo "=========================================="
echo "Job finished at $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
