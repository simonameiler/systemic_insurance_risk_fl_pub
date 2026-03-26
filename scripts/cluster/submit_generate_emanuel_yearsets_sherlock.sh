#!/bin/bash
#SBATCH --job-name=emanuel_yearsets
#SBATCH --output=logs/emanuel_yearsets_%A_%a.out
#SBATCH --error=logs/emanuel_yearsets_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

# Parallel batch script to generate year sets for Emanuel TC event sets
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
cd $HOME/systemic_insurance_risk_fl_pub

# Parameters
N_YEARS=10000
SEED=42

# Define all event sets
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
if [ -n "$1" ]; then
    # Command-line argument provided
    EVENT_SET="$1"
    echo "Using event set from command line: $EVENT_SET"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Array job - use task ID to index
    EVENT_SET="${EVENT_SETS[$SLURM_ARRAY_TASK_ID]}"
    echo "Using event set from array index $SLURM_ARRAY_TASK_ID: $EVENT_SET"
else
    echo "ERROR: No event set specified and not running as array job"
    echo "Usage: sbatch submit_generate_emanuel_yearsets_sherlock.sh <event_set>"
    echo "   OR: sbatch --array=0-25 submit_generate_emanuel_yearsets_sherlock.sh"
    exit 1
fi

echo "Generating year sets for: $EVENT_SET"
echo "Parameters: N_YEARS=$N_YEARS, SEED=$SEED"

# Run the year set generation script
python scripts/hazard/generate_emanuel_year_sets.py \
    --event_set "$EVENT_SET" \
    --n_years $N_YEARS \
    --seed $SEED \
    --sampling uniform

exit_code=$?

echo "=========================================="
echo "Job finished at $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
