#!/bin/bash
#SBATCH --job-name=windfields_emanuel
#SBATCH --output=logs/windfields_emanuel_%A_%a.out
#SBATCH --error=logs/windfields_emanuel_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# Parallel batch script to compute windfields from Kerry Emanuel TC tracks on Sherlock
# Each array job processes one track file in parallel

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

# Navigate to cats directory
cd scripts/hazard

# Define all track files (without .mat extension - will be added when calling Python)
TRACK_FILES=(
    "Simona_FLA_AL_canesm_20thcal"
    "Simona_FLA_AL_canesm_ssp245_2cal"
    "Simona_FLA_AL_canesm_ssp245cal"
    "Simona_FLA_AL_canesm_ssp585_2cal"
    "Simona_FLA_AL_canesm_ssp585cal"
    "Simona_FLA_AL_cnrm6_20thcal"
    "Simona_FLA_AL_cnrm6_ssp245_2cal"
    "Simona_FLA_AL_cnrm6_ssp245cal"
    "Simona_FLA_AL_cnrm6_ssp585_2cal"
    "Simona_FLA_AL_cnrm6_ssp585cal"
    "Simona_FLA_AL_ecearth6_20thcal"
    "Simona_FLA_AL_ecearth6_ssp245_2cal"
    "Simona_FLA_AL_ecearth6_ssp245cal"
    "Simona_FLA_AL_ecearth6_ssp585_2cal"
    "Simona_FLA_AL_ecearth6_ssp585cal"
    "Simona_FLA_AL_era5_reanalcal"
    "Simona_FLA_AL_ipsl6_20thcal"
    "Simona_FLA_AL_ipsl6_ssp245_2cal"
    "Simona_FLA_AL_ipsl6_ssp245cal"
    "Simona_FLA_AL_ipsl6_ssp585_2cal"
    "Simona_FLA_AL_ipsl6_ssp585cal"
    "Simona_FLA_AL_miroc6_20thcal"
    "Simona_FLA_AL_miroc6_ssp245_2cal"
    "Simona_FLA_AL_miroc6_ssp245cal"
    "Simona_FLA_AL_miroc6_ssp585_2cal"
    "Simona_FLA_AL_miroc6_ssp585cal"
)

# Get the track file for this array task
TRACK_FILE="${TRACK_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Processing track file: $TRACK_FILE"

# Run the windfield computation script for this specific file (add .mat extension)
python compute_windfields_emanuel.py "${TRACK_FILE}.mat"

exit_code=$?

echo "=========================================="
echo "Job finished at $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
