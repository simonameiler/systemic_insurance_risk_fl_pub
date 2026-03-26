#!/bin/bash
################################################################################
# Submit Emanuel TC Policy Suite in PARALLEL on Sherlock
#
# This script submits 4 separate jobs (one per policy) to run in parallel.
# Much faster than sequential execution (12-18h per job vs 48-72h total).
#
# Usage:
#   bash scripts/cluster/submit_policy_suite_parallel.sh FL_era5_reanalcal
#
# Or customize policies:
#   bash scripts/cluster/submit_policy_suite_parallel.sh FL_era5_reanalcal \
#       "baseline" "market_exit_moderate" "penetration_major" "building_codes_major"
################################################################################

set -e

# Check if sbatch is available (Sherlock only)
if ! command -v sbatch &> /dev/null; then
    echo "ERROR: This script must be run on Sherlock (sbatch not found)"
    echo ""
    echo "To use this script:"
    echo "  1. SSH to Sherlock: ssh <sunetid>@login.sherlock.stanford.edu"
    echo "  2. Navigate to repo: cd ~/repos/systemic_insurance_risk_fl_pub"
    echo "  3. Run: bash scripts/cluster/submit_policy_suite_parallel.sh FL_era5_reanalcal"
    exit 1
fi

# Event set (required)
if [ -z "$1" ]; then
    echo "ERROR: Event set required"
    echo "Usage: bash submit_policy_suite_parallel.sh <event_set> [policy1] [policy2] ..."
    exit 1
fi

EVENT_SET="$1"
shift  # Remove first argument

# Default policies if none specified
if [ $# -eq 0 ]; then
    POLICIES=("baseline" "market_exit_moderate" "penetration_major" "building_codes_major")
else
    POLICIES=("$@")
fi

# Path to submission script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_emanuel_policy_suite_sherlock.sh"

echo "Script directory: $SCRIPT_DIR"
echo "Submit script: $SUBMIT_SCRIPT"

if [ ! -f "$SUBMIT_SCRIPT" ]; then
    echo "ERROR: Submission script not found: $SUBMIT_SCRIPT"
    echo ""
    echo "Expected location: scripts/cluster/submit_emanuel_policy_suite_sherlock.sh"
    echo "Current directory: $(pwd)"
    ls -la "$SCRIPT_DIR" | grep submit
    exit 1
fi

echo ""
echo "==========================================================================="
echo "SUBMITTING PARALLEL POLICY JOBS"
echo "==========================================================================="
echo "Event set: $EVENT_SET"
echo "Policies:  ${POLICIES[@]}"
echo ""

# Submit jobs
JOB_IDS=()
for POLICY in "${POLICIES[@]}"; do
    echo "Submitting: $POLICY"
    echo "  Command: sbatch $SUBMIT_SCRIPT $EVENT_SET --policies $POLICY"
    
    # Submit job and capture output
    JOB_OUTPUT=$(sbatch "$SUBMIT_SCRIPT" "$EVENT_SET" --policies "$POLICY" 2>&1)
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ❌ ERROR: sbatch failed with exit code $EXIT_CODE"
        echo "  Output: $JOB_OUTPUT"
        continue
    fi
    
    # Extract job ID (format: "Submitted batch job 12345")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+' | tail -1)
    
    if [ -n "$JOB_ID" ]; then
        JOB_IDS+=("$JOB_ID")
        echo "  ✅ Submitted → Job ID: $JOB_ID"
    else
        echo "  ⚠️  WARNING: Could not extract job ID"
        echo "  Full output: $JOB_OUTPUT"
    fi
done

echo ""
echo "==========================================================================="
echo "SUBMITTED ${#JOB_IDS[@]} JOBS"
echo "==========================================================================="
echo ""

# Print job IDs
if [ ${#JOB_IDS[@]} -gt 0 ]; then
    echo "Job IDs: ${JOB_IDS[@]}"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Check specific job:"
    echo "  squeue -j ${JOB_IDS[0]}"
    echo ""
    echo "View logs:"
    echo "  tail -f logs/emanuel_policy_${JOB_IDS[0]}.out"
    echo "  tail -f logs/emanuel_policy_${JOB_IDS[0]}.err"
    echo ""
    echo "Cancel all jobs:"
    echo "  scancel ${JOB_IDS[@]}"
else
    echo "⚠️  No jobs submitted successfully"
    exit 1
fi

echo "==========================================================================="
