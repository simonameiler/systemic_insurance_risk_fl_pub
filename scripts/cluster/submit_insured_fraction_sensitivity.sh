#!/bin/bash
################################################################################
# Insured Fraction Sensitivity Analysis — Parallel Launcher
#
# Submits one SLURM job per insured fraction value (all run in parallel).
# Each job runs 10K seasons for a single fraction → ~8–10h each.
#
# Usage:
#   # Default: fractions 0.1 0.2 0.3 0.4 0.5 (5 parallel jobs)
#   bash scripts/cluster/submit_insured_fraction_sensitivity.sh
#
#   # Custom fractions
#   bash scripts/cluster/submit_insured_fraction_sensitivity.sh "0.1 0.3 0.5"
#
#   # Limit seasons (for testing)
#   bash scripts/cluster/submit_insured_fraction_sensitivity.sh "0.1 0.2 0.3 0.4 0.5" 500
################################################################################

set -e
set -u

FRACTIONS="${1:-0.1 0.2 0.3 0.4 0.5}"
N_YEARS="${2:-}"

PROJECT_DIR="${HOME}/repos/systemic_insurance_risk_fl_pub"
cd "${PROJECT_DIR}"
mkdir -p logs

echo "======================================================================"
echo "Submitting insured fraction sensitivity jobs"
echo "  Fractions: ${FRACTIONS}"
echo "  N_YEARS:   ${N_YEARS:-all}"
echo "======================================================================"
echo ""

for FRAC in ${FRACTIONS}; do
    N_YEARS_ARG=""
    if [[ -n "${N_YEARS}" ]]; then
        N_YEARS_ARG="--n_years ${N_YEARS}"
    fi

    JOB_ID=$(sbatch --parsable \
        --job-name="ins_f${FRAC}" \
        --output="logs/ins_frac_${FRAC}_%j.out" \
        --error="logs/ins_frac_${FRAC}_%j.err" \
        --time=48:00:00 \
        --mem=64G \
        --cpus-per-task=1 \
        --partition=serc \
        --wrap="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate climada_env
cd ${PROJECT_DIR}
echo \"Fraction: ${FRAC}  Job: \${SLURM_JOB_ID}  Node: \${SLURM_NODELIST}  Start: \$(date)\"
python scripts/run/run_insured_fraction_sensitivity.py --fractions ${FRAC} ${N_YEARS_ARG}
echo \"Finished: \$(date)\"
")

    echo "  Submitted frac=${FRAC}  →  SLURM job ${JOB_ID}"
done

echo ""
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
