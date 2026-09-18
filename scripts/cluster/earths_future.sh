#!/bin/bash
################################################################################
# earths_future.sh -- thin Slurm wrapper for the FHCF-patched cluster
# campaign (Earth's Future revision).
#
# Commands:
#   check          Read-only preflight: code revision, environment, all 26
#                  required event-set caches, active-input hashes, the
#                  existing test suite. Never regenerates hazard data.
#   pilot          Submit the paired ERA5 comparison (old_both_bugs vs.
#                  both_fixed, first 200 year IDs, seed 42) as ONE Slurm job.
#   pilot-report   Check that job's Slurm status and financial reconciliation.
#                  Refuses to pass while the job is pending/running/failed.
#   production     Submit the full 106-run inventory as a bounded-concurrency
#                  Slurm array job. Gated on a passing pilot-report for the
#                  SAME code revision and inputs; refuses accidental
#                  duplicate submission.
#   status         Query Slurm for pilot/production job states. Never infers
#                  success from output-directory existence alone.
#   postprocess    Regenerate the section5-9 tables from resolved, validated
#                  production outputs. Reports precisely what still needs
#                  manual work rather than claiming full completion.
#
# This is a thin wrapper: the actual science lives in
# scripts/run/run_*.py and scripts/earths_future_revision/section*.py.
# scripts/cluster/earths_future_lib.py is a small helper for the JSON
# manifest / hashing / CSV-validation / squeue-sacct bookkeeping a shell
# script does awkwardly; it does not itself submit any Slurm job.
#
# Nothing here runs "git pull" or switches branches. Every submitted job
# records its own code revision (commit + dirty-tracked-source check) at
# EXECUTION time, in addition to the revision `check`/`pilot`/`production`
# record at SUBMISSION time -- compare the two if you suspect the checkout
# changed while jobs were queued.
#
# Usage:
#   scripts/cluster/earths_future.sh check
#   scripts/cluster/earths_future.sh pilot
#   scripts/cluster/earths_future.sh pilot-report [--manifest PATH]
#   scripts/cluster/earths_future.sh production [--force] [--concurrency N]
#   scripts/cluster/earths_future.sh status [--manifest PATH]
#   scripts/cluster/earths_future.sh postprocess [--manifest PATH]
#
# Environment overrides (all optional; `check` validates, does not assume):
#   EF_CAMPAIGN      fhcf (default) or catbond; prefer catbond_revision.sh.
#   EF_PROJECT_DIR   Sherlock checkout root. Default: derived from this
#                    script's own location (../.. from scripts/cluster/).
#   EF_IMPACT_ROOT   Impact-cache root. Default:
#                    /home/groups/bakerjw/smeiler/climada_data/data/impact/impacts
#   EF_PARTITION     Slurm partition. Default: serc
#   EF_CONCURRENCY   Max concurrent production array tasks. Default: 20
#   EF_PYTHON        Python used to run earths_future_lib.py (not the jobs
#                    themselves, which always activate climada_env
#                    explicitly). Default: python3
################################################################################
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${EF_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
IMPACT_ROOT="${EF_IMPACT_ROOT:-/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts}"
PARTITION="${EF_PARTITION:-serc}"
CONCURRENCY="${EF_CONCURRENCY:-20}"
PY="${EF_PYTHON:-python3}"
LIB="${SCRIPT_DIR}/earths_future_lib.py"
CAMPAIGN="${EF_CAMPAIGN:-fhcf}"
case "${CAMPAIGN}" in
  fhcf|catbond) ;;
  *) echo "[ef] Unknown campaign: ${CAMPAIGN}" >&2; exit 1 ;;
esac
ENTRY_POINT="${SCRIPT_DIR}/earths_future.sh"
if [[ "${CAMPAIGN}" == catbond ]]; then
  ENTRY_POINT="${SCRIPT_DIR}/catbond_revision.sh"
fi

LOG_DIR="${PROJECT_DIR}/logs"
CLUSTER_OUT="${PROJECT_DIR}/results/earths_future_revision/${CAMPAIGN}_cluster"
MC_OUT_ROOT="${PROJECT_DIR}/results/mc_runs_${CAMPAIGN}_patched"
MANIFEST_DIR="${CLUSTER_OUT}/manifests"
REPORT_DIR="${CLUSTER_OUT}/reports"
PILOT_LATEST="${MANIFEST_DIR}/pilot_manifest_latest.json"
PRODUCTION_LATEST="${MANIFEST_DIR}/production_manifest_latest.json"

mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}" "${REPORT_DIR}"

usage() {
  cat <<EOF
Usage: ${ENTRY_POINT} {check|pilot|pilot-report|production|status|postprocess} [options]
See the header comment of this file, or
docs/earths_future_revision/cluster_runbook.md, for details on each command.
EOF
}

require_sbatch() {
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "[ef] ERROR: 'sbatch' not found on PATH. This command submits Slurm jobs and" >&2
    echo "[ef]        must run on a Sherlock login or dev node, not locally." >&2
    return 1
  fi
}

conda_activate_snippet() {
  cat <<'EOF'
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh" || exit 1
  conda activate climada_env || exit 1
else
  echo "[ef] ERROR: could not find conda / climada_env" >&2
  exit 1
fi
EOF
}

# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
cmd_check() {
  echo "[ef] check: code revision, environment, 26 required event-set"
  echo "[ef]        caches, active-input hashes, existing test suite."
  echo "[ef]   PROJECT_DIR=${PROJECT_DIR}"
  echo "[ef]   IMPACT_ROOT=${IMPACT_ROOT}  (validated below, not assumed correct)"
  local report="${REPORT_DIR}/check_$(date +%Y%m%d_%H%M%S).json"
  "${PY}" "${LIB}" preflight --impact-root "${IMPACT_ROOT}" --report-out "${report}"
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[ef] check PASSED. Safe to run 'pilot' next."
  else
    echo "[ef] check FAILED -- see problems above and ${report}. Not safe to run pilot/production." >&2
  fi
  return $rc
}

# --------------------------------------------------------------------------- #
# pilot
# --------------------------------------------------------------------------- #
cmd_pilot() {
  require_sbatch || return 1
  local ts out_root script job_id
  ts="$(date +%Y%m%d_%H%M%S)"
  out_root="${MC_OUT_ROOT}/pilot_era5_${ts}"
  mkdir -p "${out_root}"

  local impact_dir="${IMPACT_ROOT}/FL_era5_reanalcal"
  echo "[ef] pilot: paired ERA5 ${CAMPAIGN} comparison, first 200 year IDs, seed 42"
  echo "[ef]   impact_dir=${impact_dir}"
  echo "[ef]   out_root=${out_root}"

  script="${out_root}/submit_pilot.sh"
  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=ef_pilot_era5"
    echo "#SBATCH --output=${LOG_DIR}/ef_pilot_era5_%j.out"
    echo "#SBATCH --error=${LOG_DIR}/ef_pilot_era5_%j.err"
    echo "#SBATCH --time=12:00:00"
    echo "#SBATCH --mem=32G"
    echo "#SBATCH --cpus-per-task=1"
    echo "#SBATCH --partition=${PARTITION}"
    echo "set -u"
    echo 'echo "[ef] pilot job=${SLURM_JOB_ID} node=${SLURM_NODELIST} start=$(date)"'
    echo "cd '${PROJECT_DIR}' || exit 1"
    echo 'EF_JOB_COMMIT=$(git rev-parse HEAD) || exit 1'
    echo 'EF_JOB_DESCRIBE=$(git describe --always --dirty) || exit 1'
    echo 'echo "[ef] code_revision_at_execution commit=${EF_JOB_COMMIT} describe=${EF_JOB_DESCRIBE}"'
    conda_activate_snippet
    echo "cd '${PROJECT_DIR}'"
    if [[ "${CAMPAIGN}" == catbond ]]; then
      echo "python scripts/earths_future_revision/catbond_pilot_era5.py --impact-dir '${impact_dir}' --n-years 200 --seed 42 --out-root '${out_root}'"
    else
    echo "python scripts/earths_future_revision/fhcf_pilot_era5.py \\"
    echo "  --event-set FL_era5_reanalcal \\"
    echo "  --impact-dir '${impact_dir}' \\"
    echo "  --n-years 200 --seed 42 \\"
    echo "  --out-root '${out_root}' \\"
    echo "  --variants old_both_bugs both_fixed"
    fi
    echo 'EXIT_CODE=$?'
    echo 'echo "[ef] pilot finished=$(date) exit=${EXIT_CODE}"'
    echo 'exit ${EXIT_CODE}'
  } > "${script}"
  chmod +x "${script}"

  job_id=$(sbatch --parsable "${script}")
  if [[ -z "${job_id}" ]]; then
    echo "[ef] ERROR: sbatch did not return a job id" >&2
    return 1
  fi
  echo "[ef] submitted Slurm job ${job_id}"

  local manifest="${MANIFEST_DIR}/pilot_manifest_${ts}.json"
  "${PY}" "${LIB}" manifest-init --manifest "${manifest}" --campaign pilot \
    --seed 42 --out-root "${out_root}" || return 1
  "${PY}" "${LIB}" manifest-add-job --manifest "${manifest}" \
    --name era5_pilot_paired --job-id "${job_id}" --expected-seasons 200 \
    --output-dir "${out_root}" || return 1
  ln -sf "$(basename "${manifest}")" "${PILOT_LATEST}"

  echo "[ef] wrote manifest: ${manifest}"
  echo "[ef]   (symlinked as $(basename "${PILOT_LATEST}"))"
  echo "[ef] check progress with: ${ENTRY_POINT} status"
  echo "[ef] once Slurm shows it completed, run: ${ENTRY_POINT} pilot-report"
}

# --------------------------------------------------------------------------- #
# pilot-report (also used internally as production's gate)
# --------------------------------------------------------------------------- #
run_pilot_report() {
  local manifest="${1:-${PILOT_LATEST}}"
  if [[ ! -e "${manifest}" ]]; then
    echo "[ef] ERROR: pilot manifest not found: ${manifest}. Run '${ENTRY_POINT} pilot' first." >&2
    return 1
  fi
  echo "[ef] pilot-report: ${manifest}"

  local status_json queue_state job_id out_root
  status_json=$("${PY}" "${LIB}" job-status --manifest "${manifest}")
  echo "${status_json}"
  queue_state=$(printf '%s' "${status_json}" | "${PY}" -c "import json,sys; print(json.load(sys.stdin)['jobs'][0]['queue_state'])")
  job_id=$(printf '%s' "${status_json}" | "${PY}" -c "import json,sys; print(json.load(sys.stdin)['jobs'][0]['slurm_job_id'])")
  out_root=$("${PY}" -c "import json; print(json.load(open('${manifest}'))['jobs'][0]['output_dir'])")

  if [[ "${queue_state}" != completed* ]]; then
    echo "[ef] pilot job ${job_id} is not completed yet (queue_state=${queue_state})." >&2
    echo "[ef] pilot-report FAILS while pending/running/failed by design." >&2
    return 1
  fi

  local report="${REPORT_DIR}/pilot_report_$(date +%Y%m%d_%H%M%S).json"
  if [[ "${CAMPAIGN}" == catbond ]]; then
    "${PY}" "${PROJECT_DIR}/scripts/earths_future_revision/catbond_pilot_era5.py" \
      --report --out-root "${out_root}" --n-years 200 --report-out "${report}"
  else
  "${PY}" "${LIB}" compare-pilot \
    --old-dir "${out_root}/old_both_bugs" \
    --new-dir "${out_root}/both_fixed" \
    --expected-seasons 200 \
    --report-out "${report}"
  fi
  local rc=$?
  echo "[ef] full pilot-report: ${report}"
  if [[ $rc -eq 0 ]]; then
    echo "[ef] pilot-report PASSED (job ${job_id}): row counts match, no error rows, upstream"
    echo "[ef] draws (gross loss / NFIP) identical between variants as required."
  else
    echo "[ef] pilot-report FAILED -- see ${report}. Do not run production yet." >&2
  fi
  return $rc
}

cmd_pilot_report() {
  local manifest="${PILOT_LATEST}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --manifest) manifest="$2"; shift 2 ;;
      *) echo "[ef] unknown pilot-report option: $1" >&2; return 1 ;;
    esac
  done
  run_pilot_report "${manifest}"
}

# --------------------------------------------------------------------------- #
# production
# --------------------------------------------------------------------------- #
cmd_production() {
  require_sbatch || return 1
  local force_flag="" concurrency="${CONCURRENCY}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force) force_flag="--force"; shift ;;
      --concurrency) concurrency="$2"; shift 2 ;;
      *) echo "[ef] unknown production option: $1" >&2; return 1 ;;
    esac
  done

  echo "[ef] production: gating on a passing pilot-report (same code revision + inputs)"
  if ! run_pilot_report "${PILOT_LATEST}"; then
    echo "[ef] ERROR: pilot-report did not pass. Run '${ENTRY_POINT} pilot' then '${ENTRY_POINT} pilot-report' and" >&2
    echo "[ef]        resolve any failures before '${ENTRY_POINT} production'." >&2
    return 1
  fi

  echo "[ef] production: checking for a duplicate in-flight/completed campaign"
  if ! "${PY}" "${LIB}" guard-duplicate --manifest "${PRODUCTION_LATEST}" ${force_flag}; then
    echo "[ef] Refusing to submit. Pass '${ENTRY_POINT} production --force' if this is intentional" >&2
    echo "[ef] (e.g. deliberately re-running under the same code+inputs)." >&2
    return 1
  fi

  local ts out_root
  ts="$(date +%Y%m%d_%H%M%S)"
  out_root="${MC_OUT_ROOT}/production_${ts}"
  mkdir -p "${out_root}"

  local tasks_tsv="${MANIFEST_DIR}/production_tasks_${ts}.tsv"
  "${PY}" "${LIB}" list-jobs --out-root "${out_root}" --impact-root "${IMPACT_ROOT}" > "${tasks_tsv}"
  local n_jobs
  n_jobs=$(wc -l < "${tasks_tsv}" | tr -d ' ')
  if [[ "${n_jobs}" -ne 106 ]]; then
    echo "[ef] ERROR: expected 106 production jobs, list-jobs produced ${n_jobs}. Refusing to submit." >&2
    return 1
  fi
  echo "[ef] production: ${n_jobs} jobs -> ${tasks_tsv}"
  echo "[ef]   out_root=${out_root}"
  echo "[ef]   concurrency=${concurrency}  partition=${PARTITION}"

  local array_script="${MANIFEST_DIR}/production_array_${ts}.sh"
  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=ef_production"
    echo "#SBATCH --output=${LOG_DIR}/ef_production_%A_%a.out"
    echo "#SBATCH --error=${LOG_DIR}/ef_production_%A_%a.err"
    echo "#SBATCH --time=48:00:00"
    echo "#SBATCH --mem=64G"
    echo "#SBATCH --cpus-per-task=1"
    echo "#SBATCH --partition=${PARTITION}"
    echo "#SBATCH --array=0-$((n_jobs - 1))%${concurrency}"
    echo "set -u"
    echo "TASKS_TSV='${tasks_tsv}'"
    echo 'LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${TASKS_TSV}")'
    echo 'NAME=$(printf "%s" "${LINE}" | cut -f1)'
    echo 'OUT_DIR=$(printf "%s" "${LINE}" | cut -f3)'
    echo 'CMD=$(printf "%s" "${LINE}" | cut -f4-)'
    echo 'echo "[ef] task=${SLURM_ARRAY_TASK_ID} name=${NAME} job=${SLURM_JOB_ID} node=${SLURM_NODELIST} start=$(date)"'
    echo "cd '${PROJECT_DIR}' || exit 1"
    echo 'EF_JOB_COMMIT=$(git rev-parse HEAD) || exit 1'
    echo 'EF_JOB_DESCRIBE=$(git describe --always --dirty) || exit 1'
    echo 'echo "[ef] code_revision_at_execution commit=${EF_JOB_COMMIT} describe=${EF_JOB_DESCRIBE}"'
    conda_activate_snippet
    echo "cd '${PROJECT_DIR}'"
    echo 'mkdir -p "${OUT_DIR}"'
    echo 'echo "[ef] command: ${CMD}"'
    echo 'eval "${CMD}"'
    echo 'EXIT_CODE=$?'
    echo 'echo "[ef] task=${SLURM_ARRAY_TASK_ID} name=${NAME} finished=$(date) exit=${EXIT_CODE}"'
    echo 'exit ${EXIT_CODE}'
  } > "${array_script}"
  chmod +x "${array_script}"

  local job_id
  job_id=$(sbatch --parsable "${array_script}")
  if [[ -z "${job_id}" ]]; then
    echo "[ef] ERROR: sbatch did not return a job id" >&2
    return 1
  fi
  echo "[ef] submitted Slurm array job ${job_id} (array 0-$((n_jobs - 1))%${concurrency})"

  local manifest="${MANIFEST_DIR}/production_manifest_${ts}.json"
  "${PY}" "${LIB}" manifest-init --manifest "${manifest}" --campaign production \
    --seed 42 --out-root "${out_root}" --concurrency-limit "${concurrency}" || return 1

  local i=0
  while IFS=$'\t' read -r name expected out_dir _cmd; do
    "${PY}" "${LIB}" manifest-add-job --manifest "${manifest}" \
      --name "${name}" --job-id "${job_id}_${i}" --expected-seasons "${expected}" \
      --output-dir "${out_dir}" || return 1
    i=$((i + 1))
  done < "${tasks_tsv}"

  ln -sf "$(basename "${manifest}")" "${PRODUCTION_LATEST}"
  echo "[ef] wrote manifest: ${manifest}"
  echo "[ef]   (symlinked as $(basename "${PRODUCTION_LATEST}"))"
  echo "[ef] recorded ${i} jobs. Check progress with: ${ENTRY_POINT} status"
}

# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
cmd_status() {
  local manifest=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --manifest) manifest="$2"; shift 2 ;;
      *) echo "[ef] unknown status option: $1" >&2; return 1 ;;
    esac
  done

  if [[ -n "${manifest}" ]]; then
    "${PY}" "${LIB}" job-status --manifest "${manifest}"
    return $?
  fi

  local any=0
  for m in "${PILOT_LATEST}" "${PRODUCTION_LATEST}"; do
    if [[ -e "${m}" ]]; then
      any=1
      echo "=== ${m} ==="
      "${PY}" "${LIB}" job-status --manifest "${m}"
      echo ""
    fi
  done
  if [[ ${any} -eq 0 ]]; then
    echo "[ef] no pilot or production manifest found yet. Run '${ENTRY_POINT} pilot' or '${ENTRY_POINT} production' first."
  fi
}

# --------------------------------------------------------------------------- #
# postprocess
# --------------------------------------------------------------------------- #
cmd_postprocess() {
  local manifest="${PRODUCTION_LATEST}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --manifest) manifest="$2"; shift 2 ;;
      *) echo "[ef] unknown postprocess option: $1" >&2; return 1 ;;
    esac
  done
  if [[ ! -e "${manifest}" ]]; then
    echo "[ef] ERROR: no production manifest found: ${manifest}. Run '${ENTRY_POINT} production' first." >&2
    return 1
  fi

  local ts resolved_json
  ts="$(date +%Y%m%d_%H%M%S)"
  resolved_json="${REPORT_DIR}/postprocess_resolved_${ts}.json"
  echo "[ef] postprocess: resolving completed+validated run directories from ${manifest}"
  "${PY}" "${LIB}" resolve-manifest --manifest "${manifest}" --report-out "${resolved_json}"
  local resolve_rc=$?
  if [[ ${resolve_rc} -ne 0 ]]; then
    echo "[ef] NOTE: not every production job is completed+valid yet (see ${resolved_json})." >&2
    echo "[ef]       Proceeding with whichever tables/sections have everything they need;" >&2
    echo "[ef]       anything that doesn't will be listed as skipped below, not silently" >&2
    echo "[ef]       substituted or reported as done." >&2
  fi

  local pp_out="${CLUSTER_OUT}/postprocess_${ts}"
  mkdir -p "${pp_out}"
  local maps_dir="${pp_out}/scenario_maps"
  "${PY}" "${LIB}" build-scenario-maps --resolved "${resolved_json}" --out-dir "${maps_dir}"
  local maps_rc=$?

  local skipped=()
  local ran=()

  local era5_baseline_dir
  era5_baseline_dir=$("${PY}" -c "
import json
d = json.load(open('${resolved_json}'))['resolved']
print(d.get('era5_baseline', ''))
")

  cd "${PROJECT_DIR}/scripts/earths_future_revision" || return 1

  if [[ -n "${era5_baseline_dir}" ]]; then
    echo "[ef] postprocess: section5 (seasonal return-level table)"
    if "${PY}" section5_return_periods.py --iterations "${era5_baseline_dir}/iterations.csv" \
        --out-dir "${pp_out}/tables"; then
      ran+=("section5_return_periods")
    else
      skipped+=("section5_return_periods (script failed; see output above)")
    fi

    echo "[ef] postprocess: section6 (decomposition)"
    if "${PY}" section6_decomposition.py --iterations "${era5_baseline_dir}/iterations.csv" \
        --out-dir "${pp_out}/decomposition" --fig-dir "${pp_out}/figures"; then
      ran+=("section6_decomposition")
    else
      skipped+=("section6_decomposition (script failed; see output above)")
    fi
  else
    skipped+=("section5_return_periods, section6_decomposition (need job 'era5_baseline', not yet resolved)")
  fi

  echo "[ef] postprocess: section7 NFIP allocation (independent of MC run outputs)"
  if "${PY}" section7_nfip_allocation.py --out-dir "${pp_out}/nfip_allocation"; then
    ran+=("section7_nfip_allocation")
  else
    skipped+=("section7_nfip_allocation (script failed; see output above)")
  fi

  local frac_consolidated="${pp_out}/insured_fraction_consolidated"
  mkdir -p "${frac_consolidated}"
  local n_frac_found=0
  for f in 0.1 0.2 0.3 0.4 0.5; do
    local job_name="insured_fraction_${f}"
    local frac_dir
    frac_dir=$("${PY}" -c "
import json
d = json.load(open('${resolved_json}'))['resolved']
print(d.get('${job_name}', ''))
")
    if [[ -n "${frac_dir}" ]]; then
      local src="${frac_dir}/iterations_frac_$(printf '%.2f' "${f}").csv"
      if [[ -f "${src}" ]]; then
        cp "${src}" "${frac_consolidated}/"
        n_frac_found=$((n_frac_found + 1))
      fi
    fi
  done
  if [[ -n "${era5_baseline_dir}" && ${n_frac_found} -eq 5 ]]; then
    echo "[ef] postprocess: section7 insured-fraction sensitivity (5/5 fractions found)"
    if "${PY}" section7_insured_fraction.py --frac-dir "${frac_consolidated}" \
        --baseline "${era5_baseline_dir}/iterations.csv" --out-dir "${pp_out}/insured_fraction"; then
      ran+=("section7_insured_fraction")
    else
      skipped+=("section7_insured_fraction (script failed; see output above)")
    fi
  else
    local have_baseline_word="no"
    [[ -n "${era5_baseline_dir}" ]] && have_baseline_word="a"
    skipped+=("section7_insured_fraction (found ${n_frac_found}/5 fraction runs and ${have_baseline_word} baseline; need all 5 plus era5_baseline)")
  fi

  if [[ -f "${maps_dir}/historical_scenario_map.json" ]]; then
    local n_hist
    n_hist=$("${PY}" -c "import json; print(len(json.load(open('${maps_dir}/historical_scenario_map.json'))))")
    echo "[ef] postprocess: section8 historical scenarios (${n_hist}/7 resolved)"
    if "${PY}" section8_historical_and_variance.py --scenario-map "${maps_dir}/historical_scenario_map.json" \
        --out-dir "${pp_out}/historical"; then
      ran+=("section8_historical_and_variance (${n_hist}/7 scenarios)")
      if [[ "${n_hist}" -lt 7 ]]; then
        skipped+=("section8_historical_and_variance: only ${n_hist}/7 historical scenarios were available; table_S3 is partial, not the full corrected SI Table S3")
      fi
    else
      skipped+=("section8_historical_and_variance (script failed; see output above)")
    fi
  else
    skipped+=("section8_historical_and_variance (no historical scenario resolved yet)")
  fi

  if [[ -f "${maps_dir}/climate_policy_scenario_map.json" ]]; then
    local n_cp
    n_cp=$("${PY}" -c "import json; print(len(json.load(open('${maps_dir}/climate_policy_scenario_map.json'))))")
    echo "[ef] postprocess: section9 climate/policy means + exceedance probabilities (${n_cp}/4 resolved)"
    if "${PY}" section9_climate_policy_tables.py --scenario-map "${maps_dir}/climate_policy_scenario_map.json" \
        --out-dir "${pp_out}/climate_policy"; then
      ran+=("section9_climate_policy_tables (${n_cp}/4 scenarios)")
      if [[ "${n_cp}" -lt 4 ]]; then
        skipped+=("section9_climate_policy_tables: only ${n_cp}/4 ERA5 baseline+policy runs were available; table_S4/S5 are partial")
      fi
    else
      skipped+=("section9_climate_policy_tables (script failed; see output above)")
    fi
  else
    skipped+=("section9_climate_policy_tables (era5_baseline / policy runs not yet resolved)")
  fi

  cd "${PROJECT_DIR}" || return 1

  echo ""
  echo "======================================================================"
  echo "[ef] postprocess summary -> ${pp_out}"
  echo "======================================================================"
  echo "Ran:"
  for r in "${ran[@]:-}"; do [[ -n "$r" ]] && echo "  [OK] $r"; done
  echo ""
  echo "NOT run / needs manual attention:"
  for s in "${skipped[@]:-}"; do [[ -n "$s" ]] && echo "  [--] $s"; done
  echo "  [--] building-code curves / offset estimates (13 levels x 5 GCMs): no existing"
  echo "       script in scripts/analysis/ takes a fresh output-directory override; the"
  echo "       closest candidate (analyze_emanuel_comprehensive.py) hardcodes an unrelated"
  echo "       absolute path and has no CLI. Needs a dedicated script or a rewrite of that"
  echo "       one before this can be automated here."
  echo "  [--] main/SI figure workflows: notebook-driven"
  echo "       (notebooks/probabilistic_risk_analysis_pub.ipynb); not executed by this"
  echo "       wrapper. Re-run it manually against the tables produced above."
  echo ""
  echo "[ef] Do NOT treat the publication update as complete until the two items above"
  echo "[ef] are also done and every 'NOT run' line here is resolved."

  if [[ ${#skipped[@]:-0} -gt 0 ]]; then
    return 2
  fi
  return 0
}

# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
case "${1:-}" in
  check) shift; cmd_check "$@" ;;
  pilot) shift; cmd_pilot "$@" ;;
  pilot-report) shift; cmd_pilot_report "$@" ;;
  production) shift; cmd_production "$@" ;;
  status) shift; cmd_status "$@" ;;
  postprocess) shift; cmd_postprocess "$@" ;;
  -h|--help|"") usage; exit 1 ;;
  *) echo "[ef] unknown command: ${1}" >&2; usage; exit 1 ;;
esac
