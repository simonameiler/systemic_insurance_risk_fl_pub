#!/usr/bin/env python3
"""Small helper library backing scripts/cluster/earths_future.sh.

Not a general workflow framework: this module only does the things a shell
script does awkwardly (JSON manifests, SHA-256 hashing, CSV row/column
validation, squeue/sacct parsing) so earths_future.sh can stay a thin
wrapper around sbatch and the existing scientific drivers. Every subcommand
here is read-only or writes only to the manifest/report paths passed to it;
none of them submit Slurm jobs (sbatch is invoked from earths_future.sh).

Subcommands (see `--help` on each):
    preflight            environment/input/code checks for `check`
    list-jobs            enumerate the fixed production experiment inventory
                         (or the single pilot job) as ready-to-submit commands
    resolve-output-dir   find the one run directory a driver created inside
                         a dedicated fresh per-job output root
    manifest-init        create a fresh campaign manifest (pilot or production)
    manifest-add-job     record one submitted Slurm job in a manifest
    manifest-set-status  update a job's status (used by `status`)
    guard-duplicate      refuse to start a production campaign that duplicates
                         an existing one for the same code+inputs
    validate-run         check one run directory's row count / error rows
    compare-pilot        pair two (old vs. corrected) run directories
    job-status           classify manifest jobs via squeue/sacct
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "earths_future_revision"))

REVIEWED_PATCH_COMMIT = "ef7706513e9437a47c33dc6b4972297f8c235975"

# Impact-cache root and its expected per-event-set filenames (Sherlock
# defaults; overridable). 26 = 5 GCMs x 5 periods + ERA5.
DEFAULT_IMPACT_ROOT = "/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts"
GCMS = ["canesm", "cnrm6", "ecearth6", "ipsl6", "miroc6"]
PERIODS = ["20thcal", "ssp245cal", "ssp245_2cal", "ssp585cal", "ssp585_2cal"]
REQUIRED_EVENT_SETS = ["FL_era5_reanalcal"] + [f"FL_{g}_{p}" for g in GCMS for p in PERIODS]
assert len(REQUIRED_EVENT_SETS) == 26

YEAR_SETS_FILENAME = "year_sets_N10000_seed42.csv"
EVENT_METADATA_FILENAME = "event_metadata.csv"

# Active financial inputs the production/pilot financial-model call path
# reads (independent review, item 3: hash every active input, not just
# hazard/premium/surplus).
ACTIVE_DATA_INPUTS = [
    "24fin_fhcf.csv", "fhcf_terms_keyed.csv", "company_keys.csv",
    "FL HO Market Share Report_6.10.25.xlsx",
    "20250805 FL Surplus Capital, Group v Entity.xlsx",
    "FHCF_2024_Exposure_byCounty.xlsx",
    "citizens_capital_pml.csv", "citizens_county_data.csv",
    "citizens_county_data_all_harmonized.csv",
    "NfipResidentialPenetrationRates.csv",
    "nfip_FL_coverage_premium_by_year.csv", "nfip_FL_claims_by_year.csv",
    "catbonds_2024.csv", "catbonds_2024_reviewed.csv",
    "county_region.csv", "fl_county_fips.csv", "florida_coastal_counties.csv",
    "florida_log_contribution_p95_present.csv",
]

HISTORICAL_HAZARD_FILES = [
    "1926255N15314.csv", "1928250N14343.csv", "1992230N11325.csv", "2017242N16333.csv",
    "great_miami_twice.csv", "great_miami_then_andrew.csv",
    "andrew_then_great_miami.csv", "irma_twice.csv",
]

FRESH_OUTPUT_ROOTS = [
    "results/mc_runs_fhcf_patched",
    "results/earths_future_revision/fhcf_cluster",
    "results/mc_runs_catbond_patched",
    "results/earths_future_revision/catbond_cluster",
]
MARCH_ARCHIVE_ROOT = "results/mc_runs"

# Required production coverage (brief's exact inventory table). 1 + 3 + 25 +
# 65 + 7 + 5 = 106 runs total. Excludes the unused andrew_then_gm reversed
# sequence and the wrong 50-task building-code design on purpose.
POLICY_NAMES = ["market_exit_moderate", "penetration_major", "building_codes_major"]
HISTORICAL_SCENARIOS = ["great_miami", "andrew", "gm_then_andrew", "double_gm",
                         "lake_okeechobee", "irma", "double_irma"]
INSURED_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]
BUILDING_CODE_LEVELS = list(range(13))
BUILDING_CODE_PERIOD = "ssp245cal"  # matches the reviewed 65-task design


def build_job_list(out_root: Path, impact_root: Path) -> list[dict]:
    """The fixed production experiment inventory as ready-to-run driver
    commands. Each job gets its OWN dedicated, otherwise-empty output root
    (out_dir below) so the single run directory it creates can be found
    later with `resolve-output-dir` -- no glob over a shared, populated
    results/mc_runs the way the pre-patch building-code driver did."""
    jobs = []

    def emanuel_cmd(event_set, out_dir, policy=None):
        # "python" (not sys.executable): these commands are shlex-joined and
        # run inside a batch job's own `conda activate climada_env`, which
        # may be a different absolute interpreter path than whatever is
        # running earths_future_lib.py right now (e.g. a login/dev shell).
        cmd = ["python", "scripts/run/run_emanuel_monte_carlo.py",
               "--event_set", event_set,
               "--impact_dir", str(impact_root / event_set),
               "--seed", "42", "--out", str(out_dir)]
        if policy:
            cmd += ["--policy", policy]
        return cmd

    era5_dir = out_root / "era5_baseline"
    jobs.append({"name": "era5_baseline", "expected_seasons": 10000, "out_dir": era5_dir,
                 "cmd": emanuel_cmd("FL_era5_reanalcal", era5_dir)})

    for policy in POLICY_NAMES:
        d = out_root / f"era5_policy_{policy}"
        jobs.append({"name": f"era5_policy_{policy}", "expected_seasons": 10000, "out_dir": d,
                     "cmd": emanuel_cmd("FL_era5_reanalcal", d, policy=policy)})

    for gcm in GCMS:
        for period in PERIODS:
            event_set = f"FL_{gcm}_{period}"
            d = out_root / "gcm_baseline" / f"{gcm}_{period}"
            jobs.append({"name": f"gcm_baseline_{gcm}_{period}", "expected_seasons": 10000,
                         "out_dir": d, "cmd": emanuel_cmd(event_set, d)})

    for gcm in GCMS:
        for level in BUILDING_CODE_LEVELS:
            event_set = f"FL_{gcm}_{BUILDING_CODE_PERIOD}"
            d = out_root / "buildingcode" / f"{gcm}_L{level:02d}"
            cmd = ["python", "scripts/run/run_climate_buildingcode_sensitivity_windfloods.py",
                   "--code_level", str(level), "--event_set", event_set,
                   "--impact_dir", str(impact_root / event_set),
                   "--seed", "42", "--out_dir", str(d)]
            jobs.append({"name": f"buildingcode_{gcm}_L{level:02d}", "expected_seasons": 10000,
                         "out_dir": d, "cmd": cmd})

    for scenario in HISTORICAL_SCENARIOS:
        d = out_root / "historical" / scenario
        cmd = ["python", "scripts/run/run_historical_scenarios_mc.py",
               "--scenario", scenario, "--n_iter", "1000", "--seed", "42",
               "--out", str(d), "--skip_report"]
        jobs.append({"name": f"historical_{scenario}", "expected_seasons": 1000,
                     "out_dir": d, "cmd": cmd})

    for frac in INSURED_FRACTIONS:
        d = out_root / "insured_fraction" / f"frac_{frac}"
        cmd = ["python", "scripts/run/run_insured_fraction_sensitivity.py",
               "--fractions", str(frac), "--seed", "42", "--out_dir", str(d)]
        jobs.append({"name": f"insured_fraction_{frac}", "expected_seasons": 10000,
                     "out_dir": d, "cmd": cmd})

    assert len(jobs) == 106, f"expected 106 production jobs, built {len(jobs)}"
    return jobs


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def _run(cmd: list[str], *, cwd=None) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          check=True).stdout.strip()


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def git_revision(require_ancestor_of_patch: bool = True) -> dict:
    # Sherlock's system Git predates -C. Run Git in the checkout instead.
    commit = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    describe = _run(["git", "describe", "--always", "--dirty"], cwd=REPO_ROOT)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT)
    # Distinguish source changes from ignored/untracked output files: only
    # tracked paths under fl_risk_model/ and scripts/ count as "dirty" for
    # the purposes of pinning a code revision. Untracked results/ output is
    # expected and irrelevant to which CODE ran.
    source_status = _run(["git", "status", "--porcelain", "--untracked-files=no",
                         "--", "fl_risk_model", "scripts", "pyproject.toml"],
                        cwd=REPO_ROOT)
    contains_patch = None
    if require_ancestor_of_patch:
        proc = subprocess.run(
            ["git", "merge-base", REVIEWED_PATCH_COMMIT, "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if proc.returncode not in (0, 1):
            raise subprocess.CalledProcessError(proc.returncode, proc.args,
                                                output=proc.stdout, stderr=proc.stderr)
        contains_patch = (proc.returncode == 0 and
                          proc.stdout.strip() == REVIEWED_PATCH_COMMIT)
    return {
        "commit": commit,
        "describe": describe,
        "branch": branch,
        "is_dirty_tracked_source": bool(source_status),
        "dirty_tracked_source_paths": source_status.splitlines() if source_status else [],
        "contains_reviewed_fhcf_patch": contains_patch,
        "reviewed_patch_commit": REVIEWED_PATCH_COMMIT,
    }


def hash_active_inputs() -> dict:
    return {f"data/{f}": file_sha256(REPO_ROOT / "fl_risk_model" / "data" / f)
            for f in ACTIVE_DATA_INPUTS}


def hash_historical_hazard_inputs() -> dict:
    d = REPO_ROOT / "fl_risk_model" / "data" / "hazard" / "historical_events"
    return {f"hazard/{f}": file_sha256(d / f) for f in HISTORICAL_HAZARD_FILES}


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

def cmd_preflight(args) -> int:
    impact_root = Path(args.impact_root)
    report: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "problems": [], "ok": []}

    # 1. Code revision.
    try:
        rev = git_revision(require_ancestor_of_patch=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (getattr(exc, "stderr", "") or str(exc)).strip()
        report["code_revision"] = {"error": detail}
        report["problems"].append(f"could not verify code revision: {detail}")
    else:
        report["code_revision"] = rev
        if rev["contains_reviewed_fhcf_patch"] is False:
            report["problems"].append(
                f"current HEAD ({rev['commit']}) does not contain the reviewed FHCF patch "
                f"{REVIEWED_PATCH_COMMIT}"
            )
        else:
            report["ok"].append("branch contains the reviewed FHCF patch")
        if rev["is_dirty_tracked_source"]:
            report["problems"].append(
                "tracked source files (fl_risk_model/, scripts/) have uncommitted changes: "
                f"{rev['dirty_tracked_source_paths']}"
            )
        else:
            report["ok"].append("no uncommitted changes to tracked source files")

    # 2. Environment.
    try:
        import fl_risk_model.mc_run_events  # noqa: F401
        import fl_risk_model.fhcf as fhcf_mod
        has_patch_signature = "CompanyGrossWindLossUSD" in Path(fhcf_mod.__file__).read_text()
        if has_patch_signature:
            report["ok"].append("fl_risk_model.fhcf contains the patched apply_fhcf_recovery")
        else:
            report["problems"].append(
                "fl_risk_model/fhcf.py does not contain the patched apply_fhcf_recovery "
                "(CompanyGrossWindLossUSD not found) -- wrong checkout or reverted patch"
            )
    except Exception as e:
        report["problems"].append(f"could not import fl_risk_model.mc_run_events: {e}")

    # A login shell can auto-activate base and report a different environment.
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    report["conda_env"] = conda_env or None
    report["conda_prefix"] = os.environ.get("CONDA_PREFIX")
    report["python_executable"] = sys.executable
    if conda_env == "climada_env":
        report["ok"].append("climada_env is the active conda environment")
    else:
        report["problems"].append(
            f"climada_env is not the active conda environment (got: {conda_env!r}); "
            "activate climada_env before rerunning the check"
        )

    # 3. Required event-set impact caches (26).
    missing_event_sets = []
    for es in REQUIRED_EVENT_SETS:
        d = impact_root / es
        year_sets = d / YEAR_SETS_FILENAME
        metadata = d / EVENT_METADATA_FILENAME
        if not (d.is_dir() and year_sets.exists() and metadata.exists()):
            missing_event_sets.append({
                "event_set": es,
                "dir_exists": d.is_dir(),
                "year_sets_exists": year_sets.exists(),
                "event_metadata_exists": metadata.exists(),
                "expected_path": str(d),
            })
    report["required_event_sets"] = {
        "impact_root": str(impact_root),
        "required": len(REQUIRED_EVENT_SETS),
        "present": len(REQUIRED_EVENT_SETS) - len(missing_event_sets),
        "missing": missing_event_sets,
    }
    if missing_event_sets:
        report["problems"].append(
            f"{len(missing_event_sets)}/{len(REQUIRED_EVENT_SETS)} required event-set "
            f"directories are missing or incomplete under {impact_root} "
            "(concrete list in required_event_sets.missing; NOT regenerating hazard data)"
        )
    else:
        report["ok"].append(f"all {len(REQUIRED_EVENT_SETS)} required event-set directories present")

    # 4. Active financial inputs.
    input_hashes = hash_active_inputs()
    missing_inputs = [k for k, v in input_hashes.items() if v == "MISSING"]
    report["active_input_hashes"] = input_hashes
    if missing_inputs:
        report["problems"].append(f"missing active financial inputs: {missing_inputs}")
    else:
        report["ok"].append(f"all {len(input_hashes)} active financial inputs present")

    # 5. Historical hazard inputs (git-tracked, should always be present).
    hist_hashes = hash_historical_hazard_inputs()
    missing_hist = [k for k, v in hist_hashes.items() if v == "MISSING"]
    report["historical_hazard_hashes"] = hist_hashes
    if missing_hist:
        report["problems"].append(f"missing historical hazard inputs: {missing_hist}")
    else:
        report["ok"].append(f"all {len(hist_hashes)} historical hazard inputs present")

    # 6. Output-root separation.
    for root in FRESH_OUTPUT_ROOTS:
        p = REPO_ROOT / root
        report.setdefault("output_roots", {})[root] = {
            "exists": p.exists(),
            "is_distinct_from_march_archive": str(p.resolve()) != str((REPO_ROOT / MARCH_ARCHIVE_ROOT).resolve()),
        }
    report["ok"].append(f"fresh output roots configured distinct from {MARCH_ARCHIVE_ROOT}")

    # 7. Tests.
    if not args.skip_tests:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "fl_risk_model/tests/earths_future", "-q"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        diagnostic_lines = (proc.stdout.strip() or proc.stderr.strip()).splitlines()
        report["tests"] = {
            "returncode": proc.returncode,
            "python_executable": sys.executable,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "summary": diagnostic_lines[-1] if diagnostic_lines else
                       f"pytest exited with status {proc.returncode} without output",
        }
        if proc.returncode != 0:
            report["problems"].append(f"pytest failed: {report['tests']['summary']}")
        else:
            report["ok"].append(f"pytest: {report['tests']['summary']}")

    report["pass"] = len(report["problems"]) == 0
    out_path = args.report_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"pass": report["pass"], "ok": report["ok"], "problems": report["problems"]}, indent=2))
    print(f"\nFull report: {out_path}")
    return 0 if report["pass"] else 1


# --------------------------------------------------------------------------- #
# list-jobs / resolve-output-dir
# --------------------------------------------------------------------------- #

def cmd_list_jobs(args) -> int:
    out_root = Path(args.out_root)
    impact_root = Path(args.impact_root)
    jobs = build_job_list(out_root, impact_root)
    for job in jobs:
        print("\t".join([
            job["name"], str(job["expected_seasons"]), str(job["out_dir"]),
            shlex.join(str(c) for c in job["cmd"]),
        ]))
    print(f"# {len(jobs)} jobs", file=sys.stderr)
    return 0


def cmd_resolve_output_dir(args) -> int:
    """Every job below writes into its OWN dedicated, otherwise-empty output
    root, so the run directory the driver actually created is unambiguous:
    the single subdirectory of that root. Zero or more than one is a real
    problem (job never ran / ran twice into the same root) and is reported
    as an error rather than guessed at."""
    root = args.job_out_root
    if not root.is_dir():
        print(f"[ERROR] job output root does not exist: {root}", file=sys.stderr)
        return 1
    children = sorted(p for p in root.iterdir() if p.is_dir())
    if len(children) == 0:
        print(f"[ERROR] no run directory found under {root} (job did not complete "
              "or wrote nothing)", file=sys.stderr)
        return 1
    if len(children) > 1:
        print(f"[ERROR] {len(children)} directories found under {root}, expected exactly "
              f"one dedicated per-job output root: {children}", file=sys.stderr)
        return 1
    print(str(children[0]))
    return 0


# --------------------------------------------------------------------------- #
# manifest-init / manifest-add-job / manifest-set-status
# --------------------------------------------------------------------------- #

def cmd_manifest_init(args) -> int:
    manifest = {
        "campaign": args.campaign,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code_revision": git_revision(),
        "seed": args.seed,
        "out_root": args.out_root,
        "concurrency_limit": args.concurrency_limit,
        "input_hashes": {**hash_active_inputs(), **hash_historical_hazard_inputs()},
        "jobs": [],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {args.manifest}")
    return 0


def _load_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def cmd_manifest_add_job(args) -> int:
    manifest = _load_manifest(args.manifest)
    manifest["jobs"].append({
        "name": args.name,
        "slurm_job_id": args.job_id,
        "expected_seasons": args.expected_seasons,
        "output_dir": args.output_dir,
        "status": "submitted",
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _save_manifest(args.manifest, manifest)
    print(f"Recorded job {args.name!r} (slurm id {args.job_id}) in {args.manifest}")
    return 0


def cmd_manifest_set_status(args) -> int:
    manifest = _load_manifest(args.manifest)
    found = False
    for job in manifest["jobs"]:
        if job["name"] == args.name:
            job["status"] = args.status
            job["output_dir"] = args.output_dir or job.get("output_dir")
            found = True
    if not found:
        print(f"[ERROR] job {args.name!r} not found in {args.manifest}", file=sys.stderr)
        return 1
    _save_manifest(args.manifest, manifest)
    return 0


# --------------------------------------------------------------------------- #
# guard-duplicate
# --------------------------------------------------------------------------- #

def cmd_guard_duplicate(args) -> int:
    """Refuse a new production submission if an existing manifest for the
    SAME code revision + input hashes already has jobs that are running or
    completed (not merely failed), unless --force."""
    if not args.manifest.exists():
        print("No existing manifest at that path -- safe to proceed.")
        return 0
    existing = _load_manifest(args.manifest)
    current_rev = git_revision()
    current_inputs = {**hash_active_inputs(), **hash_historical_hazard_inputs()}

    same_commit = existing.get("code_revision", {}).get("commit") == current_rev["commit"]
    same_inputs = existing.get("input_hashes") == current_inputs
    active_jobs = [j for j in existing.get("jobs", []) if j.get("status") in
                   ("submitted", "running", "completed")]

    if same_commit and same_inputs and active_jobs and not args.force:
        print(json.dumps({
            "duplicate": True,
            "existing_manifest": str(args.manifest),
            "existing_code_revision": existing.get("code_revision", {}).get("commit"),
            "n_active_jobs": len(active_jobs),
            "message": "Refusing duplicate production submission for the same code revision "
                       "and inputs. Pass --force to override, or point --manifest at a new path "
                       "for an intentionally separate campaign.",
        }, indent=2))
        return 1

    print(json.dumps({
        "duplicate": False,
        "same_commit": same_commit,
        "same_inputs": same_inputs,
        "n_active_jobs_in_existing_manifest": len(active_jobs),
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# validate-run
# --------------------------------------------------------------------------- #

def _resolve_run_dir(root: Path, expected_filename: str = "iterations.csv") -> Path:
    """Accept either a concrete run directory (has `expected_filename`
    directly) or a dedicated per-job output root (has exactly one
    subdirectory, which is the run directory) -- so validate-run/
    compare-pilot/job-status work the same way regardless of which one the
    manifest recorded. Falls through to the input unchanged if neither
    pattern matches; the caller then reports the concrete missing-file
    problem rather than guessing.

    `expected_filename` defaults to the standard driver output name but the
    insured-fraction driver instead writes iterations_frac_<f>.csv (one
    dedicated root can also hold more than one fraction's file if a caller
    ever points several fractions at the same root) -- resolve-manifest
    passes that name explicitly for insured_fraction_* jobs."""
    if (root / expected_filename).exists():
        return root
    if root.is_dir():
        children = sorted(p for p in root.iterdir() if p.is_dir())
        if len(children) == 1 and (children[0] / expected_filename).exists():
            return children[0]
    return root


def validate_run_dir(run_dir: Path, expected_seasons: int | None,
                      expected_filename: str = "iterations.csv") -> dict:
    run_dir = _resolve_run_dir(run_dir, expected_filename)
    result: dict = {"run_dir": str(run_dir), "exists": run_dir.exists()}
    iterations_csv = run_dir / expected_filename
    if not iterations_csv.exists():
        result["pass"] = False
        result["reason"] = f"{expected_filename} not found (directory existence alone is not success)"
        return result

    df = pd.read_csv(iterations_csv, low_memory=False)
    n_rows = len(df)
    n_error = int((df["scenario"] == "error").sum()) if "scenario" in df.columns else 0
    n_dup = int(df.duplicated(subset=["year_id"]).sum()) if "year_id" in df.columns else 0

    result.update({
        "n_rows": n_rows,
        "expected_seasons": expected_seasons,
        "n_error_rows": n_error,
        "n_duplicate_year_ids": n_dup,
    })
    problems = []
    if expected_seasons is not None and n_rows != expected_seasons:
        problems.append(f"expected {expected_seasons} rows, found {n_rows}")
    if n_error > 0:
        problems.append(f"{n_error} row(s) with scenario=='error'")
    if n_dup > 0:
        problems.append(f"{n_dup} duplicate year_id value(s)")
    if (run_dir / "errors_summary.txt").exists():
        problems.append("errors_summary.txt present")

    result["problems"] = problems
    result["pass"] = len(problems) == 0
    return result


def cmd_validate_run(args) -> int:
    result = validate_run_dir(args.run_dir, args.expected_seasons, args.expected_filename)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


# --------------------------------------------------------------------------- #
# compare-pilot
# --------------------------------------------------------------------------- #

UPSTREAM_COLUMNS = [
    "total_damage_usd", "wind_total_usd", "water_total_usd",
    "nfip_borrowed_usd", "nfip_claims_paid_usd",
]


def compare_pilot_runs(old_dir: Path, new_dir: Path, expected_seasons: int | None) -> dict:
    old_dir = _resolve_run_dir(old_dir)
    new_dir = _resolve_run_dir(new_dir)
    old_valid = validate_run_dir(old_dir, expected_seasons)
    new_valid = validate_run_dir(new_dir, expected_seasons)
    result = {"old": old_valid, "new": new_valid}

    if not (old_valid["pass"] and new_valid["pass"]):
        result["pass"] = False
        result["reason"] = "one or both runs failed validate-run; cannot declare pilot success"
        return result

    old_df = pd.read_csv(old_dir / "iterations.csv", low_memory=False)
    new_df = pd.read_csv(new_dir / "iterations.csv", low_memory=False)

    if len(old_df) != len(new_df):
        result["pass"] = False
        result["reason"] = f"row count mismatch: old={len(old_df)} new={len(new_df)}"
        return result

    mismatched_cols = []
    for col in UPSTREAM_COLUMNS:
        if col not in old_df.columns or col not in new_df.columns:
            continue
        if not old_df[col].astype(str).equals(new_df[col].astype(str)):
            mismatched_cols.append(col)

    result["upstream_columns_checked"] = [c for c in UPSTREAM_COLUMNS if c in old_df.columns]
    result["upstream_columns_mismatched"] = mismatched_cols

    means = {}
    for col in ["fhcf_shortfall_usd", "fhcf_cap_binding", "figa_residual_deficit_usd",
                "citizens_residual_deficit_usd", "nfip_borrowed_usd", "defaults_post"]:
        if col in old_df.columns and col in new_df.columns:
            means[col] = {
                "old_mean": float(pd.to_numeric(old_df[col], errors="coerce").mean()),
                "new_mean": float(pd.to_numeric(new_df[col], errors="coerce").mean()),
            }
    result["means"] = means

    # A changed mean is expected and not a failure by itself; an upstream
    # mismatch or a validation failure is.
    result["pass"] = len(mismatched_cols) == 0
    if not result["pass"]:
        result["reason"] = f"upstream columns changed between variants (should be seed-identical): {mismatched_cols}"
    return result


def cmd_compare_pilot(args) -> int:
    result = compare_pilot_runs(args.old_dir, args.new_dir, args.expected_seasons)
    out = args.report_out
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


# --------------------------------------------------------------------------- #
# job-status
# --------------------------------------------------------------------------- #

def _sacct_state(job_id: str) -> str | None:
    """Query sacct for a single job id's state. Returns None if sacct is
    unavailable or the job is unknown (caller falls back to squeue)."""
    proc = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=State", "--noheader", "--parsable2"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    # First line is usually the parent job; states like COMPLETED, FAILED,
    # RUNNING, PENDING, CANCELLED.
    return proc.stdout.strip().splitlines()[0].strip()


def _squeue_state(job_id: str) -> str | None:
    proc = subprocess.run(
        ["squeue", "-j", str(job_id), "--noheader", "--format=%T"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.strip().splitlines()[0].strip()


def classify_job(job: dict) -> str:
    job_id = job.get("slurm_job_id")
    if not job_id:
        return "unknown (no slurm_job_id recorded)"
    state = _squeue_state(job_id)  # still in the queue -> pending/running
    if state:
        return state.lower()
    state = _sacct_state(job_id)  # no longer queued -> ask accounting
    if state:
        return state.lower()
    return "unknown (not in squeue or sacct)"


_INSURED_FRACTION_JOB_RE = re.compile(r"^insured_fraction_(\d+(?:\.\d+)?)$")


def _expected_filename_for_job(name: str) -> str:
    """The insured-fraction driver writes iterations_frac_<f>.csv, not
    iterations.csv (fl_risk_model/../run_insured_fraction_sensitivity.py's
    run_sweep); every other production job name uses the standard name."""
    m = _INSURED_FRACTION_JOB_RE.match(name)
    if m:
        return f"iterations_frac_{float(m.group(1)):.2f}.csv"
    return "iterations.csv"


def cmd_resolve_manifest(args) -> int:
    """For `postprocess`: classify every job in a manifest and validate its
    output. Jobs that are not (queue-state) completed, or that fail
    validate-run, go into `missing` with a reason -- never guessed into
    `resolved`. Never declares postprocess-readiness from directory
    existence alone."""
    manifest = _load_manifest(args.manifest)
    resolved, missing = {}, []
    for job in manifest.get("jobs", []):
        state = classify_job(job)
        out_dir = job.get("output_dir")
        if not out_dir or not state.startswith("completed"):
            missing.append({"name": job["name"], "queue_state": state})
            continue
        v = validate_run_dir(Path(out_dir), job.get("expected_seasons"),
                              _expected_filename_for_job(job["name"]))
        if not v["pass"]:
            # validate_run_dir returns "reason" (not "problems") when it
            # never got as far as row-level checks, e.g. no iterations.csv
            # at all -- true for every insured-fraction job, whose output
            # is iterations_frac_<f>.csv instead. Report whichever is set.
            problems = v.get("problems") or [v.get("reason", "validate-run failed")]
            missing.append({"name": job["name"], "queue_state": state, "validate_problems": problems})
            continue
        resolved[job["name"]] = v["run_dir"]

    out = {"resolved": resolved, "missing": missing}
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0 if not missing else 1


# Job-name -> section8/section9 display-label mappings. Centralized here
# (not duplicated in the bash wrapper) because it is the one place that
# needs to know both the production job-naming convention (build_job_list)
# and the section8/9 scripts' expected --scenario-map label strings.
CLIMATE_POLICY_LABELS = {
    "era5_baseline": "Baseline (ERA5)",
    "era5_policy_market_exit_moderate": "Market Exit",
    "era5_policy_penetration_major": "Insurance Penetration",
    "era5_policy_building_codes_major": "Building Codes",
}
HISTORICAL_LABELS = {
    "historical_great_miami": "Great Miami",
    "historical_andrew": "Andrew",
    "historical_gm_then_andrew": "Great Miami then Andrew",
    "historical_double_gm": "Double Great Miami",
    "historical_lake_okeechobee": "Lake Okeechobee",
    "historical_irma": "Irma",
    "historical_double_irma": "Double Irma",
}


def cmd_build_scenario_maps(args) -> int:
    with open(args.resolved) as f:
        resolved = json.load(f)["resolved"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def build(label_map, out_name):
        present = {label: resolved[job] for job, label in label_map.items() if job in resolved}
        missing = [label for job, label in label_map.items() if job not in resolved]
        path = args.out_dir / out_name
        with open(path, "w") as f:
            json.dump(present, f, indent=2)
        return path, present, missing

    cp_path, cp_present, cp_missing = build(CLIMATE_POLICY_LABELS, "climate_policy_scenario_map.json")
    hist_path, hist_present, hist_missing = build(HISTORICAL_LABELS, "historical_scenario_map.json")
    print(json.dumps({
        "climate_policy_map": str(cp_path), "climate_policy_present": list(cp_present),
        "climate_policy_missing": cp_missing,
        "historical_map": str(hist_path), "historical_present": list(hist_present),
        "historical_missing": hist_missing,
    }, indent=2))
    return 0 if not (cp_missing or hist_missing) else 1


def cmd_job_status(args) -> int:
    manifest = _load_manifest(args.manifest)
    rows = []
    for job in manifest.get("jobs", []):
        state = classify_job(job)
        run_ok = None
        # A paired pilot has two nested outputs; the generic single-run
        # validator cannot assess its root. Use pilot-report for the pair.
        paired_pilot = job.get("name") == "era5_pilot_paired"
        if state.startswith("completed") and job.get("output_dir") and not paired_pilot:
            v = validate_run_dir(Path(job["output_dir"]), job.get("expected_seasons"))
            run_ok = v["pass"]
        rows.append({
            "name": job["name"],
            "slurm_job_id": job.get("slurm_job_id"),
            "recorded_status": job.get("status"),
            "queue_state": state,
            "output_dir": job.get("output_dir"),
            "output_valid": run_ok,
            **({"validation_note": "Use pilot-report to validate both paired outputs."}
               if paired_pilot else {}),
        })
    summary = {}
    for r in rows:
        summary[r["queue_state"]] = summary.get(r["queue_state"], 0) + 1
    print(json.dumps({"jobs": rows, "state_summary": summary}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--impact-root", default=DEFAULT_IMPACT_ROOT)
    p.add_argument("--report-out", type=Path, required=True)
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("list-jobs")
    p.add_argument("--out-root", required=True)
    p.add_argument("--impact-root", default=DEFAULT_IMPACT_ROOT)
    p.set_defaults(func=cmd_list_jobs)

    p = sub.add_parser("resolve-output-dir")
    p.add_argument("--job-out-root", type=Path, required=True)
    p.set_defaults(func=cmd_resolve_output_dir)

    p = sub.add_parser("manifest-init")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--campaign", choices=["pilot", "production"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", required=True)
    p.add_argument("--concurrency-limit", type=int, default=20)
    p.set_defaults(func=cmd_manifest_init)

    p = sub.add_parser("manifest-add-job")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--expected-seasons", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=cmd_manifest_add_job)

    p = sub.add_parser("manifest-set-status")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--output-dir", default=None)
    p.set_defaults(func=cmd_manifest_set_status)

    p = sub.add_parser("guard-duplicate")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_guard_duplicate)

    p = sub.add_parser("validate-run")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--expected-seasons", type=int, default=None)
    p.add_argument("--expected-filename", default="iterations.csv",
                    help="Use iterations_frac_<f>.csv for insured-fraction runs.")
    p.set_defaults(func=cmd_validate_run)

    p = sub.add_parser("compare-pilot")
    p.add_argument("--old-dir", type=Path, required=True)
    p.add_argument("--new-dir", type=Path, required=True)
    p.add_argument("--expected-seasons", type=int, default=None)
    p.add_argument("--report-out", type=Path, default=None)
    p.set_defaults(func=cmd_compare_pilot)

    p = sub.add_parser("resolve-manifest")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--report-out", type=Path, default=None)
    p.set_defaults(func=cmd_resolve_manifest)

    p = sub.add_parser("build-scenario-maps")
    p.add_argument("--resolved", type=Path, required=True,
                    help="Path to a resolve-manifest --report-out JSON.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=cmd_build_scenario_maps)

    p = sub.add_parser("job-status")
    p.add_argument("--manifest", type=Path, required=True)
    p.set_defaults(func=cmd_job_status)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
