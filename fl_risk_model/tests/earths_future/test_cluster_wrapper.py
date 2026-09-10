"""Tests for scripts/cluster/earths_future_lib.py, the small helper behind
scripts/cluster/earths_future.sh.

These use mocked `squeue`/`sacct` executables placed first on PATH (never
the real Slurm client tools, and never sbatch -- nothing here submits a
job) to check the one property the brief specifically asked to be tested:
a pilot or production job that is pending, running, failed, or "completed"
with a bad output (wrong row count, error rows, missing file) can never be
reported as resolved/passing by job-status, resolve-manifest, or
compare-pilot. See docs/earths_future_revision/cluster_runbook.md.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "cluster"))

import earths_future_lib as ef  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _write_iterations(path: Path, n=50, seed=0, with_error_row=False, wrong_count=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = n - 1 if wrong_count else n
    df = pd.DataFrame({
        "year_id": range(1, rows + 1),
        "scenario": ["ok"] * rows,
        "total_damage_usd": rng.uniform(0, 1e9, rows),
        "wind_total_usd": rng.uniform(0, 1e9, rows),
        "water_total_usd": rng.uniform(0, 1e8, rows),
        "nfip_borrowed_usd": np.zeros(rows),
        "nfip_claims_paid_usd": rng.uniform(0, 1e7, rows),
        "fhcf_shortfall_usd": np.zeros(rows),
    })
    if with_error_row:
        df.loc[0, "scenario"] = "error"
    df.to_csv(path, index=False)
    return path


def _mock_slurm_bin(tmp_path: Path, states: dict[str, str]) -> Path:
    """A directory with mock squeue/sacct: squeue always reports "not
    queued" (exit 1) so classify_job falls through to sacct, which returns
    `states[job_id]` if present, else exit 1 (unknown)."""
    bindir = tmp_path / "mockbin"
    bindir.mkdir()
    (bindir / "squeue").write_text("#!/bin/bash\nexit 1\n")
    states_json = json.dumps(states)
    sacct_script = f"""#!/bin/bash
python3 - "$@" <<'PYEOF'
import json, sys
states = json.loads('''{states_json}''')
jobid = None
args = sys.argv[1:]
for i, a in enumerate(args):
    if a == "-j" and i + 1 < len(args):
        jobid = args[i + 1]
if jobid in states:
    print(states[jobid])
    sys.exit(0)
sys.exit(1)
PYEOF
"""
    (bindir / "sacct").write_text(sacct_script)
    for f in ["squeue", "sacct"]:
        p = bindir / f
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


@pytest.fixture
def mock_path(tmp_path, monkeypatch):
    def _install(states: dict[str, str]):
        bindir = _mock_slurm_bin(tmp_path, states)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        return bindir
    return _install


# --------------------------------------------------------------------------- #
# validate_run_dir / _resolve_run_dir
# --------------------------------------------------------------------------- #

def test_validate_run_dir_pass(tmp_path):
    d = tmp_path / "run"
    _write_iterations(d / "iterations.csv", n=50)
    result = ef.validate_run_dir(d, expected_seasons=50)
    assert result["pass"] is True
    assert result["problems"] == []


def test_validate_run_dir_wrong_row_count_fails(tmp_path):
    d = tmp_path / "run"
    _write_iterations(d / "iterations.csv", n=50, wrong_count=True)
    result = ef.validate_run_dir(d, expected_seasons=50)
    assert result["pass"] is False
    assert any("expected 50 rows" in p for p in result["problems"])


def test_validate_run_dir_error_row_fails(tmp_path):
    d = tmp_path / "run"
    _write_iterations(d / "iterations.csv", n=50, with_error_row=True)
    result = ef.validate_run_dir(d, expected_seasons=50)
    assert result["pass"] is False
    assert any("error" in p for p in result["problems"])


def test_validate_run_dir_missing_file_fails_with_reason_not_problems(tmp_path):
    d = tmp_path / "empty_root"
    d.mkdir()
    result = ef.validate_run_dir(d, expected_seasons=50)
    assert result["pass"] is False
    assert "reason" in result and "problems" not in result


def test_resolve_run_dir_dedicated_root_with_one_subdir(tmp_path):
    root = tmp_path / "job_root"
    _write_iterations(root / "inner" / "iterations.csv", n=10)
    result = ef.validate_run_dir(root, expected_seasons=10)
    assert result["pass"] is True
    assert result["run_dir"] == str(root / "inner")


def test_resolve_run_dir_two_subdirs_does_not_guess(tmp_path):
    root = tmp_path / "job_root"
    _write_iterations(root / "a" / "iterations.csv", n=10)
    _write_iterations(root / "b" / "iterations.csv", n=10)
    result = ef.validate_run_dir(root, expected_seasons=10)
    # Neither subdirectory is picked; validate-run reports the root itself
    # as missing iterations.csv rather than silently choosing one.
    assert result["pass"] is False
    assert result["run_dir"] == str(root)


def test_insured_fraction_expected_filename_resolution(tmp_path):
    root = tmp_path / "frac_root"
    inner = root / "insured_frac_sensitivity_20260101_000000"
    _write_iterations(inner / "iterations_frac_0.30.csv", n=10)
    expected_filename = ef._expected_filename_for_job("insured_fraction_0.3")
    assert expected_filename == "iterations_frac_0.30.csv"
    result = ef.validate_run_dir(root, expected_seasons=10, expected_filename=expected_filename)
    assert result["pass"] is True
    assert result["run_dir"] == str(inner)


def test_insured_fraction_wrong_default_filename_does_not_find_it(tmp_path):
    """Guards the exact bug hit during manual testing: resolving with the
    generic default ("iterations.csv") must NOT find an insured-fraction
    run's actual file, so callers are forced to pass the right name."""
    root = tmp_path / "frac_root"
    inner = root / "insured_frac_sensitivity_20260101_000000"
    _write_iterations(inner / "iterations_frac_0.30.csv", n=10)
    result = ef.validate_run_dir(root, expected_seasons=10)  # default filename
    assert result["pass"] is False


# --------------------------------------------------------------------------- #
# compare_pilot_runs
# --------------------------------------------------------------------------- #

def test_compare_pilot_passes_when_upstream_identical(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_iterations(old / "iterations.csv", n=20, seed=7)
    _write_iterations(new / "iterations.csv", n=20, seed=7)
    result = ef.compare_pilot_runs(old, new, expected_seasons=20)
    assert result["pass"] is True
    assert result["upstream_columns_mismatched"] == []


def test_compare_pilot_fails_when_upstream_diverges(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_iterations(old / "iterations.csv", n=20, seed=1)
    _write_iterations(new / "iterations.csv", n=20, seed=2)  # different draws
    result = ef.compare_pilot_runs(old, new, expected_seasons=20)
    assert result["pass"] is False
    assert "total_damage_usd" in result["upstream_columns_mismatched"]


def test_compare_pilot_fails_if_either_side_is_invalid(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    _write_iterations(old / "iterations.csv", n=20, with_error_row=True)
    _write_iterations(new / "iterations.csv", n=20)
    result = ef.compare_pilot_runs(old, new, expected_seasons=20)
    assert result["pass"] is False
    assert "validate-run" in result["reason"]


# --------------------------------------------------------------------------- #
# classify_job / job-status: pending, running, failed must never read as
# completed; a "completed" job with bad output must never validate.
# --------------------------------------------------------------------------- #

def test_classify_job_running_via_squeue(tmp_path, monkeypatch):
    bindir = tmp_path / "mockbin"
    bindir.mkdir()
    (bindir / "squeue").write_text("#!/bin/bash\necho RUNNING\nexit 0\n")
    (bindir / "sacct").write_text("#!/bin/bash\nexit 1\n")
    for f in ["squeue", "sacct"]:
        p = bindir / f
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    assert ef.classify_job({"slurm_job_id": "42"}) == "running"


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "FAILED", "CANCELLED", "TIMEOUT"])
def test_non_completed_states_never_resolve(tmp_path, mock_path, state):
    mock_path({"77": state})
    manifest_path = tmp_path / "manifest.json"
    out_dir = tmp_path / "out"
    _write_iterations(out_dir / "iterations.csv", n=10)  # good output, but job isn't done
    manifest = {
        "jobs": [{"name": "j", "slurm_job_id": "77", "expected_seasons": 10,
                  "output_dir": str(out_dir), "status": "submitted"}],
    }
    manifest_path.write_text(json.dumps(manifest))

    class Args:
        manifest = manifest_path
        report_out = None
    rc = ef.cmd_resolve_manifest(Args())
    # resolve-manifest never puts a non-completed job into `resolved`,
    # regardless of how good its output directory looks (rc==1 means
    # `missing` is non-empty; see the report-out-based tests below for the
    # full resolved/missing contents).
    assert rc == 1


@pytest.mark.parametrize("bad_kind", ["wrong_count", "error_row"])
def test_completed_but_invalid_output_never_resolves(tmp_path, mock_path, bad_kind):
    mock_path({"88": "COMPLETED"})
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "resolved.json"
    out_dir = tmp_path / "out"
    _write_iterations(out_dir / "iterations.csv", n=10,
                       wrong_count=(bad_kind == "wrong_count"),
                       with_error_row=(bad_kind == "error_row"))
    manifest = {
        "jobs": [{"name": "j", "slurm_job_id": "88", "expected_seasons": 10,
                  "output_dir": str(out_dir), "status": "submitted"}],
    }
    manifest_path.write_text(json.dumps(manifest))

    class Args:
        manifest = manifest_path
        report_out = report_path
    rc = ef.cmd_resolve_manifest(Args())
    assert rc == 1
    result = json.loads(report_path.read_text())
    assert result["resolved"] == {}
    assert result["missing"][0]["name"] == "j"


def test_completed_and_valid_output_resolves(tmp_path, mock_path):
    mock_path({"99": "COMPLETED"})
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "resolved.json"
    out_dir = tmp_path / "out"
    _write_iterations(out_dir / "iterations.csv", n=10)
    manifest = {
        "jobs": [{"name": "era5_baseline", "slurm_job_id": "99", "expected_seasons": 10,
                  "output_dir": str(out_dir), "status": "submitted"}],
    }
    manifest_path.write_text(json.dumps(manifest))

    class Args:
        manifest = manifest_path
        report_out = report_path
    rc = ef.cmd_resolve_manifest(Args())
    assert rc == 0
    result = json.loads(report_path.read_text())
    assert result["resolved"] == {"era5_baseline": str(out_dir)}
    assert result["missing"] == []


# --------------------------------------------------------------------------- #
# guard-duplicate
# --------------------------------------------------------------------------- #

def test_guard_duplicate_allows_when_no_manifest(tmp_path):
    class Args:
        manifest = tmp_path / "nope.json"
        force = False
    assert ef.cmd_guard_duplicate(Args()) == 0


def test_guard_duplicate_refuses_same_revision_active_job(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    rev = ef.git_revision()
    inputs = {**ef.hash_active_inputs(), **ef.hash_historical_hazard_inputs()}
    manifest_path.write_text(json.dumps({
        "code_revision": rev, "input_hashes": inputs,
        "jobs": [{"name": "x", "status": "running"}],
    }))

    class Args:
        manifest = manifest_path
        force = False
    assert ef.cmd_guard_duplicate(Args()) == 1

    class ArgsForce:
        manifest = manifest_path
        force = True
    assert ef.cmd_guard_duplicate(ArgsForce()) == 0


def test_guard_duplicate_allows_when_all_prior_jobs_failed(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    rev = ef.git_revision()
    inputs = {**ef.hash_active_inputs(), **ef.hash_historical_hazard_inputs()}
    manifest_path.write_text(json.dumps({
        "code_revision": rev, "input_hashes": inputs,
        "jobs": [{"name": "x", "status": "failed"}],
    }))

    class Args:
        manifest = manifest_path
        force = False
    assert ef.cmd_guard_duplicate(Args()) == 0


# --------------------------------------------------------------------------- #
# build_job_list: exact required production coverage
# --------------------------------------------------------------------------- #

def test_build_job_list_has_106_unique_jobs():
    jobs = ef.build_job_list(Path("results/mc_runs_fhcf_patched/production"),
                              Path(ef.DEFAULT_IMPACT_ROOT))
    assert len(jobs) == 106
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names))
    out_dirs = [str(j["out_dir"]) for j in jobs]
    assert len(out_dirs) == len(set(out_dirs))  # every job gets its own dedicated root


def test_build_job_list_excludes_andrew_then_gm_and_wrong_buildingcode_design():
    jobs = ef.build_job_list(Path("out"), Path(ef.DEFAULT_IMPACT_ROOT))
    names = [j["name"] for j in jobs]
    assert not any("andrew_then_gm" in n for n in names)
    bc_jobs = [j for j in jobs if j["name"].startswith("buildingcode_")]
    assert len(bc_jobs) == 65  # 13 levels x 5 GCMs, not the 50-task design
    levels = {j["name"].split("_L")[-1] for j in bc_jobs}
    assert levels == {f"{i:02d}" for i in range(13)}
