"""Regression checks for preflight on Sherlock's older Git and shell setup.

The temporary Git repositories and mocked preflight inputs never submit
jobs or execute scientific simulations.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "cluster"))

import earths_future_lib as ef  # noqa: E402


@pytest.fixture
def old_git_checkout(tmp_path, monkeypatch):
    real_git = shutil.which("git")
    assert real_git, "Git is required to validate cluster revision tracking"
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()

    def git(*args):
        return subprocess.run(
            [real_git, *args], cwd=checkout, capture_output=True,
            text=True, check=True,
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Preflight test")
    git("config", "user.email", "preflight-test@example.invalid")
    source = checkout / "fl_risk_model" / "example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    git("add", "fl_risk_model/example.py")
    git("commit", "-m", "Initial fixture")
    initial = git("rev-parse", "HEAD")
    main_branch = git("symbolic-ref", "--short", "HEAD")

    source.write_text("value = 2\n")
    git("commit", "-am", "Reviewed patch fixture")
    patch = git("rev-parse", "HEAD")
    source.write_text("value = 3\n")
    git("commit", "-am", "Launcher fixture")
    head = git("rev-parse", "HEAD")

    git("checkout", "-b", "unrelated-patch", initial)
    source.write_text("value = 99\n")
    git("commit", "-am", "Unrelated fixture")
    unrelated = git("rev-parse", "HEAD")
    git("checkout", main_branch)

    bindir = tmp_path / "old-git-bin"
    bindir.mkdir()
    shim = bindir / "git"
    shim.write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        '    -C|--is-ancestor) echo "unsupported old Git option: $arg" >&2; exit 129;;\n'
        "  esac\n"
        "done\n"
        f"exec {shlex.quote(real_git)} \"$@\"\n"
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(ef, "REPO_ROOT", checkout)
    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(
        checkout=checkout, source=source, patch=patch, head=head,
        branch=main_branch, unrelated=unrelated,
    )


def test_revision_works_with_old_git_and_ignores_untracked_backups(
    old_git_checkout, monkeypatch,
):
    fixture = old_git_checkout
    monkeypatch.setattr(ef, "REVIEWED_PATCH_COMMIT", fixture.patch)
    fixture.source.with_suffix(".py.backup").write_text("local backup\n")

    revision = ef.git_revision()

    assert revision["commit"] == fixture.head
    assert revision["branch"] == fixture.branch
    assert revision["contains_reviewed_fhcf_patch"] is True
    assert revision["is_dirty_tracked_source"] is False
    assert revision["dirty_tracked_source_paths"] == []

    fixture.source.write_text("value = 4\n")
    changed = ef.git_revision()
    assert changed["is_dirty_tracked_source"] is True
    assert any("example.py" in p for p in changed["dirty_tracked_source_paths"])


def test_revision_rejects_patch_on_an_unmerged_branch(old_git_checkout, monkeypatch):
    monkeypatch.setattr(ef, "REVIEWED_PATCH_COMMIT", old_git_checkout.unrelated)

    revision = ef.git_revision()

    assert revision["commit"] == old_git_checkout.head
    assert revision["contains_reviewed_fhcf_patch"] is False


@pytest.fixture
def preflight_inputs(tmp_path, monkeypatch):
    """Stub only external inputs and imports; exercise actual report logic."""
    import fl_risk_model

    fake_fhcf_path = tmp_path / "fhcf.py"
    fake_fhcf_path.write_text("CompanyGrossWindLossUSD = None\n")
    fake_fhcf = ModuleType("fl_risk_model.fhcf")
    fake_fhcf.__file__ = str(fake_fhcf_path)
    fake_mc = ModuleType("fl_risk_model.mc_run_events")
    for name, module in [("fhcf", fake_fhcf), ("mc_run_events", fake_mc)]:
        monkeypatch.setitem(sys.modules, f"fl_risk_model.{name}", module)
        monkeypatch.setattr(fl_risk_model, name, module, raising=False)

    monkeypatch.setattr(ef, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ef, "git_revision", lambda **kwargs: {
        "commit": "a" * 40,
        "describe": "aaaaaaa",
        "branch": "fixture",
        "contains_reviewed_fhcf_patch": True,
        "is_dirty_tracked_source": False,
        "dirty_tracked_source_paths": [],
    })
    monkeypatch.setattr(ef, "hash_active_inputs", lambda: {"data/example": "abc"})
    monkeypatch.setattr(ef, "hash_historical_hazard_inputs", lambda: {"hazard/example": "def"})
    monkeypatch.setattr(ef, "REQUIRED_EVENT_SETS", ["fixture"])
    event_dir = tmp_path / "impact" / "fixture"
    event_dir.mkdir(parents=True)
    (event_dir / ef.YEAR_SETS_FILENAME).write_text("fixture\n")
    (event_dir / ef.EVENT_METADATA_FILENAME).write_text("fixture\n")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "climada_env")
    return SimpleNamespace(
        impact_root=tmp_path / "impact", skip_tests=True,
        report_out=tmp_path / "reports" / "check.json",
    )


def test_preflight_reads_inherited_conda_environment_without_starting_shell(
    preflight_inputs, monkeypatch,
):
    def unexpected_subprocess(*args, **kwargs):
        raise AssertionError("Environment detection must not start another shell")

    monkeypatch.setattr(ef.subprocess, "run", unexpected_subprocess)

    assert ef.cmd_preflight(preflight_inputs) == 0
    report = json.loads(preflight_inputs.report_out.read_text())
    assert report["pass"] is True
    assert report["conda_env"] == "climada_env"
    assert report["python_executable"] == sys.executable


def test_preflight_retains_stderr_when_pytest_fails_before_printing_stdout(
    preflight_inputs, monkeypatch,
):
    stderr = f"{sys.executable}: No module named pytest\n"
    calls = []

    def failed_pytest(command, **kwargs):
        assert command[:3] == [sys.executable, "-m", "pytest"]
        assert Path(kwargs["cwd"]) == ef.REPO_ROOT
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr=stderr)

    preflight_inputs.skip_tests = False
    monkeypatch.setattr(ef.subprocess, "run", failed_pytest)

    assert ef.cmd_preflight(preflight_inputs) == 1
    report = json.loads(preflight_inputs.report_out.read_text())
    assert len(calls) == 1
    assert report["conda_env"] == "climada_env"
    assert report["tests"]["returncode"] == 1
    assert report["tests"]["stdout"] == ""
    assert report["tests"]["stderr"].strip() == stderr.strip()
    assert "No module named pytest" in report["tests"]["summary"]
    assert any("No module named pytest" in item for item in report["problems"])
