"""Cluster cat-bond pilot validation; no Slurm submissions or hazard generation."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'earths_future_revision'))
import catbond_pilot_era5 as pilot


def write_pair(tmp_path):
    paths = [tmp_path / 'original', tmp_path / 'corrected']
    for i, path in enumerate(paths):
        path.mkdir()
        rows = [{col: 100.0 for col in dict.fromkeys([*pilot.UPSTREAM, *pilot.FINANCIAL])},
                {col: np.nan for col in dict.fromkeys([*pilot.UPSTREAM, *pilot.FINANCIAL])}]
        rows[0].update(year_id=1, scenario='year_1', catbond_payout_usd=10.0 + i,
                       figa_residual_deficit_usd=20.0 + i, defaults_post=1)
        rows[1].update(year_id=2, scenario='zero_events')
        pd.DataFrame(rows).to_csv(path / 'iterations.csv', index=False)
    return paths


def test_zero_event_seasons_preserved_and_financial_changes_allowed(tmp_path):
    old, new = write_pair(tmp_path)
    result = pilot.compare_outputs(old, new, 2)
    assert result['pass'], result
    assert result['means']['catbond_payout_usd'] == {'original': 5.0, 'corrected': 5.5}


@pytest.mark.parametrize('problem', ['missing', 'fhcf_changed', 'nan', 'ids', 'error', 'no_payout'])
def test_invalid_paired_outputs_block_production(tmp_path, problem):
    old, new = write_pair(tmp_path)
    df = pd.read_csv(new / 'iterations.csv')
    if problem == 'missing':
        df = df.drop(columns='fhcf_recovery_private_usd')
    elif problem == 'fhcf_changed':
        df.loc[0, 'fhcf_recovery_private_usd'] += 10
    elif problem == 'nan':
        df.loc[0, 'catbond_payout_usd'] = np.nan
    elif problem == 'ids':
        df.loc[0, 'year_id'] = 9
    elif problem == 'error':
        df.loc[0, 'scenario'] = 'error'
    elif problem == 'no_payout':
        df.catbond_payout_usd = 0
        other = pd.read_csv(old / 'iterations.csv')
        other.catbond_payout_usd = 0
        other.to_csv(old / 'iterations.csv', index=False)
    df.to_csv(new / 'iterations.csv', index=False)
    assert not pilot.compare_outputs(old, new, 2)['pass']


@pytest.mark.parametrize('changed', ['commit', 'dirty', 'inputs', 'seed', 'none'])
def test_report_requires_current_committed_source_and_inputs(tmp_path, monkeypatch, changed):
    old, new = write_pair(tmp_path)
    revision = {'commit': 'abc', 'is_dirty_tracked_source': False}
    hashes = {'data/reviewed': 'same'}
    manifest = {'output_dirs': {'original': str(old), 'corrected': str(new)},
                'code_revision': revision.copy(), 'input_hashes': hashes.copy(),
                'impact_dir': str(tmp_path), 'n_years': 2, 'seed': 42}
    if changed == 'commit':
        revision['commit'] = 'def'
    elif changed == 'dirty':
        revision['is_dirty_tracked_source'] = True
    elif changed == 'inputs':
        hashes['data/reviewed'] = 'changed'
    elif changed == 'seed':
        manifest['seed'] = 7
    (tmp_path / 'catbond_pilot_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(pilot.cluster, 'git_revision', lambda: revision)
    monkeypatch.setattr(pilot, 'input_hashes', lambda _: hashes)
    result = pilot.report(tmp_path, expected_seasons=2)
    assert result['pass'] == (changed == 'none')


def test_pilot_reuses_production_driver_and_restores_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'input_hashes', lambda _: {'input': 'present'})
    monkeypatch.setattr(pilot.cluster, 'git_revision',
                        lambda: {'commit': 'abc', 'is_dirty_tracked_source': False})
    saved = (pilot.cfg.CATBONDS_CSV, pilot.cfg.SYNTHETIC_EVENT_DIR,
             pilot.runner.apply_catbond_recovery)
    calls = []
    def fake_run(**kwargs):
        calls.append((kwargs, pilot.cfg.CATBONDS_CSV, pilot.runner.apply_catbond_recovery))
        return kwargs['out_dir'] / 'run'
    monkeypatch.setattr(pilot, 'run_stochastic_tc_monte_carlo', fake_run)
    manifest = pilot.run_pilot(tmp_path / 'cache', tmp_path / 'out')
    assert [call[0]['seed'] for call in calls] == [42, 42]
    assert [call[0]['n_years'] for call in calls] == [200, 200]
    assert calls[0][0]['year_sets_csv'] == calls[1][0]['year_sets_csv']
    assert [call[1].name for call in calls] == ['catbonds_2024.csv', 'catbonds_2024_reviewed.csv']
    assert calls[0][2] is pilot.legacy.apply_catbond_recovery
    assert calls[1][2] is pilot.catbonds.apply_catbond_recovery
    assert set(manifest['output_dirs']) == {'original', 'corrected'}
    assert saved == (pilot.cfg.CATBONDS_CSV, pilot.cfg.SYNTHETIC_EVENT_DIR,
                     pilot.runner.apply_catbond_recovery)


def test_cluster_entrypoint_submits_catbond_pilot_to_separate_root(tmp_path, monkeypatch):
    import os
    import subprocess
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    sbatch = bindir / 'sbatch'
    sbatch.write_text('#!/bin/bash\necho 12345\n')
    sbatch.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bindir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('EF_PROJECT_DIR', str(tmp_path / 'checkout'))
    monkeypatch.setenv('EF_PYTHON', sys.executable)
    result = subprocess.run(['bash', str(ROOT / 'scripts/cluster/catbond_revision.sh'), 'pilot'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    out = tmp_path / 'checkout/results/mc_runs_catbond_patched'
    script, = out.glob('pilot_era5_*/submit_pilot.sh')
    text = script.read_text()
    assert 'catbond_pilot_era5.py' in text
    assert 'old_both_bugs' not in text
    assert 'fhcf_pilot_era5.py' not in text
    manifest = tmp_path / 'checkout/results/earths_future_revision/catbond_cluster/manifests/pilot_manifest_latest.json'
    assert json.loads(manifest.read_text())['jobs'][0]['slurm_job_id'] == '12345'
