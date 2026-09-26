"""Exercise the public-input demo and its failure reporting."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('public_demo', ROOT/'scripts/demo/run_demo.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_demo_with_citizens_bonds_runs_without_licensed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.cfg, 'MARKET_SHARE_XLSX', tmp_path/'absent_market_share.xlsx')
    monkeypatch.setattr(demo.cfg, 'SURPLUS_FILE', tmp_path/'absent_surplus.xlsx')
    result = demo.run_demo(n_iter=2, seed=42)
    assert len(result) == 2
    assert not result.scenario.eq('error').any(), result.to_dict('records')
    for column in ['total_damage_usd', 'catbond_payout_usd', 'figa_residual_deficit_usd']:
        assert np.isfinite(result[column]).all()
    assert result.catbond_payout_usd.gt(0).any()


def test_failed_demo_exits_without_reference_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(demo, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(demo.sys, 'argv', ['run_demo.py', '--n_iter', '1'])
    monkeypatch.setattr(demo, 'run_demo', lambda **kwargs: pd.DataFrame([
        {'iteration': 0, 'scenario': 'error', 'error': 'fixture failure'}]))
    with pytest.raises(SystemExit, match='Demo failed'):
        demo.main()
    assert (tmp_path/'demo_output/demo_summary.csv').exists()
    assert not (tmp_path/'demo_output/expected_demo_summary.csv').exists()
