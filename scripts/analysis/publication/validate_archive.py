"""Verify a complete public result archive before publication processing."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.cluster.campaign_inventory import build_job_list

SIMULATION_COMMIT = '014ed02c1b3cad1334411b82ce440a01f5b24f1b'
REQUIRED = [
    'total_damage_usd', 'wind_total_usd', 'water_total_usd',
    'figa_residual_deficit_usd', 'citizens_residual_deficit_usd', 'nfip_borrowed_usd',
    'fhcf_shortfall_usd', 'defaults_post', 'largest_entity_deficit_usd',
]


def validate_archive(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('software_version') != '1.1.0' or manifest.get('simulation_commit') != SIMULATION_COMMIT:
        raise ValueError('Archive version does not match the v1.1.0 simulation campaign.')
    expected = {j['name']:j['expected_seasons'] for j in build_job_list(Path('.'),Path('.'))}
    records = manifest['runs']
    names = [r['name'] for r in records]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError('Archive must contain exactly the 106 distinct publication analyses.')
    if manifest['n_runs'] != 106 or manifest['total_rows'] != 997000:
        raise ValueError('Incorrect campaign totals.')
    index = {}
    for record in records:
        name = record['name']
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Archive path escapes its root: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
        if record['execution_commit'] != manifest['simulation_commit'] or record['execution_exit_code'] != 0:
            raise ValueError(f'Inconsistent source execution: {name}')
        data = pd.read_csv(path, low_memory=False)
        if len(data) != expected[name] or record['expected_rows'] != expected[name]:
            raise ValueError(f'Incomplete run: {name}')
        if 'error' in data and data.error.notna().any():
            raise ValueError(f'Error records: {name}')
        if 'scenario' not in data or data.scenario.eq('error').any():
            raise ValueError(f'Invalid scenarios: {name}')
        key = 'iter' if name.startswith('historical_') else 'year_id'
        if key not in data or data[key].isna().any() or data[key].duplicated().any():
            raise ValueError(f'Missing or duplicate realization IDs: {name}')
        ids = set(data[key])
        valid_ids = [set(range(expected[name])),set(range(1,expected[name]+1))] if key=='iter' else [set(range(1,expected[name]+1))]
        if ids not in valid_ids:
            raise ValueError(f'Incomplete realization IDs: {name}')
        zero = data.scenario.eq('zero_events') & data.total_damage_usd.eq(0)
        for col in REQUIRED:
            if col not in data:
                raise ValueError(f'Missing {col}: {name}')
            values = pd.to_numeric(data[col],errors='coerce')
            if not np.isfinite(values[~zero]).all() or (values < -1e-4).any():
                raise ValueError(f'Invalid {col}: {name}')
        if not np.allclose(data.total_damage_usd.fillna(0),data.wind_total_usd.fillna(0)+data.water_total_usd.fillna(0),rtol=1e-10,atol=1e-3):
            raise ValueError(f'Gross losses do not reconcile: {name}')
        index[name] = str(path)
    return manifest,index
