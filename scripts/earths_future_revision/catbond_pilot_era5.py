#!/usr/bin/env python3
"""Paired ERA5 financial replay with the original and reviewed cat-bond layer.

Both variants retain the corrected FHCF implementation and reuse the same cached
hazards, first 200 year IDs, and random seed. This pilot checks the production
stochastic path; it is not a source of headline probabilities or return levels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'cluster'))

import numpy as np
import pandas as pd
import earths_future_lib as cluster
from fl_risk_model import catbonds, config as cfg, runner
from fl_risk_model.mc_run_events import run_stochastic_tc_monte_carlo
import catbonds_before_review as legacy

UPSTREAM = [
    'total_damage_usd', 'wind_total_usd', 'water_total_usd',
    'insured_private_wind_pre_usd', 'insured_citizens_wind_pre_usd',
    'fhcf_recovery_private_usd', 'fhcf_recovery_citizens_usd',
    'nfip_borrowed_usd', 'nfip_claims_paid_usd',
]
FINANCIAL = ['catbond_payout_usd', 'figa_residual_deficit_usd',
             'citizens_residual_deficit_usd', 'nfip_borrowed_usd', 'defaults_post']


def input_hashes(impact_dir: Path) -> dict:
    return {
        **cluster.hash_active_inputs(),
        **{f'impact/{name}': cluster.file_sha256(impact_dir / name)
           for name in [cluster.YEAR_SETS_FILENAME, cluster.EVENT_METADATA_FILENAME]},
    }


def run_pilot(impact_dir: Path, out_root: Path, n_years: int = 200, seed: int = 42) -> dict:
    hashes = input_hashes(impact_dir)
    missing = [name for name, value in hashes.items() if value == 'MISSING']
    if missing:
        raise FileNotFoundError(f'Required cached inputs missing (no regeneration): {missing}')
    revision = cluster.git_revision()
    if revision['is_dirty_tracked_source']:
        raise RuntimeError('Commit source changes before running the paired cluster pilot.')
    manifest_path = out_root / 'catbond_pilot_manifest.json'
    if out_root.exists() and any(out_root.glob('original/*')):
        raise FileExistsError('Use a fresh pilot output directory; original output already exists.')
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        'purpose': 'paired cat-bond correction pilot; FHCF fixed in both variants',
        'code_revision': revision, 'input_hashes': hashes,
        'impact_dir': str(impact_dir.resolve()), 'n_years': n_years, 'seed': seed,
        'output_dirs': {},
    }
    saved = (cfg.SYNTHETIC_EVENT_DIR, cfg.SYNTHETIC_EVENT_METADATA_CSV,
             cfg.CATBONDS_CSV, runner.load_catbond_table, runner.apply_catbond_recovery)
    cfg.SYNTHETIC_EVENT_DIR = impact_dir
    cfg.SYNTHETIC_EVENT_METADATA_CSV = impact_dir / cluster.EVENT_METADATA_FILENAME
    try:
        for label, module, filename in [
            ('original', legacy, 'catbonds_2024.csv'),
            ('corrected', catbonds, 'catbonds_2024_reviewed.csv'),
        ]:
            cfg.CATBONDS_CSV = cfg.DATA_DIR / filename
            runner.load_catbond_table = module.load_catbond_table
            runner.apply_catbond_recovery = module.apply_catbond_recovery
            run_dir = run_stochastic_tc_monte_carlo(
                year_sets_csv=impact_dir / cluster.YEAR_SETS_FILENAME,
                n_years=n_years, seed=seed, out_dir=out_root / label,
                run_label=f'catbond_pilot_era5_{label}',
            )
            manifest['output_dirs'][label] = str(Path(run_dir).resolve())
    finally:
        (cfg.SYNTHETIC_EVENT_DIR, cfg.SYNTHETIC_EVENT_METADATA_CSV,
         cfg.CATBONDS_CSV, runner.load_catbond_table, runner.apply_catbond_recovery) = saved
    if cluster.git_revision()['commit'] != revision['commit'] or input_hashes(impact_dir) != hashes:
        raise RuntimeError('Code or inputs changed during the pilot; repeat from a fixed checkout.')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def compare_outputs(old_dir: Path, new_dir: Path, n_years: int) -> dict:
    validations = {label: cluster.validate_run_dir(path, n_years)
                   for label, path in [('original', old_dir), ('corrected', new_dir)]}
    result = {'runs': validations, 'problems': []}
    if not all(v['pass'] for v in validations.values()):
        result.update({'pass': False, 'problems': ['One or both output sets failed validation.']})
        return result
    frames = [pd.read_csv(Path(v['run_dir']) / 'iterations.csv') for v in validations.values()]
    old, new = frames
    required = ['year_id', 'scenario', *UPSTREAM, *FINANCIAL]
    for label, df in zip(validations, frames):
        missing = sorted(set(required) - set(df.columns))
        if missing:
            result['problems'].append(f'{label}: missing columns {missing}')
    if result['problems']:
        result['pass'] = False
        return result
    old, new = [df.sort_values('year_id').reset_index(drop=True) for df in frames]
    if not old.year_id.equals(new.year_id) or not old.scenario.equals(new.scenario):
        result['problems'].append('Season IDs or scenario labels differ between paired runs.')
    for df in [old, new]:
        # The production writer omits financial fields for explicit zero-event years.
        zero = df.scenario.eq('zero_events')
        for col in dict.fromkeys([*UPSTREAM, *FINANCIAL]):
            df[col] = pd.to_numeric(df[col], errors='raise')
            df.loc[zero, col] = df.loc[zero, col].fillna(0.0)
            if not np.isfinite(df[col]).all():
                result['problems'].append(f'Missing or non-finite active-season values in {col}.')
    result['upstream_columns_checked'] = UPSTREAM
    result['upstream_columns_mismatched'] = [
        col for col in UPSTREAM
        if not np.allclose(old[col], new[col], rtol=1e-12, atol=0.01, equal_nan=False)
    ]
    if result['upstream_columns_mismatched']:
        result['problems'].append('Physical losses, initial allocation, FHCF, or NFIP changed.')
    result['means'] = {
        col: {'original': float(old[col].mean()), 'corrected': float(new[col].mean())}
        for col in FINANCIAL
    }
    result['mean_residual_financing_requirement_usd'] = {
        label: float(df[['figa_residual_deficit_usd', 'citizens_residual_deficit_usd',
                         'nfip_borrowed_usd']].sum(axis=1).mean())
        for label, df in [('original', old), ('corrected', new)]
    }
    if not (old.catbond_payout_usd.gt(0).any() or new.catbond_payout_usd.gt(0).any()):
        result['problems'].append('No bond payout in either variant; this pilot did not exercise the layer.')
    result['pass'] = not result['problems']
    return result


def report(out_root: Path, expected_seasons: int = 200) -> dict:
    manifest = json.loads((out_root / 'catbond_pilot_manifest.json').read_text())
    result = compare_outputs(Path(manifest['output_dirs']['original']),
                             Path(manifest['output_dirs']['corrected']), expected_seasons)
    current = cluster.git_revision()
    if (current['is_dirty_tracked_source'] or
            manifest['code_revision']['is_dirty_tracked_source'] or
            current['commit'] != manifest['code_revision']['commit']):
        result['problems'].append('Pilot and current checkout must have the same committed source.')
    if input_hashes(Path(manifest['impact_dir'])) != manifest['input_hashes']:
        result['problems'].append('Financial inputs or ERA5 catalog changed since the pilot.')
    if manifest['n_years'] != expected_seasons or manifest['seed'] != 42:
        result['problems'].append('Pilot must use the requested season count and seed 42.')
    result['pass'] = not result['problems']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--impact-dir', type=Path,
                        default=Path(cluster.DEFAULT_IMPACT_ROOT) / 'FL_era5_reanalcal')
    parser.add_argument('--n-years', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--report-out', type=Path)
    args = parser.parse_args()
    if args.report:
        result = report(args.out_root, args.n_years)
        if args.report_out:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return 0 if result['pass'] else 1
    run_pilot(args.impact_dir, args.out_root, args.n_years, args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
