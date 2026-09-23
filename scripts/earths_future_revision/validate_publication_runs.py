"""Validate locally transferred production outputs without requiring Slurm."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--campaign-root', type=Path, help='Local copy of the production output root.')
    ap.add_argument('--log-dir', type=Path, help='Local directory containing Slurm execution logs.')
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo_root / 'scripts' / 'cluster'))
    from earths_future_lib import validate_run_dir, _expected_filename_for_job
    manifest = json.loads(args.manifest.read_text())
    records, index = [], {}
    original_root = Path(manifest['out_root'])
    campaign_root = args.campaign_root
    if campaign_root is None:
        parts = original_root.parts
        if 'results' not in parts:
            raise ValueError('Supply --campaign-root for a manifest outside results/.')
        campaign_root = args.repo_root.joinpath(*parts[parts.index('results'):])
    log_dir = args.log_dir or args.repo_root / 'logs'
    expected_commit = manifest['code_revision']['commit']
    for job in manifest['jobs']:
        relative = Path(job['output_dir']).relative_to(original_root)
        filename = _expected_filename_for_job(job['name'])
        report = validate_run_dir(campaign_root / relative, job['expected_seasons'], filename)
        report['name'] = job['name']
        path = Path(report['run_dir']) / filename
        log_path = log_dir / f"ef_production_{job['slurm_job_id']}.out"
        log = log_path.read_text(errors='replace') if log_path.exists() else ''
        commits = re.findall(r'code_revision_at_execution commit=(\S+)', log)
        exit_codes = re.findall(r'\[ef\] task=\d+ name=\S+ finished=.* exit=(\d+)', log)
        problems = report.setdefault('problems', [])
        if commits != [expected_commit]:
            problems.append(f'execution commit mismatch or missing: {commits}')
        if exit_codes != ['0']:
            problems.append(f'missing or unsuccessful log completion: {exit_codes}')
        report['execution_commit'] = commits
        report['exit_codes'] = exit_codes
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            report['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            zero = df['scenario'].eq('zero_events') if 'scenario' in df else pd.Series(False, index=df.index)
            report['zero_event_seasons'] = int(zero.sum())
            if 'error' in df and df['error'].notna().any():
                problems.append('nonempty error column')
            required = ['total_damage_usd', 'figa_residual_deficit_usd', 'citizens_residual_deficit_usd', 'nfip_borrowed_usd', 'fhcf_shortfall_usd']
            for c in required:
                if c not in df:
                    problems.append(f'missing {c}')
                    continue
                values = pd.to_numeric(df[c], errors='coerce')
                if not np.isfinite(values.loc[~zero]).all():
                    problems.append(f'nonfinite {c} in nonzero-event seasons')
                if (values < -1e-4).any():
                    problems.append(f'negative {c}')
            if 'year_id' in df and set(df['year_id']) != set(range(1, job['expected_seasons'] + 1)):
                problems.append('year_id coverage differs from expected complete seasons')
            if {'wind_total_usd', 'water_total_usd', 'total_damage_usd'} <= set(df):
                error = (df.total_damage_usd - df.wind_total_usd - df.water_total_usd).abs().fillna(0)
                report['max_gross_reconciliation_error_usd'] = float(error.max())
                if not np.allclose(df.total_damage_usd.fillna(0), df.wind_total_usd.fillna(0) + df.water_total_usd.fillna(0), rtol=1e-10, atol=1e-3):
                    problems.append('gross total does not equal wind plus flood')
            report['fhcf_shortfall_max_usd'] = float(df['fhcf_shortfall_usd'].fillna(0).max()) if 'fhcf_shortfall_usd' in df else None
            index[job['name']] = str(path)
        report['pass'] = report['pass'] and not problems
        records.append(report)
    passed = len(records) == 106 and all(x['pass'] for x in records)
    result = {'pass': passed, 'manifest': str(args.manifest), 'source_commit': expected_commit, 'n_runs': len(records), 'total_rows': sum(x.get('n_rows', 0) for x in records), 'runs': records}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    if passed:
        (args.out_dir / 'validated_runs.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'runs'}, indent=2))
    for r in records:
        if not r['pass']:
            print(r['name'], r.get('reason'), r['problems'])
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
