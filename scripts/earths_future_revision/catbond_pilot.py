"""Paired historical financial replay; no hazard generation or production overwrite."""
from pathlib import Path
import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from fl_risk_model import config as cfg, runner, catbonds
from fl_risk_model.mc_run_events import _prepare_common_inputs, run_one_iteration, SCENARIOS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', required=True, choices=SCENARIOS)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg.DEBUG_PRINTS = False
    warnings.filterwarnings('ignore', category=FutureWarning)
    legacy_path = Path(__file__).with_name('catbonds_before_review.py')
    spec = importlib.util.spec_from_file_location('catbonds_before_review', legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    args.out.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        common = _prepare_common_inputs()
    rows, bonds = [], []
    started = time.monotonic()
    for variant, mod, input_name in [('original', legacy, 'catbonds_2024.csv'),
                                     ('corrected', catbonds, 'catbonds_2024_reviewed.csv')]:
        cfg.CATBONDS_CSV = cfg.DATA_DIR / input_name
        runner.load_catbond_table = mod.load_catbond_table
        rng = np.random.default_rng(args.seed)
        for i in range(args.n):
            def record(**kwargs):
                recov, diag = mod.apply_catbond_recovery(**kwargs)
                bd = diag['bond_diag'].copy()
                bd['variant'], bd['iteration'], bd['scenario'] = variant, i + 1, args.scenario
                bonds.append(bd)
                return recov, diag
            runner.apply_catbond_recovery = record
            with contextlib.redirect_stdout(io.StringIO()):
                row = run_one_iteration(args.scenario, SCENARIOS[args.scenario], rng, common)
            row.update(variant=variant, iteration=i + 1)
            rows.append(row)
        print(args.scenario, variant, args.n, 'complete', round(time.monotonic()-started, 1), 'seconds', flush=True)
    df = pd.DataFrame(rows)
    paired = df.pivot(index='iteration', columns='variant')
    unchanged = ['total_damage_usd', 'wind_total_usd', 'water_total_usd',
                 'insured_private_wind_pre_usd', 'insured_citizens_wind_pre_usd',
                 'fhcf_recovery_private_usd', 'fhcf_recovery_citizens_usd', 'nfip_borrowed_usd']
    for c in unchanged:
        np.testing.assert_allclose(paired[c]['original'], paired[c]['corrected'], rtol=1e-12, atol=0.01)
    df.to_csv(args.out / f'{args.scenario}_paired.csv', index=False)
    pd.concat(bonds, ignore_index=True).to_csv(args.out / f'{args.scenario}_bonds.csv', index=False)
    (args.out / f'{args.scenario}_checks.json').write_text(json.dumps({
        'scenario': args.scenario, 'n_per_variant': args.n, 'seed': args.seed,
        'unchanged_fields_verified': unchanged, 'elapsed_seconds': time.monotonic()-started,
        'scope': 'Historical paired financial comparison; not revised probabilistic or climate results.'
    }, indent=2))


if __name__ == '__main__':
    main()
