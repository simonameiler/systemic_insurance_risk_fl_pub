"""Run historical stress tests and Great Miami policy scenarios on one checkout.

Usage: python scripts/code_paper_fixes_compare.py <repo_root> <out_csv> <n_iter> [legacy_surplus]
Writes nothing inside the repo; per-iteration scalar outputs go to <out_csv>.
"""
import sys
import time
import warnings
from pathlib import Path

repo, out_csv, n_iter = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, str(repo))

import numpy as np
import pandas as pd

from fl_risk_model import config as cfg
cfg.DEBUG_PRINTS = False
cfg.VERBOSE_EXPOSURE = False
if len(sys.argv) > 4 and sys.argv[4] == "legacy_surplus" and hasattr(cfg, "MISSING_SURPLUS_POLICY"):
    cfg.MISSING_SURPLUS_POLICY = "legacy_drop"

from fl_risk_model import mc_run_events as mc

assert Path(mc.__file__).resolve().is_relative_to(repo.resolve()), mc.__file__

HIST = ["great_miami", "andrew", "lake_okeechobee", "irma", "gm_then_andrew", "double_gm", "double_irma"]
POLICY = ["market_exit_moderate", "penetration_major", "building_codes_major"]

common = mc._prepare_common_inputs()
rows = []
t0 = time.time()
runs = [(s, None) for s in HIST] + [("great_miami", p) for p in POLICY]
for scen, policy in runs:
    rng = np.random.default_rng(42)
    pcfg = cfg.POLICY_SCENARIOS.get(policy) if policy else None
    for i in range(n_iter):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = mc.run_one_iteration(scen, mc.SCENARIOS[scen], rng, common,
                                       do_flood=mc.DO_FLOOD, surplus_year=mc.YEAR,
                                       policy_scenario_config=pcfg)
        row = {k: v for k, v in rec.items() if isinstance(v, (int, float, bool, np.integer, np.floating))}
        row.update(run=f"{scen}|{policy or 'baseline'}", iteration=i)
        rows.append(row)
    print(f"{scen}|{policy or 'baseline'} done ({time.time() - t0:.0f}s)", flush=True)

pd.DataFrame(rows).to_csv(out_csv, index=False)
