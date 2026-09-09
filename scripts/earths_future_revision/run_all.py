#!/usr/bin/env python3
"""Single entry point for the Earth's Future corrections and sensitivity work.

Runs, in order, every reproducible check and analysis that does not require
the proprietary hazard/exposure inputs listed in
docs/earths_future_revision/data_inventory.md. Each stage can also be run
individually; see the module docstring of each section script for its own
CLI and the exact commands recorded in
docs/earths_future_revision/correction_register.md for anything blocked.

Usage
-----
    conda activate climada_env
    cd scripts/earths_future_revision
    python run_all.py                 # everything below
    python run_all.py --tests-only    # just the pytest suite
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

STAGES = [
    ("pytest (accounting fixtures, seasonal aggregation, NFIP allocation, "
     "scenario totals -- 17 tests)",
     [sys.executable, "-m", "pytest", "fl_risk_model/tests/earths_future", "-v"],
     REPO_ROOT),

    ("Section 5: corrected return-period table (Table 1)",
     [sys.executable, "section5_return_periods.py", "--iterations",
      str(REPO_ROOT / "results/mc_runs/emanuel_era5_baseline_20260326_141913/iterations.csv")],
     HERE),

    ("Section 6: common-season burden decomposition",
     [sys.executable, "section6_decomposition.py", "--iterations",
      str(REPO_ROOT / "results/mc_runs/emanuel_era5_baseline_20260326_141913/iterations.csv")],
     HERE),

    ("Section 7: NFIP allocation county rate table",
     [sys.executable, "section7_nfip_allocation.py"],
     HERE),

    ("Section 7: insured-fraction sensitivity, corrected",
     [sys.executable, "section7_insured_fraction.py"],
     HERE),

    ("Section 8: season-count check, historical Table S3, Table S7 audit",
     [sys.executable, "section8_historical_and_variance.py"],
     HERE),

    ("Section 9: corrected scenario/climate means (Table S4)",
     [sys.executable, "section9_climate_policy_tables.py"],
     HERE),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests-only", action="store_true")
    args = ap.parse_args()

    stages = STAGES[:1] if args.tests_only else STAGES

    failures = []
    for label, cmd, cwd in stages:
        print("\n" + "=" * 88)
        print(label)
        print("=" * 88)
        result = subprocess.run(cmd, cwd=str(cwd))
        if result.returncode != 0:
            failures.append(label)

    print("\n" + "=" * 88)
    if failures:
        print(f"FAILED stages ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"All {len(stages)} stages completed. Outputs in "
              f"{REPO_ROOT / 'results/earths_future_revision'}")


if __name__ == "__main__":
    main()
