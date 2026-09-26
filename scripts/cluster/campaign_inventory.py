from __future__ import annotations

import argparse

import json

from pathlib import Path

GCMS = ["canesm", "cnrm6", "ecearth6", "ipsl6", "miroc6"]

PERIODS = ["20thcal", "ssp245cal", "ssp245_2cal", "ssp585cal", "ssp585_2cal"]

POLICY_NAMES = ["market_exit_moderate", "penetration_major", "building_codes_major"]

HISTORICAL_SCENARIOS = ["great_miami", "andrew", "gm_then_andrew", "double_gm",
                         "lake_okeechobee", "irma", "double_irma"]

INSURED_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]

BUILDING_CODE_LEVELS = list(range(13))

BUILDING_CODE_PERIOD = "ssp245cal"

def build_job_list(out_root: Path, impact_root: Path) -> list[dict]:
    """Return the 106 manuscript analyses with explicit input and output paths."""
    jobs = []

    def emanuel_cmd(event_set, out_dir, policy=None):
        cmd = ["python", "scripts/run/run_emanuel_monte_carlo.py",
               "--event_set", event_set,
               "--impact_dir", str(impact_root / event_set),
               "--seed", "42", "--out", str(out_dir)]
        if policy:
            cmd += ["--policy", policy]
        return cmd

    era5_dir = out_root / "era5_baseline"
    jobs.append({"name": "era5_baseline", "expected_seasons": 10000, "out_dir": era5_dir,
                 "cmd": emanuel_cmd("FL_era5_reanalcal", era5_dir)})

    for policy in POLICY_NAMES:
        d = out_root / f"era5_policy_{policy}"
        jobs.append({"name": f"era5_policy_{policy}", "expected_seasons": 10000, "out_dir": d,
                     "cmd": emanuel_cmd("FL_era5_reanalcal", d, policy=policy)})

    for gcm in GCMS:
        for period in PERIODS:
            event_set = f"FL_{gcm}_{period}"
            d = out_root / "gcm_baseline" / f"{gcm}_{period}"
            jobs.append({"name": f"gcm_baseline_{gcm}_{period}", "expected_seasons": 10000,
                         "out_dir": d, "cmd": emanuel_cmd(event_set, d)})

    for gcm in GCMS:
        for level in BUILDING_CODE_LEVELS:
            event_set = f"FL_{gcm}_{BUILDING_CODE_PERIOD}"
            d = out_root / "buildingcode" / f"{gcm}_L{level:02d}"
            cmd = ["python", "scripts/run/run_climate_buildingcode_sensitivity_windfloods.py",
                   "--code_level", str(level), "--event_set", event_set,
                   "--impact_dir", str(impact_root / event_set),
                   "--seed", "42", "--out_dir", str(d)]
            jobs.append({"name": f"buildingcode_{gcm}_L{level:02d}", "expected_seasons": 10000,
                         "out_dir": d, "cmd": cmd})

    for scenario in HISTORICAL_SCENARIOS:
        d = out_root / "historical" / scenario
        cmd = ["python", "scripts/run/run_historical_scenarios_mc.py",
               "--scenario", scenario, "--n_iter", "1000", "--seed", "42",
               "--out", str(d), "--skip_report"]
        jobs.append({"name": f"historical_{scenario}", "expected_seasons": 1000,
                     "out_dir": d, "cmd": cmd})

    for frac in INSURED_FRACTIONS:
        d = out_root / "insured_fraction" / f"frac_{frac}"
        cmd = ["python", "scripts/run/run_insured_fraction_sensitivity.py",
               "--fractions", str(frac), "--seed", "42", "--out_dir", str(d),
               "--impact_dir", str(impact_root / "FL_era5_reanalcal")]
        jobs.append({"name": f"insured_fraction_{frac}", "expected_seasons": 10000,
                     "out_dir": d, "cmd": cmd})

    assert len(jobs) == 106, f"expected 106 production jobs, built {len(jobs)}"
    return jobs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the complete manuscript simulation inventory.')
    parser.add_argument('--impact-root', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jobs = build_job_list(args.out_root.resolve(), args.impact_root.resolve())
    for job in jobs:
        job['out_dir'] = str(job['out_dir'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jobs, indent=2) + '\n')
    print(f'Wrote {len(jobs)} analysis commands to {args.output}')
