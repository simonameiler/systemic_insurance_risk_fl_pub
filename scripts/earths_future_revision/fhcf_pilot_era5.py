#!/usr/bin/env python3
"""Paired ERA5 pilot: pre-FHCF-patch vs. corrected calculation, through the
actual stochastic production entry point
(fl_risk_model.mc_run_events.run_stochastic_tc_monte_carlo), on the first
200 ERA5 year IDs (seed 42), including zero-event years.

This is the "paired ERA5 comparison" scripts/cluster/earths_future.sh's
`pilot` command submits on the cluster. It is a small, thin driver: it does
not reimplement the Monte Carlo loop (that stays in mc_run_events.py) and
does not reimplement the FHCF variants (those are imported from
fhcf_pilot_isolate_fixes.py, which already defines and tests them against
the historical-scenario pilot). What is new here, relative to that file, is
using run_stochastic_tc_monte_carlo -- the production stochastic entry
point -- against the ERA5 synthetic year-set catalog, instead of the
historical-scenario driver.

`n_years=200` is passed straight to run_stochastic_tc_monte_carlo, which
filters `year_sets[year_sets['year_id'] <= n_years]` (fl_risk_model/
mc_run_events.py): this selects the FIRST 200 year IDs by construction, not
a random sample, and each year_id's zero-event/nonzero-event status is
whatever the pre-generated year-set file assigned it (zero-event years are
recorded as explicit rows, not dropped -- see that function's own
docstring and the code around "Zero-event year (from Poisson) - record
it!").

This script does not, and must not, generate a year-set file or an impact
cache: if the expected files are not found at --impact-dir, it fails with a
concrete, actionable error (see `_require_inputs`), exactly like
run_emanuel_monte_carlo.py and run_climate_buildingcode_sensitivity_windfloods.py
already do, rather than attempting to regenerate hazard data.

This does not report headline return levels: 200 seasons is far too few to
support the ERA5 baseline's return-period estimates (see
docs/earths_future_revision/correction_register.md and
fhcf_patch_and_pilot_report.md). Its purpose is to confirm the cluster
environment and the stochastic (not historical-footprint) code path behave
identically to the validated local historical pilot: same upstream draws
regardless of FHCF variant, gross losses and NFIP unaffected by the patch,
and correct company/statewide FHCF reconciliation.

Usage
-----
    conda activate climada_env
    python scripts/earths_future_revision/fhcf_pilot_era5.py \\
        --event-set FL_era5_reanalcal \\
        --impact-dir /home/groups/bakerjw/smeiler/climada_data/data/impact/impacts/FL_era5_reanalcal \\
        --n-years 200 --seed 42 \\
        --out-root results/mc_runs_fhcf_patched/pilot_era5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fhcf_pilot_isolate_fixes import VARIANTS, _set_variant, _file_sha256  # noqa: E402

from fl_risk_model import config as cfg  # noqa: E402
from fl_risk_model.mc_run_events import run_stochastic_tc_monte_carlo  # noqa: E402

DEFAULT_IMPACT_ROOT = "/home/groups/bakerjw/smeiler/climada_data/data/impact/impacts"
DEFAULT_YEAR_SETS_FILE = "year_sets_N10000_seed42.csv"
# Default variant pair for the "paired ERA5 comparison" the brief specifies.
# All four variants (adding formula_fix_only / aggregation_fix_only) remain
# available via --variants for anyone who wants the same isolation the
# local historical pilot ran.
DEFAULT_VARIANTS = ["old_both_bugs", "both_fixed"]


def _require_inputs(impact_dir: Path, year_sets_file: str) -> tuple[Path, Path]:
    """Concrete, actionable missing-path report. Never regenerates hazard
    data or falls back to a synthetic substitute."""
    problems = []
    if not impact_dir.is_dir():
        problems.append(f"impact directory not found: {impact_dir}")
    year_sets_csv = impact_dir / year_sets_file
    if not year_sets_csv.exists():
        problems.append(f"year-sets file not found: {year_sets_csv}")
    metadata_csv = impact_dir / "event_metadata.csv"
    if not metadata_csv.exists():
        problems.append(f"event metadata not found: {metadata_csv}")
    if problems:
        msg = "\n".join(f"  - {p}" for p in problems)
        raise FileNotFoundError(
            "Missing required ERA5 pilot inputs; not regenerating hazard data:\n"
            f"{msg}\n"
            "Generate these on the cluster with scripts/hazard/generate_emanuel_year_sets.py "
            "and scripts/hazard/precompute_emanuel_tc_impacts.py, or point --impact-dir at an "
            "existing cache."
        )
    return year_sets_csv, metadata_csv


def _git_revision() -> dict:
    def _run(args):
        return subprocess.run(["git", "-C", str(REPO_ROOT)] + args,
                               capture_output=True, text=True).stdout.strip()
    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "describe": _run(["describe", "--always", "--dirty"]),
        "is_dirty_tracked_source": bool(_run(["status", "--porcelain",
                                               "--", "fl_risk_model", "scripts"])),
    }


# Every active financial input the ERA5 pilot's financial-model call path
# reads, verified against fl_risk_model/data/ (`ls fl_risk_model/data/*.csv
# *.xlsx`), not just the hazard/premium/surplus subset the local historical
# pilot's manifest hashed (independent review, item 3).
ACTIVE_INPUT_FILES = [
    "24fin_fhcf.csv", "fhcf_terms_keyed.csv", "company_keys.csv",
    "FL HO Market Share Report_6.10.25.xlsx",
    "20250805 FL Surplus Capital, Group v Entity.xlsx",
    "FHCF_2024_Exposure_byCounty.xlsx",
    "citizens_capital_pml.csv", "citizens_county_data.csv",
    "citizens_county_data_all_harmonized.csv",
    "NfipResidentialPenetrationRates.csv",
    "nfip_FL_coverage_premium_by_year.csv", "nfip_FL_claims_by_year.csv",
    "catbonds_2024.csv",
    "county_region.csv", "fl_county_fips.csv", "florida_coastal_counties.csv",
    "florida_log_contribution_p95_present.csv",
]


def _input_manifest(impact_dir: Path, year_sets_csv: Path, metadata_csv: Path) -> dict:
    manifest = {}
    for p in [year_sets_csv, metadata_csv]:
        manifest[f"impact/{p.name}"] = _file_sha256(p) if p.exists() else "MISSING"
    for fname in ACTIVE_INPUT_FILES:
        p = cfg.DATA_DIR / fname
        manifest[f"data/{fname}"] = _file_sha256(p) if p.exists() else "MISSING (not found by this name)"
    return manifest


def run_pilot(event_set: str, impact_dir: Path, year_sets_file: str,
              n_years: int, seed: int, out_root: Path,
              variants: list[str]) -> dict:
    year_sets_csv, metadata_csv = _require_inputs(impact_dir, year_sets_file)

    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "paired ERA5 pilot (ephemeral cluster environment/stochastic-path check, "
                   "NOT a source of headline return levels)",
        "event_set": event_set,
        "impact_dir": str(impact_dir),
        "year_sets_file": year_sets_file,
        "n_years": n_years,
        "seed": seed,
        "variants": variants,
        "code_revision": _git_revision(),
        "input_checksums_sha256_16": _input_manifest(impact_dir, year_sets_csv, metadata_csv),
        "output_dirs": {},
    }

    original_event_dir = cfg.SYNTHETIC_EVENT_DIR
    original_metadata_csv = cfg.SYNTHETIC_EVENT_METADATA_CSV
    cfg.SYNTHETIC_EVENT_DIR = impact_dir
    cfg.SYNTHETIC_EVENT_METADATA_CSV = metadata_csv

    try:
        for variant_name in variants:
            if variant_name not in VARIANTS:
                raise ValueError(f"unknown variant {variant_name!r}; choices: {list(VARIANTS)}")
            _set_variant(VARIANTS[variant_name])
            variant_out_root = out_root / variant_name
            print(f"\n=== ERA5 pilot: event_set={event_set} variant={variant_name} "
                  f"n_years={n_years} seed={seed} ===")
            run_dir = run_stochastic_tc_monte_carlo(
                year_sets_csv=year_sets_csv,
                n_years=n_years,
                seed=seed,
                out_dir=variant_out_root,
                run_label=f"fhcf_pilot_era5_{variant_name}",
            )
            manifest["output_dirs"][variant_name] = str(run_dir)
            print(f"  -> {run_dir}")
    finally:
        cfg.SYNTHETIC_EVENT_DIR = original_event_dir
        cfg.SYNTHETIC_EVENT_METADATA_CSV = original_metadata_csv
        # Restore the production (patched) implementation for any code that
        # imports fl_risk_model.fhcf/runner/branches.citizens after this
        # function returns in the same process.
        _set_variant(VARIANTS["both_fixed"])

    manifest_path = out_root / "pilot_era5_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {manifest_path}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event-set", default="FL_era5_reanalcal")
    ap.add_argument("--impact-dir", type=Path,
                     default=Path(DEFAULT_IMPACT_ROOT) / "FL_era5_reanalcal")
    ap.add_argument("--year-sets-file", default=DEFAULT_YEAR_SETS_FILE)
    ap.add_argument("--n-years", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path,
                     default=REPO_ROOT / "results" / "mc_runs_fhcf_patched" / "pilot_era5")
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                     choices=list(VARIANTS), help="Default: paired old vs. corrected only.")
    args = ap.parse_args()

    run_pilot(
        event_set=args.event_set,
        impact_dir=args.impact_dir,
        year_sets_file=args.year_sets_file,
        n_years=args.n_years,
        seed=args.seed,
        out_root=args.out_root,
        variants=args.variants,
    )


if __name__ == "__main__":
    main()
