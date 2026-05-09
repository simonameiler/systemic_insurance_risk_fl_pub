#!/usr/bin/env python3
"""
run_demo.py  —  Lightweight demo of the Florida systemic insurance risk model
==============================================================================

Purpose
-------
Run a self-contained demonstration of the full risk-propagation workflow using
only publicly redistributable data.  The demo uses:

    • Real public hazard data — Great Miami Hurricane (1926) county-level wind
      impacts derived from IBTrACS tracks via a CLIMADA pipeline (included in
      fl_risk_model/data/hazard/historical_events/).
    • Real public institutional data — FHCF county exposure, Citizens Property
      Insurance county data, NFIP premium and policy data (all included).
    • Synthetic financial data — Pareto-distributed market shares and plausible
      surplus figures for all 99 real Florida private homeowners insurers,
      substituting for the proprietary S&P Capital IQ data.  Company names and
      statutory entity keys come from the public regulatory record
      (fl_risk_model/data/company_keys.csv); only the financial values are
      synthetic.  See demo_data/README.md for details.

Scenario
--------
    Great Miami Hurricane (1926) — IBTrACS ID 1926255N15314
    This is the primary illustrative event in the manuscript (Fig. 2).
    The pre-computed county-level impact footprint is in:
        fl_risk_model/data/hazard/historical_events/1926255N15314.csv
    For full transparency on how that file was produced, see:
        scripts/demo/run_climada_hazard_pipeline.py  (requires CLIMADA)

Why Great Miami matters
-----------------------
At 2024 Florida exposure levels, the 1926 Great Miami Hurricane would generate
total gross losses on the order of $150–200 B (mean of 1 000 MC iterations),
with ~$35–45 B absorbed by the private HO insurance market — well in excess of
the ~$18 B in synthetic entity surplus assigned to private carriers.  This makes
systemic stress (multiple insurer defaults, FHCF near-depletion, FIGA stress)
visible in the demo, illustrating why the paper studies systemic risk.

This demo does NOT require:
    • WindRiskTech L.L.C. / MIT model synthetic TC event sets.
    • S&P Capital IQ licensed market-share or surplus data.
    • An HPC cluster — typical runtime is 2–5 minutes on a laptop.

Output
------
    demo_output/demo_summary.csv          — per-iteration MC results
    demo_output/expected_demo_summary.csv — reference snapshot (first run only)

Usage
-----
    python scripts/demo/run_demo.py
    python scripts/demo/run_demo.py --n_iter 50 --seed 7
    python scripts/demo/run_demo.py --out my_output_dir
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make repo root importable regardless of cwd
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Early dependency check
# ---------------------------------------------------------------------------
_MISSING_DEPS: list[str] = []
for _pkg in ["numpy", "pandas", "scipy"]:
    try:
        __import__(_pkg)
    except ImportError:
        _MISSING_DEPS.append(_pkg)

if _MISSING_DEPS:
    sys.exit(
        "\n[demo] ERROR: Missing Python dependencies: "
        + ", ".join(_MISSING_DEPS)
        + "\n\nInstall with:\n    pip install -e .\nor:\n    pip install numpy pandas scipy\n"
    )

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Check required demo and model data files
# ---------------------------------------------------------------------------
DEMO_DATA_DIR  = REPO_ROOT / "demo_data"
MODEL_DATA_DIR = REPO_ROOT / "fl_risk_model" / "data"

_REQUIRED_FILES = {
    "demo_market_share":  DEMO_DATA_DIR / "demo_market_share.csv",
    "demo_surplus":       DEMO_DATA_DIR / "demo_surplus.csv",
    "fhcf_exposure":      MODEL_DATA_DIR / "FHCF_2024_Exposure_byCounty.xlsx",
    "citizens_county":    MODEL_DATA_DIR / "citizens_county_data_all_harmonized.csv",
    "citizens_capital":   MODEL_DATA_DIR / "citizens_capital_pml.csv",
    "nfip_policies":      MODEL_DATA_DIR / "nfip_FL_coverage_premium_by_year.csv",
    "fl_county_fips":     MODEL_DATA_DIR / "fl_county_fips.csv",
    "great_miami_event":  MODEL_DATA_DIR / "hazard" / "historical_events" / "1926255N15314.csv",
    "catbonds":           MODEL_DATA_DIR / "catbonds_2024.csv",
}

_missing = {k: str(v) for k, v in _REQUIRED_FILES.items() if not v.exists()}
if _missing:
    sys.exit(
        "\n[demo] ERROR: Missing required files:\n"
        + "\n".join(f"  {k}: {v}" for k, v in _missing.items())
        + "\n\nEnsure the repository was cloned in full and "
          "'pip install -e .' was run from the repository root.\n"
    )


# ---------------------------------------------------------------------------
# Import model modules (after path setup)
# ---------------------------------------------------------------------------
try:
    from fl_risk_model import config as cfg
    from fl_risk_model.loader import load_fhcf_county_exposure, load_nfip_policy_coverage
    from fl_risk_model.capital import (
        load_citizens_capital_row_from_csv,
        apply_losses_to_surplus,
        apply_group_capital_contributions,
    )
    from fl_risk_model.mc_run_events import run_one_iteration
except ImportError as exc:
    sys.exit(
        f"\n[demo] ERROR: Could not import fl_risk_model: {exc}\n"
        "Run:  pip install -e .  from the repository root.\n"
    )

# ---------------------------------------------------------------------------
# Scenario constants
# ---------------------------------------------------------------------------
DEMO_SCENARIO   = "great_miami"
DEMO_EVENT_STEMS = ["1926255N15314"]   # IBTrACS ID for the 1926 Great Miami Hurricane


# ---------------------------------------------------------------------------
# Helpers: load and validate demo data
# ---------------------------------------------------------------------------

def _load_demo_market_share(path: Path, year: int = 2024) -> pd.DataFrame:
    """
    Load synthetic market share CSV.

    Expected columns: Company, StatEntityKey, MarketShare{year}
    Returns a DataFrame with 'Company', 'StatEntityKey', 'Share' (normalised to 1).
    """
    df = pd.read_csv(path)
    share_col = f"MarketShare{year}"
    if share_col not in df.columns:
        cands = [c for c in df.columns if c.lower().startswith("marketshare")]
        if cands:
            df = df.rename(columns={cands[0]: share_col})
        else:
            raise ValueError(
                f"demo_market_share.csv must have a 'MarketShare{year}' column. "
                f"Found: {list(df.columns)}"
            )
    df = df.rename(columns={share_col: "Share"})
    df["Share"] = pd.to_numeric(df["Share"], errors="coerce").fillna(0.0)
    total = df["Share"].sum()
    if total > 0:
        df["Share"] = df["Share"] / total   # normalise
    return df[["Company", "StatEntityKey", "Share"]]


def _load_demo_surplus(path: Path) -> pd.DataFrame:
    """
    Load synthetic surplus CSV into the schema expected by capital.py.

    Required columns: Company, StatEntityKey, SurplusUSD
    Optional: GroupSurplusUSD, GroupToEntityRatio
    """
    df = pd.read_csv(path)
    for col in ["Company", "StatEntityKey", "SurplusUSD"]:
        if col not in df.columns:
            raise ValueError(f"demo_surplus.csv is missing column '{col}'")
    df["SurplusUSD"] = pd.to_numeric(df["SurplusUSD"], errors="coerce").fillna(0.0)
    if "GroupSurplusUSD" not in df.columns:
        df["GroupSurplusUSD"] = df["SurplusUSD"] * 1.8
    else:
        df["GroupSurplusUSD"] = pd.to_numeric(df["GroupSurplusUSD"], errors="coerce").fillna(
            df["SurplusUSD"] * 1.8
        )
    if "GroupToEntityRatio" not in df.columns:
        df["GroupToEntityRatio"] = df["GroupSurplusUSD"] / df["SurplusUSD"].replace(0, np.nan)
    return df[["Company", "StatEntityKey", "SurplusUSD", "GroupSurplusUSD", "GroupToEntityRatio"]]


def _build_demo_common_inputs(
    demo_surplus_df: pd.DataFrame,
    demo_mshare_df: pd.DataFrame,
) -> tuple:
    """
    Build the common_inputs tuple expected by run_one_iteration(), using:
      - public institutional data (FHCF, Citizens, NFIP, county xwalk)
      - synthetic private-market financial data (market share, surplus)

    Returns
    -------
    (fhcf_county_df, mshare, county_xwalk, cit_cap, nfip_claims_df, nfip_exposure_df)
    """
    # 1. FHCF county exposure (public FHCF disclosure)
    fhcf_county_df = load_fhcf_county_exposure(
        cfg.EXPOSURE_FILE, sheet_name=0, header_row=4
    )

    # 2. Market share — demo CSV has 'Company', 'StatEntityKey', 'Share'
    #    The runner expects the column to be named 'Share' or MarketShare{year}.
    mshare = demo_mshare_df.copy()

    # 3. County FIPS crosswalk (public)
    county_xwalk = pd.read_csv(cfg.DATA_DIR / "fl_county_fips.csv")
    low = {c.lower(): c for c in county_xwalk.columns}
    if "county" in low:
        county_xwalk = county_xwalk.rename(columns={low["county"]: "County"})
    if "county_fips" in low:
        county_xwalk["county_fips"] = (
            county_xwalk[low["county_fips"]].astype(str)
            .str.replace(r"\D", "", regex=True).str.zfill(5)
        )
    elif "statefp" in low and "countyfp" in low:
        sf = county_xwalk[low["statefp"]].astype(str).str.replace(r"\D", "", regex=True).str.zfill(2)
        cf = county_xwalk[low["countyfp"]].astype(str).str.replace(r"\D", "", regex=True).str.zfill(3)
        county_xwalk["county_fips"] = sf + cf
    else:
        raise KeyError("fl_county_fips.csv needs county_fips or STATEFP+COUNTYFP columns.")
    county_xwalk = county_xwalk[["County", "county_fips"]].drop_duplicates()

    # 4. Citizens capital (public Citizens annual reports)
    cit_cap = load_citizens_capital_row_from_csv(
        path=str(cfg.CITIZENS_CAPITAL_CSV),
        year=int(cfg.CITIZENS_CAPITAL_YEAR),
    )

    # 5. NFIP claims — not required when NFIP_PAYOUT_MODE is 'unity' (default)
    nfip_claims_df = None

    # 6. NFIP exposure (FEMA public data)
    nfip_exposure_df = load_nfip_policy_coverage(
        path=str(cfg.NFIP_POLICIES_CSV),
        mode=getattr(cfg, "SAMPLING_MODE_NFIP_POLICIES", "FIXED_YEAR"),
        year=int(cfg.NFIP_POLICY_YEAR),
        lookback_years=int(getattr(cfg, "EWA_WINDOW_YEARS", 5)),
        half_life=float(getattr(cfg, "EWA_HALF_LIFE_YEARS", 2.0)),
        county_xwalk=county_xwalk,
    )

    return fhcf_county_df, mshare, county_xwalk, cit_cap, nfip_claims_df, nfip_exposure_df


# ---------------------------------------------------------------------------
# Monkey-patch: redirect proprietary surplus loader to demo CSV
# ---------------------------------------------------------------------------

def _patch_surplus_loader(demo_surplus_df: pd.DataFrame):
    """
    Return a context-free patch function that replaces load_surplus_data_with_groups
    with a closure returning the pre-built demo DataFrame.

    Usage:
        orig, patch = _patch_surplus_loader(df)
        <run model>
        restore(orig)
    """
    import fl_risk_model.runner  as _runner_mod
    import fl_risk_model.capital as _capital_mod

    orig_capital = _capital_mod.load_surplus_data_with_groups
    orig_runner  = _runner_mod.load_surplus_data_with_groups

    def _demo_load(**kw):
        return demo_surplus_df.copy()

    _capital_mod.load_surplus_data_with_groups = _demo_load
    _runner_mod.load_surplus_data_with_groups  = _demo_load

    def _restore():
        _capital_mod.load_surplus_data_with_groups = orig_capital
        _runner_mod.load_surplus_data_with_groups  = orig_runner

    return _restore


# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------

def run_demo(
    n_iter: int = 100,
    seed: int = 42,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run n_iter Monte Carlo iterations of the Great Miami Hurricane scenario
    through the full risk waterfall using synthetic private insurer data.

    The wind/water split is sampled each iteration from the event-specific
    Beta prior (mean=0.70 wind share, concentration=10) — the same prior
    used in the manuscript's historical scenario analysis.

    Parameters
    ----------
    n_iter : int
        Number of Monte Carlo iterations.
    seed : int
        Master RNG seed. Held fixed so results are reproducible.
    out_dir : Path, optional
        Unused here; passed through for convenience.

    Returns
    -------
    pd.DataFrame
        One row per iteration with all risk waterfall outputs.
    """
    print(f"\n{'='*60}")
    print("  Florida Systemic Insurance Risk Model — Demo Run")
    print(f"{'='*60}")
    print(f"  Scenario     : Great Miami Hurricane (1926)")
    print(f"  IBTrACS ID   : 1926255N15314")
    print(f"  Companies    : 99 FL private homeowners insurers")
    print(f"  Financials   : SYNTHETIC (illustrative only — see demo_data/)")
    print(f"  Iterations   : {n_iter}")
    print(f"  Seed         : {seed}")
    print()

    # ── Silence verbose diagnostic output during demo ──────────────────────
    cfg.DEBUG_PRINTS          = False
    cfg.VERBOSE_EXPOSURE      = False
    cfg.PRINT_MASSBALANCE_TOP5 = False

    # ── Load demo financial data ────────────────────────────────────────────
    print("Loading demo data...")
    demo_mshare_df = _load_demo_market_share(
        DEMO_DATA_DIR / "demo_market_share.csv",
        year=getattr(cfg, "MARKET_SHARE_YEAR", 2024),
    )
    demo_surplus_df = _load_demo_surplus(DEMO_DATA_DIR / "demo_surplus.csv")

    n_priv = len(demo_mshare_df)
    total_surplus = demo_surplus_df["SurplusUSD"].sum()
    print(f"  Loaded {n_priv} private insurers")
    print(f"  Total synthetic entity surplus: ${total_surplus/1e9:.1f}B")

    # ── Build model inputs (public data for institutional layers) ───────────
    print("Building model inputs (FHCF, Citizens, NFIP, county xwalk)...")
    common_inputs = _build_demo_common_inputs(demo_surplus_df, demo_mshare_df)

    # ── Patch surplus loader ────────────────────────────────────────────────
    restore_fn = _patch_surplus_loader(demo_surplus_df)

    try:
        rng = np.random.default_rng(seed)
        rows: list[dict] = []

        print(f"Running {n_iter} iterations...")
        t0 = time.time()

        for i in range(n_iter):
            iter_seed = int(rng.integers(0, 2**31))
            iter_rng  = np.random.default_rng(iter_seed)
            try:
                result = run_one_iteration(
                    scenario_name=DEMO_SCENARIO,
                    event_stems=DEMO_EVENT_STEMS,
                    rng=iter_rng,
                    common_inputs=common_inputs,
                    do_flood=True,
                    surplus_year=getattr(cfg, "FIXED_YEAR", 2024),
                    policy_scenario_config=None,
                )
                row = {
                    "iteration": i,
                    "scenario": DEMO_SCENARIO,
                    **{k: v for k, v in result.items()
                       if isinstance(v, (int, float, bool, np.integer, np.floating))},
                }
                rows.append(row)
            except Exception as exc:
                rows.append({"iteration": i, "scenario": "error", "error": str(exc)})

            if (i + 1) % max(1, n_iter // 4) == 0:
                elapsed = time.time() - t0
                print(f"  Completed {i+1:3d}/{n_iter} iterations  ({elapsed:.1f}s elapsed)")

        elapsed_total = time.time() - t0
        print(f"\n  All {n_iter} iterations completed in {elapsed_total:.1f}s")

    finally:
        restore_fn()   # always restore original loaders

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame, n_iter: int) -> None:
    """Print a concise summary of demo results to stdout."""
    valid = df[df.get("scenario", pd.Series(dtype=str)) != "error"].copy()
    n_valid = len(valid)
    n_errors = n_iter - n_valid

    print(f"\n{'='*60}")
    print("  Demo Results Summary")
    print(f"{'='*60}")
    print(f"  Iterations run   : {n_iter}")
    print(f"  Successful        : {n_valid}")
    if n_errors:
        print(f"  Errors            : {n_errors}")

    if n_valid == 0:
        print("  No valid results — check error column in demo_summary.csv")
        return

    def _mean(col):
        return valid[col].mean() if col in valid.columns else None

    def _fmt(col, scale=1e9, unit="B USD"):
        v = _mean(col)
        return f"${v/scale:.2f} {unit} (mean)" if v is not None else "N/A"

    def _pct(col, thresh=0):
        if col in valid.columns:
            return f"{(valid[col] > thresh).mean()*100:.1f}% of iterations"
        return "N/A"

    print()
    print("  Gross losses:")
    print(f"    Total damage         : {_fmt('total_damage_usd')}")
    print(f"    Wind                 : {_fmt('wind_total_usd')}")
    print(f"    Water/flood          : {_fmt('water_total_usd')}")
    print()
    print("  Insured losses:")
    print(f"    Private wind (insured): {_fmt('wind_insured_private_usd')}")
    print(f"    Citizens wind         : {_fmt('wind_insured_citizens_usd')}")
    print(f"    NFIP flood            : {_fmt('flood_insured_capped_usd')}")
    print()
    print("  Un/underinsured losses:")
    print(f"    Wind                 : {_fmt('wind_underinsured_usd')}")
    print(f"    Flood                : {_fmt('flood_underinsured_usd')}")
    print()
    print("  Systemic stress (% of iterations):")
    print(f"    Any private default  : {_pct('defaults_post', thresh=0)}")
    print(f"    Defaults > 10 firms  : {_pct('defaults_post', thresh=10)}")
    print(f"    FHCF shortfall > 0   : {_pct('fhcf_shortfall_usd', thresh=0)}")
    print(f"    Citizens deficit > 0 : {_pct('citizens_residual_deficit_usd', thresh=0)}")
    print(f"    NFIP borrowing > 0   : {_pct('nfip_borrowed_usd', thresh=0)}")
    print()
    print("  Public institutional burden (mean):")
    print(f"    FHCF shortfall       : {_fmt('fhcf_shortfall_usd')}")
    print(f"    NFIP borrowed        : {_fmt('nfip_borrowed_usd')}")
    print(f"    Citizens deficit     : {_fmt('citizens_residual_deficit_usd')}")
    print(f"    FIGA residual        : {_fmt('figa_residual_deficit_usd')}")
    print()
    print("  IMPORTANT: These outputs use SYNTHETIC insurer financial data.")
    print("  Company names are real (public regulatory record); market shares")
    print("  and surplus values are illustrative only.  See demo_data/README.md.")
    print("  Results do NOT match the manuscript figures or tables.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fl_risk_model demo — Great Miami Hurricane with synthetic "
            "insurer financial data.  No proprietary inputs required."
        )
    )
    parser.add_argument(
        "--n_iter", type=int, default=100,
        help="Number of Monte Carlo iterations (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--out", type=str, default="demo_output",
        help="Output directory, relative to repo root (default: demo_output)",
    )
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df_results = run_demo(n_iter=args.n_iter, seed=args.seed, out_dir=out_dir)

    # Save results
    summary_path = out_dir / "demo_summary.csv"
    df_results.to_csv(summary_path, index=False)
    print(f"  Results saved to: {summary_path.relative_to(REPO_ROOT)}")

    # Print human-readable summary
    _print_summary(df_results, n_iter=args.n_iter)

    # Save expected output snapshot (first run only, for reviewer comparison)
    expected_path = out_dir / "expected_demo_summary.csv"
    if not expected_path.exists():
        df_results.to_csv(expected_path, index=False)
        print(f"  Expected output snapshot saved: {expected_path.relative_to(REPO_ROOT)}")
    else:
        print(f"  Expected output already exists: {expected_path.relative_to(REPO_ROOT)}")
        print("         (delete it to regenerate from the current run)")


if __name__ == "__main__":
    main()

