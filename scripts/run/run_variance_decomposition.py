#!/usr/bin/env python3
"""
run_variance_decomposition.py - Variance decomposition analysis

Tests the claim: "In probabilistic assessment, outcome variability is
primarily driven by differences in synthetic event realizations rather
than parameter perturbations."

Two complementary analyses
--------------------------
A) Reference run   - existing 10K-season run with full parameter sampling.
B) Fixed-param run - 10K seasons, identical year-sets, all stochastic
                     parameters pinned to their means:
                       * SAMPLE_EXPOSURE = False  (no TIV perturbation)
                       * Wind-share Beta concentration -> 10 000 (~ deterministic)
                     Compare  Var(B) / Var(A)  ~ 1  ⇒  hazard dominates.

C) Nested MC       - M seasons × K parameter draws per season.
                     Decomposes total variance into between-season (hazard)
                     and within-season (parameters) via one-way ANOVA:
                       Var_total = Var_between(Ȳ_m) + E_m[Var_within(Y_mk)]
                       η² = Var_between / Var_total

Usage
-----
    # Run B only (10K seasons, fixed parameters)
    python run_variance_decomposition.py --mode fixed

    # Run nested MC (M=300 seasons × K=50 draws)
    python run_variance_decomposition.py --mode nested --M 300 --K 50

    # Run nested on fewer years for quick test
    python run_variance_decomposition.py --mode nested --M 50 --K 10

    # Run both
    python run_variance_decomposition.py --mode both --M 300 --K 50

    # Analyze results from previous runs
    python run_variance_decomposition.py --mode analyze \
        --ref_run results/mc_runs/<reference_dir> \
        --fixed_run results/mc_runs/<fixed_dir> \
        --nested_run results/mc_runs/<nested_dir>
"""

import sys
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from fl_risk_model import config as cfg
from fl_risk_model.mc_run_events import (
    run_one_iteration,
    _prepare_common_inputs,
    _USE_STOCHASTIC_EVENTS,
    DO_FLOOD,
    RNG_SEED,
)

# Lazily import YEAR (module-level constant in mc_run_events)
YEAR = int(getattr(cfg, "MARKET_SHARE_YEAR", 2024))

# Key output metrics for variance decomposition
METRICS = [
    # Total loss
    "total_damage_usd",
    # Loss by branch / insured status
    "wind_insured_private_usd",        # private insurer wind
    "wind_insured_citizens_usd",       # Citizens wind
    "flood_insured_capped_usd",        # NFIP-insured flood
    "wind_uninsured_usd",              # uninsured wind (excl. underinsured)
    "wind_underinsured_usd",           # underinsured wind
    "flood_un_derinsured_usd",         # uninsured + underinsured flood
    # Institutional stress (public burden)
    "fhcf_shortfall_usd",
    "figa_residual_deficit_usd",
    "citizens_residual_deficit_usd",
    "nfip_borrowed_usd",
    "catbond_payout_usd",
    # Capital & default outcomes
    "defaults_post",
    "largest_entity_deficit_usd",
    "group_contrib_total_usd",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_year_sets(year_sets_csv: Path | None = None) -> pd.DataFrame:
    """Load year-sets CSV (pre-drawn synthetic seasons)."""
    if year_sets_csv is None:
        year_sets_csv = Path(getattr(cfg, "SYNTHETIC_YEAR_SETS_CSV"))
    return pd.read_csv(year_sets_csv)


def _set_fixed_params():
    """Pin stochastic parameters to their means (for Run B)."""
    saved = {}

    # 1. TIV sampling off
    saved["SAMPLE_EXPOSURE"] = getattr(cfg, "SAMPLE_EXPOSURE", True)
    cfg.SAMPLE_EXPOSURE = False

    # 2. Wind-share: crank Beta concentration -> near-deterministic
    saved["DEFAULT_WIND_SHARE_CONCENTRATION"] = cfg.DEFAULT_WIND_SHARE_CONCENTRATION

    saved["EVENT_WIND_SHARE_PARAMS"] = {
        k: dict(v) for k, v in cfg.EVENT_WIND_SHARE_PARAMS.items()
    }
    saved["COMPOSITE_WIND_SHARE_PARAMS"] = {
        k: dict(v) for k, v in cfg.COMPOSITE_WIND_SHARE_PARAMS.items()
    }

    FIXED_CONC = 10_000  # effectively deterministic
    cfg.DEFAULT_WIND_SHARE_CONCENTRATION = FIXED_CONC
    for k in cfg.EVENT_WIND_SHARE_PARAMS:
        cfg.EVENT_WIND_SHARE_PARAMS[k]["concentration"] = FIXED_CONC
    for k in cfg.COMPOSITE_WIND_SHARE_PARAMS:
        cfg.COMPOSITE_WIND_SHARE_PARAMS[k]["concentration"] = FIXED_CONC

    return saved


def _restore_params(saved: dict):
    """Restore original config after Run B."""
    cfg.SAMPLE_EXPOSURE = saved["SAMPLE_EXPOSURE"]
    cfg.DEFAULT_WIND_SHARE_CONCENTRATION = saved["DEFAULT_WIND_SHARE_CONCENTRATION"]
    for k, v in saved["EVENT_WIND_SHARE_PARAMS"].items():
        cfg.EVENT_WIND_SHARE_PARAMS[k] = v
    for k, v in saved["COMPOSITE_WIND_SHARE_PARAMS"].items():
        cfg.COMPOSITE_WIND_SHARE_PARAMS[k] = v


def _run_year(year_id, event_stems, rng, common_inputs,
              group_threshold=10.0):
    """Run one year through the model, return result dict or None on error."""
    try:
        rec = run_one_iteration(
            scenario_name=f"year_{year_id}",
            event_stems=event_stems,
            rng=rng,
            common_inputs=common_inputs,
            do_flood=DO_FLOOD,
            surplus_year=YEAR,
            policy_scenario_config=None,
            group_support_eligibility_threshold=group_threshold,
        )
        rec["year_id"] = year_id
        return rec
    except Exception as e:
        print(f"  [ERROR] year {year_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Run B: fixed-parameter 10K-season sweep
# ---------------------------------------------------------------------------
def run_fixed_params(year_sets: pd.DataFrame,
                     seed: int = RNG_SEED,
                     out_dir: Path = Path("results/mc_runs"),
                     n_years: int | None = None) -> Path:
    """10K seasons with all stochastic parameters pinned to means."""
    import fl_risk_model.mc_run_events as _mce
    _mce._USE_STOCHASTIC_EVENTS = True

    saved = _set_fixed_params()
    try:
        rng = np.random.default_rng(seed)
        common_inputs = _prepare_common_inputs()

        total_years = year_sets["year_id"].nunique()
        if n_years is not None:
            total_years = min(total_years, n_years)

        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir / f"variance_fixed_params_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(run_dir / "run_config.json", "w") as f:
            json.dump({
                "mode": "fixed_params",
                "n_years": total_years,
                "seed": seed,
                "SAMPLE_EXPOSURE": False,
                "wind_share_concentration": 10_000,
            }, f, indent=2)

        rows = []
        for year_id in range(1, total_years + 1):
            ydf = year_sets[year_sets["year_id"] == year_id]
            if ydf["event_id"].isna().all():
                rows.append({
                    "year_id": year_id,
                    "scenario": "zero_events",
                    **{m: 0.0 for m in METRICS},
                })
                continue

            event_stems = ydf["event_id"].dropna().tolist()
            rec = _run_year(year_id, event_stems, rng, common_inputs)
            if rec is not None:
                rows.append(rec)

            if year_id % 500 == 0:
                print(f"  [fixed] {year_id:,}/{total_years:,}")

        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "iterations.csv", index=False)
        print(f"  Fixed-param run -> {run_dir}  ({len(df)} years)")
        return run_dir
    finally:
        _restore_params(saved)
        _mce._USE_STOCHASTIC_EVENTS = False


# ---------------------------------------------------------------------------
# Run C: nested MC (M seasons × K parameter draws)
# ---------------------------------------------------------------------------
def run_nested_mc(year_sets: pd.DataFrame,
                  M: int = 300,
                  K: int = 50,
                  seed: int = RNG_SEED,
                  out_dir: Path = Path("results/mc_runs")) -> Path:
    """
    Nested Monte Carlo for variance decomposition.

    Selects M seasons stratified across the loss distribution, then
    replays each season K times with fresh parameter draws.
    """
    import fl_risk_model.mc_run_events as _mce
    _mce._USE_STOCHASTIC_EVENTS = True

    try:
        common_inputs = _prepare_common_inputs()

        # ---- Step 1: quick pre-scan to estimate per-season total damage ----
        # Use event metadata for fast damage estimates (no full model run)
        metadata_csv = Path(getattr(cfg, "SYNTHETIC_EVENT_METADATA_CSV"))
        event_meta = pd.read_csv(metadata_csv)
        damage_map = dict(zip(event_meta["event_id"], event_meta["total_damage_usd"]))

        total_years = year_sets["year_id"].nunique()
        year_damage = {}
        year_events = {}
        for yid in range(1, total_years + 1):
            ydf = year_sets[year_sets["year_id"] == yid]
            if ydf["event_id"].isna().all():
                year_damage[yid] = 0.0
                year_events[yid] = []
            else:
                stems = ydf["event_id"].dropna().tolist()
                year_events[yid] = stems
                year_damage[yid] = sum(damage_map.get(s, 0.0) for s in stems)

        # ---- Step 2: stratified sample of M seasons ----
        # Sort by damage, pick evenly spaced quantiles to span the distribution
        dmg_series = pd.Series(year_damage)
        # Exclude zero-event years from selection (they have no parameter sensitivity)
        nonzero = dmg_series[dmg_series > 0].sort_values()
        if len(nonzero) < M:
            selected_ids = nonzero.index.tolist()
            print(f"  [nested] Only {len(nonzero)} non-zero years; using all.")
        else:
            quantile_indices = np.linspace(0, len(nonzero) - 1, M, dtype=int)
            selected_ids = nonzero.iloc[quantile_indices].index.tolist()

        M_actual = len(selected_ids)
        print(f"  [nested] Selected {M_actual} seasons × {K} draws = {M_actual * K:,} runs")

        # ---- Step 3: run nested MC ----
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = out_dir / f"variance_nested_{M_actual}x{K}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "run_config.json", "w") as f:
            json.dump({
                "mode": "nested_mc",
                "M": M_actual,
                "K": K,
                "seed": seed,
                "selected_year_ids": [int(x) for x in selected_ids],
            }, f, indent=2)

        rows = []
        master_rng = np.random.default_rng(seed)

        for i, yid in enumerate(selected_ids):
            stems = year_events[yid]
            for k in range(K):
                # Fresh rng per draw - each (season, draw) gets unique randomness
                draw_seed = int(master_rng.integers(0, 2**31))
                draw_rng = np.random.default_rng(draw_seed)

                rec = _run_year(yid, stems, draw_rng, common_inputs)
                if rec is not None:
                    rec["draw_k"] = k
                    rec["season_idx"] = i
                    rows.append(rec)

            if (i + 1) % 20 == 0:
                print(f"  [nested] {i + 1}/{M_actual} seasons done "
                      f"({(i + 1) * K:,} runs)")

        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "iterations.csv", index=False)
        print(f"  Nested MC -> {run_dir}  ({len(df)} rows)")
        return run_dir
    finally:
        _mce._USE_STOCHASTIC_EVENTS = False


# ---------------------------------------------------------------------------
# Analysis: compute variance decomposition from saved results
# ---------------------------------------------------------------------------
def analyze(ref_dir: Path | None,
            fixed_dir: Path | None,
            nested_dir: Path | None):
    """
    Compute and print variance decomposition results.

    Parameters
    ----------
    ref_dir : Path or None
        Reference run (full sampling, 10K seasons).
    fixed_dir : Path or None
        Fixed-parameter run (10K seasons).
    nested_dir : Path or None
        Nested MC run (M × K).
    """
    print("\n" + "=" * 80)
    print("VARIANCE DECOMPOSITION - RESULTS")
    print("=" * 80)

    # ---- Comparison A vs B: Var(fixed) / Var(reference) ----
    if ref_dir and fixed_dir:
        df_ref = pd.read_csv(ref_dir / "iterations.csv")
        df_fix = pd.read_csv(fixed_dir / "iterations.csv")

        print("\n── Run A (reference) vs Run B (fixed params) ──")
        print(f"  Reference: {ref_dir}")
        print(f"  Fixed:     {fixed_dir}")
        print(f"  {'Metric':<40s} {'Var(A)':>14s} {'Var(B)':>14s} {'B/A':>8s}")
        print("  " + "-" * 78)

        summary_rows = []
        for m in METRICS:
            if m not in df_ref.columns or m not in df_fix.columns:
                continue
            va = df_ref[m].astype(float).var()
            vb = df_fix[m].astype(float).var()
            ratio = vb / va if va > 0 else float("nan")
            print(f"  {m:<40s} {va:>14.4e} {vb:>14.4e} {ratio:>8.4f}")
            summary_rows.append({
                "metric": m,
                "var_reference": va,
                "var_fixed": vb,
                "ratio_fixed_over_ref": ratio,
                "pct_hazard": ratio * 100 if np.isfinite(ratio) else np.nan,
                "pct_params": (1 - ratio) * 100 if np.isfinite(ratio) else np.nan,
            })

        summary_df = pd.DataFrame(summary_rows)
        out_csv = (fixed_dir / "variance_comparison_AB.csv")
        summary_df.to_csv(out_csv, index=False)
        print(f"\n  Saved -> {out_csv}")
        print("  Interpretation: B/A ~ 1 means hazard dominates; "
              "(1 − B/A) is the parameter contribution.")

    # ---- Nested MC: one-way ANOVA decomposition ----
    if nested_dir:
        df_nest = pd.read_csv(nested_dir / "iterations.csv")

        print("\n── Nested MC: one-way ANOVA decomposition ──")
        print(f"  Source: {nested_dir}")
        n_seasons = df_nest["year_id"].nunique()
        n_draws = df_nest.groupby("year_id").size().median()
        print(f"  Seasons: {n_seasons}, draws/season: {n_draws:.0f}")
        print(f"\n  {'Metric':<40s} {'Var_between':>14s} {'E[Var_within]':>14s} {'η²':>8s}")
        print("  " + "-" * 78)

        anova_rows = []
        for m in METRICS:
            if m not in df_nest.columns:
                continue
            vals = df_nest[["year_id", m]].dropna()
            vals[m] = vals[m].astype(float)

            # Group means
            group_means = vals.groupby("year_id")[m].mean()
            grand_mean = group_means.mean()

            # Between-season variance = Var(group means)
            var_between = group_means.var(ddof=1)

            # Within-season variance = mean of per-season variances
            var_within_per_season = vals.groupby("year_id")[m].var(ddof=1)
            e_var_within = var_within_per_season.mean()

            var_total = var_between + e_var_within
            eta_sq = var_between / var_total if var_total > 0 else float("nan")

            print(f"  {m:<40s} {var_between:>14.4e} {e_var_within:>14.4e} {eta_sq:>8.4f}")
            anova_rows.append({
                "metric": m,
                "var_between_season": var_between,
                "E_var_within_season": e_var_within,
                "var_total": var_total,
                "eta_squared": eta_sq,
                "pct_hazard": eta_sq * 100 if np.isfinite(eta_sq) else np.nan,
                "pct_params": (1 - eta_sq) * 100 if np.isfinite(eta_sq) else np.nan,
            })

        anova_df = pd.DataFrame(anova_rows)
        out_csv = nested_dir / "variance_decomposition_nested.csv"
        anova_df.to_csv(out_csv, index=False)
        print(f"\n  Saved -> {out_csv}")
        print("  Interpretation: η² ~ 1 means between-season (hazard) dominates.")

        # Within-season spread by damage quantile
        print("\n  Within-season CV by damage quantile:")
        group_stats = df_nest.groupby("year_id").agg(
            damage_mean=("total_damage_usd", "mean"),
            damage_std=("total_damage_usd", "std"),
        )
        group_stats["cv"] = group_stats["damage_std"] / group_stats["damage_mean"]
        group_stats = group_stats.replace([np.inf, -np.inf], np.nan).dropna()

        if not group_stats.empty:
            for q_label, q in [("P25", 0.25), ("P50", 0.50), ("P75", 0.75),
                               ("P90", 0.90), ("P99", 0.99)]:
                threshold = group_stats["damage_mean"].quantile(q)
                subset = group_stats[group_stats["damage_mean"] <= threshold]
                if not subset.empty:
                    median_cv = subset["cv"].median()
                    print(f"    {q_label} (damage ≤ ${threshold:,.0f}): "
                          f"median CV = {median_cv:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Variance decomposition: hazard vs. parameter uncertainty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode", required=True,
        choices=["fixed", "nested", "both", "analyze"],
        help="Which analysis to run",
    )
    ap.add_argument("--M", type=int, default=300,
                    help="Number of seasons for nested MC (default: 300)")
    ap.add_argument("--K", type=int, default=50,
                    help="Parameter draws per season (default: 50)")
    ap.add_argument("--n_years", type=int, default=None,
                    help="Limit fixed-param run to first N years (for testing)")
    ap.add_argument("--seed", type=int, default=RNG_SEED,
                    help="Random seed")
    ap.add_argument("--out_dir", type=str, default="results/mc_runs",
                    help="Output directory root")
    ap.add_argument("--year_sets_csv", type=str, default=None,
                    help="Path to year-sets CSV (default: from config)")

    # For --mode analyze
    ap.add_argument("--ref_run", type=str, default=None,
                    help="Path to reference run directory (for analyze mode)")
    ap.add_argument("--fixed_run", type=str, default=None,
                    help="Path to fixed-param run directory (for analyze mode)")
    ap.add_argument("--nested_run", type=str, default=None,
                    help="Path to nested MC run directory (for analyze mode)")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    if args.mode == "analyze":
        analyze(
            ref_dir=Path(args.ref_run) if args.ref_run else None,
            fixed_dir=Path(args.fixed_run) if args.fixed_run else None,
            nested_dir=Path(args.nested_run) if args.nested_run else None,
        )
        return

    # Load year-sets
    year_sets_csv = Path(args.year_sets_csv) if args.year_sets_csv else None
    year_sets = _load_year_sets(year_sets_csv)
    print(f"Loaded year-sets: {year_sets['year_id'].nunique():,} years")

    fixed_dir = None
    nested_dir = None

    if args.mode in ("fixed", "both"):
        print("\n" + "=" * 60)
        print("RUN B: Fixed parameters (10K seasons)")
        print("=" * 60)
        fixed_dir = run_fixed_params(
            year_sets, seed=args.seed, out_dir=out_dir,
            n_years=args.n_years,
        )

    if args.mode in ("nested", "both"):
        print("\n" + "=" * 60)
        print(f"RUN C: Nested MC ({args.M} seasons × {args.K} draws)")
        print("=" * 60)
        nested_dir = run_nested_mc(
            year_sets, M=args.M, K=args.K,
            seed=args.seed, out_dir=out_dir,
        )

    # Auto-analyze if we produced results
    if fixed_dir or nested_dir:
        analyze(
            ref_dir=None,  # user can supply reference run path later
            fixed_dir=fixed_dir,
            nested_dir=nested_dir,
        )

    print("\nDone. To compare against a reference run, use:")
    print("  python run_variance_decomposition.py --mode analyze \\")
    print("    --ref_run <path_to_reference> \\")
    if fixed_dir:
        print(f"    --fixed_run {fixed_dir} \\")
    if nested_dir:
        print(f"    --nested_run {nested_dir}")


if __name__ == "__main__":
    main()
