"""
common.py - Shared utilities for the Earth's Future corrections and sensitivity work.

This module is the single place where the corrected accounting and return-level
conventions used throughout ``scripts/earths_future_revision/`` are defined, so
that every downstream table and figure uses the same definitions.

Conventions established here (see docs/earths_future_revision/correction_register.md
for the evidence and reasoning behind each one):

1. Season-level aggregation before quantiles (Section 5 of the revision brief).
   A return level for an aggregate quantity (e.g. total public burden) is the
   empirical quantile of the *season-level sum*, not the sum of the separately
   estimated component quantiles. ``season_public_burden`` builds the season-level
   series; ``empirical_return_level`` estimates its quantiles.

2. Non-overlapping public burden aggregate (Section 4).
   ``fhcf_shortfall_usd`` is the *statewide* unrecovered FHCF reimbursement demand
   once the USD 17B seasonal cap binds. Tracing the model (fl_risk_model/runner.py,
   the capital-depletion and FIGA blocks) shows that whatever portion of an
   individual insurer's FHCF shortfall is not absorbed by that insurer's own
   remaining capital already flows into ``figa_residual_deficit_usd`` (private
   insurers) or ``citizens_residual_deficit_usd`` (Citizens), because capital
   depletion is applied to *net* wind loss (gross minus FHCF and cat-bond
   recovery), and FIGA/Citizens deficits are downstream of that net loss. The
   portion that *is* absorbed by remaining capital never appears as FIGA,
   Citizens, or NFIP stress at all -- it is a private capital loss, not a public
   or quasi-public backstop obligation. Adding the full ``fhcf_shortfall_usd`` to
   FIGA + Citizens + NFIP therefore double-counts the fraction of the shortfall
   that ended up inside a default. We report both the legacy (additive-by-
   construction) sum and a corrected, non-overlapping aggregate that excludes the
   FHCF shortfall from the sum and reports it separately as an upstream
   diagnostic of FHCF capacity stress.

3. All 10,000 simulated seasons are included in every quantile and mean,
   including zero-loss seasons. No filtering on nonzero rows unless explicitly
   documented (e.g. the severity-bin decomposition in Section 6, which documents
   its own zero-loss bin).

4. Quantile / return-period convention: for return period RP (years) and N
   simulated seasons, the return level is the empirical quantile at probability
   level q = 1 - 1/RP, computed with ``numpy.quantile`` using the default linear
   interpolation between order statistics. Ties are handled implicitly by the
   interpolation (no special tie-breaking rule is needed because linear
   interpolation between sorted, possibly repeated, values is well-defined).

5. Bootstrap uncertainty: entire season *rows* are resampled with replacement
   (preserving the joint dependence between total loss and every burden
   component within a season), following the manuscript's existing convention
   of 1,000 resamples and a reported 10th-90th percentile interval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "earths_future_revision"

RETURN_PERIODS = [10, 25, 50, 100, 250, 500, 1000]

# Display name for the corrected, non-overlapping 3-component aggregate
# (FIGA residual deficit + Citizens residual deficit + NFIP financing
# requirement). Per the author's editorial decision, this is the name used
# in tables/figures; the legacy 4-component (FHCF-inclusive) sum is kept
# only for explicit reconciliation, never as the primary reported quantity.
RESIDUAL_FINANCING_REQUIREMENT_LABEL = "Residual financing requirement"

# Original (submitted) definition -- retained for reconciliation only.
LEGACY_BURDEN_COMPONENTS = [
    "fhcf_shortfall_usd",
    "figa_residual_deficit_usd",
    "citizens_residual_deficit_usd",
    "nfip_borrowed_usd",
]

# Corrected, non-overlapping definition (Section 4 / correction register item C1).
CORRECTED_BURDEN_COMPONENTS = [
    "figa_residual_deficit_usd",
    "citizens_residual_deficit_usd",
    "nfip_borrowed_usd",
]

LOSS_COLUMNS = {
    "Total loss": "total_damage_usd",
    "Insured wind -- private": "wind_insured_private_usd",
    "Citizens wind": "wind_insured_citizens_usd",
    "Insured flood -- NFIP": "flood_insured_capped_usd",
    "Un/underinsured wind": None,  # constructed: underinsured + uninsured wind
    "Un/underinsured flood": "flood_un_derinsured_usd",
}

INST_COLUMNS = {
    "FHCF shortfall": "fhcf_shortfall_usd",
    "FIGA residual": "figa_residual_deficit_usd",
    "Citizens deficit": "citizens_residual_deficit_usd",
    "NFIP Treasury borrowing": "nfip_borrowed_usd",
}


def load_iterations(path: str | Path) -> pd.DataFrame:
    """Load an ``iterations.csv`` season table and add derived columns used
    throughout this package. One row is one simulated season (verified: the
    ``year_id`` column is unique per row and every row aggregates all events
    sampled for that season -- see docs/earths_future_revision/provenance.md).
    """
    df = pd.read_csv(path, low_memory=False)
    if "year_id" in df.columns:
        n_rows = len(df)
        n_unique = df["year_id"].nunique()
        if n_rows != n_unique:
            raise ValueError(
                f"{path}: {n_rows} rows but {n_unique} unique year_id -- "
                "rows do not represent one-season-per-row as assumed."
            )

    # Un/underinsured wind = underinsured_wind_usd + uninsured_wind_usd
    if "wind_underinsured_usd" in df.columns and "wind_uninsured_usd" in df.columns:
        df["wind_un_underinsured_usd"] = (
            df["wind_underinsured_usd"].fillna(0.0) + df["wind_uninsured_usd"].fillna(0.0)
        )
    elif "underinsured_wind_usd" in df.columns and "uninsured_wind_usd" in df.columns:
        df["wind_un_underinsured_usd"] = (
            df["underinsured_wind_usd"].fillna(0.0) + df["uninsured_wind_usd"].fillna(0.0)
        )

    for col in LEGACY_BURDEN_COMPONENTS:
        if col not in df.columns:
            raise ValueError(f"{path}: expected column '{col}' not found.")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Season-level aggregates (Section 5 + Section 4 fix)
    df["public_burden_legacy_usd"] = df[LEGACY_BURDEN_COMPONENTS].sum(axis=1)
    df["public_burden_corrected_usd"] = df[CORRECTED_BURDEN_COMPONENTS].sum(axis=1)

    return df


def empirical_return_level(x: np.ndarray, return_periods: Sequence[int] = RETURN_PERIODS) -> dict:
    """Empirical return levels of a season-level series.

    q = 1 - 1/RP; quantiles use numpy's default linear interpolation. All
    seasons (including zeros) must already be present in ``x``.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = {}
    for rp in return_periods:
        q = 1.0 - 1.0 / rp
        out[f"RP{rp}"] = float(np.quantile(x, q, method="linear"))
    out["N"] = n
    return out


def bootstrap_return_levels(
    df: pd.DataFrame,
    columns: Iterable[str],
    return_periods: Sequence[int] = RETURN_PERIODS,
    n_boot: int = 1000,
    seed: int = 42,
    lo_pct: float = 10.0,
    hi_pct: float = 90.0,
) -> dict:
    """Bootstrap the empirical return levels of one or more season-level
    columns by resampling whole season *rows* with replacement, preserving the
    joint dependence between columns within a season. Returns, for each
    column and return period, the point estimate (from the full sample) and
    the [lo_pct, hi_pct] percentile interval across bootstrap resamples.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    columns = list(columns)
    arrs = {c: df[c].to_numpy(dtype=float) for c in columns}

    point = {c: empirical_return_level(arrs[c], return_periods) for c in columns}

    boot_vals = {c: {f"RP{rp}": np.empty(n_boot) for rp in return_periods} for c in columns}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for c in columns:
            sample = arrs[c][idx]
            for rp in return_periods:
                q = 1.0 - 1.0 / rp
                boot_vals[c][f"RP{rp}"][b] = np.quantile(sample, q, method="linear")

    out = {}
    for c in columns:
        out[c] = {}
        for rp in return_periods:
            key = f"RP{rp}"
            lo, hi = np.percentile(boot_vals[c][key], [lo_pct, hi_pct])
            out[c][key] = {
                "point": point[c][key],
                "lo": float(lo),
                "hi": float(hi),
            }
    return out


def to_billion(x: float) -> float:
    return x / 1e9
