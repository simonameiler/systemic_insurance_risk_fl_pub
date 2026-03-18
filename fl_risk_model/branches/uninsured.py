# fl_risk_model/branches/uninsured.py
from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from fl_risk_model import config as cfg

# -----------------------------
# Utilities
# -----------------------------
def _require_cols(df: pd.DataFrame, need: Tuple[str, ...], where: str) -> None:
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required column(s): {missing}")

def _make_rng(rng: Optional[np.random.Generator] = None,
              seed: Optional[int] = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng()

def _validate_or_sample_fractions(
    n: int,
    rates: Optional[Dict[str, float]],
    rng: np.random.Generator,
    insured_alpha: float,
    insured_beta: float,
    under_hh_alpha: float,
    under_hh_beta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce arrays (insured_frac, underinsured_frac, uninsured_frac) of length n
    that sum to ~1 row-wise.
    """
    if rates is not None:
        # Use provided fractions; normalize robustly.
        i = float(rates.get("insured", 0.0))
        u = float(rates.get("underinsured", 0.0))
        uu = float(rates.get("uninsured", 0.0))
        vec = np.array([i, u, uu], dtype=float)
        s = vec.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Provided insurance rates must be positive and sum > 0.")
        vec = vec / s
        return (np.full(n, vec[0], dtype=float),
                np.full(n, vec[1], dtype=float),
                np.full(n, vec[2], dtype=float))

    # Sample: insured ~ Beta; split household portion (1 - insured) into under/un.
    insured_frac = rng.beta(insured_alpha, insured_beta, size=n)
    hh = 1.0 - insured_frac
    under_share_of_hh = rng.beta(under_hh_alpha, under_hh_beta, size=n)
    underinsured_frac = hh * under_share_of_hh
    uninsured_frac = hh - underinsured_frac

    # Hygiene
    insured_frac = np.clip(insured_frac, 0.0, 1.0)
    underinsured_frac = np.clip(underinsured_frac, 0.0, 1.0)
    uninsured_frac = np.clip(uninsured_frac, 0.0, 1.0)
    s = insured_frac + underinsured_frac + uninsured_frac
    s[s == 0] = 1.0
    return insured_frac / s, underinsured_frac / s, uninsured_frac / s

# -----------------------------
# Wind carve-out (gross stage)
# -----------------------------
def apply_gross_carveout_wind(
    gross_df: pd.DataFrame,
    *,
    county_col: str = "County",
    loss_col: str = "GrossWindLossUSD",
    rates: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    insured_alpha: float = cfg.INSURED_ALPHA,
    insured_beta:  float = cfg.INSURED_BETA,
    under_hh_alpha: float = cfg.UNDER_HH_ALPHA,
    under_hh_beta:  float = cfg.UNDER_HH_BETA,
    return_rates: bool = True,
) -> pd.DataFrame:
    """
    Split industry-wide gross wind losses per county into insured / underinsured / uninsured.

    Parameters
    ----------
    gross_df : DataFrame
        Must contain [county_col, loss_col]; loss_col is TOTAL gross wind loss (USD) by county.
    rates : dict | None
        If provided, use {'insured','underinsured','uninsured'} (normalized if not exactly 1).
        If None, sample using Beta priors.
    rng / seed : Randomness control for reproducible sampling.
    *_alpha/_beta : Beta priors (defaults taken from config).
    return_rates : If True, include the sampled/used fractions in the output.

    Returns
    -------
    DataFrame (copy of input) plus:
      - 'InsuredWindUSD', 'UnderinsuredWindUSD', 'UninsuredWindUSD'
      - optionally 'insured_frac','underinsured_frac','uninsured_frac'
    """
    _require_cols(gross_df, (county_col, loss_col), "apply_gross_carveout_wind")
    out = gross_df.copy()
    out[loss_col] = pd.to_numeric(out[loss_col], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)

    R = _make_rng(rng, seed)
    n = len(out)
    insured_frac, underinsured_frac, uninsured_frac = _validate_or_sample_fractions(
        n, rates, R, insured_alpha, insured_beta, under_hh_alpha, under_hh_beta
    )

    base = out[loss_col].to_numpy(dtype=float)
    out["InsuredWindUSD"]      = base * insured_frac
    out["UnderinsuredWindUSD"] = base * underinsured_frac
    out["UninsuredWindUSD"]    = base * uninsured_frac

    if return_rates:
        out["insured_frac"] = insured_frac
        out["underinsured_frac"] = underinsured_frac
        out["uninsured_frac"] = uninsured_frac

    # (Mass balance check can be asserted in tests if desired.)
    return out

# -----------------------------
# Flood carve-out (gross stage)
# -----------------------------
def apply_gross_carveout_flood(
    gross_df: pd.DataFrame,
    *,
    county_col: str = "County",
    loss_col_primary: str = "FloodLossUSD_capped",
    loss_col_fallback: str = "FloodLossUSD",
    rates: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    insured_alpha: float = cfg.INSURED_ALPHA,
    insured_beta:  float = cfg.INSURED_BETA,
    under_hh_alpha: float = cfg.UNDER_HH_ALPHA,
    under_hh_beta:  float = cfg.UNDER_HH_BETA,
    return_rates: bool = True,
) -> pd.DataFrame:
    """
    Split industry-wide gross flood losses per county into insured / underinsured / uninsured.

    Notes
    -----
    - Prefers 'FloodLossUSD_capped' (if present) to represent a capped/limited gross,
      otherwise falls back to 'FloodLossUSD'.
    - Beta priors default to config; pass different values if flood needs distinct priors.

    Returns
    -------
    DataFrame (copy of input) plus:
      - 'InsuredFloodUSD', 'UnderinsuredFloodUSD', 'UninsuredFloodUSD'
      - optionally 'insured_frac_flood','underinsured_frac_flood','uninsured_frac_flood'
    """
    loss_col = loss_col_primary if loss_col_primary in gross_df.columns else loss_col_fallback
    _require_cols(gross_df, (county_col, loss_col), "apply_gross_carveout_flood")

    out = gross_df.copy()
    out[loss_col] = pd.to_numeric(out[loss_col], errors="coerce").fillna(0.0).astype(float).clip(lower=0.0)

    R = _make_rng(rng, seed)
    n = len(out)
    insured_frac, underinsured_frac, uninsured_frac = _validate_or_sample_fractions(
        n, rates, R, insured_alpha, insured_beta, under_hh_alpha, under_hh_beta
    )

    base = out[loss_col].to_numpy(dtype=float)
    out["InsuredFloodUSD"]      = base * insured_frac
    out["UnderinsuredFloodUSD"] = base * underinsured_frac
    out["UninsuredFloodUSD"]    = base * uninsured_frac

    if return_rates:
        out["insured_frac_flood"] = insured_frac
        out["underinsured_frac_flood"] = underinsured_frac
        out["uninsured_frac_flood"] = uninsured_frac

    return out
