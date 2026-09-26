from __future__ import annotations
from typing import Sequence
import numpy as np

RETURN_PERIODS = [10, 25, 50, 100, 250, 500, 1000]

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

def to_billion(x: float) -> float:
    return x / 1e9
