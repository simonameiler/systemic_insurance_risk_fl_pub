"""
diff_cat_model.validate — Gradient validation against finite differences.

Provides utilities to sanity-check that JAX AD gradients match numerical
central-difference approximations, catching any smoothing or implementation
bugs before running real analyses.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from diff_cat_model.pipeline import (
    CatModelParams,
    aal_from_params,
    gradient_aal,
)


def finite_diff_gradient(
    params: CatModelParams,
    event_distances: jax.Array,
    exposure_values: jax.Array,
    event_frequencies: jax.Array,
    eps: float = 1.0,
) -> dict[str, float]:
    """Central-difference approximation of ∂AAL/∂(each param).

    Parameters
    ----------
    params : CatModelParams
    event_distances, exposure_values, event_frequencies : arrays
    eps : float
        Perturbation size (absolute).  Tuned per-parameter internally.

    Returns
    -------
    grads : dict  {param_name: ∂AAL/∂param}
    """
    base_dict = params._asdict()
    fd_grads = {}

    # per-parameter relative step sizes (avoid 0-crossing issues)
    step_scales = {
        "v_max": 0.1,
        "r_max": 0.1,
        "hol_b": 0.01,
        "v_thresh": 0.1,
        "v_half": 0.1,
        "scale": 0.001,
        "deductible": 100.0,
        "cover": 1000.0,
        "beta": 0.1,
    }

    for name in params._fields:
        h = step_scales.get(name, eps)

        plus_dict = dict(base_dict)
        plus_dict[name] = base_dict[name] + h
        f_plus = aal_from_params(
            CatModelParams(**plus_dict), event_distances, exposure_values, event_frequencies
        )

        minus_dict = dict(base_dict)
        minus_dict[name] = base_dict[name] - h
        f_minus = aal_from_params(
            CatModelParams(**minus_dict), event_distances, exposure_values, event_frequencies
        )

        fd_grads[name] = float((f_plus - f_minus) / (2.0 * h))

    return fd_grads


def compare_gradients(
    params: CatModelParams,
    event_distances: jax.Array,
    exposure_values: jax.Array,
    event_frequencies: jax.Array,
) -> dict[str, dict]:
    """Compare AD gradients with finite differences.

    Returns
    -------
    comparison : dict of {param: {"ad": float, "fd": float, "rel_err": float}}
    """
    ad_grads = gradient_aal(params, event_distances, exposure_values, event_frequencies)
    fd_grads = finite_diff_gradient(params, event_distances, exposure_values, event_frequencies)

    comparison = {}
    for i, name in enumerate(params._fields):
        ad_val = float(ad_grads[i])
        fd_val = fd_grads[name]
        denom = max(abs(ad_val), abs(fd_val), 1e-12)
        rel_err = abs(ad_val - fd_val) / denom
        comparison[name] = {"ad": ad_val, "fd": fd_val, "rel_err": rel_err}

    return comparison
