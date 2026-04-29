"""
diff_cat_model.pipeline — End-to-end differentiable catastrophe risk pipeline.

Pipeline stages (mirroring CLIMADA but fully JAX-traceable):

  1. **Hazard**:  Parametric wind intensity at each exposure point.
                  Simplified Holland-like radial decay profile.
  2. **Vulnerability**: Sigmoid damage function  intensity → MDR ∈ [0, 1].
                        (Emanuel-style rational function, smoothed.)
  3. **Financial terms**: Softplus deductible + smooth-min cap per policy.
  4. **Aggregation**: vmap over events, frequency-weighted AAL.

Everything is parameterised via a single flat dict / NamedTuple so that
``jax.grad`` can differentiate AAL w.r.t. *any* parameter.
"""

from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp

from diff_cat_model.primitives import softplus, smooth_min, sigmoid


# ===================================================================
# Parameter container
# ===================================================================

class CatModelParams(NamedTuple):
    """All differentiable parameters of the toy cat-risk model.

    Hazard (simplified Holland-type radial wind profile)
    ----------------------------------------------------
    v_max       : max sustained wind speed at eye wall [m/s]
    r_max       : radius of maximum wind [km]
    hol_b       : Holland B parameter (controls profile shape, ∈ [1, 2.5])

    Vulnerability (Emanuel-style sigmoid)
    --------------------------------------
    v_thresh    : wind speed threshold for onset of damage [m/s]
    v_half      : wind speed at 50 % mean damage ratio [m/s]
    scale       : maximum damage fraction (typically 1.0)

    Financial terms (per-policy)
    ----------------------------
    deductible  : per-policy deductible [USD]
    cover       : per-policy insurance limit / cap [USD]

    Smoothing
    ---------
    beta        : sharpness for softplus / smooth_min (β)
    """
    v_max: jax.Array
    r_max: jax.Array
    hol_b: jax.Array
    v_thresh: jax.Array
    v_half: jax.Array
    scale: jax.Array
    deductible: jax.Array
    cover: jax.Array
    beta: jax.Array


def default_params() -> CatModelParams:
    """Sensible defaults for a Florida-like hurricane scenario."""
    return CatModelParams(
        v_max=jnp.float32(60.0),       # Cat-3 hurricane
        r_max=jnp.float32(40.0),       # 40 km RMW
        hol_b=jnp.float32(1.5),        # typical B
        v_thresh=jnp.float32(25.7),    # Emanuel threshold
        v_half=jnp.float32(74.7),      # Emanuel half-damage speed
        scale=jnp.float32(1.0),
        deductible=jnp.float32(10_000.0),
        cover=jnp.float32(500_000.0),
        beta=jnp.float32(50.0),
    )


# ===================================================================
# 1. HAZARD — Simplified Holland radial wind profile
# ===================================================================

def hazard_intensity(
    distances: jax.Array,
    v_max: jax.Array,
    r_max: jax.Array,
    hol_b: jax.Array,
) -> jax.Array:
    """Compute wind speed at each distance from the storm eye.

    Simplified Holland (1980) gradient wind (cyclostrophic, no Coriolis):

        V(r) = V_max · [(R_max / r)^B · exp(1 − (R_max/r)^B)]^{1/2}

    This is smooth for r > 0 and composed of elementary differentiable ops
    (pow, exp, sqrt).

    Parameters
    ----------
    distances : (n_points,) array
        Distances from storm centre to each exposure centroid [km].
    v_max, r_max, hol_b : scalars
        Storm parameters.

    Returns
    -------
    wind_speed : (n_points,) array  [m/s]
    """
    # Regularise r = 0 to avoid singularity (standard practice)
    r = jnp.maximum(distances, 1e-3)

    r_ratio = (r_max / r) ** hol_b                      # (R_max/r)^B
    inner = r_ratio * jnp.exp(1.0 - r_ratio)            # (R/r)^B · exp(1-(R/r)^B)
    wind = v_max * jnp.sqrt(jnp.maximum(inner, 1e-30))   # V_max · sqrt(·)
    return wind


# ===================================================================
# 2. VULNERABILITY — Sigmoid damage function
# ===================================================================

def vulnerability(
    wind: jax.Array,
    v_thresh: jax.Array,
    v_half: jax.Array,
    scale: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    """Mean damage ratio (MDR) as a smooth Emanuel-style function.

    MDR(v) = scale · [u³ / (1 + u³)]

    where u = softplus(v − v_thresh) / (v_half − v_thresh),
    using softplus instead of max(0, ·) to keep the threshold smooth.

    This matches CLIMADA's ``ImpactFunc.from_emanuel_usa`` but replaces
    the hard ``np.maximum`` with a differentiable softplus.

    Parameters
    ----------
    wind : array  [m/s]
    v_thresh, v_half, scale, beta : scalars

    Returns
    -------
    mdr : array ∈ [0, scale]
    """
    # Smooth onset: replaces max(v - v_thresh, 0)
    excess = softplus(wind - v_thresh, beta=beta)
    u = excess / (v_half - v_thresh)

    # Emanuel rational function (smooth S-shape)
    # Using u*u*u instead of u**3 for stable AD gradients near u=0
    u3 = u * u * u
    mdr = scale * u3 / (1.0 + u3)
    return mdr


# ===================================================================
# 3. FINANCIAL TERMS — Softplus deductible + smooth cap
# ===================================================================

def apply_financial_terms(
    ground_up_loss: jax.Array,
    deductible: jax.Array,
    cover: jax.Array,
    beta: jax.Array,
) -> jax.Array:
    """Apply per-policy deductible and insurance cap (smoothed).

    insured_loss = min( max(loss − ded, 0),  cap )

    Composed from softplus (deductible) and smooth_min (cap).

    Parameters
    ----------
    ground_up_loss : array [USD]
    deductible : scalar [USD]
    cover : scalar [USD]
    beta : scalar  (sharpness)

    Returns
    -------
    insured_loss : array [USD]
    """
    excess = softplus(ground_up_loss - deductible, beta=beta)
    insured = smooth_min(excess, cover, alpha=beta)
    return insured


# ===================================================================
# 4. SINGLE-EVENT LOSS (compose all stages)
# ===================================================================

def event_loss(
    params: CatModelParams,
    distances: jax.Array,
    exposure_values: jax.Array,
) -> jax.Array:
    """Total insured loss for a single event across all exposure points.

    Parameters
    ----------
    params : CatModelParams
    distances : (n_points,) — distance of each point from storm centre
    exposure_values : (n_points,) — TIV per exposure point [USD]

    Returns
    -------
    total_insured_loss : scalar [USD]
    """
    # Stage 1: hazard
    wind = hazard_intensity(distances, params.v_max, params.r_max, params.hol_b)

    # Stage 2: vulnerability
    mdr = vulnerability(wind, params.v_thresh, params.v_half, params.scale, params.beta)

    # Stage 3: ground-up loss per point
    ground_up = mdr * exposure_values

    # Stage 4: financial terms per point
    insured = apply_financial_terms(ground_up, params.deductible, params.cover, params.beta)

    # Stage 5: aggregate across portfolio
    return jnp.sum(insured)


# ===================================================================
# 5. AVERAGE ANNUAL LOSS (vmap + frequency weighting)
# ===================================================================

def average_annual_loss(
    params: CatModelParams,
    event_distances: jax.Array,
    exposure_values: jax.Array,
    event_frequencies: jax.Array,
) -> jax.Array:
    """Frequency-weighted Average Annual Loss across the event catalogue.

    AAL = Σ_e  freq_e · L(event_e)

    Parameters
    ----------
    params : CatModelParams
    event_distances : (n_events, n_points)  [km]
    exposure_values : (n_points,)           [USD]
    event_frequencies : (n_events,)         [1/year]

    Returns
    -------
    aal : scalar [USD/year]
    """
    # vmap the single-event loss over the event axis
    batched_loss = jax.vmap(
        lambda dists: event_loss(params, dists, exposure_values)
    )
    losses = batched_loss(event_distances)                # (n_events,)
    aal = jnp.dot(event_frequencies, losses)              # scalar
    return aal


# ===================================================================
# 6. CONVENIENCE: flat-call for jax.grad
# ===================================================================

def aal_from_params(
    params: CatModelParams,
    event_distances: jax.Array,
    exposure_values: jax.Array,
    event_frequencies: jax.Array,
) -> jax.Array:
    """Thin wrapper so ``jax.grad(aal_from_params)`` works directly."""
    return average_annual_loss(params, event_distances, exposure_values, event_frequencies)


# JIT-compiled gradient function: ∂AAL/∂params in a single backward pass
gradient_aal = jax.jit(jax.grad(aal_from_params))
