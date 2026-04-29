"""
diff_cat_model.primitives — Differentiable smoothing primitives for insurance.

Three core building blocks (see roadmap Table on p.4):

  Softplus:  smooth max(0, x)          → ln(1 + exp(β·x)) / β
  Neg-LSE:   smooth min(a, b)          → −ln(exp(−α·a) + exp(−α·b)) / α
  Sigmoid:   smooth indicator 1{x > 0} → 1 / (1 + exp(−k·x))

All functions accept scalar or array inputs and are fully JAX-traceable.
Numerical stability is ensured via jnp.where / log-sum-exp tricks.
"""

from __future__ import annotations
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Softplus — smooth replacement for max(0, x)
# ---------------------------------------------------------------------------

def softplus(x, beta: float = 50.0):
    """
    Smooth approximation to max(0, x).

    softplus_β(x) = ln(1 + exp(β·x)) / β

    For large β·x the naive formula overflows; we use the identity
    softplus(x) = x + softplus(-x) when β·x > 20.

    Parameters
    ----------
    x : array_like
        Input values.
    beta : float
        Sharpness parameter.  Larger β → closer to exact max(0,x).
        Max bias = +ln(2)/β  (≈ $14 per operation for β = 50).
    """
    bx = beta * x
    # Clamp input to exp to avoid inf in the non-selected branch.
    # jnp.where evaluates both branches; without clamping, exp(large)
    # overflows and poisons gradients even when that branch isn't used.
    bx_safe = jnp.clip(bx, -500.0, 20.0)
    return jnp.where(bx > 20.0, x, jnp.log1p(jnp.exp(bx_safe)) / beta)


# ---------------------------------------------------------------------------
# Smooth-min via negative LogSumExp — smooth replacement for min(a, b)
# ---------------------------------------------------------------------------

def smooth_min(a, b, alpha: float = 50.0):
    """
    Smooth approximation to min(a, b) via negative LogSumExp.

    smooth_min_α(a, b) = −ln(exp(−α·a) + exp(−α·b)) / α

    Max bias = −ln(2)/α  (always underestimates true min).

    Parameters
    ----------
    a, b : array_like
    alpha : float
        Sharpness.  Larger α → closer to exact min.
    """
    neg_a = -alpha * a
    neg_b = -alpha * b
    max_val = jnp.maximum(neg_a, neg_b)
    # log-sum-exp trick: clamp differences to avoid overflow in exp
    da = jnp.clip(neg_a - max_val, -500.0, 0.0)
    db = jnp.clip(neg_b - max_val, -500.0, 0.0)
    lse = max_val + jnp.log(jnp.exp(da) + jnp.exp(db))
    return -lse / alpha


# ---------------------------------------------------------------------------
# Sigmoid — smooth indicator function 1{x > 0}
# ---------------------------------------------------------------------------

def sigmoid(x, k: float = 50.0):
    """
    Smooth approximation to the Heaviside step 1{x > 0}.

    σ_k(x) = 1 / (1 + exp(−k·x))

    Parameters
    ----------
    x : array_like
    k : float
        Sharpness.  Larger k → steeper transition.
    """
    kx = k * x
    # Clamp to avoid exp overflow in non-selected jnp.where branches
    kx_safe = jnp.clip(kx, -20.0, 20.0)
    return jnp.where(
        kx > 20.0,
        1.0,
        jnp.where(kx < -20.0, 0.0, 1.0 / (1.0 + jnp.exp(-kx_safe))),
    )


# ---------------------------------------------------------------------------
# Composed: excess-of-loss layer  max(0, min(loss − attachment, limit))
# ---------------------------------------------------------------------------

def smooth_xl_layer(loss, attachment, limit, beta: float = 50.0):
    """
    Smooth excess-of-loss reinsurance layer recovery.

    XL(loss) = max(0, min(loss − attachment, limit))

    Composed from softplus (for the deductible) and smooth_min (for the cap).
    Max bias ≤ 2·ln(2)/β.
    """
    excess = softplus(loss - attachment, beta=beta)
    return smooth_min(excess, limit, alpha=beta)
