"""
diff_cat_model — A minimal differentiable catastrophe risk pipeline in JAX.

Implements the Phase 1 demonstrator from the AAD for Cat Risk roadmap:
  parametric hazard → sigmoid vulnerability → softplus financial terms
  → vmap over events → grad of Average Annual Loss (AAL).

All operations are JAX-traceable, enabling:
  - jax.grad   : exact sensitivities via reverse-mode AD (adjoint)
  - jax.vmap   : vectorized evaluation over the event catalogue
  - jax.jit    : XLA compilation for GPU/TPU acceleration
"""
from diff_cat_model.pipeline import (
    CatModelParams,
    hazard_intensity,
    vulnerability,
    apply_financial_terms,
    event_loss,
    average_annual_loss,
    aal_from_params,
    gradient_aal,
)
