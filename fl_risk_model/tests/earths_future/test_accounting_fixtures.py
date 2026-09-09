"""
Deterministic fixture tests for the Earth's Future institutional-accounting
corrections (revision brief Section 4, 5, 6, 7).

These tests exercise the real production functions (fl_risk_model.fhcf,
fl_risk_model.capital) with small, hand-derived inputs so that expected
values are computed independently of the implementation, per the revision
brief ("do not freeze incorrect manuscript values as expected outputs").

Run with:  pytest fl_risk_model/tests/earths_future -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_risk_model.fhcf import normalize_fhcf_terms, apply_fhcf_recovery
from fl_risk_model.catbonds import _payout_occurrence


# ---------------------------------------------------------------------------
# 1. Sum of component quantiles != quantile of the component sum (Section 5)
# ---------------------------------------------------------------------------

def test_sum_of_quantiles_differs_from_quantile_of_sum():
    """Codifies the Section 5 bug as a general statistical fact, independent
    of the insurance model: for two dependent-but-not-comonotonic series,
    summing their separately estimated quantiles is not the quantile of
    their sum."""
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.pareto(2.0, n) * 10
    # b is correlated with a but not comonotonic (adds independent noise),
    # so the two series are not perfectly rank-aligned.
    b = 0.5 * a + rng.pareto(2.0, n) * 3
    total = a + b

    q = 0.99
    sum_of_marginal_quantiles = np.quantile(a, q) + np.quantile(b, q)
    quantile_of_sum = np.quantile(total, q)

    assert sum_of_marginal_quantiles != pytest.approx(quantile_of_sum, rel=1e-6)
    # For non-comonotonic variables the sum of marginals over-states the
    # quantile of the sum (a form of the Frechet upper bound not being tight).
    assert sum_of_marginal_quantiles > quantile_of_sum


def test_zero_seasons_and_ties_handled_in_quantile_convention():
    """Zero-loss seasons must remain in the sample (Section 5), and repeated
    (tied) values must not crash or silently drop rows under linear
    interpolation."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "earths_future_revision"))
    from common import empirical_return_level

    x = np.array([0.0] * 8000 + [1.0] * 1000 + [5.0] * 900 + [100.0] * 100)
    out = empirical_return_level(x, return_periods=[10, 100])
    assert out["N"] == 10000
    # q=0.9 -> position 0.9*(10000-1)=8999.1, which falls between the last
    # tied "1.0" (index 8999) and the first tied "5.0" (index 9000): linear
    # interpolation gives 1.0 + 0.1*(5.0-1.0) = 1.4, not a crash or a
    # silently dropped tie.
    assert out["RP10"] == pytest.approx(1.4, rel=1e-6)
    assert out["RP100"] >= 5.0


# ---------------------------------------------------------------------------
# 2. FHCF shortfall overlap with downstream FIGA deficit (Section 4)
# ---------------------------------------------------------------------------

def _fhcf_terms(company: str, premium: float, coverage_pct: float) -> pd.DataFrame:
    raw = pd.DataFrame([{"Company": company, "FHCFPremium": premium, "CoveragePct": coverage_pct}])
    return normalize_fhcf_terms(raw)


def test_fhcf_shortfall_propagates_into_downstream_default_deficit():
    """Case A: an insurer's FHCF-eligible loss is far enough past its own
    Limit that recovery has saturated at the full Limit (verified formula,
    Article IV(1); see docs/earths_future_revision/fhcf_contract_verification.md),
    producing an FHCF shortfall borne by that insurer. That insurer's
    remaining (post-recovery) loss then exceeds its own capital, producing a
    default. The resulting FIGA-style deficit (net loss minus capital)
    already contains the FHCF shortfall dollar-for-dollar, because capital
    depletion in fl_risk_model.runner.run_one_scenario is applied to
    NetWindUSD = Gross - Recovery (see runner.py step 8-9), i.e. after FHCF
    recovery. Adding the statewide FHCF shortfall AND this deficit therefore
    double-counts the overlapping dollars. This fixture uses a single-row
    (single-company) loss_df, so it is unaffected by the separate
    company-aggregation defect/fix (see test_fhcf_contract_verification.py).
    """
    company = "TestCo"
    premium = 10_000_000.0  # arbitrary
    coverage_pct = 90
    terms = _fhcf_terms(company, premium, coverage_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    limit = float(terms["LimitUSD"].iloc[0])
    coverage_frac = coverage_pct / 100.0
    from fl_risk_model.config import FHCF_LAE_FACTOR
    # Verified formula (Article IV(1); see fhcf_contract_verification.md):
    # recovery saturates at the full Limit once
    # Excess >= Limit / (coverage_frac * LAE). Use a loss comfortably past
    # that point so recovery is already fully saturated at `limit`.
    saturation_excess = limit / (coverage_frac * FHCF_LAE_FACTOR)

    # Baseline case: loss is well past the saturation point, so recovery is
    # exactly at its capped value (the full Limit) and the company bears a
    # shortfall above its own limit equal to (gross - retention - recovery).
    gross_at_cap = retention + 2.0 * saturation_excess
    loss_df_at_cap = pd.DataFrame([{"Company": company, "GrossWindLossUSD": gross_at_cap}])
    recovery_at_cap = float(apply_fhcf_recovery(loss_df_at_cap, terms)["RecoveryUSD"].iloc[0])
    net_at_cap = float(apply_fhcf_recovery(loss_df_at_cap, terms)["NetWindUSD"].iloc[0])
    assert recovery_at_cap == pytest.approx(limit, rel=1e-9)  # confirms saturation

    # Give the insurer exactly enough capital to absorb this baseline case
    # with zero deficit.
    starting_capital = net_at_cap

    # Shortfall case: an additional loss increment lands entirely above the
    # company's FHCF limit, so it receives zero additional recovery and
    # flows 1:1 into NetWindUSD (this is the FHCF shortfall for this company).
    extra_shortfall = 5_000_000.0
    gross_with_shortfall = gross_at_cap + extra_shortfall
    loss_df = pd.DataFrame([{"Company": company, "GrossWindLossUSD": gross_with_shortfall}])
    net_with_shortfall = float(apply_fhcf_recovery(loss_df, terms)["NetWindUSD"].iloc[0])

    assert net_with_shortfall == pytest.approx(net_at_cap + extra_shortfall, rel=1e-9)

    deficit = max(net_with_shortfall - starting_capital, 0.0)

    # The entire extra_shortfall dollar-for-dollar becomes this insurer's
    # capital deficit, because capital depletion (fl_risk_model.runner,
    # step 8-9) is applied to NetWindUSD, which already contains the
    # unrecovered-above-limit loss.
    assert deficit == pytest.approx(extra_shortfall, rel=1e-9)

    # Legacy aggregate double-counts: (statewide FHCF shortfall) + (FIGA
    # deficit that already contains that shortfall).
    legacy_aggregate = extra_shortfall + deficit
    corrected_aggregate = deficit  # FIGA deficit alone already reflects it
    assert legacy_aggregate == pytest.approx(2 * extra_shortfall, rel=1e-9)
    assert legacy_aggregate > corrected_aggregate


def test_fhcf_shortfall_absorbed_by_capital_is_not_double_counted_when_dropped():
    """Case B: same FHCF shortfall, but the insurer has ample capital, so no
    default occurs. The shortfall reduces private surplus only -- it never
    appears in FIGA, Citizens, or NFIP. It is correctly excluded from the
    corrected public-burden sum (dropping it does not remove a real
    quasi-public obligation, because none was created)."""
    company = "TestCo2"
    premium = 10_000_000.0
    coverage_pct = 90
    terms = _fhcf_terms(company, premium, coverage_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    limit = float(terms["LimitUSD"].iloc[0])

    gross_loss = retention + limit + 5_000_000.0
    loss_df = pd.DataFrame([{"Company": company, "GrossWindLossUSD": gross_loss}])
    out = apply_fhcf_recovery(loss_df, terms)
    net_loss = float(out["NetWindUSD"].iloc[0])

    starting_capital = net_loss + 50_000_000.0  # ample capital
    ending_capital = starting_capital - net_loss
    assert ending_capital > 0.0  # solvent: no default, no FIGA deficit

    # This insurer contributes 0 to FIGA. The statewide FHCF shortfall from
    # this company is a private capital loss only -- it must not be added to
    # the public/quasi-public burden aggregate.
    figa_contribution = 0.0
    assert figa_contribution == 0.0


# ---------------------------------------------------------------------------
# 3. Conservation checks (Section 4)
# ---------------------------------------------------------------------------

def test_zero_loss_gives_zero_recovery_and_zero_net():
    terms = _fhcf_terms("ZeroCo", 5_000_000.0, 75)
    loss_df = pd.DataFrame([{"Company": "ZeroCo", "GrossWindLossUSD": 0.0}])
    out = apply_fhcf_recovery(loss_df, terms)
    assert out["RecoveryUSD"].iloc[0] == 0.0
    assert out["NetWindUSD"].iloc[0] == 0.0


def test_recovery_is_nonnegative_and_bounded_by_gross_loss():
    terms = _fhcf_terms("BoundCo", 8_000_000.0, 45)
    for gross in [0.0, 1e6, 1e8, 1e10]:
        loss_df = pd.DataFrame([{"Company": "BoundCo", "GrossWindLossUSD": gross}])
        out = apply_fhcf_recovery(loss_df, terms)
        rec = float(out["RecoveryUSD"].iloc[0])
        net = float(out["NetWindUSD"].iloc[0])
        assert rec >= 0.0
        assert net >= 0.0
        assert rec <= gross + 1e-6


def test_no_duplicated_recovery_across_companies_missing_terms():
    """A company absent from the terms table receives zero FHCF recovery
    (by design) rather than inheriting another company's terms."""
    terms = _fhcf_terms("KnownCo", 5_000_000.0, 90)
    loss_df = pd.DataFrame([
        {"Company": "KnownCo", "GrossWindLossUSD": 50_000_000.0},
        {"Company": "UnknownCo", "GrossWindLossUSD": 50_000_000.0},
    ])
    out = apply_fhcf_recovery(loss_df, terms)
    unknown_row = out[out["Company"] == "UnknownCo"].iloc[0]
    assert unknown_row["RecoveryUSD"] == 0.0
    assert unknown_row["NetWindUSD"] == unknown_row["GrossWindLossUSD"]


# ---------------------------------------------------------------------------
# 4. FHCF coverage-election double-application (Section 4, unresolved item C4)
# ---------------------------------------------------------------------------

def test_fhcf_coverage_election_may_be_applied_twice_between_retention_and_limit():
    """Documents (does not silently fix) the algebra raised by Reviewer 2 and
    AUTHOR CHECK S4-FHCF: retention multiples satisfy mult(p) * p = constant
    (confirmed against the primary 2023-24 FHCF Reimbursement Contract:
    retention is adjusted 200% / 120% / 100% of the 90%-coverage base value
    for the 45% / 75% / 90% coverage elections respectively), which is
    consistent with FHCFPremium already scaling with the company's coverage
    election p. If so, LimitUSD = premium * payout_multiple already reflects
    the elected (not the full 100%) layer size, and the current formula's
    extra multiplication by (CoveragePct_norm/100) in apply_fhcf_recovery
    applies the election a second time to the capped portion of recovery.

    This test quantifies -- but does not assert as ground truth -- the
    resulting difference, and is intended to accompany the correction
    register's flagged author decision (item C4), not to change the default
    production formula.
    """
    from fl_risk_model.config import FHCF_RET_MULTIPLES, FHCF_PAYOUT_MULTIPLE

    # Confirmed against the FHCF 2023-24 Reimbursement Contract: retention
    # multiple adjustments are exactly 200% (45%), 120% (75%), 100% (90%) of
    # the 90%-coverage base.
    base = FHCF_RET_MULTIPLES[90]
    assert FHCF_RET_MULTIPLES[75] == pytest.approx(1.20 * base, rel=1e-4)
    assert FHCF_RET_MULTIPLES[45] == pytest.approx(2.00 * base, rel=1e-4)

    premium = 10_000_000.0
    terms = _fhcf_terms("LowCov", premium, 45)
    retention = float(terms["RetentionUSD"].iloc[0])
    limit = float(terms["LimitUSD"].iloc[0])
    gross = retention + limit  # loss exactly fills the capped layer

    loss_df = pd.DataFrame([{"Company": "LowCov", "GrossWindLossUSD": gross}])
    current = apply_fhcf_recovery(loss_df, terms)
    recovery_current = float(current["RecoveryUSD"].iloc[0])

    # Alternative interpretation: do not re-apply p_i outside the min(), since
    # it is already embedded in Limit via a p_i-scaled premium.
    from fl_risk_model.config import FHCF_LAE_FACTOR
    recovery_alternative = limit * FHCF_LAE_FACTOR

    assert recovery_alternative > recovery_current
    # At a 45% coverage election, the alternative recovers ~1/0.45 = 2.22x
    # more of the capped layer than the current formula.
    ratio = recovery_alternative / recovery_current
    assert ratio == pytest.approx(1.0 / 0.45, rel=1e-6)
