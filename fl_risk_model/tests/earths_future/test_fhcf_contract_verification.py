"""
FHCF contract verification tests (docs/earths_future_revision/fhcf_contract_verification.md).

These tests are separate from fl_risk_model/tests/earths_future/test_accounting_fixtures.py
on purpose. That file's
test_fhcf_coverage_election_may_be_applied_twice_between_retention_and_limit
only quantifies a suspected discrepancy; it does not assert a verified
correction and is retained unmodified as a record of what was suspected
before this verification. The tests below assert expected values derived
directly from cited clauses of the FHCF 2023-2024 Reimbursement Contract
(FHCF-2023K, Rule 19-8.010 F.A.C., "Coverage Effective: June 1, 2023"),
Article IV(1) and Article V(17), V(19), V(26), V(27), V(28). See the
verification report for the full source table and page references.

Two independent, previously undocumented findings are established here:

1. FORMULA ORDER (Article IV(1)): the Company's Limit caps the *total*
   reimbursement (coverage-level-scaled excess plus the loss adjustment
   expense allowance), not the raw excess-over-retention before scaling.
   The current implementation (fl_risk_model.fhcf.apply_fhcf_recovery)
   caps the excess before scaling: `(1+a) * p * min(E, K)`. The contract
   requires `min((1+a) * p * E, K)`. The two formulas agree whenever
   E <= K (which includes every case up to and including E == K, since
   (1+a)*p <= 0.99 < 1 for all three coverage levels) and diverge only once
   E > K, where the current code permanently under-recovers relative to the
   Company's actual contractual Limit -- by up to 50.5% of the Limit at the
   45% coverage election.

2. AGGREGATION LEVEL (Article V(26), V(28)): Retention and Ultimate Net
   Loss are defined once per Covered Event for the Company's entire book of
   Covered Policies, not per county. fl_risk_model.runner.run_one_scenario
   calls fl_risk_model.fhcf.attach_fhcf_terms_for_losses (which explicitly
   collapses to one row per Company) and then
   fl_risk_model.fhcf.apply_fhcf_recovery on a loss_df that is still at
   Company x County granularity. apply_fhcf_recovery merges the one
   company-level RetentionUSD/LimitUSD onto every county row and then
   computes ExcessUSD/RecoverableUSD ROW BY ROW, so a company's single
   Retention and Limit are silently applied independently to every county
   row instead of once to the company's summed loss. This can either
   under-count excess (many counties each below retention alone) or, more
   importantly for tail risk, let a company recover a multiple of its
   actual contractual Limit (several counties each independently capped at
   the full Limit).

Run with: pytest fl_risk_model/tests/earths_future/test_fhcf_contract_verification.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from fl_risk_model.fhcf import (
    normalize_fhcf_terms,
    apply_fhcf_recovery,
    attach_fhcf_terms_for_losses,
)
from fl_risk_model.config import FHCF_RET_MULTIPLES, FHCF_PAYOUT_MULTIPLE, FHCF_LAE_FACTOR

PREMIUM = 10_000_000.0
LAE = FHCF_LAE_FACTOR  # 1.10, matches Article V(19)(a): 10% of reimbursed losses
LIMIT = PREMIUM * FHCF_PAYOUT_MULTIPLE  # Article V(17): Limit = Premium x Payout Multiple


def _terms(coverage_pct: int, premium: float = PREMIUM) -> pd.DataFrame:
    raw = pd.DataFrame([{"Company": "TestCo", "FHCFPremium": premium, "CoveragePct": coverage_pct}])
    return normalize_fhcf_terms(raw)


def _verified_recovery(gross: float, retention: float, limit: float, p: float, a: float = 0.10) -> float:
    """Article IV(1): reimbursement = min((1+a) * p * max(UNL - Retention, 0), Limit)."""
    excess = max(gross - retention, 0.0)
    return min((1.0 + a) * p * excess, limit)


def _current_code_recovery(gross: float, terms: pd.DataFrame) -> float:
    loss_df = pd.DataFrame([{"Company": "TestCo", "GrossWindLossUSD": gross}])
    return float(apply_fhcf_recovery(loss_df, terms)["RecoveryUSD"].iloc[0])


# ---------------------------------------------------------------------------
# 1. Formula-order verification at 45%, 75%, 90%, across five loss regions.
# ---------------------------------------------------------------------------

COVERAGE_LEVELS = {45: 0.45, 75: 0.75, 90: 0.90}


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_below_retention_current_code_matches_contract(cov_pct, p):
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    gross = retention - 1_000_000.0
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)
    assert expected == 0.0
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_at_retention_current_code_matches_contract(cov_pct, p):
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    expected = _verified_recovery(retention, retention, LIMIT, p)
    actual = _current_code_recovery(retention, terms)
    assert expected == 0.0
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_in_covered_layer_current_code_matches_contract(cov_pct, p):
    """Within the covered layer (E < K), the two formula orderings coincide
    because (1+a)*p <= 0.99 < 1 at every coverage level: scaling E down by
    (1+a)*p before or after comparing to K gives the same result. This is
    the region existing production results were computed in for the vast
    majority of loss draws, which is why this bug was not caught by the
    existing test suite."""
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    gross = retention + 30_000_000.0
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)
    assert actual == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_at_e_equals_limit_boundary_still_agrees(cov_pct, p):
    """At E == K exactly, min(E,K) == E == K in both orderings, so the two
    formulas still agree. This is the last point of agreement."""
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    gross = retention + LIMIT
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)
    assert actual == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_far_above_limit_current_code_UNDER_recovers(cov_pct, p):
    """This is the confirmed defect. Once E > K, the contract (Article
    IV(1)) requires the Company to recover up to its full Limit K. The
    current code plateaus at (1+a)*p*K, which is strictly below K for every
    coverage election (0.99K at 90%, 0.825K at 75%, 0.495K at 45%)."""
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    gross = retention + 3.0 * LIMIT  # E = 3K, deep in "far above the limit"
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)

    assert expected == pytest.approx(LIMIT, rel=1e-9)  # contract: full Limit is reached
    assert actual == pytest.approx(LAE * p * LIMIT, rel=1e-9)  # current code: plateaus below Limit
    assert actual < expected  # confirmed under-recovery
    shortfall_fraction = (expected - actual) / expected
    assert shortfall_fraction == pytest.approx(1.0 - LAE * p, rel=1e-6)


def test_shortfall_is_largest_at_the_45_percent_election():
    """Numerical headline for the report: at 45% coverage, current code
    recovers only 49.5% of the Company's actual contractual Limit once
    losses are far above it (versus 99% at 90% coverage)."""
    terms45 = _terms(45)
    terms90 = _terms(90)
    retention45 = float(terms45["RetentionUSD"].iloc[0])
    retention90 = float(terms90["RetentionUSD"].iloc[0])

    gross45 = retention45 + 3.0 * LIMIT
    gross90 = retention90 + 3.0 * LIMIT

    recovery45 = _current_code_recovery(gross45, terms45)
    recovery90 = _current_code_recovery(gross90, terms90)

    assert recovery45 / LIMIT == pytest.approx(0.495, rel=1e-6)
    assert recovery90 / LIMIT == pytest.approx(0.99, rel=1e-6)
    assert (LIMIT - recovery45) > (LIMIT - recovery90)


# ---------------------------------------------------------------------------
# 2. Aggregation-level verification: one company total row vs. multiple
#    county rows, through the ACTUAL production wrapper
#    (attach_fhcf_terms_for_losses + apply_fhcf_recovery), per runner.py's
#    call path (steps 5.1-5.3).
# ---------------------------------------------------------------------------

def _company_keys_and_market_share():
    market_share_df = pd.DataFrame([{"Company": "TestCo", "StatEntityKey": "SK1"}])
    company_crosswalk_df = pd.DataFrame([{
        "StatEntityKey": "SK1", "NAIC": "99999", "fhcf_participant": True,
    }])
    return market_share_df, company_crosswalk_df


def _terms_with_naic(coverage_pct: int, premium: float = PREMIUM) -> pd.DataFrame:
    raw = pd.DataFrame([{
        "Company": "TestCo", "NAIC": "99999", "FHCFPremium": premium, "CoveragePct": coverage_pct,
    }])
    return normalize_fhcf_terms(raw)


def _recover_via_production_wrapper(loss_df: pd.DataFrame, terms_norm: pd.DataFrame) -> float:
    """Exercises the same two-call sequence as fl_risk_model.runner
    (attach_fhcf_terms_for_losses then apply_fhcf_recovery), not just
    apply_fhcf_recovery in isolation."""
    market_share_df, company_crosswalk_df = _company_keys_and_market_share()
    terms_for_company = attach_fhcf_terms_for_losses(
        loss_df=loss_df,
        terms_df=terms_norm,
        market_share_df=market_share_df,
        company_crosswalk_df=company_crosswalk_df,
        qa_strict=False,
    )
    out = apply_fhcf_recovery(loss_df, terms_for_company)
    return float(out["RecoveryUSD"].sum())


def test_company_total_vs_county_split_diverge_under_current_code():
    """Article V(26)/(28): Retention and Ultimate Net Loss apply once to
    the Company's full book for the Covered Event, so splitting the SAME
    total loss across counties must not change total recovery. Demonstrates
    that it currently does, using the real production wrapper.

    Each of the 3 county rows is independently large enough (Retention +
    2*Limit) that, on its own, its excess-over-retention already exceeds the
    Limit -- exactly the "far above the limit" fixture, replicated per
    county. The one-row case uses the identical TOTAL gross loss (the sum
    of the three county losses) so this isolates the effect of granularity
    alone, holding total loss fixed.
    """
    cov_pct, p = 90, 0.90
    terms = _terms_with_naic(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])

    per_county_gross = retention + 2.0 * LIMIT
    county_rows = pd.DataFrame([
        {"Company": "TestCo", "County": "A", "GrossWindLossUSD": per_county_gross},
        {"Company": "TestCo", "County": "B", "GrossWindLossUSD": per_county_gross},
        {"Company": "TestCo", "County": "C", "GrossWindLossUSD": per_county_gross},
    ])
    recovery_county_split = _recover_via_production_wrapper(county_rows, terms)

    total_gross = 3.0 * per_county_gross  # same total loss, single row
    one_row = pd.DataFrame([{"Company": "TestCo", "GrossWindLossUSD": total_gross}])
    recovery_one_row = _recover_via_production_wrapper(one_row, terms)

    # One row: Excess = total_gross - retention (retention subtracted ONCE)
    # is far above K, so recovery correctly plateaus at (1+a)*p*K.
    assert recovery_one_row == pytest.approx(LAE * p * LIMIT, rel=1e-6)

    # BUG: with the SAME total loss split into 3 counties, retention is
    # subtracted once PER ROW, so each row's excess (2*Limit) already
    # exceeds Limit on its own and each row independently plateaus at
    # (1+a)*p*K -- tripling total recovery for identical total loss.
    assert recovery_county_split == pytest.approx(3.0 * LAE * p * LIMIT, rel=1e-6)
    assert recovery_county_split > recovery_one_row
    assert recovery_county_split > LIMIT  # exceeds the Company's actual contractual maximum


def test_company_total_vs_county_split_would_agree_under_aggregate_first_fix():
    """Sanity check for the proposed fix: if GrossWindLossUSD is summed to
    company level BEFORE calling apply_fhcf_recovery (as Article V(26)/(28)
    requires), the one-row and county-split representations of the same
    total loss agree exactly."""
    cov_pct, p = 90, 0.90
    terms = _terms_with_naic(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])

    per_county_gross = retention + 2.0 * LIMIT
    county_rows = pd.DataFrame([
        {"Company": "TestCo", "County": "A", "GrossWindLossUSD": per_county_gross},
        {"Company": "TestCo", "County": "B", "GrossWindLossUSD": per_county_gross},
        {"Company": "TestCo", "County": "C", "GrossWindLossUSD": per_county_gross},
    ])
    aggregated_first = (
        county_rows.groupby("Company", as_index=False)["GrossWindLossUSD"].sum()
    )
    recovery_aggregated_first = _recover_via_production_wrapper(aggregated_first, terms)

    total_gross = 3.0 * per_county_gross
    one_row = pd.DataFrame([{"Company": "TestCo", "GrossWindLossUSD": total_gross}])
    recovery_one_row = _recover_via_production_wrapper(one_row, terms)

    assert recovery_aggregated_first == pytest.approx(recovery_one_row, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Statewide cap: two insurers + Citizens, below/at/above the $17B cap.
#    Exercises fl_risk_model.runner._apply_industry_season_cap directly,
#    which is the actual production wrapper for the statewide constraint.
# ---------------------------------------------------------------------------

from fl_risk_model.runner import _apply_industry_season_cap  # noqa: E402


def _precap_frame(rows):
    return pd.DataFrame(rows)


def test_statewide_cap_below_capacity_no_scaling():
    cap = 17_000_000_000.0
    private = _precap_frame([
        {"Company": "InsA", "County": "X", "GrossWindLossUSD": 3e9, "FHCF_RecoveryPreCapUSD": 2e9},
        {"Company": "InsB", "County": "X", "GrossWindLossUSD": 4e9, "FHCF_RecoveryPreCapUSD": 3e9},
    ])
    citizens = _precap_frame([
        {"Company": "Citizens", "County": "X", "GrossWindLossUSD": 2e9, "FHCF_RecoveryPreCapUSD": 1.5e9},
    ])
    p, c, diag = _apply_industry_season_cap(private, citizens, cap)
    assert diag["fhcf_total_precap_usd"] == pytest.approx(6.5e9)
    assert diag["fhcf_scaling_factor"] == pytest.approx(1.0)
    assert diag["fhcf_cap_binding"] is False
    assert diag["fhcf_shortfall_usd"] == pytest.approx(0.0)
    # Reconciliation: recoveries unscaled, net = gross - recovery, no leakage.
    assert p["FHCF_RecoveryUSD"].sum() == pytest.approx(5e9)
    assert c["FHCF_RecoveryUSD"].sum() == pytest.approx(1.5e9)
    assert (p["NetWindUSD"] >= 0).all() and (c["NetWindUSD"] >= 0).all()


def test_statewide_cap_exactly_at_capacity():
    cap = 6.5e9
    private = _precap_frame([{"Company": "InsA", "County": "X", "GrossWindLossUSD": 5e9,
                               "FHCF_RecoveryPreCapUSD": 4.5e9}])
    citizens = _precap_frame([{"Company": "Citizens", "County": "X", "GrossWindLossUSD": 3e9,
                                "FHCF_RecoveryPreCapUSD": 2e9}])
    p, c, diag = _apply_industry_season_cap(private, citizens, cap)
    assert diag["fhcf_scaling_factor"] == pytest.approx(1.0)
    assert diag["fhcf_cap_binding"] is False  # scale==1.0 is not "binding" by the model's own definition
    assert diag["fhcf_shortfall_usd"] == pytest.approx(0.0)


def test_statewide_cap_above_capacity_prorates_private_and_citizens_together():
    cap = 5.0e9
    private = _precap_frame([
        {"Company": "InsA", "County": "X", "GrossWindLossUSD": 5e9, "FHCF_RecoveryPreCapUSD": 4.0e9},
        {"Company": "InsB", "County": "X", "GrossWindLossUSD": 5e9, "FHCF_RecoveryPreCapUSD": 4.0e9},
    ])
    citizens = _precap_frame([{"Company": "Citizens", "County": "X", "GrossWindLossUSD": 3e9,
                                "FHCF_RecoveryPreCapUSD": 2.0e9}])
    p, c, diag = _apply_industry_season_cap(private, citizens, cap)

    pre_total = 10.0e9
    expected_scale = cap / pre_total  # 0.5
    assert diag["fhcf_scaling_factor"] == pytest.approx(expected_scale)
    assert diag["fhcf_cap_binding"] is True
    assert diag["fhcf_total_postcap_usd"] == pytest.approx(cap)
    assert diag["fhcf_shortfall_usd"] == pytest.approx(pre_total - cap)

    # Both private AND Citizens are scaled by the SAME factor (single,
    # combined statewide constraint -- Article IV(3): "reduce ... uniformly
    # among all insurers").
    assert p["FHCF_RecoveryUSD"].sum() == pytest.approx(8.0e9 * expected_scale)
    assert c["FHCF_RecoveryUSD"].sum() == pytest.approx(2.0e9 * expected_scale)
    total_post = p["FHCF_RecoveryUSD"].sum() + c["FHCF_RecoveryUSD"].sum()
    assert total_post == pytest.approx(cap)

    # Reconciliation: retained (net) losses plus recovered amounts equal
    # gross losses for every row; no duplicated recoveries.
    combined = pd.concat([p, c], ignore_index=True)
    recon = (combined["NetWindUSD"] + combined["FHCF_RecoveryUSD"] - combined["GrossWindLossUSD"]).abs()
    assert (recon < 1e-6).all()


# ---------------------------------------------------------------------------
# 4. Citizens: live fallback path vs. the dead cfg-based helper.
#
# fl_risk_model.runner.run_one_scenario's ACTUAL Citizens code path
# (_citizens_terms_fallback_row, called at lines ~806-810) builds a raw
# {Company, FHCFPremium, CoveragePct} row and normalizes it with the SAME
# fl_risk_model.fhcf.normalize_fhcf_terms used for every other company --
# i.e. Limit = Premium x Payout Multiple, no extra coverage factor, matching
# Article V(17). This path is used whenever Citizens is not resolved from
# fl_risk_model/data/24fin_fhcf.csv (or CITIZENS_FHCF_FORCE_CONFIG_TERMS is
# set); in the archived production inputs, Citizens IS resolved from that
# CSV (NAIC 10064), so even this fallback is not reached in practice.
#
# fl_risk_model.branches.citizens.citizens_fhcf_terms_from_cfg_or_csv is a
# SEPARATE function that reads fl_risk_model.config.CITIZENS_FHCF_LIMIT_USD,
# which IS defined with an extra x CoveragePct factor
# (config.py: "Premium x PayoutMultiplier x CoveragePct"), inconsistent with
# Article V(17). It is imported into runner.py but its only call site there
# is commented out -- it is unreachable dead code under the current
# runner.py, not merely inactive for the current data.
# ---------------------------------------------------------------------------

from fl_risk_model.branches.citizens import citizens_fhcf_terms_from_cfg_or_csv  # noqa: E402
from fl_risk_model import config as cfg  # noqa: E402


def test_live_citizens_fallback_matches_general_path_formula():
    """The live runner.py fallback (_citizens_terms_fallback_row +
    normalize_fhcf_terms) omits the coverage factor from Limit, exactly
    like the general company path -- both are consistent with Article
    V(17)."""
    premium = cfg.CITIZENS_FHCF_PREMIUM_USD
    cov_pct = cfg.CITIZENS_FHCF_COVERAGE_PCT * 100  # 0.90 -> 90
    live_fallback_terms = normalize_fhcf_terms(pd.DataFrame([{
        "Company": cfg.CITIZENS_COMPANY_NAME if hasattr(cfg, "CITIZENS_COMPANY_NAME")
        else "Citizens Property Insurance Corporation",
        "FHCFPremium": premium,
        "CoveragePct": cov_pct,
    }]))
    general_path_terms = _terms(int(cov_pct), premium)

    assert float(live_fallback_terms["LimitUSD"].iloc[0]) == pytest.approx(
        float(general_path_terms["LimitUSD"].iloc[0])
    )
    assert float(live_fallback_terms["LimitUSD"].iloc[0]) == pytest.approx(premium * FHCF_PAYOUT_MULTIPLE)


def test_dead_cfg_helper_would_apply_coverage_factor_twice_if_ever_called():
    """Documents (does not exercise via any reachable call path) the
    inconsistency in the unreachable citizens_fhcf_terms_from_cfg_or_csv /
    CITIZENS_FHCF_LIMIT_USD combination, for the record."""
    naic = str(getattr(cfg, "CITIZENS_NAIC", "10064"))
    empty_terms_norm = pd.DataFrame(columns=["NAIC", "StatEntityKey", "CoveragePct_norm",
                                              "RetentionUSD", "LimitUSD"])
    empty_company_keys = pd.DataFrame(columns=["NAIC", "StatEntityKey"])

    dead_path_row = citizens_fhcf_terms_from_cfg_or_csv(empty_terms_norm, empty_company_keys, cfg)
    dead_path_limit = float(dead_path_row["LimitUSD"].iloc[0])

    correct_limit = cfg.CITIZENS_FHCF_PREMIUM_USD * FHCF_PAYOUT_MULTIPLE
    assert dead_path_limit == pytest.approx(cfg.CITIZENS_FHCF_LIMIT_USD)
    assert dead_path_limit != pytest.approx(correct_limit)
    assert dead_path_limit == pytest.approx(correct_limit * cfg.CITIZENS_FHCF_COVERAGE_PCT)
