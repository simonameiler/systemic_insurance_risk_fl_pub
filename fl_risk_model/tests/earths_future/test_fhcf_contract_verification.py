"""
FHCF contract verification tests (docs/earths_future_revision/fhcf_contract_verification.md).

These tests are separate from fl_risk_model/tests/earths_future/test_accounting_fixtures.py
on purpose. That file's
test_fhcf_coverage_election_may_be_applied_twice_between_retention_and_limit
only quantified a suspected discrepancy against an alternative hypothesis
that was NOT adopted (removing the coverage factor entirely); it is
retained unmodified as a record of that earlier, evidence-lighter
suspicion.

The two defects verified against the primary FHCF 2023-2024 Reimbursement
Contract (FHCF-2023K, Rule 19-8.010 F.A.C., "Coverage Effective: June 1,
2023"), Article IV(1) and Article V(17), V(19), V(26), V(27), V(28), have
now been PATCHED in fl_risk_model/fhcf.py::apply_fhcf_recovery:

1. FORMULA ORDER (Article IV(1)): the Company's Limit caps the *total*
   reimbursement (coverage-level-scaled excess plus the loss adjustment
   expense allowance), not the raw excess-over-retention before scaling.
   PATCHED: apply_fhcf_recovery now computes
   min((1+a) * p * E, K) instead of (1+a) * p * min(E, K).

2. AGGREGATION LEVEL (Article V(26), V(28)): Retention and Ultimate Net
   Loss are defined once per Covered Event for the Company's entire book,
   not per county. PATCHED: apply_fhcf_recovery now sums GrossWindLossUSD
   to one row per Company before computing recovery, then allocates the
   single company-level recovery back to the original rows in proportion
   to each row's share of the company's gross loss.

The tests below are REGRESSION tests against the verified formula: they
assert what the code SHOULD do and currently DOES do, now that both
defects are patched. They are not merely defect-characterization tests
that happen to pass -- each expected value is computed independently
(_verified_recovery, or hand-derived company totals), not copied from a
prior run of the code under test. The record of the OLD (pre-patch)
behavior that these tests used to assert is preserved in
docs/earths_future_revision/fhcf_contract_verification.md and in the
correction register, not re-derived here.

Run with: pytest fl_risk_model/tests/earths_future/test_fhcf_contract_verification.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fl_risk_model.fhcf import (
    normalize_fhcf_terms,
    apply_fhcf_recovery,
    attach_fhcf_terms_for_losses,
)
from fl_risk_model.config import FHCF_RET_MULTIPLES, FHCF_PAYOUT_MULTIPLE, FHCF_LAE_FACTOR
from fl_risk_model.runner import _apply_industry_season_cap

PREMIUM = 10_000_000.0
LAE = FHCF_LAE_FACTOR  # 1.10, i.e. (1+a); matches Article V(19)(a): 10% of reimbursed losses
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
# 1. Formula verification at 45%, 75%, 90%, across six loss regions:
#    below retention, at retention, within the layer, at E=K, at the actual
#    saturation point E = K/(1.10*p), and well above saturation.
# ---------------------------------------------------------------------------

COVERAGE_LEVELS = {45: 0.45, 75: 0.75, 90: 0.90}


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_below_retention(cov_pct, p):
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    gross = retention - 1_000_000.0
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)
    assert expected == 0.0
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_at_retention(cov_pct, p):
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    expected = _verified_recovery(retention, retention, LIMIT, p)
    actual = _current_code_recovery(retention, terms)
    assert expected == 0.0
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_within_the_covered_layer(cov_pct, p):
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    gross = retention + 30_000_000.0
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)
    assert actual == pytest.approx(expected, rel=1e-9)
    assert actual == pytest.approx(LAE * p * 30_000_000.0, rel=1e-9)


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_at_e_equals_k(cov_pct, p):
    """At E == K exactly, both formula orderings agree (this was already
    true before the patch, since (1+a)*p <= 0.99 < 1 at every coverage
    level: E==K is not yet past the true saturation point)."""
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    gross = retention + LIMIT
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)
    assert actual == pytest.approx(expected, rel=1e-9)
    assert actual == pytest.approx(LAE * p * LIMIT, rel=1e-9)
    assert actual < LIMIT  # not yet saturated


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_at_the_actual_saturation_point(cov_pct, p):
    """E = K / (1.10*p) is exactly where the verified formula's scaled
    excess first equals K -- the true saturation point, strictly beyond
    E=K for every coverage election (since 1.10*p <= 0.99 < 1). Before the
    patch, current code had already been plateaued at (1+a)*p*K since
    E=K, so this was exactly where the pre-patch code's under-recovery was
    largest in relative terms yet still small in absolute terms; after the
    patch, this is the last point where recovery is still (just) reaching
    K rather than being capped short of it."""
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    saturation_excess = LIMIT / (LAE * p)
    gross = retention + saturation_excess
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)
    assert expected == pytest.approx(LIMIT, rel=1e-9)
    assert actual == pytest.approx(LIMIT, rel=1e-9)  # patch: now reaches the true Limit


@pytest.mark.parametrize("cov_pct,p", COVERAGE_LEVELS.items())
def test_well_above_saturation_reaches_full_limit(cov_pct, p):
    """Regression test for the confirmed formula-order defect. Once E is
    well past saturation, the contract (Article IV(1)) requires the
    Company to recover its full Limit K, and the patched code now does."""
    terms = _terms(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])
    gross = retention + 3.0 * LIMIT  # E = 3K, deep past saturation
    expected = _verified_recovery(gross, retention, LIMIT, p)
    actual = _current_code_recovery(gross, terms)
    assert expected == pytest.approx(LIMIT, rel=1e-9)
    assert actual == pytest.approx(LIMIT, rel=1e-9)


def test_saturation_shortfall_no_longer_depends_on_coverage_election():
    """Before the patch, current code plateaued at (1+a)*p*K, so the
    relative shortfall against the true Limit K grew as coverage election
    fell (49.5% recovered at 45% vs. 99% at 90%). After the patch, all
    three elections correctly reach the full Limit once losses are deep
    enough past saturation."""
    for cov_pct, p in COVERAGE_LEVELS.items():
        terms = _terms(cov_pct)
        retention = float(terms["RetentionUSD"].iloc[0])
        gross = retention + 3.0 * LIMIT
        recovery = _current_code_recovery(gross, terms)
        assert recovery == pytest.approx(LIMIT, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Aggregation-level verification, through the ACTUAL production wrapper
#    (attach_fhcf_terms_for_losses + apply_fhcf_recovery), per runner.py's
#    call path (steps 5.1-5.3). Covers: unequal county splits, a case where
#    every county row is individually below retention but the company sum
#    is above it, a case that exhausts the company limit, and zero-loss
#    rows. Private insurers and combined Citizens are both covered (Section
#    4 repeats the Citizens-specific comparison against its own terms path).
# ---------------------------------------------------------------------------

def _company_keys_and_market_share(company="TestCo", key="SK1", naic="99999"):
    market_share_df = pd.DataFrame([{"Company": company, "StatEntityKey": key}])
    company_crosswalk_df = pd.DataFrame([{
        "StatEntityKey": key, "NAIC": naic, "fhcf_participant": True,
    }])
    return market_share_df, company_crosswalk_df


def _terms_with_naic(coverage_pct: int, premium: float = PREMIUM,
                      company: str = "TestCo", naic: str = "99999") -> pd.DataFrame:
    raw = pd.DataFrame([{
        "Company": company, "NAIC": naic, "FHCFPremium": premium, "CoveragePct": coverage_pct,
    }])
    return normalize_fhcf_terms(raw)


def _recover_via_production_wrapper(loss_df: pd.DataFrame, terms_norm: pd.DataFrame,
                                     company: str = "TestCo", key: str = "SK1",
                                     naic: str = "99999") -> pd.DataFrame:
    """Exercises the same two-call sequence as fl_risk_model.runner
    (attach_fhcf_terms_for_losses then apply_fhcf_recovery), not just
    apply_fhcf_recovery in isolation. Returns the full row-level output
    (not just the summed recovery) so callers can check per-row
    reconciliation."""
    market_share_df, company_crosswalk_df = _company_keys_and_market_share(company, key, naic)
    terms_for_company = attach_fhcf_terms_for_losses(
        loss_df=loss_df,
        terms_df=terms_norm,
        market_share_df=market_share_df,
        company_crosswalk_df=company_crosswalk_df,
        qa_strict=False,
    )
    return apply_fhcf_recovery(loss_df, terms_for_company)


def test_unequal_county_split_matches_one_row_total():
    """Regression test: unequal county splits of the same company total now
    give the same total recovery as a single aggregated row (the
    aggregation defect is fixed)."""
    cov_pct, p = 90, 0.90
    terms = _terms_with_naic(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])

    total_gross = retention + 3.0 * LIMIT
    # Deliberately unequal split: 60% / 30% / 10%.
    county_rows = pd.DataFrame([
        {"Company": "TestCo", "County": "A", "GrossWindLossUSD": 0.6 * total_gross},
        {"Company": "TestCo", "County": "B", "GrossWindLossUSD": 0.3 * total_gross},
        {"Company": "TestCo", "County": "C", "GrossWindLossUSD": 0.1 * total_gross},
    ])
    out_split = _recover_via_production_wrapper(county_rows, terms)

    one_row = pd.DataFrame([{"Company": "TestCo", "GrossWindLossUSD": total_gross}])
    out_one = _recover_via_production_wrapper(one_row, terms)

    expected = _verified_recovery(total_gross, retention, LIMIT, p)
    assert float(out_split["RecoveryUSD"].sum()) == pytest.approx(expected, rel=1e-9)
    assert float(out_one["RecoveryUSD"].sum()) == pytest.approx(expected, rel=1e-9)
    assert float(out_split["RecoveryUSD"].sum()) == pytest.approx(float(out_one["RecoveryUSD"].sum()), rel=1e-9)


def test_each_county_below_retention_alone_but_company_sum_above_it():
    """The case the brief specifically calls out: no single county's loss
    exceeds the company's Retention on its own, but the company's SUMMED
    loss does. Before the patch this produced zero recovery everywhere
    (each row's own excess was zero); the patch correctly recognizes the
    company-level excess."""
    cov_pct, p = 75, 0.75
    terms = _terms_with_naic(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])

    per_county = retention * 0.4  # each row alone: well below retention
    n_counties = 4  # 4 * 0.4 * retention = 1.6 * retention > retention
    county_rows = pd.DataFrame([
        {"Company": "TestCo", "County": f"C{i}", "GrossWindLossUSD": per_county}
        for i in range(n_counties)
    ])
    total_gross = n_counties * per_county
    assert total_gross > retention  # sanity check on the fixture itself

    out = _recover_via_production_wrapper(county_rows, terms)
    expected = _verified_recovery(total_gross, retention, LIMIT, p)
    assert expected > 0.0  # the company IS eligible for recovery in aggregate
    assert float(out["RecoveryUSD"].sum()) == pytest.approx(expected, rel=1e-9)
    # Every individual row shows zero excess on its own -- the recovery is
    # correctly attributed at the company level, not fabricated per row.
    assert (per_county < retention)


def test_company_limit_exhausted_case_reconciles_across_county_rows():
    """A case that exhausts the company Limit, split unevenly across
    counties with one zero-loss row included. Recovery must reconcile
    exactly to the company-level capped amount, and the zero-loss row must
    receive exactly zero recovery (defined 0/0 share, not NaN)."""
    cov_pct, p = 90, 0.90
    terms = _terms_with_naic(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])

    total_gross = retention + 5.0 * LIMIT  # deep past saturation
    county_rows = pd.DataFrame([
        {"Company": "TestCo", "County": "A", "GrossWindLossUSD": 0.7 * total_gross},
        {"Company": "TestCo", "County": "B", "GrossWindLossUSD": 0.3 * total_gross},
        {"Company": "TestCo", "County": "Zero", "GrossWindLossUSD": 0.0},
    ])
    out = _recover_via_production_wrapper(county_rows, terms)

    assert float(out["RecoveryUSD"].sum()) == pytest.approx(LIMIT, rel=1e-9)
    zero_row = out[out["County"] == "Zero"].iloc[0]
    assert zero_row["RecoveryUSD"] == 0.0
    assert not np.isnan(zero_row["RecoveryUSD"])
    assert zero_row["NetWindUSD"] == 0.0

    # Reconciliation: gross == net + recovery for every row, no negative
    # values, no NaNs, and CompanyRecoveryUSD is the same broadcast value
    # on every row of this company.
    recon = (out["NetWindUSD"] + out["RecoveryUSD"] - out["GrossWindLossUSD"]).abs()
    assert (recon < 1e-6).all()
    assert (out["NetWindUSD"] >= -1e-6).all()
    assert out["RecoveryUSD"].notna().all()
    assert out["CompanyRecoveryUSD"].nunique() == 1


def test_zero_loss_company_has_zero_recovery_and_no_nan():
    terms = _terms_with_naic(90)
    county_rows = pd.DataFrame([
        {"Company": "TestCo", "County": "A", "GrossWindLossUSD": 0.0},
        {"Company": "TestCo", "County": "B", "GrossWindLossUSD": 0.0},
    ])
    out = _recover_via_production_wrapper(county_rows, terms)
    assert (out["RecoveryUSD"] == 0.0).all()
    assert (out["NetWindUSD"] == 0.0).all()
    assert out["RecoveryUSD"].notna().all()
    assert out["CompanyGrossWindLossUSD"].eq(0.0).all()


def test_row_order_and_row_count_preserved():
    """No rows are dropped, added, or reordered by the aggregation fix."""
    terms = _terms_with_naic(90)
    county_rows = pd.DataFrame([
        {"Company": "TestCo", "County": "Z", "GrossWindLossUSD": 5_000_000.0},
        {"Company": "TestCo", "County": "A", "GrossWindLossUSD": 200_000_000.0},
        {"Company": "TestCo", "County": "M", "GrossWindLossUSD": 0.0},
    ])
    out = _recover_via_production_wrapper(county_rows, terms)
    assert len(out) == len(county_rows)
    assert list(out["County"]) == ["Z", "A", "M"]
    assert not out.duplicated(subset=["Company", "County"]).any()


def test_company_total_vs_county_split_still_agrees_after_patch():
    """Direct regression test on the exact fixture the earlier
    (pre-patch) version of this file used to demonstrate the defect: same
    total loss represented as one row vs. three equal county rows, each
    individually already past the company's own saturation point. Both
    representations now recover the company's full Limit, not a multiple
    of it."""
    cov_pct, p = 90, 0.90
    terms = _terms_with_naic(cov_pct)
    retention = float(terms["RetentionUSD"].iloc[0])

    per_county_gross = retention + 2.0 * LIMIT
    county_rows = pd.DataFrame([
        {"Company": "TestCo", "County": "A", "GrossWindLossUSD": per_county_gross},
        {"Company": "TestCo", "County": "B", "GrossWindLossUSD": per_county_gross},
        {"Company": "TestCo", "County": "C", "GrossWindLossUSD": per_county_gross},
    ])
    recovery_county_split = float(_recover_via_production_wrapper(county_rows, terms)["RecoveryUSD"].sum())

    total_gross = 3.0 * per_county_gross
    one_row = pd.DataFrame([{"Company": "TestCo", "GrossWindLossUSD": total_gross}])
    recovery_one_row = float(_recover_via_production_wrapper(one_row, terms)["RecoveryUSD"].sum())

    # Pre-patch record (see fhcf_contract_verification.md): one-row recovery
    # was LAE*p*LIMIT (~99% of Limit) and the 3-county split recovered
    # 3x that (>3x the company's actual Limit). Both are now the full Limit.
    assert recovery_one_row == pytest.approx(LIMIT, rel=1e-9)
    assert recovery_county_split == pytest.approx(LIMIT, rel=1e-9)
    assert recovery_county_split == pytest.approx(recovery_one_row, rel=1e-9)
    assert recovery_county_split <= LIMIT + 1e-6  # no longer exceeds the Company's actual maximum


# ---------------------------------------------------------------------------
# 3. Statewide cap integration: recoveries for at least two private insurers
#    AND Citizens are computed via the real apply_fhcf_recovery (not
#    supplied as dummy pre-cap frames) and then passed through
#    fl_risk_model.runner._apply_industry_season_cap, below/at/above the
#    cap.
# ---------------------------------------------------------------------------

def _computed_precap_frame(company, naic, coverage_pct, premium, gross_by_county):
    terms = _terms_with_naic(coverage_pct, premium, company=company, naic=naic)
    loss_df = pd.DataFrame([
        {"Company": company, "County": c, "GrossWindLossUSD": g}
        for c, g in gross_by_county.items()
    ])
    out = _recover_via_production_wrapper(loss_df, terms, company=company, key=f"SK-{naic}", naic=naic)
    return out.rename(columns={"RecoveryUSD": "FHCF_RecoveryPreCapUSD"})[
        ["Company", "County", "GrossWindLossUSD", "FHCF_RecoveryPreCapUSD"]
    ]


def _two_insurers_and_citizens(scale: float):
    """Two private insurers (different coverage elections, multi-county)
    plus Citizens, each with REAL computed pre-cap recovery. `scale`
    multiplies every company's loss uniformly so the same fixture can probe
    below/at/above the statewide cap."""
    # Premiums sized so Retention (Premium x retention multiple) is well
    # below the fixture's gross losses, so each company has genuine
    # recoverable excess to feed the statewide cap (a too-large premium
    # relative to loss, as in an earlier draft of this fixture, makes
    # Retention exceed Gross and every recovery zero).
    insA = _computed_precap_frame(
        "InsA", "11111", 90, PREMIUM * 3,  # larger insurer
        {"X": 200_000_000.0 * scale, "Y": 150_000_000.0 * scale},
    )
    insB = _computed_precap_frame(
        "InsB", "22222", 45, PREMIUM * 1,
        {"X": 90_000_000.0 * scale, "Z": 40_000_000.0 * scale},
    )
    private = pd.concat([insA, insB], ignore_index=True)
    citizens = _computed_precap_frame(
        "Citizens", "10064", 90, PREMIUM * 4,
        {"X": 250_000_000.0 * scale, "Y": 100_000_000.0 * scale},
    )
    return private, citizens


def test_statewide_cap_integration_below_at_and_above_capacity_with_real_recoveries():
    # Determine, from the fixture itself, the pre-cap total at scale=1.0.
    private1, citizens1 = _two_insurers_and_citizens(scale=1.0)
    pre_total_1 = float(private1["FHCF_RecoveryPreCapUSD"].sum() + citizens1["FHCF_RecoveryPreCapUSD"].sum())
    assert pre_total_1 > 0.0

    below_cap = pre_total_1 * 10.0  # cap far above pre-cap demand
    at_cap = pre_total_1  # cap exactly equal to pre-cap demand
    above_cap = pre_total_1 * 0.4  # cap well below pre-cap demand

    for cap_usd, expect_binding in [(below_cap, False), (at_cap, False), (above_cap, True)]:
        private, citizens = _two_insurers_and_citizens(scale=1.0)
        pre_total = float(private["FHCF_RecoveryPreCapUSD"].sum() + citizens["FHCF_RecoveryPreCapUSD"].sum())
        p_out, c_out, diag = _apply_industry_season_cap(private, citizens, cap_usd)

        expected_scale = 1.0 if pre_total <= cap_usd else cap_usd / pre_total
        assert diag["fhcf_scaling_factor"] == pytest.approx(expected_scale, rel=1e-9)
        assert diag["fhcf_cap_binding"] == expect_binding
        assert diag["fhcf_total_postcap_usd"] == pytest.approx(min(pre_total, cap_usd), rel=1e-9)
        assert diag["fhcf_shortfall_usd"] == pytest.approx(max(pre_total - cap_usd, 0.0), rel=1e-6)

        # Both private companies AND Citizens are scaled by the SAME factor
        # (Article IV(3): reduce uniformly among all insurers).
        combined = pd.concat([p_out, c_out], ignore_index=True)
        for company in combined["Company"].unique():
            rows = combined[combined["Company"] == company]
            pre = rows["FHCF_RecoveryPreCapUSD"].sum()
            post = rows["FHCF_RecoveryUSD"].sum()
            if pre > 0:
                assert post / pre == pytest.approx(expected_scale, rel=1e-6)

        # Each company's OWN limit is still respected: post-cap recovery
        # never exceeds what apply_fhcf_recovery already capped it at
        # (scaling can only reduce it further, never restore it above the
        # company's own pre-cap, already-Limit-respecting amount).
        assert (combined["FHCF_RecoveryUSD"] <= combined["FHCF_RecoveryPreCapUSD"] + 1e-6).all()

        # Reconciliation after scaling: gross == net + recovery, no negative
        # net, statewide total recovery equals min(pre_total, cap).
        recon = (combined["NetWindUSD"] + combined["FHCF_RecoveryUSD"] - combined["GrossWindLossUSD"]).abs()
        assert (recon < 1e-6).all()
        assert (combined["NetWindUSD"] >= -1e-6).all()
        assert combined["FHCF_RecoveryUSD"].sum() == pytest.approx(min(pre_total, cap_usd), rel=1e-9)


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
# runner.py, not merely inactive for the current data. Not patched, per the
# brief ("leave unrelated dead code alone").
# ---------------------------------------------------------------------------

from fl_risk_model.branches.citizens import citizens_fhcf_terms_from_cfg_or_csv  # noqa: E402
from fl_risk_model import config as cfg  # noqa: E402


def test_live_citizens_fallback_matches_general_path_formula():
    """The live runner.py fallback (_citizens_terms_fallback_row +
    normalize_fhcf_terms) omits the coverage factor from Limit, exactly
    like the general company path -- both are consistent with Article
    V(17), and both now use the patched, company-aggregated
    apply_fhcf_recovery."""
    premium = cfg.CITIZENS_FHCF_PREMIUM_USD
    cov_pct = cfg.CITIZENS_FHCF_COVERAGE_PCT * 100  # 0.90 -> 90
    live_fallback_terms = normalize_fhcf_terms(pd.DataFrame([{
        "Company": getattr(cfg, "CITIZENS_COMPANY_NAME", "Citizens Property Insurance Corporation"),
        "FHCFPremium": premium,
        "CoveragePct": cov_pct,
    }]))
    general_path_terms = _terms(int(cov_pct), premium)

    assert float(live_fallback_terms["LimitUSD"].iloc[0]) == pytest.approx(
        float(general_path_terms["LimitUSD"].iloc[0])
    )
    assert float(live_fallback_terms["LimitUSD"].iloc[0]) == pytest.approx(premium * FHCF_PAYOUT_MULTIPLE)


def test_citizens_multi_county_aggregation_via_live_path():
    """Citizens' combined (single-entity) representation, exercised across
    multiple counties, reconciles the same way a private insurer's does."""
    premium = cfg.CITIZENS_FHCF_PREMIUM_USD
    cov_pct = int(round(cfg.CITIZENS_FHCF_COVERAGE_PCT * 100))
    terms = normalize_fhcf_terms(pd.DataFrame([{
        "Company": "Citizens Property Insurance Corporation",
        "NAIC": "10064",
        "FHCFPremium": premium,
        "CoveragePct": cov_pct,
    }]))
    retention = float(terms["RetentionUSD"].iloc[0])
    limit = float(terms["LimitUSD"].iloc[0])
    total_gross = retention + 2.0 * limit
    county_rows = pd.DataFrame([
        {"Company": "Citizens Property Insurance Corporation", "County": "X", "GrossWindLossUSD": 0.5 * total_gross},
        {"Company": "Citizens Property Insurance Corporation", "County": "Y", "GrossWindLossUSD": 0.5 * total_gross},
    ])
    out = _recover_via_production_wrapper(
        county_rows, terms, company="Citizens Property Insurance Corporation", key="C6949", naic="10064"
    )
    p = cov_pct / 100.0
    expected = _verified_recovery(total_gross, retention, limit, p)
    assert float(out["RecoveryUSD"].sum()) == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(limit, rel=1e-9)  # this fixture is deep past saturation


def test_dead_cfg_helper_would_apply_coverage_factor_twice_if_ever_called():
    """Documents (does not exercise via any reachable call path) the
    inconsistency in the unreachable citizens_fhcf_terms_from_cfg_or_csv /
    CITIZENS_FHCF_LIMIT_USD combination, for the record. Left unpatched:
    unrelated dead code, out of scope for this correction."""
    empty_terms_norm = pd.DataFrame(columns=["NAIC", "StatEntityKey", "CoveragePct_norm",
                                              "RetentionUSD", "LimitUSD"])
    empty_company_keys = pd.DataFrame(columns=["NAIC", "StatEntityKey"])

    dead_path_row = citizens_fhcf_terms_from_cfg_or_csv(empty_terms_norm, empty_company_keys, cfg)
    dead_path_limit = float(dead_path_row["LimitUSD"].iloc[0])

    correct_limit = cfg.CITIZENS_FHCF_PREMIUM_USD * FHCF_PAYOUT_MULTIPLE
    assert dead_path_limit == pytest.approx(cfg.CITIZENS_FHCF_LIMIT_USD)
    assert dead_path_limit != pytest.approx(correct_limit)
    assert dead_path_limit == pytest.approx(correct_limit * cfg.CITIZENS_FHCF_COVERAGE_PCT)
