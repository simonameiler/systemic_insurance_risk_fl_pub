"""Solvency decisions must not amplify sub-cent cancellation into lost premiums."""
import numpy as np
import pandas as pd
import pytest
from fl_risk_model.capital import apply_group_capital_contributions


def metadata(names, surplus=1.0, group_surplus=1.0, ratio=1.0):
    return pd.DataFrame({
        'Company': names, 'SurplusUSD': surplus, 'GroupSurplusUSD': group_surplus,
        'NAICGroupNumber': 'test', 'NAICGroupName': 'Test Group', 'GroupToEntityRatio': ratio,
    })


def test_subcent_balances_keep_insurer_in_assessment_base():
    balances = [-1e-7, -0.009, 0.0, 0.009, -0.01, 0.01, -100.0]
    names = [f'C{i}' for i in range(len(balances))]
    out = apply_group_capital_contributions(
        pd.DataFrame({'Company': names, 'EndingSurplusUSD': balances}),
        metadata(names), rng=np.random.default_rng(42),
    )
    assert out.AdjustedSurplusUSD.tolist() == [0, 0, 0, 0, -0.01, 0.01, -100.0]
    assert out.DefaultFlag.tolist() == [False, False, False, False, True, False, True]
    premiums = pd.Series([1e9] * len(names), index=out.index)
    assert premiums[~out.DefaultFlag].sum() == 5e9
    np.testing.assert_allclose(out.AdjustedSurplusUSD,
        out.EndingSurplusUSD + out.GroupContributionUSD + out.SurplusRoundingAdjustmentUSD,
        rtol=0, atol=1e-12)


def test_fully_funded_group_has_no_roundoff_defaults():
    amounts = np.random.default_rng(17).uniform(1e6, 1e9, 40)
    names = [f'C{i}' for i in range(len(amounts))]
    out = apply_group_capital_contributions(
        pd.DataFrame({'Company': names, 'EndingSurplusUSD': -amounts}),
        metadata(names, group_surplus=float(amounts.sum())+len(names), ratio=20),
        rng=np.random.default_rng(42),
    )
    assert not out.DefaultFlag.any()
    assert out.AdjustedSurplusUSD.eq(0).all()
    assert out.SurplusRoundingAdjustmentUSD.abs().sum() < 0.01
    assert out.GroupContributionUSD.sum() == pytest.approx(amounts.sum(), abs=0.01)


def test_donor_and_recipient_use_same_zero_convention():
    names = ['recipient', 'donor']
    out = apply_group_capital_contributions(
        pd.DataFrame({'Company': names, 'EndingSurplusUSD': [-100.0, 100.0]}),
        metadata(names, group_surplus=2.0, ratio=20), rng=np.random.default_rng(42),
    )
    assert not out.DefaultFlag.any()
    assert out.AdjustedSurplusUSD.eq(0).all()
    assert out.GroupContributionUSD.sum() == 0


def test_publication_figa_cleanup_preserves_one_cent():
    from scripts.earths_future_revision.publication_financial_cleanup import clean_figa_roundoff
    frame = pd.DataFrame({'figa_residual_deficit_usd': [1e-7, -1e-7, 0.009, 0.01, 1.0, np.nan]})
    clean_figa_roundoff(frame)
    assert frame.figa_residual_deficit_usd.iloc[:5].tolist() == [0, 0, 0, 0.01, 1]
    assert np.isnan(frame.figa_residual_deficit_usd.iloc[5])
