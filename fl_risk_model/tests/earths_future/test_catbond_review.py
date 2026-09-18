import numpy as np
import pandas as pd
import pytest
from fl_risk_model import catbonds as cb, config as cfg


def fixture_inputs():
    claims = pd.DataFrame({'Company': ['A', 'B'], 'County': ['X', 'X'], 'NetWindUSD': [150., 300.]})
    ms = pd.DataFrame({'Company': ['A', 'B'], 'StatEntityKey': ['K1', 'K2']})
    keys = pd.DataFrame({'Company_MS': ['A', 'B'], 'Company_FHCF': ['A', 'B'],
                         'StatEntityKey': ['K1', 'K2'], 'NAIC': ['1', '2'], 'fhcf_participant': [True, True]})
    return claims, ms, keys


def payout(bonds, driver=1000.):
    claims, ms, keys = fixture_inputs()
    return cb.apply_catbond_recovery(claims, claims.iloc[:0], pd.DataFrame(bonds), ms, keys,
                                    industry_insured_wind_pre_fhcf_usd=driver)


def bond(trigger='indemnity', key='K1'):
    return {'BondID': 'test', 'Cedent_Sponsor': 'A', 'TriggerClass': trigger,
            'BeneficiaryStatEntityKeys': key, 'AttachmentUSD': 100., 'LimitUSD': 100.}


@pytest.mark.parametrize('driver,expected', [(0, 0), (99, 0), (100, 0), (140, 40), (200, 100), (500, 100)])
def test_layer_boundaries(driver, expected):
    assert cb._payout_occurrence(driver, 100, 100) == expected


def test_indemnity_pays_only_beneficiary_above_attachment():
    r, d = payout([bond()])
    assert list(r.Company) == ['A']
    assert d['catbond_payout_total'] == 50


def test_industry_trigger_does_not_make_every_insurer_a_beneficiary():
    r, d = payout([bond('industry')])
    assert list(r.Company) == ['A']
    assert d['catbond_payout_total'] == 100


def test_multiple_layers_cannot_overcredit_claims():
    r, d = payout([bond('industry'), dict(bond('industry'), BondID='second')])
    assert d['catbond_triggered_payout_total'] == 200
    assert d['catbond_payout_total'] == 150
    assert r.CatBondRecoveryUSD.sum() == 150


def test_unknown_beneficiary_fails():
    with pytest.raises(ValueError):
        payout([bond(key='unknown')])


def test_citizens_legal_name_resolves_when_absent_from_market_share_names():
    claims, ms, keys = fixture_inputs()
    claims.loc[0, 'Company'] = 'Citizens Legal Name'
    keys.loc[0, 'Company_FHCF'] = 'Citizens Legal Name'
    r, d = cb.apply_catbond_recovery(claims.iloc[1:], claims.iloc[:1], pd.DataFrame([bond()]), ms, keys)
    assert list(r.Company) == ['Citizens Legal Name']
    assert d['catbond_payout_total'] == 50


@pytest.mark.parametrize('date,expected', [('Jun.24', True), ('2024-06', True), ('Dec.24', False), ('2024-12', False), ('bad date', False)])
def test_snapshot_date(date, expected):
    assert cb._in_force_for_season(date) == expected


def test_state_list_excluding_florida_is_not_nationwide_cover():
    assert not cb._is_fl_relevant('US named storm: Alabama, Louisiana, Texas')
    assert not cb._is_fl_relevant('North Carolina named storm')
    assert cb._is_fl_relevant('US named storm')
    assert cb._is_fl_relevant('US named storm: Alabama, Florida, Texas')


def test_reviewed_inventory():
    df = cb.load_catbond_table(cfg.DATA_DIR / 'catbonds_2024_reviewed.csv')
    assert len(df) == 8
    assert df.LimitUSD.sum() == 2540000000
    assert df.TriggerClass.eq('indemnity').all()
    assert df.BeneficiaryStatEntityKeys.str.len().gt(0).all()
    assert np.all(df.AttachmentUSD == df.LimitUSD)


def test_unreviewed_inventory_rejected():
    with pytest.raises(ValueError):
        cb.load_catbond_table(cfg.DATA_DIR / 'catbonds_2024.csv')
