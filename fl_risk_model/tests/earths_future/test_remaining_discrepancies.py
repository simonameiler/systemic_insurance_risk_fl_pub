"""Regression tests for the code-paper discrepancies fixed on
fix/remaining-code-paper-discrepancies (see docs/code_paper_fixes.md):
Citizens Tier-2 premium base, surplus coverage of loss-bearing insurers,
the penetration-scenario Citizens split, the USD 0.01 roundoff rule, and
zero-event season rows.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fl_risk_model import config as cfg
from fl_risk_model.capital import (
    apply_group_capital_contributions,
    apply_losses_to_surplus,
    reconcile_surplus_with_losses,
)
from fl_risk_model.mc_run_events import _fill_zero_event_rows
from fl_risk_model.runner import _private_premium_total, citizens_private_split


def _surplus(rows):
    df = pd.DataFrame(rows, columns=["Company", "StatEntityKey", "SurplusUSD"])
    df["GroupSurplusUSD"] = df["SurplusUSD"]
    df["GroupToEntityRatio"] = 1.0
    df["NAICGroupName"] = df["Company"]
    df["NAICGroupNumber"] = df["StatEntityKey"]
    return df


# --- Citizens Tier-2 premium base ---------------------------------------------

def test_private_premium_total_excludes_citizens():
    base = pd.DataFrame({
        "Company": ["A Ins Co", "Citizens Property Ins Corp", "B Ins Co"],
        "StatEntityKey": ["C1", cfg.CITIZENS_STATKEY, "C2"],
        "PremiumUSD": [100.0, 999.0, 50.0],
    })
    assert _private_premium_total(base) == pytest.approx(150.0)


def test_private_premium_total_missing_base_is_zero():
    assert _private_premium_total(None) == 0.0


# --- surplus coverage of loss-bearing insurers ----------------------------------

def test_reconcile_matches_by_key_and_takes_loss_side_name():
    surplus = _surplus([["Hartford Ins Co. the MW", "C888", 425.0]])
    losses = pd.DataFrame({"Company": ["Hartford Ins Co. of the MW"], "StatEntityKey": ["C888"],
                           "TotalLossUSD": [10.0]})
    out = reconcile_surplus_with_losses(surplus, losses, policy="error")
    assert out.loc[out["StatEntityKey"] == "C888", "Company"].item() == "Hartford Ins Co. of the MW"
    assert not out["SurplusImputed"].any()


def test_reconcile_imputes_missing_entity_with_median_ratio():
    surplus = _surplus([["A", "C1", 200.0], ["B", "C2", 600.0]])  # ratios 2 and 3 -> median 2.5
    premium = pd.DataFrame({"StatEntityKey": ["C1", "C2", "C3"], "PremiumUSD": [100.0, 200.0, 40.0]})
    losses = pd.DataFrame({"Company": ["A", "C"], "StatEntityKey": ["C1", "C3"], "TotalLossUSD": [5.0, 7.0]})
    with pytest.warns(UserWarning, match="Imputed surplus"):
        out = reconcile_surplus_with_losses(surplus, losses, premium, policy="impute_premium_ratio")
    row = out[out["Company"] == "C"].iloc[0]
    assert row["SurplusImputed"]
    assert row["SurplusUSD"] == pytest.approx(2.5 * 40.0)
    assert row["GroupToEntityRatio"] == 1.0
    assert row["NAICGroupNumber"] == "imputed:C3"


def test_reconcile_error_policy_lists_unmatched():
    surplus = _surplus([["A", "C1", 200.0]])
    losses = pd.DataFrame({"Company": ["Slide Insurance Co."], "StatEntityKey": ["C9834"], "TotalLossUSD": [1.0]})
    with pytest.raises(ValueError, match="Slide Insurance Co."):
        reconcile_surplus_with_losses(surplus, losses, policy="error")


def test_reconcile_legacy_drop_leaves_surplus_unchanged():
    surplus = _surplus([["A", "C1", 200.0]])
    losses = pd.DataFrame({"Company": ["Z"], "StatEntityKey": ["C9"], "TotalLossUSD": [1.0]})
    out = reconcile_surplus_with_losses(surplus, losses, policy="legacy_drop")
    assert list(out["Company"]) == ["A"]


def test_all_private_net_loss_reaches_capital_after_reconcile():
    surplus = _surplus([["A", "C1", 200.0], ["B", "C2", 600.0]])
    premium = pd.DataFrame({"StatEntityKey": ["C1", "C2", "C3"], "PremiumUSD": [100.0, 200.0, 40.0]})
    losses = pd.DataFrame({"Company": ["A", "B", "C"], "StatEntityKey": ["C1", "C2", "C3"],
                           "TotalLossUSD": [5.0, 0.0, 300.0]})
    with warnings.catch_warnings():  # warned once per entity set; may already have fired
        warnings.simplefilter("ignore")
        rec = reconcile_surplus_with_losses(surplus, losses, premium, policy="impute_premium_ratio")
    cap = apply_losses_to_surplus(rec[["Company", "StatEntityKey", "SurplusUSD"]], losses, sample=False)
    assert cap["TotalLossUSD"].sum() == pytest.approx(losses["TotalLossUSD"].sum())
    assert bool(cap.loc[cap["Company"] == "C", "RuinFlag"].item())  # 300 loss vs 100 imputed surplus


def test_fill_from_market_share_group_sources():
    from fl_risk_model.capital import fill_surplus_from_market_share
    extract = _surplus([["A", "C1", 200.0]]).assign(GroupSurplusUSD=[900.0], NAICGroupNumber=[77.0],
                                                    SurplusSource="surplus_extract", GroupSurplusSource="surplus_extract")
    ms = pd.DataFrame({
        "StatEntityKey": ["C1", "C2", "C3", "C4", cfg.CITIZENS_STATKEY],
        "Company": ["A", "B", "C", "D", "Citizens Property Ins Corp."],
        "SurplusUSD": [200.0, 50.0, 40.0, 30.0, float("nan")],
        "SNLGroup": ["G-A", "G-A", "", "G-D", ""],
        "SNLGroupSurplusUSD": [950.0, 950.0, float("nan"), 120.0, float("nan")],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fill_surplus_from_market_share(extract, ms).set_index("StatEntityKey")
    assert cfg.CITIZENS_STATKEY not in out.index
    assert out.loc["C2", "GroupSurplusSource"] == "surplus_extract_sibling"
    assert out.loc["C2", "NAICGroupNumber"] == 77.0 and out.loc["C2", "GroupSurplusUSD"] == 900.0
    assert out.loc["C3", "GroupSurplusSource"] == "standalone" and out.loc["C3", "GroupToEntityRatio"] == 1.0
    assert out.loc["C4", "GroupSurplusSource"] == "market_share_group_sheet"
    assert out.loc["C4", "GroupSurplusUSD"] == 120.0 and out.loc["C4", "GroupToEntityRatio"] == pytest.approx(4.0)


_LICENSED = pytest.mark.skipif(not (Path(cfg.MARKET_SHARE_XLSX).exists() and Path(cfg.SURPLUS_FILE).exists()),
                               reason="licensed S&P inputs not present")


@_LICENSED
def test_market_share_capital_equals_extract_surplus():
    from fl_risk_model.capital import load_market_share_capital, load_surplus_data_with_groups
    ms = load_market_share_capital(cfg.MARKET_SHARE_XLSX).set_index("StatEntityKey")["SurplusUSD"]
    ex = load_surplus_data_with_groups(cfg.SURPLUS_FILE, year=2024, fill_from_market_share=False)
    ex = ex.set_index("StatEntityKey")["SurplusUSD"].drop(cfg.CITIZENS_STATKEY, errors="ignore")
    both = ms.dropna().index.intersection(ex.index)
    assert len(both) >= 70
    rel = (ms[both] - ex[both]).abs() / ex[both].abs()
    assert rel.max() < 1e-3


@_LICENSED
def test_surplus_covers_every_modeled_private_writer():
    from fl_risk_model.capital import load_surplus_data_with_groups
    from fl_risk_model.loader import load_market_share
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = load_surplus_data_with_groups(cfg.SURPLUS_FILE, year=2024)
    roster = set(load_market_share(cfg.MARKET_SHARE_XLSX, include_key=True)["StatEntityKey"].astype(str))
    assert roster - {cfg.CITIZENS_STATKEY} <= set(sp["StatEntityKey"])


# --- penetration-scenario Citizens split ----------------------------------------

def test_scaled_portfolio_split_uses_scenario_totals():
    fhcf = pd.DataFrame({"County": ["X"], "CountyTIV": [100.0]})
    # After depopulation to 8% and a 2.25x exposure scale-up: Citizens 18, private 207
    cit = pd.DataFrame({"County": ["X"], "TIV": [18.0]})
    priv = pd.DataFrame({"County": ["X"], "TIV": [207.0]})
    legacy = citizens_private_split(fhcf, cit, priv, scaled_portfolio=False)
    fixed = citizens_private_split(fhcf, cit, priv, scaled_portfolio=True)
    assert legacy["CitizensFrac"].item() == pytest.approx(0.18)  # the inflated share
    assert fixed["CitizensFrac"].item() == pytest.approx(0.08)
    assert fixed["CitizensFrac"].item() + fixed["PrivateFrac"].item() == pytest.approx(1.0)


def test_baseline_split_unchanged():
    fhcf = pd.DataFrame({"County": ["X"], "CountyTIV": [100.0]})
    cit = pd.DataFrame({"County": ["X"], "TIV": [15.0]})
    priv = pd.DataFrame({"County": ["X"], "TIV": [85.0]})
    assert citizens_private_split(fhcf, cit, priv)["CitizensFrac"].item() == pytest.approx(0.15)


# --- USD 0.01 roundoff rule (Text S4) -------------------------------------------

def test_default_flag_roundoff_rule():
    capital_post = pd.DataFrame({
        "Company": ["roundoff", "one_cent_deficit", "solvent"],
        "StatEntityKey": ["C1", "C2", "C3"],
        "SurplusUSD": [1.0, 1.0, 1.0],
        "EndingSurplusUSD": [-0.004, -0.01, 5.0],
    })
    surplus = _surplus([["roundoff", "C1", 1.0], ["one_cent_deficit", "C2", 1.0], ["solvent", "C3", 1.0]])
    out = apply_group_capital_contributions(capital_post, surplus, rng=np.random.default_rng(0))
    flags = out.set_index("Company")["DefaultFlag"]
    assert not flags["roundoff"]
    assert flags["one_cent_deficit"]
    assert not flags["solvent"]


# --- Citizens starting surplus ------------------------------------------------------

def test_citizens_surplus_parses_apostrophe_separated_dollars(tmp_path):
    from fl_risk_model.capital import load_citizens_capital_row_from_csv
    csv = tmp_path / "cap.csv"
    csv.write_text(
        "year,capital,capital_from_budget,pml10\n"
        "2023, $5'023'466'880.00 , $4'901'300'000 , $2'378'200'000 \n"
        "2024, $4'183'356'808.00 , $6'284'900'000 , $2'436'400'000 \n"
    )
    row = load_citizens_capital_row_from_csv(path=str(csv), year=2024, mode="FIXED_YEAR")
    assert row["projected_year_end_surplus_usd"] == pytest.approx(4_183_356_808.0)


def test_citizens_surplus_from_repo_file_is_positive():
    from fl_risk_model.capital import load_citizens_capital_row_from_csv
    row = load_citizens_capital_row_from_csv(year=2024, mode="FIXED_YEAR")
    assert row["projected_year_end_surplus_usd"] == pytest.approx(4_183_356_808.0)


# --- zero-event season rows -----------------------------------------------------

def test_zero_event_rows_filled_from_computed_zero_loss_season():
    df = pd.DataFrame([
        {"year_id": 1, "scenario": "year_1", "events": "e1", "total_damage_usd": 0.0,
         "figa_residual_deficit_usd": 0.0, "citizens_tier1_capacity_usd": 5.4e8,
         "fhcf_scaling_factor": 1.0, "catbond_limit_in_force_usd": 2.5e9},
        {"year_id": 2, "scenario": "zero_events", "events": "", "total_damage_usd": 0.0,
         "figa_residual_deficit_usd": 0.0},
        {"year_id": 3, "scenario": "year_3", "events": "e2", "total_damage_usd": 1e9,
         "figa_residual_deficit_usd": 1e8, "citizens_tier1_capacity_usd": 5.4e8,
         "fhcf_scaling_factor": 1.0, "catbond_limit_in_force_usd": 2.5e9},
    ])
    out = _fill_zero_event_rows(df).set_index("year_id")
    assert out.loc[2, "citizens_tier1_capacity_usd"] == 5.4e8
    assert out.loc[2, "fhcf_scaling_factor"] == 1.0
    assert out.loc[2, "catbond_limit_in_force_usd"] == 2.5e9
    assert out.loc[2, "events"] == ""
    assert out.loc[2, "scenario"] == "zero_events"
