"""Regression tests locking in the Section 9 / register item C5 fix:
reported total_damage_usd must reflect any prescribed physical loss
reduction (building codes), and must be unaffected for scenarios that only
change coverage/market structure (market exit, penetration).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "earths_future_revision"))
from common import load_iterations  # noqa: E402
from section9_climate_policy_tables import (  # noqa: E402
    add_corrected_total,
    SCENARIO_DIRS,
    MC_ROOT,
)


def _load(label):
    path = MC_ROOT / SCENARIO_DIRS[label] / "iterations.csv"
    if not path.exists():
        pytest.skip(f"archived run not present in this checkout: {path}")
    return add_corrected_total(load_iterations(path))


def test_unaffected_scenarios_reconcile_exactly():
    for label in ["Baseline (ERA5)", "Market Exit", "Insurance Penetration"]:
        df = _load(label)
        diff = (df["total_damage_usd"] - df["_corrected_total_damage_usd"]).abs()
        assert diff.max() < 1.0, f"{label}: reconstruction should equal reported total exactly"


def test_building_codes_reported_total_is_stale():
    """Documents the confirmed bug: the archived building-codes run's
    reported total_damage_usd equals the UNREDUCED baseline total, while the
    reconstructed corrected total reflects the prescribed reduction."""
    bc = _load("Building Codes")
    baseline = _load("Baseline (ERA5)")

    reported_bc_mean = bc["total_damage_usd"].mean()
    corrected_bc_mean = bc["_corrected_total_damage_usd"].mean()
    baseline_mean = baseline["total_damage_usd"].mean()

    # Bug signature: reported total for building codes matches baseline
    # (no reduction applied to the *reported* total)...
    assert reported_bc_mean == pytest.approx(baseline_mean, rel=1e-9)
    # ...while the corrected total is genuinely lower (real reduction).
    assert corrected_bc_mean < 0.8 * baseline_mean


def test_building_codes_uninsured_residual_direction_reverses_once_corrected():
    """The submitted SI Table S4 reports un/underinsured wind INCREASING
    under building codes (9.7B -> 11.7B), reconstructed as
    wind_total_usd(stale) - insured. Using the correctly-computed
    wind_uninsured_usd + wind_underinsured_usd columns, the true direction
    is a DECREASE, consistent with a genuine 30% wind loss reduction."""
    bc = _load("Building Codes")
    baseline = _load("Baseline (ERA5)")

    bc_correct_residual = (bc["wind_uninsured_usd"] + bc["wind_underinsured_usd"]).mean()
    baseline_correct_residual = (baseline["wind_uninsured_usd"] + baseline["wind_underinsured_usd"]).mean()
    assert bc_correct_residual < baseline_correct_residual

    bc_stale_residual = (bc["total_damage_usd"] - bc["wind_insured_private_usd"]
                         - bc["wind_insured_citizens_usd"]).mean()
    baseline_stale_residual = (baseline["total_damage_usd"] - baseline["wind_insured_private_usd"]
                               - baseline["wind_insured_citizens_usd"]).mean()
    assert bc_stale_residual > baseline_stale_residual  # reproduces the submitted (wrong) direction
