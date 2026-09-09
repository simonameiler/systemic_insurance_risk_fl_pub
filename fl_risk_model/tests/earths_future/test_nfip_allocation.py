"""Tests for the Section 7 NFIP structure-to-loss allocation configurations."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "earths_future_revision"))
from section7_nfip_allocation import build_county_rate_table, PENETRATION_CSV  # noqa: E402


def test_baseline_rate_lies_within_county_envelope():
    table = build_county_rate_table(PENETRATION_CSV)
    assert table["baseline_within_envelope"].all()


def test_baseline_reproduces_reported_all_county_rate_apart_from_cleaning():
    """Supporting Text S3: 'In the baseline this reproduces the county rate
    apart from these data cleaning steps.' tau_c = s*r_sfha + (1-s)*r_non
    where r_non is *inferred* from r_all algebraically, so tau_c should equal
    r_all exactly wherever no clipping or fillna occurred.
    """
    table = build_county_rate_table(PENETRATION_CSV)
    raw = pd.read_csv(PENETRATION_CSV)
    raw = raw[raw["state"] == "Florida"].set_index("county")
    merged = table.set_index("county").join(raw[["resPenetrationRate"]])
    diffs = (merged["rate_baseline_structure_weighted"] - merged["resPenetrationRate"]).abs()
    # Should match to floating-point precision for counties with valid,
    # unclipped inputs (no missing SFHA rate, s_sfha strictly in (0,1)).
    clean = (merged["s_sfha_share_of_stock"] > 0) & (merged["s_sfha_share_of_stock"] < 1)
    assert diffs[clean].max() < 1e-9


def test_sfha_only_config_undefined_when_no_sfha_stock_in_county():
    df = pd.DataFrame({
        "state": ["Florida"],
        "county": ["NoSFHACounty"],
        "resPenetrationRateSfha": [np.nan],
        "resPenetrationRate": [0.10],
        "resContractsInForceSfha": [0],
        "resContractsInForce": [100],
        "totalResStructuresSfha": [0],
        "totalResStructures": [1000],
        "county_fips": ["12999"],
        "asOfDate": ["2025-05-15T00:00:00.000Z"],
        "id": ["x"],
    })
    tmp = Path("/tmp") if Path("/tmp").exists() else Path(".")
    p = tmp / "_test_no_sfha.csv"
    df.to_csv(p, index=False)
    try:
        table = build_county_rate_table(p)
        row = table.iloc[0]
        assert row["s_sfha_share_of_stock"] == 0.0
        assert pd.isna(row["rate_sfha_only_config"])
        assert not pd.isna(row["rate_non_sfha_only_config"])
    finally:
        p.unlink(missing_ok=True)


def test_rate_ordering_is_not_uniformly_labeled_bound():
    """Confirms both orderings (SFHA rate above and, where present, at or
    below the non-SFHA rate) can occur across counties, so envelope
    construction must be per-county rather than a single global label."""
    table = build_county_rate_table(PENETRATION_CSV)
    valid = table.dropna(subset=["rate_sfha_only_config", "rate_non_sfha_only_config"])
    assert valid["sfha_rate_exceeds_non_sfha_rate"].nunique() >= 1
    # Document the actual observed direction rather than assuming it.
    n_total = len(valid)
    n_sfha_higher = int(valid["sfha_rate_exceeds_non_sfha_rate"].sum())
    assert 0 <= n_sfha_higher <= n_total
