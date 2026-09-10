"""Regression test for the severity-bin merge edge case identified in the
independent FHCF-pilot review (item 6): when the lowest occupied
positive-loss bin has a bin *code* greater than zero (because bin code 0
itself is empty for this dataset) and that lowest-occupied bin is sparse,
the merge loop used to search for "a preceding bin" among bins with code
< 0, an empty set, raising ValueError. The archived ERA5 baseline never
triggers this (its bin 0 is always populated), so this could pass silently
until a differently-distributed dataset (e.g. a short pilot sample) hit it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "earths_future_revision"))
from section6_decomposition import assign_bins, BIN_EDGES_USD  # noqa: E402


def _fake_iterations(total_damage_usd, n_boot_cols=("fhcf_shortfall_usd",
                                                      "figa_residual_deficit_usd",
                                                      "citizens_residual_deficit_usd",
                                                      "nfip_borrowed_usd",
                                                      "public_burden_corrected_usd")):
    df = pd.DataFrame({"total_damage_usd": total_damage_usd})
    for c in n_boot_cols:
        df[c] = 0.0
    return df


def test_lowest_occupied_bin_above_index_zero_does_not_raise():
    """Construct a dataset where the first positive-loss bin (index 0,
    (0, 39.5e9]) has ZERO seasons, but the second bin (index 1) has a
    sparse but nonzero count. Before the fix, merging bin 1 (sparse) would
    search for a preceding present bin among {0}, find it empty (since bin
    0 has no seasons), and raise ValueError on max() of an empty sequence.
    """
    # 3 seasons in bin 1's range (39.5e9, 147e9], well under any reasonable
    # min_count, and NOTHING in bin 0's range (0, 39.5e9].
    total_damage = [0.0] * 50 + [80e9, 90e9, 100e9] + [200e9] * 60
    df = _fake_iterations(total_damage)

    # Sanity check on the fixture itself: bin 0's range is genuinely empty.
    lo0, hi0 = BIN_EDGES_USD[0], BIN_EDGES_USD[1]
    assert not ((df["total_damage_usd"] > lo0) & (df["total_damage_usd"] <= hi0)).any()

    bins = assign_bins(df, min_count=50)  # bin-1's 3 seasons are sparse (<50)
    assert bins.isna().sum() == 0
    # The 3 sparse bin-1 seasons must be merged into SOME positive-loss bin
    # (there is no lower positive-loss bin to receive them, so they join the
    # next occupied bin upward -- whichever the merge algorithm resolves to
    # -- rather than raising or being dropped).
    nonzero_mask = df["total_damage_usd"] > 0
    assert (bins[nonzero_mask] != "Zero loss").all()
    assert bins[~nonzero_mask].eq("Zero loss").all()


def test_normal_case_still_works_bin_zero_populated():
    """Regression guard: the common case (bin 0 populated) is unaffected."""
    total_damage = [0.0] * 10 + [10e9] * 200 + [500e9] * 200
    df = _fake_iterations(total_damage)
    bins = assign_bins(df, min_count=50)
    assert (bins[df["total_damage_usd"] == 0.0] == "Zero loss").all()
    assert (bins[df["total_damage_usd"] == 10e9] != "Zero loss").all()


def test_all_zero_loss_does_not_raise():
    df = _fake_iterations([0.0] * 20)
    bins = assign_bins(df, min_count=50)
    assert (bins == "Zero loss").all()
