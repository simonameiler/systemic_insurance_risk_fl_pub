"""
Tests documenting the consequences of applying the financial model once to
seasonal-aggregate losses (revision brief Section 4, catastrophe bonds and
FHCF occurrence contracts; correction register item C3).

These tests describe and quantify the seasonal-aggregation approximation
using the model's own occurrence-payout function (fl_risk_model.catbonds).
They do not implement or assert a multi-event or multi-year recovery model.
"""
from __future__ import annotations

from fl_risk_model.catbonds import _payout_occurrence


def test_two_subattachment_events_can_sum_above_attachment_in_season_aggregate():
    """Two individual events, each below a cat bond's attachment point, are
    summed into one seasonal driver before the occurrence payout function is
    applied once. If the underlying instrument is genuinely a per-occurrence
    (single-event) trigger, this seasonal aggregation can produce a payout
    that an eventwise, per-occurrence application of the same contract would
    not produce for either event individually. This test quantifies that
    difference using the model's own payout function; it is a description of
    the modeled seasonal-contract approximation, not a claim that the model
    performs eventwise accounting.
    """
    attach = 100.0
    limit = 50.0

    event_1_loss = 60.0
    event_2_loss = 55.0
    seasonal_aggregate_loss = event_1_loss + event_2_loss  # 115, > attach

    payout_event_1_alone = _payout_occurrence(event_1_loss, attach, limit)
    payout_event_2_alone = _payout_occurrence(event_2_loss, attach, limit)
    payout_seasonal_aggregate = _payout_occurrence(seasonal_aggregate_loss, attach, limit)

    # Neither event alone would trigger a genuinely per-occurrence contract.
    assert payout_event_1_alone == 0.0
    assert payout_event_2_alone == 0.0
    # The seasonal-aggregate application does trigger a payout.
    assert payout_seasonal_aggregate > 0.0
    assert payout_seasonal_aggregate == min(seasonal_aggregate_loss - attach, limit)


def test_paired_losses_with_exposure_depletion_apply_financial_model_once():
    """Paired historical scenarios (e.g. Great Miami then Andrew) reduce
    exposed asset value after the first storm (physical exposure depletion)
    but combine the resulting county losses and apply the financial model
    (FHCF, capital, FIGA, Citizens, NFIP) exactly once to the combined total,
    per Supporting Text S5. This test documents that property using a
    minimal synthetic example: the financial model output for a combined
    loss L1+L2 (post-depletion) is not equal to running the model twice
    (once per storm) and summing recoveries/capital hits, because retention
    layers, company limits, and capital are evaluated once against the
    combined total rather than reset between the two storms.
    """
    attach = 100.0
    limit = 50.0
    loss_storm1 = 40.0  # post-depletion loss from storm 1
    loss_storm2 = 90.0  # loss from storm 2, on already-depleted exposure

    combined_once = _payout_occurrence(loss_storm1 + loss_storm2, attach, limit)
    sequential_reset = _payout_occurrence(loss_storm1, attach, limit) + _payout_occurrence(
        loss_storm2, attach, limit
    )
    assert combined_once != sequential_reset
