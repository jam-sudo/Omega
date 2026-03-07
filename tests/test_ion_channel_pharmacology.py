"""Tests for ion channel pharmacology prediction (Phase 823)."""

import pytest

from omega_pbpk.prediction.ion_channel_pharmacology import (
    IonChannelResult,
    predict_channel_block,
    screen_channels,
)

# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


def test_predict_channel_block_return_type():
    result = predict_channel_block("DrugA", "hERG", 100.0, 1000.0, 1.0, 10000.0)
    assert isinstance(result, IonChannelResult)


def test_result_fields_populated():
    result = predict_channel_block("DrugA", "hERG", 100.0, 1000.0, 1.0, 10000.0)
    assert result.drug_name == "DrugA"
    assert result.channel_name == "hERG"
    assert result.ic50_nM == 1000.0
    assert result.drug_conc_nM == 100.0
    assert result.hill_coef == 1.0


# ---------------------------------------------------------------------------
# Hill equation correctness
# ---------------------------------------------------------------------------


def test_block_pct_in_range():
    result = predict_channel_block("DrugA", "hERG", 500.0, 1000.0, 1.0, 10000.0)
    assert 0.0 <= result.block_pct <= 100.0


def test_at_ic50_block_is_50_percent():
    result = predict_channel_block("DrugA", "hERG", 1000.0, 1000.0, 1.0, 10000.0)
    assert abs(result.block_pct - 50.0) < 0.1


def test_high_concentration_approaches_100():
    """At C >> IC50, block should be close to 100%."""
    result = predict_channel_block("DrugA", "hERG", 1e8, 1000.0, 1.0, 10000.0)
    assert result.block_pct > 99.0


def test_low_concentration_approaches_0():
    """At C << IC50, block should be close to 0%."""
    result = predict_channel_block("DrugA", "hERG", 0.001, 1000.0, 1.0, 10000.0)
    assert result.block_pct < 0.01


def test_zero_concentration_gives_zero_block():
    result = predict_channel_block("DrugA", "hERG", 0.0, 1000.0, 1.0, 10000.0)
    assert result.block_pct == 0.0


def test_hill_coef_affects_steepness():
    """Higher Hill coefficient should give sharper transition."""
    r1 = predict_channel_block("DrugA", "hERG", 200.0, 1000.0, 1.0, 10000.0)
    r2 = predict_channel_block("DrugA", "hERG", 200.0, 1000.0, 3.0, 10000.0)
    # At C < IC50, higher hill means less block
    assert r2.block_pct < r1.block_pct


# ---------------------------------------------------------------------------
# Therapeutic window margin
# ---------------------------------------------------------------------------


def test_therapeutic_window_margin_calculation():
    result = predict_channel_block("DrugA", "hERG", 100.0, 500.0, 1.0, 5000.0)
    assert abs(result.therapeutic_window_margin - 10.0) < 1e-9


def test_therapeutic_window_margin_below_one():
    """When therapeutic_ic50 < ion channel IC50, margin < 1 (overlap risk)."""
    result = predict_channel_block("DrugA", "hERG", 100.0, 1000.0, 1.0, 500.0)
    assert result.therapeutic_window_margin < 1.0


# ---------------------------------------------------------------------------
# Cardiac safety risk classification
# ---------------------------------------------------------------------------


def test_herg_high_block_critical_cardiac_risk():
    """hERG block > 50% should give critical cardiac risk."""
    result = predict_channel_block("DrugA", "hERG", 10000.0, 1000.0, 1.0, 100000.0)
    assert result.block_pct > 50.0
    assert result.cardiac_safety_risk == "critical"


def test_herg_low_block_low_cardiac_risk():
    """hERG block < 10% should give low cardiac risk."""
    result = predict_channel_block("DrugA", "hERG", 1.0, 1000.0, 1.0, 10000.0)
    assert result.cardiac_safety_risk == "low"


def test_herg_cns_risk_is_low():
    result = predict_channel_block("DrugA", "hERG", 100.0, 1000.0, 1.0, 10000.0)
    assert result.cns_risk == "low"


def test_nav1_1_cardiac_risk_is_low():
    result = predict_channel_block("DrugA", "Nav1.1", 10000.0, 1000.0, 1.0, 100000.0)
    assert result.cardiac_safety_risk == "low"


def test_nav1_1_high_block_high_cns_risk():
    """Nav1.1 block > 50% should give high CNS risk."""
    result = predict_channel_block("DrugA", "Nav1.1", 10000.0, 1000.0, 1.0, 100000.0)
    assert result.block_pct > 50.0
    assert result.cns_risk == "high"


# ---------------------------------------------------------------------------
# Use dependence and state dependence
# ---------------------------------------------------------------------------


def test_use_dependence_valid_values():
    valid = {"high", "moderate", "low"}
    result = predict_channel_block("DrugA", "hERG", 500.0, 1000.0, 1.0, 10000.0)
    assert result.use_dependence in valid


def test_use_dependence_high_at_very_high_block():
    """Block > 70% should give high use dependence."""
    result = predict_channel_block("DrugA", "hERG", 50000.0, 1000.0, 1.0, 100000.0)
    assert result.block_pct > 70.0
    assert result.use_dependence == "high"


def test_state_dependent_low_ic50():
    """IC50 < 100 nM should result in state_dependent=True."""
    result = predict_channel_block("DrugA", "hERG", 50.0, 50.0, 1.0, 500.0)
    assert result.state_dependent is True


def test_state_dependent_high_ic50():
    """IC50 >= 100 nM should result in state_dependent=False."""
    result = predict_channel_block("DrugA", "hERG", 50.0, 500.0, 1.0, 5000.0)
    assert result.state_dependent is False


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_ic50_raises():
    with pytest.raises(ValueError, match="ic50_nM"):
        predict_channel_block("DrugA", "hERG", 100.0, 0.0, 1.0, 10000.0)


def test_negative_ic50_raises():
    with pytest.raises(ValueError, match="ic50_nM"):
        predict_channel_block("DrugA", "hERG", 100.0, -50.0, 1.0, 10000.0)


def test_negative_drug_conc_raises():
    with pytest.raises(ValueError, match="drug_conc_nM"):
        predict_channel_block("DrugA", "hERG", -1.0, 1000.0, 1.0, 10000.0)


def test_invalid_therapeutic_ic50_raises():
    with pytest.raises(ValueError, match="therapeutic_ic50_nM"):
        predict_channel_block("DrugA", "hERG", 100.0, 1000.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Screen channels
# ---------------------------------------------------------------------------


SAMPLE_CHANNELS = [
    {"name": "hERG", "ic50_nM": 100.0, "hill_coef": 1.0, "therapeutic_ic50_nM": 10000.0},
    {"name": "Nav1.5", "ic50_nM": 5000.0, "hill_coef": 1.0, "therapeutic_ic50_nM": 10000.0},
    {"name": "Cav1.2", "ic50_nM": 1000.0, "hill_coef": 1.0, "therapeutic_ic50_nM": 10000.0},
]


def test_screen_channels_return_list():
    results = screen_channels("DrugA", 500.0, SAMPLE_CHANNELS)
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(r, IonChannelResult) for r in results)


def test_screen_channels_sorted_by_block_pct_descending():
    """Results should be ordered from most to least blocked channel."""
    results = screen_channels("DrugA", 500.0, SAMPLE_CHANNELS)
    block_pcts = [r.block_pct for r in results]
    assert block_pcts == sorted(block_pcts, reverse=True)


def test_screen_channels_most_blocked_is_lowest_ic50():
    """Channel with lowest IC50 (hERG at 100 nM) should be most blocked at 500 nM."""
    results = screen_channels("DrugA", 500.0, SAMPLE_CHANNELS)
    assert results[0].channel_name == "hERG"
