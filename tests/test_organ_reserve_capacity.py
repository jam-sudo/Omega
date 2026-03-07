"""Tests for organ reserve capacity and safety margin assessment."""

import pytest

from omega_pbpk.risk.organ_reserve_capacity import (
    OrganReserveResult,
    assess_hepatic_reserve,
    assess_renal_reserve,
    estimate_organ_toxicity_dose,
    screen_organ_safety,
)

# ---------------------------------------------------------------------------
# OrganReserveResult is a dataclass
# ---------------------------------------------------------------------------


def test_organ_reserve_result_is_dataclass():
    result = assess_hepatic_reserve(40.0, 5.0)
    assert isinstance(result, OrganReserveResult)


# ---------------------------------------------------------------------------
# assess_hepatic_reserve
# ---------------------------------------------------------------------------


def test_hepatic_reserve_organ_name():
    result = assess_hepatic_reserve(40.0, 10.0)
    assert result.organ == "hepatic"


def test_hepatic_reserve_low_auc_high_safety_margin():
    # AUC = 5, threshold = 50 → margin = 10x
    result = assess_hepatic_reserve(40.0, 5.0, hepatotoxicity_threshold_mg_h_per_L=50.0)
    assert result.safety_margin_fold > 5.0


def test_hepatic_reserve_high_auc_exposure_fraction_clamped():
    # AUC > threshold → ef should be clamped at 1.0
    result = assess_hepatic_reserve(40.0, 100.0, hepatotoxicity_threshold_mg_h_per_L=50.0)
    assert result.exposure_fraction == pytest.approx(1.0)


def test_hepatic_reserve_high_auc_reserve_zero():
    result = assess_hepatic_reserve(40.0, 100.0, hepatotoxicity_threshold_mg_h_per_L=50.0)
    assert result.reserve_remaining == pytest.approx(0.0)


def test_hepatic_reserve_adequate_risk():
    # Low AUC → large reserve remaining → "adequate"
    result = assess_hepatic_reserve(40.0, 1.0, hepatotoxicity_threshold_mg_h_per_L=50.0)
    assert result.risk_level == "adequate"
    assert result.reserve_remaining > 0.5


def test_hepatic_reserve_critically_low_risk():
    # Very high AUC → reserve consumed → "critically_low"
    result = assess_hepatic_reserve(40.0, 60.0, hepatotoxicity_threshold_mg_h_per_L=50.0)
    assert result.risk_level == "critically_low"
    assert result.reserve_remaining < 0.2


def test_hepatic_reserve_reduced_risk():
    # ef ≈ 0.6 → reserve_remaining = 0.7 - 0.6*0.7 = 0.28 → "reduced"
    result = assess_hepatic_reserve(40.0, 30.0, hepatotoxicity_threshold_mg_h_per_L=50.0)
    assert result.risk_level == "reduced"
    assert 0.2 <= result.reserve_remaining <= 0.5


def test_hepatic_reserve_safety_margin_calculation():
    result = assess_hepatic_reserve(40.0, 10.0, hepatotoxicity_threshold_mg_h_per_L=50.0)
    assert result.safety_margin_fold == pytest.approx(5.0)


def test_hepatic_reserve_notes_nonempty():
    result = assess_hepatic_reserve(40.0, 10.0)
    assert len(result.notes) > 0


def test_hepatic_reserve_exposure_fraction_value():
    result = assess_hepatic_reserve(40.0, 25.0, hepatotoxicity_threshold_mg_h_per_L=50.0)
    assert result.exposure_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# assess_renal_reserve
# ---------------------------------------------------------------------------


def test_renal_reserve_organ_name():
    result = assess_renal_reserve(100.0, 5.0)
    assert result.organ == "renal"


def test_renal_reserve_full_egfr():
    # eGFR = 100 = baseline → renal_reserve = 1.0; low AUC → high reserve remaining
    result = assess_renal_reserve(100.0, 1.0, nephrotoxicity_threshold=30.0, baseline_egfr=100.0)
    assert result.reserve_remaining > 0.5


def test_renal_reserve_reduced_egfr():
    # eGFR = 30 → renal_reserve = 0.3; low exposure → reserve_remaining ≈ 0.3 → "reduced"
    result = assess_renal_reserve(30.0, 1.0, nephrotoxicity_threshold=30.0, baseline_egfr=100.0)
    assert result.reserve_remaining == pytest.approx(0.3 * (1.0 - 1.0 / 30.0), rel=1e-3)


def test_renal_reserve_safety_margin():
    result = assess_renal_reserve(100.0, 10.0, nephrotoxicity_threshold=30.0)
    assert result.safety_margin_fold == pytest.approx(3.0)


def test_renal_reserve_notes_nonempty():
    result = assess_renal_reserve(100.0, 5.0)
    assert len(result.notes) > 0


def test_renal_reserve_high_auc_clamped_ef():
    result = assess_renal_reserve(100.0, 100.0, nephrotoxicity_threshold=30.0)
    assert result.exposure_fraction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# screen_organ_safety
# ---------------------------------------------------------------------------


def test_screen_organ_safety_returns_list():
    compounds = [
        {"drug_auc": 5.0},
        {"drug_auc": 40.0},
        {"drug_auc": 20.0},
    ]
    results = screen_organ_safety(compounds, organ="hepatic")
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_organ_safety_sorted_ascending():
    compounds = [
        {"drug_auc": 5.0},  # margin = 10x
        {"drug_auc": 40.0},  # margin = 1.25x
        {"drug_auc": 20.0},  # margin = 2.5x
    ]
    results = screen_organ_safety(compounds, organ="hepatic")
    margins = [r.safety_margin_fold for r in results]
    assert margins == sorted(margins)


def test_screen_organ_safety_renal():
    compounds = [
        {"drug_auc": 5.0, "egfr": 100.0},
        {"drug_auc": 25.0, "egfr": 100.0},
    ]
    results = screen_organ_safety(compounds, organ="renal")
    assert results[0].safety_margin_fold <= results[1].safety_margin_fold


# ---------------------------------------------------------------------------
# estimate_organ_toxicity_dose
# ---------------------------------------------------------------------------


def test_estimate_organ_toxicity_dose_basic():
    # dose = AUC_threshold × CL
    dose = estimate_organ_toxicity_dose(cl_L_per_h=10.0, threshold_auc=50.0)
    assert dose == pytest.approx(500.0)


def test_estimate_organ_toxicity_dose_zero_cl():
    dose = estimate_organ_toxicity_dose(cl_L_per_h=0.0, threshold_auc=50.0)
    assert dose == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_hepatic_reserve_negative_auc_raises():
    with pytest.raises(ValueError):
        assess_hepatic_reserve(40.0, -1.0)


def test_hepatic_reserve_zero_threshold_raises():
    with pytest.raises(ValueError):
        assess_hepatic_reserve(40.0, 10.0, hepatotoxicity_threshold_mg_h_per_L=0.0)


def test_hepatic_reserve_negative_threshold_raises():
    with pytest.raises(ValueError):
        assess_hepatic_reserve(40.0, 10.0, hepatotoxicity_threshold_mg_h_per_L=-5.0)


def test_renal_reserve_negative_auc_raises():
    with pytest.raises(ValueError):
        assess_renal_reserve(100.0, -1.0)


def test_renal_reserve_zero_threshold_raises():
    with pytest.raises(ValueError):
        assess_renal_reserve(100.0, 10.0, nephrotoxicity_threshold=0.0)
