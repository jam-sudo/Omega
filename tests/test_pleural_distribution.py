"""Tests for prediction/pleural_distribution.py (Phase 923)."""

import math

import pytest

from omega_pbpk.prediction.pleural_distribution import (
    PleuralDistributionResult,
    compare_pleural_conditions,
    predict_pleural_distribution,
)

# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    result = predict_pleural_distribution("TestDrug")
    assert isinstance(result, PleuralDistributionResult)


def test_default_drug_name_preserved():
    result = predict_pleural_distribution("Cisplatin")
    assert result.drug_name == "Cisplatin"


def test_default_route_systemic():
    result = predict_pleural_distribution("Drug")
    assert result.route == "systemic"


def test_default_condition_normal():
    result = predict_pleural_distribution("Drug")
    assert result.condition == "normal"


# ---------------------------------------------------------------------------
# Pleural volume
# ---------------------------------------------------------------------------


def test_normal_pleural_volume():
    result = predict_pleural_distribution("Drug", condition="normal")
    assert math.isclose(result.pleural_volume_L, 0.020, rel_tol=1e-9)


def test_malignant_effusion_pleural_volume():
    result = predict_pleural_distribution("Drug", condition="malignant_effusion")
    assert math.isclose(result.pleural_volume_L, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Systemic route logic
# ---------------------------------------------------------------------------


def test_systemic_ratio_formula():
    logp = 2.0
    plasma = 1.0
    result = predict_pleural_distribution("Drug", logp=logp, plasma_conc_mg_L=plasma)
    expected_ratio = 0.3 + logp * 0.1  # = 0.5
    assert math.isclose(result.plasma_to_pleural_ratio, expected_ratio, rel_tol=1e-9)


def test_systemic_conc_from_ratio():
    result = predict_pleural_distribution("Drug", logp=2.0, plasma_conc_mg_L=2.0)
    assert math.isclose(result.predicted_pleural_conc_mg_L, 0.5 * 2.0, rel_tol=1e-9)


def test_systemic_ratio_clamped_low():
    # logp = -10 → ratio = 0.3 + (-10)*0.1 = -0.7 → clamped to 0.05
    result = predict_pleural_distribution("Drug", logp=-10.0)
    assert result.plasma_to_pleural_ratio == pytest.approx(0.05)


def test_systemic_ratio_clamped_high():
    # logp = 20 → ratio = 0.3 + 20*0.1 = 2.3 → clamped to 2.0
    result = predict_pleural_distribution("Drug", logp=20.0)
    assert result.plasma_to_pleural_ratio == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Intrapleural route logic
# ---------------------------------------------------------------------------


def test_intrapleural_conc_normal():
    dose = 10.0
    result = predict_pleural_distribution(
        "Drug", dose_mg=dose, route="intrapleural", condition="normal"
    )
    expected = dose / 0.020
    assert math.isclose(result.predicted_pleural_conc_mg_L, expected, rel_tol=1e-9)


def test_intrapleural_conc_malignant():
    dose = 100.0
    result = predict_pleural_distribution(
        "Drug", dose_mg=dose, route="intrapleural", condition="malignant_effusion"
    )
    expected = dose / 1.0
    assert math.isclose(result.predicted_pleural_conc_mg_L, expected, rel_tol=1e-9)


def test_intrapleural_ratio_clamped_min():
    # ratio = conc / plasma_conc; ensure ≥ 1.0
    result = predict_pleural_distribution(
        "Drug",
        dose_mg=0.001,
        route="intrapleural",
        condition="normal",
        plasma_conc_mg_L=1000.0,
    )
    assert result.plasma_to_pleural_ratio >= 1.0


def test_intrapleural_ratio_clamped_max():
    result = predict_pleural_distribution(
        "Drug",
        dose_mg=10.0,
        route="intrapleural",
        condition="normal",
        plasma_conc_mg_L=0.001,
    )
    assert result.plasma_to_pleural_ratio <= 500.0


# ---------------------------------------------------------------------------
# Drug amount
# ---------------------------------------------------------------------------


def test_drug_amount_in_pleural():
    result = predict_pleural_distribution("Drug", dose_mg=10.0, logp=2.0, plasma_conc_mg_L=1.0)
    expected = result.predicted_pleural_conc_mg_L * result.pleural_volume_L
    assert math.isclose(result.drug_amount_in_pleural_mg, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Therapeutic flag
# ---------------------------------------------------------------------------


def test_therapeutic_achievable_true():
    # intrapleural into small normal volume gives high conc
    result = predict_pleural_distribution(
        "Drug", dose_mg=1.0, route="intrapleural", condition="normal"
    )
    assert result.therapeutic_local_conc_achievable is True


def test_therapeutic_achievable_false():
    # Very low systemic plasma conc
    result = predict_pleural_distribution("Drug", logp=0.0, plasma_conc_mg_L=0.001)
    assert result.therapeutic_local_conc_achievable is False


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_therapeutic():
    result = predict_pleural_distribution(
        "Drug", dose_mg=1.0, route="intrapleural", condition="normal"
    )
    assert "Therapeutic pleural concentrations achieved" in result.notes


def test_notes_malignant_low_conc():
    # systemic with low logp → low ratio → low conc < 1 mg/L → malignant note
    result = predict_pleural_distribution(
        "Drug",
        logp=-5.0,
        plasma_conc_mg_L=0.1,
        condition="malignant_effusion",
        route="systemic",
    )
    assert result.predicted_pleural_conc_mg_L < 1.0
    assert "Malignant effusion" in result.notes


def test_notes_limited_systemic():
    result = predict_pleural_distribution(
        "Drug", logp=0.0, plasma_conc_mg_L=0.001, condition="normal"
    )
    assert result.predicted_pleural_conc_mg_L < 1.0
    assert "Limited pleural penetration" in result.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_pleural_distribution("Drug", dose_mg=0.0)


def test_invalid_mw():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_pleural_distribution("Drug", mw_Da=-1.0)


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        predict_pleural_distribution("Drug", route="oral")


def test_invalid_condition():
    with pytest.raises(ValueError, match="condition"):
        predict_pleural_distribution("Drug", condition="chronic")


# ---------------------------------------------------------------------------
# compare_pleural_conditions
# ---------------------------------------------------------------------------


def test_compare_returns_two_results():
    results = compare_pleural_conditions("Drug", dose_mg=10.0, logp=2.0)
    assert len(results) == 2


def test_compare_conditions_covered():
    results = compare_pleural_conditions("Drug", dose_mg=10.0, logp=2.0)
    conditions = {r.condition for r in results}
    assert conditions == {"normal", "malignant_effusion"}


def test_compare_sorted_descending():
    results = compare_pleural_conditions("Drug", dose_mg=10.0, logp=2.0)
    assert results[0].predicted_pleural_conc_mg_L >= results[1].predicted_pleural_conc_mg_L
