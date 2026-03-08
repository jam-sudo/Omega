"""
Tests for Phase 446 — Solution Viscosity Prediction for Liquid Formulations
"""

import pytest

from omega_pbpk.prediction.viscosity_prediction_p446 import (
    EXCIPIENTS,
    ViscosityResult,
    predict_formulation_viscosity,
    screen_excipients_viscosity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def water_result():
    return predict_formulation_viscosity(
        drug_name="test_drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=1.0,
        excipient_name="water",
        excipient_concentration_pct=0.0,
        temperature_C=25.0,
    )


@pytest.fixture
def hpmc_05_result():
    return predict_formulation_viscosity(
        drug_name="test_drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=1.0,
        excipient_name="HPMC_0.5pct",
        excipient_concentration_pct=1.0,
        temperature_C=25.0,
    )


@pytest.fixture
def hpmc_2_result():
    return predict_formulation_viscosity(
        drug_name="test_drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=1.0,
        excipient_name="HPMC_2pct",
        excipient_concentration_pct=1.0,
        temperature_C=25.0,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type(water_result):
    assert isinstance(water_result, ViscosityResult)


def test_drug_name_preserved(water_result):
    assert water_result.drug_name == "test_drug"


def test_excipient_name_preserved(water_result):
    assert water_result.excipient_name == "water"


# ---------------------------------------------------------------------------
# Physical plausibility tests
# ---------------------------------------------------------------------------


def test_viscosity_positive(water_result):
    assert water_result.viscosity_cP > 0


def test_water_viscosity_class(water_result):
    assert water_result.viscosity_class == "water_like"


def test_hpmc_2_higher_than_hpmc_05(hpmc_05_result, hpmc_2_result):
    assert hpmc_2_result.viscosity_cP > hpmc_05_result.viscosity_cP


def test_higher_temperature_lower_viscosity():
    """Higher temperature should give lower viscosity."""
    cold = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=1.0,
        excipient_name="glycerin",
        excipient_concentration_pct=1.0,
        temperature_C=10.0,
    )
    warm = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=1.0,
        excipient_name="glycerin",
        excipient_concentration_pct=1.0,
        temperature_C=40.0,
    )
    assert cold.viscosity_cP > warm.viscosity_cP


def test_higher_drug_conc_slightly_higher_viscosity():
    """Higher drug concentration should give slightly higher viscosity."""
    low = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=1.0,
        excipient_name="water",
        excipient_concentration_pct=0.0,
        temperature_C=25.0,
    )
    high = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=100.0,
        excipient_name="water",
        excipient_concentration_pct=0.0,
        temperature_C=25.0,
    )
    assert high.viscosity_cP > low.viscosity_cP


# ---------------------------------------------------------------------------
# Viscosity class tests
# ---------------------------------------------------------------------------


def test_viscosity_class_correct_low():
    r = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=1.0,
        excipient_name="PEG_400",
        excipient_concentration_pct=1.0,
        temperature_C=25.0,
    )
    assert r.viscosity_class in {"water_like", "low", "moderate"}


def test_viscosity_class_carbomer():
    """Carbomer at high concentration should be high/very high viscosity."""
    r = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="gel",
        drug_concentration_mg_mL=1.0,
        excipient_name="carbomer_0.5pct",
        excipient_concentration_pct=2.0,
        temperature_C=25.0,
    )
    assert r.viscosity_class in {"high", "very_high"}


# ---------------------------------------------------------------------------
# Syringeability tests
# ---------------------------------------------------------------------------


def test_water_syringeability_excellent(water_result):
    assert water_result.syringeability_class == "excellent"


def test_high_viscosity_not_injectable():
    r = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="gel",
        drug_concentration_mg_mL=1.0,
        excipient_name="carbomer_0.5pct",
        excipient_concentration_pct=5.0,
        temperature_C=25.0,
    )
    assert r.syringeability_class in {"difficult", "not_injectable"}


# ---------------------------------------------------------------------------
# Injectability rating tests
# ---------------------------------------------------------------------------


def test_injectability_rating_range(water_result):
    assert 0.0 <= water_result.injectability_rating <= 100.0


def test_high_viscosity_low_injectability():
    low_visc = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="aqueous_solution",
        drug_concentration_mg_mL=1.0,
        excipient_name="water",
        excipient_concentration_pct=0.0,
        temperature_C=25.0,
    )
    high_visc = predict_formulation_viscosity(
        drug_name="drug",
        formulation_type="gel",
        drug_concentration_mg_mL=1.0,
        excipient_name="carbomer_0.5pct",
        excipient_concentration_pct=2.0,
        temperature_C=25.0,
    )
    assert high_visc.injectability_rating < low_visc.injectability_rating


# ---------------------------------------------------------------------------
# screen_excipients_viscosity tests
# ---------------------------------------------------------------------------


def test_screen_returns_correct_length():
    results = screen_excipients_viscosity(
        drug_name="test", drug_concentration_mg_mL=1.0, temperature_C=25.0
    )
    assert len(results) == len(EXCIPIENTS)


def test_screen_sorted_ascending():
    results = screen_excipients_viscosity(
        drug_name="test", drug_concentration_mg_mL=1.0, temperature_C=25.0
    )
    for i in range(1, len(results)):
        assert results[i - 1].viscosity_cP <= results[i].viscosity_cP


def test_screen_all_return_type():
    results = screen_excipients_viscosity(
        drug_name="test", drug_concentration_mg_mL=1.0, temperature_C=25.0
    )
    assert all(isinstance(r, ViscosityResult) for r in results)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_invalid_excipient():
    with pytest.raises(ValueError, match="excipient_name"):
        predict_formulation_viscosity(
            drug_name="drug",
            formulation_type="aqueous_solution",
            drug_concentration_mg_mL=1.0,
            excipient_name="unknown_excipient",
            excipient_concentration_pct=1.0,
            temperature_C=25.0,
        )


def test_temperature_too_low():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_formulation_viscosity(
            drug_name="drug",
            formulation_type="aqueous_solution",
            drug_concentration_mg_mL=1.0,
            excipient_name="water",
            excipient_concentration_pct=0.0,
            temperature_C=2.0,
        )


def test_temperature_too_high():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_formulation_viscosity(
            drug_name="drug",
            formulation_type="aqueous_solution",
            drug_concentration_mg_mL=1.0,
            excipient_name="water",
            excipient_concentration_pct=0.0,
            temperature_C=90.0,
        )


def test_negative_drug_concentration():
    with pytest.raises(ValueError, match="drug_concentration_mg_mL"):
        predict_formulation_viscosity(
            drug_name="drug",
            formulation_type="aqueous_solution",
            drug_concentration_mg_mL=-1.0,
            excipient_name="water",
            excipient_concentration_pct=0.0,
            temperature_C=25.0,
        )
