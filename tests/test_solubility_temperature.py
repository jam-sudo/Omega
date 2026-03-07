"""Tests for Phase 1062 — Drug Aqueous Solubility Temperature Dependence."""

import pytest

from omega_pbpk.prediction.solubility_temperature import (
    SolubilityTemperatureResult,
    predict_solubility_temperature,
    solubility_temperature_profile,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logp=2.5,
        melting_point_C=150.0,
        temperature_C=25.0,
        pka=7.0,
        ionization_type="neutral",
        ph=7.0,
    )
    params.update(kwargs)
    return predict_solubility_temperature(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_frozen_dataclass():
    result = _default()
    assert isinstance(result, SolubilityTemperatureResult)


def test_frozen_dataclass_immutable():
    result = _default()
    with pytest.raises((AttributeError, TypeError)):
        result.solubility_mg_mL = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core value tests
# ---------------------------------------------------------------------------


def test_solubility_positive():
    result = _default()
    assert result.solubility_mg_mL > 0.0


def test_relative_to_25c_positive():
    result = _default()
    assert result.relative_to_25C > 0.0


def test_dissolution_heat_positive_high_logp():
    """For logP > 1, ΔH > 0 (endothermic dissolution)."""
    result = _default(logp=3.5)
    assert result.dissolution_heat_kJ_mol > 0.0


def test_enthalpy_driven_false_for_typical_drug():
    """Endothermic dissolution (ΔH > 0) → enthalpy_driven = False."""
    result = _default(logp=3.5)
    assert result.enthalpy_driven is False


def test_temperature_sensitivity_in_valid_set():
    result = _default()
    assert result.temperature_sensitivity in {"low", "moderate", "high"}


def test_crystal_habit_in_valid_set():
    result = _default()
    assert result.predicted_crystal_habit in {"prismatic", "needle", "plate"}


def test_higher_temperature_higher_solubility_endothermic():
    """For endothermic dissolution (ΔH > 0), higher T → higher solubility."""
    r_low = _default(temperature_C=15.0, logp=3.0)
    r_high = _default(temperature_C=50.0, logp=3.0)
    assert r_high.solubility_mg_mL > r_low.solubility_mg_mL


def test_high_logp_higher_dissolution_heat():
    """Higher logP → higher dissolution enthalpy."""
    r_low = _default(logp=0.5)
    r_high = _default(logp=4.0)
    assert r_high.dissolution_heat_kJ_mol > r_low.dissolution_heat_kJ_mol


def test_acid_high_ph_higher_solubility():
    """Acidic drug at high pH should be more soluble than at low pH (HH)."""
    r_lo = predict_solubility_temperature(
        "Acid", 200.0, 1.5, 100.0, temperature_C=25.0,
        pka=5.0, ionization_type="acid", ph=3.0
    )
    r_hi = predict_solubility_temperature(
        "Acid", 200.0, 1.5, 100.0, temperature_C=25.0,
        pka=5.0, ionization_type="acid", ph=9.0
    )
    assert r_hi.solubility_mg_mL > r_lo.solubility_mg_mL


def test_notes_nonempty():
    result = _default()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Crystal habit tests
# ---------------------------------------------------------------------------


def test_crystal_habit_needle_cold_lipophilic():
    result = _default(temperature_C=10.0, logp=3.0)
    assert result.predicted_crystal_habit == "needle"


def test_crystal_habit_prismatic_hot():
    result = _default(temperature_C=60.0)
    assert result.predicted_crystal_habit == "prismatic"


# ---------------------------------------------------------------------------
# solubility_temperature_profile
# ---------------------------------------------------------------------------


def test_profile_returns_list_correct_length():
    temps = [10.0, 25.0, 37.0, 60.0]
    results = solubility_temperature_profile(
        "Drug", 300.0, 2.0, 120.0, temps
    )
    assert isinstance(results, list)
    assert len(results) == len(temps)


def test_profile_all_results_are_correct_type():
    temps = [15.0, 37.0]
    results = solubility_temperature_profile("Drug", 300.0, 2.0, 120.0, temps)
    for r in results:
        assert isinstance(r, SolubilityTemperatureResult)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _default(mw_Da=0.0)


def test_validation_melting_point_zero():
    with pytest.raises(ValueError, match="melting_point_C"):
        _default(melting_point_C=0.0)


def test_validation_temperature_out_of_range():
    with pytest.raises(ValueError, match="temperature_C"):
        _default(temperature_C=150.0)


def test_validation_pka_out_of_range():
    with pytest.raises(ValueError, match="pka"):
        _default(pka=15.0)


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        _default(ionization_type="zwitterion")
