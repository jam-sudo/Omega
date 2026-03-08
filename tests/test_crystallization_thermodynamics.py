"""Tests for Phase 375 — Drug Crystallization Thermodynamics."""

import pytest

from omega_pbpk.prediction.crystallization_thermodynamics import (
    CrystallizationThermodynamicsResult,
    assess_crystallization,
    compare_temperatures,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _base_result(
    concentration=5.0,
    solubility=1.0,
    mw=400.0,
    logp=2.0,
    temperature_C=25.0,
    surface_tension_mN_per_m=30.0,
):
    return assess_crystallization(
        drug_name="TestDrug",
        initial_concentration_mg_mL=concentration,
        equilibrium_solubility_mg_mL=solubility,
        mw_g_per_mol=mw,
        logp=logp,
        temperature_C=temperature_C,
        surface_tension_mN_per_m=surface_tension_mN_per_m,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _base_result()
    assert isinstance(result, CrystallizationThermodynamicsResult)


def test_result_is_frozen():
    result = _base_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Supersaturation ratio
# ---------------------------------------------------------------------------


def test_supersaturation_ratio_correct():
    result = _base_result(concentration=6.0, solubility=2.0)
    assert abs(result.supersaturation_ratio - 3.0) < 1e-9


def test_supersaturation_ratio_less_than_one_undersaturated():
    result = _base_result(concentration=0.5, solubility=1.0)
    assert result.supersaturation_ratio < 1.0
    assert result.crystallization_tendency == "low"
    assert result.nucleation_rate_per_s == 0.0


# ---------------------------------------------------------------------------
# 3. Crystallization tendency thresholds
# ---------------------------------------------------------------------------


def test_low_tendency_s_1_2():
    result = _base_result(concentration=1.2, solubility=1.0)
    assert result.crystallization_tendency == "low"


def test_moderate_tendency_s_2():
    result = _base_result(concentration=2.0, solubility=1.0)
    assert result.crystallization_tendency == "moderate"


def test_high_tendency_s_5():
    result = _base_result(concentration=5.0, solubility=1.0)
    assert result.crystallization_tendency == "high"


def test_very_high_tendency_s_15():
    result = _base_result(concentration=15.0, solubility=1.0)
    assert result.crystallization_tendency == "very_high"


# ---------------------------------------------------------------------------
# 4. Nucleation rate and induction time
# ---------------------------------------------------------------------------


def test_nucleation_rate_positive_when_supersaturated():
    result = _base_result(concentration=5.0, solubility=1.0)
    assert result.nucleation_rate_per_s > 0.0


def test_induction_time_is_inverse_of_nucleation_rate():
    result = _base_result(concentration=5.0, solubility=1.0)
    expected = 1.0 / result.nucleation_rate_per_s
    assert abs(result.induction_time_s - expected) < 1e-6 * expected + 1e-20


def test_higher_supersaturation_higher_nucleation_rate():
    low_s = _base_result(concentration=2.0, solubility=1.0)
    high_s = _base_result(concentration=20.0, solubility=1.0)
    assert high_s.nucleation_rate_per_s >= low_s.nucleation_rate_per_s


def test_undersaturated_gives_infinite_induction_time():
    result = _base_result(concentration=0.5, solubility=1.0)
    assert result.induction_time_s == float("inf")


# ---------------------------------------------------------------------------
# 5. Delta G
# ---------------------------------------------------------------------------


def test_delta_g_nucleation_positive_when_supersaturated():
    result = _base_result(concentration=5.0, solubility=1.0)
    assert result.delta_g_nucleation_kJ_per_mol > 0.0


def test_delta_g_zero_when_undersaturated():
    result = _base_result(concentration=0.5, solubility=1.0)
    assert result.delta_g_nucleation_kJ_per_mol == 0.0


# ---------------------------------------------------------------------------
# 6. Temperature effect
# ---------------------------------------------------------------------------


def test_temperature_field_stored():
    result = _base_result(temperature_C=37.0)
    assert abs(result.temperature_K - 310.15) < 0.01


def test_temperature_K_correct():
    result = _base_result(temperature_C=25.0)
    assert abs(result.temperature_K - 298.15) < 0.01


# ---------------------------------------------------------------------------
# 7. Amorphous advantage
# ---------------------------------------------------------------------------


def test_amorphous_advantage_at_least_one():
    result = _base_result(logp=-1.0)
    assert result.amorphous_advantage_fold >= 1.0


def test_amorphous_advantage_increases_with_logp():
    low_logp = _base_result(logp=1.0)
    high_logp = _base_result(logp=5.0)
    assert high_logp.amorphous_advantage_fold > low_logp.amorphous_advantage_fold


# ---------------------------------------------------------------------------
# 8. compare_temperatures
# ---------------------------------------------------------------------------


def test_compare_temperatures_returns_list():
    results = compare_temperatures(
        drug_name="TestDrug",
        initial_concentration_mg_mL=5.0,
        equilibrium_solubility_mg_mL=1.0,
        mw_g_per_mol=400.0,
        logp=2.0,
    )
    assert isinstance(results, list)


def test_compare_temperatures_default_four_entries():
    results = compare_temperatures(
        drug_name="TestDrug",
        initial_concentration_mg_mL=5.0,
        equilibrium_solubility_mg_mL=1.0,
        mw_g_per_mol=400.0,
        logp=2.0,
    )
    assert len(results) == 4


def test_compare_temperatures_sorted_nucleation_rate_ascending():
    results = compare_temperatures(
        drug_name="TestDrug",
        initial_concentration_mg_mL=5.0,
        equilibrium_solubility_mg_mL=1.0,
        mw_g_per_mol=400.0,
        logp=2.0,
    )
    rates = [r.nucleation_rate_per_s for r in results]
    assert rates == sorted(rates)


def test_compare_temperatures_custom_list():
    results = compare_temperatures(
        drug_name="TestDrug",
        initial_concentration_mg_mL=5.0,
        equilibrium_solubility_mg_mL=1.0,
        mw_g_per_mol=400.0,
        logp=2.0,
        temperatures_C=[10.0, 40.0],
    )
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 9. Validation errors
# ---------------------------------------------------------------------------


def test_validation_negative_concentration():
    with pytest.raises(ValueError, match="initial_concentration"):
        assess_crystallization(
            drug_name="X",
            initial_concentration_mg_mL=-1.0,
            equilibrium_solubility_mg_mL=1.0,
            mw_g_per_mol=300.0,
            logp=2.0,
        )


def test_validation_zero_solubility():
    with pytest.raises(ValueError, match="equilibrium_solubility"):
        assess_crystallization(
            drug_name="X",
            initial_concentration_mg_mL=1.0,
            equilibrium_solubility_mg_mL=0.0,
            mw_g_per_mol=300.0,
            logp=2.0,
        )


def test_validation_negative_mw():
    with pytest.raises(ValueError, match="mw_g_per_mol"):
        assess_crystallization(
            drug_name="X",
            initial_concentration_mg_mL=1.0,
            equilibrium_solubility_mg_mL=1.0,
            mw_g_per_mol=-100.0,
            logp=2.0,
        )


def test_validation_temperature_out_of_range():
    with pytest.raises(ValueError, match="temperature_C"):
        assess_crystallization(
            drug_name="X",
            initial_concentration_mg_mL=1.0,
            equilibrium_solubility_mg_mL=1.0,
            mw_g_per_mol=300.0,
            logp=2.0,
            temperature_C=300.0,
        )
