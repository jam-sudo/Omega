"""Tests for Phase 294 — Formulation Stability Under Temperature Cycling."""

import pytest

from omega_pbpk.prediction.temperature_cycling_stability import (
    TemperatureCyclingResult,
    assess_cold_chain_risk,
    simulate_temperature_cycling,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="TestBiologic",
    n_cycles=3,
    t_cold_h_per_cycle=168.0,  # 1 week cold
    t_warm_h_per_cycle=2.0,  # 2 h warm excursion
    temp_cold_C=5.0,
    temp_warm_C=25.0,
    initial_purity_pct=99.0,
    k_deg_ref_per_h=1e-4,
    ea_kJ_per_mol=80.0,
)


# ---------------------------------------------------------------------------
# Return type & field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert isinstance(result, TemperatureCyclingResult)


def test_drug_name_preserved():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.drug_name == "TestBiologic"


def test_n_cycles_preserved():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.n_cycles == 3


def test_temp_cold_preserved():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.temp_cold_C == 5.0


def test_temp_warm_preserved():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.temp_warm_C == 25.0


def test_initial_purity_preserved():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.initial_purity_pct == 99.0


# ---------------------------------------------------------------------------
# Purity bounds
# ---------------------------------------------------------------------------


def test_purity_final_leq_initial():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.purity_final_pct <= result.initial_purity_pct


def test_purity_final_geq_zero():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.purity_final_pct >= 0.0


def test_degradation_pct_geq_zero():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.degradation_pct >= 0.0


def test_degradation_pct_equals_diff():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    expected = result.initial_purity_pct - result.purity_final_pct
    assert result.degradation_pct == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Temperature effect
# ---------------------------------------------------------------------------


def test_higher_temp_warm_lower_purity():
    """Higher warm temperature should yield lower final purity."""
    result_low = simulate_temperature_cycling(**{**BASE_KWARGS, "temp_warm_C": 25.0})
    result_high = simulate_temperature_cycling(**{**BASE_KWARGS, "temp_warm_C": 40.0})
    assert result_high.purity_final_pct < result_low.purity_final_pct


def test_higher_temp_warm_higher_k_warm():
    result_low = simulate_temperature_cycling(**{**BASE_KWARGS, "temp_warm_C": 25.0})
    result_high = simulate_temperature_cycling(**{**BASE_KWARGS, "temp_warm_C": 40.0})
    assert result_high.k_warm_per_h > result_low.k_warm_per_h


# ---------------------------------------------------------------------------
# Cycle count effect
# ---------------------------------------------------------------------------


def test_more_cycles_lower_purity():
    result_few = simulate_temperature_cycling(**{**BASE_KWARGS, "n_cycles": 1})
    result_many = simulate_temperature_cycling(**{**BASE_KWARGS, "n_cycles": 10})
    assert result_many.purity_final_pct < result_few.purity_final_pct


def test_total_warm_h_scales_with_cycles():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    expected_warm = 3 * 2.0
    assert result.total_warm_h == pytest.approx(expected_warm)


def test_total_cold_h_scales_with_cycles():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    expected_cold = 3 * 168.0
    assert result.total_cold_h == pytest.approx(expected_cold)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_stability_class_valid():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.stability_class in {"stable", "marginally_stable", "unstable"}


def test_failure_risk_valid():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert result.failure_risk in {"low", "moderate", "high"}


def test_stable_for_mild_conditions():
    """Very mild cycling should produce stable classification."""
    result = simulate_temperature_cycling(
        drug_name="StableDrug",
        n_cycles=1,
        t_cold_h_per_cycle=1.0,
        t_warm_h_per_cycle=0.1,
        temp_cold_C=2.0,
        temp_warm_C=8.0,
        initial_purity_pct=99.5,
        k_deg_ref_per_h=1e-6,
        ea_kJ_per_mol=50.0,
    )
    assert result.stability_class == "stable"


def test_unstable_for_harsh_conditions():
    """Very harsh cycling should produce unstable classification."""
    result = simulate_temperature_cycling(
        drug_name="UnstableDrug",
        n_cycles=50,
        t_cold_h_per_cycle=1.0,
        t_warm_h_per_cycle=100.0,
        temp_cold_C=5.0,
        temp_warm_C=60.0,
        initial_purity_pct=99.0,
        k_deg_ref_per_h=0.01,
        ea_kJ_per_mol=80.0,
    )
    assert result.stability_class == "unstable"
    assert result.failure_risk == "high"


# ---------------------------------------------------------------------------
# assess_cold_chain_risk
# ---------------------------------------------------------------------------


def test_assess_cold_chain_risk_returns_correct_type():
    result = assess_cold_chain_risk(
        drug_name="BioDrug",
        n_excursions=5,
        excursion_duration_h=4.0,
        excursion_temp_C=25.0,
        initial_purity_pct=98.0,
        k_deg_ref_per_h=5e-5,
        ea_kJ_per_mol=75.0,
    )
    assert isinstance(result, TemperatureCyclingResult)


def test_assess_cold_chain_risk_temp_cold_is_5():
    result = assess_cold_chain_risk(
        drug_name="BioDrug",
        n_excursions=5,
        excursion_duration_h=4.0,
        excursion_temp_C=25.0,
        initial_purity_pct=98.0,
        k_deg_ref_per_h=5e-5,
        ea_kJ_per_mol=75.0,
    )
    assert result.temp_cold_C == 5.0


def test_assess_cold_chain_risk_n_cycles():
    result = assess_cold_chain_risk(
        drug_name="BioDrug",
        n_excursions=7,
        excursion_duration_h=4.0,
        excursion_temp_C=30.0,
        initial_purity_pct=98.0,
        k_deg_ref_per_h=5e-5,
        ea_kJ_per_mol=75.0,
    )
    assert result.n_cycles == 7


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_n_cycles_raises():
    with pytest.raises(ValueError, match="n_cycles"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "n_cycles": 0})


def test_invalid_t_cold_raises():
    with pytest.raises(ValueError, match="t_cold_h_per_cycle"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "t_cold_h_per_cycle": 0.0})


def test_invalid_t_warm_raises():
    with pytest.raises(ValueError, match="t_warm_h_per_cycle"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "t_warm_h_per_cycle": -1.0})


def test_temp_warm_leq_cold_raises():
    with pytest.raises(ValueError, match="temp_warm_C"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "temp_warm_C": 5.0})


def test_temp_warm_equal_cold_raises():
    with pytest.raises(ValueError, match="temp_warm_C"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "temp_cold_C": 10.0, "temp_warm_C": 10.0})


def test_invalid_purity_zero_raises():
    with pytest.raises(ValueError, match="initial_purity_pct"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "initial_purity_pct": 0.0})


def test_invalid_purity_over_100_raises():
    with pytest.raises(ValueError, match="initial_purity_pct"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "initial_purity_pct": 101.0})


def test_invalid_k_deg_raises():
    with pytest.raises(ValueError, match="k_deg_ref_per_h"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "k_deg_ref_per_h": 0.0})


def test_invalid_ea_raises():
    with pytest.raises(ValueError, match="ea_kJ_per_mol"):
        simulate_temperature_cycling(**{**BASE_KWARGS, "ea_kJ_per_mol": -10.0})


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = simulate_temperature_cycling(**BASE_KWARGS)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
