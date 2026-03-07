"""Tests for Phase 844 — Drug Formulation Release Modeling."""

import math

import pytest

from omega_pbpk.core.formulation_release_modeling import (
    FormulationReleaseResult,
    compare_release_models,
    simulate_release,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type_higuchi():
    result = simulate_release("Matrix", "DrugX", dose_mg=100.0, release_model="higuchi")
    assert isinstance(result, FormulationReleaseResult)


def test_return_type_korsmeyer_peppas():
    result = simulate_release("ER Tablet", "DrugY", dose_mg=50.0, release_model="korsmeyer_peppas")
    assert isinstance(result, FormulationReleaseResult)


# ---------------------------------------------------------------------------
# Monotonically non-decreasing fraction_released
# ---------------------------------------------------------------------------


def test_fraction_monotone_higuchi():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="higuchi", k_release=0.15)
    for i in range(1, len(result.fraction_released)):
        assert result.fraction_released[i] >= result.fraction_released[i - 1] - 1e-12


def test_fraction_monotone_first_order():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="first_order", k_release=0.1)
    for i in range(1, len(result.fraction_released)):
        assert result.fraction_released[i] >= result.fraction_released[i - 1] - 1e-12


def test_fraction_monotone_zero_order():
    result = simulate_release("M", "D", dose_mg=200.0, release_model="zero_order", k_release=0.05)
    for i in range(1, len(result.fraction_released)):
        assert result.fraction_released[i] >= result.fraction_released[i - 1] - 1e-12


def test_fraction_monotone_osmotic():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="osmotic", k_release=0.2)
    for i in range(1, len(result.fraction_released)):
        assert result.fraction_released[i] >= result.fraction_released[i - 1] - 1e-12


def test_fraction_monotone_korsmeyer_peppas():
    result = simulate_release(
        "M", "D", dose_mg=100.0, release_model="korsmeyer_peppas", k_release=0.1, n_exponent=0.45
    )
    for i in range(1, len(result.fraction_released)):
        assert result.fraction_released[i] >= result.fraction_released[i - 1] - 1e-12


# ---------------------------------------------------------------------------
# fraction_released[-1] > 0
# ---------------------------------------------------------------------------


def test_final_fraction_positive_higuchi():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="higuchi")
    assert result.fraction_released[-1] > 0.0


def test_final_fraction_positive_all_models():
    for model in ("higuchi", "korsmeyer_peppas", "zero_order", "first_order", "osmotic"):
        result = simulate_release("M", "D", dose_mg=100.0, release_model=model, k_release=0.1)
        assert result.fraction_released[-1] > 0.0, f"Model {model} gave zero final fraction"


# ---------------------------------------------------------------------------
# t50 < t80 < t90 ordering
# ---------------------------------------------------------------------------


def test_t50_t80_t90_ordering_first_order():
    result = simulate_release(
        "M", "D", dose_mg=100.0, release_model="first_order", k_release=0.3, t_end_h=24.0
    )
    if result.t50_h < math.inf and result.t80_h < math.inf and result.t90_h < math.inf:
        assert result.t50_h <= result.t80_h <= result.t90_h


def test_t50_t80_t90_ordering_zero_order():
    result = simulate_release(
        "M", "D", dose_mg=100.0, release_model="zero_order", k_release=0.05, t_end_h=24.0
    )
    if result.t50_h < math.inf and result.t80_h < math.inf:
        assert result.t50_h <= result.t80_h


# ---------------------------------------------------------------------------
# first_order asymptotic behaviour
# ---------------------------------------------------------------------------


def test_first_order_approaches_one():
    result = simulate_release(
        "M", "D", dose_mg=100.0, release_model="first_order", k_release=1.0, t_end_h=24.0
    )
    # With k=1.0, after 24h: F = 1 - exp(-24) ≈ 1.0
    assert result.fraction_released[-1] > 0.99


# ---------------------------------------------------------------------------
# zero_order constant rate
# ---------------------------------------------------------------------------


def test_zero_order_constant_rate():
    result = simulate_release(
        "M", "D", dose_mg=100.0, release_model="zero_order", k_release=0.04, t_end_h=10.0, dt_h=0.1
    )
    # rate should be constant = k * dose_mg (clamped when fraction hits 1)
    expected_rate = 0.04 * 100.0
    # Check rates where fraction < 1.0
    for f, rate in zip(result.fraction_released, result.release_rate_mg_per_h):
        if f < 1.0:
            assert abs(rate - expected_rate) < 1e-6, f"Rate {rate} != expected {expected_rate}"


# ---------------------------------------------------------------------------
# Korsmeyer-Peppas release mechanism
# ---------------------------------------------------------------------------


def test_korsmeyer_peppas_diffusion_n_le_0p5():
    result = simulate_release(
        "M", "D", dose_mg=100.0, release_model="korsmeyer_peppas", n_exponent=0.45
    )
    assert result.release_mechanism == "diffusion"


def test_korsmeyer_peppas_erosion_n_gt_0p5():
    result = simulate_release(
        "M", "D", dose_mg=100.0, release_model="korsmeyer_peppas", n_exponent=0.75
    )
    assert result.release_mechanism == "erosion"


# ---------------------------------------------------------------------------
# release_mechanism assignment
# ---------------------------------------------------------------------------


def test_higuchi_mechanism_is_diffusion():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="higuchi")
    assert result.release_mechanism == "diffusion"


def test_zero_order_mechanism_is_osmotic_pumping():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="zero_order")
    assert result.release_mechanism == "osmotic_pumping"


def test_first_order_mechanism_is_first_order():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="first_order")
    assert result.release_mechanism == "first_order"


def test_osmotic_mechanism_is_osmotic_pumping():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="osmotic")
    assert result.release_mechanism == "osmotic_pumping"


# ---------------------------------------------------------------------------
# similarity_factor_f2 bounds
# ---------------------------------------------------------------------------


def test_f2_bounds():
    result = simulate_release("M", "D", dose_mg=100.0, release_model="zero_order")
    assert 0.0 <= result.similarity_factor_f2 <= 100.0


# ---------------------------------------------------------------------------
# complete_release_h is inf when not reached
# ---------------------------------------------------------------------------


def test_complete_release_inf_when_not_reached():
    # Higuchi with k=0.01 over 24h: F = 0.01 * sqrt(24) ≈ 0.049 < 0.95
    result = simulate_release(
        "M", "D", dose_mg=100.0, release_model="higuchi", k_release=0.01, t_end_h=24.0
    )
    assert result.complete_release_h == math.inf


# ---------------------------------------------------------------------------
# compare_release_models
# ---------------------------------------------------------------------------


def test_compare_release_models_returns_5():
    results = compare_release_models("FormX", "DrugZ", dose_mg=100.0, k_release=0.15)
    assert len(results) == 5


def test_compare_release_models_all_instances():
    results = compare_release_models("FormX", "DrugZ", dose_mg=100.0)
    for r in results:
        assert isinstance(r, FormulationReleaseResult)


def test_compare_release_models_sorted_by_t50():
    results = compare_release_models("FormX", "DrugZ", dose_mg=100.0, k_release=0.2)
    t50_values = [r.t50_h for r in results]
    # Sorted ascending (math.inf is fine for tail)
    for i in range(1, len(t50_values)):
        a = t50_values[i - 1]
        b = t50_values[i]
        if a == math.inf:
            assert b == math.inf
        else:
            assert a <= b


def test_compare_release_models_all_models_present():
    results = compare_release_models("FormX", "DrugZ", dose_mg=100.0)
    model_names = {r.release_model for r in results}
    expected = {"higuchi", "korsmeyer_peppas", "zero_order", "first_order", "osmotic"}
    assert model_names == expected


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_release("M", "D", dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_release("M", "D", dose_mg=-10.0)


def test_k_release_zero_raises():
    with pytest.raises(ValueError):
        simulate_release("M", "D", dose_mg=100.0, k_release=0.0)


def test_k_release_negative_raises():
    with pytest.raises(ValueError):
        simulate_release("M", "D", dose_mg=100.0, k_release=-0.1)


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        simulate_release("M", "D", dose_mg=100.0, release_model="invalid_model")
