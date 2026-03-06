"""Tests for omega_pbpk.biopharmaceutics.dissolution module."""

import math
import pytest
from omega_pbpk.biopharmaceutics.dissolution import (
    DissolutionResult,
    simulate_weibull,
    simulate_noyes_whitney,
    compare_dissolution_models,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weibull(drug_name="DrugA", dose_mg=100.0, td_h=0.5, beta=1.5, t_end_h=4.0, dt_h=0.01):
    return simulate_weibull(drug_name, dose_mg, td_h=td_h, beta=beta, t_end_h=t_end_h, dt_h=dt_h)


def _make_noyes_whitney(drug_name="DrugB", dose_mg=100.0, solubility_mg_mL=0.1):
    return simulate_noyes_whitney(drug_name, dose_mg, solubility_mg_mL)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------

def test_weibull_returns_dissolution_result():
    result = _make_weibull()
    assert isinstance(result, DissolutionResult)


def test_noyes_whitney_returns_dissolution_result():
    result = _make_noyes_whitney()
    assert isinstance(result, DissolutionResult)


# ---------------------------------------------------------------------------
# 2. drug_name preserved
# ---------------------------------------------------------------------------

def test_weibull_drug_name_preserved():
    result = simulate_weibull("Ibuprofen", 400.0)
    assert result.drug_name == "Ibuprofen"


def test_noyes_whitney_drug_name_preserved():
    result = simulate_noyes_whitney("Furosemide", 40.0, solubility_mg_mL=0.05)
    assert result.drug_name == "Furosemide"


# ---------------------------------------------------------------------------
# 3. times / pct_dissolved length consistency
# ---------------------------------------------------------------------------

def test_weibull_times_and_pct_same_length():
    result = _make_weibull()
    assert len(result.times_h) == len(result.pct_dissolved)


def test_noyes_whitney_times_and_pct_same_length():
    result = _make_noyes_whitney()
    assert len(result.times_h) == len(result.pct_dissolved)


# ---------------------------------------------------------------------------
# 4. Boundary values
# ---------------------------------------------------------------------------

def test_weibull_pct_dissolved_first_nonnegative():
    result = _make_weibull()
    assert result.pct_dissolved[0] >= 0.0


def test_weibull_pct_dissolved_last_at_most_100():
    result = _make_weibull()
    assert result.pct_dissolved[-1] <= 100.0


def test_noyes_whitney_pct_dissolved_starts_at_zero():
    result = _make_noyes_whitney()
    assert result.pct_dissolved[0] == pytest.approx(0.0, abs=1e-6)


def test_noyes_whitney_pct_dissolved_last_at_most_100():
    result = _make_noyes_whitney()
    assert result.pct_dissolved[-1] <= 100.0


# ---------------------------------------------------------------------------
# 5. Monotonically increasing
# ---------------------------------------------------------------------------

def test_weibull_pct_dissolved_monotonically_increasing():
    result = _make_weibull()
    for i in range(1, len(result.pct_dissolved)):
        assert result.pct_dissolved[i] >= result.pct_dissolved[i - 1] - 1e-9


def test_noyes_whitney_pct_dissolved_monotonically_increasing():
    result = _make_noyes_whitney()
    for i in range(1, len(result.pct_dissolved)):
        assert result.pct_dissolved[i] >= result.pct_dissolved[i - 1] - 1e-9


# ---------------------------------------------------------------------------
# 6. t80 / t90 ordering and sign
# ---------------------------------------------------------------------------

def test_weibull_t80_positive_when_reached():
    result = _make_weibull(td_h=0.5, beta=1.5, t_end_h=4.0)
    if result.t80_h is not None:
        assert result.t80_h > 0.0


def test_weibull_t90_greater_than_t80_when_both_reached():
    result = _make_weibull(td_h=0.5, beta=1.5, t_end_h=6.0)
    if result.t80_h is not None and result.t90_h is not None:
        assert result.t90_h > result.t80_h


def test_noyes_whitney_t80_positive_when_reached():
    result = _make_noyes_whitney(solubility_mg_mL=5.0)
    if result.t80_h is not None:
        assert result.t80_h > 0.0


# ---------------------------------------------------------------------------
# 7. Weibull analytical formula for t80 and t90
# ---------------------------------------------------------------------------

def test_weibull_t80_matches_analytical_formula():
    td, beta = 0.5, 1.5
    result = simulate_weibull("Drug", 100.0, td_h=td, beta=beta, t_end_h=6.0)
    expected_t80 = td * ((-math.log(0.2)) ** (1.0 / beta))
    assert result.t80_h == pytest.approx(expected_t80, rel=1e-3)


def test_weibull_t90_matches_analytical_formula():
    td, beta = 0.5, 1.5
    result = simulate_weibull("Drug", 100.0, td_h=td, beta=beta, t_end_h=6.0)
    expected_t90 = td * ((-math.log(0.1)) ** (1.0 / beta))
    assert result.t90_h == pytest.approx(expected_t90, rel=1e-3)


# ---------------------------------------------------------------------------
# 8. Parametric sensitivity — higher beta means faster dissolution (smaller t80)
# ---------------------------------------------------------------------------

def test_weibull_higher_beta_smaller_t80():
    r_low = simulate_weibull("Drug", 100.0, td_h=0.5, beta=0.8, t_end_h=8.0)
    r_high = simulate_weibull("Drug", 100.0, td_h=0.5, beta=2.5, t_end_h=8.0)
    if r_low.t80_h is not None and r_high.t80_h is not None:
        assert r_high.t80_h < r_low.t80_h


# ---------------------------------------------------------------------------
# 9. Parametric sensitivity — higher td means larger t80
# ---------------------------------------------------------------------------

def test_weibull_higher_td_larger_t80():
    r_small = simulate_weibull("Drug", 100.0, td_h=0.3, beta=1.5, t_end_h=6.0)
    r_large = simulate_weibull("Drug", 100.0, td_h=1.0, beta=1.5, t_end_h=6.0)
    if r_small.t80_h is not None and r_large.t80_h is not None:
        assert r_large.t80_h > r_small.t80_h


# ---------------------------------------------------------------------------
# 10. model_params contains expected keys
# ---------------------------------------------------------------------------

def test_weibull_model_params_has_td_and_beta():
    result = _make_weibull(td_h=0.7, beta=2.0)
    assert "td_h" in result.model_params
    assert "beta" in result.model_params


def test_weibull_model_params_values_match_inputs():
    result = simulate_weibull("Drug", 100.0, td_h=0.7, beta=2.0)
    assert result.model_params["td_h"] == pytest.approx(0.7)
    assert result.model_params["beta"] == pytest.approx(2.0)


def test_noyes_whitney_model_params_has_solubility():
    result = simulate_noyes_whitney("Drug", 100.0, solubility_mg_mL=0.25)
    assert "solubility_mg_mL" in result.model_params


# ---------------------------------------------------------------------------
# 11. dissolution_rate_pct_per_h > 0
# ---------------------------------------------------------------------------

def test_weibull_dissolution_rate_positive():
    result = _make_weibull()
    assert result.dissolution_rate_pct_per_h > 0.0


def test_noyes_whitney_dissolution_rate_positive():
    result = _make_noyes_whitney()
    assert result.dissolution_rate_pct_per_h > 0.0


# ---------------------------------------------------------------------------
# 12. dissolution_model field
# ---------------------------------------------------------------------------

def test_weibull_dissolution_model_field():
    result = _make_weibull()
    assert result.dissolution_model.lower() == "weibull"


def test_noyes_whitney_dissolution_model_field():
    result = _make_noyes_whitney()
    assert "noyes" in result.dissolution_model.lower() or "whitney" in result.dissolution_model.lower()


# ---------------------------------------------------------------------------
# 13. compare_dissolution_models
# ---------------------------------------------------------------------------

def test_compare_dissolution_models_returns_dict():
    out = compare_dissolution_models("Drug", 100.0, solubility_mg_mL=0.1)
    assert isinstance(out, dict)


def test_compare_dissolution_models_has_three_keys():
    out = compare_dissolution_models("Drug", 100.0, solubility_mg_mL=0.1)
    assert len(out) == 3


def test_compare_dissolution_models_values_are_dissolution_results():
    out = compare_dissolution_models("Drug", 100.0, solubility_mg_mL=0.1)
    for key, val in out.items():
        assert isinstance(val, DissolutionResult), f"Key '{key}' is not a DissolutionResult"


# ---------------------------------------------------------------------------
# 14. sink_conditions field
# ---------------------------------------------------------------------------

def test_weibull_sink_conditions_is_bool():
    result = _make_weibull()
    assert isinstance(result.sink_conditions, bool)


def test_noyes_whitney_sink_conditions_is_bool():
    result = _make_noyes_whitney()
    assert isinstance(result.sink_conditions, bool)


# ---------------------------------------------------------------------------
# 15. Weibull pct_dissolved at t=0 near zero
# ---------------------------------------------------------------------------

def test_weibull_pct_dissolved_at_t0_near_zero():
    result = _make_weibull(dt_h=0.01)
    # t=0 gives 100*(1-exp(0))=0
    assert result.pct_dissolved[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 16. Weibull pct formula at a specific midpoint
# ---------------------------------------------------------------------------

def test_weibull_pct_at_td_equals_63_percent():
    """At t=td, pct = 100*(1-exp(-1^beta)) = 100*(1-exp(-1)) ≈ 63.21% for any beta."""
    td, beta = 1.0, 2.0
    result = simulate_weibull("Drug", 100.0, td_h=td, beta=beta, t_end_h=5.0, dt_h=0.001)
    # Find index closest to t=td
    idx = min(range(len(result.times_h)), key=lambda i: abs(result.times_h[i] - td))
    expected = 100.0 * (1.0 - math.exp(-1.0))
    assert result.pct_dissolved[idx] == pytest.approx(expected, abs=0.5)
