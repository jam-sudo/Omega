"""Tests for concentration-effect (PK/PD) modeling — Phase 831."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.conc_effect_modeling import (
    ConcEffectResult,
    fit_emax_model,
    predict_effect,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _synthetic_sigmoid(
    concs: list[float],
    e0: float,
    emax: float,
    ec50: float,
    hill: float,
) -> list[float]:
    """Generate noise-free sigmoid Emax effects."""
    effects = []
    for c in concs:
        if c <= 0.0:
            effects.append(e0)
        else:
            cn = c**hill
            ec50n = ec50**hill
            effects.append(e0 + (emax - e0) * cn / (ec50n + cn))
    return effects


CONCS_BASIC = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
TRUE_E0 = 0.0
TRUE_EMAX = 100.0
TRUE_EC50 = 5.0
TRUE_HILL = 2.0

EFFECTS_SIGMOID = _synthetic_sigmoid(CONCS_BASIC, TRUE_E0, TRUE_EMAX, TRUE_EC50, TRUE_HILL)
EFFECTS_EMAX = _synthetic_sigmoid(CONCS_BASIC, TRUE_E0, TRUE_EMAX, TRUE_EC50, 1.0)

CONCS_NO_ZERO = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
EFFECTS_NO_ZERO = _synthetic_sigmoid(CONCS_NO_ZERO, TRUE_E0, TRUE_EMAX, TRUE_EC50, 1.0)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_fit_returns_conc_effect_result(self):
        result = fit_emax_model("TestDrug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        assert isinstance(result, ConcEffectResult)

    def test_result_is_frozen(self):
        result = fit_emax_model("TestDrug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_predicted_effects_length_matches_input(self):
        result = fit_emax_model("TestDrug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        assert len(result.predicted_effects) == len(CONCS_BASIC)

    def test_drug_name_preserved(self):
        result = fit_emax_model("DrugX", CONCS_BASIC, EFFECTS_SIGMOID, "emax")
        assert result.drug_name == "DrugX"

    def test_model_type_preserved(self):
        for mt in ("emax", "sigmoid_emax", "linear", "power"):
            result = fit_emax_model("D", CONCS_BASIC, EFFECTS_SIGMOID, mt)
            assert result.model_type == mt


# ---------------------------------------------------------------------------
# Goodness-of-fit tests
# ---------------------------------------------------------------------------


class TestGoodnessOfFit:
    def test_sigmoid_emax_r2_above_095(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        assert result.r_squared > 0.95

    def test_emax_r2_above_095(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_EMAX, "emax")
        assert result.r_squared > 0.95

    def test_rmse_is_nonnegative(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        assert result.rmse >= 0.0

    def test_linear_r2_above_08_for_low_concs(self):
        concs_lin = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        effects_lin = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        result = fit_emax_model("Drug", concs_lin, effects_lin, "linear")
        assert result.r_squared > 0.8


# ---------------------------------------------------------------------------
# EC50 accuracy
# ---------------------------------------------------------------------------


class TestEC50Accuracy:
    def test_ec50_close_to_true_for_emax(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_EMAX, "emax")
        # Allow 50% relative error given coarse grid
        assert abs(result.ec50 - TRUE_EC50) / TRUE_EC50 < 0.5

    def test_ec50_close_to_true_for_sigmoid_emax(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        assert abs(result.ec50 - TRUE_EC50) / TRUE_EC50 < 0.5


# ---------------------------------------------------------------------------
# EC10 / EC50 / EC90 ordering
# ---------------------------------------------------------------------------


class TestECOrdering:
    def test_ec10_less_than_ec50_less_than_ec90(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        assert result.ec10 < result.ec50
        assert result.ec50 < result.ec90

    def test_ec10_less_than_ec50_emax(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_EMAX, "emax")
        assert result.ec10 < result.ec50 < result.ec90

    def test_therapeutic_index_positive(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        assert result.therapeutic_index > 0.0

    def test_therapeutic_index_equals_ratio(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        expected = result.ec90 / result.ec10
        assert abs(result.therapeutic_index - expected) < 1e-9


# ---------------------------------------------------------------------------
# Model-specific properties
# ---------------------------------------------------------------------------


class TestModelSpecific:
    def test_emax_hill_coef_is_one(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_EMAX, "emax")
        assert result.hill_coef == 1.0

    def test_linear_hill_coef_is_one(self):
        concs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        effects = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        result = fit_emax_model("Drug", concs, effects, "linear")
        assert result.hill_coef == 1.0

    def test_power_hill_coef_is_one(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "power")
        assert result.hill_coef == 1.0

    def test_e0_zero_when_first_conc_nonzero(self):
        result = fit_emax_model("Drug", CONCS_NO_ZERO, EFFECTS_NO_ZERO, "emax")
        assert result.e0 == 0.0

    def test_e0_set_from_first_effect_when_conc_is_zero(self):
        concs = [0.0, 1.0, 5.0, 10.0, 20.0]
        effects = [5.0, 30.0, 70.0, 90.0, 98.0]
        result = fit_emax_model("Drug", concs, effects, "emax")
        assert result.e0 == 5.0


# ---------------------------------------------------------------------------
# predict_effect
# ---------------------------------------------------------------------------


class TestPredictEffect:
    def test_predict_effect_returns_float(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        val = predict_effect("Drug", 5.0, result)
        assert isinstance(val, float)

    def test_predict_effect_sigmoid_near_emax_at_high_conc(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "sigmoid_emax")
        val = predict_effect("Drug", 1000.0, result)
        assert val > 0.8 * result.emax

    def test_predict_effect_emax_model(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_EMAX, "emax")
        val = predict_effect("Drug", 0.0, result)
        assert isinstance(val, float)

    def test_predict_effect_linear_model(self):
        concs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        effects = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        result = fit_emax_model("Drug", concs, effects, "linear")
        val = predict_effect("Drug", 2.5, result)
        assert isinstance(val, float)

    def test_predict_effect_power_model(self):
        result = fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "power")
        val = predict_effect("Drug", 10.0, result)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mismatched_lengths_raise_value_error(self):
        with pytest.raises(ValueError, match="same length"):
            fit_emax_model("Drug", [1.0, 2.0, 3.0], [10.0, 20.0], "emax")

    def test_too_few_points_raise_value_error(self):
        with pytest.raises(ValueError, match="3"):
            fit_emax_model("Drug", [1.0, 2.0], [10.0, 20.0], "emax")

    def test_unknown_model_type_raises_value_error(self):
        with pytest.raises(ValueError):
            fit_emax_model("Drug", CONCS_BASIC, EFFECTS_SIGMOID, "unknown_model")
