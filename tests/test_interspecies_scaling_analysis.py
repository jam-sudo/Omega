"""Tests for Phase 382: interspecies_scaling_analysis module."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.interspecies_scaling_analysis import (
    AllometricScalingResult,
    analyze_allometric_scaling,
    predict_human_from_animals,
)

# ---------------------------------------------------------------------------
# Reference data: textbook allometric CL data (CL in L/h total)
# BW (kg):   0.02   0.25   3.5    10.0   70.0
# Perfect b=0.75 data: CL = 0.1 * BW^0.75
# ---------------------------------------------------------------------------

def _cl_075(bw: float) -> float:
    """Perfect CL with b=0.75."""
    return 0.1 * (bw ** 0.75)


def _vd_10(bw: float) -> float:
    """Perfect Vd with b=1.0."""
    return 0.5 * bw


SPECIES_BW = [0.02, 0.25, 3.5, 10.0, 30.0]


# ---------------------------------------------------------------------------
# Input validation tests — analyze_allometric_scaling
# ---------------------------------------------------------------------------

class TestAnalyzeAllometricScalingValidation:
    def test_empty_param_name(self):
        with pytest.raises(ValueError, match="param_name"):
            analyze_allometric_scaling([1.0, 2.0], [0.1, 0.2], "")

    def test_whitespace_param_name(self):
        with pytest.raises(ValueError, match="param_name"):
            analyze_allometric_scaling([1.0, 2.0], [0.1, 0.2], "  ")

    def test_empty_weights(self):
        with pytest.raises(ValueError):
            analyze_allometric_scaling([], [], "CL")

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            analyze_allometric_scaling([1.0, 2.0], [0.1], "CL")

    def test_single_species(self):
        with pytest.raises(ValueError, match="2 species"):
            analyze_allometric_scaling([1.0], [0.5], "CL")

    def test_negative_body_weight(self):
        with pytest.raises(ValueError, match="species_weights_kg"):
            analyze_allometric_scaling([-1.0, 2.0], [0.1, 0.2], "CL")

    def test_zero_body_weight(self):
        with pytest.raises(ValueError, match="species_weights_kg"):
            analyze_allometric_scaling([0.0, 2.0], [0.1, 0.2], "CL")

    def test_negative_param_value(self):
        with pytest.raises(ValueError, match="species_params"):
            analyze_allometric_scaling([1.0, 2.0], [-0.1, 0.2], "CL")

    def test_zero_param_value(self):
        with pytest.raises(ValueError, match="species_params"):
            analyze_allometric_scaling([1.0, 2.0], [0.0, 0.2], "CL")

    def test_negative_human_bw(self):
        with pytest.raises(ValueError, match="human_bw_kg"):
            analyze_allometric_scaling([1.0, 2.0], [0.1, 0.2], "CL", human_bw_kg=-70.0)


# ---------------------------------------------------------------------------
# Regression quality tests
# ---------------------------------------------------------------------------

class TestAllometricRegressionQuality:
    def test_perfect_075_exponent(self):
        """With perfect CL data b=0.75, regression should recover ~0.75."""
        bw = SPECIES_BW
        cl = [_cl_075(w) for w in bw]
        result = analyze_allometric_scaling(bw, cl, "CL")
        assert result.b_exponent == pytest.approx(0.75, abs=0.02)

    def test_perfect_075_r_squared_near_1(self):
        """Perfect power-law data → R² should be ~1.0."""
        bw = SPECIES_BW
        cl = [_cl_075(w) for w in bw]
        result = analyze_allometric_scaling(bw, cl, "CL")
        assert result.r_squared >= 0.999

    def test_perfect_vd_exponent(self):
        """With perfect Vd data b=1.0, regression should recover ~1.0."""
        bw = SPECIES_BW
        vd = [_vd_10(w) for w in bw]
        result = analyze_allometric_scaling(bw, vd, "Vd")
        assert result.b_exponent == pytest.approx(1.0, abs=0.02)

    def test_human_prediction_near_truth(self):
        """Predicted human CL should be close to true value for perfect data."""
        bw = SPECIES_BW
        cl = [_cl_075(w) for w in bw]
        true_human = _cl_075(70.0)
        result = analyze_allometric_scaling(bw, cl, "CL")
        assert result.predicted_human_value == pytest.approx(true_human, rel=0.05)

    def test_two_species_works(self):
        """Two species should be sufficient for regression."""
        result = analyze_allometric_scaling([0.25, 10.0], [0.05, 0.8], "CL")
        assert isinstance(result, AllometricScalingResult)
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)  # 2 pts → perfect fit

    def test_coefficient_positive(self):
        bw = SPECIES_BW
        cl = [_cl_075(w) for w in bw]
        result = analyze_allometric_scaling(bw, cl, "CL")
        assert result.a_coefficient > 0

    def test_result_type(self):
        result = analyze_allometric_scaling([0.25, 10.0], [0.05, 0.8], "CL")
        assert isinstance(result, AllometricScalingResult)

    def test_param_name_stored(self):
        result = analyze_allometric_scaling([0.25, 10.0], [0.05, 0.8], "MyParam")
        assert result.param_name == "MyParam"

    def test_species_data_stored(self):
        bw = [0.25, 10.0]
        cl = [0.05, 0.8]
        result = analyze_allometric_scaling(bw, cl, "CL")
        assert result.species_weights_kg == bw
        assert result.species_params == cl


# ---------------------------------------------------------------------------
# Expected exponent inference tests
# ---------------------------------------------------------------------------

class TestExpectedExponent:
    def test_cl_expected_exponent(self):
        result = analyze_allometric_scaling([0.25, 10.0], [0.05, 0.8], "CL")
        assert result.expected_exponent == pytest.approx(0.75)

    def test_clearance_expected_exponent(self):
        result = analyze_allometric_scaling([0.25, 10.0], [0.05, 0.8], "clearance")
        assert result.expected_exponent == pytest.approx(0.75)

    def test_vd_expected_exponent(self):
        result = analyze_allometric_scaling([0.25, 10.0], [0.1, 4.0], "Vd")
        assert result.expected_exponent == pytest.approx(1.0)

    def test_thalf_expected_exponent(self):
        result = analyze_allometric_scaling([0.25, 10.0], [0.3, 1.0], "t_half")
        assert result.expected_exponent == pytest.approx(0.25)

    def test_halflife_expected_exponent(self):
        result = analyze_allometric_scaling([0.25, 10.0], [0.3, 1.0], "half_life")
        assert result.expected_exponent == pytest.approx(0.25)

    def test_consistent_assessment_for_good_fit(self):
        """Good fit → exponent_assessment should contain 'consistent'."""
        bw = SPECIES_BW
        cl = [_cl_075(w) for w in bw]
        result = analyze_allometric_scaling(bw, cl, "CL")
        assert "consistent" in result.exponent_assessment

    def test_notes_populated(self):
        result = analyze_allometric_scaling([0.25, 10.0], [0.05, 0.8], "CL")
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# predict_human_from_animals tests
# ---------------------------------------------------------------------------

class TestPredictHumanFromAnimals:
    def _make_species_data(self, bw_list, cl_list):
        return [
            {"species": f"sp{i}", "bw_kg": bw_list[i], "cl_L_per_h": cl_list[i]}
            for i in range(len(bw_list))
        ]

    def test_empty_species_data_raises(self):
        with pytest.raises(ValueError):
            predict_human_from_animals([], "cl_L_per_h")

    def test_invalid_target_param_raises(self):
        data = self._make_species_data([0.25, 10.0], [0.05, 0.8])
        with pytest.raises(ValueError, match="target_param"):
            predict_human_from_animals(data, "unknown_param")

    def test_missing_bw_key_raises(self):
        data = [{"species": "rat", "cl_L_per_h": 0.5}]
        with pytest.raises(ValueError, match="bw_kg"):
            predict_human_from_animals(data, "cl_L_per_h")

    def test_missing_param_key_raises(self):
        data = [{"species": "rat", "bw_kg": 0.25}]
        with pytest.raises(ValueError, match="cl_L_per_h"):
            predict_human_from_animals(data, "cl_L_per_h")

    def test_single_species_raises(self):
        data = [{"bw_kg": 0.25, "cl_L_per_h": 0.05}]
        with pytest.raises(ValueError, match="2 species"):
            predict_human_from_animals(data, "cl_L_per_h")

    def test_returns_dict(self):
        bw = [0.25, 3.5, 10.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        result = predict_human_from_animals(data, "cl_L_per_h")
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        bw = [0.25, 3.5, 10.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        result = predict_human_from_animals(data, "cl_L_per_h")
        for key in ("predicted_value", "param_name", "a_coefficient", "b_exponent",
                    "r_squared", "ci_lower", "ci_upper", "n_species",
                    "exponent_assessment", "notes"):
            assert key in result

    def test_predicted_value_positive(self):
        bw = [0.25, 3.5, 10.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        result = predict_human_from_animals(data, "cl_L_per_h")
        assert result["predicted_value"] > 0

    def test_ci_lower_less_than_upper(self):
        bw = [0.25, 3.5, 10.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        result = predict_human_from_animals(data, "cl_L_per_h")
        assert result["ci_lower"] < result["ci_upper"]

    def test_predicted_value_within_ci(self):
        bw = [0.25, 3.5, 10.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        result = predict_human_from_animals(data, "cl_L_per_h")
        assert result["ci_lower"] <= result["predicted_value"] <= result["ci_upper"]

    def test_human_cl_prediction_near_truth(self):
        """For perfect b=0.75 data, predicted human CL should be close to truth."""
        bw = [0.02, 0.25, 3.5, 10.0, 30.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        result = predict_human_from_animals(data, "cl_L_per_h")
        true_human = _cl_075(70.0)
        assert result["predicted_value"] == pytest.approx(true_human, rel=0.1)

    def test_n_species_correct(self):
        bw = [0.25, 3.5, 10.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        result = predict_human_from_animals(data, "cl_L_per_h")
        assert result["n_species"] == 3

    def test_vd_prediction(self):
        """Test Vd prediction with b~1.0 data."""
        bw = [0.25, 3.5, 10.0]
        vd = [_vd_10(w) for w in bw]
        data = [
            {"bw_kg": bw[i], "vd_L": vd[i]} for i in range(len(bw))
        ]
        result = predict_human_from_animals(data, "vd_L")
        true_human = _vd_10(70.0)
        assert result["predicted_value"] == pytest.approx(true_human, rel=0.1)

    def test_thalf_prediction(self):
        """Test t_half prediction."""
        bw = [0.25, 3.5, 10.0]
        t_half = [0.5 * (w ** 0.25) for w in bw]
        data = [
            {"bw_kg": bw[i], "t_half_h": t_half[i]} for i in range(len(bw))
        ]
        result = predict_human_from_animals(data, "t_half_h")
        assert result["predicted_value"] > 0

    def test_ci_lower_nonnegative(self):
        """CI lower bound should be >= 0."""
        bw = [0.25, 3.5, 10.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        result = predict_human_from_animals(data, "cl_L_per_h")
        assert result["ci_lower"] >= 0.0

    def test_negative_human_bw_raises(self):
        bw = [0.25, 10.0]
        cl = [_cl_075(w) for w in bw]
        data = self._make_species_data(bw, cl)
        with pytest.raises(ValueError, match="human_bw_kg"):
            predict_human_from_animals(data, "cl_L_per_h", human_bw_kg=-70.0)
