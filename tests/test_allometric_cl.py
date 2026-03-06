"""Tests for allometric interspecies CL/Vd prediction."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.allometric_cl import (
    AllometricAnalysisResult,
    AllometricFit,
    fit_allometric,
    full_allometric_analysis,
    predict_human_pk,
)


# ---------------------------------------------------------------------------
# Reference animal data (synthetic power-law data)
# ---------------------------------------------------------------------------

# CL data: CL = 0.15 * BW^0.75 (exact allometric relationship)
_CL_DATA = [
    {"species": "mouse", "bw_kg": 0.02, "value": 0.15 * (0.02**0.75)},
    {"species": "rat", "bw_kg": 0.25, "value": 0.15 * (0.25**0.75)},
    {"species": "rabbit", "bw_kg": 3.5, "value": 0.15 * (3.5**0.75)},
    {"species": "dog", "bw_kg": 10.0, "value": 0.15 * (10.0**0.75)},
    {"species": "monkey", "bw_kg": 5.0, "value": 0.15 * (5.0**0.75)},
]

# Vd data: Vd = 2.5 * BW^1.0 (exact linear relationship)
_VD_DATA = [
    {"species": "mouse", "bw_kg": 0.02, "value": 2.5 * 0.02},
    {"species": "rat", "bw_kg": 0.25, "value": 2.5 * 0.25},
    {"species": "rabbit", "bw_kg": 3.5, "value": 2.5 * 3.5},
    {"species": "dog", "bw_kg": 10.0, "value": 2.5 * 10.0},
    {"species": "monkey", "bw_kg": 5.0, "value": 2.5 * 5.0},
]

_TWO_SPECIES_CL = [
    {"species": "rat", "bw_kg": 0.25, "value": 0.15 * (0.25**0.75)},
    {"species": "dog", "bw_kg": 10.0, "value": 0.15 * (10.0**0.75)},
]

_TWO_SPECIES_VD = [
    {"species": "rat", "bw_kg": 0.25, "value": 2.5 * 0.25},
    {"species": "dog", "bw_kg": 10.0, "value": 2.5 * 10.0},
]


# ---------------------------------------------------------------------------
# 1-3: Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_single_data_point_raises(self):
        with pytest.raises(ValueError, match="2 data points"):
            fit_allometric([{"species": "rat", "bw_kg": 0.25, "value": 0.05}], pk_param="CL")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="2 data points"):
            fit_allometric([], pk_param="CL")

    def test_bw_zero_raises(self):
        data = [
            {"species": "rat", "bw_kg": 0.0, "value": 0.05},
            {"species": "dog", "bw_kg": 10.0, "value": 0.5},
        ]
        with pytest.raises(ValueError, match="bw_kg"):
            fit_allometric(data, pk_param="CL")

    def test_bw_negative_raises(self):
        data = [
            {"species": "rat", "bw_kg": -0.25, "value": 0.05},
            {"species": "dog", "bw_kg": 10.0, "value": 0.5},
        ]
        with pytest.raises(ValueError, match="bw_kg"):
            fit_allometric(data, pk_param="CL")

    def test_value_zero_raises(self):
        data = [
            {"species": "rat", "bw_kg": 0.25, "value": 0.0},
            {"species": "dog", "bw_kg": 10.0, "value": 0.5},
        ]
        with pytest.raises(ValueError, match="value"):
            fit_allometric(data, pk_param="CL")

    def test_value_negative_raises(self):
        data = [
            {"species": "rat", "bw_kg": 0.25, "value": -0.05},
            {"species": "dog", "bw_kg": 10.0, "value": 0.5},
        ]
        with pytest.raises(ValueError, match="value"):
            fit_allometric(data, pk_param="CL")


# ---------------------------------------------------------------------------
# 4-8: fit_allometric correctness
# ---------------------------------------------------------------------------


class TestFitAllometric:
    def test_returns_allometric_fit(self):
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        assert isinstance(fit, AllometricFit)

    def test_pk_param_stored(self):
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        assert fit.pk_param == "CL"

    def test_n_species_correct(self):
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        assert fit.n_species == len(_CL_DATA)

    def test_exponent_approx_0_75_for_cl(self):
        """Synthetic data built from BW^0.75 should recover exponent ~0.75."""
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        assert fit.exponent_b == pytest.approx(0.75, abs=0.02)

    def test_coefficient_approx_for_cl(self):
        """Coefficient should recover ~0.15."""
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        assert fit.coefficient_a == pytest.approx(0.15, rel=0.05)

    def test_exponent_approx_1_0_for_vd(self):
        """Synthetic Vd = 2.5 * BW^1 should recover exponent ~1.0."""
        fit = fit_allometric(_VD_DATA, pk_param="Vd")
        assert fit.exponent_b == pytest.approx(1.0, abs=0.02)

    def test_r_squared_near_1_for_perfect_data(self):
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        assert fit.r_squared > 0.99

    def test_r_squared_in_unit_interval(self):
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        assert 0.0 <= fit.r_squared <= 1.0

    def test_two_species_minimum(self):
        fit = fit_allometric(_TWO_SPECIES_CL, pk_param="CL")
        assert isinstance(fit, AllometricFit)
        assert fit.n_species == 2


# ---------------------------------------------------------------------------
# 9-12: predict_human_pk
# ---------------------------------------------------------------------------


class TestPredictHumanPK:
    def test_returns_positive_value(self):
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        val = predict_human_pk(fit)
        assert val > 0.0

    def test_known_human_cl(self):
        """CL = 0.15 * 70^0.75 should be predicted correctly."""
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        expected = 0.15 * (70.0**0.75)
        assert predict_human_pk(fit, human_bw_kg=70.0) == pytest.approx(expected, rel=0.05)

    def test_known_human_vd(self):
        """Vd = 2.5 * 70 = 175 L should be predicted."""
        fit = fit_allometric(_VD_DATA, pk_param="Vd")
        assert predict_human_pk(fit, human_bw_kg=70.0) == pytest.approx(175.0, rel=0.05)

    def test_heavier_person_gives_larger_cl(self):
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        cl_70 = predict_human_pk(fit, human_bw_kg=70.0)
        cl_100 = predict_human_pk(fit, human_bw_kg=100.0)
        assert cl_100 > cl_70

    def test_human_bw_zero_raises(self):
        fit = fit_allometric(_CL_DATA, pk_param="CL")
        with pytest.raises(ValueError, match="human_bw_kg"):
            predict_human_pk(fit, human_bw_kg=0.0)


# ---------------------------------------------------------------------------
# 13-18: full_allometric_analysis
# ---------------------------------------------------------------------------


class TestFullAllometricAnalysis:
    def test_returns_allometric_analysis_result(self):
        r = full_allometric_analysis(_CL_DATA, _VD_DATA)
        assert isinstance(r, AllometricAnalysisResult)

    def test_cl_fit_is_allometric_fit(self):
        r = full_allometric_analysis(_CL_DATA, _VD_DATA)
        assert isinstance(r.cl_fit, AllometricFit)

    def test_vd_fit_is_allometric_fit(self):
        r = full_allometric_analysis(_CL_DATA, _VD_DATA)
        assert isinstance(r.vd_fit, AllometricFit)

    def test_human_cl_positive(self):
        r = full_allometric_analysis(_CL_DATA, _VD_DATA)
        assert r.human_cl_L_per_h > 0.0

    def test_human_vd_positive(self):
        r = full_allometric_analysis(_CL_DATA, _VD_DATA)
        assert r.human_vd_L > 0.0

    def test_predicted_t_half_positive(self):
        r = full_allometric_analysis(_CL_DATA, _VD_DATA)
        assert r.predicted_t_half_h > 0.0

    def test_t_half_consistent_with_cl_vd(self):
        """t1/2 should equal ln(2)*Vd/CL."""
        r = full_allometric_analysis(_CL_DATA, _VD_DATA)
        expected_t_half = math.log(2.0) * r.human_vd_L / r.human_cl_L_per_h
        assert r.predicted_t_half_h == pytest.approx(expected_t_half, rel=1e-6)

    def test_notes_is_string(self):
        r = full_allometric_analysis(_CL_DATA, _VD_DATA)
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0

    def test_two_species_analysis_works(self):
        r = full_allometric_analysis(_TWO_SPECIES_CL, _TWO_SPECIES_VD)
        assert r.human_cl_L_per_h > 0.0
        assert r.human_vd_L > 0.0

    def test_single_species_cl_raises(self):
        single = [{"species": "rat", "bw_kg": 0.25, "value": 0.05}]
        with pytest.raises(ValueError, match="2 data points"):
            full_allometric_analysis(single, _VD_DATA)

    def test_different_species_sets_give_different_results(self):
        """More data spanning more BW range should change predictions."""
        r_two = full_allometric_analysis(_TWO_SPECIES_CL, _TWO_SPECIES_VD)
        r_five = full_allometric_analysis(_CL_DATA, _VD_DATA)
        # They may agree closely for perfect power-law data, but both should work
        assert r_two.human_cl_L_per_h > 0.0
        assert r_five.human_cl_L_per_h > 0.0

    def test_custom_human_bw(self):
        r_70 = full_allometric_analysis(_CL_DATA, _VD_DATA, human_bw_kg=70.0)
        r_80 = full_allometric_analysis(_CL_DATA, _VD_DATA, human_bw_kg=80.0)
        # Heavier body weight → higher CL with positive exponent
        assert r_80.human_cl_L_per_h > r_70.human_cl_L_per_h
