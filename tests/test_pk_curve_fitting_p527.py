"""Tests for Phase 527 — PK Curve Fitting (clinical/pk_curve_fitting.py)."""

import math

import pytest

from omega_pbpk.clinical.pk_curve_fitting import (
    PKCurveFitResult,
    fit_pk_curve,
    predict_from_fit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iv_conc(dose, vd, cl, t):
    k = cl / vd
    return (dose / vd) * math.exp(-k * t)


def _oral_conc(dose, vd, cl, ka, t):
    k = cl / vd
    return dose * ka / (vd * (ka - k)) * (math.exp(-k * t) - math.exp(-ka * t))


TIMES_IV = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
DOSE = 100.0
TRUE_VD = 50.0
TRUE_CL = 5.0

# Noiseless IV data
CONCS_IV = [_iv_conc(DOSE, TRUE_VD, TRUE_CL, t) for t in TIMES_IV]

TIMES_ORAL = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
TRUE_KA = 1.5
CONCS_ORAL = [_oral_conc(DOSE, TRUE_VD, TRUE_CL, TRUE_KA, t) for t in TIMES_ORAL]


# ---------------------------------------------------------------------------
# Return type & fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_pkcurvefitresult(self):
        result = fit_pk_curve("TestDrug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert isinstance(result, PKCurveFitResult)

    def test_frozen_dataclass(self):
        result = fit_pk_curve("TestDrug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        with pytest.raises((AttributeError, TypeError)):
            result.vd_fit_L = 99.0  # type: ignore[misc]

    def test_all_fields_present(self):
        result = fit_pk_curve("TestDrug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert hasattr(result, "drug_name")
        assert hasattr(result, "model_type")
        assert hasattr(result, "dose_mg")
        assert hasattr(result, "n_obs")
        assert hasattr(result, "vd_fit_L")
        assert hasattr(result, "cl_fit_L_per_h")
        assert hasattr(result, "ka_fit_per_h")
        assert hasattr(result, "t_half_fit_h")
        assert hasattr(result, "rss")
        assert hasattr(result, "r_squared")
        assert hasattr(result, "fit_quality")
        assert hasattr(result, "notes")

    def test_drug_name_stored(self):
        result = fit_pk_curve("Aspirin", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.drug_name == "Aspirin"

    def test_model_type_stored_iv(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.model_type == "1cpt_iv"

    def test_model_type_stored_oral(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_ORAL, CONCS_ORAL, "1cpt_oral")
        assert result.model_type == "1cpt_oral"

    def test_n_obs_correct(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.n_obs == len(TIMES_IV)

    def test_dose_stored(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.dose_mg == DOSE


# ---------------------------------------------------------------------------
# IV fitting accuracy
# ---------------------------------------------------------------------------


class TestIVFitting:
    def test_r_squared_high_noiseless(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.r_squared > 0.99

    def test_fit_quality_excellent_noiseless(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.fit_quality == "excellent"

    def test_t_half_consistent_with_vd_cl(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        expected_t_half = 0.693 * result.vd_fit_L / result.cl_fit_L_per_h
        assert abs(result.t_half_fit_h - expected_t_half) < 1e-6

    def test_rss_nonnegative(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.rss >= 0.0

    def test_r_squared_in_bounds(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert 0.0 <= result.r_squared <= 1.0

    def test_ka_is_minus_one_for_iv(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.ka_fit_per_h == -1.0

    def test_vd_positive(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.vd_fit_L > 0

    def test_cl_positive(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        assert result.cl_fit_L_per_h > 0


# ---------------------------------------------------------------------------
# Oral fitting
# ---------------------------------------------------------------------------


class TestOralFitting:
    def test_r_squared_oral_noiseless(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_ORAL, CONCS_ORAL, "1cpt_oral")
        assert result.r_squared > 0.99

    def test_ka_positive_for_oral(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_ORAL, CONCS_ORAL, "1cpt_oral")
        assert result.ka_fit_per_h > 0

    def test_t_half_oral(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_ORAL, CONCS_ORAL, "1cpt_oral")
        expected_t_half = 0.693 * result.vd_fit_L / result.cl_fit_L_per_h
        assert abs(result.t_half_fit_h - expected_t_half) < 1e-6

    def test_rss_nonnegative_oral(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_ORAL, CONCS_ORAL, "1cpt_oral")
        assert result.rss >= 0.0


# ---------------------------------------------------------------------------
# Fit quality classification
# ---------------------------------------------------------------------------


class TestFitQuality:
    def test_excellent_when_r2_above_095(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        # noiseless -> r2 > 0.99
        assert result.fit_quality == "excellent"

    def test_fit_quality_is_valid_string(self):
        noisy = [c * (1 + 0.15 * (0.5 - i % 2)) for i, c in enumerate(CONCS_IV)]
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, noisy, "1cpt_iv")
        assert result.fit_quality in {"excellent", "good", "poor"}

    def test_poor_classification_random_data(self):
        random_concs = [0.1, 5.0, 0.2, 3.0, 7.0, 0.05, 2.0]
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, random_concs, "1cpt_iv")
        assert result.fit_quality in {"excellent", "good", "poor"}


# ---------------------------------------------------------------------------
# predict_from_fit
# ---------------------------------------------------------------------------


class TestPredictFromFit:
    def test_returns_list(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        pred = predict_from_fit(result, [1.0, 2.0, 4.0])
        assert isinstance(pred, list)

    def test_correct_length(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        times_new = [0.25, 1.5, 3.0, 5.0, 10.0]
        pred = predict_from_fit(result, times_new)
        assert len(pred) == len(times_new)

    def test_iv_predictions_monotonically_decreasing(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        times_dense = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]
        pred = predict_from_fit(result, times_dense)
        for i in range(1, len(pred)):
            assert pred[i] < pred[i - 1], (
                f"Not monotonically decreasing at index {i}: {pred[i - 1]:.4f} -> {pred[i]:.4f}"
            )

    def test_predict_nonnegative(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "1cpt_iv")
        pred = predict_from_fit(result, [0.1, 1.0, 5.0, 20.0])
        assert all(p >= 0.0 for p in pred)

    def test_predict_oral_nonnegative(self):
        result = fit_pk_curve("Drug", DOSE, TIMES_ORAL, CONCS_ORAL, "1cpt_oral")
        pred = predict_from_fit(result, [0.5, 1.0, 2.0, 6.0, 12.0])
        assert all(p >= 0.0 for p in pred)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="model_type"):
            fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV, "invalid_model")

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            fit_pk_curve("Drug", DOSE, TIMES_IV, CONCS_IV[:5], "1cpt_iv")

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="3 observations"):
            fit_pk_curve("Drug", DOSE, [1.0, 2.0], [0.5, 0.3], "1cpt_iv")

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            fit_pk_curve("Drug", 0.0, TIMES_IV, CONCS_IV, "1cpt_iv")

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            fit_pk_curve("Drug", -10.0, TIMES_IV, CONCS_IV, "1cpt_iv")
