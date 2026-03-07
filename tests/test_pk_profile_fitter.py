"""Tests for Phase 320 — Concentration-Time Profile Fitter."""

import math

import pytest

from omega_pbpk.clinical.pk_profile_fitter import (
    PKFitResult,
    fit_one_compartment,
    predict_from_fitted,
)

# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _iv_concs(times, dose, cl, vd):
    ke = cl / vd
    return [dose / vd * math.exp(-ke * t) for t in times]


def _oral_concs(times, dose, cl, vd, ka):
    ke = cl / vd
    denom = vd * (ka - ke)
    scale = dose * ka / denom
    return [max(0.0, scale * (math.exp(-ke * t) - math.exp(-ka * t))) for t in times]


TIMES_IV = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
TIMES_ORAL = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]

TRUE_CL = 5.0  # L/h
TRUE_VD = 50.0  # L
TRUE_KA = 1.0  # /h
DOSE = 100.0  # mg

CONCS_IV = _iv_concs(TIMES_IV, DOSE, TRUE_CL, TRUE_VD)
CONCS_ORAL = _oral_concs(TIMES_ORAL, DOSE, TRUE_CL, TRUE_VD, TRUE_KA)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_iv_return_type():
    result = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert isinstance(result, PKFitResult)


def test_oral_return_type():
    result = fit_one_compartment("DrugB", TIMES_ORAL, CONCS_ORAL, DOSE, "oral")
    assert isinstance(result, PKFitResult)


# ---------------------------------------------------------------------------
# fitted_cl > 0, fitted_vd > 0
# ---------------------------------------------------------------------------


def test_fitted_cl_positive_iv():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.fitted_cl_L_per_h > 0.0


def test_fitted_vd_positive_iv():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.fitted_vd_L > 0.0


def test_fitted_cl_positive_oral():
    r = fit_one_compartment("DrugB", TIMES_ORAL, CONCS_ORAL, DOSE, "oral")
    assert r.fitted_cl_L_per_h > 0.0


def test_fitted_vd_positive_oral():
    r = fit_one_compartment("DrugB", TIMES_ORAL, CONCS_ORAL, DOSE, "oral")
    assert r.fitted_vd_L > 0.0


# ---------------------------------------------------------------------------
# r_squared in [-1, 1]
# ---------------------------------------------------------------------------


def test_r_squared_range_iv():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert -1.0 <= r.r_squared <= 1.0


def test_r_squared_range_oral():
    r = fit_one_compartment("DrugB", TIMES_ORAL, CONCS_ORAL, DOSE, "oral")
    assert -1.0 <= r.r_squared <= 1.0


def test_r_squared_good_fit_iv():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.r_squared > 0.5  # should be a reasonable fit


def test_r_squared_good_fit_oral():
    r = fit_one_compartment("DrugB", TIMES_ORAL, CONCS_ORAL, DOSE, "oral")
    assert r.r_squared > 0.5


# ---------------------------------------------------------------------------
# Fitted parameters close to true values (within 3x for IV)
# ---------------------------------------------------------------------------


def test_fitted_cl_within_3x_iv():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert TRUE_CL / 3 <= r.fitted_cl_L_per_h <= TRUE_CL * 3


def test_fitted_vd_within_3x_iv():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert TRUE_VD / 3 <= r.fitted_vd_L <= TRUE_VD * 3


def test_fitted_ke_positive():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.fitted_ke_per_h > 0.0


def test_fitted_t_half_positive():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.fitted_t_half_h > 0.0


# ---------------------------------------------------------------------------
# IV route: ka should be None
# ---------------------------------------------------------------------------


def test_iv_ka_is_none():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.fitted_ka_per_h is None


# ---------------------------------------------------------------------------
# Oral route: ka should not be None
# ---------------------------------------------------------------------------


def test_oral_ka_not_none():
    r = fit_one_compartment("DrugB", TIMES_ORAL, CONCS_ORAL, DOSE, "oral")
    assert r.fitted_ka_per_h is not None
    assert r.fitted_ka_per_h > 0.0


# ---------------------------------------------------------------------------
# predict_from_fitted returns list of correct length
# ---------------------------------------------------------------------------


def test_predict_from_fitted_iv_length():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    pred = predict_from_fitted(r, [0.0, 5.0, 10.0, 20.0])
    assert len(pred) == 4


def test_predict_from_fitted_oral_length():
    r = fit_one_compartment("DrugB", TIMES_ORAL, CONCS_ORAL, DOSE, "oral")
    pred = predict_from_fitted(r, [0.0, 2.0, 6.0, 12.0, 24.0])
    assert len(pred) == 5


def test_predict_from_fitted_returns_list():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    pred = predict_from_fitted(r, TIMES_IV)
    assert isinstance(pred, list)


# ---------------------------------------------------------------------------
# n_observations == len(times_h)
# ---------------------------------------------------------------------------


def test_n_observations_iv():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.n_observations == len(TIMES_IV)


def test_n_observations_oral():
    r = fit_one_compartment("DrugB", TIMES_ORAL, CONCS_ORAL, DOSE, "oral")
    assert r.n_observations == len(TIMES_ORAL)


# ---------------------------------------------------------------------------
# AUC values are non-negative
# ---------------------------------------------------------------------------


def test_auc_obs_nonnegative():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.auc_obs_mg_h_per_L >= 0.0


def test_auc_pred_nonnegative():
    r = fit_one_compartment("DrugA", TIMES_IV, CONCS_IV, DOSE, "iv_bolus")
    assert r.auc_pred_mg_h_per_L >= 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_too_few_points():
    with pytest.raises(ValueError, match="at least 3"):
        fit_one_compartment("D", [0.0, 1.0], [1.0, 0.5], 100.0, "iv_bolus")


def test_validation_mismatched_lengths():
    with pytest.raises(ValueError, match="len"):
        fit_one_compartment("D", [0.0, 1.0, 2.0], [1.0, 0.5], 100.0, "iv_bolus")


def test_validation_negative_time():
    with pytest.raises(ValueError, match="times"):
        fit_one_compartment("D", [-1.0, 0.0, 1.0], [2.0, 1.0, 0.5], 100.0, "iv_bolus")


def test_validation_negative_concentration():
    with pytest.raises(ValueError, match="concentrations"):
        fit_one_compartment("D", [0.0, 1.0, 2.0], [-1.0, 0.5, 0.2], 100.0, "iv_bolus")


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        fit_one_compartment("D", TIMES_IV, CONCS_IV, 0.0, "iv_bolus")


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        fit_one_compartment("D", TIMES_IV, CONCS_IV, DOSE, "intramuscular")
