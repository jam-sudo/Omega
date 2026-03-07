"""Tests for Phase 349 — Bioequivalence Prediction from Formulation Parameters."""

import math

import pytest

from omega_pbpk.clinical.be_formulation_predictor import (
    BEFormulationResult,
    predict_be_formulation,
)

# --- Helpers ---


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        auc_test_mg_h_per_L=100.0,
        auc_ref_mg_h_per_L=100.0,
        cmax_test_mg_L=10.0,
        cmax_ref_mg_L=10.0,
        tmax_test_h=2.0,
        tmax_ref_h=2.0,
        n_subjects=24,
        cv_intra_pct=20.0,
    )
    defaults.update(kwargs)
    return predict_be_formulation(**defaults)


# --- Return type and structure ---


def test_return_type():
    result = make_result()
    assert isinstance(result, BEFormulationResult)


def test_result_is_frozen():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.gmr_auc = 99.0  # type: ignore[misc]


def test_be_auc_is_bool():
    result = make_result()
    assert isinstance(result.be_auc, bool)


def test_be_cmax_is_bool():
    result = make_result()
    assert isinstance(result.be_cmax, bool)


# --- GMR correctness ---


def test_gmr_auc_exact():
    result = predict_be_formulation(
        drug_name="Drug",
        auc_test_mg_h_per_L=120.0,
        auc_ref_mg_h_per_L=100.0,
        cmax_test_mg_L=10.0,
        cmax_ref_mg_L=10.0,
        tmax_test_h=2.0,
        tmax_ref_h=2.0,
    )
    assert result.gmr_auc == pytest.approx(1.20, rel=1e-9)


def test_gmr_cmax_exact():
    result = predict_be_formulation(
        drug_name="Drug",
        auc_test_mg_h_per_L=100.0,
        auc_ref_mg_h_per_L=100.0,
        cmax_test_mg_L=8.0,
        cmax_ref_mg_L=10.0,
        tmax_test_h=2.0,
        tmax_ref_h=2.0,
    )
    assert result.gmr_cmax == pytest.approx(0.80, rel=1e-9)


# --- Identical formulations -> bioequivalent ---


def test_identical_formulations_bioequivalent():
    result = make_result()
    assert result.be_conclusion == "bioequivalent"
    assert result.be_auc is True
    assert result.be_cmax is True


# --- GMR = 1.30 -> not_bioequivalent ---


def test_gmr_130_not_bioequivalent():
    result = predict_be_formulation(
        drug_name="Drug",
        auc_test_mg_h_per_L=130.0,
        auc_ref_mg_h_per_L=100.0,
        cmax_test_mg_L=130.0,
        cmax_ref_mg_L=100.0,
        tmax_test_h=2.0,
        tmax_ref_h=2.0,
        cv_intra_pct=10.0,
        n_subjects=24,
    )
    assert result.be_conclusion == "not_bioequivalent"


# --- CI bounds ---


def test_ci_lower_less_than_gmr_less_than_ci_upper():
    result = make_result(auc_test_mg_h_per_L=110.0, cmax_test_mg_L=11.0)
    assert result.ci90_auc_lower < result.gmr_auc < result.ci90_auc_upper
    assert result.ci90_cmax_lower < result.gmr_cmax < result.ci90_cmax_upper


def test_ci_symmetric_in_log_space():
    result = make_result()
    # log(GMR) should be midpoint of log CI
    log_mid_auc = 0.5 * (math.log(result.ci90_auc_lower) + math.log(result.ci90_auc_upper))
    assert log_mid_auc == pytest.approx(math.log(result.gmr_auc), abs=1e-10)


# --- Power estimate ---


def test_power_estimate_in_0_100():
    result = make_result()
    assert 0.0 <= result.power_estimate_pct <= 100.0


def test_power_estimate_increases_with_n():
    r_small = make_result(n_subjects=6)
    r_large = make_result(n_subjects=100)
    assert r_large.power_estimate_pct >= r_small.power_estimate_pct


# --- Required n ---


def test_required_n_positive():
    result = make_result()
    assert result.required_n_for_80pct_power > 0


def test_required_n_is_int():
    result = make_result()
    assert isinstance(result.required_n_for_80pct_power, int)


def test_required_n_increases_with_cv():
    r_low = make_result(cv_intra_pct=10.0)
    r_high = make_result(cv_intra_pct=40.0)
    assert r_high.required_n_for_80pct_power > r_low.required_n_for_80pct_power


# --- Higher CV -> wider CI ---


def test_higher_cv_wider_ci():
    r_low = make_result(cv_intra_pct=10.0)
    r_high = make_result(cv_intra_pct=40.0)
    width_low = r_low.ci90_auc_upper - r_low.ci90_auc_lower
    width_high = r_high.ci90_auc_upper - r_high.ci90_auc_lower
    assert width_high > width_low


def test_higher_cv_more_likely_not_be():
    # Formulation with slight difference: large CV should fail
    r_low = predict_be_formulation(
        drug_name="D",
        auc_test_mg_h_per_L=122.0,
        auc_ref_mg_h_per_L=100.0,
        cmax_test_mg_L=122.0,
        cmax_ref_mg_L=100.0,
        tmax_test_h=1.0,
        tmax_ref_h=1.0,
        cv_intra_pct=5.0,
        n_subjects=24,
    )
    r_high = predict_be_formulation(
        drug_name="D",
        auc_test_mg_h_per_L=122.0,
        auc_ref_mg_h_per_L=100.0,
        cmax_test_mg_L=122.0,
        cmax_ref_mg_L=100.0,
        tmax_test_h=1.0,
        tmax_ref_h=1.0,
        cv_intra_pct=50.0,
        n_subjects=24,
    )
    # High CV -> CI more likely to exceed bounds
    assert r_high.ci90_auc_upper >= r_low.ci90_auc_upper


# --- Tmax difference ---


def test_tmax_diff_calculated_correctly():
    result = predict_be_formulation(
        drug_name="D",
        auc_test_mg_h_per_L=100.0,
        auc_ref_mg_h_per_L=100.0,
        cmax_test_mg_L=10.0,
        cmax_ref_mg_L=10.0,
        tmax_test_h=3.5,
        tmax_ref_h=1.5,
    )
    assert result.tmax_diff_h == pytest.approx(2.0, abs=1e-9)


# --- Validation errors ---


def test_validation_auc_test_le_zero():
    with pytest.raises(ValueError, match="auc_test"):
        make_result(auc_test_mg_h_per_L=0.0)


def test_validation_auc_ref_le_zero():
    with pytest.raises(ValueError, match="auc_ref"):
        make_result(auc_ref_mg_h_per_L=-5.0)


def test_validation_n_subjects_lt_2():
    with pytest.raises(ValueError, match="n_subjects"):
        make_result(n_subjects=1)


def test_validation_cv_le_zero():
    with pytest.raises(ValueError, match="cv_intra_pct"):
        make_result(cv_intra_pct=0.0)


def test_validation_cmax_test_le_zero():
    with pytest.raises(ValueError):
        make_result(cmax_test_mg_L=0.0)


def test_validation_cmax_ref_le_zero():
    with pytest.raises(ValueError):
        make_result(cmax_ref_mg_L=-1.0)
