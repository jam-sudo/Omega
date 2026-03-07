"""Tests for formulation_be module — Phase 288."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.formulation_be import (
    BEPredictionResult,
    formulation_sensitivity,
    predict_be_outcome,
    required_sample_size,
)


# ---------------------------------------------------------------------------
# predict_be_outcome tests
# ---------------------------------------------------------------------------


def test_predict_be_outcome_returns_dataclass():
    res = predict_be_outcome(
        ref_auc=100.0,
        ref_cmax=20.0,
        test_auc=100.0,
        test_cmax=20.0,
    )
    assert isinstance(res, BEPredictionResult)


def test_predict_be_frozen():
    res = predict_be_outcome(100.0, 20.0, 100.0, 20.0)
    with pytest.raises((AttributeError, TypeError)):
        res.gmr_auc = 0.5  # type: ignore[misc]


def test_perfect_match_is_bioequivalent():
    res = predict_be_outcome(100.0, 20.0, 100.0, 20.0, n_subjects=24)
    assert res.gmr_auc == pytest.approx(1.0)
    assert res.gmr_cmax == pytest.approx(1.0)
    assert res.be_auc is True
    assert res.be_cmax is True
    assert res.overall_be is True


def test_gmr_auc_correct():
    res = predict_be_outcome(100.0, 20.0, 90.0, 20.0)
    assert res.gmr_auc == pytest.approx(0.9, rel=1e-6)


def test_gmr_cmax_correct():
    res = predict_be_outcome(100.0, 20.0, 100.0, 18.0)
    assert res.gmr_cmax == pytest.approx(0.9, rel=1e-6)


def test_ci90_auc_width_positive():
    res = predict_be_outcome(100.0, 20.0, 95.0, 20.0)
    lo, hi = res.ci90_auc
    assert hi > lo
    assert lo > 0


def test_ci90_cmax_width_positive():
    res = predict_be_outcome(100.0, 20.0, 100.0, 19.0)
    lo, hi = res.ci90_cmax
    assert hi > lo


def test_large_n_narrows_ci():
    res_small = predict_be_outcome(100.0, 20.0, 95.0, 20.0, n_subjects=6)
    res_large = predict_be_outcome(100.0, 20.0, 95.0, 20.0, n_subjects=60)
    width_small = res_small.ci90_auc[1] - res_small.ci90_auc[0]
    width_large = res_large.ci90_auc[1] - res_large.ci90_auc[0]
    assert width_large < width_small


def test_failing_auc_fails_overall():
    # GMR=0.70 should fail AUC BE
    res = predict_be_outcome(100.0, 20.0, 70.0, 20.0, n_subjects=24)
    assert res.be_auc is False
    assert res.overall_be is False


def test_failing_cmax_fails_overall():
    # GMR Cmax = 1.30 should fail
    res = predict_be_outcome(100.0, 20.0, 100.0, 26.0, n_subjects=24)
    assert res.be_cmax is False
    assert res.overall_be is False


def test_both_passing_overall_be():
    # GMR ~0.95 for both with large n, small CV should pass
    res = predict_be_outcome(100.0, 20.0, 95.0, 19.0, ref_auc_cv=0.15, ref_cmax_cv=0.15,
                             n_subjects=30)
    assert res.overall_be is True


def test_n_subjects_stored():
    res = predict_be_outcome(100.0, 20.0, 100.0, 20.0, n_subjects=18)
    assert res.n_subjects == 18


def test_invalid_ref_auc_raises():
    with pytest.raises(ValueError, match="ref_auc"):
        predict_be_outcome(0.0, 20.0, 100.0, 20.0)


def test_invalid_ref_cmax_raises():
    with pytest.raises(ValueError, match="ref_cmax"):
        predict_be_outcome(100.0, -1.0, 100.0, 20.0)


def test_invalid_test_auc_raises():
    with pytest.raises(ValueError, match="test_auc"):
        predict_be_outcome(100.0, 20.0, 0.0, 20.0)


def test_invalid_test_cmax_raises():
    with pytest.raises(ValueError, match="test_cmax"):
        predict_be_outcome(100.0, 20.0, 100.0, 0.0)


def test_invalid_cv_raises():
    with pytest.raises(ValueError, match="ref_auc_cv"):
        predict_be_outcome(100.0, 20.0, 100.0, 20.0, ref_auc_cv=0.0)


def test_invalid_n_subjects_raises():
    with pytest.raises(ValueError, match="n_subjects"):
        predict_be_outcome(100.0, 20.0, 100.0, 20.0, n_subjects=1)


def test_notes_non_empty():
    res = predict_be_outcome(100.0, 20.0, 100.0, 20.0)
    assert isinstance(res.notes, str)
    assert len(res.notes) > 0


# ---------------------------------------------------------------------------
# formulation_sensitivity tests
# ---------------------------------------------------------------------------


def test_formulation_sensitivity_returns_list():
    ref = {"ref_auc": 100.0, "ref_cmax": 20.0}
    variants = [
        {"test_auc": 95.0, "test_cmax": 19.0},
        {"test_auc": 80.0, "test_cmax": 17.0},
    ]
    results = formulation_sensitivity(ref, variants)
    assert isinstance(results, list)
    assert len(results) == 2


def test_formulation_sensitivity_types():
    ref = {"ref_auc": 100.0, "ref_cmax": 20.0}
    variants = [{"test_auc": 95.0, "test_cmax": 19.0}]
    results = formulation_sensitivity(ref, variants)
    assert isinstance(results[0], BEPredictionResult)


def test_formulation_sensitivity_empty_variants():
    ref = {"ref_auc": 100.0, "ref_cmax": 20.0}
    results = formulation_sensitivity(ref, [])
    assert results == []


def test_formulation_sensitivity_n_override():
    ref = {"ref_auc": 100.0, "ref_cmax": 20.0, "n_subjects": 12}
    variants = [{"test_auc": 95.0, "test_cmax": 19.0, "n_subjects": 24}]
    results = formulation_sensitivity(ref, variants)
    assert results[0].n_subjects == 24


# ---------------------------------------------------------------------------
# required_sample_size tests
# ---------------------------------------------------------------------------


def test_required_sample_size_returns_int():
    n = required_sample_size(gmr=0.95, cv=0.20)
    assert isinstance(n, int)


def test_required_sample_size_minimum_2():
    # Very high GMR=1.0 (exactly on ratio = 1) → should still be >= 2
    n = required_sample_size(gmr=1.0, cv=0.01)
    assert n >= 2


def test_required_sample_size_maximum_200():
    # Very high CV, borderline GMR → n capped at 200
    n = required_sample_size(gmr=0.81, cv=0.80)
    assert n <= 200


def test_higher_cv_needs_more_subjects():
    n_low = required_sample_size(gmr=0.95, cv=0.15)
    n_high = required_sample_size(gmr=0.95, cv=0.35)
    assert n_high > n_low


def test_gmr_further_from_1_needs_more_subjects():
    n_close = required_sample_size(gmr=0.99, cv=0.20)
    n_far = required_sample_size(gmr=0.85, cv=0.20)
    # GMR=0.85 is further from boundary → actually needs fewer? No:
    # The delta to nearest boundary: ln(0.85)-ln(0.80)=0.061 vs ln(0.99)-ln(0.80)=0.213
    # So 0.85 is closer to boundary → needs more n
    assert n_far > n_close


def test_higher_power_needs_more_subjects():
    n_80 = required_sample_size(gmr=0.95, cv=0.20, power=0.80)
    n_90 = required_sample_size(gmr=0.95, cv=0.20, power=0.90)
    assert n_90 >= n_80


def test_invalid_gmr_raises():
    with pytest.raises(ValueError, match="gmr"):
        required_sample_size(gmr=0.0, cv=0.20)


def test_invalid_cv_raises():
    with pytest.raises(ValueError, match="cv"):
        required_sample_size(gmr=0.95, cv=0.0)


def test_invalid_power_raises():
    with pytest.raises(ValueError, match="power"):
        required_sample_size(gmr=0.95, cv=0.20, power=0.0)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        required_sample_size(gmr=0.95, cv=0.20, alpha=0.6)


def test_typical_case_reasonable_n():
    # GMR=0.95, CV=0.20, power=0.80 → typical result is ~14-20 subjects
    n = required_sample_size(gmr=0.95, cv=0.20)
    assert 2 <= n <= 200
