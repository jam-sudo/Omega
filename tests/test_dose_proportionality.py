"""Tests for dose-proportionality analysis (Phase 176)."""

import pytest

from omega_pbpk.clinical.dose_proportionality import (
    DoseProportionalityResult,
    PowerModelResult,
    assess_dose_proportionality,
    power_model_fit,
)

# Perfect proportional data: AUC = 10 * dose (beta = 1.0)
DOSES = [10.0, 30.0, 100.0, 300.0]
AUCS_PROP = [d * 10.0 for d in DOSES]

# Supra-proportional: AUC = dose^1.5
AUCS_SUPRA = [d**1.5 for d in DOSES]

# Sub-proportional: AUC = dose^0.5
AUCS_SUB = [d**0.5 for d in DOSES]


def test_power_model_proportional():
    r = power_model_fit(DOSES, AUCS_PROP)
    assert abs(r.beta - 1.0) < 0.05
    assert r.r2 > 0.99


def test_power_model_supra():
    r = power_model_fit(DOSES, AUCS_SUPRA)
    assert r.beta > 1.1
    assert r.assessment == "supra_proportional"


def test_power_model_sub():
    r = power_model_fit(DOSES, AUCS_SUB)
    assert r.beta < 0.9
    assert r.assessment == "sub_proportional"


def test_power_model_r2_in_bounds():
    r = power_model_fit(DOSES, AUCS_PROP)
    assert 0.0 <= r.r2 <= 1.0


def test_power_model_returns_result():
    r = power_model_fit(DOSES, AUCS_PROP)
    assert isinstance(r, PowerModelResult)


def test_power_model_too_few_points_raises():
    with pytest.raises(ValueError, match="[Aa]t least 3"):
        power_model_fit([10.0, 30.0], [100.0, 300.0])


def test_power_model_length_mismatch_raises():
    with pytest.raises(ValueError):
        power_model_fit(DOSES, AUCS_PROP[:3])


def test_power_model_zero_dose_raises():
    with pytest.raises(ValueError, match="[Dd]ose"):
        power_model_fit([0.0, 10.0, 30.0], [0.0, 100.0, 300.0])


def test_power_model_predicted_values_count():
    r = power_model_fit(DOSES, AUCS_PROP)
    assert len(r.predicted_at_doses) == len(DOSES)


def test_power_model_predicted_close_to_actual():
    r = power_model_fit(DOSES, AUCS_PROP)
    for pred, actual in zip(r.predicted_at_doses, AUCS_PROP):
        assert abs(pred - actual) / actual < 0.05


def test_assess_proportional():
    r = assess_dose_proportionality(DOSES, AUCS_PROP)
    assert isinstance(r, DoseProportionalityResult)
    assert r.proportionality_assessment == "proportional"


def test_assess_supra_proportional():
    r = assess_dose_proportionality(DOSES, AUCS_SUPRA)
    assert r.proportionality_assessment == "supra_proportional"


def test_assess_sub_proportional():
    r = assess_dose_proportionality(DOSES, AUCS_SUB)
    assert r.proportionality_assessment == "sub_proportional"


def test_assess_with_cmax():
    r = assess_dose_proportionality(DOSES, AUCS_PROP, cmaxes=[d * 2.0 for d in DOSES])
    assert r.beta_cmax is not None
    assert abs(r.beta_cmax - 1.0) < 0.05


def test_assess_beta_auc_correct():
    r = assess_dose_proportionality(DOSES, AUCS_PROP)
    assert abs(r.beta_auc - 1.0) < 0.05


def test_dose_normalized_auc_length():
    r = assess_dose_proportionality(DOSES, AUCS_PROP)
    assert len(r.dose_normalized_auc) == len(DOSES)


def test_dose_normalized_auc_constant_proportional():
    r = assess_dose_proportionality(DOSES, AUCS_PROP)
    # Should be constant (all = 10.0) for proportional data
    vals = r.dose_normalized_auc
    assert max(vals) - min(vals) < 1e-6


def test_cv_near_zero_proportional():
    r = assess_dose_proportionality(DOSES, AUCS_PROP)
    assert r.cv_dose_normalized_auc_pct < 1.0


def test_r2_proportional():
    r = assess_dose_proportionality(DOSES, AUCS_PROP)
    assert r.r2_auc > 0.99


def test_too_few_doses_raises():
    with pytest.raises(ValueError, match="[Aa]t least 3"):
        assess_dose_proportionality([10.0, 30.0], [100.0, 300.0])


def test_cmax_length_mismatch_raises():
    with pytest.raises(ValueError):
        assess_dose_proportionality(DOSES, AUCS_PROP, cmaxes=[1.0, 2.0])


def test_notes_not_empty():
    r = assess_dose_proportionality(DOSES, AUCS_PROP)
    assert len(r.notes) > 0
