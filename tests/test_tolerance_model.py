"""Tests for clinical/tolerance_model.py — Phase 761."""

import pytest

from omega_pbpk.clinical.tolerance_model import (
    ToleranceModelResult,
    assess_tolerance_development,
    simulate_tolerance_pk_pd,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_result():
    return simulate_tolerance_pk_pd(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_per_h=1.0,
        f_oral=0.8,
        emax=100.0,
        ec50_mg_L=1.0,
        k_desens_per_h=0.1,
        k_recovery_per_h=0.05,
        dosing_interval_h=12.0,
        n_doses=10,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_tolerance_model_result(default_result):
    assert isinstance(default_result, ToleranceModelResult)


def test_drug_name_preserved(default_result):
    assert default_result.drug_name == "TestDrug"


def test_dose_mg_preserved(default_result):
    assert default_result.dose_mg == 100.0


def test_times_h_is_list(default_result):
    assert isinstance(default_result.times_h, list)
    assert len(default_result.times_h) > 0


def test_c_plasma_is_list(default_result):
    assert isinstance(default_result.c_plasma_mg_L, list)
    assert len(default_result.c_plasma_mg_L) == len(default_result.times_h)


def test_effect_pct_is_list(default_result):
    assert isinstance(default_result.effect_pct, list)
    assert len(default_result.effect_pct) == len(default_result.times_h)


def test_sensitivity_is_list(default_result):
    assert isinstance(default_result.sensitivity, list)
    assert len(default_result.sensitivity) == len(default_result.times_h)


# ---------------------------------------------------------------------------
# PD tolerance correctness
# ---------------------------------------------------------------------------


def test_dose_1_peak_effect_positive(default_result):
    assert default_result.dose_1_peak_effect > 0.0


def test_dose_n_peak_effect_less_than_dose_1(default_result):
    """Tolerance develops: last dose has lower peak effect."""
    assert default_result.dose_n_peak_effect < default_result.dose_1_peak_effect


def test_tolerance_ratio_less_than_1(default_result):
    assert default_result.tolerance_ratio < 1.0


def test_min_sensitivity_less_than_1(default_result):
    """Receptor desensitization occurs."""
    assert default_result.min_sensitivity < 1.0


def test_sensitivity_all_at_most_1(default_result):
    assert all(s <= 1.0 for s in default_result.sensitivity)


def test_sensitivity_all_non_negative(default_result):
    assert all(s >= 0.0 for s in default_result.sensitivity)


def test_effect_pct_within_bounds(default_result):
    """Effect should be between 0 and emax."""
    assert all(0.0 <= e <= 100.01 for e in default_result.effect_pct)


# ---------------------------------------------------------------------------
# Desensitization rate sensitivity
# ---------------------------------------------------------------------------


def test_higher_k_desens_more_tolerance():
    """Higher desensitization rate → lower tolerance ratio."""
    result_low = simulate_tolerance_pk_pd(
        drug_name="Drug",
        dose_mg=100.0,
        k_desens_per_h=0.05,
        k_recovery_per_h=0.05,
        n_doses=8,
        dt_h=0.1,
    )
    result_high = simulate_tolerance_pk_pd(
        drug_name="Drug",
        dose_mg=100.0,
        k_desens_per_h=0.3,
        k_recovery_per_h=0.05,
        n_doses=8,
        dt_h=0.1,
    )
    assert result_high.tolerance_ratio < result_low.tolerance_ratio


def test_k_desens_zero_no_tolerance():
    """With k_desens approaching zero (but must be > 0), tolerance should be minimal."""
    result = simulate_tolerance_pk_pd(
        drug_name="NoDesTol",
        dose_mg=50.0,
        k_desens_per_h=1e-9,
        k_recovery_per_h=0.05,
        n_doses=5,
        dt_h=0.1,
    )
    # With essentially zero desensitization, ratio should be close to 1
    assert result.tolerance_ratio > 0.95


def test_tolerance_ratio_between_0_and_1(default_result):
    assert 0.0 < default_result.tolerance_ratio <= 1.0


# ---------------------------------------------------------------------------
# Recovery time
# ---------------------------------------------------------------------------


def test_recovery_time_h_non_negative(default_result):
    assert default_result.recovery_time_h >= 0.0


# ---------------------------------------------------------------------------
# assess_tolerance_development
# ---------------------------------------------------------------------------


def test_assess_returns_dict(default_result):
    result = assess_tolerance_development(default_result)
    assert isinstance(result, dict)


def test_assess_has_severity_key(default_result):
    result = assess_tolerance_development(default_result)
    assert "severity" in result


def test_assess_severity_valid_values(default_result):
    result = assess_tolerance_development(default_result)
    assert result["severity"] in {"none", "mild", "moderate", "severe"}


def test_assess_severe_with_high_desens():
    result = simulate_tolerance_pk_pd(
        drug_name="SevereDrug",
        dose_mg=200.0,
        k_desens_per_h=0.8,
        k_recovery_per_h=0.02,
        n_doses=15,
        dt_h=0.1,
    )
    assessment = assess_tolerance_development(result)
    # High desensitization should produce moderate or severe tolerance
    assert assessment["severity"] in {"moderate", "severe"}


def test_assess_none_with_zero_desens():
    result = simulate_tolerance_pk_pd(
        drug_name="NoDrug",
        dose_mg=50.0,
        k_desens_per_h=1e-9,
        k_recovery_per_h=0.05,
        n_doses=5,
        dt_h=0.1,
    )
    assessment = assess_tolerance_development(result)
    assert assessment["severity"] == "none"


def test_assess_has_description_key(default_result):
    result = assess_tolerance_development(default_result)
    assert "description" in result


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty(default_result):
    assert len(default_result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tolerance_pk_pd(drug_name="X", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tolerance_pk_pd(drug_name="X", dose_mg=-10.0)


def test_validation_emax_zero():
    with pytest.raises(ValueError, match="emax"):
        simulate_tolerance_pk_pd(drug_name="X", dose_mg=100.0, emax=0.0)


def test_validation_emax_negative():
    with pytest.raises(ValueError, match="emax"):
        simulate_tolerance_pk_pd(drug_name="X", dose_mg=100.0, emax=-1.0)


def test_validation_k_desens_zero():
    with pytest.raises(ValueError, match="k_desens"):
        simulate_tolerance_pk_pd(drug_name="X", dose_mg=100.0, k_desens_per_h=0.0)


def test_single_dose_simulation():
    result = simulate_tolerance_pk_pd(
        drug_name="SingleDose",
        dose_mg=100.0,
        n_doses=1,
        dt_h=0.1,
    )
    assert isinstance(result, ToleranceModelResult)
    assert result.dose_1_peak_effect > 0.0
