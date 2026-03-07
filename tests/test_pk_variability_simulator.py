"""Tests for Phase 866 — Pharmacokinetic Variability Simulator."""

import pytest

from omega_pbpk.clinical.pk_variability_simulator import (
    PKVariabilityResult,
    compare_dose_variability,
    simulate_pk_variability,
)

# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_pk_variability("DrugA", dose_mg=100.0)
    assert isinstance(result, PKVariabilityResult)


def test_fields_preserved():
    result = simulate_pk_variability(
        "TestDrug",
        dose_mg=200.0,
        n_subjects=50,
        cl_pop_L_per_h=6.0,
        vd_pop_L=60.0,
        ka_pop_per_h=1.2,
        mec_mg_L=0.3,
        mtc_mg_L=4.0,
    )
    assert result.drug_name == "TestDrug"
    assert result.dose_mg == 200.0
    assert result.n_subjects == 50
    assert result.cl_pop_L_per_h == 6.0
    assert result.vd_pop_L == 60.0
    assert result.ka_pop_per_h == 1.2
    assert result.mec_mg_L == 0.3
    assert result.mtc_mg_L == 4.0


# ---------------------------------------------------------------------------
# Output value checks
# ---------------------------------------------------------------------------


def test_cmax_mean_positive():
    result = simulate_pk_variability("DrugA", dose_mg=100.0)
    assert result.cmax_mean > 0.0


def test_auc_mean_positive():
    result = simulate_pk_variability("DrugA", dose_mg=100.0)
    assert result.auc_mean > 0.0


def test_cmax_median_positive():
    result = simulate_pk_variability("DrugA", dose_mg=100.0)
    assert result.cmax_median > 0.0


def test_auc_median_positive():
    result = simulate_pk_variability("DrugA", dose_mg=100.0)
    assert result.auc_median > 0.0


# ---------------------------------------------------------------------------
# CV% checks
# ---------------------------------------------------------------------------


def test_cv_pct_positive_when_omega_positive():
    result = simulate_pk_variability(
        "DrugA", dose_mg=100.0, omega_cl_cv_pct=30.0, omega_vd_cv_pct=20.0
    )
    assert result.cmax_cv_pct > 0.0
    assert result.auc_cv_pct > 0.0


def test_cv_pct_near_zero_when_omega_zero():
    result = simulate_pk_variability(
        "DrugA",
        dose_mg=100.0,
        omega_cl_cv_pct=0.0,
        omega_vd_cv_pct=0.0,
        omega_ka_cv_pct=0.0,
        n_subjects=50,
    )
    assert result.cmax_cv_pct < 1.0
    assert result.auc_cv_pct < 1.0


# ---------------------------------------------------------------------------
# Dose-response checks
# ---------------------------------------------------------------------------


def test_higher_dose_higher_cmax():
    r_low = simulate_pk_variability("DrugA", dose_mg=50.0)
    r_high = simulate_pk_variability("DrugA", dose_mg=200.0)
    assert r_high.cmax_mean > r_low.cmax_mean


def test_higher_dose_higher_auc():
    r_low = simulate_pk_variability("DrugA", dose_mg=50.0)
    r_high = simulate_pk_variability("DrugA", dose_mg=200.0)
    assert r_high.auc_mean > r_low.auc_mean


# ---------------------------------------------------------------------------
# Percentage checks
# ---------------------------------------------------------------------------


def test_pct_below_mec_in_range():
    result = simulate_pk_variability("DrugA", dose_mg=100.0)
    assert 0.0 <= result.pct_below_mec <= 100.0


def test_pct_above_mtc_in_range():
    result = simulate_pk_variability("DrugA", dose_mg=100.0)
    assert 0.0 <= result.pct_above_mtc <= 100.0


def test_low_dose_high_pct_below_mec():
    """Very low dose should result in many subjects below MEC."""
    result = simulate_pk_variability(
        "DrugA", dose_mg=0.01, mec_mg_L=1.0, mtc_mg_L=100.0, n_subjects=50
    )
    assert result.pct_below_mec > 80.0


def test_high_dose_high_pct_above_mtc():
    """Very high dose should result in many subjects above MTC."""
    result = simulate_pk_variability(
        "DrugA", dose_mg=10000.0, mec_mg_L=0.01, mtc_mg_L=0.1, n_subjects=50
    )
    assert result.pct_above_mtc > 50.0


# ---------------------------------------------------------------------------
# Determinism and seed behavior
# ---------------------------------------------------------------------------


def test_same_seed_same_result():
    r1 = simulate_pk_variability("DrugA", dose_mg=100.0, seed=42)
    r2 = simulate_pk_variability("DrugA", dose_mg=100.0, seed=42)
    assert r1.cmax_mean == r2.cmax_mean
    assert r1.auc_mean == r2.auc_mean


def test_different_seed_different_result():
    r1 = simulate_pk_variability("DrugA", dose_mg=100.0, seed=42)
    r2 = simulate_pk_variability("DrugA", dose_mg=100.0, seed=999)
    # Results should differ in detail but be similar (within ~20%)
    assert r1.cmax_mean != r2.cmax_mean
    ratio = r1.cmax_mean / r2.cmax_mean
    assert 0.8 <= ratio <= 1.2


# ---------------------------------------------------------------------------
# compare_dose_variability
# ---------------------------------------------------------------------------


def test_compare_dose_variability_returns_list():
    results = compare_dose_variability("DrugA", doses_mg=[100.0, 200.0, 300.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_dose_variability_sorted_ascending():
    results = compare_dose_variability("DrugA", doses_mg=[300.0, 100.0, 200.0])
    doses = [r.dose_mg for r in results]
    assert doses == sorted(doses)


def test_compare_dose_variability_increasing_cmax():
    results = compare_dose_variability("DrugA", doses_mg=[50.0, 100.0, 200.0])
    cmaxes = [r.cmax_mean for r in results]
    assert cmaxes == sorted(cmaxes)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_pk_variability("DrugA", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_pk_variability("DrugA", dose_mg=-10.0)


def test_validation_n_subjects_too_small():
    with pytest.raises(ValueError, match="n_subjects must be >= 10"):
        simulate_pk_variability("DrugA", dose_mg=100.0, n_subjects=5)


def test_validation_omega_cl_negative():
    with pytest.raises(ValueError, match="omega_cl_cv_pct must be >= 0"):
        simulate_pk_variability("DrugA", dose_mg=100.0, omega_cl_cv_pct=-5.0)


def test_validation_mec_negative():
    with pytest.raises(ValueError, match="mec_mg_L must be >= 0"):
        simulate_pk_variability("DrugA", dose_mg=100.0, mec_mg_L=-1.0)
