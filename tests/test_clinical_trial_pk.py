"""Tests for clinical trial PK simulation (Phase 362)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.clinical.clinical_trial_pk import (
    ClinicalTrialPKResult,
    compare_dose_groups,
    simulate_clinical_trial_pk,
)

# ---------------------------------------------------------------------------
# Default parameters for convenience
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    study_name="test_study",
    n_subjects=10,
    dose_mg=100.0,
    cl_mean_L_per_h=5.0,
    vd_mean_L=50.0,
    ka_per_h=1.5,
    f_oral=0.9,
    omega_cl_pct=30.0,
    omega_vd_pct=20.0,
    sigma_pct=15.0,
    seed=42,
    t_end_h=48.0,
    dt_h=0.5,
)


def _run(**overrides) -> ClinicalTrialPKResult:
    kwargs = dict(DEFAULT_KWARGS)
    kwargs.update(overrides)
    return simulate_clinical_trial_pk(**kwargs)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_returns_clinical_trial_pk_result():
    r = _run()
    assert isinstance(r, ClinicalTrialPKResult)


def test_frozen_dataclass_immutable():
    r = _run()
    with pytest.raises((AttributeError, TypeError)):
        r.auc_gm_mg_h_per_L = 999.0  # type: ignore[misc]


def test_n_subjects_auc_individual_length():
    r = _run(n_subjects=10)
    assert len(r.auc_individual) == 10


def test_n_subjects_cmax_individual_length():
    r = _run(n_subjects=10)
    assert len(r.cmax_individual) == 10


def test_auc_gm_positive():
    r = _run()
    assert r.auc_gm_mg_h_per_L > 0


def test_cmax_gm_positive():
    r = _run()
    assert r.cmax_gm_mg_L > 0


def test_t_half_gm_positive():
    r = _run()
    assert r.t_half_gm_h > 0


def test_t_half_from_cl_vd():
    """t½ ≈ ln(2) * Vd / CL."""
    r = _run(cl_mean_L_per_h=5.0, vd_mean_L=50.0)
    expected = math.log(2) * 50.0 / 5.0
    assert r.t_half_gm_h == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_higher_dose_higher_cmax_gm():
    low = _run(dose_mg=50.0, seed=99)
    high = _run(dose_mg=200.0, seed=99)
    assert high.cmax_gm_mg_L > low.cmax_gm_mg_L


def test_higher_dose_higher_auc_gm():
    low = _run(dose_mg=50.0, seed=99)
    high = _run(dose_mg=200.0, seed=99)
    assert high.auc_gm_mg_h_per_L > low.auc_gm_mg_h_per_L


def test_dose_proportional_auc():
    """Doubling dose should roughly double AUC (linear PK)."""
    low = _run(dose_mg=100.0, sigma_pct=0.0, seed=1)
    high = _run(dose_mg=200.0, sigma_pct=0.0, seed=1)
    ratio = high.auc_gm_mg_h_per_L / low.auc_gm_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.10)


# ---------------------------------------------------------------------------
# Variability
# ---------------------------------------------------------------------------


def test_auc_cv_pct_positive_with_variability():
    r = _run(omega_cl_pct=30.0, n_subjects=20)
    assert r.auc_cv_pct > 0


def test_high_omega_cl_yields_high_auc_cv():
    low_var = _run(omega_cl_pct=5.0, n_subjects=30, seed=7)
    high_var = _run(omega_cl_pct=50.0, n_subjects=30, seed=7)
    assert high_var.auc_cv_pct > low_var.auc_cv_pct


def test_cmax_cv_pct_positive():
    r = _run(omega_cl_pct=30.0, n_subjects=20)
    assert r.cmax_cv_pct > 0


def test_zero_sigma_no_crash():
    r = _run(sigma_pct=0.0)
    assert r.auc_gm_mg_h_per_L > 0


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def test_auc_90ci_lower_lt_gm_lt_upper():
    r = _run(n_subjects=20)
    lo, hi = r.auc_gm_90ci
    assert lo < r.auc_gm_mg_h_per_L < hi


def test_cmax_90ci_lower_lt_gm_lt_upper():
    r = _run(n_subjects=20)
    lo, hi = r.cmax_gm_90ci
    assert lo < r.cmax_gm_mg_L < hi


def test_auc_90ci_tuple_length():
    r = _run()
    assert len(r.auc_gm_90ci) == 2


def test_cmax_90ci_tuple_length():
    r = _run()
    assert len(r.cmax_gm_90ci) == 2


# ---------------------------------------------------------------------------
# BE power
# ---------------------------------------------------------------------------


def test_n_required_for_be_positive():
    r = _run()
    assert r.n_required_for_be > 0


def test_n_required_for_be_int():
    r = _run()
    assert isinstance(r.n_required_for_be, int)


def test_higher_variability_larger_be_n():
    low = _run(omega_cl_pct=10.0)
    high = _run(omega_cl_pct=50.0)
    assert high.n_required_for_be > low.n_required_for_be


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_same_result():
    r1 = _run(seed=123)
    r2 = _run(seed=123)
    assert r1.auc_gm_mg_h_per_L == pytest.approx(r2.auc_gm_mg_h_per_L, rel=1e-9)
    assert r1.cmax_gm_mg_L == pytest.approx(r2.cmax_gm_mg_L, rel=1e-9)


def test_different_seeds_different_results():
    r1 = _run(seed=1)
    r2 = _run(seed=2)
    assert r1.auc_gm_mg_h_per_L != pytest.approx(r2.auc_gm_mg_h_per_L, rel=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_n_subjects_too_small_raises():
    with pytest.raises(ValueError, match="n_subjects"):
        _run(n_subjects=2)


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-10.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_mean_L_per_h"):
        _run(cl_mean_L_per_h=0.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_mean_L"):
        _run(vd_mean_L=0.0)


def test_sigma_negative_raises():
    with pytest.raises(ValueError, match="sigma_pct"):
        _run(sigma_pct=-1.0)


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    r = _run()
    assert len(r.notes) > 0


def test_study_name_preserved():
    r = simulate_clinical_trial_pk(
        study_name="MY_TRIAL",
        n_subjects=5,
        dose_mg=10.0,
        cl_mean_L_per_h=2.0,
        vd_mean_L=20.0,
    )
    assert r.study_name == "MY_TRIAL"


def test_tmax_median_non_negative():
    r = _run()
    assert r.tmax_median_h >= 0.0


# ---------------------------------------------------------------------------
# Multiple doses
# ---------------------------------------------------------------------------


def test_multiple_dose_higher_auc_than_single():
    single = _run(n_doses=1, dosing_regimen="single_dose", t_end_h=72.0, seed=5)
    multi = _run(n_doses=3, dosing_regimen="multiple_dose", interval_h=24.0, t_end_h=72.0, seed=5)
    assert multi.auc_gm_mg_h_per_L > single.auc_gm_mg_h_per_L


# ---------------------------------------------------------------------------
# compare_dose_groups
# ---------------------------------------------------------------------------


def test_compare_dose_groups_returns_list():
    result = compare_dose_groups("trial", [50.0, 100.0, 200.0], cl_mean=5.0, vd_mean=50.0,
                                 n_subjects_per_group=5, t_end_h=24.0, dt_h=0.5)
    assert isinstance(result, list)
    assert len(result) == 3


def test_compare_dose_groups_sorted_ascending():
    result = compare_dose_groups("trial", [200.0, 50.0, 100.0], cl_mean=5.0, vd_mean=50.0,
                                 n_subjects_per_group=5, t_end_h=24.0, dt_h=0.5)
    doses = [r.dose_mg for r in result]
    assert doses == sorted(doses)


def test_compare_dose_groups_all_results():
    result = compare_dose_groups("trial", [50.0, 100.0], cl_mean=5.0, vd_mean=50.0,
                                 n_subjects_per_group=5, t_end_h=24.0, dt_h=0.5)
    assert all(isinstance(r, ClinicalTrialPKResult) for r in result)


def test_compare_dose_groups_cmax_increases():
    result = compare_dose_groups("trial", [50.0, 100.0, 200.0], cl_mean=5.0, vd_mean=50.0,
                                 n_subjects_per_group=10, t_end_h=24.0, dt_h=0.5)
    cmax_vals = [r.cmax_gm_mg_L for r in result]
    assert cmax_vals[0] < cmax_vals[1] < cmax_vals[2]
