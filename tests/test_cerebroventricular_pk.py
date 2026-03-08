"""Tests for cerebroventricular PK model — Phase 244."""

import pytest

from omega_pbpk.core.cerebroventricular_pk import (
    CerebroventricularPKResult,
    simulate_cerebroventricular_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=1.0,
        cl_sys_L_per_h=0.5,
    )
    defaults.update(kwargs)
    return simulate_cerebroventricular_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _default()
    assert isinstance(result, CerebroventricularPKResult)


def test_times_is_list():
    result = _default()
    assert isinstance(result.times_h, list)


def test_concentrations_are_lists():
    result = _default()
    assert isinstance(result.c_lateral_mg_mL, list)
    assert isinstance(result.c_third_mg_mL, list)
    assert isinstance(result.c_fourth_mg_mL, list)
    assert isinstance(result.c_systemic_mg_L, list)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_lateral_starts_high():
    result = _default()
    assert result.c_lateral_mg_mL[0] > 0.0


def test_third_starts_at_zero():
    result = _default()
    assert result.c_third_mg_mL[0] == pytest.approx(0.0)


def test_fourth_starts_at_zero():
    result = _default()
    assert result.c_fourth_mg_mL[0] == pytest.approx(0.0)


def test_systemic_starts_at_zero():
    result = _default()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# All cmax values positive
# ---------------------------------------------------------------------------


def test_cmax_lateral_positive():
    result = _default()
    assert result.cmax_lateral_mg_mL > 0.0


def test_cmax_third_positive():
    result = _default()
    assert result.cmax_third_mg_mL > 0.0


def test_cmax_fourth_positive():
    result = _default()
    assert result.cmax_fourth_mg_mL > 0.0


# ---------------------------------------------------------------------------
# AUC positive
# ---------------------------------------------------------------------------


def test_auc_lateral_positive():
    result = _default()
    assert result.auc_lateral_mg_h_per_mL > 0.0


def test_auc_fourth_positive():
    result = _default()
    assert result.auc_fourth_mg_h_per_mL > 0.0


# ---------------------------------------------------------------------------
# Lateral concentration decreases overall
# ---------------------------------------------------------------------------


def test_lateral_decreases_overall():
    result = _default()
    # Compare average of first quarter vs last quarter
    n = len(result.c_lateral_mg_mL)
    q = n // 4
    avg_early = sum(result.c_lateral_mg_mL[:q]) / q
    avg_late = sum(result.c_lateral_mg_mL[3 * q :]) / (n - 3 * q)
    assert avg_late < avg_early


# ---------------------------------------------------------------------------
# Concentration flow: lateral starts highest (initial condition), fourth ends lowest
# ---------------------------------------------------------------------------


def test_lateral_initial_concentration_highest():
    # At t=0, lateral is the only non-zero compartment
    result = _default()
    assert result.c_lateral_mg_mL[0] > result.c_third_mg_mL[0]
    assert result.c_lateral_mg_mL[0] > result.c_fourth_mg_mL[0]


def test_cmax_third_greater_than_fourth():
    result = _default()
    assert result.cmax_third_mg_mL > result.cmax_fourth_mg_mL


# ---------------------------------------------------------------------------
# Dose linearity: 2x dose → 2x cmax
# ---------------------------------------------------------------------------


def test_dose_linearity_lateral():
    r1 = _default(dose_mg=1.0)
    r2 = _default(dose_mg=2.0)
    assert r2.cmax_lateral_mg_mL == pytest.approx(2.0 * r1.cmax_lateral_mg_mL, rel=1e-6)


def test_dose_linearity_fourth():
    r1 = _default(dose_mg=1.0)
    r2 = _default(dose_mg=2.0)
    assert r2.cmax_fourth_mg_mL == pytest.approx(2.0 * r1.cmax_fourth_mg_mL, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cerebroventricular_pk("X", dose_mg=0.0, cl_sys_L_per_h=0.5)


def test_invalid_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cerebroventricular_pk("X", dose_mg=-1.0, cl_sys_L_per_h=0.5)


def test_invalid_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_cerebroventricular_pk("X", dose_mg=1.0, cl_sys_L_per_h=0.0)


def test_invalid_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_cerebroventricular_pk("X", dose_mg=1.0, cl_sys_L_per_h=-0.5)
