"""Tests for Phase 935 — extravascular_distribution_pk."""

import pytest

from omega_pbpk.core.extravascular_distribution_pk import (
    ExtravascularDistributionResult,
    screen_tissue_distribution,
    simulate_extravascular_distribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**kwargs):
    defaults = dict(
        drug_name="DrugX",
        dose_mg=10.0,
        logp=1.0,
        route="iv",
        cl_hepatic_L_per_h=10.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_extravascular_distribution(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    res = _sim()
    assert isinstance(res, ExtravascularDistributionResult)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_iv_c_plasma_initial():
    """IV: first plasma concentration ~ dose / V_plasma."""
    res = _sim(dose_mg=10.0, v_plasma_L=4.0, route="iv")
    assert res.c_plasma_mg_L[0] == pytest.approx(10.0 / 4.0, rel=0.05)


def test_oral_c_plasma_initial_zero():
    """Oral: plasma starts at 0."""
    res = _sim(route="oral")
    assert res.c_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# PK metrics > 0
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive():
    res = _sim()
    assert res.cmax_plasma > 0


def test_auc_plasma_positive():
    res = _sim()
    assert res.auc_plasma > 0


# ---------------------------------------------------------------------------
# Tissue concentration lists
# ---------------------------------------------------------------------------


def test_c_liver_is_list_of_floats():
    res = _sim()
    assert isinstance(res.c_liver_mg_L, list)
    assert all(isinstance(v, float) for v in res.c_liver_mg_L)


def test_c_brain_is_list_of_floats():
    res = _sim()
    assert isinstance(res.c_brain_mg_L, list)
    assert all(isinstance(v, float) for v in res.c_brain_mg_L)


def test_c_muscle_is_list():
    res = _sim()
    assert isinstance(res.c_muscle_mg_L, list)
    assert len(res.c_muscle_mg_L) > 1


def test_c_fat_is_list():
    res = _sim()
    assert isinstance(res.c_fat_mg_L, list)
    assert len(res.c_fat_mg_L) > 1


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def test_brain_exposure_index_positive():
    res = _sim()
    assert res.brain_exposure_index > 0


def test_liver_muscle_ratio_greater_than_one():
    """Default liver Kp > muscle Kp → liver accumulates more."""
    res = _sim()
    assert res.liver_muscle_ratio > 1.0


# ---------------------------------------------------------------------------
# Kp values used
# ---------------------------------------------------------------------------


def test_kp_muscle_used_positive():
    res = _sim()
    assert res.kp_muscle_used > 0


def test_kp_fat_used_positive():
    res = _sim()
    assert res.kp_fat_used > 0


def test_kp_liver_used_positive():
    res = _sim()
    assert res.kp_liver_used > 0


def test_kp_brain_used_positive():
    res = _sim()
    assert res.kp_brain_used > 0


# ---------------------------------------------------------------------------
# logP effect on Kp
# ---------------------------------------------------------------------------


def test_high_logp_higher_kp_fat():
    res_low = _sim(logp=-1.0)
    res_high = _sim(logp=4.0)
    assert res_high.kp_fat_used > res_low.kp_fat_used


def test_high_logp_higher_brain_exposure_index():
    res_low = _sim(logp=-2.0)
    res_high = _sim(logp=4.0)
    assert res_high.brain_exposure_index > res_low.brain_exposure_index


# ---------------------------------------------------------------------------
# Override Kp
# ---------------------------------------------------------------------------


def test_override_kp_brain():
    res = _sim(kp_brain=0.5)
    assert res.kp_brain_used == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# screen_tissue_distribution
# ---------------------------------------------------------------------------


def test_screen_returns_correct_length():
    logp_values = [-1.0, 0.0, 1.0, 3.0, 5.0]
    results = screen_tissue_distribution("DrugX", logp_values, dose_mg=10.0, t_end_h=12.0)
    assert len(results) == len(logp_values)


def test_screen_sorted_by_brain_exposure_descending():
    logp_values = [-1.0, 1.0, 4.0]
    results = screen_tissue_distribution("DrugX", logp_values, dose_mg=10.0, t_end_h=12.0)
    scores = [r.brain_exposure_index for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Route: IV vs oral
# ---------------------------------------------------------------------------


def test_iv_first_concentration_nonzero():
    res = _sim(route="iv")
    assert res.c_plasma_mg_L[0] > 0


def test_oral_first_concentration_is_zero():
    res = _sim(route="oral")
    assert res.c_plasma_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Notes non-empty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    res = _sim()
    assert isinstance(res.notes, str) and len(res.notes) > 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_mg_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=0.0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        _sim(dose_mg=-5.0)


def test_validation_cl_hepatic_zero():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        _sim(cl_hepatic_L_per_h=0.0)


def test_validation_v_plasma_zero():
    with pytest.raises(ValueError, match="v_plasma_L"):
        _sim(v_plasma_L=0.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        _sim(route="subcutaneous")


# ---------------------------------------------------------------------------
# Linearity (2x dose → ~2x AUC)
# ---------------------------------------------------------------------------


def test_dose_doubling_doubles_auc():
    res1 = _sim(dose_mg=10.0)
    res2 = _sim(dose_mg=20.0)
    ratio = res2.auc_plasma / res1.auc_plasma
    assert ratio == pytest.approx(2.0, rel=0.05)
