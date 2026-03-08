"""Tests for Phase 356 — Drug Absorption via Lymphatics (Chylomicron Route)."""

import pytest

from omega_pbpk.core.lymphatic_chylomicron_pk import (
    LymphaticChylomicronResult,
    compute_f_chylomicron,
    simulate_lymphatic_chylomicron,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = dict(
    drug_name="LipophilicDrug",
    dose_mg=100.0,
    logP=4.0,
    ka_per_h=1.0,
    f_oral=1.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    t_end_h=24.0,
    dt_h=0.05,
)

VALID_FOOD_EFFECT_CLASSES = {"significant", "minor", "negligible"}


def default_result() -> LymphaticChylomicronResult:
    return simulate_lymphatic_chylomicron(**DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = default_result()
    assert isinstance(result, LymphaticChylomicronResult)


def test_times_list_not_empty():
    result = default_result()
    assert len(result.times_h) > 0


def test_array_lengths_equal():
    result = default_result()
    n = len(result.times_h)
    assert len(result.c_gut_mg_mL) == n
    assert len(result.c_lymph_depot_mg) == n
    assert len(result.c_systemic_mg_L) == n


def test_times_start_at_zero():
    result = default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_or_near_t_end():
    result = default_result()
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.1)


# ---------------------------------------------------------------------------
# f_chylomicron computation
# ---------------------------------------------------------------------------


def test_f_chylomicron_in_0_1():
    result = default_result()
    assert 0.0 <= result.f_chylomicron <= 1.0


def test_high_logP_higher_f_chylo():
    result_high = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "logP": 6.0})
    result_low = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "logP": 0.0})
    assert result_high.f_chylomicron > result_low.f_chylomicron


def test_logP_3_f_chylo_near_half():
    # logistic(0) = 0.5
    f = compute_f_chylomicron(3.0)
    assert f == pytest.approx(0.5, rel=1e-4)


def test_very_high_logP_f_chylo_near_one():
    f = compute_f_chylomicron(10.0)
    assert f > 0.99


def test_very_low_logP_f_chylo_near_zero():
    f = compute_f_chylomicron(-5.0)
    assert f < 0.01


# ---------------------------------------------------------------------------
# Systemic dynamics
# ---------------------------------------------------------------------------


def test_systemic_starts_at_zero():
    result = default_result()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


def test_systemic_rises():
    result = default_result()
    c_sys = result.c_systemic_mg_L
    assert max(c_sys) > 0.0


def test_cmax_positive():
    result = default_result()
    assert result.cmax_systemic_mg_L > 0.0


def test_tmax_positive():
    result = default_result()
    assert result.tmax_systemic_h > 0.0


def test_auc_positive():
    result = default_result()
    assert result.auc_systemic_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Lymphatic contribution
# ---------------------------------------------------------------------------


def test_lymphatic_contribution_pct_in_0_100():
    result = default_result()
    assert 0.0 <= result.lymphatic_contribution_pct <= 100.0


def test_high_logP_higher_lymphatic_contribution():
    result_high = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "logP": 6.0})
    result_low = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "logP": 0.0})
    assert result_high.lymphatic_contribution_pct > result_low.lymphatic_contribution_pct


# ---------------------------------------------------------------------------
# Food effect class
# ---------------------------------------------------------------------------


def test_food_effect_class_valid():
    result = default_result()
    assert result.food_effect_class in VALID_FOOD_EFFECT_CLASSES


def test_high_logP_significant_food_effect():
    result = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "logP": 6.0})
    assert result.food_effect_class == "significant"


def test_low_logP_negligible_food_effect():
    result = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "logP": 0.0})
    assert result.food_effect_class == "negligible"


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_auc():
    result_1x = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "dose_mg": 50.0})
    result_2x = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "dose_mg": 100.0})
    ratio = result_2x.auc_systemic_mg_h_per_L / result_1x.auc_systemic_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_dose_linearity_cmax():
    result_1x = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "dose_mg": 50.0})
    result_2x = simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "dose_mg": 100.0})
    ratio = result_2x.cmax_systemic_mg_L / result_1x.cmax_systemic_mg_L
    assert ratio == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_nonpositive():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "dose_mg": 0.0})


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "dose_mg": -5.0})


def test_validation_cl_nonpositive():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "cl_sys_L_per_h": 0.0})


def test_validation_vd_nonpositive():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "vd_sys_L": -1.0})


def test_validation_ka_nonpositive():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "ka_per_h": 0.0})


def test_validation_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "f_oral": 0.0})


def test_validation_f_oral_above_one():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_lymphatic_chylomicron(**{**DEFAULT_PARAMS, "f_oral": 1.5})


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_contains_drug_name():
    result = default_result()
    assert "LipophilicDrug" in result.notes
