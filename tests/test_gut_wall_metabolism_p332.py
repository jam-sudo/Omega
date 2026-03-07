"""Tests for Phase 332 — Gut Wall Metabolism Model."""

import pytest

from omega_pbpk.core.gut_wall_metabolism_p332 import (
    GutWallMetabolismResult,
    simulate_gut_wall_metabolism,
)

# --- Helper defaults ---
DEFAULT_PARAMS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    f_abs=0.9,
    cl_gut_wall_L_per_h=10.0,
    cl_hepatic_L_per_h=20.0,
    vd_L=50.0,
    t_end_h=24.0,
    dt_h=0.1,
)


def make_result(**overrides):
    params = {**DEFAULT_PARAMS, **overrides}
    return simulate_gut_wall_metabolism(**params)


# --- Return type (frozen dataclass) ---
def test_return_type():
    result = make_result()
    assert isinstance(result, GutWallMetabolismResult)


def test_frozen_dataclass():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


# --- f_oral in (0, 1] ---
def test_f_oral_in_range():
    result = make_result()
    assert 0.0 < result.f_oral <= 1.0


# --- gut_extraction_ratio in [0, 1) ---
def test_gut_extraction_ratio_in_range():
    result = make_result()
    assert 0.0 <= result.gut_extraction_ratio < 1.0


# --- hepatic_extraction_ratio in [0, 1) ---
def test_hepatic_extraction_ratio_in_range():
    result = make_result()
    assert 0.0 <= result.hepatic_extraction_ratio < 1.0


# --- fg + gut_extraction_ratio = 1 ---
def test_fg_plus_eg_equals_one():
    result = make_result()
    total = result.fg_gut_wall + result.gut_extraction_ratio
    assert abs(total - 1.0) < 1e-9


# --- fh + hepatic_extraction_ratio = 1 ---
def test_fh_plus_eh_equals_one():
    result = make_result()
    total = result.fh_hepatic + result.hepatic_extraction_ratio
    assert abs(total - 1.0) < 1e-9


# --- f_oral = f_abs * fg * fh ---
def test_f_oral_formula():
    result = make_result()
    expected = result.f_abs * result.fg_gut_wall * result.fh_hepatic
    assert abs(result.f_oral - expected) < 1e-9


# --- high_gut_metabolism True when Eg > 0.3 ---
def test_high_gut_metabolism_true():
    # Force high gut extraction: large cl_gut_wall relative to q_gut
    result = make_result(cl_gut_wall_L_per_h=20.0, q_gut_L_per_h=18.0)
    # Eg = 1 - 18/(18 + 20*1.0) = 1 - 18/38 ~ 0.526 > 0.3
    assert result.gut_extraction_ratio > 0.3
    assert result.high_gut_metabolism is True


# --- high_gut_metabolism False when cl_gut_wall near 0 ---
def test_high_gut_metabolism_false():
    result = make_result(cl_gut_wall_L_per_h=0.1)
    # Eg = 1 - 18/(18 + 0.1) ~ 0.0055, well below 0.3
    assert result.gut_extraction_ratio < 0.3
    assert result.high_gut_metabolism is False


# --- Higher cl_gut_wall -> lower f_oral ---
def test_higher_cl_gut_wall_lower_f_oral():
    r_low = make_result(cl_gut_wall_L_per_h=1.0)
    r_high = make_result(cl_gut_wall_L_per_h=50.0)
    assert r_high.f_oral < r_low.f_oral


# --- cmax > 0 ---
def test_cmax_positive():
    result = make_result()
    assert result.cmax_mg_L > 0.0


# --- auc > 0 ---
def test_auc_positive():
    result = make_result()
    assert result.auc_mg_h_per_L > 0.0


# --- t_half > 0 ---
def test_t_half_positive():
    result = make_result()
    assert result.t_half_h > 0.0


# --- Dose linearity: 2x dose -> 2x AUC ---
def test_dose_linearity_auc():
    r1 = make_result(dose_mg=100.0)
    r2 = make_result(dose_mg=200.0)
    ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert abs(ratio - 2.0) < 0.01


# --- Dose linearity: 2x dose -> 2x Cmax ---
def test_dose_linearity_cmax():
    r1 = make_result(dose_mg=100.0)
    r2 = make_result(dose_mg=200.0)
    ratio = r2.cmax_mg_L / r1.cmax_mg_L
    assert abs(ratio - 2.0) < 0.01


# --- Notes is a string ---
def test_notes_is_string():
    result = make_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# --- Validation: dose <= 0 ---
def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        make_result(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        make_result(dose_mg=-1.0)


# --- Validation: f_abs not in (0, 1] ---
def test_validation_f_abs_zero():
    with pytest.raises(ValueError, match="f_abs"):
        make_result(f_abs=0.0)


def test_validation_f_abs_greater_than_one():
    with pytest.raises(ValueError, match="f_abs"):
        make_result(f_abs=1.1)


# --- Validation: cl_gut_wall < 0 ---
def test_validation_cl_gut_wall_negative():
    with pytest.raises(ValueError, match="cl_gut_wall"):
        make_result(cl_gut_wall_L_per_h=-1.0)


# --- Validation: cl_hepatic <= 0 ---
def test_validation_cl_hepatic_zero():
    with pytest.raises(ValueError, match="cl_hepatic"):
        make_result(cl_hepatic_L_per_h=0.0)


# --- Validation: vd <= 0 ---
def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        make_result(vd_L=0.0)


# --- cl_gut_wall = 0 -> no gut extraction ---
def test_zero_gut_wall_cl_no_extraction():
    result = make_result(cl_gut_wall_L_per_h=0.0)
    assert abs(result.gut_extraction_ratio) < 1e-9
    assert abs(result.fg_gut_wall - 1.0) < 1e-9
