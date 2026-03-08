"""Tests for Phase 696 — Synovial Fluid Drug Distribution."""

import math

import pytest

from omega_pbpk.core.synovial_fluid_pk_p696 import (
    SynovialFluidPKResult,
    simulate_synovial_fluid_pk,
)

BASE = dict(
    drug_name="diclofenac",
    dose_mg=50.0,
    logP=1.5,
    mw_Da=296.0,
    fu_plasma=0.5,
    cl_sys_L_per_h=4.0,
    vd_sys_L=12.0,
)


@pytest.fixture
def result():
    return simulate_synovial_fluid_pk(**BASE)


# --- Return type / structure ---


def test_return_type(result):
    assert isinstance(result, SynovialFluidPKResult)


def test_drug_name(result):
    assert result.drug_name == "diclofenac"


def test_dose_mg(result):
    assert result.dose_mg == 50.0


def test_joint_default(result):
    assert result.joint == "knee"


def test_times_h_is_list(result):
    assert isinstance(result.times_h, list)
    assert len(result.times_h) > 0


def test_c_plasma_is_list(result):
    assert isinstance(result.c_plasma_mg_L, list)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_c_synovial_is_list(result):
    assert isinstance(result.c_synovial_mg_L, list)
    assert len(result.c_synovial_mg_L) == len(result.times_h)


# --- All joints accepted ---


@pytest.mark.parametrize("joint", ["knee", "hip", "wrist", "ankle"])
def test_all_joints_accepted(joint):
    r = simulate_synovial_fluid_pk(**{**BASE, "joint": joint})
    assert r.joint == joint


# --- Invalid joint ---


def test_invalid_joint_raises():
    with pytest.raises(ValueError):
        simulate_synovial_fluid_pk(**{**BASE, "joint": "elbow"})


# --- Non-negative concentrations ---


def test_non_negative_plasma(result):
    assert all(c >= 0 for c in result.c_plasma_mg_L)


def test_non_negative_synovial(result):
    assert all(c >= 0 for c in result.c_synovial_mg_L)


# --- Cmax and AUC positive ---


def test_cmax_plasma_positive(result):
    assert result.cmax_plasma_mg_L > 0


def test_cmax_synovial_positive(result):
    assert result.cmax_synovial_mg_L > 0


def test_auc_plasma_positive(result):
    assert result.auc_plasma_mg_h_per_L > 0


def test_auc_synovial_positive(result):
    assert result.auc_synovial_mg_h_per_L > 0


# --- Synovial volume correct for each joint ---


@pytest.mark.parametrize("joint,vol", [("knee", 1.1), ("hip", 0.5), ("wrist", 0.3), ("ankle", 0.4)])
def test_synovial_volume_correct(joint, vol):
    r = simulate_synovial_fluid_pk(**{**BASE, "joint": joint})
    assert abs(r.synovial_volume_mL - vol) < 1e-6


# --- Higher fu_plasma → higher synovial exposure ---


def test_higher_fu_higher_synovial():
    r_low = simulate_synovial_fluid_pk(**{**BASE, "fu_plasma": 0.1})
    r_high = simulate_synovial_fluid_pk(**{**BASE, "fu_plasma": 0.9})
    assert r_high.auc_synovial_mg_h_per_L > r_low.auc_synovial_mg_h_per_L


# --- Penetration index in (0, 1) ---


def test_penetration_index_range(result):
    assert 0 < result.penetration_index < 1


# --- t_half_synovial positive ---


def test_t_half_synovial_positive(result):
    assert result.t_half_synovial_h > 0


def test_t_half_synovial_value(result):
    expected = math.log(2) / 0.15
    assert abs(result.t_half_synovial_h - expected) < 1e-6


# --- 2x dose → 2x Cmax ---


def test_dose_linearity():
    r1 = simulate_synovial_fluid_pk(**{**BASE, "dose_mg": 50.0})
    r2 = simulate_synovial_fluid_pk(**{**BASE, "dose_mg": 100.0})
    ratio = r2.cmax_plasma_mg_L / r1.cmax_plasma_mg_L
    assert abs(ratio - 2.0) < 0.1


# --- Synovial to plasma ratio ---


def test_synovial_to_plasma_ratio_positive(result):
    assert result.synovial_to_plasma_ratio > 0


# --- Validation errors ---


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_synovial_fluid_pk(**{**BASE, "dose_mg": 0})


def test_validation_dose_negative():
    with pytest.raises(ValueError):
        simulate_synovial_fluid_pk(**{**BASE, "dose_mg": -10})


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError):
        simulate_synovial_fluid_pk(**{**BASE, "cl_sys_L_per_h": 0})


def test_validation_vd_sys_zero():
    with pytest.raises(ValueError):
        simulate_synovial_fluid_pk(**{**BASE, "vd_sys_L": 0})


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError):
        simulate_synovial_fluid_pk(**{**BASE, "fu_plasma": 0})


def test_validation_fu_plasma_gt_1():
    with pytest.raises(ValueError):
        simulate_synovial_fluid_pk(**{**BASE, "fu_plasma": 1.5})


# --- Notes ---


def test_notes_is_string(result):
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
