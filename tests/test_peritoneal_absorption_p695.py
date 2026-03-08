"""Tests for Phase 695 — Peritoneal Drug Absorption."""

import pytest

from omega_pbpk.core.peritoneal_absorption_p695 import (
    PeritonealAbsorptionResult,
    simulate_peritoneal_absorption,
)

BASE = dict(
    drug_name="cisplatin",
    dose_mg=100.0,
    mw_Da=300.0,
    logP=1.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=20.0,
)


@pytest.fixture
def result():
    return simulate_peritoneal_absorption(**BASE)


# --- Return type / structure ---


def test_return_type(result):
    assert isinstance(result, PeritonealAbsorptionResult)


def test_drug_name(result):
    assert result.drug_name == "cisplatin"


def test_dose_mg(result):
    assert result.dose_mg == 100.0


def test_times_h_is_list(result):
    assert isinstance(result.times_h, list)
    assert len(result.times_h) > 0


def test_c_peritoneal_is_list(result):
    assert isinstance(result.c_peritoneal_mg_L, list)
    assert len(result.c_peritoneal_mg_L) == len(result.times_h)


def test_c_plasma_is_list(result):
    assert isinstance(result.c_plasma_mg_L, list)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_c_tumor_is_list(result):
    assert isinstance(result.c_tumor_mg_g, list)
    assert len(result.c_tumor_mg_g) == len(result.times_h)


# --- Non-negative concentrations ---


def test_non_negative_peritoneal(result):
    assert all(c >= 0 for c in result.c_peritoneal_mg_L)


def test_non_negative_plasma(result):
    assert all(c >= 0 for c in result.c_plasma_mg_L)


def test_non_negative_tumor(result):
    assert all(c >= 0 for c in result.c_tumor_mg_g)


# --- Initial peritoneal concentration ---


def test_initial_peritoneal_concentration(result):
    expected = 100.0 / 2.0  # dose / peritoneal_volume_L
    assert abs(result.c_peritoneal_mg_L[0] - expected) < 1e-6


# --- Peritoneal decreases over time ---


def test_peritoneal_decreases():
    r = simulate_peritoneal_absorption(**BASE)
    assert r.c_peritoneal_mg_L[-1] < r.c_peritoneal_mg_L[0]


# --- Plasma rises then falls ---


def test_plasma_rises_then_falls():
    r = simulate_peritoneal_absorption(**BASE)
    peak_idx = r.c_plasma_mg_L.index(max(r.c_plasma_mg_L))
    assert peak_idx > 0
    assert peak_idx < len(r.c_plasma_mg_L) - 1


# --- Cmax values positive ---


def test_cmax_peritoneal_positive(result):
    assert result.cmax_peritoneal_mg_L > 0


def test_cmax_plasma_positive(result):
    assert result.cmax_plasma_mg_L > 0


def test_cmax_tumor_positive(result):
    assert result.cmax_tumor_mg_g > 0


# --- AUC values positive ---


def test_auc_peritoneal_positive(result):
    assert result.auc_peritoneal_mg_h_per_L > 0


def test_auc_plasma_positive(result):
    assert result.auc_plasma_mg_h_per_L > 0


# --- Peritoneal advantage ---


def test_peritoneal_advantage_ge_1(result):
    assert result.peritoneal_advantage >= 1.0


def test_drug_exposure_ratio(result):
    assert result.drug_exposure_ratio == result.peritoneal_advantage


# --- Higher MW → slower absorption ---


def test_higher_mw_slower_absorption():
    r_low = simulate_peritoneal_absorption(**{**BASE, "mw_Da": 200.0})
    r_high = simulate_peritoneal_absorption(**{**BASE, "mw_Da": 2000.0})
    # Higher MW means slower absorption, so more drug remains in peritoneum
    assert r_high.c_peritoneal_mg_L[-1] > r_low.c_peritoneal_mg_L[-1]


# --- 2x dose → 2x Cmax ---


def test_dose_linearity():
    r1 = simulate_peritoneal_absorption(**{**BASE, "dose_mg": 100.0})
    r2 = simulate_peritoneal_absorption(**{**BASE, "dose_mg": 200.0})
    ratio = r2.cmax_peritoneal_mg_L / r1.cmax_peritoneal_mg_L
    assert abs(ratio - 2.0) < 0.1


# --- Peritoneal volume ---


def test_peritoneal_volume(result):
    assert result.peritoneal_volume_L == 2.0


# --- Validation errors ---


def test_validation_dose_zero():
    with pytest.raises(ValueError):
        simulate_peritoneal_absorption(**{**BASE, "dose_mg": 0})


def test_validation_dose_negative():
    with pytest.raises(ValueError):
        simulate_peritoneal_absorption(**{**BASE, "dose_mg": -10})


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError):
        simulate_peritoneal_absorption(**{**BASE, "cl_sys_L_per_h": 0})


def test_validation_vd_sys_zero():
    with pytest.raises(ValueError):
        simulate_peritoneal_absorption(**{**BASE, "vd_sys_L": 0})


def test_validation_peritoneal_volume_zero():
    with pytest.raises(ValueError):
        simulate_peritoneal_absorption(**{**BASE, "peritoneal_volume_L": 0})


def test_validation_tumor_mass_zero():
    with pytest.raises(ValueError):
        simulate_peritoneal_absorption(**{**BASE, "tumor_mass_g": 0})


def test_validation_mw_zero():
    with pytest.raises(ValueError):
        simulate_peritoneal_absorption(**{**BASE, "mw_Da": 0})


# --- Notes ---


def test_notes_is_string(result):
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
