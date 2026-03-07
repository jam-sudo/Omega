"""Tests for Phase 1046 — Drug Vesicle PK."""

import pytest

from omega_pbpk.core.vesicle_pk import VesiclePKResult, simulate_vesicle_pk


# --- Return type ---

def test_returns_dataclass():
    result = simulate_vesicle_pk("DrugA", dose_mg=10.0)
    assert isinstance(result, VesiclePKResult)


# --- c_encapsulated decreases monotonically ---

def test_encapsulated_decreases():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0, t_end_h=48.0)
    enc = result.c_encapsulated_mg_L
    for i in range(1, len(enc)):
        assert enc[i] <= enc[i - 1] + 1e-12


# --- c_free initial value ---

def test_free_initial_value():
    dose = 10.0
    ee = 0.8
    vd = 50.0
    result = simulate_vesicle_pk("Drug", dose_mg=dose, encapsulation_efficiency=ee, vd_free_L=vd, t_end_h=72.0)
    expected_c0 = dose * (1 - ee) / vd
    assert abs(result.c_free_mg_L[0] - expected_c0) < 1e-9


# --- c_free has a peak > initial ---

def test_free_has_peak_above_initial():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0, encapsulation_efficiency=0.9, cl_free_L_per_h=5.0, vd_free_L=50.0, t_end_h=72.0)
    c0 = result.c_free_mg_L[0]
    assert result.cmax_free > c0


# --- AUC positivity ---

def test_auc_free_positive():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0)
    assert result.auc_free > 0.0


def test_auc_total_positive():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0)
    assert result.auc_total > 0.0


# --- cmax_free > c_free[0] ---

def test_cmax_gt_initial():
    result = simulate_vesicle_pk("Drug", dose_mg=100.0, encapsulation_efficiency=0.95, cl_free_L_per_h=2.0, t_end_h=72.0)
    assert result.cmax_free > result.c_free_mg_L[0]


# --- t_half_vesicle_h > 0 ---

def test_t_half_vesicle_positive():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0)
    assert result.t_half_vesicle_h > 0.0


# --- Exosome longer t_half than liposome ---

def test_exosome_longer_than_liposome():
    lipo = simulate_vesicle_pk("Drug", dose_mg=10.0, vesicle_type="liposome")
    exo = simulate_vesicle_pk("Drug", dose_mg=10.0, vesicle_type="exosome")
    assert exo.t_half_vesicle_h > lipo.t_half_vesicle_h


# --- Higher EE → higher initial encapsulated amount ---

def test_higher_ee_higher_encapsulated():
    r_low = simulate_vesicle_pk("Drug", dose_mg=10.0, encapsulation_efficiency=0.4)
    r_high = simulate_vesicle_pk("Drug", dose_mg=10.0, encapsulation_efficiency=0.9)
    assert r_high.c_encapsulated_mg_L[0] > r_low.c_encapsulated_mg_L[0]


# --- release_rate_per_h > 0 ---

def test_release_rate_positive():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0)
    assert result.release_rate_per_h > 0.0


# --- release_rate matches known values ---

@pytest.mark.parametrize("vtype,expected_kr", [
    ("liposome", 0.1),
    ("exosome", 0.05),
    ("niosome", 0.15),
    ("transfersome", 0.2),
])
def test_release_rate_by_type(vtype, expected_kr):
    result = simulate_vesicle_pk("Drug", dose_mg=10.0, vesicle_type=vtype)
    assert abs(result.release_rate_per_h - expected_kr) < 1e-9


# --- notes nonempty ---

def test_notes_nonempty():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0)
    assert len(result.notes) > 0


# --- Validation ---

def test_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_vesicle_pk("Drug", dose_mg=0.0)


def test_ee_zero_raises():
    with pytest.raises(ValueError):
        simulate_vesicle_pk("Drug", dose_mg=10.0, encapsulation_efficiency=0.0)


def test_ee_gt1_raises():
    with pytest.raises(ValueError):
        simulate_vesicle_pk("Drug", dose_mg=10.0, encapsulation_efficiency=1.5)


def test_invalid_type_raises():
    with pytest.raises(ValueError):
        simulate_vesicle_pk("Drug", dose_mg=10.0, vesicle_type="magic_bubble")


def test_cl_zero_raises():
    with pytest.raises(ValueError):
        simulate_vesicle_pk("Drug", dose_mg=10.0, cl_free_L_per_h=0.0)


# --- Niosome and transfersome ---

@pytest.mark.parametrize("vtype", ["niosome", "transfersome"])
def test_other_types_run(vtype):
    result = simulate_vesicle_pk("Drug", dose_mg=10.0, vesicle_type=vtype)
    assert result.auc_free > 0.0
    assert result.cmax_free > 0.0


# --- tmax is within simulation window ---

def test_tmax_within_window():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0, t_end_h=72.0)
    assert 0.0 <= result.tmax_free_h <= 72.0


# --- auc_total >= auc_free ---

def test_auc_total_gte_auc_free():
    result = simulate_vesicle_pk("Drug", dose_mg=10.0)
    assert result.auc_total >= result.auc_free
