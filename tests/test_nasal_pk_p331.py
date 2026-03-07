"""Tests for Phase 331 — Nasal Drug Delivery PK Model."""

import pytest

from omega_pbpk.core.nasal_pk_p331 import NasalPKResult, simulate_nasal_pk

# --- Helper defaults ---
DEFAULT_PARAMS = dict(
    drug_name="TestDrug",
    dose_mg=10.0,
    k_abs_nasal_per_h=0.5,
    k_mucosal_per_h=0.2,
    k_muco_clearance_per_h=0.1,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    t_end_h=12.0,
    dt_h=0.1,
)


def make_result(**overrides):
    params = {**DEFAULT_PARAMS, **overrides}
    return simulate_nasal_pk(**params)


# --- Return type ---
def test_return_type():
    result = make_result()
    assert isinstance(result, NasalPKResult)


# --- Array lengths ---
def test_array_lengths():
    result = make_result(t_end_h=12.0, dt_h=0.1)
    n = len(result.times_h)
    assert n > 0
    assert len(result.c_nasal_depot_mg_mL) == n
    assert len(result.c_mucosal_mg_mL) == n
    assert len(result.c_systemic_mg_L) == n


# --- Nasal depot starts at dose/v_nasal ---
def test_nasal_depot_initial_value():
    result = make_result(dose_mg=10.0, v_nasal_mL=10.0)
    expected = 10.0 / 10.0  # 1.0 mg/mL
    assert abs(result.c_nasal_depot_mg_mL[0] - expected) < 1e-9


# --- Nasal depot decreases monotonically (it can only go down) ---
def test_nasal_depot_decreases():
    result = make_result()
    depot = result.c_nasal_depot_mg_mL
    # Depot should be monotonically non-increasing
    for i in range(1, len(depot)):
        assert depot[i] <= depot[i - 1] + 1e-9


# --- Systemic starts at zero ---
def test_systemic_starts_at_zero():
    result = make_result()
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# --- Systemic rises then falls ---
def test_systemic_rises_then_falls():
    result = make_result()
    c_sys = result.c_systemic_mg_L
    # tmax index should not be 0 (rises from 0)
    tmax_idx = c_sys.index(max(c_sys))
    assert tmax_idx > 0
    # After tmax, concentration should generally decrease
    if tmax_idx < len(c_sys) - 1:
        assert c_sys[-1] < c_sys[tmax_idx]


# --- tmax > 0 ---
def test_tmax_positive():
    result = make_result()
    assert result.tmax_systemic_h > 0.0


# --- cmax > 0 ---
def test_cmax_positive():
    result = make_result()
    assert result.cmax_systemic_mg_L > 0.0


# --- AUC > 0 ---
def test_auc_positive():
    result = make_result()
    assert result.auc_systemic_mg_h_per_L > 0.0


# --- bioavailability_pct in [0, 100] ---
def test_bioavailability_in_range():
    result = make_result()
    assert 0.0 <= result.bioavailability_pct <= 100.0


# --- mucociliary_loss_pct in [0, 100] ---
def test_mucociliary_loss_in_range():
    result = make_result()
    assert 0.0 <= result.mucociliary_loss_pct <= 100.0


# --- Higher k_muco_clearance -> lower bioavailability ---
def test_higher_mucociliary_clearance_lower_bioavailability():
    r_low = make_result(k_muco_clearance_per_h=0.05)
    r_high = make_result(k_muco_clearance_per_h=1.0)
    assert r_high.bioavailability_pct < r_low.bioavailability_pct


# --- Higher k_muco_clearance -> higher mucociliary_loss_pct ---
def test_higher_mucociliary_clearance_more_loss():
    r_low = make_result(k_muco_clearance_per_h=0.01)
    r_high = make_result(k_muco_clearance_per_h=2.0)
    assert r_high.mucociliary_loss_pct > r_low.mucociliary_loss_pct


# --- Zero mucociliary clearance -> near-zero loss ---
def test_zero_mucociliary_clearance_no_loss():
    result = make_result(k_muco_clearance_per_h=0.0)
    assert result.mucociliary_loss_pct < 1.0  # near zero (not exactly due to numerics)


# --- Dose linearity: 2x dose -> 2x AUC ---
def test_dose_linearity_auc():
    r1 = make_result(dose_mg=10.0)
    r2 = make_result(dose_mg=20.0)
    assert abs(r2.auc_systemic_mg_h_per_L / r1.auc_systemic_mg_h_per_L - 2.0) < 0.01


# --- Dose linearity: 2x dose -> 2x Cmax ---
def test_dose_linearity_cmax():
    r1 = make_result(dose_mg=10.0)
    r2 = make_result(dose_mg=20.0)
    assert abs(r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L - 2.0) < 0.01


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
        make_result(dose_mg=-5.0)


# --- Validation: cl_sys <= 0 ---
def test_validation_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        make_result(cl_sys_L_per_h=0.0)


# --- Validation: vd <= 0 ---
def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys"):
        make_result(vd_sys_L=0.0)


# --- Validation: k_abs_nasal <= 0 ---
def test_validation_k_abs_nasal_zero():
    with pytest.raises(ValueError, match="k_abs_nasal"):
        make_result(k_abs_nasal_per_h=0.0)


# --- v_nasal_mL custom value changes depot concentration ---
def test_custom_v_nasal():
    r1 = make_result(v_nasal_mL=5.0, dose_mg=10.0)
    r2 = make_result(v_nasal_mL=20.0, dose_mg=10.0)
    # Smaller volume -> higher depot concentration initially
    assert r1.c_nasal_depot_mg_mL[0] > r2.c_nasal_depot_mg_mL[0]
