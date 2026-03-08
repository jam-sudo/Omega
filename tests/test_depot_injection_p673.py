"""Tests for Phase 673 - Drug Depot Injection PK."""

import pytest

from omega_pbpk.core.depot_injection_p673 import (
    VALID_DEPOT_TYPES,
    DepotInjectionResult,
    compare_depots,
    simulate_depot_injection,
)


@pytest.fixture
def base_params():
    return {
        "drug_name": "TestDrug",
        "dose_mg": 100.0,
        "cl_sys_L_per_h": 5.0,
        "vd_sys_L": 50.0,
    }


# --- Return type and structure ---


def test_return_type(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert isinstance(r, DepotInjectionResult)


def test_list_lengths(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im", t_end_h=100.0, dt_h=1.0)
    assert len(r.times_h) == len(r.c_depot_mg) == len(r.c_plasma_mg_L) == 101


def test_non_negative_concentrations(base_params):
    r = simulate_depot_injection(**base_params, depot_type="microsphere_sc")
    assert all(c >= 0 for c in r.c_depot_mg)
    assert all(c >= 0 for c in r.c_plasma_mg_L)


def test_drug_name_preserved(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert r.drug_name == "TestDrug"


def test_dose_preserved(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert r.dose_mg == 100.0


def test_depot_type_preserved(base_params):
    r = simulate_depot_injection(**base_params, depot_type="polymer_matrix")
    assert r.depot_type == "polymer_matrix"


# --- Depot-specific behavior ---


def test_polymer_matrix_longest_sustained(base_params):
    """polymer_matrix has lowest ka -> longest sustained duration."""
    results = {}
    for dt in VALID_DEPOT_TYPES:
        r = simulate_depot_injection(**base_params, depot_type=dt)
        results[dt] = r.sustained_duration_h
    assert results["polymer_matrix"] == max(results.values())


def test_oil_based_im_lag_time(base_params):
    """oil_based_im has lag_h=2 -> c_plasma stays 0 for first steps."""
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im", dt_h=1.0)
    # At t=0 and t=1, no absorption has occurred (lag=2h)
    assert r.c_plasma_mg_L[0] == 0.0
    assert r.c_plasma_mg_L[1] == 0.0


def test_microsphere_no_lag(base_params):
    """microsphere_sc has no lag, absorption starts immediately."""
    r = simulate_depot_injection(**base_params, depot_type="microsphere_sc", dt_h=1.0)
    # After first step, plasma should have some drug
    assert r.c_plasma_mg_L[1] > 0.0


def test_crystalline_suspension_slow(base_params):
    """crystalline_suspension has ka=0.01, slower than microsphere."""
    r_crystal = simulate_depot_injection(**base_params, depot_type="crystalline_suspension")
    r_micro = simulate_depot_injection(**base_params, depot_type="microsphere_sc")
    assert r_crystal.tmax_h > r_micro.tmax_h


# --- PK parameters ---


def test_cmax_positive(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert r.cmax_mg_L > 0


def test_tmax_positive(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert r.tmax_h > 0


def test_cmax_at_tmax(base_params):
    """Cmax is achieved at tmax."""
    r = simulate_depot_injection(**base_params, depot_type="microsphere_sc", dt_h=1.0)
    tmax_idx = r.times_h.index(r.tmax_h)
    assert r.c_plasma_mg_L[tmax_idx] == r.cmax_mg_L


def test_auc_positive(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert r.auc_mg_h_per_L > 0


def test_t_half_absorption(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert abs(r.t_half_absorption_h - 0.693 / 0.02) < 0.01


def test_t_half_elimination(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    k_el = 5.0 / 50.0
    assert abs(r.t_half_elimination_h - 0.693 / k_el) < 0.01


# --- Flip-flop ---


def test_flip_flop_present_when_ka_lt_kel(base_params):
    """ka < k_el -> flip_flop_present = True."""
    # k_el = 5/50 = 0.1, ka for oil_based_im = 0.02 < 0.1
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert r.flip_flop_present is True


def test_flip_flop_absent_when_ka_gt_kel():
    """ka > k_el -> flip_flop_present = False."""
    # Use very low clearance so k_el is tiny
    r = simulate_depot_injection(
        "Drug", 100.0, "microsphere_sc", cl_sys_L_per_h=0.001, vd_sys_L=50.0
    )
    # k_el = 0.001/50 = 0.00002, ka=0.05 > 0.00002
    assert r.flip_flop_present is False


# --- Sustained duration ---


def test_sustained_duration_positive(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert r.sustained_duration_h > 0


# --- Bioavailability ---


def test_bioavailability_oil(base_params):
    r = simulate_depot_injection(**base_params, depot_type="oil_based_im")
    assert r.bioavailability_pct == 95.0


def test_bioavailability_microsphere(base_params):
    r = simulate_depot_injection(**base_params, depot_type="microsphere_sc")
    assert r.bioavailability_pct == 90.0


# --- Dose proportionality ---


def test_dose_proportionality_cmax(base_params):
    r1 = simulate_depot_injection(**base_params, depot_type="microsphere_sc")
    params2 = {**base_params, "dose_mg": 200.0}
    r2 = simulate_depot_injection(**params2, depot_type="microsphere_sc")
    assert abs(r2.cmax_mg_L / r1.cmax_mg_L - 2.0) < 0.05


def test_dose_proportionality_auc(base_params):
    r1 = simulate_depot_injection(**base_params, depot_type="microsphere_sc")
    params2 = {**base_params, "dose_mg": 200.0}
    r2 = simulate_depot_injection(**params2, depot_type="microsphere_sc")
    assert abs(r2.auc_mg_h_per_L / r1.auc_mg_h_per_L - 2.0) < 0.05


# --- compare_depots ---


def test_compare_returns_four(base_params):
    results = compare_depots(**base_params)
    assert len(results) == 4


def test_compare_sorted_by_auc_desc(base_params):
    results = compare_depots(**base_params)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


# --- Validation errors ---


def test_invalid_dose():
    with pytest.raises(ValueError):
        simulate_depot_injection("D", -10, "oil_based_im", 5.0, 50.0)


def test_invalid_clearance():
    with pytest.raises(ValueError):
        simulate_depot_injection("D", 100, "oil_based_im", -1.0, 50.0)


def test_invalid_vd():
    with pytest.raises(ValueError):
        simulate_depot_injection("D", 100, "oil_based_im", 5.0, 0.0)


def test_invalid_depot_type():
    with pytest.raises(ValueError):
        simulate_depot_injection("D", 100, "invalid_type", 5.0, 50.0)
