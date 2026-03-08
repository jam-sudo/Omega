"""
Tests for Phase 447 — Skin Depot PK (transdermal delivery model).
"""

import pytest

from omega_pbpk.core.skin_depot_pk import (
    SkinDepotPKResult,
    compare_patch_sizes,
    simulate_skin_depot_pk,
)

# ─── Fixture ────────────────────────────────────────────────────────────────


@pytest.fixture
def base_result():
    return simulate_skin_depot_pk(
        drug_name="TestDrug",
        patch_dose_mg=10.0,
        patch_area_cm2=20.0,
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
        patch_duration_h=24.0,
        t_end_h=72.0,
        dt_h=0.1,
    )


# ─── Return type ────────────────────────────────────────────────────────────


def test_return_type(base_result):
    assert isinstance(base_result, SkinDepotPKResult)


def test_drug_name_preserved(base_result):
    assert base_result.drug_name == "TestDrug"


def test_patch_dose_preserved(base_result):
    assert base_result.patch_dose_mg == 10.0


# ─── Array consistency ───────────────────────────────────────────────────────


def test_times_start_at_zero(base_result):
    assert base_result.times_h[0] == 0.0


def test_array_lengths_consistent(base_result):
    n = len(base_result.times_h)
    assert len(base_result.c_stratum_corneum_mg_g) == n
    assert len(base_result.c_viable_epidermis_mg_mL) == n
    assert len(base_result.c_systemic_mg_L) == n


def test_times_monotone_increasing(base_result):
    for i in range(1, len(base_result.times_h)):
        assert base_result.times_h[i] > base_result.times_h[i - 1]


# ─── SC accumulation ─────────────────────────────────────────────────────────


def test_sc_starts_at_zero(base_result):
    assert base_result.c_stratum_corneum_mg_g[0] == 0.0


def test_sc_accumulates(base_result):
    # SC should accumulate above 0 during patch delivery
    max_sc = max(base_result.c_stratum_corneum_mg_g)
    assert max_sc > 0.0


# ─── Systemic PK ─────────────────────────────────────────────────────────────


def test_systemic_starts_near_zero(base_result):
    assert base_result.c_systemic_mg_L[0] == 0.0


def test_cmax_systemic_positive(base_result):
    assert base_result.cmax_systemic_mg_L > 0.0


def test_auc_systemic_positive(base_result):
    assert base_result.auc_systemic_mg_h_per_L > 0.0


def test_tmax_after_time_zero(base_result):
    assert base_result.tmax_systemic_h > 0.0


# ─── Lag time ────────────────────────────────────────────────────────────────


def test_lag_time_positive(base_result):
    assert base_result.lag_time_h > 0.0


def test_lag_time_before_tmax(base_result):
    assert base_result.lag_time_h < base_result.tmax_systemic_h


# ─── logP effect ─────────────────────────────────────────────────────────────


def test_higher_logp_shorter_lag():
    """Higher logP → more lipophilic → faster SC penetration → shorter lag."""
    r_low = simulate_skin_depot_pk(
        drug_name="LowLogP",
        patch_dose_mg=10.0,
        patch_area_cm2=20.0,
        logp=-1.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
    )
    r_high = simulate_skin_depot_pk(
        drug_name="HighLogP",
        patch_dose_mg=10.0,
        patch_area_cm2=20.0,
        logp=3.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
    )
    assert r_high.lag_time_h <= r_low.lag_time_h


def test_higher_logp_higher_penetration():
    """Higher logP → higher k_sc_to_ve → more drug reaches systemic."""
    r_low = simulate_skin_depot_pk(
        drug_name="LowLogP",
        patch_dose_mg=10.0,
        patch_area_cm2=20.0,
        logp=-1.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
    )
    r_high = simulate_skin_depot_pk(
        drug_name="HighLogP",
        patch_dose_mg=10.0,
        patch_area_cm2=20.0,
        logp=3.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
    )
    assert r_high.auc_systemic_mg_h_per_L > r_low.auc_systemic_mg_h_per_L


# ─── Efficiency ──────────────────────────────────────────────────────────────


def test_patch_efficiency_range(base_result):
    assert 0.0 <= base_result.patch_efficiency_pct <= 100.0


def test_steady_state_conc_positive(base_result):
    assert base_result.steady_state_conc_mg_L > 0.0


def test_t_half_positive(base_result):
    assert base_result.t_half_systemic_h > 0.0


# ─── Dose linearity ──────────────────────────────────────────────────────────


def test_dose_linearity_cmax():
    """2x dose → 2x Cmax."""
    r1 = simulate_skin_depot_pk(
        drug_name="Drug",
        patch_dose_mg=5.0,
        patch_area_cm2=20.0,
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
    )
    r2 = simulate_skin_depot_pk(
        drug_name="Drug",
        patch_dose_mg=10.0,
        patch_area_cm2=20.0,
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
    )
    ratio = r2.cmax_systemic_mg_L / r1.cmax_systemic_mg_L
    assert abs(ratio - 2.0) < 0.15  # within 15% due to ODE numerics


# ─── compare_patch_sizes ─────────────────────────────────────────────────────


def test_compare_patch_sizes_length():
    areas = [10.0, 20.0, 40.0]
    results = compare_patch_sizes(
        drug_name="Drug",
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
        areas_cm2=areas,
    )
    assert len(results) == 3


def test_compare_patch_sizes_sorted_descending():
    areas = [10.0, 20.0, 40.0]
    results = compare_patch_sizes(
        drug_name="Drug",
        logp=2.0,
        cl_sys_L_per_h=2.0,
        vd_sys_L=50.0,
        areas_cm2=areas,
    )
    for i in range(len(results) - 1):
        assert results[i].steady_state_conc_mg_L >= results[i + 1].steady_state_conc_mg_L


# ─── Validation errors ───────────────────────────────────────────────────────


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="patch_dose_mg"):
        simulate_skin_depot_pk(
            drug_name="D",
            patch_dose_mg=0.0,
            patch_area_cm2=10.0,
            logp=2.0,
            cl_sys_L_per_h=2.0,
        )


def test_validation_area_negative():
    with pytest.raises(ValueError, match="patch_area_cm2"):
        simulate_skin_depot_pk(
            drug_name="D",
            patch_dose_mg=10.0,
            patch_area_cm2=-5.0,
            logp=2.0,
            cl_sys_L_per_h=2.0,
        )


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_skin_depot_pk(
            drug_name="D",
            patch_dose_mg=10.0,
            patch_area_cm2=10.0,
            logp=2.0,
            cl_sys_L_per_h=0.0,
        )


def test_flux_rate_positive(base_result):
    assert base_result.flux_rate_mg_per_h >= 0.0


def test_notes_not_empty(base_result):
    assert len(base_result.notes) > 0
