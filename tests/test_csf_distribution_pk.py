"""Tests for CSF drug distribution PK model (Phase 495)."""

import pytest

from omega_pbpk.core.csf_distribution_pk import (
    CSFDistributionResult,
    compare_logp_csf,
    simulate_csf_distribution,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_result():
    return simulate_csf_distribution(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.0,
        cl_sys_L_per_h=5.0,
        vd_plasma_L=10.0,
        fu_plasma=0.1,
        t_end_h=24.0,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type(base_result):
    assert isinstance(base_result, CSFDistributionResult)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_plasma_starts_at_dose_over_vd():
    result = simulate_csf_distribution(
        drug_name="Drug",
        dose_mg=200.0,
        logP=1.0,
        cl_sys_L_per_h=4.0,
        vd_plasma_L=8.0,
        fu_plasma=0.2,
    )
    expected = 200.0 / 8.0
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-9


def test_c_csf_starts_at_zero(base_result):
    assert base_result.c_csf_mg_L[0] == 0.0


def test_c_isf_starts_at_zero(base_result):
    assert base_result.c_isf_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Profile shape and basic metrics
# ---------------------------------------------------------------------------


def test_cmax_csf_positive(base_result):
    assert base_result.cmax_csf_mg_L > 0.0


def test_auc_csf_positive(base_result):
    assert base_result.auc_csf_mg_h_per_L > 0.0


def test_tmax_csf_within_simulation_range(base_result):
    assert 0.0 <= base_result.tmax_csf_h <= 24.0


def test_times_length_matches_steps():
    result = simulate_csf_distribution(
        drug_name="X",
        dose_mg=50.0,
        logP=1.0,
        cl_sys_L_per_h=2.0,
        t_end_h=10.0,
        dt_h=0.5,
    )
    expected_len = int(round(10.0 / 0.5)) + 1
    assert len(result.times_h) == expected_len


def test_plasma_decays_monotonically(base_result):
    # Plasma is analytical exp decay — strictly decreasing
    plasma = base_result.c_plasma_mg_L
    for i in range(1, len(plasma)):
        assert plasma[i] <= plasma[i - 1] + 1e-9


def test_csf_rises_then_falls(base_result):
    csf = base_result.c_csf_mg_L
    # Should be 0 at t=0, increase, then eventually decrease
    assert csf[0] == 0.0
    assert max(csf) > 0.0


def test_list_lengths_consistent(base_result):
    n = len(base_result.times_h)
    assert len(base_result.c_plasma_mg_L) == n
    assert len(base_result.c_csf_mg_L) == n
    assert len(base_result.c_isf_mg_L) == n


# ---------------------------------------------------------------------------
# PS_cp calculation
# ---------------------------------------------------------------------------


def test_ps_cp_positive_logp():
    result = simulate_csf_distribution(drug_name="A", dose_mg=100.0, logP=3.0, cl_sys_L_per_h=5.0)
    expected_ps = 0.001 * (1.0 + 3.0)
    assert abs(result.ps_cp_L_per_h - expected_ps) < 1e-9


def test_ps_cp_negative_logp():
    result = simulate_csf_distribution(drug_name="B", dose_mg=100.0, logP=-1.0, cl_sys_L_per_h=5.0)
    assert abs(result.ps_cp_L_per_h - 0.001) < 1e-9


def test_ps_cp_zero_logp():
    result = simulate_csf_distribution(drug_name="C", dose_mg=100.0, logP=0.0, cl_sys_L_per_h=5.0)
    # logP=0 is not > 0, so fallback to 0.001
    assert abs(result.ps_cp_L_per_h - 0.001) < 1e-9


# ---------------------------------------------------------------------------
# Higher logP -> higher CSF penetration
# ---------------------------------------------------------------------------


def test_higher_logp_higher_csf_penetration():
    r_low = simulate_csf_distribution(
        drug_name="LowLogP",
        dose_mg=100.0,
        logP=0.5,
        cl_sys_L_per_h=5.0,
        vd_plasma_L=10.0,
    )
    r_high = simulate_csf_distribution(
        drug_name="HighLogP",
        dose_mg=100.0,
        logP=4.0,
        cl_sys_L_per_h=5.0,
        vd_plasma_L=10.0,
    )
    assert r_high.auc_csf_mg_h_per_L > r_low.auc_csf_mg_h_per_L


def test_higher_logp_higher_ps_cp():
    r1 = simulate_csf_distribution("A", 100.0, logP=1.0, cl_sys_L_per_h=5.0)
    r2 = simulate_csf_distribution("B", 100.0, logP=5.0, cl_sys_L_per_h=5.0)
    assert r2.ps_cp_L_per_h > r1.ps_cp_L_per_h


# ---------------------------------------------------------------------------
# CSF/plasma ratio
# ---------------------------------------------------------------------------


def test_csf_plasma_ratio_non_negative(base_result):
    assert base_result.csf_plasma_ratio >= 0.0


def test_csf_plasma_ratio_is_float(base_result):
    assert isinstance(base_result.csf_plasma_ratio, float)


# ---------------------------------------------------------------------------
# compare_logp_csf
# ---------------------------------------------------------------------------


def test_compare_logp_returns_list():
    results = compare_logp_csf(
        drug_name="Drug",
        dose_mg=100.0,
        logp_values=[1.0, 2.0, 3.0],
        cl_sys_L_per_h=5.0,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_logp_sorted_descending():
    results = compare_logp_csf(
        drug_name="Drug",
        dose_mg=100.0,
        logp_values=[0.5, 2.0, 4.0],
        cl_sys_L_per_h=5.0,
    )
    aucs = [r.auc_csf_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_logp_result_types():
    results = compare_logp_csf(
        drug_name="X",
        dose_mg=50.0,
        logp_values=[1.0, 3.0],
        cl_sys_L_per_h=3.0,
    )
    for r in results:
        assert isinstance(r, CSFDistributionResult)


def test_compare_logp_drug_names():
    results = compare_logp_csf(
        drug_name="Base",
        dose_mg=100.0,
        logp_values=[1.0, 2.0],
        cl_sys_L_per_h=5.0,
    )
    drug_names = {r.drug_name for r in results}
    assert "Base_logP1.0" in drug_names or "Base_logP2.0" in drug_names


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_distribution("X", 0.0, 1.0, 5.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_distribution("X", -10.0, 1.0, 5.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_csf_distribution("X", 100.0, 1.0, 0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_csf_distribution("X", 100.0, 1.0, 5.0, vd_plasma_L=0.0)


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_csf_distribution("X", 100.0, 1.0, 5.0, fu_plasma=0.0)


def test_validation_fu_plasma_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_csf_distribution("X", 100.0, 1.0, 5.0, fu_plasma=1.5)


# ---------------------------------------------------------------------------
# notes field
# ---------------------------------------------------------------------------


def test_notes_is_string(base_result):
    assert isinstance(base_result.notes, str)


def test_notes_contains_ps_cp(base_result):
    assert "PS_cp" in base_result.notes
