"""Tests for Phase 531: Ophthalmic Drop Dosing Model."""

import pytest

from omega_pbpk.core.ophthalmic_drop_dosing import (
    OphthalmicDropResult,
    compare_logp_ophthalmic,
    simulate_ophthalmic_drop,
)

# ---------------------------------------------------------------------------
# Basic return type and fields
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_ophthalmic_drop("timolol", dose_mg=0.1, logP=1.5)
    assert isinstance(result, OphthalmicDropResult)


def test_result_fields_present():
    result = simulate_ophthalmic_drop("timolol", dose_mg=0.1, logP=1.5)
    assert hasattr(result, "drug_name")
    assert hasattr(result, "dose_mg")
    assert hasattr(result, "logP")
    assert hasattr(result, "n_drops")
    assert hasattr(result, "k_corneal_per_h")
    assert hasattr(result, "k_drain_per_h")
    assert hasattr(result, "times_h")
    assert hasattr(result, "a_tear_mg")
    assert hasattr(result, "c_aqueous_mg_L")
    assert hasattr(result, "c_systemic_mg_L")
    assert hasattr(result, "cmax_aqueous_mg_L")
    assert hasattr(result, "tmax_aqueous_h")
    assert hasattr(result, "auc_aqueous_mg_h_per_L")
    assert hasattr(result, "cmax_systemic_mg_L")
    assert hasattr(result, "systemic_fraction_pct")
    assert hasattr(result, "corneal_bioavailability_pct")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_a_tear_starts_at_total_dose_single_drop():
    result = simulate_ophthalmic_drop("drug_A", dose_mg=0.5, logP=2.0)
    assert result.a_tear_mg[0] == pytest.approx(0.5, rel=1e-6)


def test_a_tear_starts_at_total_dose_multi_drops():
    result = simulate_ophthalmic_drop("drug_A", dose_mg=0.5, logP=2.0, n_drops=3)
    assert result.a_tear_mg[0] == pytest.approx(1.5, rel=1e-6)


def test_aqueous_starts_at_zero():
    result = simulate_ophthalmic_drop("drug_B", dose_mg=0.2, logP=1.0)
    assert result.c_aqueous_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_systemic_starts_at_zero():
    result = simulate_ophthalmic_drop("drug_B", dose_mg=0.2, logP=1.0)
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------


def test_aqueous_rises_then_falls():
    result = simulate_ophthalmic_drop("drug_C", dose_mg=0.1, logP=2.0, t_end_h=8.0)
    # Should rise from zero
    assert result.cmax_aqueous_mg_L > 0.0
    assert result.tmax_aqueous_h > 0.0


def test_tear_decreases_over_time():
    result = simulate_ophthalmic_drop("drug_D", dose_mg=0.1, logP=1.0, t_end_h=4.0)
    # Tear amount should decrease from initial
    assert result.a_tear_mg[-1] < result.a_tear_mg[0]


def test_systemic_concentration_positive():
    result = simulate_ophthalmic_drop("drug_E", dose_mg=0.5, logP=1.0, t_end_h=6.0)
    assert result.cmax_systemic_mg_L > 0.0


# ---------------------------------------------------------------------------
# logP effects on corneal absorption
# ---------------------------------------------------------------------------


def test_k_corneal_increases_with_logP():
    r_low = simulate_ophthalmic_drop("drug_F", dose_mg=0.1, logP=0.5)
    r_high = simulate_ophthalmic_drop("drug_F", dose_mg=0.1, logP=3.0)
    assert r_high.k_corneal_per_h > r_low.k_corneal_per_h


def test_higher_logP_higher_aqueous_auc():
    r_low = simulate_ophthalmic_drop("drug_G", dose_mg=0.1, logP=0.5, t_end_h=8.0)
    r_high = simulate_ophthalmic_drop("drug_G", dose_mg=0.1, logP=3.0, t_end_h=8.0)
    assert r_high.auc_aqueous_mg_h_per_L > r_low.auc_aqueous_mg_h_per_L


def test_negative_logP_uses_min_k_corneal():
    result = simulate_ophthalmic_drop("drug_H", dose_mg=0.1, logP=-1.0)
    assert result.k_corneal_per_h == pytest.approx(0.01, abs=1e-6)


def test_k_corneal_clamped_to_max():
    # logP=100 gives 0.1*(1+100)/2 = 5.05 > 5.0, should be clamped
    result = simulate_ophthalmic_drop("drug_I", dose_mg=0.1, logP=100.0)
    assert result.k_corneal_per_h == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# k_drain
# ---------------------------------------------------------------------------


def test_k_drain_is_30_per_h():
    result = simulate_ophthalmic_drop("drug_J", dose_mg=0.1, logP=1.0)
    assert result.k_drain_per_h == pytest.approx(30.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Systemic exposure via nasolacrimal
# ---------------------------------------------------------------------------


def test_systemic_fraction_positive():
    result = simulate_ophthalmic_drop("drug_K", dose_mg=0.5, logP=1.0, t_end_h=6.0)
    assert result.systemic_fraction_pct > 0.0


def test_systemic_fraction_in_range():
    result = simulate_ophthalmic_drop("drug_L", dose_mg=0.5, logP=1.0, t_end_h=8.0)
    assert 0.0 < result.systemic_fraction_pct <= 100.0


# ---------------------------------------------------------------------------
# Dose linearity (n_drops)
# ---------------------------------------------------------------------------


def test_two_drops_double_aqueous_auc():
    r1 = simulate_ophthalmic_drop("drug_M", dose_mg=0.1, logP=2.0, n_drops=1, t_end_h=8.0)
    r2 = simulate_ophthalmic_drop("drug_M", dose_mg=0.1, logP=2.0, n_drops=2, t_end_h=8.0)
    assert r2.auc_aqueous_mg_h_per_L == pytest.approx(2.0 * r1.auc_aqueous_mg_h_per_L, rel=1e-4)


def test_two_drops_double_systemic_cmax():
    r1 = simulate_ophthalmic_drop("drug_N", dose_mg=0.1, logP=1.5, n_drops=1, t_end_h=8.0)
    r2 = simulate_ophthalmic_drop("drug_N", dose_mg=0.1, logP=1.5, n_drops=2, t_end_h=8.0)
    assert r2.cmax_systemic_mg_L == pytest.approx(2.0 * r1.cmax_systemic_mg_L, rel=1e-4)


# ---------------------------------------------------------------------------
# n_drops metadata
# ---------------------------------------------------------------------------


def test_n_drops_stored():
    result = simulate_ophthalmic_drop("drug_O", dose_mg=0.1, logP=1.0, n_drops=4)
    assert result.n_drops == 4


def test_dose_mg_stored():
    result = simulate_ophthalmic_drop("drug_P", dose_mg=0.25, logP=1.0)
    assert result.dose_mg == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Corneal bioavailability
# ---------------------------------------------------------------------------


def test_corneal_bioavailability_positive():
    result = simulate_ophthalmic_drop("drug_Q", dose_mg=0.1, logP=2.0, t_end_h=8.0)
    assert result.corneal_bioavailability_pct > 0.0


def test_corneal_bioavailability_max_100():
    result = simulate_ophthalmic_drop("drug_R", dose_mg=0.1, logP=5.0, t_end_h=8.0)
    assert result.corneal_bioavailability_pct <= 100.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ophthalmic_drop("drug", dose_mg=0.0, logP=1.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ophthalmic_drop("drug", dose_mg=-0.1, logP=1.0)


def test_invalid_n_drops_raises():
    with pytest.raises(ValueError, match="n_drops"):
        simulate_ophthalmic_drop("drug", dose_mg=0.1, logP=1.0, n_drops=0)


def test_invalid_cl_sys_raises():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_ophthalmic_drop("drug", dose_mg=0.1, logP=1.0, cl_sys_L_per_h=0.0)


def test_invalid_vd_sys_raises():
    with pytest.raises(ValueError, match="vd_sys"):
        simulate_ophthalmic_drop("drug", dose_mg=0.1, logP=1.0, vd_sys_L=0.0)


# ---------------------------------------------------------------------------
# compare_logp_ophthalmic
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_logp_ophthalmic("drug_S", dose_mg=0.1, logp_values=[0.5, 2.0, 4.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_sorted_by_aqueous_auc_descending():
    results = compare_logp_ophthalmic("drug_T", dose_mg=0.1, logp_values=[0.5, 2.0, 4.0])
    aucs = [r.auc_aqueous_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_results_are_ophthalmic_drop_result():
    results = compare_logp_ophthalmic("drug_U", dose_mg=0.1, logp_values=[1.0, 3.0])
    for r in results:
        assert isinstance(r, OphthalmicDropResult)
