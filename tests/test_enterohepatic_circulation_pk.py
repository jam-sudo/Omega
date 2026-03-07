"""Tests for enterohepatic circulation (EHC) pharmacokinetics model (Phase 687)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.enterohepatic_circulation_pk import (
    EHCResult,
    compare_ehc_vs_no_ehc,
    simulate_ehc_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_ehc_result():
    result = simulate_ehc_pk("DrugA", dose_mg=100.0)
    assert isinstance(result, EHCResult)


def test_initial_plasma_concentration():
    dose_mg = 200.0
    vd_L = 40.0
    result = simulate_ehc_pk("DrugB", dose_mg=dose_mg, vd_L=vd_L)
    expected = dose_mg / vd_L
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-9


def test_initial_bile_amount_is_zero():
    result = simulate_ehc_pk("DrugC", dose_mg=100.0)
    assert result.a_bile_mg[0] == 0.0


def test_initial_intestine_amount_is_zero():
    result = simulate_ehc_pk("DrugC", dose_mg=100.0)
    assert result.a_intestine_mg[0] == 0.0


def test_cmax_positive():
    result = simulate_ehc_pk("DrugD", dose_mg=100.0)
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = simulate_ehc_pk("DrugE", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_nonnegative():
    result = simulate_ehc_pk("DrugF", dose_mg=100.0)
    assert result.tmax_h >= 0.0


def test_times_list_length():
    result = simulate_ehc_pk("DrugG", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    assert len(result.times_h) > 0
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.a_bile_mg)
    assert len(result.times_h) == len(result.a_intestine_mg)


def test_notes_nonempty():
    result = simulate_ehc_pk("DrugH", dose_mg=100.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# EHC-specific behavior
# ---------------------------------------------------------------------------


def test_no_biliary_excretion_zero_peaks():
    """With f_biliary=0 there is no EHC, so no secondary peaks."""
    result = simulate_ehc_pk("DrugI", dose_mg=100.0, f_biliary=0.0)
    assert result.n_ehc_peaks == 0


def test_fe_biliary_pct_zero_when_no_biliary():
    result = simulate_ehc_pk("DrugJ", dose_mg=100.0, f_biliary=0.0)
    assert result.fe_biliary_pct == 0.0


def test_fe_biliary_pct_positive_with_biliary():
    result = simulate_ehc_pk("DrugK", dose_mg=100.0, f_biliary=0.3, f_reabsorbed=0.8)
    # fe_biliary_pct = f_biliary * (1 - f_reabsorbed) * 100 = 0.3 * 0.2 * 100 = 6.0
    assert abs(result.fe_biliary_pct - 6.0) < 1e-9


def test_bile_accumulates_over_time():
    """Bile compartment should accumulate drug during simulation."""
    result = simulate_ehc_pk("DrugL", dose_mg=100.0, f_biliary=0.3)
    max_bile = max(result.a_bile_mg)
    assert max_bile > 0.0


def test_ehc_increases_auc_vs_no_ehc():
    """EHC (f_biliary>0, f_reabsorbed>0) should yield higher AUC than no EHC."""
    comparison = compare_ehc_vs_no_ehc("DrugM", dose_mg=100.0, f_biliary=0.3)
    assert comparison["auc_ratio"] >= 1.0


def test_auc_ratio_is_one_with_zero_reabsorption():
    """If nothing is reabsorbed, AUC ratio should be approximately 1 (no net gain)."""
    result_ehc = simulate_ehc_pk("DrugN", dose_mg=100.0, f_biliary=0.3, f_reabsorbed=0.0)
    result_noehc = simulate_ehc_pk("DrugN", dose_mg=100.0, f_biliary=0.0, f_reabsorbed=0.0)
    # With f_reabsorbed=0, all bile is lost; EHC should not increase AUC
    assert result_ehc.auc_mg_h_per_L <= result_noehc.auc_mg_h_per_L + 0.01


def test_compare_ehc_vs_no_ehc_returns_dict():
    result = compare_ehc_vs_no_ehc("DrugO", dose_mg=100.0, f_biliary=0.3)
    assert isinstance(result, dict)


def test_compare_ehc_vs_no_ehc_keys():
    result = compare_ehc_vs_no_ehc("DrugP", dose_mg=100.0, f_biliary=0.3)
    assert "ehc_result" in result
    assert "no_ehc_result" in result
    assert "auc_ratio" in result
    assert "notes" in result


def test_compare_ehc_result_is_ehc_result():
    result = compare_ehc_vs_no_ehc("DrugQ", dose_mg=100.0, f_biliary=0.3)
    assert isinstance(result["ehc_result"], EHCResult)
    assert isinstance(result["no_ehc_result"], EHCResult)


def test_compare_notes_nonempty():
    result = compare_ehc_vs_no_ehc("DrugR", dose_mg=100.0, f_biliary=0.3)
    assert len(result["notes"]) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError):
        simulate_ehc_pk("DrugS", dose_mg=0.0)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError):
        simulate_ehc_pk("DrugT", dose_mg=-50.0)


def test_raises_on_f_biliary_equals_one():
    with pytest.raises(ValueError):
        simulate_ehc_pk("DrugU", dose_mg=100.0, f_biliary=1.0)


def test_raises_on_f_biliary_negative():
    with pytest.raises(ValueError):
        simulate_ehc_pk("DrugV", dose_mg=100.0, f_biliary=-0.1)


def test_t_half_positive():
    result = simulate_ehc_pk("DrugW", dose_mg=100.0, t_end_h=48.0)
    assert result.t_half_apparent_h >= 0.0


def test_drug_name_stored():
    result = simulate_ehc_pk("Morphine", dose_mg=100.0)
    assert result.drug_name == "Morphine"


def test_dose_stored():
    result = simulate_ehc_pk("DrugX", dose_mg=250.0)
    assert result.dose_mg == 250.0
