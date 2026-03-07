"""Tests for Phase 971 — Splenic PK model."""
import pytest
from omega_pbpk.core.splenic_pk import (
    SplenicPKResult,
    simulate_splenic_pk,
    screen_spleen_trapping,
)


def test_return_type_is_splenic_pk_result():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert isinstance(result, SplenicPKResult)


def test_cmax_plasma_positive():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert result.cmax_plasma > 0


def test_cmax_spleen_positive():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert result.cmax_spleen > 0


def test_auc_plasma_positive():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert result.auc_plasma > 0


def test_auc_spleen_positive():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert result.auc_spleen > 0


def test_spleen_to_plasma_ratio_positive():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert result.spleen_to_plasma_ratio > 0


def test_kp_spleen_used_positive():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert result.kp_spleen_used > 0


def test_spleen_exposure_index_positive():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert result.spleen_exposure_index > 0


def test_high_logp_higher_kp_spleen():
    res_low = simulate_splenic_pk("DrugA", dose_mg=10.0, logp=0.0)
    res_high = simulate_splenic_pk("DrugB", dose_mg=10.0, logp=4.0)
    assert res_high.kp_spleen_used > res_low.kp_spleen_used


def test_high_logp_higher_spleen_to_plasma_ratio():
    res_low = simulate_splenic_pk("DrugA", dose_mg=10.0, logp=0.0)
    res_high = simulate_splenic_pk("DrugB", dose_mg=10.0, logp=4.0)
    assert res_high.spleen_to_plasma_ratio > res_low.spleen_to_plasma_ratio


def test_iv_initial_plasma_nonzero():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0, route="iv")
    assert result.c_plasma_mg_L[0] > 0


def test_oral_initial_plasma_zero():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0, route="oral")
    assert result.c_plasma_mg_L[0] == 0.0


def test_screen_returns_correct_length():
    logp_values = [0.0, 1.0, 2.0, 3.0]
    results = screen_spleen_trapping("DrugX", logp_values=logp_values)
    assert len(results) == 4


def test_screen_sorted_by_spleen_to_plasma_ratio_descending():
    logp_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    results = screen_spleen_trapping("DrugX", logp_values=logp_values)
    ratios = [r.spleen_to_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_notes_non_empty():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert result.notes != ""


def test_times_and_c_plasma_same_length():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_c_spleen_and_c_plasma_same_length():
    result = simulate_splenic_pk("DrugA", dose_mg=10.0)
    assert len(result.c_spleen_mg_L) == len(result.c_plasma_mg_L)


def test_validation_dose_mg_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_splenic_pk("DrugA", dose_mg=0.0)


def test_validation_dose_mg_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_splenic_pk("DrugA", dose_mg=-5.0)


def test_validation_cl_hepatic_zero():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        simulate_splenic_pk("DrugA", dose_mg=10.0, cl_hepatic_L_per_h=0.0)


def test_validation_v_plasma_zero():
    with pytest.raises(ValueError, match="v_plasma_L"):
        simulate_splenic_pk("DrugA", dose_mg=10.0, v_plasma_L=0.0)


def test_validation_v_spleen_zero():
    with pytest.raises(ValueError, match="v_spleen_L"):
        simulate_splenic_pk("DrugA", dose_mg=10.0, v_spleen_L=0.0)


def test_validation_q_spleen_zero():
    with pytest.raises(ValueError, match="q_spleen_L_per_h"):
        simulate_splenic_pk("DrugA", dose_mg=10.0, q_spleen_L_per_h=0.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_splenic_pk("DrugA", dose_mg=10.0, route="subcutaneous")


def test_oral_absorption_peak_occurs():
    """Oral dosing: plasma peaks after t=0 (absorption phase)."""
    result = simulate_splenic_pk("DrugA", dose_mg=10.0, route="oral", ka_per_h=1.0)
    # plasma should peak after t=0
    peak_idx = result.c_plasma_mg_L.index(result.cmax_plasma)
    assert peak_idx > 0


def test_drug_name_preserved():
    result = simulate_splenic_pk("TestDrug", dose_mg=10.0)
    assert result.drug_name == "TestDrug"


def test_dose_mg_preserved():
    result = simulate_splenic_pk("DrugA", dose_mg=25.0)
    assert result.dose_mg == 25.0
