"""Tests for two-compartment oral PK model."""

import pytest

from omega_pbpk.core.two_compartment_oral_pk import (
    TwoCptOralResult,
    compare_one_vs_two_cpt,
    simulate_two_cpt_oral,
)

# --- Basic return type and structure ---


def test_returns_result_type():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert isinstance(result, TwoCptOralResult)


def test_route_is_oral():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.route == "oral"


def test_drug_name_preserved():
    result = simulate_two_cpt_oral("TestDrug", 50.0)
    assert result.drug_name == "TestDrug"


def test_dose_mg_preserved():
    result = simulate_two_cpt_oral("DrugA", 200.0)
    assert result.dose_mg == 200.0


# --- Initial conditions ---


def test_c1_starts_at_zero():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.c1_mg_L[0] == 0.0


def test_c2_starts_at_zero():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.c2_mg_L[0] == 0.0


def test_times_start_at_zero():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.times_h[0] == 0.0


# --- PK properties ---


def test_cmax_c1_positive():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.cmax_c1_mg_L > 0


def test_auc_c1_positive():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.auc_c1 > 0


def test_cmax_c2_positive():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.cmax_c2_mg_L > 0


def test_auc_c2_positive():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.auc_c2 > 0


def test_tmax_c2_after_tmax_c1():
    """Peripheral compartment peaks later than central."""
    result = simulate_two_cpt_oral(
        "DrugA", 100.0, ka_per_h=2.0, q_L_per_h=2.0, v2_L=40.0, t_end_h=48.0
    )
    assert result.tmax_c2_h > result.tmax_c1_h


def test_t_half_beta_positive():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.t_half_beta_h > 0


def test_vss_greater_than_v1():
    """Vss = V1 + V2*(k12/k21) > V1."""
    result = simulate_two_cpt_oral("DrugA", 100.0, v1_L=10.0, v2_L=40.0)
    assert result.vss_L > 10.0


# --- Dose linearity ---


def test_dose_linearity_cmax():
    """Doubling dose should double Cmax."""
    r1 = simulate_two_cpt_oral("DrugA", 100.0)
    r2 = simulate_two_cpt_oral("DrugA", 200.0)
    ratio = r2.cmax_c1_mg_L / r1.cmax_c1_mg_L
    assert abs(ratio - 2.0) < 0.05


def test_dose_linearity_auc():
    """Doubling dose should double AUC."""
    r1 = simulate_two_cpt_oral("DrugA", 100.0)
    r2 = simulate_two_cpt_oral("DrugA", 200.0)
    ratio = r2.auc_c1 / r1.auc_c1
    assert abs(ratio - 2.0) < 0.05


# --- Higher Q makes compartments more similar ---


def test_high_q_similar_compartments():
    """With very high intercompartmental CL, c1 and c2 equilibrate faster."""
    r_low_q = simulate_two_cpt_oral("DrugA", 100.0, q_L_per_h=0.5, t_end_h=48.0)
    r_high_q = simulate_two_cpt_oral("DrugA", 100.0, q_L_per_h=50.0, t_end_h=48.0)

    # With high Q, ratio of AUC2/AUC1 should be closer to V1/V2 equilibrium
    ratio_low_q = r_low_q.auc_c2 / r_low_q.auc_c1 if r_low_q.auc_c1 > 0 else 0
    ratio_high_q = r_high_q.auc_c2 / r_high_q.auc_c1 if r_high_q.auc_c1 > 0 else 0

    # With high Q both compartments equilibrate faster — ratio_high_q should be > ratio_low_q
    assert ratio_high_q > ratio_low_q


# --- Notes ---


def test_notes_nonempty():
    result = simulate_two_cpt_oral("DrugA", 100.0)
    assert result.notes and len(result.notes) > 0


# --- compare_one_vs_two_cpt ---


def test_compare_returns_dict():
    result = compare_one_vs_two_cpt("DrugA", 100.0)
    assert isinstance(result, dict)


def test_compare_has_two_cpt_key():
    result = compare_one_vs_two_cpt("DrugA", 100.0)
    assert "two_cpt" in result


def test_compare_has_one_cpt_key():
    result = compare_one_vs_two_cpt("DrugA", 100.0)
    assert "one_cpt" in result


def test_compare_has_auc_difference():
    result = compare_one_vs_two_cpt("DrugA", 100.0)
    assert "auc_difference_pct" in result
    assert result["auc_difference_pct"] >= 0


def test_compare_two_cpt_has_cmax():
    result = compare_one_vs_two_cpt("DrugA", 100.0)
    assert "cmax" in result["two_cpt"]
    assert result["two_cpt"]["cmax"] > 0


def test_compare_one_cpt_has_cmax():
    result = compare_one_vs_two_cpt("DrugA", 100.0)
    assert "cmax" in result["one_cpt"]
    assert result["one_cpt"]["cmax"] > 0


# --- Validation errors ---


def test_v1_zero_raises():
    with pytest.raises(ValueError):
        simulate_two_cpt_oral("DrugA", 100.0, v1_L=0.0)


def test_v1_negative_raises():
    with pytest.raises(ValueError):
        simulate_two_cpt_oral("DrugA", 100.0, v1_L=-5.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError):
        simulate_two_cpt_oral("DrugA", 100.0, cl_L_per_h=0.0)


def test_q_zero_raises():
    with pytest.raises(ValueError):
        simulate_two_cpt_oral("DrugA", 100.0, q_L_per_h=0.0)
