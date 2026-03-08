"""
Tests for Phase 437: Multi-cycle chemotherapy PK model.
"""

import pytest

from omega_pbpk.clinical.chemotherapy_pk_model import (
    ChemotherapyPKResult,
    optimize_chemotherapy_schedule,
    simulate_chemotherapy_pk,
)

# ─────────────────────────────────────────────────────────────────────────────
# Basic return type and structure tests
# ─────────────────────────────────────────────────────────────────────────────


def test_return_type():
    result = simulate_chemotherapy_pk(
        drug_name="Doxorubicin",
        dose_per_cycle_mg=50.0,
        n_cycles=3,
        cl_L_per_h=10.0,
        vd_L=100.0,
    )
    assert isinstance(result, ChemotherapyPKResult)


def test_drug_name_preserved():
    result = simulate_chemotherapy_pk(
        drug_name="Cisplatin",
        dose_per_cycle_mg=75.0,
        n_cycles=2,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert result.drug_name == "Cisplatin"


def test_n_cycles_preserved():
    result = simulate_chemotherapy_pk(
        drug_name="Paclitaxel",
        dose_per_cycle_mg=100.0,
        n_cycles=4,
        cl_L_per_h=15.0,
        vd_L=200.0,
    )
    assert result.n_cycles == 4


def test_dose_reductions_list_length():
    n = 5
    result = simulate_chemotherapy_pk(
        drug_name="Carboplatin",
        dose_per_cycle_mg=300.0,
        n_cycles=n,
        cl_L_per_h=8.0,
        vd_L=150.0,
    )
    assert len(result.dose_reductions) == n


def test_auc_per_cycle_list_length():
    n = 6
    result = simulate_chemotherapy_pk(
        drug_name="Cyclophosphamide",
        dose_per_cycle_mg=500.0,
        n_cycles=n,
        cl_L_per_h=12.0,
        vd_L=250.0,
    )
    assert len(result.auc_per_cycle) == n


def test_cumulative_auc_positive():
    result = simulate_chemotherapy_pk(
        drug_name="Vincristine",
        dose_per_cycle_mg=2.0,
        n_cycles=3,
        cl_L_per_h=1.0,
        vd_L=10.0,
    )
    assert result.cumulative_auc_mg_h_per_L > 0.0


def test_toxicity_score_between_0_and_1():
    result = simulate_chemotherapy_pk(
        drug_name="Fluorouracil",
        dose_per_cycle_mg=500.0,
        n_cycles=3,
        cl_L_per_h=20.0,
        vd_L=100.0,
    )
    assert 0.0 <= result.toxicity_score <= 1.0


def test_toxicity_score_increases_with_cycles():
    """More cycles → higher toxicity_score."""
    result_few = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=2,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    result_many = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=8,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert result_many.toxicity_score > result_few.toxicity_score


def test_high_cycles_toxicity_above_threshold():
    """With enough cycles, toxicity_score should exceed 0.5."""
    result = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=200.0,
        n_cycles=10,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert result.toxicity_score > 0.5


def test_dose_modifications_non_negative():
    result = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=4,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert result.dose_modifications >= 0


def test_average_dose_intensity_between_0_and_1():
    result = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=6,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert 0.0 < result.average_dose_intensity <= 1.0


def test_t_half_positive():
    result = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=3,
        cl_L_per_h=10.0,
        vd_L=100.0,
    )
    assert result.t_half_h > 0.0


def test_t_half_formula():
    """t_half = 0.693 * Vd / CL."""
    cl = 10.0
    vd = 100.0
    result = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=2,
        cl_L_per_h=cl,
        vd_L=vd,
    )
    expected = 0.693 * vd / cl
    assert abs(result.t_half_h - expected) < 0.001


def test_times_days_is_list():
    result = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=2,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert isinstance(result.times_days, list)
    assert len(result.times_days) > 0


def test_c_plasma_non_negative():
    result = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=3,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


def test_dose_linearity_cumulative_auc():
    """2x dose → 2x cumulative AUC (low cycles so no modifications)."""
    result_1x = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=50.0,
        n_cycles=2,
        cl_L_per_h=10.0,
        vd_L=100.0,
    )
    result_2x = simulate_chemotherapy_pk(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        n_cycles=2,
        cl_L_per_h=10.0,
        vd_L=100.0,
    )
    ratio = result_2x.cumulative_auc_mg_h_per_L / result_1x.cumulative_auc_mg_h_per_L
    assert abs(ratio - 2.0) < 0.01


def test_optimize_returns_3_items():
    results = optimize_chemotherapy_schedule(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    assert len(results) == 3


def test_optimize_sorted_by_cumulative_auc_descending():
    results = optimize_chemotherapy_schedule(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    aucs = [r.cumulative_auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_optimize_all_results_are_chemo_pk_results():
    results = optimize_chemotherapy_schedule(
        drug_name="Drug",
        dose_per_cycle_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
    )
    for r in results:
        assert isinstance(r, ChemotherapyPKResult)


# ─────────────────────────────────────────────────────────────────────────────
# Validation / error tests
# ─────────────────────────────────────────────────────────────────────────────


def test_validation_dose_le_zero():
    with pytest.raises(ValueError, match="dose_per_cycle_mg"):
        simulate_chemotherapy_pk("Drug", 0.0, 3, 5.0, 50.0)


def test_validation_negative_dose():
    with pytest.raises(ValueError, match="dose_per_cycle_mg"):
        simulate_chemotherapy_pk("Drug", -10.0, 3, 5.0, 50.0)


def test_validation_n_cycles_lt_1():
    with pytest.raises(ValueError, match="n_cycles"):
        simulate_chemotherapy_pk("Drug", 100.0, 0, 5.0, 50.0)


def test_validation_interval_lt_7():
    with pytest.raises(ValueError, match="cycle_interval_days"):
        simulate_chemotherapy_pk("Drug", 100.0, 3, 5.0, 50.0, cycle_interval_days=6)
