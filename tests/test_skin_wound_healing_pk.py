"""Tests for Phase 260: skin_wound_healing_pk."""

import pytest

from omega_pbpk.core.skin_wound_healing_pk import (
    FORMULATION_TYPES,
    WoundHealingPKResult,
    compare_wound_formulations,
    simulate_wound_healing_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structural tests
# ---------------------------------------------------------------------------


def test_returns_wound_healing_pk_result():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel")
    assert isinstance(result, WoundHealingPKResult)


def test_drug_name_preserved():
    result = simulate_wound_healing_pk("TestDrug", 10.0, "cream")
    assert result.drug_name == "TestDrug"


def test_formulation_type_preserved():
    result = simulate_wound_healing_pk("DrugA", 10.0, "film")
    assert result.formulation_type == "film"


def test_dose_preserved():
    result = simulate_wound_healing_pk("DrugA", 25.0, "antibiotic_ointment")
    assert result.dose_mg == 25.0


# ---------------------------------------------------------------------------
# Initial conditions: wound and tissue start at 0
# ---------------------------------------------------------------------------


def test_wound_concentration_starts_at_zero():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel")
    assert result.c_wound_mg_mL[0] == pytest.approx(0.0)


def test_tissue_concentration_starts_at_zero():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel")
    assert result.c_tissue_mg_mL[0] == pytest.approx(0.0)


def test_plasma_concentration_starts_at_zero():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Positive concentrations and AUC
# ---------------------------------------------------------------------------


def test_cmax_wound_positive():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel")
    assert result.cmax_wound_mg_mL > 0.0


def test_auc_wound_positive():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel")
    assert result.auc_wound_mg_h_per_mL > 0.0


def test_auc_plasma_positive():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel")
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_local_systemic_ratio_positive():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel")
    assert result.local_systemic_ratio > 0.0


# ---------------------------------------------------------------------------
# Wound coverage score range
# ---------------------------------------------------------------------------


def test_wound_coverage_score_in_range():
    for ft in FORMULATION_TYPES:
        result = simulate_wound_healing_pk("DrugA", 10.0, ft)
        assert 0.0 <= result.wound_coverage_score <= 100.0, (
            f"formulation={ft} gave score={result.wound_coverage_score}"
        )


# ---------------------------------------------------------------------------
# Formulation comparisons
# ---------------------------------------------------------------------------


def test_film_higher_auc_wound_than_cream():
    r_film = simulate_wound_healing_pk("DrugA", 10.0, "film")
    r_cream = simulate_wound_healing_pk("DrugA", 10.0, "cream")
    assert r_film.auc_wound_mg_h_per_mL > r_cream.auc_wound_mg_h_per_mL


def test_time_series_lengths_match():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel", t_end_h=24.0, dt_h=0.1)
    n = len(result.times_h)
    assert len(result.c_wound_mg_mL) == n
    assert len(result.c_tissue_mg_mL) == n
    assert len(result.c_plasma_mg_L) == n


def test_tmax_wound_within_simulation_window():
    result = simulate_wound_healing_pk("DrugA", 10.0, "hydrogel", t_end_h=48.0)
    assert 0.0 <= result.tmax_wound_h <= 48.0


# ---------------------------------------------------------------------------
# compare_wound_formulations
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_wound_formulations("DrugA", 10.0)
    assert len(results) == 4


def test_compare_sorted_by_auc_wound_descending():
    results = compare_wound_formulations("DrugA", 10.0)
    aucs = [r.auc_wound_mg_h_per_mL for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_four_formulations_present():
    results = compare_wound_formulations("DrugA", 10.0)
    types_found = {r.formulation_type for r in results}
    assert types_found == set(FORMULATION_TYPES.keys())


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_wound_healing_pk("DrugA", 0.0, "hydrogel")


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_wound_healing_pk("DrugA", -5.0, "hydrogel")


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_wound_healing_pk("DrugA", 10.0, "hydrogel", cl_L_per_h=0.0)


def test_invalid_cl_negative():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_wound_healing_pk("DrugA", 10.0, "hydrogel", cl_L_per_h=-1.0)


def test_invalid_formulation_type():
    with pytest.raises(ValueError, match="formulation_type"):
        simulate_wound_healing_pk("DrugA", 10.0, "spray")
