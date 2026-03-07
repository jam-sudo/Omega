"""Tests for Phase 912 — Drug Rectal/Colonic Absorption."""

import pytest

from omega_pbpk.clinical.rectal_absorption import (
    RectalAbsorptionResult,
    compare_rectal_formulations,
    simulate_rectal_absorption,
)

# ---------------------------------------------------------------------------
# Basic return type and field preservation
# ---------------------------------------------------------------------------


def test_returns_dataclass():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert isinstance(result, RectalAbsorptionResult)


def test_drug_name_preserved():
    result = simulate_rectal_absorption("indomethacin", dose_mg=100.0)
    assert result.drug_name == "indomethacin"


def test_dose_preserved():
    result = simulate_rectal_absorption("diclofenac", dose_mg=50.0)
    assert result.dose_mg == 50.0


def test_default_formulation():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert result.formulation == "suppository_wax"


def test_formulation_preserved():
    result = simulate_rectal_absorption("metronidazole", dose_mg=500.0, formulation="enema")
    assert result.formulation == "enema"


def test_rectal_ph_default():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=325.0)
    assert result.rectal_ph == 7.4


def test_rectal_ph_custom():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=325.0, rectal_ph=6.8)
    assert result.rectal_ph == 6.8


# ---------------------------------------------------------------------------
# ODE-derived PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert result.cmax_mg_L > 0.0


def test_tmax_positive():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert result.tmax_h > 0.0


def test_auc_positive():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert result.auc_mg_h_per_L > 0.0


def test_times_length_consistent():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0, t_end_h=6.0, dt_h=0.1)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) > 0


def test_times_start_at_zero():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert result.times_h[0] == 0.0


def test_cmax_matches_max_concentration():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert abs(result.cmax_mg_L - max(result.c_plasma_mg_L)) < 1e-9


def test_tmax_matches_cmax_time():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    idx = list(result.c_plasma_mg_L).index(result.cmax_mg_L)
    assert abs(result.tmax_h - result.times_h[idx]) < 1e-9


def test_f_rectal_between_zero_and_one():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert 0.0 <= result.f_rectal <= 1.0


def test_first_pass_bypass_fraction():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert result.first_pass_bypass_fraction == 0.5


def test_compared_to_oral_ratio_clamped():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert 0.1 <= result.compared_to_oral_ratio <= 5.0


# ---------------------------------------------------------------------------
# Formulation differences
# ---------------------------------------------------------------------------


def test_enema_higher_auc_than_wax():
    """Enema has higher absorption fraction and faster release."""
    enema = simulate_rectal_absorption("drug", dose_mg=100.0, formulation="enema")
    wax = simulate_rectal_absorption("drug", dose_mg=100.0, formulation="suppository_wax")
    assert enema.auc_mg_h_per_L > wax.auc_mg_h_per_L


def test_peg_higher_auc_than_wax():
    peg = simulate_rectal_absorption("drug", dose_mg=100.0, formulation="suppository_PEG")
    wax = simulate_rectal_absorption("drug", dose_mg=100.0, formulation="suppository_wax")
    assert peg.auc_mg_h_per_L > wax.auc_mg_h_per_L


def test_dose_linearity():
    """Doubling the dose should roughly double AUC."""
    low = simulate_rectal_absorption("drug", dose_mg=100.0)
    high = simulate_rectal_absorption("drug", dose_mg=200.0)
    ratio = high.auc_mg_h_per_L / low.auc_mg_h_per_L
    assert 1.8 < ratio < 2.2


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_not_empty():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert len(result.notes) > 0


def test_bypass_note_present():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    assert "bypass" in result.notes.lower() or "first-pass" in result.notes.lower()


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_rectal_absorption("drug", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_rectal_absorption("drug", dose_mg=-50.0)


def test_invalid_formulation_raises():
    with pytest.raises(ValueError, match="formulation"):
        simulate_rectal_absorption("drug", dose_mg=100.0, formulation="unknown_form")


# ---------------------------------------------------------------------------
# compare_rectal_formulations
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_rectal_formulations("acetaminophen", dose_mg=500.0)
    assert len(results) == 4


def test_compare_all_formulations_represented():
    results = compare_rectal_formulations("acetaminophen", dose_mg=500.0)
    formulations = {r.formulation for r in results}
    assert formulations == {"suppository_wax", "suppository_PEG", "enema", "gel"}


def test_compare_sorted_by_auc_descending():
    results = compare_rectal_formulations("acetaminophen", dose_mg=500.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_kwargs_forwarded():
    """Custom cl_L_per_h should change AUC."""
    fast_cl = compare_rectal_formulations("drug", dose_mg=100.0, cl_L_per_h=20.0)
    slow_cl = compare_rectal_formulations("drug", dose_mg=100.0, cl_L_per_h=5.0)
    # Faster clearance → lower AUC for first formulation
    assert fast_cl[0].auc_mg_h_per_L < slow_cl[0].auc_mg_h_per_L


def test_result_is_frozen():
    result = simulate_rectal_absorption("acetaminophen", dose_mg=650.0)
    with pytest.raises(Exception):
        result.drug_name = "other"  # type: ignore[misc]
