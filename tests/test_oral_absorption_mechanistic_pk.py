"""Tests for Phase 951 — Mechanistic oral absorption model."""

import pytest

from omega_pbpk.core.oral_absorption_mechanistic_pk import (
    MechanisticOralAbsorptionResult,
    sensitivity_analysis,
    simulate_mechanistic_oral_absorption,
)

# ── Basic return type and structure ──────────────────────────────────────────


def test_return_type():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert isinstance(r, MechanisticOralAbsorptionResult)


def test_c_plasma_starts_at_zero():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert r.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_cmax_positive():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert r.cmax_mg_L > 0.0


def test_auc_positive():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert r.auc_mg_h_per_L > 0.0


def test_tmax_positive():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert r.tmax_h > 0.0


def test_bioavailability_in_range():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert 0.0 <= r.bioavailability_pct <= 100.0


def test_e_hepatic_in_range():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert 0.0 <= r.e_hepatic <= 1.0


def test_fh_in_range():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert 0.0 <= r.fh <= 1.0


def test_fg_preserved():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, fg=0.65)
    assert r.fg == pytest.approx(0.65)


def test_fraction_absorbed_in_range():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert 0.0 <= r.fraction_absorbed <= 1.0


def test_a_solid_starts_at_dose():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=200.0)
    assert r.a_solid_mg[0] == pytest.approx(200.0)


def test_a_solid_monotone_decreasing():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, t_end_h=12.0)
    for i in range(1, len(r.a_solid_mg)):
        assert r.a_solid_mg[i] <= r.a_solid_mg[i - 1] + 1e-9


def test_higher_kdiss_higher_cmax():
    r_slow = simulate_mechanistic_oral_absorption(
        "DrugA", dose_mg=100.0, kdiss_per_h=0.1, t_end_h=48.0
    )
    r_fast = simulate_mechanistic_oral_absorption(
        "DrugA", dose_mg=100.0, kdiss_per_h=5.0, t_end_h=48.0
    )
    assert r_fast.cmax_mg_L > r_slow.cmax_mg_L


def test_higher_fg_higher_bioavailability():
    r_low = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, fg=0.2, t_end_h=48.0)
    r_high = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, fg=0.9, t_end_h=48.0)
    assert r_high.bioavailability_pct > r_low.bioavailability_pct


# ── Sensitivity analysis ──────────────────────────────────────────────────────


def test_sensitivity_returns_correct_length():
    values = [0.1, 0.5, 1.0, 2.0]
    results = sensitivity_analysis(
        "DrugA", dose_mg=100.0, param_name="kdiss_per_h", param_values=values
    )
    assert len(results) == len(values)


def test_sensitivity_sorted_by_auc_descending():
    values = [0.1, 0.5, 1.0, 2.0]
    results = sensitivity_analysis(
        "DrugA", dose_mg=100.0, param_name="kdiss_per_h", param_values=values
    )
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_sensitivity_on_ka():
    values = [0.5, 1.0, 2.0]
    results = sensitivity_analysis(
        "DrugA", dose_mg=100.0, param_name="ka_per_h", param_values=values
    )
    assert len(results) == 3


def test_sensitivity_on_fg():
    values = [0.3, 0.6, 0.9]
    results = sensitivity_analysis("DrugA", dose_mg=100.0, param_name="fg", param_values=values)
    assert all(isinstance(r, MechanisticOralAbsorptionResult) for r in results)


def test_sensitivity_on_clint():
    values = [1.0, 5.0, 20.0]
    results = sensitivity_analysis(
        "DrugA", dose_mg=100.0, param_name="clint_hepatic_L_per_h", param_values=values
    )
    assert len(results) == 3


# ── Notes ─────────────────────────────────────────────────────────────────────


def test_notes_nonempty():
    r = simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0)
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ── Validation ────────────────────────────────────────────────────────────────


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mechanistic_oral_absorption("DrugA", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mechanistic_oral_absorption("DrugA", dose_mg=-10.0)


def test_validation_kdiss_zero():
    with pytest.raises(ValueError, match="kdiss_per_h"):
        simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, kdiss_per_h=0.0)


def test_validation_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, ka_per_h=0.0)


def test_validation_fg_negative():
    with pytest.raises(ValueError, match="fg"):
        simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, fg=-0.1)


def test_validation_fg_above_one():
    with pytest.raises(ValueError, match="fg"):
        simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, fg=1.1)


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, fu_plasma=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_mechanistic_oral_absorption("DrugA", dose_mg=100.0, vd_L=0.0)
