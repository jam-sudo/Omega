"""Tests for Phase 484 — Inhaled Drug Deposition Model."""

import pytest

from omega_pbpk.core.inhaled_deposition import (
    InhaledDepositionResult,
    _compute_deposition_fractions,
    optimize_particle_size,
    simulate_inhaled_deposition,
)

# ── Return type ──────────────────────────────────────────────────────────────


def test_returns_correct_type():
    result = simulate_inhaled_deposition("DrugX", total_dose_mg=1.0, MMAD_um=2.0)
    assert isinstance(result, InhaledDepositionResult)


def test_result_has_list_fields():
    result = simulate_inhaled_deposition("DrugX", total_dose_mg=1.0, MMAD_um=2.0)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_systemic_mg_L, list)


# ── Deposition fractions sum to 1 ────────────────────────────────────────────


def test_fractions_sum_to_1_mmad_2():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0)
    total = r.f_oropharyngeal + r.f_tracheobronchial + r.f_alveolar + r.f_exhaled
    assert abs(total - 1.0) < 1e-9


def test_fractions_sum_to_1_mmad_10():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=10.0)
    total = r.f_oropharyngeal + r.f_tracheobronchial + r.f_alveolar + r.f_exhaled
    assert abs(total - 1.0) < 1e-9


def test_fractions_sum_to_1_mmad_05():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=0.5)
    total = r.f_oropharyngeal + r.f_tracheobronchial + r.f_alveolar + r.f_exhaled
    assert abs(total - 1.0) < 1e-9


def test_fractions_sum_to_1_mmad_3():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=3.0)
    total = r.f_oropharyngeal + r.f_tracheobronchial + r.f_alveolar + r.f_exhaled
    assert abs(total - 1.0) < 1e-9


# ── Optimal alveolar deposition ──────────────────────────────────────────────


def test_alveolar_positive_at_mmad_2():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0)
    assert r.f_alveolar > 0.0


def test_alveolar_greater_than_zero_at_mmad_1():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=1.0)
    # At MMAD=1, alveolar = 0.3 * (1 - 1/3) = 0.2
    assert r.f_alveolar > 0.0


# ── Large particles → oropharyngeal ──────────────────────────────────────────


def test_oropharyngeal_dominates_at_mmad_10():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=10.0)
    assert r.f_oropharyngeal > r.f_alveolar


def test_alveolar_zero_at_mmad_10():
    fo, ftb, fa, fex = _compute_deposition_fractions(10.0)
    # After normalization, alveolar raw = 0.0 for MMAD >= 5
    assert fa == 0.0 or fa < 0.01  # should be 0 before normalization


# ── Fractions in [0, 1] ──────────────────────────────────────────────────────


def test_all_fractions_non_negative_mmad_2():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0)
    assert r.f_oropharyngeal >= 0.0
    assert r.f_tracheobronchial >= 0.0
    assert r.f_alveolar >= 0.0
    assert r.f_exhaled >= 0.0


def test_all_fractions_at_most_1_mmad_5():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=5.0)
    assert r.f_oropharyngeal <= 1.0
    assert r.f_tracheobronchial <= 1.0
    assert r.f_alveolar <= 1.0
    assert r.f_exhaled <= 1.0


# ── PK output ────────────────────────────────────────────────────────────────


def test_cmax_positive():
    r = simulate_inhaled_deposition("D", total_dose_mg=10.0, MMAD_um=2.0)
    assert r.cmax_systemic_mg_L > 0.0


def test_auc_positive():
    r = simulate_inhaled_deposition("D", total_dose_mg=10.0, MMAD_um=2.0)
    assert r.auc_systemic_mg_h_per_L > 0.0


def test_higher_dose_gives_higher_cmax():
    r_low = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0)
    r_high = simulate_inhaled_deposition("D", total_dose_mg=10.0, MMAD_um=2.0)
    assert r_high.cmax_systemic_mg_L > r_low.cmax_systemic_mg_L


def test_times_and_concentrations_same_length():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0)
    assert len(r.times_h) == len(r.c_systemic_mg_L)


def test_concentrations_non_negative():
    r = simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0)
    assert all(c >= 0 for c in r.c_systemic_mg_L)


def test_lung_bioavailability_pct_positive():
    r = simulate_inhaled_deposition("D", total_dose_mg=10.0, MMAD_um=2.0)
    assert r.lung_bioavailability_pct > 0.0


def test_lung_bioavailability_pct_at_most_100():
    r = simulate_inhaled_deposition("D", total_dose_mg=10.0, MMAD_um=2.0)
    assert r.lung_bioavailability_pct <= 100.0


# ── optimize_particle_size ───────────────────────────────────────────────────


def test_optimize_returns_list():
    results = optimize_particle_size("DrugO", total_dose_mg=5.0)
    assert isinstance(results, list)


def test_optimize_default_6_values():
    results = optimize_particle_size("DrugO", total_dose_mg=5.0)
    assert len(results) == 6


def test_optimize_sorted_by_auc_descending():
    results = optimize_particle_size("DrugO", total_dose_mg=5.0)
    for i in range(len(results) - 1):
        assert results[i].auc_systemic_mg_h_per_L >= results[i + 1].auc_systemic_mg_h_per_L


def test_optimize_custom_mmad_values():
    results = optimize_particle_size("DrugO", total_dose_mg=5.0, MMAD_values=[1.0, 2.0, 3.0])
    assert len(results) == 3


# ── Validation errors ─────────────────────────────────────────────────────────


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="total_dose_mg"):
        simulate_inhaled_deposition("D", total_dose_mg=0.0, MMAD_um=2.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="total_dose_mg"):
        simulate_inhaled_deposition("D", total_dose_mg=-1.0, MMAD_um=2.0)


def test_invalid_mmad_zero():
    with pytest.raises(ValueError, match="MMAD_um"):
        simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=0.0)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0, cl_sys_L_per_h=0.0)


def test_invalid_vd_negative():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0, vd_sys_L=-1.0)


def test_invalid_f_oral_above_1():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_inhaled_deposition("D", total_dose_mg=1.0, MMAD_um=2.0, f_oral=1.5)


# ── drug_name propagation ─────────────────────────────────────────────────────


def test_drug_name_in_result():
    r = simulate_inhaled_deposition("Salbutamol", total_dose_mg=0.1, MMAD_um=2.0)
    assert r.drug_name == "Salbutamol"
