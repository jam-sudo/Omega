"""Tests for Phase 959 — organ_perfusion_pk."""

import pytest

from omega_pbpk.core.organ_perfusion_pk import (
    OrganPerfusionPKResult,
    screen_organ_exposure,
    simulate_organ_perfusion,
)

# ---------------------------------------------------------------------------
# Basic return type & structure
# ---------------------------------------------------------------------------


def test_return_type():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0)
    assert isinstance(r, OrganPerfusionPKResult)


def test_c_arterial_iv_starts_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0, route="iv")
    assert r.c_arterial_mg_L[0] > 0.0


def test_c_arterial_oral_starts_zero():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0, route="oral")
    assert r.c_arterial_mg_L[0] == 0.0


def test_cmax_arterial_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0)
    assert r.cmax_arterial > 0.0


def test_auc_arterial_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0)
    assert r.auc_arterial > 0.0


def test_auc_liver_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0)
    assert r.auc_liver > 0.0


def test_auc_brain_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0)
    assert r.auc_brain > 0.0


def test_liver_to_blood_ratio_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0)
    assert r.liver_to_blood_auc_ratio > 0.0


def test_brain_to_blood_ratio_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0)
    assert r.brain_to_blood_auc_ratio > 0.0


# ---------------------------------------------------------------------------
# High logP → higher brain penetration
# ---------------------------------------------------------------------------


def test_high_logp_higher_brain_ratio():
    r_lo = simulate_organ_perfusion("DrugLo", dose_mg=10.0, logp=-1.0)
    r_hi = simulate_organ_perfusion("DrugHi", dose_mg=10.0, logp=4.0)
    assert r_hi.brain_to_blood_auc_ratio > r_lo.brain_to_blood_auc_ratio


# ---------------------------------------------------------------------------
# Liver AUC > brain AUC (higher Kp and flow)
# ---------------------------------------------------------------------------


def test_liver_auc_greater_than_brain_auc():
    r = simulate_organ_perfusion("DrugA", dose_mg=100.0, logp=2.0)
    assert r.auc_liver > r.auc_brain


# ---------------------------------------------------------------------------
# List lengths match
# ---------------------------------------------------------------------------


def test_times_length_matches_c_arterial():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=1.0, t_end_h=4.0, dt_h=0.1)
    assert len(r.times_h) == len(r.c_arterial_mg_L)


def test_c_kidney_is_list():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=1.0)
    assert isinstance(r.c_kidney_mg_L, list)


def test_c_lung_is_list():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=1.0)
    assert isinstance(r.c_lung_mg_L, list)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=1.0)
    assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Drug name preserved
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    r = simulate_organ_perfusion("TestDrug", dose_mg=5.0, logp=1.0)
    assert r.drug_name == "TestDrug"


def test_dose_preserved():
    r = simulate_organ_perfusion("TestDrug", dose_mg=50.0, logp=1.0)
    assert r.dose_mg == 50.0


# ---------------------------------------------------------------------------
# screen_organ_exposure
# ---------------------------------------------------------------------------


def test_screen_returns_correct_length():
    logp_values = [0.0, 1.0, 2.0, 3.0]
    results = screen_organ_exposure("DrugX", logp_values=logp_values, dose_mg=10.0)
    assert len(results) == 4


def test_screen_sorted_by_brain_ratio_descending():
    logp_values = [-1.0, 0.5, 2.0, 4.0]
    results = screen_organ_exposure("DrugX", logp_values=logp_values, dose_mg=10.0)
    ratios = [r.brain_to_blood_auc_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_all_results_are_correct_type():
    results = screen_organ_exposure("DrugX", logp_values=[1.0, 2.0], dose_mg=10.0)
    for r in results:
        assert isinstance(r, OrganPerfusionPKResult)


# ---------------------------------------------------------------------------
# Oral simulation
# ---------------------------------------------------------------------------


def test_oral_auc_arterial_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0, route="oral", ka_per_h=1.5)
    assert r.auc_arterial > 0.0


def test_oral_cmax_arterial_positive():
    r = simulate_organ_perfusion("DrugA", dose_mg=10.0, logp=2.0, route="oral", ka_per_h=1.5)
    assert r.cmax_arterial > 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_organ_perfusion("DrugA", dose_mg=0.0, logp=1.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_organ_perfusion("DrugA", dose_mg=-5.0, logp=1.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        simulate_organ_perfusion("DrugA", dose_mg=10.0, cl_hepatic_L_per_h=0.0)


def test_validation_cl_negative():
    with pytest.raises(ValueError, match="cl_hepatic_L_per_h"):
        simulate_organ_perfusion("DrugA", dose_mg=10.0, cl_hepatic_L_per_h=-1.0)


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_organ_perfusion("DrugA", dose_mg=10.0, route="subcutaneous")
