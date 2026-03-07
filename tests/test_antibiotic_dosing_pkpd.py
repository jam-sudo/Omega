"""Tests for Phase 957 — Antibiotic PK/PD dosing optimization."""

import pytest

from omega_pbpk.clinical.antibiotic_dosing_pkpd import (
    AntibioticDosingResult,
    optimize_antibiotic_dosing,
    screen_dose_mic_combinations,
)

# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


def test_return_type():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert isinstance(result, AntibioticDosingResult)


def test_cmax_free_positive():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert result.cmax_free_mg_L > 0


def test_auc_free_24h_positive():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert result.auc_free_24h_mg_h_per_L > 0


def test_ft_above_mic_pct_range():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert 0.0 <= result.ft_above_mic_pct <= 100.0


def test_auc_mic_ratio_positive():
    result = optimize_antibiotic_dosing("ciprofloxacin", "fluoroquinolone", dose_mg=400.0)
    assert result.auc_mic_ratio > 0


def test_cmax_mic_ratio_positive():
    result = optimize_antibiotic_dosing("gentamicin", "aminoglycoside", dose_mg=200.0)
    assert result.cmax_mic_ratio > 0


def test_target_attained_is_bool():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert isinstance(result.target_attained, bool)


def test_pkpd_index_nonempty():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert isinstance(result.pkpd_index, str) and len(result.pkpd_index) > 0


def test_notes_nonempty():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_times_h_nonempty():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert len(result.times_h) > 0


# ---------------------------------------------------------------------------
# High dose vs low dose
# ---------------------------------------------------------------------------


def test_high_dose_target_attained():
    """Very high dose should attain target for beta-lactam (fT>MIC > 50%)."""
    high = optimize_antibiotic_dosing(
        "piperacillin",
        "beta_lactam",
        dose_mg=4000.0,
        dosing_interval_h=6.0,
        mic_mg_L=0.5,
        cl_L_per_h=5.0,
        vd_L=20.0,
        fu=0.7,
    )
    low = optimize_antibiotic_dosing(
        "piperacillin",
        "beta_lactam",
        dose_mg=10.0,
        dosing_interval_h=24.0,
        mic_mg_L=8.0,
        cl_L_per_h=20.0,
        vd_L=20.0,
        fu=0.7,
    )
    assert high.target_attained is True
    assert low.target_attained is False


# ---------------------------------------------------------------------------
# PK/PD index per class
# ---------------------------------------------------------------------------


def test_beta_lactam_pkpd_index():
    result = optimize_antibiotic_dosing("amoxicillin", "beta_lactam", dose_mg=500.0)
    assert "fT" in result.pkpd_index or "time" in result.pkpd_index.lower()


def test_fluoroquinolone_pkpd_index():
    result = optimize_antibiotic_dosing("ciprofloxacin", "fluoroquinolone", dose_mg=400.0)
    assert "AUC" in result.pkpd_index or "AUIC" in result.pkpd_index


def test_aminoglycoside_pkpd_index():
    result = optimize_antibiotic_dosing("gentamicin", "aminoglycoside", dose_mg=200.0)
    assert "Cmax" in result.pkpd_index


# ---------------------------------------------------------------------------
# screen_dose_mic_combinations
# ---------------------------------------------------------------------------


def test_screen_returns_correct_length():
    doses = [250.0, 500.0, 1000.0]
    mics = [0.5, 2.0]
    results = screen_dose_mic_combinations("amoxicillin", "beta_lactam", doses, mics)
    assert len(results) == len(doses) * len(mics)


def test_screen_sorted_target_attained_first():
    doses = [50.0, 2000.0]
    mics = [0.5, 8.0]
    results = screen_dose_mic_combinations(
        "amoxicillin", "beta_lactam", doses, mics, cl_L_per_h=10.0, vd_L=20.0, fu=0.7
    )
    # All True results should come before False results
    seen_false = False
    for r in results:
        if not r.target_attained:
            seen_false = True
        if seen_false:
            assert not r.target_attained, "False target_attained found after True in sorted list"


def test_low_mic_easier_target_attained():
    """Lower MIC → easier to attain target."""
    low_mic = optimize_antibiotic_dosing(
        "ciprofloxacin",
        "fluoroquinolone",
        dose_mg=400.0,
        mic_mg_L=0.01,
        cl_L_per_h=10.0,
        vd_L=20.0,
        fu=0.7,
    )
    high_mic = optimize_antibiotic_dosing(
        "ciprofloxacin",
        "fluoroquinolone",
        dose_mg=400.0,
        mic_mg_L=100.0,
        cl_L_per_h=10.0,
        vd_L=20.0,
        fu=0.7,
    )
    assert low_mic.auc_mic_ratio > high_mic.auc_mic_ratio


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_mg():
    with pytest.raises(ValueError):
        optimize_antibiotic_dosing("drug", "beta_lactam", dose_mg=0.0)


def test_invalid_dose_mg_negative():
    with pytest.raises(ValueError):
        optimize_antibiotic_dosing("drug", "beta_lactam", dose_mg=-100.0)


def test_invalid_mic_mg_L():
    with pytest.raises(ValueError):
        optimize_antibiotic_dosing("drug", "beta_lactam", dose_mg=500.0, mic_mg_L=0.0)


def test_invalid_cl_L_per_h():
    with pytest.raises(ValueError):
        optimize_antibiotic_dosing("drug", "beta_lactam", dose_mg=500.0, cl_L_per_h=-1.0)


def test_invalid_vd_L():
    with pytest.raises(ValueError):
        optimize_antibiotic_dosing("drug", "beta_lactam", dose_mg=500.0, vd_L=0.0)


def test_invalid_fu_zero():
    with pytest.raises(ValueError):
        optimize_antibiotic_dosing("drug", "beta_lactam", dose_mg=500.0, fu=0.0)


def test_invalid_fu_gt_one():
    with pytest.raises(ValueError):
        optimize_antibiotic_dosing("drug", "beta_lactam", dose_mg=500.0, fu=1.5)


def test_invalid_antibiotic_class():
    with pytest.raises(ValueError):
        optimize_antibiotic_dosing("drug", "unknown_class", dose_mg=500.0)


# ---------------------------------------------------------------------------
# Additional checks
# ---------------------------------------------------------------------------


def test_drug_name_preserved():
    result = optimize_antibiotic_dosing("TestDrug", "generic", dose_mg=500.0)
    assert result.drug_name == "TestDrug"


def test_oral_absorption():
    """Oral absorption (ka > 0) should still give positive concentrations."""
    result = optimize_antibiotic_dosing(
        "amoxicillin_oral", "beta_lactam", dose_mg=500.0, ka_per_h=1.5, fu=0.8
    )
    assert result.cmax_free_mg_L > 0
    assert result.auc_free_24h_mg_h_per_L > 0


def test_generic_class_uses_auc_index():
    result = optimize_antibiotic_dosing("generic_drug", "generic", dose_mg=500.0)
    assert "AUC" in result.pkpd_index


def test_fluoroquinolone_high_dose_target():
    """High-dose fluoroquinolone with low MIC should hit AUIC > 25."""
    result = optimize_antibiotic_dosing(
        "ciprofloxacin",
        "fluoroquinolone",
        dose_mg=800.0,
        mic_mg_L=0.1,
        cl_L_per_h=5.0,
        vd_L=20.0,
        fu=0.6,
    )
    assert result.target_attained is True


def test_aminoglycoside_high_dose_target():
    """High-dose aminoglycoside should hit Cmax/MIC > 10."""
    result = optimize_antibiotic_dosing(
        "gentamicin",
        "aminoglycoside",
        dose_mg=600.0,
        mic_mg_L=1.0,
        cl_L_per_h=5.0,
        vd_L=10.0,
        fu=0.9,
    )
    assert result.target_attained is True
