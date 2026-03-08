"""
Tests for Phase 635 — Antimicrobial PK/PD Integration
"""

import pytest

from omega_pbpk.clinical.antimicrobial_pkpd_p635 import (
    AntimicrobialPKPDResult,
    simulate_antimicrobial_pkpd,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="ciprofloxacin",
        pathogen="E_coli",
        mic_mg_L=0.25,
        dose_mg=400.0,
        vd_L=100.0,
        cl_L_per_h=20.0,
        pkpd_index="AUC/MIC",
        tau_h=12.0,
        n_doses=3,
    )
    params.update(kwargs)
    return simulate_antimicrobial_pkpd(**params)


# ---------------------------------------------------------------------------
# Return type and structure tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = default_result()
    assert isinstance(result, AntimicrobialPKPDResult)


def test_result_fields_present():
    result = default_result()
    assert hasattr(result, "drug_name")
    assert hasattr(result, "pathogen")
    assert hasattr(result, "mic_mg_L")
    assert hasattr(result, "pkpd_index")
    assert hasattr(result, "times_h")
    assert hasattr(result, "c_drug_mg_L")
    assert hasattr(result, "log10_cfu")
    assert hasattr(result, "times_above_mic_h")
    assert hasattr(result, "fT_above_mic_pct")
    assert hasattr(result, "auc_mic_ratio")
    assert hasattr(result, "cmax_mic_ratio")
    assert hasattr(result, "predicted_outcome")
    assert hasattr(result, "resistance_risk")
    assert hasattr(result, "optimal_pkpd_target_met")
    assert hasattr(result, "notes")


def test_list_fields_are_lists():
    result = default_result()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_drug_mg_L, list)
    assert isinstance(result.log10_cfu, list)


def test_list_lengths_equal():
    result = default_result()
    assert len(result.times_h) == len(result.c_drug_mg_L) == len(result.log10_cfu)


def test_list_length_matches_simulation():
    result = default_result(tau_h=12.0, n_doses=3, dt_h=0.1)
    # t_end = 3*12+12 = 48h; steps = 480 + 1 initial
    expected_len = int(round(48.0 / 0.1)) + 1
    assert len(result.times_h) == expected_len


def test_scalar_fields_are_floats():
    result = default_result()
    assert isinstance(result.times_above_mic_h, float)
    assert isinstance(result.fT_above_mic_pct, float)
    assert isinstance(result.auc_mic_ratio, float)
    assert isinstance(result.cmax_mic_ratio, float)


def test_string_fields():
    result = default_result(drug_name="meropenem", pathogen="Klebsiella")
    assert result.drug_name == "meropenem"
    assert result.pathogen == "Klebsiella"
    assert result.mic_mg_L == 0.25


# ---------------------------------------------------------------------------
# PK correctness
# ---------------------------------------------------------------------------


def test_concentration_starts_at_zero():
    result = default_result()
    assert result.c_drug_mg_L[0] == 0.0


def test_concentration_increases_after_dose():
    result = default_result()
    assert max(result.c_drug_mg_L) > 0.0


def test_cmax_mic_ratio_positive():
    result = default_result()
    assert result.cmax_mic_ratio > 0.0


def test_auc_mic_ratio_positive():
    result = default_result()
    assert result.auc_mic_ratio > 0.0


def test_log10_cfu_starts_at_6():
    result = default_result()
    assert abs(result.log10_cfu[0] - 6.0) < 1e-9


def test_log10_cfu_bounded():
    result = default_result()
    for v in result.log10_cfu:
        assert 0.0 <= v <= 9.0


# ---------------------------------------------------------------------------
# PK/PD index tests
# ---------------------------------------------------------------------------


def test_pkpd_index_stored():
    result = default_result(pkpd_index="Cmax/MIC")
    assert result.pkpd_index == "Cmax/MIC"


def test_fT_above_mic_pct_range():
    result = default_result(pkpd_index="fT>MIC")
    assert 0.0 <= result.fT_above_mic_pct <= 100.0


def test_times_above_mic_nonnegative():
    result = default_result()
    assert result.times_above_mic_h >= 0.0


def test_bactericidal_outcome_high_dose():
    # Very high dose relative to MIC should achieve bactericidal AUC/MIC > 125
    result = simulate_antimicrobial_pkpd(
        drug_name="levofloxacin",
        pathogen="S_pneumoniae",
        mic_mg_L=0.03,
        dose_mg=750.0,
        vd_L=90.0,
        cl_L_per_h=10.0,
        pkpd_index="AUC/MIC",
        tau_h=24.0,
        n_doses=3,
    )
    assert result.predicted_outcome == "bactericidal"


def test_treatment_failure_low_dose():
    # Very low dose relative to MIC should fail
    result = simulate_antimicrobial_pkpd(
        drug_name="test_drug",
        pathogen="test_pathogen",
        mic_mg_L=10.0,
        dose_mg=1.0,
        vd_L=100.0,
        cl_L_per_h=20.0,
        pkpd_index="AUC/MIC",
        tau_h=24.0,
        n_doses=1,
    )
    assert result.predicted_outcome == "treatment_failure"


def test_cmax_mic_index_bactericidal():
    # Cmax/MIC > 10 → bactericidal
    result = simulate_antimicrobial_pkpd(
        drug_name="gentamicin",
        pathogen="Pseudomonas",
        mic_mg_L=1.0,
        dose_mg=500.0,
        vd_L=20.0,
        cl_L_per_h=5.0,
        pkpd_index="Cmax/MIC",
        tau_h=24.0,
        n_doses=1,
    )
    # Cmax ≈ 500*0.85/20 peak → should be >> 10
    assert result.cmax_mic_ratio > 10.0
    assert result.predicted_outcome == "bactericidal"


# ---------------------------------------------------------------------------
# Resistance risk tests
# ---------------------------------------------------------------------------


def test_resistance_risk_values():
    result = default_result()
    assert result.resistance_risk in ("low", "moderate", "high")


def test_high_resistance_risk_for_failed_treatment():
    # With very poor PK/PD, bacteria should not be eradicated
    result = simulate_antimicrobial_pkpd(
        drug_name="test",
        pathogen="test",
        mic_mg_L=50.0,
        dose_mg=1.0,
        vd_L=100.0,
        cl_L_per_h=20.0,
        pkpd_index="AUC/MIC",
        tau_h=24.0,
        n_doses=1,
    )
    # Bacteria grow freely → high resistance risk
    assert result.resistance_risk == "high"


# ---------------------------------------------------------------------------
# Target attainment tests
# ---------------------------------------------------------------------------


def test_optimal_target_met_boolean():
    result = default_result()
    assert isinstance(result.optimal_pkpd_target_met, bool)


def test_optimal_target_met_for_bactericidal():
    result = simulate_antimicrobial_pkpd(
        drug_name="cipro",
        pathogen="E_coli",
        mic_mg_L=0.01,
        dose_mg=500.0,
        vd_L=80.0,
        cl_L_per_h=10.0,
        pkpd_index="AUC/MIC",
        tau_h=12.0,
        n_doses=5,
    )
    # Should achieve bactericidal outcome
    assert result.predicted_outcome == "bactericidal"
    # If final CFU < 3, target should be met
    if result.log10_cfu[-1] < 3.0:
        assert result.optimal_pkpd_target_met is True


# ---------------------------------------------------------------------------
# Multiple dose tests
# ---------------------------------------------------------------------------


def test_multiple_doses_increase_auc():
    r1 = simulate_antimicrobial_pkpd(
        drug_name="drug",
        pathogen="bug",
        mic_mg_L=1.0,
        dose_mg=200.0,
        vd_L=50.0,
        cl_L_per_h=10.0,
        n_doses=1,
        tau_h=24.0,
    )
    r3 = simulate_antimicrobial_pkpd(
        drug_name="drug",
        pathogen="bug",
        mic_mg_L=1.0,
        dose_mg=200.0,
        vd_L=50.0,
        cl_L_per_h=10.0,
        n_doses=3,
        tau_h=24.0,
    )
    assert r3.auc_mic_ratio > r1.auc_mic_ratio


def test_notes_is_string():
    result = default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validation_mic_zero():
    with pytest.raises(ValueError, match="mic_mg_L"):
        simulate_antimicrobial_pkpd(
            drug_name="d",
            pathogen="p",
            mic_mg_L=0.0,
            dose_mg=100.0,
            vd_L=10.0,
            cl_L_per_h=5.0,
        )


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_antimicrobial_pkpd(
            drug_name="d",
            pathogen="p",
            mic_mg_L=1.0,
            dose_mg=0.0,
            vd_L=10.0,
            cl_L_per_h=5.0,
        )


def test_validation_n_doses_zero():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_antimicrobial_pkpd(
            drug_name="d",
            pathogen="p",
            mic_mg_L=1.0,
            dose_mg=100.0,
            vd_L=10.0,
            cl_L_per_h=5.0,
            n_doses=0,
        )


def test_validation_invalid_pkpd_index():
    with pytest.raises(ValueError, match="pkpd_index"):
        simulate_antimicrobial_pkpd(
            drug_name="d",
            pathogen="p",
            mic_mg_L=1.0,
            dose_mg=100.0,
            vd_L=10.0,
            cl_L_per_h=5.0,
            pkpd_index="invalid",
        )
