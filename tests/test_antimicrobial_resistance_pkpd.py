"""Tests for Phase 1041 — Antimicrobial Resistance PK/PD."""

import pytest
from omega_pbpk.clinical.antimicrobial_resistance_pkpd import (
    AntimicrobialPKPDResult,
    simulate_antimicrobial_pkpd,
)

VALID_OUTCOMES = {"eradication", "suppression", "failure", "amplification"}


@pytest.fixture
def default_result():
    return simulate_antimicrobial_pkpd(
        drug_name="TestAntibiotic",
        dose_mg=500.0,
        dosing_interval_h=12.0,
        n_doses=7,
        mic_sensitive_mg_L=0.5,
        mic_resistant_mg_L=4.0,
        pkpd_type="AUC_MIC",
        cl_L_per_h=5.0,
        vd_L=30.0,
        ka_per_h=1.0,
        n0_sensitive=6.0,
        n0_resistant=4.0,
        mu_max_per_h=0.8,
        n_max=9.0,
    )


def test_return_type(default_result):
    assert isinstance(default_result, AntimicrobialPKPDResult)


def test_c_plasma_nonempty(default_result):
    assert len(default_result.c_plasma_mg_L) > 0


def test_c_plasma_positive_values(default_result):
    # Should have some positive concentrations after dosing
    assert any(c > 0 for c in default_result.c_plasma_mg_L)


def test_all_concentrations_nonnegative(default_result):
    assert all(c >= 0 for c in default_result.c_plasma_mg_L)


def test_n_sensitive_nonempty(default_result):
    assert len(default_result.n_sensitive) > 0


def test_n_resistant_nonempty(default_result):
    assert len(default_result.n_resistant) > 0


def test_n_total_nonempty(default_result):
    assert len(default_result.n_total) > 0


def test_auc_mic_ratio_positive(default_result):
    assert default_result.auc_mic_ratio > 0


def test_cmax_mic_ratio_positive(default_result):
    assert default_result.cmax_mic_ratio > 0


def test_pct_time_above_mic_range(default_result):
    assert 0 <= default_result.pct_time_above_mic <= 100


def test_eradication_probability_range(default_result):
    assert 0 <= default_result.eradication_probability <= 1


def test_outcome_valid(default_result):
    assert default_result.outcome in VALID_OUTCOMES


def test_resistance_amplification_is_bool(default_result):
    assert isinstance(default_result.resistance_amplification, bool)


def test_notes_nonempty(default_result):
    assert len(default_result.notes) > 0


def test_high_dose_good_outcome():
    """Very high dose relative to MIC should lead to eradication or suppression."""
    result = simulate_antimicrobial_pkpd(
        drug_name="HighDoseAntibiotic",
        dose_mg=5000.0,
        dosing_interval_h=24.0,
        n_doses=14,
        mic_sensitive_mg_L=0.1,
        mic_resistant_mg_L=1.0,
        pkpd_type="AUC_MIC",
        cl_L_per_h=5.0,
        vd_L=30.0,
        ka_per_h=2.0,
        n0_sensitive=6.0,
        n0_resistant=3.0,
        mu_max_per_h=0.8,
        n_max=9.0,
    )
    assert result.outcome in {"eradication", "suppression"}


def test_cmax_mic_pkpd_type():
    result = simulate_antimicrobial_pkpd(
        drug_name="CmaxDrug",
        dose_mg=500.0,
        dosing_interval_h=24.0,
        n_doses=7,
        mic_sensitive_mg_L=0.5,
        mic_resistant_mg_L=4.0,
        pkpd_type="Cmax_MIC",
    )
    assert isinstance(result, AntimicrobialPKPDResult)
    assert result.eradication_probability in {0.1, 0.4, 0.7, 0.9}


def test_ft_mic_pkpd_type():
    result = simulate_antimicrobial_pkpd(
        drug_name="BetaLactam",
        dose_mg=500.0,
        dosing_interval_h=6.0,
        n_doses=14,
        mic_sensitive_mg_L=0.5,
        mic_resistant_mg_L=4.0,
        pkpd_type="fT_MIC",
    )
    assert isinstance(result, AntimicrobialPKPDResult)
    assert result.eradication_probability in {0.1, 0.4, 0.7, 0.9}


def test_times_h_nonempty(default_result):
    assert len(default_result.times_h) > 0


def test_list_lengths_consistent(default_result):
    n = len(default_result.times_h)
    assert len(default_result.c_plasma_mg_L) == n
    assert len(default_result.n_sensitive) == n
    assert len(default_result.n_resistant) == n
    assert len(default_result.n_total) == n


# Validation tests
def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_antimicrobial_pkpd(drug_name="X", dose_mg=0.0)


def test_validation_mic_sensitive_ge_resistant():
    with pytest.raises(ValueError, match="mic_resistant"):
        simulate_antimicrobial_pkpd(
            drug_name="X",
            dose_mg=100.0,
            mic_sensitive_mg_L=4.0,
            mic_resistant_mg_L=0.5,
        )


def test_validation_invalid_pkpd_type():
    with pytest.raises(ValueError, match="pkpd_type"):
        simulate_antimicrobial_pkpd(
            drug_name="X",
            dose_mg=100.0,
            pkpd_type="INVALID",
        )


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_antimicrobial_pkpd(drug_name="X", dose_mg=100.0, cl_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_antimicrobial_pkpd(drug_name="X", dose_mg=100.0, vd_L=0.0)
