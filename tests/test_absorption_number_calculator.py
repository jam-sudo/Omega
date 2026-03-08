"""Tests for Phase 576 - Oral Drug Absorption Number Calculator."""

import pytest

from omega_pbpk.prediction.absorption_number_calculator import (
    AbsorptionNumberResult,
    calculate_absorption_number,
    screen_compounds,
)


def test_return_type():
    result = calculate_absorption_number("DrugA", 100.0, 1.0, 1e-4)
    assert isinstance(result, AbsorptionNumberResult)


def test_dose_number_formula():
    result = calculate_absorption_number("DrugB", 500.0, 0.5, 1e-4)
    expected_do = 500.0 / (250.0 * 0.5)
    assert abs(result.dose_number - expected_do) < 1e-9


def test_absorption_number_formula():
    peff = 2e-4
    t_transit_min = 199.0
    r = 1.75
    result = calculate_absorption_number("DrugC", 100.0, 1.0, peff)
    expected_an = peff * 2.0 * (t_transit_min * 60.0) / r
    assert abs(result.absorption_number - expected_an) < 1e-9


def test_bcs_class_I():
    # High perm (An > 1), high sol (Do <= 1)
    result = calculate_absorption_number("DrugD", 10.0, 1.0, 1e-3)
    assert result.bcs_class == "I"
    assert result.absorption_limited_by == "neither"


def test_bcs_class_II():
    # High perm (An > 1), low sol (Do > 1)
    result = calculate_absorption_number("DrugE", 1000.0, 0.001, 1e-3)
    assert result.bcs_class == "II"
    assert result.absorption_limited_by == "solubility"


def test_bcs_class_III():
    # Low perm (An <= 1), high sol (Do <= 1)
    result = calculate_absorption_number("DrugF", 10.0, 1.0, 1e-5)
    assert result.bcs_class == "III"
    assert result.absorption_limited_by == "permeability"


def test_bcs_class_IV():
    # Low perm (An <= 1), low sol (Do > 1)
    result = calculate_absorption_number("DrugG", 5000.0, 0.001, 1e-6)
    assert result.bcs_class == "IV"
    assert result.absorption_limited_by == "both"


def test_high_an_low_do_fa_corrected_equals_fa():
    # BCS I: no solubility limitation
    result = calculate_absorption_number("DrugH", 10.0, 1.0, 1e-3)
    assert abs(result.fa_corrected - result.fraction_absorbed) < 1e-9


def test_high_do_fa_corrected_limited():
    # BCS II: solubility limits
    result = calculate_absorption_number("DrugI", 1000.0, 0.001, 1e-3)
    expected_limit = 1.0 / result.dose_number
    assert result.fa_corrected <= result.fraction_absorbed + 1e-9
    assert result.fa_corrected <= expected_limit + 1e-9


def test_fa_corrected_le_fa_perm_limited():
    result = calculate_absorption_number("DrugJ", 500.0, 0.01, 1e-4)
    assert result.fa_corrected <= result.fraction_absorbed + 1e-9


def test_fa_corrected_range():
    result = calculate_absorption_number("DrugK", 100.0, 0.5, 1e-4)
    assert 0.0 <= result.fa_corrected <= 1.0


def test_fraction_absorbed_range():
    result = calculate_absorption_number("DrugL", 100.0, 1.0, 1e-4)
    assert 0.0 <= result.fraction_absorbed <= 1.0


def test_drug_name_preserved():
    result = calculate_absorption_number("MyCompound", 100.0, 1.0, 1e-4)
    assert result.drug_name == "MyCompound"


def test_dose_preserved():
    result = calculate_absorption_number("DrugM", 250.0, 1.0, 1e-4)
    assert result.dose_mg == 250.0


def test_solubility_preserved():
    result = calculate_absorption_number("DrugN", 100.0, 0.5, 1e-4)
    assert result.solubility_mg_mL == 0.5


def test_peff_preserved():
    result = calculate_absorption_number("DrugO", 100.0, 1.0, 3e-4)
    assert result.peff_cm_per_s == 3e-4


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        calculate_absorption_number("DrugP", -10.0, 1.0, 1e-4)


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        calculate_absorption_number("DrugQ", 0.0, 1.0, 1e-4)


def test_validation_solubility_negative():
    with pytest.raises(ValueError, match="solubility"):
        calculate_absorption_number("DrugR", 100.0, -0.1, 1e-4)


def test_validation_peff_zero():
    with pytest.raises(ValueError, match="peff"):
        calculate_absorption_number("DrugS", 100.0, 1.0, 0.0)


def test_screen_compounds_sorted():
    compounds = [
        {"drug_name": "Low", "dose_mg": 5000.0, "solubility_mg_mL": 0.001, "peff_cm_per_s": 1e-6},
        {"drug_name": "High", "dose_mg": 10.0, "solubility_mg_mL": 1.0, "peff_cm_per_s": 1e-3},
        {"drug_name": "Mid", "dose_mg": 100.0, "solubility_mg_mL": 0.1, "peff_cm_per_s": 1e-4},
    ]
    results = screen_compounds(compounds)
    assert len(results) == 3
    for i in range(len(results) - 1):
        assert results[i].fa_corrected >= results[i + 1].fa_corrected


def test_screen_compounds_returns_list():
    compounds = [
        {"drug_name": "A", "dose_mg": 100.0, "solubility_mg_mL": 1.0, "peff_cm_per_s": 1e-4},
    ]
    results = screen_compounds(compounds)
    assert isinstance(results, list)
    assert all(isinstance(r, AbsorptionNumberResult) for r in results)
