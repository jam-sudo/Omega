"""Tests for Phase 1098 — Drug Erythrocyte Partitioning."""

import pytest

from omega_pbpk.prediction.erythrocyte_partitioning import (
    ErythrocytePartitioningResult,
    predict_rbc_partitioning,
    screen_compounds,
)

# --- Return type ---

def test_return_type():
    result = predict_rbc_partitioning("Chloroquine", logP=4.0, ionization_type="base", pka=10.0)
    assert isinstance(result, ErythrocytePartitioningResult)


# --- Basic drug with high pKa → high Kp_rbc ---

def test_basic_drug_high_pka_high_kp():
    """A basic drug with pKa >> 7.2 should have higher Kp_rbc than a neutral with same logP."""
    neutral = predict_rbc_partitioning("Neutral", logP=2.0, ionization_type="neutral")
    basic = predict_rbc_partitioning("Basic", logP=2.0, ionization_type="base", pka=10.0)
    assert basic.kp_rbc > neutral.kp_rbc


def test_basic_drug_pka_below_7_2_similar_to_neutral():
    """Basic drug with pKa < 7.2 should have same Kp as neutral (no trapping)."""
    neutral = predict_rbc_partitioning("Neutral", logP=2.0, ionization_type="neutral")
    basic_low_pka = predict_rbc_partitioning("BasicLow", logP=2.0, ionization_type="base", pka=6.0)
    assert abs(basic_low_pka.kp_rbc - neutral.kp_rbc) < 1e-9


# --- Neutral drug: Kp increases with logP ---

def test_neutral_kp_increases_with_logp():
    low = predict_rbc_partitioning("LowLogP", logP=-1.0, ionization_type="neutral")
    high = predict_rbc_partitioning("HighLogP", logP=5.0, ionization_type="neutral")
    assert high.kp_rbc > low.kp_rbc


def test_neutral_kp_clamped_at_10():
    result = predict_rbc_partitioning("VeryLipophilic", logP=9.0, ionization_type="neutral")
    assert result.kp_rbc <= 10.0


def test_neutral_kp_clamped_at_0_1():
    result = predict_rbc_partitioning("Hydrophilic", logP=-5.0, ionization_type="neutral")
    assert result.kp_rbc >= 0.1


# --- Rb > 0 ---

def test_rb_positive():
    result = predict_rbc_partitioning("Drug", logP=1.0, ionization_type="neutral")
    assert result.rb_blood_plasma_ratio > 0


def test_rb_equals_one_when_kp_is_one():
    """When Kp_rbc = 1: Rb = (1 - Hct) + Hct * 1 = 1.0"""
    # Find logP that gives Kp_rbc = 1.0: 0.3 + 0.2*logP = 1.0 → logP = 3.5
    result = predict_rbc_partitioning("Drug", logP=3.5, ionization_type="neutral", hematocrit=0.45)
    assert abs(result.rb_blood_plasma_ratio - 1.0) < 1e-6


# --- fu_blood <= fu_plasma ---

def test_fu_blood_le_fu_plasma_when_kp_gt1():
    """When Kp_rbc > 1, Rb > 1, so fu_blood = fu_plasma/Rb < fu_plasma."""
    result = predict_rbc_partitioning("Drug", logP=5.0, ionization_type="base", pka=10.0,
                                      fu_plasma=0.5)
    assert result.fu_blood <= result.fu_plasma


def test_fu_blood_ge_fu_plasma_when_kp_lt1():
    """When Kp_rbc < 1, Rb < 1, so fu_blood > fu_plasma (plasma has more binding)."""
    result = predict_rbc_partitioning("Drug", logP=-4.0, ionization_type="acid", fu_plasma=0.3)
    # Kp_rbc = 0.1 (clamped), Rb = (1-0.45) + 0.45*0.1 = 0.595 < 1 → fu_blood > fu_plasma
    assert result.fu_blood >= result.fu_plasma


# --- Accumulation risk ---

def test_rbc_accumulation_risk_low():
    result = predict_rbc_partitioning("Drug", logP=-4.0, ionization_type="neutral")
    assert result.rbc_accumulation_risk == "low"


def test_rbc_accumulation_risk_moderate():
    # logP=3.5 → Kp_rbc=1.0, neutral → check moderate boundary
    # logP=5.0 → Kp_rbc=1.3, neutral
    result = predict_rbc_partitioning("Drug", logP=5.0, ionization_type="neutral")
    # Kp = min(10, 0.3 + 0.2*5) = min(10, 1.3) = 1.3 → moderate
    assert result.rbc_accumulation_risk == "moderate"


def test_rbc_accumulation_risk_high():
    result = predict_rbc_partitioning("Drug", logP=4.0, ionization_type="base", pka=10.0)
    assert result.rbc_accumulation_risk == "high"


def test_high_kp_rbc_gives_high_risk():
    """A drug with very high Kp_rbc (base + high pKa) should show high risk."""
    result = predict_rbc_partitioning("ChloroquineAnalog", logP=4.0, ionization_type="base",
                                      pka=10.0)
    assert result.kp_rbc > 5.0
    assert result.rbc_accumulation_risk == "high"


# --- Clinical implication ---

def test_clinical_implication_nonempty():
    result = predict_rbc_partitioning("Drug", logP=1.0, ionization_type="neutral")
    assert len(result.clinical_implication) > 0


def test_clinical_implication_high_rbc_contains_reservoir():
    result = predict_rbc_partitioning("Drug", logP=4.0, ionization_type="base", pka=10.0)
    # High accumulation → should mention RBC as reservoir or substantial accumulation
    assert "RBC" in result.clinical_implication or "reservoir" in result.clinical_implication or \
           "accumulation" in result.clinical_implication or "blood" in result.clinical_implication.lower()


# --- screen_compounds ---

def test_screen_compounds_returns_correct_count():
    compounds = [
        {"name": "DrugA", "logP": 2.0, "ionization_type": "neutral"},
        {"name": "DrugB", "logP": 4.0, "ionization_type": "base", "pka": 9.0},
        {"name": "DrugC", "logP": -1.0, "ionization_type": "acid"},
    ]
    results = screen_compounds(compounds)
    assert len(results) == 3


def test_screen_compounds_sorted_by_kp_descending():
    compounds = [
        {"name": "DrugA", "logP": -3.0, "ionization_type": "neutral"},
        {"name": "DrugB", "logP": 4.0, "ionization_type": "base", "pka": 10.0},
        {"name": "DrugC", "logP": 2.0, "ionization_type": "neutral"},
    ]
    results = screen_compounds(compounds)
    kp_values = [r.kp_rbc for r in results]
    assert kp_values == sorted(kp_values, reverse=True), f"Not sorted descending: {kp_values}"


def test_screen_compounds_first_is_highest_kp():
    compounds = [
        {"name": "Low", "logP": -4.0, "ionization_type": "acid"},
        {"name": "High", "logP": 4.0, "ionization_type": "base", "pka": 10.0},
    ]
    results = screen_compounds(compounds)
    assert results[0].drug_name == "High"


# --- Hematocrit effect ---

def test_hematocrit_effect_high_kp():
    """Higher Hct with high Kp_rbc should give higher Rb."""
    low_hct = predict_rbc_partitioning("Drug", logP=4.0, ionization_type="base", pka=10.0,
                                       hematocrit=0.30)
    high_hct = predict_rbc_partitioning("Drug", logP=4.0, ionization_type="base", pka=10.0,
                                        hematocrit=0.60)
    assert high_hct.rb_blood_plasma_ratio > low_hct.rb_blood_plasma_ratio


# --- CL correction factor ---

def test_cl_correction_factor_equals_rb():
    result = predict_rbc_partitioning("Drug", logP=2.0, ionization_type="neutral")
    assert abs(result.cl_blood_correction_factor - result.rb_blood_plasma_ratio) < 1e-12


# --- Amphoteric ---

def test_amphoteric_high_pka_uses_base_model():
    """Amphoteric drug with high pKa should have same Kp as base with same logP/pKa."""
    amphoteric = predict_rbc_partitioning("Amphoteric", logP=3.0, ionization_type="amphoteric",
                                          pka=9.0)
    base = predict_rbc_partitioning("Base", logP=3.0, ionization_type="base", pka=9.0)
    assert abs(amphoteric.kp_rbc - base.kp_rbc) < 1e-9


# --- Validation errors ---

def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_rbc_partitioning("Drug", logP=1.0, ionization_type="zwitterion")


def test_invalid_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_rbc_partitioning("Drug", logP=1.0, ionization_type="neutral", fu_plasma=0.0)


def test_invalid_fu_plasma_greater_than_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_rbc_partitioning("Drug", logP=1.0, ionization_type="neutral", fu_plasma=1.5)


def test_invalid_hematocrit_zero():
    with pytest.raises(ValueError, match="hematocrit"):
        predict_rbc_partitioning("Drug", logP=1.0, ionization_type="neutral", hematocrit=0.0)


def test_invalid_hematocrit_one():
    with pytest.raises(ValueError, match="hematocrit"):
        predict_rbc_partitioning("Drug", logP=1.0, ionization_type="neutral", hematocrit=1.0)


def test_invalid_logp_too_low():
    with pytest.raises(ValueError, match="logP"):
        predict_rbc_partitioning("Drug", logP=-6.0, ionization_type="neutral")


def test_invalid_logp_too_high():
    with pytest.raises(ValueError, match="logP"):
        predict_rbc_partitioning("Drug", logP=11.0, ionization_type="neutral")
