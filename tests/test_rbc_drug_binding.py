"""Tests for Phase 483 — Drug Binding to Red Blood Cells."""

import pytest

from omega_pbpk.prediction.rbc_drug_binding import (
    RBCBindingResult,
    _compute_kp_rbc,
    predict_rbc_binding,
    screen_rbc_binding,
)

# ── Return type ──────────────────────────────────────────────────────────────


def test_returns_rbc_binding_result():
    result = predict_rbc_binding("DrugA", logP=2.0)
    assert isinstance(result, RBCBindingResult)


def test_result_is_frozen():
    result = predict_rbc_binding("DrugA", logP=2.0)
    with pytest.raises((AttributeError, TypeError)):
        result.kp_rbc = 99.0  # type: ignore[misc]


# ── Field presence ────────────────────────────────────────────────────────────


def test_all_fields_present():
    r = predict_rbc_binding("DrugX", logP=1.5)
    assert hasattr(r, "drug_name")
    assert hasattr(r, "kp_rbc")
    assert hasattr(r, "blood_plasma_ratio")
    assert hasattr(r, "fu_blood")
    assert hasattr(r, "cl_blood_correction_factor")
    assert hasattr(r, "rbc_binding_class")
    assert hasattr(r, "clinical_relevance")
    assert hasattr(r, "notes")


# ── B:P formula ──────────────────────────────────────────────────────────────


def test_bp_formula_neutral():
    logP = 2.0
    hct = 0.45
    r = predict_rbc_binding("Neutral", logP=logP, ionization_type="neutral", hematocrit=hct)
    kp_expected = max(0.1, min(10.0, 0.57 + 0.3 * logP))
    bp_expected = 1.0 + hct * (kp_expected - 1.0)
    assert abs(r.blood_plasma_ratio - bp_expected) < 1e-9


def test_kp_rbc_neutral_formula():
    logP = 1.0
    kp = _compute_kp_rbc(logP, pKa=7.0, ionization_type="neutral")
    assert abs(kp - (0.57 + 0.3 * logP)) < 1e-9


def test_bp_with_custom_hematocrit():
    r = predict_rbc_binding("DrugH", logP=2.0, hematocrit=0.40)
    kp = r.kp_rbc
    expected = 1.0 + 0.40 * (kp - 1.0)
    assert abs(r.blood_plasma_ratio - expected) < 1e-9


# ── fu_blood calculation ──────────────────────────────────────────────────────


def test_fu_blood_equals_fu_plasma_over_bp():
    fu_plasma = 0.2
    r = predict_rbc_binding("DrugFU", logP=2.0, fu_plasma=fu_plasma)
    assert abs(r.fu_blood - fu_plasma / r.blood_plasma_ratio) < 1e-9


def test_cl_correction_factor_is_inverse_bp():
    r = predict_rbc_binding("DrugCL", logP=2.0)
    assert abs(r.cl_blood_correction_factor - 1.0 / r.blood_plasma_ratio) < 1e-9


# ── Lipophilicity effects ─────────────────────────────────────────────────────


def test_high_logP_gives_higher_kp():
    r_low = predict_rbc_binding("LowLogP", logP=0.0)
    r_high = predict_rbc_binding("HighLogP", logP=4.0)
    assert r_high.kp_rbc >= r_low.kp_rbc


def test_very_high_logP_clamped_to_10():
    r = predict_rbc_binding("VeryLipophilic", logP=50.0)
    assert r.kp_rbc <= 10.0


def test_very_negative_logP_clamped_to_0_1():
    r = predict_rbc_binding("VeryHydrophilic", logP=-50.0)
    assert r.kp_rbc >= 0.1


# ── Ionization type effects ───────────────────────────────────────────────────


def test_base_higher_kp_than_neutral_same_logP():
    # For base formula: factor = 1 + 0.2*(7.4-pKa)
    # When pKa < 7.4 → factor > 1 → Kp_base > Kp_neutral
    logP = 2.0
    pKa = 5.0  # weakly basic drug, pKa below pH → factor = 1 + 0.2*2.4 = 1.48
    r_neutral = predict_rbc_binding("N", logP=logP, pKa=pKa, ionization_type="neutral")
    r_base = predict_rbc_binding("B", logP=logP, pKa=pKa, ionization_type="base")
    assert r_base.kp_rbc >= r_neutral.kp_rbc


def test_acid_donnan_effect_reduces_kp():
    logP = 3.0
    pKa = 3.0  # strongly acidic
    r_neutral = predict_rbc_binding("N", logP=logP, pKa=pKa, ionization_type="neutral")
    r_acid = predict_rbc_binding("A", logP=logP, pKa=pKa, ionization_type="acid")
    # Acid should have lower or equal Kp due to Donnan factor
    assert r_acid.kp_rbc <= r_neutral.kp_rbc + 1e-9


# ── RBC binding class ─────────────────────────────────────────────────────────


def test_rbc_class_low():
    # logP = -2.0 → kp = 0.57 - 0.6 = -0.03 → clamped to 0.1 → low
    r = predict_rbc_binding("Low", logP=-2.0)
    assert r.rbc_binding_class == "low"


def test_rbc_class_moderate():
    # logP = 2.0 → kp = 0.57 + 0.6 = 1.17 → moderate
    r = predict_rbc_binding("Moderate", logP=2.0)
    assert r.rbc_binding_class == "moderate"


def test_rbc_class_high():
    # logP = 10.0 → kp = 0.57 + 3.0 = 3.57 → high
    r = predict_rbc_binding("High", logP=10.0)
    assert r.rbc_binding_class == "high"


# ── Clinical relevance ────────────────────────────────────────────────────────


def test_clinical_relevance_none_when_bp_near_1():
    # logP = 0.0 → kp = 0.57, B:P = 1 + 0.45*(0.57-1) = 0.807
    # |0.807-1| = 0.193 < 0.3 → minor
    # We need B:P very close to 1 — neutral with logP that gives kp≈1
    # kp=1 when 0.57+0.3*logP=1 → logP=1.433
    r = predict_rbc_binding("Near1", logP=1.433)
    # B:P ≈ 1.0 → |B:P - 1| should be < 0.1
    assert r.clinical_relevance in ("none", "minor")


def test_clinical_relevance_significant():
    # High logP → high B:P
    r = predict_rbc_binding("Significant", logP=8.0)
    assert r.clinical_relevance == "significant"


# ── Screen function ───────────────────────────────────────────────────────────


def test_screen_returns_list():
    compounds = [
        {"drug_name": "A", "logP": 1.0},
        {"drug_name": "B", "logP": 3.0},
        {"drug_name": "C", "logP": 5.0},
    ]
    results = screen_rbc_binding(compounds)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_by_bp_descending():
    compounds = [
        {"drug_name": "A", "logP": 1.0},
        {"drug_name": "B", "logP": 3.0},
        {"drug_name": "C", "logP": 5.0},
    ]
    results = screen_rbc_binding(compounds)
    for i in range(len(results) - 1):
        assert results[i].blood_plasma_ratio >= results[i + 1].blood_plasma_ratio


def test_screen_single_compound():
    results = screen_rbc_binding([{"drug_name": "Solo", "logP": 2.0}])
    assert len(results) == 1


# ── Validation errors ─────────────────────────────────────────────────────────


def test_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_rbc_binding("X", logP=1.0, ionization_type="zwitterion")


def test_invalid_hematocrit_zero():
    with pytest.raises(ValueError, match="hematocrit"):
        predict_rbc_binding("X", logP=1.0, hematocrit=0.0)


def test_invalid_hematocrit_one():
    with pytest.raises(ValueError, match="hematocrit"):
        predict_rbc_binding("X", logP=1.0, hematocrit=1.0)


def test_invalid_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_rbc_binding("X", logP=1.0, fu_plasma=0.0)


def test_invalid_fu_plasma_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_rbc_binding("X", logP=1.0, fu_plasma=1.5)


# ── drug_name propagation ─────────────────────────────────────────────────────


def test_drug_name_in_result():
    r = predict_rbc_binding("Aspirin", logP=1.19)
    assert r.drug_name == "Aspirin"
