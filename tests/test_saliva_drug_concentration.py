"""Tests for Phase 499 — Saliva Drug Concentration Model."""

import math

import pytest

from omega_pbpk.prediction.saliva_drug_concentration import (
    SalivaDrugResult,
    compare_saliva_monitoring,
    predict_saliva_concentration,
)

_SALIVA_PH = 6.8
_PLASMA_PH = 7.4


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_saliva_concentration("TestDrug", 1.0, 0.5, 7.0, "neutral")
    assert isinstance(result, SalivaDrugResult)


def test_required_fields():
    result = predict_saliva_concentration("DrugA", 2.0, 0.4, 8.0, "base")
    assert hasattr(result, "drug_name")
    assert hasattr(result, "plasma_conc_mg_L")
    assert hasattr(result, "fu_plasma")
    assert hasattr(result, "pKa")
    assert hasattr(result, "drug_type")
    assert hasattr(result, "saliva_plasma_ratio")
    assert hasattr(result, "saliva_conc_mg_L")
    assert hasattr(result, "monitoring_utility")
    assert hasattr(result, "notes")


def test_field_values_stored():
    result = predict_saliva_concentration("Aspirin", 5.0, 0.1, 3.5, "acid")
    assert result.drug_name == "Aspirin"
    assert result.plasma_conc_mg_L == 5.0
    assert result.fu_plasma == 0.1
    assert result.pKa == 3.5
    assert result.drug_type == "acid"


# ---------------------------------------------------------------------------
# Neutral drug
# ---------------------------------------------------------------------------


def test_neutral_ratio_equals_fu():
    fu = 0.3
    result = predict_saliva_concentration("NeutralX", 10.0, fu, 10.0, "neutral")
    assert math.isclose(result.saliva_plasma_ratio, fu, rel_tol=1e-9)


def test_neutral_saliva_conc():
    fu = 0.5
    plasma = 4.0
    result = predict_saliva_concentration("NeutralY", plasma, fu, 9.0, "neutral")
    assert math.isclose(result.saliva_conc_mg_L, plasma * fu, rel_tol=1e-9)


def test_neutral_fu_one():
    result = predict_saliva_concentration("NeutralZ", 2.0, 1.0, 5.0, "neutral")
    assert math.isclose(result.saliva_plasma_ratio, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Acid drug
# ---------------------------------------------------------------------------


def test_acid_low_pKa_lower_saliva():
    """Acid with pKa=5: more ionized in saliva (pH 6.8) → ratio < 1."""
    result = predict_saliva_concentration("WeakAcid", 1.0, 1.0, 5.0, "acid")
    assert result.saliva_plasma_ratio < 1.0


def test_acid_high_pKa_approaches_neutral():
    """Acid with very high pKa: essentially uncharged → ratio ≈ fu."""
    fu = 0.6
    result = predict_saliva_concentration("HighpKaAcid", 1.0, fu, 14.0, "acid")
    assert math.isclose(result.saliva_plasma_ratio, fu, rel_tol=0.05)


def test_acid_ratio_positive():
    result = predict_saliva_concentration("AcidDrug", 3.0, 0.2, 4.0, "acid")
    assert result.saliva_plasma_ratio > 0.0


def test_acid_saliva_conc_consistent():
    result = predict_saliva_concentration("Ibuprofen", 2.0, 0.01, 4.9, "acid")
    assert math.isclose(
        result.saliva_conc_mg_L,
        result.plasma_conc_mg_L * result.saliva_plasma_ratio,
        rel_tol=1e-9,
    )


# ---------------------------------------------------------------------------
# Base drug
# ---------------------------------------------------------------------------


def test_base_high_pKa_ion_trapping():
    """Base with pKa=9: ion trapping in saliva (lower pH) → ratio > 1."""
    result = predict_saliva_concentration("StrongBase", 1.0, 1.0, 9.0, "base")
    assert result.saliva_plasma_ratio > 1.0


def test_base_saliva_conc_higher_than_plasma():
    result = predict_saliva_concentration("BaseX", 1.0, 1.0, 9.0, "base")
    assert result.saliva_conc_mg_L > result.plasma_conc_mg_L


def test_base_ratio_clamped_max():
    """Very strong base with fu=1 and pKa=14 → ratio may be clamped."""
    result = predict_saliva_concentration("StrongBase2", 1.0, 1.0, 14.0, "base")
    assert result.saliva_plasma_ratio <= 10.0


def test_base_low_pKa_approaches_neutral():
    """Base with very low pKa: essentially uncharged → ratio ≈ fu."""
    fu = 0.7
    result = predict_saliva_concentration("LowpKaBase", 1.0, fu, 1.0, "base")
    assert math.isclose(result.saliva_plasma_ratio, fu, rel_tol=0.05)


# ---------------------------------------------------------------------------
# Ratio clamping
# ---------------------------------------------------------------------------


def test_ratio_clamp_minimum():
    result = predict_saliva_concentration("TightBound", 1.0, 0.001, 3.0, "acid")
    assert result.saliva_plasma_ratio >= 0.01


def test_ratio_clamp_maximum():
    result = predict_saliva_concentration("IonTrapped", 1.0, 1.0, 14.0, "base")
    assert result.saliva_plasma_ratio <= 10.0


# ---------------------------------------------------------------------------
# monitoring_utility classification
# ---------------------------------------------------------------------------


def test_monitoring_utility_excellent():
    """Neutral drug with fu=1.0 → ratio=1.0 → excellent."""
    result = predict_saliva_concentration("PerfectDrug", 1.0, 1.0, 7.0, "neutral")
    assert result.monitoring_utility == "excellent"


def test_monitoring_utility_good():
    """Neutral drug with fu=0.6 → ratio=0.6 → good (0.5-1.5)."""
    result = predict_saliva_concentration("GoodDrug", 1.0, 0.6, 7.0, "neutral")
    assert result.monitoring_utility == "good"


def test_monitoring_utility_poor():
    """Neutral drug with fu=0.1 → ratio=0.1 → poor."""
    result = predict_saliva_concentration("PoorDrug", 1.0, 0.1, 7.0, "neutral")
    assert result.monitoring_utility == "poor"


def test_monitoring_utility_poor_high_ratio():
    """Base with strong ion trapping → ratio > 1.5 → poor."""
    result = predict_saliva_concentration("HighBase", 1.0, 1.0, 9.0, "base")
    if result.saliva_plasma_ratio > 1.5:
        assert result.monitoring_utility == "poor"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validate_plasma_conc_zero():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        predict_saliva_concentration("Drug", 0.0, 0.5, 7.0, "neutral")


def test_validate_plasma_conc_negative():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        predict_saliva_concentration("Drug", -1.0, 0.5, 7.0, "neutral")


def test_validate_fu_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_saliva_concentration("Drug", 1.0, 0.0, 7.0, "neutral")


def test_validate_fu_exceeds_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_saliva_concentration("Drug", 1.0, 1.5, 7.0, "neutral")


def test_validate_drug_type_invalid():
    with pytest.raises(ValueError, match="drug_type"):
        predict_saliva_concentration("Drug", 1.0, 0.5, 7.0, "amphoteric")


# ---------------------------------------------------------------------------
# compare_saliva_monitoring
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    compounds = [
        {
            "drug_name": "A",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.9,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
        {
            "drug_name": "B",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.1,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
    ]
    result = compare_saliva_monitoring(compounds)
    assert isinstance(result, list)
    assert len(result) == 2


def test_compare_sorted_by_ratio():
    compounds = [
        {
            "drug_name": "High",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.9,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
        {
            "drug_name": "Low",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.1,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
        {
            "drug_name": "Mid",
            "plasma_conc_mg_L": 1.0,
            "fu_plasma": 0.5,
            "pKa": 7.0,
            "drug_type": "neutral",
        },
    ]
    results = compare_saliva_monitoring(compounds)
    ratios = [r.saliva_plasma_ratio for r in results]
    assert ratios == sorted(ratios)


def test_compare_elements_are_saliva_results():
    compounds = [
        {
            "drug_name": "X",
            "plasma_conc_mg_L": 2.0,
            "fu_plasma": 0.4,
            "pKa": 6.0,
            "drug_type": "acid",
        },
    ]
    results = compare_saliva_monitoring(compounds)
    assert isinstance(results[0], SalivaDrugResult)


def test_compare_empty_list():
    result = compare_saliva_monitoring([])
    assert result == []
