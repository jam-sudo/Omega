"""Tests for Phase 1066 — Drug Formulation Compatibility Matrix."""

import pytest

from omega_pbpk.prediction.formulation_compatibility_matrix import (
    CompatibilityResult,
    compatibility_matrix,
    predict_compatibility,
)


def test_return_type():
    result = predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient="lactose")
    assert isinstance(result, CompatibilityResult)


def test_frozen_dataclass():
    result = predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient="lactose")
    with pytest.raises((AttributeError, TypeError)):
        result.excipient = "MCC"  # type: ignore[misc]


def test_compatibility_score_range():
    for exc in [
        "lactose",
        "MCC",
        "starch",
        "HPMC",
        "PVP",
        "magnesium_stearate",
        "citric_acid",
        "sodium_bicarbonate",
        "talc",
        "croscarmellose",
    ]:
        result = predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient=exc)
        assert 0 <= result.compatibility_score <= 100, f"{exc}: {result.compatibility_score}"


def test_degradation_risk_valid():
    valid = {"low", "moderate", "high"}
    result = predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient="lactose")
    assert result.degradation_risk in valid


def test_interaction_type_valid():
    valid = {"none", "physical", "chemical", "both"}
    result = predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient="lactose")
    assert result.interaction_type in valid


def test_recommended_ratio_valid():
    valid = {"1:10", "1:5", "1:1", "avoid"}
    result = predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient="lactose")
    assert result.recommended_ratio in valid


def test_stability_impact_range():
    result = predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient="lactose")
    assert 0 <= result.stability_impact_pct <= 50


def test_amine_lactose_low_score_maillard():
    """Drug with amine + lactose (reducing sugar) → Maillard → low score."""
    result = predict_compatibility(
        "AmineDrug", mw_Da=300.0, logp=2.0, has_amine=True, excipient="lactose"
    )
    assert result.compatibility_score <= 60
    assert result.interaction_type == "chemical"
    assert "Maillard" in result.mechanism


def test_ester_citric_acid_lower_than_mcc():
    """Drug with ester + citric_acid → acid hydrolysis → lower score than with MCC."""
    result_citric = predict_compatibility(
        "EsterDrug", mw_Da=300.0, logp=2.0, has_ester=True, excipient="citric_acid"
    )
    result_mcc = predict_compatibility(
        "EsterDrug", mw_Da=300.0, logp=2.0, has_ester=True, excipient="MCC"
    )
    assert result_citric.compatibility_score < result_mcc.compatibility_score


def test_mcc_no_interactions_clean_drug():
    """MCC with a drug having no reactive groups → no interactions, high score."""
    result = predict_compatibility(
        "CleanDrug",
        mw_Da=500.0,
        logp=2.0,
        has_amine=False,
        has_ester=False,
        has_aldehyde=False,
        has_acid=False,
        has_hydroxyl=False,
        excipient="MCC",
    )
    assert result.mechanism == "none"
    assert result.compatibility_score == pytest.approx(100.0)
    assert result.interaction_type == "none"


def test_maillard_chemical_interaction_type():
    result = predict_compatibility(
        "AmineDrug", mw_Da=300.0, logp=2.0, has_amine=True, excipient="lactose"
    )
    assert result.interaction_type == "chemical"


def test_adsorption_physical_small_molecule():
    """Small molecule with HPMC → adsorption → physical interaction."""
    result = predict_compatibility("SmallDrug", mw_Da=200.0, logp=2.0, excipient="HPMC")
    assert result.interaction_type == "physical"
    assert result.compatibility_score == pytest.approx(95.0)


def test_adsorption_no_effect_large_molecule():
    """Large molecule (MW >= 400) with HPMC → no adsorption."""
    result = predict_compatibility("LargeDrug", mw_Da=500.0, logp=2.0, excipient="HPMC")
    assert result.compatibility_score == pytest.approx(100.0)


def test_both_interaction_type():
    """Amine drug (small) with lactose → Maillard (chemical) + check for both."""
    # Small amine drug with HPMC → adsorption (physical)
    # But we need chemical + physical simultaneously
    # Use amine + lactose (Maillard=chemical) won't give physical since lactose isn't in adsorbing set
    # Use amine + HPMC: adsorption=physical, no chemical unless amine+reducing_sugar
    # Let's use amine + aldehyde + HPMC small molecule → chemical(aldehyde) + physical(adsorption)
    result = predict_compatibility(
        "ReactiveDrug", mw_Da=200.0, logp=2.0, has_aldehyde=True, excipient="HPMC"
    )
    # aldehyde on non-reducing sugar: -10, adsorption: -5 → score=85
    assert result.interaction_type == "both"


def test_compatibility_matrix_returns_10_items():
    results = compatibility_matrix("DrugA", mw_Da=300.0, logp=2.0)
    assert len(results) == 10


def test_compatibility_matrix_sorted_desc():
    results = compatibility_matrix("DrugA", mw_Da=300.0, logp=2.0)
    scores = [r.compatibility_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_compatibility_matrix_all_results():
    results = compatibility_matrix("DrugA", mw_Da=300.0, logp=2.0)
    excipient_names = {r.excipient for r in results}
    expected = {
        "lactose",
        "MCC",
        "starch",
        "HPMC",
        "PVP",
        "magnesium_stearate",
        "citric_acid",
        "sodium_bicarbonate",
        "talc",
        "croscarmellose",
    }
    assert excipient_names == expected


def test_notes_nonempty():
    result = predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient="MCC")
    assert result.notes
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = predict_compatibility("TestDrug", mw_Da=300.0, logp=2.0, excipient="talc")
    assert result.drug_name == "TestDrug"


def test_stability_impact_formula():
    result = predict_compatibility(
        "DrugA", mw_Da=300.0, logp=2.0, has_amine=True, excipient="lactose"
    )
    expected = (100.0 - result.compatibility_score) * 0.5
    assert result.stability_impact_pct == pytest.approx(expected)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_compatibility("DrugA", mw_Da=0.0, logp=2.0, excipient="lactose")


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_compatibility("DrugA", mw_Da=-100.0, logp=2.0, excipient="lactose")


def test_validation_invalid_excipient():
    with pytest.raises(ValueError, match="excipient"):
        predict_compatibility("DrugA", mw_Da=300.0, logp=2.0, excipient="sugar")


def test_base_saponification_magnesium_stearate_ester():
    result = predict_compatibility(
        "EsterDrug", mw_Da=400.0, logp=2.0, has_ester=True, excipient="magnesium_stearate"
    )
    assert "saponification" in result.mechanism
    assert result.compatibility_score < 100
