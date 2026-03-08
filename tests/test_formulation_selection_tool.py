"""Tests for Phase 225 — formulation_selection_tool."""

import pytest

from omega_pbpk.prediction.formulation_selection_tool import (
    FormulationSelectionResult,
    compare_formulation_strategies,
    select_formulation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bcs1_result():
    return select_formulation(
        drug_name="DrugA",
        bcs_class=1,
        dose_mg=100.0,
        solubility_mg_mL=5.0,
        stability_ph_range=4.0,
        logP=1.5,
        mw_Da=300.0,
    )


def _bcs2_high_logp_result():
    return select_formulation(
        drug_name="DrugB",
        bcs_class=2,
        dose_mg=100.0,
        solubility_mg_mL=0.05,
        stability_ph_range=4.0,
        logP=5.0,
        mw_Da=400.0,
    )


def _bcs2_moderate_logp_result():
    return select_formulation(
        drug_name="DrugC",
        bcs_class=2,
        dose_mg=100.0,
        solubility_mg_mL=0.5,
        stability_ph_range=4.0,
        logP=2.0,
        mw_Da=350.0,
    )


def _bcs3_result():
    return select_formulation(
        drug_name="DrugD",
        bcs_class=3,
        dose_mg=200.0,
        solubility_mg_mL=10.0,
        stability_ph_range=4.0,
        logP=0.5,
        mw_Da=280.0,
    )


def _bcs4_result():
    return select_formulation(
        drug_name="DrugE",
        bcs_class=4,
        dose_mg=50.0,
        solubility_mg_mL=0.01,
        stability_ph_range=3.0,
        logP=4.0,
        mw_Da=450.0,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _bcs1_result()
    assert isinstance(result, FormulationSelectionResult)


def test_result_is_frozen():
    result = _bcs1_result()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Field preservation
# ---------------------------------------------------------------------------


def test_field_preservation():
    result = select_formulation(
        drug_name="MyDrug",
        bcs_class=2,
        dose_mg=250.0,
        solubility_mg_mL=1.0,
        logP=3.5,
        mw_Da=380.0,
    )
    assert result.drug_name == "MyDrug"
    assert result.bcs_class == 2
    assert result.dose_mg == pytest.approx(250.0)
    assert result.solubility_mg_mL == pytest.approx(1.0)
    assert result.logP == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# 3. BCS I → tablet or capsule recommended
# ---------------------------------------------------------------------------


def test_bcs1_recommends_tablet_or_capsule():
    result = _bcs1_result()
    rec = result.primary_recommendation.lower()
    assert "tablet" in rec or "capsule" in rec


def test_bcs1_development_risk_low():
    result = _bcs1_result()
    assert result.development_risk == "low"


# ---------------------------------------------------------------------------
# 4. BCS II high logP → lipid-based or ASD
# ---------------------------------------------------------------------------


def test_bcs2_high_logp_recommends_lipid_or_asd():
    result = _bcs2_high_logp_result()
    rec = result.primary_recommendation.lower()
    assert "lipid" in rec or "asd" in rec or "amorphous" in rec or "smedds" in rec or "sedds" in rec


def test_bcs2_high_logp_top_formulations_contain_lipid_or_asd():
    result = _bcs2_high_logp_result()
    top = " ".join(result.top_formulations).lower()
    assert "lipid" in top or "asd" in top or "amorphous" in top or "smedds" in top or "sedds" in top


# ---------------------------------------------------------------------------
# 5. top_formulations has <= 3 entries
# ---------------------------------------------------------------------------


def test_top_formulations_at_most_3():
    for fn in [_bcs1_result, _bcs2_high_logp_result, _bcs3_result, _bcs4_result]:
        result = fn()
        assert len(result.top_formulations) <= 3


def test_top_formulations_is_tuple():
    result = _bcs1_result()
    assert isinstance(result.top_formulations, tuple)


# ---------------------------------------------------------------------------
# 6. formulation_scores each in [0, 100]
# ---------------------------------------------------------------------------


def test_formulation_scores_in_range():
    for fn in [
        _bcs1_result,
        _bcs2_high_logp_result,
        _bcs2_moderate_logp_result,
        _bcs3_result,
        _bcs4_result,
    ]:
        result = fn()
        for score in result.formulation_scores:
            assert 0.0 <= score <= 100.0, f"Score out of range: {score}"


def test_formulation_scores_is_tuple():
    result = _bcs1_result()
    assert isinstance(result.formulation_scores, tuple)


def test_formulation_scores_count_matches_top_formulations():
    result = _bcs2_high_logp_result()
    assert len(result.formulation_scores) == len(result.top_formulations)


# ---------------------------------------------------------------------------
# 7. development_risk valid values
# ---------------------------------------------------------------------------


def test_development_risk_valid():
    for fn in [_bcs1_result, _bcs2_high_logp_result, _bcs3_result, _bcs4_result]:
        result = fn()
        assert result.development_risk in ("low", "moderate", "high")


def test_bcs4_development_risk_high():
    result = _bcs4_result()
    assert result.development_risk == "high"


# ---------------------------------------------------------------------------
# 8. compare_formulation_strategies sorted descending
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    items = compare_formulation_strategies(
        drug_name="X",
        bcs_class=2,
        dose_mg=100.0,
        solubility_mg_mL=0.1,
        logP=4.0,
    )
    assert isinstance(items, list)
    assert len(items) > 0


def test_compare_sorted_descending():
    items = compare_formulation_strategies(
        drug_name="X",
        bcs_class=2,
        dose_mg=100.0,
        solubility_mg_mL=0.1,
        logP=4.0,
    )
    scores = [score for _, score in items]
    assert scores == sorted(scores, reverse=True)


def test_compare_scores_in_range():
    items = compare_formulation_strategies(
        drug_name="X",
        bcs_class=3,
        dose_mg=200.0,
        solubility_mg_mL=5.0,
        logP=1.0,
    )
    for _, score in items:
        assert 0.0 <= score <= 100.0


def test_compare_contains_all_formulation_types():
    items = compare_formulation_strategies(
        drug_name="X",
        bcs_class=1,
        dose_mg=50.0,
        solubility_mg_mL=3.0,
        logP=1.0,
    )
    # Should have multiple formulations evaluated
    assert len(items) >= 3


# ---------------------------------------------------------------------------
# 9. Narrow pH stability triggers enteric coating mention
# ---------------------------------------------------------------------------


def test_narrow_ph_stability_mentions_enteric():
    result = select_formulation(
        drug_name="AcidLabile",
        bcs_class=1,
        dose_mg=100.0,
        solubility_mg_mL=5.0,
        stability_ph_range=1.5,  # narrow
        logP=1.0,
        mw_Da=300.0,
    )
    assert "enteric" in result.notes.lower() or "enteric" in result.rationale.lower()


# ---------------------------------------------------------------------------
# 10. High dose (>500 mg) avoids SMEDDS as top recommendation
# ---------------------------------------------------------------------------


def test_high_dose_avoids_smedds_as_primary():
    result = select_formulation(
        drug_name="HighDose",
        bcs_class=2,
        dose_mg=700.0,
        solubility_mg_mL=0.1,
        logP=5.0,
        mw_Da=400.0,
    )
    # SMEDDS should not be top recommendation for high dose drug
    rec = result.primary_recommendation.lower()
    # Primary should be amorphous/ASD or tablet, not smedds/sedds
    assert "smedds" not in rec or "amorphous" in rec or "tablet" in rec or "asd" in rec


# ---------------------------------------------------------------------------
# 11. Validation errors
# ---------------------------------------------------------------------------


def test_invalid_bcs_class():
    with pytest.raises(ValueError, match="bcs_class"):
        select_formulation("X", bcs_class=5, dose_mg=100.0, solubility_mg_mL=1.0)


def test_invalid_bcs_class_zero():
    with pytest.raises(ValueError, match="bcs_class"):
        select_formulation("X", bcs_class=0, dose_mg=100.0, solubility_mg_mL=1.0)


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        select_formulation("X", bcs_class=1, dose_mg=0.0, solubility_mg_mL=1.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        select_formulation("X", bcs_class=1, dose_mg=-10.0, solubility_mg_mL=1.0)


def test_solubility_zero_raises():
    with pytest.raises(ValueError, match="solubility"):
        select_formulation("X", bcs_class=1, dose_mg=100.0, solubility_mg_mL=0.0)


def test_solubility_negative_raises():
    with pytest.raises(ValueError, match="solubility"):
        select_formulation("X", bcs_class=2, dose_mg=100.0, solubility_mg_mL=-1.0)


def test_compare_invalid_bcs_raises():
    with pytest.raises(ValueError, match="bcs_class"):
        compare_formulation_strategies(
            "X", bcs_class=9, dose_mg=100.0, solubility_mg_mL=1.0, logP=2.0
        )


# ---------------------------------------------------------------------------
# 12. BCS II moderate logP → salt or particle size
# ---------------------------------------------------------------------------


def test_bcs2_moderate_logp_top3_contains_salt_or_particle():
    result = _bcs2_moderate_logp_result()
    top = " ".join(result.top_formulations).lower()
    assert "salt" in top or "particle" in top or "crystal" in top or "amorphous" in top


# ---------------------------------------------------------------------------
# 13. rationale is non-empty string
# ---------------------------------------------------------------------------


def test_rationale_non_empty():
    result = _bcs1_result()
    assert isinstance(result.rationale, str)
    assert len(result.rationale) > 10


def test_notes_non_empty():
    result = _bcs1_result()
    assert isinstance(result.notes, str)
