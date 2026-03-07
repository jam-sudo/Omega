"""Tests for Phase 257: PediatricFormulationSelector."""

import pytest

from omega_pbpk.prediction.pediatric_formulation_selector import (
    PediatricFormulationResult,
    compare_age_groups_for_drug,
    select_pediatric_formulation,
)

DRUG = "TestDrug"
DOSE_PER_KG = 5.0  # mg/kg
SOLUBILITY = 10.0  # mg/mL


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type():
    result = select_pediatric_formulation(DRUG, 6.0, 7.0, DOSE_PER_KG, SOLUBILITY)
    assert isinstance(result, PediatricFormulationResult)


# ---------------------------------------------------------------------------
# dose_mg calculation
# ---------------------------------------------------------------------------


def test_dose_mg_exact():
    weight = 8.5
    dpk = 3.0
    result = select_pediatric_formulation(DRUG, 6.0, weight, dpk, SOLUBILITY)
    assert abs(result.dose_mg - dpk * weight) < 1e-9


def test_dose_mg_toddler():
    weight = 12.0
    dpk = 10.0
    result = select_pediatric_formulation(DRUG, 24.0, weight, dpk, SOLUBILITY)
    assert abs(result.dose_mg - dpk * weight) < 1e-9


# ---------------------------------------------------------------------------
# age_group correct
# ---------------------------------------------------------------------------


def test_age_group_neonate():
    result = select_pediatric_formulation(DRUG, 0.5, 3.5, DOSE_PER_KG)
    assert result.age_group == "neonate"


def test_age_group_infant():
    result = select_pediatric_formulation(DRUG, 6.0, 7.0, DOSE_PER_KG)
    assert result.age_group == "infant"


def test_age_group_toddler():
    result = select_pediatric_formulation(DRUG, 24.0, 12.0, DOSE_PER_KG)
    assert result.age_group == "toddler"


def test_age_group_child():
    result = select_pediatric_formulation(DRUG, 90.0, 20.0, DOSE_PER_KG)
    assert result.age_group == "child"


def test_age_group_adolescent():
    result = select_pediatric_formulation(DRUG, 180.0, 55.0, DOSE_PER_KG)
    assert result.age_group == "adolescent"


# ---------------------------------------------------------------------------
# Palatability score range
# ---------------------------------------------------------------------------


def test_palatability_score_in_range():
    for age, weight in [(0.5, 3.5), (6.0, 7.0), (24.0, 12.0), (90.0, 20.0), (180.0, 55.0)]:
        result = select_pediatric_formulation(DRUG, age, weight, DOSE_PER_KG)
        assert 0 <= result.palatability_score <= 100, f"palatability out of range for age {age}"


# ---------------------------------------------------------------------------
# Neonate gets a liquid form
# ---------------------------------------------------------------------------


def test_neonate_gets_liquid():
    result = select_pediatric_formulation(DRUG, 0.5, 3.5, DOSE_PER_KG)
    assert result.recommended_formulation in ("solution", "suspension"), (
        f"Neonate should get liquid, got {result.recommended_formulation}"
    )


def test_neonate_volume_positive():
    result = select_pediatric_formulation(DRUG, 0.5, 3.5, DOSE_PER_KG)
    assert result.volume_mL > 0


def test_neonate_concentration_positive():
    result = select_pediatric_formulation(DRUG, 0.5, 3.5, DOSE_PER_KG)
    assert result.concentration_mg_mL > 0


# ---------------------------------------------------------------------------
# Adolescent can get solid forms
# ---------------------------------------------------------------------------


def test_adolescent_recommended_form():
    result = select_pediatric_formulation(DRUG, 180.0, 55.0, DOSE_PER_KG)
    assert result.recommended_formulation in ("tablet", "capsule", "solution")


# ---------------------------------------------------------------------------
# dose_accuracy_risk valid values
# ---------------------------------------------------------------------------


def test_dose_accuracy_risk_valid():
    valid_risks = {"high", "moderate", "low"}
    for age, weight in [(0.5, 3.5), (6.0, 7.0), (24.0, 12.0), (90.0, 20.0), (180.0, 55.0)]:
        result = select_pediatric_formulation(DRUG, age, weight, DOSE_PER_KG)
        assert result.dose_accuracy_risk in valid_risks


# ---------------------------------------------------------------------------
# age_appropriate flag
# ---------------------------------------------------------------------------


def test_age_appropriate_true():
    result = select_pediatric_formulation(DRUG, 6.0, 7.0, DOSE_PER_KG)
    assert result.age_appropriate is True


# ---------------------------------------------------------------------------
# compare_age_groups_for_drug
# ---------------------------------------------------------------------------


def test_compare_returns_5():
    results = compare_age_groups_for_drug(DRUG, DOSE_PER_KG)
    assert len(results) == 5


def test_compare_sorted_descending():
    results = compare_age_groups_for_drug(DRUG, DOSE_PER_KG)
    scores = [r.palatability_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_compare_all_age_groups_represented():
    results = compare_age_groups_for_drug(DRUG, DOSE_PER_KG)
    groups = {r.age_group for r in results}
    assert groups == {"neonate", "infant", "toddler", "child", "adolescent"}


def test_compare_return_type():
    results = compare_age_groups_for_drug(DRUG, DOSE_PER_KG)
    for r in results:
        assert isinstance(r, PediatricFormulationResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_age_zero():
    with pytest.raises(ValueError, match="age_months"):
        select_pediatric_formulation(DRUG, 0, 7.0, DOSE_PER_KG)


def test_invalid_age_negative():
    with pytest.raises(ValueError, match="age_months"):
        select_pediatric_formulation(DRUG, -1.0, 7.0, DOSE_PER_KG)


def test_invalid_weight_zero():
    with pytest.raises(ValueError, match="weight_kg"):
        select_pediatric_formulation(DRUG, 6.0, 0.0, DOSE_PER_KG)


def test_invalid_dose_per_kg_zero():
    with pytest.raises(ValueError, match="dose_per_kg_mg"):
        select_pediatric_formulation(DRUG, 6.0, 7.0, 0.0)


def test_invalid_dose_per_kg_negative():
    with pytest.raises(ValueError, match="dose_per_kg_mg"):
        select_pediatric_formulation(DRUG, 6.0, 7.0, -5.0)
