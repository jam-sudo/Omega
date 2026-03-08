"""Tests for Phase 465 — Pancreatic Enzyme Effect on Drug Absorption."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.pancreatic_enzyme_absorption import (
    PancreaticEnzymeAbsorptionResult,
    predict_pancreatic_enzyme_absorption,
    screen_pancreatic_conditions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _predict(
    drug_type: str = "prodrug",
    condition: str = "normal",
    dose_mg: float = 100.0,
    base_fa: float = 0.8,
    drug_name: str = "DrugX",
) -> PancreaticEnzymeAbsorptionResult:
    return predict_pancreatic_enzyme_absorption(
        drug_name=drug_name,
        drug_type=drug_type,
        pancreatic_condition=condition,
        dose_mg=dose_mg,
        base_fa=base_fa,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type_prodrug_normal():
    result = _predict(drug_type="prodrug", condition="normal")
    assert isinstance(result, PancreaticEnzymeAbsorptionResult)


def test_return_type_stable():
    result = _predict(drug_type="stable", condition="pancreatitis")
    assert isinstance(result, PancreaticEnzymeAbsorptionResult)


def test_return_type_labile():
    result = _predict(drug_type="labile", condition="normal")
    assert isinstance(result, PancreaticEnzymeAbsorptionResult)


# ---------------------------------------------------------------------------
# Prodrug tests
# ---------------------------------------------------------------------------


def test_prodrug_normal_fa_equals_base_fa():
    result = _predict(drug_type="prodrug", condition="normal", base_fa=0.8)
    assert result.fa_adjusted == pytest.approx(0.8, rel=1e-6)


def test_prodrug_normal_recommendation_standard():
    result = _predict(drug_type="prodrug", condition="normal")
    assert result.dose_recommendation == "standard"


def test_prodrug_pancreatitis_fa_less_than_normal():
    normal = _predict(drug_type="prodrug", condition="normal")
    pancreatitis = _predict(drug_type="prodrug", condition="pancreatitis")
    assert pancreatitis.fa_adjusted < normal.fa_adjusted


def test_prodrug_insufficiency_fa_less_than_normal():
    normal = _predict(drug_type="prodrug", condition="normal")
    insuff = _predict(drug_type="prodrug", condition="pancreatic_insufficiency")
    assert insuff.fa_adjusted < normal.fa_adjusted


def test_prodrug_insufficiency_recommendation_enzyme_supplement():
    result = _predict(drug_type="prodrug", condition="pancreatic_insufficiency")
    assert result.dose_recommendation == "enzyme_supplement"


def test_prodrug_pancreatitis_recommendation_enzyme_supplement():
    result = _predict(drug_type="prodrug", condition="pancreatitis")
    assert result.dose_recommendation == "enzyme_supplement"


def test_prodrug_post_surgical_recommendation_enzyme_supplement():
    result = _predict(drug_type="prodrug", condition="post_surgical")
    assert result.dose_recommendation == "enzyme_supplement"


def test_prodrug_insufficiency_enzyme_factor():
    result = _predict(drug_type="prodrug", condition="pancreatic_insufficiency")
    assert result.enzyme_factor == pytest.approx(0.1, rel=1e-6)


# ---------------------------------------------------------------------------
# Stable drug tests
# ---------------------------------------------------------------------------


def test_stable_fa_unchanged_normal():
    result = _predict(drug_type="stable", condition="normal", base_fa=0.6)
    assert result.fa_adjusted == pytest.approx(0.6, rel=1e-6)


def test_stable_fa_unchanged_insufficiency():
    result = _predict(drug_type="stable", condition="pancreatic_insufficiency", base_fa=0.6)
    assert result.fa_adjusted == pytest.approx(0.6, rel=1e-6)


def test_stable_recommendation_always_standard():
    for cond in ("normal", "pancreatitis", "pancreatic_insufficiency", "post_surgical"):
        result = _predict(drug_type="stable", condition=cond)
        assert result.dose_recommendation == "standard", f"Failed for condition {cond}"


# ---------------------------------------------------------------------------
# Labile drug tests
# ---------------------------------------------------------------------------


def test_labile_normal_fa_less_than_base_fa():
    # In normal condition enzyme_factor=1.0: fa = base_fa * max(0.1, 1.0 - 0.5*1.0) = base_fa * 0.5
    result = _predict(drug_type="labile", condition="normal", base_fa=0.8)
    assert result.fa_adjusted < result.base_fa


def test_labile_normal_fa_equals_half_base_fa():
    result = _predict(drug_type="labile", condition="normal", base_fa=0.8)
    assert result.fa_adjusted == pytest.approx(0.8 * 0.5, rel=1e-6)


def test_labile_insufficiency_fa_higher_than_normal():
    # Low enzyme activity → less degradation → higher fa
    normal = _predict(drug_type="labile", condition="normal")
    insuff = _predict(drug_type="labile", condition="pancreatic_insufficiency")
    assert insuff.fa_adjusted > normal.fa_adjusted


def test_labile_insufficiency_fa_value():
    # enzyme_factor=0.1: fa = base_fa * max(0.1, 1.0 - 0.5*0.1) = base_fa * 0.95
    result = _predict(drug_type="labile", condition="pancreatic_insufficiency", base_fa=0.8)
    expected = 0.8 * max(0.1, 1.0 - 0.5 * 0.1)
    assert result.fa_adjusted == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# dose_absorbed_mg tests
# ---------------------------------------------------------------------------


def test_dose_absorbed_mg_prodrug_normal():
    result = _predict(drug_type="prodrug", condition="normal", dose_mg=200.0, base_fa=0.8)
    assert result.dose_absorbed_mg == pytest.approx(result.fa_adjusted * 200.0, rel=1e-6)


def test_dose_absorbed_mg_stable():
    result = _predict(drug_type="stable", condition="pancreatitis", dose_mg=50.0, base_fa=0.5)
    assert result.dose_absorbed_mg == pytest.approx(0.5 * 50.0, rel=1e-6)


def test_dose_absorbed_mg_labile():
    result = _predict(drug_type="labile", condition="normal", dose_mg=100.0, base_fa=0.8)
    assert result.dose_absorbed_mg == pytest.approx(result.fa_adjusted * 100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Screen function tests
# ---------------------------------------------------------------------------


def test_screen_returns_four_items():
    results = screen_pancreatic_conditions("DrugX", "prodrug", 100.0)
    assert len(results) == 4


def test_screen_sorted_descending_prodrug():
    results = screen_pancreatic_conditions("DrugX", "prodrug", 100.0)
    fas = [r.fa_adjusted for r in results]
    assert fas == sorted(fas, reverse=True)


def test_screen_sorted_descending_labile():
    results = screen_pancreatic_conditions("DrugX", "labile", 100.0)
    fas = [r.fa_adjusted for r in results]
    assert fas == sorted(fas, reverse=True)


def test_screen_stable_all_same_fa():
    results = screen_pancreatic_conditions("DrugX", "stable", 100.0, base_fa=0.7)
    fas = {r.fa_adjusted for r in results}
    assert len(fas) == 1
    assert list(fas)[0] == pytest.approx(0.7, rel=1e-6)


def test_screen_all_conditions_present():
    results = screen_pancreatic_conditions("DrugX", "prodrug", 100.0)
    conditions = {r.pancreatic_condition for r in results}
    assert conditions == {"normal", "pancreatitis", "pancreatic_insufficiency", "post_surgical"}


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_invalid_drug_type_raises():
    with pytest.raises(ValueError, match="drug_type"):
        predict_pancreatic_enzyme_absorption("DrugX", "unknown", "normal", 100.0)


def test_invalid_condition_raises():
    with pytest.raises(ValueError, match="pancreatic_condition"):
        predict_pancreatic_enzyme_absorption("DrugX", "prodrug", "diabetes", 100.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_pancreatic_enzyme_absorption("DrugX", "prodrug", "normal", -10.0)


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_pancreatic_enzyme_absorption("DrugX", "prodrug", "normal", 0.0)


def test_invalid_base_fa_raises():
    with pytest.raises(ValueError, match="base_fa"):
        predict_pancreatic_enzyme_absorption("DrugX", "prodrug", "normal", 100.0, base_fa=0.0)


def test_invalid_base_fa_above_one_raises():
    with pytest.raises(ValueError, match="base_fa"):
        predict_pancreatic_enzyme_absorption("DrugX", "prodrug", "normal", 100.0, base_fa=1.5)
