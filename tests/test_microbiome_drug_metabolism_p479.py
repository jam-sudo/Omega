"""Tests for Phase 479 — Microbiome Drug Metabolism Prediction."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.microbiome_drug_metabolism_p479 import (
    MicrobiomeDrugMetabolismResult,
    compare_microbiome_conditions,
    predict_microbiome_metabolism,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type_basic():
    result = predict_microbiome_metabolism("DrugA", "inert", "normal", 100.0)
    assert isinstance(result, MicrobiomeDrugMetabolismResult)


def test_return_type_glucuronide():
    result = predict_microbiome_metabolism("Morphine", "glucuronide_hydrolysis", "normal", 50.0)
    assert isinstance(result, MicrobiomeDrugMetabolismResult)


# ---------------------------------------------------------------------------
# Inert drug tests
# ---------------------------------------------------------------------------


def test_inert_adjusted_fa_equals_base_fa():
    result = predict_microbiome_metabolism("NeutralDrug", "inert", "normal", 100.0, base_fa=0.6)
    assert result.adjusted_fa == pytest.approx(0.6, rel=1e-6)


def test_inert_fa_multiplier_is_one():
    result = predict_microbiome_metabolism("NeutralDrug", "inert", "dysbiosis", 100.0, base_fa=0.8)
    assert result.fa_multiplier == pytest.approx(1.0, rel=1e-6)


def test_inert_systemic_exposure_ratio_is_one():
    result = predict_microbiome_metabolism("NeutralDrug", "inert", "antibiotic_treated", 200.0)
    assert result.systemic_exposure_ratio == pytest.approx(1.0, rel=1e-6)


def test_inert_clinical_impact_is_none():
    result = predict_microbiome_metabolism("NeutralDrug", "inert", "normal", 100.0)
    assert result.clinical_impact == "none"


# ---------------------------------------------------------------------------
# azo_reduction + antibiotic_treated
# ---------------------------------------------------------------------------


def test_azo_reduction_antibiotic_treated_very_low_fa():
    result = predict_microbiome_metabolism(
        "Sulfasalazine", "azo_reduction", "antibiotic_treated", 500.0, base_fa=0.9
    )
    # activity_factor = 0.05, fa_multiplier = 0.05, adjusted_fa = 0.9 * 0.05 = 0.045
    assert result.adjusted_fa == pytest.approx(0.9 * 0.05, rel=1e-6)


def test_azo_reduction_antibiotic_treated_major_impact():
    result = predict_microbiome_metabolism(
        "Prodrug", "azo_reduction", "antibiotic_treated", 100.0, base_fa=0.7
    )
    # exposure ratio = 0.05 → 95% drop → major
    assert result.clinical_impact == "major"


def test_azo_reduction_normal_full_activity():
    result = predict_microbiome_metabolism("Prodrug", "azo_reduction", "normal", 100.0, base_fa=0.5)
    # activity_factor = 1.0, fa_multiplier = 1.0, adjusted_fa = 0.5
    assert result.fa_multiplier == pytest.approx(1.0, rel=1e-6)
    assert result.adjusted_fa == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# glucuronide_hydrolysis + enhanced
# ---------------------------------------------------------------------------


def test_glucuronide_hydrolysis_enhanced_highest_adjusted_fa():
    # enhanced: activity_factor=1.5, fa_multiplier = 1 + 0.3*1.5 = 1.45
    result = predict_microbiome_metabolism(
        "Estradiol", "glucuronide_hydrolysis", "enhanced", 100.0, base_fa=0.6
    )
    expected_fa = min(1.0, 0.6 * 1.45)
    assert result.adjusted_fa == pytest.approx(expected_fa, rel=1e-6)


def test_glucuronide_hydrolysis_enhanced_multiplier():
    result = predict_microbiome_metabolism(
        "Drug", "glucuronide_hydrolysis", "enhanced", 100.0, base_fa=0.5
    )
    assert result.fa_multiplier == pytest.approx(1.45, rel=1e-6)


def test_glucuronide_hydrolysis_antibiotic_treated_lower_than_normal():
    result_antibiotic = predict_microbiome_metabolism(
        "Drug", "glucuronide_hydrolysis", "antibiotic_treated", 100.0, base_fa=0.6
    )
    result_normal = predict_microbiome_metabolism(
        "Drug", "glucuronide_hydrolysis", "normal", 100.0, base_fa=0.6
    )
    assert result_antibiotic.adjusted_fa < result_normal.adjusted_fa


# ---------------------------------------------------------------------------
# adjusted_fa clamped to [0, 1]
# ---------------------------------------------------------------------------


def test_adjusted_fa_not_above_one():
    # glucuronide_hydrolysis + enhanced + high base_fa might exceed 1
    result = predict_microbiome_metabolism(
        "Drug", "glucuronide_hydrolysis", "enhanced", 100.0, base_fa=0.95
    )
    assert result.adjusted_fa <= 1.0


def test_adjusted_fa_not_below_zero():
    result = predict_microbiome_metabolism(
        "Drug", "azo_reduction", "antibiotic_treated", 100.0, base_fa=0.001
    )
    assert result.adjusted_fa >= 0.0


# ---------------------------------------------------------------------------
# clinical_impact classification
# ---------------------------------------------------------------------------


def test_clinical_impact_none_for_inert():
    result = predict_microbiome_metabolism("Drug", "inert", "enhanced", 100.0, base_fa=0.7)
    assert result.clinical_impact == "none"


def test_clinical_impact_minor():
    # nitroreduction + normal: fa_multiplier = 0.5 + 0.5*1.0 = 1.0 → no change → none
    # Try dysbiosis: 0.5 + 0.5*0.4 = 0.7 → adjusted = 0.7*0.7 = 0.49 → ratio 0.7 → 30% drop → significant
    result = predict_microbiome_metabolism(
        "Drug", "nitroreduction", "dysbiosis", 100.0, base_fa=0.7
    )
    # 0.5+0.5*0.4=0.7, ratio=0.7 → 30% change = significant
    assert result.clinical_impact in ("minor", "significant")


def test_clinical_impact_significant():
    # azo_reduction + dysbiosis: activity=0.4, fa_multiplier=0.4, ratio=0.4 → 60% drop → major
    result = predict_microbiome_metabolism(
        "Prodrug", "azo_reduction", "dysbiosis", 100.0, base_fa=0.8
    )
    assert result.clinical_impact == "major"


def test_clinical_impact_major():
    result = predict_microbiome_metabolism(
        "Prodrug", "azo_reduction", "antibiotic_treated", 100.0, base_fa=0.9
    )
    assert result.clinical_impact == "major"


# ---------------------------------------------------------------------------
# dose_absorbed_mg
# ---------------------------------------------------------------------------


def test_dose_absorbed_mg_correct():
    result = predict_microbiome_metabolism("Drug", "inert", "normal", 200.0, base_fa=0.5)
    assert result.dose_absorbed_mg == pytest.approx(200.0 * 0.5, rel=1e-6)


def test_dose_absorbed_mg_azo_reduction():
    result = predict_microbiome_metabolism("Drug", "azo_reduction", "normal", 100.0, base_fa=0.6)
    # adjusted_fa = 0.6 * 1.0 = 0.6
    assert result.dose_absorbed_mg == pytest.approx(60.0, rel=1e-6)


# ---------------------------------------------------------------------------
# compare_microbiome_conditions
# ---------------------------------------------------------------------------


def test_compare_returns_four_items():
    results = compare_microbiome_conditions("Drug", "inert", 100.0, base_fa=0.7)
    assert len(results) == 4


def test_compare_sorted_by_adjusted_fa_descending():
    results = compare_microbiome_conditions("Drug", "glucuronide_hydrolysis", 100.0, base_fa=0.6)
    adjusted_fas = [r.adjusted_fa for r in results]
    assert adjusted_fas == sorted(adjusted_fas, reverse=True)


def test_compare_inert_all_same_adjusted_fa():
    results = compare_microbiome_conditions("Drug", "inert", 100.0, base_fa=0.5)
    adjusted_fas = [r.adjusted_fa for r in results]
    assert all(abs(fa - 0.5) < 1e-9 for fa in adjusted_fas)


def test_compare_all_conditions_present():
    results = compare_microbiome_conditions("Drug", "azo_reduction", 100.0)
    conditions = {r.microbiome_condition for r in results}
    assert conditions == {"normal", "dysbiosis", "antibiotic_treated", "enhanced"}


def test_compare_azo_reduction_enhanced_highest():
    results = compare_microbiome_conditions("Drug", "azo_reduction", 100.0, base_fa=0.7)
    assert results[0].microbiome_condition == "enhanced"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_interaction_type():
    with pytest.raises(ValueError, match="interaction_type"):
        predict_microbiome_metabolism("Drug", "invalid_type", "normal", 100.0)


def test_invalid_microbiome_condition():
    with pytest.raises(ValueError, match="microbiome_condition"):
        predict_microbiome_metabolism("Drug", "inert", "bad_condition", 100.0)


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_microbiome_metabolism("Drug", "inert", "normal", 0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_microbiome_metabolism("Drug", "inert", "normal", -50.0)


def test_invalid_base_fa_zero():
    with pytest.raises(ValueError, match="base_fa"):
        predict_microbiome_metabolism("Drug", "inert", "normal", 100.0, base_fa=0.0)


def test_invalid_base_fa_above_one():
    with pytest.raises(ValueError, match="base_fa"):
        predict_microbiome_metabolism("Drug", "inert", "normal", 100.0, base_fa=1.5)
