"""
Tests for Phase 233: Absorption Enhancement Predictor
~20 tests covering return types, physical constraints, edge cases, validation.
"""

import pytest

from omega_pbpk.prediction.absorption_enhancement_predictor import (
    VALID_ENHANCER_TYPES,
    AbsorptionEnhancerResult,
    predict_absorption_enhancement,
    screen_enhancers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic(enhancer_type="surfactant", f_oral=0.3):
    return predict_absorption_enhancement(
        drug_name="TestDrug",
        baseline_peff_cm_s=1e-5,
        baseline_f_oral=f_oral,
        enhancer_type=enhancer_type,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _basic()
    assert isinstance(result, AbsorptionEnhancerResult)


def test_result_is_frozen():
    result = _basic()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "changed"  # type: ignore[misc]


def test_drug_name_preserved():
    result = predict_absorption_enhancement("Aspirin", 1e-5, 0.5, "lipid")
    assert result.drug_name == "Aspirin"


def test_enhancer_type_preserved():
    result = _basic("chitosan")
    assert result.enhancer_type == "chitosan"


# ---------------------------------------------------------------------------
# Permeability enhancement
# ---------------------------------------------------------------------------


def test_enhanced_peff_greater_for_surfactant():
    result = _basic("surfactant")
    assert result.enhanced_peff_cm_s > result.baseline_peff_cm_s


def test_enhanced_peff_greater_for_lipid():
    result = _basic("lipid")
    assert result.enhanced_peff_cm_s > result.baseline_peff_cm_s


def test_enhanced_peff_greater_for_cyclodextrin():
    result = _basic("cyclodextrin")
    assert result.enhanced_peff_cm_s > result.baseline_peff_cm_s


def test_enhanced_peff_greater_for_chitosan():
    result = _basic("chitosan")
    assert result.enhanced_peff_cm_s > result.baseline_peff_cm_s


def test_none_enhancer_ratio_equals_one():
    result = _basic("none")
    assert abs(result.enhancement_ratio - 1.0) < 1e-12


def test_none_enhancer_peff_unchanged():
    result = _basic("none")
    assert abs(result.enhanced_peff_cm_s - result.baseline_peff_cm_s) < 1e-20


# ---------------------------------------------------------------------------
# Bioavailability
# ---------------------------------------------------------------------------


def test_enhanced_f_oral_gte_baseline():
    for enh in VALID_ENHANCER_TYPES:
        result = _basic(enh)
        assert result.enhanced_f_oral >= result.baseline_f_oral, f"Failed for enhancer: {enh}"


def test_enhanced_f_oral_capped_at_one():
    # baseline 0.95 + surfactant 0.15 = 1.10 → capped at 1.0
    result = predict_absorption_enhancement("Drug", 1e-5, 0.95, "surfactant")
    assert result.enhanced_f_oral <= 1.0


def test_enhanced_f_exactly_one_when_capped():
    result = predict_absorption_enhancement("Drug", 1e-5, 0.95, "surfactant")
    assert result.enhanced_f_oral == 1.0


def test_bioavailability_improvement_pct_nonneg():
    for enh in VALID_ENHANCER_TYPES:
        result = _basic(enh)
        assert result.bioavailability_improvement_pct >= 0.0, f"Failed for enhancer: {enh}"


def test_none_improvement_pct_is_zero():
    result = _basic("none")
    assert abs(result.bioavailability_improvement_pct) < 1e-12


# ---------------------------------------------------------------------------
# screen_enhancers
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    results = screen_enhancers("Drug", 1e-5, 0.3)
    assert isinstance(results, list)


def test_screen_returns_five_results():
    results = screen_enhancers("Drug", 1e-5, 0.3)
    assert len(results) == 5


def test_screen_sorted_descending_by_enhanced_f():
    results = screen_enhancers("Drug", 1e-5, 0.3)
    f_values = [r.enhanced_f_oral for r in results]
    assert f_values == sorted(f_values, reverse=True)


def test_screen_all_enhancer_types_covered():
    results = screen_enhancers("Drug", 1e-5, 0.3)
    found_types = {r.enhancer_type for r in results}
    assert found_types == set(VALID_ENHANCER_TYPES)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_peff_zero_raises():
    with pytest.raises(ValueError, match="baseline_peff_cm_s"):
        predict_absorption_enhancement("Drug", 0.0, 0.3, "surfactant")


def test_peff_negative_raises():
    with pytest.raises(ValueError, match="baseline_peff_cm_s"):
        predict_absorption_enhancement("Drug", -1e-5, 0.3, "surfactant")


def test_f_oral_above_one_raises():
    with pytest.raises(ValueError, match="baseline_f_oral"):
        predict_absorption_enhancement("Drug", 1e-5, 1.1, "surfactant")


def test_f_oral_zero_raises():
    with pytest.raises(ValueError, match="baseline_f_oral"):
        predict_absorption_enhancement("Drug", 1e-5, 0.0, "surfactant")


def test_invalid_enhancer_raises():
    with pytest.raises(ValueError, match="enhancer_type"):
        predict_absorption_enhancement("Drug", 1e-5, 0.3, "unknown_enhancer")
