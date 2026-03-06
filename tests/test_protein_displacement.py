"""Tests for omega_pbpk.clinical.protein_displacement module."""

import pytest

from omega_pbpk.clinical.protein_displacement import (
    DisplacementResult,
    _competitive_displacement,
    assess_protein_displacement,
    displacement_sensitivity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**overrides):
    kwargs = dict(
        victim_drug="warfarin",
        perpetrator_drug="ibuprofen",
        baseline_fup=0.01,
        ki_app_uM=50.0,
        competitor_conc_uM=100.0,
    )
    kwargs.update(overrides)
    return assess_protein_displacement(**kwargs)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_returns_displacement_result():
    result = _default_result()
    assert isinstance(result, DisplacementResult)


# ---------------------------------------------------------------------------
# 2. DisplacementResult dataclass fields present
# ---------------------------------------------------------------------------


def test_dataclass_fields():
    result = _default_result()
    for field in (
        "victim_drug",
        "perpetrator_drug",
        "baseline_fup",
        "displaced_fup",
        "fup_ratio",
        "cl_ratio",
        "auc_ratio",
        "is_clinically_significant",
        "binding_site",
        "mechanism",
        "recommendation",
    ):
        assert hasattr(result, field), f"Missing field: {field}"


# ---------------------------------------------------------------------------
# 3. displaced_fup >= baseline_fup
# ---------------------------------------------------------------------------


def test_displaced_fup_gte_baseline():
    result = _default_result(baseline_fup=0.05, ki_app_uM=100.0, competitor_conc_uM=50.0)
    assert result.displaced_fup >= result.baseline_fup


def test_displaced_fup_gte_baseline_zero_conc():
    result = _default_result(baseline_fup=0.1, competitor_conc_uM=0.0)
    assert result.displaced_fup >= result.baseline_fup


# ---------------------------------------------------------------------------
# 4. fup_ratio >= 1.0
# ---------------------------------------------------------------------------


def test_fup_ratio_gte_one():
    result = _default_result()
    assert result.fup_ratio >= 1.0


def test_fup_ratio_gte_one_low_conc():
    result = _default_result(competitor_conc_uM=0.0)
    assert result.fup_ratio >= 1.0


# ---------------------------------------------------------------------------
# 5. auc_ratio <= 1.0
# ---------------------------------------------------------------------------


def test_auc_ratio_lte_one():
    result = _default_result()
    assert result.auc_ratio <= 1.0


def test_auc_ratio_lte_one_high_displacement():
    result = _default_result(baseline_fup=0.01, ki_app_uM=1.0, competitor_conc_uM=1000.0)
    assert result.auc_ratio <= 1.0


# ---------------------------------------------------------------------------
# 6. Higher conc / lower ki -> higher fup_ratio
# ---------------------------------------------------------------------------


def test_higher_conc_higher_fup_ratio():
    low = assess_protein_displacement("warfarin", "ibuprofen", 0.02, 50.0, 10.0)
    high = assess_protein_displacement("warfarin", "ibuprofen", 0.02, 50.0, 500.0)
    assert high.fup_ratio > low.fup_ratio


def test_lower_ki_higher_fup_ratio():
    weak = assess_protein_displacement("warfarin", "ibuprofen", 0.02, 500.0, 100.0)
    strong = assess_protein_displacement("warfarin", "ibuprofen", 0.02, 10.0, 100.0)
    assert strong.fup_ratio > weak.fup_ratio


# ---------------------------------------------------------------------------
# 7. is_clinically_significant=True when fup_ratio>1.5 AND is_nti=True
# ---------------------------------------------------------------------------


def test_clinically_significant_nti_high_displacement():
    result = assess_protein_displacement(
        "warfarin",
        "ibuprofen",
        baseline_fup=0.01,
        ki_app_uM=1.0,
        competitor_conc_uM=1000.0,
        is_nti=True,
    )
    assert result.fup_ratio > 1.5
    assert result.is_clinically_significant is True


# ---------------------------------------------------------------------------
# 8. is_clinically_significant=False when not NTI (even if fup_ratio>1.5)
# ---------------------------------------------------------------------------


def test_not_significant_when_not_nti():
    result = assess_protein_displacement(
        "drug_a",
        "drug_b",
        baseline_fup=0.01,
        ki_app_uM=1.0,
        competitor_conc_uM=1000.0,
        is_nti=False,
    )
    assert result.fup_ratio > 1.5
    assert result.is_clinically_significant is False


# ---------------------------------------------------------------------------
# 9. is_clinically_significant=False when fup_ratio<=1.5 (even if NTI)
# ---------------------------------------------------------------------------


def test_not_significant_when_low_fup_ratio_nti():
    result = assess_protein_displacement(
        "warfarin",
        "naproxen",
        baseline_fup=0.5,
        ki_app_uM=1000.0,
        competitor_conc_uM=0.1,
        is_nti=True,
    )
    assert result.fup_ratio <= 1.5
    assert result.is_clinically_significant is False


# ---------------------------------------------------------------------------
# 10. ValueError: baseline_fup <= 0
# ---------------------------------------------------------------------------


def test_raises_value_error_baseline_fup_zero():
    with pytest.raises(ValueError):
        assess_protein_displacement(
            "w", "i", baseline_fup=0.0, ki_app_uM=50.0, competitor_conc_uM=10.0
        )


def test_raises_value_error_baseline_fup_negative():
    with pytest.raises(ValueError):
        assess_protein_displacement(
            "w", "i", baseline_fup=-0.1, ki_app_uM=50.0, competitor_conc_uM=10.0
        )


# ---------------------------------------------------------------------------
# 11. ValueError: ki_app <= 0
# ---------------------------------------------------------------------------


def test_raises_value_error_ki_app_zero():
    with pytest.raises(ValueError):
        assess_protein_displacement(
            "w", "i", baseline_fup=0.1, ki_app_uM=0.0, competitor_conc_uM=10.0
        )


def test_raises_value_error_ki_app_negative():
    with pytest.raises(ValueError):
        assess_protein_displacement(
            "w", "i", baseline_fup=0.1, ki_app_uM=-5.0, competitor_conc_uM=10.0
        )


# ---------------------------------------------------------------------------
# 12. ValueError: competitor_conc < 0
# ---------------------------------------------------------------------------


def test_raises_value_error_negative_conc():
    with pytest.raises(ValueError):
        assess_protein_displacement(
            "w", "i", baseline_fup=0.1, ki_app_uM=50.0, competitor_conc_uM=-1.0
        )


# ---------------------------------------------------------------------------
# 13. displacement_sensitivity returns list with len == len(conc_range)
# ---------------------------------------------------------------------------


def test_sensitivity_length():
    conc_range = [0.0, 10.0, 50.0, 100.0, 500.0]
    results = displacement_sensitivity("warfarin", 0.02, 50.0, conc_range)
    assert len(results) == len(conc_range)


def test_sensitivity_single_point():
    results = displacement_sensitivity("digoxin", 0.1, 200.0, [25.0])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# 14. displacement_sensitivity: higher conc -> higher fup_ratio (monotonic)
# ---------------------------------------------------------------------------


def test_sensitivity_monotone_fup_ratio():
    conc_range = [0.0, 10.0, 50.0, 200.0, 1000.0]
    results = displacement_sensitivity("phenytoin", 0.1, 80.0, conc_range)
    ratios = [r.fup_ratio for r in results]
    assert ratios == sorted(ratios)


# ---------------------------------------------------------------------------
# 15. recommendation is a non-empty string
# ---------------------------------------------------------------------------


def test_recommendation_non_empty():
    result = _default_result()
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation.strip()) > 0


def test_recommendation_non_empty_significant():
    result = assess_protein_displacement(
        "warfarin",
        "ibuprofen",
        baseline_fup=0.01,
        ki_app_uM=1.0,
        competitor_conc_uM=1000.0,
        is_nti=True,
    )
    assert isinstance(result.recommendation, str)
    assert len(result.recommendation.strip()) > 0


# ---------------------------------------------------------------------------
# 16. _competitive_displacement internal function
# ---------------------------------------------------------------------------


def test_competitive_displacement_zero_conc():
    """With zero competitor concentration, displaced fup equals baseline fup."""
    fup_baseline = 0.05
    result = _competitive_displacement(fup_baseline, ki_app=100.0, competitor_conc_uM=0.0)
    assert result == pytest.approx(fup_baseline)


def test_competitive_displacement_bounded_by_one():
    """displaced_fup must not exceed 1.0 regardless of formula output."""
    result = _competitive_displacement(0.99, ki_app=0.001, competitor_conc_uM=1e9)
    assert result <= 1.0


def test_competitive_displacement_increases_with_conc():
    low = _competitive_displacement(0.05, ki_app=50.0, competitor_conc_uM=10.0)
    high = _competitive_displacement(0.05, ki_app=50.0, competitor_conc_uM=500.0)
    assert high > low


# ---------------------------------------------------------------------------
# 17. binding_site propagated correctly
# ---------------------------------------------------------------------------


def test_binding_site_default():
    result = assess_protein_displacement("w", "i", 0.05, 50.0, 10.0)
    assert result.binding_site == "albumin_site_I"


def test_binding_site_custom():
    result = assess_protein_displacement("w", "i", 0.05, 50.0, 10.0, binding_site="albumin_site_II")
    assert result.binding_site == "albumin_site_II"


# ---------------------------------------------------------------------------
# 18. Drug names preserved in result
# ---------------------------------------------------------------------------


def test_drug_names_in_result():
    result = assess_protein_displacement("theophylline", "erythromycin", 0.4, 200.0, 50.0)
    assert result.victim_drug == "theophylline"
    assert result.perpetrator_drug == "erythromycin"


# ---------------------------------------------------------------------------
# 19. auc_ratio is inverse of fup_ratio
# ---------------------------------------------------------------------------


def test_auc_ratio_inverse_of_fup_ratio():
    result = _default_result(baseline_fup=0.1, ki_app_uM=50.0, competitor_conc_uM=50.0)
    assert result.auc_ratio == pytest.approx(1.0 / result.fup_ratio, rel=1e-6)


# ---------------------------------------------------------------------------
# 20. cl_ratio equals fup_ratio (proportional clearance increase)
# ---------------------------------------------------------------------------


def test_cl_ratio_equals_fup_ratio():
    result = _default_result(baseline_fup=0.05, ki_app_uM=100.0, competitor_conc_uM=200.0)
    assert result.cl_ratio == pytest.approx(result.fup_ratio, rel=1e-6)


# ---------------------------------------------------------------------------
# 21. Sensitivity all results are DisplacementResult
# ---------------------------------------------------------------------------


def test_sensitivity_result_types():
    results = displacement_sensitivity("lithium", 0.8, 300.0, [0.0, 25.0, 100.0])
    for r in results:
        assert isinstance(r, DisplacementResult)


# ---------------------------------------------------------------------------
# 22. displaced_fup capped at 1.0
# ---------------------------------------------------------------------------


def test_displaced_fup_capped_at_one():
    result = assess_protein_displacement(
        "w",
        "i",
        baseline_fup=0.99,
        ki_app_uM=0.001,
        competitor_conc_uM=1e9,
    )
    assert result.displaced_fup <= 1.0
