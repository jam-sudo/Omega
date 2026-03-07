"""Tests for Phase 780 — dose_rounding_v2.py."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.dose_rounding_v2 import (
    ClinicalDoseResult,
    batch_select_clinical_doses,
    find_best_tablet_combination,
    select_clinical_dose,
)

STRENGTHS = [25.0, 50.0, 100.0, 200.0, 250.0, 500.0]


# ---------------------------------------------------------------------------
# 1. Return type is ClinicalDoseResult
# ---------------------------------------------------------------------------
def test_return_type():
    result = select_clinical_dose("DrugX", 100.0)
    assert isinstance(result, ClinicalDoseResult)


# ---------------------------------------------------------------------------
# 2. Field types
# ---------------------------------------------------------------------------
def test_field_types():
    result = select_clinical_dose("DrugX", 150.0, STRENGTHS)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.target_dose_mg, float)
    assert isinstance(result.rounded_dose_mg, float)
    assert isinstance(result.available_strengths, list)
    assert isinstance(result.selected_strength_mg, float)
    assert isinstance(result.n_tablets, float)
    assert isinstance(result.dose_error_pct, float)
    assert isinstance(result.is_practical, bool)
    assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# 3. rounded_dose_mg close to target for standard strengths
# ---------------------------------------------------------------------------
def test_rounded_close_to_target():
    result = select_clinical_dose("DrugA", 100.0, STRENGTHS)
    assert abs(result.rounded_dose_mg - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# 4. dose_error_pct <= max_dose_error_pct when is_practical
# ---------------------------------------------------------------------------
def test_dose_error_within_threshold_when_practical():
    result = select_clinical_dose("DrugA", 100.0, STRENGTHS, max_dose_error_pct=15.0)
    if result.is_practical:
        assert result.dose_error_pct <= 15.0


# ---------------------------------------------------------------------------
# 5. n_tablets <= max_tablets when is_practical
# ---------------------------------------------------------------------------
def test_n_tablets_within_max_when_practical():
    result = select_clinical_dose("DrugA", 200.0, STRENGTHS, max_tablets=3)
    if result.is_practical:
        assert result.n_tablets <= 3.0


# ---------------------------------------------------------------------------
# 6. ValueError for target_dose_mg <= 0
# ---------------------------------------------------------------------------
def test_invalid_target_dose_zero():
    with pytest.raises(ValueError):
        select_clinical_dose("DrugA", 0.0, STRENGTHS)


# ---------------------------------------------------------------------------
# 7. ValueError for target_dose_mg negative
# ---------------------------------------------------------------------------
def test_invalid_target_dose_negative():
    with pytest.raises(ValueError):
        select_clinical_dose("DrugA", -50.0, STRENGTHS)


# ---------------------------------------------------------------------------
# 8. find_best_tablet_combination returns 3-tuple
# ---------------------------------------------------------------------------
def test_optimize_returns_tuple():
    result = find_best_tablet_combination(100.0, STRENGTHS)
    assert isinstance(result, tuple)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 9. find_best_tablet_combination actual_dose = selected_strength * n_tablets
# ---------------------------------------------------------------------------
def test_optimize_actual_dose_consistent():
    strength, n, actual = find_best_tablet_combination(150.0, STRENGTHS)
    assert abs(actual - strength * n) < 1e-6


# ---------------------------------------------------------------------------
# 10. find_best_tablet_combination ValueError on empty strengths
# ---------------------------------------------------------------------------
def test_optimize_empty_strengths():
    with pytest.raises(ValueError):
        find_best_tablet_combination(100.0, [])


# ---------------------------------------------------------------------------
# 11. find_best_tablet_combination ValueError on non-positive target
# ---------------------------------------------------------------------------
def test_optimize_invalid_target():
    with pytest.raises(ValueError):
        find_best_tablet_combination(-10.0, STRENGTHS)


# ---------------------------------------------------------------------------
# 12. Default strengths used when None provided
# ---------------------------------------------------------------------------
def test_default_strengths_used():
    result = select_clinical_dose("DrugB", 100.0)
    assert len(result.available_strengths) > 0
    assert 100.0 in result.available_strengths


# ---------------------------------------------------------------------------
# 13. available_strengths stored in result matches input
# ---------------------------------------------------------------------------
def test_available_strengths_stored():
    custom = [50.0, 100.0, 150.0]
    result = select_clinical_dose("DrugC", 100.0, custom)
    assert result.available_strengths == custom


# ---------------------------------------------------------------------------
# 14. is_practical True for exact strength match
# ---------------------------------------------------------------------------
def test_practical_exact_match():
    result = select_clinical_dose("DrugA", 100.0, [100.0], max_tablets=1, max_dose_error_pct=1.0)
    assert result.is_practical is True
    assert abs(result.dose_error_pct) < 1e-6


# ---------------------------------------------------------------------------
# 15. is_practical False when too many tablets required
# ---------------------------------------------------------------------------
def test_not_practical_too_many_tablets():
    # Force a situation where many tablets are needed: target=5 mg, only strength=100 mg
    # The best we can do is 0.5 tablets = 50 mg, error = 90% -> not practical
    result = select_clinical_dose("DrugA", 5.0, [100.0], max_tablets=1, max_dose_error_pct=15.0)
    assert result.is_practical is False


# ---------------------------------------------------------------------------
# 16. drug_name passthrough
# ---------------------------------------------------------------------------
def test_drug_name_passthrough():
    result = select_clinical_dose("Ibuprofen", 400.0, STRENGTHS)
    assert result.drug_name == "Ibuprofen"


# ---------------------------------------------------------------------------
# 17. notes is non-empty string
# ---------------------------------------------------------------------------
def test_notes_non_empty():
    result = select_clinical_dose("DrugD", 300.0, STRENGTHS)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 18. batch_select_clinical_doses returns list of ClinicalDoseResult
# ---------------------------------------------------------------------------
def test_batch_returns_list():
    params_list = [
        {"drug_name": "A", "target_dose_mg": 100.0, "available_strengths_mg": STRENGTHS},
        {"drug_name": "B", "target_dose_mg": 200.0, "available_strengths_mg": STRENGTHS},
    ]
    results = batch_select_clinical_doses(params_list)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, ClinicalDoseResult)


# ---------------------------------------------------------------------------
# 19. Half-tablet logic: 75 mg from 50+100 strengths
# ---------------------------------------------------------------------------
def test_half_tablet_selection():
    # 50 mg x 1.5 = 75 mg, or 100 mg x 0.5 = 50 (worse)
    result = select_clinical_dose("DrugE", 75.0, [50.0, 100.0])
    assert abs(result.rounded_dose_mg - 75.0) < 1e-6


# ---------------------------------------------------------------------------
# 20. dose_error_pct is non-negative
# ---------------------------------------------------------------------------
def test_dose_error_non_negative():
    result = select_clinical_dose("DrugF", 130.0, STRENGTHS)
    assert result.dose_error_pct >= 0.0
