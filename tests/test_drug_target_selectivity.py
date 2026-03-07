"""Tests for drug target selectivity prediction (Phase 1002)."""

import math
import pytest

from omega_pbpk.prediction.drug_target_selectivity import (
    TargetSelectivityResult,
    compare_drug_selectivity,
    predict_target_selectivity,
    screen_selectivity_panel,
)


def test_returns_correct_type():
    result = predict_target_selectivity(
        "DrugA", "TargetX", 1.0, ["OffA", "OffB"], [100.0, 200.0]
    )
    assert isinstance(result, TargetSelectivityResult)


def test_minimum_si_positive():
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1"], [50.0])
    assert result.minimum_selectivity_index > 0


def test_mean_si_positive():
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1", "O2"], [50.0, 200.0])
    assert result.mean_selectivity_index > 0


def test_selectivity_class_valid():
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1"], [50.0])
    assert result.selectivity_class in {"highly_selective", "selective", "non_selective"}


def test_therapeutic_window_factor_positive():
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1"], [50.0])
    assert result.therapeutic_window_factor > 0


def test_si_calculation_correct():
    # SI = ki_off / ki_primary = 500 / 5 = 100
    result = predict_target_selectivity("DrugA", "T", 5.0, ["O1"], [500.0])
    assert math.isclose(result.selectivity_indices[0], 100.0)


def test_min_si_consistent_with_selectivity_class_highly_selective():
    # primary=1, off=1000 → SI=1000 → highly_selective
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1"], [1000.0])
    assert result.selectivity_class == "highly_selective"
    assert result.minimum_selectivity_index == 1000.0


def test_min_si_consistent_with_selectivity_class_non_selective():
    # primary=10, off=50 → SI=5 → non_selective
    result = predict_target_selectivity("DrugA", "T", 10.0, ["O1"], [50.0])
    assert result.selectivity_class == "non_selective"
    assert math.isclose(result.minimum_selectivity_index, 5.0)


def test_selective_class_boundary():
    # primary=1, off=50 → SI=50 → selective (10 < 50 <= 100)
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1"], [50.0])
    assert result.selectivity_class == "selective"


def test_therapeutic_window_factor_formula():
    # twf = min_SI / 10
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1"], [500.0])
    assert math.isclose(result.therapeutic_window_factor, 500.0 / 10.0)


def test_off_targets_stored_as_tuple():
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1", "O2"], [50.0, 100.0])
    assert isinstance(result.off_targets, tuple)


def test_off_target_ki_stored_as_tuple():
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1", "O2"], [50.0, 100.0])
    assert isinstance(result.off_target_ki_nM, tuple)


def test_multiple_off_targets_min_si():
    result = predict_target_selectivity(
        "DrugA", "T", 2.0, ["O1", "O2", "O3"], [20.0, 40.0, 10.0]
    )
    # SIs: 10, 20, 5 → min=5
    assert math.isclose(result.minimum_selectivity_index, 5.0)


def test_mean_si_calculation():
    result = predict_target_selectivity("DrugA", "T", 1.0, ["O1", "O2"], [10.0, 20.0])
    assert math.isclose(result.mean_selectivity_index, 15.0)


def test_screen_selectivity_panel():
    panel = {"hERG": 1000.0, "Nav1.5": 5000.0}
    result = screen_selectivity_panel("DrugA", "TargetX", 10.0, panel)
    assert isinstance(result, TargetSelectivityResult)
    assert result.minimum_selectivity_index > 0


def test_screen_selectivity_panel_si_correct():
    panel = {"hERG": 100.0}
    result = screen_selectivity_panel("DrugA", "T", 1.0, panel)
    assert math.isclose(result.selectivity_indices[0], 100.0)


def test_compare_drug_selectivity_returns_list():
    drugs = [
        {"drug_name": "A", "primary_target": "T", "primary_ki_nM": 1.0,
         "off_targets": ["O1"], "off_target_ki_nM": [100.0]},
        {"drug_name": "B", "primary_target": "T", "primary_ki_nM": 1.0,
         "off_targets": ["O1"], "off_target_ki_nM": [10.0]},
    ]
    results = compare_drug_selectivity(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_drug_selectivity_sorted_descending():
    drugs = [
        {"drug_name": "Low", "primary_target": "T", "primary_ki_nM": 1.0,
         "off_targets": ["O1"], "off_target_ki_nM": [5.0]},
        {"drug_name": "High", "primary_target": "T", "primary_ki_nM": 1.0,
         "off_targets": ["O1"], "off_target_ki_nM": [1000.0]},
        {"drug_name": "Mid", "primary_target": "T", "primary_ki_nM": 1.0,
         "off_targets": ["O1"], "off_target_ki_nM": [50.0]},
    ]
    results = compare_drug_selectivity(drugs)
    sis = [r.minimum_selectivity_index for r in results]
    assert sis == sorted(sis, reverse=True)


def test_invalid_primary_ki_raises():
    with pytest.raises(ValueError):
        predict_target_selectivity("DrugA", "T", 0.0, ["O1"], [50.0])


def test_negative_primary_ki_raises():
    with pytest.raises(ValueError):
        predict_target_selectivity("DrugA", "T", -1.0, ["O1"], [50.0])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        predict_target_selectivity("DrugA", "T", 1.0, ["O1", "O2"], [50.0])


def test_zero_off_target_ki_raises():
    with pytest.raises(ValueError):
        predict_target_selectivity("DrugA", "T", 1.0, ["O1"], [0.0])


def test_empty_off_targets_raises():
    with pytest.raises(ValueError):
        predict_target_selectivity("DrugA", "T", 1.0, [], [])
