"""Tests for Phase 207 — crystal_habit_predictor.py (~25 tests)."""

import pytest

from omega_pbpk.prediction.crystal_habit_predictor import (
    CrystalHabitResult,
    compare_polymorphs,
    predict_crystal_habit,
)

VALID_HABITS = {"needle", "plate", "prism", "cube"}
VALID_FLOWABILITY = {"poor", "moderate", "good", "excellent"}
VALID_COMPRESSIBILITY = {"poor", "moderate", "good"}
VALID_PROCESSING_RISK = {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Basic return type and field preservation
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_crystal_habit(
        "TestDrug", logP=1.0, mw_Da=300.0, hbd=3, hba=4, symmetry_index=0.5
    )
    assert isinstance(result, CrystalHabitResult)


def test_field_drug_name():
    result = predict_crystal_habit(
        "Aspirin", logP=1.2, mw_Da=180.2, hbd=2, hba=3, symmetry_index=0.4
    )
    assert result.drug_name == "Aspirin"


def test_field_logP():
    result = predict_crystal_habit("DrugA", logP=2.5, mw_Da=250.0, hbd=2, hba=4, symmetry_index=0.3)
    assert result.logP == 2.5


def test_field_mw_Da():
    result = predict_crystal_habit("DrugB", logP=1.0, mw_Da=350.0, hbd=1, hba=5, symmetry_index=0.5)
    assert result.mw_Da == 350.0


def test_field_hbd_hba():
    result = predict_crystal_habit("DrugC", logP=1.0, mw_Da=300.0, hbd=3, hba=5, symmetry_index=0.4)
    assert result.hbd == 3
    assert result.hba == 5


def test_field_symmetry_index():
    result = predict_crystal_habit("DrugD", logP=1.0, mw_Da=300.0, hbd=2, hba=3, symmetry_index=0.6)
    assert result.symmetry_index == 0.6


# ---------------------------------------------------------------------------
# Habit prediction rules
# ---------------------------------------------------------------------------


def test_high_hbd_hba_gives_needle():
    # hbd + hba = 9 > 8 → needle (symmetry_index < 0.7)
    result = predict_crystal_habit(
        "NeedleDrug", logP=1.0, mw_Da=300.0, hbd=5, hba=4, symmetry_index=0.3
    )
    assert result.habit_class == "needle"


def test_high_hbd_hba_boundary_gives_needle():
    # hbd + hba = 9 (barely over threshold)
    result = predict_crystal_habit(
        "NeedleDrug2", logP=2.0, mw_Da=350.0, hbd=4, hba=5, symmetry_index=0.2
    )
    assert result.habit_class == "needle"


def test_high_logP_low_hbd_gives_plate():
    # logP > 3, hbd < 2, hbd+hba <= 8, symmetry < 0.7 → plate
    result = predict_crystal_habit(
        "PlateDrug", logP=4.0, mw_Da=280.0, hbd=1, hba=3, symmetry_index=0.3
    )
    assert result.habit_class == "plate"


def test_high_logP_low_hbd_edge():
    # logP exactly 3.1, hbd=0 → plate
    result = predict_crystal_habit(
        "PlateEdge", logP=3.1, mw_Da=200.0, hbd=0, hba=2, symmetry_index=0.4
    )
    assert result.habit_class == "plate"


def test_high_symmetry_gives_cube():
    result = predict_crystal_habit(
        "CubeDrug", logP=4.0, mw_Da=300.0, hbd=1, hba=3, symmetry_index=0.8
    )
    assert result.habit_class == "cube"


def test_default_prism():
    # Moderate logP, moderate HBD+HBA, moderate symmetry → prism
    result = predict_crystal_habit(
        "PrismDrug", logP=1.5, mw_Da=300.0, hbd=2, hba=3, symmetry_index=0.4
    )
    assert result.habit_class == "prism"


# ---------------------------------------------------------------------------
# Dissolution modifier
# ---------------------------------------------------------------------------


def test_dissolution_modifier_positive():
    result = predict_crystal_habit("DrugX", logP=1.0, mw_Da=300.0, hbd=2, hba=3, symmetry_index=0.5)
    assert result.dissolution_modifier > 0


def test_needle_dissolution_modifier():
    result = predict_crystal_habit(
        "NeedleD", logP=1.0, mw_Da=300.0, hbd=5, hba=4, symmetry_index=0.2
    )
    assert result.dissolution_modifier == 1.2


def test_plate_dissolution_modifier():
    result = predict_crystal_habit(
        "PlateD", logP=4.0, mw_Da=280.0, hbd=1, hba=3, symmetry_index=0.3
    )
    assert result.dissolution_modifier == 0.8


def test_prism_dissolution_modifier():
    result = predict_crystal_habit(
        "PrismD", logP=1.5, mw_Da=300.0, hbd=2, hba=3, symmetry_index=0.4
    )
    assert result.dissolution_modifier == 1.0


def test_cube_dissolution_modifier():
    result = predict_crystal_habit(
        "CubeD", logP=2.0, mw_Da=300.0, hbd=1, hba=3, symmetry_index=0.75
    )
    assert result.dissolution_modifier == 0.9


# ---------------------------------------------------------------------------
# Flowability valid values
# ---------------------------------------------------------------------------


def test_flowability_valid():
    result = predict_crystal_habit(
        "FlowDrug", logP=1.0, mw_Da=300.0, hbd=2, hba=3, symmetry_index=0.5
    )
    assert result.flowability in VALID_FLOWABILITY


def test_needle_flowability_poor():
    result = predict_crystal_habit(
        "NeedleFlow", logP=1.0, mw_Da=300.0, hbd=5, hba=4, symmetry_index=0.2
    )
    assert result.flowability == "poor"


# ---------------------------------------------------------------------------
# Processing risk valid values
# ---------------------------------------------------------------------------


def test_processing_risk_valid():
    result = predict_crystal_habit(
        "RiskDrug", logP=1.0, mw_Da=300.0, hbd=2, hba=3, symmetry_index=0.5
    )
    assert result.processing_risk in VALID_PROCESSING_RISK


def test_needle_processing_risk_high():
    result = predict_crystal_habit(
        "NeedleRisk", logP=1.0, mw_Da=300.0, hbd=5, hba=5, symmetry_index=0.2
    )
    assert result.processing_risk == "high"


def test_prism_processing_risk_low():
    result = predict_crystal_habit(
        "PrismRisk", logP=1.5, mw_Da=300.0, hbd=2, hba=3, symmetry_index=0.4
    )
    assert result.processing_risk == "low"


# ---------------------------------------------------------------------------
# compare_polymorphs
# ---------------------------------------------------------------------------


def test_compare_polymorphs_sorted_descending():
    conditions = [
        {"logP": 4.0, "mw_Da": 280.0, "hbd": 1, "hba": 3, "symmetry_index": 0.3},  # plate 0.8
        {"logP": 1.0, "mw_Da": 300.0, "hbd": 5, "hba": 4, "symmetry_index": 0.2},  # needle 1.2
        {"logP": 1.5, "mw_Da": 300.0, "hbd": 2, "hba": 3, "symmetry_index": 0.4},  # prism 1.0
    ]
    results = compare_polymorphs("DrugPoly", conditions)
    modifiers = [r.dissolution_modifier for r in results]
    assert modifiers == sorted(modifiers, reverse=True)


def test_compare_polymorphs_returns_list():
    conditions = [
        {"logP": 2.0, "mw_Da": 300.0, "hbd": 2, "hba": 3, "symmetry_index": 0.5},
    ]
    results = compare_polymorphs("DrugSingle", conditions)
    assert isinstance(results, list)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_crystal_habit("BadDrug", logP=1.0, mw_Da=0.0, hbd=2, hba=3, symmetry_index=0.5)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_crystal_habit("BadDrug", logP=1.0, mw_Da=-100.0, hbd=2, hba=3, symmetry_index=0.5)


def test_symmetry_index_above_1_raises():
    with pytest.raises(ValueError, match="symmetry_index"):
        predict_crystal_habit("BadDrug", logP=1.0, mw_Da=300.0, hbd=2, hba=3, symmetry_index=1.5)


def test_symmetry_index_negative_raises():
    with pytest.raises(ValueError, match="symmetry_index"):
        predict_crystal_habit("BadDrug", logP=1.0, mw_Da=300.0, hbd=2, hba=3, symmetry_index=-0.1)
