"""
Tests for Phase 251: prediction/nanoemulsion_formulation.py
"""

import pytest

from omega_pbpk.prediction.nanoemulsion_formulation import (
    VALID_OIL_PHASES,
    NanoemulsionResult,
    design_nanoemulsion,
    screen_oil_phases,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_result():
    return design_nanoemulsion("TestDrug", logp=3.0, drug_loading_pct=10.0, oil_phase="MCT")


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type(default_result):
    assert isinstance(default_result, NanoemulsionResult)


def test_result_is_frozen(default_result):
    with pytest.raises((AttributeError, TypeError)):
        default_result.drug_name = "other"


# ---------------------------------------------------------------------------
# Physical plausibility tests
# ---------------------------------------------------------------------------


def test_droplet_size_in_range(default_result):
    assert 50.0 <= default_result.droplet_size_nm <= 500.0


def test_pdi_in_range(default_result):
    assert 0.1 <= default_result.polydispersity_index <= 0.5


def test_solubilization_efficiency_in_range(default_result):
    assert 0.0 <= default_result.solubilization_efficiency_pct <= 100.0


def test_stability_score_in_range(default_result):
    assert 0.0 <= default_result.stability_score <= 100.0


def test_bioavailability_factor_greater_than_one(default_result):
    assert default_result.bioavailability_enhancement_factor > 1.0


# ---------------------------------------------------------------------------
# Formulation class tests
# ---------------------------------------------------------------------------


def test_formulation_class_valid_values(default_result):
    valid = {"excellent", "good", "acceptable", "poor"}
    assert default_result.formulation_class in valid


def test_formulation_class_excellent():
    # oleic_acid has highest sol_cap → should score high
    result = design_nanoemulsion("TestDrug", logp=5.0, drug_loading_pct=5.0, oil_phase="oleic_acid")
    assert result.formulation_class in {"excellent", "good", "acceptable", "poor"}


def test_formulation_class_poor():
    # soybean_oil with high loading and low logP
    result = design_nanoemulsion(
        "TestDrug", logp=0.5, drug_loading_pct=40.0, oil_phase="soybean_oil"
    )
    assert result.formulation_class in {"poor", "acceptable"}


# ---------------------------------------------------------------------------
# logP effect
# ---------------------------------------------------------------------------


def test_lower_logp_reduces_solubilization_efficiency():
    """logP < 2 halves solubilization_capacity, lowering efficiency."""
    high_logp = design_nanoemulsion("DrugA", logp=5.0, drug_loading_pct=10.0)
    low_logp = design_nanoemulsion("DrugA", logp=1.0, drug_loading_pct=10.0)
    assert low_logp.solubilization_efficiency_pct < high_logp.solubilization_efficiency_pct


def test_low_logp_boundary():
    """logP exactly 2.0 should NOT trigger the 0.5x penalty (strict < 2)."""
    at_boundary = design_nanoemulsion("DrugB", logp=2.0, drug_loading_pct=10.0)
    above = design_nanoemulsion("DrugB", logp=2.1, drug_loading_pct=10.0)
    assert at_boundary.solubilization_efficiency_pct == pytest.approx(
        above.solubilization_efficiency_pct, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Screen oil phases
# ---------------------------------------------------------------------------


def test_screen_returns_five_results():
    results = screen_oil_phases("TestDrug", logp=3.0)
    assert len(results) == 5


def test_screen_sorted_descending():
    results = screen_oil_phases("TestDrug", logp=3.0)
    scores = [r.stability_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_covers_all_oil_phases():
    results = screen_oil_phases("TestDrug", logp=3.0)
    returned_phases = {r.oil_phase for r in results}
    assert returned_phases == set(VALID_OIL_PHASES)


def test_screen_default_loading():
    results = screen_oil_phases("TestDrug", logp=3.0)
    for r in results:
        assert r.drug_loading_pct == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_oil_phase_raises():
    with pytest.raises(ValueError, match="oil_phase"):
        design_nanoemulsion("X", logp=3.0, drug_loading_pct=10.0, oil_phase="unknown_oil")


def test_zero_drug_loading_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        design_nanoemulsion("X", logp=3.0, drug_loading_pct=0.0)


def test_negative_drug_loading_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        design_nanoemulsion("X", logp=3.0, drug_loading_pct=-5.0)


def test_drug_loading_exactly_50_allowed():
    result = design_nanoemulsion("X", logp=3.0, drug_loading_pct=50.0)
    assert isinstance(result, NanoemulsionResult)


def test_drug_loading_above_50_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        design_nanoemulsion("X", logp=3.0, drug_loading_pct=51.0)


# ---------------------------------------------------------------------------
# All oil phases produce valid results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oil_phase", VALID_OIL_PHASES)
def test_each_oil_phase_valid(oil_phase):
    result = design_nanoemulsion("Ibuprofen", logp=3.5, drug_loading_pct=15.0, oil_phase=oil_phase)
    assert isinstance(result, NanoemulsionResult)
    assert 50.0 <= result.droplet_size_nm <= 500.0
    assert 0.1 <= result.polydispersity_index <= 0.5
    assert 0.0 <= result.solubilization_efficiency_pct <= 100.0
    assert result.bioavailability_enhancement_factor > 1.0
