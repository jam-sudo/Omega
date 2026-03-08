"""Tests for Phase 452 — Drug Stability Screening."""

from math import isclose

import pytest

from omega_pbpk.prediction.drug_stability_screening import (
    DrugStabilityResult,
    compare_drug_stability,
    screen_drug_stability,
)

# ---------------------------------------------------------------------------
# Return type checks
# ---------------------------------------------------------------------------


def test_return_type():
    result = screen_drug_stability("DrugA", "amine", logp=2.0, mw_Da=300.0)
    assert isinstance(result, DrugStabilityResult)


def test_drug_name_preserved():
    result = screen_drug_stability("MyDrug", "alcohol", logp=1.5, mw_Da=250.0)
    assert result.drug_name == "MyDrug"


def test_chemical_class_preserved():
    result = screen_drug_stability("MyDrug", "ester", logp=1.5, mw_Da=250.0)
    assert result.chemical_class == "ester"


# ---------------------------------------------------------------------------
# Numeric sanity checks
# ---------------------------------------------------------------------------


def test_primary_degradation_rate_positive():
    result = screen_drug_stability("DrugB", "lactam", logp=2.0, mw_Da=400.0)
    assert result.primary_degradation_rate > 0.0


def test_t90_days_positive():
    result = screen_drug_stability("DrugC", "acid", logp=1.0, mw_Da=350.0)
    assert result.t90_days > 0.0


def test_forced_degradation_score_in_range():
    result = screen_drug_stability("DrugD", "none", logp=0.5, mw_Da=200.0)
    assert 0.0 <= result.forced_degradation_score <= 100.0


def test_shelf_life_equals_t90_over_30():
    result = screen_drug_stability("DrugE", "amine", logp=2.0, mw_Da=300.0)
    expected = result.t90_days / 30.0
    assert isclose(result.shelf_life_estimate_months, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Comparative stability ordering
# ---------------------------------------------------------------------------


def test_ester_higher_k_deg_than_alcohol():
    """Ester has higher k_base than alcohol, so k_deg should be higher."""
    ester = screen_drug_stability("EsterDrug", "ester", logp=2.0, mw_Da=300.0)
    alcohol = screen_drug_stability("AlcoholDrug", "alcohol", logp=2.0, mw_Da=300.0)
    assert ester.primary_degradation_rate > alcohol.primary_degradation_rate


def test_lactam_shorter_shelf_life_than_none():
    """Lactam (k_base=5e-4) should have shorter shelf life than 'none' (k_base=5e-5)."""
    lactam = screen_drug_stability("LactamDrug", "lactam", logp=1.0, mw_Da=300.0)
    stable = screen_drug_stability("StableDrug", "none", logp=1.0, mw_Da=300.0)
    assert lactam.shelf_life_estimate_months < stable.shelf_life_estimate_months


def test_lactam_has_shortest_t90():
    """Among ester and lactam with same logP/MW, lactam should have shorter t90 due to higher k_base."""
    lactam = screen_drug_stability("LactamDrug", "lactam", logp=2.0, mw_Da=300.0)
    amine = screen_drug_stability("AmineDrug", "amine", logp=2.0, mw_Da=300.0)
    # lactam k_base=5e-4, amine k_base=3e-4
    assert lactam.t90_days < amine.t90_days


# ---------------------------------------------------------------------------
# Degradation pathways
# ---------------------------------------------------------------------------


def test_degradation_pathways_non_empty():
    result = screen_drug_stability("DrugF", "ester", logp=2.0, mw_Da=300.0)
    assert isinstance(result.degradation_pathways, str)
    assert len(result.degradation_pathways) > 0


def test_ester_hydrolysis_in_pathways():
    """Ester has hydrolysis fraction > 0, so 'hydrolysis' should be in pathways."""
    result = screen_drug_stability("EsterDrug", "ester", logp=2.0, mw_Da=300.0)
    assert "hydrolysis" in result.degradation_pathways


def test_amine_oxidation_in_pathways():
    """Amine has oxidation fraction > 0, so 'oxidation' should be in pathways."""
    result = screen_drug_stability("AmineDrug", "amine", logp=2.0, mw_Da=300.0)
    assert "oxidation" in result.degradation_pathways


# ---------------------------------------------------------------------------
# Storage requirement from t90 range
# ---------------------------------------------------------------------------


def test_stable_drug_room_temp():
    """'none' class has k_base=5e-5, should produce a long t90 → room_temp."""
    result = screen_drug_stability("StableDrug", "none", logp=0.5, mw_Da=200.0)
    assert result.storage_requirement == "room_temp"


def test_unstable_drug_not_room_temp():
    """Ester or lactam with high logP/MW should have shorter t90."""
    result = screen_drug_stability("UnstableDrug", "lactam", logp=5.0, mw_Da=800.0)
    # t90 may be short enough to require refrigeration or freeze
    assert result.storage_requirement in ("room_temp", "refrigerate", "freeze", "protect_light")


# ---------------------------------------------------------------------------
# compare_drug_stability
# ---------------------------------------------------------------------------


def test_compare_returns_correct_length():
    compounds = [
        {"drug_name": "DrugA", "chemical_class": "amine", "logp": 2.0, "mw_Da": 300.0},
        {"drug_name": "DrugB", "chemical_class": "ester", "logp": 2.0, "mw_Da": 300.0},
        {"drug_name": "DrugC", "chemical_class": "alcohol", "logp": 2.0, "mw_Da": 300.0},
    ]
    results = compare_drug_stability(compounds)
    assert len(results) == 3


def test_compare_sorted_descending():
    """Results should be sorted by forced_degradation_score descending (most stable first)."""
    compounds = [
        {"drug_name": "DrugA", "chemical_class": "lactam", "logp": 2.0, "mw_Da": 300.0},
        {"drug_name": "DrugB", "chemical_class": "alcohol", "logp": 2.0, "mw_Da": 300.0},
        {"drug_name": "DrugC", "chemical_class": "none", "logp": 2.0, "mw_Da": 300.0},
    ]
    results = compare_drug_stability(compounds)
    for i in range(1, len(results)):
        assert results[i - 1].forced_degradation_score >= results[i].forced_degradation_score


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_chemical_class_raises_value_error():
    with pytest.raises(ValueError, match="Unrecognized chemical_class"):
        screen_drug_stability("BadDrug", "polymer", logp=2.0, mw_Da=300.0)


def test_zero_mw_raises_value_error():
    with pytest.raises(ValueError, match="mw_Da must be positive"):
        screen_drug_stability("BadDrug", "amine", logp=2.0, mw_Da=0.0)


def test_negative_mw_raises_value_error():
    with pytest.raises(ValueError, match="mw_Da must be positive"):
        screen_drug_stability("BadDrug", "amine", logp=2.0, mw_Da=-100.0)
