"""Tests for Phase 896 — Adipocyte Cellular Drug Uptake."""

import math
import pytest

from omega_pbpk.prediction.adipocyte_uptake import (
    AdipocyteUptakeResult,
    predict_adipocyte_uptake,
    screen_adipocyte_uptake,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_frozen_dataclass():
    result = predict_adipocyte_uptake("TestDrug")
    assert isinstance(result, AdipocyteUptakeResult)


def test_result_is_frozen():
    result = predict_adipocyte_uptake("TestDrug")
    with pytest.raises((AttributeError, TypeError)):
        result.logp = 99.0  # type: ignore[misc]


def test_basic_fields():
    result = predict_adipocyte_uptake("DrugX", logp=2.0, mw_Da=350.0)
    assert result.drug_name == "DrugX"
    assert result.logp == 2.0
    assert result.mw_Da == 350.0


def test_lipid_droplet_kp_positive():
    result = predict_adipocyte_uptake("Drug", logp=2.0)
    assert result.lipid_droplet_kp >= 1.0


def test_membrane_permeability_in_range():
    result = predict_adipocyte_uptake("Drug", logp=2.0)
    assert 1e-8 <= result.membrane_permeability_cm_s <= 1e-4


def test_uptake_rate_in_range():
    result = predict_adipocyte_uptake("Drug", logp=2.0)
    assert 0.001 <= result.uptake_rate_constant_per_min <= 10.0


def test_intracellular_conc_factor_at_least_one():
    result = predict_adipocyte_uptake("Drug", logp=0.0)
    assert result.intracellular_conc_factor >= 1.0


def test_equilibration_time_positive():
    result = predict_adipocyte_uptake("Drug", logp=2.0)
    assert result.equilibration_time_min > 0.0


# ---------------------------------------------------------------------------
# Physicochemical relationships
# ---------------------------------------------------------------------------


def test_higher_logp_higher_lipid_kp():
    r_low = predict_adipocyte_uptake("Drug", logp=1.0)
    r_high = predict_adipocyte_uptake("Drug", logp=4.0)
    assert r_high.lipid_droplet_kp > r_low.lipid_droplet_kp


def test_higher_logp_higher_permeability():
    r_low = predict_adipocyte_uptake("Drug", logp=0.0)
    r_high = predict_adipocyte_uptake("Drug", logp=3.0)
    assert r_high.membrane_permeability_cm_s >= r_low.membrane_permeability_cm_s


def test_higher_logp_higher_icf():
    r_low = predict_adipocyte_uptake("Drug", logp=1.0)
    r_high = predict_adipocyte_uptake("Drug", logp=5.0)
    assert r_high.intracellular_conc_factor >= r_low.intracellular_conc_factor


def test_large_mw_reduces_permeability():
    r_small = predict_adipocyte_uptake("Drug", logp=2.0, mw_Da=200.0)
    r_large = predict_adipocyte_uptake("Drug", logp=2.0, mw_Da=700.0)
    assert r_small.membrane_permeability_cm_s >= r_large.membrane_permeability_cm_s


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_extensive_class_for_very_high_logp():
    result = predict_adipocyte_uptake("Drug", logp=6.0)
    # lipid_droplet_kp = 10^(6*0.7) = 10^4.2 → clamped to 1e4
    # icf = 1e4 * 0.3 = 3000 → clamped to 500 > 50
    assert result.adipocyte_accumulation_class == "extensive"


def test_minimal_class_for_low_logp():
    result = predict_adipocyte_uptake("Drug", logp=0.0)
    # kp = 1.0, icf = 1.0 * 0.3 = 0.3 → clamped to 1.0 ≤ 10
    assert result.adipocyte_accumulation_class == "minimal"


def test_bioaccumulation_risk_true_for_high_logp_and_high_icf():
    result = predict_adipocyte_uptake("Drug", logp=5.0)
    # logp > 3, icf should be > 20
    assert result.bioaccumulation_risk is True


def test_bioaccumulation_risk_false_for_low_logp():
    result = predict_adipocyte_uptake("Drug", logp=1.0)
    assert result.bioaccumulation_risk is False


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_bioaccumulation_risk():
    result = predict_adipocyte_uptake("Drug", logp=5.0)
    assert "Bioaccumulation risk" in result.notes


def test_notes_acceptable_for_minimal():
    result = predict_adipocyte_uptake("Drug", logp=0.0)
    assert "Acceptable" in result.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_zero_mw():
    with pytest.raises(ValueError, match="mw_Da"):
        predict_adipocyte_uptake("Drug", mw_Da=0.0)


def test_raises_invalid_ionization_type():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_adipocyte_uptake("Drug", ionization_type="zwitterion")


# ---------------------------------------------------------------------------
# Ionization types
# ---------------------------------------------------------------------------


def test_acid_ionization_type_accepted():
    result = predict_adipocyte_uptake("Drug", logp=2.0, ionization_type="acid")
    assert result.drug_name == "Drug"


def test_base_ionization_type_accepted():
    result = predict_adipocyte_uptake("Drug", logp=2.0, ionization_type="base")
    assert result.drug_name == "Drug"


# ---------------------------------------------------------------------------
# screen_adipocyte_uptake
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    candidates = [
        {"name": "DrugA", "logp": 1.0},
        {"name": "DrugB", "logp": 4.0},
        {"name": "DrugC", "logp": 2.5},
    ]
    results = screen_adipocyte_uptake(candidates)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_by_icf_descending():
    candidates = [
        {"name": "Low", "logp": 0.5},
        {"name": "High", "logp": 5.0},
        {"name": "Mid", "logp": 2.5},
    ]
    results = screen_adipocyte_uptake(candidates)
    icfs = [r.intracellular_conc_factor for r in results]
    assert icfs == sorted(icfs, reverse=True)


def test_screen_uses_default_mw():
    candidates = [{"name": "DrugA", "logp": 2.0}]
    results = screen_adipocyte_uptake(candidates)
    assert results[0].mw_Da == 350.0


def test_screen_optional_ionization_type():
    candidates = [
        {"name": "DrugA", "logp": 2.0, "ionization_type": "acid"},
    ]
    results = screen_adipocyte_uptake(candidates)
    assert len(results) == 1
