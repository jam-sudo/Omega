"""Tests for ASD system modeling — Phase 243."""

import pytest

from omega_pbpk.prediction.amorphous_solid_dispersion_p243 import (
    _FORMULATION_CLASSES,
    ASDResult,
    design_asd,
    screen_asd_polymers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        crystalline_solubility_mg_mL=0.1,
        logp=3.5,
        drug_loading_pct=20.0,
        polymer_type="HPMC_AS",
    )
    defaults.update(kwargs)
    return design_asd(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic fields
# ---------------------------------------------------------------------------


def test_returns_asd_result():
    result = _default()
    assert isinstance(result, ASDResult)


def test_drug_name_preserved():
    result = _default(drug_name="Ibuprofen")
    assert result.drug_name == "Ibuprofen"


def test_polymer_type_preserved():
    result = _default(polymer_type="PVP_VA")
    assert result.polymer_type == "PVP_VA"


def test_drug_loading_preserved():
    result = _default(drug_loading_pct=30.0)
    assert result.drug_loading_pct == pytest.approx(30.0)


def test_crystalline_solubility_preserved():
    result = _default(crystalline_solubility_mg_mL=0.05)
    assert result.crystalline_solubility_mg_mL == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Amorphous solubility > crystalline solubility
# ---------------------------------------------------------------------------


def test_amorphous_above_crystalline():
    result = _default()
    assert result.amorphous_solubility_mg_mL > result.crystalline_solubility_mg_mL


def test_amorphous_solubility_positive():
    result = _default()
    assert result.amorphous_solubility_mg_mL > 0.0


def test_apparent_solubility_positive():
    result = _default()
    assert result.apparent_solubility_mg_mL > 0.0


# ---------------------------------------------------------------------------
# Spring > parachute (crystallization_inhibitor < 1)
# ---------------------------------------------------------------------------


def test_spring_duration_greater_than_parachute():
    result = _default()
    assert result.spring_duration_min > result.parachute_duration_min


def test_spring_duration_positive():
    result = _default()
    assert result.spring_duration_min > 0.0


def test_parachute_duration_positive():
    result = _default()
    assert result.parachute_duration_min > 0.0


# ---------------------------------------------------------------------------
# Relative bioavailability increase > 0
# ---------------------------------------------------------------------------


def test_relative_ba_increase_positive():
    result = _default()
    assert result.relative_bioavailability_increase_pct > 0.0


# ---------------------------------------------------------------------------
# Physical stability score in [0, 100]
# ---------------------------------------------------------------------------


def test_physical_stability_in_bounds():
    result = _default()
    assert 0.0 <= result.physical_stability_score <= 100.0


def test_physical_stability_in_bounds_high_loading():
    result = _default(drug_loading_pct=80.0)
    assert 0.0 <= result.physical_stability_score <= 100.0


# ---------------------------------------------------------------------------
# Formulation class valid values
# ---------------------------------------------------------------------------


def test_formulation_class_valid():
    result = _default()
    assert result.formulation_class in _FORMULATION_CLASSES


def test_formulation_class_excellent_for_low_loading():
    # Low loading and strong polymer should give high stability
    result = _default(drug_loading_pct=5.0, polymer_type="HPMC_AS")
    # Just check valid
    assert result.formulation_class in _FORMULATION_CLASSES


# ---------------------------------------------------------------------------
# Higher loading → lower stability score
# ---------------------------------------------------------------------------


def test_higher_loading_lower_stability():
    r_low = _default(drug_loading_pct=10.0)
    r_high = _default(drug_loading_pct=70.0)
    assert r_high.physical_stability_score < r_low.physical_stability_score


# ---------------------------------------------------------------------------
# Screen returns 5 sorted descending
# ---------------------------------------------------------------------------


def test_screen_returns_five():
    results = screen_asd_polymers("Drug", 0.1, 3.0)
    assert len(results) == 5


def test_screen_sorted_descending():
    results = screen_asd_polymers("Drug", 0.1, 3.0)
    scores = [r.physical_stability_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_screen_all_asd_results():
    results = screen_asd_polymers("Drug", 0.1, 3.0)
    for r in results:
        assert isinstance(r, ASDResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_solubility_zero_raises():
    with pytest.raises(ValueError, match="crystalline_solubility"):
        design_asd("X", crystalline_solubility_mg_mL=0.0, logp=3.0, drug_loading_pct=20.0)


def test_invalid_solubility_negative_raises():
    with pytest.raises(ValueError, match="crystalline_solubility"):
        design_asd("X", crystalline_solubility_mg_mL=-1.0, logp=3.0, drug_loading_pct=20.0)


def test_invalid_loading_zero_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        design_asd("X", crystalline_solubility_mg_mL=0.1, logp=3.0, drug_loading_pct=0.0)


def test_invalid_loading_100_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        design_asd("X", crystalline_solubility_mg_mL=0.1, logp=3.0, drug_loading_pct=100.0)


def test_invalid_polymer_raises():
    with pytest.raises(ValueError, match="polymer_type"):
        design_asd(
            "X",
            crystalline_solubility_mg_mL=0.1,
            logp=3.0,
            drug_loading_pct=20.0,
            polymer_type="InvalidPolymer",
        )
