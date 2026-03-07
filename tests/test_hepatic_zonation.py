"""Tests for Phase 1077 -- Hepatic Zone-Specific Metabolism."""

import pytest
from omega_pbpk.core.hepatic_zonation import (
    HepaticZonationResult,
    compare_cyp_enzymes,
    simulate_hepatic_zonation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_result(cyp: str = "CYP3A4", **kwargs) -> HepaticZonationResult:
    return simulate_hepatic_zonation(
        drug_name="TestDrug",
        cyp_enzyme=cyp,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_return_type():
    result = default_result()
    assert isinstance(result, HepaticZonationResult)


def test_frozen_dataclass():
    result = default_result()
    with pytest.raises((AttributeError, TypeError)):
        result.e_total = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Zone extraction relationships
# ---------------------------------------------------------------------------

def test_cyp3a4_pericentral_zone3_gt_zone1():
    """CYP3A4 has higher activity in zone 3, so e_zone3 > e_zone1."""
    result = default_result("CYP3A4")
    assert result.e_zone3 > result.e_zone1


def test_cyp2d6_periportal_zone1_gt_zone3():
    """CYP2D6 has higher activity in zone 1, so e_zone1 > e_zone3."""
    result = default_result("CYP2D6")
    assert result.e_zone1 > result.e_zone3


def test_cyp2e1_pericentral_zone3_gt_zone1():
    """CYP2E1 is pericentral dominant."""
    result = default_result("CYP2E1")
    assert result.e_zone3 > result.e_zone1


def test_cyp2c9_uniform_zones_approx_equal():
    """CYP2C9 has uniform expression, all zones should be approximately equal."""
    result = default_result("CYP2C9")
    assert abs(result.e_zone1 - result.e_zone3) < 1e-9


# ---------------------------------------------------------------------------
# Total extraction
# ---------------------------------------------------------------------------

def test_e_total_less_than_one():
    result = default_result()
    assert result.e_total < 1.0


def test_e_total_positive():
    result = default_result()
    assert result.e_total > 0.0


def test_e_total_gte_each_zone():
    """Total extraction must be >= any single zone extraction."""
    result = default_result()
    assert result.e_total >= result.e_zone1
    assert result.e_total >= result.e_zone2
    assert result.e_total >= result.e_zone3


def test_e_total_formula():
    """e_total == 1 - (1-e1)*(1-e2)*(1-e3)."""
    result = default_result()
    expected = 1.0 - (1.0 - result.e_zone1) * (1.0 - result.e_zone2) * (1.0 - result.e_zone3)
    assert abs(result.e_total - expected) < 1e-10


# ---------------------------------------------------------------------------
# CL hepatic
# ---------------------------------------------------------------------------

def test_cl_hepatic_positive():
    result = default_result()
    assert result.cl_hepatic_L_per_h > 0.0


def test_cl_hepatic_equals_q_times_e_total():
    result = default_result()
    expected = result.q_portal_L_per_h * result.e_total
    assert abs(result.cl_hepatic_L_per_h - expected) < 1e-8


# ---------------------------------------------------------------------------
# First-pass effect classification
# ---------------------------------------------------------------------------

def test_first_pass_effect_valid_values():
    result = default_result()
    assert result.first_pass_effect in {"low", "intermediate", "high"}


def test_high_clint_gives_high_first_pass():
    """Very high CLint should give high extraction."""
    result = simulate_hepatic_zonation(
        drug_name="HighCL",
        cyp_enzyme="CYP3A4",
        clint_total_mL_per_min_per_g=500.0,
        fu_plasma=1.0,
    )
    assert result.first_pass_effect == "high"


def test_low_clint_gives_low_first_pass():
    """Very low CLint should give low extraction."""
    result = simulate_hepatic_zonation(
        drug_name="LowCL",
        cyp_enzyme="CYP3A4",
        clint_total_mL_per_min_per_g=0.01,
        fu_plasma=0.01,
    )
    assert result.first_pass_effect == "low"


# ---------------------------------------------------------------------------
# Zone sensitivity
# ---------------------------------------------------------------------------

def test_zone_sensitivity_valid_values():
    result = default_result()
    assert result.zone_sensitivity in {"periportal", "pericentral"}


def test_cyp3a4_zone_sensitivity_pericentral():
    result = default_result("CYP3A4")
    assert result.zone_sensitivity == "pericentral"


def test_cyp2d6_zone_sensitivity_periportal():
    result = default_result("CYP2D6")
    assert result.zone_sensitivity == "periportal"


def test_cyp2e1_zone_sensitivity_pericentral():
    result = default_result("CYP2E1")
    assert result.zone_sensitivity == "pericentral"


def test_cyp1a2_zone_sensitivity_pericentral():
    result = default_result("CYP1A2")
    assert result.zone_sensitivity == "pericentral"


# ---------------------------------------------------------------------------
# compare_cyp_enzymes
# ---------------------------------------------------------------------------

def test_compare_cyp_enzymes_returns_five():
    results = compare_cyp_enzymes("TestDrug")
    assert len(results) == 5


def test_compare_cyp_enzymes_sorted_descending():
    results = compare_cyp_enzymes("TestDrug")
    for i in range(len(results) - 1):
        assert results[i].cl_hepatic_L_per_h >= results[i + 1].cl_hepatic_L_per_h


def test_compare_cyp_enzymes_all_results_valid():
    results = compare_cyp_enzymes("TestDrug")
    for r in results:
        assert isinstance(r, HepaticZonationResult)
        assert r.first_pass_effect in {"low", "intermediate", "high"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_cyp_raises():
    with pytest.raises(ValueError, match="Unsupported CYP"):
        simulate_hepatic_zonation("Drug", "CYP99X")


def test_invalid_clint_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_zonation("Drug", "CYP3A4", clint_total_mL_per_min_per_g=-1.0)


def test_invalid_q_portal_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_zonation("Drug", "CYP3A4", q_portal_L_per_h=0.0)


def test_invalid_fu_plasma_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_zonation("Drug", "CYP3A4", fu_plasma=0.0)


def test_invalid_fu_plasma_gt_one_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_zonation("Drug", "CYP3A4", fu_plasma=1.5)


def test_invalid_fu_mic_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_zonation("Drug", "CYP3A4", fu_mic=0.0)


def test_invalid_liver_weight_raises():
    with pytest.raises(ValueError):
        simulate_hepatic_zonation("Drug", "CYP3A4", liver_weight_g=-100.0)


# ---------------------------------------------------------------------------
# Notes / strings
# ---------------------------------------------------------------------------

def test_notes_nonempty():
    result = default_result()
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = simulate_hepatic_zonation("Midazolam", "CYP3A4")
    assert result.drug_name == "Midazolam"


def test_fu_mic_passed_through():
    result = simulate_hepatic_zonation("Drug", "CYP3A4", fu_mic=0.3)
    assert result.fu_mic == pytest.approx(0.3)
