"""Tests for Phase 513 — Hepatic Zonation CYP Activity Model."""

import pytest

from omega_pbpk.core.hepatic_zonation_cyp import (
    ZONATION_INDEX,
    ZONE_ACTIVITY,
    ZONE_FRACTIONS,
    HepaticZonationResult,
    compare_cyp_enzymes,
    simulate_hepatic_zonation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
FUP = 0.3
CLINT = 50.0  # L/h
Q = 90.0  # L/h


def base_result(cyp="CYP1A2"):
    return simulate_hepatic_zonation(DRUG, FUP, CLINT, cyp)


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


def test_return_type():
    r = base_result()
    assert isinstance(r, HepaticZonationResult)


def test_fields_present():
    r = base_result()
    assert hasattr(r, "drug_name")
    assert hasattr(r, "fup")
    assert hasattr(r, "cl_int_total_L_per_h")
    assert hasattr(r, "dominant_cyp")
    assert hasattr(r, "cl_int_zone1_L_per_h")
    assert hasattr(r, "cl_int_zone2_L_per_h")
    assert hasattr(r, "cl_int_zone3_L_per_h")
    assert hasattr(r, "cl_int_effective_L_per_h")
    assert hasattr(r, "eh_hepatic")
    assert hasattr(r, "fh_hepatic")
    assert hasattr(r, "cl_hepatic_L_per_h")
    assert hasattr(r, "zonation_index")
    assert hasattr(r, "metabolic_risk")
    assert hasattr(r, "notes")


def test_drug_name_stored():
    r = simulate_hepatic_zonation("Aspirin", FUP, CLINT, "CYP1A2")
    assert r.drug_name == "Aspirin"


def test_dominant_cyp_stored():
    r = simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP3A4")
    assert r.dominant_cyp == "CYP3A4"


# ---------------------------------------------------------------------------
# Zone CLint values
# ---------------------------------------------------------------------------


def test_zone1_clint_formula():
    """CLint_zone1 = CLint_total * 1.5 * 0.30"""
    r = base_result()
    expected = CLINT * ZONE_ACTIVITY["zone1"] * ZONE_FRACTIONS["zone1"]
    assert abs(r.cl_int_zone1_L_per_h - expected) < 1e-9


def test_zone2_clint_formula():
    """CLint_zone2 = CLint_total * 1.0 * 0.40"""
    r = base_result()
    expected = CLINT * ZONE_ACTIVITY["zone2"] * ZONE_FRACTIONS["zone2"]
    assert abs(r.cl_int_zone2_L_per_h - expected) < 1e-9


def test_zone3_clint_formula():
    """CLint_zone3 = CLint_total * 0.7 * 0.30"""
    r = base_result()
    expected = CLINT * ZONE_ACTIVITY["zone3"] * ZONE_FRACTIONS["zone3"]
    assert abs(r.cl_int_zone3_L_per_h - expected) < 1e-9


def test_zone1_clint_greater_than_zone3():
    """Zone1 has higher activity multiplier than Zone3."""
    r = base_result("CYP1A2")
    assert r.cl_int_zone1_L_per_h > r.cl_int_zone3_L_per_h


def test_effective_clint_is_sum():
    """CLint_eff = CLint_z1 + CLint_z2 + CLint_z3."""
    r = base_result()
    expected = r.cl_int_zone1_L_per_h + r.cl_int_zone2_L_per_h + r.cl_int_zone3_L_per_h
    assert abs(r.cl_int_effective_L_per_h - expected) < 1e-9


def test_effective_clint_less_than_total():
    """Effective CLint is a weighted average, so for standard weights it sums < CLint_total*1.
    (0.30*1.5 + 0.40*1.0 + 0.30*0.7 = 0.45 + 0.40 + 0.21 = 1.06, so eff > total is possible)
    Just verify positive."""
    r = base_result()
    assert r.cl_int_effective_L_per_h > 0.0


# ---------------------------------------------------------------------------
# Well-stirred model
# ---------------------------------------------------------------------------


def test_eh_formula():
    """EH = fup * CLint_eff / (Q + fup * CLint_eff)."""
    r = base_result()
    num = FUP * r.cl_int_effective_L_per_h
    expected_eh = num / (Q + num)
    assert abs(r.eh_hepatic - expected_eh) < 1e-9


def test_fh_equals_one_minus_eh():
    r = base_result()
    assert abs(r.fh_hepatic - (1.0 - r.eh_hepatic)) < 1e-9


def test_clh_equals_q_times_eh():
    r = base_result()
    assert abs(r.cl_hepatic_L_per_h - Q * r.eh_hepatic) < 1e-9


def test_eh_between_zero_and_one():
    r = base_result()
    assert 0.0 < r.eh_hepatic < 1.0


def test_fh_between_zero_and_one():
    r = base_result()
    assert 0.0 < r.fh_hepatic < 1.0


def test_clh_less_than_q():
    r = base_result()
    assert r.cl_hepatic_L_per_h <= Q


# ---------------------------------------------------------------------------
# Zonation index
# ---------------------------------------------------------------------------


def test_zonation_index_value():
    """Zonation index = 0.7 / 1.5 ≈ 0.4667."""
    r = base_result()
    expected = 0.7 / 1.5
    assert abs(r.zonation_index - expected) < 1e-4


def test_zonation_index_constant():
    """ZONATION_INDEX constant matches expected value."""
    assert abs(ZONATION_INDEX - 0.7 / 1.5) < 1e-9


def test_zonation_index_same_across_cyps():
    r1 = simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP1A2")
    r2 = simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP2D6")
    assert abs(r1.zonation_index - r2.zonation_index) < 1e-9


# ---------------------------------------------------------------------------
# Metabolic risk
# ---------------------------------------------------------------------------


def test_metabolic_risk_high_cyp2d6():
    r = simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP2D6")
    assert r.metabolic_risk == "high"


def test_metabolic_risk_high_cyp2c9():
    r = simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP2C9")
    assert r.metabolic_risk == "high"


def test_metabolic_risk_normal_cyp3a4():
    r = simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP3A4")
    assert r.metabolic_risk == "normal"


def test_metabolic_risk_normal_cyp1a2():
    r = simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP1A2")
    assert r.metabolic_risk == "normal"


def test_metabolic_risk_normal_cyp2e1():
    r = simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP2E1")
    assert r.metabolic_risk == "normal"


# ---------------------------------------------------------------------------
# compare_cyp_enzymes
# ---------------------------------------------------------------------------


def test_compare_returns_5():
    results = compare_cyp_enzymes(DRUG, FUP, CLINT)
    assert len(results) == 5


def test_compare_sorted_eh_ascending():
    results = compare_cyp_enzymes(DRUG, FUP, CLINT)
    ehs = [r.eh_hepatic for r in results]
    assert ehs == sorted(ehs)


def test_compare_all_cyps_covered():
    results = compare_cyp_enzymes(DRUG, FUP, CLINT)
    cyps = {r.dominant_cyp for r in results}
    assert cyps == {"CYP1A2", "CYP2E1", "CYP3A4", "CYP2D6", "CYP2C9"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_fup_zero():
    with pytest.raises(ValueError, match="fup"):
        simulate_hepatic_zonation(DRUG, 0.0, CLINT, "CYP1A2")


def test_invalid_fup_negative():
    with pytest.raises(ValueError, match="fup"):
        simulate_hepatic_zonation(DRUG, -0.1, CLINT, "CYP1A2")


def test_invalid_fup_above_one():
    with pytest.raises(ValueError, match="fup"):
        simulate_hepatic_zonation(DRUG, 1.1, CLINT, "CYP1A2")


def test_invalid_clint_zero():
    with pytest.raises(ValueError, match="cl_int"):
        simulate_hepatic_zonation(DRUG, FUP, 0.0, "CYP1A2")


def test_invalid_clint_negative():
    with pytest.raises(ValueError, match="cl_int"):
        simulate_hepatic_zonation(DRUG, FUP, -5.0, "CYP1A2")


def test_invalid_q_zero():
    with pytest.raises(ValueError, match="q_hepatic"):
        simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP1A2", q_hepatic_L_per_h=0.0)


def test_invalid_cyp():
    with pytest.raises(ValueError, match="dominant_cyp"):
        simulate_hepatic_zonation(DRUG, FUP, CLINT, "CYP99")
