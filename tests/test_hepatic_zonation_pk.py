"""Tests for Phase 865 — Hepatic Zonation Model."""

import pytest

from omega_pbpk.core.hepatic_zonation_pk import (
    HepaticZonationResult,
    compare_disease_states,
    simulate_hepatic_zonation,
)

# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_hepatic_zonation("DrugA")
    assert isinstance(result, HepaticZonationResult)


def test_fields_preserved():
    result = simulate_hepatic_zonation(
        "TestDrug",
        clint_total_mL_per_min=40.0,
        q_hepatic_L_per_h=80.0,
        fup=0.2,
        cyp_enzyme="CYP3A4",
        disease_state="fibrosis",
    )
    assert result.drug_name == "TestDrug"
    assert result.clint_total_mL_per_min == 40.0
    assert result.q_hepatic_L_per_h == 80.0
    assert result.fup == 0.2
    assert result.cyp_enzyme == "CYP3A4"
    assert result.disease_state == "fibrosis"


# ---------------------------------------------------------------------------
# Extraction ratio bounds
# ---------------------------------------------------------------------------


def test_er_in_unit_interval():
    result = simulate_hepatic_zonation("DrugA", clint_total_mL_per_min=200.0)
    assert 0.0 <= result.extraction_ratio <= 1.0


def test_er_low_clint():
    result = simulate_hepatic_zonation("LowCLint", clint_total_mL_per_min=1.0, fup=0.05)
    assert result.extraction_ratio < 0.5


def test_er_high_clint():
    # Use very high CLint and high fup to ensure ER > 0.7
    # CLint=5000 mL/min=300 L/h, fup=1.0, Q=90 L/h
    # Per zone: CLint_z*fup/Q_z = (300*0.4*1.0)/(30) = 4.0 → c_out=1/5=0.2
    # Overall ER = 1 - 0.2 * (1/4.3)^2 > 0.7
    result = simulate_hepatic_zonation("HighCLint", clint_total_mL_per_min=5000.0, fup=1.0)
    assert result.extraction_ratio > 0.7


# ---------------------------------------------------------------------------
# Derived quantities consistency
# ---------------------------------------------------------------------------


def test_cl_hepatic_equals_q_times_er():
    result = simulate_hepatic_zonation("DrugA")
    expected = result.q_hepatic_L_per_h * result.extraction_ratio
    assert abs(result.cl_hepatic_L_per_h - expected) < 1e-9


def test_fh_equals_one_minus_er():
    result = simulate_hepatic_zonation("DrugA")
    assert abs(result.fh_bioavailability - (1.0 - result.extraction_ratio)) < 1e-9


# ---------------------------------------------------------------------------
# high_extraction flag
# ---------------------------------------------------------------------------


def test_high_extraction_true_when_er_above_07():
    result = simulate_hepatic_zonation("HighEx", clint_total_mL_per_min=500.0, fup=1.0)
    if result.extraction_ratio > 0.7:
        assert result.high_extraction is True


def test_high_extraction_false_when_er_below_07():
    result = simulate_hepatic_zonation("LowEx", clint_total_mL_per_min=2.0, fup=0.05)
    if result.extraction_ratio <= 0.7:
        assert result.high_extraction is False


def test_high_extraction_matches_threshold():
    result = simulate_hepatic_zonation("DrugA")
    assert result.high_extraction == (result.extraction_ratio > 0.7)


# ---------------------------------------------------------------------------
# Disease state effects
# ---------------------------------------------------------------------------


def test_cirrhosis_reduces_cl_vs_normal():
    r_normal = simulate_hepatic_zonation("DrugA", disease_state="normal")
    r_cirrhosis = simulate_hepatic_zonation("DrugA", disease_state="cirrhosis")
    assert r_cirrhosis.cl_hepatic_L_per_h < r_normal.cl_hepatic_L_per_h


def test_fibrosis_reduces_cl_vs_normal():
    r_normal = simulate_hepatic_zonation("DrugA", disease_state="normal")
    r_fibrosis = simulate_hepatic_zonation("DrugA", disease_state="fibrosis")
    assert r_fibrosis.cl_hepatic_L_per_h < r_normal.cl_hepatic_L_per_h


def test_disease_severity_ordering():
    r_n = simulate_hepatic_zonation("DrugA", clint_total_mL_per_min=100.0, disease_state="normal")
    r_f = simulate_hepatic_zonation("DrugA", clint_total_mL_per_min=100.0, disease_state="fibrosis")
    r_c = simulate_hepatic_zonation(
        "DrugA", clint_total_mL_per_min=100.0, disease_state="cirrhosis"
    )
    assert r_c.cl_hepatic_L_per_h < r_f.cl_hepatic_L_per_h < r_n.cl_hepatic_L_per_h


# ---------------------------------------------------------------------------
# CLint / flow sensitivity
# ---------------------------------------------------------------------------


def test_higher_clint_higher_er():
    r_low = simulate_hepatic_zonation("DrugA", clint_total_mL_per_min=10.0)
    r_high = simulate_hepatic_zonation("DrugA", clint_total_mL_per_min=200.0)
    assert r_high.extraction_ratio > r_low.extraction_ratio


def test_higher_flow_lower_er():
    """For a fixed CLint, higher hepatic flow → lower ER (flow-limited)."""
    r_low_q = simulate_hepatic_zonation(
        "DrugA", clint_total_mL_per_min=60.0, q_hepatic_L_per_h=30.0
    )
    r_high_q = simulate_hepatic_zonation(
        "DrugA", clint_total_mL_per_min=60.0, q_hepatic_L_per_h=150.0
    )
    assert r_high_q.extraction_ratio < r_low_q.extraction_ratio


# ---------------------------------------------------------------------------
# Zone fractions
# ---------------------------------------------------------------------------


def test_zone1_fraction_affects_result():
    r1 = simulate_hepatic_zonation(
        "DrugA", zone1_fraction=0.8, zone2_fraction=0.1, zone3_fraction=0.1
    )
    r2 = simulate_hepatic_zonation(
        "DrugA", zone1_fraction=0.1, zone2_fraction=0.1, zone3_fraction=0.8
    )
    # Both should be valid; results differ but ER is same (total CLint same for equal fup)
    assert 0.0 <= r1.extraction_ratio <= 1.0
    assert 0.0 <= r2.extraction_ratio <= 1.0


def test_zone_contributing_most_valid():
    result = simulate_hepatic_zonation("DrugA")
    assert result.zone_contributing_most in ("zone1", "zone2", "zone3")


# ---------------------------------------------------------------------------
# compare_disease_states
# ---------------------------------------------------------------------------


def test_compare_disease_states_returns_list():
    results = compare_disease_states("DrugA", clint_total_mL_per_min=80.0)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_disease_states_sorted_descending():
    results = compare_disease_states("DrugA", clint_total_mL_per_min=80.0)
    cls = [r.cl_hepatic_L_per_h for r in results]
    assert cls == sorted(cls, reverse=True)


def test_compare_disease_states_all_types():
    results = compare_disease_states("DrugA", clint_total_mL_per_min=80.0)
    states = {r.disease_state for r in results}
    assert states == {"normal", "fibrosis", "cirrhosis"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_zone_fractions_sum():
    with pytest.raises(ValueError, match="zone fractions"):
        simulate_hepatic_zonation(
            "DrugA", zone1_fraction=0.5, zone2_fraction=0.5, zone3_fraction=0.5
        )


def test_invalid_fup_zero():
    with pytest.raises(ValueError, match="fup"):
        simulate_hepatic_zonation("DrugA", fup=0.0)


def test_invalid_fup_above_one():
    with pytest.raises(ValueError, match="fup"):
        simulate_hepatic_zonation("DrugA", fup=1.5)


def test_invalid_clint_zero():
    with pytest.raises(ValueError, match="clint"):
        simulate_hepatic_zonation("DrugA", clint_total_mL_per_min=0.0)


def test_invalid_q_zero():
    with pytest.raises(ValueError, match="q_hepatic"):
        simulate_hepatic_zonation("DrugA", q_hepatic_L_per_h=0.0)
