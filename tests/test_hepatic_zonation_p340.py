"""Tests for Phase 340 — Hepatic Zonation Model."""

import math

import pytest

from omega_pbpk.core.hepatic_zonation_p340 import (
    HepaticZonationResult,
    simulate_hepatic_zonation,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_total_L_per_h=20.0,
        vd_L=50.0,
        cyp_isoform="CYP3A4",
        t_end_h=8.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_hepatic_zonation(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, HepaticZonationResult)


def test_lists_are_lists():
    result = make_result()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.c_zone1_mg_L, list)
    assert isinstance(result.c_zone2_mg_L, list)
    assert isinstance(result.c_zone3_mg_L, list)


# ---------------------------------------------------------------------------
# Array length consistency
# ---------------------------------------------------------------------------


def test_array_lengths_consistent():
    result = make_result()
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_zone1_mg_L) == n
    assert len(result.c_zone2_mg_L) == n
    assert len(result.c_zone3_mg_L) == n


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------


def test_plasma_starts_at_dose_over_vd():
    result = make_result(dose_mg=100.0, vd_L=50.0)
    c0_expected = 100.0 / 50.0  # 2.0 mg/L
    assert math.isclose(result.c_plasma_mg_L[0], c0_expected, rel_tol=1e-6)


def test_plasma_decreases_over_time():
    result = make_result()
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


# ---------------------------------------------------------------------------
# Zone concentration ordering
# ---------------------------------------------------------------------------


def test_zone_concentrations_positive():
    result = make_result()
    assert all(c >= 0 for c in result.c_zone1_mg_L)
    assert all(c >= 0 for c in result.c_zone2_mg_L)
    assert all(c >= 0 for c in result.c_zone3_mg_L)


def test_zone1_ge_zone2_ge_zone3():
    """Zone concentrations decrease through sequential extraction."""
    result = make_result()
    for z1, z2, z3 in zip(result.c_zone1_mg_L, result.c_zone2_mg_L, result.c_zone3_mg_L):
        assert z1 >= z2 >= z3


# ---------------------------------------------------------------------------
# CYP-specific metabolism fraction
# ---------------------------------------------------------------------------


def test_cyp3a4_zone3_highest_metabolism():
    """CYP3A4: zone3 has highest extraction → highest metabolism %."""
    result = make_result(cyp_isoform="CYP3A4")
    assert result.zone3_metabolism_pct > result.zone1_metabolism_pct
    assert result.zone3_metabolism_pct > result.zone2_metabolism_pct


def test_cyp1a2_zone1_highest_metabolism():
    """CYP1A2: zone1 has highest extraction → highest metabolism %."""
    result = make_result(cyp_isoform="CYP1A2")
    assert result.zone1_metabolism_pct > result.zone2_metabolism_pct
    assert result.zone1_metabolism_pct > result.zone3_metabolism_pct


def test_cyp2e1_zone3_highest_metabolism():
    """CYP2E1: zone3 has highest extraction."""
    result = make_result(cyp_isoform="CYP2E1")
    assert result.zone3_metabolism_pct > result.zone1_metabolism_pct


# ---------------------------------------------------------------------------
# Zone metabolism sum
# ---------------------------------------------------------------------------


def test_zone_metabolism_pct_sum_100():
    """Zone metabolism percentages should sum to ~100."""
    for isoform in ("CYP3A4", "CYP2E1", "CYP1A2"):
        result = make_result(cyp_isoform=isoform)
        total = (
            result.zone1_metabolism_pct + result.zone2_metabolism_pct + result.zone3_metabolism_pct
        )
        assert math.isclose(total, 100.0, rel_tol=1e-4), f"{isoform}: sum={total}"


# ---------------------------------------------------------------------------
# Overall hepatic extraction
# ---------------------------------------------------------------------------


def test_overall_hepatic_extraction_in_0_1():
    result = make_result()
    assert 0.0 <= result.overall_hepatic_extraction < 1.0


def test_higher_cl_higher_extraction():
    r_low_cl = make_result(cl_total_L_per_h=10.0)
    r_high_cl = make_result(cl_total_L_per_h=80.0)
    assert r_high_cl.overall_hepatic_extraction > r_low_cl.overall_hepatic_extraction


# ---------------------------------------------------------------------------
# AUC and half-life
# ---------------------------------------------------------------------------


def test_auc_positive():
    result = make_result()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_t_half_positive():
    result = make_result()
    assert result.t_half_h > 0.0


def test_t_half_formula():
    """t_half = ln(2) / ke = ln(2) * Vd / CL."""
    result = make_result(cl_total_L_per_h=20.0, vd_L=50.0)
    expected = math.log(2.0) * 50.0 / 20.0
    assert math.isclose(result.t_half_h, expected, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# isoform in result
# ---------------------------------------------------------------------------


def test_cyp_isoform_in_result():
    result = make_result(cyp_isoform="CYP3A4")
    assert result.cyp_isoform == "CYP3A4"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        make_result(dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        make_result(dose_mg=-10.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_total_L_per_h"):
        make_result(cl_total_L_per_h=0.0)


def test_validation_cl_negative():
    with pytest.raises(ValueError, match="cl_total_L_per_h"):
        make_result(cl_total_L_per_h=-5.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        make_result(vd_L=0.0)


def test_validation_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        make_result(vd_L=-1.0)


def test_validation_invalid_cyp_isoform():
    with pytest.raises(ValueError, match="cyp_isoform"):
        make_result(cyp_isoform="CYP2C9")


def test_validation_zone_fractions_not_summing_to_1():
    with pytest.raises(ValueError, match="sum to 1"):
        make_result(f_zone1=0.5, f_zone2=0.4, f_zone3=0.3)
