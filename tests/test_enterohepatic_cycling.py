"""Tests for Phase 337: Advanced enterohepatic cycling model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.enterohepatic_cycling import (
    EnterohepatiCyclingResult,
    compare_biliary_fractions,
    simulate_enterohepatic_cycling,
)

# ---------------------------------------------------------------------------
# Helper: default valid inputs
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    cl_L_per_h=5.0,
    vd_L=50.0,
    f_biliary=0.3,
    ka_per_h=1.2,
    f_oral=0.8,
    gallbladder_empty_interval_h=4.0,
    gallbladder_empty_fraction=0.8,
    t_end_h=48.0,
    dt_h=0.1,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_enterohepatic_cycling("X", 100.0, 0.0, 50.0)


def test_invalid_cl_negative():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_enterohepatic_cycling("X", 100.0, -1.0, 50.0)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 0.0)


def test_invalid_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, -10.0)


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_enterohepatic_cycling("X", 0.0, 5.0, 50.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_enterohepatic_cycling("X", -50.0, 5.0, 50.0)


def test_invalid_f_biliary_equals_one():
    with pytest.raises(ValueError, match="f_biliary"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 50.0, f_biliary=1.0)


def test_invalid_f_biliary_negative():
    with pytest.raises(ValueError, match="f_biliary"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 50.0, f_biliary=-0.1)


def test_invalid_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 50.0, ka_per_h=0.0)


def test_invalid_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 50.0, f_oral=0.0)


def test_invalid_f_oral_greater_than_one():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 50.0, f_oral=1.5)


def test_invalid_gb_interval_zero():
    with pytest.raises(ValueError, match="gallbladder_empty_interval_h"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 50.0, gallbladder_empty_interval_h=0.0)


def test_invalid_gb_fraction_zero():
    with pytest.raises(ValueError, match="gallbladder_empty_fraction"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 50.0, gallbladder_empty_fraction=0.0)


def test_invalid_gb_fraction_greater_than_one():
    with pytest.raises(ValueError, match="gallbladder_empty_fraction"):
        simulate_enterohepatic_cycling("X", 100.0, 5.0, 50.0, gallbladder_empty_fraction=1.5)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_enterohepatic_cycling(**DEFAULTS)
    assert isinstance(result, EnterohepatiCyclingResult)


def test_frozen_dataclass():
    result = simulate_enterohepatic_cycling(**DEFAULTS)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]


def test_plasma_concentrations_nonnegative():
    result = simulate_enterohepatic_cycling(**DEFAULTS)
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


def test_bile_concentrations_nonnegative():
    result = simulate_enterohepatic_cycling(**DEFAULTS)
    assert all(c >= 0.0 for c in result.c_bile_mg_L)


def test_gallbladder_mass_nonnegative():
    result = simulate_enterohepatic_cycling(**DEFAULTS)
    assert all(m >= 0.0 for m in result.m_gallbladder_mg)


def test_auc_total_positive():
    result = simulate_enterohepatic_cycling(**DEFAULTS)
    assert result.auc_total_mg_h_L > 0.0


def test_secondary_peaks_nonnegative():
    result = simulate_enterohepatic_cycling(**DEFAULTS)
    assert result.secondary_peaks >= 0


# ---------------------------------------------------------------------------
# f_biliary = 0: no recycling
# ---------------------------------------------------------------------------


def test_no_biliary_ehr_auc_ratio_near_one():
    result = simulate_enterohepatic_cycling(
        drug_name="NoBile", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_biliary=0.0
    )
    assert abs(result.ehr_auc_ratio - 1.0) < 0.05


def test_no_biliary_zero_secondary_peaks():
    result = simulate_enterohepatic_cycling(
        drug_name="NoBile", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_biliary=0.0
    )
    assert result.secondary_peaks == 0


def test_no_biliary_t_half_reasonable():
    result = simulate_enterohepatic_cycling(
        drug_name="NoBile", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_biliary=0.0
    )
    # t_half should be close to VD/CL * ln2
    theoretical = math.log(2.0) / (5.0 / 50.0)
    assert abs(result.t_half_h - theoretical) < theoretical * 0.5


# ---------------------------------------------------------------------------
# f_biliary > 0: recycling effects
# ---------------------------------------------------------------------------


def test_high_biliary_ehr_auc_ratio_greater_than_one():
    result = simulate_enterohepatic_cycling(
        drug_name="HighBile", dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, f_biliary=0.5
    )
    assert result.ehr_auc_ratio > 1.0


def test_ehr_extends_half_life():
    """With biliary recycling, apparent t_half should be longer."""
    r_no_ehr = simulate_enterohepatic_cycling(
        drug_name="Drug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_biliary=0.0,
        t_end_h=96.0,
    )
    r_ehr = simulate_enterohepatic_cycling(
        drug_name="Drug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        f_biliary=0.5,
        t_end_h=96.0,
    )
    assert r_ehr.t_half_h >= r_no_ehr.t_half_no_ehr_h


def test_notes_contain_drug_name():
    result = simulate_enterohepatic_cycling(
        drug_name="Warfarin", dose_mg=5.0, cl_L_per_h=0.2, vd_L=9.0, f_biliary=0.3
    )
    assert "Warfarin" in result.notes


def test_result_fields_drug_name():
    result = simulate_enterohepatic_cycling(
        drug_name="Morphine", dose_mg=10.0, cl_L_per_h=2.0, vd_L=30.0
    )
    assert result.drug_name == "Morphine"
    assert result.dose_mg == 10.0


def test_times_length_matches_concentrations():
    result = simulate_enterohepatic_cycling(**DEFAULTS)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_bile_mg_L)
    assert len(result.times_h) == len(result.m_gallbladder_mg)


# ---------------------------------------------------------------------------
# compare_biliary_fractions
# ---------------------------------------------------------------------------


def test_compare_biliary_fractions_default_length():
    results = compare_biliary_fractions("Drug", 100.0, 5.0, 50.0)
    assert len(results) == 5


def test_compare_biliary_fractions_custom_length():
    fb_list = [0.0, 0.2, 0.4]
    results = compare_biliary_fractions("Drug", 100.0, 5.0, 50.0, f_biliary_values=fb_list)
    assert len(results) == 3


def test_compare_biliary_fractions_sorted_descending():
    results = compare_biliary_fractions("Drug", 100.0, 5.0, 50.0)
    ratios = [r.ehr_auc_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_biliary_fractions_all_results_type():
    results = compare_biliary_fractions("Drug", 100.0, 5.0, 50.0)
    for r in results:
        assert isinstance(r, EnterohepatiCyclingResult)
