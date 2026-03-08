"""Tests for Phase 477 — Bile Acid Transporter PK (EHC simulation)."""

import pytest

from omega_pbpk.core.bile_acid_transporter_pk import (
    BileAcidTransporterResult,
    compare_biliary_fractions,
    simulate_bile_acid_transporter_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, BileAcidTransporterResult)


def test_result_has_expected_fields():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0)
    assert hasattr(result, "drug_name")
    assert hasattr(result, "dose_mg")
    assert hasattr(result, "f_biliary")
    assert hasattr(result, "times_h")
    assert hasattr(result, "c_systemic_mg_L")
    assert hasattr(result, "cmax_primary_mg_L")
    assert hasattr(result, "tmax_primary_h")
    assert hasattr(result, "cmax_secondary_mg_L")
    assert hasattr(result, "tmax_secondary_h")
    assert hasattr(result, "ehc_detected")
    assert hasattr(result, "auc_total_mg_h_per_L")
    assert hasattr(result, "notes")


def test_drug_name_stored():
    result = simulate_bile_acid_transporter_pk("Morphine", dose_mg=50.0)
    assert result.drug_name == "Morphine"


def test_dose_mg_stored():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=200.0)
    assert result.dose_mg == 200.0


def test_f_biliary_stored():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0, f_biliary=0.4)
    assert result.f_biliary == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_c_systemic_starts_at_zero():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0)
    assert result.c_systemic_mg_L[0] == pytest.approx(0.0)


def test_times_start_at_zero():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0, t_end_h=24.0, dt_h=0.1)
    # last time should be close to 24.0
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.11)


# ---------------------------------------------------------------------------
# AUC and concentration values
# ---------------------------------------------------------------------------


def test_auc_total_positive():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0)
    assert result.auc_total_mg_h_per_L > 0.0


def test_cmax_primary_positive():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_primary_mg_L > 0.0


def test_larger_dose_larger_auc():
    result_low = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=50.0)
    result_high = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=200.0)
    assert result_high.auc_total_mg_h_per_L > result_low.auc_total_mg_h_per_L


def test_larger_dose_larger_cmax():
    result_low = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=50.0)
    result_high = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=200.0)
    assert result_high.cmax_primary_mg_L > result_low.cmax_primary_mg_L


# ---------------------------------------------------------------------------
# EHC detection
# ---------------------------------------------------------------------------


def test_high_f_biliary_ehc_detected():
    """High biliary fraction with meal trigger produces secondary peak (EHC detected).

    Uses fast absorption (ka=3 /h) and clearance (cl=10 L/h) so the primary peak
    occurs early (~0.8 h) and systemic concentration drops before the meal at t=6 h
    triggers bile bolus release back to GI, creating a measurable secondary peak.
    """
    result = simulate_bile_acid_transporter_pk(
        "EHCDrug",
        dose_mg=500.0,
        f_biliary=0.5,
        ka_per_h=3.0,
        k_bile_release_per_h=0.5,
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
        t_end_h=48.0,
        dt_h=0.1,
        meal_times_h=[6.0],
    )
    assert result.ehc_detected is True


def test_zero_f_biliary_no_ehc():
    """No biliary excretion means no bile pool accumulation, no secondary peak."""
    result = simulate_bile_acid_transporter_pk(
        "TestDrug",
        dose_mg=100.0,
        f_biliary=0.0,
        t_end_h=48.0,
        dt_h=0.1,
        meal_times_h=[6.0],
    )
    assert result.ehc_detected is False


def test_secondary_peak_after_primary():
    """If EHC detected, secondary peak must occur after primary."""
    result = simulate_bile_acid_transporter_pk(
        "EHCDrug",
        dose_mg=500.0,
        f_biliary=0.5,
        ka_per_h=3.0,
        k_bile_release_per_h=0.5,
        cl_sys_L_per_h=10.0,
        vd_sys_L=50.0,
        t_end_h=48.0,
        dt_h=0.1,
        meal_times_h=[6.0],
    )
    if result.ehc_detected:
        assert result.tmax_secondary_h > result.tmax_primary_h


def test_no_ehc_secondary_values_zero():
    """When EHC not detected, secondary peak values should be 0."""
    result = simulate_bile_acid_transporter_pk(
        "TestDrug",
        dose_mg=100.0,
        f_biliary=0.0,
    )
    if not result.ehc_detected:
        assert result.cmax_secondary_mg_L == pytest.approx(0.0)
        assert result.tmax_secondary_h == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = simulate_bile_acid_transporter_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_notes_contains_drug_name():
    result = simulate_bile_acid_transporter_pk("Rifampicin", dose_mg=600.0)
    assert "Rifampicin" in result.notes


# ---------------------------------------------------------------------------
# compare_biliary_fractions
# ---------------------------------------------------------------------------


def test_compare_biliary_fractions_returns_list():
    results = compare_biliary_fractions("TestDrug", dose_mg=100.0)
    assert isinstance(results, list)


def test_compare_biliary_fractions_default_length():
    results = compare_biliary_fractions("TestDrug", dose_mg=100.0)
    assert len(results) == 4


def test_compare_biliary_fractions_all_results():
    results = compare_biliary_fractions("TestDrug", dose_mg=100.0)
    for r in results:
        assert isinstance(r, BileAcidTransporterResult)


def test_compare_biliary_fractions_sorted_by_auc_descending():
    results = compare_biliary_fractions("TestDrug", dose_mg=100.0)
    aucs = [r.auc_total_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_biliary_fractions_custom_fractions():
    fractions = [0.0, 0.2, 0.4]
    results = compare_biliary_fractions("TestDrug", dose_mg=100.0, fractions=fractions)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_bile_acid_transporter_pk("X", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_bile_acid_transporter_pk("X", dose_mg=-10.0)


def test_validation_cl_sys_zero():
    with pytest.raises(ValueError, match="cl_sys"):
        simulate_bile_acid_transporter_pk("X", dose_mg=100.0, cl_sys_L_per_h=0.0)


def test_validation_vd_sys_zero():
    with pytest.raises(ValueError, match="vd_sys"):
        simulate_bile_acid_transporter_pk("X", dose_mg=100.0, vd_sys_L=0.0)


def test_validation_f_biliary_out_of_range():
    with pytest.raises(ValueError, match="f_biliary"):
        simulate_bile_acid_transporter_pk("X", dose_mg=100.0, f_biliary=1.5)


def test_validation_f_oral_out_of_range():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_bile_acid_transporter_pk("X", dose_mg=100.0, f_oral=0.0)


def test_validation_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_bile_acid_transporter_pk("X", dose_mg=100.0, ka_per_h=0.0)
