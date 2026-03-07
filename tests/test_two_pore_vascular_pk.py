"""Tests for Phase 864 — Two-Pore Vascular Permeability Model."""

import pytest

from omega_pbpk.core.two_pore_vascular_pk import (
    TwoPoreVascularPKResult,
    compare_mw_classes,
    simulate_two_pore_vascular_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and fields
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_two_pore_vascular_pk("mAb", 100.0, 150.0)
    assert isinstance(result, TwoPoreVascularPKResult)


def test_fields_preserved():
    result = simulate_two_pore_vascular_pk("TestBiologic", 200.0, 50.0)
    assert result.drug_name == "TestBiologic"
    assert result.dose_mg == 200.0
    assert result.mw_kDa == 50.0


def test_list_fields_non_empty():
    result = simulate_two_pore_vascular_pk("mAb", 100.0, 150.0, t_end_h=24.0, dt_h=1.0)
    assert len(result.times_h) > 0
    assert len(result.c_plasma_mg_L) == len(result.times_h)
    assert len(result.c_tissue_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# MW-based auto-assignment
# ---------------------------------------------------------------------------


def test_small_mw_low_sigma():
    """Small molecule (0.5 kDa) gets sigma_reflection = 0.05."""
    result = simulate_two_pore_vascular_pk("SmallMol", 100.0, 0.5)
    assert result.sigma_reflection == pytest.approx(0.05)


def test_antibody_mw_high_sigma():
    """Antibody-range MW (150 kDa) gets sigma_reflection = 0.7."""
    result = simulate_two_pore_vascular_pk("mAb", 100.0, 150.0)
    assert result.sigma_reflection == pytest.approx(0.7)


def test_large_protein_highest_sigma():
    """Large protein (>150 kDa) gets sigma_reflection = 0.95."""
    result = simulate_two_pore_vascular_pk("LargeProtein", 100.0, 200.0)
    assert result.sigma_reflection == pytest.approx(0.95)


def test_large_mw_higher_sigma_than_small_mw():
    """Larger MW → higher sigma_reflection."""
    small = simulate_two_pore_vascular_pk("Small", 100.0, 0.5)
    large = simulate_two_pore_vascular_pk("Large", 100.0, 150.0)
    assert large.sigma_reflection > small.sigma_reflection


# ---------------------------------------------------------------------------
# Custom PS and sigma override
# ---------------------------------------------------------------------------


def test_custom_ps_and_sigma():
    """Custom PS and sigma_reflection are used instead of auto values."""
    result = simulate_two_pore_vascular_pk(
        "Drug",
        100.0,
        150.0,
        ps_total_L_per_h=1.0,
        sigma_reflection=0.5,
    )
    assert result.ps_total_L_per_h == pytest.approx(1.0)
    assert result.sigma_reflection == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# IV bolus initial condition
# ---------------------------------------------------------------------------


def test_cmax_plasma_at_t0():
    """For IV bolus, plasma Cmax should be at t=0."""
    result = simulate_two_pore_vascular_pk("mAb", 100.0, 150.0, t_end_h=168.0, dt_h=0.5)
    assert result.tmax_plasma_h == pytest.approx(0.0)


def test_initial_plasma_concentration():
    """C_plasma(0) = dose / Vd_plasma."""
    dose = 300.0
    vd_plasma = 3.0
    result = simulate_two_pore_vascular_pk(
        "Drug", dose, 150.0, vd_plasma_L=vd_plasma, t_end_h=24.0, dt_h=0.5
    )
    expected_c0 = dose / vd_plasma
    assert result.c_plasma_mg_L[0] == pytest.approx(expected_c0)


def test_tissue_peaks_after_plasma():
    """Tissue Tmax should be later than plasma Tmax (t=0)."""
    result = simulate_two_pore_vascular_pk("Drug", 100.0, 5.0, t_end_h=72.0, dt_h=0.5)
    assert result.tmax_tissue_h > result.tmax_plasma_h


# ---------------------------------------------------------------------------
# Tissue AUC comparisons across MW classes
# ---------------------------------------------------------------------------


def test_small_mw_higher_tissue_auc_relative_to_plasma():
    """Small MW has higher tissue/plasma AUC ratio than large MW."""
    small = simulate_two_pore_vascular_pk("Small", 100.0, 0.5)
    large = simulate_two_pore_vascular_pk("Large", 100.0, 150.0)
    assert small.tissue_to_plasma_auc_ratio > large.tissue_to_plasma_auc_ratio


def test_tissue_to_plasma_auc_ratio_positive():
    result = simulate_two_pore_vascular_pk("mAb", 100.0, 150.0)
    assert result.tissue_to_plasma_auc_ratio > 0.0


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_double_dose_doubles_cmax():
    """Doubling dose approximately doubles Cmax plasma (linear model)."""
    result1 = simulate_two_pore_vascular_pk("Drug", 100.0, 50.0)
    result2 = simulate_two_pore_vascular_pk("Drug", 200.0, 50.0)
    ratio = result2.cmax_plasma / result1.cmax_plasma
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_double_dose_doubles_auc():
    """Doubling dose approximately doubles AUC plasma."""
    result1 = simulate_two_pore_vascular_pk("Drug", 100.0, 50.0)
    result2 = simulate_two_pore_vascular_pk("Drug", 200.0, 50.0)
    ratio = result2.auc_plasma / result1.auc_plasma
    assert ratio == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Vss
# ---------------------------------------------------------------------------


def test_vss_greater_than_vd_plasma():
    """Vss should be >= Vd_plasma since tissue distribution adds volume."""
    vd_plasma = 3.0  # default value
    result = simulate_two_pore_vascular_pk("Drug", 100.0, 5.0, vd_plasma_L=vd_plasma)
    assert result.vss_L >= vd_plasma


def test_vss_field_exists_and_positive():
    result = simulate_two_pore_vascular_pk("mAb", 100.0, 150.0)
    assert result.vss_L > 0.0


# ---------------------------------------------------------------------------
# compare_mw_classes
# ---------------------------------------------------------------------------


def test_compare_mw_classes_returns_correct_count():
    results = compare_mw_classes("Drug", 100.0, [0.5, 10.0, 150.0])
    assert len(results) == 3


def test_compare_mw_classes_sorted_by_auc_tissue_desc():
    results = compare_mw_classes("Drug", 100.0, [0.5, 10.0, 50.0, 150.0, 200.0])
    auc_tissues = [r.auc_tissue for r in results]
    assert auc_tissues == sorted(auc_tissues, reverse=True)


def test_compare_mw_classes_all_correct_type():
    results = compare_mw_classes("Drug", 100.0, [0.5, 150.0])
    for r in results:
        assert isinstance(r, TwoPoreVascularPKResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_two_pore_vascular_pk("Drug", 0.0, 50.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_two_pore_vascular_pk("Drug", -100.0, 50.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw_kDa"):
        simulate_two_pore_vascular_pk("Drug", 100.0, 0.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw_kDa"):
        simulate_two_pore_vascular_pk("Drug", 100.0, -1.0)


def test_validation_vd_plasma_zero():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_two_pore_vascular_pk("Drug", 100.0, 50.0, vd_plasma_L=0.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_two_pore_vascular_pk("Drug", 100.0, 50.0, cl_L_per_h=0.0)
