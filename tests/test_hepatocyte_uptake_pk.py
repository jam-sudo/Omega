"""Tests for Phase 770 — Drug Hepatocyte Uptake PK."""

import pytest

from omega_pbpk.core.hepatocyte_uptake_pk import (
    HepatocyteUptakePKResult,
    assess_transporter_contribution,
    simulate_hepatocyte_uptake_pk,
)

# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


def test_returns_hepatocyte_uptake_pk_result():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert isinstance(result, HepatocyteUptakePKResult)


def test_times_list_non_empty():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert len(result.times_h) > 0


def test_c_plasma_list_matches_times():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_c_hep_list_matches_times():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert len(result.c_hep_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Scalar outputs
# ---------------------------------------------------------------------------


def test_cmax_plasma_positive():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert result.cmax_plasma > 0.0


def test_cmax_hep_positive():
    """Drug must accumulate in hepatocytes."""
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert result.cmax_hep > 0.0


def test_auc_plasma_positive():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert result.auc_plasma > 0.0


def test_auc_hep_positive():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert result.auc_hep > 0.0


def test_hep_to_plasma_ratio_positive():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert result.hep_to_plasma_ratio > 0.0


def test_cl_hepatic_estimated_positive():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert result.cl_hepatic_estimated_L_per_h >= 0.0


def test_transporter_saturation_is_bool():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert isinstance(result.transporter_saturation, bool)


def test_notes_nonempty():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_drug_name_stored():
    result = simulate_hepatocyte_uptake_pk("Rosuvastatin", dose_mg=5.0)
    assert result.drug_name == "Rosuvastatin"


# ---------------------------------------------------------------------------
# Physical behaviour
# ---------------------------------------------------------------------------


def test_higher_vmax_lowers_plasma_auc():
    """Higher transporter Vmax → more hepatic uptake → lower systemic AUC."""
    res_low = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0, vmax_uptake_mg_per_h=2.0)
    res_high = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0, vmax_uptake_mg_per_h=20.0)
    assert res_high.auc_plasma < res_low.auc_plasma


def test_higher_vmax_raises_hep_auc():
    """Higher Vmax → more drug in hepatocytes."""
    res_low = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0, vmax_uptake_mg_per_h=2.0)
    res_high = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0, vmax_uptake_mg_per_h=20.0)
    assert res_high.auc_hep > res_low.auc_hep


def test_dose_scaling_plasma_approximately_linear():
    """Doubling the dose should approximately double plasma AUC (linear regime)."""
    res_low = simulate_hepatocyte_uptake_pk(
        "Statin", dose_mg=5.0, vmax_uptake_mg_per_h=1.0, km_uptake_mg_L=10.0
    )
    res_high = simulate_hepatocyte_uptake_pk(
        "Statin", dose_mg=10.0, vmax_uptake_mg_per_h=1.0, km_uptake_mg_L=10.0
    )
    ratio = res_high.auc_plasma / res_low.auc_plasma
    # Allow 15% deviation from exact linear scaling
    assert 1.7 <= ratio <= 2.3


def test_high_dose_triggers_saturation():
    """Very high dose relative to Km should mark saturation."""
    result = simulate_hepatocyte_uptake_pk(
        "Statin",
        dose_mg=200.0,
        vd_L=10.0,  # small Vd → high Cmax
        vmax_uptake_mg_per_h=10.0,
        km_uptake_mg_L=1.0,
        pdif_per_h=0.01,
    )
    assert result.transporter_saturation is True


# ---------------------------------------------------------------------------
# assess_transporter_contribution
# ---------------------------------------------------------------------------


def test_assess_returns_dict():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    contrib = assess_transporter_contribution(result)
    assert isinstance(contrib, dict)


def test_assess_has_oatp_fraction_key():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    contrib = assess_transporter_contribution(result)
    assert "oatp_fraction" in contrib


def test_assess_has_passive_fraction_key():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    contrib = assess_transporter_contribution(result)
    assert "passive_fraction" in contrib


def test_assess_fractions_sum_to_one():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    contrib = assess_transporter_contribution(result)
    total = contrib["oatp_fraction"] + contrib["passive_fraction"]
    assert abs(total - 1.0) < 1e-6


def test_assess_has_notes_key():
    result = simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0)
    contrib = assess_transporter_contribution(result)
    assert "notes" in contrib


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_hepatocyte_uptake_pk("Statin", dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_hepatocyte_uptake_pk("Statin", dose_mg=-10.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError):
        simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0, vd_L=0.0)


def test_cl_sys_negative_raises():
    with pytest.raises(ValueError):
        simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0, cl_sys_L_per_h=-1.0)


def test_vmax_zero_raises():
    with pytest.raises(ValueError):
        simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0, vmax_uptake_mg_per_h=0.0)


def test_km_zero_raises():
    with pytest.raises(ValueError):
        simulate_hepatocyte_uptake_pk("Statin", dose_mg=10.0, km_uptake_mg_L=0.0)
