"""Tests for Phase 731 — Drug Efflux Transporter PK."""

import pytest

from omega_pbpk.core.efflux_transporter_pk import (
    EfluxTransporterPKResult,
    assess_efflux_saturation,
    simulate_efflux_transporter_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = simulate_efflux_transporter_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, EfluxTransporterPKResult)


def test_cmax_plasma_positive():
    result = simulate_efflux_transporter_pk("DrugA", dose_mg=100.0)
    assert result.cmax_plasma > 0


def test_cmax_tissue_positive():
    result = simulate_efflux_transporter_pk("DrugA", dose_mg=100.0)
    assert result.cmax_tissue > 0


def test_auc_plasma_positive():
    result = simulate_efflux_transporter_pk("DrugA", dose_mg=100.0)
    assert result.auc_plasma > 0


def test_auc_tissue_positive():
    result = simulate_efflux_transporter_pk("DrugA", dose_mg=100.0)
    assert result.auc_tissue > 0


def test_efflux_saturation_index_positive():
    result = simulate_efflux_transporter_pk("DrugA", dose_mg=100.0)
    assert result.efflux_saturation_index > 0


def test_tissue_to_plasma_auc_ratio_positive():
    result = simulate_efflux_transporter_pk("DrugA", dose_mg=100.0)
    assert result.tissue_to_plasma_auc_ratio > 0


def test_notes_nonempty():
    result = simulate_efflux_transporter_pk("DrugA", dose_mg=100.0)
    assert result.notes and len(result.notes) > 0


def test_times_list_length():
    result = simulate_efflux_transporter_pk("DrugA", dose_mg=100.0, t_end_h=12.0, dt_h=0.1)
    assert len(result.times_h) > 0
    assert len(result.c_plasma_mg_L) == len(result.times_h)
    assert len(result.c_tissue_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Physiological behaviour
# ---------------------------------------------------------------------------


def test_low_vmax_tissue_accumulates_more():
    """Very low Vmax → less efflux → tissue accumulates more drug."""
    low_efflux = simulate_efflux_transporter_pk("DrugB", dose_mg=100.0, vmax_efflux_mg_per_h=0.1)
    high_efflux = simulate_efflux_transporter_pk("DrugB", dose_mg=100.0, vmax_efflux_mg_per_h=50.0)
    assert low_efflux.auc_tissue > high_efflux.auc_tissue


def test_high_vmax_reduces_tissue_concentration():
    """Very high Vmax → strong efflux → lower tissue AUC."""
    moderate = simulate_efflux_transporter_pk("DrugC", dose_mg=100.0, vmax_efflux_mg_per_h=5.0)
    very_high = simulate_efflux_transporter_pk("DrugC", dose_mg=100.0, vmax_efflux_mg_per_h=200.0)
    assert moderate.auc_tissue > very_high.auc_tissue


def test_linear_dose_scaling_auc():
    """Doubling dose should approximately double AUC (unsaturated regime)."""
    r100 = simulate_efflux_transporter_pk("DrugD", dose_mg=10.0, km_efflux_mg_L=100.0)
    r200 = simulate_efflux_transporter_pk("DrugD", dose_mg=20.0, km_efflux_mg_L=100.0)
    ratio = r200.auc_plasma / r100.auc_plasma
    assert 1.8 < ratio < 2.2


def test_saturation_index_matches_definition():
    result = simulate_efflux_transporter_pk("DrugE", dose_mg=100.0, km_efflux_mg_L=0.5)
    expected = result.cmax_tissue / 0.5
    assert abs(result.efflux_saturation_index - expected) < 1e-9


def test_drug_name_preserved():
    result = simulate_efflux_transporter_pk("MyDrug", dose_mg=50.0)
    assert result.drug_name == "MyDrug"


def test_dose_preserved():
    result = simulate_efflux_transporter_pk("DrugX", dose_mg=75.0)
    assert result.dose_mg == 75.0


# ---------------------------------------------------------------------------
# assess_efflux_saturation
# ---------------------------------------------------------------------------


def test_assess_returns_dict_with_required_keys():
    result = assess_efflux_saturation(cmax_tissue_mg_L=0.1, km_efflux_mg_L=0.5)
    assert "saturation_index" in result
    assert "risk_level" in result


def test_assess_low_risk():
    result = assess_efflux_saturation(cmax_tissue_mg_L=0.01, km_efflux_mg_L=0.5)
    assert result["risk_level"] == "low"
    assert result["saturation_index"] < 0.1


def test_assess_moderate_risk():
    result = assess_efflux_saturation(cmax_tissue_mg_L=0.3, km_efflux_mg_L=0.5)
    assert result["risk_level"] == "moderate"


def test_assess_high_risk():
    result = assess_efflux_saturation(cmax_tissue_mg_L=5.0, km_efflux_mg_L=0.5)
    assert result["risk_level"] == "high"
    assert result["saturation_index"] >= 1.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_efflux_transporter_pk("DrugV", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_efflux_transporter_pk("DrugV", dose_mg=-10.0)


def test_validation_km_zero_raises():
    with pytest.raises(ValueError, match="km_efflux_mg_L"):
        simulate_efflux_transporter_pk("DrugV", dose_mg=100.0, km_efflux_mg_L=0.0)
