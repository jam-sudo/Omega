"""Tests for Phase 998 — hepatic_zonal_pk.py"""

import pytest

from omega_pbpk.core.hepatic_zonal_pk import (
    HepaticZonalPKResult,
    simulate_hepatic_zonal_pk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_hepatic_zonal_pk_result():
    result = simulate_hepatic_zonal_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, HepaticZonalPKResult)


def test_drug_name_preserved():
    result = simulate_hepatic_zonal_pk("Acetaminophen", dose_mg=500.0)
    assert result.drug_name == "Acetaminophen"


# ---------------------------------------------------------------------------
# Numeric sanity
# ---------------------------------------------------------------------------


def test_cmax_systemic_positive():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0)
    assert result.cmax_systemic_mg_L > 0


def test_auc_systemic_positive():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0)
    assert result.auc_systemic_mg_h_per_L > 0


def test_zone1_extraction_in_range():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0)
    assert 0 <= result.zone1_extraction <= 1


def test_zone2_extraction_in_range():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0)
    assert 0 <= result.zone2_extraction <= 1


def test_total_hepatic_extraction_in_range():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0)
    assert 0 <= result.total_hepatic_extraction <= 1


def test_total_extraction_gte_individual_zones():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0)
    assert result.total_hepatic_extraction >= max(result.zone1_extraction, result.zone2_extraction)


# ---------------------------------------------------------------------------
# Primary zone of metabolism
# ---------------------------------------------------------------------------


def test_primary_zone_valid_values():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0)
    assert result.primary_zone_of_metabolism in {"zone1", "zone2", "distributed"}


def test_hepatotoxicity_zone_valid_values():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0)
    assert result.hepatotoxicity_zone in {"periportal", "centrilobular", "panlobular"}


def test_high_clint_zone1_gives_zone1_primary():
    """Very high zone1 CLint relative to zone2 → primary_zone == 'zone1'."""
    result = simulate_hepatic_zonal_pk(
        "Drug",
        dose_mg=100.0,
        clint_zone1_mL_min=200.0,
        clint_zone2_mL_min=5.0,
    )
    assert result.primary_zone_of_metabolism == "zone1"


def test_high_clint_zone2_gives_zone2_primary():
    """Very high zone2 CLint relative to zone1 → primary_zone == 'zone2'."""
    result = simulate_hepatic_zonal_pk(
        "Drug",
        dose_mg=100.0,
        clint_zone1_mL_min=5.0,
        clint_zone2_mL_min=200.0,
    )
    assert result.primary_zone_of_metabolism == "zone2"


def test_hepatotoxicity_zone_periportal_when_zone1_primary():
    result = simulate_hepatic_zonal_pk(
        "Drug",
        dose_mg=100.0,
        clint_zone1_mL_min=200.0,
        clint_zone2_mL_min=5.0,
    )
    assert result.hepatotoxicity_zone == "periportal"


def test_hepatotoxicity_zone_centrilobular_when_zone2_primary():
    result = simulate_hepatic_zonal_pk(
        "Drug",
        dose_mg=100.0,
        clint_zone1_mL_min=5.0,
        clint_zone2_mL_min=200.0,
    )
    assert result.hepatotoxicity_zone == "centrilobular"


# ---------------------------------------------------------------------------
# List length consistency
# ---------------------------------------------------------------------------


def test_c_zone1_same_length_as_times():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0, t_end_h=4.0, dt_h=0.1)
    assert len(result.c_zone1_mg_L) == len(result.times_h)


def test_c_zone2_same_length_as_times():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0, t_end_h=4.0, dt_h=0.1)
    assert len(result.c_zone2_mg_L) == len(result.times_h)


def test_c_systemic_same_length_as_times():
    result = simulate_hepatic_zonal_pk("Drug", dose_mg=100.0, t_end_h=4.0, dt_h=0.1)
    assert len(result.c_systemic_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_hepatic_zonal_pk("Drug", dose_mg=0.0)


def test_negative_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_hepatic_zonal_pk("Drug", dose_mg=-50.0)


def test_invalid_q_portal_raises_value_error():
    with pytest.raises(ValueError, match="q_portal"):
        simulate_hepatic_zonal_pk("Drug", dose_mg=100.0, q_portal_L_per_h=0.0)


def test_invalid_vd_systemic_raises_value_error():
    with pytest.raises(ValueError, match="vd_systemic"):
        simulate_hepatic_zonal_pk("Drug", dose_mg=100.0, vd_systemic_L=0.0)
