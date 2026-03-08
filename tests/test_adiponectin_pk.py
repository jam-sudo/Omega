"""Tests for Phase 218 — adiponectin_pk.py"""

import pytest

from omega_pbpk.core.adiponectin_pk import (
    AdiponectinPKResult,
    assess_target_engagement,
    simulate_adiponectin_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iv_result():
    return simulate_adiponectin_pk("Adipomab", 100.0, "iv", vd_L=10.0, t_end_h=168.0)


@pytest.fixture
def sc_result():
    return simulate_adiponectin_pk("Adipomab", 100.0, "sc", vd_L=10.0, t_end_h=168.0)


# ---------------------------------------------------------------------------
# Return type & field preservation
# ---------------------------------------------------------------------------


def test_return_type_iv(iv_result):
    assert isinstance(iv_result, AdiponectinPKResult)


def test_return_type_sc(sc_result):
    assert isinstance(sc_result, AdiponectinPKResult)


def test_field_drug_name(iv_result):
    assert iv_result.drug_name == "Adipomab"


def test_field_dose_mg(iv_result):
    assert iv_result.dose_mg == 100.0


def test_field_route_iv(iv_result):
    assert iv_result.route == "iv"


def test_field_route_sc(sc_result):
    assert sc_result.route == "sc"


# ---------------------------------------------------------------------------
# List fields
# ---------------------------------------------------------------------------


def test_times_h_is_list(iv_result):
    assert isinstance(iv_result.times_h, list)
    assert len(iv_result.times_h) > 0


def test_c_drug_is_list(iv_result):
    assert isinstance(iv_result.c_drug_mg_L, list)
    assert len(iv_result.c_drug_mg_L) == len(iv_result.times_h)


def test_c_complex_is_list(iv_result):
    assert isinstance(iv_result.c_complex_mg_L, list)
    assert len(iv_result.c_complex_mg_L) == len(iv_result.times_h)


def test_c_peripheral_is_list(iv_result):
    assert isinstance(iv_result.c_peripheral_mg_L, list)
    assert len(iv_result.c_peripheral_mg_L) == len(iv_result.times_h)


# ---------------------------------------------------------------------------
# Initial condition behaviour
# ---------------------------------------------------------------------------


def test_iv_starts_at_dose_over_vd(iv_result):
    """IV: first concentration = dose / Vd."""
    expected = 100.0 / 10.0  # 10.0 mg/L
    assert iv_result.c_drug_mg_L[0] == pytest.approx(expected, rel=1e-6)


def test_sc_starts_at_zero(sc_result):
    """SC: drug concentration starts at 0."""
    assert sc_result.c_drug_mg_L[0] == pytest.approx(0.0, abs=1e-9)


def test_sc_rises_from_zero(sc_result):
    """SC: drug concentration should rise after t=0."""
    peak = max(sc_result.c_drug_mg_L)
    assert peak > 0.01  # at least some absorption occurs


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


def test_cmax_positive_iv(iv_result):
    assert iv_result.cmax_drug_mg_L > 0.0


def test_cmax_positive_sc(sc_result):
    assert sc_result.cmax_drug_mg_L > 0.0


def test_auc_positive_iv(iv_result):
    assert iv_result.auc_drug_mg_h_per_L > 0.0


def test_auc_positive_sc(sc_result):
    assert sc_result.auc_drug_mg_h_per_L > 0.0


def test_iv_auc_greater_than_sc_auc(iv_result, sc_result):
    """IV typically has higher AUC than SC due to F < 1."""
    assert iv_result.auc_drug_mg_h_per_L > sc_result.auc_drug_mg_h_per_L


def test_c_complex_rises_then_falls(iv_result):
    """Complex should increase from 0 and eventually decline."""
    peak_idx = iv_result.c_complex_mg_L.index(max(iv_result.c_complex_mg_L))
    # peak should not be at the last time point
    assert peak_idx < len(iv_result.c_complex_mg_L) - 1
    assert iv_result.c_complex_mg_L[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Receptor occupancy
# ---------------------------------------------------------------------------


def test_receptor_occupancy_at_cmax_in_range_iv(iv_result):
    assert 0.0 <= iv_result.receptor_occupancy_at_cmax <= 1.0


def test_receptor_occupancy_at_cmax_in_range_sc(sc_result):
    assert 0.0 <= sc_result.receptor_occupancy_at_cmax <= 1.0


# ---------------------------------------------------------------------------
# assess_target_engagement
# ---------------------------------------------------------------------------


def test_assess_target_engagement_returns_dict(iv_result):
    te = assess_target_engagement(iv_result)
    assert isinstance(te, dict)


def test_assess_keys_present(iv_result):
    te = assess_target_engagement(iv_result)
    assert "max_receptor_occupancy_pct" in te
    assert "duration_above_50pct_h" in te
    assert "clinical_relevance" in te


def test_assess_max_ro_pct_non_negative(iv_result):
    te = assess_target_engagement(iv_result)
    assert te["max_receptor_occupancy_pct"] >= 0.0


def test_assess_duration_non_negative(iv_result):
    te = assess_target_engagement(iv_result)
    assert te["duration_above_50pct_h"] >= 0.0


def test_assess_clinical_relevance_is_str(iv_result):
    te = assess_target_engagement(iv_result)
    assert isinstance(te["clinical_relevance"], str)
    assert len(te["clinical_relevance"]) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adiponectin_pk("Drug", 0.0, "iv")


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adiponectin_pk("Drug", -50.0, "sc")


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_adiponectin_pk("Drug", 100.0, "intramuscular")
