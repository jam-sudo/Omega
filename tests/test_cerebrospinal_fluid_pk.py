"""Tests for Phase 1145 — CSF PK model."""

import math
import pytest

from omega_pbpk.core.cerebrospinal_fluid_pk import simulate_csf_pk, CSFPKResult


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_csf_pk_result():
    result = simulate_csf_pk("drug_a", dose_mg=500.0, logP=2.0)
    assert isinstance(result, CSFPKResult)


def test_iv_route_csf_starts_at_zero():
    """IV route: CSF concentration should start at 0."""
    result = simulate_csf_pk("drug_a", dose_mg=500.0, logP=2.0, route="iv")
    assert result.c_csf_mg_L[0] == pytest.approx(0.0)


def test_iv_route_plasma_starts_at_dose_over_vd():
    result = simulate_csf_pk("drug_a", dose_mg=500.0, logP=2.0, route="iv", vd_L=50.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(10.0)  # 500/50


def test_iv_route_csf_rises_then_falls():
    """IV route: CSF should rise from 0 to a peak and then decline."""
    result = simulate_csf_pk("drug_a", dose_mg=500.0, logP=2.0, route="iv", t_end_h=48.0)
    csf = result.c_csf_mg_L
    # tmax should not be at time 0
    assert result.tmax_csf_h > 0
    # cmax should be positive
    assert result.cmax_csf_mg_L > 0


def test_intrathecal_csf_starts_high():
    """Intrathecal: CSF(0) = dose / V_csf = 500 / 0.15."""
    result = simulate_csf_pk(
        "drug_a", dose_mg=500.0, logP=2.0, route="intrathecal"
    )
    expected = 500.0 / 0.15
    assert result.c_csf_mg_L[0] == pytest.approx(expected, rel=1e-6)


def test_intrathecal_plasma_starts_at_zero():
    result = simulate_csf_pk("drug_b", dose_mg=100.0, logP=1.0, route="intrathecal")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_intrathecal_csf_declines_over_time():
    result = simulate_csf_pk("drug_b", dose_mg=100.0, logP=1.0, route="intrathecal")
    # CSF should decline overall (net drainage > influx since plasma starts at 0)
    assert result.c_csf_mg_L[-1] < result.c_csf_mg_L[0]


# ---------------------------------------------------------------------------
# logP effect on CSF penetration
# ---------------------------------------------------------------------------


def test_higher_logP_gives_higher_csf_penetration_iv():
    """Higher logP → higher kpc → higher CSF/plasma AUC ratio (IV route)."""
    result_low = simulate_csf_pk("low", dose_mg=500.0, logP=1.0, route="iv", fu_plasma=0.5)
    result_high = simulate_csf_pk("high", dose_mg=500.0, logP=5.0, route="iv", fu_plasma=0.5)
    assert result_high.csf_to_plasma_auc_ratio > result_low.csf_to_plasma_auc_ratio


def test_higher_fu_gives_higher_csf_penetration():
    """Higher fu_plasma → higher kpc → more CSF penetration."""
    result_low = simulate_csf_pk("d1", dose_mg=500.0, logP=2.0, route="iv", fu_plasma=0.05)
    result_high = simulate_csf_pk("d2", dose_mg=500.0, logP=2.0, route="iv", fu_plasma=0.5)
    assert result_high.csf_to_plasma_auc_ratio > result_low.csf_to_plasma_auc_ratio


# ---------------------------------------------------------------------------
# AUC and ratio checks
# ---------------------------------------------------------------------------


def test_auc_csf_positive_iv():
    result = simulate_csf_pk("d", dose_mg=500.0, logP=2.0, route="iv")
    assert result.auc_csf_mg_h_per_L > 0


def test_auc_plasma_positive_iv():
    result = simulate_csf_pk("d", dose_mg=500.0, logP=2.0, route="iv")
    assert result.auc_plasma_mg_h_per_L > 0


def test_csf_to_plasma_ratio_positive():
    result = simulate_csf_pk("d", dose_mg=500.0, logP=2.0, route="iv")
    assert result.csf_to_plasma_auc_ratio > 0


def test_csf_to_plasma_ratio_matches_auc():
    result = simulate_csf_pk("d", dose_mg=500.0, logP=2.0, route="iv")
    expected = result.auc_csf_mg_h_per_L / result.auc_plasma_mg_h_per_L
    assert result.csf_to_plasma_auc_ratio == pytest.approx(expected, rel=1e-6)


def test_auc_csf_positive_intrathecal():
    result = simulate_csf_pk("d", dose_mg=100.0, logP=2.0, route="intrathecal")
    assert result.auc_csf_mg_h_per_L > 0


# ---------------------------------------------------------------------------
# Penetration class
# ---------------------------------------------------------------------------


def test_penetration_class_poor_low_logP():
    """Very low logP, low fu → lower CSF/plasma ratio than high logP drug."""
    result_low = simulate_csf_pk(
        "bad", dose_mg=500.0, logP=-2.0, route="iv",
        fu_plasma=0.01, cl_L_per_h=30.0, vd_L=50.0, t_end_h=24.0
    )
    result_high = simulate_csf_pk(
        "good", dose_mg=500.0, logP=5.0, route="iv",
        fu_plasma=0.8, cl_L_per_h=5.0, vd_L=50.0, t_end_h=24.0
    )
    # Low logP/fu gives lower penetration than high logP/fu
    assert result_low.csf_to_plasma_auc_ratio < result_high.csf_to_plasma_auc_ratio
    # Penetration class is valid
    assert result_low.csf_penetration_class in {"poor", "moderate", "good"}


def test_penetration_class_good_high_logP():
    """High logP, high fu → good penetration."""
    result = simulate_csf_pk(
        "good_drug", dose_mg=500.0, logP=5.0, route="iv",
        fu_plasma=0.8, cl_L_per_h=2.0, vd_L=50.0, t_end_h=48.0
    )
    assert result.csf_penetration_class in {"moderate", "good"}


def test_penetration_class_values_valid():
    for logP in [-2.0, 1.5, 5.0]:
        r = simulate_csf_pk("x", dose_mg=200.0, logP=logP, route="iv")
        assert r.csf_penetration_class in {"poor", "moderate", "good"}


# ---------------------------------------------------------------------------
# Meningitis utility
# ---------------------------------------------------------------------------


def test_meningitis_utility_true_when_ratio_above_01():
    result = simulate_csf_pk(
        "good_drug", dose_mg=500.0, logP=5.0, route="iv",
        fu_plasma=0.8, cl_L_per_h=2.0, vd_L=50.0, t_end_h=48.0
    )
    if result.csf_to_plasma_auc_ratio > 0.1:
        assert result.meningitis_utility is True


def test_meningitis_utility_false_when_ratio_below_01():
    result = simulate_csf_pk(
        "bad", dose_mg=500.0, logP=-2.0, route="iv",
        fu_plasma=0.01, cl_L_per_h=30.0, vd_L=50.0
    )
    if result.csf_to_plasma_auc_ratio <= 0.1:
        assert result.meningitis_utility is False


def test_meningitis_utility_consistent_with_ratio():
    result = simulate_csf_pk("d", dose_mg=200.0, logP=3.0, route="iv", fu_plasma=0.5)
    expected = result.csf_to_plasma_auc_ratio > 0.1
    assert result.meningitis_utility == expected


# ---------------------------------------------------------------------------
# Output structure checks
# ---------------------------------------------------------------------------


def test_times_list_length_matches_steps():
    result = simulate_csf_pk("d", dose_mg=100.0, logP=2.0, t_end_h=10.0, dt_h=0.5)
    # n_steps = int(10 / 0.5) + 1 = 21
    assert len(result.times_h) == 21
    assert len(result.c_plasma_mg_L) == 21
    assert len(result.c_csf_mg_L) == 21


def test_times_start_at_zero():
    result = simulate_csf_pk("d", dose_mg=100.0, logP=2.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_cmax_csf_equals_max_of_list():
    result = simulate_csf_pk("d", dose_mg=100.0, logP=2.0)
    assert result.cmax_csf_mg_L == pytest.approx(max(result.c_csf_mg_L), rel=1e-9)


def test_tmax_csf_corresponds_to_cmax():
    result = simulate_csf_pk("d", dose_mg=100.0, logP=2.0)
    idx = result.c_csf_mg_L.index(result.cmax_csf_mg_L)
    assert result.tmax_csf_h == pytest.approx(result.times_h[idx], rel=1e-9)


def test_notes_non_empty():
    result = simulate_csf_pk("d", dose_mg=100.0, logP=2.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_pk("d", dose_mg=0.0, logP=2.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_pk("d", dose_mg=-10.0, logP=2.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_csf_pk("d", dose_mg=100.0, logP=2.0, cl_L_per_h=0.0)


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_csf_pk("d", dose_mg=100.0, logP=2.0, vd_L=0.0)


def test_validation_logP_out_of_range():
    with pytest.raises(ValueError, match="logP"):
        simulate_csf_pk("d", dose_mg=100.0, logP=15.0)


def test_validation_logP_too_low():
    with pytest.raises(ValueError, match="logP"):
        simulate_csf_pk("d", dose_mg=100.0, logP=-10.0)


def test_validation_route_invalid():
    with pytest.raises(ValueError, match="route"):
        simulate_csf_pk("d", dose_mg=100.0, logP=2.0, route="oral")


def test_validation_fu_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_csf_pk("d", dose_mg=100.0, logP=2.0, fu_plasma=0.0)


def test_validation_fu_above_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        simulate_csf_pk("d", dose_mg=100.0, logP=2.0, fu_plasma=1.5)


# ---------------------------------------------------------------------------
# Mass / physical sanity
# ---------------------------------------------------------------------------


def test_concentrations_non_negative():
    result = simulate_csf_pk("d", dose_mg=500.0, logP=2.0, route="iv", t_end_h=72.0)
    assert all(c >= 0 for c in result.c_plasma_mg_L)
    assert all(c >= 0 for c in result.c_csf_mg_L)


def test_drug_name_preserved():
    result = simulate_csf_pk("meropenem", dose_mg=1000.0, logP=-0.5, route="iv")
    assert result.drug_name == "meropenem"
    assert result.dose_mg == pytest.approx(1000.0)
    assert result.route == "iv"
