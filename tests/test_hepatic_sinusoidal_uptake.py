"""Tests for Phase 317 — Hepatic Sinusoidal Uptake Transporter Model."""

import pytest

from omega_pbpk.core.hepatic_sinusoidal_uptake import (
    HepaticUptakeResult,
    compare_oatp_inhibition,
    simulate_hepatic_uptake,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMMON_KWARGS = dict(
    drug_name="atorvastatin",
    dose_mg=40.0,
    route="oral",
    cl_passive_L_per_h=1.0,
    vmax_oatp_mg_per_h=20.0,
    km_oatp_mg_per_L=0.5,
    vd_plasma_L=5.0,
    vd_hepatic_L=1.0,
    t_end_h=24.0,
    dt_h=0.05,
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert isinstance(result, HepaticUptakeResult)


def test_drug_name_preserved():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.drug_name == "atorvastatin"


def test_dose_preserved():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.dose_mg == 40.0


def test_route_preserved():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.route == "oral"


def test_vmax_km_preserved():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.vmax_oatp_mg_per_h == 20.0
    assert result.km_oatp_mg_per_L == 0.5


# ---------------------------------------------------------------------------
# Time arrays
# ---------------------------------------------------------------------------


def test_times_length_matches_concentrations():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_hepatic_mg_L)


def test_times_start_at_zero():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_are_positive():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert all(t >= 0.0 for t in result.times_h)


# ---------------------------------------------------------------------------
# Core PK outputs
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_cmax_plasma_positive():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.cmax_plasma_mg_L > 0.0


def test_auc_hepatic_positive():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.auc_hepatic_mg_h_per_L > 0.0


def test_cmax_hepatic_positive():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.cmax_hepatic_mg_L > 0.0


def test_hepatic_to_plasma_ratio_positive():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.hepatic_to_plasma_ratio > 0.0


def test_hepatic_concentrating_effect():
    """Drug should accumulate in liver (ratio > 1 for OATP substrates)."""
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert result.hepatic_to_plasma_ratio > 1.0


# ---------------------------------------------------------------------------
# Pharmacological relationships
# ---------------------------------------------------------------------------


def test_higher_vmax_lower_plasma_auc():
    """Higher Vmax means more uptake from plasma → lower plasma AUC."""
    low_vmax = simulate_hepatic_uptake(**{**COMMON_KWARGS, "vmax_oatp_mg_per_h": 5.0})
    high_vmax = simulate_hepatic_uptake(**{**COMMON_KWARGS, "vmax_oatp_mg_per_h": 50.0})
    assert high_vmax.auc_plasma_mg_h_per_L < low_vmax.auc_plasma_mg_h_per_L


def test_higher_vmax_higher_hepatic_auc():
    """Higher Vmax means more uptake into liver → higher hepatic AUC."""
    low_vmax = simulate_hepatic_uptake(**{**COMMON_KWARGS, "vmax_oatp_mg_per_h": 5.0})
    high_vmax = simulate_hepatic_uptake(**{**COMMON_KWARGS, "vmax_oatp_mg_per_h": 50.0})
    assert high_vmax.auc_hepatic_mg_h_per_L > low_vmax.auc_hepatic_mg_h_per_L


def test_iv_route_no_gut_absorption():
    """IV bolus should yield higher initial plasma concentration than oral."""
    iv_result = simulate_hepatic_uptake(**{**COMMON_KWARGS, "route": "iv_bolus"})
    oral_result = simulate_hepatic_uptake(**{**COMMON_KWARGS, "route": "oral"})
    assert iv_result.cmax_plasma_mg_L > oral_result.cmax_plasma_mg_L


# ---------------------------------------------------------------------------
# compare_oatp_inhibition
# ---------------------------------------------------------------------------


def test_compare_oatp_inhibition_returns_list():
    fracs = [0.0, 0.5, 0.9]
    results = compare_oatp_inhibition(
        drug_name="rosuvastatin",
        dose_mg=20.0,
        inhibition_fractions=fracs,
        cl_passive_L_per_h=0.5,
        vmax_oatp_mg_per_h=15.0,
        km_oatp_mg_per_L=0.3,
        vd_plasma_L=4.0,
        vd_hepatic_L=0.8,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_oatp_inhibition_sorted_descending():
    """Results must be sorted by auc_plasma descending."""
    fracs = [0.0, 0.5, 0.9]
    results = compare_oatp_inhibition(
        drug_name="rosuvastatin",
        dose_mg=20.0,
        inhibition_fractions=fracs,
        cl_passive_L_per_h=0.5,
        vmax_oatp_mg_per_h=15.0,
        km_oatp_mg_per_L=0.3,
        vd_plasma_L=4.0,
        vd_hepatic_L=0.8,
    )
    aucs = [r.auc_plasma_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_higher_inhibition_higher_plasma_auc():
    """90% inhibition should yield higher plasma AUC than no inhibition."""
    fracs = [0.0, 0.9]
    results = compare_oatp_inhibition(
        drug_name="rosuvastatin",
        dose_mg=20.0,
        inhibition_fractions=fracs,
        cl_passive_L_per_h=0.5,
        vmax_oatp_mg_per_h=15.0,
        km_oatp_mg_per_L=0.3,
        vd_plasma_L=4.0,
        vd_hepatic_L=0.8,
    )
    # After sort descending, first entry has highest plasma AUC
    assert results[0].auc_plasma_mg_h_per_L > results[1].auc_plasma_mg_h_per_L


def test_compare_returns_hepatic_uptake_results():
    fracs = [0.0, 0.5]
    results = compare_oatp_inhibition(
        drug_name="simvastatin",
        dose_mg=20.0,
        inhibition_fractions=fracs,
        cl_passive_L_per_h=0.5,
        vmax_oatp_mg_per_h=10.0,
        km_oatp_mg_per_L=0.4,
        vd_plasma_L=3.0,
        vd_hepatic_L=0.6,
    )
    for r in results:
        assert isinstance(r, HepaticUptakeResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "route": "subcutaneous"})


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "dose_mg": 0.0})


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "dose_mg": -5.0})


def test_invalid_cl_passive_negative():
    with pytest.raises(ValueError, match="cl_passive_L_per_h"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "cl_passive_L_per_h": -1.0})


def test_invalid_vmax_zero():
    with pytest.raises(ValueError, match="vmax_oatp_mg_per_h"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "vmax_oatp_mg_per_h": 0.0})


def test_invalid_km_zero():
    with pytest.raises(ValueError, match="km_oatp_mg_per_L"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "km_oatp_mg_per_L": 0.0})


def test_invalid_vd_plasma_zero():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "vd_plasma_L": 0.0})


def test_invalid_vd_hepatic_zero():
    with pytest.raises(ValueError, match="vd_hepatic_L"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "vd_hepatic_L": 0.0})


def test_invalid_t_end_zero():
    with pytest.raises(ValueError, match="t_end_h"):
        simulate_hepatic_uptake(**{**COMMON_KWARGS, "t_end_h": 0.0})


def test_notes_is_string():
    result = simulate_hepatic_uptake(**COMMON_KWARGS)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
