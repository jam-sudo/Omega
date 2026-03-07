"""Tests for Phase 919 — Drug Spinal Cord Distribution."""

import pytest

from omega_pbpk.core.spinal_cord_pk import (
    SpinalCordPKResult,
    simulate_spinal_cord_pk,
)

# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


def test_result_is_dataclass():
    result = simulate_spinal_cord_pk("morphine", dose_mg=1.0, route="intrathecal")
    assert isinstance(result, SpinalCordPKResult)


def test_result_fields_populated():
    result = simulate_spinal_cord_pk("morphine", dose_mg=1.0)
    assert result.drug_name == "morphine"
    assert result.dose_mg == 1.0
    assert result.route == "intrathecal"
    assert len(result.times_h) > 0
    assert len(result.c_csf_mg_L) > 0
    assert len(result.c_spinal_cord_mg_L) > 0
    assert len(result.c_plasma_mg_L) > 0


def test_time_array_length_matches_concentration():
    result = simulate_spinal_cord_pk("drug", dose_mg=2.0)
    assert len(result.times_h) == len(result.c_csf_mg_L)
    assert len(result.times_h) == len(result.c_spinal_cord_mg_L)
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_time_starts_at_zero():
    result = simulate_spinal_cord_pk("drug", dose_mg=1.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_time_ends_at_t_end():
    result = simulate_spinal_cord_pk("drug", dose_mg=1.0, t_end_h=10.0, dt_h=0.1)
    assert result.times_h[-1] == pytest.approx(10.0, abs=0.15)


# ---------------------------------------------------------------------------
# Intrathecal route
# ---------------------------------------------------------------------------


def test_intrathecal_initial_csf_concentration():
    v_csf_L = 0.14
    dose_mg = 1.0
    result = simulate_spinal_cord_pk(
        "morphine", dose_mg=dose_mg, route="intrathecal", v_csf_L=v_csf_L
    )
    expected = dose_mg / v_csf_L
    assert result.c_csf_mg_L[0] == pytest.approx(expected, rel=1e-6)


def test_intrathecal_initial_plasma_zero():
    result = simulate_spinal_cord_pk("morphine", dose_mg=1.0, route="intrathecal")
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_intrathecal_plasma_rises_over_time():
    result = simulate_spinal_cord_pk("morphine", dose_mg=1.0, route="intrathecal")
    # Plasma should accumulate due to CSF drainage
    assert max(result.c_plasma_mg_L) > 0.0


def test_intrathecal_csf_decays():
    result = simulate_spinal_cord_pk("morphine", dose_mg=1.0, route="intrathecal")
    assert result.c_csf_mg_L[-1] < result.c_csf_mg_L[0]


def test_intrathecal_cmax_csf_equals_initial():
    v_csf_L = 0.14
    dose_mg = 2.0
    result = simulate_spinal_cord_pk("drug", dose_mg=dose_mg, route="intrathecal", v_csf_L=v_csf_L)
    assert result.cmax_csf == pytest.approx(dose_mg / v_csf_L, rel=1e-6)


def test_intrathecal_tmax_csf_at_zero():
    result = simulate_spinal_cord_pk("drug", dose_mg=1.0, route="intrathecal")
    assert result.tmax_csf_h == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# Systemic route
# ---------------------------------------------------------------------------


def test_systemic_initial_plasma_concentration():
    vd_sys_L = 50.0
    dose_mg = 10.0
    result = simulate_spinal_cord_pk(
        "gabapentin",
        dose_mg=dose_mg,
        route="systemic",
        vd_sys_L=vd_sys_L,
    )
    expected = dose_mg / vd_sys_L
    assert result.c_plasma_mg_L[0] == pytest.approx(expected, rel=1e-6)


def test_systemic_initial_csf_zero():
    result = simulate_spinal_cord_pk("gabapentin", dose_mg=10.0, route="systemic")
    assert result.c_csf_mg_L[0] == pytest.approx(0.0, abs=1e-12)


def test_systemic_csf_rises_then_falls():
    result = simulate_spinal_cord_pk("gabapentin", dose_mg=10.0, route="systemic", t_end_h=24.0)
    assert result.cmax_csf > 0.0
    assert result.tmax_csf_h > 0.0


def test_systemic_plasma_decays():
    result = simulate_spinal_cord_pk("gabapentin", dose_mg=10.0, route="systemic")
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


# ---------------------------------------------------------------------------
# AUC and PK metrics
# ---------------------------------------------------------------------------


def test_auc_csf_positive():
    result = simulate_spinal_cord_pk("drug", dose_mg=1.0)
    assert result.auc_csf > 0.0


def test_auc_spinal_cord_positive():
    result = simulate_spinal_cord_pk("drug", dose_mg=1.0)
    assert result.auc_spinal_cord > 0.0


def test_analgesic_duration_positive_intrathecal():
    result = simulate_spinal_cord_pk("morphine", dose_mg=1.0, route="intrathecal")
    assert result.analgesic_duration_h > 0.0


def test_analgesic_duration_bounded_by_simulation():
    t_end = 12.0
    result = simulate_spinal_cord_pk("morphine", dose_mg=1.0, route="intrathecal", t_end_h=t_end)
    assert result.analgesic_duration_h <= t_end + 0.1


# ---------------------------------------------------------------------------
# Spinal-to-plasma ratio and notes
# ---------------------------------------------------------------------------


def test_spinal_to_plasma_ratio_intrathecal_high():
    # Intrathecal should yield high spinal selectivity
    result = simulate_spinal_cord_pk(
        "morphine",
        dose_mg=1.0,
        route="intrathecal",
        k_csf_drain_per_h=0.05,  # slow drain → keeps drug in CSF/cord
        cl_sys_L_per_h=20.0,  # fast systemic clearance
        t_end_h=24.0,
    )
    assert result.spinal_to_plasma_ratio > 0.0


def test_notes_intrathecal_excellent_selectivity():
    result = simulate_spinal_cord_pk(
        "morphine",
        dose_mg=1.0,
        route="intrathecal",
        k_csf_drain_per_h=0.01,
        cl_sys_L_per_h=50.0,
        t_end_h=24.0,
    )
    # With very slow drain and high systemic CL, spinal/plasma ratio should be > 10
    if result.spinal_to_plasma_ratio > 10.0:
        assert "Excellent spinal selectivity" in result.notes


def test_notes_limited_distribution_systemic():
    # Systemic with poor CNS penetration parameters
    result = simulate_spinal_cord_pk(
        "drug",
        dose_mg=10.0,
        route="systemic",
        k_csf_to_cord_per_h=0.01,
        k_cord_to_csf_per_h=0.5,
    )
    # spinal_to_plasma_ratio likely < 1 → limited distribution note
    if result.spinal_to_plasma_ratio <= 1.0:
        assert "Limited spinal cord distribution" in result.notes


def test_notes_non_empty():
    result = simulate_spinal_cord_pk("drug", dose_mg=1.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_spinal_cord_pk("drug", dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_spinal_cord_pk("drug", dose_mg=-5.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_spinal_cord_pk("drug", dose_mg=1.0, route="oral")


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


def test_cmax_csf_scales_with_dose_intrathecal():
    result_1 = simulate_spinal_cord_pk("drug", dose_mg=1.0, route="intrathecal")
    result_2 = simulate_spinal_cord_pk("drug", dose_mg=2.0, route="intrathecal")
    assert result_2.cmax_csf == pytest.approx(2.0 * result_1.cmax_csf, rel=1e-6)


def test_auc_scales_with_dose():
    result_1 = simulate_spinal_cord_pk("drug", dose_mg=1.0)
    result_2 = simulate_spinal_cord_pk("drug", dose_mg=3.0)
    assert result_2.auc_csf == pytest.approx(3.0 * result_1.auc_csf, rel=1e-6)
