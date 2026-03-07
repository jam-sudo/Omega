"""Tests for core/pba_model.py — Physiologically-Based Absorption Model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.pba_model import PBAResult, compare_particle_sizes, simulate_pba

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.5,
    pka=4.0,
    compound_type="acid",
    solubility_mg_mL_pH7=1.0,
    particle_radius_um=50.0,
    cl_L_per_h=10.0,
    vd_L=50.0,
    t_end_h=24.0,
    dt_h=0.1,
)


@pytest.fixture()
def default_result() -> PBAResult:
    return simulate_pba(**DEFAULT_KWARGS)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pba(**{**DEFAULT_KWARGS, "dose_mg": 0.0})


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_pba(**{**DEFAULT_KWARGS, "dose_mg": -50.0})


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_pba(**{**DEFAULT_KWARGS, "cl_L_per_h": 0.0})


def test_invalid_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_pba(**{**DEFAULT_KWARGS, "vd_L": -1.0})


def test_invalid_solubility_zero():
    with pytest.raises(ValueError, match="solubility"):
        simulate_pba(**{**DEFAULT_KWARGS, "solubility_mg_mL_pH7": 0.0})


def test_invalid_particle_radius_zero():
    with pytest.raises(ValueError, match="particle_radius"):
        simulate_pba(**{**DEFAULT_KWARGS, "particle_radius_um": 0.0})


def test_invalid_compound_type():
    with pytest.raises(ValueError, match="compound_type"):
        simulate_pba(**{**DEFAULT_KWARGS, "compound_type": "zwitterion"})


# ---------------------------------------------------------------------------
# Result field types
# ---------------------------------------------------------------------------


def test_result_is_pbaresult(default_result):
    assert isinstance(default_result, PBAResult)


def test_times_h_is_list_of_float(default_result):
    assert isinstance(default_result.times_h, list)
    assert all(isinstance(t, float) for t in default_result.times_h)


def test_c_plasma_is_list_of_float(default_result):
    assert isinstance(default_result.c_plasma_mg_L, list)
    assert all(isinstance(c, float) for c in default_result.c_plasma_mg_L)


def test_f_absorbed_per_segment_is_dict(default_result):
    assert isinstance(default_result.f_absorbed_per_segment, dict)


def test_notes_is_list(default_result):
    assert isinstance(default_result.notes, list)


# ---------------------------------------------------------------------------
# Biological plausibility tests
# ---------------------------------------------------------------------------


def test_f_absorbed_total_in_0_1(default_result):
    assert 0.0 <= default_result.f_absorbed_total <= 1.0


def test_segment_fractions_non_negative(default_result):
    for seg, frac in default_result.f_absorbed_per_segment.items():
        assert frac >= 0.0, f"Negative fraction for segment {seg}"


def test_segment_fractions_sum_approx_f_total(default_result):
    total_from_segs = sum(default_result.f_absorbed_per_segment.values())
    assert abs(total_from_segs - default_result.f_absorbed_total) < 1e-6


def test_cmax_positive(default_result):
    assert default_result.cmax_mg_L > 0.0


def test_auc_positive(default_result):
    assert default_result.auc_mg_h_per_L > 0.0


def test_tmax_within_simulation_time(default_result):
    assert 0.0 <= default_result.tmax_h <= 24.0


def test_colon_absorption_smaller_than_si_proximal():
    """Colon has much lower absorption than SI proximal for permeable drug."""
    result = simulate_pba(
        drug_name="HighPerm",
        dose_mg=100.0,
        logP=3.0,
        pka=7.0,
        compound_type="neutral",
        solubility_mg_mL_pH7=5.0,
        particle_radius_um=10.0,
        cl_L_per_h=10.0,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    f = result.f_absorbed_per_segment
    assert f["colon"] < f["si_proximal"], (
        f"Expected colon ({f['colon']:.4f}) < si_proximal ({f['si_proximal']:.4f})"
    )


def test_high_logP_most_absorption_in_si_proximal():
    """High logP drug should show most absorption in SI proximal."""
    result = simulate_pba(
        drug_name="HighLogP",
        dose_mg=100.0,
        logP=4.0,
        pka=4.0,
        compound_type="acid",
        solubility_mg_mL_pH7=2.0,
        particle_radius_um=5.0,
    )
    f = result.f_absorbed_per_segment
    # SI proximal should dominate
    assert f["si_proximal"] > f["colon"]
    assert f["si_proximal"] > f["stomach"]


def test_neutral_compound_no_pH_effect():
    """Neutral compound: absorption should not depend on segment pH (only Peff scaling)."""
    result = simulate_pba(
        drug_name="Neutral",
        dose_mg=100.0,
        logP=2.0,
        pka=7.0,
        compound_type="neutral",
        solubility_mg_mL_pH7=10.0,
        particle_radius_um=5.0,
    )
    # Neutral: solubility is uniform, absorption driven only by Peff×segment scale
    # SI proximal > SI distal > colon
    f = result.f_absorbed_per_segment
    assert f["si_proximal"] >= f["si_distal"] >= f["colon"]


# ---------------------------------------------------------------------------
# compare_particle_sizes tests
# ---------------------------------------------------------------------------


def test_compare_particle_sizes_returns_sorted():
    """Smaller particles → higher f_absorbed (sorted descending)."""
    radii = [100.0, 50.0, 10.0]
    results = compare_particle_sizes(
        drug_name="Drug",
        dose_mg=100.0,
        logP=2.0,
        pka=7.0,
        compound_type="neutral",
        solubility_mg_mL_pH7=0.5,  # low solubility to make dissolution limiting
        particle_radii_um=radii,
    )
    assert len(results) == 3
    # Check sorted descending by f_absorbed_total
    for i in range(len(results) - 1):
        assert results[i].f_absorbed_total >= results[i + 1].f_absorbed_total


def test_compare_particle_sizes_smaller_higher_f():
    """Smallest particle should yield highest (or equal) f_absorbed."""
    radii = [200.0, 10.0]
    results = compare_particle_sizes(
        drug_name="Drug",
        dose_mg=100.0,
        logP=2.0,
        pka=7.0,
        compound_type="neutral",
        solubility_mg_mL_pH7=0.3,
        particle_radii_um=radii,
    )
    # First result should be the one with highest f_absorbed (smallest particle)
    assert results[0].f_absorbed_total >= results[1].f_absorbed_total


def test_compare_particle_sizes_count():
    radii = [5.0, 10.0, 50.0, 100.0]
    results = compare_particle_sizes(
        drug_name="Drug",
        dose_mg=50.0,
        logP=1.5,
        pka=5.0,
        compound_type="base",
        solubility_mg_mL_pH7=1.0,
        particle_radii_um=radii,
    )
    assert len(results) == 4
