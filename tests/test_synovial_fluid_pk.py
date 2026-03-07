"""Tests for Phase 766: Synovial fluid PK model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.synovial_fluid_pk import (
    SynovialFluidPKResult,
    compare_antiarthritic_drugs,
    simulate_synovial_fluid_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_KWARGS = dict(
    drug_name="Diclofenac",
    dose_mg=100.0,
    logp=2.0,
    mw_da=300.0,
    cl_L_per_h=5.0,
    vd_plasma_L=10.0,
    ps_syn_L_per_h=0.05,
    v_sf_L=0.002,
    ke_sf_per_h=0.1,
    t_end_h=24.0,
    dt_h=0.05,
)


def _default_result() -> SynovialFluidPKResult:
    return simulate_synovial_fluid_pk(**DEFAULT_KWARGS)


# ---------------------------------------------------------------------------
# Basic return-type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _default_result()
    assert isinstance(result, SynovialFluidPKResult)


def test_times_list_nonempty():
    result = _default_result()
    assert len(result.times_h) > 0


def test_c_plasma_list_matches_times():
    result = _default_result()
    assert len(result.c_plasma_mg_L) == len(result.times_h)


def test_c_sf_list_matches_times():
    result = _default_result()
    assert len(result.c_sf_mg_L) == len(result.times_h)


# ---------------------------------------------------------------------------
# Scalar metric sanity
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = _default_result()
    assert result.auc_plasma > 0.0


def test_auc_sf_positive():
    result = _default_result()
    assert result.auc_sf > 0.0


def test_cmax_plasma_positive():
    result = _default_result()
    assert result.cmax_plasma > 0.0


def test_cmax_sf_positive():
    result = _default_result()
    assert result.cmax_sf > 0.0


def test_cmax_plasma_greater_than_cmax_sf():
    """Plasma drives distribution; plasma Cmax should exceed SF Cmax.

    Use a finer dt to avoid forward-Euler overshoot in the tiny SF volume.
    """
    result = simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "dt_h": 0.001})
    assert result.cmax_plasma > result.cmax_sf


def test_sf_to_plasma_auc_ratio_between_0_and_2():
    result = _default_result()
    assert 0.0 < result.sf_to_plasma_auc_ratio < 2.0


def test_kp_sf_between_0_and_2():
    result = _default_result()
    assert 0.0 < result.kp_sf < 2.0


def test_sf_penetration_index_between_0_and_200():
    result = _default_result()
    assert 0.0 < result.sf_penetration_index < 200.0


def test_therapeutic_adequacy_valid():
    result = _default_result()
    assert result.therapeutic_adequacy in {"poor", "moderate", "good"}


def test_notes_nonempty():
    result = _default_result()
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ---------------------------------------------------------------------------
# Physical / pharmacological behaviour
# ---------------------------------------------------------------------------


def test_tmax_sf_greater_than_tmax_plasma():
    """SF equilibration is delayed; tmax in SF should be later than in plasma."""
    result = _default_result()
    # For IV bolus plasma tmax is 0 (initial peak), SF tmax is later
    assert result.tmax_sf_h > result.tmax_plasma_h


def test_higher_ps_syn_gives_higher_sf_penetration():
    """Higher synovium permeability → more drug enters SF."""
    r_low = simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "ps_syn_L_per_h": 0.01})
    r_high = simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "ps_syn_L_per_h": 0.5})
    assert r_high.sf_to_plasma_auc_ratio > r_low.sf_to_plasma_auc_ratio


def test_linear_dose_scaling():
    """Doubling the dose should double AUC in plasma and SF."""
    r1 = simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "dose_mg": 100.0})
    r2 = simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "dose_mg": 200.0})
    assert abs(r2.auc_plasma / r1.auc_plasma - 2.0) < 0.05
    assert abs(r2.auc_sf / r1.auc_sf - 2.0) < 0.05


def test_sf_to_plasma_auc_ratio_dose_invariant():
    """SF/plasma AUC ratio should be independent of dose (linear PK)."""
    r1 = simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "dose_mg": 50.0})
    r2 = simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "dose_mg": 200.0})
    assert abs(r1.sf_to_plasma_auc_ratio - r2.sf_to_plasma_auc_ratio) < 0.01


# ---------------------------------------------------------------------------
# compare_antiarthritic_drugs
# ---------------------------------------------------------------------------


def test_compare_antiarthritic_drugs_returns_list():
    drugs = [
        {"drug_name": "DrugA", "dose_mg": 100.0, "ps_syn_L_per_h": 0.1},
        {"drug_name": "DrugB", "dose_mg": 100.0, "ps_syn_L_per_h": 0.3},
        {"drug_name": "DrugC", "dose_mg": 100.0, "ps_syn_L_per_h": 0.05},
    ]
    results = compare_antiarthritic_drugs(drugs)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_antiarthritic_drugs_sorted_descending():
    """Results sorted by sf_to_plasma_auc_ratio descending."""
    drugs = [
        {"drug_name": "DrugA", "dose_mg": 100.0, "ps_syn_L_per_h": 0.05},
        {"drug_name": "DrugB", "dose_mg": 100.0, "ps_syn_L_per_h": 0.5},
        {"drug_name": "DrugC", "dose_mg": 100.0, "ps_syn_L_per_h": 0.2},
    ]
    results = compare_antiarthritic_drugs(drugs)
    ratios = [r.sf_to_plasma_auc_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


# ---------------------------------------------------------------------------
# Validation (errors)
# ---------------------------------------------------------------------------


def test_validation_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "dose_mg": 0.0})


def test_validation_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "dose_mg": -50.0})


def test_validation_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "vd_plasma_L": 0.0})


def test_validation_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "cl_L_per_h": 0.0})


def test_validation_zero_ps_raises():
    with pytest.raises(ValueError, match="ps_syn_L_per_h"):
        simulate_synovial_fluid_pk(**{**DEFAULT_KWARGS, "ps_syn_L_per_h": 0.0})


# ---------------------------------------------------------------------------
# Initial conditions (IV bolus)
# ---------------------------------------------------------------------------


def test_plasma_starts_at_dose_over_vd():
    result = _default_result()
    expected = DEFAULT_KWARGS["dose_mg"] / DEFAULT_KWARGS["vd_plasma_L"]
    assert abs(result.c_plasma_mg_L[0] - expected) < 1e-9


def test_sf_starts_at_zero():
    result = _default_result()
    assert result.c_sf_mg_L[0] == 0.0
