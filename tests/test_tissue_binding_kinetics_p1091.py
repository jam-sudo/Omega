"""Tests for core/tissue_binding_kinetics_p1091.py — Phase 1091.

Two-site tissue binding kinetics: non-specific (site 1) + specific/receptor (site 2).
"""

from __future__ import annotations

import pytest

from omega_pbpk.core.tissue_binding_kinetics_p1091 import (
    TissueBindingResult,
    compare_affinities,
    simulate_tissue_binding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_result(**kwargs) -> TissueBindingResult:
    return simulate_tissue_binding(drug_name="TestDrug", dose_mg=100.0, **kwargs)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_return_type():
    r = default_result()
    assert isinstance(r, TissueBindingResult)


def test_result_fields_present():
    r = default_result()
    assert hasattr(r, "c_free_mg_L")
    assert hasattr(r, "b_site1_mg_L")
    assert hasattr(r, "b_site2_mg_L")
    assert hasattr(r, "c_total_mg_L")
    assert hasattr(r, "fu_tissue_list")
    assert hasattr(r, "notes")


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def test_c_free_starts_at_dose_over_v():
    r = simulate_tissue_binding(drug_name="DrugA", dose_mg=50.0, v_tissue_L=5.0)
    assert abs(r.c_free_mg_L[0] - 50.0 / 5.0) < 1e-10


def test_c_total_starts_at_dose_over_v():
    """At t=0 there is no binding yet (B1=B2=0), so c_total = c_free."""
    r = simulate_tissue_binding(drug_name="DrugA", dose_mg=80.0, v_tissue_L=8.0)
    assert abs(r.c_total_mg_L[0] - 80.0 / 8.0) < 1e-10


def test_b1_b2_start_at_zero():
    r = default_result()
    assert r.b_site1_mg_L[0] == pytest.approx(0.0)
    assert r.b_site2_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Binding dynamics
# ---------------------------------------------------------------------------

def test_site2_occupancy_increases_in_first_hour():
    r = default_result()
    # occupancy at 1h should be > 0 (binding occurs)
    assert r.occupancy_site2_at_1h > 0.0


def test_fu_tissue_at_1h_less_than_1():
    """Some binding occurs so fu < 1.0."""
    r = default_result()
    assert r.fu_tissue_at_1h < 1.0


def test_binding_occurs_positive_b1():
    r = default_result()
    max_b1 = max(r.b_site1_mg_L)
    assert max_b1 > 0.0


def test_binding_occurs_positive_b2():
    r = default_result()
    max_b2 = max(r.b_site2_mg_L)
    assert max_b2 > 0.0


def test_c_total_decreases_over_time():
    """Overall drug is eliminated, so c_total should trend down."""
    r = default_result()
    assert r.c_total_mg_L[-1] < r.c_total_mg_L[0]


# ---------------------------------------------------------------------------
# List lengths
# ---------------------------------------------------------------------------

def test_fu_tissue_list_same_length_as_times():
    r = default_result()
    assert len(r.fu_tissue_list) == len(r.times_h)


def test_c_free_same_length_as_times():
    r = default_result()
    assert len(r.c_free_mg_L) == len(r.times_h)


def test_c_total_same_length_as_times():
    r = default_result()
    assert len(r.c_total_mg_L) == len(r.times_h)


def test_b1_b2_same_length_as_times():
    r = default_result()
    assert len(r.b_site1_mg_L) == len(r.times_h)
    assert len(r.b_site2_mg_L) == len(r.times_h)


# ---------------------------------------------------------------------------
# Equilibrium & metrics
# ---------------------------------------------------------------------------

def test_t_binding_equilibrium_positive():
    r = default_result()
    assert r.t_binding_equilibrium_h > 0.0


def test_kp_apparent_at_1h_positive():
    r = default_result()
    assert r.kp_apparent_at_1h > 0.0


def test_occupancy_site2_in_range():
    r = default_result()
    assert 0.0 <= r.occupancy_site2_at_1h <= 1.0
    assert 0.0 <= r.occupancy_site2_at_24h <= 1.0


def test_notes_nonempty():
    r = default_result()
    assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Affinity comparison
# ---------------------------------------------------------------------------

def test_higher_koff2_lower_occupancy_at_24h():
    """Higher koff2 = weaker affinity = lower occupancy."""
    r_high_affinity = simulate_tissue_binding(
        drug_name="Drug", dose_mg=100.0, koff2=0.01
    )
    r_low_affinity = simulate_tissue_binding(
        drug_name="Drug", dose_mg=100.0, koff2=2.0
    )
    assert r_high_affinity.occupancy_site2_at_24h >= r_low_affinity.occupancy_site2_at_24h


def test_compare_affinities_returns_correct_count():
    results = compare_affinities("Drug", 100.0, [0.01, 0.1, 1.0, 5.0])
    assert len(results) == 4


def test_compare_affinities_sorted_descending():
    results = compare_affinities("Drug", 100.0, [0.5, 0.05, 5.0])
    occupancies = [r.occupancy_site2_at_24h for r in results]
    assert occupancies == sorted(occupancies, reverse=True)


def test_compare_affinities_all_tbresults():
    results = compare_affinities("Drug", 100.0, [0.1, 1.0])
    for r in results:
        assert isinstance(r, TissueBindingResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tissue_binding(drug_name="D", dose_mg=0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tissue_binding(drug_name="D", dose_mg=-10.0)


def test_invalid_v_tissue():
    with pytest.raises(ValueError, match="v_tissue_L"):
        simulate_tissue_binding(drug_name="D", dose_mg=100.0, v_tissue_L=0.0)


def test_invalid_cl_tissue():
    with pytest.raises(ValueError, match="cl_tissue_L_per_h"):
        simulate_tissue_binding(drug_name="D", dose_mg=100.0, cl_tissue_L_per_h=-1.0)


def test_invalid_kon1():
    with pytest.raises(ValueError, match="kon1"):
        simulate_tissue_binding(drug_name="D", dose_mg=100.0, kon1=0.0)


def test_invalid_koff2():
    with pytest.raises(ValueError, match="koff2"):
        simulate_tissue_binding(drug_name="D", dose_mg=100.0, koff2=-0.1)
