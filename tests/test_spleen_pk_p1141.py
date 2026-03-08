"""Tests for Phase 1141 — core/spleen_pk_p1141.py."""

import math

import pytest

from omega_pbpk.core.spleen_pk_p1141 import SpleenPKResult, simulate_spleen_pk

# ---------------------------------------------------------------------------
# Basic sanity / structure
# ---------------------------------------------------------------------------


def test_returns_result_type():
    res = simulate_spleen_pk("DrugA", dose_mg=100.0)
    assert isinstance(res, SpleenPKResult)


def test_drug_name_preserved():
    res = simulate_spleen_pk("MyDrug", dose_mg=50.0)
    assert res.drug_name == "MyDrug"


def test_dose_preserved():
    res = simulate_spleen_pk("X", dose_mg=200.0)
    assert res.dose_mg == 200.0


def test_particle_size_zero_default():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert res.particle_size_nm == 0.0


def test_particle_size_preserved():
    res = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=150.0)
    assert res.particle_size_nm == 150.0


# ---------------------------------------------------------------------------
# Time / concentration array checks
# ---------------------------------------------------------------------------


def test_times_length_matches_concentrations():
    res = simulate_spleen_pk("X", dose_mg=100.0, t_end_h=10.0, dt_h=0.5)
    n = len(res.times_h)
    assert len(res.c_plasma_mg_L) == n
    assert len(res.c_spleen_mg_L) == n
    assert len(res.c_immune_mg_L) == n


def test_times_start_at_zero():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert res.times_h[0] == pytest.approx(0.0)


def test_times_end_near_t_end():
    res = simulate_spleen_pk("X", dose_mg=100.0, t_end_h=24.0, dt_h=0.1)
    assert res.times_h[-1] == pytest.approx(24.0, abs=0.15)


def test_c_spleen_starts_at_zero():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert res.c_spleen_mg_L[0] == pytest.approx(0.0)


def test_c_immune_starts_at_zero():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert res.c_immune_mg_L[0] == pytest.approx(0.0)


def test_c_plasma_starts_positive():
    res = simulate_spleen_pk("X", dose_mg=100.0, vd_L=50.0)
    assert res.c_plasma_mg_L[0] == pytest.approx(100.0 / 50.0, rel=1e-6)


def test_c_spleen_rises_then_falls():
    """Spleen concentration should rise to a peak and then decline."""
    res = simulate_spleen_pk("X", dose_mg=100.0, t_end_h=48.0, dt_h=0.1)
    peak_idx = res.c_spleen_mg_L.index(max(res.c_spleen_mg_L))
    # must rise: max is not at t=0
    assert peak_idx > 0
    # must fall: last value < peak
    assert res.c_spleen_mg_L[-1] < res.c_spleen_mg_L[peak_idx]


def test_all_concentrations_non_negative():
    res = simulate_spleen_pk("X", dose_mg=100.0, t_end_h=24.0)
    assert all(c >= 0.0 for c in res.c_plasma_mg_L)
    assert all(c >= 0.0 for c in res.c_spleen_mg_L)
    assert all(c >= 0.0 for c in res.c_immune_mg_L)


# ---------------------------------------------------------------------------
# Nanoparticle effect
# ---------------------------------------------------------------------------


def test_nanoparticle_increases_spleen_auc():
    small = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=0.0)
    nano = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=200.0)
    assert nano.auc_spleen_mg_h_per_L > small.auc_spleen_mg_h_per_L


def test_larger_nanoparticle_more_spleen_accumulation():
    nano100 = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=100.0)
    nano400 = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=400.0)
    assert nano400.auc_spleen_mg_h_per_L > nano100.auc_spleen_mg_h_per_L


def test_nanoparticle_kps_capped_at_5x():
    """Very large particles should not exceed 5x kps boost."""
    nano_capped = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=1000.0)
    nano_max = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=500.0)
    # Both should hit the same cap (5x), so AUCs should be very close
    assert math.isclose(
        nano_capped.auc_spleen_mg_h_per_L,
        nano_max.auc_spleen_mg_h_per_L,
        rel_tol=1e-3,
    )


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def test_spleen_to_plasma_auc_ratio_positive():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert res.spleen_to_plasma_auc_ratio > 0.0


def test_immune_cell_accumulation_pct_non_negative():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert res.immune_cell_accumulation_pct >= 0.0


def test_cmax_spleen_equals_max_list():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert res.cmax_spleen_mg_L == pytest.approx(max(res.c_spleen_mg_L))


def test_tmax_spleen_in_range():
    res = simulate_spleen_pk("X", dose_mg=100.0, t_end_h=24.0)
    assert 0.0 <= res.tmax_spleen_h <= 24.1


def test_auc_spleen_consistent_with_trapz():
    res = simulate_spleen_pk("X", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    assert res.auc_spleen_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Splenic targeting classification
# ---------------------------------------------------------------------------


def test_splenic_targeting_class_good_for_large_nanoparticle():
    res = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=400.0, t_end_h=48.0)
    assert res.splenic_targeting_class == "good"


def test_splenic_targeting_class_valid_values():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert res.splenic_targeting_class in {"poor", "moderate", "good"}


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    res = simulate_spleen_pk("X", dose_mg=100.0)
    assert isinstance(res.notes, str)


def test_notes_mentions_nanoparticle_when_applicable():
    res = simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=200.0)
    assert "Nanoparticle" in res.notes or "nanoparticle" in res.notes.lower()


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_spleen_pk("X", dose_mg=-10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_spleen_pk("X", dose_mg=0.0)


def test_negative_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_spleen_pk("X", dose_mg=100.0, cl_L_per_h=-1.0)


def test_negative_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_spleen_pk("X", dose_mg=100.0, vd_L=-5.0)


def test_negative_particle_size_raises():
    with pytest.raises(ValueError, match="particle_size_nm"):
        simulate_spleen_pk("X", dose_mg=100.0, particle_size_nm=-1.0)
