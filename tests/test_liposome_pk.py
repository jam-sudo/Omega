"""Tests for Phase 199: liposome_pk module."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.core.liposome_pk import (
    LiposomePKResult,
    compare_liposome_sizes,
    simulate_liposome_pk,
)


# ---------------------------------------------------------------------------
# Basic smoke test
# ---------------------------------------------------------------------------

def test_simulate_returns_result_type():
    result = simulate_liposome_pk("TestLipo", dose_mg_lipid=100.0, size_nm=100.0)
    assert isinstance(result, LiposomePKResult)


def test_result_fields_present():
    r = simulate_liposome_pk("TestLipo", dose_mg_lipid=50.0, size_nm=80.0)
    assert r.formulation_name == "TestLipo"
    assert r.dose_mg_lipid == 50.0
    assert r.size_nm == 80.0
    assert len(r.times_h) > 0
    assert len(r.c_liposome_mg_L) == len(r.times_h)
    assert len(r.c_free_drug_mg_L) == len(r.times_h)
    assert len(r.c_encapsulated_mg_L) == len(r.times_h)


# ---------------------------------------------------------------------------
# PEGylation effect: longer t_half with more PEG
# ---------------------------------------------------------------------------

def test_pegylated_has_longer_thalf_than_unpegylated():
    r_pegged = simulate_liposome_pk(
        "PEGylated", dose_mg_lipid=100.0, size_nm=100.0, peg_fraction=0.1
    )
    r_bare = simulate_liposome_pk(
        "Bare", dose_mg_lipid=100.0, size_nm=100.0, peg_fraction=0.0
    )
    assert r_pegged.t_half_liposome_h > r_bare.t_half_liposome_h


def test_peg_fraction_zero_vs_max():
    r0 = simulate_liposome_pk("NoPEG", dose_mg_lipid=100.0, size_nm=100.0, peg_fraction=0.0)
    r1 = simulate_liposome_pk("MaxPEG", dose_mg_lipid=100.0, size_nm=100.0, peg_fraction=0.09)
    assert r1.t_half_liposome_h > r0.t_half_liposome_h


# ---------------------------------------------------------------------------
# Size effect: larger liposomes → faster MPS clearance
# ---------------------------------------------------------------------------

def test_larger_size_faster_clearance():
    r_small = simulate_liposome_pk("Small", dose_mg_lipid=100.0, size_nm=80.0)
    r_large = simulate_liposome_pk("Large", dose_mg_lipid=100.0, size_nm=300.0)
    # Larger → higher k_cl_lipo → shorter t_half
    assert r_large.t_half_liposome_h < r_small.t_half_liposome_h


def test_larger_liposome_higher_liver_uptake():
    r_small = simulate_liposome_pk("Small", dose_mg_lipid=100.0, size_nm=80.0, t_end_h=240.0)
    r_large = simulate_liposome_pk("Large", dose_mg_lipid=100.0, size_nm=400.0, t_end_h=240.0)
    assert r_large.liver_uptake_pct > r_small.liver_uptake_pct


# ---------------------------------------------------------------------------
# Drug loading linearity
# ---------------------------------------------------------------------------

def test_drug_loading_linearly_affects_cmax_free():
    r_lo = simulate_liposome_pk(
        "Low", dose_mg_lipid=100.0, size_nm=100.0, drug_loading_pct=5.0, t_end_h=48.0
    )
    r_hi = simulate_liposome_pk(
        "High", dose_mg_lipid=100.0, size_nm=100.0, drug_loading_pct=20.0, t_end_h=48.0
    )
    # 4× more drug should produce ≈4× higher Cmax (linear ODE)
    ratio = r_hi.cmax_free_drug / r_lo.cmax_free_drug
    assert 3.5 < ratio < 4.5


def test_auc_free_drug_proportional_to_loading():
    r_lo = simulate_liposome_pk(
        "Low", dose_mg_lipid=100.0, size_nm=100.0, drug_loading_pct=10.0
    )
    r_hi = simulate_liposome_pk(
        "High", dose_mg_lipid=100.0, size_nm=100.0, drug_loading_pct=20.0
    )
    ratio = r_hi.auc_free_drug / r_lo.auc_free_drug
    assert 1.8 < ratio < 2.2


# ---------------------------------------------------------------------------
# Release rate effect
# ---------------------------------------------------------------------------

def test_faster_release_higher_early_cmax():
    r_slow = simulate_liposome_pk(
        "Slow", dose_mg_lipid=100.0, size_nm=100.0, k_release_per_h=0.01, t_end_h=48.0
    )
    r_fast = simulate_liposome_pk(
        "Fast", dose_mg_lipid=100.0, size_nm=100.0, k_release_per_h=0.2, t_end_h=48.0
    )
    assert r_fast.cmax_free_drug > r_slow.cmax_free_drug


def test_release_thalf_from_k_release():
    k = 0.1
    r = simulate_liposome_pk("Test", dose_mg_lipid=100.0, size_nm=100.0, k_release_per_h=k)
    expected = math.log(2) / k
    assert abs(r.release_t_half_h - expected) < 0.01


# ---------------------------------------------------------------------------
# compare_liposome_sizes
# ---------------------------------------------------------------------------

def test_compare_returns_sorted_desc():
    results = compare_liposome_sizes(
        "LipoSeries",
        dose_mg_lipid=100.0,
        sizes_nm=[50.0, 100.0, 200.0, 400.0],
    )
    assert len(results) == 4
    aucs = [r.auc_free_drug for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_formulation_names_contain_size():
    results = compare_liposome_sizes(
        "Lipo", dose_mg_lipid=100.0, sizes_nm=[100.0, 200.0]
    )
    names = {r.formulation_name for r in results}
    assert any("100" in n for n in names)
    assert any("200" in n for n in names)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_peg_fraction_above_one_raises():
    with pytest.raises(ValueError, match="peg_fraction"):
        simulate_liposome_pk("Bad", dose_mg_lipid=100.0, size_nm=100.0, peg_fraction=1.5)


def test_invalid_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg_lipid"):
        simulate_liposome_pk("Bad", dose_mg_lipid=0.0, size_nm=100.0)


def test_invalid_size_zero_raises():
    with pytest.raises(ValueError, match="size_nm"):
        simulate_liposome_pk("Bad", dose_mg_lipid=100.0, size_nm=0.0)


def test_invalid_drug_loading_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        simulate_liposome_pk("Bad", dose_mg_lipid=100.0, size_nm=100.0, drug_loading_pct=0.0)
    with pytest.raises(ValueError, match="drug_loading_pct"):
        simulate_liposome_pk("Bad", dose_mg_lipid=100.0, size_nm=100.0, drug_loading_pct=100.0)


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------

def test_liposome_concentration_decreases_monotonically():
    r = simulate_liposome_pk("Test", dose_mg_lipid=100.0, size_nm=100.0)
    # c_lipo starts at peak and should generally decrease
    assert r.c_liposome_mg_L[0] >= r.c_liposome_mg_L[-1]


def test_liver_uptake_bounded():
    r = simulate_liposome_pk("Test", dose_mg_lipid=100.0, size_nm=100.0)
    assert 0.0 < r.liver_uptake_pct < 100.0


def test_initial_concentration_matches_dose():
    dose = 100.0
    V_lipo = 3.0
    r = simulate_liposome_pk("Test", dose_mg_lipid=dose, size_nm=100.0)
    assert abs(r.c_liposome_mg_L[0] - dose / V_lipo) < 1e-6
