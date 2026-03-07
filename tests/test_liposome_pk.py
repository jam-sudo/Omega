"""Tests for liposomal PK simulation module (Phase 385)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.liposome_pk import (
    LiposomePKResult,
    compare_liposome_types,
    simulate_liposome_pk,
)

# Shared parameter set for convenience
BASE_PARAMS = dict(
    drug_name="DoxorubicinLipo",
    dose_mg=50.0,
    cl_free_L_per_h=5.0,
    vd_free_L=30.0,
    k_release_per_h=0.1,
    cl_liposome_L_per_h=2.0,
    vd_liposome_L=5.0,
    t_end_h=48.0,
    dt_h=0.1,
)


# ── Basic smoke tests ──────────────────────────────────────────────────────────

def test_simulate_conventional_returns_result():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert isinstance(result, LiposomePKResult)


def test_simulate_pegylated_returns_result():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="pegylated")
    assert isinstance(result, LiposomePKResult)


def test_simulate_targeted_returns_result():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="targeted")
    assert isinstance(result, LiposomePKResult)


def test_result_fields_populated():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert result.drug_name == "DoxorubicinLipo"
    assert result.dose_mg == 50.0
    assert result.liposome_type == "conventional"
    assert len(result.times_h) > 0
    assert len(result.c_liposome_mg_L) == len(result.times_h)
    assert len(result.c_free_mg_L) == len(result.times_h)


# ── PEGylated liposome has longer circulation (higher AUC_liposome) ───────────

def test_pegylated_has_higher_liposome_auc_than_conventional():
    conv = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    peg = simulate_liposome_pk(**BASE_PARAMS, liposome_type="pegylated")
    assert peg.auc_liposome > conv.auc_liposome


def test_pegylated_longer_liposome_half_life():
    conv = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    peg = simulate_liposome_pk(**BASE_PARAMS, liposome_type="pegylated")
    assert peg.t_half_liposome_h > conv.t_half_liposome_h


# ── Targeted liposome has faster release ──────────────────────────────────────

def test_targeted_has_higher_k_release():
    targeted = simulate_liposome_pk(**BASE_PARAMS, liposome_type="targeted")
    assert math.isclose(targeted.k_release_per_h, BASE_PARAMS["k_release_per_h"] * 2.0)


def test_targeted_faster_tmax_than_conventional():
    """Targeted with 2x release rate should reach Cmax_free earlier or equal."""
    conv = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    targeted = simulate_liposome_pk(**BASE_PARAMS, liposome_type="targeted")
    assert targeted.tmax_free_h <= conv.tmax_free_h + 1.0  # allow 1h tolerance for dt artifacts


# ── Free drug concentration starts at 0 ──────────────────────────────────────

def test_free_drug_starts_at_zero():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert result.c_free_mg_L[0] == pytest.approx(0.0, abs=1e-6)


def test_liposome_concentration_starts_at_dose_over_vd():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    expected_c0 = BASE_PARAMS["dose_mg"] / BASE_PARAMS["vd_liposome_L"]
    assert result.c_liposome_mg_L[0] == pytest.approx(expected_c0, rel=1e-4)


# ── Liposome concentration decreases monotonically ───────────────────────────

def test_liposome_concentration_decreases():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    for i in range(1, len(result.c_liposome_mg_L)):
        assert result.c_liposome_mg_L[i] <= result.c_liposome_mg_L[i - 1] + 1e-9


# ── f_released approaches 1 at large t ───────────────────────────────────────

def test_f_released_approaches_1_at_large_t():
    long_params = {**BASE_PARAMS, "t_end_h": 200.0}
    result = simulate_liposome_pk(**long_params, liposome_type="conventional")
    assert result.f_released > 0.9


def test_f_released_between_0_and_1():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert 0.0 <= result.f_released <= 1.0


# ── AUC is positive ───────────────────────────────────────────────────────────

def test_auc_free_positive():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert result.auc_free > 0.0


def test_auc_liposome_positive():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert result.auc_liposome > 0.0


# ── Cmax and tmax are within simulation range ──────────────────────────────────

def test_cmax_free_positive():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert result.cmax_free > 0.0


def test_tmax_free_within_simulation_range():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert 0.0 <= result.tmax_free_h <= BASE_PARAMS["t_end_h"]


# ── compare_liposome_types returns 3 keys ────────────────────────────────────

def test_compare_returns_three_keys():
    results = compare_liposome_types(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_free_L_per_h=5.0,
        vd_free_L=30.0,
        k_release_per_h=0.1,
    )
    assert set(results.keys()) == {"conventional", "pegylated", "targeted"}


def test_compare_all_results_are_liposome_pk_result():
    results = compare_liposome_types(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_free_L_per_h=5.0,
        vd_free_L=30.0,
        k_release_per_h=0.1,
    )
    for key, val in results.items():
        assert isinstance(val, LiposomePKResult), f"{key} is not LiposomePKResult"


def test_compare_pegylated_auc_liposome_greatest():
    results = compare_liposome_types(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_free_L_per_h=5.0,
        vd_free_L=30.0,
        k_release_per_h=0.1,
    )
    assert results["pegylated"].auc_liposome > results["conventional"].auc_liposome


# ── Validation errors ─────────────────────────────────────────────────────────

def test_invalid_liposome_type_raises():
    with pytest.raises(ValueError, match="liposome_type"):
        simulate_liposome_pk(**BASE_PARAMS, liposome_type="stealth")


def test_negative_dose_raises():
    params = {**BASE_PARAMS, "dose_mg": -10.0}
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_liposome_pk(**params, liposome_type="conventional")


def test_zero_cl_free_raises():
    params = {**BASE_PARAMS, "cl_free_L_per_h": 0.0}
    with pytest.raises(ValueError, match="cl_free_L_per_h"):
        simulate_liposome_pk(**params, liposome_type="conventional")


def test_empty_drug_name_raises():
    params = {**BASE_PARAMS, "drug_name": ""}
    with pytest.raises(ValueError, match="drug_name"):
        simulate_liposome_pk(**params, liposome_type="conventional")


def test_negative_t_end_raises():
    params = {**BASE_PARAMS, "t_end_h": -1.0}
    with pytest.raises(ValueError, match="t_end_h"):
        simulate_liposome_pk(**params, liposome_type="conventional")


def test_negative_vd_liposome_raises():
    params = {**BASE_PARAMS, "vd_liposome_L": -5.0}
    with pytest.raises(ValueError, match="vd_liposome_L"):
        simulate_liposome_pk(**params, liposome_type="conventional")


# ── Notes field populated ─────────────────────────────────────────────────────

def test_notes_mention_pegylation():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="pegylated")
    assert "PEG" in result.notes or "pegylated" in result.notes.lower()


def test_notes_not_empty():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert len(result.notes) > 0


# ── Time array consistency ────────────────────────────────────────────────────

def test_time_array_starts_at_zero():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert result.times_h[0] == pytest.approx(0.0)


def test_time_array_ends_at_or_near_t_end():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert result.times_h[-1] == pytest.approx(BASE_PARAMS["t_end_h"], rel=0.05)


# ── Dose proportionality ──────────────────────────────────────────────────────

def test_doubled_dose_doubles_auc_free():
    params_lo = {**BASE_PARAMS, "dose_mg": 50.0}
    params_hi = {**BASE_PARAMS, "dose_mg": 100.0}
    r_lo = simulate_liposome_pk(**params_lo, liposome_type="conventional")
    r_hi = simulate_liposome_pk(**params_hi, liposome_type="conventional")
    ratio = r_hi.auc_free / r_lo.auc_free
    assert 1.9 < ratio < 2.1


# ── Half-life plausibility ────────────────────────────────────────────────────

def test_liposome_half_life_positive():
    result = simulate_liposome_pk(**BASE_PARAMS, liposome_type="conventional")
    assert result.t_half_liposome_h > 0.0
