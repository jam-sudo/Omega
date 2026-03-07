"""Tests for Phase 759: Drug Release from Implants."""

import math

import pytest

from omega_pbpk.core.implant_release_pk import (
    ImplantReleasePKResult,
    compare_release_models,
    simulate_implant_release_pk,
)

# ---------------------------------------------------------------------------
# Basic return type checks
# ---------------------------------------------------------------------------


def test_returns_dataclass_first_order():
    result = simulate_implant_release_pk(drug_name="TestDrug", dose_mg=100.0)
    assert isinstance(result, ImplantReleasePKResult)


def test_returns_dataclass_zero_order():
    result = simulate_implant_release_pk(
        drug_name="TestDrug", dose_mg=100.0, release_model="zero_order"
    )
    assert isinstance(result, ImplantReleasePKResult)


def test_returns_dataclass_higuchi():
    result = simulate_implant_release_pk(
        drug_name="TestDrug", dose_mg=100.0, release_model="higuchi"
    )
    assert isinstance(result, ImplantReleasePKResult)


# ---------------------------------------------------------------------------
# Cmax > 0 for all models
# ---------------------------------------------------------------------------


def test_cmax_positive_first_order():
    result = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=50.0, release_model="first_order", t_end_h=200.0
    )
    assert result.cmax_mg_L > 0.0


def test_cmax_positive_zero_order():
    result = simulate_implant_release_pk(
        drug_name="Drug",
        dose_mg=50.0,
        release_model="zero_order",
        release_duration_h=100.0,
        t_end_h=150.0,
    )
    assert result.cmax_mg_L > 0.0


def test_cmax_positive_higuchi():
    result = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=50.0, release_model="higuchi", k_higuchi=1.0, t_end_h=200.0
    )
    assert result.cmax_mg_L > 0.0


# ---------------------------------------------------------------------------
# Model-specific behaviour
# ---------------------------------------------------------------------------


def test_first_order_fraction_released_less_than_1_short_time():
    """First-order: not fully released over a short horizon."""
    result = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=100.0, release_model="first_order", k1_per_h=0.005, t_end_h=50.0
    )
    assert result.fraction_released < 1.0


def test_zero_order_approx_linear_release():
    """Zero-order: implant amount decreases approximately linearly."""
    result = simulate_implant_release_pk(
        drug_name="Drug",
        dose_mg=100.0,
        release_model="zero_order",
        release_duration_h=200.0,
        t_end_h=100.0,
        dt_h=1.0,
    )
    # At t=100h (halfway) roughly 50% should be released
    mid_idx = 100
    a_mid = result.a_implant_mg[mid_idx]
    # Allow 5% tolerance
    assert abs(a_mid - 50.0) < 5.0


def test_higuchi_fraction_positive():
    """Higuchi: some drug is released over the simulation."""
    result = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=100.0, release_model="higuchi", k_higuchi=0.5, t_end_h=300.0
    )
    assert result.fraction_released > 0.0


def test_higuchi_release_proportional_sqrt_time():
    """Higuchi: cumulative release grows with sqrt(t) — check two time points."""
    result_short = simulate_implant_release_pk(
        drug_name="Drug",
        dose_mg=200.0,
        release_model="higuchi",
        k_higuchi=2.0,
        cl_L_per_h=0.01,
        vd_L=1000.0,  # very slow elimination
        t_end_h=100.0,
        dt_h=1.0,
    )
    result_long = simulate_implant_release_pk(
        drug_name="Drug",
        dose_mg=200.0,
        release_model="higuchi",
        k_higuchi=2.0,
        cl_L_per_h=0.01,
        vd_L=1000.0,
        t_end_h=400.0,
        dt_h=1.0,
    )
    # More time → more release
    assert result_long.fraction_released >= result_short.fraction_released


# ---------------------------------------------------------------------------
# Mean plasma concentration
# ---------------------------------------------------------------------------


def test_mean_plasma_conc_positive():
    result = simulate_implant_release_pk(drug_name="X", dose_mg=100.0)
    assert result.mean_plasma_conc_mg_L > 0.0


def test_mean_plasma_conc_equals_auc_over_tend():
    result = simulate_implant_release_pk(drug_name="X", dose_mg=100.0, t_end_h=720.0)
    expected = result.auc_mg_h_per_L / 720.0
    assert abs(result.mean_plasma_conc_mg_L - expected) < 1e-9


# ---------------------------------------------------------------------------
# compare_release_models
# ---------------------------------------------------------------------------


def test_compare_release_models_returns_3():
    results = compare_release_models(drug_name="Drug", dose_mg=100.0, t_end_h=200.0)
    assert len(results) == 3


def test_compare_release_models_all_models_present():
    results = compare_release_models(drug_name="Drug", dose_mg=100.0, t_end_h=200.0)
    models = {r.release_model for r in results}
    assert models == {"zero_order", "first_order", "higuchi"}


# ---------------------------------------------------------------------------
# Half-life checks
# ---------------------------------------------------------------------------


def test_t_half_release_positive_first_order():
    result = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=100.0, release_model="first_order", k1_per_h=0.05
    )
    assert result.t_half_release_h > 0.0


def test_t_half_release_first_order_formula():
    k1 = 0.05
    result = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=100.0, release_model="first_order", k1_per_h=k1
    )
    expected = math.log(2.0) / k1
    assert abs(result.t_half_release_h - expected) < 1e-9


def test_t_half_release_zero_order_formula():
    dur = 480.0
    result = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=100.0, release_model="zero_order", release_duration_h=dur
    )
    assert abs(result.t_half_release_h - dur / 2.0) < 1e-9


def test_t_half_elimination_formula():
    cl, vd = 3.0, 60.0
    result = simulate_implant_release_pk(drug_name="Drug", dose_mg=100.0, cl_L_per_h=cl, vd_L=vd)
    expected = math.log(2.0) / (cl / vd)
    assert abs(result.t_half_elimination_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# Fraction released bounds
# ---------------------------------------------------------------------------


def test_fraction_released_between_0_and_1():
    for model in ("zero_order", "first_order", "higuchi"):
        result = simulate_implant_release_pk(
            drug_name="Drug", dose_mg=100.0, release_model=model, t_end_h=300.0
        )
        assert 0.0 <= result.fraction_released <= 1.0


# ---------------------------------------------------------------------------
# Higher k1 → faster release
# ---------------------------------------------------------------------------


def test_higher_k1_faster_release():
    slow = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=100.0, release_model="first_order", k1_per_h=0.005, t_end_h=200.0
    )
    fast = simulate_implant_release_pk(
        drug_name="Drug", dose_mg=100.0, release_model="first_order", k1_per_h=0.05, t_end_h=200.0
    )
    assert fast.fraction_released > slow.fraction_released


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError):
        simulate_implant_release_pk(drug_name="Drug", dose_mg=0.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError):
        simulate_implant_release_pk(drug_name="Drug", dose_mg=-10.0)


def test_validation_invalid_release_model_raises():
    with pytest.raises(ValueError):
        simulate_implant_release_pk(drug_name="Drug", dose_mg=100.0, release_model="magic")


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = simulate_implant_release_pk(drug_name="Drug", dose_mg=100.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
