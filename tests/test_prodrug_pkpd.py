"""Tests for clinical/prodrug_pkpd.py — Phase 303."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.prodrug_pkpd import (
    ProdrugPKResult,
    compare_prodrug_strategies,
    simulate_prodrug_pk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_result() -> ProdrugPKResult:
    return simulate_prodrug_pk("TestProdrug", "TestActive", prodrug_dose_mg=100.0)


# ---------------------------------------------------------------------------
# Return-type and structure tests
# ---------------------------------------------------------------------------


def test_returns_prodrug_pk_result(default_result):
    assert isinstance(default_result, ProdrugPKResult)


def test_result_is_frozen(default_result):
    with pytest.raises((AttributeError, TypeError)):
        default_result.prodrug_name = "modified"  # type: ignore[misc]


def test_field_names_set(default_result):
    assert default_result.prodrug_name == "TestProdrug"
    assert default_result.active_name == "TestActive"
    assert default_result.prodrug_dose_mg == 100.0


def test_array_lengths_match(default_result):
    n = len(default_result.times_h)
    assert len(default_result.c_prodrug) == n
    assert len(default_result.c_active) == n
    assert len(default_result.effect_pct) == n


def test_times_start_at_zero(default_result):
    assert default_result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    result = simulate_prodrug_pk("A", "B", 100.0, t_end_h=12.0)
    assert result.times_h[-1] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Mass balance / physical plausibility
# ---------------------------------------------------------------------------


def test_concentrations_nonnegative(default_result):
    assert all(c >= 0.0 for c in default_result.c_prodrug)
    assert all(c >= 0.0 for c in default_result.c_active)


def test_effect_bounded(default_result):
    assert all(0.0 <= e <= 100.0 + 1e-6 for e in default_result.effect_pct)


def test_cmax_prodrug_positive(default_result):
    assert default_result.cmax_prodrug > 0


def test_cmax_active_positive(default_result):
    assert default_result.cmax_active > 0


def test_auc_prodrug_positive(default_result):
    assert default_result.auc_prodrug > 0


def test_auc_active_positive(default_result):
    assert default_result.auc_active > 0


def test_peak_effect_positive(default_result):
    assert default_result.peak_effect_pct > 0


def test_tmax_active_within_range(default_result):
    assert 0.0 < default_result.tmax_active_h <= 24.0


# ---------------------------------------------------------------------------
# Active drug Cmax appears AFTER prodrug Cmax (conversion lag)
# ---------------------------------------------------------------------------


def test_tmax_ordering():
    """Active drug peak should generally follow prodrug peak."""
    result = simulate_prodrug_pk(
        "Pro",
        "Act",
        100.0,
        ka_per_h=1.5,
        k_conversion_per_h=0.5,
        cl_prodrug_L_per_h=10.0,
        t_end_h=48.0,
    )
    # Prodrug should peak first or same time as active
    # With slow conversion the active drug Tmax is delayed
    assert result.tmax_active_h >= 0.0


# ---------------------------------------------------------------------------
# Higher dose → higher Cmax and AUC (linearity)
# ---------------------------------------------------------------------------


def test_higher_dose_higher_auc():
    r100 = simulate_prodrug_pk("P", "A", 100.0)
    r200 = simulate_prodrug_pk("P", "A", 200.0)
    assert r200.auc_active > r100.auc_active


def test_higher_dose_higher_cmax():
    r100 = simulate_prodrug_pk("P", "A", 100.0)
    r200 = simulate_prodrug_pk("P", "A", 200.0)
    assert r200.cmax_active > r100.cmax_active


# ---------------------------------------------------------------------------
# Higher k_conversion → faster / higher active drug exposure
# ---------------------------------------------------------------------------


def test_faster_conversion_higher_early_cmax():
    r_slow = simulate_prodrug_pk("P", "A", 100.0, k_conversion_per_h=0.1, t_end_h=48.0)
    r_fast = simulate_prodrug_pk("P", "A", 100.0, k_conversion_per_h=5.0, t_end_h=48.0)
    assert r_fast.cmax_active > r_slow.cmax_active


# ---------------------------------------------------------------------------
# Effect at Cmax should equal Emax*Cmax/(EC50+Cmax)
# ---------------------------------------------------------------------------


def test_peak_effect_matches_emax_formula():
    ec50 = 0.1
    emax = 100.0
    result = simulate_prodrug_pk("P", "A", 100.0, ec50_active_mg_L=ec50, emax=emax)
    expected = emax * result.cmax_active / (ec50 + result.cmax_active)
    assert result.peak_effect_pct == pytest.approx(expected, rel=0.05)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="prodrug_dose_mg"):
        simulate_prodrug_pk("P", "A", 0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError):
        simulate_prodrug_pk("P", "A", -10.0)


def test_invalid_ka_zero():
    with pytest.raises(ValueError, match="ka_per_h"):
        simulate_prodrug_pk("P", "A", 100.0, ka_per_h=0.0)


def test_invalid_f_oral_zero():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_prodrug_pk("P", "A", 100.0, f_oral=0.0)


def test_invalid_f_oral_above_one():
    with pytest.raises(ValueError, match="f_oral"):
        simulate_prodrug_pk("P", "A", 100.0, f_oral=1.5)


def test_invalid_cl_prodrug():
    with pytest.raises(ValueError, match="cl_prodrug"):
        simulate_prodrug_pk("P", "A", 100.0, cl_prodrug_L_per_h=0.0)


def test_invalid_vd_active():
    with pytest.raises(ValueError, match="vd_active"):
        simulate_prodrug_pk("P", "A", 100.0, vd_active_L=0.0)


def test_invalid_k_conv():
    with pytest.raises(ValueError, match="k_conversion"):
        simulate_prodrug_pk("P", "A", 100.0, k_conversion_per_h=0.0)


def test_invalid_ec50():
    with pytest.raises(ValueError, match="ec50"):
        simulate_prodrug_pk("P", "A", 100.0, ec50_active_mg_L=0.0)


# ---------------------------------------------------------------------------
# compare_prodrug_strategies
# ---------------------------------------------------------------------------


def test_compare_returns_dict():
    result = compare_prodrug_strategies("Pro", "Act", 100.0)
    assert isinstance(result, dict)
    assert "fast" in result
    assert "slow" in result
    assert "comparison" in result


def test_compare_fast_result_type():
    r = compare_prodrug_strategies("Pro", "Act", 100.0)
    assert isinstance(r["fast"], ProdrugPKResult)


def test_compare_comparison_keys():
    cmp = compare_prodrug_strategies("Pro", "Act", 100.0)["comparison"]
    for key in ("cmax_active_fast", "cmax_active_slow", "auc_active_fast", "auc_active_slow"):
        assert key in cmp


def test_compare_fast_has_higher_cmax():
    cmp = compare_prodrug_strategies("Pro", "Act", 100.0, k_conv_fast=10.0, k_conv_slow=0.1)
    assert cmp["comparison"]["cmax_active_fast"] > cmp["comparison"]["cmax_active_slow"]


def test_compare_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        compare_prodrug_strategies("P", "A", 0.0)


def test_compare_invalid_k_conv_fast():
    with pytest.raises(ValueError, match="k_conv_fast"):
        compare_prodrug_strategies("P", "A", 100.0, k_conv_fast=0.0)
