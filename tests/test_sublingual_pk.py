"""Tests for sublingual/buccal absorption model (Phase 221)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.sublingual_pk import (
    SublingualPKResult,
    compare_mucosal_oral,
    simulate_sublingual_pk,
)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sublingual_pk("drug", 0.0, 5.0, 30.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_sublingual_pk("drug", -10.0, 5.0, 30.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_sublingual_pk("drug", 10.0, 0.0, 30.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_sublingual_pk("drug", 10.0, 5.0, 0.0)


def test_ka_mucosal_zero_raises():
    with pytest.raises(ValueError, match="ka_mucosal"):
        simulate_sublingual_pk("drug", 10.0, 5.0, 30.0, ka_mucosal_per_h=0.0)


def test_k_swallow_zero_raises():
    with pytest.raises(ValueError, match="k_swallow"):
        simulate_sublingual_pk("drug", 10.0, 5.0, 30.0, k_swallow_per_h=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        simulate_sublingual_pk("drug", 10.0, 5.0, 30.0, route="intravenous")


# ---------------------------------------------------------------------------
# Sublingual simulation correctness
# ---------------------------------------------------------------------------


@pytest.fixture()
def sl_result():
    return simulate_sublingual_pk(
        drug_name="nitroglycerin",
        dose_mg=0.4,
        cl_L_per_h=100.0,
        vd_L=200.0,
        ka_mucosal_per_h=3.0,
        k_swallow_per_h=0.5,
        route="sublingual",
        t_end_h=4.0,
    )


def test_sublingual_returns_dataclass(sl_result):
    assert isinstance(sl_result, SublingualPKResult)


def test_sublingual_drug_name(sl_result):
    assert sl_result.drug_name == "nitroglycerin"


def test_sublingual_fast_onset(sl_result):
    """Sublingual tmax should be < 1 h for fast mucosal ka."""
    assert sl_result.tmax_h < 1.0


def test_sublingual_onset_before_tmax(sl_result):
    """Onset time (20% Cmax) must be < tmax."""
    assert sl_result.onset_time_h < sl_result.tmax_h


def test_sublingual_cmax_positive(sl_result):
    assert sl_result.cmax_mg_L > 0.0


def test_sublingual_auc_positive(sl_result):
    assert sl_result.auc_mg_h_per_L > 0.0


def test_f_absorbed_mucosal_in_range(sl_result):
    assert 0.0 <= sl_result.f_absorbed_mucosal <= 1.0


def test_sublingual_route_field(sl_result):
    assert sl_result.route == "sublingual"


# ---------------------------------------------------------------------------
# Buccal vs sublingual: buccal should be slower
# ---------------------------------------------------------------------------


def test_buccal_slower_than_sublingual():
    common = dict(
        drug_name="drug",
        dose_mg=10.0,
        cl_L_per_h=5.0,
        vd_L=40.0,
        ka_mucosal_per_h=3.0,
        k_swallow_per_h=0.3,
        t_end_h=8.0,
    )
    sl = simulate_sublingual_pk(**common, route="sublingual")
    buc = simulate_sublingual_pk(**common, route="buccal")
    # Buccal has lower ka → later tmax or lower Cmax
    assert buc.tmax_h >= sl.tmax_h or buc.cmax_mg_L <= sl.cmax_mg_L


# ---------------------------------------------------------------------------
# Linearity: 2× dose → 2× Cmax and 2× AUC
# ---------------------------------------------------------------------------


def test_dose_linearity():
    base = simulate_sublingual_pk("d", 10.0, 5.0, 40.0, t_end_h=8.0)
    double = simulate_sublingual_pk("d", 20.0, 5.0, 40.0, t_end_h=8.0)
    assert abs(double.cmax_mg_L / base.cmax_mg_L - 2.0) < 0.05
    assert abs(double.auc_mg_h_per_L / base.auc_mg_h_per_L - 2.0) < 0.05


# ---------------------------------------------------------------------------
# High k_swallow → lower f_absorbed_mucosal
# ---------------------------------------------------------------------------


def test_high_swallow_reduces_mucosal_fraction():
    low_sw = simulate_sublingual_pk("d", 10.0, 5.0, 40.0, k_swallow_per_h=0.1)
    high_sw = simulate_sublingual_pk("d", 10.0, 5.0, 40.0, k_swallow_per_h=5.0)
    assert high_sw.f_absorbed_mucosal < low_sw.f_absorbed_mucosal


# ---------------------------------------------------------------------------
# compare_mucosal_oral
# ---------------------------------------------------------------------------


def test_compare_relative_ba_high_extraction():
    """For high f_hepatic drug, sublingual should give higher relative BA."""
    result = compare_mucosal_oral(
        drug_name="morphine",
        dose_mg=10.0,
        cl_L_per_h=60.0,
        vd_L=200.0,
        f_hepatic=0.7,
    )
    assert result["relative_bioavailability"] > 1.0


def test_compare_relative_ba_zero_extraction():
    """For f_hepatic=0, relative BA should be approximately 1."""
    result = compare_mucosal_oral(
        drug_name="lorazepam",
        dose_mg=2.0,
        cl_L_per_h=5.0,
        vd_L=60.0,
        f_hepatic=0.0,
        ka_mucosal_per_h=1.0,
        k_swallow_per_h=0.01,
    )
    # With no first-pass and near-zero swallowing, both routes absorb ~100%
    assert 0.5 <= result["relative_bioavailability"] <= 2.0


def test_compare_returns_both_results():
    result = compare_mucosal_oral("drug", 10.0, 5.0, 40.0, 0.5)
    assert "sublingual_result" in result
    assert "oral_result" in result
    assert isinstance(result["sublingual_result"], SublingualPKResult)
    assert isinstance(result["oral_result"], SublingualPKResult)


def test_compare_f_hepatic_in_result():
    result = compare_mucosal_oral("drug", 10.0, 5.0, 40.0, 0.6)
    assert result["f_hepatic"] == pytest.approx(0.6)
    assert result["f_oral"] == pytest.approx(0.4)


def test_compare_invalid_f_hepatic():
    with pytest.raises(ValueError, match="f_hepatic"):
        compare_mucosal_oral("drug", 10.0, 5.0, 40.0, f_hepatic=1.5)


# ---------------------------------------------------------------------------
# Result field types
# ---------------------------------------------------------------------------


def test_result_field_types(sl_result):
    assert isinstance(sl_result.times_h, list)
    assert isinstance(sl_result.m_mucosal_mg, list)
    assert isinstance(sl_result.c_plasma_mg_L, list)
    assert isinstance(sl_result.cmax_mg_L, float)
    assert isinstance(sl_result.tmax_h, float)
    assert isinstance(sl_result.auc_mg_h_per_L, float)
    assert isinstance(sl_result.f_absorbed_mucosal, float)
    assert isinstance(sl_result.onset_time_h, float)
    assert isinstance(sl_result.notes, list)
