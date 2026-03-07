"""Tests for inflammation PK/PD model — Phase 269."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.inflammation_pd import InflammationPDResult, simulate_inflammation_pd

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_inflammation_pd("Drug", dose_mg=0, cl_L_per_h=1.0, vd_L=10.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_inflammation_pd("Drug", dose_mg=-1.0, cl_L_per_h=1.0, vd_L=10.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
        simulate_inflammation_pd("Drug", dose_mg=100.0, cl_L_per_h=0.0, vd_L=10.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        simulate_inflammation_pd("Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=0.0)


def test_ic50_zero_raises():
    with pytest.raises(ValueError, match="ic50_mg_L must be > 0"):
        simulate_inflammation_pd("Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0, ic50_mg_L=0.0)


def test_imax_zero_raises():
    with pytest.raises(ValueError, match="imax must be in"):
        simulate_inflammation_pd("Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0, imax=0.0)


def test_imax_gt_one_raises():
    with pytest.raises(ValueError, match="imax must be in"):
        simulate_inflammation_pd("Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0, imax=1.5)


def test_k_cyto_prod_zero_raises():
    with pytest.raises(ValueError, match="k_cyto_prod must be > 0"):
        simulate_inflammation_pd("Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0, k_cyto_prod=0.0)


def test_k_cyto_deg_zero_raises():
    with pytest.raises(ValueError, match="k_cyto_deg must be > 0"):
        simulate_inflammation_pd("Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0, k_cyto_deg=0.0)


def test_baseline_cyto_zero_raises():
    with pytest.raises(ValueError, match="baseline_cyto must be > 0"):
        simulate_inflammation_pd(
            "Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0, baseline_cyto=0.0
        )


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route must be"):
        simulate_inflammation_pd("Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0, route="sc")


def test_f_oral_zero_raises():
    with pytest.raises(ValueError, match="f_oral must be in"):
        simulate_inflammation_pd(
            "Drug", dose_mg=100.0, cl_L_per_h=1.0, vd_L=10.0, route="oral", f_oral=0.0
        )


# ---------------------------------------------------------------------------
# Basic correctness — IV route
# ---------------------------------------------------------------------------


@pytest.fixture
def iv_result():
    return simulate_inflammation_pd(
        "TestDrug",
        dose_mg=100.0,
        cl_L_per_h=2.0,
        vd_L=20.0,
        ic50_mg_L=0.5,
        imax=0.9,
        t_end_h=24.0,
        dt_h=0.05,
    )


def test_iv_initial_concentration(iv_result):
    """IV dose/vd gives correct initial plasma concentration."""
    expected = 100.0 / 20.0  # 5 mg/L
    assert abs(iv_result.c_plasma_mg_L[0] - expected) < 1e-9


def test_cmax_positive(iv_result):
    assert iv_result.cmax > 0.0


def test_auc_positive(iv_result):
    assert iv_result.auc_plasma > 0.0


def test_cytokine_starts_at_baseline(iv_result):
    assert abs(iv_result.cytokine_level[0] - 1.0) < 1e-9


def test_peak_suppression_positive_with_high_dose(iv_result):
    """High-dose simulation should show meaningful cytokine suppression."""
    assert iv_result.peak_cytokine_suppression > 0.0


def test_drug_effect_within_bounds(iv_result):
    imax = 0.9
    for eff in iv_result.drug_effect:
        assert 0.0 <= eff <= imax + 1e-9


def test_array_lengths_equal(iv_result):
    n = len(iv_result.times_h)
    assert len(iv_result.c_plasma_mg_L) == n
    assert len(iv_result.cytokine_level) == n
    assert len(iv_result.drug_effect) == n


def test_peak_suppression_in_valid_range(iv_result):
    assert 0.0 <= iv_result.peak_cytokine_suppression <= 100.0


def test_time_below_50pct_nonnegative(iv_result):
    assert iv_result.time_cytokine_below_50pct_h >= 0.0


def test_drug_name_stored(iv_result):
    assert iv_result.drug_name == "TestDrug"


# ---------------------------------------------------------------------------
# Higher dose → greater suppression
# ---------------------------------------------------------------------------


def test_higher_dose_greater_suppression():
    common = dict(
        drug_name="Drug",
        cl_L_per_h=1.0,
        vd_L=10.0,
        ic50_mg_L=0.5,
        imax=0.95,
        t_end_h=24.0,
        dt_h=0.1,
    )
    low = simulate_inflammation_pd(dose_mg=10.0, **common)
    high = simulate_inflammation_pd(dose_mg=200.0, **common)
    assert high.peak_cytokine_suppression > low.peak_cytokine_suppression


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------


def test_oral_initial_concentration_zero():
    result = simulate_inflammation_pd(
        "OralDrug",
        dose_mg=50.0,
        cl_L_per_h=1.0,
        vd_L=10.0,
        route="oral",
        f_oral=0.7,
        t_end_h=24.0,
        dt_h=0.1,
    )
    # Oral: gut depot, so plasma starts at 0
    assert result.c_plasma_mg_L[0] == 0.0


def test_imax_exactly_one_accepted():
    result = simulate_inflammation_pd(
        "Drug", dose_mg=50.0, cl_L_per_h=1.0, vd_L=10.0, imax=1.0, t_end_h=10.0, dt_h=0.1
    )
    assert isinstance(result, InflammationPDResult)
