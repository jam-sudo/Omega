"""Tests for Phase 367 — Exposure Distribution Metric."""

import math

import pytest

from omega_pbpk.clinical.exposure_distribution_metric import (
    ExposureMetricsResult,
    compute_exposure_metrics,
)

# --- Baseline parameters ---
BASE = dict(
    drug_name="DrugA",
    cmax_mg_L=10.0,
    cmin_mg_L=2.0,
    auc_mg_h_per_L=60.0,
    tau_h=12.0,
    t_half_h=6.0,
    dose_mg=300.0,
    cl_L_per_h=5.0,
    vd_L=30.0,
    n_doses_simulated=5,
)


def make_result(**overrides):
    params = {**BASE, **overrides}
    return compute_exposure_metrics(**params)


# --- Return type ---
def test_returns_exposure_metrics_result():
    r = make_result()
    assert isinstance(r, ExposureMetricsResult)


def test_result_is_frozen():
    r = make_result()
    with pytest.raises((AttributeError, TypeError)):
        r.drug_name = "other"  # type: ignore[misc]


# --- Core metric correctness ---
def test_cmax_auc_ratio_positive():
    r = make_result()
    assert r.cmax_auc_ratio_per_h > 0


def test_cmax_auc_ratio_value():
    r = make_result()
    assert math.isclose(r.cmax_auc_ratio_per_h, 10.0 / 60.0, rel_tol=1e-9)


def test_fluctuation_pct_positive():
    r = make_result()
    assert r.fluctuation_pct > 0


def test_fluctuation_pct_value():
    # (cmax - cmin) / auc * tau * 100 = (10-2)/60*12*100 = 160%
    r = make_result()
    expected = (10.0 - 2.0) / 60.0 * 12.0 * 100.0
    assert math.isclose(r.fluctuation_pct, expected, rel_tol=1e-9)


def test_peak_trough_ratio_positive():
    r = make_result()
    assert r.peak_trough_ratio > 0


def test_peak_trough_ratio_value():
    r = make_result()
    assert math.isclose(r.peak_trough_ratio, 10.0 / 2.0, rel_tol=1e-9)


def test_peak_trough_ratio_zero_cmin():
    r = make_result(cmin_mg_L=0.0)
    assert r.peak_trough_ratio == math.inf


def test_css_avg_equals_auc_over_tau():
    r = make_result()
    assert math.isclose(r.css_avg_mg_L, 60.0 / 12.0, rel_tol=1e-9)


def test_accumulation_ratio_ge_1():
    r = make_result()
    assert r.accumulation_ratio >= 1.0


def test_accumulation_ratio_formula():
    # ke = 0.693/6 = 0.1155; R = 1/(1-exp(-0.1155*12))
    ke = 0.693 / 6.0
    expected = 1.0 / (1.0 - math.exp(-ke * 12.0))
    r = make_result()
    assert math.isclose(r.accumulation_ratio, expected, rel_tol=1e-6)


def test_auc_predicted_value():
    r = make_result()
    assert math.isclose(r.auc_predicted_mg_h_per_L, 300.0 / 5.0, rel_tol=1e-9)


def test_recovery_pct_positive():
    r = make_result()
    assert r.recovery_pct > 0


def test_recovery_pct_value():
    r = make_result()
    expected = 60.0 / (300.0 / 5.0) * 100.0
    assert math.isclose(r.recovery_pct, expected, rel_tol=1e-9)


# --- Classification ---
def test_fluctuation_class_valid_set():
    r = make_result()
    assert r.fluctuation_class in {"low", "moderate", "high"}


def test_fluctuation_class_high():
    # fluctuation = 160% → high
    r = make_result()
    assert r.fluctuation_class == "high"


def test_fluctuation_class_low():
    # cmax=5, cmin=4.5, auc=60, tau=12 → (0.5/60)*12*100=10% → low
    r = make_result(cmax_mg_L=5.0, cmin_mg_L=4.5, auc_mg_h_per_L=60.0, tau_h=12.0)
    assert r.fluctuation_class == "low"


def test_fluctuation_class_moderate():
    # cmax=10, cmin=5, auc=60, tau=6 → (5/60)*6*100=50% → moderate
    r = make_result(cmax_mg_L=10.0, cmin_mg_L=5.0, auc_mg_h_per_L=60.0, tau_h=6.0)
    assert r.fluctuation_class == "moderate"


def test_pk_interpretation_valid_set():
    r = make_result()
    assert r.pk_interpretation in {
        "high_fluctuation",
        "high_peak_trough",
        "normal",
        "rapid_distribution",
    }


def test_pk_interpretation_high_fluctuation():
    # BASE already gives fluctuation=160% → high_fluctuation
    r = make_result()
    assert r.pk_interpretation == "high_fluctuation"


def test_pk_interpretation_high_peak_trough():
    # cmax=110, cmin=10, auc=600, tau=6 → fluctuation=(100/600)*6*100=100% (border, not >100%)
    # use small fluctuation but high PTR
    r = make_result(
        cmax_mg_L=110.0,
        cmin_mg_L=10.0,
        auc_mg_h_per_L=600.0,
        tau_h=6.0,
    )
    # fluctuation = (110-10)/600*6*100 = 100% (not >100, so "moderate")
    # peak_trough = 11 > 10 → "high_peak_trough"
    assert r.pk_interpretation == "high_peak_trough"


def test_pk_interpretation_rapid_distribution():
    # cmax/auc > 0.5: cmax=5, auc=8, tau=12 → ratio=0.625
    # fluctuation = (5-0.5)/8*12*100=675% → high_fluctuation takes precedence
    # need low fluctuation AND high ratio
    # cmax=5, cmin=4.8, auc=8, tau=12 → fluctuation=(0.2/8)*12*100=30% → low
    # cmax/auc = 5/8 = 0.625 > 0.5
    # peak_trough = 5/4.8 ≈ 1.04 < 10
    r = make_result(
        cmax_mg_L=5.0,
        cmin_mg_L=4.8,
        auc_mg_h_per_L=8.0,
        tau_h=12.0,
    )
    assert r.pk_interpretation == "rapid_distribution"


# --- Validation errors ---
def test_error_cmax_zero():
    with pytest.raises(ValueError, match="cmax_mg_L"):
        make_result(cmax_mg_L=0.0)


def test_error_auc_zero():
    with pytest.raises(ValueError, match="auc_mg_h_per_L"):
        make_result(auc_mg_h_per_L=0.0)


def test_error_tau_zero():
    with pytest.raises(ValueError, match="tau_h"):
        make_result(tau_h=0.0)


def test_error_t_half_zero():
    with pytest.raises(ValueError, match="t_half_h"):
        make_result(t_half_h=0.0)


def test_error_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        make_result(cl_L_per_h=0.0)
