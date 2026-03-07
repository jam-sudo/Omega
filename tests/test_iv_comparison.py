"""Tests for Phase 372: IV bolus vs infusion comparison."""

import math

import pytest

from omega_pbpk.core.iv_comparison import (
    compare_iv_bolus_infusion,
    optimize_infusion_duration,
)

# ---------------------------------------------------------------------------
# Basic construction / frozen dataclass
# ---------------------------------------------------------------------------


def test_result_is_frozen_dataclass():
    res = compare_iv_bolus_infusion("TestDrug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    with pytest.raises((AttributeError, TypeError)):
        res.drug_name = "Other"  # type: ignore[misc]


def test_result_attributes_accessible():
    res = compare_iv_bolus_infusion("DrugX", total_dose_mg=200.0, cl_L_per_h=10.0, vd_L=100.0)
    assert res.drug_name == "DrugX"
    assert res.total_dose_mg == 200.0


# ---------------------------------------------------------------------------
# Bolus Cmax = dose/vd
# ---------------------------------------------------------------------------


def test_cmax_bolus_equals_dose_over_vd():
    dose, vd = 100.0, 50.0
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=dose, cl_L_per_h=5.0, vd_L=vd)
    assert abs(res.cmax_bolus_mg_L - dose / vd) < 1e-9


def test_cmax_bolus_scales_with_dose():
    res1 = compare_iv_bolus_infusion("D", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    res2 = compare_iv_bolus_infusion("D", total_dose_mg=200.0, cl_L_per_h=5.0, vd_L=50.0)
    assert abs(res2.cmax_bolus_mg_L / res1.cmax_bolus_mg_L - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# AUC = dose/CL for both routes (same total dose)
# ---------------------------------------------------------------------------


def test_auc_bolus_equals_dose_over_cl():
    dose, cl = 100.0, 5.0
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=dose, cl_L_per_h=cl, vd_L=50.0)
    assert abs(res.auc_bolus_mg_h_per_L - dose / cl) < 1e-9


def test_auc_infusion_equals_dose_over_cl():
    dose, cl = 100.0, 5.0
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=dose, cl_L_per_h=cl, vd_L=50.0)
    assert abs(res.auc_infusion_mg_h_per_L - dose / cl) < 1e-9


def test_auc_ratio_approximately_one():
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert abs(res.auc_ratio - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Css = R/CL
# ---------------------------------------------------------------------------


def test_css_infusion_equals_rate_over_cl():
    dose, cl, t_inf = 100.0, 5.0, 1.0  # R = 100 mg/h
    res = compare_iv_bolus_infusion(
        "Drug", total_dose_mg=dose, cl_L_per_h=cl, vd_L=50.0, infusion_duration_h=t_inf
    )
    expected_css = (dose / t_inf) / cl  # R/CL
    assert abs(res.css_infusion_mg_L - expected_css) < 1e-9


# ---------------------------------------------------------------------------
# Longer infusion → lower Cmax_infusion
# ---------------------------------------------------------------------------


def test_longer_infusion_lower_cmax():
    base = dict(drug_name="Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    short = compare_iv_bolus_infusion(**base, infusion_duration_h=0.25)
    long_ = compare_iv_bolus_infusion(**base, infusion_duration_h=2.0)
    assert long_.cmax_infusion_mg_L < short.cmax_infusion_mg_L


# ---------------------------------------------------------------------------
# Cmax ratio
# ---------------------------------------------------------------------------


def test_cmax_ratio_greater_than_one():
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert res.cmax_ratio > 1.0


def test_cmax_ratio_large_for_long_infusion():
    """Long infusion with fast clearance → Cmax_infusion << Cmax_bolus → large ratio.

    ke = 5/50 = 0.1/h, t_inf = 8h → Cmax_inf = (100/8/5)*(1-exp(-0.8)) ≈ 1.38
    Cmax_bolus = 2.0 → ratio ≈ 1.45 > 1.0
    """
    res = compare_iv_bolus_infusion(
        "Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, infusion_duration_h=8.0
    )
    assert res.cmax_ratio > 1.0


# ---------------------------------------------------------------------------
# t_half
# ---------------------------------------------------------------------------


def test_t_half_correct():
    cl, vd = 5.0, 50.0
    ke = cl / vd
    expected_t_half = math.log(2) / ke
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=100.0, cl_L_per_h=cl, vd_L=vd)
    assert abs(res.t_half_h - expected_t_half) < 1e-9


# ---------------------------------------------------------------------------
# Concentrations non-negative
# ---------------------------------------------------------------------------


def test_c_bolus_non_negative():
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert all(c >= 0.0 for c in res.c_bolus_mg_L)


def test_c_infusion_non_negative():
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert all(c >= 0.0 for c in res.c_infusion_mg_L)


def test_times_start_at_zero():
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert res.times_h[0] == 0.0


# ---------------------------------------------------------------------------
# Toxicity risk / infusion_preferred
# ---------------------------------------------------------------------------


def test_infusion_preferred_when_bolus_high_risk():
    """High bolus Cmax relative to threshold → infusion preferred."""
    res = compare_iv_bolus_infusion(
        "Drug",
        total_dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        toxic_threshold_mg_L=0.5,  # threshold << Cmax_bolus of 2.0
    )
    assert res.bolus_toxicity_risk == "high"
    assert res.infusion_preferred is True


def test_infusion_not_preferred_when_bolus_low_risk():
    """Very high threshold → bolus is below → risk low → infusion not preferred."""
    res = compare_iv_bolus_infusion(
        "Drug",
        total_dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        toxic_threshold_mg_L=100.0,  # threshold >> Cmax_bolus of 2.0
    )
    assert res.bolus_toxicity_risk == "low"
    assert res.infusion_preferred is False


def test_bolus_toxicity_risk_values():
    res = compare_iv_bolus_infusion("Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0)
    assert res.bolus_toxicity_risk in {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# below_toxic flags
# ---------------------------------------------------------------------------


def test_below_toxic_bolus_true_when_dose_small():
    """Small dose → Cmax bolus < threshold."""
    res = compare_iv_bolus_infusion(
        "Drug", total_dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0, toxic_threshold_mg_L=5.0
    )
    # Cmax_bolus = 10/50 = 0.2 < 5.0
    assert res.below_toxic_bolus is True


def test_below_toxic_bolus_false_when_dose_large():
    res = compare_iv_bolus_infusion(
        "Drug", total_dose_mg=500.0, cl_L_per_h=5.0, vd_L=50.0, toxic_threshold_mg_L=2.0
    )
    # Cmax_bolus = 500/50 = 10 > 2.0
    assert res.below_toxic_bolus is False


def test_below_toxic_infusion_can_be_true_when_bolus_false():
    """Long infusion may bring Cmax below threshold when bolus is above."""
    res = compare_iv_bolus_infusion(
        "Drug",
        total_dose_mg=100.0,
        cl_L_per_h=10.0,
        vd_L=50.0,
        infusion_duration_h=4.0,  # long infusion
        toxic_threshold_mg_L=1.5,
    )
    # Cmax_bolus = 100/50 = 2.0 > 1.5, but infusion Cmax may be < 1.5
    assert res.below_toxic_bolus is False
    # infusion Cmax = (100/4/10) * (1-exp(-0.2*4)) = 2.5*(1-exp(-0.8)) ≈ 1.38 < 1.5
    assert res.below_toxic_infusion is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="total_dose_mg"):
        compare_iv_bolus_infusion("D", total_dose_mg=0.0, cl_L_per_h=5.0, vd_L=50.0)


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        compare_iv_bolus_infusion("D", total_dose_mg=100.0, cl_L_per_h=0.0, vd_L=50.0)


def test_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        compare_iv_bolus_infusion("D", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=0.0)


def test_zero_infusion_duration_raises():
    with pytest.raises(ValueError, match="infusion_duration_h"):
        compare_iv_bolus_infusion(
            "D", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, infusion_duration_h=0.0
        )


# ---------------------------------------------------------------------------
# optimize_infusion_duration
# ---------------------------------------------------------------------------


def test_optimize_returns_dict_with_required_keys():
    result = optimize_infusion_duration(
        "Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, toxic_threshold_mg_L=1.5
    )
    assert "optimal_duration_h" in result
    assert "achievable_cmax_mg_L" in result
    assert "recommendation" in result


def test_optimize_achievable_cmax_below_threshold():
    threshold = 1.5
    result = optimize_infusion_duration(
        "Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, toxic_threshold_mg_L=threshold
    )
    # Cmax bolus = 100/50 = 2.0 > 1.5, so optimization needed
    assert result["achievable_cmax_mg_L"] <= threshold + 1e-4


def test_optimize_no_infusion_when_bolus_below_threshold():
    """If Cmax_bolus < threshold, no extended infusion needed."""
    result = optimize_infusion_duration(
        "Drug", total_dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0, toxic_threshold_mg_L=5.0
    )
    # Cmax_bolus = 10/50 = 0.2 < 5.0
    assert result["optimal_duration_h"] == 0.0


def test_optimize_with_target_cmax_override():
    """target_cmax_mg_L overrides toxic_threshold_mg_L."""
    result = optimize_infusion_duration(
        "Drug",
        total_dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        toxic_threshold_mg_L=0.0,
        target_cmax_mg_L=1.0,
    )
    assert result["achievable_cmax_mg_L"] <= 1.0 + 1e-4


def test_optimize_optimal_duration_positive_when_bolus_above_threshold():
    result = optimize_infusion_duration(
        "Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, toxic_threshold_mg_L=1.5
    )
    assert result["optimal_duration_h"] > 0.0


def test_optimize_recommendation_is_string():
    result = optimize_infusion_duration(
        "Drug", total_dose_mg=100.0, cl_L_per_h=5.0, vd_L=50.0, toxic_threshold_mg_L=1.5
    )
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 0
