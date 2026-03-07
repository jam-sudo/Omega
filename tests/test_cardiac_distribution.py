"""Tests for Phase 252: Cardiac Output-Dependent Drug Distribution."""

import math
import pytest

from omega_pbpk.clinical.cardiac_distribution import (
    CardiacPKResult,
    HFDoseResult,
    heart_failure_dose_adjustment,
    scale_organ_flows,
    simulate_cardiac_pk,
)


# ---------------------------------------------------------------------------
# scale_organ_flows tests
# ---------------------------------------------------------------------------


def test_scale_organ_flows_normal():
    """At CO_frac=1.0, all flows should be ~1.0 (normal)."""
    flows = scale_organ_flows(1.0)
    assert isinstance(flows, dict)
    assert abs(flows["hepatic"] - 1.0) < 0.05
    assert abs(flows["renal"] - 1.0) < 0.05
    assert abs(flows["brain"] - 1.0) < 0.05


def test_scale_organ_flows_returns_expected_keys():
    flows = scale_organ_flows(1.0)
    for key in ("hepatic", "renal", "muscle", "fat", "brain", "splanchnic"):
        assert key in flows


def test_scale_organ_flows_hf_hepatic_preserved():
    """At CO_frac=0.5 (HF), hepatic flow should be preserved (>0.5 of normal)."""
    flows = scale_organ_flows(0.5)
    assert flows["hepatic"] > 0.5


def test_scale_organ_flows_hf_renal_preserved():
    """At CO_frac=0.5 (HF), renal flow should be better preserved than muscle."""
    flows = scale_organ_flows(0.5)
    assert flows["renal"] > flows["muscle"]


def test_scale_organ_flows_hf_muscle_reduced():
    """At CO_frac=0.5 (HF), muscle flow should be significantly reduced."""
    flows = scale_organ_flows(0.5)
    assert flows["muscle"] < 0.5  # Muscle is preferentially reduced in HF


def test_scale_organ_flows_exercise_increased():
    """At CO_frac=2.0 (exercise), all flows should be > 1.0."""
    flows = scale_organ_flows(2.0)
    assert flows["renal"] > 1.0
    assert flows["muscle"] > 1.0
    assert flows["splanchnic"] > 1.0


def test_scale_organ_flows_exercise_hepatic_above_normal():
    flows = scale_organ_flows(2.0)
    assert flows["hepatic"] > 1.0


def test_scale_organ_flows_severe_hf():
    """At CO_frac=0.3 (severe HF), flows should be reduced but hepatic/renal most preserved."""
    flows = scale_organ_flows(0.3)
    assert flows["hepatic"] > flows["muscle"]
    assert flows["renal"] > flows["muscle"]
    assert flows["brain"] > flows["muscle"]


def test_scale_organ_flows_invalid_zero():
    with pytest.raises(ValueError):
        scale_organ_flows(0.0)


def test_scale_organ_flows_invalid_negative():
    with pytest.raises(ValueError):
        scale_organ_flows(-0.5)


def test_scale_organ_flows_invalid_too_high():
    with pytest.raises(ValueError):
        scale_organ_flows(3.5)


# ---------------------------------------------------------------------------
# simulate_cardiac_pk tests
# ---------------------------------------------------------------------------


def test_simulate_cardiac_pk_returns_result():
    result = simulate_cardiac_pk(
        drug_name="digoxin",
        dose_mg=0.25,
        cl_normal_L_per_h=5.0,
        vd_L=500.0,
        cardiac_output_fraction=0.5,
    )
    assert isinstance(result, CardiacPKResult)


def test_simulate_cardiac_pk_field_types():
    result = simulate_cardiac_pk("drug", dose_mg=10.0, cl_normal_L_per_h=5.0, vd_L=100.0)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.dose_mg, float)
    assert isinstance(result.cardiac_output_fraction, float)
    assert isinstance(result.route, str)
    assert isinstance(result.cl_effective_L_per_h, float)
    assert isinstance(result.vd_effective_L, float)
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.cmax, float)
    assert isinstance(result.tmax_h, float)
    assert isinstance(result.auc, float)
    assert isinstance(result.t_half_h, float)
    assert isinstance(result.notes, str)


def test_hf_lower_cl_higher_auc():
    """CO_frac=0.5 should produce higher AUC than CO_frac=1.0 (lower CL)."""
    common = dict(drug_name="drug", dose_mg=10.0, cl_normal_L_per_h=5.0, vd_L=100.0, t_end_h=24.0)
    r_normal = simulate_cardiac_pk(**common, cardiac_output_fraction=1.0)
    r_hf = simulate_cardiac_pk(**common, cardiac_output_fraction=0.5)
    assert r_hf.auc > r_normal.auc


def test_hf_longer_half_life():
    """Heart failure (CO=0.5) should produce longer t½ than normal (CO=1.0)."""
    common = dict(drug_name="drug", dose_mg=10.0, cl_normal_L_per_h=5.0, vd_L=100.0)
    r_normal = simulate_cardiac_pk(**common, cardiac_output_fraction=1.0)
    r_hf = simulate_cardiac_pk(**common, cardiac_output_fraction=0.5)
    assert r_hf.t_half_h > r_normal.t_half_h


def test_normal_co_approx_normal_cl():
    """At CO_frac=1.0, CL_effective should be close to CL_normal * 0.8."""
    cl_normal = 5.0
    result = simulate_cardiac_pk(
        "drug", dose_mg=10.0, cl_normal_L_per_h=cl_normal, vd_L=100.0, cardiac_output_fraction=1.0
    )
    # CL_effective = CL_normal * 1.0 * 0.8 = 4.0
    assert abs(result.cl_effective_L_per_h - cl_normal * 0.8) < 0.1


def test_oral_route_works():
    result = simulate_cardiac_pk(
        "drug", dose_mg=10.0, cl_normal_L_per_h=5.0, vd_L=100.0, route="oral", ka_per_h=1.5
    )
    assert result.cmax > 0.0
    assert result.tmax_h > 0.0  # Oral has positive tmax


def test_simulate_cardiac_pk_invalid_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_cardiac_pk("drug", dose_mg=0.0, cl_normal_L_per_h=5.0, vd_L=100.0)


def test_simulate_cardiac_pk_invalid_cl():
    with pytest.raises(ValueError, match="cl_normal_L_per_h"):
        simulate_cardiac_pk("drug", dose_mg=10.0, cl_normal_L_per_h=-1.0, vd_L=100.0)


def test_simulate_cardiac_pk_invalid_co():
    with pytest.raises(ValueError, match="cardiac_output_fraction"):
        simulate_cardiac_pk(
            "drug", dose_mg=10.0, cl_normal_L_per_h=5.0, vd_L=100.0, cardiac_output_fraction=0.0
        )


def test_simulate_cardiac_pk_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_cardiac_pk("drug", dose_mg=10.0, cl_normal_L_per_h=5.0, vd_L=100.0, route="im")


# ---------------------------------------------------------------------------
# heart_failure_dose_adjustment tests
# ---------------------------------------------------------------------------


def test_hf_dose_adjustment_returns_result():
    result = heart_failure_dose_adjustment(
        drug_name="digoxin",
        normal_dose_mg=0.25,
        cl_normal_L_per_h=5.0,
        vd_L=500.0,
        nyha_class=3,
    )
    assert isinstance(result, HFDoseResult)


def test_hf_dose_result_field_types():
    result = heart_failure_dose_adjustment("drug", 100.0, 5.0, 100.0, nyha_class=2)
    assert isinstance(result.drug_name, str)
    assert isinstance(result.normal_dose_mg, float)
    assert isinstance(result.nyha_class, int)
    assert isinstance(result.cardiac_output_fraction, float)
    assert isinstance(result.cl_effective_L_per_h, float)
    assert isinstance(result.adjusted_dose_mg, float)
    assert isinstance(result.dose_reduction_pct, float)
    assert isinstance(result.t_half_h, float)
    assert isinstance(result.rationale, str)


def test_nyha_iv_greatest_dose_reduction():
    """NYHA IV should produce the greatest dose reduction."""
    common = dict(drug_name="drug", normal_dose_mg=100.0, cl_normal_L_per_h=5.0, vd_L=100.0)
    r1 = heart_failure_dose_adjustment(**common, nyha_class=1)
    r2 = heart_failure_dose_adjustment(**common, nyha_class=2)
    r3 = heart_failure_dose_adjustment(**common, nyha_class=3)
    r4 = heart_failure_dose_adjustment(**common, nyha_class=4)
    assert r4.dose_reduction_pct > r3.dose_reduction_pct
    assert r3.dose_reduction_pct > r2.dose_reduction_pct
    assert r2.dose_reduction_pct > r1.dose_reduction_pct


def test_dose_reduction_pct_in_bounds():
    result = heart_failure_dose_adjustment("drug", 100.0, 5.0, 100.0, nyha_class=4)
    assert 0.0 <= result.dose_reduction_pct <= 100.0


def test_t_half_increases_with_nyha_class():
    """Higher NYHA class -> longer t½ (lower CL)."""
    common = dict(drug_name="drug", normal_dose_mg=100.0, cl_normal_L_per_h=5.0, vd_L=100.0)
    r1 = heart_failure_dose_adjustment(**common, nyha_class=1)
    r4 = heart_failure_dose_adjustment(**common, nyha_class=4)
    assert r4.t_half_h > r1.t_half_h


def test_nyha_class_i_mild_adjustment():
    """NYHA I (CO_frac=0.9) should have a modest dose adjustment (<35%)."""
    result = heart_failure_dose_adjustment("drug", 100.0, 5.0, 100.0, nyha_class=1)
    # NYHA I: CO_frac=0.9, CL = 0.9 * 0.8 = 72% of normal -> 28% reduction
    assert result.dose_reduction_pct < 35.0


def test_adjusted_dose_proportional_to_cl():
    """Adjusted dose should scale with CL ratio."""
    normal_dose = 100.0
    result = heart_failure_dose_adjustment("drug", normal_dose, 5.0, 100.0, nyha_class=3)
    cl_ratio = result.cl_effective_L_per_h / 5.0
    expected_dose = normal_dose * cl_ratio
    assert abs(result.adjusted_dose_mg - expected_dose) < 0.1


def test_invalid_nyha_class():
    with pytest.raises(ValueError, match="nyha_class"):
        heart_failure_dose_adjustment("drug", 100.0, 5.0, 100.0, nyha_class=5)


def test_invalid_normal_dose():
    with pytest.raises(ValueError, match="normal_dose_mg"):
        heart_failure_dose_adjustment("drug", -10.0, 5.0, 100.0, nyha_class=2)


def test_co_fraction_decreases_with_nyha():
    """Higher NYHA class -> lower CO fraction."""
    common = dict(drug_name="drug", normal_dose_mg=100.0, cl_normal_L_per_h=5.0, vd_L=100.0)
    co_fracs = [
        heart_failure_dose_adjustment(**common, nyha_class=c).cardiac_output_fraction
        for c in (1, 2, 3, 4)
    ]
    assert co_fracs[0] > co_fracs[1] > co_fracs[2] > co_fracs[3]
