"""Tests for COPBPKResult / simulate_co_pbpk / cardiac_output_sensitivity (Phase 435)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.cardiac_output_pbpk import (
    COPBPKResult,
    cardiac_output_sensitivity,
    simulate_co_pbpk,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_result() -> COPBPKResult:
    return simulate_co_pbpk(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_hep_L_per_h=10.0,
        cardiac_output_fraction=1.0,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


@pytest.fixture
def hf_result() -> COPBPKResult:
    return simulate_co_pbpk(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_hep_L_per_h=10.0,
        cardiac_output_fraction=0.5,
        vd_L=50.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# 1. Return type and structure
# ---------------------------------------------------------------------------


def test_returns_copbpk_result(normal_result):
    assert isinstance(normal_result, COPBPKResult)


def test_result_has_required_fields(normal_result):
    assert hasattr(normal_result, "drug_name")
    assert hasattr(normal_result, "dose_mg")
    assert hasattr(normal_result, "cardiac_output_fraction")
    assert hasattr(normal_result, "cl_hep_effective_L_per_h")
    assert hasattr(normal_result, "times_h")
    assert hasattr(normal_result, "c_plasma_mg_L")
    assert hasattr(normal_result, "tissue_concs")
    assert hasattr(normal_result, "auc_plasma")
    assert hasattr(normal_result, "t_half_h")
    assert hasattr(normal_result, "notes")


def test_tissue_concs_contains_all_organs(normal_result):
    assert set(normal_result.tissue_concs.keys()) == {"liver", "kidney", "muscle", "fat"}


def test_times_and_plasma_same_length(normal_result):
    assert len(normal_result.times_h) == len(normal_result.c_plasma_mg_L)


def test_tissue_concs_same_length_as_times(normal_result):
    n = len(normal_result.times_h)
    for org, concs in normal_result.tissue_concs.items():
        assert len(concs) == n, f"Organ {org} has mismatched length"


# ---------------------------------------------------------------------------
# 2. Initial conditions
# ---------------------------------------------------------------------------


def test_initial_plasma_conc_equals_dose_over_vd():
    result = simulate_co_pbpk(
        drug_name="Drug",
        dose_mg=200.0,
        cl_hep_L_per_h=5.0,
        cardiac_output_fraction=1.0,
        vd_L=100.0,
        t_end_h=24.0,
        dt_h=0.1,
    )
    expected_c0 = 200.0 / 100.0
    assert abs(result.c_plasma_mg_L[0] - expected_c0) < 0.01


def test_times_start_at_zero(normal_result):
    assert normal_result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end(normal_result):
    assert normal_result.times_h[-1] == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# 3. Cardiac output effects on CL
# ---------------------------------------------------------------------------


def test_lower_co_reduces_effective_hep_cl(normal_result, hf_result):
    assert hf_result.cl_hep_effective_L_per_h < normal_result.cl_hep_effective_L_per_h


def test_lower_co_increases_auc(normal_result, hf_result):
    """Heart failure (CO=0.5) -> lower CL -> higher AUC."""
    assert hf_result.auc_plasma > normal_result.auc_plasma


def test_lower_co_increases_t_half(normal_result, hf_result):
    assert hf_result.t_half_h > normal_result.t_half_h


def test_higher_co_decreases_auc():
    result_high_co = simulate_co_pbpk(
        drug_name="Drug",
        dose_mg=100.0,
        cl_hep_L_per_h=10.0,
        cardiac_output_fraction=2.0,
        vd_L=50.0,
    )
    result_normal = simulate_co_pbpk(
        drug_name="Drug",
        dose_mg=100.0,
        cl_hep_L_per_h=10.0,
        cardiac_output_fraction=1.0,
        vd_L=50.0,
    )
    assert result_high_co.auc_plasma < result_normal.auc_plasma


# ---------------------------------------------------------------------------
# 4. Well-stirred hepatic model
# ---------------------------------------------------------------------------


def test_effective_cl_less_than_or_equal_to_liver_flow():
    # CL_eff <= Q_liver (flow-limited ceiling)
    result = simulate_co_pbpk(
        drug_name="Drug",
        dose_mg=100.0,
        cl_hep_L_per_h=200.0,  # very high intrinsic CL
        cardiac_output_fraction=1.0,
        organ_flows={"liver": 54.0, "kidney": 72.0, "muscle": 15.0, "fat": 5.0},
        vd_L=50.0,
    )
    assert result.cl_hep_effective_L_per_h <= 54.0 + 0.01  # tiny float tolerance


def test_effective_cl_scales_with_co_fraction():
    r1 = simulate_co_pbpk("D", 100, 10, 1.0, vd_L=50)
    r2 = simulate_co_pbpk("D", 100, 10, 0.5, vd_L=50)
    assert r2.cl_hep_effective_L_per_h < r1.cl_hep_effective_L_per_h


# ---------------------------------------------------------------------------
# 5. Plasma concentration sanity
# ---------------------------------------------------------------------------


def test_plasma_concs_non_negative(normal_result):
    assert all(c >= 0.0 for c in normal_result.c_plasma_mg_L)


def test_tissue_concs_non_negative(normal_result):
    for org, concs in normal_result.tissue_concs.items():
        assert all(c >= 0.0 for c in concs), f"Negative concentration in {org}"


def test_auc_positive(normal_result):
    assert normal_result.auc_plasma > 0.0


# ---------------------------------------------------------------------------
# 6. t_half sanity
# ---------------------------------------------------------------------------


def test_t_half_positive(normal_result):
    assert normal_result.t_half_h > 0.0


def test_t_half_formula():
    """t1/2 = ln(2) * Vd / CL_eff; verify the well-stirred CL is used."""
    q_h = 54.0  # default liver flow * CO=1
    cl_int = 10.0
    cl_eff = q_h * cl_int / (q_h + cl_int)
    expected_t_half = math.log(2) * 50.0 / cl_eff
    result = simulate_co_pbpk("D", 100, cl_int, 1.0, vd_L=50.0)
    assert abs(result.t_half_h - expected_t_half) < 0.01


# ---------------------------------------------------------------------------
# 7. Custom organ flows and Kp
# ---------------------------------------------------------------------------


def test_custom_organ_flows_accepted():
    result = simulate_co_pbpk(
        drug_name="X",
        dose_mg=100,
        cl_hep_L_per_h=5,
        cardiac_output_fraction=1.0,
        organ_flows={"liver": 60, "kidney": 70, "muscle": 20, "fat": 8},
        organ_kp={"liver": 4, "kidney": 6, "muscle": 2, "fat": 1},
        vd_L=50,
    )
    assert isinstance(result, COPBPKResult)
    assert result.auc_plasma > 0


def test_higher_kp_increases_liver_tissue_conc():
    r_low = simulate_co_pbpk(
        "D", 100, 10, 1.0,
        organ_kp={"liver": 1, "kidney": 5, "muscle": 1, "fat": 0.5},
        vd_L=50,
    )
    r_high = simulate_co_pbpk(
        "D", 100, 10, 1.0,
        organ_kp={"liver": 10, "kidney": 5, "muscle": 1, "fat": 0.5},
        vd_L=50,
    )
    # Higher Kp -> more drug partitions into liver tissue
    assert r_high.tissue_concs["liver"][1] > r_low.tissue_concs["liver"][1]


# ---------------------------------------------------------------------------
# 8. cardiac_output_sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity_returns_list():
    results = cardiac_output_sensitivity("Drug", 100, 10, 50, [0.5, 1.0, 1.5])
    assert isinstance(results, list)
    assert len(results) == 3


def test_sensitivity_has_expected_keys():
    results = cardiac_output_sensitivity("Drug", 100, 10, 50, [1.0])
    assert set(results[0].keys()) == {
        "co_fraction", "auc_plasma", "t_half_h", "cl_hep_effective_L_per_h"
    }


def test_sensitivity_auc_monotone_decreasing_with_co():
    """Higher CO -> higher CL -> lower AUC."""
    fracs = [0.3, 0.5, 1.0, 1.5, 2.0]
    results = cardiac_output_sensitivity("Drug", 100, 15, 50, fracs)
    aucs = [r["auc_plasma"] for r in results]
    for i in range(len(aucs) - 1):
        assert aucs[i] > aucs[i + 1], f"AUC not decreasing at index {i}"


def test_sensitivity_cl_monotone_increasing_with_co():
    fracs = [0.5, 1.0, 2.0]
    results = cardiac_output_sensitivity("Drug", 100, 10, 50, fracs)
    cls = [r["cl_hep_effective_L_per_h"] for r in results]
    assert cls[0] < cls[1] < cls[2]


def test_sensitivity_co_fraction_stored_correctly():
    fracs = [0.4, 1.0, 1.8]
    results = cardiac_output_sensitivity("Drug", 100, 10, 50, fracs)
    for res, frac in zip(results, fracs, strict=True):
        assert res["co_fraction"] == frac


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------


def test_raises_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_co_pbpk("D", 0, 10, 1.0, vd_L=50)


def test_raises_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_co_pbpk("D", -10, 10, 1.0, vd_L=50)


def test_raises_on_zero_cl_hep():
    with pytest.raises(ValueError, match="cl_hep_L_per_h"):
        simulate_co_pbpk("D", 100, 0.0, 1.0, vd_L=50)


def test_raises_on_zero_co_fraction():
    with pytest.raises(ValueError, match="cardiac_output_fraction"):
        simulate_co_pbpk("D", 100, 10, 0.0, vd_L=50)


def test_raises_on_negative_co_fraction():
    with pytest.raises(ValueError, match="cardiac_output_fraction"):
        simulate_co_pbpk("D", 100, 10, -0.5, vd_L=50)


def test_raises_on_zero_vd():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_co_pbpk("D", 100, 10, 1.0, vd_L=0)


def test_raises_on_unknown_organ_flow():
    with pytest.raises(ValueError, match="Unknown organ"):
        simulate_co_pbpk("D", 100, 10, 1.0, organ_flows={"spleen": 5}, vd_L=50)


def test_raises_on_unknown_organ_kp():
    with pytest.raises(ValueError, match="Unknown organ"):
        simulate_co_pbpk("D", 100, 10, 1.0, organ_kp={"brain": 2}, vd_L=50)


def test_raises_on_empty_co_fractions():
    with pytest.raises(ValueError, match="co_fractions"):
        cardiac_output_sensitivity("Drug", 100, 10, 50, [])


def test_raises_on_invalid_co_fraction_in_sensitivity():
    with pytest.raises(ValueError):
        cardiac_output_sensitivity("Drug", 100, 10, 50, [0.0])
