"""Tests for Phase 516 — Cerebrospinal Fluid Drug Dynamics (Lumbar)."""

import pytest

from omega_pbpk.core.csf_lumbar_dynamics_p516 import (
    CSFLumbarDynamicsResult,
    compare_csf_penetration,
    simulate_csf_lumbar_dynamics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
DOSE = 100.0
CL = 5.0
VD = 20.0
PS = 0.1


def base_result(**kwargs):
    defaults = dict(
        drug_name=DRUG,
        dose_mg=DOSE,
        cl_L_per_h=CL,
        vd_plasma_L=VD,
        ps_csf_L_per_h=PS,
    )
    defaults.update(kwargs)
    return simulate_csf_lumbar_dynamics(**defaults)


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_return_type():
    r = base_result()
    assert isinstance(r, CSFLumbarDynamicsResult)


def test_required_fields_present():
    r = base_result()
    for attr in (
        "drug_name",
        "dose_mg",
        "ps_csf_L_per_h",
        "times_h",
        "c_plasma_mg_L",
        "c_csf_mg_L",
        "cmax_plasma",
        "tmax_plasma_h",
        "auc_plasma",
        "cmax_csf",
        "tmax_csf_h",
        "auc_csf",
        "csf_plasma_auc_ratio",
        "csf_penetration_pct",
        "t_half_plasma_h",
        "t_half_csf_h",
        "notes",
    ):
        assert hasattr(r, attr), f"Missing field: {attr}"


def test_field_values_match_inputs():
    r = base_result(ps_csf_L_per_h=0.2)
    assert r.drug_name == DRUG
    assert r.dose_mg == DOSE
    assert r.ps_csf_L_per_h == 0.2


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_plasma_starts_at_dose_over_vd():
    r = base_result()
    expected = DOSE / VD
    assert abs(r.c_plasma_mg_L[0] - expected) < 1e-9


def test_csf_starts_at_zero():
    r = base_result()
    assert r.c_csf_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Time series properties
# ---------------------------------------------------------------------------


def test_time_series_same_length():
    r = base_result()
    n = len(r.times_h)
    assert len(r.c_plasma_mg_L) == n
    assert len(r.c_csf_mg_L) == n


def test_time_starts_at_zero():
    r = base_result()
    assert r.times_h[0] == 0.0


def test_time_ends_near_t_end():
    r = base_result(t_end_h=24.0)
    assert abs(r.times_h[-1] - 24.0) < 0.5


def test_plasma_declines_overall():
    r = base_result()
    assert r.c_plasma_mg_L[0] > r.c_plasma_mg_L[-1]


def test_csf_rises_from_zero():
    r = base_result()
    assert r.cmax_csf > 0.0


def test_csf_peak_after_plasma_peak():
    r = base_result()
    # CSF Tmax must be at or after plasma Tmax (CSF fills after plasma dose)
    assert r.tmax_csf_h >= r.tmax_plasma_h


# ---------------------------------------------------------------------------
# CSF penetration metrics
# ---------------------------------------------------------------------------


def test_csf_plasma_auc_ratio_positive():
    r = base_result()
    assert r.csf_plasma_auc_ratio > 0.0


def test_csf_penetration_pct_between_0_and_100():
    r = base_result()
    assert 0.0 < r.csf_penetration_pct <= 100.0


def test_higher_ps_higher_csf_ratio():
    r_low = base_result(ps_csf_L_per_h=0.01)
    r_high = base_result(ps_csf_L_per_h=0.5)
    assert r_high.csf_plasma_auc_ratio > r_low.csf_plasma_auc_ratio


def test_higher_ps_higher_csf_penetration_pct():
    r_low = base_result(ps_csf_L_per_h=0.01)
    r_high = base_result(ps_csf_L_per_h=0.5)
    assert r_high.csf_penetration_pct > r_low.csf_penetration_pct


# ---------------------------------------------------------------------------
# Half-life metrics
# ---------------------------------------------------------------------------


def test_t_half_plasma_positive():
    # Use a long simulation and moderate PS so plasma clearly halves
    r = simulate_csf_lumbar_dynamics(DRUG, DOSE, CL, VD, ps_csf_L_per_h=0.05, t_end_h=72.0)
    assert r.t_half_plasma_h > 0.0


def test_t_half_csf_nonnegative():
    r = base_result(t_end_h=72.0)
    assert r.t_half_csf_h >= 0.0


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_plasma():
    r1 = simulate_csf_lumbar_dynamics(DRUG, 100.0, CL, VD, PS)
    r2 = simulate_csf_lumbar_dynamics(DRUG, 200.0, CL, VD, PS)
    ratio = r2.auc_plasma / r1.auc_plasma
    assert abs(ratio - 2.0) < 0.01


def test_dose_linearity_csf():
    r1 = simulate_csf_lumbar_dynamics(DRUG, 100.0, CL, VD, PS)
    r2 = simulate_csf_lumbar_dynamics(DRUG, 200.0, CL, VD, PS)
    ratio = r2.auc_csf / r1.auc_csf
    assert abs(ratio - 2.0) < 0.01


def test_dose_linearity_cmax_csf():
    r1 = simulate_csf_lumbar_dynamics(DRUG, 50.0, CL, VD, PS)
    r2 = simulate_csf_lumbar_dynamics(DRUG, 100.0, CL, VD, PS)
    ratio = r2.cmax_csf / r1.cmax_csf
    assert abs(ratio - 2.0) < 0.01


# ---------------------------------------------------------------------------
# AUC consistency
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    r = base_result()
    assert r.auc_plasma > 0.0


def test_auc_csf_positive():
    r = base_result()
    assert r.auc_csf > 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_lumbar_dynamics(DRUG, 0.0, CL, VD, PS)


def test_error_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_csf_lumbar_dynamics(DRUG, -50.0, CL, VD, PS)


def test_error_on_zero_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_csf_lumbar_dynamics(DRUG, DOSE, 0.0, VD, PS)


def test_error_on_zero_vd():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_csf_lumbar_dynamics(DRUG, DOSE, CL, 0.0, PS)


def test_error_on_zero_ps():
    with pytest.raises(ValueError, match="ps_csf_L_per_h"):
        simulate_csf_lumbar_dynamics(DRUG, DOSE, CL, VD, 0.0)


# ---------------------------------------------------------------------------
# compare_csf_penetration
# ---------------------------------------------------------------------------


def test_compare_csf_penetration_returns_list():
    result = compare_csf_penetration(DRUG, DOSE, CL, VD, [0.01, 0.1, 0.5])
    assert isinstance(result, list)
    assert len(result) == 3


def test_compare_csf_penetration_sorted_descending():
    result = compare_csf_penetration(DRUG, DOSE, CL, VD, [0.01, 0.05, 0.2])
    ratios = [r.csf_plasma_auc_ratio for r in result]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_csf_penetration_all_results():
    result = compare_csf_penetration(DRUG, DOSE, CL, VD, [0.1, 0.3])
    for r in result:
        assert isinstance(r, CSFLumbarDynamicsResult)
