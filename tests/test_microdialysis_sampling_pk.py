"""Tests for Phase 514 — Microdialysis Sampling PK."""

import math

import pytest

from omega_pbpk.core.microdialysis_sampling_pk import (
    MicrodialysisPKResult,
    _kp_isf_from_logp,
    _relative_recovery,
    compare_probe_recoveries,
    simulate_microdialysis_pk,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
DOSE = 10.0  # mg
CL = 5.0  # L/h
VD = 20.0  # L
FUP = 0.2
LOGP = 1.5
PS = 0.5  # mL/min
FLOW = 2.0  # mL/min


def base_result(**kwargs):
    defaults = dict(
        drug_name=DRUG,
        dose_mg=DOSE,
        cl_L_per_h=CL,
        vd_L=VD,
        fup=FUP,
        logP=LOGP,
        probe_ps_mL_per_min=PS,
        flow_rate_mL_per_min=FLOW,
        t_end_h=12.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_microdialysis_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


def test_return_type():
    r = base_result()
    assert isinstance(r, MicrodialysisPKResult)


def test_fields_present():
    r = base_result()
    for field_name in [
        "drug_name",
        "dose_mg",
        "logP",
        "fup",
        "probe_ps_mL_per_min",
        "flow_rate_mL_per_min",
        "relative_recovery",
        "kp_isf",
        "times_h",
        "c_plasma_mg_L",
        "c_isf_mg_L",
        "c_dialysate_mg_L",
        "cmax_plasma",
        "cmax_isf",
        "cmax_dialysate",
        "auc_plasma",
        "auc_isf",
        "auc_dialysate",
        "isf_to_plasma_ratio",
        "t_half_h",
        "notes",
    ]:
        assert hasattr(r, field_name), f"Missing field: {field_name}"


def test_drug_name_stored():
    r = simulate_microdialysis_pk("Morphine", DOSE, CL, VD, FUP, LOGP)
    assert r.drug_name == "Morphine"


# ---------------------------------------------------------------------------
# Probe recovery
# ---------------------------------------------------------------------------


def test_relative_recovery_formula():
    """R = 1 - exp(-PS / flow)."""
    expected = 1.0 - math.exp(-PS / FLOW)
    assert abs(_relative_recovery(PS, FLOW) - expected) < 1e-9


def test_recovery_in_result():
    r = base_result()
    expected = 1.0 - math.exp(-PS / FLOW)
    assert abs(r.relative_recovery - expected) < 1e-9


def test_recovery_between_0_and_1():
    r = base_result()
    assert 0.0 < r.relative_recovery < 1.0


def test_dialysate_less_than_isf():
    """C_dialysate = C_ISF * R, and R < 1, so dialysate < ISF."""
    r = base_result()
    assert r.cmax_dialysate < r.cmax_isf


def test_dialysate_equals_isf_times_recovery():
    r = base_result()
    assert abs(r.cmax_dialysate - r.cmax_isf * r.relative_recovery) < 1e-6


# ---------------------------------------------------------------------------
# Kp_isf
# ---------------------------------------------------------------------------


def test_kp_isf_logp_below_2():
    """For logP <= 2, Kp_isf = 1.0."""
    kp = _kp_isf_from_logp(1.0)
    assert abs(kp - 1.0) < 1e-9


def test_kp_isf_logp_above_2():
    """For logP=4, Kp_isf = 1.0 + 0.1*(4-2) = 1.2."""
    kp = _kp_isf_from_logp(4.0)
    assert abs(kp - 1.2) < 1e-9


def test_kp_isf_stored():
    r = simulate_microdialysis_pk(DRUG, DOSE, CL, VD, FUP, 4.0)
    assert abs(r.kp_isf - 1.2) < 1e-9


# ---------------------------------------------------------------------------
# ISF / plasma relationship
# ---------------------------------------------------------------------------


def test_isf_less_than_plasma_low_fup():
    """For low fup (0.1), C_ISF < C_plasma (since fup * Kp_isf < 1)."""
    r = simulate_microdialysis_pk(DRUG, DOSE, CL, VD, fup=0.1, logP=1.0)
    # Kp_isf = 1.0 for logP=1, fup=0.1 → ratio = 0.1 < 1
    assert r.cmax_isf < r.cmax_plasma


def test_isf_to_plasma_ratio_formula():
    """isf_to_plasma_ratio = fup * Kp_isf."""
    r = base_result()
    expected = FUP * r.kp_isf
    assert abs(r.isf_to_plasma_ratio - expected) < 1e-6


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def test_plasma_declines_over_time():
    r = base_result()
    assert r.c_plasma_mg_L[0] > r.c_plasma_mg_L[-1]


def test_times_start_at_zero():
    r = base_result()
    assert r.times_h[0] == 0.0


def test_times_end_at_t_end():
    r = base_result(t_end_h=6.0, dt_h=0.1)
    assert abs(r.times_h[-1] - 6.0) < 0.15


def test_list_lengths_match():
    r = base_result()
    n = len(r.times_h)
    assert len(r.c_plasma_mg_L) == n
    assert len(r.c_isf_mg_L) == n
    assert len(r.c_dialysate_mg_L) == n


# ---------------------------------------------------------------------------
# AUC ordering
# ---------------------------------------------------------------------------


def test_auc_dialysate_less_than_auc_isf():
    """For recovery < 1, AUC_dialysate < AUC_ISF."""
    r = base_result()
    assert r.auc_dialysate < r.auc_isf


def test_auc_isf_less_than_auc_plasma_low_fup():
    """For low fup, AUC_ISF < AUC_plasma."""
    r = simulate_microdialysis_pk(DRUG, DOSE, CL, VD, fup=0.1, logP=1.0)
    assert r.auc_isf < r.auc_plasma


# ---------------------------------------------------------------------------
# Half-life
# ---------------------------------------------------------------------------


def test_half_life_formula():
    """t_half = ln(2) * Vd / CL."""
    r = base_result()
    expected = math.log(2.0) * VD / CL
    assert abs(r.t_half_h - expected) < 1e-6


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


def test_dose_linearity_cmax():
    """Doubling dose doubles Cmax (linear PK)."""
    r1 = base_result(dose_mg=10.0)
    r2 = base_result(dose_mg=20.0)
    assert abs(r2.cmax_plasma / r1.cmax_plasma - 2.0) < 1e-6


def test_dose_linearity_auc():
    """Doubling dose doubles AUC."""
    r1 = base_result(dose_mg=10.0)
    r2 = base_result(dose_mg=20.0)
    assert abs(r2.auc_plasma / r1.auc_plasma - 2.0) < 1e-4


# ---------------------------------------------------------------------------
# Higher PS → higher recovery
# ---------------------------------------------------------------------------


def test_higher_ps_higher_recovery():
    r_low = base_result(probe_ps_mL_per_min=0.5)
    r_high = base_result(probe_ps_mL_per_min=2.0)
    assert r_high.relative_recovery > r_low.relative_recovery


# ---------------------------------------------------------------------------
# compare_probe_recoveries
# ---------------------------------------------------------------------------


def test_compare_returns_correct_length():
    ps_list = [0.5, 1.0, 2.0, 4.0]
    results = compare_probe_recoveries(DRUG, DOSE, CL, VD, FUP, LOGP, ps_list)
    assert len(results) == 4


def test_compare_sorted_descending():
    ps_list = [0.5, 1.0, 2.0, 4.0]
    results = compare_probe_recoveries(DRUG, DOSE, CL, VD, FUP, LOGP, ps_list)
    recoveries = [r.relative_recovery for r in results]
    assert recoveries == sorted(recoveries, reverse=True)


def test_compare_all_results_type():
    ps_list = [0.5, 1.0]
    results = compare_probe_recoveries(DRUG, DOSE, CL, VD, FUP, LOGP, ps_list)
    for r in results:
        assert isinstance(r, MicrodialysisPKResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose"):
        simulate_microdialysis_pk(DRUG, 0.0, CL, VD, FUP, LOGP)


def test_invalid_cl_negative():
    with pytest.raises(ValueError, match="cl_L"):
        simulate_microdialysis_pk(DRUG, DOSE, -1.0, VD, FUP, LOGP)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_microdialysis_pk(DRUG, DOSE, CL, 0.0, FUP, LOGP)


def test_invalid_fup_zero():
    with pytest.raises(ValueError, match="fup"):
        simulate_microdialysis_pk(DRUG, DOSE, CL, VD, 0.0, LOGP)


def test_invalid_fup_above_one():
    with pytest.raises(ValueError, match="fup"):
        simulate_microdialysis_pk(DRUG, DOSE, CL, VD, 1.5, LOGP)


def test_invalid_ps_zero():
    with pytest.raises(ValueError, match="probe_ps"):
        simulate_microdialysis_pk(DRUG, DOSE, CL, VD, FUP, LOGP, probe_ps_mL_per_min=0.0)


def test_invalid_flow_negative():
    with pytest.raises(ValueError, match="flow_rate"):
        simulate_microdialysis_pk(DRUG, DOSE, CL, VD, FUP, LOGP, flow_rate_mL_per_min=-1.0)
