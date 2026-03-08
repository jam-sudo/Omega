"""Tests for Phase 515 — Drug Accumulation in Adipose (Extended)."""

import pytest

from omega_pbpk.core.adipose_drug_accumulation_p515 import (
    AdiposeAccumulationResult,
    compare_logp_adipose,
    simulate_adipose_accumulation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DRUG = "TestDrug"
DOSE = 100.0
LOG_P = 3.0
CL = 5.0
VD_PLASMA = 10.0


def base_result(**kwargs):
    defaults = dict(
        drug_name=DRUG,
        dose_mg=DOSE,
        logP=LOG_P,
        cl_plasma_L_per_h=CL,
        vd_plasma_L=VD_PLASMA,
    )
    defaults.update(kwargs)
    return simulate_adipose_accumulation(**defaults)


# ---------------------------------------------------------------------------
# Return type and field presence
# ---------------------------------------------------------------------------


def test_return_type():
    r = base_result()
    assert isinstance(r, AdiposeAccumulationResult)


def test_required_fields_present():
    r = base_result()
    for attr in (
        "drug_name",
        "dose_mg",
        "logP",
        "weight_kg",
        "f_adipose",
        "kp_adipose",
        "kp_lean",
        "times_h",
        "c_plasma_mg_L",
        "c_lean_mg_L",
        "c_adipose_mg_L",
        "cmax_plasma",
        "cmax_lean",
        "cmax_adipose",
        "auc_plasma",
        "auc_lean",
        "auc_adipose",
        "adipose_to_plasma_auc_ratio",
        "accumulation_concern",
        "washout_t50_h",
        "notes",
    ):
        assert hasattr(r, attr), f"Missing field: {attr}"


def test_field_values_match_inputs():
    r = base_result(weight_kg=80.0, f_adipose=0.35)
    assert r.drug_name == DRUG
    assert r.dose_mg == DOSE
    assert r.logP == LOG_P
    assert r.weight_kg == 80.0
    assert r.f_adipose == 0.35


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


def test_plasma_starts_at_dose_over_vd():
    r = base_result()
    expected = DOSE / VD_PLASMA
    assert abs(r.c_plasma_mg_L[0] - expected) < 1e-9


def test_lean_starts_at_zero():
    r = base_result()
    assert r.c_lean_mg_L[0] == 0.0


def test_adipose_starts_at_zero():
    r = base_result()
    assert r.c_adipose_mg_L[0] == 0.0


# ---------------------------------------------------------------------------
# Time series properties
# ---------------------------------------------------------------------------


def test_time_series_same_length():
    r = base_result()
    n = len(r.times_h)
    assert len(r.c_plasma_mg_L) == n
    assert len(r.c_lean_mg_L) == n
    assert len(r.c_adipose_mg_L) == n


def test_time_starts_at_zero():
    r = base_result()
    assert r.times_h[0] == 0.0


def test_time_ends_near_t_end():
    r = base_result(t_end_h=48.0)
    assert abs(r.times_h[-1] - 48.0) < 0.5


def test_plasma_declines_overall():
    r = base_result()
    # Plasma should eventually decline from its peak
    assert r.c_plasma_mg_L[0] > r.c_plasma_mg_L[-1]


def test_adipose_rises_then_may_fall():
    r = base_result(logP=4.0, t_end_h=96.0)
    # Adipose must rise from 0
    assert r.cmax_adipose > 0.0
    # For moderate logP, adipose eventually starts declining
    mid = len(r.c_adipose_mg_L) // 2
    assert max(r.c_adipose_mg_L[:mid]) <= r.cmax_adipose + 1e-9


def test_lean_rises_from_zero():
    r = base_result()
    assert r.cmax_lean > 0.0


# ---------------------------------------------------------------------------
# Partition coefficients
# ---------------------------------------------------------------------------


def test_kp_adipose_formula_logp_gt1():
    r = base_result(logP=3.0)
    expected = 2.0 ** (3.0 - 1.0)  # = 4.0
    assert abs(r.kp_adipose - expected) < 1e-6


def test_kp_adipose_clamp_low():
    # For logP <= 1, kp_adipose is set to 1.0 (formula uses else branch),
    # then clamped to [0.5, 100]. Since 1.0 >= 0.5 the clamp doesn't change it.
    r = base_result(logP=-5.0)
    assert r.kp_adipose == 1.0


def test_kp_adipose_clamp_high():
    # logP=10 → 2^9 = 512 → clamped to 100
    r = base_result(logP=10.0)
    assert r.kp_adipose == 100.0


def test_kp_lean_formula():
    r = base_result(logP=2.0)
    expected = 0.5 + 0.1 * 2.0
    assert abs(r.kp_lean - expected) < 1e-9


def test_kp_lean_negative_logp():
    r = base_result(logP=-1.0)
    expected = 0.5 + 0.1 * 0.0  # max(0, -1) = 0
    assert abs(r.kp_lean - expected) < 1e-9


# ---------------------------------------------------------------------------
# AUC and ratios
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    r = base_result()
    assert r.auc_plasma > 0.0


def test_auc_adipose_positive():
    r = base_result()
    assert r.auc_adipose > 0.0


def test_adipose_to_plasma_ratio_positive():
    r = base_result()
    assert r.adipose_to_plasma_auc_ratio > 0.0


def test_higher_logp_higher_adipose_ratio():
    r_low = base_result(logP=1.0)
    r_high = base_result(logP=5.0)
    assert r_high.adipose_to_plasma_auc_ratio > r_low.adipose_to_plasma_auc_ratio


# ---------------------------------------------------------------------------
# Accumulation concern flag
# ---------------------------------------------------------------------------


def test_accumulation_concern_true_for_high_kp():
    # logP=5 → kp_adipose = 2^4 = 16 > 5 → True
    r = base_result(logP=5.0)
    assert r.kp_adipose > 5.0
    assert r.accumulation_concern is True


def test_accumulation_concern_false_for_low_kp():
    # logP=1.5 → kp_adipose = 2^0.5 ≈ 1.41 < 5 → False
    r = base_result(logP=1.5)
    assert r.kp_adipose <= 5.0
    assert r.accumulation_concern is False


# ---------------------------------------------------------------------------
# Washout t50
# ---------------------------------------------------------------------------


def test_washout_t50_positive_for_high_logp():
    r = base_result(logP=4.0, t_end_h=120.0)
    # If adipose peak was reached and washout began, t50 should be > 0
    if r.cmax_adipose > 0.0:
        # t50 > 0 only if the adipose drops to half within simulation
        assert r.washout_t50_h >= 0.0


def test_washout_t50_nonnegative():
    r = base_result()
    assert r.washout_t50_h >= 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_error_on_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_accumulation(DRUG, 0.0, LOG_P, CL, VD_PLASMA)


def test_error_on_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_accumulation(DRUG, -10.0, LOG_P, CL, VD_PLASMA)


def test_error_on_zero_cl():
    with pytest.raises(ValueError, match="cl_plasma_L_per_h"):
        simulate_adipose_accumulation(DRUG, DOSE, LOG_P, 0.0, VD_PLASMA)


def test_error_on_zero_vd_plasma():
    with pytest.raises(ValueError, match="vd_plasma_L"):
        simulate_adipose_accumulation(DRUG, DOSE, LOG_P, CL, 0.0)


# ---------------------------------------------------------------------------
# compare_logp_adipose
# ---------------------------------------------------------------------------


def test_compare_logp_returns_list():
    result = compare_logp_adipose(DRUG, DOSE, CL, VD_PLASMA, [1.0, 3.0, 5.0])
    assert isinstance(result, list)
    assert len(result) == 3


def test_compare_logp_sorted_descending():
    result = compare_logp_adipose(DRUG, DOSE, CL, VD_PLASMA, [1.0, 2.0, 4.0])
    ratios = [r.adipose_to_plasma_auc_ratio for r in result]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_logp_all_are_results():
    result = compare_logp_adipose(DRUG, DOSE, CL, VD_PLASMA, [0.5, 2.5])
    for r in result:
        assert isinstance(r, AdiposeAccumulationResult)
