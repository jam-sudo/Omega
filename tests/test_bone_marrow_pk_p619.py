"""Tests for Phase 619 — Drug Bone Marrow Distribution."""

import math

import pytest

from omega_pbpk.core.bone_marrow_pk_p619 import (
    BoneMarrowPKResult,
    simulate_bone_marrow_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.0,
    mw_Da=300.0,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
    marrow_weight_g=1500.0,
    t_end_h=24.0,
    dt_h=0.1,
)


def run_base(**overrides):
    kwargs = {**BASE_KWARGS, **overrides}
    return simulate_bone_marrow_pk(**kwargs)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = run_base()
    assert isinstance(result, BoneMarrowPKResult)


def test_list_fields_are_lists():
    result = run_base()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.c_marrow_mg_g, list)


def test_list_lengths_match():
    result = run_base()
    n = len(result.times_h)
    assert n == len(result.c_plasma_mg_L)
    assert n == len(result.c_marrow_mg_g)


def test_expected_step_count():
    result = run_base(t_end_h=10.0, dt_h=0.1)
    # n_steps = int(10.0/0.1) + 1 = 101
    assert len(result.times_h) == 101


def test_times_start_at_zero():
    result = run_base()
    assert result.times_h[0] == pytest.approx(0.0)


def test_times_end_at_t_end():
    result = run_base(t_end_h=24.0, dt_h=0.1)
    assert result.times_h[-1] == pytest.approx(24.0, abs=0.2)


def test_drug_name_preserved():
    result = run_base(drug_name="Ibuprofen")
    assert result.drug_name == "Ibuprofen"


def test_dose_preserved():
    result = run_base(dose_mg=200.0)
    assert result.dose_mg == 200.0


# ---------------------------------------------------------------------------
# Concentration positivity and plausibility
# ---------------------------------------------------------------------------


def test_plasma_concentrations_non_negative():
    result = run_base()
    assert all(c >= 0.0 for c in result.c_plasma_mg_L)


def test_marrow_concentrations_non_negative():
    result = run_base()
    assert all(c >= 0.0 for c in result.c_marrow_mg_g)


def test_cmax_plasma_positive():
    result = run_base()
    assert result.cmax_plasma_mg_L > 0.0


def test_cmax_marrow_positive():
    result = run_base()
    assert result.cmax_marrow_mg_g > 0.0


def test_cmax_plasma_matches_list_max():
    result = run_base()
    assert result.cmax_plasma_mg_L == pytest.approx(max(result.c_plasma_mg_L))


def test_cmax_marrow_matches_list_max():
    result = run_base()
    assert result.cmax_marrow_mg_g == pytest.approx(max(result.c_marrow_mg_g))


# ---------------------------------------------------------------------------
# logP effect on Kp and accumulation
# ---------------------------------------------------------------------------


def test_higher_logP_increases_kp_marrow():
    r_low = run_base(logP=0.0)
    r_high = run_base(logP=4.0)
    assert r_high.kp_marrow > r_low.kp_marrow


def test_kp_marrow_within_bounds():
    for logP in [-5.0, 0.0, 2.0, 5.0, 10.0]:
        r = run_base(logP=logP)
        assert 0.1 <= r.kp_marrow <= 15.0


def test_kp_marrow_reduced_by_high_mw():
    r_low_mw = run_base(mw_Da=100.0, logP=2.0)
    r_high_mw = run_base(mw_Da=900.0, logP=2.0)
    assert r_low_mw.kp_marrow >= r_high_mw.kp_marrow


def test_higher_logP_increases_marrow_auc():
    r_low = run_base(logP=0.5)
    r_high = run_base(logP=4.0)
    assert r_high.auc_marrow_mg_h_per_g > r_low.auc_marrow_mg_h_per_g


# ---------------------------------------------------------------------------
# AUC and ratios
# ---------------------------------------------------------------------------


def test_auc_plasma_positive():
    result = run_base()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_marrow_positive():
    result = run_base()
    assert result.auc_marrow_mg_h_per_g > 0.0


def test_marrow_to_plasma_ratio_positive():
    result = run_base()
    assert result.marrow_to_plasma_ratio > 0.0


def test_marrow_to_plasma_ratio_formula():
    result = run_base()
    expected = result.auc_marrow_mg_h_per_g / result.auc_plasma_mg_h_per_L
    assert result.marrow_to_plasma_ratio == pytest.approx(expected, rel=1e-6)


def test_higher_dose_scales_auc_linearly():
    r1 = run_base(dose_mg=100.0)
    r2 = run_base(dose_mg=200.0)
    ratio = r2.auc_plasma_mg_h_per_L / r1.auc_plasma_mg_h_per_L
    assert ratio == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Hematotoxicity index
# ---------------------------------------------------------------------------


def test_hematotoxicity_index_positive():
    result = run_base()
    assert result.hematotoxicity_index > 0.0


def test_hematotoxicity_index_clamped():
    result = run_base()
    assert 0.01 <= result.hematotoxicity_index <= 100.0


def test_hematotoxicity_higher_for_high_logP():
    r_low = run_base(logP=1.0, dose_mg=100.0)
    r_high = run_base(logP=5.0, dose_mg=100.0)
    assert r_high.hematotoxicity_index >= r_low.hematotoxicity_index


# ---------------------------------------------------------------------------
# t_half_marrow
# ---------------------------------------------------------------------------


def test_t_half_marrow_positive():
    result = run_base()
    assert result.t_half_marrow_h > 0.0


def test_t_half_marrow_formula():
    result = run_base()
    # k_out = Qm / Vd_marrow = 3.0 / (marrow_weight_g * kp_marrow / 1000)
    Qm = 3.0
    Vd_marrow = 1500.0 * result.kp_marrow / 1000.0
    k_out = Qm / Vd_marrow
    expected_t_half = math.log(2.0) / k_out
    assert result.t_half_marrow_h == pytest.approx(expected_t_half, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        run_base(dose_mg=-10.0)


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        run_base(dose_mg=0.0)


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="cl_sys"):
        run_base(cl_sys_L_per_h=0.0)


def test_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_sys"):
        run_base(vd_sys_L=0.0)


def test_zero_marrow_weight_raises():
    with pytest.raises(ValueError, match="marrow_weight"):
        run_base(marrow_weight_g=0.0)


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = run_base()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0
