"""Tests for Phase 937 — Critical Care PK (critical_care_pk_p937.py)."""
import math
import pytest

from omega_pbpk.clinical.critical_care_pk_p937 import (
    CriticalCarePKResult,
    simulate_critical_care_pk,
    compare_icu_states,
)


# ── Basic return type ──────────────────────────────────────────────────────────

def test_return_type():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0)
    assert isinstance(result, CriticalCarePKResult)


def test_cmax_positive():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0)
    assert result.cmax_mg_L > 0


def test_auc_positive():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0)
    assert result.auc_mg_h_per_L > 0


def test_t_half_positive():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0)
    assert result.t_half_h > 0


def test_tmax_nonnegative():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0)
    assert result.tmax_h >= 0


def test_notes_nonempty():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0)
    assert isinstance(result.notes, str) and len(result.notes) > 0


# ── Normal ICU fold changes (reference) ───────────────────────────────────────

def test_normal_icu_cl_fold_change():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="normal_icu")
    assert result.cl_fold_change == 1.0


def test_normal_icu_vd_fold_change():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="normal_icu")
    assert result.vd_fold_change == 1.0


# ── State-specific fold changes ────────────────────────────────────────────────

def test_sepsis_early_cl_fold_change_gt1():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="sepsis_early")
    assert result.cl_fold_change > 1.0


def test_sepsis_late_aki_cl_fold_change_lt1():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="sepsis_late_aki")
    assert result.cl_fold_change < 1.0


def test_sepsis_early_vd_fold_change():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="sepsis_early")
    assert result.vd_fold_change == 2.0


def test_cl_effective_matches_reference_times_multiplier():
    cl_ref = 12.0
    result = simulate_critical_care_pk(
        "Drug_A", dose_mg=500.0, icu_state="sepsis_early",
        cl_ref_L_per_h=cl_ref
    )
    # sepsis_early multiplier = 1.8
    assert abs(result.cl_effective_L_per_h - cl_ref * 1.8) < 1e-9


# ── AUC comparison ─────────────────────────────────────────────────────────────

def test_sepsis_late_aki_auc_greater_than_normal():
    """Lower CL in AKI -> more drug accumulation -> higher AUC."""
    aki = simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="sepsis_late_aki",
                                    t_end_h=96.0)
    normal = simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="normal_icu",
                                       t_end_h=96.0)
    assert aki.auc_mg_h_per_L > normal.auc_mg_h_per_L


# ── IV infusion vs IV bolus ────────────────────────────────────────────────────

def test_iv_infusion_cmax_less_than_bolus():
    """Same dose spread over 24 h -> lower Cmax than bolus."""
    bolus = simulate_critical_care_pk(
        "Drug_A", dose_mg=500.0, route="iv_bolus"
    )
    infusion = simulate_critical_care_pk(
        "Drug_A", dose_mg=500.0, route="iv_infusion",
        infusion_duration_h=24.0, t_end_h=48.0
    )
    assert infusion.cmax_mg_L < bolus.cmax_mg_L


# ── Half-life formula ──────────────────────────────────────────────────────────

def test_t_half_formula():
    """t_half should equal ln(2) / ke = ln(2) * Vd_eff / CL_eff."""
    cl_ref, vd_ref = 10.0, 50.0
    result = simulate_critical_care_pk(
        "Drug_A", dose_mg=500.0, icu_state="normal_icu",
        cl_ref_L_per_h=cl_ref, vd_ref_L=vd_ref
    )
    expected_t_half = math.log(2) * vd_ref / cl_ref
    assert abs(result.t_half_h - expected_t_half) < 1e-9


# ── compare_icu_states ─────────────────────────────────────────────────────────

def test_compare_returns_4_results():
    results = compare_icu_states("Drug_A", dose_mg=500.0)
    assert len(results) == 4


def test_compare_sorted_by_auc_descending():
    results = compare_icu_states("Drug_A", dose_mg=500.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_states_represented():
    results = compare_icu_states("Drug_A", dose_mg=500.0)
    states = {r.icu_state for r in results}
    assert states == {"normal_icu", "sepsis_early", "sepsis_late_aki", "ards"}


# ── Validation ─────────────────────────────────────────────────────────────────

def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_critical_care_pk("Drug_A", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_critical_care_pk("Drug_A", dose_mg=-100.0)


def test_validation_cl_ref_zero():
    with pytest.raises(ValueError, match="cl_ref_L_per_h"):
        simulate_critical_care_pk("Drug_A", dose_mg=500.0, cl_ref_L_per_h=0.0)


def test_validation_vd_ref_zero():
    with pytest.raises(ValueError, match="vd_ref_L"):
        simulate_critical_care_pk("Drug_A", dose_mg=500.0, vd_ref_L=0.0)


def test_validation_invalid_icu_state():
    with pytest.raises(ValueError, match="icu_state"):
        simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="chronic_kidney_disease")


def test_validation_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_critical_care_pk("Drug_A", dose_mg=500.0, route="oral")


# ── Additional sanity checks ───────────────────────────────────────────────────

def test_ards_cl_fold_change():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0, icu_state="ards")
    assert abs(result.cl_fold_change - 0.7) < 1e-9


def test_times_length_correct():
    result = simulate_critical_care_pk(
        "Drug_A", dose_mg=500.0, t_end_h=10.0, dt_h=1.0
    )
    assert len(result.times_h) == 11  # 0..10 inclusive


def test_drug_name_preserved():
    result = simulate_critical_care_pk("Vancomycin", dose_mg=1000.0)
    assert result.drug_name == "Vancomycin"


def test_route_preserved():
    result = simulate_critical_care_pk("Drug_A", dose_mg=500.0, route="iv_infusion",
                                       infusion_duration_h=8.0, t_end_h=24.0)
    assert result.route == "iv_infusion"
