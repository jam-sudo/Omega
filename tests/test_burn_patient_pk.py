"""Tests for Phase 933: Burn Patient PK Model."""

import pytest

from omega_pbpk.clinical.burn_patient_pk import (
    BurnPatientPKResult,
    compare_burn_severity,
    simulate_burn_pk,
)

# ---- Basic return type and shape ----


def test_return_type():
    result = simulate_burn_pk("VancomycinA", dose_mg=1000.0, tbsa_pct=30.0)
    assert isinstance(result, BurnPatientPKResult)


def test_c_plasma_starts_positive():
    result = simulate_burn_pk("DrugA", dose_mg=500.0, tbsa_pct=20.0)
    # IV bolus: concentration should be > 0 after first step
    assert result.c_plasma_mg_L[1] > 0


def test_cmax_positive():
    result = simulate_burn_pk("DrugB", dose_mg=500.0, tbsa_pct=20.0)
    assert result.cmax_mg_L > 0


def test_auc_positive():
    result = simulate_burn_pk("DrugC", dose_mg=500.0, tbsa_pct=20.0)
    assert result.auc_mg_h_per_L > 0


def test_vd_at_24h_expanded():
    result = simulate_burn_pk("DrugD", dose_mg=500.0, tbsa_pct=30.0, vd_normal_L=50.0)
    assert result.vd_at_24h_L > 50.0  # expanded in burn


def test_vd_fold_change_gt1_any_tbsa():
    result = simulate_burn_pk("DrugE", dose_mg=500.0, tbsa_pct=10.0)
    assert result.vd_fold_change > 1.0


def test_cl_fold_change_gt1_hypermetabolic():
    # At 96h (hypermetabolic phase), CL should be elevated
    result = simulate_burn_pk("DrugF", dose_mg=500.0, tbsa_pct=30.0)
    assert result.cl_fold_change > 1.0


def test_severe_burn_higher_vd_than_mild():
    mild = simulate_burn_pk("DrugG", dose_mg=500.0, tbsa_pct=10.0)
    severe = simulate_burn_pk("DrugG", dose_mg=500.0, tbsa_pct=60.0)
    assert severe.vd_at_24h_L > mild.vd_at_24h_L


def test_tbsa0_fold_change_is_1():
    result = simulate_burn_pk("DrugH", dose_mg=500.0, tbsa_pct=0.0)
    assert result.vd_fold_change == pytest.approx(1.0, abs=1e-9)


def test_compare_returns_4_by_default():
    results = compare_burn_severity("DrugI", dose_mg=500.0)
    assert len(results) == 4


def test_compare_sorted_by_auc_desc():
    results = compare_burn_severity("DrugJ", dose_mg=500.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_notes_nonempty():
    result = simulate_burn_pk("DrugK", dose_mg=500.0, tbsa_pct=20.0)
    assert result.notes and len(result.notes) > 0


def test_n_doses_3_no_error():
    result = simulate_burn_pk(
        "DrugL", dose_mg=500.0, tbsa_pct=20.0, n_doses=3, dosing_interval_h=24.0, t_end_h=168.0
    )
    assert result.cmax_mg_L > 0


def test_tmax_nonnegative():
    result = simulate_burn_pk("DrugM", dose_mg=500.0, tbsa_pct=20.0)
    assert result.tmax_h >= 0


def test_times_h_nonempty():
    result = simulate_burn_pk("DrugN", dose_mg=500.0, tbsa_pct=20.0)
    assert len(result.times_h) > 0


# ---- Validation errors ----


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_burn_pk("DrugV", dose_mg=0.0, tbsa_pct=20.0)


def test_validation_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_burn_pk("DrugV", dose_mg=-100.0, tbsa_pct=20.0)


def test_validation_tbsa_negative_raises():
    with pytest.raises(ValueError, match="tbsa_pct"):
        simulate_burn_pk("DrugV", dose_mg=500.0, tbsa_pct=-5.0)


def test_validation_tbsa_over100_raises():
    with pytest.raises(ValueError, match="tbsa_pct"):
        simulate_burn_pk("DrugV", dose_mg=500.0, tbsa_pct=101.0)


def test_validation_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_normal_L_per_h"):
        simulate_burn_pk("DrugV", dose_mg=500.0, tbsa_pct=20.0, cl_normal_L_per_h=0.0)


def test_validation_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_normal_L_per_h"):
        simulate_burn_pk("DrugV", dose_mg=500.0, tbsa_pct=20.0, cl_normal_L_per_h=-1.0)


def test_validation_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_normal_L"):
        simulate_burn_pk("DrugV", dose_mg=500.0, tbsa_pct=20.0, vd_normal_L=0.0)


def test_validation_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_normal_L"):
        simulate_burn_pk("DrugV", dose_mg=500.0, tbsa_pct=20.0, vd_normal_L=-10.0)


def test_validation_n_doses_zero_raises():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_burn_pk("DrugV", dose_mg=500.0, tbsa_pct=20.0, n_doses=0)


def test_validation_n_doses_negative_raises():
    with pytest.raises(ValueError, match="n_doses"):
        simulate_burn_pk("DrugV", dose_mg=500.0, tbsa_pct=20.0, n_doses=-1)


# ---- compare_burn_severity custom levels ----


def test_compare_custom_tbsa_levels():
    results = compare_burn_severity("DrugX", dose_mg=500.0, tbsa_levels=[20.0, 40.0])
    assert len(results) == 2


# ---- Additional checks ----


def test_drug_name_preserved():
    result = simulate_burn_pk("Vancomycin", dose_mg=1000.0, tbsa_pct=25.0)
    assert result.drug_name == "Vancomycin"


def test_tbsa_preserved():
    result = simulate_burn_pk("DrugY", dose_mg=500.0, tbsa_pct=45.0)
    assert result.tbsa_pct == pytest.approx(45.0)


def test_c_plasma_length_matches_times():
    result = simulate_burn_pk("DrugZ", dose_mg=500.0, tbsa_pct=20.0)
    assert len(result.c_plasma_mg_L) == len(result.times_h)
