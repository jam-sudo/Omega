"""Tests for Phase 895 — Adipose Tissue Drug Distribution."""

import math
import pytest

from omega_pbpk.core.adipose_distribution_pk import (
    AdiposeDistributionResult,
    compare_bmi_groups,
    simulate_adipose_distribution,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


def test_returns_dataclass_instance():
    result = simulate_adipose_distribution("TestDrug", dose_mg=100.0)
    assert isinstance(result, AdiposeDistributionResult)


def test_basic_fields_populated():
    result = simulate_adipose_distribution("DrugA", dose_mg=200.0, logp=3.0, bmi=25.0)
    assert result.drug_name == "DrugA"
    assert result.dose_mg == 200.0
    assert result.logp == 3.0
    assert result.bmi == 25.0


def test_time_series_length_consistent():
    result = simulate_adipose_distribution("DrugA", dose_mg=100.0, t_end_h=10.0, dt_h=0.1)
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_adipose_mg_L) == n


def test_time_starts_at_zero():
    result = simulate_adipose_distribution("DrugA", dose_mg=100.0)
    assert result.times_h[0] == 0.0


def test_cmax_plasma_positive():
    result = simulate_adipose_distribution("DrugA", dose_mg=100.0)
    assert result.cmax_plasma > 0.0


def test_cmax_adipose_positive():
    result = simulate_adipose_distribution("DrugA", dose_mg=100.0, logp=3.0)
    assert result.cmax_adipose > 0.0


def test_auc_plasma_positive():
    result = simulate_adipose_distribution("DrugA", dose_mg=100.0)
    assert result.auc_plasma > 0.0


def test_auc_adipose_positive_for_lipophilic():
    result = simulate_adipose_distribution("DrugA", dose_mg=100.0, logp=3.0)
    assert result.auc_adipose > 0.0


# ---------------------------------------------------------------------------
# Physicochemical relationships
# ---------------------------------------------------------------------------


def test_higher_logp_increases_vd():
    r_low = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=1.0, bmi=25.0)
    r_high = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=4.0, bmi=25.0)
    assert r_high.vd_apparent_L > r_low.vd_apparent_L


def test_higher_logp_increases_t_half():
    r_low = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=1.0, bmi=25.0)
    r_high = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=4.0, bmi=25.0)
    assert r_high.t_half_terminal_h > r_low.t_half_terminal_h


def test_obese_patient_higher_vd_than_lean():
    r_lean = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=3.0, bmi=22.0)
    r_obese = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=3.0, bmi=35.0)
    assert r_obese.vd_apparent_L > r_lean.vd_apparent_L


def test_obesity_dose_adjustment_exceeds_one_for_obese():
    r = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=3.0, bmi=35.0)
    assert r.obesity_dose_adjustment > 1.0


def test_obesity_dose_adjustment_approx_one_for_normal_bmi():
    r = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=3.0, bmi=25.0)
    assert abs(r.obesity_dose_adjustment - 1.0) < 0.05


def test_dose_scaling_linear_cmax():
    r1 = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=2.0, bmi=25.0)
    r2 = simulate_adipose_distribution("Drug", dose_mg=200.0, logp=2.0, bmi=25.0)
    ratio = r2.cmax_plasma / r1.cmax_plasma
    assert abs(ratio - 2.0) < 0.1


# ---------------------------------------------------------------------------
# Notes classification
# ---------------------------------------------------------------------------


def test_notes_highly_lipophilic():
    result = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=5.0)
    assert "Highly lipophilic" in result.notes


def test_notes_moderate_lipophilicity():
    result = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=3.0)
    assert "Moderate" in result.notes


def test_notes_low_lipophilicity():
    result = simulate_adipose_distribution("Drug", dose_mg=100.0, logp=0.5)
    assert "Low" in result.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_negative_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_distribution("Drug", dose_mg=-50.0)


def test_raises_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_adipose_distribution("Drug", dose_mg=0.0)


def test_raises_low_bmi():
    with pytest.raises(ValueError, match="bmi"):
        simulate_adipose_distribution("Drug", dose_mg=100.0, bmi=5.0)


def test_raises_zero_clearance():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_adipose_distribution("Drug", dose_mg=100.0, cl_L_per_h=0.0)


# ---------------------------------------------------------------------------
# compare_bmi_groups
# ---------------------------------------------------------------------------


def test_compare_bmi_groups_returns_list():
    results = compare_bmi_groups("Drug", dose_mg=100.0, logp=3.0)
    assert isinstance(results, list)
    assert len(results) == 5


def test_compare_bmi_groups_sorted_ascending():
    results = compare_bmi_groups("Drug", dose_mg=100.0, logp=3.0)
    bmis = [r.bmi for r in results]
    assert bmis == sorted(bmis)


def test_compare_bmi_groups_custom_bmis():
    results = compare_bmi_groups("Drug", dose_mg=100.0, logp=2.0, bmis=[20.0, 30.0])
    assert len(results) == 2
    assert results[0].bmi < results[1].bmi


def test_compare_bmi_groups_vd_increases_with_bmi():
    results = compare_bmi_groups("Drug", dose_mg=100.0, logp=3.0)
    vds = [r.vd_apparent_L for r in results]
    # Each successive BMI should have equal or higher Vd
    for i in range(len(vds) - 1):
        assert vds[i + 1] >= vds[i]
