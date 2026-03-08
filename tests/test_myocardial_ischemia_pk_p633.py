"""Tests for Phase 633 — Drug Myocardial Ischemia PK"""

import math

import pytest

from omega_pbpk.clinical.myocardial_ischemia_pk_p633 import (
    MyocardialIschemiaPKResult,
    ischemia_severity_comparison,
    predict_ischemia_pk,
)

LN2 = math.log(2.0)

# ─── Fixtures ────────────────────────────────────────────────────────────────

DRUG = "TestDrug"
CL_NORM = 10.0  # L/h
VD_NORM = 50.0  # L
F_HEP = 0.6
F_REN = 0.3


def result_mild():
    return predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, "mild", F_HEP, F_REN)


def result_moderate():
    return predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, "moderate", F_HEP, F_REN)


def result_severe():
    return predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, "severe", F_HEP, F_REN)


# ─── Return type & frozen dataclass ──────────────────────────────────────────


def test_returns_correct_type():
    res = result_mild()
    assert isinstance(res, MyocardialIschemiaPKResult)


def test_frozen_dataclass_immutable():
    res = result_mild()
    with pytest.raises((AttributeError, TypeError)):
        res.drug_name = "other"  # type: ignore[misc]


def test_drug_name_preserved():
    res = predict_ischemia_pk("Aspirin", CL_NORM, VD_NORM, "mild")
    assert res.drug_name == "Aspirin"


def test_ischemia_severity_preserved():
    for sev in ("mild", "moderate", "severe"):
        res = predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, sev)
        assert res.ischemia_severity == sev


# ─── Cardiac output factors ───────────────────────────────────────────────────


def test_cardiac_output_factor_mild():
    assert result_mild().cardiac_output_factor == pytest.approx(0.85)


def test_cardiac_output_factor_moderate():
    assert result_moderate().cardiac_output_factor == pytest.approx(0.65)


def test_cardiac_output_factor_severe():
    assert result_severe().cardiac_output_factor == pytest.approx(0.40)


# ─── CL reduction by severity ────────────────────────────────────────────────


def _expected_cl(co_factor):
    hepatic_perf = co_factor
    renal_perf = max(0.1, min(1.0, co_factor * 0.8))
    cl_hep = CL_NORM * F_HEP * hepatic_perf
    cl_ren = CL_NORM * F_REN * renal_perf
    cl_other = CL_NORM * (1 - F_HEP - F_REN)
    return cl_hep + cl_ren + cl_other


def test_cl_ischemia_mild():
    expected = _expected_cl(0.85)
    assert result_mild().cl_ischemia_L_per_h == pytest.approx(expected, rel=1e-6)


def test_cl_ischemia_moderate():
    expected = _expected_cl(0.65)
    assert result_moderate().cl_ischemia_L_per_h == pytest.approx(expected, rel=1e-6)


def test_cl_ischemia_severe():
    expected = _expected_cl(0.40)
    assert result_severe().cl_ischemia_L_per_h == pytest.approx(expected, rel=1e-6)


def test_cl_ischemia_less_than_normal():
    for sev in ("mild", "moderate", "severe"):
        res = predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, sev)
        assert res.cl_ischemia_L_per_h < res.cl_normal_L_per_h


def test_cl_decreases_with_severity():
    mild = result_mild()
    moderate = result_moderate()
    severe = result_severe()
    assert mild.cl_ischemia_L_per_h > moderate.cl_ischemia_L_per_h > severe.cl_ischemia_L_per_h


# ─── Perfusion reductions ─────────────────────────────────────────────────────


def test_hepatic_perfusion_mild():
    assert result_mild().hepatic_perfusion_reduction == pytest.approx(0.85)


def test_renal_perfusion_mild():
    expected = max(0.1, min(1.0, 0.85 * 0.8))
    assert result_mild().renal_perfusion_reduction == pytest.approx(expected)


def test_renal_perfusion_severe_clamped():
    # severe: co=0.40, renal_perf = 0.40*0.8 = 0.32 (above 0.1)
    assert result_severe().renal_perfusion_reduction == pytest.approx(0.32, rel=1e-6)


def test_renal_perfusion_never_below_0_1():
    # Extreme: simulate with a very low co scenario manually
    # Severe co=0.40 => renal=0.32; still above 0.1; clamp only if co < 0.125
    res = result_severe()
    assert res.renal_perfusion_reduction >= 0.1


# ─── Vd changes ──────────────────────────────────────────────────────────────


def test_vd_unchanged_mild():
    assert result_mild().vd_ischemia_L == pytest.approx(VD_NORM * 1.0)


def test_vd_increased_moderate():
    assert result_moderate().vd_ischemia_L == pytest.approx(VD_NORM * 1.1)


def test_vd_decreased_severe():
    assert result_severe().vd_ischemia_L == pytest.approx(VD_NORM * 0.9)


# ─── Half-life ────────────────────────────────────────────────────────────────


def test_t_half_normal():
    expected = LN2 * VD_NORM / CL_NORM
    assert result_mild().t_half_normal_h == pytest.approx(expected)


def test_t_half_ischemia_longer_than_normal():
    for sev in ("mild", "moderate", "severe"):
        res = predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, sev)
        assert res.t_half_ischemia_h > res.t_half_normal_h


# ─── AUC ratio ────────────────────────────────────────────────────────────────


def test_auc_ratio_is_cl_normal_over_cl_ischemia():
    res = result_mild()
    expected = res.cl_normal_L_per_h / res.cl_ischemia_L_per_h
    assert res.auc_ratio == pytest.approx(expected)


def test_auc_ratio_greater_than_1():
    for sev in ("mild", "moderate", "severe"):
        res = predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, sev)
        assert res.auc_ratio > 1.0


# ─── Dose adjustment factor ───────────────────────────────────────────────────


def test_dose_adjustment_factor_less_than_1():
    for sev in ("mild", "moderate", "severe"):
        res = predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, sev)
        assert res.dose_adjustment_factor < 1.0


def test_dose_adjustment_times_auc_ratio_equals_1():
    res = result_moderate()
    assert res.dose_adjustment_factor * res.auc_ratio == pytest.approx(1.0, rel=1e-6)


# ─── Severity comparison ──────────────────────────────────────────────────────


def test_severity_comparison_returns_list():
    results = ischemia_severity_comparison(DRUG, CL_NORM, VD_NORM)
    assert isinstance(results, list)
    assert len(results) == 3


def test_severity_comparison_sorted_by_cl_descending():
    results = ischemia_severity_comparison(DRUG, CL_NORM, VD_NORM)
    cls = [r.cl_ischemia_L_per_h for r in results]
    assert cls == sorted(cls, reverse=True)


def test_severity_comparison_all_severities_present():
    results = ischemia_severity_comparison(DRUG, CL_NORM, VD_NORM)
    severities = {r.ischemia_severity for r in results}
    assert severities == {"mild", "moderate", "severe"}


# ─── Validation ───────────────────────────────────────────────────────────────


def test_invalid_severity_raises():
    with pytest.raises(ValueError, match="ischemia_severity"):
        predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, "extreme")


def test_negative_cl_raises():
    with pytest.raises(ValueError, match="cl_normal_L_per_h"):
        predict_ischemia_pk(DRUG, -1.0, VD_NORM, "mild")


def test_zero_vd_raises():
    with pytest.raises(ValueError, match="vd_normal_L"):
        predict_ischemia_pk(DRUG, CL_NORM, 0.0, "mild")


def test_f_hepatic_out_of_range_raises():
    with pytest.raises(ValueError, match="f_hepatic"):
        predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, "mild", f_hepatic=1.5)


def test_f_renal_out_of_range_raises():
    with pytest.raises(ValueError, match="f_renal"):
        predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, "mild", f_renal=-0.1)


def test_f_sum_exceeds_1_raises():
    with pytest.raises(ValueError, match=r"f_hepatic \+ f_renal"):
        predict_ischemia_pk(DRUG, CL_NORM, VD_NORM, "mild", f_hepatic=0.7, f_renal=0.5)


def test_notes_is_nonempty_string():
    res = result_mild()
    assert isinstance(res.notes, str)
    assert len(res.notes) > 0
