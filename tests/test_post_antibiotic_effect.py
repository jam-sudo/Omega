"""Tests for Phase 532: Antibiotic Post-Antibiotic Effect PK/PD."""

import math

import pytest

from omega_pbpk.clinical.post_antibiotic_effect import (
    PostAntibioticEffectResult,
    compare_mic_levels,
    simulate_post_antibiotic_effect,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(
    drug="gentamicin",
    cls="aminoglycoside",
    dose=500.0,
    mic=1.0,
    cl=5.0,
    vd=20.0,
    t_end=24.0,
):
    return simulate_post_antibiotic_effect(drug, cls, dose, mic, cl, vd, t_end)


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run()
    assert isinstance(result, PostAntibioticEffectResult)


def test_result_fields_present():
    result = _run()
    assert hasattr(result, "drug_name")
    assert hasattr(result, "antibiotic_class")
    assert hasattr(result, "dose_mg")
    assert hasattr(result, "mic_mg_L")
    assert hasattr(result, "cmax_mg_L")
    assert hasattr(result, "cmax_mic_ratio")
    assert hasattr(result, "auc_mg_h_per_L")
    assert hasattr(result, "auc_mic_ratio")
    assert hasattr(result, "t_above_mic_h")
    assert hasattr(result, "pae_h")
    assert hasattr(result, "effective_duration_h")
    assert hasattr(result, "pk_pd_index")
    assert hasattr(result, "efficacy_prediction")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Analytical PK
# ---------------------------------------------------------------------------


def test_cmax_equals_dose_over_vd():
    result = _run(dose=400.0, vd=20.0)
    assert result.cmax_mg_L == pytest.approx(400.0 / 20.0, rel=1e-6)


def test_auc_equals_dose_over_cl():
    result = _run(dose=400.0, cl=8.0)
    assert result.auc_mg_h_per_L == pytest.approx(400.0 / 8.0, rel=1e-6)


def test_cmax_mic_ratio_above_one_when_above_mic():
    # dose=500, vd=20 -> Cmax=25; mic=1 -> ratio=25
    result = _run(dose=500.0, vd=20.0, mic=1.0)
    assert result.cmax_mic_ratio > 1.0


def test_cmax_mic_ratio_below_one_when_below_mic():
    # dose=10, vd=20 -> Cmax=0.5; mic=1 -> ratio=0.5
    result = _run(dose=10.0, vd=20.0, mic=1.0)
    assert result.cmax_mic_ratio < 1.0


# ---------------------------------------------------------------------------
# Time above MIC
# ---------------------------------------------------------------------------


def test_t_above_mic_positive_when_cmax_exceeds_mic():
    # Cmax=25 > MIC=1
    result = _run(dose=500.0, vd=20.0, mic=1.0)
    assert result.t_above_mic_h > 0.0


def test_t_above_mic_zero_when_cmax_below_mic():
    # Cmax=0.5 < MIC=1
    result = _run(dose=10.0, vd=20.0, mic=1.0)
    assert result.t_above_mic_h == pytest.approx(0.0, abs=1e-12)


def test_t_above_mic_analytical():
    # C(t) = Cmax * exp(-k_el * t); k_el = CL/Vd = 5/20 = 0.25/h
    # t_mic = (1/k_el) * ln(Cmax/MIC) = 4 * ln(25/1)
    dose, vd, cl, mic = 500.0, 20.0, 5.0, 1.0
    cmax = dose / vd  # 25
    k_el = cl / vd  # 0.25
    expected = (1.0 / k_el) * math.log(cmax / mic)
    result = _run(dose=dose, vd=vd, cl=cl, mic=mic)
    assert result.t_above_mic_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# PAE duration
# ---------------------------------------------------------------------------


def test_aminoglycoside_pae_greater_than_beta_lactam_pae():
    r_ag = simulate_post_antibiotic_effect("gent", "aminoglycoside", 500.0, 1.0, 5.0, 20.0)
    r_bl = simulate_post_antibiotic_effect("amp", "beta_lactam", 500.0, 1.0, 5.0, 20.0)
    assert r_ag.pae_h > r_bl.pae_h


def test_macrolide_pae_in_range():
    result = simulate_post_antibiotic_effect("erythro", "macrolide", 500.0, 1.0, 5.0, 20.0)
    assert 2.0 <= result.pae_h <= 8.0


def test_beta_lactam_pae_in_range():
    result = simulate_post_antibiotic_effect("amp", "beta_lactam", 500.0, 1.0, 5.0, 20.0)
    assert 0.0 <= result.pae_h <= 1.0


def test_oxazolidinone_pae_in_range():
    result = simulate_post_antibiotic_effect("linezolid", "oxazolidinone", 600.0, 2.0, 6.0, 40.0)
    assert 1.0 <= result.pae_h <= 3.0


# ---------------------------------------------------------------------------
# Effective duration
# ---------------------------------------------------------------------------


def test_effective_duration_gte_t_above_mic():
    result = _run()
    assert result.effective_duration_h >= result.t_above_mic_h


def test_effective_duration_gte_pae():
    result = _run()
    assert result.effective_duration_h >= result.pae_h


def test_effective_duration_equals_t_plus_pae():
    result = _run()
    assert result.effective_duration_h == pytest.approx(
        result.t_above_mic_h + result.pae_h, rel=1e-9
    )


# ---------------------------------------------------------------------------
# PK/PD index
# ---------------------------------------------------------------------------


def test_pk_pd_index_aminoglycoside():
    result = _run(cls="aminoglycoside")
    assert result.pk_pd_index == "AUC/MIC"


def test_pk_pd_index_fluoroquinolone():
    result = simulate_post_antibiotic_effect("cipro", "fluoroquinolone", 400.0, 0.5, 5.0, 30.0)
    assert result.pk_pd_index == "AUC/MIC"


def test_pk_pd_index_beta_lactam():
    result = simulate_post_antibiotic_effect("amp", "beta_lactam", 2000.0, 1.0, 20.0, 15.0)
    assert result.pk_pd_index == "T>MIC"


# ---------------------------------------------------------------------------
# Efficacy classification
# ---------------------------------------------------------------------------


def test_optimal_efficacy_high_auc_mic():
    # AUC/MIC = (dose/CL) / mic; need > 125
    # dose=2500, cl=5, mic=1 -> AUC=500, AUC/MIC=500 -> optimal (aminoglycoside)
    result = simulate_post_antibiotic_effect("gent", "aminoglycoside", 2500.0, 1.0, 5.0, 20.0)
    assert result.efficacy_prediction == "optimal"


def test_suboptimal_efficacy_low_auc_mic():
    # AUC/MIC = (50/5)/1 = 10 < 125 -> suboptimal (aminoglycoside)
    result = simulate_post_antibiotic_effect("gent", "aminoglycoside", 50.0, 1.0, 5.0, 20.0)
    assert result.efficacy_prediction == "suboptimal"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_class_raises():
    with pytest.raises(ValueError, match="antibiotic_class"):
        simulate_post_antibiotic_effect("drug", "invalid_class", 500.0, 1.0, 5.0, 20.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_post_antibiotic_effect("drug", "aminoglycoside", 0.0, 1.0, 5.0, 20.0)


def test_invalid_mic_raises():
    with pytest.raises(ValueError, match="mic_mg_L"):
        simulate_post_antibiotic_effect("drug", "aminoglycoside", 500.0, 0.0, 5.0, 20.0)


def test_invalid_cl_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_post_antibiotic_effect("drug", "aminoglycoside", 500.0, 1.0, 0.0, 20.0)


def test_invalid_vd_raises():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_post_antibiotic_effect("drug", "aminoglycoside", 500.0, 1.0, 5.0, 0.0)


# ---------------------------------------------------------------------------
# compare_mic_levels
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    results = compare_mic_levels("gent", "aminoglycoside", 500.0, [0.5, 1.0, 4.0], 5.0, 20.0)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_sorted_by_effective_duration_descending():
    results = compare_mic_levels("gent", "aminoglycoside", 500.0, [0.5, 1.0, 4.0], 5.0, 20.0)
    durations = [r.effective_duration_h for r in results]
    assert durations == sorted(durations, reverse=True)


def test_compare_all_are_pae_result():
    results = compare_mic_levels("amp", "beta_lactam", 2000.0, [0.25, 1.0], 20.0, 15.0)
    for r in results:
        assert isinstance(r, PostAntibioticEffectResult)
