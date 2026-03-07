"""Tests for lymph node drug distribution — Phase 734."""

from __future__ import annotations

import pytest

from omega_pbpk.core.lymph_node_pk import (
    LymphNodePKResult,
    assess_lymph_targeting,
    simulate_lymph_node_pk,
)


def _run(**kwargs):
    defaults = dict(drug_name="TestDrug", dose_mg=10.0)
    defaults.update(kwargs)
    return simulate_lymph_node_pk(**defaults)


# --- Basic result ---


def test_returns_result_instance():
    result = _run()
    assert isinstance(result, LymphNodePKResult)


def test_cmax_plasma_positive():
    result = _run()
    assert result.cmax_plasma > 0


def test_cmax_node_mg_positive():
    result = _run()
    assert result.cmax_node_mg > 0


def test_auc_plasma_positive():
    result = _run()
    assert result.auc_plasma > 0


def test_cumulative_node_exposure_positive():
    result = _run()
    assert result.cumulative_node_exposure_mg_h > 0


def test_notes_nonempty():
    result = _run()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_f_via_lymphatics_stored():
    result = _run(f_lymph=0.2)
    assert result.f_via_lymphatics == pytest.approx(0.2)


# --- Lymphatic fraction effects ---


def test_higher_f_lymph_higher_node_accumulation():
    low = _run(f_lymph=0.05)
    high = _run(f_lymph=0.5)
    assert high.cmax_node_mg > low.cmax_node_mg


def test_f_lymph_zero_gives_zero_node_accumulation():
    result = _run(f_lymph=0.0)
    assert result.cmax_node_mg == pytest.approx(0.0, abs=1e-10)


# --- Linear dose scaling ---


def test_linear_dose_scaling_plasma():
    r1 = _run(dose_mg=5.0)
    r2 = _run(dose_mg=10.0)
    ratio = r2.cmax_plasma / r1.cmax_plasma
    assert abs(ratio - 2.0) < 0.05


def test_linear_dose_scaling_node():
    r1 = _run(dose_mg=5.0)
    r2 = _run(dose_mg=10.0)
    ratio = r2.cmax_node_mg / r1.cmax_node_mg
    assert abs(ratio - 2.0) < 0.05


# --- Time arrays ---


def test_times_and_lists_same_length():
    result = _run()
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.a_node_mg)
    assert len(result.times_h) == len(result.a_inj_mg)


def test_injection_site_decreases():
    result = _run()
    # Injection site amount should decrease over time
    assert result.a_inj_mg[0] >= result.a_inj_mg[-1]


# --- Validation ---


def test_dose_le_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _run(dose_mg=-1.0)


def test_f_lymph_negative_raises():
    with pytest.raises(ValueError, match="f_lymph"):
        _run(f_lymph=-0.1)


def test_f_lymph_gt_1_raises():
    with pytest.raises(ValueError, match="f_lymph"):
        _run(f_lymph=1.5)


def test_cl_le_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        _run(cl_L_per_h=0.0)


def test_vd_le_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        _run(vd_L=0.0)


# --- assess_lymph_targeting ---


def test_assess_returns_dict():
    result = _run()
    assessment = assess_lymph_targeting(result)
    assert isinstance(assessment, dict)


def test_assess_has_lymph_enrichment_key():
    result = _run()
    assessment = assess_lymph_targeting(result)
    assert "lymph_enrichment" in assessment


def test_assess_enrichment_is_string():
    result = _run()
    assessment = assess_lymph_targeting(result)
    assert isinstance(assessment["lymph_enrichment"], str)


def test_assess_enrichment_valid_values():
    result = _run()
    assessment = assess_lymph_targeting(result)
    assert assessment["lymph_enrichment"] in ("poor", "moderate", "good")


def test_assess_has_cmax_node():
    result = _run()
    assessment = assess_lymph_targeting(result)
    assert "cmax_node_mg" in assessment
    assert assessment["cmax_node_mg"] == result.cmax_node_mg


def test_assess_high_f_lymph_gives_good_enrichment():
    result = _run(f_lymph=0.9, dose_mg=100.0, t_end_h=120.0)
    assessment = assess_lymph_targeting(result)
    assert assessment["lymph_enrichment"] in ("moderate", "good")
