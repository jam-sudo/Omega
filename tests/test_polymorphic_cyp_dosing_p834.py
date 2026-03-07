"""Tests for polymorphic CYP substrate dosing model (Phase 834)."""

import pytest

from omega_pbpk.clinical.polymorphic_cyp_dosing_p834 import (
    PolymorphicCYPDosingResult,
    predict_polymorphic_dosing,
    screen_phenotypes,
)

# ── Return type ───────────────────────────────────────────────────────────────


def test_return_type():
    result = predict_polymorphic_dosing(
        drug_name="Codeine",
        standard_dose_mg=30.0,
        cyp_enzyme="CYP2D6",
        phenotype="EM",
        fm_cyp=0.8,
        standard_cmax_mg_L=0.1,
    )
    assert isinstance(result, PolymorphicCYPDosingResult)


# ── EM baseline ───────────────────────────────────────────────────────────────


def test_em_cl_ratio_is_one():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "EM", 0.9, 1.0)
    assert result.cl_ratio_vs_em == pytest.approx(1.0)


def test_em_auc_ratio_is_one():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "EM", 0.9, 1.0)
    assert result.auc_ratio_vs_em == pytest.approx(1.0)


def test_em_recommended_dose_equals_standard():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "EM", 0.9, 1.0)
    assert result.recommended_dose_mg == pytest.approx(100.0)


def test_em_safety_concern_none():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "EM", 0.9, 1.0)
    assert result.safety_concern == "none"


# ── PM phenotype ──────────────────────────────────────────────────────────────


def test_pm_cl_ratio_less_than_em():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "PM", 0.5, 1.0)
    assert result.cl_ratio_vs_em < 1.0


def test_pm_auc_ratio_greater_than_one():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "PM", 0.5, 1.0)
    assert result.auc_ratio_vs_em > 1.0


def test_pm_high_fm_toxicity_risk():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "PM", 0.9, 1.0)
    assert result.safety_concern == "toxicity_risk"


def test_pm_low_fm_no_safety_concern():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "PM", 0.3, 1.0)
    assert result.safety_concern == "none"


def test_pm_cl_ratio_formula_at_max_fm():
    """PM with fm_cyp=1.0: cl_ratio = 1 - 1.0 * 0.9 = 0.1 (above the 0.05 clamp floor)."""
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "PM", 1.0, 1.0)
    assert result.cl_ratio_vs_em == pytest.approx(0.1, rel=1e-6)


# ── UM phenotype ──────────────────────────────────────────────────────────────


def test_um_cl_ratio_greater_than_one():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "UM", 0.5, 1.0)
    assert result.cl_ratio_vs_em > 1.0


def test_um_auc_ratio_less_than_one():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "UM", 0.5, 1.0)
    assert result.auc_ratio_vs_em < 1.0


def test_um_high_fm_efficacy_risk():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "UM", 0.9, 1.0)
    assert result.safety_concern == "efficacy_risk"


def test_um_cl_ratio_formula():
    fm = 0.6
    expected_cl = 1.0 + fm * 0.3
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "UM", fm, 1.0)
    assert result.cl_ratio_vs_em == pytest.approx(expected_cl)


# ── IM phenotype ──────────────────────────────────────────────────────────────


def test_im_cl_ratio_between_pm_and_em():
    pm = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "PM", 0.8, 1.0)
    im = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "IM", 0.8, 1.0)
    em = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "EM", 0.8, 1.0)
    assert pm.cl_ratio_vs_em < im.cl_ratio_vs_em < em.cl_ratio_vs_em


def test_im_high_fm_toxicity_risk():
    result = predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "IM", 0.9, 1.0)
    assert result.safety_concern == "toxicity_risk"


# ── screen_phenotypes ─────────────────────────────────────────────────────────


def test_screen_phenotypes_returns_four():
    results = screen_phenotypes("Drug", 100.0, "CYP2D6", 0.8, 1.0)
    assert len(results) == 4


def test_screen_phenotypes_order():
    results = screen_phenotypes("Drug", 100.0, "CYP2D6", 0.8, 1.0)
    assert [r.phenotype for r in results] == ["EM", "IM", "PM", "UM"]


def test_screen_phenotypes_result_type():
    results = screen_phenotypes("Drug", 100.0, "CYP2D6", 0.8, 1.0)
    for r in results:
        assert isinstance(r, PolymorphicCYPDosingResult)


# ── Validation errors ─────────────────────────────────────────────────────────


def test_invalid_fm_cyp_above_one():
    with pytest.raises(ValueError, match="fm_cyp"):
        predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "EM", 1.1, 1.0)


def test_invalid_fm_cyp_negative():
    with pytest.raises(ValueError, match="fm_cyp"):
        predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "EM", -0.1, 1.0)


def test_invalid_phenotype():
    with pytest.raises(ValueError, match="phenotype"):
        predict_polymorphic_dosing("Drug", 100.0, "CYP2D6", "NM", 0.5, 1.0)


def test_invalid_standard_dose():
    with pytest.raises(ValueError, match="standard_dose"):
        predict_polymorphic_dosing("Drug", 0.0, "CYP2D6", "EM", 0.5, 1.0)
