"""Tests for omega_pbpk.core.pbpk_tissue_binding module."""

import pytest

from omega_pbpk.core.pbpk_tissue_binding import (
    TissueBindingResult,
    _fut_from_kp_and_fup,
    _tissue_binding_correction,
    compare_binding_species,
    predict_tissue_binding,
)

# ---------------------------------------------------------------------------
# Basic result type and fields
# ---------------------------------------------------------------------------


def test_predict_returns_tissue_binding_result():
    result = predict_tissue_binding("DrugA", logP=1.0, fup=0.5)
    assert isinstance(result, TissueBindingResult)


def test_result_stores_drug_name():
    result = predict_tissue_binding("DrugA", logP=1.0, fup=0.5)
    assert result.drug_name == "DrugA"


def test_result_stores_fup():
    result = predict_tissue_binding("DrugA", logP=1.0, fup=0.5)
    assert result.fup == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# fut formula correctness
# ---------------------------------------------------------------------------


def test_fut_muscle_formula():
    logP = 2.0
    fup = 0.6
    expected = fup / (1 + 0.3 * logP)
    result = predict_tissue_binding("DrugA", logP=logP, fup=fup)
    assert result.fut_muscle == pytest.approx(expected, rel=1e-6)


def test_fut_adipose_formula():
    logP = 2.0
    fup = 0.6
    expected = fup / (1 + 0.5 * logP)
    result = predict_tissue_binding("DrugA", logP=logP, fup=fup)
    assert result.fut_adipose == pytest.approx(expected, rel=1e-6)


def test_fut_liver_formula():
    fup = 0.6
    result = predict_tissue_binding("DrugA", logP=0.0, fup=fup)
    assert result.fut_liver == pytest.approx(fup * 0.8, rel=1e-6)


def test_fut_brain_formula():
    logP = 2.0
    fup = 0.6
    expected = fup / (1 + 0.2 * logP)
    result = predict_tissue_binding("DrugA", logP=logP, fup=fup)
    assert result.fut_brain == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Lower bound clamping to 0.001
# ---------------------------------------------------------------------------


def test_fut_muscle_clamped_lower():
    # Very high logP should clamp fut_muscle to 0.001
    result = predict_tissue_binding("DrugX", logP=100.0, fup=0.01)
    assert result.fut_muscle >= 0.001


def test_fut_adipose_clamped_lower():
    result = predict_tissue_binding("DrugX", logP=100.0, fup=0.01)
    assert result.fut_adipose >= 0.001


def test_fut_liver_clamped_lower():
    # Very small fup*0.8 should still be >= 0.001
    result = predict_tissue_binding("DrugX", logP=0.0, fup=0.0001)
    # fup=0.0001 would give fut_liver=0.00008 -> clamped to 0.001
    assert result.fut_liver >= 0.001


def test_fut_brain_clamped_lower():
    result = predict_tissue_binding("DrugX", logP=100.0, fup=0.01)
    assert result.fut_brain >= 0.001


# ---------------------------------------------------------------------------
# Upper bound clamping to 1.0
# ---------------------------------------------------------------------------


def test_fut_liver_clamped_upper():
    # fup=1.0 gives fut_liver=0.8 which is below 1.0, use fup=1.0
    # but use a scenario where it would exceed 1 (shouldn't with valid fup<=1)
    result = predict_tissue_binding("DrugY", logP=0.0, fup=1.0)
    assert result.fut_liver <= 1.0


def test_fut_muscle_not_exceed_one():
    result = predict_tissue_binding("DrugY", logP=0.0, fup=1.0)
    assert result.fut_muscle <= 1.0


# ---------------------------------------------------------------------------
# Negative or zero logP: use max(logP, 0)
# ---------------------------------------------------------------------------


def test_negative_logP_treated_as_zero_for_fut_muscle():
    fup = 0.4
    # max(-2, 0) = 0 => fut_muscle = fup / 1 = fup
    result = predict_tissue_binding("DrugB", logP=-2.0, fup=fup)
    assert result.fut_muscle == pytest.approx(fup, rel=1e-6)


def test_negative_logP_treated_as_zero_for_fut_brain():
    fup = 0.4
    result = predict_tissue_binding("DrugB", logP=-3.0, fup=fup)
    assert result.fut_brain == pytest.approx(fup, rel=1e-6)


# ---------------------------------------------------------------------------
# kpu_to_kp_ratio
# ---------------------------------------------------------------------------


def test_kpu_to_kp_ratio_positive():
    result = predict_tissue_binding("DrugA", logP=1.0, fup=0.5)
    assert result.kpu_to_kp_ratio > 0


def test_kpu_to_kp_ratio_formula():
    logP = 2.0
    fup = 0.6
    result = predict_tissue_binding("DrugA", logP=logP, fup=fup)
    expected = fup / result.fut_muscle
    assert result.kpu_to_kp_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# corrected_kp populated when kp_dict provided
# ---------------------------------------------------------------------------


def test_corrected_kp_empty_when_no_kp_dict():
    result = predict_tissue_binding("DrugA", logP=1.0, fup=0.5)
    assert result.corrected_kp == {} or result.corrected_kp is None or len(result.corrected_kp) == 0


def test_corrected_kp_populated_when_kp_dict_given():
    kp_dict = {"muscle": 2.0, "liver": 5.0}
    result = predict_tissue_binding("DrugA", logP=1.0, fup=0.5, kp_dict=kp_dict)
    assert len(result.corrected_kp) > 0


def test_corrected_kp_has_correct_keys():
    kp_dict = {"muscle": 2.0, "liver": 5.0}
    result = predict_tissue_binding("DrugA", logP=1.0, fup=0.5, kp_dict=kp_dict)
    for key in kp_dict:
        assert key in result.corrected_kp


# ---------------------------------------------------------------------------
# Higher logP -> lower fut_muscle
# ---------------------------------------------------------------------------


def test_higher_logP_gives_lower_fut_muscle():
    fup = 0.5
    result_low = predict_tissue_binding("DrugA", logP=1.0, fup=fup)
    result_high = predict_tissue_binding("DrugA", logP=5.0, fup=fup)
    assert result_high.fut_muscle < result_low.fut_muscle


def test_higher_logP_gives_lower_fut_adipose():
    fup = 0.5
    result_low = predict_tissue_binding("DrugA", logP=1.0, fup=fup)
    result_high = predict_tissue_binding("DrugA", logP=5.0, fup=fup)
    assert result_high.fut_adipose < result_low.fut_adipose


# ---------------------------------------------------------------------------
# ValueError for invalid inputs
# ---------------------------------------------------------------------------


def test_raises_value_error_fup_zero():
    with pytest.raises(ValueError):
        predict_tissue_binding("DrugA", logP=1.0, fup=0.0)


def test_raises_value_error_fup_negative():
    with pytest.raises(ValueError):
        predict_tissue_binding("DrugA", logP=1.0, fup=-0.1)


def test_raises_value_error_fup_greater_than_one():
    with pytest.raises(ValueError):
        predict_tissue_binding("DrugA", logP=1.0, fup=1.1)


def test_raises_value_error_empty_drug_name():
    with pytest.raises(ValueError):
        predict_tissue_binding("", logP=1.0, fup=0.5)


# ---------------------------------------------------------------------------
# compare_binding_species
# ---------------------------------------------------------------------------


def test_compare_binding_species_human_only():
    result = compare_binding_species("DrugA", logP=1.0, fup_human=0.5)
    assert "human" in result
    assert isinstance(result["human"], TissueBindingResult)


def test_compare_binding_species_rat_included_when_given():
    result = compare_binding_species("DrugA", logP=1.0, fup_human=0.5, fup_rat=0.3)
    assert "rat" in result


def test_compare_binding_species_dog_included_when_given():
    result = compare_binding_species("DrugA", logP=1.0, fup_human=0.5, fup_dog=0.4)
    assert "dog" in result


def test_compare_binding_species_excludes_none_species():
    result = compare_binding_species("DrugA", logP=1.0, fup_human=0.5)
    assert "rat" not in result
    assert "dog" not in result


def test_compare_binding_species_all_three_species():
    result = compare_binding_species("DrugA", logP=1.0, fup_human=0.5, fup_rat=0.3, fup_dog=0.4)
    assert set(result.keys()) == {"human", "rat", "dog"}


def test_compare_binding_species_values_are_tissue_binding_results():
    result = compare_binding_species("DrugA", logP=1.0, fup_human=0.5, fup_rat=0.3)
    for v in result.values():
        assert isinstance(v, TissueBindingResult)


# ---------------------------------------------------------------------------
# _fut_from_kp_and_fup helper
# ---------------------------------------------------------------------------


def test_fut_from_kp_and_fup_formula():
    kp = 4.0
    fup = 0.8
    result = _fut_from_kp_and_fup(kp, fup)
    assert result == pytest.approx(fup / kp, rel=1e-6)


def test_fut_from_kp_and_fup_clamped_lower():
    # Very large kp -> fup/kp near zero -> clamped to 0.001
    result = _fut_from_kp_and_fup(kp_in_vivo=10000.0, fup=0.01)
    assert result >= 0.001


def test_fut_from_kp_and_fup_clamped_upper():
    # Very small kp -> fup/kp > 1 -> clamped to 1.0
    result = _fut_from_kp_and_fup(kp_in_vivo=0.1, fup=0.5)
    # 0.5/0.1 = 5.0 -> clamped to 1.0
    assert result <= 1.0


# ---------------------------------------------------------------------------
# _tissue_binding_correction helper
# ---------------------------------------------------------------------------


def test_tissue_binding_correction_formula():
    kpu = 3.0
    fup = 0.5
    fut = 0.25
    result = _tissue_binding_correction(kpu, fup, fut)
    assert result == pytest.approx(kpu * fup / fut, rel=1e-6)


def test_tissue_binding_correction_increases_with_kpu():
    fup = 0.5
    fut = 0.25
    result_low = _tissue_binding_correction(kpu=1.0, fup=fup, fut=fut)
    result_high = _tissue_binding_correction(kpu=5.0, fup=fup, fut=fut)
    assert result_high > result_low


def test_tissue_binding_correction_positive():
    result = _tissue_binding_correction(kpu=2.0, fup=0.4, fut=0.2)
    assert result > 0
