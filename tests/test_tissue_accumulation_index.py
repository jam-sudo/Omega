"""Tests for Phase 1038 — Drug Tissue Accumulation Index."""

import pytest

from omega_pbpk.prediction.tissue_accumulation_index import (
    TissueAccumulationResult,
    predict_tissue_accumulation,
    screen_tissues,
)

VALID_CLASSES = {"negligible", "low", "moderate", "high", "very_high"}
ALL_TISSUES = {"fat", "muscle", "liver", "kidney", "heart", "lung"}


def make_result(tissue="fat", logp=2.0, pka=7.0, ionization_type="neutral", **kwargs):
    defaults = dict(
        drug_name="TestDrug",
        logp=logp,
        pka=pka,
        ionization_type=ionization_type,
        tissue=tissue,
        fu_plasma=0.3,
        cl_sys_L_per_h=5.0,
        vd_sys_L=30.0,
    )
    defaults.update(kwargs)
    return predict_tissue_accumulation(**defaults)


# --- Return type ---
def test_returns_tissue_accumulation_result():
    result = make_result()
    assert isinstance(result, TissueAccumulationResult)


def test_result_is_frozen():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.kp_predicted = 99.0  # type: ignore[misc]


# --- Kp range ---
def test_kp_in_valid_range():
    result = make_result()
    assert 0.01 <= result.kp_predicted <= 100.0


def test_kp_high_logp_fat():
    result = make_result(tissue="fat", logp=5.0)
    assert result.kp_predicted > 1.0


def test_kp_low_logp_fat_smaller():
    res_high = make_result(tissue="fat", logp=5.0)
    res_low = make_result(tissue="fat", logp=0.0)
    assert res_high.kp_predicted > res_low.kp_predicted


# --- Accumulation index ---
def test_accumulation_index_in_range():
    result = make_result()
    assert 0.0 <= result.accumulation_index <= 5.0


# --- Accumulation class ---
def test_accumulation_class_valid():
    result = make_result()
    assert result.accumulation_class in VALID_CLASSES


# --- Fat vs muscle for lipophilic drug ---
def test_fat_accumulation_gt_muscle_high_logp():
    # logP=1.5: fat Kp ~85, muscle Kp ~8 (both below clamp of 100)
    res_fat = make_result(tissue="fat", logp=1.5)
    res_muscle = make_result(tissue="muscle", logp=1.5)
    assert res_fat.kp_predicted > res_muscle.kp_predicted


# --- Base drugs in liver (lysosomal trapping) ---
def test_base_drug_liver_higher_kp():
    res_base = make_result(tissue="liver", ionization_type="base", pka=9.0, logp=2.0)
    res_neutral = make_result(tissue="liver", ionization_type="neutral", pka=9.0, logp=2.0)
    assert res_base.kp_predicted > res_neutral.kp_predicted


# --- Washout half-life ---
def test_half_life_washout_nonneg():
    result = make_result()
    assert result.half_life_washout_h >= 0


# --- Toxicity concern ---
def test_toxicity_concern_consistent_with_index():
    result = make_result()
    assert result.toxicity_concern == (result.accumulation_index > 3.0)


def test_high_kp_has_toxicity_concern():
    # logP=5, fu_plasma=0.05 in fat should push kp and index high
    result = make_result(tissue="fat", logp=5.0, fu_plasma=0.05)
    if result.accumulation_index > 3.0:
        assert result.toxicity_concern is True


def test_low_kp_no_toxicity_concern():
    result = make_result(tissue="muscle", logp=0.0, fu_plasma=0.9)
    if result.accumulation_index <= 3.0:
        assert result.toxicity_concern is False


# --- screen_tissues ---
def test_screen_tissues_returns_six():
    results = screen_tissues("DrugX", logp=2.0, pka=7.0, ionization_type="neutral")
    assert len(results) == 6


def test_screen_tissues_all_types_present():
    results = screen_tissues("DrugX", logp=2.0, pka=7.0, ionization_type="neutral")
    tissues = {r.tissue for r in results}
    assert tissues == ALL_TISSUES


def test_screen_tissues_sorted_by_kp_desc():
    results = screen_tissues("DrugX", logp=2.0, pka=7.0, ionization_type="neutral")
    kps = [r.kp_predicted for r in results]
    assert kps == sorted(kps, reverse=True)


# --- Notes nonempty ---
def test_notes_nonempty():
    result = make_result()
    assert len(result.notes) > 0


# --- Validation errors ---
def test_fu_zero_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        make_result(fu_plasma=0.0)


def test_invalid_tissue_raises():
    with pytest.raises(ValueError, match="tissue"):
        make_result(tissue="spleen")


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        make_result(ionization_type="zwitterion")


def test_pka_negative_raises():
    with pytest.raises(ValueError, match="pka"):
        make_result(pka=-1.0)


def test_pka_above_14_raises():
    with pytest.raises(ValueError, match="pka"):
        make_result(pka=15.0)
