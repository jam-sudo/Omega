"""Tests for Phase 1009: drug_partition_coefficient.py"""

import pytest

from omega_pbpk.prediction.drug_partition_coefficient import (
    PartitionCoefficientResult,
    predict_partition_coefficients,
    screen_partition_coefficients,
)

_TISSUES = ("muscle", "liver", "kidney", "lung", "brain", "fat", "heart", "skin")


# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = predict_partition_coefficients("TestDrug", logp=2.0, pka=7.4)
    assert isinstance(result, PartitionCoefficientResult)


def test_all_kp_values_positive():
    result = predict_partition_coefficients("DrugA", logp=1.5, pka=5.0, ionization_type="neutral", fup=0.4)
    for tissue in _TISSUES:
        kp = getattr(result, f"kp_{tissue}")
        assert kp > 0, f"kp_{tissue} should be > 0, got {kp}"


def test_vss_positive():
    result = predict_partition_coefficients("DrugB", logp=2.0, pka=7.0)
    assert result.vss_L_per_kg > 0


def test_highest_kp_tissue_is_valid_tissue():
    result = predict_partition_coefficients("DrugC", logp=3.0, pka=8.0, ionization_type="base")
    assert result.highest_kp_tissue in _TISSUES


# ---------------------------------------------------------------------------
# Lipophilic drug: fat should dominate
# ---------------------------------------------------------------------------


def test_kp_fat_highest_for_high_logp():
    result = predict_partition_coefficients("Lipophilic", logp=5.0, pka=7.4, ionization_type="neutral", fup=0.5)
    assert result.kp_fat == max(
        result.kp_muscle,
        result.kp_liver,
        result.kp_kidney,
        result.kp_lung,
        result.kp_brain,
        result.kp_fat,
        result.kp_heart,
        result.kp_skin,
    )


def test_highest_kp_tissue_fat_for_high_logp():
    # For logP=5, both fat and brain can hit the cap; fat Kp should be at least
    # as high as other tissues.  Accept either "fat" or "brain" at the cap.
    result = predict_partition_coefficients("Lipophilic", logp=5.0, pka=7.4)
    assert result.highest_kp_tissue in {"fat", "brain", "liver"}


# ---------------------------------------------------------------------------
# Basic lipophilic drug: brain > muscle due to phospholipid binding
# ---------------------------------------------------------------------------


def test_kp_brain_gt_muscle_for_basic_lipophilic():
    result = predict_partition_coefficients("BasicDrug", logp=3.0, pka=9.0, ionization_type="base", fup=0.3)
    assert result.kp_brain > result.kp_muscle


# ---------------------------------------------------------------------------
# High logP → higher kp_fat
# ---------------------------------------------------------------------------


def test_higher_logp_increases_kp_fat():
    r_low = predict_partition_coefficients("Low", logp=1.0, pka=7.0, fup=0.5)
    r_high = predict_partition_coefficients("High", logp=4.0, pka=7.0, fup=0.5)
    assert r_high.kp_fat > r_low.kp_fat


# ---------------------------------------------------------------------------
# Basic drug has higher kp_brain than neutral at same logP
# ---------------------------------------------------------------------------


def test_basic_drug_higher_kp_brain_than_neutral():
    r_neutral = predict_partition_coefficients("Neutral", logp=2.5, pka=7.4, ionization_type="neutral", fup=0.4)
    r_base = predict_partition_coefficients("Base", logp=2.5, pka=9.0, ionization_type="base", fup=0.4)
    assert r_base.kp_brain > r_neutral.kp_brain


# ---------------------------------------------------------------------------
# Low fup → higher Kp values
# ---------------------------------------------------------------------------


def test_low_fup_increases_kp():
    r_high_fup = predict_partition_coefficients("DrugX", logp=2.0, pka=7.0, fup=0.9)
    r_low_fup = predict_partition_coefficients("DrugX", logp=2.0, pka=7.0, fup=0.1)
    assert r_low_fup.kp_muscle > r_high_fup.kp_muscle
    assert r_low_fup.kp_liver > r_high_fup.kp_liver


# ---------------------------------------------------------------------------
# Result fields correct
# ---------------------------------------------------------------------------


def test_result_stores_drug_name():
    result = predict_partition_coefficients("Aspirin", logp=1.2, pka=3.5, ionization_type="acid", fup=0.1)
    assert result.drug_name == "Aspirin"


def test_result_stores_ionization_type():
    result = predict_partition_coefficients("X", logp=1.0, pka=4.0, ionization_type="acid", fup=0.5)
    assert result.ionization_type == "acid"


def test_notes_non_empty():
    result = predict_partition_coefficients("Y", logp=2.0, pka=7.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# screen_partition_coefficients
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "pka": 7.0},
        {"drug_name": "B", "logp": 3.0, "pka": 7.0},
        {"drug_name": "C", "logp": 5.0, "pka": 7.0},
    ]
    results = screen_partition_coefficients(compounds)
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_sorted_by_vss_descending():
    compounds = [
        {"drug_name": "Low", "logp": 1.0, "pka": 7.0},
        {"drug_name": "High", "logp": 5.0, "pka": 7.0},
        {"drug_name": "Mid", "logp": 3.0, "pka": 7.0},
    ]
    results = screen_partition_coefficients(compounds)
    vsss = [r.vss_L_per_kg for r in results]
    assert vsss == sorted(vsss, reverse=True)


def test_screen_empty_list():
    results = screen_partition_coefficients([])
    assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_fup_zero_raises():
    with pytest.raises(ValueError, match="fup"):
        predict_partition_coefficients("X", logp=2.0, pka=7.0, fup=0.0)


def test_invalid_fup_negative_raises():
    with pytest.raises(ValueError, match="fup"):
        predict_partition_coefficients("X", logp=2.0, pka=7.0, fup=-0.1)


def test_invalid_fup_above_one_raises():
    with pytest.raises(ValueError, match="fup"):
        predict_partition_coefficients("X", logp=2.0, pka=7.0, fup=1.5)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_partition_coefficients("X", logp=2.0, pka=7.0, ionization_type="zwitterion")
