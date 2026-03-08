"""Tests for Phase 992 — Red Blood Cell Partitioning."""

import pytest

from omega_pbpk.core.red_blood_cell_pk import (
    RBCPartitioningResult,
    compare_hematocrit_effect,
    predict_rbc_partitioning,
)

# ---------------------------------------------------------------------------
# Basic return-type and field sanity
# ---------------------------------------------------------------------------


def test_returns_result_type():
    result = predict_rbc_partitioning("TestDrug", logp=2.0, pka=7.0)
    assert isinstance(result, RBCPartitioningResult)


def test_kp_rbc_positive():
    result = predict_rbc_partitioning("TestDrug", logp=2.0, pka=7.0)
    assert result.kp_rbc > 0


def test_blood_plasma_ratio_positive():
    result = predict_rbc_partitioning("TestDrug", logp=2.0, pka=7.0)
    assert result.blood_plasma_ratio > 0


def test_f_in_rbc_in_range():
    result = predict_rbc_partitioning("TestDrug", logp=2.0, pka=7.0)
    assert 0.0 <= result.f_in_rbc <= 1.0


def test_plasma_to_blood_correction_positive():
    result = predict_rbc_partitioning("TestDrug", logp=2.0, pka=7.0)
    assert result.plasma_to_blood_correction > 0


def test_rbc_partitioning_class_valid():
    result = predict_rbc_partitioning("TestDrug", logp=2.0, pka=7.0)
    assert result.rbc_partitioning_class in {"RBC-avid", "balanced", "plasma-preferred"}


def test_hemolysis_risk_is_bool():
    result = predict_rbc_partitioning("TestDrug", logp=2.0, pka=7.0)
    assert isinstance(result.hemolysis_risk, bool)


# ---------------------------------------------------------------------------
# Physicochemical relationships
# ---------------------------------------------------------------------------


def test_lipophilic_drug_higher_kp_than_hydrophilic():
    lipophilic = predict_rbc_partitioning("LipoDrug", logp=4.0, pka=7.0)
    hydrophilic = predict_rbc_partitioning("HydroDrug", logp=-2.0, pka=7.0)
    assert lipophilic.kp_rbc > hydrophilic.kp_rbc


def test_basic_drug_higher_kp_than_neutral_same_logp():
    # Basic drug with high pKa → RBC sequestration
    basic = predict_rbc_partitioning("BasicDrug", logp=2.0, pka=9.5, ionization_type="base")
    neutral = predict_rbc_partitioning("NeutralDrug", logp=2.0, pka=9.5, ionization_type="neutral")
    assert basic.kp_rbc >= neutral.kp_rbc


def test_high_hematocrit_higher_blood_plasma_ratio_for_rbc_avid():
    """For a lipophilic (RBC-avid) drug, higher hct should increase Rb."""
    low_hct = predict_rbc_partitioning("Drug", logp=4.0, pka=7.0, hematocrit=0.30)
    high_hct = predict_rbc_partitioning("Drug", logp=4.0, pka=7.0, hematocrit=0.55)
    # Only meaningful if kp_rbc > 1 (drug prefers RBCs)
    if low_hct.kp_rbc > 1.0:
        assert high_hct.blood_plasma_ratio > low_hct.blood_plasma_ratio


def test_plasma_to_blood_correction_inverse_of_rb():
    result = predict_rbc_partitioning("TestDrug", logp=2.0, pka=7.0)
    expected = 1.0 / result.blood_plasma_ratio
    assert abs(result.plasma_to_blood_correction - expected) < 1e-9


def test_hemolysis_risk_for_very_lipophilic():
    # Very lipophilic basic drug likely has kp_rbc > 10
    result = predict_rbc_partitioning("HighRisk", logp=5.0, pka=10.0, ionization_type="base")
    # kp_rbc driven by logP=5 → log_kp = 2.5-0.3=2.2 → kp~158, then *enhancement
    # hemolysis_risk should be True
    assert result.hemolysis_risk is True


def test_no_hemolysis_risk_for_hydrophilic():
    result = predict_rbc_partitioning("SafeDrug", logp=-1.0, pka=7.0)
    # kp_rbc = max(0.1, 10^(-0.5-0.3)) = max(0.1, 10^-0.8) ~0.16
    assert result.hemolysis_risk is False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_rbc_avid_classification():
    # Need blood_plasma_ratio > 2: e.g. hct=0.45, kp_rbc needs > ~3.2
    result = predict_rbc_partitioning("RBCAvid", logp=4.0, pka=10.0, ionization_type="base")
    if result.blood_plasma_ratio > 2.0:
        assert result.rbc_partitioning_class == "RBC-avid"


def test_plasma_preferred_classification():
    # Need blood_plasma_ratio < 0.5: e.g. low logP acidic drug
    result = predict_rbc_partitioning("PlasmaDrug", logp=-2.0, pka=3.0, ionization_type="acid")
    if result.blood_plasma_ratio < 0.5:
        assert result.rbc_partitioning_class == "plasma-preferred"


# ---------------------------------------------------------------------------
# compare_hematocrit_effect
# ---------------------------------------------------------------------------


def test_compare_hematocrit_returns_correct_length():
    hct_values = [0.30, 0.40, 0.45, 0.50, 0.60]
    results = compare_hematocrit_effect(
        "Drug", logp=2.0, pka=7.0, ionization_type="neutral", hct_values=hct_values
    )
    assert len(results) == len(hct_values)


def test_compare_hematocrit_returns_result_types():
    hct_values = [0.35, 0.45]
    results = compare_hematocrit_effect(
        "Drug", logp=2.0, pka=7.0, ionization_type="neutral", hct_values=hct_values
    )
    assert all(isinstance(r, RBCPartitioningResult) for r in results)


def test_compare_hematocrit_empty_list():
    results = compare_hematocrit_effect(
        "Drug", logp=2.0, pka=7.0, ionization_type="neutral", hct_values=[]
    )
    assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_hematocrit_raises():
    with pytest.raises(ValueError, match="hematocrit"):
        predict_rbc_partitioning("Drug", logp=2.0, pka=7.0, hematocrit=1.2)


def test_invalid_hematocrit_zero_raises():
    with pytest.raises(ValueError, match="hematocrit"):
        predict_rbc_partitioning("Drug", logp=2.0, pka=7.0, hematocrit=0.0)


def test_invalid_pka_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_rbc_partitioning("Drug", logp=2.0, pka=-1.0)


def test_invalid_pka_above_14_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_rbc_partitioning("Drug", logp=2.0, pka=15.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_rbc_partitioning("Drug", logp=2.0, pka=7.0, ionization_type="cation")
