"""Tests for Phase 976 — Drug Membrane Partitioning Prediction."""

import pytest

from omega_pbpk.prediction.drug_membrane_partitioning import (
    MembranePartitioningResult,
    compare_membrane_partitioning,
    predict_membrane_partitioning,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = predict_membrane_partitioning("TestDrug", logp=2.0, pka=7.0)
    assert isinstance(result, MembranePartitioningResult)


def test_kp_mem_positive():
    result = predict_membrane_partitioning("Drug", logp=2.0, pka=7.0)
    assert result.kp_mem > 0.0


def test_f_unbound_membrane_in_range():
    result = predict_membrane_partitioning("Drug", logp=2.0, pka=7.0)
    assert 0.0 < result.f_unbound_membrane <= 1.0


def test_mic_binding_factor_gte_one():
    result = predict_membrane_partitioning("Drug", logp=2.0, pka=7.0)
    assert result.mic_binding_factor >= 1.0


def test_accumulation_risk_valid_values():
    result = predict_membrane_partitioning("Drug", logp=2.0, pka=7.0)
    assert result.accumulation_risk in {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Ionization and lipophilicity effects
# ---------------------------------------------------------------------------


def test_basic_lipophilic_higher_kp_than_neutral():
    """Basic drug (base) with same logP has higher kp_mem than neutral drug."""
    basic = predict_membrane_partitioning("BasicDrug", logp=3.0, pka=9.0, ionization_type="base")
    neutral = predict_membrane_partitioning(
        "NeutralDrug", logp=3.0, pka=9.0, ionization_type="neutral"
    )
    assert basic.kp_mem > neutral.kp_mem


def test_lipophilic_higher_kp_than_hydrophilic():
    """logP=4 drug has higher kp_mem than logP=-1 drug."""
    lipophilic = predict_membrane_partitioning("Lipo", logp=4.0, pka=7.0)
    hydrophilic = predict_membrane_partitioning("Hydro", logp=-1.0, pka=7.0)
    assert lipophilic.kp_mem > hydrophilic.kp_mem


def test_log_kp_mem_monotone_with_logp():
    """Increasing logP should increase log_kp_mem for neutral drug."""
    r1 = predict_membrane_partitioning("D1", logp=1.0, pka=7.0)
    r2 = predict_membrane_partitioning("D2", logp=3.0, pka=7.0)
    r3 = predict_membrane_partitioning("D3", logp=5.0, pka=7.0)
    assert r1.log_kp_mem < r2.log_kp_mem < r3.log_kp_mem


# ---------------------------------------------------------------------------
# Lysosomal trapping
# ---------------------------------------------------------------------------


def test_lysosomal_trapping_true_for_basic_lipophilic():
    """Basic drug with logP > 1 → trapping risk True."""
    result = predict_membrane_partitioning("BasicLipo", logp=2.0, pka=9.0, ionization_type="base")
    assert result.lysosomal_trapping_risk is True


def test_lysosomal_trapping_false_for_neutral():
    """Neutral drug → no lysosomal trapping."""
    result = predict_membrane_partitioning("Neutral", logp=3.0, pka=7.0, ionization_type="neutral")
    assert result.lysosomal_trapping_risk is False


def test_lysosomal_trapping_false_for_basic_hydrophilic():
    """Basic drug with logP <= 1 → no lysosomal trapping risk."""
    result = predict_membrane_partitioning("BasicHydro", logp=0.5, pka=9.0, ionization_type="base")
    assert result.lysosomal_trapping_risk is False


def test_lysosomal_trapping_false_for_acid():
    """Acid drug → no lysosomal trapping."""
    result = predict_membrane_partitioning("AcidDrug", logp=3.0, pka=4.5, ionization_type="acid")
    assert result.lysosomal_trapping_risk is False


# ---------------------------------------------------------------------------
# Accumulation risk levels
# ---------------------------------------------------------------------------


def test_high_accumulation_risk():
    """Very lipophilic basic drug → high accumulation risk."""
    result = predict_membrane_partitioning("VeryLipo", logp=8.0, pka=9.0, ionization_type="base")
    assert result.accumulation_risk == "high"


def test_low_accumulation_risk():
    """Hydrophilic drug → low accumulation risk."""
    result = predict_membrane_partitioning("Hydro", logp=-1.0, pka=7.0, ionization_type="neutral")
    assert result.accumulation_risk == "low"


# ---------------------------------------------------------------------------
# compare_membrane_partitioning
# ---------------------------------------------------------------------------


def test_compare_returns_list():
    drugs = [
        {"drug_name": "A", "logp": 1.0, "pka": 7.0},
        {"drug_name": "B", "logp": 3.0, "pka": 7.0},
    ]
    results = compare_membrane_partitioning(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_compare_sorted_by_kp_mem_descending():
    drugs = [
        {"drug_name": "Hydro", "logp": -1.0, "pka": 7.0},
        {"drug_name": "Mid", "logp": 2.0, "pka": 7.0},
        {"drug_name": "Lipo", "logp": 5.0, "pka": 7.0},
    ]
    results = compare_membrane_partitioning(drugs)
    kp_mems = [r.kp_mem for r in results]
    assert kp_mems == sorted(kp_mems, reverse=True)


def test_compare_empty_list():
    results = compare_membrane_partitioning([])
    assert results == []


def test_compare_preserves_results_type():
    drugs = [{"drug_name": "X", "logp": 2.0, "pka": 7.0, "ionization_type": "base"}]
    results = compare_membrane_partitioning(drugs)
    assert isinstance(results[0], MembranePartitioningResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_pka_below_zero_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_membrane_partitioning("Drug", logp=2.0, pka=-1.0)


def test_invalid_pka_above_14_raises():
    with pytest.raises(ValueError, match="pka"):
        predict_membrane_partitioning("Drug", logp=2.0, pka=15.0)


def test_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        predict_membrane_partitioning("Drug", logp=2.0, pka=7.0, ionization_type="zwitterion")


def test_invalid_logp_below_range_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_membrane_partitioning("Drug", logp=-6.0, pka=7.0)


def test_invalid_logp_above_range_raises():
    with pytest.raises(ValueError, match="logp"):
        predict_membrane_partitioning("Drug", logp=11.0, pka=7.0)
