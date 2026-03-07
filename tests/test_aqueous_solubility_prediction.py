"""Tests for Phase 958 — Aqueous solubility prediction."""

import math

import pytest

from omega_pbpk.prediction.aqueous_solubility_prediction import (
    AqueousSolubilityResult,
    predict_aqueous_solubility,
    screen_solubility,
)

_VALID_CLASSES = {
    "highly_soluble",
    "soluble",
    "sparingly_soluble",
    "poorly_soluble",
    "practically_insoluble",
}


# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


def test_return_type():
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    assert isinstance(result, AqueousSolubilityResult)


def test_frozen_dataclass():
    """AqueousSolubilityResult should be immutable (frozen)."""
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    with pytest.raises(Exception):
        result.drug_name = "other"  # type: ignore


def test_yalkowsky_positive():
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    assert result.solubility_yalkowsky_mg_mL > 0


def test_esol_positive():
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    assert result.solubility_esol_mg_mL > 0


def test_consensus_positive():
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    assert result.solubility_consensus_mg_mL > 0


def test_solubility_class_valid():
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    assert result.solubility_class in _VALID_CLASSES


def test_bcs_dose_number_positive():
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    assert result.bcs_dose_number > 0


def test_high_solubility_bcs_is_bool():
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    assert isinstance(result.high_solubility_bcs, bool)


def test_notes_nonempty():
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_drug_name_preserved():
    result = predict_aqueous_solubility("TestCompound", logp=2.0, mw=300.0)
    assert result.drug_name == "TestCompound"


# ---------------------------------------------------------------------------
# Physical relationships
# ---------------------------------------------------------------------------


def test_high_logp_lower_solubility():
    """Higher logP → lower solubility."""
    low_logp = predict_aqueous_solubility("compound_A", logp=0.5, mw=200.0)
    high_logp = predict_aqueous_solubility("compound_B", logp=5.0, mw=200.0)
    assert low_logp.solubility_consensus_mg_mL > high_logp.solubility_consensus_mg_mL


def test_high_mp_lower_yalkowsky_solubility():
    """Higher melting point → lower Yalkowsky solubility."""
    low_mp = predict_aqueous_solubility("cpd_low_mp", logp=2.0, mw=300.0, melting_point_c=50.0)
    high_mp = predict_aqueous_solubility("cpd_high_mp", logp=2.0, mw=300.0, melting_point_c=300.0)
    assert low_mp.solubility_yalkowsky_mg_mL > high_mp.solubility_yalkowsky_mg_mL


def test_low_logp_higher_consensus():
    """Low logP compound should have higher consensus solubility."""
    low_logp = predict_aqueous_solubility("hydrophilic", logp=-1.0, mw=200.0)
    high_logp = predict_aqueous_solubility("lipophilic", logp=5.0, mw=200.0)
    assert low_logp.solubility_consensus_mg_mL > high_logp.solubility_consensus_mg_mL


# ---------------------------------------------------------------------------
# BCS dose number
# ---------------------------------------------------------------------------


def test_bcs_dose_number_formula():
    """Dn = dose / (S_consensus * 250) within tolerance."""
    result = predict_aqueous_solubility("test", logp=1.5, mw=250.0, dose_mg=100.0)
    expected_dn = 100.0 / (result.solubility_consensus_mg_mL * 250.0)
    assert abs(result.bcs_dose_number - expected_dn) < 1e-6


def test_high_solubility_bcs_true_when_dn_le_1():
    """high_solubility_bcs should be True when Dn <= 1."""
    # Very hydrophilic compound with small dose → Dn <= 1
    result = predict_aqueous_solubility("hydrophilic", logp=-3.0, mw=100.0, dose_mg=1.0)
    if result.bcs_dose_number <= 1.0:
        assert result.high_solubility_bcs is True
    else:
        assert result.high_solubility_bcs is False


def test_high_solubility_bcs_false_when_dn_gt_1():
    """Poorly soluble compound with large dose → Dn > 1 → high_solubility_bcs = False."""
    result = predict_aqueous_solubility("lipophilic_large_dose", logp=6.0, mw=500.0, dose_mg=1000.0)
    if result.bcs_dose_number > 1.0:
        assert result.high_solubility_bcs is False


# ---------------------------------------------------------------------------
# Log solubility consistency
# ---------------------------------------------------------------------------


def test_log_solubility_consensus_matches_log10():
    """log_solubility_consensus should equal log10(solubility_consensus_mg_mL)."""
    result = predict_aqueous_solubility("aspirin", logp=1.19, mw=180.16)
    expected = math.log10(result.solubility_consensus_mg_mL)
    assert abs(result.log_solubility_consensus - expected) < 1e-9


def test_log_solubility_yalk_matches():
    result = predict_aqueous_solubility("naproxen", logp=3.18, mw=230.26)
    expected = math.log10(result.solubility_yalkowsky_mg_mL)
    assert abs(result.log_solubility_yalk - expected) < 1e-9


def test_log_solubility_esol_matches():
    result = predict_aqueous_solubility("naproxen", logp=3.18, mw=230.26)
    expected = math.log10(result.solubility_esol_mg_mL)
    assert abs(result.log_solubility_esol - expected) < 1e-9


# ---------------------------------------------------------------------------
# screen_solubility
# ---------------------------------------------------------------------------


def test_screen_correct_length():
    compounds = [
        {"drug_name": "A", "logp": 1.0, "mw": 200.0},
        {"drug_name": "B", "logp": 3.0, "mw": 350.0},
        {"drug_name": "C", "logp": 5.0, "mw": 500.0},
    ]
    results = screen_solubility(compounds)
    assert len(results) == 3


def test_screen_sorted_descending():
    """screen_solubility should be sorted by consensus solubility descending."""
    compounds = [
        {"drug_name": "high_logp", "logp": 6.0, "mw": 500.0},
        {"drug_name": "low_logp", "logp": -1.0, "mw": 150.0},
        {"drug_name": "mid_logp", "logp": 2.5, "mw": 300.0},
    ]
    results = screen_solubility(compounds)
    for i in range(len(results) - 1):
        assert results[i].solubility_consensus_mg_mL >= results[i + 1].solubility_consensus_mg_mL


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_aqueous_solubility("drug", logp=1.0, mw=0.0)


def test_mw_negative_raises():
    with pytest.raises(ValueError, match="mw"):
        predict_aqueous_solubility("drug", logp=1.0, mw=-100.0)


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_aqueous_solubility("drug", logp=1.0, mw=200.0, dose_mg=0.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_aqueous_solubility("drug", logp=1.0, mw=200.0, dose_mg=-50.0)


def test_aromatic_proportion_negative_raises():
    with pytest.raises(ValueError, match="aromatic_proportion"):
        predict_aqueous_solubility("drug", logp=1.0, mw=200.0, aromatic_proportion=-0.1)


def test_aromatic_proportion_gt_one_raises():
    with pytest.raises(ValueError, match="aromatic_proportion"):
        predict_aqueous_solubility("drug", logp=1.0, mw=200.0, aromatic_proportion=1.5)


def test_rotatable_bonds_negative_raises():
    with pytest.raises(ValueError, match="rotatable_bonds"):
        predict_aqueous_solubility("drug", logp=1.0, mw=200.0, rotatable_bonds=-1)
