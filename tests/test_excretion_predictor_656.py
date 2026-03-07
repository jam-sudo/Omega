"""Tests for Phase 656 additions to prediction/excretion_predictor.py."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.excretion_predictor import (
    ExcretionPrediction,
    predict_excretion,
    screen_excretion,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_prediction(
    compound_name: str = "TestDrug",
    logP: float = 1.0,
    mw: float = 300.0,
    psa: float = 60.0,
    fup: float = 0.5,
    clint_uL_per_min_per_mg: float = 10.0,
    n_hbd: int = 2,
    n_hba: int = 4,
    **kwargs,
) -> ExcretionPrediction:
    return predict_excretion(
        compound_name=compound_name,
        logP=logP,
        mw=mw,
        psa=psa,
        fup=fup,
        clint_uL_per_min_per_mg=clint_uL_per_min_per_mg,
        n_hbd=n_hbd,
        n_hba=n_hba,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test: return type
# ---------------------------------------------------------------------------


def test_returns_excretion_prediction():
    result = make_prediction()
    assert isinstance(result, ExcretionPrediction)


# ---------------------------------------------------------------------------
# Test: fraction ranges
# ---------------------------------------------------------------------------


def test_fe_urine_in_0_1():
    result = make_prediction()
    assert 0.0 <= result.fe_urine_pred <= 1.0


def test_fe_feces_in_0_1():
    result = make_prediction()
    assert 0.0 <= result.fe_feces_pred <= 1.0


def test_fe_metabolism_in_0_1():
    result = make_prediction()
    assert 0.0 <= result.fe_metabolism_pred <= 1.0


def test_total_recovery_approx_1():
    result = make_prediction()
    assert result.total_recovery == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Test: dominant route
# ---------------------------------------------------------------------------


def test_dominant_route_in_valid_set():
    result = make_prediction()
    assert result.dominant_route in {"renal", "metabolic", "biliary", "mixed"}


# ---------------------------------------------------------------------------
# Test: molecular weight class
# ---------------------------------------------------------------------------


def test_mw_class_small_for_mw_200():
    result = make_prediction(mw=200.0)
    assert result.molecular_weight_class == "small"


def test_mw_class_medium_for_mw_400():
    result = make_prediction(mw=400.0)
    assert result.molecular_weight_class == "medium"


def test_mw_class_large_for_mw_600():
    result = make_prediction(mw=600.0)
    assert result.molecular_weight_class == "large"


# ---------------------------------------------------------------------------
# Test: Lipinski compliance
# ---------------------------------------------------------------------------


def test_lipinski_compliant_for_small_drug():
    result = make_prediction(mw=200.0, logP=2.0, n_hbd=2, n_hba=4)
    assert result.lipinski_compliant is True


def test_lipinski_non_compliant_for_large_drug():
    result = make_prediction(mw=600.0, logP=6.0, n_hbd=2, n_hba=4)
    assert result.lipinski_compliant is False


# ---------------------------------------------------------------------------
# Test: biliary excretion flag
# ---------------------------------------------------------------------------


def test_biliary_flag_true_for_mw500_logP3():
    result = make_prediction(mw=500.0, logP=3.0)
    assert result.biliary_excretion_flag is True


def test_biliary_flag_false_for_mw200_logP1():
    result = make_prediction(mw=200.0, logP=1.0)
    assert result.biliary_excretion_flag is False


# ---------------------------------------------------------------------------
# Test: physicochemical effects on excretion
# ---------------------------------------------------------------------------


def test_fe_urine_positive_for_hydrophilic_drug():
    # logP=-1, high fup → high renal excretion
    result = make_prediction(logP=-1.0, fup=0.9, clint_uL_per_min_per_mg=1.0)
    assert result.fe_urine_pred > 0


def test_fe_metabolism_positive_for_lipophilic_drug():
    # logP=5, high CLint → high metabolic elimination
    result = make_prediction(logP=5.0, clint_uL_per_min_per_mg=50.0, fup=0.1)
    assert result.fe_metabolism_pred > 0


def test_high_clint_gives_higher_fe_metabolism():
    low_clint = make_prediction(clint_uL_per_min_per_mg=1.0, logP=2.0, fup=0.3)
    high_clint = make_prediction(clint_uL_per_min_per_mg=60.0, logP=2.0, fup=0.3)
    assert high_clint.fe_metabolism_pred > low_clint.fe_metabolism_pred


def test_renal_cl_positive_for_renally_eliminated_drug():
    result = make_prediction(logP=-1.0, fup=0.8, clint_uL_per_min_per_mg=1.0)
    assert result.renal_cl_pred_mL_per_min > 0


# ---------------------------------------------------------------------------
# Test: notes
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = make_prediction()
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Test: screen_excretion
# ---------------------------------------------------------------------------


def test_screen_excretion_returns_sorted_by_fe_urine_desc():
    compounds = [
        {
            "compound_name": "HydroA",
            "logP": -1.0,
            "mw": 200.0,
            "fup": 0.9,
            "clint_uL_per_min_per_mg": 1.0,
            "psa": 80.0,
        },
        {
            "compound_name": "LipoB",
            "logP": 4.0,
            "mw": 350.0,
            "fup": 0.1,
            "clint_uL_per_min_per_mg": 40.0,
            "psa": 30.0,
        },
        {
            "compound_name": "MidC",
            "logP": 1.5,
            "mw": 280.0,
            "fup": 0.5,
            "clint_uL_per_min_per_mg": 10.0,
            "psa": 60.0,
        },
    ]
    results = screen_excretion(compounds)
    assert len(results) == 3
    fe_urines = [r.fe_urine_pred for r in results]
    assert fe_urines == sorted(fe_urines, reverse=True)


def test_screen_excretion_empty_returns_empty():
    results = screen_excretion([])
    assert results == []


# ---------------------------------------------------------------------------
# Test: validation errors
# ---------------------------------------------------------------------------


def test_raises_error_fup_zero():
    with pytest.raises(ValueError):
        predict_excretion("X", logP=1.0, mw=300.0, psa=60.0, fup=0.0)


def test_raises_error_mw_zero():
    with pytest.raises(ValueError):
        predict_excretion("X", logP=1.0, mw=0.0, psa=60.0, fup=0.5)


def test_raises_error_gfr_zero():
    with pytest.raises(ValueError):
        predict_excretion("X", logP=1.0, mw=300.0, psa=60.0, fup=0.5, gfr_mL_per_min=0.0)


def test_raises_error_n_hbd_negative():
    with pytest.raises(ValueError):
        predict_excretion("X", logP=1.0, mw=300.0, psa=60.0, fup=0.5, n_hbd=-1)
