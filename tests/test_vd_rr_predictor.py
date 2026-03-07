"""Tests for omega_pbpk.prediction.vd_rr_predictor module."""

import pytest

from omega_pbpk.prediction.vd_rr_predictor import (
    VdPrediction,
    predict_vd,
    screen_vd,
)

VALID_VD_CLASSES = {"low", "moderate", "high", "very_high"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(**kwargs):
    defaults = dict(
        compound_name="TestDrug",
        logP=2.0,
        mw=350.0,
        psa=60.0,
        fup=0.1,
    )
    defaults.update(kwargs)
    return predict_vd(**defaults)


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


def test_returns_vd_prediction():
    result = _base()
    assert isinstance(result, VdPrediction)


def test_predicted_vd_L_gt_0():
    result = _base()
    assert result.predicted_vd_L > 0


def test_predicted_vd_L_per_kg_gt_0():
    result = _base()
    assert result.predicted_vd_L_per_kg > 0


def test_vd_class_is_valid():
    result = _base()
    assert result.vd_class in VALID_VD_CLASSES


def test_kp_muscle_gt_0():
    result = _base()
    assert result.kp_muscle > 0


def test_kp_adipose_gt_0():
    result = _base()
    assert result.kp_adipose > 0


def test_kp_liver_gt_0():
    result = _base()
    assert result.kp_liver > 0


def test_kp_brain_ge_0():
    result = _base()
    assert result.kp_brain >= 0


def test_predicted_vss_L_gt_0():
    result = _base()
    assert result.predicted_vss_L > 0


def test_notes_nonempty():
    result = _base()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Physicochemical relationships (QSPR method)
# ---------------------------------------------------------------------------


def test_high_logP_gives_higher_vd_than_low_logP():
    low = predict_vd("LowLogP", logP=-2.0, mw=300.0, psa=60.0, fup=0.1)
    high = predict_vd("HighLogP", logP=5.0, mw=300.0, psa=60.0, fup=0.1)
    assert high.predicted_vd_L_per_kg > low.predicted_vd_L_per_kg


def test_basic_drug_gives_higher_vd():
    neutral = predict_vd("Neutral", logP=2.0, mw=300.0, psa=60.0, fup=0.1)
    basic = predict_vd("Basic", logP=2.0, mw=300.0, psa=60.0, fup=0.1, pka_base=9.0)
    assert basic.predicted_vd_L_per_kg > neutral.predicted_vd_L_per_kg


def test_acidic_drug_gives_lower_vd():
    neutral = predict_vd("Neutral", logP=2.0, mw=300.0, psa=60.0, fup=0.3)
    acidic = predict_vd("Acidic", logP=2.0, mw=300.0, psa=60.0, fup=0.3, pka_acid=3.0)
    assert acidic.predicted_vd_L_per_kg < neutral.predicted_vd_L_per_kg


def test_low_fup_gives_higher_vd():
    low_fup = predict_vd("LowFup", logP=2.0, mw=300.0, psa=60.0, fup=0.01)
    high_fup = predict_vd("HighFup", logP=2.0, mw=300.0, psa=60.0, fup=0.9)
    assert low_fup.predicted_vd_L_per_kg > high_fup.predicted_vd_L_per_kg


def test_high_psa_gives_lower_vd():
    low_psa = predict_vd("LowPSA", logP=2.0, mw=300.0, psa=30.0, fup=0.1)
    high_psa = predict_vd("HighPSA", logP=2.0, mw=300.0, psa=150.0, fup=0.1)
    assert high_psa.predicted_vd_L_per_kg < low_psa.predicted_vd_L_per_kg


# ---------------------------------------------------------------------------
# vd_class thresholds
# ---------------------------------------------------------------------------


def test_vd_class_low_for_hydrophilic():
    # logP=-2 → base_vd ≈ 0.7, psa=150 penalty large, fup=1.0
    result = predict_vd("HydroPhilic", logP=-2.0, mw=200.0, psa=150.0, fup=1.0)
    assert result.vd_class == "low"


def test_vd_class_high_for_lipophilic_basic():
    # logP=5, pKa=9, fup=0.01 → very high Vd
    result = predict_vd("Lipophilic", logP=5.0, mw=350.0, psa=40.0, fup=0.01, pka_base=9.0)
    assert result.vd_class in {"high", "very_high"}


# ---------------------------------------------------------------------------
# Rodgers-Rowland method
# ---------------------------------------------------------------------------


def test_rodgers_rowland_returns_positive_vd():
    result = predict_vd("RR_Drug", logP=2.0, mw=300.0, psa=60.0, fup=0.1, method="rodgers_rowland")
    assert result.predicted_vd_L > 0


def test_rodgers_rowland_basic_drug_positive_vd():
    result = predict_vd(
        "BasicRR", logP=3.0, mw=300.0, psa=50.0, fup=0.05, pka_base=9.0, method="rodgers_rowland"
    )
    assert result.predicted_vd_L > 0


# ---------------------------------------------------------------------------
# screen_vd
# ---------------------------------------------------------------------------


def test_screen_returns_list():
    compounds = [
        dict(compound_name="A", logP=2.0, mw=300.0, psa=60.0, fup=0.1),
        dict(compound_name="B", logP=4.0, mw=350.0, psa=40.0, fup=0.05),
    ]
    results = screen_vd(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_vd_descending():
    compounds = [
        dict(compound_name="Low", logP=-2.0, mw=200.0, psa=150.0, fup=1.0),
        dict(compound_name="High", logP=5.0, mw=350.0, psa=30.0, fup=0.01),
        dict(compound_name="Mid", logP=2.0, mw=300.0, psa=60.0, fup=0.1),
    ]
    results = screen_vd(compounds)
    vds = [r.predicted_vd_L_per_kg for r in results]
    assert vds == sorted(vds, reverse=True)


def test_screen_empty_returns_empty():
    assert screen_vd([]) == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_raises_when_fup_zero():
    with pytest.raises(ValueError):
        predict_vd("X", logP=2.0, mw=300.0, psa=60.0, fup=0.0)


def test_raises_when_fup_gt_1():
    with pytest.raises(ValueError):
        predict_vd("X", logP=2.0, mw=300.0, psa=60.0, fup=1.2)


def test_raises_when_body_weight_zero():
    with pytest.raises(ValueError):
        predict_vd("X", logP=2.0, mw=300.0, psa=60.0, fup=0.1, body_weight_kg=0.0)


def test_raises_when_method_invalid():
    with pytest.raises(ValueError):
        predict_vd("X", logP=2.0, mw=300.0, psa=60.0, fup=0.1, method="compartmental")
