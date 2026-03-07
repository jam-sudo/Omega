"""Tests for Phase 654 — total_clearance_predictor module."""

import pytest

from omega_pbpk.prediction.total_clearance_predictor import (
    TotalClearancePrediction,
    predict_total_clearance,
    screen_clearance,
)


def default_pred(**kwargs) -> TotalClearancePrediction:
    """Return a prediction with default parameters, overriding with kwargs."""
    params = dict(
        compound_name="TestCompound",
        logP=2.0,
        mw=350.0,
        psa=60.0,
        fup=0.1,
        clint_uL_per_min_per_mg=20.0,
    )
    params.update(kwargs)
    return predict_total_clearance(**params)


# ── basic structure ──────────────────────────────────────────────────────────


def test_returns_total_clearance_prediction():
    pred = default_pred()
    assert isinstance(pred, TotalClearancePrediction)


def test_cl_total_positive():
    pred = default_pred()
    assert pred.cl_total_L_per_h > 0


def test_cl_hepatic_nonnegative():
    pred = default_pred()
    assert pred.cl_hepatic_L_per_h >= 0


def test_cl_renal_nonnegative():
    pred = default_pred()
    assert pred.cl_renal_L_per_h >= 0


def test_cl_biliary_nonnegative():
    pred = default_pred()
    assert pred.cl_biliary_L_per_h >= 0


# ── additive clearance ───────────────────────────────────────────────────────


def test_cl_total_equals_sum():
    pred = default_pred()
    expected = pred.cl_hepatic_L_per_h + pred.cl_renal_L_per_h + pred.cl_biliary_L_per_h
    assert abs(pred.cl_total_L_per_h - expected) < 1e-6


# ── extraction ratios ────────────────────────────────────────────────────────


def test_hepatic_er_in_range():
    pred = default_pred()
    assert 0 <= pred.hepatic_er <= 1


def test_renal_er_in_range():
    pred = default_pred()
    assert 0 <= pred.renal_er <= 1


# ── fractions ────────────────────────────────────────────────────────────────


def test_fractions_sum_to_one():
    pred = default_pred()
    total = pred.fraction_hepatic + pred.fraction_renal + pred.fraction_biliary
    assert abs(total - 1.0) < 1e-6


# ── high clearance flag ──────────────────────────────────────────────────────


def test_high_clearance_flag_true_high_clint():
    """High CLint should yield ER > 0.7 → high_clearance_hepatic True."""
    pred = predict_total_clearance(
        "HighCL",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        fup=1.0,
        clint_uL_per_min_per_mg=2000.0,
    )
    assert pred.high_clearance_hepatic is True


def test_high_clearance_flag_false_low_clint():
    """Very low CLint should give ER < 0.7."""
    pred = predict_total_clearance(
        "LowCL",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        fup=0.01,
        clint_uL_per_min_per_mg=0.1,
    )
    assert pred.high_clearance_hepatic is False


# ── route of elimination ─────────────────────────────────────────────────────


def test_route_of_elimination_valid():
    pred = default_pred()
    valid = {"hepatic", "renal", "biliary", "mixed"}
    assert pred.route_of_elimination in valid


# ── t half ───────────────────────────────────────────────────────────────────


def test_t_half_positive():
    pred = default_pred()
    assert pred.t_half_predicted_h > 0


# ── physicochemical effects ───────────────────────────────────────────────────


def test_high_clint_higher_cl_hepatic():
    """Higher CLint → higher hepatic clearance."""
    pred_low = default_pred(clint_uL_per_min_per_mg=1.0)
    pred_high = default_pred(clint_uL_per_min_per_mg=500.0)
    assert pred_high.cl_hepatic_L_per_h > pred_low.cl_hepatic_L_per_h


def test_high_fup_higher_cl_hepatic():
    """Higher fup → higher hepatic extraction."""
    pred_low = default_pred(fup=0.01)
    pred_high = default_pred(fup=1.0)
    assert pred_high.cl_hepatic_L_per_h > pred_low.cl_hepatic_L_per_h


def test_high_mw_high_logp_nonzero_biliary():
    """MW>400 and logP>2 → non-zero biliary clearance."""
    pred = predict_total_clearance(
        "BigLipophilic",
        logP=4.0,
        mw=550.0,
        psa=80.0,
        fup=0.1,
        clint_uL_per_min_per_mg=10.0,
    )
    assert pred.cl_biliary_L_per_h > 0


def test_zero_clint_negligible_hepatic():
    """Zero CLint → effectively zero hepatic CL."""
    pred = predict_total_clearance(
        "ZeroCLint",
        logP=1.0,
        mw=250.0,
        psa=50.0,
        fup=0.5,
        clint_uL_per_min_per_mg=0.0,
    )
    assert pred.cl_hepatic_L_per_h == pytest.approx(0.0, abs=1e-6)


def test_zero_fe_urine_no_secretion_low_renal():
    """No renal excretion parameters → low renal CL."""
    pred = predict_total_clearance(
        "LowRenal",
        logP=3.0,
        mw=300.0,
        psa=50.0,
        fup=0.01,
        clint_uL_per_min_per_mg=5.0,
        fe_urine=0.0,
        active_secretion_cl_mL_per_min=0.0,
    )
    assert pred.cl_renal_L_per_h < 1.0


# ── notes ────────────────────────────────────────────────────────────────────


def test_notes_nonempty():
    pred = default_pred()
    assert pred.notes and len(pred.notes) > 0


# ── screen_clearance ─────────────────────────────────────────────────────────


def test_screen_clearance_returns_list():
    compounds = [
        {
            "compound_name": "A",
            "logP": 1.0,
            "mw": 200.0,
            "psa": 40.0,
            "fup": 0.1,
            "clint_uL_per_min_per_mg": 5.0,
        },
        {
            "compound_name": "B",
            "logP": 3.0,
            "mw": 500.0,
            "psa": 80.0,
            "fup": 0.5,
            "clint_uL_per_min_per_mg": 100.0,
        },
    ]
    results = screen_clearance(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_clearance_sorted_descending():
    compounds = [
        {
            "compound_name": "LowCL",
            "logP": 1.0,
            "mw": 200.0,
            "psa": 40.0,
            "fup": 0.01,
            "clint_uL_per_min_per_mg": 1.0,
        },
        {
            "compound_name": "HighCL",
            "logP": 3.0,
            "mw": 350.0,
            "psa": 60.0,
            "fup": 0.8,
            "clint_uL_per_min_per_mg": 500.0,
        },
    ]
    results = screen_clearance(compounds)
    assert results[0].cl_total_L_per_h >= results[1].cl_total_L_per_h


def test_screen_clearance_empty():
    results = screen_clearance([])
    assert results == []


# ── validation ───────────────────────────────────────────────────────────────


def test_validation_fup_zero():
    with pytest.raises(ValueError, match="fup"):
        predict_total_clearance(
            "X", logP=1.0, mw=300.0, psa=60.0, fup=0.0, clint_uL_per_min_per_mg=10.0
        )


def test_validation_fup_too_high():
    with pytest.raises(ValueError, match="fup"):
        predict_total_clearance(
            "X", logP=1.0, mw=300.0, psa=60.0, fup=1.2, clint_uL_per_min_per_mg=10.0
        )


def test_validation_clint_negative():
    with pytest.raises(ValueError, match="clint"):
        predict_total_clearance(
            "X", logP=1.0, mw=300.0, psa=60.0, fup=0.1, clint_uL_per_min_per_mg=-1.0
        )


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_total_clearance(
            "X", logP=1.0, mw=0.0, psa=60.0, fup=0.1, clint_uL_per_min_per_mg=10.0
        )
