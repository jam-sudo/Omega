"""Tests for hepatic microsomal stability prediction (Phase 651)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.microsomal_stability_p651 import (
    MicrosomalStabilityResult,
    predict_microsomal_stability,
    screen_microsomal_stability,
)

# ── Basic return type ────────────────────────────────────────────────


def test_return_type():
    r = predict_microsomal_stability("DrugA", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=20.0)
    assert isinstance(r, MicrosomalStabilityResult)


def test_frozen_dataclass():
    r = predict_microsomal_stability("DrugA", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=20.0)
    with pytest.raises(AttributeError):
        r.drug_name = "other"


def test_drug_name_preserved():
    r = predict_microsomal_stability("TestDrug", logP=1.0, mw_Da=250.0, t_half_in_vitro_min=40.0)
    assert r.drug_name == "TestDrug"


# ── CLint relationship with t_half ──────────────────────────────────


def test_higher_thalf_lower_clint():
    r_short = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=5.0)
    r_long = predict_microsomal_stability("B", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=60.0)
    assert r_short.clint_mL_per_min_per_mg > r_long.clint_mL_per_min_per_mg


def test_higher_thalf_lower_extraction():
    r_short = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=5.0)
    r_long = predict_microsomal_stability("B", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=60.0)
    assert r_short.hepatic_extraction_ratio > r_long.hepatic_extraction_ratio


def test_higher_thalf_higher_f_hepatic():
    r_short = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=5.0)
    r_long = predict_microsomal_stability("B", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=60.0)
    assert r_long.predicted_f_hepatic > r_short.predicted_f_hepatic


# ── fu_mic ───────────────────────────────────────────────────────────


def test_low_logP_fumic_one():
    r = predict_microsomal_stability("A", logP=1.0, mw_Da=300.0, t_half_in_vitro_min=20.0)
    assert r.fu_mic == 1.0


def test_high_logP_fumic_less_than_one():
    r = predict_microsomal_stability("A", logP=4.0, mw_Da=300.0, t_half_in_vitro_min=20.0)
    assert r.fu_mic < 1.0


def test_fumic_in_range():
    for lp in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        r = predict_microsomal_stability("A", logP=lp, mw_Da=300.0, t_half_in_vitro_min=20.0)
        assert 0.01 <= r.fu_mic <= 1.0


def test_lower_logP_fumic_closer_to_one():
    r_low = predict_microsomal_stability("A", logP=2.5, mw_Da=300.0, t_half_in_vitro_min=20.0)
    r_high = predict_microsomal_stability("B", logP=5.0, mw_Da=300.0, t_half_in_vitro_min=20.0)
    assert r_low.fu_mic > r_high.fu_mic


# ── Hepatic extraction ratio and predicted_f_hepatic ─────────────────


def test_extraction_ratio_range():
    for thalf in [2.0, 10.0, 30.0, 100.0]:
        r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=thalf)
        assert 0.0 <= r.hepatic_extraction_ratio <= 1.0


def test_predicted_f_hepatic_range():
    for thalf in [2.0, 10.0, 30.0, 100.0]:
        r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=thalf)
        assert 0.0 <= r.predicted_f_hepatic <= 1.0


def test_f_hepatic_plus_extraction_equals_one():
    r = predict_microsomal_stability("A", logP=3.0, mw_Da=400.0, t_half_in_vitro_min=15.0)
    assert abs(r.predicted_f_hepatic + r.hepatic_extraction_ratio - 1.0) < 1e-10


# ── Stability class ─────────────────────────────────────────────────


def test_stability_class_high():
    r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=30.0)
    assert r.stability_class == "high"


def test_stability_class_medium():
    r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=20.0)
    assert r.stability_class == "medium"


def test_stability_class_low():
    r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=10.0)
    assert r.stability_class == "low"


def test_stability_class_boundary_30():
    r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=30.0)
    assert r.stability_class == "high"


def test_stability_class_boundary_15():
    r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=15.0)
    assert r.stability_class == "medium"


# ── CLint scaled positive ───────────────────────────────────────────


def test_clint_scaled_positive():
    r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=20.0)
    assert r.clint_scaled_L_per_h > 0


# ── Hepatic CL <= hepatic blood flow ────────────────────────────────


def test_hepatic_cl_bounded():
    r = predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=1.0)
    assert r.hepatic_cl_L_per_h <= 90.0


def test_hepatic_cl_custom_flow():
    r = predict_microsomal_stability(
        "A",
        logP=2.0,
        mw_Da=300.0,
        t_half_in_vitro_min=1.0,
        hepatic_blood_flow_L_per_h=50.0,
    )
    assert r.hepatic_cl_L_per_h <= 50.0


# ── Screen function ─────────────────────────────────────────────────


def test_screen_returns_list():
    compounds = [
        {"drug_name": "A", "logP": 2.0, "mw_Da": 300.0, "t_half_in_vitro_min": 10.0},
        {"drug_name": "B", "logP": 2.0, "mw_Da": 300.0, "t_half_in_vitro_min": 40.0},
    ]
    results = screen_microsomal_stability(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_sorted_by_thalf_descending():
    compounds = [
        {"drug_name": "A", "logP": 2.0, "mw_Da": 300.0, "t_half_in_vitro_min": 5.0},
        {"drug_name": "B", "logP": 2.0, "mw_Da": 300.0, "t_half_in_vitro_min": 50.0},
        {"drug_name": "C", "logP": 2.0, "mw_Da": 300.0, "t_half_in_vitro_min": 20.0},
    ]
    results = screen_microsomal_stability(compounds)
    assert results[0].drug_name == "B"
    assert results[1].drug_name == "C"
    assert results[2].drug_name == "A"


# ── Validation errors ───────────────────────────────────────────────


def test_validation_thalf_zero():
    with pytest.raises(ValueError):
        predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=0.0)


def test_validation_thalf_negative():
    with pytest.raises(ValueError):
        predict_microsomal_stability("A", logP=2.0, mw_Da=300.0, t_half_in_vitro_min=-5.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError):
        predict_microsomal_stability("A", logP=2.0, mw_Da=0.0, t_half_in_vitro_min=20.0)


def test_validation_protein_zero():
    with pytest.raises(ValueError):
        predict_microsomal_stability(
            "A",
            logP=2.0,
            mw_Da=300.0,
            t_half_in_vitro_min=20.0,
            microsomal_protein_mg_mL=0.0,
        )


def test_notes_contains_info():
    r = predict_microsomal_stability("A", logP=3.0, mw_Da=300.0, t_half_in_vitro_min=20.0)
    assert "Well-stirred" in r.notes
