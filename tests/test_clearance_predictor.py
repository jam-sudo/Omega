"""Tests for hepatic clearance predictor (Phase 390)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.clearance_predictor import (
    ClearancePredictionResult,
    predict_hepatic_cl,
    rank_clearance_compounds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(**kw) -> ClearancePredictionResult:
    defaults = dict(
        clint_uL_per_min_per_mg_protein=10.0,
        fu_mic=0.5,
        hepatocyte_scaling=1.0,
        method="microsomal",
        body_weight_kg=70.0,
    )
    defaults.update(kw)
    return predict_hepatic_cl(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_clearance_prediction_result(self):
        assert isinstance(_pred(), ClearancePredictionResult)

    def test_method_stored(self):
        r = _pred(method="microsomal")
        assert r.method == "microsomal"

    def test_fu_mic_stored(self):
        r = _pred(fu_mic=0.3)
        assert abs(r.fu_mic - 0.3) < 1e-9

    def test_body_weight_stored(self):
        r = _pred(body_weight_kg=60.0)
        assert abs(r.body_weight_kg - 60.0) < 1e-9

    def test_cl_hepatic_positive(self):
        r = _pred()
        assert r.cl_hepatic_L_per_h > 0.0

    def test_extraction_ratio_between_0_and_1(self):
        r = _pred()
        assert 0.0 < r.extraction_ratio < 1.0

    def test_qh_positive(self):
        r = _pred()
        assert r.qh_mL_per_min > 0.0

    def test_clint_invivo_positive(self):
        r = _pred()
        assert r.clint_invivo_mL_per_min > 0.0

    def test_notes_non_empty(self):
        r = _pred()
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Well-stirred model physics
# ---------------------------------------------------------------------------


class TestPhysics:
    def test_higher_clint_gives_higher_cl(self):
        r_high = _pred(clint_uL_per_min_per_mg_protein=100.0)
        r_low = _pred(clint_uL_per_min_per_mg_protein=1.0)
        assert r_high.cl_hepatic_L_per_h > r_low.cl_hepatic_L_per_h

    def test_higher_fu_mic_gives_higher_cl(self):
        r_high = _pred(fu_mic=0.9)
        r_low = _pred(fu_mic=0.1)
        assert r_high.cl_hepatic_L_per_h > r_low.cl_hepatic_L_per_h

    def test_cl_cannot_exceed_qh(self):
        # Even with very high CLint, CL <= QH
        r = _pred(clint_uL_per_min_per_mg_protein=1e6, fu_mic=1.0)
        cl_mL_min = r.cl_hepatic_L_per_h * 1000.0 / 60.0
        assert cl_mL_min <= r.qh_mL_per_min + 1e-6

    def test_extraction_ratio_approaches_1_high_clint(self):
        r = _pred(clint_uL_per_min_per_mg_protein=1e6, fu_mic=1.0)
        assert r.extraction_ratio > 0.9

    def test_qh_allometric_scaling(self):
        r_heavy = _pred(body_weight_kg=140.0)
        r_light = _pred(body_weight_kg=35.0)
        assert r_heavy.qh_mL_per_min > r_light.qh_mL_per_min

    def test_microsomal_vs_hepatocyte_same_order_of_magnitude(self):
        r_micro = _pred(method="microsomal", clint_uL_per_min_per_mg_protein=10.0)
        r_hepa = _pred(method="hepatocyte", clint_uL_per_min_per_mg_protein=10.0)
        # Both should be positive; ratio should be within 3 orders of magnitude
        ratio = r_micro.cl_hepatic_L_per_h / r_hepa.cl_hepatic_L_per_h
        assert 1e-3 < ratio < 1e3


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_high_extraction(self):
        r = _pred(clint_uL_per_min_per_mg_protein=1e5, fu_mic=1.0)
        assert r.cl_classification == "high"

    def test_low_extraction(self):
        r = _pred(clint_uL_per_min_per_mg_protein=0.01, fu_mic=0.1)
        assert r.cl_classification == "low"

    def test_classification_in_valid_set(self):
        r = _pred()
        assert r.cl_classification in {"high", "intermediate", "low"}

    def test_extraction_ratio_matches_classification_high(self):
        r = _pred(clint_uL_per_min_per_mg_protein=1e5, fu_mic=1.0)
        assert r.extraction_ratio > 0.7

    def test_extraction_ratio_matches_classification_low(self):
        r = _pred(clint_uL_per_min_per_mg_protein=0.01, fu_mic=0.1)
        assert r.extraction_ratio < 0.3


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_clint_raises(self):
        with pytest.raises(ValueError):
            _pred(clint_uL_per_min_per_mg_protein=0.0)

    def test_negative_clint_raises(self):
        with pytest.raises(ValueError):
            _pred(clint_uL_per_min_per_mg_protein=-1.0)

    def test_zero_fu_mic_raises(self):
        with pytest.raises(ValueError):
            _pred(fu_mic=0.0)

    def test_fu_mic_greater_than_1_raises(self):
        with pytest.raises(ValueError):
            _pred(fu_mic=1.5)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            _pred(method="unknown_method")

    def test_zero_body_weight_raises(self):
        with pytest.raises(ValueError):
            _pred(body_weight_kg=0.0)

    def test_negative_body_weight_raises(self):
        with pytest.raises(ValueError):
            _pred(body_weight_kg=-70.0)

    def test_zero_hepatocyte_scaling_raises(self):
        with pytest.raises(ValueError):
            _pred(hepatocyte_scaling=0.0, method="hepatocyte")


# ---------------------------------------------------------------------------
# rank_clearance_compounds
# ---------------------------------------------------------------------------


class TestRankClearanceCompounds:
    _compounds = [
        {"name": "DrugA", "clint": 5.0, "fu_mic": 0.3},
        {"name": "DrugB", "clint": 50.0, "fu_mic": 0.8},
        {"name": "DrugC", "clint": 0.5, "fu_mic": 0.1},
    ]

    def test_returns_list(self):
        result = rank_clearance_compounds(self._compounds)
        assert isinstance(result, list)

    def test_length_preserved(self):
        result = rank_clearance_compounds(self._compounds)
        assert len(result) == 3

    def test_sorted_descending_by_cl(self):
        result = rank_clearance_compounds(self._compounds)
        cls = [r["cl_hepatic_L_per_h"] for r in result]
        assert cls == sorted(cls, reverse=True)

    def test_all_have_extraction_ratio(self):
        result = rank_clearance_compounds(self._compounds)
        assert all("extraction_ratio" in r for r in result)

    def test_all_have_cl_classification(self):
        result = rank_clearance_compounds(self._compounds)
        assert all(r["cl_classification"] in {"high", "intermediate", "low"} for r in result)

    def test_empty_compounds_raises(self):
        with pytest.raises(ValueError):
            rank_clearance_compounds([])

    def test_missing_fu_mic_raises(self):
        with pytest.raises((ValueError, KeyError, TypeError)):
            rank_clearance_compounds([{"name": "X", "clint": 10.0}])

    def test_missing_clint_raises(self):
        with pytest.raises((ValueError, KeyError, TypeError)):
            rank_clearance_compounds([{"name": "X", "fu_mic": 0.5}])
