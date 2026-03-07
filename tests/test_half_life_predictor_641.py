"""Tests for Phase 641 QSPR-based half-life predictor."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.half_life_predictor import (
    HalfLifePrediction,
    predict_half_life_from_props,
    screen_half_lives,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_BCS_CLASSES = {"I", "II", "III", "IV"}
_VD_CLASSES = {"low", "moderate", "high"}
_CL_CLASSES = {"low", "moderate", "high"}


def _default(**kw) -> HalfLifePrediction:
    defaults = dict(
        compound_name="TestDrug",
        logP=2.0,
        mw=300.0,
        psa=60.0,
        clint_uL_min_per_mg=10.0,
        solubility_mg_mL=1.0,
    )
    defaults.update(kw)
    return predict_half_life_from_props(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic properties
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_half_life_prediction(self):
        result = _default()
        assert isinstance(result, HalfLifePrediction)

    def test_predicted_t_half_positive(self):
        result = _default()
        assert result.predicted_t_half_h > 0.0

    def test_ci_bounds_surround_prediction(self):
        result = _default()
        assert result.ci_lower_h < result.predicted_t_half_h < result.ci_upper_h

    def test_predicted_ke_per_h_matches_formula(self):
        result = _default()
        expected_ke = math.log(2.0) / result.predicted_t_half_h
        assert abs(result.predicted_ke_per_h - expected_ke) < 1e-9

    def test_bcs_class_valid(self):
        result = _default()
        assert result.bcs_class in _BCS_CLASSES

    def test_vd_class_valid(self):
        result = _default()
        assert result.vd_class in _VD_CLASSES

    def test_cl_class_valid(self):
        result = _default()
        assert result.cl_class in _CL_CLASSES

    def test_elimination_half_lives_in_24h_nonneg(self):
        result = _default()
        assert result.elimination_half_lives_in_24h >= 0.0

    def test_notes_nonempty(self):
        result = _default()
        assert result.notes != ""

    def test_compound_name_preserved(self):
        result = _default(compound_name="Aspirin")
        assert result.compound_name == "Aspirin"


# ---------------------------------------------------------------------------
# Physicochemical property effects
# ---------------------------------------------------------------------------


class TestPropertyEffects:
    def test_high_logp_longer_t_half(self):
        """Lipophilic drugs accumulate → longer t½."""
        low = _default(logP=0.0)
        high = _default(logP=5.0)
        assert high.predicted_t_half_h > low.predicted_t_half_h

    def test_high_clint_shorter_t_half(self):
        """Faster metabolism → shorter t½."""
        low_cl = _default(clint_uL_min_per_mg=1.0)
        high_cl = _default(clint_uL_min_per_mg=200.0)
        assert high_cl.predicted_t_half_h < low_cl.predicted_t_half_h

    def test_acidic_drug_longer_t_half(self):
        """Acidic drug (pKa_acid=3) → ionization_factor=1.3 → longer t½."""
        neutral = _default()
        acidic = _default(pka_acid=3.0)
        assert acidic.predicted_t_half_h > neutral.predicted_t_half_h

    def test_basic_drug_shorter_t_half(self):
        """Basic drug (pKa_base=9) → ionization_factor=0.9 → shorter t½."""
        neutral = _default()
        basic = _default(pka_base=9.0)
        assert basic.predicted_t_half_h < neutral.predicted_t_half_h


# ---------------------------------------------------------------------------
# BCS classification
# ---------------------------------------------------------------------------


class TestBCSClassification:
    def test_bcs_i_narrow_ci_vs_bcs_iv(self):
        """BCS I (high perm, high sol) → narrower CI than BCS IV."""
        bcs_i = predict_half_life_from_props(
            compound_name="BCSI",
            logP=1.5,
            mw=300.0,
            psa=60.0,
            solubility_mg_mL=5.0,
        )
        bcs_iv = predict_half_life_from_props(
            compound_name="BCSIV",
            logP=-2.0,
            mw=600.0,
            psa=180.0,
            solubility_mg_mL=0.01,
        )
        ci_i = bcs_i.ci_upper_h - bcs_i.ci_lower_h
        ci_iv = bcs_iv.ci_upper_h - bcs_iv.ci_lower_h
        # CI should be narrower for BCS I (20% vs 60% of predicted)
        assert ci_i / bcs_i.predicted_t_half_h < ci_iv / bcs_iv.predicted_t_half_h


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


class TestScreenHalfLives:
    def test_returns_sorted_ascending(self):
        compounds = [
            {
                "compound_name": "A",
                "logP": 5.0,
                "mw": 400.0,
                "psa": 50.0,
                "clint_uL_min_per_mg": 1.0,
            },
            {
                "compound_name": "B",
                "logP": 1.0,
                "mw": 250.0,
                "psa": 80.0,
                "clint_uL_min_per_mg": 100.0,
            },
            {
                "compound_name": "C",
                "logP": 3.0,
                "mw": 300.0,
                "psa": 60.0,
                "clint_uL_min_per_mg": 20.0,
            },
        ]
        results = screen_half_lives(compounds)
        t_halfs = [r.predicted_t_half_h for r in results]
        assert t_halfs == sorted(t_halfs)

    def test_empty_list_returns_empty(self):
        results = screen_half_lives([])
        assert results == []

    def test_returns_list_of_predictions(self):
        compounds = [
            {"compound_name": "X", "logP": 2.0, "mw": 300.0, "psa": 70.0},
        ]
        results = screen_half_lives(compounds)
        assert len(results) == 1
        assert isinstance(results[0], HalfLifePrediction)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _default(mw=0.0)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _default(mw=-100.0)

    def test_clint_negative_raises(self):
        with pytest.raises(ValueError, match="clint"):
            _default(clint_uL_min_per_mg=-1.0)

    def test_body_weight_zero_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            _default(body_weight_kg=0.0)

    def test_psa_negative_raises(self):
        with pytest.raises(ValueError, match="psa"):
            _default(psa=-5.0)
