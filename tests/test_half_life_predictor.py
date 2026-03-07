"""Tests for half-life predictor (Phase 378)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.half_life_predictor import (
    HalfLifeResult,
    classify_half_life,
    predict_half_life,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default(**kw):
    defaults = dict(
        mw=300.0,
        logP=2.0,
        psa=60.0,
        n_hbd=2,
        fu_plasma=0.2,
        route_of_elimination="hepatic",
    )
    defaults.update(kw)
    return predict_half_life(**defaults)


# ---------------------------------------------------------------------------
# classify_half_life boundary tests
# ---------------------------------------------------------------------------


class TestClassifyHalfLife:
    def test_ultra_short(self):
        assert classify_half_life(0.5) == "ultra-short"

    def test_boundary_ultra_short_to_short(self):
        assert classify_half_life(1.0) == "short"

    def test_short(self):
        assert classify_half_life(3.0) == "short"

    def test_boundary_short_to_medium(self):
        assert classify_half_life(6.0) == "medium"

    def test_medium(self):
        assert classify_half_life(12.0) == "medium"

    def test_boundary_medium_to_long(self):
        assert classify_half_life(24.0) == "long"

    def test_long(self):
        assert classify_half_life(48.0) == "long"

    def test_boundary_long_to_very_long(self):
        assert classify_half_life(72.0) == "very-long"

    def test_very_long(self):
        assert classify_half_life(100.0) == "very-long"

    def test_near_zero_ultra_short(self):
        assert classify_half_life(0.01) == "ultra-short"

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="t_half_h"):
            classify_half_life(-1.0)


# ---------------------------------------------------------------------------
# Return type and basic properties
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_half_life_result(self):
        assert isinstance(_default(), HalfLifeResult)

    def test_t_half_positive(self):
        r = _default()
        assert r.t_half_h > 0.0

    def test_cl_per_kg_positive(self):
        r = _default()
        assert r.cl_L_per_h_per_kg > 0.0

    def test_vd_per_kg_positive(self):
        r = _default()
        assert r.vd_L_per_kg > 0.0

    def test_vd_L_is_70kg_scaled(self):
        r = _default()
        assert abs(r.vd_L - r.vd_L_per_kg * 70.0) < 1e-6

    def test_cl_L_per_h_is_70kg_scaled(self):
        r = _default()
        assert abs(r.cl_L_per_h - r.cl_L_per_h_per_kg * 70.0) < 1e-6

    def test_half_life_formula(self):
        r = _default()
        expected_t_half = 0.693147 * r.vd_L_per_kg / r.cl_L_per_h_per_kg
        assert abs(r.t_half_h - expected_t_half) < 1e-6

    def test_half_life_class_matches_classify(self):
        r = _default()
        assert r.half_life_class == classify_half_life(r.t_half_h)

    def test_route_preserved(self):
        r = _default(route_of_elimination="renal")
        assert r.route_of_elimination == "renal"


# ---------------------------------------------------------------------------
# Physicochemical property effects
# ---------------------------------------------------------------------------


class TestPhysicochemicalEffects:
    def test_higher_logP_higher_vd(self):
        r_low = _default(logP=0.0)
        r_high = _default(logP=5.0)
        assert r_high.vd_L_per_kg > r_low.vd_L_per_kg

    def test_higher_logP_lower_hepatic_cl(self):
        # hepatic CL = 15 * fup * exp(-0.3 * logP)
        r_low = _default(logP=0.0, route_of_elimination="hepatic")
        r_high = _default(logP=5.0, route_of_elimination="hepatic")
        assert r_high.cl_L_per_h_per_kg < r_low.cl_L_per_h_per_kg

    def test_higher_logP_longer_t_half_hepatic(self):
        r_low = _default(logP=0.0, route_of_elimination="hepatic")
        r_high = _default(logP=4.0, route_of_elimination="hepatic")
        assert r_high.t_half_h > r_low.t_half_h

    def test_higher_fu_higher_cl_hepatic(self):
        r_low = _default(fu_plasma=0.05)
        r_high = _default(fu_plasma=0.80)
        assert r_high.cl_L_per_h_per_kg > r_low.cl_L_per_h_per_kg

    def test_higher_mw_higher_vd(self):
        r_low = _default(mw=100.0)
        r_high = _default(mw=600.0)
        assert r_high.vd_L_per_kg > r_low.vd_L_per_kg

    def test_higher_psa_lower_vd(self):
        r_low = _default(psa=20.0)
        r_high = _default(psa=180.0)
        assert r_high.vd_L_per_kg < r_low.vd_L_per_kg

    def test_vd_clamped_minimum(self):
        # Very high psa and n_hbd with low logP/mw → Vd clamped to 0.3
        r = _default(mw=50.0, logP=-5.0, psa=300.0, n_hbd=20, fu_plasma=0.01)
        assert r.vd_L_per_kg >= 0.3

    def test_vd_clamped_maximum(self):
        # Very high logP and mw → Vd clamped to 200
        r = _default(mw=2000.0, logP=20.0, psa=0.0, n_hbd=0, fu_plasma=0.99)
        assert r.vd_L_per_kg <= 200.0


# ---------------------------------------------------------------------------
# Route of elimination
# ---------------------------------------------------------------------------


class TestRouteOfElimination:
    def test_hepatic_route(self):
        r = _default(route_of_elimination="hepatic")
        assert r.t_half_h > 0.0

    def test_renal_route(self):
        r = _default(route_of_elimination="renal")
        assert r.t_half_h > 0.0

    def test_mixed_route(self):
        r = _default(route_of_elimination="mixed")
        assert r.t_half_h > 0.0

    def test_mixed_cl_between_hepatic_and_renal(self):
        r_h = _default(route_of_elimination="hepatic")
        r_r = _default(route_of_elimination="renal")
        r_m = _default(route_of_elimination="mixed")
        cl_min = min(r_h.cl_L_per_h_per_kg, r_r.cl_L_per_h_per_kg)
        cl_max = max(r_h.cl_L_per_h_per_kg, r_r.cl_L_per_h_per_kg)
        assert cl_min <= r_m.cl_L_per_h_per_kg <= cl_max + 1e-9


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_logP_in_range_high_confidence(self):
        r = _default(logP=2.0)
        assert r.confidence == "high"

    def test_logP_boundary_high_confidence_lower(self):
        r = _default(logP=-1.0)
        assert r.confidence == "high"

    def test_logP_boundary_high_confidence_upper(self):
        r = _default(logP=5.0)
        assert r.confidence == "high"

    def test_logP_outside_range_moderate_confidence(self):
        r = _default(logP=6.5)
        assert r.confidence == "moderate"

    def test_logP_very_extreme_low_confidence(self):
        r = _default(logP=10.0)
        assert r.confidence == "low"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _default(mw=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _default(mw=-50.0)

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa"):
            _default(psa=-5.0)

    def test_negative_n_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            _default(n_hbd=-1)

    def test_zero_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _default(fu_plasma=0.0)

    def test_fu_plasma_above_1_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _default(fu_plasma=1.5)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route_of_elimination"):
            _default(route_of_elimination="biliary")

    def test_negative_t_half_classify_raises(self):
        with pytest.raises(ValueError):
            classify_half_life(-0.1)
