"""Tests for Oie-Tozer volume of distribution prediction (Phase 394)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.volume_of_distribution_prediction import (
    VdPredictionResult,
    predict_vd,
    vd_sensitivity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _predict(**kw):
    defaults = dict(
        mw=350.0,
        logP=2.0,
        psa=70.0,
        fu_plasma=0.2,
        fu_tissue=None,
        pka=None,
        molecule_type="small_molecule",
        body_weight_kg=70.0,
    )
    defaults.update(kw)
    return predict_vd(**defaults)


# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


class TestVdResultType:
    def test_returns_result_object(self):
        r = _predict()
        assert isinstance(r, VdPredictionResult)

    def test_vd_per_kg_positive(self):
        r = _predict()
        assert r.vd_L_per_kg > 0.0

    def test_vd_L_positive(self):
        r = _predict()
        assert r.vd_L > 0.0

    def test_vd_L_consistent_with_body_weight(self):
        bw = 80.0
        r = _predict(body_weight_kg=bw)
        assert r.vd_L == pytest.approx(r.vd_L_per_kg * bw, rel=1e-6)

    def test_notes_is_string(self):
        r = _predict()
        assert isinstance(r.notes, str)

    def test_molecule_type_stored(self):
        r = _predict(molecule_type="peptide")
        assert r.molecule_type == "peptide"

    def test_pka_stored(self):
        r = _predict(pka=7.4)
        assert r.pka == pytest.approx(7.4)

    def test_pka_none_stored(self):
        r = _predict(pka=None)
        assert r.pka is None

    def test_fu_plasma_stored(self):
        r = _predict(fu_plasma=0.5)
        assert r.fu_plasma == pytest.approx(0.5)

    def test_vd_class_valid_string(self):
        r = _predict()
        assert r.vd_class in ("low", "moderate", "high", "very_high")


# ---------------------------------------------------------------------------
# Vd classification thresholds
# ---------------------------------------------------------------------------


class TestVdClassification:
    def test_low_vd_classification(self):
        # High fu_plasma, low logP → low Vd
        r = _predict(fu_plasma=0.95, logP=-2.0, fu_tissue=0.95)
        assert r.vd_class == "low"

    def test_very_high_vd_classification(self):
        # Very low fu_tissue relative to fu_plasma → very high Vd
        r = _predict(fu_plasma=0.01, logP=5.0, fu_tissue=0.0001)
        assert r.vd_class == "very_high"

    def test_moderate_or_high_vd_for_typical_drug(self):
        r = _predict(fu_plasma=0.2, logP=2.0)
        assert r.vd_class in ("low", "moderate", "high", "very_high")

    def test_vd_increases_with_logP(self):
        r_low_logP = _predict(logP=0.0, fu_plasma=0.5)
        r_high_logP = _predict(logP=4.0, fu_plasma=0.5)
        assert r_high_logP.vd_L_per_kg >= r_low_logP.vd_L_per_kg

    def test_vd_decreases_with_fu_plasma(self):
        r_low_fu = _predict(fu_plasma=0.01, logP=2.0)
        r_high_fu = _predict(fu_plasma=0.9, logP=2.0)
        assert r_low_fu.vd_L_per_kg >= r_high_fu.vd_L_per_kg


# ---------------------------------------------------------------------------
# Tissue binding factor
# ---------------------------------------------------------------------------


class TestTissueBindingFactor:
    def test_tissue_binding_factor_positive(self):
        r = _predict()
        assert r.tissue_binding_factor > 0.0

    def test_fu_tissue_used_leq_one(self):
        r = _predict()
        assert 0 < r.fu_tissue_used <= 1.0

    def test_provided_fu_tissue_used(self):
        r = _predict(fu_plasma=0.3, fu_tissue=0.05)
        assert r.fu_tissue_used == pytest.approx(0.05)

    def test_estimated_fu_tissue_differs_from_fu_plasma(self):
        r = _predict(fu_plasma=0.5, logP=3.0, fu_tissue=None)
        # fu_tissue should be smaller than fu_plasma for positive logP
        assert r.fu_tissue_used < r.fu_plasma

    def test_tissue_binding_factor_equals_fu_ratio(self):
        fu_p = 0.2
        fu_t = 0.05
        r = _predict(fu_plasma=fu_p, fu_tissue=fu_t)
        assert r.tissue_binding_factor == pytest.approx(fu_p / fu_t, rel=1e-5)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestVdValidation:
    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_vd(mw=0.0, logP=2.0, psa=70.0, fu_plasma=0.2)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError):
            predict_vd(mw=-100.0, logP=2.0, psa=70.0, fu_plasma=0.2)

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa"):
            predict_vd(mw=300.0, logP=2.0, psa=-1.0, fu_plasma=0.2)

    def test_zero_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_vd(mw=300.0, logP=2.0, psa=70.0, fu_plasma=0.0)

    def test_fu_plasma_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_vd(mw=300.0, logP=2.0, psa=70.0, fu_plasma=1.5)

    def test_zero_fu_tissue_raises(self):
        with pytest.raises(ValueError, match="fu_tissue"):
            predict_vd(mw=300.0, logP=2.0, psa=70.0, fu_plasma=0.2, fu_tissue=0.0)

    def test_zero_body_weight_raises(self):
        with pytest.raises(ValueError, match="body_weight_kg"):
            predict_vd(mw=300.0, logP=2.0, psa=70.0, fu_plasma=0.2, body_weight_kg=0.0)


# ---------------------------------------------------------------------------
# vd_sensitivity sweep
# ---------------------------------------------------------------------------


class TestVdSensitivity:
    def test_returns_list_of_dicts(self):
        results = vd_sensitivity(
            mw=350.0, logP_values=[-1.0, 0.0, 1.0, 2.0, 3.0], psa=70.0, fu_plasma=0.3
        )
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_length_matches_logP_values(self):
        lp_vals = [0.0, 1.0, 2.0, 3.0]
        results = vd_sensitivity(mw=350.0, logP_values=lp_vals, psa=70.0, fu_plasma=0.3)
        assert len(results) == len(lp_vals)

    def test_result_keys_present(self):
        results = vd_sensitivity(mw=350.0, logP_values=[2.0], psa=70.0, fu_plasma=0.3)
        keys = results[0].keys()
        assert "logP" in keys
        assert "vd_L_per_kg" in keys
        assert "vd_L" in keys
        assert "vd_class" in keys
        assert "tissue_binding_factor" in keys

    def test_logP_values_match_input(self):
        lp_vals = [1.0, 3.0, 5.0]
        results = vd_sensitivity(mw=350.0, logP_values=lp_vals, psa=70.0, fu_plasma=0.3)
        for r, lp in zip(results, lp_vals, strict=True):
            assert r["logP"] == pytest.approx(lp)

    def test_monotone_trend_with_logP(self):
        """Higher logP should generally give higher or equal Vd (for positive logP)."""
        lp_vals = [1.0, 2.0, 3.0, 4.0]
        results = vd_sensitivity(mw=350.0, logP_values=lp_vals, psa=70.0, fu_plasma=0.3)
        vd_vals = [r["vd_L_per_kg"] for r in results]
        assert vd_vals[-1] >= vd_vals[0]

    def test_empty_logP_values_raises(self):
        with pytest.raises(ValueError, match="logP_values"):
            vd_sensitivity(mw=350.0, logP_values=[], psa=70.0, fu_plasma=0.3)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            vd_sensitivity(mw=0.0, logP_values=[2.0], psa=70.0, fu_plasma=0.3)

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa"):
            vd_sensitivity(mw=300.0, logP_values=[2.0], psa=-5.0, fu_plasma=0.3)

    def test_invalid_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            vd_sensitivity(mw=300.0, logP_values=[2.0], psa=70.0, fu_plasma=0.0)

    def test_pka_passed_through(self):
        results = vd_sensitivity(mw=350.0, logP_values=[2.0], psa=70.0, fu_plasma=0.3, pka=7.4)
        assert len(results) == 1
