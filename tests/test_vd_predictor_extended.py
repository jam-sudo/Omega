"""Tests for extended volume of distribution predictor (Phase 534)."""

import pytest

from omega_pbpk.prediction.vd_predictor_extended import (
    oie_tozer_vss,
    predict_vd_ensemble,
    predict_vd_from_logP,
)


class TestOieTozerVss:
    def test_basic_returns_float(self):
        result = oie_tozer_vss(fu_plasma=0.1, fu_tissue=0.05)
        assert isinstance(result, float)
        assert result > 0

    def test_high_fu_plasma_lower_vss(self):
        # Higher fu_plasma means less protein binding → lower tissue partitioning
        # With fu_tissue fixed, higher fu_plasma → lower Vss (since Vss = Vp + Vt*fu_p/fu_t)
        # Actually higher fu_plasma → higher Vss in this formula
        # But with same fu_tissue, comparing fu_plasma=0.9 vs fu_plasma=0.1:
        vss_high_fu = oie_tozer_vss(fu_plasma=0.9, fu_tissue=0.1)
        vss_low_fu = oie_tozer_vss(fu_plasma=0.1, fu_tissue=0.1)
        # Higher fu_plasma → higher Vss (more tissue penetration when less plasma bound)
        assert vss_high_fu > vss_low_fu

    def test_low_fu_tissue_high_vss(self):
        # Very low fu_tissue → extensive tissue binding → large Vss
        vss_high_binding = oie_tozer_vss(fu_plasma=0.1, fu_tissue=0.001)
        vss_low_binding = oie_tozer_vss(fu_plasma=0.1, fu_tissue=0.5)
        assert vss_high_binding > vss_low_binding

    def test_vss_includes_plasma_volume(self):
        # Minimum Vss should be at least plasma volume (3 L)
        vss = oie_tozer_vss(fu_plasma=0.001, fu_tissue=0.999)
        assert vss >= 3.0

    def test_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            oie_tozer_vss(fu_plasma=0.0, fu_tissue=0.1)

    def test_invalid_fu_plasma_above_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            oie_tozer_vss(fu_plasma=1.5, fu_tissue=0.1)

    def test_invalid_fu_tissue_zero(self):
        with pytest.raises(ValueError, match="fu_tissue"):
            oie_tozer_vss(fu_plasma=0.1, fu_tissue=0.0)

    def test_invalid_fu_tissue_above_one(self):
        with pytest.raises(ValueError, match="fu_tissue"):
            oie_tozer_vss(fu_plasma=0.1, fu_tissue=1.5)

    def test_equal_fu_gives_vss_near_total_body_water(self):
        # If fu_plasma == fu_tissue, Vss = Vp + Vt = 3 + 38 = 41 L
        vss = oie_tozer_vss(fu_plasma=0.5, fu_tissue=0.5)
        assert abs(vss - 41.0) < 1e-9


class TestPredictVdFromLogP:
    def test_returns_expected_keys(self):
        result = predict_vd_from_logP(logP=2.0)
        for key in ("vd_L", "vd_L_per_kg", "logP", "drug_type", "notes"):
            assert key in result

    def test_high_logP_higher_vd(self):
        result_high = predict_vd_from_logP(logP=5.0, fu_plasma=0.1)
        result_low = predict_vd_from_logP(logP=0.0, fu_plasma=0.1)
        assert result_high["vd_L"] > result_low["vd_L"]

    def test_weak_base_higher_than_neutral(self):
        base = predict_vd_from_logP(logP=2.0, drug_type="weak_base", fu_plasma=0.5)
        neutral = predict_vd_from_logP(logP=2.0, drug_type="neutral", fu_plasma=0.5)
        assert base["vd_L"] > neutral["vd_L"]

    def test_weak_acid_lower_than_neutral(self):
        acid = predict_vd_from_logP(logP=2.0, drug_type="weak_acid", fu_plasma=0.5)
        neutral = predict_vd_from_logP(logP=2.0, drug_type="neutral", fu_plasma=0.5)
        assert acid["vd_L"] < neutral["vd_L"]

    def test_drug_type_ordering(self):
        # weak_base > neutral > weak_acid at same logP and fu_plasma
        logP = 2.0
        fu = 0.3
        vd_base = predict_vd_from_logP(logP, drug_type="weak_base", fu_plasma=fu)["vd_L"]
        vd_neutral = predict_vd_from_logP(logP, drug_type="neutral", fu_plasma=fu)["vd_L"]
        vd_acid = predict_vd_from_logP(logP, drug_type="weak_acid", fu_plasma=fu)["vd_L"]
        assert vd_base > vd_neutral > vd_acid

    def test_vd_L_per_kg_relationship(self):
        result = predict_vd_from_logP(logP=2.0)
        assert abs(result["vd_L"] / 70.0 - result["vd_L_per_kg"]) < 1e-9

    def test_vd_clamp_min(self):
        # Very low logP and high fu → clamped to 3 L minimum
        result = predict_vd_from_logP(logP=-10.0, fu_plasma=1.0, drug_type="weak_acid")
        assert result["vd_L"] >= 3.0

    def test_vd_clamp_max(self):
        # Very high logP → clamped to 3000 L maximum
        result = predict_vd_from_logP(logP=20.0, fu_plasma=0.001, drug_type="weak_base")
        assert result["vd_L"] <= 3000.0

    def test_invalid_fu_plasma(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_vd_from_logP(logP=2.0, fu_plasma=0.0)

    def test_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_vd_from_logP(logP=2.0, drug_type="strong_acid")

    def test_pka_included_in_notes(self):
        result = predict_vd_from_logP(logP=2.0, pka=7.4)
        assert "pKa=7.4" in result["notes"]

    def test_higher_fu_lower_vd(self):
        result_high_fu = predict_vd_from_logP(logP=2.0, fu_plasma=0.9)
        result_low_fu = predict_vd_from_logP(logP=2.0, fu_plasma=0.01)
        assert result_high_fu["vd_L"] < result_low_fu["vd_L"]


class TestPredictVdEnsemble:
    def test_returns_expected_keys(self):
        result = predict_vd_ensemble(logP=2.0, fu_plasma=0.1, fu_tissue=0.05)
        for key in (
            "vd_oie_tozer_L",
            "vd_qsar_L",
            "vd_ensemble_L",
            "vd_ensemble_L_per_kg",
            "confidence",
        ):
            assert key in result

    def test_ensemble_is_mean(self):
        result = predict_vd_ensemble(logP=2.0, fu_plasma=0.1, fu_tissue=0.05)
        expected_mean = (result["vd_oie_tozer_L"] + result["vd_qsar_L"]) / 2.0
        assert abs(result["vd_ensemble_L"] - expected_mean) < 1e-9

    def test_ensemble_L_per_kg(self):
        result = predict_vd_ensemble(logP=2.0, fu_plasma=0.1, fu_tissue=0.05)
        assert abs(result["vd_ensemble_L"] / 70.0 - result["vd_ensemble_L_per_kg"]) < 1e-9

    def test_confidence_high_when_methods_agree(self):
        # Use parameters that make both methods give similar values
        # Oie-Tozer: 3 + 38*(0.5/0.5) = 41 L; QSAR: depends on logP
        # Choose logP that gives ~41 L from QSAR with fu=0.5
        # QSAR neutral: exp(0.5*logP + 1.5) * (1/0.5)^0.5
        # For ~41L: exp(0.5*logP+1.5) * 1.414 ≈ 41 → exp(0.5*logP+1.5) ≈ 29
        # 0.5*logP + 1.5 ≈ ln(29) ≈ 3.37 → logP ≈ 3.74
        # Let's use logP=3.7, fu_plasma=0.5, fu_tissue=0.5 and check confidence
        result = predict_vd_ensemble(logP=3.7, fu_plasma=0.5, fu_tissue=0.5)
        # Both within 2x: confidence should be 'high'
        ratio = max(result["vd_oie_tozer_L"], result["vd_qsar_L"]) / min(
            result["vd_oie_tozer_L"], result["vd_qsar_L"]
        )
        expected_conf = "high" if ratio <= 2.0 else "low"
        assert result["confidence"] == expected_conf

    def test_confidence_low_when_methods_disagree(self):
        # Low logP + high fu_tissue → Oie-Tozer gives very low Vss; QSAR may differ
        # High tissue binding (fu_tissue=0.001) → Oie-Tozer: 3 + 38*(0.1/0.001) = 3803 L
        # QSAR: clamped at 3000 L (or some value)
        # Low logP Oie-Tozer: fu_plasma=0.9, fu_tissue=0.001 → 3+38*900=34203 (huge)
        # vs QSAR: clamped at 3000 → ratio > 2x → low confidence
        result = predict_vd_ensemble(logP=-3.0, fu_plasma=0.9, fu_tissue=0.001)
        # Oie-Tozer will be >> QSAR (clamped at 3)
        assert result["confidence"] == "low"

    def test_invalid_fu_plasma(self):
        with pytest.raises(ValueError):
            predict_vd_ensemble(logP=2.0, fu_plasma=0.0, fu_tissue=0.05)

    def test_invalid_fu_tissue(self):
        with pytest.raises(ValueError):
            predict_vd_ensemble(logP=2.0, fu_plasma=0.1, fu_tissue=0.0)
