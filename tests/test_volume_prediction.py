"""Tests for volume of distribution prediction (Phase 116)."""

from __future__ import annotations

from omega_pbpk.prediction.volume_prediction import (
    VolumePredictionResult,
    predict_vd_tissue,
    screen_vd,
)


class TestPredictVdTissue:
    def test_returns_result_type(self):
        r = predict_vd_tissue("Drug", logP=2.0, fup=0.5)
        assert isinstance(r, VolumePredictionResult)

    def test_vd_positive(self):
        r = predict_vd_tissue("Drug", logP=2.0, fup=0.5)
        assert r.vd_L > 0.0

    def test_vd_per_kg_positive(self):
        r = predict_vd_tissue("Drug", logP=2.0, fup=0.5)
        assert r.vd_per_kg_L > 0.0

    def test_high_logP_increases_vd(self):
        r_low = predict_vd_tissue("Drug", logP=0.0, fup=0.5)
        r_high = predict_vd_tissue("Drug", logP=5.0, fup=0.5)
        assert r_high.vd_L > r_low.vd_L

    def test_low_fup_decreases_vd(self):
        # At low fup, Vd = Vp + Σ(Vt * Kp * fup) → decreases
        r_high_fup = predict_vd_tissue("Drug", logP=2.0, fup=0.9)
        r_low_fup = predict_vd_tissue("Drug", logP=2.0, fup=0.1)
        assert r_low_fup.vd_L < r_high_fup.vd_L

    def test_classification_valid(self):
        r = predict_vd_tissue("Drug", logP=2.0, fup=0.5)
        assert r.classification in ("low", "moderate", "high")

    def test_kp_tissue_has_all_tissues(self):
        r = predict_vd_tissue("Drug", logP=2.0, fup=0.5)
        expected = {"adipose", "muscle", "liver", "brain", "kidney"}
        assert expected.issubset(r.kp_tissue.keys())

    def test_all_kp_positive(self):
        r = predict_vd_tissue("Drug", logP=2.0, fup=0.5)
        assert all(v > 0 for v in r.kp_tissue.values())

    def test_basic_drug_higher_vd(self):
        r_neutral = predict_vd_tissue("Drug", logP=2.0, fup=0.5, drug_type="neutral")
        r_basic = predict_vd_tissue("Drug", logP=2.0, fup=0.5, drug_type="basic", pka_basic=9.0)
        assert r_basic.vd_L > r_neutral.vd_L

    def test_method_is_string(self):
        r = predict_vd_tissue("Drug", logP=2.0, fup=0.5)
        assert isinstance(r.method, str)


class TestScreenVd:
    def test_returns_list(self):
        compounds = [
            {"drug_name": "A", "logP": 2.0, "fup": 0.5},
            {"drug_name": "B", "logP": 4.0, "fup": 0.2},
        ]
        results = screen_vd(compounds)
        assert len(results) == 2

    def test_empty_returns_empty(self):
        assert screen_vd([]) == []
