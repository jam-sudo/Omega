"""Tests for first-pass effect calculator (Phase 538)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.first_pass_effect import (
    compare_routes_bioavailability,
    gut_extraction_ratio,
    hepatic_extraction_ratio,
    oral_bioavailability,
    predict_food_effect_on_bioavailability,
)


# ---------------------------------------------------------------------------
# hepatic_extraction_ratio
# ---------------------------------------------------------------------------
class TestHepaticExtractionRatio:
    def test_high_clint_er_near_one(self):
        er = hepatic_extraction_ratio(clint_L_per_h=10000.0, fu_plasma=1.0)
        assert er > 0.99

    def test_zero_clint_er_zero(self):
        er = hepatic_extraction_ratio(clint_L_per_h=0.0, fu_plasma=0.5)
        assert er == pytest.approx(0.0)

    def test_well_stirred_formula(self):
        clint, fu, qh, rbp = 45.0, 0.5, 90.0, 1.0
        fu_clint = fu * clint / rbp
        expected = fu_clint / (qh + fu_clint)
        er = hepatic_extraction_ratio(clint, fu, qh_L_per_h=qh, rbp=rbp, model="well_stirred")
        assert er == pytest.approx(expected, rel=1e-6)

    def test_parallel_tube_formula(self):
        import math

        clint, fu, qh, rbp = 45.0, 0.5, 90.0, 1.0
        fu_clint = fu * clint / rbp
        expected = 1.0 - math.exp(-fu_clint / qh)
        er = hepatic_extraction_ratio(clint, fu, qh_L_per_h=qh, rbp=rbp, model="parallel_tube")
        assert er == pytest.approx(expected, rel=1e-6)

    def test_er_in_zero_one_range(self):
        er = hepatic_extraction_ratio(50.0, 0.3)
        assert 0.0 <= er <= 1.0

    def test_invalid_fu_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            hepatic_extraction_ratio(10.0, 0.0)

    def test_invalid_fu_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            hepatic_extraction_ratio(10.0, 1.5)

    def test_invalid_clint_negative_raises(self):
        with pytest.raises(ValueError, match="clint_L_per_h"):
            hepatic_extraction_ratio(-1.0, 0.5)

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            hepatic_extraction_ratio(10.0, 0.5, model="unknown")

    def test_low_fu_reduces_er(self):
        er_high_fu = hepatic_extraction_ratio(20.0, 1.0)
        er_low_fu = hepatic_extraction_ratio(20.0, 0.01)
        assert er_low_fu < er_high_fu


# ---------------------------------------------------------------------------
# gut_extraction_ratio
# ---------------------------------------------------------------------------
class TestGutExtractionRatio:
    def test_zero_clint_gut_eg_zero(self):
        eg = gut_extraction_ratio(peff_cm_s=1e-4, clint_gut_L_per_h=0.0)
        assert eg == pytest.approx(0.0)

    def test_high_clint_gut_eg_near_one(self):
        eg = gut_extraction_ratio(peff_cm_s=1e-6, clint_gut_L_per_h=1e6)
        assert eg > 0.99

    def test_qgut_auto_calculated_from_peff(self):
        peff = 1e-4  # cm/s
        # Qgut = peff * 200 cm^2 * 3600 s/h / 1000 mL/L = 0.072 L/h
        clint = 0.072  # L/h — equal to auto-calculated qgut → EG = 0.5
        eg = gut_extraction_ratio(peff_cm_s=peff, clint_gut_L_per_h=clint)
        assert eg == pytest.approx(0.5, rel=1e-4)

    def test_explicit_qgut_used(self):
        eg = gut_extraction_ratio(peff_cm_s=0.0, clint_gut_L_per_h=10.0, qgut_L_per_h=10.0)
        assert eg == pytest.approx(0.5, rel=1e-6)

    def test_eg_in_zero_one_range(self):
        eg = gut_extraction_ratio(1e-5, 5.0)
        assert 0.0 <= eg <= 1.0

    def test_invalid_peff_negative_raises(self):
        with pytest.raises(ValueError, match="peff_cm_s"):
            gut_extraction_ratio(-1e-4, 1.0)

    def test_invalid_clint_negative_raises(self):
        with pytest.raises(ValueError, match="clint_gut_L_per_h"):
            gut_extraction_ratio(1e-4, -1.0)


# ---------------------------------------------------------------------------
# oral_bioavailability
# ---------------------------------------------------------------------------
class TestOralBioavailability:
    def test_f_oral_equals_fa_fg_fh(self):
        result = oral_bioavailability(fa=0.9, fg=0.8, fh=0.7)
        assert result["F_oral"] == pytest.approx(0.9 * 0.8 * 0.7, rel=1e-8)

    def test_perfect_absorption_f_oral_one(self):
        result = oral_bioavailability(fa=1.0, fg=1.0, fh=1.0)
        assert result["F_oral"] == pytest.approx(1.0)

    def test_zero_fa_zero_bioavailability(self):
        result = oral_bioavailability(fa=0.0, fg=0.8, fh=0.9)
        assert result["F_oral"] == pytest.approx(0.0)

    def test_loss_fields_correct(self):
        result = oral_bioavailability(fa=0.8, fg=0.7, fh=0.6)
        assert result["loss_gut_lumen"] == pytest.approx(0.2, rel=1e-8)
        assert result["loss_gut_wall"] == pytest.approx(0.3, rel=1e-8)
        assert result["loss_liver"] == pytest.approx(0.4, rel=1e-8)

    def test_notes_returned(self):
        result = oral_bioavailability(0.9, 0.9, 0.9)
        assert "notes" in result
        assert isinstance(result["notes"], str)

    def test_invalid_fa_raises(self):
        with pytest.raises(ValueError, match="fa"):
            oral_bioavailability(fa=1.5, fg=0.8, fh=0.7)

    def test_invalid_fg_raises(self):
        with pytest.raises(ValueError, match="fg"):
            oral_bioavailability(fa=0.9, fg=-0.1, fh=0.7)

    def test_invalid_fh_raises(self):
        with pytest.raises(ValueError, match="fh"):
            oral_bioavailability(fa=0.9, fg=0.8, fh=1.1)


# ---------------------------------------------------------------------------
# predict_food_effect_on_bioavailability
# ---------------------------------------------------------------------------
class TestPredictFoodEffect:
    def test_high_logP_f_fed_higher(self):
        result = predict_food_effect_on_bioavailability(f_fasted=0.3, logP=5.0)
        assert result["F_fed"] > result["F_fasted"]

    def test_low_logP_no_enhancement(self):
        result = predict_food_effect_on_bioavailability(f_fasted=0.8, logP=0.0)
        assert result["food_effect_ratio"] == pytest.approx(1.0)

    def test_direction_increase_for_high_logP(self):
        result = predict_food_effect_on_bioavailability(f_fasted=0.2, logP=6.0)
        assert result["direction"] == "increase"

    def test_direction_no_effect_for_low_logP(self):
        result = predict_food_effect_on_bioavailability(f_fasted=0.9, logP=1.0)
        assert result["direction"] == "no_effect"

    def test_f_fed_capped_at_one(self):
        result = predict_food_effect_on_bioavailability(f_fasted=0.9, logP=10.0)
        assert result["F_fed"] <= 1.0

    def test_recommendation_returned(self):
        result = predict_food_effect_on_bioavailability(f_fasted=0.5, logP=4.0)
        assert "recommendation" in result
        assert isinstance(result["recommendation"], str)

    def test_invalid_f_fasted_raises(self):
        with pytest.raises(ValueError, match="f_fasted"):
            predict_food_effect_on_bioavailability(f_fasted=1.5, logP=3.0)

    def test_bile_acid_disabled(self):
        result_on = predict_food_effect_on_bioavailability(0.3, 4.0, bile_acid_enhancement=True)
        result_off = predict_food_effect_on_bioavailability(0.3, 4.0, bile_acid_enhancement=False)
        assert result_on["F_fed"] >= result_off["F_fed"]


# ---------------------------------------------------------------------------
# compare_routes_bioavailability
# ---------------------------------------------------------------------------
class TestCompareRoutesBioavailability:
    def test_equal_dose_equal_auc_f_100pct(self):
        result = compare_routes_bioavailability(
            iv_auc=100.0, oral_auc=100.0, dose_iv=100.0, dose_oral=100.0
        )
        assert result["F_oral_pct"] == pytest.approx(100.0, rel=1e-6)

    def test_oral_auc_half_f_50pct(self):
        result = compare_routes_bioavailability(
            iv_auc=100.0, oral_auc=50.0, dose_iv=100.0, dose_oral=100.0
        )
        assert result["F_oral_pct"] == pytest.approx(50.0, rel=1e-6)

    def test_dose_normalization_applied(self):
        result = compare_routes_bioavailability(
            iv_auc=100.0, oral_auc=200.0, dose_iv=100.0, dose_oral=200.0
        )
        assert result["F_oral_pct"] == pytest.approx(100.0, rel=1e-6)

    def test_result_fields_present(self):
        result = compare_routes_bioavailability(100.0, 80.0, 100.0, 100.0)
        for key in ("F_oral_pct", "oral_auc", "iv_auc", "dose_oral", "dose_iv"):
            assert key in result

    def test_invalid_dose_iv_zero_raises(self):
        with pytest.raises(ValueError, match="dose_iv"):
            compare_routes_bioavailability(100.0, 50.0, dose_iv=0.0, dose_oral=100.0)

    def test_invalid_dose_oral_zero_raises(self):
        with pytest.raises(ValueError, match="dose_oral"):
            compare_routes_bioavailability(100.0, 50.0, dose_iv=100.0, dose_oral=0.0)

    def test_negative_iv_auc_raises(self):
        with pytest.raises(ValueError, match="iv_auc"):
            compare_routes_bioavailability(-10.0, 50.0, 100.0, 100.0)

    def test_negative_oral_auc_raises(self):
        with pytest.raises(ValueError, match="oral_auc"):
            compare_routes_bioavailability(100.0, -50.0, 100.0, 100.0)
