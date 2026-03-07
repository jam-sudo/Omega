"""Tests for hepatic first-pass extraction module (Phase 891)."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.clinical.hepatic_first_pass import (
    HepaticFirstPassResult,
    predict_hepatic_first_pass,
    screen_hepatic_fp,
)


class TestPredictHepaticFirstPass:
    """Tests for predict_hepatic_first_pass function."""

    def test_returns_dataclass_instance(self):
        result = predict_hepatic_first_pass("Drug_A")
        assert isinstance(result, HepaticFirstPassResult)

    def test_drug_name_stored(self):
        result = predict_hepatic_first_pass("Lidocaine")
        assert result.drug_name == "Lidocaine"

    def test_dose_mg_stored(self):
        result = predict_hepatic_first_pass("X", dose_mg=100.0)
        assert result.dose_mg == 100.0

    def test_fa_fg_stored(self):
        result = predict_hepatic_first_pass("X", fa=0.8, fg=0.7)
        assert result.fa == pytest.approx(0.8)
        assert result.fg == pytest.approx(0.7)

    def test_e_hepatic_well_stirred_formula(self):
        # E_H = CLint / (Q + CLint)
        cl_int = 45.0
        q = 90.0
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=cl_int, q_hepatic_L_per_h=q)
        expected_e = cl_int / (q + cl_int)
        assert result.e_hepatic == pytest.approx(expected_e, rel=1e-6)

    def test_fh_is_one_minus_e_hepatic(self):
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=45.0)
        assert result.fh == pytest.approx(1.0 - result.e_hepatic, rel=1e-9)

    def test_cl_hepatic_is_q_times_e(self):
        q = 90.0
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=30.0, q_hepatic_L_per_h=q)
        assert result.cl_hepatic_L_per_h == pytest.approx(q * result.e_hepatic, rel=1e-9)

    def test_bioavailability_formula(self):
        result = predict_hepatic_first_pass("X", fa=0.9, fg=0.85, cl_intrinsic_L_per_h=10.0)
        expected_f = result.fa * result.fg * result.fh
        assert result.f_bioavailability == pytest.approx(expected_f, rel=1e-9)

    def test_high_bioavailability_class(self):
        # Low CLint → high F
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=1.0, fa=0.95, fg=0.95)
        assert result.f_bioavailability > 0.6
        assert result.bioavailability_class == "high"

    def test_intermediate_bioavailability_class(self):
        # Tune CLint to land F between 0.3 and 0.6
        # E_H = 45/135 = 0.333; fh = 0.667; F = 0.95*0.9*0.667 ≈ 0.57
        result = predict_hepatic_first_pass(
            "X", cl_intrinsic_L_per_h=45.0, fa=0.95, fg=0.9, q_hepatic_L_per_h=90.0
        )
        assert 0.3 < result.f_bioavailability <= 0.6
        assert result.bioavailability_class == "intermediate"

    def test_low_bioavailability_class(self):
        # Very high CLint → low F
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=900.0, fa=0.95, fg=0.9)
        assert result.f_bioavailability <= 0.3
        assert result.bioavailability_class == "low"

    def test_first_pass_loss_fraction(self):
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=30.0)
        assert result.first_pass_loss_fraction == pytest.approx(1.0 - result.fh, rel=1e-9)

    def test_hepatic_blood_flow_stored(self):
        q = 75.0
        result = predict_hepatic_first_pass("X", q_hepatic_L_per_h=q)
        assert result.hepatic_blood_flow_L_per_h == pytest.approx(q)

    def test_high_extraction_notes(self):
        # E_H > 0.7 when CLint >> Q
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=300.0, q_hepatic_L_per_h=90.0)
        assert result.e_hepatic > 0.7
        assert "High hepatic extraction" in result.notes
        assert "IV vs oral" in result.notes

    def test_moderate_extraction_notes(self):
        # E_H between 0.3 and 0.7
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=45.0, q_hepatic_L_per_h=90.0)
        assert 0.3 < result.e_hepatic <= 0.7
        assert "Moderate hepatic extraction" in result.notes

    def test_low_extraction_notes(self):
        # E_H < 0.3 when CLint is small
        result = predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=5.0, q_hepatic_L_per_h=90.0)
        assert result.e_hepatic < 0.3
        assert "Low hepatic extraction" in result.notes

    def test_bioavailability_clamped_to_zero_one(self):
        # Even with extreme parameters, F must be in [0, 1]
        result = predict_hepatic_first_pass("X", fa=1.0, fg=1.0, cl_intrinsic_L_per_h=0.001)
        assert 0.0 <= result.f_bioavailability <= 1.0

    def test_validation_fa_out_of_range(self):
        with pytest.raises(ValueError, match="fa"):
            predict_hepatic_first_pass("X", fa=1.5)

    def test_validation_fg_out_of_range(self):
        with pytest.raises(ValueError, match="fg"):
            predict_hepatic_first_pass("X", fg=-0.1)

    def test_validation_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_hepatic_first_pass("X", fu_plasma=0.0)

    def test_validation_cl_intrinsic_non_positive(self):
        with pytest.raises(ValueError, match="cl_intrinsic"):
            predict_hepatic_first_pass("X", cl_intrinsic_L_per_h=0.0)

    def test_frozen_dataclass(self):
        result = predict_hepatic_first_pass("X")
        with pytest.raises(Exception):
            result.drug_name = "Y"  # type: ignore[misc]


class TestScreenHepaticFp:
    """Tests for screen_hepatic_fp function."""

    def test_returns_list(self):
        candidates = [
            {"name": "A", "cl_intrinsic_L_per_h": 5.0},
            {"name": "B", "cl_intrinsic_L_per_h": 50.0},
        ]
        results = screen_hepatic_fp(candidates)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_by_bioavailability_descending(self):
        candidates = [
            {"name": "HighExtraction", "cl_intrinsic_L_per_h": 500.0},
            {"name": "LowExtraction", "cl_intrinsic_L_per_h": 1.0},
            {"name": "MidExtraction", "cl_intrinsic_L_per_h": 45.0},
        ]
        results = screen_hepatic_fp(candidates)
        f_vals = [r.f_bioavailability for r in results]
        assert f_vals == sorted(f_vals, reverse=True)

    def test_optional_fa_fg_defaults(self):
        candidates = [{"name": "X", "cl_intrinsic_L_per_h": 10.0}]
        result = screen_hepatic_fp(candidates)[0]
        assert result.fa == pytest.approx(0.95)
        assert result.fg == pytest.approx(0.9)

    def test_optional_fa_fg_overridden(self):
        candidates = [{"name": "X", "cl_intrinsic_L_per_h": 10.0, "fa": 0.7, "fg": 0.6}]
        result = screen_hepatic_fp(candidates)[0]
        assert result.fa == pytest.approx(0.7)
        assert result.fg == pytest.approx(0.6)

    def test_empty_list(self):
        assert screen_hepatic_fp([]) == []
