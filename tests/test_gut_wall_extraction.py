"""Tests for gut wall extraction model."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.gut_wall_extraction import (
    GutExtractionResult,
    gut_wall_extraction,
    predict_oral_f,
    sensitivity_analysis,
)


class TestGutExtractionResult:
    """Test GutExtractionResult dataclass."""

    def test_returns_correct_type(self):
        r = gut_wall_extraction(peff_cm_s=1e-4)
        assert isinstance(r, GutExtractionResult)

    def test_frozen_dataclass(self):
        r = gut_wall_extraction(peff_cm_s=1e-4)
        with pytest.raises(Exception):
            r.fg = 0.5  # type: ignore[misc]

    def test_fields_present(self):
        r = gut_wall_extraction(peff_cm_s=2e-4)
        assert hasattr(r, "peff_cm_s")
        assert hasattr(r, "fg")
        assert hasattr(r, "eg")
        assert hasattr(r, "fa")
        assert hasattr(r, "f_combined")
        assert hasattr(r, "qgut_L_per_h")
        assert hasattr(r, "notes")


class TestFaFromPermeability:
    """Test sigmoid fa estimation from Peff."""

    def test_high_peff_gives_high_fa(self):
        # Peff = 5e-4 cm/s → peff_scaled = 5 → should be well above 0.5
        r = gut_wall_extraction(peff_cm_s=5e-4)
        assert r.fa > 0.9

    def test_low_peff_gives_low_fa(self):
        # Peff = 1e-5 cm/s → very low absorption
        r = gut_wall_extraction(peff_cm_s=1e-5)
        assert r.fa < 0.2

    def test_mid_peff_gives_mid_fa(self):
        # Peff = 2e-4 cm/s → peff_scaled = 2 → fa ≈ 0.5
        r = gut_wall_extraction(peff_cm_s=2e-4)
        assert 0.4 < r.fa < 0.6

    def test_fa_in_zero_one_range(self):
        for peff in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            r = gut_wall_extraction(peff_cm_s=peff)
            assert 0.0 <= r.fa <= 1.0, f"fa out of range for peff={peff}"


class TestFgFromClintGut:
    """Test well-stirred gut model for fg."""

    def test_no_clint_gives_fg_one(self):
        r = gut_wall_extraction(peff_cm_s=1e-4)
        assert r.fg == pytest.approx(1.0)
        assert r.eg == pytest.approx(0.0)

    def test_high_clint_gives_low_fg(self):
        # High metabolic clearance → high extraction
        r = gut_wall_extraction(peff_cm_s=1e-4, clint_gut_L_per_h=1000.0, qgut_L_per_h=18.0)
        assert r.fg < 0.1
        assert r.eg > 0.9

    def test_zero_clint_gives_fg_one(self):
        r = gut_wall_extraction(peff_cm_s=1e-4, clint_gut_L_per_h=0.0, qgut_L_per_h=18.0)
        assert r.fg == pytest.approx(1.0)
        assert r.eg == pytest.approx(0.0)

    def test_clint_equals_qgut_gives_eg_half(self):
        # CLint*fu = Qgut → EG = 0.5, fg = 0.5
        qgut = 18.0
        r = gut_wall_extraction(peff_cm_s=1e-4, clint_gut_L_per_h=qgut, qgut_L_per_h=qgut,
                                 fu_gut=1.0)
        assert r.eg == pytest.approx(0.5, rel=1e-6)
        assert r.fg == pytest.approx(0.5, rel=1e-6)

    def test_fu_gut_scales_effective_clint(self):
        # fu_gut=0.5 should halve the effective CLint
        r_full = gut_wall_extraction(peff_cm_s=1e-4, clint_gut_L_per_h=36.0,
                                      qgut_L_per_h=18.0, fu_gut=1.0)
        r_half = gut_wall_extraction(peff_cm_s=1e-4, clint_gut_L_per_h=72.0,
                                      qgut_L_per_h=18.0, fu_gut=0.5)
        assert r_full.eg == pytest.approx(r_half.eg, rel=1e-6)


class TestFgFactor:
    """Test directly provided fg_factor."""

    def test_fg_factor_used_directly(self):
        r = gut_wall_extraction(peff_cm_s=1e-4, fg_factor=0.7)
        assert r.fg == pytest.approx(0.7)
        assert r.eg == pytest.approx(0.3, rel=1e-6)

    def test_fg_factor_zero(self):
        r = gut_wall_extraction(peff_cm_s=1e-4, fg_factor=0.0)
        assert r.fg == pytest.approx(0.0)
        assert r.eg == pytest.approx(1.0)

    def test_fg_factor_one(self):
        r = gut_wall_extraction(peff_cm_s=1e-4, fg_factor=1.0)
        assert r.fg == pytest.approx(1.0)
        assert r.eg == pytest.approx(0.0)


class TestFCombined:
    """Test f_combined = fa * fg."""

    def test_f_combined_equals_fa_times_fg(self):
        r = gut_wall_extraction(peff_cm_s=3e-4, clint_gut_L_per_h=10.0)
        assert r.f_combined == pytest.approx(r.fa * r.fg, rel=1e-6)

    def test_f_combined_in_zero_one_range(self):
        r = gut_wall_extraction(peff_cm_s=1e-4, clint_gut_L_per_h=5.0)
        assert 0.0 <= r.f_combined <= 1.0

    def test_high_peff_no_metabolism_gives_high_f(self):
        r = gut_wall_extraction(peff_cm_s=1e-3, clint_gut_L_per_h=0.0)
        assert r.f_combined > 0.9


class TestValidation:
    """Test input validation."""

    def test_zero_peff_raises(self):
        with pytest.raises(ValueError, match="peff_cm_s"):
            gut_wall_extraction(peff_cm_s=0.0)

    def test_negative_peff_raises(self):
        with pytest.raises(ValueError, match="peff_cm_s"):
            gut_wall_extraction(peff_cm_s=-1e-4)

    def test_zero_qgut_raises(self):
        with pytest.raises(ValueError, match="qgut_L_per_h"):
            gut_wall_extraction(peff_cm_s=1e-4, qgut_L_per_h=0.0)

    def test_fu_gut_zero_raises(self):
        with pytest.raises(ValueError, match="fu_gut"):
            gut_wall_extraction(peff_cm_s=1e-4, fu_gut=0.0)

    def test_fu_gut_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_gut"):
            gut_wall_extraction(peff_cm_s=1e-4, fu_gut=1.5)

    def test_negative_clint_raises(self):
        with pytest.raises(ValueError, match="clint_gut_L_per_h"):
            gut_wall_extraction(peff_cm_s=1e-4, clint_gut_L_per_h=-1.0)

    def test_fg_factor_out_of_range_raises(self):
        with pytest.raises(ValueError, match="fg_factor"):
            gut_wall_extraction(peff_cm_s=1e-4, fg_factor=1.5)


class TestPredictOralF:
    """Tests for predict_oral_f function."""

    def test_basic_calculation(self):
        result = predict_oral_f(fa=0.9, fg=0.8, fh=0.7)
        expected_F = 0.9 * 0.8 * 0.7
        assert result["F"] == pytest.approx(expected_F, rel=1e-6)

    def test_all_ones_gives_one(self):
        result = predict_oral_f(fa=1.0, fg=1.0, fh=1.0)
        assert result["F"] == pytest.approx(1.0)

    def test_all_zeros_gives_zero(self):
        result = predict_oral_f(fa=0.0, fg=0.0, fh=0.0)
        assert result["F"] == pytest.approx(0.0)

    def test_high_f_classification(self):
        result = predict_oral_f(fa=0.95, fg=0.95, fh=0.95)
        assert "High" in result["bioavailability_class"]

    def test_very_low_f_classification(self):
        result = predict_oral_f(fa=0.2, fg=0.3, fh=0.2)
        F = result["F"]
        if F < 0.2:
            assert "Very low" in result["bioavailability_class"]

    def test_result_contains_required_keys(self):
        result = predict_oral_f(fa=0.5, fg=0.6, fh=0.7)
        assert "F" in result
        assert "fa" in result
        assert "fg" in result
        assert "fh" in result
        assert "bioavailability_class" in result
        assert "limiting_step" in result
        assert "notes" in result

    def test_clamps_inputs_above_one(self):
        result = predict_oral_f(fa=1.5, fg=1.0, fh=1.0)
        assert result["F"] == pytest.approx(1.0)


class TestSensitivityAnalysis:
    """Tests for sensitivity_analysis function."""

    def test_returns_list(self):
        rows = sensitivity_analysis(
            peff_range=[1e-4, 5e-4],
            fg_range=[0.5, 1.0],
            fh_range=[0.7],
        )
        assert isinstance(rows, list)

    def test_correct_number_of_rows(self):
        rows = sensitivity_analysis(
            peff_range=[1e-4, 5e-4],
            fg_range=[0.5, 1.0],
            fh_range=[0.7, 0.9],
        )
        assert len(rows) == 2 * 2 * 2  # 8 rows

    def test_row_has_required_keys(self):
        rows = sensitivity_analysis(
            peff_range=[2e-4], fg_range=[0.8], fh_range=[0.6]
        )
        assert len(rows) == 1
        row = rows[0]
        assert "peff_cm_s" in row
        assert "fa" in row
        assert "fg" in row
        assert "fh" in row
        assert "F" in row

    def test_f_equals_fa_times_fg_times_fh(self):
        rows = sensitivity_analysis(
            peff_range=[3e-4], fg_range=[0.7], fh_range=[0.5]
        )
        row = rows[0]
        assert row["F"] == pytest.approx(row["fa"] * row["fg"] * row["fh"], rel=1e-6)

    def test_higher_peff_gives_higher_f(self):
        rows = sensitivity_analysis(
            peff_range=[1e-5, 1e-3],
            fg_range=[1.0],
            fh_range=[1.0],
        )
        assert rows[1]["F"] > rows[0]["F"]
