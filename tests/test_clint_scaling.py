"""Tests for core/clint_scaling.py — IVIVE intrinsic clearance scaling."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.clint_scaling import (
    IVIVEResult,
    fu_mic_correction,
    hepatocyte_clint_scale,
    intrinsic_clearance_from_t_half,
    ivive_pipeline,
    microsomal_clint_scale,
    well_stirred_cl,
)

# ---------------------------------------------------------------------------
# microsomal_clint_scale
# ---------------------------------------------------------------------------


class TestMicrosomalClintScale:
    def test_basic_calculation(self):
        # CLint = 1 µL/min/mg, protein=0.5 mg/mL, liver=1500g, mppgl=45
        # = 1 * 0.5 * 1500 * 45 * 60 / 1e6
        expected = 1.0 * 0.5 * 1500.0 * 45.0 * 60.0 / 1e6
        result = microsomal_clint_scale(1.0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_known_value(self):
        # CLint=10, defaults → 10 * 0.5 * 1500 * 45 * 60 / 1e6 = 20.25 L/h
        result = microsomal_clint_scale(10.0)
        assert math.isclose(result, 20.25, rel_tol=1e-6)

    def test_custom_params(self):
        result = microsomal_clint_scale(
            clint_uL_per_min_per_mg=5.0,
            protein_conc_mg_per_mL=1.0,
            liver_weight_g=1000.0,
            mppgl=40.0,
        )
        expected = 5.0 * 1.0 * 1000.0 * 40.0 * 60.0 / 1e6
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_returns_float(self):
        assert isinstance(microsomal_clint_scale(2.0), float)

    def test_raises_on_zero_clint(self):
        with pytest.raises(ValueError, match="clint_uL_per_min_per_mg"):
            microsomal_clint_scale(0.0)

    def test_raises_on_negative_clint(self):
        with pytest.raises(ValueError):
            microsomal_clint_scale(-1.0)

    def test_raises_on_zero_protein(self):
        with pytest.raises(ValueError, match="protein_conc"):
            microsomal_clint_scale(1.0, protein_conc_mg_per_mL=0.0)

    def test_raises_on_zero_liver_weight(self):
        with pytest.raises(ValueError, match="liver_weight"):
            microsomal_clint_scale(1.0, liver_weight_g=0.0)

    def test_proportional_to_clint(self):
        r1 = microsomal_clint_scale(1.0)
        r2 = microsomal_clint_scale(2.0)
        assert math.isclose(r2, 2.0 * r1, rel_tol=1e-9)

    def test_proportional_to_mppgl(self):
        r1 = microsomal_clint_scale(1.0, mppgl=45.0)
        r2 = microsomal_clint_scale(1.0, mppgl=90.0)
        assert math.isclose(r2, 2.0 * r1, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# hepatocyte_clint_scale
# ---------------------------------------------------------------------------


class TestHepatocyteClintScale:
    def test_basic_calculation(self):
        # CLint=1 µL/min/10^6 cells, liver=1500g, hpgl=120e6
        # = 1 * (120e6/1e6) * 1500 * 60 / 1e6 = 1 * 120 * 1500 * 60 / 1e6
        expected = 1.0 * 120.0 * 1500.0 * 60.0 / 1e6
        result = hepatocyte_clint_scale(1.0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_known_value(self):
        # CLint=1, defaults → 10.8 L/h
        result = hepatocyte_clint_scale(1.0)
        assert math.isclose(result, 10.8, rel_tol=1e-6)

    def test_returns_float(self):
        assert isinstance(hepatocyte_clint_scale(2.0), float)

    def test_raises_on_zero_clint(self):
        with pytest.raises(ValueError):
            hepatocyte_clint_scale(0.0)

    def test_raises_on_negative_liver_weight(self):
        with pytest.raises(ValueError, match="liver_weight"):
            hepatocyte_clint_scale(1.0, liver_weight_g=-100.0)

    def test_raises_on_zero_hpgl(self):
        with pytest.raises(ValueError, match="hpgl"):
            hepatocyte_clint_scale(1.0, hpgl=0.0)

    def test_proportional_to_clint(self):
        r1 = hepatocyte_clint_scale(1.0)
        r2 = hepatocyte_clint_scale(3.0)
        assert math.isclose(r2, 3.0 * r1, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# fu_mic_correction
# ---------------------------------------------------------------------------


class TestFuMicCorrection:
    def test_no_correction(self):
        # fu_mic=1 → no change
        assert math.isclose(fu_mic_correction(10.0, 1.0), 10.0)

    def test_half_binding(self):
        # fu_mic=0.5 → CLint doubles
        assert math.isclose(fu_mic_correction(10.0, 0.5), 20.0)

    def test_strong_binding(self):
        result = fu_mic_correction(5.0, 0.1)
        assert math.isclose(result, 50.0)

    def test_raises_on_zero_clint(self):
        with pytest.raises(ValueError, match="clint_obs"):
            fu_mic_correction(0.0, 0.5)

    def test_raises_on_zero_fu_mic(self):
        with pytest.raises(ValueError, match="fu_mic"):
            fu_mic_correction(10.0, 0.0)

    def test_raises_on_fu_mic_greater_than_one(self):
        with pytest.raises(ValueError, match="fu_mic"):
            fu_mic_correction(10.0, 1.5)


# ---------------------------------------------------------------------------
# well_stirred_cl
# ---------------------------------------------------------------------------


class TestWellStirredCl:
    def test_low_extraction(self):
        # Very small CLint → CL_H ≈ fu * CLint (flow-independent)
        cl_h = well_stirred_cl(0.1, fu_blood=0.1, qh_L_per_h=90.0)
        # Expect ≈ 0.1 * 0.01 * ... well below QH
        assert cl_h < 0.015

    def test_high_extraction(self):
        # Very large CLint → CL_H → QH
        cl_h = well_stirred_cl(1e6, fu_blood=1.0, qh_L_per_h=90.0)
        assert math.isclose(cl_h, 90.0, rel_tol=0.01)

    def test_bounded_by_qh(self):
        cl_h = well_stirred_cl(1000.0, fu_blood=1.0, qh_L_per_h=90.0)
        assert cl_h < 90.0

    def test_known_value(self):
        # CLint=90, fu=1, QH=90 → CL_H = 90*1*90/(90+90) = 45
        cl_h = well_stirred_cl(90.0, fu_blood=1.0, qh_L_per_h=90.0)
        assert math.isclose(cl_h, 45.0, rel_tol=1e-6)

    def test_fu_effect(self):
        cl1 = well_stirred_cl(10.0, fu_blood=1.0, qh_L_per_h=90.0)
        cl2 = well_stirred_cl(10.0, fu_blood=0.1, qh_L_per_h=90.0)
        assert cl2 < cl1

    def test_raises_on_zero_clint(self):
        with pytest.raises(ValueError, match="clint"):
            well_stirred_cl(0.0, 0.5)

    def test_raises_on_invalid_fu(self):
        with pytest.raises(ValueError, match="fu_blood"):
            well_stirred_cl(10.0, fu_blood=0.0)

    def test_raises_on_zero_qh(self):
        with pytest.raises(ValueError, match="qh"):
            well_stirred_cl(10.0, fu_blood=0.5, qh_L_per_h=0.0)


# ---------------------------------------------------------------------------
# intrinsic_clearance_from_t_half
# ---------------------------------------------------------------------------


class TestIntrinsicClearanceFromTHalf:
    def test_basic(self):
        # t_half=60 min, protein=0.5 mg/mL, volume=0.5 mL
        ke = 0.693 / 60.0
        protein_amount = 0.5 * 0.5  # 0.25 mg
        clint_mL = ke * 0.5 / protein_amount
        expected = clint_mL * 1000.0
        result = intrinsic_clearance_from_t_half(60.0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_proportional_to_volume(self):
        r1 = intrinsic_clearance_from_t_half(60.0, incubation_volume_mL=0.5)
        r2 = intrinsic_clearance_from_t_half(60.0, incubation_volume_mL=1.0)
        # Doubling volume doubles CLint (same protein concentration)
        assert math.isclose(r2, r1, rel_tol=1e-6)

    def test_shorter_t_half_higher_clint(self):
        r1 = intrinsic_clearance_from_t_half(120.0)
        r2 = intrinsic_clearance_from_t_half(30.0)
        assert r2 > r1

    def test_raises_on_zero_t_half(self):
        with pytest.raises(ValueError, match="t_half"):
            intrinsic_clearance_from_t_half(0.0)

    def test_raises_on_negative_protein(self):
        with pytest.raises(ValueError, match="protein"):
            intrinsic_clearance_from_t_half(60.0, protein_mg_per_mL=-1.0)

    def test_returns_float(self):
        assert isinstance(intrinsic_clearance_from_t_half(60.0), float)

    def test_units_positive(self):
        result = intrinsic_clearance_from_t_half(45.0)
        assert result > 0


# ---------------------------------------------------------------------------
# ivive_pipeline
# ---------------------------------------------------------------------------


class TestIVIVEPipeline:
    def test_returns_ivive_result(self):
        result = ivive_pipeline("test_drug", clint_uL_per_min_per_mg=5.0, fu_plasma=0.1)
        assert isinstance(result, IVIVEResult)

    def test_frozen_dataclass(self):
        result = ivive_pipeline("test_drug", 5.0, 0.1)
        with pytest.raises((AttributeError, TypeError)):
            result.compound_name = "other"  # type: ignore[misc]

    def test_compound_name(self):
        result = ivive_pipeline("warfarin", 5.0, 0.1)
        assert result.compound_name == "warfarin"

    def test_clint_obs_preserved(self):
        result = ivive_pipeline("drug", 7.5, 0.5)
        assert math.isclose(result.clint_obs, 7.5)

    def test_fu_mic_correction_applied(self):
        r1 = ivive_pipeline("drug", 5.0, 0.1, fu_mic=1.0)
        r2 = ivive_pipeline("drug", 5.0, 0.1, fu_mic=0.5)
        assert r2.clint_corrected > r1.clint_corrected
        assert math.isclose(r2.clint_corrected, 2.0 * r1.clint_corrected)

    def test_no_fu_mic_correction_when_one(self):
        result = ivive_pipeline("drug", 5.0, 0.1, fu_mic=1.0)
        assert math.isclose(result.clint_obs, result.clint_corrected)

    def test_extraction_ratio_range(self):
        result = ivive_pipeline("drug", 5.0, 0.5)
        assert 0.0 < result.extraction_ratio <= 1.0

    def test_high_clint_high_er(self):
        result = ivive_pipeline("drug", 500.0, 1.0)
        assert result.extraction_ratio > 0.7

    def test_low_clint_low_er(self):
        result = ivive_pipeline("drug", 0.01, 0.01)
        assert result.extraction_ratio < 0.3

    def test_cl_h_less_than_qh(self):
        result = ivive_pipeline("drug", 5.0, 0.5, qh_L_per_h=90.0)
        assert result.cl_hepatic_L_per_h < 90.0

    def test_raises_on_invalid_fu_plasma(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            ivive_pipeline("drug", 5.0, fu_plasma=1.5)

    def test_raises_on_invalid_fu_mic(self):
        with pytest.raises(ValueError, match="fu_mic"):
            ivive_pipeline("drug", 5.0, 0.1, fu_mic=0.0)

    def test_raises_on_zero_clint(self):
        with pytest.raises(ValueError):
            ivive_pipeline("drug", 0.0, 0.1)

    def test_notes_nonempty(self):
        result = ivive_pipeline("drug", 5.0, 0.1)
        assert len(result.notes) > 0

    def test_fu_plasma_stored(self):
        result = ivive_pipeline("drug", 5.0, fu_plasma=0.3)
        assert math.isclose(result.fu_plasma, 0.3)

    def test_fu_mic_stored(self):
        result = ivive_pipeline("drug", 5.0, 0.1, fu_mic=0.7)
        assert math.isclose(result.fu_mic, 0.7)
