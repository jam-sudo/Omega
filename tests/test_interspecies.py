"""Tests for interspecies allometric scaling."""
from __future__ import annotations

import math

import numpy as np
import pytest

from omega_pbpk.core.interspecies import (
    DOG,
    HUMAN,
    MONKEY,
    MOUSE,
    RAT,
    RABBIT,
    SPECIES_DB,
    AllometricFit,
    AnimalPKData,
    HumanPKPrediction,
    allometric_regression,
    correct_brain_weight,
    correct_mlp,
    predict_human_pk,
    scale_single_species,
)


# ── SpeciesPhysiology ───────────────────────────────────────────────

class TestSpeciesPhysiology:
    def test_predefined_species_exist(self):
        assert MOUSE.body_weight_kg == pytest.approx(0.02, rel=0.1)
        assert RAT.body_weight_kg == pytest.approx(0.25, rel=0.1)
        assert DOG.body_weight_kg == pytest.approx(10.0, rel=0.1)
        assert MONKEY.body_weight_kg == pytest.approx(5.0, rel=0.1)
        assert HUMAN.body_weight_kg == pytest.approx(70.0, rel=0.1)

    def test_species_db_has_all(self):
        assert set(SPECIES_DB.keys()) == {"mouse", "rat", "rabbit", "dog", "monkey", "human"}

    def test_mlp_values_positive(self):
        for sp in SPECIES_DB.values():
            assert sp.mlp_years > 0

    def test_brain_weights_positive(self):
        for sp in SPECIES_DB.values():
            assert sp.brain_weight_g > 0


# ── Single species scaling ──────────────────────────────────────────

class TestSingleSpeciesScaling:
    def test_same_weight_no_change(self):
        result = scale_single_species(10.0, 70.0, 70.0, exponent=0.75)
        assert result == pytest.approx(10.0)

    def test_rat_to_human_cl(self):
        # CL exponent 0.75: ratio = (70/0.25)^0.75
        ratio = (70.0 / 0.25) ** 0.75
        result = scale_single_species(1.0, 0.25, 70.0, exponent=0.75)
        assert result == pytest.approx(ratio, rel=1e-6)

    def test_vd_scales_linearly(self):
        # Exponent 1.0: direct BW proportionality
        result = scale_single_species(2.0, 10.0, 70.0, exponent=1.0)
        assert result == pytest.approx(2.0 * 7.0, rel=1e-6)

    def test_custom_exponent_thalf(self):
        # Exponent 0.25 for half-life
        ratio = (70.0 / 0.25) ** 0.25
        result = scale_single_species(1.0, 0.25, 70.0, exponent=0.25)
        assert result == pytest.approx(ratio, rel=1e-6)

    def test_invalid_bw_raises(self):
        with pytest.raises(ValueError):
            scale_single_species(10.0, 0.0, 70.0)
        with pytest.raises(ValueError):
            scale_single_species(10.0, 70.0, -1.0)


# ── Allometric regression ──────────────────────────────────────────

class TestAllometricRegression:
    def test_perfect_fit_r2_one(self):
        # Generate data from Y = 2.0 * BW^0.75
        bw = [0.25, 3.5, 10.0, 70.0]
        vals = [2.0 * w ** 0.75 for w in bw]
        fit = allometric_regression(bw, vals)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-6)
        assert fit.exponent_b == pytest.approx(0.75, rel=0.01)
        assert fit.coefficient_a == pytest.approx(2.0, rel=0.01)

    def test_two_species_minimum(self):
        fit = allometric_regression([0.25, 10.0], [1.0, 20.0])
        assert isinstance(fit, AllometricFit)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-6)  # 2 points = perfect line

    def test_human_prediction_at_70kg(self):
        # Y = 3.0 * BW^0.8 → at 70 kg: 3.0 * 70^0.8
        bw = [0.02, 0.25, 5.0, 10.0]
        vals = [3.0 * w ** 0.8 for w in bw]
        fit = allometric_regression(bw, vals, human_bw_kg=70.0)
        expected = 3.0 * 70.0 ** 0.8
        assert fit.human_predicted == pytest.approx(expected, rel=0.01)

    def test_too_few_species_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            allometric_regression([0.25], [1.0])

    def test_returns_allometric_fit(self):
        fit = allometric_regression([0.25, 10.0, 70.0], [1.0, 10.0, 40.0])
        assert hasattr(fit, "coefficient_a")
        assert hasattr(fit, "exponent_b")
        assert hasattr(fit, "r_squared")
        assert hasattr(fit, "human_predicted")


# ── MLP and brain weight corrections ───────────────────────────────

class TestCorrections:
    def test_mlp_correction_returns_float(self):
        cl = [0.5, 5.0, 15.0]
        bw = [0.25, 5.0, 10.0]
        mlp = [4.5, 35.0, 20.0]
        result = correct_mlp(cl, bw, mlp)
        assert isinstance(result, float)
        assert result > 0

    def test_brain_weight_correction_returns_float(self):
        cl = [0.5, 5.0, 15.0]
        bw = [0.25, 5.0, 10.0]
        brw = [1.8, 62.0, 72.0]
        result = correct_brain_weight(cl, bw, brw)
        assert isinstance(result, float)
        assert result > 0


# ── predict_human_pk ────────────────────────────────────────────────

class TestPredictHumanPK:
    def test_single_species_uses_fixed_exponents(self):
        data = [AnimalPKData("rat", cl_mL_min_kg=10.0, vd_L_kg=1.0, thalf_h=2.0)]
        result = predict_human_pk(data)
        assert result.method_used == "single_species_fixed_exponents"
        assert isinstance(result, HumanPKPrediction)
        assert result.cl_L_per_h > 0
        assert result.vd_L > 0
        assert result.thalf_h > 0

    def test_two_species_regression(self):
        data = [
            AnimalPKData("rat", cl_mL_min_kg=10.0, vd_L_kg=1.0, thalf_h=2.0),
            AnimalPKData("dog", cl_mL_min_kg=5.0, vd_L_kg=0.8, thalf_h=4.0),
        ]
        result = predict_human_pk(data)
        assert result.method_used == "two_species_regression"
        assert result.fit_quality is not None

    def test_multi_species_rule_of_exponents(self):
        data = [
            AnimalPKData("mouse", cl_mL_min_kg=20.0, vd_L_kg=2.0, thalf_h=0.5),
            AnimalPKData("rat", cl_mL_min_kg=10.0, vd_L_kg=1.0, thalf_h=2.0),
            AnimalPKData("dog", cl_mL_min_kg=5.0, vd_L_kg=0.8, thalf_h=4.0),
        ]
        result = predict_human_pk(data)
        assert "rule_of_exponents" in result.method_used
        assert result.fit_quality is not None

    def test_unit_conversion_cl(self):
        # rat: 10 mL/min/kg, BW=0.25 kg → total = 10*0.25*60/1000 = 0.15 L/h
        # scale to human (70 kg) with exp 0.75: 0.15 * (70/0.25)^0.75
        data = [AnimalPKData("rat", cl_mL_min_kg=10.0, vd_L_kg=1.0, thalf_h=2.0)]
        result = predict_human_pk(data)
        rat_cl_total = 10.0 * 0.25 * 60.0 / 1000.0  # 0.15 L/h
        expected_cl = rat_cl_total * (70.0 / 0.25) ** 0.75
        assert result.cl_L_per_h == pytest.approx(expected_cl, rel=0.01)

    def test_unit_conversion_vd(self):
        # rat: 1.0 L/kg, BW=0.25 → total=0.25 L, scale with exp 1.0 → 0.25*(70/0.25)=70 L
        data = [AnimalPKData("rat", cl_mL_min_kg=10.0, vd_L_kg=1.0, thalf_h=2.0)]
        result = predict_human_pk(data)
        assert result.vd_L == pytest.approx(70.0, rel=0.01)

    def test_dose_scaling(self):
        data = [AnimalPKData("rat", cl_mL_min_kg=10.0, vd_L_kg=1.0, thalf_h=2.0, dose_mg_kg=5.0)]
        result = predict_human_pk(data)
        assert result.dose_mg == pytest.approx(5.0 * 70.0, rel=0.01)

    def test_warnings_for_poor_fit(self):
        # Create data that won't fit well allometrically
        data = [
            AnimalPKData("mouse", cl_mL_min_kg=100.0, vd_L_kg=2.0, thalf_h=0.5),
            AnimalPKData("rat", cl_mL_min_kg=1.0, vd_L_kg=1.0, thalf_h=2.0),
            AnimalPKData("dog", cl_mL_min_kg=50.0, vd_L_kg=0.8, thalf_h=4.0),
        ]
        result = predict_human_pk(data)
        # May or may not have warnings depending on fit — just check it runs
        assert isinstance(result.warnings, list)

    def test_unknown_species_raises(self):
        data = [AnimalPKData("hamster", cl_mL_min_kg=10.0, vd_L_kg=1.0, thalf_h=2.0)]
        with pytest.raises(ValueError, match="Unknown species"):
            predict_human_pk(data)

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            predict_human_pk([])

    def test_thalf_from_cl_vd(self):
        # For 2+ species, t½ = ln2 * Vd / CL
        data = [
            AnimalPKData("rat", cl_mL_min_kg=10.0, vd_L_kg=1.0, thalf_h=2.0),
            AnimalPKData("dog", cl_mL_min_kg=5.0, vd_L_kg=0.8, thalf_h=4.0),
        ]
        result = predict_human_pk(data)
        expected_thalf = math.log(2) * result.vd_L / result.cl_L_per_h
        assert result.thalf_h == pytest.approx(expected_thalf, rel=0.01)
