"""Tests for oral absorption prediction — BCS fa from Peff/solubility/dose."""

import pytest

from omega_pbpk.prediction.absorption_predictor import (
    AbsorptionResult,
    _absorption_number,
    _dose_number,
    predict_absorption,
    screen_absorption,
)


class TestHelpers:
    def test_absorption_number(self):
        """An = Peff * 2 * t * 3600 / R."""
        an = _absorption_number(1e-4, transit_h=3.5)
        expected = 1e-4 * 2 * 3.5 * 3600 / 1.75
        assert abs(an - expected) < 1e-9

    def test_dose_number(self):
        """Do = dose / (Cs * 250)."""
        do = _dose_number(250.0, 1.0)
        assert abs(do - 1.0) < 1e-9

    def test_dose_number_high(self):
        do = _dose_number(1000.0, 0.1)
        assert do == pytest.approx(1000.0 / (0.1 * 250.0))


class TestPredictAbsorption:
    def test_input_validation(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_absorption("X", dose_mg=0, peff_cm_s=1e-4, solubility_mg_mL=1.0)
        with pytest.raises(ValueError, match="peff"):
            predict_absorption("X", dose_mg=100, peff_cm_s=0, solubility_mg_mL=1.0)
        with pytest.raises(ValueError, match="solubility"):
            predict_absorption("X", dose_mg=100, peff_cm_s=1e-4, solubility_mg_mL=0)

    def test_bcs_class_i(self):
        """High perm + high sol → BCS I, high fa."""
        result = predict_absorption("aspirin", dose_mg=100, peff_cm_s=5e-4, solubility_mg_mL=10.0)
        assert result.bcs_like == "I"
        assert result.fa > 0.8

    def test_bcs_class_ii(self):
        """High perm + low sol → BCS II, dissolution-limited."""
        result = predict_absorption(
            "poorly_soluble", dose_mg=100, peff_cm_s=5e-4, solubility_mg_mL=0.001
        )
        assert result.bcs_like == "II"
        assert result.limiting_factor == "dissolution"
        assert result.fa < 0.5

    def test_bcs_class_iii(self):
        """Low perm + high sol → BCS III, permeability-limited."""
        result = predict_absorption(
            "poorly_permeable", dose_mg=50, peff_cm_s=1e-6, solubility_mg_mL=100.0
        )
        assert result.bcs_like == "III"
        assert result.limiting_factor == "permeability"
        assert result.fa < 0.5

    def test_bcs_class_iv(self):
        """Low perm + low sol → BCS IV, low fa."""
        result = predict_absorption("bcs4", dose_mg=200, peff_cm_s=1e-6, solubility_mg_mL=0.001)
        assert result.bcs_like == "IV"
        assert result.fa < 0.3

    def test_fa_bounds(self):
        result = predict_absorption("drug", dose_mg=100, peff_cm_s=1e-4, solubility_mg_mL=1.0)
        assert 0.0 <= result.fa <= 1.0
        assert 0.0 <= result.fa_permeability <= 1.0
        assert 0.0 <= result.fa_dissolution <= 1.0

    def test_fa_is_min(self):
        """fa = min(fa_perm, fa_diss)."""
        result = predict_absorption("drug", dose_mg=500, peff_cm_s=5e-4, solubility_mg_mL=0.01)
        assert result.fa == pytest.approx(min(result.fa_permeability, result.fa_dissolution))

    def test_high_peff_approaches_1(self):
        """Very high permeability: fa_perm approaches 1."""
        result = predict_absorption("drug", dose_mg=10, peff_cm_s=1e-2, solubility_mg_mL=1.0)
        assert result.fa_permeability > 0.99

    def test_result_structure(self):
        result = predict_absorption("test", dose_mg=100, peff_cm_s=1e-4, solubility_mg_mL=1.0)
        assert isinstance(result, AbsorptionResult)
        assert result.drug_name == "test"
        assert result.bcs_like in ("I", "II", "III", "IV")
        assert result.limiting_factor in ("dissolution", "permeability", "none")
        assert result.t_abs_h >= 0
        assert 0.0 < result.confidence <= 1.0

    def test_confidence_in_range(self):
        result = predict_absorption("drug", dose_mg=100, peff_cm_s=1e-4, solubility_mg_mL=1.0)
        assert result.confidence == pytest.approx(0.80)

    def test_dose_number_stored(self):
        result = predict_absorption("drug", dose_mg=250, peff_cm_s=1e-4, solubility_mg_mL=1.0)
        expected_do = 250.0 / (1.0 * 250.0)
        assert result.dose_number == pytest.approx(expected_do)


class TestScreenAbsorption:
    def test_screen_multiple(self):
        compounds = [
            {"drug_name": "A", "dose_mg": 100, "peff_cm_s": 5e-4, "solubility_mg_mL": 10.0},
            {"drug_name": "B", "dose_mg": 200, "peff_cm_s": 1e-6, "solubility_mg_mL": 0.001},
        ]
        results = screen_absorption(compounds)
        assert len(results) == 2
        assert results[0].drug_name == "A"
        assert results[1].drug_name == "B"
        assert results[0].fa > results[1].fa
