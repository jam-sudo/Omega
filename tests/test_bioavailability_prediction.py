"""Tests for oral bioavailability prediction (F = fa × fg × fh)."""

import pytest

from omega_pbpk.prediction.bioavailability_prediction import (
    BioavailabilityResult,
    predict_bioavailability,
)


class TestFaFormula:
    """Unit tests for the fa (fraction absorbed) calculation."""

    def test_high_peff_gives_fa_near_one(self):
        drug = {
            "peff": 7.0,
            "solubility_mg_mL": 20.0,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 1.0,
        }
        result = predict_bioavailability(drug, dose_mg=100.0)
        assert result.fa >= 0.9

    def test_low_peff_gives_low_fa(self):
        drug = {
            "peff": 0.01,
            "solubility_mg_mL": 20.0,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 1.0,
        }
        result = predict_bioavailability(drug, dose_mg=100.0)
        assert result.fa < 0.5

    def test_low_solubility_reduces_fa(self):
        drug = {
            "peff": 7.0,
            "solubility_mg_mL": 0.01,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 1.0,
        }
        # dose_number = 100 / (0.01 * 250) = 40 → fa_sol = 1/40 = 0.025
        result = predict_bioavailability(drug, dose_mg=100.0)
        assert result.fa < 0.1
        assert result.dose_number == pytest.approx(40.0)

    def test_fa_clamped_to_unit_interval(self):
        drug = {
            "peff": 100.0,
            "solubility_mg_mL": 100.0,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 1.0,
        }
        result = predict_bioavailability(drug, dose_mg=1.0)
        assert 0.0 <= result.fa <= 1.0


class TestFgFormula:
    """Unit tests for the fg (gut-wall survival) calculation."""

    def test_no_gut_clint_gives_fg_one(self):
        drug = {
            "peff": 5.0,
            "solubility_mg_mL": 1.0,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 1.0,
            "clint_gut_L_per_h": 0.0,
        }
        result = predict_bioavailability(drug)
        assert result.fg == 1.0

    def test_high_gut_clint_reduces_fg(self):
        drug = {
            "peff": 5.0,
            "solubility_mg_mL": 1.0,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 1.0,
            "clint_gut_L_per_h": 500.0,
        }
        # fg = 18 / (18 + 0.5*500) = 18 / 268 ≈ 0.067
        result = predict_bioavailability(drug)
        assert result.fg < 0.1


class TestFhFormula:
    """Unit tests for fh (hepatic first-pass survival) and extraction ratio."""

    def test_low_clint_gives_high_fh(self):
        drug = {
            "peff": 5.0,
            "solubility_mg_mL": 1.0,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 1.0,
        }
        result = predict_bioavailability(drug, q_liver_L_per_h=90.0)
        assert result.fh > 0.8
        assert result.extraction_ratio < 0.2

    def test_high_clint_gives_low_fh(self):
        drug = {
            "peff": 5.0,
            "solubility_mg_mL": 1.0,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 5000.0,
        }
        result = predict_bioavailability(drug, q_liver_L_per_h=90.0)
        assert result.fh < 0.3
        assert result.extraction_ratio > 0.7

    def test_f_total_equals_product(self):
        drug = {
            "peff": 3.0,
            "solubility_mg_mL": 0.5,
            "fup": 0.3,
            "clint_hepatic_L_per_h": 100.0,
            "clint_gut_L_per_h": 50.0,
        }
        result = predict_bioavailability(drug, dose_mg=200.0)
        assert result.F_total == pytest.approx(result.fa * result.fg * result.fh)

    def test_limiting_factor_valid_string(self):
        drug = {
            "peff": 3.0,
            "solubility_mg_mL": 1.0,
            "fup": 0.5,
            "clint_hepatic_L_per_h": 10.0,
        }
        result = predict_bioavailability(drug)
        assert result.limiting_factor in {
            "absorption",
            "gut_metabolism",
            "hepatic",
            "none",
        }


@pytest.mark.integration
class TestIntegration:
    """Integration tests with real Drug objects."""

    def test_caffeine_high_bioavailability(self):
        from omega_pbpk.drugs.caffeine import CAFFEINE

        result = predict_bioavailability(CAFFEINE, dose_mg=200.0)
        assert result.fa >= 0.9
        assert result.fg == 1.0
        assert result.fh > 0.8
        assert result.F_total > 0.8
        assert result.confidence == "high"
        assert result.F_total == pytest.approx(result.fa * result.fg * result.fh)

    def test_midazolam_low_bioavailability(self):
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        result = predict_bioavailability(MIDAZOLAM, dose_mg=7.5)
        assert result.fa >= 0.7  # high peff but low solubility limits fa
        assert result.fg < 0.5  # high gut CLint
        assert result.F_total < 0.5
        assert result.limiting_factor == "gut_metabolism"
        assert result.confidence == "high"

    def test_midazolam_extraction_ratio(self):
        from omega_pbpk.drugs.midazolam import MIDAZOLAM

        result = predict_bioavailability(MIDAZOLAM, dose_mg=7.5, q_liver_L_per_h=90.0)
        # fup=0.032, CLint=750 → E_h = 0.032*750 / (90 + 0.032*750) = 24/114 ≈ 0.21
        assert 0.1 < result.extraction_ratio < 0.4

    def test_result_is_frozen_dataclass(self):
        drug = {"peff": 5.0, "fup": 0.5, "clint_hepatic_L_per_h": 1.0}
        result = predict_bioavailability(drug)
        assert isinstance(result, BioavailabilityResult)
        with pytest.raises(AttributeError):
            result.fa = 0.5  # type: ignore[misc]
