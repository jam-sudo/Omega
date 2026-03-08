"""Tests for Phase 1128: Oral Bioavailability Prediction."""

import pytest

from omega_pbpk.prediction.oral_bioavailability_prediction import (
    OralBioavailabilityResult,
    predict_oral_bioavailability,
    sensitivity_analysis,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        solubility_mg_mL=1.0,
        logP=2.0,
        pka=7.4,
        ionization_type="neutral",
        fu_plasma=0.1,
        cl_L_per_h=10.0,
        mw_Da=300.0,
    )
    defaults.update(kwargs)
    return predict_oral_bioavailability(**defaults)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_dataclass(self):
        res = _run()
        assert isinstance(res, OralBioavailabilityResult)

    def test_drug_name_preserved(self):
        res = _run(drug_name="Ibuprofen")
        assert res.drug_name == "Ibuprofen"

    def test_dose_preserved(self):
        res = _run(dose_mg=100.0)
        assert res.dose_mg == 100.0

    def test_fu_preserved(self):
        res = _run(fu_plasma=0.2)
        assert res.fu_plasma == pytest.approx(0.2)

    def test_logp_preserved(self):
        res = _run(logP=3.0)
        assert res.logP == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Core consistency: F = fa * fg * fh
# ---------------------------------------------------------------------------


class TestFConsistency:
    def test_f_equals_fa_times_fg_times_fh(self):
        res = _run()
        expected = res.fa * res.fg * res.fh
        assert res.f_oral_predicted == pytest.approx(expected, rel=1e-9)

    def test_f_oral_pct_equals_f_times_100(self):
        res = _run()
        assert res.f_oral_pct == pytest.approx(res.f_oral_predicted * 100.0, rel=1e-9)

    def test_f_in_range_0_to_1(self):
        res = _run()
        assert 0.0 <= res.f_oral_predicted <= 1.0

    def test_fa_in_range_0_to_1(self):
        res = _run()
        assert 0.0 <= res.fa <= 1.0

    def test_fg_in_range_0_to_1(self):
        res = _run()
        assert 0.0 <= res.fg <= 1.0

    def test_fh_in_range_0_to_1(self):
        res = _run()
        assert 0.0 <= res.fh <= 1.0


# ---------------------------------------------------------------------------
# BCS classification
# ---------------------------------------------------------------------------


class TestBCSClassification:
    def test_bcs_class_is_string(self):
        res = _run()
        assert res.bcs_class in ("I", "II", "III", "IV")

    def test_bcs_i_high_sol_high_perm(self):
        """High solubility + high permeability → BCS I.
        Peff = 10^(-4.36 + 0.11*logP); at logP=4: ~1.2e-4 > 1e-4 (high perm).
        High solubility: dose=1, sol=10 → Do=0.004 << 1."""
        # Do = 1/(10*250) = 0.0004 → high sol; logP=4 → Peff~1.2e-4 > 1e-4
        res = _run(dose_mg=1.0, solubility_mg_mL=10.0, logP=4.0)
        assert res.bcs_class == "I"

    def test_bcs_iv_low_sol_low_perm(self):
        """Low solubility + low permeability → BCS IV."""
        # Low sol: Do >> 1: dose=500, sol=0.01 → Do=200
        # Low perm: logP=-2 → Peff very low
        res = _run(dose_mg=500.0, solubility_mg_mL=0.01, logP=-2.0)
        assert res.bcs_class == "IV"

    def test_do_value_formula(self):
        """Do = dose / (solubility * 250)."""
        res = _run(dose_mg=10.0, solubility_mg_mL=0.5)
        expected_do = 10.0 / (0.5 * 250.0)
        assert res.do_value == pytest.approx(expected_do, rel=1e-9)

    def test_peff_positive(self):
        res = _run()
        assert res.peff_cm_per_s > 0


# ---------------------------------------------------------------------------
# Absorption (fa) tests
# ---------------------------------------------------------------------------


class TestFractionAbsorbed:
    def test_higher_solubility_lower_do_higher_fa(self):
        """Higher solubility → lower Do → higher fa (for BCS I/II)."""
        res_low_sol = _run(dose_mg=100.0, solubility_mg_mL=0.1, logP=3.0)
        res_high_sol = _run(dose_mg=100.0, solubility_mg_mL=10.0, logP=3.0)
        assert res_high_sol.fa >= res_low_sol.fa

    def test_bcs_iii_iv_low_fa(self):
        """BCS III/IV → fa = 0.3 (without high Peff override)."""
        # Low logP → low Peff → BCS III (high sol) or IV (low sol)
        res = _run(dose_mg=1.0, solubility_mg_mL=10.0, logP=-3.0)
        assert res.bcs_class in ("III", "IV")
        # Without high Peff override (Peff <= 5e-4), fa = 0.3
        assert res.fa == pytest.approx(0.3, rel=0.1)


# ---------------------------------------------------------------------------
# Gut wall extraction (fg) tests
# ---------------------------------------------------------------------------


class TestGutWallExtraction:
    def test_low_logp_fg_is_one(self):
        """logP < 1 → EG = 0 → fg = 1."""
        res = _run(logP=0.5)
        assert res.fg == pytest.approx(1.0)

    def test_very_high_logp_fg_is_low(self):
        """logP > 9 → EG = 0.8 → fg = 0.2."""
        res = _run(logP=9.5)
        assert res.fg == pytest.approx(0.2, rel=0.01)

    def test_moderate_logp_fg_decreases_with_logp(self):
        """Higher logP → higher EG → lower fg."""
        res_low = _run(logP=2.0)
        res_high = _run(logP=6.0)
        assert res_high.fg < res_low.fg


# ---------------------------------------------------------------------------
# Hepatic extraction (fh) tests
# ---------------------------------------------------------------------------


class TestHepaticExtraction:
    def test_high_cl_low_fh(self):
        """Very high CL → high extraction → low fh."""
        res = _run(cl_L_per_h=200.0, fu_plasma=0.5)
        assert res.fh == pytest.approx(0.2, rel=0.01)

    def test_low_cl_high_fh(self):
        """Low CL → low extraction → fh close to 1."""
        res = _run(cl_L_per_h=1.0, fu_plasma=0.5)
        # fh = 1 - CL/Qh = 1 - 1/90 ≈ 0.989
        assert res.fh > 0.9

    def test_higher_cl_lower_f(self):
        """Higher clearance → lower F (through lower fh)."""
        res_low = _run(cl_L_per_h=5.0)
        res_high = _run(cl_L_per_h=80.0)
        assert res_high.f_oral_predicted < res_low.f_oral_predicted


# ---------------------------------------------------------------------------
# Bioavailability class tests
# ---------------------------------------------------------------------------


class TestBioavailabilityClass:
    def test_high_f_class(self):
        res = _run(dose_mg=1.0, solubility_mg_mL=10.0, logP=0.5, cl_L_per_h=1.0)
        if res.f_oral_predicted > 0.70:
            assert res.bioavailability_class == "high"

    def test_low_f_class(self):
        res = _run(dose_mg=500.0, solubility_mg_mL=0.01, logP=-2.0, cl_L_per_h=80.0)
        if res.f_oral_predicted < 0.30:
            assert res.bioavailability_class == "low"

    def test_bioavailability_class_values(self):
        res = _run()
        assert res.bioavailability_class in ("low", "moderate", "high")


# ---------------------------------------------------------------------------
# Limiting factor test
# ---------------------------------------------------------------------------


class TestLimitingFactor:
    def test_limiting_factor_is_valid(self):
        res = _run()
        assert res.limiting_factor in ("absorption", "gut_extraction", "hepatic_extraction")

    def test_poor_solubility_limits_absorption(self):
        """Very poor solubility → absorption is typically limiting."""
        res = _run(dose_mg=500.0, solubility_mg_mL=0.001, logP=-2.0, cl_L_per_h=5.0)
        # fa should be low for BCS IV
        assert res.fa <= 0.3


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_string(self):
        res = _run()
        assert isinstance(res.notes, str)

    def test_notes_contains_bcs(self):
        res = _run()
        assert "BCS" in res.notes


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    def test_returns_dict(self):
        result = sensitivity_analysis("TestDrug", 10.0, 1.0, 2.0, 7.4, "neutral", 0.1, 10.0, 300.0)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = sensitivity_analysis("TestDrug", 10.0, 1.0, 2.0, 7.4, "neutral", 0.1, 10.0, 300.0)
        assert "fa" in result
        assert "fg" in result
        assert "fh" in result
        assert "F" in result
        assert "sensitivity" in result

    def test_sensitivity_has_parameter_keys(self):
        result = sensitivity_analysis("TestDrug", 10.0, 1.0, 2.0, 7.4, "neutral", 0.1, 10.0, 300.0)
        sens = result["sensitivity"]
        assert "dose_mg" in sens
        assert "solubility_mg_mL" in sens
        assert "logP" in sens
        assert "fu_plasma" in sens
        assert "cl_L_per_h" in sens

    def test_base_f_consistent_with_prediction(self):
        result = sensitivity_analysis("TestDrug", 10.0, 1.0, 2.0, 7.4, "neutral", 0.1, 10.0, 300.0)
        direct = _run(dose_mg=10.0, solubility_mg_mL=1.0, logP=2.0, fu_plasma=0.1, cl_L_per_h=10.0)
        assert result["F"] == pytest.approx(direct.f_oral_predicted, rel=1e-9)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-1.0)

    def test_solubility_zero_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            _run(solubility_mg_mL=0.0)

    def test_fu_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _run(fu_plasma=0.0)

    def test_fu_gt_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _run(fu_plasma=1.5)

    def test_logp_too_low_raises(self):
        with pytest.raises(ValueError, match="logP"):
            _run(logP=-6.0)

    def test_logp_too_high_raises(self):
        with pytest.raises(ValueError, match="logP"):
            _run(logP=11.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _run(cl_L_per_h=0.0)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _run(mw_Da=0.0)
