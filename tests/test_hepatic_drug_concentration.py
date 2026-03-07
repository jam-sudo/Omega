"""
Tests for Phase 929: hepatic drug concentration prediction.
"""

import pytest
from omega_pbpk.prediction.hepatic_drug_concentration import (
    HepaticDrugConcentrationResult,
    predict_hepatic_concentration,
    screen_hepatic_risk,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _base_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        logp=2.0,
        fu_plasma=0.1,
        pka=8.0,
        ionization_type="neutral",
        c_plasma_mg_L=1.0,
        clint_L_per_h=5.0,
        q_liver_L_per_h=97.0,
    )
    defaults.update(kwargs)
    return predict_hepatic_concentration(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_correct_type(self):
        result = _base_result()
        assert isinstance(result, HepaticDrugConcentrationResult)

    def test_result_is_frozen(self):
        result = _base_result()
        with pytest.raises((AttributeError, TypeError)):
            result.kp_liver = 999  # type: ignore


# ---------------------------------------------------------------------------
# Basic values
# ---------------------------------------------------------------------------

class TestBasicValues:
    def test_kp_liver_positive(self):
        result = _base_result()
        assert result.kp_liver > 0

    def test_c_liver_positive_when_plasma_nonzero(self):
        result = _base_result(c_plasma_mg_L=1.0)
        assert result.c_liver_mg_L > 0

    def test_c_liver_equals_kp_times_plasma(self):
        result = _base_result(c_plasma_mg_L=2.0)
        assert abs(result.c_liver_mg_L - result.kp_liver * result.c_plasma_mg_L) < 1e-9

    def test_liver_to_plasma_ratio_equals_kp(self):
        result = _base_result()
        assert abs(result.liver_to_plasma_ratio - result.kp_liver) < 1e-9

    def test_e_hepatic_in_range(self):
        result = _base_result()
        assert 0 <= result.e_hepatic <= 1

    def test_c_hepatocyte_positive(self):
        result = _base_result(c_plasma_mg_L=1.0)
        assert result.c_hepatocyte_mg_L > 0

    def test_c_plasma_zero_gives_zero_liver(self):
        result = _base_result(c_plasma_mg_L=0.0)
        assert result.c_liver_mg_L == 0.0
        assert result.c_hepatocyte_mg_L == 0.0


# ---------------------------------------------------------------------------
# Drug property effects
# ---------------------------------------------------------------------------

class TestDrugPropertyEffects:
    def test_high_logp_increases_kp_liver(self):
        low = _base_result(logp=0.0)
        high = _base_result(logp=4.0)
        assert high.kp_liver > low.kp_liver

    def test_low_fu_plasma_increases_kp_liver(self):
        # Lower fu means less unbound = higher tissue partitioning
        high_fu = _base_result(fu_plasma=0.9)
        low_fu = _base_result(fu_plasma=0.01)
        assert low_fu.kp_liver > high_fu.kp_liver


# ---------------------------------------------------------------------------
# DILI risk
# ---------------------------------------------------------------------------

class TestDiliRisk:
    def test_dili_risk_valid_value(self):
        result = _base_result()
        assert result.dili_risk in {"low", "moderate", "high"}

    def test_high_kp_gives_high_dili(self):
        # Very high logP + very low fu -> kp_liver > 100 -> high risk
        result = _base_result(logp=6.0, fu_plasma=0.001)
        assert result.dili_risk == "high"

    def test_low_logp_gives_low_dili(self):
        result = _base_result(logp=-1.0, fu_plasma=0.9)
        assert result.dili_risk == "low"


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

class TestNotes:
    def test_notes_non_empty(self):
        result = _base_result()
        assert result.notes != ""


# ---------------------------------------------------------------------------
# Ionization type effects
# ---------------------------------------------------------------------------

class TestIonizationEffects:
    def test_neutral_hepatocyte_equals_liver(self):
        result = _base_result(ionization_type="neutral", c_plasma_mg_L=1.0)
        assert abs(result.c_hepatocyte_mg_L - result.c_liver_mg_L) < 1e-9

    def test_base_hepatocyte_greater_than_liver(self):
        # Bases accumulate in acidic cell compartment (ion trapping)
        result = _base_result(ionization_type="base", pka=9.0, c_plasma_mg_L=1.0)
        assert result.c_hepatocyte_mg_L > result.c_liver_mg_L

    def test_acid_hepatocyte_leq_liver(self):
        # Acids are less ionized in acidic cell -> less trapping
        result = _base_result(ionization_type="acid", pka=5.0, c_plasma_mg_L=1.0)
        assert result.c_hepatocyte_mg_L <= result.c_liver_mg_L


# ---------------------------------------------------------------------------
# Drug name preserved
# ---------------------------------------------------------------------------

class TestDrugNamePreserved:
    def test_drug_name_preserved(self):
        result = _base_result(drug_name="Acetaminophen")
        assert result.drug_name == "Acetaminophen"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _base_result(fu_plasma=0.0)

    def test_fu_plasma_negative_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _base_result(fu_plasma=-0.1)

    def test_fu_plasma_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _base_result(fu_plasma=1.1)

    def test_c_plasma_negative_raises(self):
        with pytest.raises(ValueError, match="c_plasma_mg_L"):
            _base_result(c_plasma_mg_L=-1.0)

    def test_clint_negative_raises(self):
        with pytest.raises(ValueError, match="clint_L_per_h"):
            _base_result(clint_L_per_h=-1.0)

    def test_q_liver_zero_raises(self):
        with pytest.raises(ValueError, match="q_liver_L_per_h"):
            _base_result(q_liver_L_per_h=0.0)

    def test_q_liver_negative_raises(self):
        with pytest.raises(ValueError, match="q_liver_L_per_h"):
            _base_result(q_liver_L_per_h=-10.0)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            _base_result(ionization_type="zwitterion")


# ---------------------------------------------------------------------------
# screen_hepatic_risk
# ---------------------------------------------------------------------------

COMPOUNDS = [
    dict(drug_name="DrugA", logp=1.0, fu_plasma=0.5, pka=7.0, ionization_type="neutral", c_plasma_mg_L=1.0),
    dict(drug_name="DrugB", logp=3.0, fu_plasma=0.1, pka=8.0, ionization_type="base", c_plasma_mg_L=1.0),
    dict(drug_name="DrugC", logp=0.5, fu_plasma=0.8, pka=4.0, ionization_type="acid", c_plasma_mg_L=1.0),
]


class TestScreenHepaticRisk:
    def test_returns_correct_length(self):
        results = screen_hepatic_risk(COMPOUNDS)
        assert len(results) == len(COMPOUNDS)

    def test_sorted_by_liver_to_plasma_descending(self):
        results = screen_hepatic_risk(COMPOUNDS)
        ratios = [r.liver_to_plasma_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_all_results_correct_type(self):
        results = screen_hepatic_risk(COMPOUNDS)
        for r in results:
            assert isinstance(r, HepaticDrugConcentrationResult)
