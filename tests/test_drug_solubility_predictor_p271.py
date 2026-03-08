"""Tests for Phase 271: Drug Solubility Predictor (GSE + Henderson-Hasselbalch)."""

import pytest

from omega_pbpk.prediction.drug_solubility_predictor_p271 import (
    SolubilityPredictionResult,
    predict_solubility,
    screen_solubility_conditions,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _ibuprofen_like():
    """Ibuprofen-like acid: MW=206, logP=3.97, MP=75C, pKa=4.4."""
    return dict(
        drug_name="ibuprofen", mw=206.0, logp=3.97, mp_C=75.0, pka=4.4, ionization_type="acid"
    )


def _metformin_like():
    """Metformin-like base: MW=129, logP=-1.43, MP=232C, pKa=11.5."""
    return dict(
        drug_name="metformin", mw=129.0, logp=-1.43, mp_C=232.0, pka=11.5, ionization_type="base"
    )


def _neutral_drug():
    """Neutral drug: MW=300, logP=2.0, MP=150C."""
    return dict(drug_name="neutral_x", mw=300.0, logp=2.0, mp_C=150.0)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        result = predict_solubility("aspirin", mw=180.0, logp=1.2, mp_C=135.0)
        assert isinstance(result, SolubilityPredictionResult)

    def test_drug_name_preserved(self):
        result = predict_solubility("aspirin", mw=180.0, logp=1.2, mp_C=135.0)
        assert result.drug_name == "aspirin"

    def test_parameters_stored(self):
        result = predict_solubility("aspirin", mw=180.0, logp=1.2, mp_C=135.0)
        assert result.mw == pytest.approx(180.0)
        assert result.logp == pytest.approx(1.2)
        assert result.mp_C == pytest.approx(135.0)


# ---------------------------------------------------------------------------
# Intrinsic solubility tests
# ---------------------------------------------------------------------------


class TestIntrinsicSolubility:
    def test_sw_intrinsic_positive(self):
        result = predict_solubility(**_neutral_drug())
        assert result.sw_intrinsic_mg_mL > 0.0

    def test_higher_logp_lower_solubility(self):
        """Higher logP should give lower solubility (GSE)."""
        r_low = predict_solubility("low_logp", mw=200.0, logp=1.0, mp_C=100.0)
        r_high = predict_solubility("high_logp", mw=200.0, logp=4.0, mp_C=100.0)
        assert r_low.sw_intrinsic_mg_mL > r_high.sw_intrinsic_mg_mL

    def test_higher_mp_lower_solubility(self):
        """Higher melting point should give lower solubility (GSE)."""
        r_low = predict_solubility("low_mp", mw=200.0, logp=2.0, mp_C=50.0)
        r_high = predict_solubility("high_mp", mw=200.0, logp=2.0, mp_C=200.0)
        assert r_low.sw_intrinsic_mg_mL > r_high.sw_intrinsic_mg_mL

    def test_all_solubility_values_positive(self):
        result = predict_solubility(**_ibuprofen_like())
        assert result.sw_gastric_mg_mL > 0.0
        assert result.sw_intestinal_mg_mL > 0.0
        assert result.sw_colon_mg_mL > 0.0


# ---------------------------------------------------------------------------
# pH-dependent solubility tests
# ---------------------------------------------------------------------------


class TestPHDependentSolubility:
    def test_acid_higher_solubility_intestinal_vs_gastric(self):
        """Acid (pKa=4) should be more soluble at intestinal pH 6.8 than gastric pH 1.2."""
        result = predict_solubility(
            "weak_acid", mw=200.0, logp=2.0, mp_C=100.0, pka=4.0, ionization_type="acid"
        )
        assert result.sw_intestinal_mg_mL > result.sw_gastric_mg_mL

    def test_base_higher_solubility_gastric_vs_intestinal(self):
        """Base (pKa=9) should be more soluble at gastric pH 1.2 than intestinal pH 6.8."""
        result = predict_solubility(
            "weak_base", mw=200.0, logp=2.0, mp_C=100.0, pka=9.0, ionization_type="base"
        )
        assert result.sw_gastric_mg_mL > result.sw_intestinal_mg_mL

    def test_neutral_solubility_same_across_ph(self):
        """Neutral drug should have same solubility regardless of pH."""
        result = predict_solubility(**_neutral_drug())
        assert result.sw_gastric_mg_mL == pytest.approx(result.sw_intestinal_mg_mL)
        assert result.sw_intestinal_mg_mL == pytest.approx(result.sw_colon_mg_mL)

    def test_none_pka_treated_as_neutral(self):
        """None pKa should behave identically to neutral ionization."""
        r_none = predict_solubility(
            "x", mw=200.0, logp=2.0, mp_C=100.0, pka=None, ionization_type="acid"
        )
        r_neut = predict_solubility("x", mw=200.0, logp=2.0, mp_C=100.0, ionization_type="neutral")
        assert r_none.sw_intestinal_mg_mL == pytest.approx(r_neut.sw_intestinal_mg_mL)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassification:
    def test_solubility_class_valid_values(self):
        valid = {"highly_soluble", "moderately_soluble", "poorly_soluble"}
        result = predict_solubility("x", mw=200.0, logp=2.0, mp_C=100.0)
        assert result.solubility_class in valid

    def test_highly_soluble_drug(self):
        """Low logP, low MP -> highly soluble."""
        result = predict_solubility("soluble", mw=150.0, logp=-1.0, mp_C=50.0)
        assert result.solubility_class == "highly_soluble"

    def test_poorly_soluble_drug(self):
        """High logP, high MP -> poorly soluble."""
        result = predict_solubility("insoluble", mw=500.0, logp=6.0, mp_C=200.0)
        assert result.solubility_class == "poorly_soluble"

    def test_bcs_dose_number_positive(self):
        result = predict_solubility("x", mw=200.0, logp=2.0, mp_C=100.0)
        assert result.bcs_dose_number > 0.0

    def test_bcs_class_valid_values(self):
        result = predict_solubility("x", mw=200.0, logp=2.0, mp_C=100.0)
        assert result.bcs_class in {"I/III", "II/IV"}

    def test_bcs_do_high_solubility_class_I_III(self):
        """High solubility -> Do < 1 -> BCS I/III."""
        result = predict_solubility("x", mw=150.0, logp=-1.0, mp_C=50.0, dose_mg=10.0)
        assert result.bcs_class == "I/III"

    def test_bcs_do_low_solubility_class_II_IV(self):
        """Low solubility + high dose -> Do >= 1 -> BCS II/IV."""
        result = predict_solubility("x", mw=500.0, logp=6.0, mp_C=200.0, dose_mg=500.0)
        assert result.bcs_class == "II/IV"


# ---------------------------------------------------------------------------
# Screen function tests
# ---------------------------------------------------------------------------


class TestScreenFunction:
    def test_returns_dict(self):
        result = screen_solubility_conditions("x", mw=200.0, logp=2.0, mp_C=100.0)
        assert isinstance(result, dict)

    def test_correct_keys(self):
        result = screen_solubility_conditions("x", mw=200.0, logp=2.0, mp_C=100.0)
        assert set(result.keys()) == {"gastric", "intestinal", "colon", "plasma"}

    def test_all_values_positive(self):
        result = screen_solubility_conditions("x", mw=200.0, logp=2.0, mp_C=100.0)
        for key, val in result.items():
            assert val > 0.0, f"Expected positive solubility for {key}, got {val}"


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_mw_zero(self):
        with pytest.raises(ValueError, match="mw must be > 0"):
            predict_solubility("x", mw=0.0, logp=2.0, mp_C=100.0)

    def test_invalid_mw_negative(self):
        with pytest.raises(ValueError):
            predict_solubility("x", mw=-100.0, logp=2.0, mp_C=100.0)

    def test_invalid_logp_too_high(self):
        with pytest.raises(ValueError, match="logp must be in"):
            predict_solubility("x", mw=200.0, logp=11.0, mp_C=100.0)

    def test_invalid_logp_too_low(self):
        with pytest.raises(ValueError, match="logp must be in"):
            predict_solubility("x", mw=200.0, logp=-6.0, mp_C=100.0)

    def test_invalid_mp_too_high(self):
        with pytest.raises(ValueError, match="mp_C must be in"):
            predict_solubility("x", mw=200.0, logp=2.0, mp_C=350.0)

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_solubility("x", mw=200.0, logp=2.0, mp_C=100.0, ionization_type="zwitterion")
