"""Tests for pH-solubility profile prediction."""

import pytest

from omega_pbpk.prediction.ph_solubility import (
    pHSolubilityResult,
    predict_ph_solubility,
    screen_ph_solubility,
)


class TestInputValidation:
    def test_negative_solubility(self):
        with pytest.raises(ValueError, match="intrinsic_solubility"):
            predict_ph_solubility("D", -1.0, "acid")

    def test_zero_solubility(self):
        with pytest.raises(ValueError, match="intrinsic_solubility"):
            predict_ph_solubility("D", 0.0, "acid")

    def test_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_ph_solubility("D", 1.0, "zwitterion_invalid")


class TestNeutralDrug:
    def test_constant_solubility(self):
        r = predict_ph_solubility("neutral_drug", 1.0, "neutral")
        # All solubility values should equal S0
        assert all(abs(s - 1.0) < 1e-6 for s in r.solubility_mg_mL)

    def test_drug_type_stored(self):
        r = predict_ph_solubility("neutral_drug", 1.0, "neutral")
        assert r.drug_type == "neutral"


class TestAcidDrug:
    def setup_method(self):
        # Ibuprofen-like: S0=0.021, pKa=4.4
        self.r = predict_ph_solubility("ibuprofen", 0.021, "acid", pka1=4.4)

    def test_result_type(self):
        assert isinstance(self.r, pHSolubilityResult)

    def test_higher_ph_higher_solubility(self):
        """Acids ionize more at high pH → higher solubility."""
        assert self.r.solubility_at_ph_7_4 > self.r.solubility_at_ph_1_2

    def test_at_pka_double_s0(self):
        """At pH=pKa: S = S0 * (1 + 10^0) = 2*S0."""
        r_at_pka = predict_ph_solubility(
            "test", 1.0, "acid", pka1=4.0, ph_range=(3.9, 4.1), n_points=3
        )
        # Middle point closest to pKa=4.0
        mid = r_at_pka.solubility_mg_mL[1]
        assert mid == pytest.approx(2.0, rel=0.01)

    def test_low_ph_near_s0(self):
        """At very low pH (pH << pKa), S ≈ S0."""
        r = predict_ph_solubility("test", 1.0, "acid", pka1=7.0, ph_range=(1.0, 2.0), n_points=5)
        # pH=1, pKa=7: S = 1*(1+10^(1-7)) ≈ 1
        assert r.solubility_mg_mL[0] == pytest.approx(1.0, rel=0.01)

    def test_hh_formula(self):
        """S(pH) = S0 * (1 + 10^(pH-pKa)) for acid."""
        S0 = 0.021
        pka = 4.4
        ph = 7.4
        expected = S0 * (1 + 10 ** (ph - pka))
        assert self.r.solubility_at_ph_7_4 == pytest.approx(expected, rel=1e-3)

    def test_max_at_high_ph(self):
        """Max solubility should be at highest pH in range."""
        assert self.r.ph_max_solubility == pytest.approx(9.0, abs=0.2)

    def test_pka_stored(self):
        assert 4.4 in self.r.pka_values


class TestBaseDrug:
    def setup_method(self):
        # Propranolol-like: pKa=9.5 (base)
        self.r = predict_ph_solubility("propranolol", 0.05, "base", pka1=9.5)

    def test_lower_ph_higher_solubility(self):
        """Bases ionize more at low pH → higher solubility."""
        assert self.r.solubility_at_ph_1_2 > self.r.solubility_at_ph_7_4

    def test_hh_formula_base(self):
        """S(pH) = S0 * (1 + 10^(pKa-pH)) for base."""
        S0 = 0.05
        pka = 9.5
        ph = 7.4
        expected = S0 * (1 + 10 ** (pka - ph))
        assert self.r.solubility_at_ph_7_4 == pytest.approx(expected, rel=1e-3)

    def test_max_at_low_ph(self):
        assert self.r.ph_max_solubility == pytest.approx(1.0, abs=0.2)


class TestAmphotericDrug:
    def setup_method(self):
        # Ciprofloxacin-like: pKa_acid=6.1, pKa_base=8.7
        self.r = predict_ph_solubility("cipro", 0.03, "amphoteric", pka1=6.1, pka2=8.7)

    def test_higher_solubility_at_extremes(self):
        """Amphoteric: high solubility at both low and high pH."""
        assert self.r.solubility_at_ph_1_2 > self.r.min_solubility_mg_mL
        assert self.r.solubility_at_ph_7_4 >= self.r.min_solubility_mg_mL

    def test_two_pka_stored(self):
        assert len(self.r.pka_values) == 2

    def test_min_between_pkas(self):
        """Minimum solubility should be between the two pKa values."""
        pka_acid, pka_base = 6.1, 8.7
        assert pka_acid - 1 <= self.r.ph_min_solubility <= pka_base + 1


class TestOutputFormat:
    def test_ph_range_length(self):
        r = predict_ph_solubility("D", 1.0, "neutral", n_points=50)
        assert len(r.ph_values) == 50
        assert len(r.solubility_mg_mL) == 50

    def test_all_solubilities_positive(self):
        r = predict_ph_solubility("D", 1.0, "acid", pka1=4.0)
        assert all(s > 0 for s in r.solubility_mg_mL)

    def test_intrinsic_solubility_stored(self):
        r = predict_ph_solubility("D", 0.5, "neutral")
        assert r.intrinsic_solubility_mg_mL == 0.5

    def test_specific_ph_fields(self):
        r = predict_ph_solubility("D", 1.0, "neutral")
        assert r.solubility_at_ph_7_4 > 0
        assert r.solubility_at_ph_6_8 > 0
        assert r.solubility_at_ph_1_2 > 0


class TestScreening:
    def test_screen_multiple(self):
        compounds = [
            {
                "drug_name": "acid_drug",
                "intrinsic_solubility_mg_mL": 0.1,
                "drug_type": "acid",
                "pka1": 4.0,
            },
            {
                "drug_name": "base_drug",
                "intrinsic_solubility_mg_mL": 0.5,
                "drug_type": "base",
                "pka1": 9.0,
            },
        ]
        results = screen_ph_solubility(compounds)
        assert len(results) == 2
        assert results[0].drug_name == "acid_drug"
        assert results[1].drug_name == "base_drug"
