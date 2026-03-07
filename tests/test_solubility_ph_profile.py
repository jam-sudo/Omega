"""Tests for biopharmaceutics/solubility_ph_profile.py — Phase 551."""

import pytest

from omega_pbpk.biopharmaceutics.solubility_ph_profile import (
    dose_number,
    minimum_volume_for_dissolution,
    ph_solubility_profile,
    solubility_at_ph,
    solubility_bcs_classification,
)

# ---------------------------------------------------------------------------
# solubility_at_ph — weak_acid
# ---------------------------------------------------------------------------


class TestSolubilityAtPhWeakAcid:
    def test_high_ph_higher_solubility(self):
        """Weak acid: higher pH → higher solubility."""
        s_low = solubility_at_ph(0.1, pka=4.0, drug_type="weak_acid", ph=2.0)
        s_high = solubility_at_ph(0.1, pka=4.0, drug_type="weak_acid", ph=7.0)
        assert s_high > s_low

    def test_at_pka_doubles(self):
        """At pH = pKa, S = 2 * S0 for weak acid."""
        s = solubility_at_ph(1.0, pka=5.0, drug_type="weak_acid", ph=5.0)
        assert abs(s - 2.0) < 1e-10

    def test_formula_correctness(self):
        """S(pH) = S0 * (1 + 10^(pH - pKa))."""
        s0, pka, ph = 0.5, 4.0, 6.0
        expected = s0 * (1.0 + 10.0 ** (ph - pka))
        result = solubility_at_ph(s0, pka=pka, drug_type="weak_acid", ph=ph)
        assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# solubility_at_ph — weak_base
# ---------------------------------------------------------------------------


class TestSolubilityAtPhWeakBase:
    def test_low_ph_higher_solubility(self):
        """Weak base: lower pH → higher solubility."""
        s_low = solubility_at_ph(0.1, pka=8.0, drug_type="weak_base", ph=2.0)
        s_high = solubility_at_ph(0.1, pka=8.0, drug_type="weak_base", ph=7.0)
        assert s_low > s_high

    def test_at_pka_doubles(self):
        """At pH = pKa, S = 2 * S0 for weak base."""
        s = solubility_at_ph(2.0, pka=6.0, drug_type="weak_base", ph=6.0)
        assert abs(s - 4.0) < 1e-10

    def test_formula_correctness(self):
        """S(pH) = S0 * (1 + 10^(pKa - pH))."""
        s0, pka, ph = 0.3, 7.0, 4.0
        expected = s0 * (1.0 + 10.0 ** (pka - ph))
        result = solubility_at_ph(s0, pka=pka, drug_type="weak_base", ph=ph)
        assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# solubility_at_ph — neutral
# ---------------------------------------------------------------------------


class TestSolubilityAtPhNeutral:
    def test_constant_across_ph(self):
        """Neutral drug: solubility is constant regardless of pH."""
        s0 = 1.5
        phs = [1.0, 4.0, 7.4, 8.0]
        results = [solubility_at_ph(s0, pka=7.0, drug_type="neutral", ph=p) for p in phs]
        assert all(abs(r - s0) < 1e-12 for r in results)


# ---------------------------------------------------------------------------
# solubility_at_ph — amphoteric
# ---------------------------------------------------------------------------


class TestSolubilityAtPhAmphoteric:
    def test_u_shaped_profile(self):
        """Amphoteric drug: U-shaped profile (minimum between the two pKas)."""
        s0, pka_acid, pka_base = 0.1, 7.0, 4.0
        ph_values = [1.0, 4.0, 5.5, 7.0, 8.0]
        sols = [
            solubility_at_ph(s0, pka=pka_acid, drug_type="amphoteric", ph=p, pka2=pka_base)
            for p in ph_values
        ]
        # Ends should be higher than the middle
        assert sols[0] > sols[2]
        assert sols[-1] > sols[2]

    def test_requires_pka2(self):
        """Amphoteric without pka2 raises ValueError."""
        with pytest.raises(ValueError, match="pka2"):
            solubility_at_ph(1.0, pka=7.0, drug_type="amphoteric", ph=5.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestSolubilityInputValidation:
    def test_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            solubility_at_ph(1.0, pka=5.0, drug_type="zwitterion", ph=7.0)

    def test_negative_s0(self):
        with pytest.raises(ValueError, match="s0_mg_mL"):
            solubility_at_ph(-1.0, pka=5.0, drug_type="weak_acid", ph=7.0)


# ---------------------------------------------------------------------------
# ph_solubility_profile
# ---------------------------------------------------------------------------


class TestPhSolubilityProfile:
    def test_returns_correct_keys(self):
        result = ph_solubility_profile(1.0, pka=5.0, drug_type="weak_acid")
        assert "ph_values" in result
        assert "solubility_mg_mL" in result
        assert "ph_min_solubility" in result
        assert "s_min_mg_mL" in result
        assert "s_max_mg_mL" in result

    def test_correct_number_of_points(self):
        result = ph_solubility_profile(1.0, pka=5.0, drug_type="weak_acid", n_points=71)
        assert len(result["ph_values"]) == 71
        assert len(result["solubility_mg_mL"]) == 71

    def test_weak_acid_min_at_low_ph(self):
        """Weak acid minimum solubility occurs at low pH."""
        result = ph_solubility_profile(
            1.0, pka=5.0, drug_type="weak_acid", ph_range=(1.0, 8.0), n_points=71
        )
        assert result["ph_min_solubility"] < 4.0

    def test_weak_base_min_at_high_ph(self):
        """Weak base minimum solubility occurs at high pH."""
        result = ph_solubility_profile(
            1.0, pka=5.0, drug_type="weak_base", ph_range=(1.0, 8.0), n_points=71
        )
        assert result["ph_min_solubility"] > 5.0

    def test_s_min_leq_s_max(self):
        result = ph_solubility_profile(0.5, pka=6.0, drug_type="weak_acid")
        assert result["s_min_mg_mL"] <= result["s_max_mg_mL"]

    def test_invalid_ph_range(self):
        with pytest.raises(ValueError):
            ph_solubility_profile(1.0, pka=5.0, drug_type="weak_acid", ph_range=(8.0, 1.0))

    def test_amphoteric_needs_pka2(self):
        with pytest.raises(ValueError):
            ph_solubility_profile(1.0, pka=7.0, drug_type="amphoteric")


# ---------------------------------------------------------------------------
# dose_number
# ---------------------------------------------------------------------------


class TestDoseNumber:
    def test_high_dose_low_solubility_class(self):
        """High dose relative to solubility → Dn > 1 → low_solubility."""
        result = dose_number(dose_mg=1000.0, volume_mL=250.0, s0_mg_mL=1.0, drug_type="neutral")
        assert result["dose_number"] > 1.0
        assert result["bcs_solubility_class"] == "low_solubility"

    def test_low_dose_high_solubility_class(self):
        """Low dose relative to high solubility → Dn ≤ 1 → high_solubility."""
        result = dose_number(dose_mg=1.0, volume_mL=250.0, s0_mg_mL=10.0, drug_type="neutral")
        assert result["dose_number"] <= 1.0
        assert result["bcs_solubility_class"] == "high_solubility"

    def test_with_pka_adjustment(self):
        """Dose number computed with pH-adjusted solubility."""
        result = dose_number(
            dose_mg=100.0,
            volume_mL=250.0,
            s0_mg_mL=0.01,
            ph=7.4,
            pka=4.0,
            drug_type="weak_acid",
        )
        assert "dose_number" in result
        assert result["solubility_at_ph_mg_mL"] > 0.01  # pH-adjusted > intrinsic

    def test_missing_s0_raises(self):
        with pytest.raises(ValueError):
            dose_number(dose_mg=100.0, volume_mL=250.0)


# ---------------------------------------------------------------------------
# minimum_volume_for_dissolution
# ---------------------------------------------------------------------------


class TestMinimumVolumeForDissolution:
    def test_proportional_to_dose(self):
        """Double the dose → double the minimum volume."""
        v1 = minimum_volume_for_dissolution(100.0, pka=5.0, drug_type="weak_acid")
        v2 = minimum_volume_for_dissolution(200.0, pka=5.0, drug_type="weak_acid")
        assert abs(v2 / v1 - 2.0) < 1e-10

    def test_higher_solubility_lower_volume(self):
        """Higher s0 → lower minimum volume needed."""
        v_low = minimum_volume_for_dissolution(100.0, pka=5.0, drug_type="weak_acid", s0_mg_mL=0.1)
        v_high = minimum_volume_for_dissolution(
            100.0, pka=5.0, drug_type="weak_acid", s0_mg_mL=10.0
        )
        assert v_low > v_high

    def test_result_is_positive(self):
        v = minimum_volume_for_dissolution(50.0, pka=6.0, drug_type="weak_base", ph=5.0)
        assert v > 0


# ---------------------------------------------------------------------------
# solubility_bcs_classification
# ---------------------------------------------------------------------------


class TestSolubilityBcsClassification:
    def test_class_i(self):
        """High solubility + high permeability → Class I."""
        result = solubility_bcs_classification(s_mg_mL_ph65=1.0, peff_cm_s=5e-4, dose_mg=100.0)
        assert result["bcs_class"] == "I"
        assert result["is_high_solubility"]
        assert result["is_high_permeability"]

    def test_class_ii(self):
        """Low solubility + high permeability → Class II."""
        result = solubility_bcs_classification(s_mg_mL_ph65=0.001, peff_cm_s=5e-4, dose_mg=500.0)
        assert result["bcs_class"] == "II"

    def test_class_iii(self):
        """High solubility + low permeability → Class III."""
        result = solubility_bcs_classification(s_mg_mL_ph65=5.0, peff_cm_s=1e-5, dose_mg=50.0)
        assert result["bcs_class"] == "III"

    def test_class_iv(self):
        """Low solubility + low permeability → Class IV."""
        result = solubility_bcs_classification(s_mg_mL_ph65=0.001, peff_cm_s=1e-5, dose_mg=500.0)
        assert result["bcs_class"] == "IV"
        assert not result["is_high_solubility"]
        assert not result["is_high_permeability"]

    def test_dose_number_in_result(self):
        result = solubility_bcs_classification(s_mg_mL_ph65=1.0, peff_cm_s=3e-4, dose_mg=100.0)
        assert "dose_number" in result
        expected_dn = 100.0 / (250.0 * 1.0)
        assert abs(result["dose_number"] - expected_dn) < 1e-10
