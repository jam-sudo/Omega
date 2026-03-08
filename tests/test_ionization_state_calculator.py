"""
Tests for Phase 374 — Drug Ionization State Calculator
"""

import math

import pytest

from omega_pbpk.prediction.ionization_state_calculator import (
    IonizationStateResult,
    calculate_ionization,
    screen_ionization_window,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _acid(**kw):
    defaults = dict(drug_name="WeakAcid", drug_type="acid", pka_acid=4.5)
    defaults.update(kw)
    return defaults


def _base(**kw):
    defaults = dict(drug_name="WeakBase", drug_type="base", pka_base=8.5)
    defaults.update(kw)
    return defaults


# ---------------------------------------------------------------------------
# Return type and field existence
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_ionization_state_result_acid(self):
        result = calculate_ionization(**_acid())
        assert isinstance(result, IonizationStateResult)

    def test_returns_ionization_state_result_base(self):
        result = calculate_ionization(**_base())
        assert isinstance(result, IonizationStateResult)

    def test_drug_name_preserved(self):
        result = calculate_ionization(**_acid(drug_name="Aspirin"))
        assert result.drug_name == "Aspirin"

    def test_drug_type_preserved(self):
        result = calculate_ionization(**_acid())
        assert result.drug_type == "acid"

    def test_ph_values_list_length(self):
        result = calculate_ionization(**_acid(), n_points=30)
        assert len(result.ph_values) == 30

    def test_fraction_neutral_list_length(self):
        result = calculate_ionization(**_acid(), n_points=20)
        assert len(result.fraction_neutral) == 20

    def test_fraction_ionized_list_length(self):
        result = calculate_ionization(**_acid(), n_points=20)
        assert len(result.fraction_ionized) == 20


# ---------------------------------------------------------------------------
# Acid ionization physics
# ---------------------------------------------------------------------------


class TestAcidIonization:
    def test_acid_mostly_neutral_below_pka(self):
        """At pH << pKa, acid is mostly neutral (protonated, HA)."""
        result = calculate_ionization(**_acid(pka_acid=7.0), ph_range=(1.0, 5.0), n_points=20)
        # At pH=1, fraction_neutral should be close to 1.0
        assert result.fraction_neutral[0] > 0.99

    def test_acid_mostly_ionized_above_pka(self):
        """At pH >> pKa, acid is mostly ionized (deprotonated, A-)."""
        result = calculate_ionization(**_acid(pka_acid=4.0), ph_range=(7.0, 10.0), n_points=20)
        # At pH=10, fraction_ionized should be close to 1.0
        assert result.fraction_ionized[-1] > 0.99

    def test_acid_50_percent_ionized_at_pka(self):
        """At pH = pKa, acid is 50% ionized."""
        pka = 6.0
        result = calculate_ionization(
            drug_name="TestAcid", drug_type="acid", pka_acid=pka, ph_range=(5.0, 7.0), n_points=50
        )
        # Find closest pH to pKa
        closest_idx = min(
            range(len(result.ph_values)), key=lambda i: abs(result.ph_values[i] - pka)
        )
        assert result.fraction_ionized[closest_idx] == pytest.approx(0.5, abs=0.02)

    def test_acid_anionic_equals_ionized(self):
        """For acid, anionic fraction == ionized fraction."""
        result = calculate_ionization(**_acid())
        for fi, fa in zip(result.fraction_ionized, result.fraction_anionic):
            assert fi == pytest.approx(fa, abs=1e-10)

    def test_acid_no_cationic_fraction(self):
        """For acid, cationic fraction is always 0."""
        result = calculate_ionization(**_acid())
        assert all(fc == pytest.approx(0.0) for fc in result.fraction_cationic)

    def test_acid_ph50_equals_pka(self):
        result = calculate_ionization(**_acid(pka_acid=5.5))
        assert result.ph_50_ionized == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# Base ionization physics
# ---------------------------------------------------------------------------


class TestBaseIonization:
    def test_base_mostly_ionized_below_pka(self):
        """At pH << pKa, base is mostly ionized (protonated, BH+)."""
        result = calculate_ionization(**_base(pka_base=8.0), ph_range=(1.0, 5.0), n_points=20)
        assert result.fraction_ionized[-1] > 0.99

    def test_base_mostly_neutral_above_pka(self):
        """At pH >> pKa, base is mostly neutral (deprotonated, B)."""
        result = calculate_ionization(**_base(pka_base=5.0), ph_range=(8.0, 10.0), n_points=20)
        assert result.fraction_neutral[-1] > 0.99

    def test_base_cationic_equals_ionized(self):
        """For base, cationic fraction == ionized fraction."""
        result = calculate_ionization(**_base())
        for fi, fc in zip(result.fraction_ionized, result.fraction_cationic):
            assert fi == pytest.approx(fc, abs=1e-10)

    def test_base_no_anionic_fraction(self):
        """For base, anionic fraction is always 0."""
        result = calculate_ionization(**_base())
        assert all(fa == pytest.approx(0.0) for fa in result.fraction_anionic)

    def test_base_ph50_equals_pka(self):
        result = calculate_ionization(**_base(pka_base=9.2))
        assert result.ph_50_ionized == pytest.approx(9.2)


# ---------------------------------------------------------------------------
# Neutral drug
# ---------------------------------------------------------------------------


class TestNeutralDrug:
    def test_neutral_always_neutral(self):
        result = calculate_ionization(drug_name="Neutral", drug_type="neutral")
        assert all(fn == pytest.approx(1.0) for fn in result.fraction_neutral)

    def test_neutral_no_ionization(self):
        result = calculate_ionization(drug_name="Neutral", drug_type="neutral")
        assert all(fi == pytest.approx(0.0) for fi in result.fraction_ionized)

    def test_neutral_ph50_is_nan(self):
        result = calculate_ionization(drug_name="Neutral", drug_type="neutral")
        assert math.isnan(result.ph_50_ionized)


# ---------------------------------------------------------------------------
# Zwitterion
# ---------------------------------------------------------------------------


class TestZwitterion:
    def test_zwitterion_has_both_cationic_and_anionic(self):
        result = calculate_ionization(
            drug_name="Zwitterion",
            drug_type="zwitterion",
            pka_acid=4.0,
            pka_base=9.0,
            ph_range=(1.0, 12.0),
            n_points=50,
        )
        # Should have non-zero cationic at low pH and non-zero anionic at high pH
        assert result.fraction_cationic[0] > 0.1
        assert result.fraction_anionic[-1] > 0.1

    def test_zwitterion_neutral_fraction_nonneg(self):
        result = calculate_ionization(
            drug_name="Zwitterion",
            drug_type="zwitterion",
            pka_acid=4.0,
            pka_base=9.0,
        )
        assert all(fn >= 0.0 for fn in result.fraction_neutral)

    def test_zwitterion_ph50_is_min_pka(self):
        result = calculate_ionization(
            drug_name="Zwitterion",
            drug_type="zwitterion",
            pka_acid=4.0,
            pka_base=9.0,
        )
        assert result.ph_50_ionized == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# optimal_absorption_ph and gi_ionization_fraction
# ---------------------------------------------------------------------------


class TestDerivedMetrics:
    def test_optimal_absorption_ph_in_range(self):
        result = calculate_ionization(**_acid(), ph_range=(1.0, 10.0), n_points=50)
        ph_min, ph_max = 1.0, 10.0
        assert ph_min <= result.optimal_absorption_ph <= ph_max

    def test_gi_ionization_fraction_in_range(self):
        result = calculate_ionization(**_acid())
        assert 0.0 <= result.gi_ionization_fraction <= 1.0

    def test_acid_gi_ionization_at_gi_ph(self):
        """Manually verify GI ionization for acid with known pKa."""
        pka = 4.5
        gi_ph = 6.5
        expected = 1.0 / (1.0 + 10.0 ** (pka - gi_ph))
        result = calculate_ionization(drug_name="Acid", drug_type="acid", pka_acid=pka)
        assert result.gi_ionization_fraction == pytest.approx(expected, rel=0.001)

    def test_acid_low_pka_optimal_absorption_at_low_ph(self):
        """For a very strong acid (pKa=1), neutral form dominates at low pH."""
        result = calculate_ionization(
            drug_name="StrongAcid",
            drug_type="acid",
            pka_acid=1.0,
            ph_range=(1.0, 10.0),
            n_points=100,
        )
        # At pH 1, acid with pKa=1 is 50% neutral — optimal should be near low end
        assert result.optimal_absorption_ph < 5.0


# ---------------------------------------------------------------------------
# screen_ionization_window
# ---------------------------------------------------------------------------


class TestScreenIonizationWindow:
    def test_returns_dict(self):
        result = screen_ionization_window(
            drug_name="Drug",
            drug_type="acid",
            pka_acid=4.5,
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = screen_ionization_window(
            drug_name="Drug",
            drug_type="acid",
            pka_acid=4.5,
        )
        expected_keys = {
            "mean_neutral_fraction",
            "min_neutral_fraction",
            "max_neutral_fraction",
            "ph_values",
            "neutral_fractions",
        }
        assert expected_keys.issubset(result.keys())

    def test_mean_neutral_fraction_in_range(self):
        result = screen_ionization_window(
            drug_name="Drug",
            drug_type="acid",
            pka_acid=4.5,
        )
        assert 0.0 <= result["mean_neutral_fraction"] <= 1.0

    def test_min_leq_mean_leq_max(self):
        result = screen_ionization_window(
            drug_name="Drug",
            drug_type="base",
            pka_base=7.5,
        )
        assert result["min_neutral_fraction"] <= result["mean_neutral_fraction"]
        assert result["mean_neutral_fraction"] <= result["max_neutral_fraction"]

    def test_ph_values_list_length_50(self):
        result = screen_ionization_window(
            drug_name="Drug",
            drug_type="acid",
            pka_acid=4.5,
        )
        assert len(result["ph_values"]) == 50

    def test_neutral_fractions_list_length_50(self):
        result = screen_ionization_window(
            drug_name="Drug",
            drug_type="acid",
            pka_acid=4.5,
        )
        assert len(result["neutral_fractions"]) == 50


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            calculate_ionization(drug_name="Drug", drug_type="ampholyte", pka_acid=5.0)

    def test_acid_without_pka_acid_raises(self):
        with pytest.raises(ValueError, match="pka_acid"):
            calculate_ionization(drug_name="Drug", drug_type="acid")

    def test_base_without_pka_base_raises(self):
        with pytest.raises(ValueError, match="pka_base"):
            calculate_ionization(drug_name="Drug", drug_type="base")

    def test_zwitterion_missing_pka_acid_raises(self):
        with pytest.raises(ValueError):
            calculate_ionization(drug_name="Drug", drug_type="zwitterion", pka_base=9.0)

    def test_zwitterion_missing_pka_base_raises(self):
        with pytest.raises(ValueError):
            calculate_ionization(drug_name="Drug", drug_type="zwitterion", pka_acid=4.0)
