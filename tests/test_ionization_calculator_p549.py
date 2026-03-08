"""Tests for ionization state calculator — Phase 549."""

import pytest

from omega_pbpk.prediction.ionization_calculator_p549 import (
    IonizationResult,
    calculate_ionization,
    compare_ionization_profiles,
    ionization_at_ph,
)


class TestIonizationAtPh:
    """Tests for the ionization_at_ph helper."""

    def test_acid_at_pka_gives_50_pct(self):
        frac = ionization_at_ph(pKa=4.5, drug_type="acid", ph=4.5)
        assert abs(frac - 0.5) < 1e-9

    def test_base_at_pka_gives_50_pct(self):
        frac = ionization_at_ph(pKa=8.0, drug_type="base", ph=8.0)
        assert abs(frac - 0.5) < 1e-9

    def test_acid_low_ph_mostly_unionized(self):
        frac = ionization_at_ph(pKa=4.5, drug_type="acid", ph=1.0)
        assert frac < 0.01

    def test_acid_high_ph_mostly_ionized(self):
        frac = ionization_at_ph(pKa=4.5, drug_type="acid", ph=8.0)
        assert frac > 0.99

    def test_base_high_ph_mostly_unionized(self):
        frac = ionization_at_ph(pKa=8.0, drug_type="base", ph=12.0)
        assert frac < 0.01

    def test_base_low_ph_mostly_ionized(self):
        frac = ionization_at_ph(pKa=8.0, drug_type="base", ph=4.0)
        assert frac > 0.99

    def test_amphoteric_uses_base_formula(self):
        frac_ampho = ionization_at_ph(pKa=6.0, drug_type="amphoteric", ph=7.0)
        frac_base = ionization_at_ph(pKa=6.0, drug_type="base", ph=7.0)
        assert abs(frac_ampho - frac_base) < 1e-12

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="Invalid drug_type"):
            ionization_at_ph(pKa=4.5, drug_type="cation", ph=7.0)


class TestCalculateIonization:
    """Tests for the main calculate_ionization function."""

    def test_return_type(self):
        result = calculate_ionization("Aspirin", 3.5, "acid")
        assert isinstance(result, IonizationResult)

    def test_fields_are_tuples(self):
        result = calculate_ionization("Aspirin", 3.5, "acid")
        assert isinstance(result.ph_values, tuple)
        assert isinstance(result.ionized_fractions, tuple)
        assert isinstance(result.unionized_fractions, tuple)

    def test_default_n_points(self):
        result = calculate_ionization("Aspirin", 3.5, "acid")
        assert len(result.ph_values) == 19
        assert len(result.ionized_fractions) == 19
        assert len(result.unionized_fractions) == 19

    def test_ph_range_endpoints(self):
        result = calculate_ionization("Drug", 5.0, "acid", ph_range=(2.0, 8.0), n_points=7)
        assert abs(result.ph_values[0] - 2.0) < 1e-9
        assert abs(result.ph_values[-1] - 8.0) < 1e-9

    def test_ionized_plus_unionized_equals_one(self):
        result = calculate_ionization("Drug", 5.0, "base")
        for ion, union in zip(result.ionized_fractions, result.unionized_fractions):
            assert abs(ion + union - 1.0) < 1e-12

    def test_ph_50_pct_ionized_equals_pka(self):
        result = calculate_ionization("Drug", 6.2, "acid")
        assert abs(result.ph_50_pct_ionized - 6.2) < 1e-12

    def test_dominant_form_acid_low_pka(self):
        # Acid with pKa=3.5: at pH 7.4, mostly ionized
        result = calculate_ionization("Aspirin", 3.5, "acid")
        assert result.dominant_form_at_physiological_ph == "ionized"

    def test_dominant_form_acid_high_pka(self):
        # Acid with pKa=9.0: at pH 7.4, mostly unionized
        result = calculate_ionization("Phenol", 9.0, "acid")
        assert result.dominant_form_at_physiological_ph == "unionized"

    def test_dominant_form_base_high_pka(self):
        # Base with pKa=9.0: at pH 7.4, mostly ionized
        result = calculate_ionization("Amphetamine", 9.0, "base")
        assert result.dominant_form_at_physiological_ph == "ionized"

    def test_dominant_form_base_low_pka(self):
        # Base with pKa=5.0: at pH 7.4, mostly unionized
        result = calculate_ionization("Caffeine-like", 5.0, "base")
        assert result.dominant_form_at_physiological_ph == "unionized"

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="Invalid drug_type"):
            calculate_ionization("Drug", 5.0, "zwitterion")

    def test_n_points_less_than_2_raises(self):
        with pytest.raises(ValueError):
            calculate_ionization("Drug", 5.0, "acid", n_points=1)


class TestCompareIonizationProfiles:
    """Tests for compare_ionization_profiles."""

    def test_sorted_ascending_by_ph_50(self):
        compounds = [
            {"drug_name": "B", "pKa": 8.0, "drug_type": "acid"},
            {"drug_name": "A", "pKa": 3.5, "drug_type": "acid"},
            {"drug_name": "C", "pKa": 5.5, "drug_type": "base"},
        ]
        results = compare_ionization_profiles(compounds)
        pka_values = [r.ph_50_pct_ionized for r in results]
        assert pka_values == sorted(pka_values)

    def test_returns_list_of_ionization_results(self):
        compounds = [
            {"drug_name": "A", "pKa": 4.0, "drug_type": "acid"},
            {"drug_name": "B", "pKa": 9.0, "drug_type": "base"},
        ]
        results = compare_ionization_profiles(compounds)
        assert isinstance(results, list)
        assert all(isinstance(r, IonizationResult) for r in results)

    def test_empty_list(self):
        assert compare_ionization_profiles([]) == []
