"""Tests for first-in-human dose prediction."""

import pytest

from omega_pbpk.clinical.fih_dose import (
    FIHDoseResult,
    compare_species,
    most_conservative_mrsd,
    predict_fih_dose,
)


class TestInputValidation:
    def test_noael_zero(self):
        with pytest.raises(ValueError, match="noael"):
            predict_fih_dose("D", 0, "rat")

    def test_noael_negative(self):
        with pytest.raises(ValueError, match="noael"):
            predict_fih_dose("D", -1, "rat")

    def test_unknown_species(self):
        with pytest.raises(ValueError, match="Unknown species"):
            predict_fih_dose("D", 100, "elephant")


class TestBSAMethod:
    def setup_method(self):
        self.r = predict_fih_dose("drug_x", noael_mg_kg=100, species="rat")

    def test_result_type(self):
        assert isinstance(self.r, FIHDoseResult)

    def test_species_stored(self):
        assert self.r.species == "rat"

    def test_noael_stored(self):
        assert self.r.noael_mg_kg == 100.0

    def test_safety_factor_rat(self):
        """FDA default safety factor for rat = 6."""
        assert self.r.safety_factor == pytest.approx(6.0)

    def test_hed_less_than_noael(self):
        """HED (mg/kg) should be less than NOAEL for small animal species."""
        assert self.r.hed_mg_kg < self.r.noael_mg_kg

    def test_mrsd_less_than_hed(self):
        assert self.r.mrsd_mg < self.r.hed_mg

    def test_mrsd_positive(self):
        assert self.r.mrsd_mg > 0

    def test_hed_mg_equals_hed_mg_kg_times_bw(self):
        assert self.r.hed_mg == pytest.approx(self.r.hed_mg_kg * self.r.human_bw_kg, rel=1e-6)

    def test_noael_mg_m2_positive(self):
        assert self.r.noael_mg_m2 > 0

    def test_method_stored(self):
        assert "bsa" in self.r.method

    def test_notes_contains_mrsd(self):
        assert "MRSD" in self.r.notes


class TestSpeciesComparison:
    def test_dog_lower_safety_factor(self):
        """Dog has SF=2 vs rat SF=6 → dog gives larger MRSD for same NOAEL."""
        r_rat = predict_fih_dose("D", 100, "rat")
        r_dog = predict_fih_dose("D", 100, "dog")
        assert r_dog.mrsd_mg > r_rat.mrsd_mg

    def test_higher_noael_higher_mrsd(self):
        r1 = predict_fih_dose("D", 50, "rat")
        r2 = predict_fih_dose("D", 100, "rat")
        assert r2.mrsd_mg > r1.mrsd_mg

    def test_linear_scaling(self):
        """Doubling NOAEL should double MRSD."""
        r1 = predict_fih_dose("D", 50, "rat")
        r2 = predict_fih_dose("D", 100, "rat")
        assert r2.mrsd_mg == pytest.approx(r1.mrsd_mg * 2, rel=1e-6)

    def test_custom_safety_factor(self):
        r = predict_fih_dose("D", 100, "rat", safety_factor=10)
        assert r.safety_factor == 10.0

    def test_custom_sf_reduces_mrsd(self):
        r_sf6 = predict_fih_dose("D", 100, "rat", safety_factor=6)
        r_sf10 = predict_fih_dose("D", 100, "rat", safety_factor=10)
        assert r_sf10.mrsd_mg < r_sf6.mrsd_mg

    def test_all_species_work(self):
        for sp in ["mouse", "rat", "rabbit", "dog", "monkey"]:
            r = predict_fih_dose("D", 100, sp)
            assert r.mrsd_mg > 0


class TestAllometricMethod:
    def test_allometric_gives_result(self):
        r = predict_fih_dose("D", 100, "rat", method="allometric")
        assert isinstance(r, FIHDoseResult)
        assert r.mrsd_mg > 0

    def test_method_stored_as_allometric(self):
        r = predict_fih_dose("D", 100, "rat", method="allometric")
        assert "allometric" in r.method

    def test_allometric_vs_bsa_differ(self):
        r_bsa = predict_fih_dose("D", 100, "rat", method="bsa")
        r_allo = predict_fih_dose("D", 100, "rat", method="allometric")
        # They should differ (different methods)
        assert r_bsa.hed_mg != pytest.approx(r_allo.hed_mg, rel=0.01)


class TestCompareSpecies:
    def setup_method(self):
        self.results = compare_species("drug_x", {"rat": 100, "dog": 30})

    def test_returns_dict(self):
        assert "rat" in self.results
        assert "dog" in self.results

    def test_all_fih_results(self):
        for r in self.results.values():
            assert isinstance(r, FIHDoseResult)

    def test_most_conservative(self):
        best = most_conservative_mrsd(self.results)
        assert isinstance(best, FIHDoseResult)
        # Should be the one with lowest MRSD
        all_mrsd = [r.mrsd_mg for r in self.results.values()]
        assert best.mrsd_mg == min(all_mrsd)
