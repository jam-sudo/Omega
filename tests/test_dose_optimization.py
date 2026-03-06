"""Tests for clinical dose optimization module (DoseOptimizer, MultiDoseSimulator)."""

import pytest

from omega_pbpk.clinical.dose_optimization import (
    DoseOptimizer,
    FIHDoseResult,
)


@pytest.fixture(scope="module")
def optimizer():
    return DoseOptimizer()


class TestFIHDose:
    def test_result_type(self, optimizer):
        r = optimizer.fih_dose(noael_mg_kg=100, species="rat")
        assert isinstance(r, FIHDoseResult)

    def test_fih_dose_positive(self, optimizer):
        r = optimizer.fih_dose(noael_mg_kg=100, species="rat")
        assert r.fih_dose_mg > 0

    def test_fih_dose_less_than_noael_bw(self, optimizer):
        """FIH dose must be less than NOAEL × body_weight (safety factor > 1)."""
        r = optimizer.fih_dose(
            noael_mg_kg=100, species="rat", body_weight_kg=70.0, safety_factor=10.0
        )
        assert r.fih_dose_mg < 100.0 * 70.0

    def test_known_rat_value(self, optimizer):
        """rat Km=6.2, human Km=37 → HED = 100 * 6.2/37; FIH = HED*70/10."""
        r = optimizer.fih_dose(
            noael_mg_kg=100, species="rat", body_weight_kg=70.0, safety_factor=10.0
        )
        expected_hed = 100.0 * 6.2 / 37.0
        expected_fih = expected_hed * 70.0 / 10.0
        assert r.fih_dose_mg == pytest.approx(expected_fih, rel=1e-4)

    def test_hek_rat(self, optimizer):
        r = optimizer.fih_dose(noael_mg_kg=100, species="rat")
        # HEK = Km_rat / Km_human = 6.2 / 37
        assert r.hek == pytest.approx(6.2 / 37.0, abs=1e-3)

    def test_higher_noael_higher_dose(self, optimizer):
        r1 = optimizer.fih_dose(noael_mg_kg=50, species="rat")
        r2 = optimizer.fih_dose(noael_mg_kg=100, species="rat")
        assert r2.fih_dose_mg > r1.fih_dose_mg

    def test_linear_scaling_noael(self, optimizer):
        r1 = optimizer.fih_dose(noael_mg_kg=50, species="rat")
        r2 = optimizer.fih_dose(noael_mg_kg=100, species="rat")
        assert r2.fih_dose_mg == pytest.approx(r1.fih_dose_mg * 2, rel=1e-4)

    def test_higher_safety_factor_lower_dose(self, optimizer):
        r_sf5 = optimizer.fih_dose(noael_mg_kg=100, species="rat", safety_factor=5.0)
        r_sf10 = optimizer.fih_dose(noael_mg_kg=100, species="rat", safety_factor=10.0)
        assert r_sf10.fih_dose_mg < r_sf5.fih_dose_mg

    def test_dog_higher_hed_than_rat(self, optimizer):
        """Dog has higher Km than rat → larger HEK → higher FIH dose."""
        r_rat = optimizer.fih_dose(noael_mg_kg=100, species="rat")
        r_dog = optimizer.fih_dose(noael_mg_kg=100, species="dog")
        assert r_dog.fih_dose_mg > r_rat.fih_dose_mg

    def test_all_species_work(self, optimizer):
        for sp in ["mouse", "rat", "rabbit", "dog", "monkey"]:
            r = optimizer.fih_dose(100, sp)
            assert r.fih_dose_mg > 0

    def test_unknown_species_uses_rat_default(self, optimizer):
        """Unknown species falls back to rat Km=6.2."""
        r_unknown = optimizer.fih_dose(100, "elephant")
        r_rat = optimizer.fih_dose(100, "rat")
        assert r_unknown.fih_dose_mg == pytest.approx(r_rat.fih_dose_mg, rel=1e-4)

    def test_safety_factor_stored(self, optimizer):
        r = optimizer.fih_dose(100, "rat", safety_factor=5.0)
        assert r.safety_factor == 5.0

    def test_method_string_contains_species(self, optimizer):
        r = optimizer.fih_dose(100, "dog")
        assert "dog" in r.method.lower()

    def test_body_weight_stored(self, optimizer):
        r = optimizer.fih_dose(100, "rat", body_weight_kg=80.0)
        assert r.body_weight_kg == 80.0

    def test_noael_stored(self, optimizer):
        r = optimizer.fih_dose(75.0, "rat")
        assert r.noael_mg_kg == 75.0


class TestFIHScalingProperties:
    def test_fih_proportional_to_body_weight(self, optimizer):
        """FIH dose scales linearly with body weight."""
        r70 = optimizer.fih_dose(100, "rat", body_weight_kg=70.0)
        r80 = optimizer.fih_dose(100, "rat", body_weight_kg=80.0)
        ratio = r80.fih_dose_mg / r70.fih_dose_mg
        assert ratio == pytest.approx(80.0 / 70.0, rel=1e-4)

    def test_fih_inversely_proportional_to_sf(self, optimizer):
        """Doubling safety factor halves FIH dose."""
        r_sf5 = optimizer.fih_dose(100, "rat", safety_factor=5.0)
        r_sf10 = optimizer.fih_dose(100, "rat", safety_factor=10.0)
        assert r_sf10.fih_dose_mg == pytest.approx(r_sf5.fih_dose_mg / 2.0, rel=1e-4)
