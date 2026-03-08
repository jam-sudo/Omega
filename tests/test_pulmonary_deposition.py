"""Tests for Phase 568 - Pulmonary Drug Deposition Model."""

import pytest

from omega_pbpk.core.pulmonary_deposition import (
    PulmonaryDepositionResult,
    compare_particle_sizes,
    predict_pulmonary_deposition,
)


def _make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        particle_size_um=3.0,
        device_type="MDI",
    )
    defaults.update(kwargs)
    return predict_pulmonary_deposition(**defaults)


class TestReturnType:
    def test_returns_pulmonary_deposition_result(self):
        r = _make_result()
        assert isinstance(r, PulmonaryDepositionResult)

    def test_all_fields_present(self):
        r = _make_result()
        assert r.drug_name == "TestDrug"
        assert isinstance(r.particle_size_um, float)
        assert isinstance(r.device_type, str)
        assert isinstance(r.f_oropharynx, float)
        assert isinstance(r.f_tracheobronchial, float)
        assert isinstance(r.f_alveolar, float)
        assert isinstance(r.f_exhaled, float)
        assert isinstance(r.fine_particle_fraction, float)
        assert isinstance(r.lung_deposition_total, float)
        assert isinstance(r.systemic_absorption_fraction, float)
        assert isinstance(r.oropharyngeal_dose_mg, float)
        assert isinstance(r.lung_dose_mg, float)
        assert isinstance(r.notes, str)

    def test_frozen_dataclass(self):
        r = _make_result()
        with pytest.raises(AttributeError):
            r.drug_name = "other"


class TestFractions:
    def test_all_fractions_non_negative(self):
        r = _make_result()
        assert r.f_oropharynx >= 0
        assert r.f_tracheobronchial >= 0
        assert r.f_alveolar >= 0
        assert r.f_exhaled >= 0

    def test_fractions_sum_to_one(self):
        r = _make_result()
        total = r.f_oropharynx + r.f_tracheobronchial + r.f_alveolar + r.f_exhaled
        assert abs(total - 1.0) < 1e-10

    def test_fractions_sum_to_one_large_particle(self):
        r = _make_result(particle_size_um=15.0)
        total = r.f_oropharynx + r.f_tracheobronchial + r.f_alveolar + r.f_exhaled
        assert abs(total - 1.0) < 1e-10

    def test_fractions_sum_to_one_small_particle(self):
        r = _make_result(particle_size_um=0.5)
        total = r.f_oropharynx + r.f_tracheobronchial + r.f_alveolar + r.f_exhaled
        assert abs(total - 1.0) < 1e-10

    def test_lung_deposition_total(self):
        r = _make_result()
        expected = r.f_tracheobronchial + r.f_alveolar
        assert abs(r.lung_deposition_total - expected) < 1e-10


class TestLargeParticles:
    def test_high_oropharyngeal_deposition(self):
        r = _make_result(particle_size_um=15.0)
        assert r.f_oropharynx > 0.1

    def test_large_vs_small_oropharyngeal(self):
        r_large = _make_result(particle_size_um=15.0)
        r_small = _make_result(particle_size_um=2.0)
        assert r_large.f_oropharynx > r_small.f_oropharynx


class TestSmallParticles:
    def test_highest_alveolar_for_small(self):
        r1 = _make_result(particle_size_um=2.0)
        r10 = _make_result(particle_size_um=10.0)
        assert r1.f_alveolar > r10.f_alveolar

    def test_optimal_range_1_3_um(self):
        r2 = _make_result(particle_size_um=2.0)
        r8 = _make_result(particle_size_um=8.0)
        assert r2.f_alveolar > r8.f_alveolar


class TestDeviceType:
    def test_nebulizer_higher_lung_deposition_than_mdi(self):
        r_mdi = _make_result(device_type="MDI", particle_size_um=3.0)
        r_neb = _make_result(device_type="nebulizer", particle_size_um=3.0)
        assert r_neb.lung_deposition_total > r_mdi.lung_deposition_total

    def test_dpi_between_mdi_and_nebulizer(self):
        r_mdi = _make_result(device_type="MDI", particle_size_um=3.0)
        r_dpi = _make_result(device_type="DPI", particle_size_um=3.0)
        r_neb = _make_result(device_type="nebulizer", particle_size_um=3.0)
        assert r_mdi.lung_deposition_total < r_dpi.lung_deposition_total
        assert r_dpi.lung_deposition_total < r_neb.lung_deposition_total

    def test_all_device_types_accepted(self):
        for dt in ["MDI", "DPI", "nebulizer"]:
            r = _make_result(device_type=dt)
            assert r.device_type == dt


class TestDoses:
    def test_lung_dose_positive(self):
        r = _make_result()
        assert r.lung_dose_mg > 0

    def test_oropharyngeal_dose_positive(self):
        r = _make_result()
        assert r.oropharyngeal_dose_mg > 0

    def test_doses_scale_with_input(self):
        r1 = _make_result(dose_mg=100.0)
        r2 = _make_result(dose_mg=200.0)
        assert abs(r2.lung_dose_mg / r1.lung_dose_mg - 2.0) < 1e-10


class TestSystemicAbsorption:
    def test_systemic_absorption_non_negative(self):
        r = _make_result()
        assert r.systemic_absorption_fraction >= 0

    def test_fpf_in_range(self):
        r = _make_result()
        assert 0.0 <= r.fine_particle_fraction <= 1.0


class TestCompareParticleSizes:
    def test_returns_list(self):
        results = compare_particle_sizes("TestDrug", 100.0)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_sorted_descending_by_lung_deposition(self):
        results = compare_particle_sizes("TestDrug", 100.0)
        depositions = [r.lung_deposition_total for r in results]
        assert depositions == sorted(depositions, reverse=True)


class TestValidation:
    def test_invalid_device_type_raises(self):
        with pytest.raises(ValueError):
            _make_result(device_type="inhaler")

    def test_negative_particle_size_raises(self):
        with pytest.raises(ValueError):
            _make_result(particle_size_um=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _make_result(dose_mg=0.0)

    def test_zero_particle_size_raises(self):
        with pytest.raises(ValueError):
            _make_result(particle_size_um=0.0)
