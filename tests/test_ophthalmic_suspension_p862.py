"""Tests for Phase 862 — Topical Ophthalmic Suspension PK."""

import math
import pytest
from omega_pbpk.core.ophthalmic_suspension_p862 import (
    OphthalmicSuspensionResult,
    simulate_ophthalmic_suspension,
)

DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=0.5,
    logP=2.0,
    mw_Da=300.0,
    solubility_mg_mL=1.0,
    particle_size_um=10.0,
)


def _run(**overrides):
    params = {**DEFAULTS, **overrides}
    return simulate_ophthalmic_suspension(**params)


class TestReturnType:
    def test_returns_dataclass(self):
        r = _run()
        assert isinstance(r, OphthalmicSuspensionResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 0.5

    def test_particle_size(self):
        r = _run()
        assert r.particle_size_um == 10.0

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 10

    def test_all_lists_same_length(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_suspension_mg_mL) == n
        assert len(r.c_dissolved_mg_mL) == n
        assert len(r.c_cornea_mg_g) == n
        assert len(r.c_aqueous_mg_L) == n
        assert len(r.dissolution_rate_mg_h) == n


class TestConcentrations:
    def test_non_negative_suspension(self):
        r = _run()
        assert all(c >= 0 for c in r.c_suspension_mg_mL)

    def test_non_negative_dissolved(self):
        r = _run()
        assert all(c >= 0 for c in r.c_dissolved_mg_mL)

    def test_non_negative_cornea(self):
        r = _run()
        assert all(c >= 0 for c in r.c_cornea_mg_g)

    def test_non_negative_aqueous(self):
        r = _run()
        assert all(c >= 0 for c in r.c_aqueous_mg_L)

    def test_dissolution_rate_non_negative(self):
        r = _run()
        assert all(d >= 0 for d in r.dissolution_rate_mg_h)


class TestDissolution:
    def test_smaller_particles_faster_dissolution(self):
        r_large = _run(particle_size_um=20.0)
        r_small = _run(particle_size_um=5.0)
        assert r_small.f_dissolved >= r_large.f_dissolved

    def test_higher_solubility_faster_dissolution(self):
        r_low = _run(solubility_mg_mL=0.5)
        r_high = _run(solubility_mg_mL=5.0)
        assert r_high.f_dissolved >= r_low.f_dissolved

    def test_f_dissolved_in_range(self):
        r = _run()
        assert 0 <= r.f_dissolved <= 1


class TestMetrics:
    def test_cmax_aqueous_positive(self):
        r = _run()
        assert r.cmax_aqueous_mg_L > 0

    def test_auc_aqueous_positive(self):
        r = _run()
        assert r.auc_aqueous_mg_h_per_L > 0

    def test_tmax_positive(self):
        r = _run()
        assert r.tmax_aqueous_h >= 0

    def test_bioavailability_in_range(self):
        r = _run()
        assert 0 <= r.bioavailability_ocular_pct <= 100


class TestDoseLinearity:
    def test_double_dose_double_cmax(self):
        r1 = _run(dose_mg=0.5)
        r2 = _run(dose_mg=1.0)
        ratio = r2.cmax_aqueous_mg_L / r1.cmax_aqueous_mg_L
        assert abs(ratio - 2.0) < 0.2


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _run(dose_mg=-1)

    def test_zero_particle_size_raises(self):
        with pytest.raises(ValueError):
            _run(particle_size_um=0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            _run(solubility_mg_mL=0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError):
            _run(mw_Da=0)


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0
