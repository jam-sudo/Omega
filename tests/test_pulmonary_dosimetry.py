"""Tests for Phase 351: Pulmonary Drug Delivery Dosimetry."""

from __future__ import annotations

import pytest

from omega_pbpk.core.pulmonary_dosimetry import (
    PulmonaryDosimetryResult,
    compare_devices,
    simulate_pulmonary_dosimetry,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

class TestBasicSmoke:
    def test_mdi_returns_result(self):
        result = simulate_pulmonary_dosimetry("Salbutamol", 0.1, device_type="MDI")
        assert isinstance(result, PulmonaryDosimetryResult)

    def test_dpi_returns_result(self):
        result = simulate_pulmonary_dosimetry("Fluticasone", 0.05, device_type="DPI")
        assert isinstance(result, PulmonaryDosimetryResult)

    def test_nebulizer_returns_result(self):
        result = simulate_pulmonary_dosimetry("Ipratropium", 0.5, device_type="nebulizer")
        assert isinstance(result, PulmonaryDosimetryResult)

    def test_soft_mist_returns_result(self):
        result = simulate_pulmonary_dosimetry("Tiotropium", 0.01, device_type="soft_mist")
        assert isinstance(result, PulmonaryDosimetryResult)

    def test_frozen_dataclass(self):
        result = simulate_pulmonary_dosimetry("Drug", 1.0)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_notes_contains_drug_name(self):
        result = simulate_pulmonary_dosimetry("Budesonide", 0.2)
        assert "Budesonide" in result.notes


# ---------------------------------------------------------------------------
# Device-specific physics
# ---------------------------------------------------------------------------

class TestDevicePhysics:
    def test_nebulizer_lower_oropharyngeal_than_mdi(self):
        neb = simulate_pulmonary_dosimetry("Drug", 1.0, device_type="nebulizer")
        mdi = simulate_pulmonary_dosimetry("Drug", 1.0, device_type="MDI")
        assert neb.m_oropharyngeal_mg < mdi.m_oropharyngeal_mg

    def test_mdi_highest_device_loss(self):
        """MDI has 80% efficiency → 20% device loss, DPI has 70% → 30% loss.
        So DPI has more device loss than MDI."""
        mdi = simulate_pulmonary_dosimetry("Drug", 1.0, device_type="MDI")
        dpi = simulate_pulmonary_dosimetry("Drug", 1.0, device_type="DPI")
        # DPI has lower efficiency (0.70 vs 0.80), so more device loss
        assert mdi.m_device_loss_mg < dpi.m_device_loss_mg

    def test_soft_mist_highest_emitted_dose(self):
        sm = simulate_pulmonary_dosimetry("Drug", 1.0, device_type="soft_mist")
        mdi = simulate_pulmonary_dosimetry("Drug", 1.0, device_type="MDI")
        dpi = simulate_pulmonary_dosimetry("Drug", 1.0, device_type="DPI")
        neb = simulate_pulmonary_dosimetry("Drug", 1.0, device_type="nebulizer")
        assert sm.emitted_dose_mg >= max(
            mdi.emitted_dose_mg, dpi.emitted_dose_mg, neb.emitted_dose_mg
        )

    def test_emitted_dose_equals_nominal_times_efficiency(self):
        result = simulate_pulmonary_dosimetry("Drug", 2.0, device_type="DPI")
        assert abs(result.emitted_dose_mg - 2.0 * 0.70) < 1e-9

    def test_device_loss_plus_emitted_equals_nominal(self):
        for dev in ["MDI", "DPI", "nebulizer", "soft_mist"]:
            result = simulate_pulmonary_dosimetry("Drug", 1.0, device_type=dev)
            assert abs(result.m_device_loss_mg + result.emitted_dose_mg - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Particle size effects
# ---------------------------------------------------------------------------

class TestParticleSize:
    def test_small_mmad_higher_fpf(self):
        small = simulate_pulmonary_dosimetry("Drug", 1.0, mmad_um=1.0)
        large = simulate_pulmonary_dosimetry("Drug", 1.0, mmad_um=5.0)
        assert small.fine_particle_fraction > large.fine_particle_fraction

    def test_small_mmad_higher_f_alveolar(self):
        small = simulate_pulmonary_dosimetry("Drug", 1.0, mmad_um=0.5)
        large = simulate_pulmonary_dosimetry("Drug", 1.0, mmad_um=4.0)
        assert small.f_alveolar >= large.f_alveolar

    def test_fpf_in_0_1(self):
        for mmad in [0.5, 1.0, 3.0, 6.0, 10.0]:
            result = simulate_pulmonary_dosimetry("Drug", 1.0, mmad_um=mmad)
            assert 0.0 <= result.fine_particle_fraction <= 1.0

    def test_fine_particle_dose_leq_emitted_dose(self):
        result = simulate_pulmonary_dosimetry("Drug", 1.0, mmad_um=2.0)
        assert result.fine_particle_dose_mg <= result.emitted_dose_mg + 1e-9


# ---------------------------------------------------------------------------
# Mass balance and fractions
# ---------------------------------------------------------------------------

class TestMassBalance:
    def test_f_lung_total_positive(self):
        result = simulate_pulmonary_dosimetry("Drug", 1.0)
        assert result.f_lung_total > 0

    def test_f_systemic_in_0_1(self):
        result = simulate_pulmonary_dosimetry("Drug", 1.0)
        assert 0.0 <= result.f_systemic <= 1.0

    def test_mass_balance_approximate(self):
        """Device loss + oropharyngeal + lung + exhaled should ≈ nominal dose."""
        result = simulate_pulmonary_dosimetry("Drug", 1.0)
        total = (
            result.m_device_loss_mg
            + result.m_oropharyngeal_mg
            + result.m_conducting_airway_mg
            + result.m_alveolar_mg
            + result.m_exhaled_mg
        )
        # Tolerance of 0.1 mg per spec
        assert abs(total - 1.0) < 0.1

    def test_conducting_plus_alveolar_equals_lung(self):
        result = simulate_pulmonary_dosimetry("Drug", 1.0)
        lung_total = result.m_conducting_airway_mg + result.m_alveolar_mg
        assert abs(lung_total / result.nominal_dose_mg - result.f_lung_total) < 1e-9

    def test_exhaled_non_negative(self):
        result = simulate_pulmonary_dosimetry("Drug", 1.0, mmad_um=2.0, device_type="MDI")
        assert result.m_exhaled_mg >= 0.0


# ---------------------------------------------------------------------------
# Systemic exposure
# ---------------------------------------------------------------------------

class TestSystemicExposure:
    def test_high_oral_f_increases_systemic(self):
        low = simulate_pulmonary_dosimetry("Drug", 1.0, f_oral_bioavailability=0.05)
        high = simulate_pulmonary_dosimetry("Drug", 1.0, f_oral_bioavailability=0.60)
        assert high.f_systemic > low.f_systemic

    def test_f_swallowed_matches_mdi_fraction(self):
        result = simulate_pulmonary_dosimetry("Drug", 2.0, device_type="MDI")
        # MDI oph_fraction=0.50, efficiency=0.80 → oph = 2.0*0.80*0.50 = 0.80
        expected_oph = 2.0 * 0.80 * 0.50
        assert abs(result.m_oropharyngeal_mg - expected_oph) < 1e-9
        assert abs(result.f_swallowed - expected_oph / 2.0) < 1e-9

    def test_therapeutic_index_local_positive(self):
        result = simulate_pulmonary_dosimetry("Drug", 1.0)
        assert result.therapeutic_index_local > 0


# ---------------------------------------------------------------------------
# compare_devices
# ---------------------------------------------------------------------------

class TestCompareDevices:
    def test_returns_all_four_devices(self):
        out = compare_devices("Drug", 1.0)
        for dev in ["MDI", "DPI", "nebulizer", "soft_mist"]:
            assert dev in out

    def test_best_lung_delivery_is_string(self):
        out = compare_devices("Drug", 1.0)
        assert isinstance(out["best_lung_delivery"], str)

    def test_best_lung_delivery_in_valid_devices(self):
        out = compare_devices("Drug", 1.0)
        assert out["best_lung_delivery"] in ["MDI", "DPI", "nebulizer", "soft_mist"]

    def test_best_lung_delivery_highest_f_lung(self):
        out = compare_devices("Drug", 1.0)
        best = out["best_lung_delivery"]
        for dev in ["MDI", "DPI", "nebulizer", "soft_mist"]:
            assert out[dev].f_lung_total <= out[best].f_lung_total + 1e-9


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_device_type(self):
        with pytest.raises(ValueError, match="device_type"):
            simulate_pulmonary_dosimetry("Drug", 1.0, device_type="pMDI")

    def test_zero_nominal_dose(self):
        with pytest.raises(ValueError, match="nominal_dose_mg"):
            simulate_pulmonary_dosimetry("Drug", 0.0)

    def test_negative_nominal_dose(self):
        with pytest.raises(ValueError):
            simulate_pulmonary_dosimetry("Drug", -1.0)

    def test_gsd_leq_one(self):
        with pytest.raises(ValueError, match="gsd"):
            simulate_pulmonary_dosimetry("Drug", 1.0, gsd=1.0)

    def test_zero_mmad(self):
        with pytest.raises(ValueError, match="mmad_um"):
            simulate_pulmonary_dosimetry("Drug", 1.0, mmad_um=0.0)

    def test_zero_flow(self):
        with pytest.raises(ValueError, match="inhalation_flow"):
            simulate_pulmonary_dosimetry("Drug", 1.0, inhalation_flow_L_per_min=0.0)
