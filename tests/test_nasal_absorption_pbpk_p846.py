"""Tests for 4-Region Nasal Absorption PBPK Model — Phase 846."""

import pytest

from omega_pbpk.core.nasal_absorption_pbpk_p846 import (
    NasalAbsorptionResult,
    simulate_nasal_absorption,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> NasalAbsorptionResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=1.0,
        particle_size_um=5.0,
        logp=2.0,
        mw_da=350.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        t_end_h=6.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_nasal_absorption(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default_result()
    assert isinstance(result, NasalAbsorptionResult)


# ---------------------------------------------------------------------------
# 4 regions
# ---------------------------------------------------------------------------


def test_four_region_names():
    result = _default_result()
    assert len(result.region_names) == 4


def test_region_names_values():
    result = _default_result()
    assert result.region_names == ["anterior", "turbinate", "olfactory", "nasopharynx"]


def test_deposition_fractions_length():
    result = _default_result()
    assert len(result.deposition_fractions) == 4


def test_absorption_fractions_length():
    result = _default_result()
    assert len(result.absorption_fractions) == 4


# ---------------------------------------------------------------------------
# sum(deposition_fractions) ≈ 1.0
# ---------------------------------------------------------------------------


def test_deposition_fractions_sum_to_one_default():
    result = _default_result()
    assert sum(result.deposition_fractions) == pytest.approx(1.0, abs=1e-9)


def test_deposition_fractions_sum_to_one_large_particle():
    result = _default_result(particle_size_um=15.0)
    assert sum(result.deposition_fractions) == pytest.approx(1.0, abs=1e-9)


def test_deposition_fractions_sum_to_one_small_particle():
    result = _default_result(particle_size_um=2.0)
    assert sum(result.deposition_fractions) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# cmax > 0, auc > 0
# ---------------------------------------------------------------------------


def test_cmax_positive():
    result = _default_result()
    assert result.cmax_mg_L > 0.0


def test_auc_positive():
    result = _default_result()
    assert result.auc_mg_h_per_L > 0.0


def test_tmax_non_negative():
    result = _default_result()
    assert result.tmax_h >= 0.0


# ---------------------------------------------------------------------------
# Smaller particles → higher olfactory delivery fraction
# ---------------------------------------------------------------------------


def test_small_particles_higher_olfactory_delivery():
    small = _default_result(particle_size_um=2.0)
    large = _default_result(particle_size_um=15.0)
    assert small.olfactory_delivery_fraction > large.olfactory_delivery_fraction


def test_olfactory_delivery_fraction_non_negative():
    result = _default_result()
    assert result.olfactory_delivery_fraction >= 0.0


# ---------------------------------------------------------------------------
# systemic_bioavailability in (0, 1]
# ---------------------------------------------------------------------------


def test_systemic_bioavailability_positive():
    result = _default_result()
    assert result.systemic_bioavailability > 0.0


def test_systemic_bioavailability_at_most_one():
    result = _default_result()
    assert result.systemic_bioavailability <= 1.0


def test_systemic_bioavailability_range_high_logp():
    result = _default_result(logp=4.0)
    assert 0.0 < result.systemic_bioavailability <= 1.0


# ---------------------------------------------------------------------------
# Mucociliary clearance rate
# ---------------------------------------------------------------------------


def test_mucociliary_clearance_rate():
    result = _default_result()
    assert result.mucociliary_clearance_rate_per_h == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Trajectory sanity
# ---------------------------------------------------------------------------


def test_times_length_consistent_with_c_plasma():
    result = _default_result()
    assert len(result.times_h) == len(result.c_plasma_mg_L)


def test_times_start_at_zero():
    result = _default_result()
    assert result.times_h[0] == pytest.approx(0.0)


def test_plasma_starts_at_zero():
    result = _default_result()
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nasal_absorption("Drug", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_nasal_absorption("Drug", dose_mg=-1.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_nasal_absorption("Drug", dose_mg=1.0, cl_L_per_h=0.0)


def test_validation_vd_negative():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_nasal_absorption("Drug", dose_mg=1.0, vd_L=-10.0)


def test_validation_particle_size_zero():
    with pytest.raises(ValueError, match="particle_size_um"):
        simulate_nasal_absorption("Drug", dose_mg=1.0, particle_size_um=0.0)
