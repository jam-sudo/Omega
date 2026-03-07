"""Tests for biorelevant dissolution media effect (Phase 389)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.dissolution_media_effect import (
    VALID_MEDIA,
    DissolutionMediaResult,
    compare_media,
    simulate_dissolution_media,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sim(**kw) -> DissolutionMediaResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        media_type="FaSSIF",
        solubility_fasted_mg_mL=0.5,
        particle_radius_um=10.0,
        diffusion_coeff_cm2_s=5e-6,
        t_end_min=120.0,
        dt_min=0.5,
    )
    defaults.update(kw)
    return simulate_dissolution_media(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dissolution_media_result(self):
        assert isinstance(_sim(), DissolutionMediaResult)

    def test_drug_name_stored(self):
        r = _sim(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_dose_stored(self):
        r = _sim(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_media_type_stored(self):
        r = _sim(media_type="FeSSIF")
        assert r.media_type == "FeSSIF"

    def test_times_and_pct_same_length(self):
        r = _sim()
        assert len(r.times_min) == len(r.pct_dissolved)

    def test_times_start_at_zero(self):
        r = _sim()
        assert r.times_min[0] == 0.0

    def test_pct_dissolved_starts_at_zero(self):
        r = _sim()
        assert r.pct_dissolved[0] == 0.0

    def test_pct_dissolved_between_0_and_100(self):
        r = _sim()
        assert all(0.0 <= p <= 100.0 for p in r.pct_dissolved)

    def test_notes_non_empty(self):
        r = _sim()
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Physics / monotonicity
# ---------------------------------------------------------------------------

class TestPhysics:
    def test_pct_dissolved_non_decreasing(self):
        r = _sim()
        for i in range(1, len(r.pct_dissolved)):
            assert r.pct_dissolved[i] >= r.pct_dissolved[i - 1] - 1e-9

    def test_fesSIF_dissolves_faster_than_water(self):
        r_fessif = _sim(media_type="FeSSIF", t_end_min=60.0, dt_min=0.2)
        r_water = _sim(media_type="water", t_end_min=60.0, dt_min=0.2)
        assert r_fessif.pct_dissolved[-1] >= r_water.pct_dissolved[-1] - 1e-6

    def test_fassif_dissolves_faster_than_hcl(self):
        r_fassif = _sim(media_type="FaSSIF", t_end_min=60.0, dt_min=0.2)
        r_hcl = _sim(media_type="HCl_pH_1_2", t_end_min=60.0, dt_min=0.2)
        assert r_fassif.pct_dissolved[-1] >= r_hcl.pct_dissolved[-1] - 1e-6

    def test_smaller_particle_dissolves_faster(self):
        r_small = _sim(particle_radius_um=2.0, t_end_min=60.0, dt_min=0.2)
        r_large = _sim(particle_radius_um=50.0, t_end_min=60.0, dt_min=0.2)
        assert r_small.pct_dissolved[-1] >= r_large.pct_dissolved[-1] - 1e-6

    def test_higher_solubility_dissolves_faster(self):
        r_high = _sim(solubility_fasted_mg_mL=5.0, t_end_min=60.0, dt_min=0.2)
        r_low = _sim(solubility_fasted_mg_mL=0.05, t_end_min=60.0, dt_min=0.2)
        assert r_high.pct_dissolved[-1] >= r_low.pct_dissolved[-1] - 1e-6

    def test_fesSIF_effective_solubility_4x_water(self):
        r_fessif = _sim(media_type="FeSSIF", solubility_fasted_mg_mL=1.0)
        r_water = _sim(media_type="water", solubility_fasted_mg_mL=1.0)
        ratio = r_fessif.solubility_effective_mg_mL / r_water.solubility_effective_mg_mL
        assert abs(ratio - 4.0) < 1e-9

    def test_hcl_effective_solubility_half_water(self):
        r_hcl = _sim(media_type="HCl_pH_1_2", solubility_fasted_mg_mL=1.0)
        assert abs(r_hcl.solubility_effective_mg_mL - 0.5) < 1e-9

    def test_pbs_effective_solubility_equals_fasted(self):
        r_pbs = _sim(media_type="PBS_pH_6_8", solubility_fasted_mg_mL=2.0)
        assert abs(r_pbs.solubility_effective_mg_mL - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Milestone times
# ---------------------------------------------------------------------------

class TestMilestoneTimes:
    def test_t50_none_when_not_reached(self):
        # Very large dose, low solubility — unlikely to hit 50 %
        r = _sim(dose_mg=10000.0, solubility_fasted_mg_mL=0.001, t_end_min=5.0, dt_min=0.1)
        assert r.t50_min is None

    def test_t50_set_when_reached(self):
        # Small dose, high solubility, fine particles — should dissolve
        r = _sim(dose_mg=1.0, solubility_fasted_mg_mL=10.0, particle_radius_um=1.0, t_end_min=60.0)
        assert r.t50_min is not None

    def test_t80_geq_t50(self):
        r = _sim(dose_mg=1.0, solubility_fasted_mg_mL=10.0, particle_radius_um=1.0, t_end_min=60.0)
        if r.t50_min is not None and r.t80_min is not None:
            assert r.t80_min >= r.t50_min - 1e-9

    def test_t_complete_geq_t80(self):
        r = _sim(dose_mg=1.0, solubility_fasted_mg_mL=10.0, particle_radius_um=1.0, t_end_min=60.0)
        if r.t80_min is not None and r.t_complete_min is not None:
            assert r.t_complete_min >= r.t80_min - 1e-9


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            _sim(drug_name="")

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            _sim(dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _sim(dose_mg=0.0)

    def test_invalid_media_raises(self):
        with pytest.raises(ValueError):
            _sim(media_type="ethanol")

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            _sim(solubility_fasted_mg_mL=0.0)

    def test_negative_solubility_raises(self):
        with pytest.raises(ValueError):
            _sim(solubility_fasted_mg_mL=-1.0)

    def test_zero_particle_radius_raises(self):
        with pytest.raises(ValueError):
            _sim(particle_radius_um=0.0)

    def test_zero_diffusion_coeff_raises(self):
        with pytest.raises(ValueError):
            _sim(diffusion_coeff_cm2_s=0.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError):
            _sim(t_end_min=0.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError):
            _sim(dt_min=0.0)


# ---------------------------------------------------------------------------
# compare_media
# ---------------------------------------------------------------------------

class TestCompareMedia:
    def test_returns_dict(self):
        result = compare_media(
            drug_name="TestDrug",
            dose_mg=100.0,
            solubility_fasted_mg_mL=0.5,
            particle_radius_um=10.0,
            diffusion_coeff_cm2_s=5e-6,
        )
        assert isinstance(result, dict)

    def test_all_five_media_present(self):
        result = compare_media(
            drug_name="TestDrug",
            dose_mg=100.0,
            solubility_fasted_mg_mL=0.5,
            particle_radius_um=10.0,
            diffusion_coeff_cm2_s=5e-6,
        )
        assert set(result.keys()) == VALID_MEDIA

    def test_all_values_are_results(self):
        result = compare_media(
            drug_name="TestDrug",
            dose_mg=100.0,
            solubility_fasted_mg_mL=0.5,
            particle_radius_um=10.0,
            diffusion_coeff_cm2_s=5e-6,
        )
        assert all(isinstance(v, DissolutionMediaResult) for v in result.values())

    def test_fesSIF_faster_than_water_in_compare(self):
        result = compare_media(
            drug_name="TestDrug",
            dose_mg=50.0,
            solubility_fasted_mg_mL=0.3,
            particle_radius_um=10.0,
            diffusion_coeff_cm2_s=5e-6,
        )
        pct_fessif = result["FeSSIF"].pct_dissolved[-1]
        pct_water = result["water"].pct_dissolved[-1]
        assert pct_fessif >= pct_water - 1e-6
