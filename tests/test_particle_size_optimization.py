"""Tests for particle size optimization (Phase 450)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.particle_size_optimization import (
    ParticleSizeResult,
    ParticleSizeOptResult,
    dissolution_rate,
    time_to_dissolve,
    dissolution_profile,
    optimize_particle_size,
    biopharmaceutics_classification,
    _optimize_particle_size_phase450,
)


# ---------------------------------------------------------------------------
# Helpers — Phase 450 new optimize_particle_size API
# ---------------------------------------------------------------------------


def _new_opt(**kw) -> ParticleSizeResult:
    defaults = dict(
        target_dissolution_time_h=1.0,
        drug_solubility_mg_mL=1.0,
        radius_range_um=(0.1, 500.0),
        diffusion_coeff_cm2_s=1e-6,
        boundary_layer_um=10.0,
        density_g_cm3=1.2,
    )
    defaults.update(kw)
    return _optimize_particle_size_phase450(**defaults)


# ---------------------------------------------------------------------------
# Helpers — legacy optimize_particle_size API
# ---------------------------------------------------------------------------


def _default_opt(**kw) -> ParticleSizeOptResult:
    defaults = dict(
        drug_name="TestDrug",
        target_dissolved_pct_at_60min=80.0,
        solubility_mg_mL=1.0,
        diffusion_coeff_cm2_s=5e-6,
        dose_mg=100.0,
        density_g_cm3=1.2,
        particle_sizes_um=[5.0, 10.0, 20.0, 50.0, 100.0],
        media_volume_mL=900.0,
    )
    defaults.update(kw)
    return optimize_particle_size(**defaults)


# ---------------------------------------------------------------------------
# Phase 450: dissolution_rate
# ---------------------------------------------------------------------------


class TestDissolutionRate:
    def test_returns_float(self):
        rate = dissolution_rate(10.0, 1.0)
        assert isinstance(rate, float)

    def test_positive_rate(self):
        rate = dissolution_rate(10.0, 1.0)
        assert rate > 0.0

    def test_smaller_radius_higher_rate(self):
        rate_small = dissolution_rate(1.0, 1.0)
        rate_large = dissolution_rate(100.0, 1.0)
        assert rate_small > rate_large

    def test_higher_solubility_higher_rate(self):
        rate_lo = dissolution_rate(10.0, 0.1)
        rate_hi = dissolution_rate(10.0, 10.0)
        assert rate_hi > rate_lo

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            dissolution_rate(0.0, 1.0)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            dissolution_rate(-5.0, 1.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            dissolution_rate(10.0, 0.0)

    def test_zero_diffusion_raises(self):
        with pytest.raises(ValueError):
            dissolution_rate(10.0, 1.0, diffusion_coeff_cm2_s=0.0)

    def test_zero_boundary_layer_raises(self):
        with pytest.raises(ValueError):
            dissolution_rate(10.0, 1.0, boundary_layer_um=0.0)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError):
            dissolution_rate(10.0, 1.0, density_g_cm3=0.0)


# ---------------------------------------------------------------------------
# Phase 450: time_to_dissolve
# ---------------------------------------------------------------------------


class TestTimeToDissolveFn:
    def test_returns_float(self):
        t = time_to_dissolve(10.0, 1.0)
        assert isinstance(t, float)

    def test_positive_time(self):
        t = time_to_dissolve(10.0, 1.0)
        assert t > 0.0

    def test_smaller_radius_faster(self):
        t_small = time_to_dissolve(1.0, 1.0)
        t_large = time_to_dissolve(100.0, 1.0)
        assert t_small < t_large

    def test_higher_solubility_faster(self):
        t_lo = time_to_dissolve(10.0, 0.1)
        t_hi = time_to_dissolve(10.0, 10.0)
        assert t_hi < t_lo

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            time_to_dissolve(0.0, 1.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            time_to_dissolve(10.0, 0.0)


# ---------------------------------------------------------------------------
# Phase 450: _optimize_particle_size_phase450 (ParticleSizeResult)
# ---------------------------------------------------------------------------


class TestNewOptimizeParticleSize:
    def test_returns_result_type(self):
        assert isinstance(_new_opt(), ParticleSizeResult)

    def test_optimal_radius_positive(self):
        r = _new_opt()
        assert r.optimal_radius_um > 0.0

    def test_predicted_time_close_to_target(self):
        target = 1.0
        r = _new_opt(target_dissolution_time_h=target)
        assert abs(r.predicted_dissolution_time_h - target) / target < 0.05

    def test_shorter_target_smaller_radius(self):
        r_short = _new_opt(target_dissolution_time_h=0.5)
        r_long = _new_opt(target_dissolution_time_h=4.0)
        assert r_short.optimal_radius_um < r_long.optimal_radius_um

    def test_higher_solubility_larger_optimal_radius(self):
        # Higher Cs allows a larger particle to dissolve in the same time (Higuchi)
        r_lo = _new_opt(drug_solubility_mg_mL=0.1)
        r_hi = _new_opt(drug_solubility_mg_mL=10.0)
        assert r_hi.optimal_radius_um > r_lo.optimal_radius_um

    def test_solubility_preserved(self):
        r = _new_opt(drug_solubility_mg_mL=2.5)
        assert r.solubility_mg_mL == 2.5

    def test_target_time_preserved(self):
        r = _new_opt(target_dissolution_time_h=2.0)
        assert r.target_dissolution_time_h == 2.0

    def test_notes_is_str(self):
        r = _new_opt()
        assert isinstance(r.notes, str)

    def test_clamped_to_min_radius(self):
        # Very short target -> optimal might be below min
        r = _new_opt(
            target_dissolution_time_h=1e-9,
            radius_range_um=(10.0, 500.0),
        )
        assert r.optimal_radius_um >= 10.0

    def test_clamped_to_max_radius(self):
        # Very long target -> optimal might exceed max
        r = _new_opt(
            target_dissolution_time_h=1e9,
            radius_range_um=(0.1, 50.0),
        )
        assert r.optimal_radius_um <= 50.0

    def test_zero_target_time_raises(self):
        with pytest.raises(ValueError):
            _new_opt(target_dissolution_time_h=0.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            _new_opt(drug_solubility_mg_mL=0.0)

    def test_invalid_radius_range_raises(self):
        with pytest.raises(ValueError):
            _new_opt(radius_range_um=(100.0, 10.0))


# ---------------------------------------------------------------------------
# Phase 450: dissolution_profile
# ---------------------------------------------------------------------------


class TestDissolutionProfile:
    def test_returns_dict(self):
        result = dissolution_profile(10.0, 1.0)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = dissolution_profile(10.0, 1.0)
        assert "times_h" in result
        assert "fraction_dissolved" in result

    def test_starts_at_zero(self):
        result = dissolution_profile(10.0, 1.0)
        assert result["fraction_dissolved"][0] == pytest.approx(0.0)

    def test_fraction_nondecreasing(self):
        result = dissolution_profile(10.0, 1.0)
        fracs = result["fraction_dissolved"]
        assert all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1))

    def test_fraction_max_1(self):
        result = dissolution_profile(10.0, 1.0)
        assert max(result["fraction_dissolved"]) <= 1.0 + 1e-9

    def test_small_radius_reaches_high_fraction(self):
        result = dissolution_profile(0.5, 10.0, t_end_h=2.0)
        assert result["fraction_dissolved"][-1] > 0.5

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            dissolution_profile(0.0, 1.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            dissolution_profile(10.0, 0.0)

    def test_times_and_fracs_same_length(self):
        result = dissolution_profile(10.0, 1.0)
        assert len(result["times_h"]) == len(result["fraction_dissolved"])


# ---------------------------------------------------------------------------
# Legacy tests — existing optimize_particle_size API
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_type(self):
        assert isinstance(_default_opt(), ParticleSizeOptResult)

    def test_drug_name_preserved(self):
        r = _default_opt(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_target_preserved(self):
        r = _default_opt(target_dissolved_pct_at_60min=75.0)
        assert r.target_dissolved_pct_at_60min == 75.0

    def test_solubility_preserved(self):
        r = _default_opt(solubility_mg_mL=2.0)
        assert r.solubility_mg_mL == 2.0

    def test_dose_preserved(self):
        r = _default_opt(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_particle_sizes_sorted(self):
        r = _default_opt(particle_sizes_um=[50.0, 5.0, 20.0])
        assert r.particle_sizes_um == sorted(r.particle_sizes_um)

    def test_dissolved_pct_list_length_matches_sizes(self):
        r = _default_opt(particle_sizes_um=[5.0, 10.0, 50.0])
        assert len(r.dissolved_pct_at_60min) == 3

    def test_notes_is_list(self):
        r = _default_opt()
        assert isinstance(r.notes, list)

    def test_optimal_size_in_candidate_list(self):
        r = _default_opt()
        assert r.optimal_size_um in r.particle_sizes_um


class TestPhysicalPlausibility:
    def test_dissolved_pct_between_0_and_100(self):
        r = _default_opt()
        assert all(0.0 <= p <= 100.0 for p in r.dissolved_pct_at_60min)

    def test_optimal_dissolved_pct_positive(self):
        r = _default_opt()
        assert r.optimal_dissolved_pct >= 0.0

    def test_smaller_particle_dissolves_more(self):
        r = _default_opt(particle_sizes_um=[5.0, 100.0])
        diss_small = r.dissolved_pct_at_60min[0]
        diss_large = r.dissolved_pct_at_60min[1]
        assert diss_small >= diss_large - 1e-6

    def test_higher_solubility_faster_dissolution(self):
        r_low = _default_opt(solubility_mg_mL=0.01, particle_sizes_um=[10.0])
        r_high = _default_opt(solubility_mg_mL=10.0, particle_sizes_um=[10.0])
        assert r_high.dissolved_pct_at_60min[0] >= r_low.dissolved_pct_at_60min[0]

    def test_single_size_returns_that_as_optimal(self):
        r = _default_opt(particle_sizes_um=[10.0])
        assert r.optimal_size_um == 10.0

    def test_highly_soluble_drug_achieves_high_dissolution(self):
        r = _default_opt(
            solubility_mg_mL=100.0,
            particle_sizes_um=[1.0, 5.0],
            dose_mg=10.0,
        )
        assert max(r.dissolved_pct_at_60min) > 50.0

    def test_optimal_pct_matches_entry_in_list(self):
        r = _default_opt()
        idx = r.particle_sizes_um.index(r.optimal_size_um)
        assert abs(r.optimal_dissolved_pct - r.dissolved_pct_at_60min[idx]) < 1e-6

    def test_note_added_when_no_size_meets_target(self):
        r = _default_opt(
            solubility_mg_mL=0.0001,
            target_dissolved_pct_at_60min=99.0,
            dose_mg=500.0,
        )
        assert any("No particle size" in note for note in r.notes)

    def test_rapid_dissolution_note_when_high_pct(self):
        r = _default_opt(
            solubility_mg_mL=200.0,
            dose_mg=1.0,
            particle_sizes_um=[1.0],
            target_dissolved_pct_at_60min=85.0,
        )
        if r.optimal_dissolved_pct >= 85.0:
            assert any("rapid dissolution" in note.lower() for note in r.notes)

    def test_more_sizes_more_dissolved_entries(self):
        r = _default_opt(particle_sizes_um=[5.0, 10.0, 20.0, 50.0])
        assert len(r.dissolved_pct_at_60min) == 4

    def test_dose_number_note_for_poorly_soluble(self):
        r = _default_opt(
            solubility_mg_mL=0.01,
            dose_mg=500.0,
            particle_sizes_um=[10.0],
        )
        assert any("dose number" in note.lower() for note in r.notes)


class TestValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            _default_opt(drug_name="")

    def test_negative_target_raises(self):
        with pytest.raises(ValueError):
            _default_opt(target_dissolved_pct_at_60min=-5.0)

    def test_target_gt_100_raises(self):
        with pytest.raises(ValueError):
            _default_opt(target_dissolved_pct_at_60min=101.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            _default_opt(solubility_mg_mL=0.0)

    def test_negative_diffusion_coeff_raises(self):
        with pytest.raises(ValueError):
            _default_opt(diffusion_coeff_cm2_s=-1e-6)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _default_opt(dose_mg=0.0)

    def test_zero_density_raises(self):
        with pytest.raises(ValueError):
            _default_opt(density_g_cm3=0.0)

    def test_empty_particle_sizes_raises(self):
        with pytest.raises(ValueError):
            _default_opt(particle_sizes_um=[])

    def test_negative_particle_size_raises(self):
        with pytest.raises(ValueError):
            _default_opt(particle_sizes_um=[-5.0, 10.0])

    def test_zero_media_volume_raises(self):
        with pytest.raises(ValueError):
            _default_opt(media_volume_mL=0.0)


class TestBiopharmaceuticsClassification:
    def test_class_i_high_both(self):
        assert biopharmaceutics_classification(90.0, 0.90) == "BCS Class I"

    def test_class_ii_low_dissolution(self):
        assert biopharmaceutics_classification(60.0, 0.90) == "BCS Class II"

    def test_class_iii_low_permeability(self):
        assert biopharmaceutics_classification(90.0, 0.50) == "BCS Class III"

    def test_class_iv_both_low(self):
        assert biopharmaceutics_classification(60.0, 0.50) == "BCS Class IV"

    def test_boundary_85pct_dissolution(self):
        assert biopharmaceutics_classification(85.0, 0.90) == "BCS Class II"

    def test_boundary_85_1_pct_dissolution(self):
        assert biopharmaceutics_classification(85.01, 0.90) == "BCS Class I"

    def test_boundary_permeability_085(self):
        assert biopharmaceutics_classification(90.0, 0.85) == "BCS Class I"

    def test_negative_dissolved_pct_raises(self):
        with pytest.raises(ValueError):
            biopharmaceutics_classification(-1.0, 0.9)

    def test_dissolved_pct_gt_100_raises(self):
        with pytest.raises(ValueError):
            biopharmaceutics_classification(101.0, 0.9)

    def test_negative_fa_raises(self):
        with pytest.raises(ValueError):
            biopharmaceutics_classification(80.0, -0.1)

    def test_fa_gt_1_raises(self):
        with pytest.raises(ValueError):
            biopharmaceutics_classification(80.0, 1.1)
