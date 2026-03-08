"""Tests for Phase 600 — Drug Nanocrystal Dissolution."""

import pytest

from omega_pbpk.core.nanocrystal_dissolution_p600 import (
    NanocrystalDissolutionResult,
    simulate_nanocrystal_dissolution,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_result():
    return simulate_nanocrystal_dissolution(
        drug_name="NanoDrug",
        dose_mg=10.0,
        initial_radius_nm=200.0,
        bulk_solubility_mg_mL=0.1,
    )


@pytest.fixture
def small_nano():
    return simulate_nanocrystal_dissolution(
        drug_name="NanoDrug",
        dose_mg=10.0,
        initial_radius_nm=50.0,
        bulk_solubility_mg_mL=0.1,
    )


@pytest.fixture
def large_nano():
    return simulate_nanocrystal_dissolution(
        drug_name="NanoDrug",
        dose_mg=10.0,
        initial_radius_nm=5000.0,
        bulk_solubility_mg_mL=0.1,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self, base_result):
        assert isinstance(base_result, NanocrystalDissolutionResult)

    def test_times_is_list(self, base_result):
        assert isinstance(base_result.times_min, list)

    def test_radius_is_list(self, base_result):
        assert isinstance(base_result.particle_radius_nm, list)

    def test_c_dissolved_is_list(self, base_result):
        assert isinstance(base_result.c_dissolved_mg_mL, list)

    def test_supersaturation_is_list(self, base_result):
        assert isinstance(base_result.c_supersaturation, list)

    def test_fraction_is_list(self, base_result):
        assert isinstance(base_result.fraction_dissolved, list)

    def test_drug_name_stored(self, base_result):
        assert base_result.drug_name == "NanoDrug"

    def test_initial_radius_stored(self, base_result):
        assert base_result.initial_particle_radius_nm == 200.0


# ---------------------------------------------------------------------------
# List lengths
# ---------------------------------------------------------------------------


class TestListLengths:
    def test_all_lists_same_length(self, base_result):
        n = len(base_result.times_min)
        assert len(base_result.particle_radius_nm) == n
        assert len(base_result.c_dissolved_mg_mL) == n
        assert len(base_result.c_supersaturation) == n
        assert len(base_result.fraction_dissolved) == n

    def test_times_starts_at_zero(self, base_result):
        assert base_result.times_min[0] == pytest.approx(0.0, abs=1e-10)

    def test_times_ends_at_t_end(self):
        r = simulate_nanocrystal_dissolution("D", 10.0, 200.0, 0.1, t_end_min=60.0, dt_min=1.0)
        assert r.times_min[-1] == pytest.approx(60.0, abs=1.5)


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------


class TestPhysicalConstraints:
    def test_radius_nonnegative(self, base_result):
        assert all(r >= 0 for r in base_result.particle_radius_nm)

    def test_radius_nonincreasing(self, base_result):
        rad = base_result.particle_radius_nm
        for i in range(1, len(rad)):
            assert rad[i] <= rad[i - 1] + 1e-10

    def test_c_dissolved_nonnegative(self, base_result):
        assert all(c >= 0 for c in base_result.c_dissolved_mg_mL)

    def test_fraction_in_range(self, base_result):
        for f in base_result.fraction_dissolved:
            assert -1e-9 <= f <= 1.0 + 1e-9

    def test_fraction_starts_at_zero(self, base_result):
        assert base_result.fraction_dissolved[0] == pytest.approx(0.0, abs=1e-10)

    def test_supersaturation_nonnegative(self, base_result):
        assert all(s >= 0 for s in base_result.c_supersaturation)


# ---------------------------------------------------------------------------
# Smaller particles dissolve faster
# ---------------------------------------------------------------------------


class TestSizeEffect:
    def test_smaller_particles_dissolve_faster(self, small_nano, large_nano):
        # Smaller particles should have lower t_50 or t_90
        assert small_nano.t_90pct_dissolved_min <= large_nano.t_90pct_dissolved_min

    def test_smaller_particles_more_dissolved_at_early_time(self, small_nano, large_nano):
        # At midpoint, small nanocrystals should have higher fraction
        mid_idx = len(small_nano.fraction_dissolved) // 4
        if mid_idx > 0:
            assert small_nano.fraction_dissolved[mid_idx] >= large_nano.fraction_dissolved[mid_idx]

    def test_enhancement_ratio_gt_one_for_small_particles(self, small_nano):
        assert small_nano.dissolution_rate_enhancement >= 1.0


# ---------------------------------------------------------------------------
# Supersaturation
# ---------------------------------------------------------------------------


class TestSupersaturation:
    def test_peak_supersaturation_positive(self, base_result):
        assert base_result.peak_supersaturation > 0

    def test_supersaturation_consistent_with_list(self, base_result):
        assert base_result.peak_supersaturation == pytest.approx(
            max(base_result.c_supersaturation), abs=1e-12
        )

    def test_supersaturation_increases_with_smaller_particles(self, small_nano, large_nano):
        # Smaller particles can create more supersaturation due to Ostwald-Freundlich
        assert small_nano.peak_supersaturation >= large_nano.peak_supersaturation


# ---------------------------------------------------------------------------
# t_50 and t_90
# ---------------------------------------------------------------------------


class TestDissolutionTimes:
    def test_t50_less_than_t90(self, base_result):
        assert base_result.t_50pct_dissolved_min <= base_result.t_90pct_dissolved_min

    def test_t50_nonnegative(self, base_result):
        assert base_result.t_50pct_dissolved_min >= 0

    def test_t90_at_most_t_end(self, base_result):
        assert base_result.t_90pct_dissolved_min <= 120.0 + 0.5

    def test_t50_smaller_for_small_particles(self, small_nano, large_nano):
        assert small_nano.t_50pct_dissolved_min <= large_nano.t_50pct_dissolved_min


# ---------------------------------------------------------------------------
# Enhancement ratio
# ---------------------------------------------------------------------------


class TestEnhancementRatio:
    def test_enhancement_nonnegative(self, base_result):
        assert base_result.dissolution_rate_enhancement >= 0

    def test_enhancement_greater_for_smaller_particles(self, small_nano, large_nano):
        assert small_nano.dissolution_rate_enhancement >= large_nano.dissolution_rate_enhancement


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


class TestAUC:
    def test_auc_positive(self, base_result):
        assert base_result.auc_dissolved_mg_h_per_mL > 0

    def test_auc_in_mg_h_per_mL_units(self, base_result):
        # AUC should be in mg·h/mL — a reasonable range for dose 10 mg in 250 mL
        # max concentration = 10/250 = 0.04 mg/mL; over 2h → AUC < ~0.08 mg·h/mL
        assert base_result.auc_dissolved_mg_h_per_mL < 0.1  # sanity check

    def test_higher_solubility_higher_auc(self):
        r1 = simulate_nanocrystal_dissolution("D", 10.0, 200.0, 0.05)
        r2 = simulate_nanocrystal_dissolution("D", 10.0, 200.0, 0.5)
        assert r2.auc_dissolved_mg_h_per_mL >= r1.auc_dissolved_mg_h_per_mL


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_radius_raises(self):
        with pytest.raises(ValueError, match="initial_radius_nm"):
            simulate_nanocrystal_dissolution("D", 10.0, 0.0, 0.1)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="initial_radius_nm"):
            simulate_nanocrystal_dissolution("D", 10.0, -100.0, 0.1)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_nanocrystal_dissolution("D", 0.0, 200.0, 0.1)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError, match="bulk_solubility"):
            simulate_nanocrystal_dissolution("D", 10.0, 200.0, 0.0)

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError, match="volume_fluid"):
            simulate_nanocrystal_dissolution("D", 10.0, 200.0, 0.1, volume_fluid_mL=0.0)

    def test_notes_is_string(self, base_result):
        assert isinstance(base_result.notes, str)
        assert len(base_result.notes) > 0
