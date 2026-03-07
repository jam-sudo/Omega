"""Tests for Phase 290 — Drug Release from Nanoparticles."""

import pytest

from omega_pbpk.prediction.nanoparticle_drug_release import (
    VALID_MECHANISMS,
    NanoparticleReleaseResult,
    compare_mechanisms,
    simulate_nanoparticle_release,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

VALID_RELEASE_CLASSES = {"fast", "sustained", "extended"}


def make_result(**kwargs) -> NanoparticleReleaseResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        particle_diameter_nm=200.0,
        drug_loading_pct=20.0,
        release_mechanism="diffusion",
        k_release_per_h=0.2,
        burst_fraction=0.1,
        t_end_h=24.0,
        dt_h=0.1,
    )
    defaults.update(kwargs)
    return simulate_nanoparticle_release(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_nanoparticle_release_result(self):
        result = make_result()
        assert isinstance(result, NanoparticleReleaseResult)

    def test_drug_name_preserved(self):
        result = make_result(drug_name="Paclitaxel")
        assert result.drug_name == "Paclitaxel"

    def test_dose_mg_preserved(self):
        result = make_result(dose_mg=5.0)
        assert result.dose_mg == pytest.approx(5.0)

    def test_particle_diameter_preserved(self):
        result = make_result(particle_diameter_nm=100.0)
        assert result.particle_diameter_nm == pytest.approx(100.0)

    def test_drug_loading_preserved(self):
        result = make_result(drug_loading_pct=30.0)
        assert result.drug_loading_pct == pytest.approx(30.0)

    def test_release_mechanism_preserved(self):
        result = make_result(release_mechanism="erosion")
        assert result.release_mechanism == "erosion"

    def test_burst_fraction_preserved(self):
        result = make_result(burst_fraction=0.2)
        assert result.burst_fraction == pytest.approx(0.2)

    def test_times_and_pct_same_length(self):
        result = make_result()
        assert len(result.times_h) == len(result.cumulative_release_pct)

    def test_times_starts_at_zero(self):
        result = make_result()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_ends_at_t_end(self):
        result = make_result(t_end_h=12.0)
        assert result.times_h[-1] == pytest.approx(12.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Release profile physics
# ---------------------------------------------------------------------------


class TestReleasePropfile:
    def test_cumulative_release_starts_at_least_burst_pct(self):
        burst = 0.2
        result = make_result(burst_fraction=burst)
        # First value should equal burst*100
        assert result.cumulative_release_pct[0] == pytest.approx(burst * 100.0, rel=1e-6)

    def test_zero_burst_starts_at_zero(self):
        result = make_result(burst_fraction=0.0)
        assert result.cumulative_release_pct[0] == pytest.approx(0.0, abs=1e-9)

    def test_monotonically_nondecreasing(self):
        result = make_result()
        pcts = result.cumulative_release_pct
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1] - 1e-9

    def test_cumulative_release_bounded_100(self):
        result = make_result(k_release_per_h=5.0)
        assert all(pct <= 100.0 + 1e-9 for pct in result.cumulative_release_pct)

    def test_total_released_pct_positive(self):
        result = make_result()
        assert result.total_released_pct > 0

    def test_total_released_pct_matches_last_value(self):
        result = make_result()
        assert result.total_released_pct == pytest.approx(
            result.cumulative_release_pct[-1], rel=1e-9
        )


# ---------------------------------------------------------------------------
# t50 and t80
# ---------------------------------------------------------------------------


class TestTimpoints:
    def test_t50_positive_when_reached(self):
        result = make_result(k_release_per_h=0.5, t_end_h=24.0)
        if result.t50_h is not None:
            assert result.t50_h > 0

    def test_t50_within_simulation_when_reached(self):
        result = make_result(k_release_per_h=0.5, t_end_h=24.0)
        if result.t50_h is not None:
            assert result.t50_h <= 24.0 + 1e-9

    def test_t80_after_t50(self):
        result = make_result(k_release_per_h=0.5, t_end_h=24.0)
        if result.t50_h is not None and result.t80_h is not None:
            assert result.t80_h >= result.t50_h

    def test_higher_k_release_gives_earlier_t50(self):
        r_slow = make_result(k_release_per_h=0.1, burst_fraction=0.0)
        r_fast = make_result(k_release_per_h=0.8, burst_fraction=0.0)
        if r_slow.t50_h is not None and r_fast.t50_h is not None:
            assert r_fast.t50_h < r_slow.t50_h

    def test_higher_burst_gives_earlier_t50(self):
        r_low_burst = make_result(burst_fraction=0.0, k_release_per_h=0.3)
        r_high_burst = make_result(burst_fraction=0.4, k_release_per_h=0.3)
        if r_low_burst.t50_h is not None and r_high_burst.t50_h is not None:
            assert r_high_burst.t50_h <= r_low_burst.t50_h

    def test_burst_above_50pct_means_t50_is_zero(self):
        # burst_fraction=0.6 → immediately 60% released → t50 = 0
        result = make_result(burst_fraction=0.6)
        # Note: burst_fraction < 1 is validated; 0.6 is valid
        # t50 should be at time=0 (first timepoint)
        assert result.t50_h == pytest.approx(0.0, abs=0.2)


# ---------------------------------------------------------------------------
# Release class
# ---------------------------------------------------------------------------


class TestReleaseClass:
    def test_release_class_valid(self):
        for mech in VALID_MECHANISMS:
            result = make_result(release_mechanism=mech)
            assert result.release_class in VALID_RELEASE_CLASSES

    def test_fast_release_class_for_high_k(self):
        result = make_result(k_release_per_h=5.0, burst_fraction=0.0)
        assert result.release_class == "fast"

    def test_extended_class_for_very_low_k(self):
        result = make_result(k_release_per_h=0.01, burst_fraction=0.0, t_end_h=48.0)
        assert result.release_class in {"sustained", "extended"}


# ---------------------------------------------------------------------------
# compare_mechanisms
# ---------------------------------------------------------------------------


class TestCompareMechanisms:
    def test_returns_three_results(self):
        results = compare_mechanisms("TestDrug", 10.0, 200.0, 20.0, 0.3)
        assert len(results) == 3

    def test_all_mechanisms_present(self):
        results = compare_mechanisms("TestDrug", 10.0, 200.0, 20.0, 0.3)
        mechs = {r.release_mechanism for r in results}
        assert mechs == VALID_MECHANISMS

    def test_sorted_by_t50_ascending(self):
        results = compare_mechanisms("TestDrug", 10.0, 200.0, 20.0, 0.3)
        t50s = [r.t50_h if r.t50_h is not None else float("inf") for r in results]
        assert t50s == sorted(t50s)

    def test_first_is_fastest(self):
        results = compare_mechanisms("TestDrug", 10.0, 200.0, 20.0, 0.3)
        first_t50 = results[0].t50_h if results[0].t50_h is not None else float("inf")
        for r in results[1:]:
            other_t50 = r.t50_h if r.t50_h is not None else float("inf")
            assert first_t50 <= other_t50


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=0.0)

    def test_dose_negative(self):
        with pytest.raises(ValueError, match="dose_mg"):
            make_result(dose_mg=-1.0)

    def test_diameter_zero(self):
        with pytest.raises(ValueError, match="particle_diameter_nm"):
            make_result(particle_diameter_nm=0.0)

    def test_diameter_too_large(self):
        with pytest.raises(ValueError, match="particle_diameter_nm"):
            make_result(particle_diameter_nm=1001.0)

    def test_drug_loading_zero(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            make_result(drug_loading_pct=0.0)

    def test_drug_loading_over_100(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            make_result(drug_loading_pct=101.0)

    def test_k_release_zero(self):
        with pytest.raises(ValueError, match="k_release_per_h"):
            make_result(k_release_per_h=0.0)

    def test_burst_fraction_negative(self):
        with pytest.raises(ValueError, match="burst_fraction"):
            make_result(burst_fraction=-0.1)

    def test_burst_fraction_one(self):
        with pytest.raises(ValueError, match="burst_fraction"):
            make_result(burst_fraction=1.0)

    def test_t_end_zero(self):
        with pytest.raises(ValueError, match="t_end_h"):
            make_result(t_end_h=0.0)

    def test_invalid_mechanism(self):
        with pytest.raises(ValueError, match="release_mechanism"):
            make_result(release_mechanism="swelling")


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_string(self):
        result = make_result()
        assert isinstance(result.notes, str)

    def test_notes_nonempty(self):
        result = make_result()
        assert len(result.notes) > 0
