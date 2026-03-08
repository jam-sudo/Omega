"""Tests for Phase 396: Drug Release from Polymeric Nanoparticles."""

import pytest

from omega_pbpk.core.polymeric_nanoparticle_release import (
    _POLYMER_PARAMS,
    NanoparticleReleaseResult,
    compare_polymers,
    simulate_nanoparticle_release,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_result(**kwargs):
    defaults = dict(
        drug_name="paclitaxel",
        total_drug_mg=10.0,
        polymer_type="PLGA",
        particle_size_nm=200.0,
        drug_loading_pct=10.0,
        t_end_h=720.0,
        dt_h=1.0,
    )
    defaults.update(kwargs)
    return simulate_nanoparticle_release(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = _base_result()
        assert isinstance(result, NanoparticleReleaseResult)

    def test_field_names_present(self):
        result = _base_result()
        for field in [
            "drug_name",
            "polymer_type",
            "release_mechanism",
            "total_drug_mg",
            "particle_size_nm",
            "times_h",
            "cumulative_release_pct",
            "release_rate_pct_per_h",
            "t50_h",
            "t80_h",
            "burst_fraction_pct",
            "sustained_duration_h",
            "higuchi_fit_r2",
            "korsmeyer_n",
            "release_class",
            "notes",
        ]:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_times_and_release_same_length(self):
        result = _base_result()
        assert len(result.times_h) == len(result.cumulative_release_pct)
        assert len(result.times_h) == len(result.release_rate_pct_per_h)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_cumulative_release_monotonically_nondecreasing(self):
        result = _base_result()
        pcts = result.cumulative_release_pct
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1] - 1e-9, (
                f"Release decreased at step {i}: {pcts[i - 1]:.4f} -> {pcts[i]:.4f}"
            )

    def test_starts_with_burst_at_t0(self):
        result = _base_result()
        assert result.times_h[0] == pytest.approx(0.0)
        assert result.cumulative_release_pct[0] > 0.0


# ---------------------------------------------------------------------------
# Near-complete release
# ---------------------------------------------------------------------------


class TestCompleteRelease:
    @pytest.mark.parametrize("polymer", list(_POLYMER_PARAMS.keys()))
    def test_reaches_high_release_by_end(self, polymer):
        result = simulate_nanoparticle_release(
            drug_name="drug",
            total_drug_mg=5.0,
            polymer_type=polymer,
            t_end_h=2000.0,
            dt_h=1.0,
        )
        assert result.cumulative_release_pct[-1] > 90.0, (
            f"{polymer}: final release {result.cumulative_release_pct[-1]:.1f}%"
        )


# ---------------------------------------------------------------------------
# Polymer comparison (PLGA slower than PEG)
# ---------------------------------------------------------------------------


class TestPolymerComparison:
    def test_peg_faster_than_plga(self):
        plga = _base_result(polymer_type="PLGA")
        peg = _base_result(polymer_type="PEG")
        assert peg.t50_h <= plga.t50_h

    def test_chitosan_faster_than_pla(self):
        chitosan = _base_result(polymer_type="chitosan")
        pla = _base_result(polymer_type="PLA")
        assert chitosan.t50_h <= pla.t50_h


# ---------------------------------------------------------------------------
# Particle size effect
# ---------------------------------------------------------------------------


class TestParticleSizeEffect:
    def test_smaller_particles_faster_release(self):
        small = _base_result(particle_size_nm=50.0)
        large = _base_result(particle_size_nm=500.0)
        assert small.t50_h <= large.t50_h

    def test_larger_particles_larger_t50(self):
        result_200 = _base_result(particle_size_nm=200.0)
        result_400 = _base_result(particle_size_nm=400.0)
        assert result_400.t50_h >= result_200.t50_h


# ---------------------------------------------------------------------------
# Burst fraction
# ---------------------------------------------------------------------------


class TestBurstFraction:
    def test_burst_fraction_pct_positive(self):
        result = _base_result()
        assert result.burst_fraction_pct > 0.0

    def test_burst_fraction_below_100(self):
        result = _base_result()
        assert result.burst_fraction_pct < 100.0

    def test_peg_higher_burst_than_pla(self):
        peg = _base_result(polymer_type="PEG")
        pla = _base_result(polymer_type="PLA")
        assert peg.burst_fraction_pct >= pla.burst_fraction_pct


# ---------------------------------------------------------------------------
# t50 < t80
# ---------------------------------------------------------------------------


class TestTimepoints:
    def test_t50_less_than_t80(self):
        result = _base_result()
        assert result.t50_h < result.t80_h

    def test_t50_positive(self):
        result = _base_result()
        assert result.t50_h > 0.0

    def test_sustained_duration_equals_t80(self):
        result = _base_result()
        assert result.sustained_duration_h == pytest.approx(result.t80_h)


# ---------------------------------------------------------------------------
# compare_polymers
# ---------------------------------------------------------------------------


class TestComparePolymers:
    def test_returns_four_results(self):
        results = compare_polymers("paclitaxel", total_drug_mg=10.0)
        assert len(results) == 4

    def test_all_polymers_represented(self):
        results = compare_polymers("paclitaxel", total_drug_mg=10.0)
        polymer_types = {r.polymer_type for r in results}
        assert polymer_types == set(_POLYMER_PARAMS.keys())

    def test_sorted_by_t50_ascending(self):
        results = compare_polymers("paclitaxel", total_drug_mg=10.0)
        t50_values = [r.t50_h for r in results]
        assert t50_values == sorted(t50_values)

    def test_all_correct_type(self):
        results = compare_polymers("paclitaxel", total_drug_mg=10.0)
        for r in results:
            assert isinstance(r, NanoparticleReleaseResult)


# ---------------------------------------------------------------------------
# Korsmeyer-Peppas n
# ---------------------------------------------------------------------------


class TestKorsmeyerN:
    def test_n_in_valid_range(self):
        for polymer in _POLYMER_PARAMS:
            result = _base_result(polymer_type=polymer)
            assert 0.0 < result.korsmeyer_n <= 1.0, f"{polymer}: korsmeyer_n={result.korsmeyer_n}"

    def test_erosion_polymer_has_high_n(self):
        pla = _base_result(polymer_type="PLA")
        peg = _base_result(polymer_type="PEG")
        assert pla.korsmeyer_n > peg.korsmeyer_n

    def test_n_values_per_mechanism(self):
        peg = _base_result(polymer_type="PEG")  # burst: n=0.3
        chitosan = _base_result(polymer_type="chitosan")  # diffusion: n=0.45
        plga = _base_result(polymer_type="PLGA")  # combined: n=0.6
        pla = _base_result(polymer_type="PLA")  # erosion: n=0.9
        assert peg.korsmeyer_n == pytest.approx(0.3)
        assert chitosan.korsmeyer_n == pytest.approx(0.45)
        assert plga.korsmeyer_n == pytest.approx(0.6)
        assert pla.korsmeyer_n == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Higuchi R2
# ---------------------------------------------------------------------------


class TestHiguchiR2:
    def test_r2_in_range(self):
        for polymer in _POLYMER_PARAMS:
            result = _base_result(polymer_type=polymer)
            assert 0.0 <= result.higuchi_fit_r2 <= 1.0

    def test_diffusion_polymer_highest_r2(self):
        chitosan = _base_result(polymer_type="chitosan")
        peg = _base_result(polymer_type="PEG")
        assert chitosan.higuchi_fit_r2 >= peg.higuchi_fit_r2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_total_drug_zero(self):
        with pytest.raises(ValueError, match="total_drug_mg"):
            _base_result(total_drug_mg=0.0)

    def test_invalid_total_drug_negative(self):
        with pytest.raises(ValueError, match="total_drug_mg"):
            _base_result(total_drug_mg=-1.0)

    def test_invalid_particle_size_zero(self):
        with pytest.raises(ValueError, match="particle_size_nm"):
            _base_result(particle_size_nm=0.0)

    def test_invalid_drug_loading_zero(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            _base_result(drug_loading_pct=0.0)

    def test_invalid_drug_loading_above_100(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            _base_result(drug_loading_pct=101.0)

    def test_invalid_polymer_type(self):
        with pytest.raises(ValueError, match="polymer_type"):
            _base_result(polymer_type="PBCA")

    def test_invalid_t_end_zero(self):
        with pytest.raises(ValueError, match="t_end_h"):
            _base_result(t_end_h=0.0)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_string(self):
        result = _base_result()
        assert isinstance(result.notes, str)

    def test_notes_nonempty(self):
        result = _base_result()
        assert len(result.notes) > 0

    def test_notes_mentions_polymer(self):
        result = _base_result(polymer_type="PLGA")
        assert "PLGA" in result.notes
