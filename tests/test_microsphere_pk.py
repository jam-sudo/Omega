"""Tests for Phase 250: Microsphere depot PK model."""

import pytest

from omega_pbpk.core.microsphere_pk import (
    MicrospherePKResult,
    compare_polymers,
    simulate_microsphere_pk,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_microsphere_pk_result(self):
        result = simulate_microsphere_pk("DrugA", dose_mg=100.0, polymer_type="PLGA_fast")
        assert isinstance(result, MicrospherePKResult)

    def test_drug_name_stored(self):
        result = simulate_microsphere_pk("Leuprolide", dose_mg=10.0, polymer_type="PLGA_slow")
        assert result.drug_name == "Leuprolide"

    def test_polymer_type_stored(self):
        result = simulate_microsphere_pk("Drug", dose_mg=50.0, polymer_type="PLA")
        assert result.polymer_type == "PLA"

    def test_dose_mg_stored(self):
        result = simulate_microsphere_pk("Drug", dose_mg=75.0, polymer_type="chitosan")
        assert result.dose_mg == 75.0


# ---------------------------------------------------------------------------
# Burst effect: plasma starts positive
# ---------------------------------------------------------------------------


class TestBurstEffect:
    def test_plasma_starts_positive_plga_fast(self):
        result = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="PLGA_fast")
        assert result.c_plasma_mg_L[0] > 0.0

    def test_plasma_starts_positive_chitosan(self):
        result = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="chitosan")
        assert result.c_plasma_mg_L[0] > 0.0

    def test_plasma_starts_positive_pla(self):
        # PLA burst_fraction=0.05, still > 0
        result = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="PLA")
        assert result.c_plasma_mg_L[0] > 0.0


# ---------------------------------------------------------------------------
# AUC and Cmax tests
# ---------------------------------------------------------------------------


class TestAUCAndCmax:
    def test_auc_positive(self):
        result = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="PLGA_fast")
        assert result.auc_plasma_mg_h_per_L > 0.0

    def test_cmax_positive(self):
        result = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="PLGA_slow")
        assert result.cmax_plasma_mg_L > 0.0

    def test_higher_dose_higher_auc(self):
        r_low = simulate_microsphere_pk("Drug", dose_mg=50.0, polymer_type="PLGA_fast")
        r_high = simulate_microsphere_pk("Drug", dose_mg=200.0, polymer_type="PLGA_fast")
        assert r_high.auc_plasma_mg_h_per_L > r_low.auc_plasma_mg_h_per_L

    def test_cmax_equals_max_of_plasma_list(self):
        result = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="albumin")
        assert abs(result.cmax_plasma_mg_L - max(result.c_plasma_mg_L)) < 1e-9


# ---------------------------------------------------------------------------
# Release timing tests
# ---------------------------------------------------------------------------


class TestReleaseTiming:
    def test_t50_less_than_t90(self):
        result = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="PLGA_fast")
        assert result.t50_release_h <= result.t90_release_h

    def test_faster_polymer_shorter_t50(self):
        r_fast = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="albumin")
        r_slow = simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="PLA")
        assert r_fast.t50_release_h <= r_slow.t50_release_h

    def test_particle_size_effect_smaller_faster(self):
        r_small = simulate_microsphere_pk(
            "Drug", dose_mg=100.0, polymer_type="PLGA_slow", particle_size_um=10.0
        )
        r_large = simulate_microsphere_pk(
            "Drug", dose_mg=100.0, polymer_type="PLGA_slow", particle_size_um=100.0
        )
        assert r_small.t50_release_h <= r_large.t50_release_h


# ---------------------------------------------------------------------------
# Duration of action tests
# ---------------------------------------------------------------------------


class TestDurationOfAction:
    def test_duration_non_negative(self):
        result = simulate_microsphere_pk(
            "Drug", dose_mg=100.0, polymer_type="PLGA_fast", mic_threshold_mg_L=0.1
        )
        assert result.duration_of_action_h >= 0.0

    def test_lower_mic_longer_duration(self):
        r_low_mic = simulate_microsphere_pk(
            "Drug", dose_mg=100.0, polymer_type="PLGA_slow", mic_threshold_mg_L=0.01
        )
        r_high_mic = simulate_microsphere_pk(
            "Drug", dose_mg=100.0, polymer_type="PLGA_slow", mic_threshold_mg_L=1.0
        )
        assert r_low_mic.duration_of_action_h >= r_high_mic.duration_of_action_h


# ---------------------------------------------------------------------------
# Compare polymers tests
# ---------------------------------------------------------------------------


class TestComparePolymers:
    def test_compare_returns_list(self):
        results = compare_polymers("Drug", dose_mg=100.0)
        assert isinstance(results, list)

    def test_compare_returns_five_polymers(self):
        results = compare_polymers("Drug", dose_mg=100.0)
        assert len(results) == 5

    def test_compare_sorted_by_duration_descending(self):
        results = compare_polymers("Drug", dose_mg=100.0)
        durations = [r.duration_of_action_h for r in results]
        assert durations == sorted(durations, reverse=True)

    def test_compare_all_result_types(self):
        results = compare_polymers("Drug", dose_mg=100.0)
        for r in results:
            assert isinstance(r, MicrospherePKResult)

    def test_compare_all_polymer_types_represented(self):
        results = compare_polymers("Drug", dose_mg=100.0)
        polymer_names = {r.polymer_type for r in results}
        assert "PLGA_fast" in polymer_names
        assert "PLA" in polymer_names
        assert "albumin" in polymer_names


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_microsphere_pk("Drug", dose_mg=-10.0, polymer_type="PLGA_fast")

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_microsphere_pk("Drug", dose_mg=0.0, polymer_type="PLGA_fast")

    def test_invalid_polymer_raises(self):
        with pytest.raises(ValueError, match="polymer_type"):
            simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="unknown_polymer")

    def test_zero_clearance_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="PLGA_fast", cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_microsphere_pk("Drug", dose_mg=100.0, polymer_type="PLGA_fast", vd_L=0.0)
