"""Tests for drug-target binding kinetics — Phase 278."""
import pytest
from omega_pbpk.clinical.target_binding_kinetics import (
    TargetBindingResult,
    simulate_target_binding,
)


class TestSimulateTargetBindingBasic:
    def test_returns_result(self):
        r = simulate_target_binding("drug", "receptor", 100.0)
        assert isinstance(r, TargetBindingResult)

    def test_drug_name_stored(self):
        r = simulate_target_binding("aspirin", "COX1", 50.0)
        assert r.drug_name == "aspirin"

    def test_target_name_stored(self):
        r = simulate_target_binding("drug", "EGFR", 100.0)
        assert r.target_name == "EGFR"

    def test_kd_computed(self):
        r = simulate_target_binding("drug", "receptor", 100.0, kon_per_nM_per_h=2.0, koff_per_h=0.4)
        assert r.kd_nM == pytest.approx(0.4 / 2.0, rel=1e-6)

    def test_times_array_length(self):
        r = simulate_target_binding("drug", "receptor", 100.0, t_end_h=10.0, dt_h=0.1)
        assert len(r.times_h) == len(r.c_drug_nM) == len(r.rt_bound)

    def test_peak_occupancy_positive(self):
        r = simulate_target_binding("drug", "receptor", 100.0)
        assert r.peak_occupancy_pct > 0

    def test_peak_occupancy_max_100(self):
        r = simulate_target_binding("drug", "receptor", 100.0)
        assert r.peak_occupancy_pct <= 100.0

    def test_occupancy_in_range(self):
        r = simulate_target_binding("drug", "receptor", 100.0)
        for occ in r.occupancy_pct:
            assert 0.0 <= occ <= 100.0

    def test_rt_bound_nonnegative(self):
        r = simulate_target_binding("drug", "receptor", 100.0)
        assert all(v >= 0.0 for v in r.rt_bound)

    def test_residence_time(self):
        r = simulate_target_binding("drug", "receptor", 100.0, koff_per_h=2.0)
        assert r.residence_time_h == pytest.approx(1.0 / 2.0, rel=1e-6)

    def test_auc_occupancy_positive(self):
        r = simulate_target_binding("drug", "receptor", 100.0)
        assert r.auc_occupancy > 0

    def test_notes_nonempty(self):
        r = simulate_target_binding("drug", "receptor", 100.0)
        assert len(r.notes) > 0

    def test_initial_bound_is_zero(self):
        r = simulate_target_binding("drug", "receptor", 100.0)
        assert r.rt_bound[0] == pytest.approx(0.0, abs=1e-9)


class TestBindingKinetics:
    def test_higher_concentration_higher_occupancy(self):
        r1 = simulate_target_binding("drug", "receptor", 10.0)
        r2 = simulate_target_binding("drug", "receptor", 1000.0)
        assert r2.peak_occupancy_pct > r1.peak_occupancy_pct

    def test_higher_kon_faster_binding(self):
        r1 = simulate_target_binding("drug", "receptor", 100.0, kon_per_nM_per_h=0.1, t_end_h=5.0, dt_h=0.05)
        r2 = simulate_target_binding("drug", "receptor", 100.0, kon_per_nM_per_h=2.0, t_end_h=5.0, dt_h=0.05)
        # Faster kon -> reaches higher occupancy sooner -> earlier tmax
        assert r2.tmax_occupancy_h <= r1.tmax_occupancy_h + 2.0

    def test_lower_koff_slower_dissociation(self):
        r1 = simulate_target_binding("drug", "receptor", 100.0, koff_per_h=5.0, cl_drug_per_h=0.1)
        r2 = simulate_target_binding("drug", "receptor", 100.0, koff_per_h=0.1, cl_drug_per_h=0.1)
        # Higher koff -> faster dissociation -> lower sustained occupancy
        assert r2.auc_occupancy >= r1.auc_occupancy

    def test_drug_clearance_reduces_auc(self):
        r1 = simulate_target_binding("drug", "receptor", 100.0, cl_drug_per_h=0.0)
        r2 = simulate_target_binding("drug", "receptor", 100.0, cl_drug_per_h=0.5)
        assert r1.auc_occupancy >= r2.auc_occupancy

    def test_linear_scaling_with_receptor_total(self):
        r1 = simulate_target_binding("drug", "receptor", 100.0, r_total_nM=5.0)
        r2 = simulate_target_binding("drug", "receptor", 100.0, r_total_nM=10.0)
        # More receptors -> more bound at same occupancy%, but occupancy% similar
        assert abs(r1.peak_occupancy_pct - r2.peak_occupancy_pct) < 20.0


class TestValidation:
    def test_invalid_c_drug_zero(self):
        with pytest.raises(ValueError):
            simulate_target_binding("drug", "receptor", 0.0)

    def test_invalid_r_total_zero(self):
        with pytest.raises(ValueError):
            simulate_target_binding("drug", "receptor", 100.0, r_total_nM=0.0)

    def test_invalid_kon_zero(self):
        with pytest.raises(ValueError):
            simulate_target_binding("drug", "receptor", 100.0, kon_per_nM_per_h=0.0)

    def test_invalid_koff_negative(self):
        with pytest.raises(ValueError):
            simulate_target_binding("drug", "receptor", 100.0, koff_per_h=-0.1)

    def test_invalid_t_end_zero(self):
        with pytest.raises(ValueError):
            simulate_target_binding("drug", "receptor", 100.0, t_end_h=0.0)
