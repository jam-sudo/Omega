"""Tests for Phase 573 - Drug Tissue Binding Kinetics."""

import pytest

from omega_pbpk.core.tissue_binding_kinetics_p573 import (
    TissueBindingKineticsResult,
    compare_tissues,
    simulate_tissue_binding,
)


def _default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        tissue="liver",
        bmax_mg_L=10.0,
        kd_mg_L=1.0,
        kon_per_h_per_mg_L=0.5,
        initial_free_tissue_mg_L=5.0,
    )
    params.update(kwargs)
    return simulate_tissue_binding(**params)


class TestReturnTypeAndStructure:
    def test_return_type(self):
        r = _default_result()
        assert isinstance(r, TissueBindingKineticsResult)

    def test_list_lengths_match(self):
        r = _default_result()
        n = len(r.times_h)
        assert len(r.c_free_tissue_mg_L) == n
        assert len(r.c_bound_tissue_mg_L) == n
        assert len(r.c_total_tissue_mg_L) == n

    def test_times_start_at_zero(self):
        r = _default_result()
        assert r.times_h[0] == 0.0

    def test_times_end_near_t_end(self):
        r = _default_result(t_end_h=12.0)
        assert r.times_h[-1] >= 11.9


class TestBindingDynamics:
    def test_c_free_decreases(self):
        r = _default_result()
        assert r.c_free_tissue_mg_L[-1] < r.c_free_tissue_mg_L[0]

    def test_c_bound_increases(self):
        r = _default_result()
        assert r.c_bound_tissue_mg_L[-1] > r.c_bound_tissue_mg_L[0]

    def test_c_bound_starts_at_zero(self):
        r = _default_result()
        assert r.c_bound_tissue_mg_L[0] == 0.0

    def test_c_total_equals_free_plus_bound(self):
        r = _default_result()
        for i in range(len(r.times_h)):
            expected = r.c_free_tissue_mg_L[i] + r.c_bound_tissue_mg_L[i]
            assert abs(r.c_total_tissue_mg_L[i] - expected) < 1e-9

    def test_c_bound_plateaus(self):
        """C_bound should plateau: late values should be close together."""
        r = _default_result(t_end_h=48.0)
        n = len(r.c_bound_tissue_mg_L)
        last_quarter = r.c_bound_tissue_mg_L[3 * n // 4 :]
        spread = max(last_quarter) - min(last_quarter)
        assert spread < 0.5  # near plateau


class TestBmaxEffect:
    def test_larger_bmax_higher_f_bound(self):
        r_low = _default_result(bmax_mg_L=5.0)
        r_high = _default_result(bmax_mg_L=20.0)
        assert r_high.f_bound_at_equilibrium > r_low.f_bound_at_equilibrium


class TestKonEffect:
    def test_higher_kon_shorter_t_half(self):
        r_slow = _default_result(kon_per_h_per_mg_L=0.1)
        r_fast = _default_result(kon_per_h_per_mg_L=2.0)
        assert r_fast.t_half_binding_h < r_slow.t_half_binding_h


class TestMetrics:
    def test_t_half_binding_positive(self):
        r = _default_result()
        assert r.t_half_binding_h > 0

    def test_f_bound_in_range(self):
        r = _default_result()
        assert 0.0 <= r.f_bound_at_equilibrium <= 1.0

    def test_tissue_accumulation_ratio_positive(self):
        r = _default_result()
        assert r.tissue_accumulation_ratio > 0

    def test_kd_stored(self):
        r = _default_result(kd_mg_L=2.5)
        assert r.kd_equilibrium_mg_L == 2.5


class TestBindingClass:
    def test_rapid_class(self):
        r = _default_result(kon_per_h_per_mg_L=5.0, bmax_mg_L=5.0, kd_mg_L=1.0)
        assert r.binding_class == "rapid"

    def test_slow_class(self):
        r = _default_result(
            kon_per_h_per_mg_L=0.001,
            bmax_mg_L=100.0,
            kd_mg_L=0.5,
            initial_free_tissue_mg_L=0.5,
            t_end_h=72.0,
        )
        assert r.binding_class == "slow"

    def test_binding_class_valid_string(self):
        r = _default_result()
        assert r.binding_class in ("rapid", "intermediate", "slow")


class TestCompareTissues:
    def test_compare_returns_list(self):
        results = compare_tissues("DrugA", 10.0, 1.0, 5.0)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_compare_sorted_descending(self):
        results = compare_tissues("DrugA", 10.0, 1.0, 5.0)
        f_bounds = [r.f_bound_at_equilibrium for r in results]
        for i in range(1, len(f_bounds)):
            assert f_bounds[i] <= f_bounds[i - 1]


class TestValidation:
    def test_bmax_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(bmax_mg_L=0)

    def test_kd_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(kd_mg_L=-1)

    def test_kon_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(kon_per_h_per_mg_L=0)

    def test_initial_c_must_be_positive(self):
        with pytest.raises(ValueError):
            _default_result(initial_free_tissue_mg_L=-0.5)
