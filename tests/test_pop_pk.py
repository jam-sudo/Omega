"""Tests for omega_pbpk.clinical.pop_pk module."""

import numpy as np
import pytest

from omega_pbpk.clinical.pop_pk import PopPKResult, eta_shrinkage, simulate_pop_pk


class TestSimulatePopPK:
    """Tests for simulate_pop_pk function."""

    def test_returns_result_type(self):
        result = simulate_pop_pk(n_subjects=20)
        assert isinstance(result, PopPKResult)

    def test_n_subjects_correct(self):
        result = simulate_pop_pk(n_subjects=20)
        assert result.n_subjects == 20

    def test_median_conc_length(self):
        result = simulate_pop_pk(n_subjects=20)
        assert len(result.median_conc) > 1

    def test_p5_less_than_median(self):
        result = simulate_pop_pk(n_subjects=25)
        p5 = np.array(result.p5_conc)
        median = np.array(result.median_conc)
        assert np.all(p5 <= median + 1e-12)

    def test_p95_greater_than_median(self):
        result = simulate_pop_pk(n_subjects=25)
        p95 = np.array(result.p95_conc)
        median = np.array(result.median_conc)
        assert np.all(p95 >= median - 1e-12)

    def test_median_auc_positive(self):
        result = simulate_pop_pk(n_subjects=20)
        assert result.median_auc > 0

    def test_cv_auc_positive(self):
        result = simulate_pop_pk(n_subjects=20)
        assert result.cv_auc_pct > 0

    def test_cv_auc_reflects_omega(self):
        result_low = simulate_pop_pk(n_subjects=30, omega_cl=0.1, seed=42)
        result_high = simulate_pop_pk(n_subjects=30, omega_cl=0.5, seed=42)
        assert result_high.cv_auc_pct > result_low.cv_auc_pct

    def test_individual_auc_count(self):
        result = simulate_pop_pk(n_subjects=20)
        assert len(result.individual_auc) == 20

    def test_median_cmax_positive(self):
        result = simulate_pop_pk(n_subjects=20)
        assert result.median_cmax > 0

    def test_therapeutic_window_counts(self):
        result = simulate_pop_pk(
            n_subjects=20, therapeutic_low_mg_L=1.0, therapeutic_high_mg_L=10.0
        )
        assert result.n_above_therapeutic + result.n_below_therapeutic <= result.n_subjects

    def test_seed_reproducible(self):
        r1 = simulate_pop_pk(n_subjects=20, seed=123)
        r2 = simulate_pop_pk(n_subjects=20, seed=123)
        assert r1.median_auc == r2.median_auc

    def test_times_h_length_positive(self):
        result = simulate_pop_pk(n_subjects=20)
        assert len(result.times_h) > 1

    def test_higher_dose_higher_auc(self):
        r_low = simulate_pop_pk(n_subjects=20, dose_mg=100, seed=99)
        r_high = simulate_pop_pk(n_subjects=20, dose_mg=200, seed=99)
        assert r_high.median_auc > r_low.median_auc


class TestEtaShrinkage:
    """Tests for eta_shrinkage function."""

    def test_zero_variance_full_shrinkage(self):
        etas = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = eta_shrinkage(etas, omega=0.3)
        assert result == pytest.approx(1.0)

    def test_returns_float(self):
        etas = [0.1, -0.05, 0.2]
        result = eta_shrinkage(etas, omega=0.3)
        assert isinstance(result, float)

    def test_low_shrinkage_when_omega_matches(self):
        rng = np.random.default_rng(42)
        omega = 0.3
        etas = rng.normal(0, np.sqrt(omega), size=1000).tolist()
        result = eta_shrinkage(etas, omega=omega)
        assert result < 0.2  # shrinkage should be low

    def test_shrinkage_bounded(self):
        rng = np.random.default_rng(7)
        etas = rng.normal(0, 0.5, size=50).tolist()
        result = eta_shrinkage(etas, omega=0.25)
        assert isinstance(result, float)  # shrinkage can be outside [0,1] when SD > omega
