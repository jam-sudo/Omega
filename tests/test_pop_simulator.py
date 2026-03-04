"""Tests for PopulationSimulator."""

import numpy as np
import pytest


def _make_drug():
    from omega_pbpk.drugs.midazolam import MIDAZOLAM

    return MIDAZOLAM


@pytest.mark.unit
class TestPopulationSimulator:
    def test_smoke_run(self):
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        sim = PopulationSimulator(drug=_make_drug())
        result = sim.run(n_subjects=3, dose_mg=5.0, route="oral", seed=0)
        assert result.n_subjects >= 1

    def test_cp_matrix_shape(self):
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        sim = PopulationSimulator(drug=_make_drug())
        result = sim.run(n_subjects=3, dose_mg=5.0, n_timepoints=25, seed=0)
        assert result.cp_matrix.shape[1] == 25
        assert result.cp_matrix.shape[0] == result.n_subjects

    def test_cmax_positive(self):
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        sim = PopulationSimulator(drug=_make_drug())
        result = sim.run(n_subjects=3, dose_mg=5.0, seed=0)
        assert np.all(result.cmax_samples > 0)

    def test_cmax_stats_keys(self):
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        sim = PopulationSimulator(drug=_make_drug())
        result = sim.run(n_subjects=3, dose_mg=5.0, seed=0)
        stats = result.cmax_stats()
        assert all(k in stats for k in ["mean", "median", "p5", "p95", "cv_pct"])

    def test_percentile_bounds(self):
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        sim = PopulationSimulator(drug=_make_drug())
        result = sim.run(n_subjects=5, dose_mg=5.0, seed=0)
        p5 = result.percentile_cp(5)
        p95 = result.percentile_cp(95)
        assert np.all(p5 <= p95)

    def test_reproducible_with_seed(self):
        from omega_pbpk.population.pop_simulator import PopulationSimulator

        sim = PopulationSimulator(drug=_make_drug())
        r1 = sim.run(n_subjects=3, dose_mg=5.0, seed=42)
        r2 = sim.run(n_subjects=3, dose_mg=5.0, seed=42)
        np.testing.assert_allclose(r1.cmax_samples, r2.cmax_samples, rtol=1e-6)

    def test_import_from_population(self):
        from omega_pbpk.population import PopulationSimulator

        assert PopulationSimulator is not None
