"""Tests for population variability simulation (Phase 610)."""

from __future__ import annotations

import math

from omega_pbpk.clinical.population_variability import (
    PopulationVariabilityResult,
    SubjectPKResult,
    lcg_random,
    lognormal_sample,
    simulate_population_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**kw):
    defaults = dict(
        dose_mg=100.0,
        typical_cl_L_per_h=5.0,
        typical_vd_L=30.0,
        n_subjects=100,
        seed=42,
        route="oral",
        t_end_h=24.0,
        dt_h=0.2,
        target_auc_mg_h_L=50.0,
        mic_mg_L=0.5,
    )
    defaults.update(kw)
    return simulate_population_pk(**defaults)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestReturnsResult:
    def test_returns_result(self):
        r = _sim()
        assert isinstance(r, PopulationVariabilityResult)


class TestNSubjects:
    def test_n_subjects_correct(self):
        r = _sim(n_subjects=50)
        assert r.n_subjects == 50

    def test_subjects_tuple_length(self):
        r = _sim(n_subjects=50)
        assert len(r.subjects) == 50


# ---------------------------------------------------------------------------
# Population statistics
# ---------------------------------------------------------------------------


class TestMeans:
    def test_mean_cmax_positive(self):
        r = _sim()
        assert r.mean_cmax > 0

    def test_mean_auc_positive(self):
        r = _sim()
        assert r.mean_auc > 0


class TestCV:
    def test_cv_cmax_in_range(self):
        r = _sim()
        assert 0 < r.cv_cmax_pct < 200


class TestPercentiles:
    def test_p5_lt_mean_lt_p95(self):
        r = _sim()
        assert r.p5_cmax < r.mean_cmax < r.p95_cmax

    def test_p5_auc_lt_mean_auc(self):
        r = _sim()
        assert r.p5_auc < r.mean_auc


# ---------------------------------------------------------------------------
# Fraction statistics
# ---------------------------------------------------------------------------


class TestFractions:
    def test_fraction_above_target_in_0_1(self):
        r = _sim()
        assert 0.0 <= r.fraction_above_target <= 1.0

    def test_fraction_below_mic_in_0_1(self):
        r = _sim()
        assert 0.0 <= r.fraction_below_mic <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_seed_reproducibility(self):
        r1 = _sim(seed=42)
        r2 = _sim(seed=42)
        assert r1.mean_cmax == r2.mean_cmax

    def test_different_seeds_different_results(self):
        r1 = _sim(seed=42)
        r2 = _sim(seed=99)
        assert r1.mean_cmax != r2.mean_cmax


# ---------------------------------------------------------------------------
# Omega effect
# ---------------------------------------------------------------------------


class TestOmegaEffect:
    def test_higher_omega_higher_cv(self):
        r_lo = _sim(omega_cl=0.1)
        r_hi = _sim(omega_cl=0.5)
        assert r_hi.cv_cmax_pct > r_lo.cv_cmax_pct


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_dose_proportionality(self):
        r1 = _sim(dose_mg=100.0)
        r2 = _sim(dose_mg=200.0)
        ratio = r2.mean_cmax / r1.mean_cmax
        assert abs(ratio - 2.0) < 0.1 * 2.0  # within 10%


# ---------------------------------------------------------------------------
# Subject result types
# ---------------------------------------------------------------------------


class TestSubjectResults:
    def test_subject_results_type(self):
        r = _sim()
        for s in r.subjects:
            assert isinstance(s, SubjectPKResult)

    def test_subjects_cl_positive(self):
        r = _sim()
        assert all(s.cl_L_per_h > 0 for s in r.subjects)

    def test_subjects_auc_positive(self):
        r = _sim()
        assert all(s.auc_mg_h_L > 0 for s in r.subjects)


# ---------------------------------------------------------------------------
# LCG random tests
# ---------------------------------------------------------------------------


class TestLCGRandom:
    def test_lcg_random_length(self):
        vals = lcg_random(42, 10)
        assert len(vals) == 10

    def test_lcg_random_in_0_1(self):
        vals = lcg_random(42, 100)
        assert all(0.0 <= v < 1.0 for v in vals)


# ---------------------------------------------------------------------------
# Lognormal sample tests
# ---------------------------------------------------------------------------


class TestLognormalSample:
    def test_lognormal_sample_positive(self):
        for u in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert lognormal_sample(5.0, 0.3, u) > 0

    def test_lognormal_median(self):
        # With many samples, geometric mean should approximate mu
        mu = 5.0
        omega = 0.3
        n = 2000
        seed = 123
        vals = lcg_random(seed, n)
        samples = [lognormal_sample(mu, omega, u) for u in vals]
        # Geometric mean = exp(mean(log(samples)))
        log_mean = sum(math.log(s) for s in samples) / n
        geo_mean = math.exp(log_mean)
        # Should be within 10% of mu
        assert abs(geo_mean - mu) / mu < 0.10


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_nonempty(self):
        r = _sim()
        assert r.notes and len(r.notes) > 0
