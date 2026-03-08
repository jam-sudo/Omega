"""Tests for phase 672 — Pharmacokinetic Variability Simulator."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.pk_variability_p672 import (
    PKVariabilityResult,
    simulate_pk_variability,
)

# ── helpers ──────────────────────────────────────────────────────────────


def _default(**kw):
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        mean_cl_L_per_h=5.0,
        cv_cl_pct=30.0,
        mean_vd_L=50.0,
        cv_vd_pct=25.0,
        n_subjects=500,
        seed=42,
    )
    defaults.update(kw)
    return simulate_pk_variability(**defaults)


# ── type & immutability ─────────────────────────────────────────────────


class TestResultType:
    def test_returns_correct_type(self):
        r = _default()
        assert isinstance(r, PKVariabilityResult)

    def test_frozen_raises_on_mutation(self):
        r = _default()
        with pytest.raises(AttributeError):
            r.drug_name = "X"  # type: ignore[misc]

    def test_frozen_raises_on_new_field(self):
        r = _default()
        with pytest.raises(AttributeError):
            r.new_field = 42  # type: ignore[attr-defined]


# ── basic properties ────────────────────────────────────────────────────


class TestBasicProperties:
    def test_n_subjects_preserved(self):
        r = _default(n_subjects=200)
        assert r.n_subjects == 200

    def test_cmax_mean_positive(self):
        r = _default()
        assert r.cmax_mean_mg_L > 0

    def test_auc_mean_positive(self):
        r = _default()
        assert r.auc_mean_mg_h_per_L > 0

    def test_dose_preserved(self):
        r = _default()
        assert r.dose_mg == 100.0


# ── variability ─────────────────────────────────────────────────────────


class TestVariability:
    def test_cv_positive_with_variability(self):
        r = _default()
        assert r.cmax_cv_pct > 0
        assert r.auc_cv_pct > 0

    def test_low_cv_with_no_variability(self):
        r = _default(cv_cl_pct=0, cv_vd_pct=0)
        # With zero variability all subjects identical → CV ~ 0
        assert r.cmax_cv_pct < 1.0
        assert r.auc_cv_pct < 1.0

    def test_higher_cv_input_higher_cv_output(self):
        low = _default(cv_cl_pct=10, cv_vd_pct=10)
        high = _default(cv_cl_pct=50, cv_vd_pct=50)
        assert high.cmax_cv_pct > low.cmax_cv_pct


# ── percentiles ─────────────────────────────────────────────────────────


class TestPercentiles:
    def test_5th_lt_mean_lt_95th_cmax(self):
        r = _default()
        assert r.cmax_5th_pct_mg_L < r.cmax_mean_mg_L < r.cmax_95th_pct_mg_L

    def test_5th_lt_mean_lt_95th_auc(self):
        r = _default()
        assert r.auc_5th_pct_mg_h_per_L < r.auc_mean_mg_h_per_L < r.auc_95th_pct_mg_h_per_L

    def test_percentiles_non_negative(self):
        r = _default()
        assert r.cmax_5th_pct_mg_L >= 0
        assert r.auc_5th_pct_mg_h_per_L >= 0


# ── therapeutic window ──────────────────────────────────────────────────


class TestTherapeuticWindow:
    def test_coverage_in_range(self):
        r = _default()
        assert 0 <= r.therapeutic_window_coverage_pct <= 100

    def test_counts_sum_to_n(self):
        r = _default()
        in_window = int(round(r.therapeutic_window_coverage_pct / 100 * r.n_subjects))
        assert r.n_below_mec + r.n_above_mtc + in_window == r.n_subjects

    def test_wide_window_high_coverage(self):
        r = _default(mec_mg_L=0.001, mtc_mg_L=1000.0)
        assert r.therapeutic_window_coverage_pct > 90

    def test_narrow_window_lower_coverage(self):
        wide = _default(mec_mg_L=0.001, mtc_mg_L=1000.0)
        narrow = _default(mec_mg_L=0.5, mtc_mg_L=0.6)
        assert wide.therapeutic_window_coverage_pct >= narrow.therapeutic_window_coverage_pct


# ── reproducibility ─────────────────────────────────────────────────────


class TestReproducibility:
    def test_same_seed_same_result(self):
        r1 = _default(seed=123)
        r2 = _default(seed=123)
        assert r1.cmax_mean_mg_L == r2.cmax_mean_mg_L
        assert r1.auc_mean_mg_h_per_L == r2.auc_mean_mg_h_per_L

    def test_different_seed_different_result(self):
        r1 = _default(seed=1)
        r2 = _default(seed=2)
        assert r1.cmax_mean_mg_L != r2.cmax_mean_mg_L


# ── validation ──────────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_n_subjects(self):
        with pytest.raises(ValueError, match="n_subjects"):
            _default(n_subjects=5)

    def test_negative_cv_cl(self):
        with pytest.raises(ValueError, match="cv_cl_pct"):
            _default(cv_cl_pct=-1)

    def test_negative_cv_vd(self):
        with pytest.raises(ValueError, match="cv_vd_pct"):
            _default(cv_vd_pct=-1)

    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default(dose_mg=0)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="mean_cl_L_per_h"):
            _default(mean_cl_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="mean_vd_L"):
            _default(mean_vd_L=0)

    def test_zero_mec(self):
        with pytest.raises(ValueError, match="mec_mg_L"):
            _default(mec_mg_L=0)

    def test_mtc_le_mec(self):
        with pytest.raises(ValueError, match="mtc_mg_L"):
            _default(mec_mg_L=2.0, mtc_mg_L=1.0)
