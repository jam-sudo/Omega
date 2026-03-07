"""Tests for analysis/pk_variability_sources.py (Phase 680)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_variability_sources import (
    PKVariabilityResult,
    decompose_pk_variability,
    simulate_iiv_profiles,
    variability_summary,
)

# ---------------------------------------------------------------------------
# simulate_iiv_profiles
# ---------------------------------------------------------------------------


class TestSimulateIivProfiles:
    def test_returns_correct_length(self):
        profiles = simulate_iiv_profiles(20, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        assert len(profiles) == 20

    def test_each_profile_has_required_keys(self):
        profiles = simulate_iiv_profiles(5, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        required = {"subject_id", "CL", "Vd", "ka", "cmax", "tmax", "auc"}
        for p in profiles:
            assert required.issubset(p.keys())

    def test_all_cmax_positive(self):
        profiles = simulate_iiv_profiles(10, cl_mean=5.0, vd_mean=50.0, ka_mean=1.5)
        assert all(p["cmax"] > 0 for p in profiles)

    def test_all_auc_positive(self):
        profiles = simulate_iiv_profiles(10, cl_mean=5.0, vd_mean=50.0, ka_mean=1.5)
        assert all(p["auc"] > 0 for p in profiles)

    def test_cv_positive_when_omega_nonzero(self):
        profiles = simulate_iiv_profiles(30, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, omega_cl=0.3)
        cl_vals = [p["CL"] for p in profiles]
        mean_cl = sum(cl_vals) / len(cl_vals)
        std_cl = math.sqrt(sum((v - mean_cl) ** 2 for v in cl_vals) / len(cl_vals))
        cv = std_cl / mean_cl * 100
        assert cv > 0

    def test_zero_omega_gives_constant_cl(self):
        profiles = simulate_iiv_profiles(
            10,
            cl_mean=5.0,
            vd_mean=50.0,
            ka_mean=1.0,
            omega_cl=0.0,
            omega_vd=0.0,
            omega_ka=0.0,
        )
        cl_vals = [p["CL"] for p in profiles]
        assert all(abs(v - 5.0) < 1e-10 for v in cl_vals)

    def test_reproducible_with_same_seed(self):
        p1 = simulate_iiv_profiles(10, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, seed=99)
        p2 = simulate_iiv_profiles(10, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, seed=99)
        assert p1[0]["CL"] == p2[0]["CL"]
        assert p1[5]["auc"] == p2[5]["auc"]

    def test_different_seeds_give_different_results(self):
        p1 = simulate_iiv_profiles(10, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, seed=1)
        p2 = simulate_iiv_profiles(10, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, seed=2)
        assert p1[0]["CL"] != p2[0]["CL"]

    def test_subject_ids_sequential(self):
        profiles = simulate_iiv_profiles(5, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        assert [p["subject_id"] for p in profiles] == [0, 1, 2, 3, 4]

    def test_n_subjects_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_subjects"):
            simulate_iiv_profiles(1, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)

    def test_cl_mean_zero_raises(self):
        with pytest.raises(ValueError, match="cl_mean"):
            simulate_iiv_profiles(5, cl_mean=0.0, vd_mean=50.0, ka_mean=1.0)

    def test_cl_mean_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_iiv_profiles(5, cl_mean=-1.0, vd_mean=50.0, ka_mean=1.0)

    def test_dose_mg_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_iiv_profiles(5, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, dose_mg=0.0)


# ---------------------------------------------------------------------------
# variability_summary
# ---------------------------------------------------------------------------


class TestVariabilitySummary:
    def test_returns_dict(self):
        profiles = simulate_iiv_profiles(20, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        summary = variability_summary(profiles)
        assert isinstance(summary, dict)

    def test_expected_keys_present(self):
        profiles = simulate_iiv_profiles(20, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        summary = variability_summary(profiles)
        expected_keys = {
            "n_subjects",
            "cl_cv_pct",
            "vd_cv_pct",
            "ka_cv_pct",
            "cmax_cv_pct",
            "auc_cv_pct",
            "auc_p5",
            "auc_p50",
            "auc_p95",
            "cmax_p5",
            "cmax_p50",
            "cmax_p95",
        }
        assert expected_keys.issubset(summary.keys())

    def test_n_subjects_matches(self):
        profiles = simulate_iiv_profiles(15, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        summary = variability_summary(profiles)
        assert summary["n_subjects"] == 15

    def test_cl_cv_positive_when_omega_nonzero(self):
        profiles = simulate_iiv_profiles(20, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, omega_cl=0.5)
        summary = variability_summary(profiles)
        assert summary["cl_cv_pct"] > 0

    def test_auc_percentiles_ordered(self):
        profiles = simulate_iiv_profiles(
            50, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, omega_cl=0.4, omega_vd=0.4
        )
        summary = variability_summary(profiles)
        assert summary["auc_p5"] < summary["auc_p50"] < summary["auc_p95"]

    def test_cmax_percentiles_ordered(self):
        profiles = simulate_iiv_profiles(
            50, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, omega_cl=0.4, omega_vd=0.4
        )
        summary = variability_summary(profiles)
        assert summary["cmax_p5"] <= summary["cmax_p50"] <= summary["cmax_p95"]


# ---------------------------------------------------------------------------
# decompose_pk_variability
# ---------------------------------------------------------------------------


class TestDecomposePkVariability:
    def test_returns_pk_variability_result(self):
        profiles = simulate_iiv_profiles(20, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        result = decompose_pk_variability(profiles, ["CL", "Vd"])
        assert isinstance(result, PKVariabilityResult)

    def test_n_subjects_correct(self):
        profiles = simulate_iiv_profiles(25, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        result = decompose_pk_variability(profiles, ["CL"])
        assert result.n_subjects == 25

    def test_cl_cv_pct_positive(self):
        profiles = simulate_iiv_profiles(20, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, omega_cl=0.3)
        result = decompose_pk_variability(profiles, ["CL", "Vd"])
        assert result.cl_cv_pct > 0

    def test_auc_cv_pct_positive(self):
        profiles = simulate_iiv_profiles(20, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0, omega_cl=0.4)
        result = decompose_pk_variability(profiles, ["CL"])
        assert result.auc_cv_pct > 0

    def test_cmax_cv_pct_non_negative(self):
        profiles = simulate_iiv_profiles(20, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        result = decompose_pk_variability(profiles, ["CL"])
        assert result.cmax_cv_pct >= 0

    def test_notes_contains_n_subjects(self):
        profiles = simulate_iiv_profiles(12, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        result = decompose_pk_variability(profiles, ["CL"])
        assert "12" in result.notes

    def test_fewer_than_2_profiles_raises(self):
        profiles = simulate_iiv_profiles(2, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        with pytest.raises(ValueError):
            decompose_pk_variability(profiles[:1], ["CL"])

    def test_result_is_frozen_dataclass(self):
        profiles = simulate_iiv_profiles(10, cl_mean=5.0, vd_mean=50.0, ka_mean=1.0)
        result = decompose_pk_variability(profiles, ["CL"])
        with pytest.raises(Exception):
            result.n_subjects = 999  # type: ignore[misc]
