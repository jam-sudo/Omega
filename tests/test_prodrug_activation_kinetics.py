"""Tests for Phase 381 — Prodrug Activation Kinetics."""

from __future__ import annotations

import pytest

from omega_pbpk.core.prodrug_activation_kinetics import (
    ProdrugActivationResult,
    compare_activation_sites,
    simulate_prodrug_activation,
)

# ---------------------------------------------------------------------------
# Return type and basic structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_prodrug_activation_result(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert isinstance(result, ProdrugActivationResult)

    def test_drug_name_stored(self):
        result = simulate_prodrug_activation("oseltamivir", 75.0)
        assert result.drug_name == "oseltamivir"

    def test_dose_stored(self):
        result = simulate_prodrug_activation("codeine", 30.0)
        assert result.dose_mg == pytest.approx(30.0)

    def test_activation_site_stored(self):
        result = simulate_prodrug_activation("enalapril", 10.0, activation_site="intestinal")
        assert result.activation_site == "intestinal"

    def test_times_starts_at_zero(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_ends_near_t_end(self):
        result = simulate_prodrug_activation("enalapril", 10.0, t_end_h=12.0)
        assert result.times_h[-1] == pytest.approx(12.0, abs=0.1)

    def test_lists_same_length(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert len(result.times_h) == len(result.c_prodrug_mg_L) == len(result.c_active_mg_L)


# ---------------------------------------------------------------------------
# Concentration profiles
# ---------------------------------------------------------------------------


class TestConcentrationProfiles:
    def test_initial_concentrations_zero(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert result.c_prodrug_mg_L[0] == pytest.approx(0.0)
        assert result.c_active_mg_L[0] == pytest.approx(0.0)

    def test_concentrations_non_negative(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert all(c >= 0.0 for c in result.c_prodrug_mg_L)
        assert all(c >= 0.0 for c in result.c_active_mg_L)

    def test_active_metabolite_appears(self):
        result = simulate_prodrug_activation("enalapril", 10.0, activation_site="hepatic")
        assert max(result.c_active_mg_L) > 0.0

    def test_prodrug_decreases_after_peak(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        # prodrug should eventually decline (last value < max)
        assert result.c_prodrug_mg_L[-1] < max(result.c_prodrug_mg_L)

    def test_active_appears_for_intestinal_site(self):
        result = simulate_prodrug_activation(
            "enalapril", 10.0, activation_site="intestinal", f_activation=0.8
        )
        assert max(result.c_active_mg_L) > 0.0

    def test_active_appears_for_systemic_site(self):
        result = simulate_prodrug_activation("enalapril", 10.0, activation_site="systemic")
        assert max(result.c_active_mg_L) > 0.0


# ---------------------------------------------------------------------------
# AUC and Cmax
# ---------------------------------------------------------------------------


class TestAucCmax:
    def test_auc_prodrug_positive(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert result.auc_prodrug_mg_h_per_L > 0.0

    def test_auc_active_positive_hepatic(self):
        result = simulate_prodrug_activation("enalapril", 10.0, activation_site="hepatic")
        assert result.auc_active_mg_h_per_L > 0.0

    def test_cmax_prodrug_positive(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert result.cmax_prodrug_mg_L > 0.0

    def test_cmax_active_positive(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert result.cmax_active_mg_L > 0.0

    def test_tmax_active_positive(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        assert result.tmax_active_h >= 0.0

    def test_dose_linearity_cmax_active(self):
        """Cmax of active should scale proportionally with dose."""
        r_low = simulate_prodrug_activation("enalapril", 10.0)
        r_high = simulate_prodrug_activation("enalapril", 100.0)
        ratio = r_high.cmax_active_mg_L / r_low.cmax_active_mg_L
        assert ratio == pytest.approx(10.0, rel=0.05)


# ---------------------------------------------------------------------------
# Activation efficiency and site effects
# ---------------------------------------------------------------------------


class TestActivationEfficiency:
    def test_activation_efficiency_between_0_and_100(self):
        for site in ("intestinal", "hepatic", "systemic"):
            result = simulate_prodrug_activation("enalapril", 10.0, activation_site=site)
            assert 0.0 <= result.activation_efficiency_pct <= 100.0

    def test_intestinal_efficiency_equals_f_activation_pct(self):
        result = simulate_prodrug_activation(
            "enalapril", 10.0, activation_site="intestinal", f_activation=0.7
        )
        assert result.activation_efficiency_pct == pytest.approx(70.0, rel=0.01)

    def test_higher_f_activation_higher_active_auc_intestinal(self):
        r_low = simulate_prodrug_activation(
            "enalapril", 10.0, activation_site="intestinal", f_activation=0.2
        )
        r_high = simulate_prodrug_activation(
            "enalapril", 10.0, activation_site="intestinal", f_activation=0.9
        )
        assert r_high.auc_active_mg_h_per_L > r_low.auc_active_mg_h_per_L

    def test_higher_k_activation_more_active_hepatic(self):
        r_slow = simulate_prodrug_activation(
            "enalapril", 10.0, activation_site="hepatic", k_activation_per_h=0.1
        )
        r_fast = simulate_prodrug_activation(
            "enalapril", 10.0, activation_site="hepatic", k_activation_per_h=2.0
        )
        assert r_fast.auc_active_mg_h_per_L > r_slow.auc_active_mg_h_per_L

    def test_active_to_prodrug_ratio_computed_correctly(self):
        result = simulate_prodrug_activation("enalapril", 10.0)
        expected = result.auc_active_mg_h_per_L / result.auc_prodrug_mg_h_per_L
        assert result.active_to_prodrug_auc_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Compare activation sites
# ---------------------------------------------------------------------------


class TestCompareActivationSites:
    def test_returns_list(self):
        results = compare_activation_sites("enalapril", 10.0)
        assert isinstance(results, list)

    def test_returns_three_results(self):
        results = compare_activation_sites("enalapril", 10.0)
        assert len(results) == 3

    def test_all_sites_represented(self):
        results = compare_activation_sites("enalapril", 10.0)
        sites = {r.activation_site for r in results}
        assert sites == {"intestinal", "hepatic", "systemic"}

    def test_sorted_by_auc_active_descending(self):
        results = compare_activation_sites("enalapril", 10.0)
        aucs = [r.auc_active_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_results_are_prodrug_activation_result(self):
        results = compare_activation_sites("enalapril", 10.0)
        for r in results:
            assert isinstance(r, ProdrugActivationResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_prodrug_activation("X", 0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_prodrug_activation("X", -10.0)

    def test_invalid_site_raises(self):
        with pytest.raises(ValueError, match="activation_site"):
            simulate_prodrug_activation("X", 10.0, activation_site="renal")

    def test_zero_cl_prodrug_raises(self):
        with pytest.raises(ValueError, match="cl_prodrug_L_per_h"):
            simulate_prodrug_activation("X", 10.0, cl_prodrug_L_per_h=0.0)

    def test_zero_cl_active_raises(self):
        with pytest.raises(ValueError, match="cl_active_L_per_h"):
            simulate_prodrug_activation("X", 10.0, cl_active_L_per_h=0.0)

    def test_zero_vd_prodrug_raises(self):
        with pytest.raises(ValueError, match="vd_prodrug_L"):
            simulate_prodrug_activation("X", 10.0, vd_prodrug_L=0.0)

    def test_zero_vd_active_raises(self):
        with pytest.raises(ValueError, match="vd_active_L"):
            simulate_prodrug_activation("X", 10.0, vd_active_L=0.0)

    def test_f_activation_above_one_raises(self):
        with pytest.raises(ValueError, match="f_activation"):
            simulate_prodrug_activation("X", 10.0, f_activation=1.5)

    def test_f_activation_negative_raises(self):
        with pytest.raises(ValueError, match="f_activation"):
            simulate_prodrug_activation("X", 10.0, f_activation=-0.1)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError, match="ka_absorption_per_h"):
            simulate_prodrug_activation("X", 10.0, ka_absorption_per_h=0.0)

    def test_zero_k_activation_raises(self):
        with pytest.raises(ValueError, match="k_activation_per_h"):
            simulate_prodrug_activation("X", 10.0, k_activation_per_h=0.0)
