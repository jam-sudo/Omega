"""
Tests for Phase 889 — Prodrug Activation Kinetics
"""

import math
import pytest

from omega_pbpk.core.prodrug_activation_pk import (
    ProdrugPKResult,
    simulate_prodrug_pk,
    compare_activation_sites,
)


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

class TestSimulateProdrugPK:

    def test_returns_prodrug_pk_result(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        assert isinstance(result, ProdrugPKResult)

    def test_drug_name_stored(self):
        result = simulate_prodrug_pk("codeine", "morphine", 30.0)
        assert result.drug_name == "codeine"
        assert result.active_metabolite_name == "morphine"

    def test_dose_stored(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 20.0)
        assert result.dose_mg == 20.0

    def test_activation_site_default_liver(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        assert result.activation_site == "liver"

    def test_activation_site_plasma(self):
        result = simulate_prodrug_pk(
            "enalapril", "enalaprilat", 10.0, activation_site="plasma"
        )
        assert result.activation_site == "plasma"

    def test_times_starts_at_zero(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_ends_at_t_end(self):
        result = simulate_prodrug_pk(
            "enalapril", "enalaprilat", 10.0, t_end_h=12.0
        )
        assert result.times_h[-1] == pytest.approx(12.0, abs=0.2)

    def test_concentrations_non_negative(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        assert all(c >= 0 for c in result.c_prodrug_mg_L)
        assert all(c >= 0 for c in result.c_active_mg_L)

    def test_initial_concentrations_zero(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        assert result.c_prodrug_mg_L[0] == pytest.approx(0.0)
        assert result.c_active_mg_L[0] == pytest.approx(0.0)

    def test_cmax_prodrug_positive(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        assert result.cmax_prodrug > 0

    def test_cmax_active_positive(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        assert result.cmax_active > 0

    def test_tmax_active_after_tmax_prodrug(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        # Active metabolite peaks after or around the prodrug peak
        assert result.tmax_active_h >= result.tmax_prodrug_h

    def test_auc_both_positive(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        assert result.auc_prodrug > 0
        assert result.auc_active > 0

    def test_f_activation_stored(self):
        result = simulate_prodrug_pk(
            "enalapril", "enalaprilat", 10.0, f_activation=0.85
        )
        assert result.f_activation == pytest.approx(0.85)

    def test_f_activation_zero_gives_no_active(self):
        result = simulate_prodrug_pk(
            "enalapril", "enalaprilat", 10.0, f_activation=0.0
        )
        assert result.auc_active == pytest.approx(0.0, abs=1e-6)

    def test_active_to_prodrug_ratio_computed(self):
        result = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        expected = result.auc_active / result.auc_prodrug
        assert result.active_to_prodrug_auc_ratio == pytest.approx(expected, rel=1e-6)

    def test_notes_dominant_active(self):
        # Very high activation rate → active dominates
        result = simulate_prodrug_pk(
            "enalapril", "enalaprilat", 10.0,
            k_act_per_h=5.0,
            f_activation=1.0,
            cl_prodrug_L_per_h=20.0,
            cl_active_L_per_h=1.0,
        )
        assert "dominant" in result.notes.lower()

    def test_notes_prodrug_dominant(self):
        # Very low activation → prodrug dominates
        result = simulate_prodrug_pk(
            "enalapril", "enalaprilat", 10.0,
            k_act_per_h=0.01,
            f_activation=0.05,
        )
        assert "prodrug-dominant" in result.notes.lower() or "prodrug" in result.notes.lower()

    def test_higher_dose_higher_concentrations(self):
        r_low = simulate_prodrug_pk("enalapril", "enalaprilat", 10.0)
        r_high = simulate_prodrug_pk("enalapril", "enalaprilat", 100.0)
        assert r_high.cmax_prodrug > r_low.cmax_prodrug
        assert r_high.auc_active > r_low.auc_active


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_prodrug_pk("X", "Y", 0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_prodrug_pk("X", "Y", -5.0)

    def test_f_activation_above_one_raises(self):
        with pytest.raises(ValueError, match="f_activation"):
            simulate_prodrug_pk("X", "Y", 10.0, f_activation=1.5)

    def test_f_activation_negative_raises(self):
        with pytest.raises(ValueError):
            simulate_prodrug_pk("X", "Y", 10.0, f_activation=-0.1)

    def test_invalid_activation_site_raises(self):
        with pytest.raises(ValueError, match="activation_site"):
            simulate_prodrug_pk("X", "Y", 10.0, activation_site="kidney")

    def test_zero_clearance_raises(self):
        with pytest.raises(ValueError, match="cl_prodrug"):
            simulate_prodrug_pk("X", "Y", 10.0, cl_prodrug_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_prodrug_pk("X", "Y", 10.0, vd_L=0.0)


# ---------------------------------------------------------------------------
# Compare activation sites
# ---------------------------------------------------------------------------

class TestCompareActivationSites:

    def test_returns_four_results(self):
        results = compare_activation_sites("enalapril", "enalaprilat", 10.0)
        assert len(results) == 4

    def test_sorted_by_auc_active_descending(self):
        results = compare_activation_sites("enalapril", "enalaprilat", 10.0)
        aucs = [r.auc_active for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_sites_represented(self):
        results = compare_activation_sites("enalapril", "enalaprilat", 10.0)
        sites = {r.activation_site for r in results}
        assert sites == {"plasma", "liver", "intestine", "gut_wall"}

    def test_kwargs_forwarded(self):
        results = compare_activation_sites(
            "enalapril", "enalaprilat", 10.0, f_activation=0.9
        )
        for r in results:
            assert r.f_activation == pytest.approx(0.9)
