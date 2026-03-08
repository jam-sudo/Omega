"""Tests for Phase 601 — Prodrug Activation Kinetics."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.prodrug_activation_p601 import (
    ProdrugActivationResult,
    compare_activation_rates,
    simulate_prodrug_activation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_sim(**kwargs):
    """Return a default simulation result with optional overrides."""
    defaults = dict(
        drug_name="enalaprilat",
        prodrug_name="enalapril",
        dose_mg=10.0,
        ka_absorption_per_h=1.0,
        f_oral=0.65,
        k_activation_per_h=0.5,
        k_el_prodrug_per_h=0.1,
        k_el_active_per_h=0.3,
        vd_prodrug_L=20.0,
        vd_active_L=15.0,
        fm_activation=0.9,
        t_end_h=24.0,
        dt_h=0.05,
    )
    defaults.update(kwargs)
    return simulate_prodrug_activation(**defaults)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _default_sim()
        assert isinstance(result, ProdrugActivationResult)

    def test_drug_names_stored(self):
        result = _default_sim(drug_name="myDrug", prodrug_name="myProdrug")
        assert result.drug_name == "myDrug"
        assert result.prodrug_name == "myProdrug"

    def test_dose_stored(self):
        result = _default_sim(dose_mg=50.0)
        assert result.dose_mg == pytest.approx(50.0)

    def test_times_is_list(self):
        result = _default_sim()
        assert isinstance(result.times_h, list)

    def test_c_prodrug_is_list(self):
        result = _default_sim()
        assert isinstance(result.c_prodrug_mg_L, list)

    def test_c_active_is_list(self):
        result = _default_sim()
        assert isinstance(result.c_active_mg_L, list)

    def test_list_lengths_equal(self):
        result = _default_sim()
        assert len(result.times_h) == len(result.c_prodrug_mg_L)
        assert len(result.times_h) == len(result.c_active_mg_L)

    def test_list_length_matches_steps(self):
        """Expect n_steps + 1 entries (initial point + n_steps)."""
        result = _default_sim(t_end_h=10.0, dt_h=0.1)
        # n_steps = ceil(10.0 / 0.1) = 100, so 101 entries
        assert len(result.times_h) == 101

    def test_times_starts_at_zero(self):
        result = _default_sim()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_initial_concentrations_zero(self):
        result = _default_sim()
        assert result.c_prodrug_mg_L[0] == pytest.approx(0.0)
        assert result.c_active_mg_L[0] == pytest.approx(0.0)

    def test_notes_is_str(self):
        result = _default_sim()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# PK plausibility
# ---------------------------------------------------------------------------


class TestPKPlausibility:
    def test_cmax_prodrug_positive(self):
        result = _default_sim()
        assert result.cmax_prodrug_mg_L > 0.0

    def test_cmax_active_positive(self):
        result = _default_sim()
        assert result.cmax_active_mg_L > 0.0

    def test_auc_prodrug_positive(self):
        result = _default_sim()
        assert result.auc_prodrug_mg_h_per_L > 0.0

    def test_auc_active_positive(self):
        result = _default_sim()
        assert result.auc_active_mg_h_per_L > 0.0

    def test_tmax_active_after_tmax_prodrug(self):
        """Active drug concentration peaks later than prodrug (lag effect)."""
        result = _default_sim(
            ka_absorption_per_h=2.0,
            k_activation_per_h=0.3,
            k_el_prodrug_per_h=0.05,
            k_el_active_per_h=0.1,
        )
        assert result.tmax_active_h >= result.tmax_prodrug_h

    def test_concentrations_nonnegative(self):
        result = _default_sim()
        assert all(c >= 0.0 for c in result.c_prodrug_mg_L)
        assert all(c >= 0.0 for c in result.c_active_mg_L)

    def test_concentrations_near_zero_at_end(self):
        """Concentrations decay toward zero by 72 h."""
        result = _default_sim(t_end_h=72.0)
        assert result.c_prodrug_mg_L[-1] < 0.01
        assert result.c_active_mg_L[-1] < 0.01

    def test_higher_dose_scales_cmax(self):
        """Doubling dose should approximately double Cmax."""
        r1 = _default_sim(dose_mg=10.0)
        r2 = _default_sim(dose_mg=20.0)
        ratio = r2.cmax_active_mg_L / r1.cmax_active_mg_L
        assert 1.8 <= ratio <= 2.2

    def test_higher_dose_scales_auc(self):
        r1 = _default_sim(dose_mg=10.0)
        r2 = _default_sim(dose_mg=20.0)
        ratio = r2.auc_active_mg_h_per_L / r1.auc_active_mg_h_per_L
        assert 1.8 <= ratio <= 2.2


# ---------------------------------------------------------------------------
# Half-life calculations
# ---------------------------------------------------------------------------


class TestHalfLife:
    def test_t_half_prodrug_formula(self):
        k_act = 0.5
        k_el_prod = 0.1
        result = _default_sim(k_activation_per_h=k_act, k_el_prodrug_per_h=k_el_prod)
        expected = math.log(2.0) / (k_act + k_el_prod)
        assert result.t_half_prodrug_h == pytest.approx(expected, rel=1e-6)

    def test_t_half_active_formula(self):
        k_el_act = 0.3
        result = _default_sim(k_el_active_per_h=k_el_act)
        expected = math.log(2.0) / k_el_act
        assert result.t_half_active_h == pytest.approx(expected, rel=1e-6)

    def test_t_half_positive(self):
        result = _default_sim()
        assert result.t_half_prodrug_h > 0.0
        assert result.t_half_active_h > 0.0


# ---------------------------------------------------------------------------
# Activation efficiency
# ---------------------------------------------------------------------------


class TestActivationEfficiency:
    def test_efficiency_between_0_and_1(self):
        result = _default_sim()
        assert 0.0 <= result.activation_efficiency <= 1.0

    def test_high_fm_high_efficiency(self):
        r_high = _default_sim(fm_activation=1.0)
        r_low = _default_sim(fm_activation=0.1)
        assert r_high.activation_efficiency > r_low.activation_efficiency

    def test_efficiency_formula(self):
        k_act = 0.5
        k_el_prod = 0.1
        fm = 0.8
        result = _default_sim(
            k_activation_per_h=k_act,
            k_el_prodrug_per_h=k_el_prod,
            fm_activation=fm,
        )
        expected = min(1.0, fm * k_act / (k_act + k_el_prod))
        assert result.activation_efficiency == pytest.approx(expected, rel=1e-6)

    def test_higher_activation_rate_higher_efficiency(self):
        r_slow = _default_sim(k_activation_per_h=0.1, fm_activation=0.9)
        r_fast = _default_sim(k_activation_per_h=2.0, fm_activation=0.9)
        assert r_fast.activation_efficiency > r_slow.activation_efficiency


# ---------------------------------------------------------------------------
# Effect of activation rate on AUC
# ---------------------------------------------------------------------------


class TestActivationRateEffect:
    def test_higher_activation_rate_increases_active_auc(self):
        """Faster activation → more active drug AUC."""
        r_slow = _default_sim(k_activation_per_h=0.1)
        r_fast = _default_sim(k_activation_per_h=2.0)
        assert r_fast.auc_active_mg_h_per_L > r_slow.auc_active_mg_h_per_L

    def test_higher_activation_rate_increases_cmax_active(self):
        r_slow = _default_sim(k_activation_per_h=0.1)
        r_fast = _default_sim(k_activation_per_h=2.0)
        assert r_fast.cmax_active_mg_L > r_slow.cmax_active_mg_L


# ---------------------------------------------------------------------------
# compare_activation_rates
# ---------------------------------------------------------------------------


class TestCompareActivationRates:
    def test_returns_list(self):
        result = compare_activation_rates(
            drug_name="drug",
            prodrug_name="prodrug",
            dose_mg=10.0,
            k_activation_values=[0.1, 0.5, 1.0],
            vd_prodrug_L=20.0,
            vd_active_L=15.0,
        )
        assert isinstance(result, list)

    def test_length_matches_inputs(self):
        values = [0.1, 0.5, 1.0, 2.0]
        result = compare_activation_rates(
            drug_name="drug",
            prodrug_name="prodrug",
            dose_mg=10.0,
            k_activation_values=values,
            vd_prodrug_L=20.0,
            vd_active_L=15.0,
        )
        assert len(result) == len(values)

    def test_sorted_by_auc_active_descending(self):
        result = compare_activation_rates(
            drug_name="drug",
            prodrug_name="prodrug",
            dose_mg=10.0,
            k_activation_values=[0.1, 0.5, 1.0, 2.0],
            vd_prodrug_L=20.0,
            vd_active_L=15.0,
        )
        aucs = [r.auc_active_mg_h_per_L for r in result]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_items_are_results(self):
        result = compare_activation_rates(
            drug_name="drug",
            prodrug_name="prodrug",
            dose_mg=10.0,
            k_activation_values=[0.2, 1.0],
            vd_prodrug_L=20.0,
            vd_active_L=15.0,
        )
        for r in result:
            assert isinstance(r, ProdrugActivationResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_sim(dose_mg=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_sim(dose_mg=0.0)

    def test_f_oral_zero_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _default_sim(f_oral=0.0)

    def test_f_oral_above_one_raises(self):
        with pytest.raises(ValueError, match="f_oral"):
            _default_sim(f_oral=1.1)

    def test_ka_zero_raises(self):
        with pytest.raises(ValueError, match="ka_absorption_per_h"):
            _default_sim(ka_absorption_per_h=0.0)

    def test_k_activation_zero_raises(self):
        with pytest.raises(ValueError, match="k_activation_per_h"):
            _default_sim(k_activation_per_h=0.0)

    def test_k_el_prodrug_zero_raises(self):
        with pytest.raises(ValueError, match="k_el_prodrug_per_h"):
            _default_sim(k_el_prodrug_per_h=0.0)

    def test_k_el_active_zero_raises(self):
        with pytest.raises(ValueError, match="k_el_active_per_h"):
            _default_sim(k_el_active_per_h=0.0)

    def test_vd_prodrug_zero_raises(self):
        with pytest.raises(ValueError, match="vd_prodrug_L"):
            _default_sim(vd_prodrug_L=0.0)

    def test_vd_active_zero_raises(self):
        with pytest.raises(ValueError, match="vd_active_L"):
            _default_sim(vd_active_L=0.0)

    def test_fm_zero_raises(self):
        with pytest.raises(ValueError, match="fm_activation"):
            _default_sim(fm_activation=0.0)

    def test_fm_above_one_raises(self):
        with pytest.raises(ValueError, match="fm_activation"):
            _default_sim(fm_activation=1.5)
