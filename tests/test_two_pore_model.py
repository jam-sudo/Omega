"""Tests for Phase 402: two-pore model for biologic vascular permeability."""

from __future__ import annotations

import pytest

from omega_pbpk.core.two_pore_model import (
    TwoPorePKResult,
    mw_effect_on_pk,
    simulate_two_pore_pk,
)

# ---------------------------------------------------------------------------
# Default parameter set (IgG-like)
# ---------------------------------------------------------------------------

IGG_PARAMS = dict(
    drug_name="test-IgG",
    mw_kDa=150.0,
    dose_mg=100.0,
    cl_L_per_day=0.2,
    vd_central_L=3.0,
    sigma_small=0.95,
    sigma_large=0.10,
    lymph_flow_L_per_day=2.0,
    t_end_days=30.0,
    dt_days=0.5,
)

SMALL_PROTEIN = dict(
    drug_name="small-protein",
    mw_kDa=15.0,
    dose_mg=50.0,
    cl_L_per_day=1.5,
    vd_central_L=5.0,
    sigma_small=0.70,
    sigma_large=0.10,
    lymph_flow_L_per_day=2.0,
    t_end_days=10.0,
    dt_days=0.2,
)


# ---------------------------------------------------------------------------
# TwoPorePKResult dataclass
# ---------------------------------------------------------------------------


class TestTwoPorePKResultDataclass:
    def test_result_is_dataclass_instance(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert isinstance(r, TwoPorePKResult)

    def test_result_is_mutable(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        # Should NOT raise — @dataclass (not frozen)
        r.drug_name = "modified"
        assert r.drug_name == "modified"

    def test_has_required_fields(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        for attr in [
            "drug_name", "mw_kDa", "dose_mg", "sigma_small", "sigma_large",
            "times_days", "c_central_mg_L", "c_tissue_mg_L",
            "cmax", "t_half_days", "auc_mg_day_per_L", "notes",
        ]:
            assert hasattr(r, attr), f"Missing attribute: {attr}"


# ---------------------------------------------------------------------------
# simulate_two_pore_pk — basic correctness
# ---------------------------------------------------------------------------


class TestSimulateTwoPorekBasic:
    def test_cmax_equals_initial_concentration(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        expected_c0 = IGG_PARAMS["dose_mg"] / IGG_PARAMS["vd_central_L"]
        assert abs(r.cmax - expected_c0) < 0.01

    def test_concentrations_non_negative(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert all(c >= 0.0 for c in r.c_central_mg_L)
        assert all(c >= 0.0 for c in r.c_tissue_mg_L)

    def test_central_concentration_decreases_overall(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert r.c_central_mg_L[-1] < r.c_central_mg_L[0]

    def test_tissue_concentration_rises_then_falls(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        # Tissue should show some accumulation before declining
        assert max(r.c_tissue_mg_L) > r.c_tissue_mg_L[0]

    def test_time_array_length_matches_concentration_arrays(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert len(r.times_days) == len(r.c_central_mg_L)
        assert len(r.times_days) == len(r.c_tissue_mg_L)

    def test_time_starts_at_zero(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert r.times_days[0] == 0.0

    def test_time_ends_near_t_end(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert abs(r.times_days[-1] - IGG_PARAMS["t_end_days"]) < 0.01

    def test_auc_positive(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert r.auc_mg_day_per_L > 0.0

    def test_t_half_positive(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert r.t_half_days > 0.0

    def test_stored_parameters_match_inputs(self):
        r = simulate_two_pore_pk(**IGG_PARAMS)
        assert r.drug_name == IGG_PARAMS["drug_name"]
        assert r.mw_kDa == IGG_PARAMS["mw_kDa"]
        assert r.dose_mg == IGG_PARAMS["dose_mg"]
        assert r.sigma_small == IGG_PARAMS["sigma_small"]
        assert r.sigma_large == IGG_PARAMS["sigma_large"]


# ---------------------------------------------------------------------------
# simulate_two_pore_pk — PK characteristics
# ---------------------------------------------------------------------------


class TestSimulateTwoPoreKPKChar:
    def test_higher_cl_gives_lower_auc(self):
        p_low = dict(IGG_PARAMS, cl_L_per_day=0.1)
        p_high = dict(IGG_PARAMS, cl_L_per_day=0.5)
        r_low = simulate_two_pore_pk(**p_low)
        r_high = simulate_two_pore_pk(**p_high)
        assert r_low.auc_mg_day_per_L > r_high.auc_mg_day_per_L

    def test_higher_dose_gives_proportionally_higher_auc(self):
        p_low = dict(IGG_PARAMS, dose_mg=50.0)
        p_high = dict(IGG_PARAMS, dose_mg=100.0)
        r_low = simulate_two_pore_pk(**p_low)
        r_high = simulate_two_pore_pk(**p_high)
        ratio = r_high.auc_mg_day_per_L / r_low.auc_mg_day_per_L
        assert abs(ratio - 2.0) < 0.3  # roughly proportional

    def test_higher_sigma_small_means_less_tissue_distribution(self):
        # sigma_small=0.99 (almost no small-pore transport) vs 0.80
        p_high_sigma = dict(IGG_PARAMS, sigma_small=0.99, t_end_days=10.0)
        p_low_sigma = dict(IGG_PARAMS, sigma_small=0.80, t_end_days=10.0)
        r_high = simulate_two_pore_pk(**p_high_sigma)
        r_low = simulate_two_pore_pk(**p_low_sigma)
        # Lower sigma_small => more transcapillary flux => more tissue accumulation
        assert max(r_low.c_tissue_mg_L) > max(r_high.c_tissue_mg_L)

    def test_small_protein_clears_faster_than_igg(self):
        r_igg = simulate_two_pore_pk(**IGG_PARAMS)
        r_small = simulate_two_pore_pk(**SMALL_PROTEIN)
        # Small protein has higher CL/dose ratio; normalised AUC should be lower
        auc_per_dose_igg = r_igg.auc_mg_day_per_L / IGG_PARAMS["dose_mg"]
        auc_per_dose_small = r_small.auc_mg_day_per_L / SMALL_PROTEIN["dose_mg"]
        assert auc_per_dose_small < auc_per_dose_igg


# ---------------------------------------------------------------------------
# simulate_two_pore_pk — input validation
# ---------------------------------------------------------------------------


class TestSimulateTwoPoreKValidation:
    def test_raises_for_empty_drug_name(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, drug_name=""))

    def test_raises_for_zero_mw(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, mw_kDa=0.0))

    def test_raises_for_negative_dose(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, dose_mg=-1.0))

    def test_raises_for_zero_cl(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, cl_L_per_day=0.0))

    def test_raises_for_zero_vd(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, vd_central_L=0.0))

    def test_raises_for_sigma_small_out_of_range(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, sigma_small=1.5))

    def test_raises_for_sigma_large_negative(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, sigma_large=-0.1))

    def test_raises_for_zero_lymph_flow(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, lymph_flow_L_per_day=0.0))

    def test_raises_for_zero_t_end(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, t_end_days=0.0))

    def test_raises_for_zero_dt(self):
        with pytest.raises(ValueError):
            simulate_two_pore_pk(**dict(IGG_PARAMS, dt_days=0.0))


# ---------------------------------------------------------------------------
# mw_effect_on_pk
# ---------------------------------------------------------------------------


class TestMWEffectOnPK:
    def test_returns_list_of_correct_length(self):
        mw_vals = [50.0, 100.0, 150.0]
        results = mw_effect_on_pk("drug", 100.0, mw_vals)
        assert len(results) == 3

    def test_each_result_has_required_keys(self):
        results = mw_effect_on_pk("drug", 100.0, [150.0])
        keys = {"mw_kDa", "sigma_small", "cl_L_per_day", "auc_mg_day_per_L", "t_half_days", "cmax"}
        assert keys.issubset(results[0].keys())

    def test_sigma_small_clamps_at_0_90_for_small_mw(self):
        # 0.6 + 0.004 * 10 = 0.64, below 0.90 => clamped to 0.90
        results = mw_effect_on_pk("drug", 100.0, [10.0])
        assert results[0]["sigma_small"] == pytest.approx(0.90)

    def test_sigma_small_clamps_at_0_99_for_large_mw(self):
        # 0.6 + 0.004 * 150 = 1.2, above 0.99 => clamped to 0.99
        results = mw_effect_on_pk("drug", 100.0, [150.0])
        assert results[0]["sigma_small"] == pytest.approx(0.99)

    def test_sigma_small_mid_range(self):
        # 0.6 + 0.004 * 80 = 0.92 => within [0.90, 0.99]
        results = mw_effect_on_pk("drug", 100.0, [80.0])
        assert abs(results[0]["sigma_small"] - 0.92) < 0.001

    def test_raises_for_empty_drug_name(self):
        with pytest.raises(ValueError):
            mw_effect_on_pk("", 100.0, [150.0])

    def test_raises_for_zero_dose(self):
        with pytest.raises(ValueError):
            mw_effect_on_pk("drug", 0.0, [150.0])

    def test_raises_for_empty_mw_list(self):
        with pytest.raises(ValueError):
            mw_effect_on_pk("drug", 100.0, [])

    def test_raises_for_negative_mw(self):
        with pytest.raises(ValueError):
            mw_effect_on_pk("drug", 100.0, [-10.0, 150.0])

    def test_raises_for_nonpositive_cl_factor(self):
        with pytest.raises(ValueError):
            mw_effect_on_pk("drug", 100.0, [150.0], cl_per_mw_factor=0.0)

    def test_auc_values_positive(self):
        results = mw_effect_on_pk("drug", 100.0, [50.0, 100.0, 150.0])
        assert all(r["auc_mg_day_per_L"] > 0 for r in results)

    def test_t_half_positive(self):
        results = mw_effect_on_pk("drug", 100.0, [50.0, 100.0, 150.0])
        assert all(r["t_half_days"] >= 0 for r in results)
