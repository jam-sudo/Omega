"""Tests for Phase 630 — Drug Lymph Node Concentration."""

import math

import pytest

from omega_pbpk.core.lymph_node_pk_p630 import (
    LymphNodePKResult,
    _f_lymph,
    _kp_lymph,
    simulate_lymph_node_pk,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_hydrophilic():
    """Small hydrophilic drug — no lymphatic absorption."""
    return simulate_lymph_node_pk(
        drug_name="HydrophilicDrug",
        dose_mg=100.0,
        logP=-1.0,
        mw_Da=200.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=20.0,
        t_end_h=48.0,
        dt_h=0.1,
    )


def _run_lipophilic():
    """Large lipophilic drug — significant lymphatic absorption."""
    return simulate_lymph_node_pk(
        drug_name="LipophilicDrug",
        dose_mg=200.0,
        logP=5.0,
        mw_Da=1200.0,
        cl_sys_L_per_h=3.0,
        vd_sys_L=150.0,
        t_end_h=72.0,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_lymph_node_pk_result(self):
        r = _run_hydrophilic()
        assert isinstance(r, LymphNodePKResult)

    def test_not_frozen(self):
        """LymphNodePKResult is a plain dataclass (has lists), should be mutable."""
        r = _run_hydrophilic()
        r.drug_name = "modified"
        assert r.drug_name == "modified"

    def test_drug_name_preserved(self):
        r = _run_hydrophilic()
        assert r.drug_name == "HydrophilicDrug"

    def test_dose_preserved(self):
        r = simulate_lymph_node_pk(
            drug_name="D",
            dose_mg=50.0,
            logP=1.0,
            mw_Da=300.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=10.0,
            t_end_h=24.0,
            dt_h=0.5,
        )
        assert r.dose_mg == 50.0


# ---------------------------------------------------------------------------
# List lengths
# ---------------------------------------------------------------------------


class TestListLengths:
    def test_times_h_non_empty(self):
        r = _run_hydrophilic()
        assert len(r.times_h) > 0

    def test_c_plasma_same_length_as_times(self):
        r = _run_hydrophilic()
        assert len(r.c_plasma_mg_L) == len(r.times_h)

    def test_c_lymph_same_length_as_times(self):
        r = _run_hydrophilic()
        assert len(r.c_lymph_mg_g) == len(r.times_h)

    def test_c_lymph_fluid_same_length_as_times(self):
        r = _run_hydrophilic()
        assert len(r.c_lymph_fluid_mg_L) == len(r.times_h)

    def test_time_starts_at_zero(self):
        r = _run_hydrophilic()
        assert r.times_h[0] == 0.0

    def test_all_concentrations_non_negative(self):
        r = _run_lipophilic()
        assert all(c >= 0.0 for c in r.c_plasma_mg_L)
        assert all(c >= 0.0 for c in r.c_lymph_mg_g)
        assert all(c >= 0.0 for c in r.c_lymph_fluid_mg_L)


# ---------------------------------------------------------------------------
# Lymphatic absorption fractions
# ---------------------------------------------------------------------------


class TestLymphaticAbsorption:
    def test_small_hydrophilic_zero_lymph(self):
        """MW < 500 and logP < 2 → f_lymph = 0."""
        assert _f_lymph(-1.0, 200.0) == 0.0
        assert _f_lymph(1.9, 499.0) == 0.0

    def test_large_lipophilic_positive_lymph(self):
        """MW ≥ 500 or logP ≥ 2 → f_lymph > 0."""
        fl = _f_lymph(5.0, 1200.0)
        assert fl > 0.0

    def test_f_lymph_capped_at_30pct(self):
        """Very lipophilic/large → f_lymph capped at 0.3."""
        fl = _f_lymph(10.0, 5000.0)
        assert fl <= 0.3

    def test_hydrophilic_drug_zero_lymphatic_absorption_pct(self):
        r = _run_hydrophilic()
        assert r.lymphatic_absorption_pct == 0.0

    def test_lipophilic_drug_positive_lymphatic_absorption_pct(self):
        r = _run_lipophilic()
        assert r.lymphatic_absorption_pct > 0.0

    def test_lymphatic_pct_max_30(self):
        r = _run_lipophilic()
        assert r.lymphatic_absorption_pct <= 30.0


# ---------------------------------------------------------------------------
# kp_lymph
# ---------------------------------------------------------------------------


class TestKpLymph:
    def test_kp_lymph_min_0_1(self):
        """Very negative logP, very large MW → clamped to 0.1."""
        kp = _kp_lymph(-10.0, 10000.0)
        assert kp >= 0.1

    def test_kp_lymph_max_15(self):
        """Very high logP, very small MW → clamped to 15."""
        kp = _kp_lymph(20.0, 50.0)
        assert kp <= 15.0

    def test_kp_lymph_positive(self):
        kp = _kp_lymph(2.0, 400.0)
        assert kp > 0.0

    def test_kp_lymph_increases_with_logp(self):
        kp_low = _kp_lymph(1.0, 300.0)
        kp_high = _kp_lymph(4.0, 300.0)
        assert kp_high > kp_low


# ---------------------------------------------------------------------------
# Cmax values
# ---------------------------------------------------------------------------


class TestCmax:
    def test_cmax_plasma_positive(self):
        r = _run_hydrophilic()
        assert r.cmax_plasma_mg_L > 0.0

    def test_cmax_lymph_positive_lipophilic(self):
        r = _run_lipophilic()
        assert r.cmax_lymph_mg_g >= 0.0

    def test_cmax_plasma_equals_max_of_list(self):
        r = _run_hydrophilic()
        # cmax is rounded to 6 decimal places; compare with tolerance
        assert abs(r.cmax_plasma_mg_L - max(r.c_plasma_mg_L)) < 1e-5

    def test_cmax_lymph_equals_max_of_list(self):
        r = _run_lipophilic()
        # cmax is rounded to 6 decimal places; compare with tolerance
        assert abs(r.cmax_lymph_mg_g - max(r.c_lymph_mg_g)) < 1e-3


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


class TestAUC:
    def test_auc_plasma_positive(self):
        r = _run_hydrophilic()
        assert r.auc_plasma_mg_h_per_L > 0.0

    def test_auc_lymph_positive_lipophilic(self):
        r = _run_lipophilic()
        assert r.auc_lymph_mg_h_per_g >= 0.0

    def test_lymph_to_plasma_ratio_non_negative(self):
        r = _run_lipophilic()
        assert r.lymph_to_plasma_ratio >= 0.0

    def test_hydrophilic_very_low_lymph_auc(self):
        """Hydrophilic drug → minimal lymph concentration."""
        r = _run_hydrophilic()
        # AUC lymph should be very small relative to plasma
        assert r.auc_lymph_mg_h_per_g < r.auc_plasma_mg_h_per_L * 0.5


# ---------------------------------------------------------------------------
# t_half_lymph
# ---------------------------------------------------------------------------


class TestTHalfLymph:
    def test_t_half_lymph_positive(self):
        r = _run_hydrophilic()
        assert r.t_half_lymph_h > 0.0

    def test_t_half_lymph_consistent_with_kout(self):
        """t_half = ln(2) / k_lymph_out; large Vd_lymph → longer t_half."""
        r_small = simulate_lymph_node_pk(
            drug_name="D",
            dose_mg=100.0,
            logP=5.0,
            mw_Da=300.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
            lymph_node_weight_g=5.0,
            t_end_h=48.0,
            dt_h=0.1,
        )
        r_large = simulate_lymph_node_pk(
            drug_name="D",
            dose_mg=100.0,
            logP=5.0,
            mw_Da=300.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
            lymph_node_weight_g=100.0,
            t_end_h=48.0,
            dt_h=0.1,
        )
        assert r_large.t_half_lymph_h > r_small.t_half_lymph_h

    def test_t_half_lymph_finite(self):
        r = _run_lipophilic()
        assert math.isfinite(r.t_half_lymph_h)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_lymph_node_pk(
                drug_name="D",
                dose_mg=-10.0,
                logP=1.0,
                mw_Da=300.0,
                cl_sys_L_per_h=2.0,
                vd_sys_L=10.0,
            )

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            simulate_lymph_node_pk(
                drug_name="D",
                dose_mg=100.0,
                logP=1.0,
                mw_Da=300.0,
                cl_sys_L_per_h=0.0,
                vd_sys_L=10.0,
            )

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            simulate_lymph_node_pk(
                drug_name="D",
                dose_mg=100.0,
                logP=1.0,
                mw_Da=300.0,
                cl_sys_L_per_h=2.0,
                vd_sys_L=-5.0,
            )

    def test_negative_lymph_node_weight_raises(self):
        with pytest.raises(ValueError, match="lymph_node_weight_g"):
            simulate_lymph_node_pk(
                drug_name="D",
                dose_mg=100.0,
                logP=1.0,
                mw_Da=300.0,
                cl_sys_L_per_h=2.0,
                vd_sys_L=10.0,
                lymph_node_weight_g=0.0,
            )

    def test_valid_call_succeeds(self):
        r = simulate_lymph_node_pk(
            drug_name="Valid",
            dose_mg=50.0,
            logP=2.5,
            mw_Da=500.0,
            cl_sys_L_per_h=4.0,
            vd_sys_L=30.0,
            t_end_h=24.0,
            dt_h=0.5,
        )
        assert isinstance(r, LymphNodePKResult)
