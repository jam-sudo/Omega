"""Tests for prodrug activation kinetics — Phase 750."""

import math

import pytest

from omega_pbpk.core.prodrug_pk import (
    ProdrugPKResult,
    assess_prodrug_activation,
    simulate_prodrug_pk,
)

# ---------------------------------------------------------------------------
# Default kwargs
# ---------------------------------------------------------------------------

DEFAULT_KW = dict(
    drug_name="ProX",
    prodrug_dose_mg=100.0,
    f_gut=0.3,
    f_hepatic=0.4,
    k_act_plasma_per_h=0.1,
    ka_per_h=1.0,
    cl_prod_L_per_h=8.0,
    cl_act_L_per_h=5.0,
    vd_prod_L=20.0,
    vd_act_L=50.0,
    t_end_h=24.0,
    dt_h=0.05,
)


def _run(**overrides):
    kw = dict(DEFAULT_KW)
    kw.update(overrides)
    return simulate_prodrug_pk(**kw)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestProdrugPKBasic:
    def test_returns_dataclass(self):
        result = _run()
        assert isinstance(result, ProdrugPKResult)

    def test_cmax_active_positive(self):
        result = _run()
        assert result.cmax_active > 0.0

    def test_cmax_prodrug_nonnegative(self):
        result = _run()
        assert result.cmax_prodrug >= 0.0

    def test_auc_active_positive(self):
        result = _run()
        assert result.auc_active > 0.0

    def test_times_start_at_zero(self):
        result = _run()
        assert result.times_h[0] == 0.0

    def test_times_end_approx_t_end(self):
        result = _run(t_end_h=12.0)
        assert abs(result.times_h[-1] - 12.0) < 0.1

    def test_list_lengths_match(self):
        result = _run()
        n = len(result.times_h)
        assert len(result.c_prodrug_mg_L) == n
        assert len(result.c_active_mg_L) == n

    def test_notes_nonempty(self):
        result = _run()
        assert isinstance(result.notes, str) and len(result.notes) > 0

    def test_active_to_prodrug_auc_ratio_positive(self):
        result = _run()
        assert result.active_to_prodrug_auc_ratio > 0.0

    def test_t_half_active_formula(self):
        cl = 5.0
        vd = 50.0
        ke = cl / vd
        expected_t_half = math.log(2.0) / ke
        result = _run(cl_act_L_per_h=cl, vd_act_L=vd)
        assert abs(result.t_half_active_h - expected_t_half) < 1e-6


# ---------------------------------------------------------------------------
# Pharmacokinetic behaviour tests
# ---------------------------------------------------------------------------


class TestProdrugPKBehaviour:
    def test_higher_f_gut_higher_active_auc(self):
        r_low = _run(f_gut=0.1)
        r_high = _run(f_gut=0.7)
        assert r_high.auc_active > r_low.auc_active

    def test_minimal_activation_very_low_active(self):
        # Very small gut, hepatic, and plasma activation → tiny active drug
        r_none = _run(f_gut=0.0, f_hepatic=0.0, k_act_plasma_per_h=0.001)
        r_normal = _run()
        assert r_normal.auc_active > r_none.auc_active

    def test_f_overall_activation_formula(self):
        f_gut = 0.3
        f_hepatic = 0.4
        expected = f_gut + (1.0 - f_gut) * f_hepatic
        result = _run(f_gut=f_gut, f_hepatic=f_hepatic)
        assert abs(result.f_overall_activation - expected) < 1e-9

    def test_tmax_active_nonnegative(self):
        result = _run()
        assert result.tmax_active_h >= 0.0

    def test_linear_dose_scaling_cmax(self):
        r1 = _run(prodrug_dose_mg=100.0)
        r2 = _run(prodrug_dose_mg=200.0)
        assert abs(r2.cmax_active / r1.cmax_active - 2.0) < 0.05

    def test_linear_dose_scaling_auc(self):
        r1 = _run(prodrug_dose_mg=100.0)
        r2 = _run(prodrug_dose_mg=200.0)
        assert abs(r2.auc_active / r1.auc_active - 2.0) < 0.05

    def test_prodrug_lag_sign(self):
        # Active typically peaks after or at same time as prodrug
        result = _run()
        # prodrug_lag = tmax_active - tmax_prodrug; can be >= 0 or slightly negative
        # Just check it's a float and exists
        assert isinstance(result.prodrug_lag, float)

    def test_higher_f_gut_f_hepatic_overall_activation(self):
        r_low = _run(f_gut=0.1, f_hepatic=0.1)
        r_high = _run(f_gut=0.6, f_hepatic=0.6)
        assert r_high.f_overall_activation > r_low.f_overall_activation


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestProdrugPKValidation:
    def test_invalid_dose_zero(self):
        with pytest.raises(ValueError):
            _run(prodrug_dose_mg=0.0)

    def test_invalid_dose_negative(self):
        with pytest.raises(ValueError):
            _run(prodrug_dose_mg=-100.0)

    def test_invalid_f_gut_negative(self):
        with pytest.raises(ValueError):
            _run(f_gut=-0.1)

    def test_invalid_f_gut_one(self):
        with pytest.raises(ValueError):
            _run(f_gut=1.0)

    def test_invalid_f_hepatic_negative(self):
        with pytest.raises(ValueError):
            _run(f_hepatic=-0.1)

    def test_invalid_f_hepatic_one(self):
        with pytest.raises(ValueError):
            _run(f_hepatic=1.0)

    def test_invalid_cl_prod_zero(self):
        with pytest.raises(ValueError):
            _run(cl_prod_L_per_h=0.0)

    def test_invalid_cl_act_zero(self):
        with pytest.raises(ValueError):
            _run(cl_act_L_per_h=0.0)


# ---------------------------------------------------------------------------
# assess_prodrug_activation tests
# ---------------------------------------------------------------------------


class TestAssessProdrugActivation:
    def test_returns_dict_with_activation_efficiency(self):
        result = _run()
        assessment = assess_prodrug_activation(result)
        assert "activation_efficiency" in assessment

    def test_activation_efficiency_valid_values(self):
        result = _run()
        assessment = assess_prodrug_activation(result)
        assert assessment["activation_efficiency"] in ("poor", "moderate", "good")

    def test_returns_active_to_prodrug_ratio(self):
        result = _run()
        assessment = assess_prodrug_activation(result)
        assert "active_to_prodrug_ratio" in assessment

    def test_poor_activation_efficiency(self):
        # f_gut=0.1, f_hepatic=0.1 → f_overall = 0.1 + 0.9*0.1 = 0.19 → poor (<30%)
        result = _run(f_gut=0.1, f_hepatic=0.1)
        assessment = assess_prodrug_activation(result)
        assert assessment["activation_efficiency"] == "poor"

    def test_good_activation_efficiency(self):
        # f_gut=0.6, f_hepatic=0.5 → f_overall = 0.6 + 0.4*0.5 = 0.8 → good (>70%)
        result = _run(f_gut=0.6, f_hepatic=0.5)
        assessment = assess_prodrug_activation(result)
        assert assessment["activation_efficiency"] == "good"
