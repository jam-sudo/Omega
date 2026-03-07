"""Tests for intravitreal PK simulation — Phase 1003."""

import math

import pytest

from omega_pbpk.core.intravitreal_pk import (
    IntravitreaPKResult,
    simulate_intravitreal_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_default(**kwargs):
    defaults = dict(drug_name="RanibizumabTest", dose_mcg=500.0, mw_kDa=48.0)
    defaults.update(kwargs)
    return simulate_intravitreal_pk(**defaults)


# ---------------------------------------------------------------------------
# Basic return type and fields
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_result_instance(self):
        r = _run_default()
        assert isinstance(r, IntravitreaPKResult)

    def test_lists_same_length(self):
        r = _run_default(t_end_h=100.0, dt_h=1.0)
        assert len(r.times_h) == len(r.c_vitreous_mg_L)
        assert len(r.times_h) == len(r.c_retina_mg_L)
        assert len(r.times_h) == len(r.c_aqueous_mg_L)

    def test_times_start_at_zero(self):
        r = _run_default(t_end_h=48.0, dt_h=1.0)
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        r = _run_default(t_end_h=48.0, dt_h=1.0)
        assert r.times_h[-1] == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# Scalar output correctness
# ---------------------------------------------------------------------------

class TestScalarOutputs:
    def test_cmax_vitreous_positive(self):
        r = _run_default()
        assert r.cmax_vitreous_mg_L > 0.0

    def test_cmax_vitreous_approx_initial_conc(self):
        # C_vit[0] = dose_mcg/1000 / 0.004
        dose_mcg = 500.0
        expected_c0 = dose_mcg / 1000.0 / 0.004
        r = _run_default(dose_mcg=dose_mcg)
        assert r.cmax_vitreous_mg_L == pytest.approx(expected_c0, rel=1e-6)

    def test_cmax_retina_positive(self):
        r = _run_default()
        assert r.cmax_retina_mg_L > 0.0

    def test_tmax_retina_positive(self):
        r = _run_default()
        assert r.tmax_retina_h > 0.0

    def test_auc_vitreous_positive(self):
        r = _run_default()
        assert r.auc_vitreous_mg_h_per_L > 0.0

    def test_auc_retina_positive(self):
        r = _run_default()
        assert r.auc_retina_mg_h_per_L > 0.0

    def test_t_half_vitreous_positive(self):
        r = _run_default()
        assert r.t_half_vitreous_h > 0.0

    def test_time_above_therapeutic_nonnegative(self):
        r = _run_default()
        assert r.time_above_therapeutic_h >= 0.0

    def test_c_retina_starts_at_zero(self):
        r = _run_default()
        assert r.c_retina_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pharmacological behaviour
# ---------------------------------------------------------------------------

class TestPKBehaviour:
    def test_large_molecule_longer_half_life_than_small(self):
        r_large = simulate_intravitreal_pk(
            "Large", 500.0, mw_kDa=48.0, t_end_h=720.0
        )
        r_small = simulate_intravitreal_pk(
            "Small", 500.0, mw_kDa=0.5, t_end_h=720.0
        )
        assert r_large.t_half_vitreous_h > r_small.t_half_vitreous_h

    def test_dose_proportional_cmax(self):
        r1 = simulate_intravitreal_pk("Drug", 500.0, mw_kDa=48.0)
        r2 = simulate_intravitreal_pk("Drug", 1000.0, mw_kDa=48.0)
        ratio = r2.cmax_vitreous_mg_L / r1.cmax_vitreous_mg_L
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_vitreous_mostly_decreasing(self):
        r = _run_default(t_end_h=720.0, dt_h=1.0)
        cv = r.c_vitreous_mg_L
        # Check that last value < first value (monotone net decline)
        assert cv[-1] < cv[0]

    def test_retina_rises_before_falling(self):
        r = _run_default(t_end_h=720.0, dt_h=1.0)
        # tmax_retina should not be at time 0 or last step
        assert r.tmax_retina_h > 0.0
        assert r.tmax_retina_h < r.times_h[-1]

    def test_t_half_analytical(self):
        # For large molecule: k_total = 0.007 + 0.001 + k_ret
        # k_ret = max(0.001, 0.02 - 0.003 * max(0, logp))
        # logp = -2.0 -> k_ret = 0.02 - 0.003*0 = 0.02
        logp = -2.0
        k_vit_ant = 0.007
        k_vit_post = 0.001
        k_ret = max(0.001, 0.02 - 0.003 * max(0.0, logp))
        k_total = k_vit_ant + k_vit_post + k_ret
        expected_t_half = math.log(2.0) / k_total
        r = simulate_intravitreal_pk(
            "Ranibizumab", 500.0, mw_kDa=48.0, logp=logp
        )
        assert r.t_half_vitreous_h == pytest.approx(expected_t_half, rel=1e-6)


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mcg"):
            simulate_intravitreal_pk("Drug", dose_mcg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mcg"):
            simulate_intravitreal_pk("Drug", dose_mcg=-100.0)

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            simulate_intravitreal_pk("Drug", dose_mcg=500.0, mw_kDa=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_kDa"):
            simulate_intravitreal_pk("Drug", dose_mcg=500.0, mw_kDa=-1.0)
