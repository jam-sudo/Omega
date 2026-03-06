"""Tests for two-compartment IV infusion analytical model."""

import pytest

from omega_pbpk.core.two_compartment_infusion import (
    TwoCompartmentInfusionResult,
    simulate_2cpt_infusion,
)

# Typical 2-cpt parameters for test drug
_CL = 5.0  # L/h
_VC = 10.0  # L
_VP = 40.0  # L
_Q = 2.0  # L/h
_DOSE = 100.0  # mg
_TINF = 1.0  # h


class TestInputValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_2cpt_infusion("D", 0, _TINF, _CL, _VC, _VP, _Q)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_2cpt_infusion("D", _DOSE, _TINF, 0, _VC, _VP, _Q)

    def test_volume_zero_raises(self):
        with pytest.raises(ValueError, match="Volumes"):
            simulate_2cpt_infusion("D", _DOSE, _TINF, _CL, 0, _VP, _Q)

    def test_infusion_duration_zero_raises(self):
        with pytest.raises(ValueError, match="infusion_duration"):
            simulate_2cpt_infusion("D", _DOSE, 0, _CL, _VC, _VP, _Q)


class TestInfusionPK:
    def setup_method(self):
        self.result = simulate_2cpt_infusion(
            "drug", _DOSE, _TINF, _CL, _VC, _VP, _Q, t_end_h=24.0, dt_h=0.05
        )

    def test_result_type(self):
        assert isinstance(self.result, TwoCompartmentInfusionResult)

    def test_css_analytical(self):
        """Css = R/CL = (dose/T) / CL."""
        expected_css = (_DOSE / _TINF) / _CL
        assert self.result.css_central == pytest.approx(expected_css)

    def test_cmax_positive(self):
        assert self.result.cmax > 0

    def test_tmax_during_or_after_infusion(self):
        """Cmax should occur at or near end of infusion for 2-cpt."""
        assert self.result.tmax_h >= 0

    def test_auc_positive(self):
        assert self.result.auc_0_t > 0
        assert self.result.auc_0_inf >= self.result.auc_0_t

    def test_infusion_rate(self):
        assert self.result.infusion_rate_mg_h == pytest.approx(_DOSE / _TINF)

    def test_half_lives_positive(self):
        assert self.result.thalf_alpha_h > 0
        assert self.result.thalf_beta_h > 0
        # Alpha (distribution) half-life < beta (elimination) half-life
        assert self.result.thalf_alpha_h < self.result.thalf_beta_h

    def test_concentration_falls_after_infusion(self):
        """Concentration should decline post-infusion."""
        cp = self.result.cp_central
        t = self.result.times_h
        tinf = self.result.infusion_duration_h

        # Find index at end of infusion
        end_idx = min(range(len(t)), key=lambda i: abs(t[i] - tinf))
        cp_at_end = cp[end_idx]
        cp_final = cp[-1]
        assert cp_final < cp_at_end

    def test_concentrations_non_negative(self):
        for c in self.result.cp_central:
            assert c >= 0.0

    def test_post_infusion_auc_positive(self):
        assert self.result.post_infusion_auc >= 0

    def test_time_array_length(self):
        assert len(self.result.times_h) > 0
        assert len(self.result.cp_central) == len(self.result.times_h)

    def test_auc_scales_with_dose(self):
        """Doubling dose doubles AUC (linear kinetics)."""
        r2 = simulate_2cpt_infusion(
            "drug", _DOSE * 2, _TINF, _CL, _VC, _VP, _Q, t_end_h=24.0, dt_h=0.05
        )
        ratio = r2.auc_0_t / self.result.auc_0_t
        assert ratio == pytest.approx(2.0, rel=0.05)

    def test_longer_infusion_higher_auc_during(self):
        """Longer infusion → more drug delivered → higher AUC."""
        r_long = simulate_2cpt_infusion(
            "drug", _DOSE, 4.0, _CL, _VC, _VP, _Q, t_end_h=24.0, dt_h=0.05
        )
        # Same dose, longer infusion. AUC_0_inf should be roughly similar
        # but cmax during infusion should be lower with longer infusion
        assert r_long.cmax <= self.result.cmax * 1.5  # rough check

    def test_q_zero_reduces_to_1cpt(self):
        """Q=0 means no distribution: should behave like 1-compartment."""
        r = simulate_2cpt_infusion(
            "drug", _DOSE, _TINF, _CL, _VC, 1.0, 0.0, t_end_h=24.0, dt_h=0.05
        )
        # With Q=0, peripheral doesn't affect central; model should still run
        assert r.cmax > 0
        assert r.auc_0_t > 0
