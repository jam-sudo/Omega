"""Tests for absorption rate constant calculator (Phase 300)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.ka_calculator import (
    KaResult,
    absorption_window_ka,
    effective_ka,
    flip_flop_check,
    ka_from_dissolution,
    ka_from_peff,
)

_S_PER_H = 3600.0


# ---------------------------------------------------------------------------
# ka_from_peff
# ---------------------------------------------------------------------------


class TestKaFromPeff:
    def test_basic_calculation(self):
        # ka = 2 * Peff / R * 3600
        peff = 1e-4  # cm/s
        r = 1.75  # cm
        expected = 2.0 * peff / r * _S_PER_H
        assert ka_from_peff(peff, r) == pytest.approx(expected, rel=1e-6)

    def test_default_radius(self):
        peff = 1e-4
        result = ka_from_peff(peff)
        expected = 2.0 * peff / 1.75 * _S_PER_H
        assert result == pytest.approx(expected, rel=1e-6)

    def test_returns_per_hour(self):
        # For typical Peff ~1e-4 cm/s → ka should be ~0.4 h^-1
        result = ka_from_peff(1e-4)
        assert result == pytest.approx(0.411, rel=0.01)

    def test_zero_peff_raises(self):
        with pytest.raises(ValueError):
            ka_from_peff(0.0)

    def test_negative_peff_raises(self):
        with pytest.raises(ValueError):
            ka_from_peff(-1e-4)

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            ka_from_peff(1e-4, radius_intestine_cm=0.0)

    def test_ka_scales_linearly_with_peff(self):
        ka1 = ka_from_peff(1e-4)
        ka2 = ka_from_peff(2e-4)
        assert ka2 == pytest.approx(2.0 * ka1, rel=1e-6)

    def test_ka_decreases_with_larger_radius(self):
        ka_small = ka_from_peff(1e-4, radius_intestine_cm=1.0)
        ka_large = ka_from_peff(1e-4, radius_intestine_cm=2.0)
        assert ka_small > ka_large


# ---------------------------------------------------------------------------
# ka_from_dissolution
# ---------------------------------------------------------------------------


class TestKaFromDissolution:
    def test_basic_calculation(self):
        # k_diss = rate / (sol * vol), ka = min(k_diss * 60, 10.0)
        rate = 1.0  # mg/min
        sol = 0.1  # mg/mL
        vol = 250.0  # mL
        k_diss_per_min = rate / (sol * vol)
        expected = min(k_diss_per_min * 60.0, 10.0)
        assert ka_from_dissolution(rate, 100.0, sol, vol) == pytest.approx(expected, rel=1e-6)

    def test_cap_at_10(self):
        # Very high dissolution rate → capped at 10 h^-1
        result = ka_from_dissolution(1000.0, 100.0, 0.001, 1.0)
        assert result == pytest.approx(10.0)

    def test_zero_rate_raises(self):
        with pytest.raises(ValueError):
            ka_from_dissolution(0.0, 100.0, 1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            ka_from_dissolution(1.0, 0.0, 1.0)

    def test_zero_solubility_raises(self):
        with pytest.raises(ValueError):
            ka_from_dissolution(1.0, 100.0, 0.0)

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError):
            ka_from_dissolution(1.0, 100.0, 1.0, volume_mL=0.0)

    def test_result_non_negative(self):
        result = ka_from_dissolution(0.5, 50.0, 0.5, 250.0)
        assert result >= 0


# ---------------------------------------------------------------------------
# effective_ka
# ---------------------------------------------------------------------------


class TestEffectiveKa:
    def _default_args(self, bcs_class):
        return dict(
            peff_cm_s=1e-4,
            dissolution_rate_mg_per_min=0.5,
            dose_mg=100.0,
            solubility_mg_mL=0.5,
            bcs_class=bcs_class,
        )

    def test_returns_ka_result(self):
        result = effective_ka(**self._default_args(1))
        assert isinstance(result, KaResult)

    def test_bcs1_both_fast(self):
        result = effective_ka(**self._default_args(1))
        assert result.rate_limiting_step == "both_fast"

    def test_bcs2_dissolution_limited(self):
        result = effective_ka(**self._default_args(2))
        assert result.rate_limiting_step == "dissolution"
        assert result.effective_ka_per_h == pytest.approx(result.ka_diss_per_h)

    def test_bcs3_permeability_limited(self):
        result = effective_ka(**self._default_args(3))
        assert result.rate_limiting_step == "permeability"
        assert result.effective_ka_per_h == pytest.approx(result.ka_peff_per_h)

    def test_bcs4_both_slow(self):
        result = effective_ka(**self._default_args(4))
        assert result.rate_limiting_step == "both_slow"

    def test_bcs1_effective_ka_min_of_both(self):
        result = effective_ka(**self._default_args(1))
        assert result.effective_ka_per_h == pytest.approx(
            min(result.ka_peff_per_h, result.ka_diss_per_h), rel=1e-6
        )

    def test_bcs4_effective_ka_min_of_both(self):
        result = effective_ka(**self._default_args(4))
        assert result.effective_ka_per_h == pytest.approx(
            min(result.ka_peff_per_h, result.ka_diss_per_h), rel=1e-6
        )

    def test_invalid_bcs_class_raises(self):
        with pytest.raises(ValueError):
            effective_ka(1e-4, 0.5, 100.0, 0.5, bcs_class=5)

    def test_zero_bcs_class_raises(self):
        with pytest.raises(ValueError):
            effective_ka(1e-4, 0.5, 100.0, 0.5, bcs_class=0)

    def test_ka_peff_correct(self):
        result = effective_ka(
            peff_cm_s=1e-4,
            dissolution_rate_mg_per_min=0.5,
            dose_mg=100.0,
            solubility_mg_mL=0.5,
            bcs_class=1,
        )
        assert result.ka_peff_per_h == pytest.approx(ka_from_peff(1e-4), rel=1e-6)

    def test_frozen_dataclass(self):
        result = effective_ka(**self._default_args(1))
        with pytest.raises(Exception):
            result.effective_ka_per_h = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# flip_flop_check
# ---------------------------------------------------------------------------


class TestFlipFlopCheck:
    def test_no_flip_flop_when_ka_lt_ke(self):
        result = flip_flop_check(ka=0.5, ke=1.0)
        assert result["flip_flop"] is False

    def test_flip_flop_when_ka_gt_ke(self):
        result = flip_flop_check(ka=2.0, ke=0.5)
        assert result["flip_flop"] is True

    def test_apparent_t_half_normal(self):
        # Normal: apparent t_half = ln(2)/ke
        result = flip_flop_check(ka=0.5, ke=1.0)
        expected = math.log(2) / min(0.5, 1.0)
        assert result["apparent_t_half_h"] == pytest.approx(expected, rel=1e-6)

    def test_apparent_t_half_flip_flop(self):
        # Flip-flop: apparent t_half = ln(2)/min(ka,ke) = ln(2)/ke
        result = flip_flop_check(ka=2.0, ke=0.5)
        expected = math.log(2) / 0.5
        assert result["apparent_t_half_h"] == pytest.approx(expected, rel=1e-6)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError):
            flip_flop_check(ka=0.0, ke=1.0)

    def test_zero_ke_raises(self):
        with pytest.raises(ValueError):
            flip_flop_check(ka=1.0, ke=0.0)

    def test_notes_string_present(self):
        result = flip_flop_check(ka=0.5, ke=1.0)
        assert isinstance(result["notes"], str)
        assert len(result["notes"]) > 0

    def test_equal_ka_ke(self):
        # ka == ke → no flip-flop (ka not > ke)
        result = flip_flop_check(ka=1.0, ke=1.0)
        assert result["flip_flop"] is False


# ---------------------------------------------------------------------------
# absorption_window_ka
# ---------------------------------------------------------------------------


class TestAbsorptionWindowKa:
    def test_basic_fa_calculation(self):
        result = absorption_window_ka(ka_per_h=1.0, transit_time_h=4.0)
        expected = 1.0 - math.exp(-1.0 * 4.0)
        assert result["fa"] == pytest.approx(expected, rel=1e-6)

    def test_fa_between_0_and_1(self):
        result = absorption_window_ka(ka_per_h=0.2, transit_time_h=4.0)
        assert 0 < result["fa"] < 1

    def test_high_ka_high_fa(self):
        result = absorption_window_ka(ka_per_h=10.0, transit_time_h=4.0)
        assert result["fa"] > 0.9

    def test_low_ka_low_fa(self):
        result = absorption_window_ka(ka_per_h=0.05, transit_time_h=1.0)
        assert result["fa"] < 0.1

    def test_default_transit_time(self):
        result = absorption_window_ka(ka_per_h=1.0)
        expected = 1.0 - math.exp(-1.0 * 4.0)
        assert result["fa"] == pytest.approx(expected, rel=1e-6)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError):
            absorption_window_ka(ka_per_h=0.0)

    def test_zero_transit_time_raises(self):
        with pytest.raises(ValueError):
            absorption_window_ka(ka_per_h=1.0, transit_time_h=0.0)

    def test_ka_preserved_in_result(self):
        result = absorption_window_ka(ka_per_h=1.5, transit_time_h=3.0)
        assert result["ka"] == pytest.approx(1.5)

    def test_transit_time_preserved_in_result(self):
        result = absorption_window_ka(ka_per_h=1.5, transit_time_h=3.0)
        assert result["transit_time_h"] == pytest.approx(3.0)

    def test_notes_string_present(self):
        result = absorption_window_ka(ka_per_h=1.0)
        assert isinstance(result["notes"], str)
        assert len(result["notes"]) > 0
