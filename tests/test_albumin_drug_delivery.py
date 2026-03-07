"""Tests for albumin-mediated drug delivery model (Phase 285)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.albumin_drug_delivery import (
    AlbuminBindingResult,
    AlbuminDeliveryResult,
    albumin_binding,
    compare_facilitation,
    simulate_albumin_facilitated_pk,
)


# ---------------------------------------------------------------------------
# albumin_binding — unit tests
# ---------------------------------------------------------------------------

class TestAlbuminBinding:
    """Tests for the albumin_binding() function."""

    def test_returns_albumin_binding_result(self):
        result = albumin_binding(1.0, 0.01)
        assert isinstance(result, AlbuminBindingResult)

    def test_zero_concentration(self):
        result = albumin_binding(0.0, 0.01)
        assert result.c_free == 0.0
        assert result.c_bound == 0.0
        assert result.fu == 1.0

    def test_mass_balance(self):
        result = albumin_binding(5.0, 0.05)
        assert math.isclose(result.c_free + result.c_bound, result.c_total, rel_tol=1e-6)

    def test_fu_in_unit_interval(self):
        for c in [0.1, 1.0, 10.0, 100.0]:
            result = albumin_binding(c, 0.01)
            assert 0.0 <= result.fu <= 1.0

    def test_high_ka_binds_most_drug(self):
        """Very high association constant → most drug is bound."""
        result = albumin_binding(1.0, 100.0)
        assert result.fu < 0.5  # mostly bound

    def test_low_ka_leaves_most_drug_free(self):
        """Very low association constant → most drug is free."""
        result = albumin_binding(1.0, 1e-6)
        assert result.fu > 0.99

    def test_fu_decreases_with_higher_ka(self):
        """Higher affinity should reduce free fraction."""
        r1 = albumin_binding(1.0, 0.001)
        r2 = albumin_binding(1.0, 0.1)
        assert r2.fu < r1.fu

    def test_fu_increases_with_higher_concentration(self):
        """At high drug concentration, albumin saturates → fu increases."""
        r_low = albumin_binding(1.0, 0.05)
        r_high = albumin_binding(10000.0, 0.05)  # saturating
        assert r_high.fu > r_low.fu

    def test_default_bmax(self):
        """Default bmax is 630 mg/L (albumin ~0.63 mM)."""
        result = albumin_binding(1.0, 0.01, bmax_albumin_mg_L=630.0)
        assert result.c_bound >= 0.0

    def test_custom_bmax(self):
        """Higher bmax should bind more drug (lower fu)."""
        r1 = albumin_binding(1.0, 0.01, bmax_albumin_mg_L=100.0)
        r2 = albumin_binding(1.0, 0.01, bmax_albumin_mg_L=1000.0)
        assert r2.fu < r1.fu

    def test_result_is_frozen(self):
        result = albumin_binding(1.0, 0.01)
        with pytest.raises((AttributeError, TypeError)):
            result.fu = 0.5  # type: ignore[misc]

    def test_notes_is_string(self):
        result = albumin_binding(1.0, 0.01)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="c_total_mg_L"):
            albumin_binding(-1.0, 0.01)

    def test_nonpositive_ka_raises(self):
        with pytest.raises(ValueError, match="ka_L_per_mg"):
            albumin_binding(1.0, 0.0)

    def test_nonpositive_bmax_raises(self):
        with pytest.raises(ValueError, match="bmax_albumin_mg_L"):
            albumin_binding(1.0, 0.01, bmax_albumin_mg_L=0.0)


# ---------------------------------------------------------------------------
# simulate_albumin_facilitated_pk — unit tests
# ---------------------------------------------------------------------------

class TestSimulateAlbuminFacilitatedPk:
    """Tests for the simulate_albumin_facilitated_pk() function."""

    def _default(self, **kwargs) -> AlbuminDeliveryResult:
        defaults = dict(
            dose_mg=100.0,
            vd_L=50.0,
            cl_intrinsic_L_per_h=5.0,
            ka_albumin_L_per_mg=0.01,
            bmax_albumin_mg_L=630.0,
            facilitation_factor=0.3,
            ka_abs_per_h=1.0,
            t_end_h=24.0,
            dt_h=0.1,
        )
        defaults.update(kwargs)
        return simulate_albumin_facilitated_pk(**defaults)

    def test_returns_albumin_delivery_result(self):
        result = self._default()
        assert isinstance(result, AlbuminDeliveryResult)

    def test_result_is_frozen(self):
        result = self._default()
        with pytest.raises((AttributeError, TypeError)):
            result.cmax_total = 0.0  # type: ignore[misc]

    def test_times_start_at_zero(self):
        result = self._default()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = self._default(t_end_h=12.0)
        assert result.times_h[-1] == pytest.approx(12.0)

    def test_initial_concentration_zero(self):
        result = self._default()
        assert result.c_total[0] == pytest.approx(0.0)

    def test_cmax_positive(self):
        result = self._default()
        assert result.cmax_total > 0.0

    def test_auc_positive(self):
        result = self._default()
        assert result.auc_total > 0.0

    def test_profile_length_matches_times(self):
        result = self._default()
        assert len(result.c_total) == len(result.times_h)
        assert len(result.c_free_profile) == len(result.times_h)
        assert len(result.fu_profile) == len(result.times_h)

    def test_fu_profile_in_unit_interval(self):
        result = self._default()
        for fu in result.fu_profile:
            assert 0.0 <= fu <= 1.0

    def test_free_le_total(self):
        """Free concentration must be <= total concentration at all times."""
        result = self._default()
        for cf, ct in zip(result.c_free_profile, result.c_total):
            assert cf <= ct + 1e-10

    def test_facilitation_factor_stored(self):
        result = self._default(facilitation_factor=0.5)
        assert result.facilitation_factor == pytest.approx(0.5)

    def test_higher_facilitation_increases_clearance(self):
        """Higher facilitation → faster elimination → lower AUC."""
        r_low = self._default(facilitation_factor=0.0)
        r_high = self._default(facilitation_factor=1.0)
        assert r_high.auc_total < r_low.auc_total

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            self._default(dose_mg=-1.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            self._default(vd_L=0.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_intrinsic"):
            self._default(cl_intrinsic_L_per_h=-5.0)

    def test_invalid_ka_albumin_raises(self):
        with pytest.raises(ValueError, match="ka_albumin"):
            self._default(ka_albumin_L_per_mg=0.0)

    def test_invalid_facilitation_factor_raises(self):
        with pytest.raises(ValueError, match="facilitation_factor"):
            self._default(facilitation_factor=1.5)

    def test_invalid_ka_abs_raises(self):
        with pytest.raises(ValueError, match="ka_abs_per_h"):
            self._default(ka_abs_per_h=-1.0)

    def test_notes_is_string(self):
        result = self._default()
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# compare_facilitation — tests
# ---------------------------------------------------------------------------

class TestCompareFacilitation:
    """Tests for compare_facilitation() function."""

    def test_returns_dict_with_three_keys(self):
        result = compare_facilitation(
            dose_mg=100.0,
            vd_L=50.0,
            cl_intrinsic_L_per_h=5.0,
            ka_albumin=0.01,
        )
        assert set(result.keys()) == {"no_facilitation", "partial_facilitation", "full_facilitation"}

    def test_each_value_is_albumin_delivery_result(self):
        result = compare_facilitation(
            dose_mg=100.0,
            vd_L=50.0,
            cl_intrinsic_L_per_h=5.0,
            ka_albumin=0.01,
        )
        for key, val in result.items():
            assert isinstance(val, AlbuminDeliveryResult), f"{key} is not AlbuminDeliveryResult"

    def test_facilitation_factors_are_correct(self):
        result = compare_facilitation(
            dose_mg=100.0,
            vd_L=50.0,
            cl_intrinsic_L_per_h=5.0,
            ka_albumin=0.01,
        )
        assert result["no_facilitation"].facilitation_factor == pytest.approx(0.0)
        assert result["partial_facilitation"].facilitation_factor == pytest.approx(0.3)
        assert result["full_facilitation"].facilitation_factor == pytest.approx(1.0)

    def test_auc_ordering(self):
        """AUC: no facilitation > partial > full facilitation."""
        result = compare_facilitation(
            dose_mg=100.0,
            vd_L=50.0,
            cl_intrinsic_L_per_h=5.0,
            ka_albumin=0.01,
        )
        auc_none = result["no_facilitation"].auc_total
        auc_partial = result["partial_facilitation"].auc_total
        auc_full = result["full_facilitation"].auc_total
        assert auc_none >= auc_partial >= auc_full
