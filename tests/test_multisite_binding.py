"""Tests for Phase 557 — core/multisite_binding.py"""

import math

import pytest

from omega_pbpk.core.multisite_binding import (
    BindingResult,
    competitive_binding_ki,
    fit_binding_curve,
    hill_binding,
    receptor_occupancy_curve,
    receptor_reserve,
    two_site_binding,
)

# ---------------------------------------------------------------------------
# hill_binding
# ---------------------------------------------------------------------------


class TestHillBinding:
    def test_half_occupancy_at_kd(self):
        """At C == Kd, occupancy should be 0.5 for Hill=1."""
        assert hill_binding(1.0, 1.0, 1.0) == pytest.approx(0.5, rel=1e-6)

    def test_half_occupancy_at_kd_any_value(self):
        """At C == Kd, occupancy should be 0.5 regardless of Kd value."""
        for kd in [0.1, 1.0, 10.0, 100.0]:
            assert hill_binding(kd, kd, 1.0) == pytest.approx(0.5, rel=1e-6)

    def test_zero_concentration_returns_zero(self):
        assert hill_binding(0.0, 1.0) == 0.0

    def test_high_concentration_approaches_one(self):
        occ = hill_binding(1e6, 1.0, 1.0)
        assert occ > 0.999

    def test_default_hill_coefficient_is_one(self):
        """Default n_hill=1 should give simple Langmuir binding."""
        assert hill_binding(2.0, 2.0) == pytest.approx(0.5, rel=1e-6)

    def test_high_hill_coefficient_steeper_curve(self):
        """n_hill > 1 gives steeper sigmoidal curve: low C → less than n=1."""
        occ_n1 = hill_binding(0.5, 1.0, 1.0)
        occ_n3 = hill_binding(0.5, 1.0, 3.0)
        assert occ_n3 < occ_n1

    def test_high_hill_high_concentration_more_than_n1(self):
        """Above Kd, high Hill coefficient gives higher occupancy than n=1."""
        occ_n1 = hill_binding(2.0, 1.0, 1.0)
        occ_n3 = hill_binding(2.0, 1.0, 3.0)
        assert occ_n3 > occ_n1

    def test_occupancy_in_unit_interval(self):
        for c in [0.0, 0.1, 1.0, 10.0, 1000.0]:
            occ = hill_binding(c, 1.0, 2.0)
            assert 0.0 <= occ <= 1.0

    def test_invalid_negative_concentration(self):
        with pytest.raises(ValueError):
            hill_binding(-1.0, 1.0)

    def test_invalid_zero_kd(self):
        with pytest.raises(ValueError):
            hill_binding(1.0, 0.0)

    def test_invalid_negative_kd(self):
        with pytest.raises(ValueError):
            hill_binding(1.0, -1.0)

    def test_invalid_zero_n_hill(self):
        with pytest.raises(ValueError):
            hill_binding(1.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# two_site_binding
# ---------------------------------------------------------------------------


class TestTwoSiteBinding:
    def test_total_equals_sum_of_sites(self):
        result = two_site_binding(5.0, kd1=1.0, bmax1=10.0, kd2=10.0, bmax2=5.0)
        assert result["total_bound"] == pytest.approx(
            result["bound_site1"] + result["bound_site2"], rel=1e-9
        )

    def test_zero_concentration_all_zero(self):
        result = two_site_binding(0.0, kd1=1.0, bmax1=10.0, kd2=5.0, bmax2=5.0)
        assert result["bound_site1"] == 0.0
        assert result["bound_site2"] == 0.0
        assert result["total_bound"] == 0.0

    def test_occupancies_in_unit_interval(self):
        result = two_site_binding(5.0, kd1=1.0, bmax1=10.0, kd2=10.0, bmax2=5.0)
        assert 0.0 <= result["occupancy_site1"] <= 1.0
        assert 0.0 <= result["occupancy_site2"] <= 1.0

    def test_high_affinity_site_more_occupied(self):
        """Site 1 (lower Kd = higher affinity) should have greater occupancy."""
        result = two_site_binding(2.0, kd1=0.5, bmax1=10.0, kd2=20.0, bmax2=10.0)
        assert result["occupancy_site1"] > result["occupancy_site2"]

    def test_invalid_negative_concentration(self):
        with pytest.raises(ValueError):
            two_site_binding(-1.0, 1.0, 10.0, 5.0, 5.0)

    def test_invalid_kd1_zero(self):
        with pytest.raises(ValueError):
            two_site_binding(1.0, 0.0, 10.0, 5.0, 5.0)

    def test_invalid_bmax_zero(self):
        with pytest.raises(ValueError):
            two_site_binding(1.0, 1.0, 0.0, 5.0, 5.0)


# ---------------------------------------------------------------------------
# receptor_occupancy_curve
# ---------------------------------------------------------------------------


class TestReceptorOccupancyCurve:
    def test_returns_correct_number_of_points(self):
        curve = receptor_occupancy_curve(kd=1.0, n_points=50)
        assert len(curve) == 50

    def test_all_occupancies_in_unit_interval(self):
        curve = receptor_occupancy_curve(kd=1.0, n_hill=2.0)
        for pt in curve:
            assert 0.0 <= pt["occupancy"] <= 1.0

    def test_occupancy_increasing_with_concentration(self):
        curve = receptor_occupancy_curve(kd=1.0)
        occupancies = [pt["occupancy"] for pt in curve]
        for i in range(1, len(occupancies)):
            assert occupancies[i] >= occupancies[i - 1]

    def test_kd_at_half_occupancy(self):
        """The point closest to Kd should have occupancy ~0.5."""
        kd = 1.0
        curve = receptor_occupancy_curve(kd=kd, c_range_min=0.01, c_range_max=100.0)
        # Find point with concentration closest to kd
        closest = min(curve, key=lambda pt: abs(pt["concentration"] - kd))
        assert closest["occupancy"] == pytest.approx(0.5, abs=0.05)

    def test_has_concentration_and_occupancy_keys(self):
        curve = receptor_occupancy_curve(kd=1.0)
        assert "concentration" in curve[0]
        assert "occupancy" in curve[0]

    def test_invalid_kd(self):
        with pytest.raises(ValueError):
            receptor_occupancy_curve(kd=0.0)

    def test_invalid_n_points(self):
        with pytest.raises(ValueError):
            receptor_occupancy_curve(kd=1.0, n_points=1)


# ---------------------------------------------------------------------------
# competitive_binding_ki
# ---------------------------------------------------------------------------


class TestCompetitiveBindingKi:
    def test_cheng_prusoff_at_kd_radioligand(self):
        """When [L] == Kd_radioligand, Ki = IC50 / 2."""
        kd_rl = 5.0
        ic50 = 10.0
        ki = competitive_binding_ki(ic50, kd_radioligand=kd_rl, conc_radioligand=kd_rl)
        assert ki == pytest.approx(ic50 / 2.0, rel=1e-9)

    def test_no_radioligand_ki_equals_ic50(self):
        """When [L] == 0, Ki == IC50."""
        ki = competitive_binding_ki(10.0, kd_radioligand=5.0, conc_radioligand=0.0)
        assert ki == pytest.approx(10.0, rel=1e-9)

    def test_ki_less_than_ic50(self):
        ki = competitive_binding_ki(10.0, kd_radioligand=1.0, conc_radioligand=5.0)
        assert ki < 10.0

    def test_invalid_ic50(self):
        with pytest.raises(ValueError):
            competitive_binding_ki(0.0, 1.0, 1.0)

    def test_invalid_kd_radioligand(self):
        with pytest.raises(ValueError):
            competitive_binding_ki(10.0, 0.0, 1.0)

    def test_invalid_negative_conc(self):
        with pytest.raises(ValueError):
            competitive_binding_ki(10.0, 1.0, -1.0)


# ---------------------------------------------------------------------------
# receptor_reserve
# ---------------------------------------------------------------------------


class TestReceptorReserve:
    def test_positive_reserve_when_response_ec50_lower(self):
        rr = receptor_reserve(ec50_receptor=10.0, ec50_response=1.0)
        assert rr == pytest.approx(90.0, rel=1e-9)

    def test_zero_reserve_when_equal(self):
        rr = receptor_reserve(ec50_receptor=5.0, ec50_response=5.0)
        assert rr == 0.0

    def test_zero_reserve_when_response_ec50_higher(self):
        rr = receptor_reserve(ec50_receptor=1.0, ec50_response=10.0)
        assert rr == 0.0

    def test_result_in_range_0_100(self):
        rr = receptor_reserve(ec50_receptor=100.0, ec50_response=0.1)
        assert 0.0 <= rr <= 100.0

    def test_invalid_ec50_receptor_zero(self):
        with pytest.raises(ValueError):
            receptor_reserve(0.0, 1.0)

    def test_invalid_ec50_response_zero(self):
        with pytest.raises(ValueError):
            receptor_reserve(1.0, 0.0)


# ---------------------------------------------------------------------------
# fit_binding_curve
# ---------------------------------------------------------------------------


class TestFitBindingCurve:
    def _make_hill_data(self, kd, n_hill, n_pts=20):
        log_min, log_max = math.log10(kd * 0.01), math.log10(kd * 100)
        step = (log_max - log_min) / (n_pts - 1)
        concs = [10 ** (log_min + i * step) for i in range(n_pts)]
        occs = [hill_binding(c, kd, n_hill) for c in concs]
        return concs, occs

    def test_returns_binding_result_instance(self):
        concs, occs = self._make_hill_data(1.0, 1.0)
        result = fit_binding_curve(concs, occs)
        assert isinstance(result, BindingResult)

    def test_recovers_kd_approximately(self):
        true_kd = 2.0
        concs, occs = self._make_hill_data(true_kd, 1.0)
        result = fit_binding_curve(concs, occs)
        # Within a factor of 3 of true kd
        assert result.kd == pytest.approx(true_kd, rel=0.5)

    def test_ec50_equals_kd(self):
        concs, occs = self._make_hill_data(5.0, 1.0)
        result = fit_binding_curve(concs, occs)
        assert result.ec50 == result.kd

    def test_bmax_is_one(self):
        concs, occs = self._make_hill_data(1.0, 1.0)
        result = fit_binding_curve(concs, occs)
        assert result.bmax == pytest.approx(1.0, rel=1e-9)

    def test_r_squared_high_for_perfect_data(self):
        concs, occs = self._make_hill_data(1.0, 1.0)
        result = fit_binding_curve(concs, occs)
        assert result.r_squared > 0.95

    def test_r_squared_in_unit_interval(self):
        concs, occs = self._make_hill_data(1.0, 2.0)
        result = fit_binding_curve(concs, occs)
        assert 0.0 <= result.r_squared <= 1.0

    def test_notes_not_empty(self):
        concs, occs = self._make_hill_data(1.0, 1.0)
        result = fit_binding_curve(concs, occs)
        assert len(result.notes) > 0

    def test_invalid_mismatched_lengths(self):
        with pytest.raises(ValueError):
            fit_binding_curve([1.0, 2.0], [0.5])

    def test_invalid_too_few_points(self):
        with pytest.raises(ValueError):
            fit_binding_curve([1.0], [0.5])

    def test_invalid_non_positive_concentration(self):
        with pytest.raises(ValueError):
            fit_binding_curve([0.0, 1.0], [0.0, 0.5])

    def test_invalid_occupancy_out_of_range(self):
        with pytest.raises(ValueError):
            fit_binding_curve([1.0, 2.0], [0.5, 1.5])
