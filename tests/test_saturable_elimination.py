"""Tests for core/saturable_elimination.py — Phase 585."""

import pytest

from omega_pbpk.core.saturable_elimination import (
    apparent_ke,
    dose_proportionality_check,
    mm_elimination_rate,
    simulate_mm_1cpt,
)

# ---------------------------------------------------------------------------
# mm_elimination_rate
# ---------------------------------------------------------------------------


class TestMmEliminationRate:
    def test_at_km_gives_half_vmax(self):
        """v = Vmax/2 when C = Km."""
        vmax = 10.0
        km = 5.0
        rate = mm_elimination_rate(km, vmax, km)
        assert abs(rate - vmax / 2) < 1e-9

    def test_at_high_concentration_approaches_vmax(self):
        """v -> Vmax when C >> Km."""
        vmax = 10.0
        km = 1.0
        c_high = 1000.0
        rate = mm_elimination_rate(c_high, vmax, km)
        assert rate > 0.99 * vmax

    def test_at_zero_concentration_gives_zero(self):
        """v = 0 when C = 0."""
        rate = mm_elimination_rate(0.0, 10.0, 5.0)
        assert rate == 0.0

    def test_proportional_at_low_concentration(self):
        """v ~ (Vmax/Km)*C when C << Km (linear regime)."""
        vmax = 10.0
        km = 100.0
        c = 1.0
        rate = mm_elimination_rate(c, vmax, km)
        expected = vmax * c / km
        assert abs(rate - expected) / expected < 0.02

    def test_validation_vmax_zero(self):
        with pytest.raises(ValueError, match="vmax"):
            mm_elimination_rate(5.0, 0.0, 5.0)

    def test_validation_km_zero(self):
        with pytest.raises(ValueError, match="km"):
            mm_elimination_rate(5.0, 10.0, 0.0)

    def test_validation_negative_concentration(self):
        with pytest.raises(ValueError, match="concentration"):
            mm_elimination_rate(-1.0, 10.0, 5.0)


# ---------------------------------------------------------------------------
# apparent_ke
# ---------------------------------------------------------------------------


class TestApparentKe:
    def test_decreases_with_increasing_concentration(self):
        """ke_apparent decreases as C increases (saturation)."""
        vmax, km, vd = 10.0, 5.0, 20.0
        ke_low = apparent_ke(0.1, vmax, km, vd)
        ke_mid = apparent_ke(km, vmax, km, vd)
        ke_high = apparent_ke(100.0, vmax, km, vd)
        assert ke_low > ke_mid > ke_high

    def test_formula_at_km(self):
        """ke = Vmax / (Vd * 2*Km) at C = Km."""
        vmax, km, vd = 8.0, 4.0, 10.0
        ke = apparent_ke(km, vmax, km, vd)
        expected = vmax / (vd * 2 * km)
        assert abs(ke - expected) < 1e-9

    def test_validation_vd_zero(self):
        with pytest.raises(ValueError, match="vd"):
            apparent_ke(5.0, 10.0, 5.0, 0.0)

    def test_validation_vmax_negative(self):
        with pytest.raises(ValueError, match="vmax"):
            apparent_ke(5.0, -1.0, 5.0, 10.0)


# ---------------------------------------------------------------------------
# simulate_mm_1cpt — IV route
# ---------------------------------------------------------------------------


class TestSimulateMm1cptIV:
    def test_iv_concentration_starts_at_dose_over_vd(self):
        sim = simulate_mm_1cpt("drug", 100.0, 20.0, 5.0, 2.0, route="iv")
        cp0 = sim["c_plasma_mg_L"][0]
        assert abs(cp0 - 100.0 / 20.0) < 1e-6

    def test_iv_concentration_monotonically_decreasing(self):
        sim = simulate_mm_1cpt("drug", 100.0, 20.0, 5.0, 2.0, route="iv")
        concs = sim["c_plasma_mg_L"]
        # Allow tiny floating point noise but overall must decrease
        for i in range(1, len(concs)):
            assert concs[i] <= concs[i - 1] + 1e-9

    def test_iv_auc_positive(self):
        sim = simulate_mm_1cpt("drug", 100.0, 20.0, 5.0, 2.0, route="iv")
        assert sim["auc"] > 0.0

    def test_iv_tmax_is_zero(self):
        """For IV bolus, Cmax is at t=0."""
        sim = simulate_mm_1cpt("drug", 100.0, 20.0, 5.0, 2.0, route="iv")
        assert sim["tmax_h"] == pytest.approx(0.0, abs=0.1)

    def test_iv_t_half_positive_and_finite(self):
        sim = simulate_mm_1cpt("drug", 100.0, 20.0, 5.0, 2.0, route="iv")
        assert 0 < sim["t_half_initial_h"] < float("inf")

    def test_returns_correct_keys(self):
        sim = simulate_mm_1cpt("mydrg", 50.0, 10.0, 3.0, 1.0, route="iv")
        for key in (
            "drug_name",
            "route",
            "dose_mg",
            "times_h",
            "c_plasma_mg_L",
            "cmax",
            "tmax_h",
            "auc",
            "t_half_initial_h",
        ):
            assert key in sim


# ---------------------------------------------------------------------------
# simulate_mm_1cpt — oral route
# ---------------------------------------------------------------------------


class TestSimulateMm1cptOral:
    def test_oral_cmax_positive(self):
        sim = simulate_mm_1cpt("oral_drug", 100.0, 20.0, 5.0, 2.0, ka_per_h=1.0, route="oral")
        assert sim["cmax"] > 0.0

    def test_oral_starts_at_zero(self):
        """Plasma starts at 0 for oral."""
        sim = simulate_mm_1cpt("oral_drug", 100.0, 20.0, 5.0, 2.0, ka_per_h=1.0, route="oral")
        assert sim["c_plasma_mg_L"][0] == pytest.approx(0.0, abs=1e-9)

    def test_oral_tmax_positive(self):
        """Tmax > 0 for oral."""
        sim = simulate_mm_1cpt("oral_drug", 100.0, 20.0, 5.0, 2.0, ka_per_h=1.0, route="oral")
        assert sim["tmax_h"] > 0.0

    def test_oral_validation_ka_missing(self):
        with pytest.raises(ValueError, match="ka_per_h"):
            simulate_mm_1cpt("x", 100.0, 20.0, 5.0, 2.0, ka_per_h=0.0, route="oral")


# ---------------------------------------------------------------------------
# Dose proportionality (non-linearity due to MM saturation)
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_double_dose_auc_more_than_doubles(self):
        """Super-proportional AUC increase due to MM saturation."""
        vmax, km, vd = 5.0, 2.0, 10.0
        res = dose_proportionality_check([50.0, 100.0], vd, vmax, km, t_end_h=48.0)
        auc_low = res[0]["auc"]
        auc_high = res[1]["auc"]
        # AUC should more than double when dose doubles (saturable elimination)
        assert auc_high > 2 * auc_low

    def test_dose_normalized_auc_increases_with_dose(self):
        """Dose-normalized AUC increases with dose (non-linear)."""
        vmax, km, vd = 5.0, 2.0, 10.0
        doses = [10.0, 50.0, 200.0]
        res = dose_proportionality_check(doses, vd, vmax, km, t_end_h=48.0)
        norm_aucs = [r["dose_normalized_auc"] for r in res]
        # Higher doses → larger normalized AUC (saturation)
        assert norm_aucs[0] < norm_aucs[1] < norm_aucs[2]

    def test_non_linear_flag_set_for_saturating_doses(self):
        vmax, km, vd = 5.0, 1.0, 10.0
        res = dose_proportionality_check([5.0, 500.0], vd, vmax, km, t_end_h=72.0)
        assert res[0]["non_linear"] is True

    def test_returns_all_required_keys(self):
        res = dose_proportionality_check([10.0, 20.0], 10.0, 5.0, 2.0)
        for r in res:
            for key in (
                "dose",
                "cmax",
                "auc",
                "dose_normalized_cmax",
                "dose_normalized_auc",
                "non_linear",
            ):
                assert key in r

    def test_validation_empty_doses(self):
        with pytest.raises(ValueError):
            dose_proportionality_check([], 10.0, 5.0, 2.0)

    def test_validation_zero_vd(self):
        with pytest.raises(ValueError):
            dose_proportionality_check([10.0], 0.0, 5.0, 2.0)
