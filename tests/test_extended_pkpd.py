"""Tests for clinical/extended_pkpd.py — Phase 502."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.extended_pkpd import (
    PKPDResult,
    calculate_pdp_index,
    emax_model,
    sigmoidal_emax,
    simulate_pkpd,
    time_above_ec50,
)

# ---------------------------------------------------------------------------
# emax_model
# ---------------------------------------------------------------------------


class TestEmaxModel:
    def test_at_ec50_half_emax(self):
        # E = E0 + Emax/2 when C = EC50
        result = emax_model(conc=5.0, emax=10.0, ec50=5.0, e0=0.0)
        assert result == pytest.approx(5.0)

    def test_at_ec50_with_baseline(self):
        result = emax_model(conc=5.0, emax=10.0, ec50=5.0, e0=2.0)
        assert result == pytest.approx(7.0)

    def test_zero_conc_returns_baseline(self):
        result = emax_model(conc=0.0, emax=10.0, ec50=5.0, e0=3.0)
        assert result == pytest.approx(3.0)

    def test_high_conc_approaches_emax_plus_e0(self):
        result = emax_model(conc=1e9, emax=10.0, ec50=5.0, e0=1.0)
        assert result == pytest.approx(11.0, rel=1e-4)

    def test_invalid_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50 must be positive"):
            emax_model(5.0, 10.0, ec50=0.0)

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="conc must be"):
            emax_model(-1.0, 10.0, ec50=5.0)


# ---------------------------------------------------------------------------
# sigmoidal_emax
# ---------------------------------------------------------------------------


class TestSigmoidalEmax:
    def test_hill_n1_same_as_emax_model(self):
        c, emax, ec50, e0 = 3.0, 10.0, 5.0, 1.0
        sig = sigmoidal_emax(c, emax, ec50, hill_n=1.0, e0=e0)
        direct = emax_model(c, emax, ec50, e0=e0)
        assert sig == pytest.approx(direct, rel=1e-6)

    def test_at_ec50_half_emax_regardless_of_hill(self):
        result = sigmoidal_emax(conc=5.0, emax=10.0, ec50=5.0, hill_n=3.0, e0=0.0)
        assert result == pytest.approx(5.0)

    def test_hill_n2_steeper_than_n1_below_ec50(self):
        # Below EC50, higher Hill n → less effect
        e_n1 = sigmoidal_emax(2.0, 10.0, 5.0, hill_n=1.0, e0=0.0)
        e_n2 = sigmoidal_emax(2.0, 10.0, 5.0, hill_n=2.0, e0=0.0)
        assert e_n2 < e_n1

    def test_hill_n2_steeper_above_ec50(self):
        # Above EC50, higher Hill n → more effect
        e_n1 = sigmoidal_emax(8.0, 10.0, 5.0, hill_n=1.0, e0=0.0)
        e_n2 = sigmoidal_emax(8.0, 10.0, 5.0, hill_n=2.0, e0=0.0)
        assert e_n2 > e_n1

    def test_zero_conc_returns_baseline(self):
        result = sigmoidal_emax(0.0, 10.0, 5.0, hill_n=2.0, e0=3.0)
        assert result == pytest.approx(3.0)

    def test_invalid_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50 must be positive"):
            sigmoidal_emax(5.0, 10.0, ec50=0.0, hill_n=1.0)

    def test_invalid_hill_n_raises(self):
        with pytest.raises(ValueError, match="hill_n must be positive"):
            sigmoidal_emax(5.0, 10.0, ec50=5.0, hill_n=0.0)

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="conc must be"):
            sigmoidal_emax(-1.0, 10.0, ec50=5.0, hill_n=1.0)


# ---------------------------------------------------------------------------
# simulate_pkpd — basic output
# ---------------------------------------------------------------------------


class TestSimulatePKPD:
    def _base_result(self, model="direct", route="iv") -> PKPDResult:
        return simulate_pkpd(
            drug_name="DrugA",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            emax=10.0,
            ec50_mg_L=0.5,
            hill_n=1.0,
            e0=0.0,
            route=route,
            t_end_h=24.0,
            dt_h=0.1,
            model=model,
        )

    def test_returns_pkpdresult(self):
        r = self._base_result()
        assert isinstance(r, PKPDResult)

    def test_times_starts_at_zero(self):
        r = self._base_result()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_ends_at_t_end(self):
        r = self._base_result()
        assert r.times_h[-1] == pytest.approx(24.0)

    def test_direct_model_effect_mirrors_pk(self):
        r = self._base_result(model="direct")
        # For direct model, effect and PK should both decrease monotonically (IV)
        # Check effect is positive at t=0 and declines
        assert r.effect[0] > r.effect[-1]

    def test_ce_model_peak_effect_after_cmax_iv(self):
        """For effect compartment model, peak effect lags behind Cmax."""
        r_iv = self._base_result(model="ce", route="iv")
        # IV Cmax is at t=0; effect peak should be after t=0
        assert r_iv.time_to_max_effect_h > 0.0

    def test_direct_iv_peak_at_t0(self):
        """Direct IV: max effect at t=0 since concentration is highest at t=0."""
        r = self._base_result(model="direct", route="iv")
        assert r.time_to_max_effect_h == pytest.approx(0.0)

    def test_oral_route_peak_after_zero(self):
        r = self._base_result(model="direct", route="oral")
        assert r.time_to_max_effect_h > 0.0

    def test_effect_auc_positive(self):
        r = self._base_result()
        assert r.effect_auc > 0.0

    def test_time_above_ec50_positive_when_dose_exceeds_ec50(self):
        r = self._base_result()
        # Dose = 100 mg, Vd = 50 L → C0 = 2 mg/L >> EC50 = 0.5
        assert r.time_above_ec50_h > 0.0

    def test_time_above_ec50_zero_when_all_concs_below(self):
        r = simulate_pkpd(
            "Low",
            dose_mg=0.001,
            cl_L_per_h=50.0,
            vd_L=100.0,
            emax=10.0,
            ec50_mg_L=10.0,
            route="iv",
            t_end_h=24.0,
            dt_h=0.1,
            model="direct",
        )
        assert r.time_above_ec50_h == pytest.approx(0.0, abs=0.2)

    def test_double_dose_longer_time_above_ec50(self):
        r1 = simulate_pkpd(
            "D",
            100.0,
            5.0,
            50.0,
            10.0,
            0.5,
            route="iv",
            t_end_h=24.0,
            dt_h=0.1,
            model="direct",
        )
        r2 = simulate_pkpd(
            "D",
            200.0,
            5.0,
            50.0,
            10.0,
            0.5,
            route="iv",
            t_end_h=24.0,
            dt_h=0.1,
            model="direct",
        )
        assert r2.time_above_ec50_h > r1.time_above_ec50_h

    def test_baseline_effect_preserved(self):
        r = simulate_pkpd(
            "D",
            100.0,
            5.0,
            50.0,
            10.0,
            0.5,
            e0=2.0,
            route="iv",
            t_end_h=24.0,
            dt_h=0.1,
            model="direct",
        )
        assert r.baseline_effect == pytest.approx(2.0)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_pkpd("D", -1.0, 5.0, 50.0, 10.0, 0.5)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_pkpd("D", 100.0, 0.0, 50.0, 10.0, 0.5)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_pkpd("D", 100.0, 5.0, 0.0, 10.0, 0.5)

    def test_invalid_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50_mg_L"):
            simulate_pkpd("D", 100.0, 5.0, 50.0, 10.0, 0.0)

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_pkpd("D", 100.0, 5.0, 50.0, 10.0, 0.5, route="inhaled")

    def test_invalid_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            simulate_pkpd("D", 100.0, 5.0, 50.0, 10.0, 0.5, model="indirect")


# ---------------------------------------------------------------------------
# time_above_ec50
# ---------------------------------------------------------------------------


class TestTimeAboveEC50:
    def test_all_above_ec50_returns_full_duration(self):
        times = [0.0, 1.0, 2.0]
        concs = [10.0, 10.0, 10.0]
        result = time_above_ec50(concs, times, ec50_mg_L=1.0)
        assert result == pytest.approx(2.0)

    def test_all_below_ec50_returns_zero(self):
        times = [0.0, 1.0, 2.0]
        concs = [0.1, 0.1, 0.1]
        result = time_above_ec50(concs, times, ec50_mg_L=1.0)
        assert result == pytest.approx(0.0)

    def test_half_above_half_below(self):
        times = [0.0, 1.0, 2.0, 3.0, 4.0]
        concs = [2.0, 2.0, 0.5, 0.5, 0.5]  # above for first 2 h
        result = time_above_ec50(concs, times, ec50_mg_L=1.0)
        assert 0.5 < result < 2.5

    def test_invalid_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50_mg_L must be positive"):
            time_above_ec50([1.0, 2.0], [0.0, 1.0], ec50_mg_L=0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            time_above_ec50([1.0, 2.0, 3.0], [0.0, 1.0], ec50_mg_L=1.0)


# ---------------------------------------------------------------------------
# calculate_pdp_index
# ---------------------------------------------------------------------------


class TestCalculatePDPIndex:
    def test_flat_effect_at_baseline_returns_zero(self):
        times = [0.0, 1.0, 2.0, 3.0]
        effect = [1.0, 1.0, 1.0, 1.0]
        result = calculate_pdp_index(effect, times, baseline_effect=1.0)
        assert result == pytest.approx(0.0)

    def test_constant_effect_above_baseline(self):
        times = [0.0, 1.0, 2.0]
        effect = [5.0, 5.0, 5.0]  # 4 units above baseline=1
        result = calculate_pdp_index(effect, times, baseline_effect=1.0)
        assert result == pytest.approx(4.0)

    def test_effect_below_baseline_clamped(self):
        times = [0.0, 1.0, 2.0]
        effect = [0.0, 0.0, 0.0]
        result = calculate_pdp_index(effect, times, baseline_effect=1.0)
        assert result == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            calculate_pdp_index([1.0, 2.0], [0.0, 1.0, 2.0], 0.0)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="At least 2 time points"):
            calculate_pdp_index([1.0], [0.0], 0.0)

    def test_pdp_increases_with_larger_effect(self):
        times = [0.0, 1.0, 2.0]
        e_low = [2.0, 2.0, 2.0]
        e_high = [5.0, 5.0, 5.0]
        pdp_low = calculate_pdp_index(e_low, times, baseline_effect=0.0)
        pdp_high = calculate_pdp_index(e_high, times, baseline_effect=0.0)
        assert pdp_high > pdp_low
