"""Tests for Phase 613: Monoclonal Antibody Dosing simulation."""

import math

import pytest

from omega_pbpk.clinical.mab_dosing_p613 import MABDosingResult, simulate_mab_dosing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_iv(**kwargs):
    params = dict(
        drug_name="TestMAB",
        dose_mg=600.0,
        route="iv",
        dosing_interval_weeks=4.0,
        kd_nM=1.0,
        target_conc_nM=10.0,
        cl_L_per_day=0.24,
        vd_central_L=6.0,
        n_doses=4,
    )
    params.update(kwargs)
    return simulate_mab_dosing(**params)


def _default_sc(**kwargs):
    params = dict(
        drug_name="TestMAB_SC",
        dose_mg=600.0,
        route="sc",
        dosing_interval_weeks=4.0,
        kd_nM=1.0,
        target_conc_nM=10.0,
        cl_L_per_day=0.24,
        vd_central_L=6.0,
        n_doses=4,
    )
    params.update(kwargs)
    return simulate_mab_dosing(**params)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_mab_dosing_result(self):
        result = _default_iv()
        assert isinstance(result, MABDosingResult)

    def test_result_has_list_fields(self):
        result = _default_iv()
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_free_mab_ug_mL, list)
        assert isinstance(result.c_target_bound_ug_mL, list)
        assert isinstance(result.c_free_target_nM, list)
        assert isinstance(result.receptor_occupancy, list)

    def test_list_lengths_equal(self):
        result = _default_iv()
        n = len(result.times_h)
        assert len(result.c_free_mab_ug_mL) == n
        assert len(result.c_target_bound_ug_mL) == n
        assert len(result.c_free_target_nM) == n
        assert len(result.receptor_occupancy) == n

    def test_scalar_fields_are_float(self):
        result = _default_iv()
        assert isinstance(result.cmax_ug_mL, float)
        assert isinstance(result.ctrough_ug_mL, float)
        assert isinstance(result.auc_ug_h_per_mL, float)
        assert isinstance(result.target_coverage_pct, float)
        assert isinstance(result.accumulation_ratio, float)

    def test_notes_is_str(self):
        result = _default_iv()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_drug_name_preserved(self):
        result = _default_iv()
        assert result.drug_name == "TestMAB"

    def test_route_preserved(self):
        result_iv = _default_iv()
        result_sc = _default_sc()
        assert result_iv.route == "iv"
        assert result_sc.route == "sc"


# ---------------------------------------------------------------------------
# Time array
# ---------------------------------------------------------------------------


class TestTimeArray:
    def test_time_starts_at_zero(self):
        result = _default_iv()
        assert result.times_h[0] == 0.0

    def test_time_monotonically_increasing(self):
        result = _default_iv()
        for i in range(1, len(result.times_h)):
            assert result.times_h[i] > result.times_h[i - 1]

    def test_time_covers_expected_duration(self):
        result = _default_iv(n_doses=2, dosing_interval_weeks=2.0)
        tau_h = 2.0 * 7.0 * 24.0
        t_end = 2 * tau_h + 168.0
        assert result.times_h[-1] >= t_end - 1.0


# ---------------------------------------------------------------------------
# IV vs SC differences
# ---------------------------------------------------------------------------


class TestIVvsSC:
    def test_iv_has_instant_peak(self):
        """IV bolus: concentration should be non-zero immediately at t=0."""
        # Use n_doses=1 to avoid accumulation pushing global max to later dose
        result = simulate_mab_dosing(
            drug_name="TestMAB",
            dose_mg=600.0,
            route="iv",
            dosing_interval_weeks=4.0,
            kd_nM=1.0,
            target_conc_nM=10.0,
            cl_L_per_day=0.24,
            vd_central_L=6.0,
            n_doses=1,
        )
        # First time point after t=0 should have positive concentration
        assert result.c_free_mab_ug_mL[0] > 0.0
        # And the max should be within the first 2 hours
        idx_max = result.c_free_mab_ug_mL.index(max(result.c_free_mab_ug_mL))
        assert result.times_h[idx_max] < 2.0

    def test_sc_peak_delayed_vs_iv(self):
        """SC absorption delays Cmax vs IV."""
        iv = _default_iv()
        sc = _default_sc()
        # Find time of first-dose peak
        # IV peak within first dosing interval
        tau_h = 4.0 * 7.0 * 24.0
        iv_first = [
            (iv.times_h[i], iv.c_free_mab_ug_mL[i])
            for i in range(len(iv.times_h))
            if iv.times_h[i] < tau_h
        ]
        sc_first = [
            (sc.times_h[i], sc.c_free_mab_ug_mL[i])
            for i in range(len(sc.times_h))
            if sc.times_h[i] < tau_h
        ]
        t_iv_peak = max(iv_first, key=lambda x: x[1])[0]
        t_sc_peak = max(sc_first, key=lambda x: x[1])[0]
        assert t_sc_peak > t_iv_peak

    def test_sc_cmax_less_than_iv(self):
        """SC bioavailability < 1 means lower Cmax for same dose."""
        iv = _default_iv()
        sc = _default_sc()
        assert sc.cmax_ug_mL < iv.cmax_ug_mL


# ---------------------------------------------------------------------------
# Concentration values
# ---------------------------------------------------------------------------


class TestConcentrations:
    def test_c_free_non_negative(self):
        result = _default_iv()
        assert all(c >= 0.0 for c in result.c_free_mab_ug_mL)

    def test_cmax_equals_max_c_free(self):
        result = _default_iv()
        assert math.isclose(result.cmax_ug_mL, max(result.c_free_mab_ug_mL), rel_tol=1e-6)

    def test_auc_positive(self):
        result = _default_iv()
        assert result.auc_ug_h_per_mL > 0.0

    def test_ctrough_le_cmax(self):
        result = _default_iv()
        assert result.ctrough_ug_mL <= result.cmax_ug_mL


# ---------------------------------------------------------------------------
# Receptor occupancy
# ---------------------------------------------------------------------------


class TestReceptorOccupancy:
    def test_ro_between_0_and_1(self):
        result = _default_iv()
        assert all(0.0 <= r <= 1.0 for r in result.receptor_occupancy)

    def test_ro_increases_with_dose(self):
        """Higher dose should give higher peak RO."""
        low = _default_iv(dose_mg=10.0)
        high = _default_iv(dose_mg=1000.0)
        assert max(high.receptor_occupancy) > max(low.receptor_occupancy)

    def test_high_kd_lower_occupancy(self):
        """Higher Kd (weaker binding) → lower RO at same dose."""
        tight = _default_iv(kd_nM=0.1)
        weak = _default_iv(kd_nM=100.0)
        assert max(tight.receptor_occupancy) > max(weak.receptor_occupancy)

    def test_target_coverage_pct_in_range(self):
        result = _default_iv()
        assert 0.0 <= result.target_coverage_pct <= 100.0


# ---------------------------------------------------------------------------
# Free target concentration
# ---------------------------------------------------------------------------


class TestFreeTarget:
    def test_free_target_in_nM(self):
        result = _default_iv()
        # At t=0 before dose applied, free target starts near total target
        # After dosing, should be suppressed
        assert all(ft >= 0.0 for ft in result.c_free_target_nM)

    def test_free_target_sum_with_bound_approx_total(self):
        """Free target + bound target (in nM) should not exceed total target."""
        result = _default_iv(target_conc_nM=10.0, mw_kDa=150.0)
        for i in range(len(result.times_h)):
            # bound in nM: c_target_bound_ug_mL * 1000 / mw_kDa
            bound_nM = result.c_target_bound_ug_mL[i] * 1000.0 / 150.0
            free_nM = result.c_free_target_nM[i]
            assert free_nM + bound_nM <= 10.0 + 1e-9


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


class TestAccumulation:
    def test_accumulation_ratio_ge_1(self):
        """Multiple dosing should lead to accumulation."""
        result = _default_iv(n_doses=4)
        assert result.accumulation_ratio >= 1.0

    def test_accumulation_increases_with_doses(self):
        """More doses → higher accumulation."""
        r2 = _default_iv(n_doses=2)
        r6 = _default_iv(n_doses=6)
        assert r6.accumulation_ratio >= r2.accumulation_ratio


# ---------------------------------------------------------------------------
# Single dose
# ---------------------------------------------------------------------------


class TestSingleDose:
    def test_single_dose_works(self):
        result = simulate_mab_dosing(
            drug_name="SingleMAB",
            dose_mg=500.0,
            route="iv",
            dosing_interval_weeks=4.0,
            kd_nM=2.0,
            target_conc_nM=5.0,
            cl_L_per_day=0.3,
            vd_central_L=5.0,
            n_doses=1,
        )
        assert isinstance(result, MABDosingResult)
        assert result.cmax_ug_mL > 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            _default_iv(route="im")

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_iv(dose_mg=-100.0)

    def test_zero_interval(self):
        with pytest.raises(ValueError, match="dosing_interval_weeks"):
            _default_iv(dosing_interval_weeks=0.0)

    def test_zero_doses(self):
        with pytest.raises(ValueError, match="n_doses"):
            _default_iv(n_doses=0)

    def test_negative_kd(self):
        with pytest.raises(ValueError, match="kd_nM"):
            _default_iv(kd_nM=-1.0)

    def test_zero_target(self):
        with pytest.raises(ValueError, match="target_conc_nM"):
            _default_iv(target_conc_nM=0.0)
