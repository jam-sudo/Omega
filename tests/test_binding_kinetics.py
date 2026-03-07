"""Tests for core/binding_kinetics.py — Phase 195."""

from __future__ import annotations

import pytest

from omega_pbpk.core.binding_kinetics import (
    BindingKineticsResult,
    residence_time_analysis,
    simulate_binding_kinetics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KON = 1e6  # M^-1 s^-1  (typical fast binder)
KOFF_SLOW = 1e-4  # s^-1  → RT = 10 000 s
KOFF_FAST = 0.1  # s^-1  → RT = 10 s


def _default_result(**kwargs) -> BindingKineticsResult:
    defaults = dict(
        drug_name="TestDrug",
        kon_per_M_per_s=KON,
        koff_per_s=KOFF_SLOW,
        drug_conc_nM=100.0,
        target_conc_nM=10.0,
        t_end_s=100.0,
        dt_s=1.0,
    )
    defaults.update(kwargs)
    return simulate_binding_kinetics(**defaults)


# ---------------------------------------------------------------------------
# Basic return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = _default_result()
        assert isinstance(r, BindingKineticsResult)

    def test_drug_name_preserved(self):
        r = _default_result(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_times_and_occupancy_same_length(self):
        r = _default_result(t_end_s=50.0, dt_s=5.0)
        assert len(r.times_s) == len(r.occupancy_pct)

    def test_times_start_at_zero(self):
        r = _default_result()
        assert r.times_s[0] == 0.0

    def test_notes_is_string(self):
        r = _default_result()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Kd calculation
# ---------------------------------------------------------------------------


class TestKdCalculation:
    def test_kd_equals_koff_over_kon(self):
        r = simulate_binding_kinetics(
            drug_name="Drug",
            kon_per_M_per_s=1e6,
            koff_per_s=1e-3,
            drug_conc_nM=100.0,
            target_conc_nM=10.0,
            t_end_s=10.0,
        )
        expected_kd_nM = (1e-3 / 1e6) * 1e9  # = 1.0 nM
        assert abs(r.kd_nM - expected_kd_nM) < 1e-6

    def test_kd_units_nM(self):
        r = simulate_binding_kinetics(
            drug_name="X",
            kon_per_M_per_s=1e5,
            koff_per_s=0.5,
            drug_conc_nM=100.0,
            target_conc_nM=10.0,
            t_end_s=10.0,
        )
        # koff/kon * 1e9 = 0.5 / 1e5 * 1e9 = 5000 nM
        assert abs(r.kd_nM - 5000.0) < 1e-3


# ---------------------------------------------------------------------------
# Residence time
# ---------------------------------------------------------------------------


class TestResidenceTime:
    def test_high_koff_short_rt(self):
        r_fast = _default_result(koff_per_s=KOFF_FAST)
        r_slow = _default_result(koff_per_s=KOFF_SLOW)
        assert r_fast.residence_time_s < r_slow.residence_time_s

    def test_rt_equals_one_over_koff(self):
        r = _default_result(koff_per_s=0.01)
        assert abs(r.residence_time_s - 100.0) < 1e-9

    def test_t_half_off_lt_rt(self):
        r = _default_result()
        # t_half_off = 0.693/koff < 1/koff = RT
        assert r.t_half_off_s < r.residence_time_s

    def test_t_half_off_formula(self):
        koff = 0.02
        r = _default_result(koff_per_s=koff)
        expected = 0.693 / koff
        assert abs(r.t_half_off_s - expected) < 1e-9


# ---------------------------------------------------------------------------
# Equilibrium occupancy
# ---------------------------------------------------------------------------


class TestEquilibriumOccupancy:
    def test_drug_much_greater_kd_gives_high_occupancy(self):
        # drug_conc >> Kd → ~100% occupancy
        # Kd = koff/kon = 0.001/1e6 = 1e-9 M = 1 nM
        # drug_conc = 10 000 nM >> Kd → near 100%
        r = simulate_binding_kinetics(
            drug_name="X",
            kon_per_M_per_s=1e6,
            koff_per_s=0.001,
            drug_conc_nM=10_000.0,
            target_conc_nM=10.0,
            t_end_s=50_000.0,
            dt_s=10.0,
        )
        assert r.equilibrium_occupancy_pct > 99.0

    def test_drug_much_less_kd_gives_low_occupancy(self):
        # Kd = 0.1/1e4 = 10 uM = 10_000 nM; drug = 10 nM << Kd → <<1%
        r = simulate_binding_kinetics(
            drug_name="X",
            kon_per_M_per_s=1e4,
            koff_per_s=0.1,
            drug_conc_nM=10.0,
            target_conc_nM=10.0,
            t_end_s=1000.0,
            dt_s=1.0,
        )
        assert r.equilibrium_occupancy_pct < 1.0

    def test_drug_conc_zero_gives_zero_occupancy(self):
        r = _default_result(drug_conc_nM=0.0)
        assert r.equilibrium_occupancy_pct == 0.0
        assert all(occ == 0.0 for occ in r.occupancy_pct)

    def test_equilibrium_occupancy_between_0_and_100(self):
        r = _default_result()
        assert 0.0 <= r.equilibrium_occupancy_pct <= 100.0


# ---------------------------------------------------------------------------
# t_half_on
# ---------------------------------------------------------------------------


class TestTHalfOn:
    def test_t_half_on_formula(self):
        kon = 1e6
        koff = 0.01
        drug_nM = 100.0
        drug_M = drug_nM * 1e-9
        r = simulate_binding_kinetics("X", kon, koff, drug_nM, 10.0, t_end_s=10.0)
        k_obs = kon * drug_M + koff
        expected = 0.693 / k_obs
        assert abs(r.t_half_on_s - expected) < 1e-9

    def test_t_half_on_positive(self):
        r = _default_result()
        assert r.t_half_on_s > 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_kon_raises(self):
        with pytest.raises(ValueError, match="kon"):
            simulate_binding_kinetics(
                "X", kon_per_M_per_s=0.0, koff_per_s=0.01, drug_conc_nM=100.0, target_conc_nM=10.0
            )

    def test_negative_kon_raises(self):
        with pytest.raises(ValueError):
            simulate_binding_kinetics(
                "X", kon_per_M_per_s=-1e6, koff_per_s=0.01, drug_conc_nM=100.0, target_conc_nM=10.0
            )

    def test_invalid_koff_raises(self):
        with pytest.raises(ValueError, match="koff"):
            simulate_binding_kinetics(
                "X", kon_per_M_per_s=1e6, koff_per_s=0.0, drug_conc_nM=100.0, target_conc_nM=10.0
            )

    def test_negative_drug_conc_raises(self):
        with pytest.raises(ValueError):
            simulate_binding_kinetics(
                "X", kon_per_M_per_s=1e6, koff_per_s=0.01, drug_conc_nM=-1.0, target_conc_nM=10.0
            )

    def test_invalid_target_conc_raises(self):
        with pytest.raises(ValueError):
            simulate_binding_kinetics(
                "X", kon_per_M_per_s=1e6, koff_per_s=0.01, drug_conc_nM=100.0, target_conc_nM=0.0
            )


# ---------------------------------------------------------------------------
# residence_time_analysis
# ---------------------------------------------------------------------------


class TestResidenceTimeAnalysis:
    def test_returns_list_of_results(self):
        koffs = [0.001, 0.01, 0.1]
        results = residence_time_analysis("Drug", KON, koffs)
        assert len(results) == 3
        assert all(isinstance(r, BindingKineticsResult) for r in results)

    def test_sorted_descending_by_rt(self):
        koffs = [0.1, 0.001, 0.01]
        results = residence_time_analysis("Drug", KON, koffs)
        rts = [r.residence_time_s for r in results]
        assert rts == sorted(rts, reverse=True)

    def test_single_koff_returns_single_result(self):
        results = residence_time_analysis("Drug", KON, [0.05])
        assert len(results) == 1

    def test_first_result_has_longest_rt(self):
        koffs = [0.001, 0.01, 1.0]
        results = residence_time_analysis("Drug", KON, koffs)
        assert results[0].residence_time_s == pytest.approx(1000.0, rel=1e-3)
