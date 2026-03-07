"""Tests for receptor binding kinetics simulation — Phase 467."""

import math
import pytest

from omega_pbpk.core.receptor_binding import (
    ReceptorBindingResult,
    calculate_kd,
    occupancy_at_steady_state,
    residence_time,
    simulate_receptor_binding,
    simulate_with_pk,
)


# ---------------------------------------------------------------------------
# Unit tests: calculate_kd
# ---------------------------------------------------------------------------

class TestCalculateKd:
    def test_basic_kd(self):
        kd = calculate_kd(kon_per_nM_per_h=0.1, koff_per_h=1.0)
        assert abs(kd - 10.0) < 1e-9

    def test_kd_units(self):
        # Kd = koff / kon => nM
        kd = calculate_kd(kon_per_nM_per_h=2.0, koff_per_h=4.0)
        assert abs(kd - 2.0) < 1e-9

    def test_kon_zero_raises(self):
        with pytest.raises(ValueError, match="kon"):
            calculate_kd(kon_per_nM_per_h=0.0, koff_per_h=1.0)

    def test_koff_zero_gives_zero_kd(self):
        kd = calculate_kd(kon_per_nM_per_h=0.5, koff_per_h=0.0)
        assert kd == 0.0

    def test_kon_negative_raises(self):
        with pytest.raises(ValueError):
            calculate_kd(kon_per_nM_per_h=-0.1, koff_per_h=1.0)

    def test_koff_negative_raises(self):
        with pytest.raises(ValueError, match="koff"):
            calculate_kd(kon_per_nM_per_h=0.1, koff_per_h=-0.5)


# ---------------------------------------------------------------------------
# Unit tests: residence_time
# ---------------------------------------------------------------------------

class TestResidenceTime:
    def test_basic(self):
        mrt = residence_time(koff_per_h=2.0)
        assert abs(mrt - 0.5) < 1e-9

    def test_slow_dissociation(self):
        mrt = residence_time(koff_per_h=0.01)
        assert abs(mrt - 100.0) < 1e-9

    def test_zero_koff_raises(self):
        with pytest.raises(ValueError):
            residence_time(koff_per_h=0.0)

    def test_negative_koff_raises(self):
        with pytest.raises(ValueError):
            residence_time(koff_per_h=-1.0)

    def test_higher_koff_shorter_mrt(self):
        mrt_fast = residence_time(koff_per_h=10.0)
        mrt_slow = residence_time(koff_per_h=1.0)
        assert mrt_fast < mrt_slow


# ---------------------------------------------------------------------------
# Unit tests: occupancy_at_steady_state
# ---------------------------------------------------------------------------

class TestOccupancyAtSteadyState:
    def test_at_kd_concentration(self):
        # At [D] = Kd → occupancy = 0.5
        occ = occupancy_at_steady_state(drug_conc_nM=10.0, kd_nM=10.0)
        assert abs(occ - 0.5) < 1e-9

    def test_high_conc_approaches_1(self):
        occ = occupancy_at_steady_state(drug_conc_nM=10000.0, kd_nM=1.0)
        assert occ > 0.99

    def test_zero_drug_zero_occupancy(self):
        occ = occupancy_at_steady_state(drug_conc_nM=0.0, kd_nM=5.0)
        assert occ == 0.0

    def test_double_conc_approx_double_at_low(self):
        # At low [D] << Kd: occupancy ≈ [D]/Kd, linear regime
        occ1 = occupancy_at_steady_state(drug_conc_nM=0.1, kd_nM=100.0)
        occ2 = occupancy_at_steady_state(drug_conc_nM=0.2, kd_nM=100.0)
        assert abs(occ2 / occ1 - 2.0) < 0.05

    def test_langmuir_formula(self):
        D, Kd = 3.0, 7.0
        expected = D / (Kd + D)
        assert abs(occupancy_at_steady_state(D, Kd) - expected) < 1e-9

    def test_negative_drug_raises(self):
        with pytest.raises(ValueError):
            occupancy_at_steady_state(drug_conc_nM=-1.0, kd_nM=10.0)

    def test_zero_kd_raises(self):
        with pytest.raises(ValueError):
            occupancy_at_steady_state(drug_conc_nM=5.0, kd_nM=0.0)


# ---------------------------------------------------------------------------
# Integration tests: simulate_receptor_binding
# ---------------------------------------------------------------------------

class TestSimulateReceptorBinding:
    def test_returns_result_type(self):
        result = simulate_receptor_binding("DrugA", 100.0, 0.1, 0.5)
        assert isinstance(result, ReceptorBindingResult)

    def test_kd_correct(self):
        result = simulate_receptor_binding("DrugA", 100.0, 0.1, 0.5)
        assert abs(result.kd_nM - 5.0) < 1e-9

    def test_residence_time_correct(self):
        result = simulate_receptor_binding("DrugA", 100.0, 0.1, 0.5)
        assert abs(result.residence_time_h - 2.0) < 1e-6

    def test_equilibrium_occupancy_matches_langmuir(self):
        # Run long enough to reach equilibrium
        dose_nM = 100.0
        kon = 0.01
        koff = 0.1
        kd = koff / kon  # 10 nM
        result = simulate_receptor_binding(
            "DrugB", dose_nM, kon, koff, receptor_conc_nM=1.0, t_end_h=200.0, dt_h=0.05
        )
        final_occ = result.occupancy_pct[-1] / 100.0
        # Langmuir equilibrium: [D]_eq ≈ dose_nM (receptor << drug)
        expected_occ = dose_nM / (kd + dose_nM)
        assert abs(final_occ - expected_occ) < 0.05

    def test_higher_kon_faster_association(self):
        r_fast = simulate_receptor_binding(
            "DrugFast", 10.0, 0.5, 1.0, receptor_conc_nM=10.0, t_end_h=5.0, dt_h=0.01
        )
        r_slow = simulate_receptor_binding(
            "DrugSlow", 10.0, 0.05, 1.0, receptor_conc_nM=10.0, t_end_h=5.0, dt_h=0.01
        )
        # Fast should reach 50% occupancy sooner
        assert r_fast.time_to_50pct_occupancy_h < r_slow.time_to_50pct_occupancy_h or (
            r_slow.time_to_50pct_occupancy_h == float("inf")
        )

    def test_higher_koff_lower_equilibrium_occupancy(self):
        dose = 10.0
        kon = 0.1
        r_tight = simulate_receptor_binding(
            "Tight", dose, kon, 0.1, receptor_conc_nM=1.0, t_end_h=200.0, dt_h=0.1
        )
        r_weak = simulate_receptor_binding(
            "Weak", dose, kon, 5.0, receptor_conc_nM=1.0, t_end_h=200.0, dt_h=0.1
        )
        assert r_tight.max_occupancy_pct > r_weak.max_occupancy_pct

    def test_mass_conservation(self):
        R0 = 10.0
        result = simulate_receptor_binding(
            "DrugC", 1000.0, 0.01, 0.1, receptor_conc_nM=R0, t_end_h=10.0, dt_h=0.01
        )
        # At each time point: R_free + DR ≈ R_total
        for rf, dr in zip(result.receptor_free_nM, result.complex_nM):
            total = rf + dr
            assert abs(total - R0) < 0.5, f"Mass balance violated: R_free+DR={total:.3f} != {R0}"

    def test_occupancy_pct_range(self):
        result = simulate_receptor_binding("DrugD", 5.0, 0.05, 0.5)
        for occ in result.occupancy_pct:
            assert 0.0 <= occ <= 100.0

    def test_times_monotone(self):
        result = simulate_receptor_binding("DrugE", 50.0, 0.1, 0.5)
        for i in range(1, len(result.times_h)):
            assert result.times_h[i] >= result.times_h[i - 1]

    def test_drug_name_preserved(self):
        result = simulate_receptor_binding("Imatinib", 50.0, 0.1, 0.5)
        assert result.drug_name == "Imatinib"

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_receptor_binding("DrugX", -1.0, 0.1, 0.5)

    def test_invalid_receptor_conc_raises(self):
        with pytest.raises(ValueError):
            simulate_receptor_binding("DrugX", 10.0, 0.1, 0.5, receptor_conc_nM=0.0)

    def test_max_occupancy_equals_max_of_list(self):
        result = simulate_receptor_binding("DrugF", 200.0, 0.2, 0.3)
        assert abs(result.max_occupancy_pct - max(result.occupancy_pct)) < 1e-9


# ---------------------------------------------------------------------------
# Integration tests: simulate_with_pk
# ---------------------------------------------------------------------------

class TestSimulateWithPk:
    def test_returns_result_type(self):
        result = simulate_with_pk(
            "DrugPK", 100.0, cl_L_per_h=5.0, vd_L=50.0,
            kon_per_nM_per_h=0.1, koff_per_h=0.5
        )
        assert isinstance(result, ReceptorBindingResult)

    def test_drug_conc_decreases_due_to_elimination(self):
        result = simulate_with_pk(
            "DrugPK", 100.0, cl_L_per_h=10.0, vd_L=20.0,
            kon_per_nM_per_h=0.001, koff_per_h=0.01, t_end_h=10.0, dt_h=0.01
        )
        # With rapid elimination, drug conc should decrease
        assert result.drug_free_nM[-1] < result.drug_free_nM[0]

    def test_fast_elimination_reduces_occupancy(self):
        """High CL → drug cleared quickly → lower eventual occupancy."""
        r_fast_cl = simulate_with_pk(
            "DrugFastCL", 200.0, cl_L_per_h=20.0, vd_L=10.0,
            kon_per_nM_per_h=0.05, koff_per_h=0.2, t_end_h=48.0, dt_h=0.01
        )
        r_slow_cl = simulate_with_pk(
            "DrugSlowCL", 200.0, cl_L_per_h=0.5, vd_L=10.0,
            kon_per_nM_per_h=0.05, koff_per_h=0.2, t_end_h=48.0, dt_h=0.01
        )
        # Slow CL → drug persists → higher sustained occupancy → higher final complex
        assert r_slow_cl.complex_nM[-1] >= r_fast_cl.complex_nM[-1]

    def test_kd_correct_in_pk_simulation(self):
        result = simulate_with_pk(
            "DrugKd", 50.0, cl_L_per_h=1.0, vd_L=10.0,
            kon_per_nM_per_h=0.2, koff_per_h=2.0
        )
        assert abs(result.kd_nM - 10.0) < 1e-9

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_with_pk(
                "DrugX", 0.0, cl_L_per_h=1.0, vd_L=10.0,
                kon_per_nM_per_h=0.1, koff_per_h=0.5
            )

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError):
            simulate_with_pk(
                "DrugX", 10.0, cl_L_per_h=1.0, vd_L=0.0,
                kon_per_nM_per_h=0.1, koff_per_h=0.5
            )

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            simulate_with_pk(
                "DrugX", 10.0, cl_L_per_h=1.0, vd_L=10.0,
                kon_per_nM_per_h=0.1, koff_per_h=0.5, mw_g_mol=0.0
            )

    def test_occupancy_range(self):
        result = simulate_with_pk(
            "DrugRange", 100.0, cl_L_per_h=2.0, vd_L=20.0,
            kon_per_nM_per_h=0.1, koff_per_h=1.0, t_end_h=24.0
        )
        for occ in result.occupancy_pct:
            assert 0.0 <= occ <= 100.0

    def test_notes_not_empty(self):
        result = simulate_with_pk(
            "DrugNotes", 50.0, cl_L_per_h=3.0, vd_L=30.0,
            kon_per_nM_per_h=0.05, koff_per_h=0.5
        )
        assert len(result.notes) > 0
