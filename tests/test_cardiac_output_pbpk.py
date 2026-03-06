"""Tests for cardiac output-dependent PBPK (Phase 202)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.cardiac_output_pbpk import (
    CARDIAC_STATES,
    CardiacOutputPKResult,
    compare_cardiac_states,
    simulate_cardiac_output_pk,
)


class TestSimulateCardiacOutputPK:
    def _base_kwargs(self):
        return dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_nominal_L_per_h=5.0,
            vd_L=50.0,
        )

    def test_returns_result_type(self):
        r = simulate_cardiac_output_pk(**self._base_kwargs())
        assert isinstance(r, CardiacOutputPKResult)

    def test_normal_state_default(self):
        r = simulate_cardiac_output_pk(**self._base_kwargs())
        assert r.cardiac_state == "normal"

    def test_heart_failure_higher_auc_than_normal(self):
        """Heart failure reduces CL → higher AUC."""
        r_normal = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="normal")
        r_hf = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="heart_failure")
        assert r_hf.auc_plasma > r_normal.auc_plasma

    def test_exercise_lower_auc_than_normal(self):
        """Exercise increases CL → lower AUC."""
        r_normal = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="normal")
        r_ex = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="exercise")
        assert r_ex.auc_plasma < r_normal.auc_plasma

    def test_sepsis_altered_vd(self):
        """Sepsis has vd_factor=1.3 → different Vd from normal."""
        r_normal = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="normal")
        r_sepsis = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="sepsis")
        # Sepsis: cl_factor=0.7, vd_factor=1.3 → ke changes
        assert r_sepsis.cardiac_output_L_per_min == CARDIAC_STATES["sepsis"]["co_L_per_min"]
        # Different AUC due to different cl_factor
        assert r_sepsis.auc_plasma != r_normal.auc_plasma

    def test_cl_effective_heart_failure(self):
        """Heart failure CL should be 0.6 * nominal."""
        r = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="heart_failure")
        expected_cl = 5.0 * 0.6
        assert abs(r.cl_effective_L_per_h - expected_cl) < 1e-6

    def test_cl_effective_exercise(self):
        """Exercise CL should be 1.3 * nominal."""
        r = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="exercise")
        expected_cl = 5.0 * 1.3
        assert abs(r.cl_effective_L_per_h - expected_cl) < 1e-6

    def test_cardiac_output_values(self):
        """Verify CO values match CARDIAC_STATES dict."""
        for state, params in CARDIAC_STATES.items():
            r = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state=state)
            assert r.cardiac_output_L_per_min == params["co_L_per_min"]

    def test_c_liver_greater_than_c_plasma(self):
        """kp_liver=3.0 and normal CO → liver conc should exceed plasma."""
        r = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="normal")
        # At Cmax time, liver > plasma since kp_liver*co_ratio = 3.0*1.0 > 1
        max_plasma = max(r.c_plasma_mg_L)
        max_liver = max(r.c_liver_mg_L)
        assert max_liver > max_plasma

    def test_c_kidney_greater_than_c_plasma(self):
        """kp_kidney=2.0 at normal CO → kidney > plasma."""
        r = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="normal")
        assert max(r.c_kidney_mg_L) > max(r.c_plasma_mg_L)

    def test_c_muscle_less_than_c_plasma_normal(self):
        """At normal CO, muscle conc < plasma (kp_muscle=0.5 * co_ratio_inv = 0.5*1 = 0.5)."""
        r = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="normal")
        assert max(r.c_muscle_mg_L) < max(r.c_plasma_mg_L)

    def test_times_length_matches_concentrations(self):
        r = simulate_cardiac_output_pk(**self._base_kwargs())
        assert len(r.times_h) == len(r.c_plasma_mg_L)
        assert len(r.times_h) == len(r.c_liver_mg_L)
        assert len(r.times_h) == len(r.c_kidney_mg_L)
        assert len(r.times_h) == len(r.c_muscle_mg_L)

    def test_plasma_conc_non_negative(self):
        r = simulate_cardiac_output_pk(**self._base_kwargs())
        assert all(c >= 0.0 for c in r.c_plasma_mg_L)

    def test_cmax_plasma_positive(self):
        r = simulate_cardiac_output_pk(**self._base_kwargs())
        assert r.cmax_plasma > 0.0

    def test_auc_plasma_positive(self):
        r = simulate_cardiac_output_pk(**self._base_kwargs())
        assert r.auc_plasma > 0.0

    def test_oral_route(self):
        """Oral route should produce delayed absorption peak."""
        r_iv = simulate_cardiac_output_pk(**self._base_kwargs(), route="iv")
        r_oral = simulate_cardiac_output_pk(
            **self._base_kwargs(), route="oral", ka_per_h=1.0, f_oral=1.0
        )
        # IV starts with max conc at t=0; oral should have lower or equal Cmax
        assert r_oral.cmax_plasma <= r_iv.cmax_plasma * 1.05  # Allow small tolerance

    def test_invalid_cardiac_state_raises(self):
        with pytest.raises(ValueError, match="cardiac_state"):
            simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="coma")

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_cardiac_output_pk(
                drug_name="X", dose_mg=0.0, cl_nominal_L_per_h=5.0, vd_L=50.0
            )

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_nominal"):
            simulate_cardiac_output_pk(
                drug_name="X", dose_mg=100.0, cl_nominal_L_per_h=-1.0, vd_L=50.0
            )

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_cardiac_output_pk(
                drug_name="X", dose_mg=100.0, cl_nominal_L_per_h=5.0, vd_L=0.0
            )

    def test_redistribution_index_normal_zero(self):
        """Normal state: cl_factor=1.0, vd_factor=1.0 → index = 0."""
        r = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="normal")
        assert r.redistribution_index == 0.0

    def test_redistribution_index_heart_failure_positive(self):
        r = simulate_cardiac_output_pk(**self._base_kwargs(), cardiac_state="heart_failure")
        assert r.redistribution_index > 0.0


class TestCompareCardiacStates:
    def _base_kwargs(self):
        return dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_nominal_L_per_h=5.0,
            vd_L=50.0,
        )

    def test_returns_four_results(self):
        results = compare_cardiac_states(**self._base_kwargs())
        assert len(results) == 4

    def test_all_states_present(self):
        results = compare_cardiac_states(**self._base_kwargs())
        states = {r.cardiac_state for r in results}
        assert states == set(CARDIAC_STATES.keys())

    def test_sorted_by_auc_descending(self):
        results = compare_cardiac_states(**self._base_kwargs())
        aucs = [r.auc_plasma for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_heart_failure_highest_auc(self):
        """Heart failure (lowest CL) should have highest AUC."""
        results = compare_cardiac_states(**self._base_kwargs())
        assert results[0].cardiac_state == "heart_failure"

    def test_exercise_lower_auc_than_normal(self):
        results = compare_cardiac_states(**self._base_kwargs())
        auc_by_state = {r.cardiac_state: r.auc_plasma for r in results}
        assert auc_by_state["exercise"] < auc_by_state["normal"]

    def test_all_results_correct_type(self):
        results = compare_cardiac_states(**self._base_kwargs())
        assert all(isinstance(r, CardiacOutputPKResult) for r in results)
