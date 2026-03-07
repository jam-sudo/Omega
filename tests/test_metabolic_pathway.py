"""Tests for metabolic pathway simulation — Phase 461."""

from __future__ import annotations

import pytest

from omega_pbpk.core.metabolic_pathway import (
    MetabolicPathwayResult,
    calculate_metabolite_exposure_ratio,
    predict_active_metabolite_contribution,
    simulate_sequential_metabolism,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_steps(n_metabolites: int = 1) -> list[dict]:
    """Return steps for parent + n_metabolites (all CL=1, Vd=10, fm=1)."""
    steps = [{"name": "parent_pk", "cl_L_per_h": 1.0, "vd_L": 10.0, "fm": 1.0}]
    for i in range(n_metabolites):
        steps.append({"name": f"M{i + 1}", "cl_L_per_h": 2.0, "vd_L": 10.0, "fm": 1.0})
    return steps


# ---------------------------------------------------------------------------
# simulate_sequential_metabolism
# ---------------------------------------------------------------------------


class TestSimulateSequentialMetabolism:
    def test_returns_result_instance(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(1))
        assert isinstance(res, MetabolicPathwayResult)

    def test_n_steps_matches_metabolite_count(self):
        for n in [1, 2, 3]:
            res = simulate_sequential_metabolism(100.0, _simple_steps(n))
            assert res.n_steps == n

    def test_step_names_includes_parent(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(1))
        assert res.step_names[0] == "parent"

    def test_step_names_length(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(2))
        # parent + 2 metabolites = 3 total
        assert len(res.step_names) == 3

    def test_metabolite_names_in_step_names(self):
        steps = _simple_steps(2)
        res = simulate_sequential_metabolism(100.0, steps)
        assert "M1" in res.step_names
        assert "M2" in res.step_names

    def test_times_start_at_zero(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps())
        assert res.times_h[0] == pytest.approx(0.0)

    def test_times_end_near_t_end(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(), t_end_h=12.0)
        assert res.times_h[-1] == pytest.approx(12.0, abs=0.15)

    def test_concentrations_count_equals_species(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(2))
        assert len(res.concentrations) == 3  # parent + 2 metabolites

    def test_parent_starts_at_c0(self):
        dose = 100.0
        vd = 10.0
        steps = [
            {"name": "pk", "cl_L_per_h": 1.0, "vd_L": vd, "fm": 1.0},
            {"name": "M1", "cl_L_per_h": 2.0, "vd_L": 10.0, "fm": 1.0},
        ]
        res = simulate_sequential_metabolism(dose, steps)
        assert res.concentrations[0][0] == pytest.approx(dose / vd, rel=1e-6)

    def test_metabolite_starts_at_zero(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(1))
        assert res.concentrations[1][0] == pytest.approx(0.0, abs=1e-12)

    def test_parent_concentration_declines(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(1), dt_h=0.05)
        parent = res.concentrations[0]
        assert parent[-1] < parent[0]

    def test_auc_values_length(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(2))
        # parent + 2 metabolites = 3 AUC values
        assert len(res.auc_values) == 3

    def test_parent_auc_positive(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps())
        assert res.auc_values[0] > 0

    def test_metabolite_auc_positive(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps())
        assert res.auc_values[1] > 0

    def test_metabolite_exposure_ratios_length(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(2))
        assert len(res.metabolite_exposure_ratios) == 2

    def test_metabolite_exposure_ratio_matches_calculation(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps(), t_end_h=48.0, dt_h=0.05)
        expected = res.auc_values[1] / res.auc_values[0]
        assert res.metabolite_exposure_ratios[0] == pytest.approx(expected, rel=1e-6)

    def test_mass_balance_fm_1(self):
        """With fm=1 and same Vd, mass balance: CL_parent * AUC_parent ≈ CL_M1 * AUC_M1 * (Vd_M1/Vd_parent)."""
        # Dose conservation: parent amount metabolised = formation of M1
        # AUC_parent * ke_parent * fm * Vd_parent ≈ AUC_M1 * ke_M1 * Vd_M1
        # Simplified: AUC_parent * CL_parent * fm ≈ AUC_M1 * CL_M1
        steps = [
            {"name": "pk", "cl_L_per_h": 1.0, "vd_L": 10.0, "fm": 1.0},
            {"name": "M1", "cl_L_per_h": 2.0, "vd_L": 10.0, "fm": 0.0},
        ]
        res = simulate_sequential_metabolism(100.0, steps, t_end_h=72.0, dt_h=0.05)
        # Amount metabolised from parent = AUC_parent * CL_parent * fm = AUC_parent * 1.0
        # Amount entering M1 = AUC_M1 * CL_M1 = AUC_M1 * 2.0
        # So AUC_parent * 1 ≈ AUC_M1 * 2 (within numeric integration tolerance)
        lhs = res.auc_values[0] * 1.0
        rhs = res.auc_values[1] * 2.0
        assert lhs == pytest.approx(rhs, rel=0.05)

    def test_metabolite_cmax_after_parent_cmax(self):
        """Flip-flop: metabolite Cmax should occur after parent Cmax."""
        steps = [
            {"name": "pk", "cl_L_per_h": 2.0, "vd_L": 10.0, "fm": 1.0},
            {"name": "M1", "cl_L_per_h": 0.5, "vd_L": 10.0, "fm": 0.0},
        ]
        res = simulate_sequential_metabolism(100.0, steps, t_end_h=24.0, dt_h=0.05)
        parent = res.concentrations[0]
        m1 = res.concentrations[1]
        t = res.times_h
        parent_tmax = t[parent.index(max(parent))]
        m1_tmax = t[m1.index(max(m1))]
        assert m1_tmax >= parent_tmax

    def test_parent_only_step_raises(self):
        """Only one step means no metabolites — raise if steps is empty."""
        with pytest.raises(ValueError, match="steps"):
            simulate_sequential_metabolism(100.0, [])

    def test_t_end_zero_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_sequential_metabolism(100.0, _simple_steps(), t_end_h=0.0)

    def test_negative_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_sequential_metabolism(100.0, _simple_steps(), t_end_h=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="parent_dose_mg"):
            simulate_sequential_metabolism(0.0, _simple_steps())

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="parent_dose_mg"):
            simulate_sequential_metabolism(-50.0, _simple_steps())

    def test_notes_is_string(self):
        res = simulate_sequential_metabolism(100.0, _simple_steps())
        assert isinstance(res.notes, str)
        assert len(res.notes) > 0

    def test_two_step_pathway(self):
        """Parent → M1 → M2 chain produces 3 concentration profiles."""
        steps = [
            {"name": "pk", "cl_L_per_h": 1.0, "vd_L": 10.0, "fm": 1.0},
            {"name": "M1", "cl_L_per_h": 2.0, "vd_L": 10.0, "fm": 0.8},
            {"name": "M2", "cl_L_per_h": 3.0, "vd_L": 10.0, "fm": 0.0},
        ]
        res = simulate_sequential_metabolism(100.0, steps, t_end_h=48.0, dt_h=0.1)
        assert res.n_steps == 2
        assert len(res.concentrations) == 3
        assert res.concentrations[2][0] == pytest.approx(0.0, abs=1e-9)

    def test_higher_fm_increases_metabolite_exposure(self):
        """Increasing fm should increase metabolite AUC."""

        def run(fm_val):
            steps = [
                {"name": "pk", "cl_L_per_h": 1.0, "vd_L": 10.0, "fm": fm_val},
                {"name": "M1", "cl_L_per_h": 1.0, "vd_L": 10.0, "fm": 0.0},
            ]
            return simulate_sequential_metabolism(100.0, steps, t_end_h=48.0, dt_h=0.1).auc_values[
                1
            ]

        auc_low = run(0.3)
        auc_high = run(0.9)
        assert auc_high > auc_low

    def test_invalid_cl_raises(self):
        steps = [
            {"name": "pk", "cl_L_per_h": 0.0, "vd_L": 10.0, "fm": 1.0},
            {"name": "M1", "cl_L_per_h": 1.0, "vd_L": 10.0, "fm": 0.0},
        ]
        with pytest.raises(ValueError):
            simulate_sequential_metabolism(100.0, steps)


# ---------------------------------------------------------------------------
# calculate_metabolite_exposure_ratio
# ---------------------------------------------------------------------------


class TestCalculateMetaboliteExposureRatio:
    def test_equal_aucs(self):
        assert calculate_metabolite_exposure_ratio(100.0, 100.0) == pytest.approx(1.0)

    def test_ratio_less_than_one(self):
        assert calculate_metabolite_exposure_ratio(100.0, 50.0) == pytest.approx(0.5)

    def test_ratio_greater_than_one(self):
        assert calculate_metabolite_exposure_ratio(50.0, 100.0) == pytest.approx(2.0)

    def test_zero_parent_auc_raises(self):
        with pytest.raises(ValueError, match="parent_auc"):
            calculate_metabolite_exposure_ratio(0.0, 50.0)

    def test_negative_parent_auc_raises(self):
        with pytest.raises(ValueError, match="parent_auc"):
            calculate_metabolite_exposure_ratio(-10.0, 50.0)

    def test_zero_metabolite_auc(self):
        assert calculate_metabolite_exposure_ratio(100.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# predict_active_metabolite_contribution
# ---------------------------------------------------------------------------


class TestPredictActiveMetaboliteContribution:
    def _run(self, fm=0.5, potency=1.0, met_cl=1.0, par_cl=1.0):
        return predict_active_metabolite_contribution(
            parent_cl_L_per_h=par_cl,
            fm=fm,
            metabolite_potency_ratio=potency,
            metabolite_cl_L_per_h=met_cl,
            dose_mg=100.0,
            vd_L=10.0,
        )

    def test_returns_dict(self):
        res = self._run()
        assert isinstance(res, dict)

    def test_keys_present(self):
        res = self._run()
        assert "metabolite_auc_ratio" in res
        assert "effective_contribution" in res
        assert "dominant_species" in res
        assert "notes" in res

    def test_fm_zero_means_no_metabolite(self):
        res = self._run(fm=0.0)
        assert res["metabolite_auc_ratio"] == pytest.approx(0.0, abs=1e-9)

    def test_fm_one_same_cl_ratio_one(self):
        """fm=1, same CL → AUC_M = AUC_parent → ratio=1."""
        res = self._run(fm=1.0, met_cl=1.0, par_cl=1.0)
        assert res["metabolite_auc_ratio"] == pytest.approx(1.0)

    def test_higher_fm_increases_contribution(self):
        res_low = self._run(fm=0.1, potency=2.0)
        res_high = self._run(fm=0.9, potency=2.0)
        assert res_high["effective_contribution"] > res_low["effective_contribution"]

    def test_higher_potency_increases_contribution(self):
        res_low = self._run(fm=0.5, potency=0.5)
        res_high = self._run(fm=0.5, potency=5.0)
        assert res_high["effective_contribution"] > res_low["effective_contribution"]

    def test_dominant_species_parent_for_low_fm_potency(self):
        res = self._run(fm=0.05, potency=0.5)
        assert res["dominant_species"] == "parent"

    def test_dominant_species_metabolite_for_high_fm_potency(self):
        res = self._run(fm=1.0, potency=10.0, met_cl=0.1, par_cl=1.0)
        assert res["dominant_species"] == "metabolite"

    def test_invalid_parent_cl_raises(self):
        with pytest.raises(ValueError):
            predict_active_metabolite_contribution(0.0, 0.5, 1.0, 1.0, 100.0, 10.0)

    def test_invalid_fm_raises(self):
        with pytest.raises(ValueError):
            predict_active_metabolite_contribution(1.0, 1.5, 1.0, 1.0, 100.0, 10.0)

    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError):
            predict_active_metabolite_contribution(1.0, 0.5, 1.0, 1.0, -100.0, 10.0)

    def test_notes_is_str(self):
        res = self._run()
        assert isinstance(res["notes"], str)
