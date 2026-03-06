"""Tests for compartmental model selection (Phase 99)."""

from __future__ import annotations

import numpy as np

from omega_pbpk.core.compartmental_analysis import (
    CompartmentResult,
    calculate_aic,
    calculate_bic,
    compare_subjects,
    select_compartment_model,
)


def _one_cpt_data(n=25):
    t = np.linspace(0, 24, n).tolist()
    c = (10.0 * np.exp(-0.2 * np.linspace(0, 24, n))).tolist()
    return t, c


def _two_cpt_data(n=25):
    t = np.linspace(0.1, 24, n).tolist()
    c = (
        8.0 * np.exp(-1.5 * np.linspace(0.1, 24, n)) + 4.0 * np.exp(-0.1 * np.linspace(0.1, 24, n))
    ).tolist()
    return t, c


class TestOneCompartmentFit:
    def test_returns_result_type(self):
        t, c = _one_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert isinstance(r, CompartmentResult)

    def test_n_compartments_equals_1_for_monoexp(self):
        t, c = _one_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert r.n_compartments == 1

    def test_r_squared_above_0_9(self):
        t, c = _one_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert r.r_squared > 0.9

    def test_half_lives_positive(self):
        t, c = _one_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert all(h > 0 for h in r.half_lives)

    def test_half_lives_one_value(self):
        t, c = _one_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert len(r.half_lives) == 1

    def test_predicted_conc_same_length(self):
        t, c = _one_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert len(r.predicted_conc) == len(t)


class TestTwoCompartmentFit:
    def test_two_cpt_r_squared_above_0_9(self):
        t, c = _two_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert r.r_squared > 0.9

    def test_half_lives_positive(self):
        t, c = _two_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert all(h > 0 for h in r.half_lives)

    def test_two_cpt_has_at_least_one_half_life(self):
        t, c = _two_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert len(r.half_lives) >= 1

    def test_residuals_length_matches(self):
        t, c = _two_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert len(r.residuals) == len(t)


class TestAICBIC:
    def test_aic_finite(self):
        t, c = _one_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert np.isfinite(r.aic)

    def test_bic_finite(self):
        t, c = _one_cpt_data()
        r = select_compartment_model("Drug", t, c)
        assert np.isfinite(r.bic)

    def test_more_params_increases_aic_same_sse(self):
        aic_2 = calculate_aic(n_points=20, n_params=2, sse=1.0)
        aic_4 = calculate_aic(n_points=20, n_params=4, sse=1.0)
        assert aic_4 > aic_2

    def test_more_params_increases_bic_same_sse(self):
        bic_2 = calculate_bic(n_points=20, n_params=2, sse=1.0)
        bic_4 = calculate_bic(n_points=20, n_params=4, sse=1.0)
        assert bic_4 > bic_2


class TestCompareSubjects:
    def test_returns_dict_with_keys(self):
        t, c = _one_cpt_data()
        subjects = [{"subject_id": f"S{i}", "times": t, "conc": c} for i in range(3)]
        result = compare_subjects(subjects)
        assert "best_model_counts" in result
        assert "mean_aic" in result
        assert "results" in result

    def test_results_length_matches_subjects(self):
        t, c = _one_cpt_data()
        subjects = [{"subject_id": f"S{i}", "times": t, "conc": c} for i in range(4)]
        result = compare_subjects(subjects)
        assert len(result["results"]) == 4

    def test_mean_aic_finite(self):
        t, c = _one_cpt_data()
        subjects = [{"subject_id": "S1", "times": t, "conc": c}]
        result = compare_subjects(subjects)
        assert np.isfinite(result["mean_aic"])
