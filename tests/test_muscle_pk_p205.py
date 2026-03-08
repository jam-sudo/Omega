"""Tests for Phase 205 — core/muscle_pk_p205.py (Skeletal Muscle PK)."""

import pytest

from omega_pbpk.core.muscle_pk_p205 import (
    MusclePKResult,
    compare_activity_states,
    simulate_muscle_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> MusclePKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_plasma_L=20.0,
        muscle_fraction=0.40,
        body_weight_kg=70.0,
        activity_state="resting",
    )
    defaults.update(kwargs)
    return simulate_muscle_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_muscle_pk_result(self):
        r = _default_result()
        assert isinstance(r, MusclePKResult)

    def test_drug_name_preserved(self):
        r = _default_result(drug_name="Ibuprofen")
        assert r.drug_name == "Ibuprofen"

    def test_dose_mg_preserved(self):
        r = _default_result(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_activity_state_preserved(self):
        r = _default_result(activity_state="exercise")
        assert r.activity_state == "exercise"

    def test_muscle_fraction_preserved(self):
        r = _default_result(muscle_fraction=0.35)
        assert r.muscle_fraction == pytest.approx(0.35)

    def test_body_weight_preserved(self):
        r = _default_result(body_weight_kg=80.0)
        assert r.body_weight_kg == pytest.approx(80.0)

    def test_notes_is_string(self):
        r = _default_result()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# List field correctness
# ---------------------------------------------------------------------------


class TestListFields:
    def test_times_h_is_list(self):
        r = _default_result()
        assert isinstance(r.times_h, list)

    def test_c_plasma_is_list(self):
        r = _default_result()
        assert isinstance(r.c_plasma_mg_L, list)

    def test_c_muscle_is_list(self):
        r = _default_result()
        assert isinstance(r.c_muscle_mg_kg, list)

    def test_same_length(self):
        r = _default_result()
        assert len(r.times_h) == len(r.c_plasma_mg_L) == len(r.c_muscle_mg_kg)

    def test_times_start_at_zero(self):
        r = _default_result()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_c_plasma_starts_at_dose_over_vd(self):
        r = _default_result(dose_mg=100.0, vd_plasma_L=20.0)
        # C(0) = dose / Vd = 100/20 = 5 mg/L
        assert r.c_plasma_mg_L[0] == pytest.approx(5.0, rel=1e-3)

    def test_c_muscle_starts_at_zero(self):
        r = _default_result()
        assert r.c_muscle_mg_kg[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# PK metric correctness
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_cmax_muscle_positive(self):
        r = _default_result()
        assert r.cmax_muscle_mg_kg > 0.0

    def test_auc_plasma_positive(self):
        r = _default_result()
        assert r.auc_plasma > 0.0

    def test_auc_muscle_positive(self):
        r = _default_result()
        assert r.auc_muscle > 0.0

    def test_muscle_to_plasma_ratio_positive(self):
        r = _default_result()
        assert r.muscle_to_plasma_ratio > 0.0

    def test_tmax_muscle_is_non_negative(self):
        r = _default_result()
        assert r.tmax_muscle_h >= 0.0

    def test_tmax_muscle_within_simulation_time(self):
        r = _default_result(t_end_h=24.0)
        assert r.tmax_muscle_h <= 24.0

    def test_muscle_to_plasma_ratio_equals_auc_ratio(self):
        r = _default_result()
        expected = r.auc_muscle / r.auc_plasma
        assert r.muscle_to_plasma_ratio == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Exercise vs. resting
# ---------------------------------------------------------------------------


class TestActivityState:
    def test_exercise_higher_muscle_auc_than_resting(self):
        resting = _default_result(activity_state="resting")
        exercise = _default_result(activity_state="exercise")
        assert exercise.auc_muscle > resting.auc_muscle

    def test_exercise_higher_cmax_muscle_than_resting(self):
        resting = _default_result(activity_state="resting")
        exercise = _default_result(activity_state="exercise")
        assert exercise.cmax_muscle_mg_kg > resting.cmax_muscle_mg_kg


# ---------------------------------------------------------------------------
# compare_activity_states
# ---------------------------------------------------------------------------


class TestCompareActivityStates:
    def test_returns_two_results(self):
        results = compare_activity_states("Drug", 100.0)
        assert len(results) == 2

    def test_exercise_first(self):
        results = compare_activity_states("Drug", 100.0)
        assert results[0].activity_state == "exercise"

    def test_resting_second(self):
        results = compare_activity_states("Drug", 100.0)
        assert results[1].activity_state == "resting"

    def test_sorted_by_auc_muscle_descending(self):
        results = compare_activity_states("Drug", 100.0)
        assert results[0].auc_muscle >= results[1].auc_muscle

    def test_all_results_are_muscle_pk_result(self):
        results = compare_activity_states("Drug", 100.0)
        for r in results:
            assert isinstance(r, MusclePKResult)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_muscle_pk("Drug", dose_mg=0.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_muscle_pk("Drug", dose_mg=-50.0)

    def test_muscle_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="muscle_fraction"):
            simulate_muscle_pk("Drug", dose_mg=100.0, muscle_fraction=0.0)

    def test_muscle_fraction_one_raises(self):
        with pytest.raises(ValueError, match="muscle_fraction"):
            simulate_muscle_pk("Drug", dose_mg=100.0, muscle_fraction=1.0)

    def test_muscle_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="muscle_fraction"):
            simulate_muscle_pk("Drug", dose_mg=100.0, muscle_fraction=-0.1)

    def test_invalid_activity_state_raises(self):
        with pytest.raises(ValueError, match="activity_state"):
            simulate_muscle_pk("Drug", dose_mg=100.0, activity_state="jogging")
