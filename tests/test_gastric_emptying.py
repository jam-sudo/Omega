"""Tests for gastric emptying and GI transit simulation (Phase 106)."""

from __future__ import annotations

import pytest

from omega_pbpk.biopharmaceutics.gastric_emptying import (
    GastricEmptyingResult,
    compare_food_states,
    simulate_gastric_emptying,
)


def _default(**kw):
    defaults = dict(
        state="fasted",
        dose_mg=100.0,
        ka_absorption_per_h=1.0,
        t_end_h=12.0,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_gastric_emptying(**defaults)


class TestGastricEmptyingResult:
    def test_returns_result_type(self):
        assert isinstance(_default(), GastricEmptyingResult)

    def test_fraction_remaining_between_0_and_1(self):
        r = _default()
        assert all(0.0 <= f <= 1.0 for f in r.fraction_remaining_gut)

    def test_fraction_absorbed_between_0_and_1(self):
        r = _default()
        assert all(0.0 <= f <= 1.0 for f in r.fraction_absorbed)

    def test_times_and_fractions_same_length(self):
        r = _default()
        assert len(r.times_h) == len(r.fraction_remaining_gut) == len(r.fraction_absorbed)

    def test_fraction_absorbed_increases(self):
        r = _default()
        fa = r.fraction_absorbed
        assert fa[-1] >= fa[0]

    def test_f_absorbed_1h_positive(self):
        r = _default()
        assert r.fraction_absorbed_1h >= 0.0

    def test_f_absorbed_4h_geq_1h(self):
        r = _default()
        assert r.fraction_absorbed_4h >= r.fraction_absorbed_1h - 1e-9

    def test_t_lag_positive(self):
        r = _default()
        assert r.t_lag_h >= 0.0

    def test_t_emptying_positive(self):
        r = _default()
        assert r.t_emptying_h > 0.0

    def test_ka_effective_positive(self):
        r = _default()
        assert r.ka_effective_per_h > 0.0

    def test_note_is_string(self):
        r = _default()
        assert isinstance(r.note, str)


class TestFastedVsFed:
    def test_fasted_empties_faster(self):
        r_fasted = _default(state="fasted")
        r_fed = _default(state="fed")
        assert r_fasted.t_emptying_h < r_fed.t_emptying_h

    def test_fasted_lag_less_than_fed(self):
        r_fasted = _default(state="fasted")
        r_fed = _default(state="fed")
        assert r_fasted.t_lag_h < r_fed.t_lag_h

    def test_fasted_absorbs_more_at_1h(self):
        r_fasted = _default(state="fasted")
        r_fed = _default(state="fed")
        assert r_fasted.fraction_absorbed_1h >= r_fed.fraction_absorbed_1h - 1e-6


class TestValidation:
    def test_invalid_state_raises(self):
        with pytest.raises(ValueError):
            _default(state="unknown")

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0.0)

    def test_zero_ka_raises(self):
        with pytest.raises(ValueError):
            _default(ka_absorption_per_h=0.0)


class TestCompareFoodStates:
    def test_returns_dict_with_fasted_and_fed(self):
        result = compare_food_states()
        assert "fasted" in result
        assert "fed" in result

    def test_both_result_types(self):
        result = compare_food_states()
        assert isinstance(result["fasted"], GastricEmptyingResult)
        assert isinstance(result["fed"], GastricEmptyingResult)
