"""Tests for Phase 187 — formulation_stability_predictor_p187.py."""

import math

import pytest

from omega_pbpk.prediction.formulation_stability_predictor_p187 import (
    StabilityResult,
    compare_storage_conditions,
    predict_stability,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_DEFAULT_KWARGS = dict(
    drug_name="DrugA",
    activation_energy_kJ_mol=80.0,
    k_ref_per_day=1e-4,
    t_ref_C=25.0,
    storage_condition="25C_60RH",
)


def _make_result(**overrides) -> StabilityResult:
    kw = {**_DEFAULT_KWARGS, **overrides}
    return predict_stability(**kw)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_stability_result(self):
        r = _make_result()
        assert isinstance(r, StabilityResult)

    def test_drug_name_preserved(self):
        r = _make_result(drug_name="Aspirin")
        assert r.drug_name == "Aspirin"

    def test_storage_condition_preserved(self):
        r = _make_result(storage_condition="40C_75RH")
        assert r.storage_condition == "40C_75RH"

    def test_temperature_correct_25C(self):
        r = _make_result(storage_condition="25C_60RH")
        assert r.temperature_C == pytest.approx(25.0)

    def test_temperature_correct_30C(self):
        r = _make_result(storage_condition="30C_65RH")
        assert r.temperature_C == pytest.approx(30.0)

    def test_temperature_correct_40C(self):
        r = _make_result(storage_condition="40C_75RH")
        assert r.temperature_C == pytest.approx(40.0)

    def test_temperature_correct_5C(self):
        r = _make_result(storage_condition="5C")
        assert r.temperature_C == pytest.approx(5.0)

    def test_stability_class_is_string(self):
        r = _make_result()
        assert isinstance(r.stability_class, str)

    def test_notes_is_string(self):
        r = _make_result()
        assert isinstance(r.notes, str)


# ---------------------------------------------------------------------------
# k_per_day Arrhenius behaviour
# ---------------------------------------------------------------------------


class TestArrheniusKinetics:
    def test_k_increases_with_temperature(self):
        r_25 = _make_result(storage_condition="25C_60RH")
        r_40 = _make_result(storage_condition="40C_75RH")
        assert r_40.k_per_day > r_25.k_per_day

    def test_k_refrigerated_lower_than_room(self):
        r_25 = _make_result(storage_condition="25C_60RH")
        r_5 = _make_result(storage_condition="5C")
        assert r_5.k_per_day < r_25.k_per_day

    def test_k_25_equals_k_ref_when_t_ref_equals_storage(self):
        r = predict_stability(
            drug_name="X",
            activation_energy_kJ_mol=80.0,
            k_ref_per_day=5e-4,
            t_ref_C=25.0,
            storage_condition="25C_60RH",
        )
        assert r.k_per_day == pytest.approx(5e-4, rel=1e-6)

    def test_k_positive(self):
        r = _make_result()
        assert r.k_per_day > 0.0


# ---------------------------------------------------------------------------
# % remaining values
# ---------------------------------------------------------------------------


class TestPctRemaining:
    def test_pct_remaining_decreases_over_time(self):
        r = _make_result()
        assert r.pct_remaining_6m > r.pct_remaining_12m
        assert r.pct_remaining_12m > r.pct_remaining_24m
        assert r.pct_remaining_24m > r.pct_remaining_36m

    def test_pct_remaining_in_range(self):
        r = _make_result()
        for val in (
            r.pct_remaining_6m,
            r.pct_remaining_12m,
            r.pct_remaining_24m,
            r.pct_remaining_36m,
        ):
            assert 0.0 <= val <= 100.0

    def test_higher_temp_lower_pct_remaining(self):
        r_25 = _make_result(storage_condition="25C_60RH")
        r_40 = _make_result(storage_condition="40C_75RH")
        assert r_40.pct_remaining_24m < r_25.pct_remaining_24m

    def test_refrigerated_highest_pct_remaining(self):
        r_5 = _make_result(storage_condition="5C")
        r_40 = _make_result(storage_condition="40C_75RH")
        assert r_5.pct_remaining_24m > r_40.pct_remaining_24m


# ---------------------------------------------------------------------------
# Shelf life
# ---------------------------------------------------------------------------


class TestShelfLife:
    def test_shelf_life_positive(self):
        r = _make_result()
        assert r.shelf_life_months > 0.0

    def test_lower_temp_longer_shelf_life(self):
        r_25 = _make_result(storage_condition="25C_60RH")
        r_40 = _make_result(storage_condition="40C_75RH")
        assert r_25.shelf_life_months > r_40.shelf_life_months

    def test_refrigerated_longest_shelf_life(self):
        r_5 = _make_result(storage_condition="5C")
        r_40 = _make_result(storage_condition="40C_75RH")
        assert r_5.shelf_life_months > r_40.shelf_life_months

    def test_shelf_life_consistent_with_90_pct(self):
        # At shelf_life_months, ~90% should remain (to within numerical tolerance)
        r = _make_result(storage_condition="25C_60RH")
        days = r.shelf_life_months * 30.4375
        pct = 100.0 * math.exp(-r.k_per_day * days)
        assert pct == pytest.approx(90.0, abs=1.0)


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


class TestStabilityClass:
    def test_stable_class_slow_degradation(self):
        # Very small k → >95% at 24m → stable
        r = predict_stability(
            drug_name="Stable",
            activation_energy_kJ_mol=60.0,
            k_ref_per_day=1e-6,
            t_ref_C=25.0,
            storage_condition="25C_60RH",
        )
        assert r.stability_class == "stable"

    def test_unstable_class_fast_degradation(self):
        # Large k → <90% at 24m → unstable
        r = predict_stability(
            drug_name="Unstable",
            activation_energy_kJ_mol=60.0,
            k_ref_per_day=1e-2,
            t_ref_C=25.0,
            storage_condition="40C_75RH",
        )
        assert r.stability_class == "unstable"

    def test_valid_stability_class_values(self):
        valid = {"stable", "acceptable", "unstable"}
        for cond in ("25C_60RH", "30C_65RH", "40C_75RH", "5C"):
            r = _make_result(storage_condition=cond)
            assert r.stability_class in valid


# ---------------------------------------------------------------------------
# compare_storage_conditions
# ---------------------------------------------------------------------------


class TestCompareStorageConditions:
    def test_returns_list(self):
        results = compare_storage_conditions("DrugB", 80.0, 1e-4, 25.0)
        assert isinstance(results, list)

    def test_returns_four_results(self):
        results = compare_storage_conditions("DrugB", 80.0, 1e-4, 25.0)
        assert len(results) == 4

    def test_all_stability_results(self):
        results = compare_storage_conditions("DrugB", 80.0, 1e-4, 25.0)
        for r in results:
            assert isinstance(r, StabilityResult)

    def test_sorted_descending_by_shelf_life(self):
        results = compare_storage_conditions("DrugB", 80.0, 1e-4, 25.0)
        shelf_lives = [r.shelf_life_months for r in results]
        assert shelf_lives == sorted(shelf_lives, reverse=True)

    def test_all_four_conditions_present(self):
        results = compare_storage_conditions("DrugB", 80.0, 1e-4, 25.0)
        conds = {r.storage_condition for r in results}
        assert conds == {"25C_60RH", "30C_65RH", "40C_75RH", "5C"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_ea_raises(self):
        with pytest.raises(ValueError):
            _make_result(activation_energy_kJ_mol=-10.0)

    def test_zero_ea_raises(self):
        with pytest.raises(ValueError):
            _make_result(activation_energy_kJ_mol=0.0)

    def test_negative_k_ref_raises(self):
        with pytest.raises(ValueError):
            _make_result(k_ref_per_day=-1e-4)

    def test_zero_k_ref_raises(self):
        with pytest.raises(ValueError):
            _make_result(k_ref_per_day=0.0)

    def test_invalid_storage_condition_raises(self):
        with pytest.raises(ValueError):
            _make_result(storage_condition="invalid_condition")

    def test_unknown_storage_condition_raises(self):
        with pytest.raises(ValueError):
            _make_result(storage_condition="60C_90RH")
