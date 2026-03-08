"""Tests for Phase 692 — Drug-Food Interaction Kinetics."""

import pytest

from omega_pbpk.clinical.drug_food_interaction_p692 import (
    DrugFoodInteractionResult,
    simulate_drug_food_interaction,
)

_DEFAULTS = dict(
    drug_name="TestDrug",
    dose_mg=100.0,
    logP=2.5,
    solubility_mg_mL=0.1,
    cl_sys_L_per_h=5.0,
    vd_sys_L=50.0,
)


def _run(**overrides):
    params = {**_DEFAULTS, **overrides}
    return simulate_drug_food_interaction(**params)


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, DrugFoodInteractionResult)

    def test_drug_name(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_dose_mg(self):
        r = _run()
        assert r.dose_mg == 100.0

    def test_food_state(self):
        r = _run()
        assert r.food_state == "high_fat"

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)
        assert len(r.times_h) > 0

    def test_all_lists_same_length(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_plasma_fasted_mg_L) == n
        assert len(r.c_plasma_fed_mg_L) == n


class TestConcentrations:
    def test_fasted_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_fasted_mg_L)

    def test_fed_non_negative(self):
        r = _run()
        assert all(c >= 0 for c in r.c_plasma_fed_mg_L)

    def test_cmax_fasted_positive(self):
        r = _run()
        assert r.cmax_fasted_mg_L > 0

    def test_cmax_fed_positive(self):
        r = _run()
        assert r.cmax_fed_mg_L > 0

    def test_auc_fasted_positive(self):
        r = _run()
        assert r.auc_fasted_mg_h_per_L > 0

    def test_auc_fed_positive(self):
        r = _run()
        assert r.auc_fed_mg_h_per_L > 0


class TestBCS:
    def test_bcs_class_valid(self):
        r = _run()
        assert r.bcs_class in {"BCS I", "BCS II", "BCS III", "BCS IV"}

    def test_bcs_ii_low_sol_high_perm(self):
        # logP=2.5 (high perm), sol=0.1 (low sol) -> BCS II
        r = _run(logP=2.5, solubility_mg_mL=0.1)
        assert r.bcs_class == "BCS II"

    def test_bcs_i_high_sol_high_perm(self):
        r = _run(logP=2.0, solubility_mg_mL=5.0)
        assert r.bcs_class == "BCS I"

    def test_bcs_iii_high_sol_low_perm(self):
        r = _run(logP=0.5, solubility_mg_mL=5.0)
        assert r.bcs_class == "BCS III"

    def test_bcs_iv_low_sol_low_perm(self):
        r = _run(logP=0.5, solubility_mg_mL=0.1)
        assert r.bcs_class == "BCS IV"


class TestFoodEffect:
    def test_food_effect_cmax_is_float(self):
        r = _run()
        assert isinstance(r.food_effect_cmax, float)
        assert r.food_effect_cmax > 0

    def test_food_effect_auc_is_float(self):
        r = _run()
        assert isinstance(r.food_effect_auc, float)
        assert r.food_effect_auc > 0

    def test_high_fat_increases_auc_bcs_ii(self):
        # BCS II drug (low sol, high logP) should see positive food effect
        r = _run(logP=3.0, solubility_mg_mL=0.05, food_state="high_fat")
        assert r.food_effect_auc > 1.0

    def test_fasted_no_food_effect(self):
        r = _run(food_state="fasted")
        assert r.food_effect_cmax == 1.0
        assert r.food_effect_auc == 1.0

    def test_fasted_fed_equals_fasted(self):
        r = _run(food_state="fasted")
        assert r.c_plasma_fed_mg_L == r.c_plasma_fasted_mg_L


class TestRecommendation:
    def test_recommendation_valid_string(self):
        r = _run()
        valid = {
            "Take with food",
            "Take on empty stomach",
            "May be taken with or without food",
        }
        assert r.recommendation in valid

    def test_take_with_food_when_positive_effect(self):
        # High logP, low sol -> big food effect -> "Take with food"
        r = _run(logP=4.0, solubility_mg_mL=0.01, food_state="high_fat")
        assert r.recommendation == "Take with food"


class TestDoseScaling:
    def test_double_dose_doubles_cmax_fasted(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_fasted_mg_L / r1.cmax_fasted_mg_L
        assert abs(ratio - 2.0) < 0.15

    def test_double_dose_doubles_cmax_fed(self):
        r1 = _run(dose_mg=100.0)
        r2 = _run(dose_mg=200.0)
        ratio = r2.cmax_fed_mg_L / r1.cmax_fed_mg_L
        assert abs(ratio - 2.0) < 0.15


class TestValidation:
    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _run(dose_mg=-1)

    def test_zero_cl(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _run(cl_sys_L_per_h=0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _run(vd_sys_L=0)

    def test_zero_solubility(self):
        with pytest.raises(ValueError, match="solubility"):
            _run(solubility_mg_mL=0)

    def test_invalid_food_state(self):
        with pytest.raises(ValueError, match="food_state"):
            _run(food_state="brunch")


class TestNotes:
    def test_notes_is_string(self):
        r = _run()
        assert isinstance(r.notes, str)

    def test_notes_not_empty(self):
        r = _run()
        assert len(r.notes) > 0
