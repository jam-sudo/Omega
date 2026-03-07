"""Tests for Phase 581 — predict_excretion_route functions."""

import pytest

from omega_pbpk.prediction.excretion_predictor import (
    biliary_excretion_likelihood,
    predict_excretion_route,
    renal_excretion_mechanism,
    urine_ph_effect_on_excretion,
)

# ---------------------------------------------------------------------------
# predict_excretion_route
# ---------------------------------------------------------------------------


class TestPredictExcretionRoute:
    def test_fe_sums_to_one(self):
        r = predict_excretion_route(300.0, 1.0, 0.5, 5.0, 3.0)
        assert abs(r["fe_renal"] + r["fe_biliary"] - 1.0) < 1e-9

    def test_primary_route_renal_when_high_cl_renal(self):
        r = predict_excretion_route(200.0, 0.5, 0.8, 10.0, 1.0)
        assert r["primary_route"] == "renal"

    def test_primary_route_hepatic_when_high_cl_hepatic(self):
        r = predict_excretion_route(450.0, 2.5, 0.3, 1.0, 15.0)
        assert r["primary_route"] == "hepatic"

    def test_primary_route_mixed_equal_clearances(self):
        r = predict_excretion_route(300.0, 1.0, 0.5, 5.0, 5.0)
        assert r["primary_route"] == "mixed"

    def test_total_cl_is_sum(self):
        r = predict_excretion_route(300.0, 1.0, 0.5, 4.0, 6.0)
        assert abs(r["total_cl_L_per_h"] - 10.0) < 1e-9

    def test_zero_total_cl_returns_undetermined(self):
        r = predict_excretion_route(300.0, 1.0, 0.5, 0.0, 0.0)
        assert r["primary_route"] == "undetermined"
        assert r["fe_renal"] == 0.0
        assert r["fe_biliary"] == 0.0

    def test_negative_cl_renal_raises(self):
        with pytest.raises(ValueError):
            predict_excretion_route(300.0, 1.0, 0.5, -1.0, 5.0)

    def test_negative_cl_hepatic_raises(self):
        with pytest.raises(ValueError):
            predict_excretion_route(300.0, 1.0, 0.5, 5.0, -1.0)

    def test_invalid_fu_raises(self):
        with pytest.raises(ValueError):
            predict_excretion_route(300.0, 1.0, 1.5, 5.0, 5.0)

    def test_notes_returned(self):
        r = predict_excretion_route(300.0, 1.0, 0.5, 5.0, 5.0)
        assert isinstance(r["notes"], str)

    def test_high_mw_note_present_when_biliary(self):
        r = predict_excretion_route(600.0, 3.0, 0.5, 2.0, 8.0)
        assert "MW" in r["notes"]


# ---------------------------------------------------------------------------
# renal_excretion_mechanism
# ---------------------------------------------------------------------------


class TestRenalExcretionMechanism:
    def test_high_logP_high_reabsorption(self):
        _r = predict_excretion_route(300.0, 1.0, 0.5, 5.0, 5.0)
        rem = renal_excretion_mechanism(0.5, 4.0, 350.0)
        assert rem["reabsorption_fraction"] > 0.3

    def test_cl_renal_net_positive(self):
        rem = renal_excretion_mechanism(0.8, 0.5, 300.0)
        assert rem["cl_renal_net_L_per_h"] > 0

    def test_negative_logP_zero_reabsorption(self):
        rem = renal_excretion_mechanism(0.8, -1.0, 300.0)
        assert rem["reabsorption_fraction"] == 0.0

    def test_active_secretion_triggered(self):
        rem = renal_excretion_mechanism(0.8, 0.5, 200.0)
        assert rem["active_secretion_bonus"] > 0

    def test_no_active_secretion_high_mw(self):
        rem = renal_excretion_mechanism(0.8, 0.5, 600.0)
        assert rem["active_secretion_bonus"] == 0.0

    def test_cl_filtration_proportional_to_fu(self):
        rem1 = renal_excretion_mechanism(0.2, -0.5, 300.0)
        rem2 = renal_excretion_mechanism(0.4, -0.5, 300.0)
        assert abs(rem2["cl_filtration"] / rem1["cl_filtration"] - 2.0) < 1e-6

    def test_invalid_fu_raises(self):
        with pytest.raises(ValueError):
            renal_excretion_mechanism(-0.1, 1.0, 300.0)

    def test_reabsorption_capped_at_0p9(self):
        rem = renal_excretion_mechanism(0.8, 8.0, 300.0)
        assert rem["reabsorption_fraction"] <= 0.9


# ---------------------------------------------------------------------------
# biliary_excretion_likelihood
# ---------------------------------------------------------------------------


class TestBiliaryExcretionLikelihood:
    def test_high_mw_high_logP_high_hba_gives_high(self):
        r = biliary_excretion_likelihood(mw=600.0, logP=3.0, n_hba=6)
        assert r["likelihood"] == "high"

    def test_score_3_for_all_criteria(self):
        r = biliary_excretion_likelihood(mw=600.0, logP=3.0, n_hba=6)
        assert r["score"] == 3

    def test_low_mw_low_logP_low_hba_gives_low(self):
        r = biliary_excretion_likelihood(mw=150.0, logP=-1.0, n_hba=1)
        assert r["likelihood"] == "low"
        assert r["score"] == 0

    def test_moderate_mw_range_and_criteria(self):
        r = biliary_excretion_likelihood(mw=400.0, logP=2.0, n_hba=3)
        # MW=400 (not >500, score 0); logP=2 (not >2, score 0); HBA=3 (not >4, score 0)
        # but mw>=300, logP>=1 and <=4 — only if score>=1; score=0 so stays low
        assert r["likelihood"] == "low"

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError):
            biliary_excretion_likelihood(-100.0, 1.0, 3)

    def test_negative_hba_raises(self):
        with pytest.raises(ValueError):
            biliary_excretion_likelihood(400.0, 1.0, -1)

    def test_notes_contain_mw_when_high(self):
        r = biliary_excretion_likelihood(mw=600.0, logP=3.0, n_hba=6)
        assert "MW" in r["notes"]


# ---------------------------------------------------------------------------
# urine_ph_effect_on_excretion
# ---------------------------------------------------------------------------


class TestUrinePHEffect:
    def test_acidic_urine_decreases_cl_for_acid_drug(self):
        # Acid pKa=5, acidic urine pH=4.5 → less ionized → more reabsorption → lower CL
        r_acid = urine_ph_effect_on_excretion(5.0, "acid", 4.5, 2.0)
        r_base_ph = urine_ph_effect_on_excretion(5.0, "acid", 8.0, 2.0)
        assert r_acid["cl_renal_adjusted_L_per_h"] < r_base_ph["cl_renal_adjusted_L_per_h"]

    def test_effect_direction_for_acid_in_basic_urine(self):
        r = urine_ph_effect_on_excretion(5.0, "acid", 8.0, 2.0)
        assert r["effect_direction"] == "increased"

    def test_effect_direction_for_acid_in_acidic_urine(self):
        r = urine_ph_effect_on_excretion(5.0, "acid", 4.0, 2.0)
        assert r["effect_direction"] == "decreased"

    def test_effect_direction_for_base_in_acidic_urine(self):
        r = urine_ph_effect_on_excretion(9.0, "base", 4.5, 2.0)
        assert r["effect_direction"] == "increased"

    def test_fi_urine_in_0_1_range(self):
        r = urine_ph_effect_on_excretion(7.0, "acid", 6.0, 3.0)
        assert 0.0 <= r["fi_urine"] <= 1.0

    def test_cl_adjusted_positive(self):
        r = urine_ph_effect_on_excretion(5.0, "acid", 6.0, 2.0)
        assert r["cl_renal_adjusted_L_per_h"] > 0

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError):
            urine_ph_effect_on_excretion(5.0, "neutral", 6.0, 2.0)

    def test_invalid_urine_ph_raises(self):
        with pytest.raises(ValueError):
            urine_ph_effect_on_excretion(5.0, "acid", 15.0, 2.0)

    def test_negative_cl_baseline_raises(self):
        with pytest.raises(ValueError):
            urine_ph_effect_on_excretion(5.0, "acid", 6.0, -1.0)
