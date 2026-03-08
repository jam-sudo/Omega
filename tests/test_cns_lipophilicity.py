"""Tests for Phase 526: CNS lipophilicity penetration prediction."""

import math

import pytest

from omega_pbpk.prediction.cns_lipophilicity import (
    CNSLipophilicityResult,
    predict_cns_penetration,
    screen_cns_penetration,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _pred(drug="TestDrug", logP=2.0, fu_plasma=0.1, c_plasma=1.0):
    return predict_cns_penetration(drug, logP, fu_plasma, c_plasma)


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_object(self):
        r = _pred()
        assert isinstance(r, CNSLipophilicityResult)

    def test_drug_name_stored(self):
        r = predict_cns_penetration("Morphine", 0.9)
        assert r.drug_name == "Morphine"

    def test_logP_stored(self):
        r = _pred(logP=3.5)
        assert r.logP == pytest.approx(3.5)

    def test_fu_plasma_stored(self):
        r = _pred(fu_plasma=0.2)
        assert r.fu_plasma == pytest.approx(0.2)

    def test_ps_bbb_positive(self):
        r = _pred()
        assert r.ps_bbb_L_per_h > 0.0

    def test_fu_brain_in_valid_range(self):
        r = _pred()
        assert 0.01 <= r.fu_brain <= 1.0

    def test_kp_brain_in_valid_range(self):
        r = _pred()
        assert 0.0001 <= r.kp_brain <= 10.0

    def test_c_plasma_stored(self):
        r = _pred(c_plasma=2.5)
        assert r.c_plasma_mg_L == pytest.approx(2.5)

    def test_notes_nonempty(self):
        r = _pred()
        assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Physics / model correctness
# ---------------------------------------------------------------------------


class TestModelCorrectness:
    def test_c_brain_equals_c_plasma_times_kp(self):
        c_plasma = 3.0
        r = _pred(c_plasma=c_plasma)
        assert r.c_brain_mg_L == pytest.approx(r.c_plasma_mg_L * r.kp_brain, rel=1e-5)

    def test_logBB_equals_log10_kp_brain(self):
        r = _pred()
        assert r.logBB == pytest.approx(math.log10(r.kp_brain), rel=1e-5)

    def test_fu_brain_decreases_with_logP(self):
        r_low = _pred(logP=1.0)
        r_high = _pred(logP=4.0)
        assert r_high.fu_brain < r_low.fu_brain

    def test_higher_logP_higher_ps_bbb_before_clamping(self):
        """For logP values that don't saturate the clamp, higher logP → higher PS."""
        r_low = _pred(logP=-1.0)
        r_mid = _pred(logP=1.0)
        assert r_mid.ps_bbb_L_per_h >= r_low.ps_bbb_L_per_h

    def test_ps_bbb_clamped_at_max(self):
        """Very high logP should hit the upper clamp."""
        r = _pred(logP=10.0)
        assert r.ps_bbb_L_per_h <= 100.0

    def test_ps_bbb_clamped_at_min(self):
        """Very low logP should hit the lower clamp."""
        r = _pred(logP=-10.0)
        assert r.ps_bbb_L_per_h >= 0.001

    def test_kp_brain_clamped_at_max(self):
        r = _pred(logP=10.0, fu_plasma=1.0)
        assert r.kp_brain <= 10.0

    def test_fu_brain_clamped_at_min(self):
        r = _pred(logP=50.0)
        assert r.fu_brain >= 0.01


# ---------------------------------------------------------------------------
# CNS classification
# ---------------------------------------------------------------------------


class TestCNSClassification:
    def test_poor_class_low_logP(self):
        """Very hydrophilic compound → poor CNS penetration."""
        r = _pred(logP=-3.0, fu_plasma=0.05)
        assert r.cns_class == "poor"

    def test_excellent_class_high_logP(self):
        """Highly lipophilic with good plasma fu → excellent CNS penetration."""
        r = predict_cns_penetration("Lipophilic", logP=5.0, fu_plasma=0.8)
        assert r.cns_class in ("excellent", "good")

    def test_classification_thresholds(self):
        # Build a result with known Kp and check class
        r = _pred(logP=2.0, fu_plasma=0.5)
        if r.kp_brain > 0.5:
            assert r.cns_class == "excellent"
        elif r.kp_brain > 0.1:
            assert r.cns_class == "good"
        else:
            assert r.cns_class == "poor"

    def test_cns_class_is_string(self):
        r = _pred()
        assert isinstance(r.cns_class, str)

    def test_cns_class_valid_values(self):
        for logP in [-2.0, 1.0, 3.0, 5.0]:
            r = _pred(logP=logP)
            assert r.cns_class in ("excellent", "good", "poor")


# ---------------------------------------------------------------------------
# Brain binding concern
# ---------------------------------------------------------------------------


class TestBrainBindingConcern:
    def test_brain_binding_concern_high_logP(self):
        """High logP → low fu_brain → brain binding concern."""
        # fu_brain = 1/(1 + 0.5*5) = 1/3.5 ≈ 0.286 > 0.1, no concern
        # logP=10 → 1/(1+5) = 0.167, > 0.1
        # logP=20 clamped to 0.01 → concern
        r_extreme = _pred(logP=20.0)
        assert r_extreme.brain_binding_concern is True

    def test_brain_binding_concern_low_logP(self):
        """Low logP → high fu_brain → no concern."""
        r = _pred(logP=0.0)
        # fu_brain = 1/(1+0) = 1.0 → no concern
        assert r.brain_binding_concern is False

    def test_brain_binding_concern_threshold(self):
        # fu_brain threshold is 0.1
        r = _pred(logP=2.0)
        assert r.brain_binding_concern == (r.fu_brain < 0.1)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_cns_penetration("D", 2.0, fu_plasma=0.0)

    def test_fu_plasma_negative_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_cns_penetration("D", 2.0, fu_plasma=-0.1)

    def test_fu_plasma_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_cns_penetration("D", 2.0, fu_plasma=1.1)

    def test_c_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="c_plasma"):
            predict_cns_penetration("D", 2.0, c_plasma_mg_L=0.0)

    def test_c_plasma_negative_raises(self):
        with pytest.raises(ValueError, match="c_plasma"):
            predict_cns_penetration("D", 2.0, c_plasma_mg_L=-1.0)


# ---------------------------------------------------------------------------
# screen_cns_penetration
# ---------------------------------------------------------------------------


class TestScreenCNSPenetration:
    def _compounds(self):
        return [
            {"drug_name": "DrugA", "logP": 1.0, "fu_plasma": 0.1},
            {"drug_name": "DrugB", "logP": 3.0, "fu_plasma": 0.2},
            {"drug_name": "DrugC", "logP": -1.0, "fu_plasma": 0.5},
        ]

    def test_returns_list(self):
        results = screen_cns_penetration(self._compounds())
        assert isinstance(results, list)

    def test_correct_count(self):
        results = screen_cns_penetration(self._compounds())
        assert len(results) == 3

    def test_sorted_by_kp_brain_descending(self):
        results = screen_cns_penetration(self._compounds())
        kps = [r.kp_brain for r in results]
        assert kps == sorted(kps, reverse=True)

    def test_all_result_objects(self):
        results = screen_cns_penetration(self._compounds())
        for r in results:
            assert isinstance(r, CNSLipophilicityResult)

    def test_default_fu_plasma_and_c_plasma(self):
        """screen should work without optional keys."""
        compounds = [{"drug_name": "X", "logP": 2.0}]
        results = screen_cns_penetration(compounds)
        assert len(results) == 1
        assert results[0].fu_plasma == pytest.approx(0.1)
        assert results[0].c_plasma_mg_L == pytest.approx(1.0)

    def test_empty_list_returns_empty(self):
        results = screen_cns_penetration([])
        assert results == []
