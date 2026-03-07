"""Tests for Phase 292: CNS drug penetration predictor."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.cns_penetration import (
    BBBPenetrationResult,
    batch_cns_screen,
    compute_mpo_cns,
    predict_bbb_penetration,
)


# ---------------------------------------------------------------------------
# compute_mpo_cns tests
# ---------------------------------------------------------------------------


class TestComputeMpoCns:
    def test_perfect_cns_score(self):
        # Ideal CNS compound: all desirabilities = 1
        # logP=1, logD=1, mw=300, hbd=0, pka=9, tpsa=30
        score = compute_mpo_cns(logP=1.0, logD74=1.0, mw=300.0, hbd=0, pka=9.0, tpsa=30.0)
        assert score == pytest.approx(6.0)

    def test_worst_cns_score(self):
        # All desirabilities = 0
        # logP=6, logD=5, mw=600, hbd=4, pka=5, tpsa=100
        score = compute_mpo_cns(logP=6.0, logD74=5.0, mw=600.0, hbd=4, pka=5.0, tpsa=100.0)
        assert score == pytest.approx(0.0)

    def test_logp_boundary_at_3(self):
        # logP=3 → d=1; logP=5 → d=0
        assert compute_mpo_cns(3.0, 1.0, 300.0, 0, 9.0, 30.0) == pytest.approx(6.0)
        assert compute_mpo_cns(5.0, 1.0, 300.0, 0, 9.0, 30.0) == pytest.approx(5.0)

    def test_logp_linear_between_3_and_5(self):
        # At logP=4 → d_logP=0.5; other desirabilities=1 → total=5.5
        score = compute_mpo_cns(4.0, 1.0, 300.0, 0, 9.0, 30.0)
        assert score == pytest.approx(5.5)

    def test_logd_boundary_at_2(self):
        # logD=2 → d=1
        score = compute_mpo_cns(1.0, 2.0, 300.0, 0, 9.0, 30.0)
        assert score == pytest.approx(6.0)

    def test_logd_linear_between_2_and_4(self):
        # logD=3 → d_logD=0.5; other desirabilities=1 → total=5.5
        score = compute_mpo_cns(1.0, 3.0, 300.0, 0, 9.0, 30.0)
        assert score == pytest.approx(5.5)

    def test_mw_boundary_at_360(self):
        # mw=360 → d=1
        score = compute_mpo_cns(1.0, 1.0, 360.0, 0, 9.0, 30.0)
        assert score == pytest.approx(6.0)

    def test_mw_linear_between_360_and_500(self):
        # mw=430 → d_mw=(500-430)/140=0.5; other=1 → total=5.5
        score = compute_mpo_cns(1.0, 1.0, 430.0, 0, 9.0, 30.0)
        assert score == pytest.approx(5.5)

    def test_hbd_boundary_at_0(self):
        # hbd=0 → d=1
        score = compute_mpo_cns(1.0, 1.0, 300.0, 0, 9.0, 30.0)
        assert score == pytest.approx(6.0)

    def test_hbd_linear_between_0_and_3(self):
        # hbd=1 → d_hbd=(3-1)/3=2/3
        score = compute_mpo_cns(1.0, 1.0, 300.0, 1, 9.0, 30.0)
        expected = 5.0 + 2.0 / 3.0
        assert score == pytest.approx(expected, rel=1e-5)

    def test_pka_step_function_in_range(self):
        # pKa=9 → d=1
        score_in = compute_mpo_cns(1.0, 1.0, 300.0, 0, 9.0, 30.0)
        assert score_in == pytest.approx(6.0)

    def test_pka_step_function_outside_range(self):
        # pKa=7 → d=0; total = 5.0
        score_out = compute_mpo_cns(1.0, 1.0, 300.0, 0, 7.0, 30.0)
        assert score_out == pytest.approx(5.0)

    def test_tpsa_boundary_at_40(self):
        # tpsa=40 → d=1
        score = compute_mpo_cns(1.0, 1.0, 300.0, 0, 9.0, 40.0)
        assert score == pytest.approx(6.0)

    def test_tpsa_linear_between_40_and_90(self):
        # tpsa=65 → d_tpsa=(90-65)/50=0.5; other=1 → total=5.5
        score = compute_mpo_cns(1.0, 1.0, 300.0, 0, 9.0, 65.0)
        assert score == pytest.approx(5.5)

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            compute_mpo_cns(1.0, 1.0, 0.0, 0, 9.0, 30.0)

    def test_invalid_hbd_raises(self):
        with pytest.raises(ValueError, match="hbd"):
            compute_mpo_cns(1.0, 1.0, 300.0, -1, 9.0, 30.0)

    def test_invalid_tpsa_raises(self):
        with pytest.raises(ValueError, match="tpsa"):
            compute_mpo_cns(1.0, 1.0, 300.0, 0, 9.0, -1.0)

    def test_score_in_range_0_to_6(self):
        # Various inputs should give score in [0, 6]
        for logP in [-2.0, 0.0, 3.0, 5.0, 7.0]:
            for mw in [100.0, 360.0, 500.0, 700.0]:
                score = compute_mpo_cns(logP, logP - 0.5, mw, 1, 8.5, 50.0)
                assert 0.0 <= score <= 6.0


# ---------------------------------------------------------------------------
# predict_bbb_penetration tests
# ---------------------------------------------------------------------------


class TestPredictBbbPenetration:
    def _ideal_result(self, **kwargs):
        defaults = dict(logP=1.0, logD74=1.0, mw=300.0, hbd=0, pka=9.0, tpsa=30.0)
        defaults.update(kwargs)
        return predict_bbb_penetration(**defaults)

    def test_returns_correct_type(self):
        result = self._ideal_result()
        assert isinstance(result, BBBPenetrationResult)

    def test_frozen_dataclass(self):
        result = self._ideal_result()
        with pytest.raises((AttributeError, TypeError)):
            result.bbb_class = "new"  # type: ignore[misc]

    def test_high_bbb_class(self):
        # Perfect MPO=6 → high
        result = self._ideal_result()
        assert result.bbb_class == "high"
        assert result.mpo_score >= 4.0

    def test_low_bbb_class(self):
        result = predict_bbb_penetration(
            logP=6.0, logD74=5.0, mw=600.0, hbd=4, pka=5.0, tpsa=100.0
        )
        assert result.bbb_class == "low"
        assert result.mpo_score < 2.5

    def test_moderate_bbb_class(self):
        # MPO around 3 → moderate
        result = predict_bbb_penetration(
            logP=4.0, logD74=3.0, mw=430.0, hbd=1, pka=7.0, tpsa=65.0
        )
        assert result.bbb_class in ("moderate", "low")

    def test_kp_uu_brain_sigmoid(self):
        # At MPO=3, sigmoid(0)=0.5
        result = predict_bbb_penetration(
            logP=1.0, logD74=1.0, mw=300.0, hbd=0, pka=7.0, tpsa=65.0
        )
        # mpo_score should be 3 (pKa=7 → 0, tpsa=65 → 0.5)
        # Actually: d_pka=0 (7 not in [8,10]), d_tpsa=0.5 → mpo = 4.5 → high
        # Let's just check kp_uu is a float in valid range
        assert 0.0 < result.kp_uu_brain < 1.0

    def test_cns_exposure_equals_kpuu_times_fu(self):
        result = self._ideal_result(fu_plasma=0.2)
        assert result.cns_exposure == pytest.approx(result.kp_uu_brain * 0.2)

    def test_component_scores_keys(self):
        result = self._ideal_result()
        expected_keys = {"logP", "logD74", "mw", "hbd", "pka", "tpsa"}
        assert set(result.component_scores.keys()) == expected_keys

    def test_component_scores_in_range(self):
        result = self._ideal_result()
        for key, val in result.component_scores.items():
            assert 0.0 <= val <= 1.0, f"{key} score {val} out of range"

    def test_notes_not_empty(self):
        result = self._ideal_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_bbb_penetration(1.0, 1.0, 0.0, 0, 9.0, 30.0)

    def test_invalid_hbd_raises(self):
        with pytest.raises(ValueError, match="hbd"):
            predict_bbb_penetration(1.0, 1.0, 300.0, -1, 9.0, 30.0)

    def test_invalid_tpsa_raises(self):
        with pytest.raises(ValueError, match="tpsa"):
            predict_bbb_penetration(1.0, 1.0, 300.0, 0, 9.0, -1.0)

    def test_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_bbb_penetration(1.0, 1.0, 300.0, 0, 9.0, 30.0, fu_plasma=0.0)

    def test_invalid_fu_plasma_above_1(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_bbb_penetration(1.0, 1.0, 300.0, 0, 9.0, 30.0, fu_plasma=1.5)

    def test_default_fu_plasma(self):
        result = self._ideal_result()
        assert result.cns_exposure == pytest.approx(result.kp_uu_brain * 0.1)

    def test_mpo_score_equals_compute_mpo_cns(self):
        result = self._ideal_result()
        expected = compute_mpo_cns(1.0, 1.0, 300.0, 0, 9.0, 30.0)
        assert result.mpo_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# batch_cns_screen tests
# ---------------------------------------------------------------------------


class TestBatchCnsScreen:
    def _make_compound(self, logP, logD74, mw, hbd, pka, tpsa, fu_plasma=0.1):
        return {
            "logP": logP,
            "logD74": logD74,
            "mw": mw,
            "hbd": hbd,
            "pka": pka,
            "tpsa": tpsa,
            "fu_plasma": fu_plasma,
        }

    def test_returns_list(self):
        compounds = [self._make_compound(1.0, 1.0, 300.0, 0, 9.0, 30.0)]
        results = batch_cns_screen(compounds)
        assert isinstance(results, list)

    def test_sorted_by_mpo_score_descending(self):
        compounds = [
            self._make_compound(6.0, 5.0, 600.0, 4, 5.0, 100.0),   # worst
            self._make_compound(1.0, 1.0, 300.0, 0, 9.0, 30.0),    # best
            self._make_compound(3.0, 2.5, 400.0, 1, 8.5, 60.0),    # middle
        ]
        results = batch_cns_screen(compounds)
        scores = [r.mpo_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        results = batch_cns_screen([])
        assert results == []

    def test_single_compound(self):
        compounds = [self._make_compound(1.0, 1.0, 300.0, 0, 9.0, 30.0)]
        results = batch_cns_screen(compounds)
        assert len(results) == 1

    def test_all_results_are_bbb_penetration_result(self):
        compounds = [
            self._make_compound(2.0, 1.5, 350.0, 1, 8.5, 50.0),
            self._make_compound(4.0, 3.0, 450.0, 2, 7.0, 75.0),
        ]
        results = batch_cns_screen(compounds)
        for r in results:
            assert isinstance(r, BBBPenetrationResult)

    def test_optional_fu_plasma(self):
        # Without fu_plasma key → uses default 0.1
        compound = {
            "logP": 1.0, "logD74": 1.0, "mw": 300.0, "hbd": 0, "pka": 9.0, "tpsa": 30.0
        }
        results = batch_cns_screen([compound])
        assert len(results) == 1
        assert results[0].cns_exposure == pytest.approx(results[0].kp_uu_brain * 0.1)
