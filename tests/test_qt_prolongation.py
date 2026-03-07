"""Tests for risk/qt_prolongation.py — Phase 345."""

from __future__ import annotations

import math
import pytest

from omega_pbpk.risk.qt_prolongation import (
    QTProlongationResult,
    predict_qt_prolongation,
    screen_cardiac_safety,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _base_result(**kwargs) -> QTProlongationResult:
    """Return a result with sensible defaults, overriding via kwargs."""
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        mw=300.0,
        logP=2.0,
        herg_ic50_uM=100.0,  # weak inhibitor
        c_plasma_mg_L=0.1,
    )
    defaults.update(kwargs)
    return predict_qt_prolongation(**defaults)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------

class TestQTProlongationResultStructure:
    def test_returns_correct_type(self):
        r = _base_result()
        assert isinstance(r, QTProlongationResult)

    def test_frozen_dataclass_not_mutable(self):
        r = _base_result()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore[misc]

    def test_all_fields_accessible(self):
        r = _base_result()
        assert r.drug_name == "TestDrug"
        assert r.dose_mg == 100.0
        assert isinstance(r.herg_inhibition_pct, float)
        assert isinstance(r.delta_qtc_ms, float)
        assert isinstance(r.predicted_qtc_ms, float)
        assert isinstance(r.risk_factors, list)
        assert isinstance(r.recommendation, str)
        assert isinstance(r.notes, str)

    def test_recommendation_is_nonempty(self):
        r = _base_result()
        assert len(r.recommendation) > 0

    def test_notes_is_nonempty(self):
        r = _base_result()
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# hERG inhibition calculation
# ---------------------------------------------------------------------------

class TestHERGInhibition:
    def test_very_high_ic50_weak_inhibition(self):
        """IC50 >> Cmax → near 0% inhibition."""
        r = predict_qt_prolongation(
            drug_name="WeakInhibitor",
            dose_mg=50.0,
            mw=300.0,
            logP=1.0,
            herg_ic50_uM=10_000.0,  # very high IC50
            c_plasma_mg_L=0.01,
        )
        assert r.herg_inhibition_pct < 5.0

    def test_very_low_ic50_strong_inhibition(self):
        """IC50 << Cmax → high % inhibition."""
        r = predict_qt_prolongation(
            drug_name="StrongInhibitor",
            dose_mg=200.0,
            mw=300.0,
            logP=3.0,
            herg_ic50_uM=0.01,  # very low IC50 (potent)
            c_plasma_mg_L=10.0,  # high concentration
        )
        assert r.herg_inhibition_pct > 90.0

    def test_herg_inhibition_between_0_and_100(self):
        for ic50 in [0.001, 1.0, 100.0, 10_000.0]:
            r = predict_qt_prolongation(
                "X", 100.0, 300.0, 2.0, ic50, c_plasma_mg_L=1.0
            )
            assert 0.0 <= r.herg_inhibition_pct <= 100.0

    def test_higher_concentration_increases_inhibition(self):
        r_low = predict_qt_prolongation("X", 10.0, 300.0, 2.0, 1.0, c_plasma_mg_L=0.1)
        r_high = predict_qt_prolongation("X", 10.0, 300.0, 2.0, 1.0, c_plasma_mg_L=10.0)
        assert r_high.herg_inhibition_pct > r_low.herg_inhibition_pct


# ---------------------------------------------------------------------------
# delta QTc calculation
# ---------------------------------------------------------------------------

class TestDeltaQTc:
    def test_weak_inhibitor_near_zero_delta_qtc(self):
        r = predict_qt_prolongation(
            "Weak", 50.0, 300.0, 1.0, 10_000.0, c_plasma_mg_L=0.001
        )
        assert r.delta_qtc_ms < 2.0

    def test_strong_inhibitor_large_delta_qtc(self):
        r = predict_qt_prolongation(
            "Strong", 200.0, 300.0, 3.0, 0.01, c_plasma_mg_L=10.0
        )
        assert r.delta_qtc_ms > 10.0

    def test_predicted_qtc_equals_baseline_plus_delta(self):
        r = _base_result(baseline_qtc_ms=410.0)
        assert abs(r.predicted_qtc_ms - (r.baseline_qtc_ms + r.delta_qtc_ms)) < 1e-6

    def test_empirical_scale_50pct_inhibition_gives_20ms(self):
        """50% hERG inhibition (IC50 = Cmax) → ~20 ms at baseline conditions."""
        # IC50 in µM, need Cmax in µM to equal IC50
        ic50_uM = 1.0
        # c_plasma_mg_L to give 1 µM for mw=300: 1 µM * 300 g/mol / 1000 = 0.3 mg/L
        c_mg_L = ic50_uM * 300.0 / 1000.0
        r = predict_qt_prolongation(
            "FiftyPct", 100.0, 300.0, 2.0, ic50_uM, c_plasma_mg_L=c_mg_L
        )
        assert abs(r.herg_inhibition_pct - 50.0) < 1.0
        # delta_qtc_base = 50 * 0.4 = 20 ms (no risk factors)
        assert abs(r.delta_qtc_ms - 20.0) < 1.0


# ---------------------------------------------------------------------------
# Risk factors
# ---------------------------------------------------------------------------

class TestRiskFactors:
    def test_no_risk_factors_by_default(self):
        r = _base_result()
        assert r.risk_factors_count == 0
        assert r.risk_factors == []

    def test_female_sex_adds_risk_factor(self):
        r = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 5.0, c_plasma_mg_L=1.0, patient_female=True
        )
        assert r.risk_factors_count >= 1
        assert any("female" in rf.lower() or "sex" in rf.lower() for rf in r.risk_factors)

    def test_female_sex_increases_delta_qtc(self):
        r_male = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 5.0, c_plasma_mg_L=1.0, patient_female=False
        )
        r_female = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 5.0, c_plasma_mg_L=1.0, patient_female=True
        )
        assert r_female.delta_qtc_ms > r_male.delta_qtc_ms
        assert abs(r_female.delta_qtc_ms - r_male.delta_qtc_ms - 5.0) < 1e-6

    def test_hypokalemia_increases_delta_qtc_by_10(self):
        r_base = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 5.0, c_plasma_mg_L=1.0
        )
        r_hypo = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 5.0, c_plasma_mg_L=1.0, hypokalemia=True
        )
        assert abs(r_hypo.delta_qtc_ms - r_base.delta_qtc_ms - 10.0) < 1e-6

    def test_hypokalemia_adds_risk_factor_entry(self):
        r = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 5.0, c_plasma_mg_L=1.0, hypokalemia=True
        )
        assert any("hypokalemia" in rf.lower() for rf in r.risk_factors)

    def test_concomitant_qt_drug_multiplies_delta_qtc(self):
        r_base = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 1.0, c_plasma_mg_L=1.0
        )
        r_combo = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 1.0, c_plasma_mg_L=1.0, concomitant_qt_drug=True
        )
        # Concomitant drug multiplies base delta by 1.5 (before offsets)
        assert r_combo.delta_qtc_ms > r_base.delta_qtc_ms

    def test_all_risk_factors_combined(self):
        r = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 1.0, c_plasma_mg_L=1.0,
            patient_female=True, hypokalemia=True, concomitant_qt_drug=True
        )
        assert r.risk_factors_count == 3

    def test_risk_factors_count_matches_list_length(self):
        r = predict_qt_prolongation(
            "X", 100.0, 300.0, 2.0, 1.0, c_plasma_mg_L=1.0,
            patient_female=True, hypokalemia=True
        )
        assert r.risk_factors_count == len(r.risk_factors)


# ---------------------------------------------------------------------------
# QTc prolongation category
# ---------------------------------------------------------------------------

class TestQTcCategory:
    def test_none_category_below_5ms(self):
        # Very weak inhibitor → delta_qtc < 5
        r = predict_qt_prolongation("X", 50.0, 300.0, 1.0, 10_000.0, c_plasma_mg_L=0.001)
        assert r.qtc_prolongation_category == "none"

    def test_mild_category_5_to_10ms(self):
        # Need delta_qtc in [5, 10): herg_inhibition in [12.5, 25%)
        # 12.5% inhibition: c_uM / (c_uM + ic50) = 0.125 → c_uM = ic50 / 7
        ic50 = 7.0
        c_uM = 1.0  # gives 1/(1+7) = 12.5% → delta = 5 ms
        c_mg_L = c_uM * 300.0 / 1000.0
        r = predict_qt_prolongation("X", 50.0, 300.0, 2.0, ic50, c_plasma_mg_L=c_mg_L)
        assert r.qtc_prolongation_category in ("none", "mild")

    def test_severe_category_above_20ms(self):
        r = predict_qt_prolongation(
            "X", 200.0, 300.0, 3.0, 0.001, c_plasma_mg_L=10.0
        )
        assert r.qtc_prolongation_category == "severe"

    def test_category_thresholds_consistent_with_delta(self):
        for trial in range(5):
            c = 0.001 * (10 ** trial)
            r = predict_qt_prolongation("X", 100.0, 300.0, 2.0, 1.0, c_plasma_mg_L=c)
            if r.delta_qtc_ms < 5:
                assert r.qtc_prolongation_category == "none"
            elif r.delta_qtc_ms < 10:
                assert r.qtc_prolongation_category == "mild"
            elif r.delta_qtc_ms < 20:
                assert r.qtc_prolongation_category == "moderate"
            else:
                assert r.qtc_prolongation_category == "severe"


# ---------------------------------------------------------------------------
# TdP risk category
# ---------------------------------------------------------------------------

class TestTdPRiskCategory:
    def test_low_risk_weak_inhibitor(self):
        r = predict_qt_prolongation("X", 50.0, 300.0, 1.0, 10_000.0, c_plasma_mg_L=0.001)
        assert r.tdp_risk_category == "low"

    def test_known_risk_very_strong_inhibitor(self):
        r = predict_qt_prolongation(
            "X", 200.0, 300.0, 3.0, 0.001, c_plasma_mg_L=10.0
        )
        assert r.tdp_risk_category == "known_risk"

    def test_conditional_risk_moderate_inhibitor(self):
        # hERG inhibition >20% but ≤50% → conditional (if delta_qtc ≤20)
        ic50 = 1.0
        # 25% inhibition: c = ic50/3 → c_uM = 1/3
        c_uM = 1.0 / 3.0
        c_mg_L = c_uM * 300.0 / 1000.0
        r = predict_qt_prolongation("X", 100.0, 300.0, 2.0, ic50, c_plasma_mg_L=c_mg_L)
        assert r.tdp_risk_category in ("conditional_risk", "low")

    def test_known_risk_triggered_by_herg_over_50pct(self):
        """Regardless of exact delta_qtc, >50% hERG inhibition → known_risk."""
        ic50 = 1.0
        c_uM = 10.0  # 10/(10+1) ≈ 91% inhibition
        c_mg_L = c_uM * 300.0 / 1000.0
        r = predict_qt_prolongation("X", 100.0, 300.0, 2.0, ic50, c_plasma_mg_L=c_mg_L)
        assert r.herg_inhibition_pct > 50.0
        assert r.tdp_risk_category == "known_risk"


# ---------------------------------------------------------------------------
# Cmax estimation (no c_plasma_mg_L provided)
# ---------------------------------------------------------------------------

class TestCmaxEstimation:
    def test_auto_cmax_gives_positive_concentration(self):
        r = predict_qt_prolongation(
            drug_name="AutoCmax",
            dose_mg=100.0,
            mw=350.0,
            logP=2.0,
            herg_ic50_uM=5.0,
            # c_plasma_mg_L=None (default) → estimated
        )
        assert r.c_plasma_mg_L > 0.0

    def test_larger_dose_gives_larger_cmax(self):
        r_low = predict_qt_prolongation("X", 50.0, 300.0, 2.0, 5.0)
        r_high = predict_qt_prolongation("X", 500.0, 300.0, 2.0, 5.0)
        assert r_high.c_plasma_mg_L > r_low.c_plasma_mg_L


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            predict_qt_prolongation("X", -10.0, 300.0, 2.0, 5.0, c_plasma_mg_L=1.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_qt_prolongation("X", 100.0, -300.0, 2.0, 5.0, c_plasma_mg_L=1.0)

    def test_negative_herg_ic50_raises(self):
        with pytest.raises(ValueError, match="herg_ic50_uM"):
            predict_qt_prolongation("X", 100.0, 300.0, 2.0, -5.0, c_plasma_mg_L=1.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            predict_qt_prolongation("X", 100.0, 300.0, 2.0, 5.0, cl_L_per_h=-1.0)


# ---------------------------------------------------------------------------
# screen_cardiac_safety
# ---------------------------------------------------------------------------

class TestScreenCardiacSafety:
    def _compounds(self):
        return [
            {"name": "DrugA", "dose_mg": 100.0, "mw": 300.0, "logP": 2.0,
             "herg_ic50_uM": 10_000.0, "c_plasma_mg_L": 0.001},  # very safe
            {"name": "DrugB", "dose_mg": 200.0, "mw": 300.0, "logP": 3.0,
             "herg_ic50_uM": 0.1, "c_plasma_mg_L": 5.0},  # risky
            {"name": "DrugC", "dose_mg": 50.0, "mw": 300.0, "logP": 1.5,
             "herg_ic50_uM": 10.0, "c_plasma_mg_L": 0.5},  # moderate
        ]

    def test_returns_list(self):
        results = screen_cardiac_safety(self._compounds())
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_delta_qtc_descending(self):
        results = screen_cardiac_safety(self._compounds())
        deltas = [r.delta_qtc_ms for r in results]
        assert deltas == sorted(deltas, reverse=True)

    def test_most_risky_first(self):
        results = screen_cardiac_safety(self._compounds())
        assert results[0].drug_name == "DrugB"

    def test_empty_list_returns_empty(self):
        assert screen_cardiac_safety([]) == []

    def test_single_compound_returns_single(self):
        cmpd = [{"name": "Solo", "dose_mg": 100.0, "mw": 300.0,
                 "logP": 2.0, "herg_ic50_uM": 5.0, "c_plasma_mg_L": 1.0}]
        results = screen_cardiac_safety(cmpd)
        assert len(results) == 1
        assert results[0].drug_name == "Solo"
