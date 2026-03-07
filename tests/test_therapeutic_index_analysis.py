"""Tests for therapeutic index and safety margin analysis (Phase 432)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.therapeutic_index_analysis import (
    TherapeuticIndexResult,
    TIPopulationResult,
    compute_therapeutic_index,
    simulate_ti_population,
)

# ---------------------------------------------------------------------------
# Default test parameters
# ---------------------------------------------------------------------------
_DRUG = "warfarin"
_ED50 = 5.0   # mg
_TD50 = 15.0  # mg
_ED50_AUC = 10.0  # mg*h/L
_TD50_AUC = 30.0  # mg*h/L
_RANGE = (1.0, 5.0)  # (MEC, MTC) in mg/L


# ---------------------------------------------------------------------------
# 1. TherapeuticIndexResult — compute_therapeutic_index
# ---------------------------------------------------------------------------


class TestComputeTIInputValidation:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            compute_therapeutic_index("", _ED50, _TD50, _ED50_AUC, _TD50_AUC, _RANGE)

    def test_ed50_zero(self):
        with pytest.raises(ValueError, match="ed50_mg"):
            compute_therapeutic_index(_DRUG, 0.0, _TD50, _ED50_AUC, _TD50_AUC, _RANGE)

    def test_td50_zero(self):
        with pytest.raises(ValueError, match="td50_mg"):
            compute_therapeutic_index(_DRUG, _ED50, 0.0, _ED50_AUC, _TD50_AUC, _RANGE)

    def test_td50_le_ed50(self):
        with pytest.raises(ValueError, match="td50_mg"):
            compute_therapeutic_index(_DRUG, 10.0, 10.0, _ED50_AUC, _TD50_AUC, _RANGE)

    def test_ed50_auc_zero(self):
        with pytest.raises(ValueError, match="ed50_auc"):
            compute_therapeutic_index(_DRUG, _ED50, _TD50, 0.0, _TD50_AUC, _RANGE)

    def test_td50_auc_le_ed50_auc(self):
        with pytest.raises(ValueError, match="td50_auc"):
            compute_therapeutic_index(_DRUG, _ED50, _TD50, 20.0, 10.0, _RANGE)

    def test_invalid_range_length(self):
        with pytest.raises(ValueError, match="therapeutic_range"):
            compute_therapeutic_index(_DRUG, _ED50, _TD50, _ED50_AUC, _TD50_AUC, (1.0,))

    def test_mec_zero(self):
        with pytest.raises(ValueError, match="MEC"):
            compute_therapeutic_index(
                _DRUG, _ED50, _TD50, _ED50_AUC, _TD50_AUC, (0.0, 5.0)
            )

    def test_mtc_le_mec(self):
        with pytest.raises(ValueError, match="MTC"):
            compute_therapeutic_index(
                _DRUG, _ED50, _TD50, _ED50_AUC, _TD50_AUC, (5.0, 3.0)
            )


class TestComputeTIResults:
    def setup_method(self):
        self.result = compute_therapeutic_index(
            _DRUG, _ED50, _TD50, _ED50_AUC, _TD50_AUC, _RANGE
        )

    def test_returns_correct_type(self):
        assert isinstance(self.result, TherapeuticIndexResult)

    def test_drug_name_stored(self):
        assert self.result.drug_name == _DRUG

    def test_ti_dose_calculation(self):
        expected = _TD50 / _ED50
        assert self.result.ti_dose == pytest.approx(expected, rel=1e-9)

    def test_ti_auc_calculation(self):
        expected = _TD50_AUC / _ED50_AUC
        assert self.result.ti_auc == pytest.approx(expected, rel=1e-9)

    def test_safety_margin_calculation(self):
        mec, mtc = _RANGE
        expected = (mtc - mec) / mec * 100.0
        assert self.result.safety_margin_pct == pytest.approx(expected, rel=1e-9)

    def test_is_narrow_ti_false_for_wide(self):
        # TI_dose = 3.0, TI_AUC = 3.0 — not narrow
        assert not self.result.is_narrow_ti

    def test_is_narrow_ti_true_for_narrow(self):
        result = compute_therapeutic_index(
            _DRUG, 10.0, 15.0, 10.0, 15.0, _RANGE
        )
        # TI_dose = 1.5 < 2.0 → narrow
        assert result.is_narrow_ti

    def test_monitoring_recommendation_non_empty(self):
        assert len(self.result.monitoring_recommendation) > 0

    def test_notes_non_empty(self):
        assert len(self.result.notes) > 0

    def test_narrow_ti_monitoring_mentions_tdm(self):
        result = compute_therapeutic_index(
            _DRUG, 10.0, 15.0, 10.0, 15.0, _RANGE
        )
        rec = result.monitoring_recommendation.lower()
        assert "tdm" in rec or "monitoring" in rec

    def test_wide_ti_monitoring_standard(self):
        # TI_dose = 3.0 → moderate → routine
        assert "monitoring" in self.result.monitoring_recommendation.lower()

    def test_result_is_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            self.result.ti_dose = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. TIPopulationResult — simulate_ti_population
# ---------------------------------------------------------------------------


class TestSimulateTIInputValidation:
    def test_empty_drug_name(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_ti_population("", _ED50, _TD50)

    def test_ed50_zero(self):
        with pytest.raises(ValueError, match="ed50_mg"):
            simulate_ti_population(_DRUG, 0.0, _TD50)

    def test_td50_le_ed50(self):
        with pytest.raises(ValueError, match="td50_mg"):
            simulate_ti_population(_DRUG, 10.0, 5.0)

    def test_n_subjects_zero(self):
        with pytest.raises(ValueError, match="n_subjects"):
            simulate_ti_population(_DRUG, _ED50, _TD50, n_subjects=0)

    def test_ed50_cv_negative(self):
        with pytest.raises(ValueError, match="ed50_cv_pct"):
            simulate_ti_population(_DRUG, _ED50, _TD50, ed50_cv_pct=-5.0)

    def test_td50_cv_negative(self):
        with pytest.raises(ValueError, match="td50_cv_pct"):
            simulate_ti_population(_DRUG, _ED50, _TD50, td50_cv_pct=-1.0)

    def test_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_ti_population(_DRUG, _ED50, _TD50, dose_mg=0.0)


class TestSimulateTIResults:
    def setup_method(self):
        self.result = simulate_ti_population(
            _DRUG, _ED50, _TD50, n_subjects=500, seed=123
        )

    def test_returns_correct_type(self):
        assert isinstance(self.result, TIPopulationResult)

    def test_drug_name_stored(self):
        assert self.result.drug_name == _DRUG

    def test_n_subjects_stored(self):
        assert self.result.n_subjects == 500

    def test_individual_ed50_correct_length(self):
        assert len(self.result.individual_ed50) == 500

    def test_individual_td50_correct_length(self):
        assert len(self.result.individual_td50) == 500

    def test_ed50_samples_positive(self):
        for v in self.result.individual_ed50:
            assert v > 0.0

    def test_td50_samples_positive(self):
        for v in self.result.individual_td50:
            assert v > 0.0

    def test_pct_effective_in_unit_interval(self):
        assert 0.0 <= self.result.pct_effective <= 1.0

    def test_pct_toxic_in_unit_interval(self):
        assert 0.0 <= self.result.pct_toxic <= 1.0

    def test_therapeutic_window_is_difference(self):
        expected = self.result.pct_effective - self.result.pct_toxic
        assert self.result.therapeutic_window_pct == pytest.approx(expected, rel=1e-9)

    def test_notes_is_list(self):
        assert isinstance(self.result.notes, list)

    def test_reproducibility(self):
        r1 = simulate_ti_population(_DRUG, _ED50, _TD50, n_subjects=200, seed=42)
        r2 = simulate_ti_population(_DRUG, _ED50, _TD50, n_subjects=200, seed=42)
        assert r1.pct_effective == pytest.approx(r2.pct_effective)
        assert r1.pct_toxic == pytest.approx(r2.pct_toxic)

    def test_different_seeds_may_differ(self):
        r1 = simulate_ti_population(_DRUG, _ED50, _TD50, n_subjects=200, seed=1)
        r2 = simulate_ti_population(_DRUG, _ED50, _TD50, n_subjects=200, seed=999)
        # With different seeds, individual values should differ (probabilistic check)
        assert r1.individual_ed50 != r2.individual_ed50

    def test_zero_cv_deterministic(self):
        r = simulate_ti_population(
            _DRUG, _ED50, _TD50, n_subjects=100,
            ed50_cv_pct=0.0, td50_cv_pct=0.0, seed=0
        )
        # All ED50 values should equal the mean
        for v in r.individual_ed50:
            assert v == pytest.approx(_ED50, rel=1e-9)

    def test_default_dose_is_geometric_mean(self):
        """Default dose should be sqrt(ED50 * TD50)."""
        r = simulate_ti_population(
            _DRUG, _ED50, _TD50, n_subjects=100, seed=0
        )
        expected_dose = math.sqrt(_ED50 * _TD50)
        assert r.dose_mg == pytest.approx(expected_dose, rel=1e-9)

    def test_high_dose_high_toxicity_note(self):
        """At a very high dose, most subjects should be toxic."""
        r = simulate_ti_population(
            _DRUG, _ED50, _TD50, n_subjects=1000,
            ed50_cv_pct=10.0, td50_cv_pct=10.0,
            dose_mg=_TD50 * 5,  # way above TD50
            seed=42,
        )
        assert r.pct_toxic > 0.5

    def test_low_dose_low_effectiveness(self):
        """At a very low dose, few subjects should respond."""
        r = simulate_ti_population(
            _DRUG, _ED50, _TD50, n_subjects=1000,
            ed50_cv_pct=10.0, td50_cv_pct=10.0,
            dose_mg=0.01,  # far below ED50
            seed=42,
        )
        assert r.pct_effective < 0.1
