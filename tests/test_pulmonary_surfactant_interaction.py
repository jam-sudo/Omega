"""Tests for pulmonary surfactant interaction prediction (Phase 880)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.pulmonary_surfactant_interaction import (
    SurfactantInteractionResult,
    predict_surfactant_interaction,
    screen_inhaled_drugs,
)

# ---------------------------------------------------------------------------
# Basic result structure tests
# ---------------------------------------------------------------------------


class TestSurfactantInteractionResultStructure:
    def test_returns_correct_type(self):
        result = predict_surfactant_interaction("TestDrug", logp=2.0)
        assert isinstance(result, SurfactantInteractionResult)

    def test_drug_name_preserved(self):
        result = predict_surfactant_interaction("Budesonide", logp=3.2)
        assert result.drug_name == "Budesonide"

    def test_logp_preserved(self):
        result = predict_surfactant_interaction("Drug", logp=1.5)
        assert result.logp == pytest.approx(1.5)

    def test_mw_preserved(self):
        result = predict_surfactant_interaction("Drug", logp=2.0, mw_Da=430.0)
        assert result.mw_Da == pytest.approx(430.0)

    def test_pka_none_when_not_provided(self):
        result = predict_surfactant_interaction("Drug", logp=2.0)
        assert result.pka is None

    def test_pka_preserved_when_provided(self):
        result = predict_surfactant_interaction("Drug", logp=2.0, pka=7.4)
        assert result.pka == pytest.approx(7.4)

    def test_frozen_dataclass(self):
        result = predict_surfactant_interaction("Drug", logp=2.0)
        with pytest.raises(Exception):
            result.logp = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Surfactant binding fraction tests
# ---------------------------------------------------------------------------


class TestSurfactantBindingFraction:
    def test_binding_at_logp_2_is_half(self):
        """Sigmoid centered at logP=2: binding fraction should be ~0.5."""
        result = predict_surfactant_interaction("Drug", logp=2.0)
        assert result.surfactant_binding_fraction == pytest.approx(0.5, abs=1e-6)

    def test_high_logp_high_binding(self):
        """High logP drugs bind strongly to surfactant."""
        result = predict_surfactant_interaction("LipophilicDrug", logp=6.0)
        assert result.surfactant_binding_fraction > 0.9

    def test_low_logp_low_binding(self):
        """Low logP drugs have minimal surfactant binding."""
        result = predict_surfactant_interaction("HydrophilicDrug", logp=-2.0)
        assert result.surfactant_binding_fraction < 0.1

    def test_binding_fraction_between_0_and_1(self):
        for logp in [-3.0, 0.0, 2.0, 5.0, 8.0]:
            result = predict_surfactant_interaction("Drug", logp=logp)
            assert 0.0 <= result.surfactant_binding_fraction <= 1.0

    def test_effective_free_fraction_complements_binding(self):
        result = predict_surfactant_interaction("Drug", logp=3.0)
        assert result.effective_free_fraction == pytest.approx(
            1.0 - result.surfactant_binding_fraction, rel=1e-9
        )

    def test_binding_monotonically_increases_with_logp(self):
        logps = [-2.0, 0.0, 2.0, 4.0, 6.0]
        bindings = [
            predict_surfactant_interaction("Drug", logp=lp).surfactant_binding_fraction
            for lp in logps
        ]
        for i in range(len(bindings) - 1):
            assert bindings[i] < bindings[i + 1]


# ---------------------------------------------------------------------------
# Derived property tests
# ---------------------------------------------------------------------------


class TestDerivedProperties:
    def test_dissolution_rate_factor_ge_1(self):
        """Surfactant aids dissolution: factor should be >= 1."""
        for logp in [-5.0, 0.0, 2.0, 5.0]:
            result = predict_surfactant_interaction("Drug", logp=logp)
            assert result.dissolution_rate_factor >= 1.0

    def test_dissolution_rate_factor_max_at_high_logp(self):
        r_low = predict_surfactant_interaction("Drug", logp=-3.0)
        r_high = predict_surfactant_interaction("Drug", logp=8.0)
        assert r_high.dissolution_rate_factor > r_low.dissolution_rate_factor

    def test_absorption_rate_modifier_equals_free_fraction(self):
        result = predict_surfactant_interaction("Drug", logp=3.0)
        assert result.absorption_rate_modifier == pytest.approx(
            result.effective_free_fraction, rel=1e-9
        )

    def test_lung_t_half_increases_with_binding(self):
        """Higher binding -> lower absorption rate -> longer half-life."""
        r_low = predict_surfactant_interaction("Drug", logp=-2.0)
        r_high = predict_surfactant_interaction("Drug", logp=6.0)
        assert r_high.predicted_lung_t_half_h > r_low.predicted_lung_t_half_h

    def test_lung_t_half_clamped_to_72h(self):
        """Extremely high logP should clamp t_half at 72h."""
        result = predict_surfactant_interaction("Drug", logp=20.0)
        assert result.predicted_lung_t_half_h <= 72.0

    def test_lung_t_half_clamped_min_0_1h(self):
        result = predict_surfactant_interaction("Drug", logp=-10.0)
        assert result.predicted_lung_t_half_h >= 0.1

    def test_retention_score_between_0_and_100(self):
        for logp in [-5.0, 0.0, 2.0, 5.0]:
            result = predict_surfactant_interaction("Drug", logp=logp)
            assert 0.0 <= result.retention_score <= 100.0

    def test_retention_score_high_for_large_lipophilic(self):
        result = predict_surfactant_interaction("Drug", logp=5.0, mw_Da=600.0)
        assert result.retention_score > 60.0


# ---------------------------------------------------------------------------
# Notes tests
# ---------------------------------------------------------------------------


class TestNotes:
    def test_high_retention_note(self):
        """logP=5, large MW should give high retention note."""
        result = predict_surfactant_interaction("Drug", logp=5.0, mw_Da=600.0)
        assert "High lung retention" in result.notes

    def test_low_retention_note(self):
        """Very hydrophilic, small MW drug should give low retention note."""
        result = predict_surfactant_interaction("Drug", logp=-3.0, mw_Da=100.0)
        assert "Low lung retention" in result.notes

    def test_notes_is_nonempty_string(self):
        result = predict_surfactant_interaction("Drug", logp=2.0)
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_surfactant_interaction("Drug", logp=2.0, mw_Da=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_surfactant_interaction("Drug", logp=2.0, mw_Da=-100.0)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_surfactant_interaction("Drug", logp=2.0, ionization_type="zwitterion")

    def test_valid_ionization_types_accepted(self):
        for ion_type in ["acid", "base", "neutral"]:
            result = predict_surfactant_interaction("Drug", logp=2.0, ionization_type=ion_type)
            assert isinstance(result, SurfactantInteractionResult)


# ---------------------------------------------------------------------------
# screen_inhaled_drugs tests
# ---------------------------------------------------------------------------


class TestScreenInhaledDrugs:
    def test_returns_list(self):
        candidates = [
            {"name": "DrugA", "logp": 1.0},
            {"name": "DrugB", "logp": 4.0},
        ]
        results = screen_inhaled_drugs(candidates)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_by_retention_score_descending(self):
        candidates = [
            {"name": "LowLogP", "logp": -2.0, "mw_Da": 150.0},
            {"name": "HighLogP", "logp": 5.0, "mw_Da": 500.0},
            {"name": "MidLogP", "logp": 2.0, "mw_Da": 300.0},
        ]
        results = screen_inhaled_drugs(candidates)
        scores = [r.retention_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_optional_fields_have_defaults(self):
        candidates = [{"name": "MinimalDrug", "logp": 2.0}]
        results = screen_inhaled_drugs(candidates)
        assert len(results) == 1
        assert results[0].mw_Da == pytest.approx(300.0)
        assert results[0].pka is None

    def test_empty_candidates_returns_empty(self):
        results = screen_inhaled_drugs([])
        assert results == []

    def test_optional_fields_used_when_provided(self):
        candidates = [
            {"name": "Drug", "logp": 3.0, "pka": 6.5, "mw_Da": 450.0, "ionization_type": "acid"}
        ]
        results = screen_inhaled_drugs(candidates)
        assert results[0].pka == pytest.approx(6.5)
        assert results[0].mw_Da == pytest.approx(450.0)
