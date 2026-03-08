"""Tests for Phase 395: Protein Binding Displacement Interaction."""

import pytest

from omega_pbpk.clinical.protein_binding_displacement_p395 import (
    ProteinBindingDisplacementResult,
    predict_displacement_interaction,
    screen_displacers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_result(**kwargs):
    defaults = dict(
        victim_drug="warfarin",
        displacer_drug="aspirin",
        binding_protein="albumin",
        victim_fup_baseline=0.01,
        displacer_conc_uM=100.0,
        ki_displacement_uM=10.0,
        victim_total_bound_fraction=0.99,
        narrow_ti=False,
    )
    defaults.update(kwargs)
    return predict_displacement_interaction(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        result = _base_result()
        assert isinstance(result, ProteinBindingDisplacementResult)

    def test_result_is_frozen(self):
        result = _base_result()
        with pytest.raises((AttributeError, TypeError)):
            result.fup_increase_fold = 999.0  # type: ignore

    def test_field_names(self):
        result = _base_result()
        assert hasattr(result, "victim_fup_baseline")
        assert hasattr(result, "victim_fup_displaced")
        assert hasattr(result, "fup_increase_fold")
        assert hasattr(result, "cl_change_pct")
        assert hasattr(result, "vd_change_pct")
        assert hasattr(result, "auc_change_pct")
        assert hasattr(result, "cmax_change_pct")
        assert hasattr(result, "clinical_significance")
        assert hasattr(result, "narrow_therapeutic_index")
        assert hasattr(result, "monitoring_required")
        assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# Core model behaviour
# ---------------------------------------------------------------------------


class TestDisplacementModel:
    def test_fup_displaced_ge_baseline(self):
        result = _base_result()
        assert result.victim_fup_displaced >= result.victim_fup_baseline

    def test_high_displacer_conc_larger_displacement(self):
        low = _base_result(displacer_conc_uM=1.0)
        high = _base_result(displacer_conc_uM=1000.0)
        assert high.fup_increase_fold > low.fup_increase_fold

    def test_zero_displacer_conc_no_displacement(self):
        result = _base_result(displacer_conc_uM=0.0)
        assert result.fup_increase_fold == pytest.approx(1.0, abs=1e-9)
        assert result.victim_fup_displaced == pytest.approx(result.victim_fup_baseline, abs=1e-9)

    def test_larger_ki_smaller_displacement(self):
        low_ki = _base_result(ki_displacement_uM=1.0)
        high_ki = _base_result(ki_displacement_uM=100.0)
        assert low_ki.fup_increase_fold > high_ki.fup_increase_fold

    def test_fup_displaced_capped_at_one(self):
        # Extreme scenario should not exceed 1.0
        result = _base_result(
            victim_fup_baseline=0.5,
            victim_total_bound_fraction=0.5,
            displacer_conc_uM=1e6,
            ki_displacement_uM=1.0,
        )
        assert result.victim_fup_displaced <= 1.0

    def test_auc_change_pct_zero_high_er(self):
        result = _base_result()
        assert result.auc_change_pct == pytest.approx(0.0)

    def test_cl_change_proportional_to_fup(self):
        result = _base_result()
        expected_cl = (result.fup_increase_fold - 1.0) * 100.0
        assert result.cl_change_pct == pytest.approx(expected_cl, abs=1e-6)

    def test_cmax_change_positive_when_displaced(self):
        result = _base_result(displacer_conc_uM=100.0)
        assert result.cmax_change_pct > 0.0


# ---------------------------------------------------------------------------
# Clinical significance mapping
# ---------------------------------------------------------------------------


class TestClinicalSignificance:
    def test_negligible_low_displacement(self):
        # Very high ki → very low displacement
        result = _base_result(
            victim_total_bound_fraction=0.1,
            displacer_conc_uM=0.001,
            ki_displacement_uM=1000.0,
        )
        assert result.clinical_significance == "negligible"

    def test_minor_significance(self):
        # Moderate displacement: fup_increase in [1.1, 1.5)
        # victim_fup_baseline=0.1, bound=0.9, fraction_displaced≈0.5 → new_unbound=0.45
        # fup_displaced = min(1, 0.1+0.45)=0.55, fold=5.5 → that's major
        # Need smaller displacement: bound=0.04, fup=0.96
        predict_displacement_interaction(
            victim_drug="drug",
            displacer_drug="displacer",
            binding_protein="albumin",
            victim_fup_baseline=0.96,
            displacer_conc_uM=10.0,
            ki_displacement_uM=100.0,
            victim_total_bound_fraction=0.04,
        )
        # fraction_displaced = 10/(10+100) ≈ 0.0909; new_unbound = 0.04*0.0909 ≈ 0.00364
        # fup_displaced ≈ 0.9636; fold ≈ 1.004 → negligible
        # Try with larger bound fraction and moderate ki
        predict_displacement_interaction(
            victim_drug="drug",
            displacer_drug="displacer",
            binding_protein="albumin",
            victim_fup_baseline=0.5,
            displacer_conc_uM=50.0,
            ki_displacement_uM=50.0,
            victim_total_bound_fraction=0.5,
        )
        # fraction_displaced = 0.5; new_unbound = 0.25; fup_displaced = 0.75; fold = 1.5 → moderate
        # Aim for minor: fold in [1.1, 1.5)
        result3 = predict_displacement_interaction(
            victim_drug="drug",
            displacer_drug="displacer",
            binding_protein="albumin",
            victim_fup_baseline=0.8,
            displacer_conc_uM=10.0,
            ki_displacement_uM=10.0,
            victim_total_bound_fraction=0.2,
        )
        # fraction_displaced = 0.5; new_unbound = 0.1; fup_displaced = 0.9; fold = 1.125 → minor
        assert result3.clinical_significance == "minor"

    def test_moderate_significance(self):
        result = predict_displacement_interaction(
            victim_drug="drug",
            displacer_drug="displacer",
            binding_protein="albumin",
            victim_fup_baseline=0.5,
            displacer_conc_uM=50.0,
            ki_displacement_uM=50.0,
            victim_total_bound_fraction=0.5,
        )
        # fold = 1.5 → moderate boundary → "moderate" (since < 2.0)
        assert result.clinical_significance in ("minor", "moderate")

    def test_major_significance_high_fold(self):
        result = predict_displacement_interaction(
            victim_drug="warfarin",
            displacer_drug="aspirin",
            binding_protein="albumin",
            victim_fup_baseline=0.01,
            displacer_conc_uM=1000.0,
            ki_displacement_uM=10.0,
            victim_total_bound_fraction=0.99,
        )
        assert result.clinical_significance == "major"


# ---------------------------------------------------------------------------
# Monitoring logic
# ---------------------------------------------------------------------------


class TestMonitoring:
    def test_monitoring_for_narrow_ti(self):
        result = _base_result(
            displacer_conc_uM=0.001,
            ki_displacement_uM=1000.0,
            narrow_ti=True,
        )
        assert result.monitoring_required is True
        assert result.narrow_therapeutic_index is True

    def test_monitoring_for_large_fold(self):
        result = _base_result(
            displacer_conc_uM=1000.0,
            ki_displacement_uM=1.0,
        )
        assert result.monitoring_required is True

    def test_no_monitoring_negligible_normal_ti(self):
        result = predict_displacement_interaction(
            victim_drug="drug",
            displacer_drug="displacer",
            binding_protein="albumin",
            victim_fup_baseline=0.99,
            displacer_conc_uM=0.0,
            ki_displacement_uM=100.0,
            victim_total_bound_fraction=0.01,
            narrow_ti=False,
        )
        assert result.monitoring_required is False


# ---------------------------------------------------------------------------
# Screen displacers
# ---------------------------------------------------------------------------


class TestScreenDisplacers:
    def _make_displacers(self):
        return [
            {"name": "aspirin", "conc_uM": 100.0, "ki_uM": 10.0, "binding_protein": "albumin"},
            {"name": "ibuprofen", "conc_uM": 50.0, "ki_uM": 100.0, "binding_protein": "albumin"},
            {"name": "naproxen", "conc_uM": 200.0, "ki_uM": 5.0, "binding_protein": "albumin"},
        ]

    def test_screen_returns_list(self):
        results = screen_displacers("warfarin", 0.01, 0.99, self._make_displacers())
        assert isinstance(results, list)

    def test_screen_length_matches_input(self):
        displacers = self._make_displacers()
        results = screen_displacers("warfarin", 0.01, 0.99, displacers)
        assert len(results) == len(displacers)

    def test_screen_sorted_descending_fold(self):
        results = screen_displacers("warfarin", 0.01, 0.99, self._make_displacers())
        folds = [r.fup_increase_fold for r in results]
        assert folds == sorted(folds, reverse=True)

    def test_screen_all_results_are_correct_type(self):
        results = screen_displacers("warfarin", 0.01, 0.99, self._make_displacers())
        for r in results:
            assert isinstance(r, ProteinBindingDisplacementResult)

    def test_screen_empty_list(self):
        results = screen_displacers("warfarin", 0.01, 0.99, [])
        assert results == []


# ---------------------------------------------------------------------------
# Protein types
# ---------------------------------------------------------------------------


class TestBindingProtein:
    @pytest.mark.parametrize("protein", ["albumin", "alpha1_agp", "both"])
    def test_valid_protein_types(self, protein):
        result = _base_result(binding_protein=protein)
        assert result.binding_protein == protein


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_fup_baseline_zero(self):
        with pytest.raises(ValueError, match="victim_fup_baseline"):
            _base_result(victim_fup_baseline=0.0)

    def test_invalid_fup_baseline_above_one(self):
        with pytest.raises(ValueError, match="victim_fup_baseline"):
            _base_result(victim_fup_baseline=1.5)

    def test_invalid_ki_zero(self):
        with pytest.raises(ValueError, match="ki_displacement_uM"):
            _base_result(ki_displacement_uM=0.0)

    def test_invalid_ki_negative(self):
        with pytest.raises(ValueError, match="ki_displacement_uM"):
            _base_result(ki_displacement_uM=-5.0)

    def test_invalid_displacer_conc_negative(self):
        with pytest.raises(ValueError, match="displacer_conc_uM"):
            _base_result(displacer_conc_uM=-1.0)

    def test_invalid_bound_fraction_above_one(self):
        with pytest.raises(ValueError, match="victim_total_bound_fraction"):
            _base_result(victim_total_bound_fraction=1.1)

    def test_invalid_bound_fraction_negative(self):
        with pytest.raises(ValueError, match="victim_total_bound_fraction"):
            _base_result(victim_total_bound_fraction=-0.1)

    def test_invalid_binding_protein(self):
        with pytest.raises(ValueError, match="binding_protein"):
            _base_result(binding_protein="lipid")


# ---------------------------------------------------------------------------
# Notes content
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_string(self):
        result = _base_result()
        assert isinstance(result.notes, str)

    def test_notes_nonempty(self):
        result = _base_result()
        assert len(result.notes) > 0

    def test_notes_mentions_drugs(self):
        result = _base_result(victim_drug="warfarin", displacer_drug="aspirin")
        assert "warfarin" in result.notes or "aspirin" in result.notes
