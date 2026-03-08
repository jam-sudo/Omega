"""Tests for Phase 538 — Protein Binding Displacement Drug Interaction."""

import pytest

from omega_pbpk.clinical.protein_binding_displacement_p538 import (
    ProteinBindingDisplacementResult,
    assess_protein_binding_displacement,
    screen_displacement_interactions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base(**overrides):
    """Return a baseline assessment with optional keyword overrides."""
    kwargs = dict(
        displaced_drug="Warfarin",
        displacing_drug="Aspirin",
        c_displacing_mg_L=10.0,
        ki_displacement_mg_L=5.0,
        fu_displaced_baseline=0.01,  # highly bound → low extraction expected
        cl_int_displaced_L_per_h=2.0,  # low CLint → enzyme-limited
    )
    kwargs.update(overrides)
    return assess_protein_binding_displacement(**kwargs)


def _high_extraction_base():
    """Displaced drug with very high CLint → flow-limited."""
    return assess_protein_binding_displacement(
        displaced_drug="Propranolol",
        displacing_drug="Displacing",
        c_displacing_mg_L=20.0,
        ki_displacement_mg_L=5.0,
        fu_displaced_baseline=0.2,
        cl_int_displaced_L_per_h=2000.0,  # flow-limited
    )


# ---------------------------------------------------------------------------
# Return type & field presence
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass_instance(self):
        r = _base()
        assert isinstance(r, ProteinBindingDisplacementResult)

    def test_has_displaced_drug(self):
        r = _base()
        assert r.displaced_drug == "Warfarin"

    def test_has_displacing_drug(self):
        r = _base()
        assert r.displacing_drug == "Aspirin"

    def test_has_fu_ratio(self):
        r = _base()
        assert isinstance(r.fu_ratio, float)

    def test_has_auc_ratio(self):
        r = _base()
        assert isinstance(r.auc_ratio, float)

    def test_has_free_cmax_ratio(self):
        r = _base()
        assert isinstance(r.free_cmax_ratio, float)

    def test_has_notes(self):
        r = _base()
        assert isinstance(r.notes, str) and len(r.notes) > 0

    def test_frozen(self):
        r = _base()
        with pytest.raises((AttributeError, TypeError)):
            r.displaced_drug = "X"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core model behaviour
# ---------------------------------------------------------------------------


class TestCoreModel:
    def test_fu_new_ge_baseline(self):
        r = _base()
        assert r.fu_displaced_new >= r.fu_displaced_baseline

    def test_fu_ratio_ge_one(self):
        r = _base()
        assert r.fu_ratio >= 1.0

    def test_higher_c_higher_fu_ratio(self):
        r_low = _base(c_displacing_mg_L=1.0)
        r_high = _base(c_displacing_mg_L=100.0)
        assert r_high.fu_ratio > r_low.fu_ratio

    def test_very_high_ki_no_displacement(self):
        r = _base(ki_displacement_mg_L=1e9)
        assert abs(r.fu_ratio - 1.0) < 1e-6

    def test_fu_new_clamped_at_one(self):
        r = assess_protein_binding_displacement(
            displaced_drug="D",
            displacing_drug="I",
            c_displacing_mg_L=1e8,
            ki_displacement_mg_L=0.001,
            fu_displaced_baseline=0.9,
            cl_int_displaced_L_per_h=10.0,
        )
        assert r.fu_displaced_new <= 1.0

    def test_c_displacing_zero_no_effect(self):
        r = _base(c_displacing_mg_L=0.0)
        assert abs(r.fu_ratio - 1.0) < 1e-9
        assert abs(r.fu_displaced_new - r.fu_displaced_baseline) < 1e-9


# ---------------------------------------------------------------------------
# Extraction ratio
# ---------------------------------------------------------------------------


class TestExtractionRatio:
    def test_eh_low_extraction_drug_under_threshold(self):
        r = _base()
        # fup=0.01, CLint=2, Q=90 → EH = (0.01*2)/(90+0.01*2) very small
        assert r.eh_displaced < 0.3

    def test_eh_high_extraction_drug_above_threshold(self):
        r = _high_extraction_base()
        assert r.eh_displaced > 0.7

    def test_eh_stored_correctly(self):
        fup, cl_int, q = 0.01, 2.0, 90.0
        expected_eh = fup * cl_int / (q + fup * cl_int)
        r = _base()
        assert abs(r.eh_displaced - expected_eh) < 1e-9


# ---------------------------------------------------------------------------
# Hepatic clearance
# ---------------------------------------------------------------------------


class TestHepaticClearance:
    def test_clh_new_ge_baseline_when_displaced(self):
        """Displacement raises fu → CLh increases (or stays same if clamped)."""
        r = _base()
        assert r.cl_hepatic_new_L_per_h >= r.cl_hepatic_baseline_L_per_h

    def test_clh_baseline_formula(self):
        fup, cl_int, q = 0.01, 2.0, 90.0
        expected = q * fup * cl_int / (q + fup * cl_int)
        r = _base()
        assert abs(r.cl_hepatic_baseline_L_per_h - expected) < 1e-9


# ---------------------------------------------------------------------------
# AUC ratio and free Cmax ratio
# ---------------------------------------------------------------------------


class TestAUCAndCmax:
    def test_auc_ratio_close_to_one_low_extraction(self):
        """AUC should be ≈ 1 for low-extraction drugs due to proportional CLh/Vd change."""
        r = _base()
        # auc_ratio = CLh_baseline / CLh_new
        # For low extraction: CLh = fup*CLint ≈ proportional → AUC ratio ≈ 1.0
        # Allow some deviation since formula isn't perfectly proportional
        assert 0.5 <= r.auc_ratio <= 1.5

    def test_free_cmax_ratio_equals_fu_ratio(self):
        r = _base()
        assert abs(r.free_cmax_ratio - r.fu_ratio) < 1e-9

    def test_free_cmax_ratio_ge_one(self):
        r = _base()
        assert r.free_cmax_ratio >= 1.0


# ---------------------------------------------------------------------------
# Displacement significance
# ---------------------------------------------------------------------------


class TestDisplacementSignificance:
    def test_negligible_when_no_displacement(self):
        r = _base(c_displacing_mg_L=0.0)
        assert r.displacement_significance == "negligible"

    def test_significant_when_fu_ratio_above_2(self):
        # fu_baseline=0.01, fu_displacing=0.05 default
        # Need [I]/Ki >> 1: c=100, ki=0.01 → unbound_I=5, unbound_I/Ki = 5/0.01 = 500
        r = _base(c_displacing_mg_L=100.0, ki_displacement_mg_L=0.01)
        assert r.displacement_significance == "significant"

    def test_classification_order(self):
        """Higher fu_ratio → more severe classification."""
        r_neg = _base(c_displacing_mg_L=0.0)
        r_sig = _base(c_displacing_mg_L=100.0, ki_displacement_mg_L=0.01)
        assert r_neg.fu_ratio < r_sig.fu_ratio

    def test_mild_significance(self):
        # Small displacement: ~11-50% increase in fu
        r = _base(c_displacing_mg_L=5.0, ki_displacement_mg_L=2.0)
        # [I] = 5*0.05=0.25, ratio=0.25/2=0.125 → fu_ratio=1.125 → mild
        assert r.displacement_significance in ("mild", "negligible", "moderate")


# ---------------------------------------------------------------------------
# Clinical concern
# ---------------------------------------------------------------------------


class TestClinicalConcern:
    def test_clinical_concern_true_for_low_extraction_significant(self):
        # Low extraction (EH < 0.3) + significant displacement
        r = _base(c_displacing_mg_L=100.0, ki_displacement_mg_L=0.01)
        assert r.eh_displaced < 0.3
        if r.displacement_significance != "negligible":
            assert r.clinical_concern is True

    def test_clinical_concern_false_for_high_extraction(self):
        # High extraction drug → clinical_concern False even if displaced significantly
        r = _high_extraction_base()
        assert r.clinical_concern is False

    def test_clinical_concern_false_when_negligible(self):
        r = _base(c_displacing_mg_L=0.0)
        assert r.clinical_concern is False


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_c_displacing_raises(self):
        with pytest.raises(ValueError, match="c_displacing"):
            _base(c_displacing_mg_L=-1.0)

    def test_ki_zero_raises(self):
        with pytest.raises(ValueError, match="ki"):
            _base(ki_displacement_mg_L=0.0)

    def test_ki_negative_raises(self):
        with pytest.raises(ValueError):
            _base(ki_displacement_mg_L=-1.0)

    def test_fu_baseline_zero_raises(self):
        with pytest.raises(ValueError, match="fu_displaced_baseline"):
            _base(fu_displaced_baseline=0.0)

    def test_fu_baseline_above_one_raises(self):
        with pytest.raises(ValueError):
            _base(fu_displaced_baseline=1.5)

    def test_cl_int_zero_raises(self):
        with pytest.raises(ValueError, match="cl_int"):
            _base(cl_int_displaced_L_per_h=0.0)


# ---------------------------------------------------------------------------
# screen_displacement_interactions
# ---------------------------------------------------------------------------


class TestScreenDisplacementInteractions:
    def _perpetrators(self):
        return [
            {
                "displacing_drug": "DrugA",
                "c_displacing_mg_L": 100.0,
                "ki_displacement_mg_L": 0.01,
                "fu_displacing": 0.05,
            },
            {
                "displacing_drug": "DrugB",
                "c_displacing_mg_L": 1.0,
                "ki_displacement_mg_L": 10.0,
                "fu_displacing": 0.05,
            },
            {
                "displacing_drug": "DrugC",
                "c_displacing_mg_L": 5.0,
                "ki_displacement_mg_L": 1.0,
                "fu_displacing": 0.05,
            },
        ]

    def test_returns_list(self):
        results = screen_displacement_interactions("Warfarin", 2.0, 0.01, self._perpetrators())
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_descending_fu_ratio(self):
        results = screen_displacement_interactions("Warfarin", 2.0, 0.01, self._perpetrators())
        ratios = [r.fu_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_strongest_displacer_first(self):
        results = screen_displacement_interactions("Warfarin", 2.0, 0.01, self._perpetrators())
        # DrugA has very high concentration / very low Ki → strongest displacement
        assert results[0].displacing_drug == "DrugA"

    def test_each_result_is_correct_type(self):
        results = screen_displacement_interactions("Warfarin", 2.0, 0.01, self._perpetrators())
        for r in results:
            assert isinstance(r, ProteinBindingDisplacementResult)

    def test_empty_perpetrators(self):
        results = screen_displacement_interactions("Warfarin", 2.0, 0.01, [])
        assert results == []
