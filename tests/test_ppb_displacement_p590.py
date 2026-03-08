"""
Tests for Phase 590 — Plasma Protein Binding Displacement.
"""

import pytest

from omega_pbpk.prediction.ppb_displacement_p590 import (
    PPBDisplacementResult,
    predict_ppb_displacement,
)

# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------


def _base(**kwargs):
    defaults = dict(
        drug_name="Warfarin",
        fu_drug=0.01,  # highly protein-bound
        vd_L=10.0,
        cl_L_per_h=0.5,
        displacer_name="Aspirin",
        fu_displacer=0.1,  # also highly bound → displaces
        protein_binding_site="albumin",
    )
    defaults.update(kwargs)
    return predict_ppb_displacement(**defaults)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _base()
        assert isinstance(result, PPBDisplacementResult)

    def test_frozen_dataclass(self):
        result = _base()
        with pytest.raises(Exception):
            result.fu_baseline = 0.99  # type: ignore

    def test_has_all_fields(self):
        result = _base()
        for field in [
            "drug_name",
            "displacer_name",
            "fu_baseline",
            "fu_displaced",
            "fu_ratio",
            "delta_auc_pct",
            "clinical_significance",
            "narrow_therapeutic_index",
            "notes",
        ]:
            assert hasattr(result, field)

    def test_drug_name_preserved(self):
        result = _base(drug_name="TestDrug")
        assert result.drug_name == "TestDrug"

    def test_displacer_name_preserved(self):
        result = _base(displacer_name="Phenytoin")
        assert result.displacer_name == "Phenytoin"


# ---------------------------------------------------------------------------
# 2. Displacement calculation
# ---------------------------------------------------------------------------


class TestDisplacementCalculation:
    def test_fu_displaced_greater_than_baseline(self):
        result = _base()
        assert result.fu_displaced > result.fu_baseline

    def test_fu_baseline_matches_input(self):
        result = _base(fu_drug=0.05)
        assert abs(result.fu_baseline - 0.05) < 1e-9

    def test_fu_ratio_consistent(self):
        result = _base()
        assert abs(result.fu_ratio - result.fu_displaced / result.fu_baseline) < 1e-9

    def test_fu_displaced_capped_at_one(self):
        # Very weak binder with strong displacer
        result = _base(fu_drug=0.99, fu_displacer=0.01)
        assert result.fu_displaced <= 1.0

    def test_albumin_and_alpha1agp_same_displacement(self):
        r_alb = _base(protein_binding_site="albumin")
        r_agp = _base(protein_binding_site="alpha1_agp")
        assert abs(r_alb.fu_displaced - r_agp.fu_displaced) < 1e-9

    def test_different_site_less_displacement(self):
        # Using "unknown" site — tests that the "else" branch gives less displacement
        # We force the else branch by passing a value not in the explicit list
        r_same = _base(protein_binding_site="albumin")
        r_diff = _base(protein_binding_site="other")
        assert r_diff.fu_displaced < r_same.fu_displaced

    def test_highly_bound_displacer_more_effect(self):
        r_low = _base(fu_displacer=0.8)  # displacer not very bound → less effect
        r_high = _base(fu_displacer=0.01)  # displacer very bound → more displacement
        assert r_high.fu_displaced >= r_low.fu_displaced

    def test_both_site_uses_high_displacement(self):
        r_both = _base(protein_binding_site="both")
        r_diff = _base(protein_binding_site="other")
        assert r_both.fu_displaced > r_diff.fu_displaced


# ---------------------------------------------------------------------------
# 3. High extraction / high Vd effects (restricted AUC change)
# ---------------------------------------------------------------------------


class TestHighExtractionHighVd:
    def test_high_extraction_restricts_auc_change(self):
        # CL = 80 L/h → extraction = 80/(80+90) ≈ 0.47  (not high)
        # CL = 200 L/h → extraction = 200/(200+90) ≈ 0.69  (not > 0.7 yet)
        # CL = 500 L/h → extraction = 500/(500+90) ≈ 0.847  (high)
        r_low_cl = _base(cl_L_per_h=0.5, vd_L=10.0)
        r_high_cl = _base(cl_L_per_h=500.0, vd_L=10.0)
        assert abs(r_high_cl.delta_auc_pct) < abs(r_low_cl.delta_auc_pct)

    def test_high_vd_restricts_auc_change(self):
        r_low_vd = _base(vd_L=10.0, cl_L_per_h=0.5)
        r_high_vd = _base(vd_L=200.0, cl_L_per_h=0.5)
        assert abs(r_high_vd.delta_auc_pct) < abs(r_low_vd.delta_auc_pct)

    def test_high_extraction_delta_auc_formula(self):
        cl = 500.0
        extraction = cl / (cl + 90.0)
        assert extraction > 0.7  # confirm high extraction
        # Use weakly bound drug (fu=0.5) with weak displacer (fu=0.8) → small fu_ratio
        result = predict_ppb_displacement(
            drug_name="Drug",
            fu_drug=0.5,
            vd_L=10.0,
            cl_L_per_h=cl,
            displacer_name="D",
            fu_displacer=0.8,
            protein_binding_site="albumin",
        )
        # delta_auc_pct = (fu_ratio - 1) * 5 — small for weak displacement
        # displacement = 0.3*(1-0.8)=0.06; fu_displaced=0.5+0.06*0.5=0.53; ratio=1.06
        # delta_auc = (1.06-1)*5 = 0.3 < 1.0
        assert abs(result.delta_auc_pct) < 1.0

    def test_low_extraction_normal_auc_formula(self):
        cl = 0.5
        extraction = cl / (cl + 90.0)
        assert extraction < 0.7
        result = _base(cl_L_per_h=cl, vd_L=10.0, fu_drug=0.01, fu_displacer=0.01)
        # Should see meaningful AUC change
        assert abs(result.delta_auc_pct) > 1.0


# ---------------------------------------------------------------------------
# 4. Narrow therapeutic index flag
# ---------------------------------------------------------------------------


class TestNarrowTI:
    def test_narrow_ti_true_when_delta_auc_above_20(self):
        # Highly bound drug + strong displacer + low Vd + low CL → large delta_auc_pct
        result = predict_ppb_displacement(
            drug_name="NTIDrug",
            fu_drug=0.005,
            vd_L=5.0,
            cl_L_per_h=0.2,
            displacer_name="Displacer",
            fu_displacer=0.01,
            protein_binding_site="albumin",
        )
        if result.delta_auc_pct > 20.0:
            assert result.narrow_therapeutic_index is True

    def test_narrow_ti_false_when_delta_auc_below_20(self):
        # Low displacement scenario
        result = predict_ppb_displacement(
            drug_name="SafeDrug",
            fu_drug=0.5,
            vd_L=10.0,
            cl_L_per_h=0.5,
            displacer_name="WeakDisplacer",
            fu_displacer=0.9,  # not very bound → little displacement
            protein_binding_site="albumin",
        )
        if result.delta_auc_pct <= 20.0:
            assert result.narrow_therapeutic_index is False

    def test_narrow_ti_consistent_with_delta_auc(self):
        result = _base()
        if result.delta_auc_pct > 20.0:
            assert result.narrow_therapeutic_index is True
        else:
            assert result.narrow_therapeutic_index is False


# ---------------------------------------------------------------------------
# 5. Clinical significance levels
# ---------------------------------------------------------------------------


class TestClinicalSignificance:
    def test_negligible_significance(self):
        # High Vd + weakly bound drug + weakly bound displacer → restricted, negligible AUC change
        result = predict_ppb_displacement(
            drug_name="Drug",
            fu_drug=0.5,
            vd_L=500.0,
            cl_L_per_h=0.5,
            displacer_name="D",
            fu_displacer=0.8,
            protein_binding_site="albumin",
        )
        assert result.clinical_significance == "negligible"

    def test_significance_values_are_valid(self):
        valid = {"negligible", "minor", "moderate", "major"}
        for vd in [5.0, 10.0, 50.0, 200.0]:
            r = _base(vd_L=vd)
            assert r.clinical_significance in valid

    def test_notes_is_string(self):
        result = _base()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fu_drug_zero_raises(self):
        with pytest.raises(ValueError, match="fu_drug"):
            _base(fu_drug=0.0)

    def test_fu_drug_negative_raises(self):
        with pytest.raises(ValueError, match="fu_drug"):
            _base(fu_drug=-0.1)

    def test_fu_drug_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_drug"):
            _base(fu_drug=1.1)

    def test_fu_drug_exactly_one_ok(self):
        result = _base(fu_drug=1.0)
        assert result.fu_baseline == 1.0

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _base(vd_L=0.0)

    def test_vd_negative_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            _base(vd_L=-10.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _base(cl_L_per_h=0.0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _base(cl_L_per_h=-5.0)

    def test_fu_displacer_zero_raises(self):
        with pytest.raises(ValueError, match="fu_displacer"):
            _base(fu_displacer=0.0)

    def test_fu_displacer_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_displacer"):
            _base(fu_displacer=1.5)

    def test_fu_displacer_exactly_one_ok(self):
        result = _base(fu_displacer=1.0)
        # No binding → no displacement
        assert result.fu_displaced == pytest.approx(result.fu_baseline)
