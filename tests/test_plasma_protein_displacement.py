"""Tests for Phase 296 — Plasma Protein Binding Displacement."""

import pytest

from omega_pbpk.clinical.plasma_protein_displacement import (
    PPBDisplacementResult,
    assess_ppb_displacement,
    screen_displacement_pairs,
)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

VICTIM = "Warfarin"
DISPLACER = "Aspirin"
FU_BASE = 0.01  # highly protein-bound
CL_RESTRICT = 5.0  # L/h — restrictive
CL_NONRESTRICT = 60.0  # L/h — non-restrictive (> 30)
VD = 10.0  # L
DOSE = 5.0  # mg
CONC = 1.0  # mg/L displacing drug
IC50 = 1.0  # mg/L


# ---------------------------------------------------------------------------
# assess_ppb_displacement
# ---------------------------------------------------------------------------


class TestAssessPPBDisplacement:
    def test_returns_correct_type(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert isinstance(r, PPBDisplacementResult)

    def test_victim_drug_name_preserved(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.victim_drug == VICTIM

    def test_displacing_drug_name_preserved(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.displacing_drug == DISPLACER

    def test_fu_displaced_ge_baseline(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.fu_displaced >= r.victim_fu_baseline

    def test_fu_displaced_le_one(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, 1000.0, IC50)
        assert r.fu_displaced <= 1.0

    def test_fu_displaced_gt_zero(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.fu_displaced > 0.0

    def test_fu_change_fold_ge_one(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.fu_change_fold >= 1.0

    def test_higher_conc_higher_fu_displaced(self):
        r_low = assess_ppb_displacement(
            VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, 0.1, IC50
        )
        r_high = assess_ppb_displacement(
            VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, 10.0, IC50
        )
        assert r_high.fu_displaced > r_low.fu_displaced

    def test_zero_conc_no_displacement(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, 0.0, IC50)
        assert r.fu_displaced == pytest.approx(FU_BASE, rel=1e-6)

    def test_restrictive_auc_ratio_approx_one(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.restrictive_clearance is True
        assert r.auc_ratio == pytest.approx(1.0, abs=1e-9)

    def test_nonrestrict_auc_ratio_gt_one(self):
        """For high CL drug with binding site displaced, AUC ratio > 1."""
        r = assess_ppb_displacement(
            VICTIM, DISPLACER, FU_BASE, CL_NONRESTRICT, VD, DOSE, CONC, IC50
        )
        assert r.restrictive_clearance is False
        assert r.auc_ratio > 1.0

    def test_cmax_ratio_equals_fu_change_fold(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.cmax_ratio == pytest.approx(r.fu_change_fold, rel=1e-9)

    def test_clinical_significance_valid_set(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.clinical_significance in {"minor", "moderate", "significant"}

    def test_highly_displaced_is_significant(self):
        """When concentration >> IC50, fu nearly doubles, should be significant."""
        r = assess_ppb_displacement(VICTIM, DISPLACER, 0.01, CL_RESTRICT, VD, DOSE, 100.0, 0.5)
        assert r.clinical_significance == "significant"

    def test_minimal_displacement_is_minor(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, 0.5, CL_RESTRICT, VD, DOSE, 0.001, IC50)
        assert r.clinical_significance == "minor"

    def test_notes_is_string(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert isinstance(r.notes, str)

    def test_fu_baseline_preserved(self):
        r = assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, IC50)
        assert r.victim_fu_baseline == pytest.approx(FU_BASE)


# ---------------------------------------------------------------------------
# screen_displacement_pairs
# ---------------------------------------------------------------------------


class TestScreenDisplacementPairs:
    def _make_data(self) -> list[dict]:
        return [
            {"drug_name": "DrugA", "concentration_mg_L": 0.1, "ic50_mg_L": 1.0},
            {"drug_name": "DrugB", "concentration_mg_L": 5.0, "ic50_mg_L": 1.0},
            {"drug_name": "DrugC", "concentration_mg_L": 20.0, "ic50_mg_L": 1.0},
        ]

    def test_returns_list(self):
        result = screen_displacement_pairs(
            VICTIM, FU_BASE, CL_RESTRICT, VD, DOSE, self._make_data()
        )
        assert isinstance(result, list)

    def test_correct_length(self):
        data = self._make_data()
        result = screen_displacement_pairs(VICTIM, FU_BASE, CL_RESTRICT, VD, DOSE, data)
        assert len(result) == len(data)

    def test_sorted_by_fu_change_fold_descending(self):
        result = screen_displacement_pairs(
            VICTIM, FU_BASE, CL_RESTRICT, VD, DOSE, self._make_data()
        )
        folds = [r.fu_change_fold for r in result]
        assert folds == sorted(folds, reverse=True)

    def test_empty_input_returns_empty(self):
        result = screen_displacement_pairs(VICTIM, FU_BASE, CL_RESTRICT, VD, DOSE, [])
        assert result == []

    def test_drug_names_preserved(self):
        data = self._make_data()
        expected_names = {d["drug_name"] for d in data}
        result = screen_displacement_pairs(VICTIM, FU_BASE, CL_RESTRICT, VD, DOSE, data)
        returned_names = {r.displacing_drug for r in result}
        assert returned_names == expected_names

    def test_all_fu_displaced_ge_baseline(self):
        result = screen_displacement_pairs(
            VICTIM, FU_BASE, CL_RESTRICT, VD, DOSE, self._make_data()
        )
        assert all(r.fu_displaced >= FU_BASE for r in result)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_fu_zero_raises(self):
        with pytest.raises(ValueError, match="victim_fu_baseline"):
            assess_ppb_displacement(VICTIM, DISPLACER, 0.0, CL_RESTRICT, VD, DOSE, CONC, IC50)

    def test_fu_above_one_raises(self):
        with pytest.raises(ValueError, match="victim_fu_baseline"):
            assess_ppb_displacement(VICTIM, DISPLACER, 1.1, CL_RESTRICT, VD, DOSE, CONC, IC50)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="victim_cl_L_per_h"):
            assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, 0.0, VD, DOSE, CONC, IC50)

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="victim_vd_L"):
            assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, 0.0, DOSE, CONC, IC50)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="victim_dose_mg"):
            assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, 0.0, CONC, IC50)

    def test_negative_conc_raises(self):
        with pytest.raises(ValueError, match="displacing_concentration_mg_L"):
            assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, -1.0, IC50)

    def test_ic50_zero_raises(self):
        with pytest.raises(ValueError, match="ic50_displacement_mg_L"):
            assess_ppb_displacement(VICTIM, DISPLACER, FU_BASE, CL_RESTRICT, VD, DOSE, CONC, 0.0)
