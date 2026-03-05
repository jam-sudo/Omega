"""Tests for protein binding displacement DDI module."""
from __future__ import annotations

import math
import pytest

from omega_pbpk.clinical.protein_binding_ddi import (
    BindingDrug,
    DisplacementResult,
    multi_drug_displacement,
    predict_displacement,
)

# Reference drugs
WARFARIN = BindingDrug(
    name="warfarin",
    fup=0.01,       # 99% bound
    fu_tissue=0.05,
    vd_L=10.0,
    cl_L_per_h=0.2,
)

DIAZEPAM = BindingDrug(
    name="diazepam",
    fup=0.02,       # 98% bound
    fu_tissue=0.1,
    vd_L=70.0,
    cl_L_per_h=1.8,
)

LIDOCAINE = BindingDrug(
    name="lidocaine",
    fup=0.3,        # moderate binding, high extraction
    fu_tissue=0.5,
    vd_L=90.0,
    cl_L_per_h=60.0,  # high extraction ~0.67
)


class TestBindingDrug:
    def test_dataclass_fields(self):
        assert WARFARIN.name == "warfarin"
        assert WARFARIN.fup == pytest.approx(0.01)
        assert WARFARIN.vd_L == pytest.approx(10.0)

    def test_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            WARFARIN.fup = 0.05  # type: ignore


class TestPredictDisplacement:
    def test_returns_result_type(self):
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02)
        assert isinstance(result, DisplacementResult)

    def test_fup_ratio_correct(self):
        # 0.01 → 0.02 = 2x increase
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02)
        assert result.fup_ratio == pytest.approx(2.0)

    def test_no_displacement_warning(self):
        # fup_new <= fup_base → warning
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.005)
        assert any("no displacement" in w for w in result.warnings)

    def test_low_extraction_cl_scales_with_fup(self):
        # Low-extraction: CL ∝ fup; doubling fup doubles CL
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02)
        assert result.cl_displaced_L_per_h == pytest.approx(result.fup_ratio * WARFARIN.cl_L_per_h, rel=0.1)

    def test_low_extraction_auc_unchanged(self):
        # For low-extraction drug: AUC ≈ unchanged (CL increases proportionally)
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02)
        assert result.auc_ratio == pytest.approx(1.0, rel=0.2)

    def test_vd_increases_with_fup(self):
        # Higher fup → more tissue distribution → larger Vd
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02)
        assert result.vd_displaced_L > WARFARIN.vd_L

    def test_high_extraction_warning(self):
        # Lidocaine: high-extraction drug
        result = predict_displacement(LIDOCAINE, perpetrator_fup_effect=0.5)
        assert result.is_high_extraction
        assert any("blood-flow limited" in w for w in result.warnings)

    def test_clinical_significance_none(self):
        # <25% increase → none
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.011)
        assert result.clinical_significance == "none"

    def test_clinical_significance_minor(self):
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.013)
        assert result.clinical_significance == "minor"

    def test_clinical_significance_moderate(self):
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.016)
        assert result.clinical_significance == "moderate"

    def test_clinical_significance_major(self):
        # 3x fup → major
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.03)
        assert result.clinical_significance == "major"

    def test_victim_name_stored(self):
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02,
                                      perpetrator_name="aspirin")
        assert result.victim_name == "warfarin"
        assert result.perpetrator_name == "aspirin"

    def test_extraction_ratio_range(self):
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02)
        assert 0.0 <= result.extraction_ratio <= 1.0

    def test_custom_cl_int(self):
        # Explicit CLint should not raise
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02,
                                      cl_int_L_per_h=50.0)
        assert result.cl_displaced_L_per_h > 0

    def test_fup_displaced_stored(self):
        result = predict_displacement(WARFARIN, perpetrator_fup_effect=0.02)
        assert result.fup_displaced == pytest.approx(0.02)
        assert result.fup_baseline == pytest.approx(0.01)


class TestMultiDrugDisplacement:
    def test_two_perpetrators(self):
        results = multi_drug_displacement(
            WARFARIN,
            perpetrators=[("aspirin", 0.005), ("ibuprofen", 0.005)],
        )
        assert len(results) == 2

    def test_fup_accumulates(self):
        results = multi_drug_displacement(
            WARFARIN,
            perpetrators=[("drug_a", 0.005), ("drug_b", 0.005)],
        )
        # Second perpetrator builds on displaced fup of first
        assert results[1].fup_displaced > results[0].fup_displaced

    def test_fup_capped_at_one(self):
        results = multi_drug_displacement(
            WARFARIN,
            perpetrators=[("drug_a", 0.5), ("drug_b", 0.6)],
        )
        assert results[-1].fup_displaced <= 1.0

    def test_empty_perpetrators(self):
        results = multi_drug_displacement(WARFARIN, perpetrators=[])
        assert results == []
