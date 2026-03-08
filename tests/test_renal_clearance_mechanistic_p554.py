"""Tests for Phase 554 — Renal Clearance Mechanistic Model."""

import pytest

from omega_pbpk.core.renal_clearance_mechanistic_p554 import (
    RenalClearanceResult,
    predict_renal_clearance,
)


class TestPredictRenalClearance:
    """Tests for predict_renal_clearance."""

    def test_returns_renal_clearance_result(self):
        r = predict_renal_clearance("DrugA", 120.0, 0.5, 1.0)
        assert isinstance(r, RenalClearanceResult)

    def test_filtration_equals_gfr_times_fup(self):
        r = predict_renal_clearance("DrugA", 120.0, 0.3, 0.5)
        assert abs(r.cl_filtration_mL_per_min - 120.0 * 0.3) < 1e-9

    def test_secretion_adds_to_total(self):
        r_no_sec = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5, vmax_secretion_mL_per_min=0.0)
        r_sec = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5, vmax_secretion_mL_per_min=50.0)
        assert r_sec.cl_renal_total_mL_per_min > r_no_sec.cl_renal_total_mL_per_min

    def test_secretion_is_half_vmax(self):
        r = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5, vmax_secretion_mL_per_min=40.0)
        assert abs(r.cl_secretion_mL_per_min - 20.0) < 1e-9

    def test_no_secretion_when_vmax_zero(self):
        r = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5)
        assert r.cl_secretion_mL_per_min == 0.0

    def test_reabsorption_reduces_total(self):
        # logP=0 -> no reabsorption, logP=4 -> reabsorption
        r_no_reab = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5)
        r_reab = predict_renal_clearance("DrugA", 120.0, 0.5, 4.0)
        assert r_reab.cl_renal_total_mL_per_min < r_no_reab.cl_renal_total_mL_per_min

    def test_no_reabsorption_low_logP(self):
        r = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5)
        assert r.cl_reabsorption_mL_per_min == 0.0

    def test_reabsorption_increases_with_logP(self):
        r1 = predict_renal_clearance("DrugA", 120.0, 0.5, 2.0)
        r2 = predict_renal_clearance("DrugA", 120.0, 0.5, 5.0)
        assert r2.cl_reabsorption_mL_per_min > r1.cl_reabsorption_mL_per_min

    def test_reabsorption_capped_at_95pct(self):
        # Very high logP
        r = predict_renal_clearance("DrugA", 120.0, 0.5, 100.0)
        max_reab = (r.cl_filtration_mL_per_min + r.cl_secretion_mL_per_min) * 0.95
        assert r.cl_reabsorption_mL_per_min <= max_reab + 1e-9

    def test_cl_renal_non_negative(self):
        r = predict_renal_clearance("DrugA", 120.0, 0.1, 10.0)
        assert r.cl_renal_total_mL_per_min >= 0.0

    def test_fe_urine_in_range(self):
        r = predict_renal_clearance("DrugA", 120.0, 0.5, 1.0)
        assert 0 <= r.fe_urine <= 1

    def test_fe_urine_higher_with_lower_hepatic(self):
        r_high_hep = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5, cl_hepatic_mL_per_min=200.0)
        r_low_hep = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5, cl_hepatic_mL_per_min=10.0)
        assert r_low_hep.fe_urine > r_high_hep.fe_urine

    def test_classification_filtration_only(self):
        # No secretion, low logP -> filtration only
        r = predict_renal_clearance("DrugA", 120.0, 0.5, 0.5)
        assert r.renal_elimination_class == "filtration_only"

    def test_classification_secretion_dominant(self):
        # High secretion relative to filtration
        r = predict_renal_clearance("DrugA", 120.0, 0.1, 0.5, vmax_secretion_mL_per_min=100.0)
        # cl_filtration = 12, cl_secretion = 50, 50 > 1.5*12=18
        assert r.renal_elimination_class == "secretion_dominant"

    def test_classification_reabsorption_dominant(self):
        # High logP -> high reabsorption fraction
        # logP=8 -> reab_frac = min(0.95, 0.15*7) = 0.95 > 0.5
        r = predict_renal_clearance("DrugA", 120.0, 0.5, 8.0)
        assert r.renal_elimination_class == "reabsorption_dominant"

    def test_drug_name_stored(self):
        r = predict_renal_clearance("Metformin", 120.0, 0.5, 0.5)
        assert r.drug_name == "Metformin"

    def test_gfr_stored(self):
        r = predict_renal_clearance("DrugA", 90.0, 0.5, 1.0)
        assert r.gfr_mL_per_min == 90.0

    # Validation errors
    def test_negative_gfr_raises(self):
        with pytest.raises(ValueError):
            predict_renal_clearance("DrugA", -10.0, 0.5, 1.0)

    def test_zero_gfr_raises(self):
        with pytest.raises(ValueError):
            predict_renal_clearance("DrugA", 0.0, 0.5, 1.0)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            predict_renal_clearance("DrugA", 120.0, 0.0, 1.0)

    def test_fup_above_one_raises(self):
        with pytest.raises(ValueError):
            predict_renal_clearance("DrugA", 120.0, 1.5, 1.0)

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            predict_renal_clearance("", 120.0, 0.5, 1.0)
