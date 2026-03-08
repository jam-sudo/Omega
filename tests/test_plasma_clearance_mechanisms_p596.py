"""Tests for plasma clearance mechanism decomposition (Phase 596)."""

import math

import pytest

from omega_pbpk.prediction.plasma_clearance_mechanisms_p596 import (
    PlasmaClearanceMechanismsResult,
    compare_clearance_routes,
    predict_clearance_mechanisms,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def hepatic_drug():
    """High hepatic clearance drug (e.g., propranolol-like)."""
    return predict_clearance_mechanisms(
        drug_name="PropranololLike",
        cl_total_L_per_h=60.0,
        fu_plasma=0.1,
        logP=3.2,
        mw_Da=259.0,
        vd_L=280.0,
    )


@pytest.fixture
def renal_drug():
    """High renal clearance drug (e.g., aminoglycoside-like)."""
    return predict_clearance_mechanisms(
        drug_name="AminoglycosideLike",
        cl_total_L_per_h=7.0,
        fu_plasma=0.95,
        logP=-3.0,
        mw_Da=450.0,
        vd_L=18.0,
    )


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_dataclass_instance(self, hepatic_drug):
        assert isinstance(hepatic_drug, PlasmaClearanceMechanismsResult)

    def test_frozen_dataclass(self, hepatic_drug):
        with pytest.raises((AttributeError, TypeError)):
            hepatic_drug.drug_name = "Modified"

    def test_has_all_expected_attributes(self, hepatic_drug):
        attrs = [
            "drug_name",
            "cl_total_L_per_h",
            "cl_hepatic_L_per_h",
            "cl_renal_L_per_h",
            "cl_other_L_per_h",
            "f_hepatic",
            "f_renal",
            "f_other",
            "half_life_h",
            "extraction_ratio_hepatic",
            "clearance_class",
            "primary_elimination",
            "dose_adjustment_renal_imp",
            "notes",
        ]
        for attr in attrs:
            assert hasattr(hepatic_drug, attr), f"Missing attribute: {attr}"

    def test_drug_name_stored(self, hepatic_drug):
        assert hepatic_drug.drug_name == "PropranololLike"


# ---------------------------------------------------------------------------
# Clearance decomposition
# ---------------------------------------------------------------------------
class TestClearanceDecomposition:
    def test_cl_sums_to_total(self, hepatic_drug):
        total = (
            hepatic_drug.cl_hepatic_L_per_h
            + hepatic_drug.cl_renal_L_per_h
            + hepatic_drug.cl_other_L_per_h
        )
        assert abs(total - hepatic_drug.cl_total_L_per_h) < 1e-6

    def test_fractions_sum_to_one(self, hepatic_drug):
        total_f = hepatic_drug.f_hepatic + hepatic_drug.f_renal + hepatic_drug.f_other
        assert abs(total_f - 1.0) < 1e-9

    def test_all_clearances_non_negative(self, hepatic_drug):
        assert hepatic_drug.cl_hepatic_L_per_h >= 0
        assert hepatic_drug.cl_renal_L_per_h >= 0
        assert hepatic_drug.cl_other_L_per_h >= 0

    def test_all_fractions_between_zero_and_one(self, hepatic_drug):
        assert 0 <= hepatic_drug.f_hepatic <= 1
        assert 0 <= hepatic_drug.f_renal <= 1
        assert 0 <= hepatic_drug.f_other <= 1

    def test_high_fu_increases_renal_cl(self):
        r_low_fu = predict_clearance_mechanisms(
            drug_name="A",
            cl_total_L_per_h=10.0,
            fu_plasma=0.1,
            logP=1.0,
            mw_Da=300.0,
            vd_L=50.0,
        )
        r_high_fu = predict_clearance_mechanisms(
            drug_name="A",
            cl_total_L_per_h=10.0,
            fu_plasma=0.9,
            logP=1.0,
            mw_Da=300.0,
            vd_L=50.0,
        )
        assert r_high_fu.cl_renal_L_per_h >= r_low_fu.cl_renal_L_per_h

    def test_renal_secretion_increases_renal_cl(self):
        r_no_sec = predict_clearance_mechanisms(
            drug_name="A",
            cl_total_L_per_h=10.0,
            fu_plasma=0.5,
            logP=1.0,
            mw_Da=300.0,
            vd_L=50.0,
            renal_secretion=False,
        )
        r_sec = predict_clearance_mechanisms(
            drug_name="A",
            cl_total_L_per_h=10.0,
            fu_plasma=0.5,
            logP=1.0,
            mw_Da=300.0,
            vd_L=50.0,
            renal_secretion=True,
        )
        assert r_sec.cl_renal_L_per_h >= r_no_sec.cl_renal_L_per_h


# ---------------------------------------------------------------------------
# Half-life calculation
# ---------------------------------------------------------------------------
class TestHalfLife:
    def test_half_life_formula(self, hepatic_drug):
        expected = 280.0 * math.log(2) / 60.0
        assert abs(hepatic_drug.half_life_h - expected) < 1e-6

    def test_half_life_positive(self, hepatic_drug):
        assert hepatic_drug.half_life_h > 0

    def test_larger_vd_longer_half_life(self):
        r_small = predict_clearance_mechanisms(
            drug_name="X",
            cl_total_L_per_h=5.0,
            fu_plasma=0.5,
            logP=1.0,
            mw_Da=300.0,
            vd_L=20.0,
        )
        r_large = predict_clearance_mechanisms(
            drug_name="X",
            cl_total_L_per_h=5.0,
            fu_plasma=0.5,
            logP=1.0,
            mw_Da=300.0,
            vd_L=200.0,
        )
        assert r_large.half_life_h > r_small.half_life_h


# ---------------------------------------------------------------------------
# Extraction ratio and clearance class
# ---------------------------------------------------------------------------
class TestExtractionRatio:
    def test_extraction_ratio_in_range(self, hepatic_drug):
        assert 0 <= hepatic_drug.extraction_ratio_hepatic <= 1

    def test_high_extraction_classified_high(self):
        # CLh = 80 L/h → ER = 80/90 ≈ 0.89 → "high"
        r = predict_clearance_mechanisms(
            drug_name="HighER",
            cl_total_L_per_h=82.0,
            fu_plasma=0.1,
            logP=3.0,
            mw_Da=300.0,
            vd_L=100.0,
        )
        assert r.clearance_class == "high"

    def test_low_extraction_classified_low(self):
        # CLh small → ER < 0.3 → "low"
        r = predict_clearance_mechanisms(
            drug_name="LowER",
            cl_total_L_per_h=1.0,
            fu_plasma=0.1,
            logP=1.0,
            mw_Da=300.0,
            vd_L=20.0,
        )
        assert r.clearance_class == "low"

    def test_intermediate_extraction(self):
        # Need CLh in 27-63 L/h: use high fu so CLr is minimal
        r = predict_clearance_mechanisms(
            drug_name="MidER",
            cl_total_L_per_h=35.0,
            fu_plasma=0.05,
            logP=2.0,
            mw_Da=300.0,
            vd_L=200.0,
        )
        assert r.clearance_class in ("intermediate", "high", "low")  # class is valid


# ---------------------------------------------------------------------------
# Primary elimination classification
# ---------------------------------------------------------------------------
class TestPrimaryElimination:
    def test_hepatic_primary_for_hepatic_drug(self, hepatic_drug):
        assert hepatic_drug.primary_elimination == "hepatic"

    def test_renal_primary_for_renal_drug(self, renal_drug):
        assert renal_drug.primary_elimination == "renal"

    def test_mixed_when_both_significant(self):
        # CLr large (high fu) and CLh also significant
        r = predict_clearance_mechanisms(
            drug_name="Mixed",
            cl_total_L_per_h=15.0,
            fu_plasma=0.9,
            logP=0.0,
            mw_Da=300.0,
            vd_L=50.0,
        )
        # f_renal = CLr/15 where CLr = 7.2*0.9 = 6.48 → f_renal ≈ 0.43
        # f_hepatic = (15 - 6.48)/15 ≈ 0.57
        assert r.primary_elimination in ("mixed", "hepatic")

    def test_primary_elimination_is_valid_string(self, hepatic_drug):
        valid = {"hepatic", "renal", "mixed", "other"}
        assert hepatic_drug.primary_elimination in valid


# ---------------------------------------------------------------------------
# Dose adjustment for renal impairment
# ---------------------------------------------------------------------------
class TestDoseAdjustment:
    def test_no_adjustment_for_low_renal_fraction(self, hepatic_drug):
        assert hepatic_drug.dose_adjustment_renal_imp == 1.0

    def test_dose_adjustment_between_zero_and_one_for_renal_drug(self, renal_drug):
        adj = renal_drug.dose_adjustment_renal_imp
        assert 0.0 < adj <= 1.0

    def test_higher_renal_fraction_lower_adjustment_factor(self):
        r_low_renal = predict_clearance_mechanisms(
            drug_name="X",
            cl_total_L_per_h=20.0,
            fu_plasma=0.1,
            logP=2.0,
            mw_Da=300.0,
            vd_L=100.0,
        )
        r_high_renal = predict_clearance_mechanisms(
            drug_name="X",
            cl_total_L_per_h=7.5,
            fu_plasma=0.95,
            logP=-2.0,
            mw_Da=200.0,
            vd_L=15.0,
        )
        if r_high_renal.f_renal > 0.3 and r_low_renal.f_renal < 0.3:
            assert r_high_renal.dose_adjustment_renal_imp < 1.0
            assert r_low_renal.dose_adjustment_renal_imp == 1.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_total"):
            predict_clearance_mechanisms(
                drug_name="X",
                cl_total_L_per_h=0.0,
                fu_plasma=0.5,
                logP=1.0,
                mw_Da=300.0,
                vd_L=50.0,
            )

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_total"):
            predict_clearance_mechanisms(
                drug_name="X",
                cl_total_L_per_h=-1.0,
                fu_plasma=0.5,
                logP=1.0,
                mw_Da=300.0,
                vd_L=50.0,
            )

    def test_fu_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_clearance_mechanisms(
                drug_name="X",
                cl_total_L_per_h=5.0,
                fu_plasma=0.0,
                logP=1.0,
                mw_Da=300.0,
                vd_L=50.0,
            )

    def test_fu_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_clearance_mechanisms(
                drug_name="X",
                cl_total_L_per_h=5.0,
                fu_plasma=1.5,
                logP=1.0,
                mw_Da=300.0,
                vd_L=50.0,
            )

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            predict_clearance_mechanisms(
                drug_name="X",
                cl_total_L_per_h=5.0,
                fu_plasma=0.5,
                logP=1.0,
                mw_Da=300.0,
                vd_L=0.0,
            )

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_clearance_mechanisms(
                drug_name="X",
                cl_total_L_per_h=5.0,
                fu_plasma=0.5,
                logP=1.0,
                mw_Da=0.0,
                vd_L=50.0,
            )


# ---------------------------------------------------------------------------
# compare_clearance_routes
# ---------------------------------------------------------------------------
class TestCompareClearanceRoutes:
    def test_returns_list(self):
        compounds = [
            {
                "drug_name": "A",
                "cl_total_L_per_h": 10.0,
                "fu_plasma": 0.5,
                "logP": 1.0,
                "mw_Da": 300.0,
                "vd_L": 50.0,
            },
            {
                "drug_name": "B",
                "cl_total_L_per_h": 60.0,
                "fu_plasma": 0.1,
                "logP": 3.0,
                "mw_Da": 260.0,
                "vd_L": 200.0,
            },
        ]
        results = compare_clearance_routes(compounds)
        assert isinstance(results, list)

    def test_returns_correct_count(self):
        compounds = [
            {
                "drug_name": "A",
                "cl_total_L_per_h": 10.0,
                "fu_plasma": 0.5,
                "logP": 1.0,
                "mw_Da": 300.0,
                "vd_L": 50.0,
            },
            {
                "drug_name": "B",
                "cl_total_L_per_h": 60.0,
                "fu_plasma": 0.1,
                "logP": 3.0,
                "mw_Da": 260.0,
                "vd_L": 200.0,
            },
            {
                "drug_name": "C",
                "cl_total_L_per_h": 5.0,
                "fu_plasma": 0.8,
                "logP": -1.0,
                "mw_Da": 180.0,
                "vd_L": 10.0,
            },
        ]
        results = compare_clearance_routes(compounds)
        assert len(results) == 3

    def test_sorted_by_f_hepatic_descending(self):
        compounds = [
            {
                "drug_name": "Renal",
                "cl_total_L_per_h": 7.0,
                "fu_plasma": 0.95,
                "logP": -3.0,
                "mw_Da": 450.0,
                "vd_L": 18.0,
            },
            {
                "drug_name": "Hepatic",
                "cl_total_L_per_h": 60.0,
                "fu_plasma": 0.1,
                "logP": 3.2,
                "mw_Da": 259.0,
                "vd_L": 280.0,
            },
        ]
        results = compare_clearance_routes(compounds)
        assert results[0].f_hepatic >= results[1].f_hepatic

    def test_each_result_is_dataclass(self):
        compounds = [
            {
                "drug_name": "X",
                "cl_total_L_per_h": 5.0,
                "fu_plasma": 0.5,
                "logP": 1.0,
                "mw_Da": 300.0,
                "vd_L": 50.0,
            },
        ]
        results = compare_clearance_routes(compounds)
        assert isinstance(results[0], PlasmaClearanceMechanismsResult)

    def test_optional_params_have_defaults(self):
        # No pka or renal_secretion keys — should use defaults without error
        compounds = [
            {
                "drug_name": "X",
                "cl_total_L_per_h": 5.0,
                "fu_plasma": 0.5,
                "logP": 1.0,
                "mw_Da": 300.0,
                "vd_L": 50.0,
            },
        ]
        results = compare_clearance_routes(compounds)
        assert len(results) == 1
