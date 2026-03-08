"""Tests for Phase 345 — Renal Drug Clearance Prediction."""

import math

import pytest

from omega_pbpk.prediction.renal_clearance_predictor import (
    RenalClearancePredictionResult,
    predict_renal_clearance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basic(**kwargs):
    """Return a result with sensible defaults, overriding with kwargs."""
    defaults = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logP=1.5,
        pKa=7.4,
        drug_type="neutral",
        fu_plasma=0.5,
        gfr_mL_per_min=120.0,
        tubular_secretion_clint_mL_per_min=0.0,
        reabsorption_fraction=0.0,
        protein_binding_pct=50.0,
    )
    defaults.update(kwargs)
    return predict_renal_clearance(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_frozen_dataclass(self):
        r = _basic()
        assert isinstance(r, RenalClearancePredictionResult)

    def test_result_is_immutable(self):
        r = _basic()
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        r = _basic(drug_name="Metformin")
        assert r.drug_name == "Metformin"


# ---------------------------------------------------------------------------
# Filtration clearance
# ---------------------------------------------------------------------------


class TestFiltrationClearance:
    def test_cl_filtration_equals_gfr_times_fu(self):
        fu = 0.4
        gfr = 100.0
        r = _basic(fu_plasma=fu, gfr_mL_per_min=gfr)
        assert math.isclose(r.cl_filtration_mL_per_min, gfr * fu, rel_tol=1e-9)

    def test_cl_filtration_zero_fu_boundary_rejected(self):
        with pytest.raises(ValueError):
            _basic(fu_plasma=0.0)

    def test_higher_gfr_increases_filtration(self):
        r_low = _basic(gfr_mL_per_min=60.0)
        r_high = _basic(gfr_mL_per_min=120.0)
        assert r_high.cl_filtration_mL_per_min > r_low.cl_filtration_mL_per_min


# ---------------------------------------------------------------------------
# Renal clearance direction
# ---------------------------------------------------------------------------


class TestRenalClearanceDirection:
    def test_cl_renal_positive(self):
        r = _basic()
        assert r.cl_renal_mL_per_min > 0

    def test_higher_fu_increases_cl_renal(self):
        r_low = _basic(fu_plasma=0.2)
        r_high = _basic(fu_plasma=0.8)
        assert r_high.cl_renal_mL_per_min > r_low.cl_renal_mL_per_min

    def test_higher_reabsorption_decreases_cl_renal(self):
        r_none = _basic(reabsorption_fraction=0.0)
        r_high = _basic(reabsorption_fraction=0.8)
        assert r_high.cl_renal_mL_per_min < r_none.cl_renal_mL_per_min

    def test_higher_secretion_increases_cl_renal(self):
        r_no_sec = _basic(tubular_secretion_clint_mL_per_min=0.0)
        r_sec = _basic(tubular_secretion_clint_mL_per_min=50.0)
        assert r_sec.cl_renal_mL_per_min > r_no_sec.cl_renal_mL_per_min

    def test_cl_renal_L_per_h_conversion(self):
        r = _basic()
        expected = r.cl_renal_mL_per_min * 60.0 / 1000.0
        assert math.isclose(r.cl_renal_L_per_h, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Fraction excreted
# ---------------------------------------------------------------------------


class TestFractionExcreted:
    def test_fe_in_unit_interval(self):
        r = _basic()
        assert 0.0 <= r.fe_renal <= 1.0

    def test_fe_with_explicit_nonrenal(self):
        # CL_renal_L_h known, nonrenal given → fe should be consistent
        r = _basic(
            fu_plasma=1.0,
            gfr_mL_per_min=120.0,
            nonrenal_cl_L_per_h=0.0,
        )
        # With zero nonrenal CL, fe should be 1.0
        assert math.isclose(r.fe_renal, 1.0, rel_tol=1e-6)

    def test_fe_default_nonrenal_is_3x_renal(self):
        r = _basic(fu_plasma=0.5, gfr_mL_per_min=120.0, reabsorption_fraction=0.0)
        # nonrenal = 3 * cl_renal → fe = cl_renal / (cl_renal + 3*cl_renal) = 0.25
        assert math.isclose(r.fe_renal, 0.25, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_renal_class_primary(self):
        # Force high fe by setting nonrenal to zero
        r = predict_renal_clearance(
            drug_name="PrimaryDrug",
            mw_Da=200.0,
            logP=0.0,
            pKa=7.0,
            drug_type="neutral",
            fu_plasma=1.0,
            gfr_mL_per_min=120.0,
            nonrenal_cl_L_per_h=0.0,
        )
        assert r.renal_elimination_class == "primary"
        assert r.ckd_sensitivity == "high"

    def test_renal_class_significant(self):
        # fe ~ 0.33 (default nonrenal = 3x renal gives fe=0.25; adjust nonrenal)
        r = predict_renal_clearance(
            drug_name="SigDrug",
            mw_Da=200.0,
            logP=0.0,
            pKa=7.0,
            drug_type="neutral",
            fu_plasma=1.0,
            gfr_mL_per_min=120.0,
            nonrenal_cl_L_per_h=7.2 * 1.5,  # fe ~ 0.4
        )
        assert r.renal_elimination_class == "significant"
        assert r.ckd_sensitivity == "moderate"

    def test_renal_class_minor(self):
        _basic(fu_plasma=0.1)  # noqa: F841
        # With default nonrenal = 3x renal → fe = 0.25 exactly
        # Need to confirm minor here with large nonrenal
        r2 = predict_renal_clearance(
            drug_name="MinorDrug",
            mw_Da=300.0,
            logP=1.0,
            pKa=7.0,
            drug_type="neutral",
            fu_plasma=0.1,
            gfr_mL_per_min=120.0,
            nonrenal_cl_L_per_h=100.0,
        )
        assert r2.renal_elimination_class == "minor"
        assert r2.ckd_sensitivity == "low"

    def test_valid_class_set(self):
        valid = {"primary", "significant", "minor"}
        r = _basic()
        assert r.renal_elimination_class in valid

    def test_valid_ckd_sensitivity_set(self):
        valid = {"high", "moderate", "low"}
        r = _basic()
        assert r.ckd_sensitivity in valid


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_gfr_zero_raises(self):
        with pytest.raises(ValueError, match="gfr"):
            _basic(gfr_mL_per_min=0.0)

    def test_gfr_negative_raises(self):
        with pytest.raises(ValueError, match="gfr"):
            _basic(gfr_mL_per_min=-1.0)

    def test_fu_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _basic(fu_plasma=0.0)

    def test_fu_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _basic(fu_plasma=1.5)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError, match="mw"):
            _basic(mw_Da=0.0)

    def test_reabsorption_above_one_raises(self):
        with pytest.raises(ValueError, match="reabsorption"):
            _basic(reabsorption_fraction=1.1)

    def test_reabsorption_negative_raises(self):
        with pytest.raises(ValueError, match="reabsorption"):
            _basic(reabsorption_fraction=-0.1)

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            _basic(drug_type="zwitterion")


# ---------------------------------------------------------------------------
# pH-dependent reabsorption
# ---------------------------------------------------------------------------


class TestPHDependentReabsorption:
    def test_acid_high_pka_less_ionized_at_tubular_ph(self):
        # At tubular pH=5, an acid with high pKa (e.g., 10) is mostly un-ionized
        # → ionization_factor ≈ 1 → more reabsorption → lower CL
        r_high_pka = predict_renal_clearance(
            drug_name="AcidHighPka",
            mw_Da=200.0,
            logP=1.0,
            pKa=10.0,
            drug_type="acid",
            fu_plasma=0.5,
            gfr_mL_per_min=120.0,
            reabsorption_fraction=0.5,
            tubular_pH=5.0,
        )
        # Low pKa → ionized at tubular pH → less reabsorption → higher CL
        r_low_pka = predict_renal_clearance(
            drug_name="AcidLowPka",
            mw_Da=200.0,
            logP=1.0,
            pKa=3.0,
            drug_type="acid",
            fu_plasma=0.5,
            gfr_mL_per_min=120.0,
            reabsorption_fraction=0.5,
            tubular_pH=5.0,
        )
        assert r_low_pka.cl_renal_mL_per_min > r_high_pka.cl_renal_mL_per_min

    def test_neutral_drug_unaffected_by_pka(self):
        r1 = predict_renal_clearance(
            drug_name="Neutral1",
            mw_Da=200.0,
            logP=1.0,
            pKa=5.0,
            drug_type="neutral",
            fu_plasma=0.5,
            gfr_mL_per_min=120.0,
            reabsorption_fraction=0.3,
        )
        r2 = predict_renal_clearance(
            drug_name="Neutral2",
            mw_Da=200.0,
            logP=1.0,
            pKa=9.0,
            drug_type="neutral",
            fu_plasma=0.5,
            gfr_mL_per_min=120.0,
            reabsorption_fraction=0.3,
        )
        # Both neutral → same clearance regardless of pKa
        assert math.isclose(r1.cl_renal_mL_per_min, r2.cl_renal_mL_per_min, rel_tol=1e-9)
