"""
Tests for Phase 209 — Biowaiver Eligibility Assessment
src/omega_pbpk/prediction/biowaivers_eligibility.py
"""

import pytest

from omega_pbpk.prediction.biowaivers_eligibility import (
    BiowaiverResult,
    assess_biowaiver,
    screen_formulation_variants,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bcs1_eligible(**overrides):
    """BCS I drug that meets all biowaiver criteria."""
    kwargs = dict(
        drug_name="DrugA",
        bcs_class=1,
        solubility_mg_mL=2.0,  # dose_number = 100/(2*250) = 0.2 ≤ 1
        dose_mg=100.0,
        permeability_high=True,
        dissolution_85pct_in_30min=True,
    )
    kwargs.update(overrides)
    return assess_biowaiver(**kwargs)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_biowaiver_result(self):
        result = _bcs1_eligible()
        assert isinstance(result, BiowaiverResult)

    def test_frozen_dataclass(self):
        result = _bcs1_eligible()
        with pytest.raises((AttributeError, TypeError)):
            result.eligible = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        r = _bcs1_eligible(drug_name="TestDrug")
        assert r.drug_name == "TestDrug"

    def test_bcs_class_preserved(self):
        r = _bcs1_eligible()
        assert r.bcs_class == 1

    def test_dose_mg_preserved(self):
        r = _bcs1_eligible(dose_mg=250.0)
        assert r.dose_mg == pytest.approx(250.0)

    def test_solubility_preserved(self):
        r = _bcs1_eligible(solubility_mg_mL=5.0)
        assert r.solubility_mg_mL == pytest.approx(5.0)

    def test_permeability_high_preserved(self):
        r = _bcs1_eligible(permeability_high=True)
        assert r.permeability_high is True

    def test_dissolution_flag_preserved(self):
        r = _bcs1_eligible(dissolution_85pct_in_30min=True)
        assert r.dissolution_85pct_in_30min is True


# ---------------------------------------------------------------------------
# 3. BCS Class I eligibility rules
# ---------------------------------------------------------------------------


class TestBCSClassI:
    def test_bcs1_all_criteria_met_eligible(self):
        r = _bcs1_eligible()
        assert r.eligible is True

    def test_bcs1_eligible_regulatory_path(self):
        r = _bcs1_eligible()
        assert r.regulatory_path == "no_be_study"

    def test_bcs1_eligible_confidence_high(self):
        r = _bcs1_eligible()
        assert r.confidence_level == "high"

    def test_bcs1_low_dissolution_not_eligible(self):
        r = _bcs1_eligible(dissolution_85pct_in_30min=False)
        assert r.eligible is False

    def test_bcs1_low_dissolution_full_be(self):
        r = _bcs1_eligible(dissolution_85pct_in_30min=False)
        assert r.regulatory_path == "full_be"

    def test_bcs1_low_permeability_not_eligible(self):
        r = _bcs1_eligible(permeability_high=False)
        assert r.eligible is False

    def test_bcs1_high_dose_number_not_eligible(self):
        # dose_number = 1000/(0.5*250) = 8.0 > 1.0
        r = _bcs1_eligible(dose_mg=1000.0, solubility_mg_mL=0.5)
        assert r.eligible is False


# ---------------------------------------------------------------------------
# 4. BCS Class II
# ---------------------------------------------------------------------------


class TestBCSClassII:
    def test_bcs2_not_eligible(self):
        r = assess_biowaiver(
            drug_name="DrugB",
            bcs_class=2,
            solubility_mg_mL=0.1,
            dose_mg=200.0,
            permeability_high=True,
            dissolution_85pct_in_30min=True,
        )
        assert r.eligible is False

    def test_bcs2_full_be_required(self):
        r = assess_biowaiver(
            drug_name="DrugB",
            bcs_class=2,
            solubility_mg_mL=0.1,
            dose_mg=200.0,
            permeability_high=True,
            dissolution_85pct_in_30min=True,
        )
        assert r.regulatory_path == "full_be"

    def test_bcs2_confidence_high(self):
        r = assess_biowaiver(
            drug_name="DrugB",
            bcs_class=2,
            solubility_mg_mL=0.1,
            dose_mg=200.0,
            permeability_high=True,
            dissolution_85pct_in_30min=True,
        )
        assert r.confidence_level == "high"


# ---------------------------------------------------------------------------
# 5. BCS Class III
# ---------------------------------------------------------------------------


class TestBCSClassIII:
    def test_bcs3_highly_soluble_rapid_dissolution_eligible(self):
        r = assess_biowaiver(
            drug_name="DrugC",
            bcs_class=3,
            solubility_mg_mL=10.0,  # dose_number = 50/(10*250) = 0.02
            dose_mg=50.0,
            permeability_high=False,  # low permeability is expected for III
            dissolution_85pct_in_30min=True,
        )
        assert r.eligible is True

    def test_bcs3_eligible_regulatory_path(self):
        r = assess_biowaiver(
            drug_name="DrugC",
            bcs_class=3,
            solubility_mg_mL=10.0,
            dose_mg=50.0,
            permeability_high=False,
            dissolution_85pct_in_30min=True,
        )
        assert r.regulatory_path == "no_be_study"

    def test_bcs3_slow_dissolution_not_eligible(self):
        r = assess_biowaiver(
            drug_name="DrugC",
            bcs_class=3,
            solubility_mg_mL=10.0,
            dose_mg=50.0,
            permeability_high=False,
            dissolution_85pct_in_30min=False,
        )
        assert r.eligible is False


# ---------------------------------------------------------------------------
# 6. BCS Class IV
# ---------------------------------------------------------------------------


class TestBCSClassIV:
    def test_bcs4_not_eligible(self):
        r = assess_biowaiver(
            drug_name="DrugD",
            bcs_class=4,
            solubility_mg_mL=0.05,
            dose_mg=100.0,
            permeability_high=False,
            dissolution_85pct_in_30min=False,
        )
        assert r.eligible is False

    def test_bcs4_full_be_required(self):
        r = assess_biowaiver(
            drug_name="DrugD",
            bcs_class=4,
            solubility_mg_mL=0.05,
            dose_mg=100.0,
            permeability_high=False,
            dissolution_85pct_in_30min=False,
        )
        assert r.regulatory_path == "full_be"


# ---------------------------------------------------------------------------
# 7. Dose number computation
# ---------------------------------------------------------------------------


class TestDoseNumber:
    def test_dose_number_correct(self):
        # Do = 500 / (4.0 * 250) = 0.5
        r = assess_biowaiver(
            drug_name="X",
            bcs_class=1,
            solubility_mg_mL=4.0,
            dose_mg=500.0,
            permeability_high=True,
            dissolution_85pct_in_30min=True,
        )
        assert r.dose_number == pytest.approx(0.5, rel=1e-6)

    def test_dose_number_boundary_exactly_1(self):
        # Do = 250 / (1.0 * 250) = 1.0 (boundary: still highly soluble)
        r = assess_biowaiver(
            drug_name="X",
            bcs_class=1,
            solubility_mg_mL=1.0,
            dose_mg=250.0,
            permeability_high=True,
            dissolution_85pct_in_30min=True,
        )
        assert r.dose_number == pytest.approx(1.0, rel=1e-6)
        assert r.eligible is True

    def test_dose_number_above_1_not_highly_soluble(self):
        # Do = 300 / (1.0 * 250) = 1.2 → not highly soluble
        r = assess_biowaiver(
            drug_name="X",
            bcs_class=1,
            solubility_mg_mL=1.0,
            dose_mg=300.0,
            permeability_high=True,
            dissolution_85pct_in_30min=True,
        )
        assert r.dose_number > 1.0
        assert r.eligible is False


# ---------------------------------------------------------------------------
# 8. Regulatory path valid values
# ---------------------------------------------------------------------------


class TestRegulatoryPath:
    _valid = {"no_be_study", "reduced_be", "full_be"}

    def test_bcs1_eligible_valid_path(self):
        assert _bcs1_eligible().regulatory_path in self._valid

    def test_bcs2_valid_path(self):
        r = assess_biowaiver("D", 2, 0.1, 50.0, True, True)
        assert r.regulatory_path in self._valid

    def test_bcs3_eligible_valid_path(self):
        r = assess_biowaiver("D", 3, 5.0, 50.0, False, True)
        assert r.regulatory_path in self._valid

    def test_bcs4_valid_path(self):
        r = assess_biowaiver("D", 4, 0.05, 50.0, False, False)
        assert r.regulatory_path in self._valid


# ---------------------------------------------------------------------------
# 9. screen_formulation_variants
# ---------------------------------------------------------------------------


class TestScreenFormulationVariants:
    def test_correct_count(self):
        doses = [50.0, 100.0, 200.0, 400.0]
        results = screen_formulation_variants(
            drug_name="DrugE",
            bcs_class=1,
            solubility_mg_mL=5.0,
            dose_options_mg=doses,
            target_dissolution_pct=90.0,
        )
        assert len(results) == len(doses)

    def test_all_returns_are_biowaiver_result(self):
        results = screen_formulation_variants(
            drug_name="DrugE",
            bcs_class=1,
            solubility_mg_mL=5.0,
            dose_options_mg=[100.0, 200.0],
            target_dissolution_pct=90.0,
        )
        for r in results:
            assert isinstance(r, BiowaiverResult)

    def test_order_matches_dose_options(self):
        doses = [50.0, 200.0, 800.0]
        results = screen_formulation_variants(
            drug_name="DrugE",
            bcs_class=1,
            solubility_mg_mL=5.0,
            dose_options_mg=doses,
            target_dissolution_pct=90.0,
        )
        for r, d in zip(results, doses):
            assert r.dose_mg == pytest.approx(d)

    def test_high_dose_becomes_ineligible(self):
        # At solubility=1.0 mg/mL: Do=1250/(1*250)=5 → not eligible
        results = screen_formulation_variants(
            drug_name="DrugF",
            bcs_class=1,
            solubility_mg_mL=1.0,
            dose_options_mg=[50.0, 1250.0],
            target_dissolution_pct=90.0,
        )
        assert results[0].eligible is True  # Do=0.2
        assert results[1].eligible is False  # Do=5.0

    def test_bcs2_all_ineligible(self):
        results = screen_formulation_variants(
            drug_name="DrugG",
            bcs_class=2,
            solubility_mg_mL=0.1,
            dose_options_mg=[100.0, 200.0, 400.0],
            target_dissolution_pct=95.0,
        )
        assert all(not r.eligible for r in results)


# ---------------------------------------------------------------------------
# 10. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_bcs_class_raises(self):
        with pytest.raises(ValueError, match="bcs_class"):
            assess_biowaiver("X", 5, 1.0, 100.0, True, True)

    def test_bcs_class_zero_raises(self):
        with pytest.raises(ValueError, match="bcs_class"):
            assess_biowaiver("X", 0, 1.0, 100.0, True, True)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            assess_biowaiver("X", 1, 1.0, 0.0, True, True)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            assess_biowaiver("X", 1, 1.0, -10.0, True, True)

    def test_solubility_zero_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            assess_biowaiver("X", 1, 0.0, 100.0, True, True)

    def test_solubility_negative_raises(self):
        with pytest.raises(ValueError, match="solubility_mg_mL"):
            assess_biowaiver("X", 1, -1.0, 100.0, True, True)
