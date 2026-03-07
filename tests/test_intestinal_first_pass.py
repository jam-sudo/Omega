"""Tests for intestinal first-pass metabolism predictor (Phase 297)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.intestinal_first_pass import (
    IntestinalFirstPassResult,
    predict_intestinal_first_pass,
    screen_substrates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_drug(
    drug_name="TestDrug",
    logp=2.0,
    mw=350.0,
    psa_A2=60.0,
    pka=8.0,
    cyp3a4_substrate=False,
    pgp_substrate=False,
    cl_int_gut_mL_per_min=10.0,
    qgut_L_per_h=18.0,
) -> IntestinalFirstPassResult:
    return predict_intestinal_first_pass(
        drug_name=drug_name,
        logp=logp,
        mw=mw,
        psa_A2=psa_A2,
        pka=pka,
        cyp3a4_substrate=cyp3a4_substrate,
        pgp_substrate=pgp_substrate,
        cl_int_gut_mL_per_min=cl_int_gut_mL_per_min,
        qgut_L_per_h=qgut_L_per_h,
    )


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _make_drug()
        assert isinstance(result, IntestinalFirstPassResult)

    def test_drug_name_preserved(self):
        result = _make_drug(drug_name="Midazolam")
        assert result.drug_name == "Midazolam"

    def test_logp_preserved(self):
        result = _make_drug(logp=3.5)
        assert result.logp == 3.5

    def test_mw_preserved(self):
        result = _make_drug(mw=400.0)
        assert result.mw == 400.0

    def test_psa_preserved(self):
        result = _make_drug(psa_A2=80.0)
        assert result.psa_A2 == 80.0

    def test_cyp3a4_substrate_preserved(self):
        result = _make_drug(cyp3a4_substrate=True)
        assert result.cyp3a4_substrate is True

    def test_pgp_substrate_preserved(self):
        result = _make_drug(pgp_substrate=True)
        assert result.pgp_substrate is True

    def test_result_is_frozen(self):
        result = _make_drug()
        with pytest.raises((AttributeError, TypeError)):
            result.fa = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Physiological range checks
# ---------------------------------------------------------------------------


class TestPhysiologicalRanges:
    def test_fa_in_range(self):
        result = _make_drug()
        assert 0.0 <= result.fa <= 1.0

    def test_fg_in_range(self):
        result = _make_drug()
        assert 0.0 <= result.fg <= 1.0

    def test_f_oral_in_range(self):
        result = _make_drug()
        assert 0.0 <= result.f_oral_estimate <= 1.0

    def test_eg_cyp_non_negative_without_substrate(self):
        result = _make_drug(cyp3a4_substrate=False)
        assert result.eg_cyp == 0.0

    def test_eg_cyp_positive_with_substrate(self):
        result = _make_drug(cyp3a4_substrate=True, cl_int_gut_mL_per_min=30.0)
        assert result.eg_cyp > 0.0

    def test_pgp_factor_no_substrate(self):
        result = _make_drug(pgp_substrate=False)
        assert result.pgp_factor == 1.0

    def test_pgp_factor_substrate(self):
        result = _make_drug(pgp_substrate=True)
        assert result.pgp_factor == 0.7

    def test_first_pass_gut_pct_non_negative(self):
        result = _make_drug()
        assert result.first_pass_gut_pct >= 0.0

    def test_first_pass_gut_pct_le_100(self):
        result = _make_drug(cyp3a4_substrate=True, pgp_substrate=True)
        assert result.first_pass_gut_pct <= 100.0


# ---------------------------------------------------------------------------
# CYP3A4 and P-gp substrate effects
# ---------------------------------------------------------------------------


class TestSubstrateEffects:
    def test_cyp3a4_substrate_lower_fg(self):
        no_cyp = _make_drug(cyp3a4_substrate=False, pgp_substrate=False)
        with_cyp = _make_drug(
            cyp3a4_substrate=True, pgp_substrate=False, cl_int_gut_mL_per_min=30.0
        )
        assert with_cyp.fg < no_cyp.fg

    def test_pgp_substrate_lower_fg(self):
        no_pgp = _make_drug(pgp_substrate=False, cyp3a4_substrate=False)
        with_pgp = _make_drug(pgp_substrate=True, cyp3a4_substrate=False)
        assert with_pgp.fg < no_pgp.fg

    def test_both_substrates_lowest_fg(self):
        neither = _make_drug(cyp3a4_substrate=False, pgp_substrate=False)
        both = _make_drug(cyp3a4_substrate=True, pgp_substrate=True, cl_int_gut_mL_per_min=30.0)
        assert both.fg <= neither.fg

    def test_no_cyp_substrate_no_gut_extraction(self):
        result = _make_drug(cyp3a4_substrate=False)
        assert result.eg_cyp == 0.0

    def test_high_clint_increases_extraction(self):
        low_cl = _make_drug(cyp3a4_substrate=True, cl_int_gut_mL_per_min=5.0)
        high_cl = _make_drug(cyp3a4_substrate=True, cl_int_gut_mL_per_min=50.0)
        assert high_cl.eg_cyp > low_cl.eg_cyp


# ---------------------------------------------------------------------------
# Bioavailability class
# ---------------------------------------------------------------------------


class TestBioavailabilityClass:
    _VALID_CLASSES = {"high", "moderate", "low"}

    def test_class_in_valid_set(self):
        result = _make_drug()
        assert result.bioavailability_class in self._VALID_CLASSES

    def test_high_logp_no_substrates_high_class(self):
        # High logP → high Peff → high fa; no gut extraction → high fg
        result = _make_drug(logp=4.0, psa_A2=30.0, cyp3a4_substrate=False, pgp_substrate=False)
        assert result.bioavailability_class in {"high", "moderate"}

    def test_cyp3a4_pgp_combined_low_class(self):
        result = _make_drug(
            cyp3a4_substrate=True,
            pgp_substrate=True,
            cl_int_gut_mL_per_min=50.0,
        )
        assert result.bioavailability_class in {"low", "moderate"}


# ---------------------------------------------------------------------------
# Dominant mechanism
# ---------------------------------------------------------------------------


class TestDominantMechanism:
    _VALID_MECHANISMS = {"cyp3a4", "pgp_efflux", "minimal"}

    def test_dominant_mechanism_in_valid_set(self):
        result = _make_drug()
        assert result.dominant_mechanism in self._VALID_MECHANISMS

    def test_high_cyp_extraction_dominant(self):
        result = _make_drug(
            cyp3a4_substrate=True,
            pgp_substrate=False,
            cl_int_gut_mL_per_min=100.0,
            qgut_L_per_h=18.0,
        )
        assert result.dominant_mechanism == "cyp3a4"

    def test_pgp_only_mechanism(self):
        result = _make_drug(
            cyp3a4_substrate=False,
            pgp_substrate=True,
        )
        assert result.dominant_mechanism == "pgp_efflux"

    def test_no_substrates_minimal(self):
        result = _make_drug(cyp3a4_substrate=False, pgp_substrate=False)
        assert result.dominant_mechanism == "minimal"


# ---------------------------------------------------------------------------
# screen_substrates
# ---------------------------------------------------------------------------


class TestScreenSubstrates:
    def _make_list(self):
        return [
            {
                "drug_name": "DrugA",
                "logp": 1.0,
                "mw": 300.0,
                "psa_A2": 80.0,
                "pka": 7.0,
                "cyp3a4_substrate": True,
                "pgp_substrate": True,
                "cl_int_gut_mL_per_min": 40.0,
            },
            {
                "drug_name": "DrugB",
                "logp": 3.0,
                "mw": 350.0,
                "psa_A2": 40.0,
                "pka": 8.0,
                "cyp3a4_substrate": False,
                "pgp_substrate": False,
            },
            {
                "drug_name": "DrugC",
                "logp": 2.0,
                "mw": 320.0,
                "psa_A2": 60.0,
                "pka": 9.0,
                "cyp3a4_substrate": True,
                "pgp_substrate": False,
                "cl_int_gut_mL_per_min": 20.0,
            },
        ]

    def test_returns_list(self):
        results = screen_substrates(self._make_list())
        assert isinstance(results, list)

    def test_returns_all_results(self):
        results = screen_substrates(self._make_list())
        assert len(results) == 3

    def test_sorted_fg_ascending(self):
        results = screen_substrates(self._make_list())
        fg_values = [r.fg for r in results]
        assert fg_values == sorted(fg_values)

    def test_all_results_are_correct_type(self):
        results = screen_substrates(self._make_list())
        for r in results:
            assert isinstance(r, IntestinalFirstPassResult)

    def test_empty_list_returns_empty(self):
        results = screen_substrates([])
        assert results == []


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_logp_too_low(self):
        with pytest.raises(ValueError, match="logp"):
            _make_drug(logp=-6.0)

    def test_logp_too_high(self):
        with pytest.raises(ValueError, match="logp"):
            _make_drug(logp=11.0)

    def test_mw_zero(self):
        with pytest.raises(ValueError, match="mw"):
            _make_drug(mw=0.0)

    def test_mw_negative(self):
        with pytest.raises(ValueError, match="mw"):
            _make_drug(mw=-100.0)

    def test_psa_negative(self):
        with pytest.raises(ValueError, match="psa_A2"):
            _make_drug(psa_A2=-1.0)

    def test_clint_negative(self):
        with pytest.raises(ValueError, match="cl_int_gut"):
            _make_drug(cl_int_gut_mL_per_min=-5.0)

    def test_qgut_zero(self):
        with pytest.raises(ValueError, match="qgut"):
            _make_drug(qgut_L_per_h=0.0)
