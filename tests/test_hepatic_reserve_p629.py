"""Tests for Phase 629 — Drug Hepatic Reserve Assessment."""

import pytest

from omega_pbpk.clinical.hepatic_reserve_p629 import (
    HepaticReserveResult,
    assess_hepatic_reserve,
    dose_recommendation_by_cp_class,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DRUG_A = "DrugA"  # normal extraction
DRUG_B = "DrugB"  # high extraction
CL_NORMAL = 20.0  # L/h
ER_LOW = 0.15
ER_HIGH = 0.85

CHILD_A_PARAMS = dict(
    bilirubin_umol_L=20.0,  # < 34 → 1pt
    albumin_g_dL=38.0,  # > 35 → 1pt
    inr=1.2,  # < 1.7 → 1pt
    ascites="none",  # → 1pt
    encephalopathy_grade=0,  # → 1pt  total = 5
)

CHILD_B_PARAMS = dict(
    bilirubin_umol_L=40.0,  # 34-50 → 2pt
    albumin_g_dL=30.0,  # 28-35 → 2pt
    inr=2.0,  # 1.7-2.3 → 2pt
    ascites="mild",  # → 2pt
    encephalopathy_grade=1,  # → 2pt  total = 10 → but we need 7-9 so adjust
    # Actually total = 10 → class C. Use milder params:
)

CHILD_B_PARAMS = dict(
    bilirubin_umol_L=40.0,  # 34-50 → 2pt
    albumin_g_dL=30.0,  # 28-35 → 2pt
    inr=1.5,  # < 1.7 → 1pt
    ascites="mild",  # → 2pt
    encephalopathy_grade=1,  # → 2pt  total = 9 → class B
)

CHILD_C_PARAMS = dict(
    bilirubin_umol_L=60.0,  # > 50 → 3pt
    albumin_g_dL=25.0,  # < 28 → 3pt
    inr=2.5,  # > 2.3 → 3pt
    ascites="severe",  # → 3pt
    encephalopathy_grade=2,  # → 3pt  total = 15 → class C
)


def _run(params, drug=DRUG_A, cl=CL_NORMAL, er=ER_LOW):
    return assess_hepatic_reserve(drug, cl, er, **params)


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_hepatic_reserve_result(self):
        r = _run(CHILD_A_PARAMS)
        assert isinstance(r, HepaticReserveResult)

    def test_frozen_dataclass_cannot_mutate(self):
        r = _run(CHILD_A_PARAMS)
        with pytest.raises((AttributeError, TypeError)):
            r.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        r = _run(CHILD_A_PARAMS, drug="Tacrolimus")
        assert r.drug_name == "Tacrolimus"


# ---------------------------------------------------------------------------
# Child-Pugh Score Calculation
# ---------------------------------------------------------------------------


class TestChildPughScoring:
    def test_class_a_score_5(self):
        r = _run(CHILD_A_PARAMS)
        assert r.child_pugh_score == 5
        assert r.child_pugh_class == "A"

    def test_class_b_score_9(self):
        r = _run(CHILD_B_PARAMS)
        assert r.child_pugh_score == 9
        assert r.child_pugh_class == "B"

    def test_class_c_score_15(self):
        r = _run(CHILD_C_PARAMS)
        assert r.child_pugh_score == 15
        assert r.child_pugh_class == "C"

    def test_class_a_max_score_6(self):
        params = dict(
            bilirubin_umol_L=33.0,  # < 34 → 1pt
            albumin_g_dL=36.0,  # > 35 → 1pt
            inr=1.6,  # < 1.7 → 1pt
            ascites="mild",  # → 2pt
            encephalopathy_grade=1,  # → 2pt  total = 7 → B
        )
        r = _run(params)
        assert r.child_pugh_class == "B"

    def test_bilirubin_boundary_34(self):
        """bilirubin == 34 → 2pt (34-50 range, inclusive)."""
        params = dict(**CHILD_A_PARAMS)
        params["bilirubin_umol_L"] = 34.0
        r = _run(params)
        assert r.child_pugh_score == 6  # was 5, now bilirubin adds 2 instead of 1

    def test_inr_boundary_1_7(self):
        """INR == 1.7 → 2pt (1.7-2.3 range)."""
        params = dict(**CHILD_A_PARAMS)
        params["inr"] = 1.7
        r = _run(params)
        assert r.child_pugh_score == 6  # 1 + 1 + 2 + 1 + 1


# ---------------------------------------------------------------------------
# MELD Score
# ---------------------------------------------------------------------------


class TestMELDScore:
    def test_meld_class_a(self):
        r = _run(CHILD_A_PARAMS)
        assert 6.0 <= r.meld_score <= 40.0

    def test_meld_class_c_higher_than_a(self):
        r_a = _run(CHILD_A_PARAMS)
        r_c = _run(CHILD_C_PARAMS)
        assert r_c.meld_score > r_a.meld_score

    def test_meld_minimum_clamped_to_6(self):
        # Very low bilirubin/INR/creatinine should not go below 6
        params = dict(**CHILD_A_PARAMS)
        r = _run(params)
        assert r.meld_score >= 6.0

    def test_meld_maximum_clamped_to_40(self):
        params = dict(**CHILD_C_PARAMS)
        r = _run(params, cl=CL_NORMAL, er=ER_HIGH)
        assert r.meld_score <= 40.0


# ---------------------------------------------------------------------------
# CL Reduction and Dose Factors
# ---------------------------------------------------------------------------


class TestCLReduction:
    def test_cl_reduction_class_a_75pct(self):
        r = _run(CHILD_A_PARAMS)
        assert abs(r.cl_reduction_factor - 0.75) < 1e-9
        assert abs(r.cl_hepatic_reduced - CL_NORMAL * 0.75) < 1e-9

    def test_cl_reduction_class_b_50pct(self):
        r = _run(CHILD_B_PARAMS)
        assert abs(r.cl_reduction_factor - 0.50) < 1e-9
        assert abs(r.cl_hepatic_reduced - CL_NORMAL * 0.50) < 1e-9

    def test_cl_reduction_class_c_25pct(self):
        r = _run(CHILD_C_PARAMS)
        assert abs(r.cl_reduction_factor - 0.25) < 1e-9
        assert abs(r.cl_hepatic_reduced - CL_NORMAL * 0.25) < 1e-9

    def test_dose_factor_equals_reduction_factor(self):
        r = _run(CHILD_B_PARAMS)
        assert abs(r.recommended_dose_factor - r.cl_reduction_factor) < 1e-9


# ---------------------------------------------------------------------------
# Contraindication Logic
# ---------------------------------------------------------------------------


class TestContraindication:
    def test_not_contraindicated_class_a_high_er(self):
        r = _run(CHILD_A_PARAMS, er=ER_HIGH)
        assert r.contraindicated is False

    def test_not_contraindicated_class_b_high_er(self):
        r = _run(CHILD_B_PARAMS, er=ER_HIGH)
        assert r.contraindicated is False

    def test_contraindicated_class_c_high_er(self):
        r = _run(CHILD_C_PARAMS, er=ER_HIGH)
        assert r.contraindicated is True

    def test_not_contraindicated_class_c_low_er(self):
        r = _run(CHILD_C_PARAMS, er=ER_LOW)
        assert r.contraindicated is False

    def test_not_contraindicated_class_c_er_exactly_0_6(self):
        """ER == 0.6: not contraindicated (must be > 0.6)."""
        r = _run(CHILD_C_PARAMS, er=0.6)
        assert r.contraindicated is False

    def test_contraindicated_class_c_er_0_61(self):
        r = _run(CHILD_C_PARAMS, er=0.61)
        assert r.contraindicated is True


# ---------------------------------------------------------------------------
# Exposure Increase
# ---------------------------------------------------------------------------


class TestExposureIncrease:
    def test_exposure_increase_class_a(self):
        r = _run(CHILD_A_PARAMS)
        assert abs(r.exposure_increase_fold - 1.0 / 0.75) < 1e-4

    def test_exposure_increase_class_b(self):
        r = _run(CHILD_B_PARAMS)
        assert abs(r.exposure_increase_fold - 2.0) < 1e-4

    def test_exposure_increase_class_c(self):
        r = _run(CHILD_C_PARAMS)
        assert abs(r.exposure_increase_fold - 4.0) < 1e-4

    def test_exposure_greater_than_1(self):
        for params in (CHILD_A_PARAMS, CHILD_B_PARAMS, CHILD_C_PARAMS):
            r = _run(params)
            assert r.exposure_increase_fold >= 1.0


# ---------------------------------------------------------------------------
# Monitoring Classification
# ---------------------------------------------------------------------------


class TestMonitoringRequired:
    def test_class_a_standard_monitoring(self):
        r = _run(CHILD_A_PARAMS)
        assert r.monitoring_required == "standard"

    def test_class_b_intensive_monitoring(self):
        r = _run(CHILD_B_PARAMS)
        assert r.monitoring_required == "intensive"

    def test_class_c_low_er_intensive(self):
        r = _run(CHILD_C_PARAMS, er=ER_LOW)
        assert r.monitoring_required == "intensive"

    def test_class_c_high_er_avoid(self):
        r = _run(CHILD_C_PARAMS, er=ER_HIGH)
        assert r.monitoring_required == "avoid"


# ---------------------------------------------------------------------------
# dose_recommendation_by_cp_class
# ---------------------------------------------------------------------------


class TestDoseRecommendationByCPClass:
    def test_returns_dict_with_abc_keys(self):
        d = dose_recommendation_by_cp_class(DRUG_A, CL_NORMAL, ER_LOW)
        assert set(d.keys()) == {"A", "B", "C"}

    def test_class_a_factor_0_75(self):
        d = dose_recommendation_by_cp_class(DRUG_A, CL_NORMAL, ER_LOW)
        assert abs(d["A"] - 0.75) < 1e-9

    def test_class_b_factor_0_50(self):
        d = dose_recommendation_by_cp_class(DRUG_A, CL_NORMAL, ER_LOW)
        assert abs(d["B"] - 0.50) < 1e-9

    def test_class_c_factor_0_25(self):
        d = dose_recommendation_by_cp_class(DRUG_A, CL_NORMAL, ER_LOW)
        assert abs(d["C"] - 0.25) < 1e-9

    def test_monotonically_decreasing(self):
        d = dose_recommendation_by_cp_class(DRUG_A, CL_NORMAL, ER_LOW)
        assert d["A"] > d["B"] > d["C"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_bilirubin_raises(self):
        params = dict(**CHILD_A_PARAMS)
        params["bilirubin_umol_L"] = -1.0
        with pytest.raises(ValueError, match="bilirubin_umol_L"):
            _run(params)

    def test_zero_albumin_raises(self):
        params = dict(**CHILD_A_PARAMS)
        params["albumin_g_dL"] = 0.0
        with pytest.raises(ValueError, match="albumin_g_dL"):
            _run(params)

    def test_invalid_ascites_raises(self):
        params = dict(**CHILD_A_PARAMS)
        params["ascites"] = "moderate"
        with pytest.raises(ValueError, match="ascites"):
            _run(params)

    def test_invalid_encephalopathy_grade_raises(self):
        params = dict(**CHILD_A_PARAMS)
        params["encephalopathy_grade"] = 3
        with pytest.raises(ValueError, match="encephalopathy_grade"):
            _run(params)

    def test_negative_inr_raises(self):
        params = dict(**CHILD_A_PARAMS)
        params["inr"] = -0.5
        with pytest.raises(ValueError, match="inr"):
            _run(params)
