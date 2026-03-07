"""Tests for Phase 199 — cardiac_pk_p199.py."""

import pytest

from omega_pbpk.core.cardiac_pk_p199 import (
    CardiacPKResult,
    assess_cardiotoxicity_risk,
    simulate_cardiac_pk,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> CardiacPKResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_plasma_L=20.0,
        logP=2.0,
    )
    defaults.update(kwargs)
    return simulate_cardiac_pk(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_cardiac_pk_result(self):
        result = _default_result()
        assert isinstance(result, CardiacPKResult)

    def test_drug_name_preserved(self):
        result = _default_result(drug_name="Amiodarone")
        assert result.drug_name == "Amiodarone"

    def test_dose_mg_preserved(self):
        result = _default_result(dose_mg=200.0)
        assert result.dose_mg == 200.0

    def test_logP_preserved(self):
        result = _default_result(logP=3.5)
        assert result.logP == 3.5

    def test_times_h_is_list(self):
        result = _default_result()
        assert isinstance(result.times_h, list)

    def test_c_plasma_mg_L_is_list(self):
        result = _default_result()
        assert isinstance(result.c_plasma_mg_L, list)

    def test_c_cardiac_mg_L_is_list(self):
        result = _default_result()
        assert isinstance(result.c_cardiac_mg_L, list)

    def test_notes_is_str(self):
        result = _default_result()
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# ODE initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_cardiac_starts_at_zero(self):
        result = _default_result()
        assert result.c_cardiac_mg_L[0] == pytest.approx(0.0)

    def test_c_plasma_starts_at_dose_over_vd(self):
        result = _default_result(dose_mg=100.0, vd_plasma_L=20.0)
        expected = 100.0 / 20.0  # 5 mg/L
        assert result.c_plasma_mg_L[0] == pytest.approx(expected, rel=1e-4)

    def test_times_h_starts_at_zero(self):
        result = _default_result()
        assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PK metrics
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_cmax_cardiac_positive(self):
        result = _default_result()
        assert result.cmax_cardiac_mg_L > 0.0

    def test_auc_cardiac_positive(self):
        result = _default_result()
        assert result.auc_cardiac_mg_h_per_L > 0.0

    def test_auc_plasma_positive(self):
        result = _default_result()
        assert result.auc_plasma_mg_h_per_L > 0.0

    def test_cardiac_to_plasma_auc_ratio_positive(self):
        result = _default_result()
        assert result.cardiac_to_plasma_auc_ratio > 0.0

    def test_tmax_cardiac_non_negative(self):
        result = _default_result()
        assert result.tmax_cardiac_h >= 0.0

    def test_list_lengths_consistent(self):
        result = _default_result(t_end_h=12.0)
        assert len(result.times_h) == len(result.c_plasma_mg_L)
        assert len(result.times_h) == len(result.c_cardiac_mg_L)


# ---------------------------------------------------------------------------
# logP effect on cardiac distribution
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logP_increases_cardiac_distribution(self):
        low_logP = _default_result(logP=0.5)
        high_logP = _default_result(logP=4.0)
        assert high_logP.cardiac_to_plasma_auc_ratio > low_logP.cardiac_to_plasma_auc_ratio

    def test_high_logP_increases_cmax_cardiac(self):
        low = _default_result(logP=0.5)
        high = _default_result(logP=4.0)
        assert high.cmax_cardiac_mg_L > low.cmax_cardiac_mg_L

    def test_lipophilicity_factor_clamped_at_low_logP(self):
        # logP=-2 → lipophilicity_factor=0.5 (floor); should still run
        result = _default_result(logP=-2.0)
        assert result.cmax_cardiac_mg_L > 0.0

    def test_lipophilicity_factor_clamped_at_high_logP(self):
        # logP=10 → lipophilicity_factor=3.0 (ceiling)
        result = _default_result(logP=10.0)
        assert result.cmax_cardiac_mg_L > 0.0


# ---------------------------------------------------------------------------
# assess_cardiotoxicity_risk
# ---------------------------------------------------------------------------


class TestAssessCardiotoxicityRisk:
    def test_returns_dict(self):
        result = _default_result()
        risk = assess_cardiotoxicity_risk(result)
        assert isinstance(risk, dict)

    def test_expected_keys(self):
        result = _default_result()
        risk = assess_cardiotoxicity_risk(result)
        assert "risk_level" in risk
        assert "cmax_cardiac_mg_L" in risk
        assert "cardiac_to_plasma_ratio" in risk
        assert "recommendation" in risk

    def test_risk_level_valid_values(self):
        result = _default_result()
        risk = assess_cardiotoxicity_risk(result)
        assert risk["risk_level"] in ("low", "moderate", "high")

    def test_high_ratio_gives_high_risk(self):
        # Simulate with very high logP to push ratio > 2.0
        result = _default_result(
            logP=8.0,
            dose_mg=1000.0,
            vd_cardiac_L=0.1,
            t_end_h=48.0,
        )
        risk = assess_cardiotoxicity_risk(result)
        assert risk["risk_level"] == "high"

    def test_low_distribution_gives_low_or_moderate_risk(self):
        # Low logP, small dose, large vd_cardiac → low/moderate cardiac accumulation
        result = _default_result(
            logP=-1.0,
            dose_mg=10.0,
            cl_L_per_h=20.0,
            vd_cardiac_L=5.0,
        )
        risk = assess_cardiotoxicity_risk(result)
        assert risk["risk_level"] in ("low", "moderate")

    def test_cmax_cardiac_matches_result(self):
        result = _default_result()
        risk = assess_cardiotoxicity_risk(result)
        assert risk["cmax_cardiac_mg_L"] == pytest.approx(result.cmax_cardiac_mg_L)

    def test_cardiac_to_plasma_ratio_matches_result(self):
        result = _default_result()
        risk = assess_cardiotoxicity_risk(result)
        assert risk["cardiac_to_plasma_ratio"] == pytest.approx(result.cardiac_to_plasma_auc_ratio)

    def test_recommendation_is_str(self):
        result = _default_result()
        risk = assess_cardiotoxicity_risk(result)
        assert isinstance(risk["recommendation"], str)
        assert len(risk["recommendation"]) > 0


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


class TestValidation:
    def test_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_cardiac_pk("X", 0.0, 5.0, 20.0, 2.0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_cardiac_pk("X", -50.0, 5.0, 20.0, 2.0)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_cardiac_pk("X", 100.0, 0.0, 20.0, 2.0)

    def test_vd_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="vd_plasma_L"):
            simulate_cardiac_pk("X", 100.0, 5.0, 0.0, 2.0)

    def test_vd_cardiac_zero_raises(self):
        with pytest.raises(ValueError, match="vd_cardiac_L"):
            simulate_cardiac_pk("X", 100.0, 5.0, 20.0, 2.0, vd_cardiac_L=0.0)
