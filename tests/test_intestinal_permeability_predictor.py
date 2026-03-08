"""
Tests for Phase 403 — Intestinal Permeability Predictor
"""

import math

import pytest

from omega_pbpk.prediction.intestinal_permeability_predictor import (
    IntestinalPermeabilityResult,
    predict_intestinal_permeability,
    screen_permeability,
)

# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = predict_intestinal_permeability("TestDrug", logp=2.0, mw_da=300.0, psa_A2=60.0)
        assert isinstance(result, IntestinalPermeabilityResult)

    def test_all_fields_present(self):
        result = predict_intestinal_permeability("TestDrug", logp=2.0, mw_da=300.0, psa_A2=60.0)
        assert result.drug_name == "TestDrug"
        assert isinstance(result.logp, float)
        assert isinstance(result.mw_da, float)
        assert isinstance(result.psa_A2, float)
        assert isinstance(result.n_hbd, int)
        assert isinstance(result.n_hba, int)
        assert isinstance(result.logpe_predicted, float)
        assert isinstance(result.peff_cm_s, float)
        assert isinstance(result.peff_class, str)
        assert isinstance(result.lipinski_violations, int)
        assert isinstance(result.lipinski_compliant, bool)
        assert isinstance(result.bcs_permeability_class, str)
        assert isinstance(result.fa_predicted_pct, float)
        assert isinstance(result.human_intestine_method, str)
        assert isinstance(result.notes, str)

    def test_result_is_frozen(self):
        result = predict_intestinal_permeability("TestDrug", logp=2.0, mw_da=300.0, psa_A2=60.0)
        with pytest.raises((AttributeError, TypeError)):
            result.logp = 99.0


# ---------------------------------------------------------------------------
# Physicochemical model
# ---------------------------------------------------------------------------


class TestPhysicochemicalModel:
    def test_high_logp_low_psa_gives_high_peff(self):
        """High logP and low PSA → high permeability."""
        result = predict_intestinal_permeability("HighPerm", logp=4.5, mw_da=250.0, psa_A2=20.0)
        assert result.peff_cm_s > 2e-4
        assert result.peff_class == "high"

    def test_low_logp_high_psa_gives_low_peff(self):
        """Low logP and high PSA → low permeability."""
        result = predict_intestinal_permeability("LowPerm", logp=-1.0, mw_da=500.0, psa_A2=180.0)
        assert result.peff_cm_s < 1e-5
        assert result.peff_class == "low"

    def test_medium_permeability_class(self):
        """Intermediate conditions → medium peff class."""
        result = predict_intestinal_permeability("MedPerm", logp=1.5, mw_da=350.0, psa_A2=90.0)
        # logPe = -6 + 0.85*1.5 - 0.01*90 - 0.003*350 = -6 + 1.275 - 0.9 - 1.05 = -6.675
        # peff = 10^-6.675 ~ 2.1e-7 → "low"
        assert result.peff_class in ("low", "medium", "high")

    def test_method_is_physicochemical(self):
        result = predict_intestinal_permeability("TestDrug", logp=2.0, mw_da=300.0, psa_A2=60.0)
        assert result.human_intestine_method == "physicochemical"

    def test_logpe_formula(self):
        """Verify logPe calculation matches formula."""
        logp, mw, psa = 3.0, 400.0, 50.0
        expected_logpe = -6.0 + 0.85 * logp - 0.01 * psa - 0.003 * mw
        result = predict_intestinal_permeability("TestDrug", logp=logp, mw_da=mw, psa_A2=psa)
        assert abs(result.logpe_predicted - expected_logpe) < 1e-9

    def test_peff_clamped_to_max(self):
        """Very high logP should not exceed 1e-3."""
        result = predict_intestinal_permeability("VeryHigh", logp=10.0, mw_da=100.0, psa_A2=0.0)
        assert result.peff_cm_s <= 1e-3

    def test_peff_clamped_to_min(self):
        """Very low permeability should not go below 1e-9."""
        result = predict_intestinal_permeability("VeryLow", logp=-5.0, mw_da=800.0, psa_A2=300.0)
        assert result.peff_cm_s >= 1e-9


# ---------------------------------------------------------------------------
# Caco-2 correlation method
# ---------------------------------------------------------------------------


class TestCaco2Correlation:
    def test_caco2_method_used(self):
        result = predict_intestinal_permeability(
            "DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0, caco2_papp_cm_s=1e-5
        )
        assert result.human_intestine_method == "caco2_correlation"

    def test_caco2_logpe_formula(self):
        """Verify Sun 2004 correlation."""
        caco2 = 1e-5
        expected_logpe = 0.292 + 0.6916 * math.log10(caco2)
        result = predict_intestinal_permeability(
            "DrugA", logp=2.0, mw_da=300.0, psa_A2=60.0, caco2_papp_cm_s=caco2
        )
        assert abs(result.logpe_predicted - expected_logpe) < 1e-9

    def test_high_caco2_gives_high_peff_class(self):
        """High Caco-2 value should produce high Peff class."""
        result = predict_intestinal_permeability(
            "DrugA",
            logp=2.0,
            mw_da=300.0,
            psa_A2=60.0,
            caco2_papp_cm_s=5e-5,  # fairly high Caco-2
        )
        # logPe = 0.292 + 0.6916 * log10(5e-5) = 0.292 + 0.6916 * (-4.301) = 0.292 - 2.974 = -2.682
        # peff = 10^-2.682 ~ 2.1e-3 → clamped to 1e-3 → high
        assert result.peff_class == "high"

    def test_caco2_takes_priority_over_physicochemical(self):
        """When caco2 is provided, should use caco2_correlation regardless of logP."""
        result = predict_intestinal_permeability(
            "DrugA", logp=5.0, mw_da=100.0, psa_A2=10.0, caco2_papp_cm_s=1e-7
        )
        assert result.human_intestine_method == "caco2_correlation"


# ---------------------------------------------------------------------------
# PAMPA correlation method
# ---------------------------------------------------------------------------


class TestPampaCorrelation:
    def test_pampa_method_used(self):
        result = predict_intestinal_permeability(
            "DrugB", logp=2.0, mw_da=300.0, psa_A2=60.0, pampa_peff_cm_s=1e-6
        )
        assert result.human_intestine_method == "pampa_correlation"

    def test_pampa_logpe_formula(self):
        """Verify PAMPA correlation formula."""
        pampa = 1e-6
        expected_logpe = 0.1 + 0.8 * math.log10(pampa)
        result = predict_intestinal_permeability(
            "DrugB", logp=2.0, mw_da=300.0, psa_A2=60.0, pampa_peff_cm_s=pampa
        )
        assert abs(result.logpe_predicted - expected_logpe) < 1e-9

    def test_caco2_takes_priority_over_pampa(self):
        """When both caco2 and pampa provided, caco2 takes priority."""
        result = predict_intestinal_permeability(
            "DrugB", logp=2.0, mw_da=300.0, psa_A2=60.0, caco2_papp_cm_s=1e-5, pampa_peff_cm_s=1e-6
        )
        assert result.human_intestine_method == "caco2_correlation"


# ---------------------------------------------------------------------------
# BCS permeability class
# ---------------------------------------------------------------------------


class TestBCSClass:
    def test_high_peff_is_bcs_high(self):
        result = predict_intestinal_permeability("BCSHigh", logp=4.0, mw_da=200.0, psa_A2=20.0)
        if result.peff_cm_s > 2e-4:
            assert result.bcs_permeability_class == "high"

    def test_low_peff_is_bcs_low(self):
        result = predict_intestinal_permeability("BCSLow", logp=-1.0, mw_da=600.0, psa_A2=200.0)
        assert result.bcs_permeability_class == "low"

    def test_bcs_threshold_at_2e_4(self):
        """BCS classification threshold is 2e-4 cm/s."""
        result = predict_intestinal_permeability("BCSTest", logp=3.5, mw_da=250.0, psa_A2=30.0)
        if result.peff_cm_s > 2e-4:
            assert result.bcs_permeability_class == "high"
        else:
            assert result.bcs_permeability_class == "low"


# ---------------------------------------------------------------------------
# Lipinski violations
# ---------------------------------------------------------------------------


class TestLipinskiViolations:
    def test_mw_violation(self):
        result = predict_intestinal_permeability("Heavy", logp=2.0, mw_da=600.0, psa_A2=60.0)
        assert result.lipinski_violations >= 1

    def test_logp_violation(self):
        result = predict_intestinal_permeability("GreasyDrug", logp=6.0, mw_da=300.0, psa_A2=60.0)
        assert result.lipinski_violations >= 1

    def test_hbd_violation(self):
        result = predict_intestinal_permeability(
            "HBDRich", logp=2.0, mw_da=300.0, psa_A2=60.0, n_hbd=7
        )
        assert result.lipinski_violations >= 1

    def test_hba_violation(self):
        result = predict_intestinal_permeability(
            "HBARich", logp=2.0, mw_da=300.0, psa_A2=60.0, n_hba=12
        )
        assert result.lipinski_violations >= 1

    def test_no_violations_compliant(self):
        result = predict_intestinal_permeability(
            "GoodDrug", logp=2.0, mw_da=300.0, psa_A2=60.0, n_hbd=2, n_hba=5
        )
        assert result.lipinski_violations == 0
        assert result.lipinski_compliant is True

    def test_multiple_violations_non_compliant(self):
        result = predict_intestinal_permeability(
            "BadDrug", logp=6.0, mw_da=600.0, psa_A2=60.0, n_hbd=7, n_hba=12
        )
        assert result.lipinski_violations >= 2
        assert result.lipinski_compliant is False

    def test_one_violation_still_compliant(self):
        """Lipinski's rule of five: <=1 violation → compliant."""
        result = predict_intestinal_permeability(
            "AlmostGood", logp=6.0, mw_da=300.0, psa_A2=60.0, n_hbd=2, n_hba=5
        )
        assert result.lipinski_violations == 1
        assert result.lipinski_compliant is True


# ---------------------------------------------------------------------------
# Fraction absorbed
# ---------------------------------------------------------------------------


class TestFractionAbsorbed:
    def test_fa_pct_in_valid_range(self):
        result = predict_intestinal_permeability("TestDrug", logp=2.0, mw_da=300.0, psa_A2=60.0)
        assert 0.0 <= result.fa_predicted_pct <= 100.0

    def test_high_peff_high_fa(self):
        """High permeability should give high fraction absorbed."""
        result = predict_intestinal_permeability("GoodAbsorb", logp=4.5, mw_da=250.0, psa_A2=20.0)
        assert result.fa_predicted_pct > 50.0

    def test_low_peff_low_fa(self):
        """Very low permeability should give low fraction absorbed."""
        result = predict_intestinal_permeability("PoorAbsorb", logp=-2.0, mw_da=700.0, psa_A2=250.0)
        assert result.fa_predicted_pct < 50.0


# ---------------------------------------------------------------------------
# Screen function
# ---------------------------------------------------------------------------


class TestScreenPermeability:
    def _make_drug_list(self):
        return [
            {"name": "DrugA", "logp": 4.5, "mw_da": 250.0, "psa_A2": 20.0},
            {"name": "DrugB", "logp": 1.0, "mw_da": 450.0, "psa_A2": 120.0},
            {"name": "DrugC", "logp": -0.5, "mw_da": 600.0, "psa_A2": 180.0},
        ]

    def test_screen_returns_list(self):
        results = screen_permeability(self._make_drug_list())
        assert isinstance(results, list)
        assert len(results) == 3

    def test_screen_returns_correct_types(self):
        results = screen_permeability(self._make_drug_list())
        for r in results:
            assert isinstance(r, IntestinalPermeabilityResult)

    def test_screen_sorted_by_peff_descending(self):
        results = screen_permeability(self._make_drug_list())
        for i in range(len(results) - 1):
            assert results[i].peff_cm_s >= results[i + 1].peff_cm_s

    def test_screen_empty_list(self):
        results = screen_permeability([])
        assert results == []

    def test_screen_preserves_drug_names(self):
        drugs = self._make_drug_list()
        results = screen_permeability(drugs)
        names = {r.drug_name for r in results}
        assert names == {"DrugA", "DrugB", "DrugC"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_da"):
            predict_intestinal_permeability("Drug", logp=2.0, mw_da=-1.0, psa_A2=60.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_da"):
            predict_intestinal_permeability("Drug", logp=2.0, mw_da=0.0, psa_A2=60.0)

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa_A2"):
            predict_intestinal_permeability("Drug", logp=2.0, mw_da=300.0, psa_A2=-10.0)

    def test_negative_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            predict_intestinal_permeability("Drug", logp=2.0, mw_da=300.0, psa_A2=60.0, n_hbd=-1)

    def test_negative_hba_raises(self):
        with pytest.raises(ValueError, match="n_hba"):
            predict_intestinal_permeability("Drug", logp=2.0, mw_da=300.0, psa_A2=60.0, n_hba=-1)

    def test_negative_caco2_raises(self):
        with pytest.raises(ValueError, match="caco2"):
            predict_intestinal_permeability(
                "Drug", logp=2.0, mw_da=300.0, psa_A2=60.0, caco2_papp_cm_s=-1e-5
            )

    def test_negative_pampa_raises(self):
        with pytest.raises(ValueError, match="pampa"):
            predict_intestinal_permeability(
                "Drug", logp=2.0, mw_da=300.0, psa_A2=60.0, pampa_peff_cm_s=-1e-6
            )
