"""Tests for Phase 903 — CSF vs Brain Parenchyma Distribution Predictor."""

import pytest

from omega_pbpk.prediction.csf_brain_distribution import (
    CSFBrainDistributionResult,
    predict_csf_brain_distribution,
    screen_cns_penetration,
)

# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------


class TestPredictBasicDefaults:
    def test_returns_dataclass(self):
        result = predict_csf_brain_distribution("TestDrug")
        assert isinstance(result, CSFBrainDistributionResult)

    def test_drug_name_stored(self):
        result = predict_csf_brain_distribution("Aspirin")
        assert result.drug_name == "Aspirin"

    def test_default_parameters_stored(self):
        result = predict_csf_brain_distribution("X")
        assert result.logp == 2.0
        assert result.psa_A2 == 60.0
        assert result.mw_Da == 350.0

    def test_result_is_frozen(self):
        result = predict_csf_brain_distribution("X")
        with pytest.raises((AttributeError, TypeError)):
            result.kp_uu_brain = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BBB permeability class
# ---------------------------------------------------------------------------


class TestBBBPermeabilityClass:
    def test_high_permeability(self):
        # score = logp - psa/100 - mw/500 = 4 - 0.2 - 0.4 = 3.4 > 1
        result = predict_csf_brain_distribution("HighPerm", logp=4.0, psa_A2=20.0, mw_Da=200.0)
        assert result.bbb_permeability_class == "high"

    def test_moderate_permeability(self):
        # score = 2 - 0.6 - 0.7 = 0.7, in (-1, 1]
        result = predict_csf_brain_distribution("ModPerm", logp=2.0, psa_A2=60.0, mw_Da=350.0)
        assert result.bbb_permeability_class == "moderate"

    def test_low_permeability(self):
        # score = 0 - 1.5 - 1.0 = -2.5 < -1
        result = predict_csf_brain_distribution("LowPerm", logp=0.0, psa_A2=150.0, mw_Da=500.0)
        assert result.bbb_permeability_class == "low"


# ---------------------------------------------------------------------------
# fu_brain calculation
# ---------------------------------------------------------------------------


class TestFuBrain:
    def test_fu_brain_in_range(self):
        result = predict_csf_brain_distribution("X")
        assert 0.01 <= result.fu_brain <= 1.0

    def test_fu_brain_decreases_with_logp(self):
        low = predict_csf_brain_distribution("X", logp=0.0)
        high = predict_csf_brain_distribution("X", logp=4.0)
        assert high.fu_brain < low.fu_brain

    def test_fu_brain_clamped_above(self):
        # Very hydrophilic: logp = -5 → fu_brain should approach 1 but clamped at 1
        result = predict_csf_brain_distribution("X", logp=-5.0)
        assert result.fu_brain <= 1.0

    def test_fu_brain_clamped_below(self):
        # Very lipophilic
        result = predict_csf_brain_distribution("X", logp=8.0)
        assert result.fu_brain >= 0.01


# ---------------------------------------------------------------------------
# kp_uu_brain and efflux
# ---------------------------------------------------------------------------


class TestKpUuBrain:
    def test_kp_uu_brain_in_range(self):
        result = predict_csf_brain_distribution("X")
        assert 0.001 <= result.kp_uu_brain <= 1.0

    def test_pgp_reduces_kp_uu(self):
        base = predict_csf_brain_distribution("X", logp=4.0)
        pgp = predict_csf_brain_distribution("X", logp=4.0, has_pgp=True)
        assert pgp.kp_uu_brain < base.kp_uu_brain

    def test_bcrp_reduces_kp_uu(self):
        base = predict_csf_brain_distribution("X", logp=4.0)
        bcrp = predict_csf_brain_distribution("X", logp=4.0, has_bcrp=True)
        assert bcrp.kp_uu_brain < base.kp_uu_brain

    def test_both_efflux_reduces_more(self):
        pgp_only = predict_csf_brain_distribution("X", logp=4.0, has_pgp=True)
        both = predict_csf_brain_distribution("X", logp=4.0, has_pgp=True, has_bcrp=True)
        assert both.kp_uu_brain < pgp_only.kp_uu_brain

    def test_efflux_flag_true_when_pgp(self):
        result = predict_csf_brain_distribution("X", has_pgp=True)
        assert result.efflux_flag is True

    def test_efflux_flag_true_when_bcrp(self):
        result = predict_csf_brain_distribution("X", has_bcrp=True)
        assert result.efflux_flag is True

    def test_efflux_flag_false_without_efflux(self):
        result = predict_csf_brain_distribution("X")
        assert result.efflux_flag is False


# ---------------------------------------------------------------------------
# CSF metrics
# ---------------------------------------------------------------------------


class TestCSFMetrics:
    def test_csf_plasma_ratio_is_70_percent_kp_uu(self):
        result = predict_csf_brain_distribution("X", logp=4.0)
        expected = result.kp_uu_brain * 0.7
        assert abs(
            result.csf_plasma_ratio - expected
        ) < 1e-9 or result.csf_plasma_ratio == pytest.approx(expected, abs=1e-6)

    def test_csf_brain_ratio_approx_0_7(self):
        # csf_brain_ratio = csf_plasma_ratio / kp_uu_brain = 0.7 (when not clamped)
        result = predict_csf_brain_distribution("X", logp=2.0, psa_A2=30.0)
        # Allow for clamping effects; ratio should be close to 0.7
        assert 0.5 <= result.csf_brain_ratio <= 1.0

    def test_csf_plasma_ratio_clamped(self):
        result = predict_csf_brain_distribution("X")
        assert 0.001 <= result.csf_plasma_ratio <= 1.0


# ---------------------------------------------------------------------------
# kp_brain
# ---------------------------------------------------------------------------


class TestKpBrain:
    def test_kp_brain_in_range(self):
        result = predict_csf_brain_distribution("X")
        assert 0.001 <= result.kp_brain <= 100.0

    def test_kp_brain_increases_with_fu_plasma(self):
        low_fu = predict_csf_brain_distribution("X", fu_plasma=0.05)
        high_fu = predict_csf_brain_distribution("X", fu_plasma=0.5)
        assert high_fu.kp_brain > low_fu.kp_brain


# ---------------------------------------------------------------------------
# Target site exposure
# ---------------------------------------------------------------------------


class TestTargetSiteExposure:
    def test_adequate_when_kp_uu_above_03(self):
        result = predict_csf_brain_distribution("X", logp=4.0)
        if result.kp_uu_brain > 0.3:
            assert result.target_site_exposure_adequate is True

    def test_inadequate_when_low_permeability(self):
        result = predict_csf_brain_distribution("X", logp=0.0, psa_A2=150.0, mw_Da=500.0)
        assert result.target_site_exposure_adequate is False


# ---------------------------------------------------------------------------
# Notes content
# ---------------------------------------------------------------------------


class TestNotes:
    def test_adequate_note_when_high_kp_uu(self):
        result = predict_csf_brain_distribution("X", logp=4.0, psa_A2=20.0, mw_Da=200.0)
        assert "Adequate CNS target site exposure" in result.notes

    def test_efflux_note_when_efflux_and_inadequate(self):
        # Low permeability + P-gp → efflux note
        result = predict_csf_brain_distribution(
            "X", logp=0.0, psa_A2=150.0, mw_Da=500.0, has_pgp=True
        )
        assert "efflux" in result.notes.lower() or "Active efflux" in result.notes

    def test_limited_penetration_note_no_efflux(self):
        result = predict_csf_brain_distribution("X", logp=0.0, psa_A2=150.0, mw_Da=500.0)
        assert "Limited CNS penetration" in result.notes


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa_A2"):
            predict_csf_brain_distribution("X", psa_A2=-1.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_csf_brain_distribution("X", mw_Da=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_csf_brain_distribution("X", mw_Da=-100.0)


# ---------------------------------------------------------------------------
# screen_cns_penetration
# ---------------------------------------------------------------------------


class TestScreenCNSPenetration:
    def test_returns_list(self):
        candidates = [
            {"name": "DrugA", "logp": 3.0},
            {"name": "DrugB", "logp": 1.0},
        ]
        results = screen_cns_penetration(candidates)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_by_kp_uu_descending(self):
        candidates = [
            {"name": "Low", "logp": 0.0, "psa_A2": 150.0},
            {"name": "High", "logp": 4.0, "psa_A2": 20.0},
            {"name": "Mid", "logp": 2.0, "psa_A2": 60.0},
        ]
        results = screen_cns_penetration(candidates)
        kp_uu_values = [r.kp_uu_brain for r in results]
        assert kp_uu_values == sorted(kp_uu_values, reverse=True)

    def test_optional_fields_default(self):
        candidates = [{"name": "Simple", "logp": 2.0}]
        results = screen_cns_penetration(candidates)
        assert results[0].psa_A2 == 60.0
        assert results[0].mw_Da == 350.0

    def test_pgp_field_used(self):
        candidates = [
            {"name": "NoPgp", "logp": 4.0, "has_pgp": False},
            {"name": "Pgp", "logp": 4.0, "has_pgp": True},
        ]
        results = screen_cns_penetration(candidates)
        no_pgp = next(r for r in results if r.drug_name == "NoPgp")
        pgp = next(r for r in results if r.drug_name == "Pgp")
        assert no_pgp.kp_uu_brain > pgp.kp_uu_brain
