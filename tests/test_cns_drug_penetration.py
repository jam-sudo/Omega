"""Tests for CNS drug penetration and target engagement prediction — Phase 508."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.cns_drug_penetration import (
    BBBPenetrationResult,
    CNSEfficacyResult,
    calculate_kp_uu_brain,
    cns_efficacy_likelihood,
    predict_bbb_penetration,
    predict_cns_target_engagement,
)

# ---------------------------------------------------------------------------
# calculate_kp_uu_brain
# ---------------------------------------------------------------------------


class TestCalculateKpUuBrain:
    def test_returns_float(self):
        result = calculate_kp_uu_brain(2.0, 300.0, 60.0)
        assert isinstance(result, float)

    def test_kp_uu_in_valid_range(self):
        result = calculate_kp_uu_brain(2.0, 300.0, 60.0)
        assert 0.01 <= result <= 5.0

    def test_pgp_substrate_lowers_kp_uu(self):
        no_pgp = calculate_kp_uu_brain(2.0, 300.0, 60.0, pgp_substrate=False)
        with_pgp = calculate_kp_uu_brain(2.0, 300.0, 60.0, pgp_substrate=True)
        assert with_pgp < no_pgp

    def test_pgp_substrate_factor_approx_0_2(self):
        no_pgp = calculate_kp_uu_brain(2.0, 300.0, 60.0, pgp_substrate=False)
        with_pgp = calculate_kp_uu_brain(2.0, 300.0, 60.0, pgp_substrate=True)
        # Factor should be ~0.2 unless already at bounds
        ratio = with_pgp / no_pgp
        assert ratio <= 0.25

    def test_high_psa_reduces_kp_uu(self):
        low_psa = calculate_kp_uu_brain(2.0, 300.0, 40.0)
        high_psa = calculate_kp_uu_brain(2.0, 300.0, 120.0)
        assert high_psa < low_psa

    def test_high_mw_reduces_kp_uu(self):
        low_mw = calculate_kp_uu_brain(2.0, 300.0, 60.0)
        high_mw = calculate_kp_uu_brain(2.0, 600.0, 60.0)
        assert high_mw < low_mw

    def test_minimum_clamped_at_0_01(self):
        # Extreme high PSA + high MW + P-gp should clamp to floor
        result = calculate_kp_uu_brain(-3.0, 800.0, 200.0, pgp_substrate=True)
        assert result >= 0.01

    def test_maximum_clamped_at_5(self):
        # Optimal logP with no penalty factors
        result = calculate_kp_uu_brain(2.5, 250.0, 30.0, pgp_substrate=False)
        assert result <= 5.0

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            calculate_kp_uu_brain(2.0, 0.0, 60.0)

    def test_invalid_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            calculate_kp_uu_brain(2.0, 300.0, 60.0, fu_plasma=0.0)


# ---------------------------------------------------------------------------
# predict_bbb_penetration
# ---------------------------------------------------------------------------


class TestPredictBBBPenetration:
    def test_returns_correct_type(self):
        result = predict_bbb_penetration("Drug-A", 2.0, 300.0, 60.0, 2)
        assert isinstance(result, BBBPenetrationResult)

    def test_bbb_score_in_range(self):
        result = predict_bbb_penetration("Drug-A", 2.0, 300.0, 60.0, 2)
        assert 0.0 <= result.bbb_score <= 100.0

    def test_low_psa_moderate_logp_high_bbb_score(self):
        result = predict_bbb_penetration("CNS-Drug", 2.5, 280.0, 40.0, 1)
        assert result.bbb_score > 60.0

    def test_high_psa_minimal_cns_exposure(self):
        # High PSA + non-optimal logP (very high) + P-gp substrate → low/minimal
        result = predict_bbb_penetration("PolarDrug", 6.0, 300.0, 150.0, 5, pgp_substrate=True)
        assert result.predicted_cns_exposure in ("low", "minimal")

    def test_pgp_substrate_flag_reflected(self):
        result = predict_bbb_penetration("PgpDrug", 2.0, 300.0, 60.0, 2, pgp_substrate=True)
        assert result.pgp_efflux_concern is True

    def test_pgp_substrate_lower_kp_uu(self):
        no_pgp = predict_bbb_penetration("DrugA", 2.0, 300.0, 60.0, 2, pgp_substrate=False)
        with_pgp = predict_bbb_penetration("DrugB", 2.0, 300.0, 60.0, 2, pgp_substrate=True)
        assert with_pgp.kp_uu_brain < no_pgp.kp_uu_brain

    def test_kp_uu_in_valid_range(self):
        result = predict_bbb_penetration("Drug-B", 2.0, 300.0, 60.0, 2)
        assert 0.01 <= result.kp_uu_brain <= 5.0

    def test_exposure_categories_ordered(self):
        high = predict_bbb_penetration("HighCNS", 2.5, 250.0, 40.0, 1)
        low = predict_bbb_penetration("LowCNS", 2.5, 600.0, 140.0, 6, pgp_substrate=True)
        # High penetration drug should have better exposure
        order = ["minimal", "low", "moderate", "high"]
        assert order.index(high.predicted_cns_exposure) >= order.index(low.predicted_cns_exposure)

    def test_notes_non_empty(self):
        result = predict_bbb_penetration("Drug-C", 2.0, 300.0, 60.0, 2)
        assert len(result.notes) > 0

    def test_drug_name_preserved(self):
        result = predict_bbb_penetration("MyDrug", 2.0, 300.0, 60.0, 2)
        assert result.drug_name == "MyDrug"

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_bbb_penetration("X", 2.0, 0.0, 60.0, 2)

    def test_invalid_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_bbb_penetration("X", 2.0, 300.0, 60.0, 2, fu_plasma=1.5)

    def test_invalid_n_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            predict_bbb_penetration("X", 2.0, 300.0, 60.0, -1)


# ---------------------------------------------------------------------------
# predict_cns_target_engagement
# ---------------------------------------------------------------------------


class TestPredictCNSTargetEngagement:
    def test_returns_float(self):
        result = predict_cns_target_engagement(1.0, 0.1, 10.0, 300.0, 10.0)
        assert isinstance(result, float)

    def test_occupancy_in_range(self):
        result = predict_cns_target_engagement(1.0, 0.1, 10.0, 300.0, 10.0)
        assert 0.0 <= result <= 100.0

    def test_high_brain_conc_near_100pct_occupancy(self):
        # Very high plasma conc, good Kp_uu, low EC50 → near 100%
        result = predict_cns_target_engagement(2.0, 0.5, 500.0, 300.0, 1.0)
        assert result > 90.0

    def test_low_brain_conc_near_zero_occupancy(self):
        # Very low plasma conc, poor Kp_uu, high EC50 → near 0%
        result = predict_cns_target_engagement(0.01, 0.01, 0.001, 300.0, 1000.0)
        assert result < 5.0

    def test_invalid_kp_uu_raises(self):
        with pytest.raises(ValueError, match="kp_uu"):
            predict_cns_target_engagement(0.0, 0.1, 10.0, 300.0, 10.0)

    def test_invalid_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_cns_target_engagement(1.0, 0.0, 10.0, 300.0, 10.0)

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_cns_target_engagement(1.0, 0.1, 10.0, 0.0, 10.0)

    def test_invalid_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50_nM"):
            predict_cns_target_engagement(1.0, 0.1, 10.0, 300.0, 0.0)


# ---------------------------------------------------------------------------
# cns_efficacy_likelihood
# ---------------------------------------------------------------------------


class TestCNSEfficacyLikelihood:
    def test_returns_correct_type(self):
        result = cns_efficacy_likelihood(2.0, 300.0, 60.0, 2, 10.0, 100.0)
        assert isinstance(result, CNSEfficacyResult)

    def test_mpo_score_max_6(self):
        result = cns_efficacy_likelihood(2.0, 300.0, 60.0, 1, 10.0, 100.0)
        assert result.cns_mpo_score <= 6.0

    def test_mpo_score_non_negative(self):
        result = cns_efficacy_likelihood(2.0, 300.0, 60.0, 2, 10.0, 100.0)
        assert result.cns_mpo_score >= 0.0

    def test_good_cns_properties_high_mpo(self):
        # Ideal CNS drug: logP 2, MW 300, PSA 40, HBD 1
        result = cns_efficacy_likelihood(2.0, 300.0, 40.0, 1, 10.0, 1000.0)
        assert result.cns_mpo_score >= 4.0

    def test_poor_cns_properties_low_mpo(self):
        # Bad CNS: logP 6, MW 600, PSA 150, HBD 6
        result = cns_efficacy_likelihood(6.0, 600.0, 150.0, 6, 10.0, 100.0)
        assert result.cns_mpo_score < 3.0

    def test_mw_concern_above_450(self):
        result = cns_efficacy_likelihood(2.0, 500.0, 60.0, 2, 10.0, 100.0)
        assert result.mw_concern is True

    def test_no_mw_concern_below_450(self):
        result = cns_efficacy_likelihood(2.0, 350.0, 60.0, 2, 10.0, 100.0)
        assert result.mw_concern is False

    def test_psa_concern_above_90(self):
        result = cns_efficacy_likelihood(2.0, 300.0, 100.0, 2, 10.0, 100.0)
        assert result.psa_concern is True

    def test_no_psa_concern_below_90(self):
        result = cns_efficacy_likelihood(2.0, 300.0, 60.0, 2, 10.0, 100.0)
        assert result.psa_concern is False

    def test_optimal_logp_1_to_4(self):
        result = cns_efficacy_likelihood(2.5, 300.0, 60.0, 2, 10.0, 100.0)
        assert result.optimal_logP is True

    def test_non_optimal_logp(self):
        result = cns_efficacy_likelihood(6.0, 300.0, 60.0, 2, 10.0, 100.0)
        assert result.optimal_logP is False

    def test_high_cmax_low_ec50_efficacious(self):
        # Very high Cmax relative to EC50 → should be efficacious
        result = cns_efficacy_likelihood(2.0, 300.0, 40.0, 1, 1.0, 10000.0)
        assert result.is_efficacious is True

    def test_low_cmax_high_ec50_not_efficacious(self):
        result = cns_efficacy_likelihood(2.0, 300.0, 120.0, 5, 10000.0, 1.0)
        assert result.is_efficacious is False

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            cns_efficacy_likelihood(2.0, 0.0, 60.0, 2, 10.0, 100.0)

    def test_invalid_ec50_raises(self):
        with pytest.raises(ValueError, match="ec50_nM"):
            cns_efficacy_likelihood(2.0, 300.0, 60.0, 2, 0.0, 100.0)

    def test_invalid_n_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            cns_efficacy_likelihood(2.0, 300.0, 60.0, -1, 10.0, 100.0)

    def test_notes_non_empty(self):
        result = cns_efficacy_likelihood(2.0, 300.0, 60.0, 2, 10.0, 100.0)
        assert len(result.notes) > 0

    def test_occupancy_in_range(self):
        result = cns_efficacy_likelihood(2.0, 300.0, 60.0, 2, 10.0, 100.0)
        assert 0.0 <= result.predicted_occupancy_at_cmax_pct <= 100.0
