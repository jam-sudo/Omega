"""Tests for P-glycoprotein efflux model (Phase 544)."""

import pytest

from omega_pbpk.core.pgp_efflux_model import (
    cns_penetration_with_pgp,
    inhibition_effect_on_pgp,
    intestinal_absorption_with_pgp,
    predict_pgp_efflux_ratio,
)


class TestPredictPgpEffluxRatio:
    def test_small_lipophilic_drug_er_near_one(self):
        # MW=200, logP=2, PSA=30, HBD=1, HBA=3 → ER=1.0
        er = predict_pgp_efflux_ratio(mw=200, logP=2.0, psa_ang2=30.0, n_hbd=1, n_hba=3)
        assert er == 1.0

    def test_large_mw_increases_er(self):
        er_large = predict_pgp_efflux_ratio(mw=600, logP=2.0, psa_ang2=30.0, n_hbd=1, n_hba=3)
        er_small = predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=30.0, n_hbd=1, n_hba=3)
        assert er_large > er_small

    def test_high_psa_increases_er(self):
        er_high = predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=120.0, n_hbd=1, n_hba=3)
        er_low = predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=40.0, n_hbd=1, n_hba=3)
        assert er_high > er_low

    def test_high_hbd_increases_er(self):
        er_high = predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=30.0, n_hbd=6, n_hba=3)
        er_low = predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=30.0, n_hbd=1, n_hba=3)
        assert er_high > er_low

    def test_high_hba_increases_er(self):
        er_high = predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=30.0, n_hbd=1, n_hba=10)
        er_low = predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=30.0, n_hbd=1, n_hba=3)
        assert er_high > er_low

    def test_er_clamped_min_one(self):
        er = predict_pgp_efflux_ratio(mw=100, logP=1.0, psa_ang2=10.0, n_hbd=0, n_hba=0)
        assert er >= 1.0

    def test_er_clamped_max_twenty(self):
        er = predict_pgp_efflux_ratio(mw=5000, logP=10.0, psa_ang2=500.0, n_hbd=20, n_hba=20)
        assert er <= 20.0

    def test_mw_exactly_400_no_bonus(self):
        er = predict_pgp_efflux_ratio(mw=400, logP=2.0, psa_ang2=30.0, n_hbd=1, n_hba=3)
        assert er == 1.0

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            predict_pgp_efflux_ratio(mw=-100, logP=2.0, psa_ang2=30.0, n_hbd=1, n_hba=3)

    def test_invalid_psa_raises(self):
        with pytest.raises(ValueError):
            predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=-10.0, n_hbd=1, n_hba=3)

    def test_invalid_hbd_raises(self):
        with pytest.raises(ValueError):
            predict_pgp_efflux_ratio(mw=300, logP=2.0, psa_ang2=30.0, n_hbd=-1, n_hba=3)


class TestCnsPenetrationWithPgp:
    def test_er_one_returns_kp_passive(self):
        kp_uu = cns_penetration_with_pgp(kp_passive=0.5, efflux_ratio=1.0)
        assert abs(kp_uu - 0.5) < 1e-9

    def test_high_er_reduces_penetration(self):
        kp_uu = cns_penetration_with_pgp(kp_passive=0.5, efflux_ratio=10.0)
        assert kp_uu < 0.5

    def test_kp_uu_cannot_exceed_kp_passive(self):
        kp_uu = cns_penetration_with_pgp(kp_passive=0.5, efflux_ratio=1.0)
        assert kp_uu <= 0.5

    def test_kp_uu_clamped_above_0_01(self):
        kp_uu = cns_penetration_with_pgp(kp_passive=0.5, efflux_ratio=20.0)
        assert kp_uu >= 0.01

    def test_higher_pgp_expression_lower_kp_uu(self):
        kp_high = cns_penetration_with_pgp(kp_passive=0.5, efflux_ratio=5.0, pgp_expression=2.0)
        kp_low = cns_penetration_with_pgp(kp_passive=0.5, efflux_ratio=5.0, pgp_expression=1.0)
        assert kp_high < kp_low

    def test_invalid_kp_passive_raises(self):
        with pytest.raises(ValueError):
            cns_penetration_with_pgp(kp_passive=0, efflux_ratio=2.0)

    def test_invalid_efflux_ratio_raises(self):
        with pytest.raises(ValueError):
            cns_penetration_with_pgp(kp_passive=0.5, efflux_ratio=0.5)


class TestIntestinalAbsorptionWithPgp:
    def test_high_dose_more_saturation(self):
        result_high = intestinal_absorption_with_pgp(
            fa_passive=0.8, efflux_ratio=3.0, dose_mg=1000.0
        )
        result_low = intestinal_absorption_with_pgp(fa_passive=0.8, efflux_ratio=3.0, dose_mg=1.0)
        assert result_high["saturation_factor"] > result_low["saturation_factor"]

    def test_high_dose_less_pgp_effect(self):
        result_high = intestinal_absorption_with_pgp(
            fa_passive=0.8, efflux_ratio=5.0, dose_mg=5000.0
        )
        result_low = intestinal_absorption_with_pgp(fa_passive=0.8, efflux_ratio=5.0, dose_mg=1.0)
        assert result_high["fa_adjusted"] > result_low["fa_adjusted"]

    def test_fa_adjusted_le_fa_passive(self):
        result = intestinal_absorption_with_pgp(fa_passive=0.8, efflux_ratio=3.0, dose_mg=100.0)
        assert result["fa_adjusted"] <= result["fa_passive"]

    def test_fa_adjusted_ge_0_01(self):
        result = intestinal_absorption_with_pgp(fa_passive=0.8, efflux_ratio=20.0, dose_mg=0.001)
        assert result["fa_adjusted"] >= 0.01

    def test_pgp_impact_classification(self):
        # High ER, low dose → significant reduction
        result = intestinal_absorption_with_pgp(fa_passive=0.9, efflux_ratio=10.0, dose_mg=0.1)
        assert result["pgp_impact"] in ("high", "moderate", "low")

    def test_er_one_minimal_effect(self):
        result = intestinal_absorption_with_pgp(fa_passive=0.8, efflux_ratio=1.0, dose_mg=100.0)
        assert abs(result["fa_adjusted"] - 0.8) < 1e-9

    def test_invalid_fa_raises(self):
        with pytest.raises(ValueError):
            intestinal_absorption_with_pgp(fa_passive=1.5, efflux_ratio=2.0, dose_mg=100.0)

    def test_invalid_efflux_ratio_raises(self):
        with pytest.raises(ValueError):
            intestinal_absorption_with_pgp(fa_passive=0.8, efflux_ratio=0.5, dose_mg=100.0)

    def test_saturation_factor_bounds(self):
        result = intestinal_absorption_with_pgp(fa_passive=0.8, efflux_ratio=3.0, dose_mg=100.0)
        assert 0.0 <= result["saturation_factor"] <= 1.0


class TestInhibitionEffectOnPgp:
    def test_inhibitor_raises_kp_uu(self):
        result = inhibition_effect_on_pgp(
            kp_uu_baseline=0.05,
            efflux_ratio=10.0,
            inhibitor_ki_uM=1.0,
            inhibitor_conc_uM=10.0,
        )
        assert result["kp_uu_inhibited"] > result["kp_uu_baseline"]

    def test_fold_increase_gt_one_with_inhibitor(self):
        result = inhibition_effect_on_pgp(
            kp_uu_baseline=0.1,
            efflux_ratio=5.0,
            inhibitor_ki_uM=1.0,
            inhibitor_conc_uM=10.0,
        )
        assert result["fold_increase_penetration"] > 1.0

    def test_zero_inhibitor_no_effect(self):
        result = inhibition_effect_on_pgp(
            kp_uu_baseline=0.1,
            efflux_ratio=5.0,
            inhibitor_ki_uM=1.0,
            inhibitor_conc_uM=0.0,
        )
        assert abs(result["fold_increase_penetration"] - 1.0) < 1e-9

    def test_fractional_inhibition_bounds(self):
        result = inhibition_effect_on_pgp(
            kp_uu_baseline=0.1,
            efflux_ratio=5.0,
            inhibitor_ki_uM=1.0,
            inhibitor_conc_uM=5.0,
        )
        assert 0.0 <= result["fractional_inhibition"] <= 1.0

    def test_effective_er_reduced_by_inhibitor(self):
        result = inhibition_effect_on_pgp(
            kp_uu_baseline=0.1,
            efflux_ratio=5.0,
            inhibitor_ki_uM=1.0,
            inhibitor_conc_uM=4.0,
        )
        assert result["effective_er"] < 5.0

    def test_effective_er_minimum_one(self):
        result = inhibition_effect_on_pgp(
            kp_uu_baseline=0.1,
            efflux_ratio=2.0,
            inhibitor_ki_uM=0.001,
            inhibitor_conc_uM=1000.0,
        )
        assert result["effective_er"] >= 1.0

    def test_invalid_kp_uu_baseline_raises(self):
        with pytest.raises(ValueError):
            inhibition_effect_on_pgp(
                kp_uu_baseline=0,
                efflux_ratio=5.0,
                inhibitor_ki_uM=1.0,
                inhibitor_conc_uM=1.0,
            )

    def test_invalid_ki_raises(self):
        with pytest.raises(ValueError):
            inhibition_effect_on_pgp(
                kp_uu_baseline=0.1,
                efflux_ratio=5.0,
                inhibitor_ki_uM=0,
                inhibitor_conc_uM=1.0,
            )
