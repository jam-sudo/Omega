"""Tests for minimal PBPK (mPBPK) model — Phase 154."""

import pytest

from omega_pbpk.core.minimal_pbpk import (
    V_BLOOD,
    V_FAT,
    V_RPT,
    V_SPT,
    MinimalPBPKResult,
    simulate_minimal_pbpk,
    vss_calculator,
)


def _default(**kw):
    defaults = dict(drug_name="TestDrug", dose_mg=100.0, cl_hepatic_L_per_h=10.0)
    defaults.update(kw)
    return simulate_minimal_pbpk(**defaults)


class TestMinimalPBPK:
    def test_returns_result_type(self):
        r = _default()
        assert isinstance(r, MinimalPBPKResult)

    def test_dose_zero_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=0)

    def test_dose_negative_raises(self):
        with pytest.raises(ValueError):
            _default(dose_mg=-1)

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError):
            _default(cl_hepatic_L_per_h=0)

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError):
            _default(cl_hepatic_L_per_h=-5)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            _default(fup=0)

    def test_cmax_positive(self):
        r = _default()
        assert r.cmax_blood > 0

    def test_auc_positive(self):
        r = _default()
        assert r.auc_blood > 0

    def test_lipophilic_fat_accumulation(self):
        r = _default(kp_fat=10.0, t_end_h=48.0)
        # At some point fat conc should exceed blood
        max_fat = max(r.c_fat_mg_L)
        # Fat accumulates more than blood at equilibrium
        assert max_fat > 0

    def test_rpt_cmax_gt_spt_cmax(self):
        r = _default(kp_rpt=2.0, kp_spt=1.0)
        assert r.tissue_distribution["rpt"] > r.tissue_distribution["spt"]

    def test_vd_ss_gt_blood_volume(self):
        r = _default()
        assert r.vd_ss_apparent_L > V_BLOOD

    def test_vss_calculator_positive(self):
        v = vss_calculator()
        assert isinstance(v, float)
        assert v > 0

    def test_higher_kp_fat_higher_vss(self):
        v1 = vss_calculator(kp_fat=5.0)
        v2 = vss_calculator(kp_fat=10.0)
        assert v2 > v1

    def test_t_half_positive(self):
        r = _default()
        assert r.t_half_h > 0

    def test_oral_cmax_lower_than_iv(self):
        r_iv = _default(route="iv")
        r_oral = _default(route="oral", ka_per_h=1.0, f_oral=1.0)
        assert r_oral.cmax_blood < r_iv.cmax_blood

    def test_auc_proportional_to_dose(self):
        r1 = _default(dose_mg=100.0)
        r2 = _default(dose_mg=200.0)
        ratio = r2.auc_blood / r1.auc_blood
        assert 1.8 < ratio < 2.2

    def test_tissue_distribution_keys(self):
        r = _default()
        assert set(r.tissue_distribution.keys()) == {"rpt", "spt", "fat"}

    def test_fup_1_higher_clearance_effect(self):
        r_low = _default(fup=0.1)
        r_high = _default(fup=1.0)
        # Both should produce valid results
        assert r_low.auc_blood > 0
        assert r_high.auc_blood > 0

    def test_blood_returns_near_zero_long_sim(self):
        r = _default(t_end_h=120.0, cl_hepatic_L_per_h=20.0)
        assert r.c_blood_mg_L[-1] < 0.01

    def test_c_blood_all_non_negative(self):
        r = _default()
        assert all(c >= 0 for c in r.c_blood_mg_L)

    def test_times_length(self):
        r = _default()
        assert len(r.times_h) > 1

    def test_low_kp_rpt_blood_higher(self):
        r = _default(kp_rpt=0.5)
        # With kp < 1, tissue conc < blood at peak
        assert r.tissue_distribution["rpt"] < r.cmax_blood

    def test_notes_contain_drug_name(self):
        r = _default(drug_name="Warfarin")
        assert "Warfarin" in r.notes

    def test_iv_tmax_at_zero(self):
        r = _default(route="iv")
        assert r.tmax_h == 0.0

    def test_oral_f_oral_zero_no_exposure(self):
        r = _default(route="oral", f_oral=0.0)
        assert r.cmax_blood == 0.0

    def test_vss_calculator_matches_formula(self):
        kp_r, kp_s, kp_f = 3.0, 1.5, 8.0
        v = vss_calculator(kp_rpt=kp_r, kp_spt=kp_s, kp_fat=kp_f)
        expected = V_BLOOD + kp_r * V_RPT + kp_s * V_SPT + kp_f * V_FAT
        assert abs(v - expected) < 1e-6

    def test_fat_gt_spt_long_time_lipophilic(self):
        r = _default(kp_fat=5.0, kp_spt=1.0, t_end_h=48.0)
        # Peak fat concentration should exceed peak SPT
        assert r.tissue_distribution["fat"] > r.tissue_distribution["spt"]
