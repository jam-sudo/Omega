"""Tests for tissue distribution calculator — Phase 155."""

import pytest

from omega_pbpk.core.tissue_distribution import (
    TISSUE_VOLUMES,
    V_BLOOD,
    TissueDistributionResult,
    accumulation_risk_screen,
    predict_tissue_distribution,
    vss_from_kp,
)


def _default(**kw):
    defaults = dict(drug_name="TestDrug", c_plasma_mg_L=1.0, logP=2.0, fup=0.1)
    defaults.update(kw)
    return predict_tissue_distribution(**defaults)


class TestTissueDistribution:
    def test_returns_result_type(self):
        r = _default()
        assert isinstance(r, TissueDistributionResult)

    def test_c_plasma_zero_raises(self):
        with pytest.raises(ValueError):
            _default(c_plasma_mg_L=0)

    def test_c_plasma_negative_raises(self):
        with pytest.raises(ValueError):
            _default(c_plasma_mg_L=-1)

    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            _default(fup=0)

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError):
            _default(fup=-0.1)

    def test_13_tissue_entries(self):
        r = _default()
        assert len(r.tissue_concentrations) == 13

    def test_all_ct_positive(self):
        r = _default()
        for tc in r.tissue_concentrations:
            assert tc.ct_total_mg_L >= r.c_plasma_mg_L * 0.01

    def test_adipose_kp_high_lipophilic(self):
        r = _default(logP=4.0, fup=0.1)
        adipose = next(tc for tc in r.tissue_concentrations if tc.tissue == "adipose")
        assert adipose.kp > 5

    def test_brain_kp_lt_adipose_kp(self):
        r = _default(logP=4.0, fup=0.1)
        adipose = next(tc for tc in r.tissue_concentrations if tc.tissue == "adipose")
        brain = next(tc for tc in r.tissue_concentrations if tc.tissue == "brain")
        assert brain.kp < adipose.kp

    def test_vss_gt_blood_volume(self):
        r = _default()
        assert r.vss_L > V_BLOOD

    def test_higher_logp_larger_vss(self):
        r1 = _default(logP=1.0)
        r2 = _default(logP=4.0)
        assert r2.vss_L > r1.vss_L

    def test_higher_fup_smaller_vss(self):
        r1 = _default(fup=0.05)
        r2 = _default(fup=0.5)
        assert r1.vss_L > r2.vss_L

    def test_highest_kp_tissue_valid(self):
        r = _default()
        assert r.highest_kp_tissue in TISSUE_VOLUMES

    def test_kp_overrides_work(self):
        r = _default(kp_overrides={"liver": 10.0})
        liver = next(tc for tc in r.tissue_concentrations if tc.tissue == "liver")
        assert liver.kp == 10.0

    def test_no_high_risk_hydrophilic(self):
        r = _default(logP=-2.0, fup=0.9)
        assert len(r.tissues_at_high_risk) == 0

    def test_plasma_fraction_in_range(self):
        r = _default()
        assert 0 < r.plasma_fraction_pct < 100

    def test_total_drug_positive(self):
        r = _default()
        assert r.total_drug_in_tissues_mg > 0

    def test_vss_from_kp_matches(self):
        r = _default()
        kp_dict = {tc.tissue: tc.kp for tc in r.tissue_concentrations}
        v = vss_from_kp(kp_dict)
        assert abs(v - r.vss_L) < 1e-6

    def test_vss_from_kp_empty_dict(self):
        v = vss_from_kp({})
        assert v == V_BLOOD

    def test_accumulation_screen_length(self):
        drugs = [
            {"name": "A", "c_plasma_mg_L": 1.0, "logP": 2.0, "fup": 0.1},
            {"name": "B", "c_plasma_mg_L": 1.0, "logP": 0.5, "fup": 0.5},
        ]
        results = accumulation_risk_screen(drugs)
        assert len(results) == 2

    def test_accumulation_screen_sorted_desc(self):
        drugs = [
            {"name": "Hydro", "c_plasma_mg_L": 1.0, "logP": -1.0, "fup": 0.9},
            {"name": "Lipo", "c_plasma_mg_L": 1.0, "logP": 4.0, "fup": 0.05},
        ]
        results = accumulation_risk_screen(drugs)
        assert results[0].vss_L >= results[1].vss_L

    def test_ct_total_equals_kp_times_cplasma(self):
        r = _default()
        liver = next(tc for tc in r.tissue_concentrations if tc.tissue == "liver")
        assert abs(liver.ct_total_mg_L - liver.kp * r.c_plasma_mg_L) < 1e-10

    def test_notes_contain_drug_name(self):
        r = _default(drug_name="Diazepam")
        assert "Diazepam" in r.notes

    def test_all_kp_in_range(self):
        r = _default()
        for tc in r.tissue_concentrations:
            assert 0.05 <= tc.kp <= 100
