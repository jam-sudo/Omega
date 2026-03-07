"""Tests for BBB penetration predictor (Phase 430)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.bbb_penetration_predictor import (
    BBBPenetrationResult,
    predict_bbb_penetration,
    simulate_cns_pk,
)


class TestPredictBBBValidation:
    def _default(self, **kwargs):
        base = dict(
            drug_name="TestDrug",
            mw=300.0,
            logP=2.0,
            psa=60.0,
            n_hbd=2,
            is_pgp_substrate=False,
            logD_pH74=1.5,
        )
        base.update(kwargs)
        return predict_bbb_penetration(**base)

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            self._default(drug_name="")

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            self._default(mw=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            self._default(mw=-100.0)

    def test_negative_psa_raises(self):
        with pytest.raises(ValueError, match="psa"):
            self._default(psa=-1.0)

    def test_negative_n_hbd_raises(self):
        with pytest.raises(ValueError, match="n_hbd"):
            self._default(n_hbd=-1)


class TestPredictBBBResult:
    def _ideal_cns_plus(self):
        return predict_bbb_penetration(
            drug_name="IdealCNS",
            mw=250.0,
            logP=2.0,
            psa=50.0,
            n_hbd=1,
            is_pgp_substrate=False,
            logD_pH74=1.8,
        )

    def _poor_cns(self):
        return predict_bbb_penetration(
            drug_name="PoorCNS",
            mw=600.0,
            logP=5.0,
            psa=150.0,
            n_hbd=6,
            is_pgp_substrate=True,
            logD_pH74=3.0,
        )

    def test_returns_correct_type(self):
        assert isinstance(self._ideal_cns_plus(), BBBPenetrationResult)

    def test_ideal_drug_score_5(self):
        result = self._ideal_cns_plus()
        assert result.cns_score == 5

    def test_ideal_drug_kp_uu_1(self):
        result = self._ideal_cns_plus()
        assert result.kp_uu == pytest.approx(1.0)

    def test_ideal_drug_cns_plus_classification(self):
        result = self._ideal_cns_plus()
        assert result.cns_classification == "CNS+"

    def test_poor_cns_low_score(self):
        result = self._poor_cns()
        assert result.cns_score == 0

    def test_poor_cns_low_kp_uu(self):
        result = self._poor_cns()
        assert result.kp_uu == pytest.approx(0.01)

    def test_poor_cns_classification(self):
        result = self._poor_cns()
        assert result.cns_classification == "CNS-"

    def test_score_3_intermediate_classification(self):
        result = predict_bbb_penetration(
            drug_name="Mid",
            mw=300.0,
            logP=2.0,
            psa=60.0,
            n_hbd=2,
            is_pgp_substrate=True,  # miss one
            logD_pH74=1.5,
        )
        # score: mw(+1), logP(+1), psa(+1), hbd(+1), pgp(-) = 4 → CNS+
        assert result.cns_score == 4
        assert result.cns_classification == "CNS+"

    def test_score_3_cns_intermediate(self):
        result = predict_bbb_penetration(
            drug_name="Mid3",
            mw=300.0,
            logP=2.0,
            psa=60.0,
            n_hbd=3,  # miss hbd (>=3 fails)
            is_pgp_substrate=True,  # miss pgp
            logD_pH74=1.5,
        )
        assert result.cns_score == 3
        assert result.cns_classification == "CNS+/-"

    def test_pgp_substrate_reduces_score(self):
        r_no_pgp = predict_bbb_penetration(
            drug_name="NoPgp", mw=300.0, logP=2.0, psa=60.0,
            n_hbd=2, is_pgp_substrate=False, logD_pH74=1.5,
        )
        r_pgp = predict_bbb_penetration(
            drug_name="Pgp", mw=300.0, logP=2.0, psa=60.0,
            n_hbd=2, is_pgp_substrate=True, logD_pH74=1.5,
        )
        assert r_pgp.cns_score == r_no_pgp.cns_score - 1

    def test_log_bb_formula(self):
        result = predict_bbb_penetration(
            drug_name="LogBBTest",
            mw=300.0,
            logP=2.0,
            psa=60.0,
            n_hbd=2,
            is_pgp_substrate=False,
            logD_pH74=1.5,
        )
        expected = 0.152 * 2.0 - 0.0148 * 60.0 + 0.139
        assert result.log_bb == pytest.approx(expected, rel=1e-6)

    def test_high_psa_low_log_bb(self):
        r_low = predict_bbb_penetration(
            drug_name="LowPSA", mw=300.0, logP=2.0, psa=30.0,
            n_hbd=1, is_pgp_substrate=False, logD_pH74=1.5,
        )
        r_high = predict_bbb_penetration(
            drug_name="HighPSA", mw=300.0, logP=2.0, psa=120.0,
            n_hbd=1, is_pgp_substrate=False, logD_pH74=1.5,
        )
        assert r_high.log_bb < r_low.log_bb

    def test_stored_fields(self):
        result = predict_bbb_penetration(
            drug_name="StoredTest", mw=350.0, logP=1.5, psa=75.0,
            n_hbd=2, is_pgp_substrate=False, logD_pH74=1.2,
        )
        assert result.drug_name == "StoredTest"
        assert result.mw == 350.0
        assert result.logP == 1.5
        assert result.psa == 75.0
        assert result.n_hbd == 2
        assert result.is_pgp_substrate is False
        assert result.logD_pH74 == 1.2

    def test_score_range_0_to_5(self):
        for logP_val, psa_val, mw_val, hbd, pgp in [
            (2.0, 50.0, 300.0, 1, False),
            (5.0, 120.0, 600.0, 5, True),
            (0.0, 50.0, 300.0, 1, False),
        ]:
            result = predict_bbb_penetration(
                drug_name="RangeTest", mw=mw_val, logP=logP_val, psa=psa_val,
                n_hbd=hbd, is_pgp_substrate=pgp, logD_pH74=logP_val,
            )
            assert 0 <= result.cns_score <= 5

    def test_notes_is_string(self):
        result = self._ideal_cns_plus()
        assert isinstance(result.notes, str)

    def test_result_is_frozen(self):
        result = self._ideal_cns_plus()
        with pytest.raises((AttributeError, TypeError)):
            result.cns_score = 0  # type: ignore[misc]


class TestSimulateCNSPK:
    def _run(self, **kwargs):
        defaults = dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_L=50.0,
            kp_uu=0.5,
            t_end_h=24.0,
            dt_h=0.05,
        )
        defaults.update(kwargs)
        return simulate_cns_pk(**defaults)

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            self._run(drug_name="")

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            self._run(dose_mg=0.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            self._run(cl_L_per_h=0.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            self._run(vd_L=0.0)

    def test_negative_kp_uu_raises(self):
        with pytest.raises(ValueError, match="kp_uu"):
            self._run(kp_uu=-0.1)

    def test_returns_dict(self):
        result = self._run()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = self._run()
        for key in ["times_h", "c_plasma_mg_L", "c_brain_mg_L", "kp_uu",
                    "cmax_plasma_mg_L", "cmax_brain_mg_L", "auc_plasma_mg_h_L",
                    "auc_brain_mg_h_L", "auc_brain_plasma_ratio"]:
            assert key in result

    def test_plasma_starts_at_dose_over_vd(self):
        result = self._run(dose_mg=100.0, vd_L=50.0)
        assert result["c_plasma_mg_L"][0] == pytest.approx(100.0 / 50.0, rel=0.01)

    def test_brain_starts_at_kp_uu_times_plasma(self):
        result = self._run(kp_uu=0.3)
        expected_brain = result["c_plasma_mg_L"][0] * 0.3
        assert result["c_brain_mg_L"][0] == pytest.approx(expected_brain, rel=1e-6)

    def test_zero_kp_uu_zero_brain(self):
        result = self._run(kp_uu=0.0)
        for c in result["c_brain_mg_L"]:
            assert c == pytest.approx(0.0)

    def test_auc_brain_plasma_ratio_equals_kp_uu(self):
        result = self._run(kp_uu=0.5)
        assert result["auc_brain_plasma_ratio"] == pytest.approx(0.5, rel=1e-6)

    def test_plasma_declines_over_time(self):
        result = self._run()
        assert result["c_plasma_mg_L"][-1] < result["c_plasma_mg_L"][0]

    def test_list_lengths_match(self):
        result = self._run()
        n = len(result["times_h"])
        assert len(result["c_plasma_mg_L"]) == n
        assert len(result["c_brain_mg_L"]) == n

    def test_times_start_zero(self):
        result = self._run()
        assert result["times_h"][0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = self._run(t_end_h=12.0)
        assert result["times_h"][-1] == pytest.approx(12.0)

    def test_dose_proportionality(self):
        r1 = self._run(dose_mg=10.0)
        r2 = self._run(dose_mg=20.0)
        assert r2["auc_plasma_mg_h_L"] == pytest.approx(r1["auc_plasma_mg_h_L"] * 2.0, rel=0.01)

    def test_kp_uu_1_auc_ratio_1(self):
        result = self._run(kp_uu=1.0)
        assert result["auc_brain_plasma_ratio"] == pytest.approx(1.0, rel=1e-6)
