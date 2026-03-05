"""Tests for CNS PBPK module — BBB penetration prediction and CNS PK simulation."""
from __future__ import annotations

import pytest
import numpy as np

from omega_pbpk.core.cns_pbpk import (
    CNSPenetrationResult,
    predict_bbb_penetration,
    simulate_cns_pk,
    assess_cns_exposure,
)

_TIMES = [float(i) for i in range(25)]
_CONC = [5.0 * np.exp(-0.2 * t) for t in _TIMES]


class TestPredictBBB:
    def test_low_logP_low_penetration(self):
        r = predict_bbb_penetration("X", logP=0, MW=200, psa=30)
        assert r.kp_uu_brain < 0.3

    def test_high_logP_high_penetration(self):
        r = predict_bbb_penetration("X", logP=4, MW=200, psa=30)
        assert r.kp_uu_brain > 0.1

    def test_high_MW_reduces_penetration(self):
        low_mw = predict_bbb_penetration("X", logP=2, MW=200, psa=60)
        high_mw = predict_bbb_penetration("X", logP=2, MW=600, psa=60)
        assert high_mw.kp_uu_brain < low_mw.kp_uu_brain

    def test_high_PSA_reduces_penetration(self):
        low_psa = predict_bbb_penetration("X", logP=2, MW=300, psa=30)
        high_psa = predict_bbb_penetration("X", logP=2, MW=300, psa=120)
        assert high_psa.kp_uu_brain < low_psa.kp_uu_brain

    def test_pgp_substrate_reduces_penetration(self):
        no_pgp = predict_bbb_penetration("X", logP=3, MW=300, psa=60, is_pgp_substrate=False)
        with_pgp = predict_bbb_penetration("X", logP=3, MW=300, psa=60, is_pgp_substrate=True)
        assert with_pgp.kp_uu_brain < no_pgp.kp_uu_brain

    def test_cns_plus_classification(self):
        r = predict_bbb_penetration("X", logP=4, MW=200, psa=30)
        if r.kp_uu_brain > 0.3:
            assert r.cns_classification == "CNS+"

    def test_cns_minus_classification(self):
        r = predict_bbb_penetration("X", logP=0, MW=600, psa=120, is_pgp_substrate=True)
        assert r.cns_classification == "CNS-"

    def test_risk_factors_populated(self):
        r = predict_bbb_penetration("X", logP=1, MW=600, psa=120, is_pgp_substrate=True)
        assert len(r.risk_factors) > 0

    def test_kp_uu_bounded(self):
        for logP, mw, psa in [(0, 200, 30), (4, 200, 30), (2, 600, 120), (-1, 500, 100)]:
            r = predict_bbb_penetration("X", logP=logP, MW=mw, psa=psa)
            assert 0.001 <= r.kp_uu_brain <= 1.0

    def test_returns_result_type(self):
        r = predict_bbb_penetration("Aspirin", logP=2, MW=300, psa=60)
        assert isinstance(r, CNSPenetrationResult)

    def test_drug_name_stored(self):
        r = predict_bbb_penetration("Aspirin", logP=1.2, MW=180, psa=63)
        assert r.drug_name == "Aspirin"


class TestSimulateCNSPK:
    def test_returns_dict_with_keys(self):
        result = simulate_cns_pk(_TIMES, _CONC, kp_uu_brain=0.3)
        expected = {"times_h", "brain_conc", "plasma_conc", "kp_uu_ss",
                    "auc_brain", "auc_plasma", "brain_plasma_auc_ratio"}
        assert expected.issubset(set(result.keys()))

    def test_auc_brain_positive(self):
        result = simulate_cns_pk(_TIMES, _CONC, kp_uu_brain=0.3)
        assert result["auc_brain"] > 0

    def test_auc_ratio_reasonable(self):
        result = simulate_cns_pk(_TIMES, _CONC, kp_uu_brain=0.3, fup=1.0)
        assert 0 < result["brain_plasma_auc_ratio"] < 2

    def test_higher_kp_uu_higher_brain_auc(self):
        high_kp = simulate_cns_pk(_TIMES, _CONC, kp_uu_brain=0.5)
        low_kp = simulate_cns_pk(_TIMES, _CONC, kp_uu_brain=0.1)
        assert low_kp["auc_brain"] < high_kp["auc_brain"]

    def test_brain_conc_length(self):
        result = simulate_cns_pk(_TIMES, _CONC, kp_uu_brain=0.3)
        assert len(result["brain_conc"]) > 1

    def test_kp_uu_ss_matches_input(self):
        result = simulate_cns_pk(_TIMES, _CONC, kp_uu_brain=0.25)
        assert result["kp_uu_ss"] == 0.25


class TestAssessCNSExposure:
    def test_returns_dict_with_penetration_and_pk(self):
        r = assess_cns_exposure("Drug", logP=2, MW=300, psa=60,
                                plasma_times=_TIMES, plasma_conc=_CONC)
        assert "penetration" in r and "pk" in r

    def test_penetration_is_result_type(self):
        r = assess_cns_exposure("Drug", logP=2, MW=300, psa=60,
                                plasma_times=_TIMES, plasma_conc=_CONC)
        assert isinstance(r["penetration"], CNSPenetrationResult)

    def test_pk_has_auc_keys(self):
        r = assess_cns_exposure("Drug", logP=2, MW=300, psa=60,
                                plasma_times=_TIMES, plasma_conc=_CONC)
        assert "auc_brain" in r["pk"]

    def test_full_workflow_aspirin(self):
        r = assess_cns_exposure("Aspirin", logP=1.2, MW=180, psa=63,
                                plasma_times=_TIMES, plasma_conc=_CONC, fup=0.01)
        pen = r["penetration"]
        assert 0.001 <= pen.kp_uu_brain <= 1.0
