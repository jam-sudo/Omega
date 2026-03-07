"""Tests for Phase 511: Drug Efflux Ratio Predictor."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.efflux_ratio_predictor import (
    EffluxImpactResult,
    assess_efflux_impact,
    predict_bcrp_substrate,
    predict_efflux_ratio,
    predict_pgp_substrate,
    rank_efflux_concern,
)

# ---------------------------------------------------------------------------
# predict_efflux_ratio
# ---------------------------------------------------------------------------


class TestPredictEffluxRatio:
    def test_base_er_small_low_psa_drug(self):
        """Small MW, low PSA, moderate logP → ER near 1."""
        er = predict_efflux_ratio(logP=2.5, mw=200.0, psa_A2=30.0, n_hbd=1)
        assert er == pytest.approx(1.0)

    def test_high_mw_high_psa_gives_high_er(self):
        """MW=700, PSA=80 → high ER."""
        er = predict_efflux_ratio(logP=1.5, mw=700.0, psa_A2=80.0, n_hbd=2)
        # MW contribution: 2*(300/100)=6, PSA+logP: +1.5, HBD amine: +1.0 → 9.5
        assert er > 6.0

    def test_mw_600_psa_70_gives_high_er(self):
        """MW=600, PSA=70, logP=1 → ER substantially > 1."""
        er = predict_efflux_ratio(logP=1.0, mw=600.0, psa_A2=70.0, n_hbd=3)
        assert er > 5.0

    def test_high_logp_reduces_er(self):
        """High logP (>4) applies -0.5 penalty to ER."""
        er_low = predict_efflux_ratio(logP=2.0, mw=450.0, psa_A2=50.0, n_hbd=1)
        er_high = predict_efflux_ratio(logP=5.0, mw=450.0, psa_A2=50.0, n_hbd=1)
        assert er_high < er_low

    def test_high_logp_minus_05(self):
        """logP > 4 subtracts 0.5 from ER (use MW=500 so base ER is 3.0 to avoid clamping)."""
        # MW=500: base=1 + 2*(100/100)=3.0; logP=3.0 → 3.0; logP=5.0 → 2.5
        er_ref = predict_efflux_ratio(logP=3.0, mw=500.0, psa_A2=30.0, n_hbd=0)
        er_high = predict_efflux_ratio(logP=5.0, mw=500.0, psa_A2=30.0, n_hbd=0)
        assert er_high == pytest.approx(er_ref - 0.5, abs=1e-9)

    def test_psa_logp_threshold(self):
        """PSA > 60 and logP < 2 adds 1.5."""
        er_no_polar = predict_efflux_ratio(logP=3.0, mw=300.0, psa_A2=50.0, n_hbd=0)
        er_polar = predict_efflux_ratio(logP=1.0, mw=300.0, psa_A2=70.0, n_hbd=0)
        assert er_polar == pytest.approx(er_no_polar + 1.5, abs=1e-9)

    def test_has_amine_adds_1(self):
        """has_amine=True adds +1.0."""
        er_no = predict_efflux_ratio(logP=3.0, mw=300.0, psa_A2=30.0, n_hbd=0)
        er_yes = predict_efflux_ratio(logP=3.0, mw=300.0, psa_A2=30.0, n_hbd=0, has_amine=True)
        assert er_yes == pytest.approx(er_no + 1.0, abs=1e-9)

    def test_er_clamped_minimum_1(self):
        """ER is always >= 1.0."""
        er = predict_efflux_ratio(logP=10.0, mw=100.0, psa_A2=10.0, n_hbd=0)
        assert er >= 1.0

    def test_er_clamped_maximum_20(self):
        """Very large MW drugs are clamped at ER=20."""
        er = predict_efflux_ratio(logP=0.0, mw=5000.0, psa_A2=200.0, n_hbd=10, has_amine=True)
        assert er == pytest.approx(20.0)

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError, match="mw must be positive"):
            predict_efflux_ratio(logP=2.0, mw=0.0, psa_A2=50.0, n_hbd=1)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError):
            predict_efflux_ratio(logP=2.0, mw=-100.0, psa_A2=50.0, n_hbd=1)

    def test_hbd_heuristic_amine(self):
        """n_hbd >= 2 and PSA > 40 triggers amine heuristic."""
        er_low_hbd = predict_efflux_ratio(logP=2.0, mw=300.0, psa_A2=50.0, n_hbd=1)
        er_high_hbd = predict_efflux_ratio(logP=2.0, mw=300.0, psa_A2=50.0, n_hbd=2)
        assert er_high_hbd > er_low_hbd

    def test_mw_exactly_400_no_mw_contribution(self):
        """MW exactly 400 → no MW contribution."""
        er = predict_efflux_ratio(logP=3.0, mw=400.0, psa_A2=30.0, n_hbd=0)
        assert er == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# predict_pgp_substrate
# ---------------------------------------------------------------------------


class TestPredictPgpSubstrate:
    def test_pgp_positive(self):
        """MW > 400 and PSA > 60 → P-gp substrate."""
        assert predict_pgp_substrate(logP=2.0, mw=500.0, psa_A2=70.0, n_hbd=2) is True

    def test_pgp_negative_low_mw(self):
        """MW <= 400 → not P-gp substrate."""
        assert predict_pgp_substrate(logP=2.0, mw=350.0, psa_A2=80.0, n_hbd=2) is False

    def test_pgp_negative_low_psa(self):
        """PSA <= 60 → not P-gp substrate."""
        assert predict_pgp_substrate(logP=2.0, mw=500.0, psa_A2=50.0, n_hbd=2) is False

    def test_pgp_boundary_mw(self):
        """MW = 400.1, PSA > 60 → True."""
        assert predict_pgp_substrate(logP=1.0, mw=400.1, psa_A2=65.0, n_hbd=1) is True

    def test_pgp_invalid_mw(self):
        with pytest.raises(ValueError):
            predict_pgp_substrate(logP=1.0, mw=0.0, psa_A2=70.0, n_hbd=1)


# ---------------------------------------------------------------------------
# predict_bcrp_substrate
# ---------------------------------------------------------------------------


class TestPredictBcrpSubstrate:
    def test_bcrp_positive_low_psa(self):
        """MW > 350 and PSA < 40 → BCRP substrate."""
        assert predict_bcrp_substrate(logP=2.0, mw=400.0, psa_A2=30.0) is True

    def test_bcrp_positive_high_logp(self):
        """MW > 350 and logP > 3 → BCRP substrate."""
        assert predict_bcrp_substrate(logP=4.0, mw=400.0, psa_A2=50.0) is True

    def test_bcrp_negative(self):
        """MW <= 350 → not BCRP substrate."""
        assert predict_bcrp_substrate(logP=2.0, mw=300.0, psa_A2=30.0) is False

    def test_bcrp_negative_high_psa_low_logp(self):
        """MW > 350 but PSA >= 40 and logP <= 3 → False."""
        assert predict_bcrp_substrate(logP=2.0, mw=400.0, psa_A2=50.0) is False

    def test_bcrp_invalid_mw(self):
        with pytest.raises(ValueError):
            predict_bcrp_substrate(logP=1.0, mw=-1.0, psa_A2=40.0)


# ---------------------------------------------------------------------------
# assess_efflux_impact
# ---------------------------------------------------------------------------


class TestAssessEffluxImpact:
    def test_er_10_severe_concern(self):
        """ER=10 → 'severe' concern."""
        result = assess_efflux_impact(efflux_ratio=10.0)
        assert result.pgp_concern == "severe"

    def test_er_1_no_concern(self):
        """ER=1 → 'none' concern."""
        result = assess_efflux_impact(efflux_ratio=1.0)
        assert result.pgp_concern == "none"

    def test_er_3_mild_concern(self):
        """ER=3 → 'mild' concern."""
        result = assess_efflux_impact(efflux_ratio=3.0)
        assert result.pgp_concern == "mild"

    def test_er_7_moderate_concern(self):
        """ER=7 → 'moderate' concern."""
        result = assess_efflux_impact(efflux_ratio=7.0)
        assert result.pgp_concern == "moderate"

    def test_kp_uu_effective(self):
        """kp_uu_effective = kp_uu_baseline / ER."""
        result = assess_efflux_impact(efflux_ratio=4.0, kp_uu_baseline=0.8)
        assert result.kp_uu_effective == pytest.approx(0.8 / 4.0)

    def test_gut_bioavailability_impact(self):
        """gut_bioavailability_impact = 1 / ER."""
        result = assess_efflux_impact(efflux_ratio=5.0)
        assert result.gut_bioavailability_impact == pytest.approx(1.0 / 5.0)

    def test_gut_impact_less_than_1_when_er_gt_1(self):
        """gut_bioavailability_impact < 1 when ER > 1."""
        result = assess_efflux_impact(efflux_ratio=2.0)
        assert result.gut_bioavailability_impact < 1.0

    def test_cns_penetration_impact(self):
        """cns_penetration_impact = kp_uu_effective / kp_uu_baseline."""
        result = assess_efflux_impact(efflux_ratio=2.0, kp_uu_baseline=1.0)
        assert result.cns_penetration_impact == pytest.approx(0.5)

    def test_er_below_1_raises(self):
        with pytest.raises(ValueError, match="efflux_ratio"):
            assess_efflux_impact(efflux_ratio=0.5)

    def test_invalid_fu_plasma_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            assess_efflux_impact(efflux_ratio=2.0, fu_plasma=0.0)

    def test_invalid_kp_uu_raises(self):
        with pytest.raises(ValueError, match="kp_uu_baseline"):
            assess_efflux_impact(efflux_ratio=2.0, kp_uu_baseline=-0.1)

    def test_returns_frozen_dataclass(self):
        result = assess_efflux_impact(efflux_ratio=3.0)
        assert isinstance(result, EffluxImpactResult)
        with pytest.raises(Exception):
            result.efflux_ratio = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# rank_efflux_concern
# ---------------------------------------------------------------------------


class TestRankEffluxConcern:
    def _drugs(self):
        return [
            {"name": "DrugA", "logP": 1.0, "mw": 200.0, "psa_A2": 30.0, "n_hbd": 0},
            {"name": "DrugB", "logP": 1.0, "mw": 700.0, "psa_A2": 90.0, "n_hbd": 3},
            {"name": "DrugC", "logP": 5.0, "mw": 450.0, "psa_A2": 55.0, "n_hbd": 1},
        ]

    def test_sorted_descending(self):
        """rank_efflux_concern returns drugs sorted by ER descending."""
        ranked = rank_efflux_concern(self._drugs())
        ers = [d["efflux_ratio"] for d in ranked]
        assert ers == sorted(ers, reverse=True)

    def test_high_mw_drug_ranked_first(self):
        """DrugB (MW=700) should have highest ER."""
        ranked = rank_efflux_concern(self._drugs())
        assert ranked[0]["name"] == "DrugB"

    def test_efflux_ratio_added(self):
        """Each drug dict has efflux_ratio added."""
        ranked = rank_efflux_concern(self._drugs())
        for d in ranked:
            assert "efflux_ratio" in d
            assert d["efflux_ratio"] >= 1.0

    def test_pgp_bcrp_added(self):
        """pgp_substrate and bcrp_substrate are added."""
        ranked = rank_efflux_concern(self._drugs())
        for d in ranked:
            assert "pgp_substrate" in d
            assert "bcrp_substrate" in d

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert rank_efflux_concern([]) == []

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            rank_efflux_concern("not a list")  # type: ignore[arg-type]

    def test_single_drug(self):
        ranked = rank_efflux_concern(
            [{"name": "X", "logP": 2.0, "mw": 300.0, "psa_A2": 40.0, "n_hbd": 1}]
        )
        assert len(ranked) == 1
        assert ranked[0]["name"] == "X"
