"""Tests for peptide PK predictor — Phase 468."""

import math
import pytest

from omega_pbpk.prediction.peptide_pk_predictor import (
    PeptidePKResult,
    estimate_peptide_clearance,
    estimate_peptide_half_life,
    peptide_route_suitability,
    predict_peptide_pk,
)


# ---------------------------------------------------------------------------
# Unit tests: estimate_peptide_clearance
# ---------------------------------------------------------------------------

class TestEstimatePeptideClearance:
    def test_returns_positive(self):
        cl = estimate_peptide_clearance(mw_Da=1000.0)
        assert cl > 0

    def test_larger_mw_higher_cl(self):
        cl_small = estimate_peptide_clearance(mw_Da=500.0)
        cl_large = estimate_peptide_clearance(mw_Da=5000.0)
        assert cl_large > cl_small

    def test_cyclic_lower_cl_than_linear(self):
        cl_linear = estimate_peptide_clearance(mw_Da=1000.0, is_cyclic=False)
        cl_cyclic = estimate_peptide_clearance(mw_Da=1000.0, is_cyclic=True)
        assert cl_cyclic < cl_linear
        # ~40% reduction
        assert cl_cyclic == pytest.approx(cl_linear * 0.60, rel=0.01)

    def test_pegylation_dramatically_reduces_cl(self):
        cl_unpegged = estimate_peptide_clearance(mw_Da=2000.0, pegylated=False)
        cl_pegged = estimate_peptide_clearance(mw_Da=2000.0, pegylated=True, peg_kDa=20.0)
        assert cl_pegged < cl_unpegged * 0.5

    def test_disulfide_reduces_cl(self):
        cl_no_ds = estimate_peptide_clearance(mw_Da=1500.0, n_disulfide=0)
        cl_with_ds = estimate_peptide_clearance(mw_Da=1500.0, n_disulfide=2)
        assert cl_with_ds < cl_no_ds

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError):
            estimate_peptide_clearance(mw_Da=0.0)

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError):
            estimate_peptide_clearance(mw_Da=-500.0)

    def test_negative_disulfide_raises(self):
        with pytest.raises(ValueError):
            estimate_peptide_clearance(mw_Da=1000.0, n_disulfide=-1)

    def test_negative_peg_kda_raises(self):
        with pytest.raises(ValueError):
            estimate_peptide_clearance(mw_Da=1000.0, pegylated=True, peg_kDa=-5.0)

    def test_larger_peg_lower_cl(self):
        cl_small_peg = estimate_peptide_clearance(mw_Da=2000.0, pegylated=True, peg_kDa=5.0)
        cl_large_peg = estimate_peptide_clearance(mw_Da=2000.0, pegylated=True, peg_kDa=40.0)
        assert cl_large_peg < cl_small_peg


# ---------------------------------------------------------------------------
# Unit tests: estimate_peptide_half_life
# ---------------------------------------------------------------------------

class TestEstimatePeptideHalfLife:
    def test_basic(self):
        # t_half = (Vd * ln2) / CL; with Vd=1.0, CL=1.0 → t_half = ln(2)
        t_half = estimate_peptide_half_life(cl_L_per_h=1.0, vd_L=1.0)
        assert abs(t_half - math.log(2)) < 1e-9

    def test_higher_cl_shorter_half_life(self):
        t1 = estimate_peptide_half_life(cl_L_per_h=5.0, vd_L=10.0)
        t2 = estimate_peptide_half_life(cl_L_per_h=1.0, vd_L=10.0)
        assert t1 < t2

    def test_higher_vd_longer_half_life(self):
        t1 = estimate_peptide_half_life(cl_L_per_h=2.0, vd_L=20.0)
        t2 = estimate_peptide_half_life(cl_L_per_h=2.0, vd_L=5.0)
        assert t1 > t2

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError):
            estimate_peptide_half_life(cl_L_per_h=0.0, vd_L=10.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError):
            estimate_peptide_half_life(cl_L_per_h=1.0, vd_L=0.0)


# ---------------------------------------------------------------------------
# Unit tests: peptide_route_suitability
# ---------------------------------------------------------------------------

class TestPeptideRouteSuitability:
    def test_returns_required_keys(self):
        scores = peptide_route_suitability(mw_Da=1000.0, n_aa=8)
        assert set(scores.keys()) == {"iv", "sc", "oral", "inhaled"}

    def test_scores_in_range(self):
        scores = peptide_route_suitability(mw_Da=1500.0, n_aa=12)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_iv_highest_for_large_peptide(self):
        scores = peptide_route_suitability(mw_Da=50000.0, n_aa=400)
        assert scores["iv"] >= scores["sc"]
        assert scores["iv"] >= scores["oral"]
        assert scores["iv"] >= scores["inhaled"]

    def test_oral_lowest_for_large_peptide(self):
        scores = peptide_route_suitability(mw_Da=20000.0, n_aa=180)
        assert scores["oral"] < scores["iv"]

    def test_cyclic_increases_oral_score(self):
        s_linear = peptide_route_suitability(mw_Da=800.0, n_aa=6, is_cyclic=False)
        s_cyclic = peptide_route_suitability(mw_Da=800.0, n_aa=6, is_cyclic=True)
        assert s_cyclic["oral"] >= s_linear["oral"]

    def test_invalid_mw_raises(self):
        with pytest.raises(ValueError):
            peptide_route_suitability(mw_Da=0.0, n_aa=5)

    def test_invalid_n_aa_raises(self):
        with pytest.raises(ValueError):
            peptide_route_suitability(mw_Da=1000.0, n_aa=0)

    def test_invalid_stability_score_raises(self):
        with pytest.raises(ValueError):
            peptide_route_suitability(mw_Da=1000.0, n_aa=8, stability_score=1.5)

    def test_higher_stability_higher_inhaled_score(self):
        s_low = peptide_route_suitability(mw_Da=1000.0, n_aa=8, stability_score=0.1)
        s_high = peptide_route_suitability(mw_Da=1000.0, n_aa=8, stability_score=0.9)
        assert s_high["inhaled"] >= s_low["inhaled"]


# ---------------------------------------------------------------------------
# Integration tests: predict_peptide_pk
# ---------------------------------------------------------------------------

class TestPredictPeptidePk:
    def test_returns_dataclass(self):
        result = predict_peptide_pk("ACDEFGHIK", mw_Da=1000.0, n_aa=9)
        assert isinstance(result, PeptidePKResult)

    def test_frozen_dataclass(self):
        result = predict_peptide_pk("ACDEFGHIK", mw_Da=1000.0, n_aa=9)
        with pytest.raises((AttributeError, TypeError)):
            result.cl_L_per_h = 999.0  # type: ignore[misc]

    def test_larger_peptide_higher_cl(self):
        r_small = predict_peptide_pk("ACDE", mw_Da=500.0, n_aa=4)
        r_large = predict_peptide_pk("ACDEFGHIKLMNPQRST", mw_Da=2000.0, n_aa=17)
        assert r_large.cl_L_per_h > r_small.cl_L_per_h

    def test_cyclic_lower_cl(self):
        r_linear = predict_peptide_pk("ACDEFGH", mw_Da=800.0, n_aa=7, is_cyclic=False)
        r_cyclic = predict_peptide_pk("ACDEFGH", mw_Da=800.0, n_aa=7, is_cyclic=True)
        assert r_cyclic.cl_L_per_h < r_linear.cl_L_per_h

    def test_pegylated_longer_half_life(self):
        r_unpeg = predict_peptide_pk("ACDEFGH", mw_Da=1500.0, n_aa=12, pegylated=False)
        r_peg = predict_peptide_pk("ACDEFGH", mw_Da=1500.0, n_aa=12, pegylated=True, peg_kDa=20.0)
        assert r_peg.t_half_h > r_unpeg.t_half_h

    def test_oral_bioavailability_decreases_with_chain_length(self):
        r_short = predict_peptide_pk("ACDE", mw_Da=480.0, n_aa=4)
        r_long = predict_peptide_pk("ACDEFGHIKLMNPQRST", mw_Da=2100.0, n_aa=18)
        assert r_short.oral_bioavailability_pct >= r_long.oral_bioavailability_pct

    def test_oral_bioavailability_less_than_5_for_long(self):
        r = predict_peptide_pk("X" * 30, mw_Da=3500.0, n_aa=30)
        assert r.oral_bioavailability_pct <= 5.0

    def test_stability_score_in_range(self):
        result = predict_peptide_pk("ACDEFGH", mw_Da=900.0, n_aa=7)
        assert 0.0 <= result.stability_score <= 1.0

    def test_route_scores_in_range(self):
        result = predict_peptide_pk("ACDEFGH", mw_Da=900.0, n_aa=7)
        for score in result.route_scores.values():
            assert 0.0 <= score <= 1.0

    def test_route_scores_has_all_keys(self):
        result = predict_peptide_pk("ACDEFGH", mw_Da=900.0, n_aa=7)
        assert {"iv", "sc", "oral", "inhaled"} == set(result.route_scores.keys())

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            predict_peptide_pk("", mw_Da=1000.0, n_aa=9)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError):
            predict_peptide_pk("ACDE", mw_Da=0.0, n_aa=4)

    def test_zero_n_aa_raises(self):
        with pytest.raises(ValueError):
            predict_peptide_pk("ACDE", mw_Da=480.0, n_aa=0)

    def test_negative_n_disulfide_raises(self):
        with pytest.raises(ValueError):
            predict_peptide_pk("ACDE", mw_Da=480.0, n_aa=4, n_disulfide=-1)

    def test_notes_not_empty(self):
        result = predict_peptide_pk("ACDEFGH", mw_Da=900.0, n_aa=7)
        assert len(result.notes) > 0

    def test_sequence_preserved(self):
        seq = "ACDEFGHIKLM"
        result = predict_peptide_pk(seq, mw_Da=1300.0, n_aa=11)
        assert result.sequence == seq

    def test_pegylated_dramatic_cl_reduction(self):
        cl_base = estimate_peptide_clearance(mw_Da=2000.0, pegylated=False)
        cl_peg = estimate_peptide_clearance(mw_Da=2000.0, pegylated=True, peg_kDa=40.0)
        ratio = cl_base / cl_peg
        assert ratio > 2.0  # At least 2x reduction

    def test_disulfide_stabilizes(self):
        r_no_ds = predict_peptide_pk("ACDEFGH", mw_Da=900.0, n_aa=7, n_disulfide=0)
        r_with_ds = predict_peptide_pk("ACDEFGH", mw_Da=900.0, n_aa=7, n_disulfide=2)
        assert r_with_ds.stability_score > r_no_ds.stability_score

    def test_t_half_consistency(self):
        result = predict_peptide_pk("ACDEFGH", mw_Da=900.0, n_aa=7)
        expected = (result.vd_L * math.log(2)) / result.cl_L_per_h
        assert abs(result.t_half_h - expected) < 0.01
