"""Tests for Phase 569 — Tissue Binding Predictor."""

import pytest

from omega_pbpk.prediction.tissue_binding_predictor import (
    kp_ratio,
    predict_fu_tissue,
    screen_tissue_distribution,
    vd_from_tissue_binding,
)

# ---------------------------------------------------------------------------
# predict_fu_tissue
# ---------------------------------------------------------------------------


class TestPredictFuTissue:
    def test_returns_dict_with_required_keys(self):
        result = predict_fu_tissue(logP=2.0, drug_type="neutral", tissue="liver")
        assert set(result.keys()) == {
            "tissue",
            "drug_type",
            "kp_tissue",
            "fu_tissue",
            "distribution_category",
        }

    def test_tissue_echoed(self):
        result = predict_fu_tissue(logP=1.0, drug_type="neutral", tissue="muscle")
        assert result["tissue"] == "muscle"

    def test_drug_type_echoed(self):
        result = predict_fu_tissue(logP=1.0, drug_type="acid", tissue="muscle")
        assert result["drug_type"] == "acid"

    def test_fu_tissue_between_0_and_1(self):
        for tissue in ["muscle", "adipose", "liver", "brain", "kidney", "lung", "heart"]:
            r = predict_fu_tissue(logP=3.0, drug_type="neutral", tissue=tissue)
            assert 0.0 < r["fu_tissue"] < 1.0, f"fu_tissue out of range for {tissue}"

    def test_adipose_highest_kp_for_lipophilic(self):
        """High-logP drug: adipose should have the highest Kp among tissues."""
        tissues = ["muscle", "adipose", "liver", "brain", "kidney", "lung", "heart"]
        kps = {
            t: predict_fu_tissue(logP=4.0, drug_type="neutral", tissue=t)["kp_tissue"]
            for t in tissues
        }
        assert kps["adipose"] == max(kps.values())

    def test_brain_phospholipid_binding_notable(self):
        """Brain has the highest phospholipid fraction (0.35) so it should rank
        second (after adipose) for moderately lipophilic compounds where the
        phospholipid term dominates over neutral-lipid differences."""
        # At logP=1 P=10, P^0.3 ≈ 2. Brain: 10*0.04 + 2*0.35 = 0.4+0.7=1.1+0.9=2.0
        # Liver: 10*0.04 + 2*0.10 = 0.4+0.2=0.6+0.9=1.5; Kidney: 10*0.02+2*0.10=0.2+0.2+0.9=1.3
        r_brain = predict_fu_tissue(logP=1.0, drug_type="neutral", tissue="brain")
        r_kidney = predict_fu_tissue(logP=1.0, drug_type="neutral", tissue="kidney")
        assert r_brain["kp_tissue"] > r_kidney["kp_tissue"]

    def test_higher_logP_lower_fu_tissue(self):
        """More lipophilic drug → lower fu_tissue (more bound)."""
        r_low = predict_fu_tissue(logP=0.0, drug_type="neutral", tissue="adipose")
        r_high = predict_fu_tissue(logP=4.0, drug_type="neutral", tissue="adipose")
        assert r_high["fu_tissue"] < r_low["fu_tissue"]

    def test_strong_base_higher_kp_than_neutral(self):
        """strong_base multiplies kp by 2.0 due to lysosomal trapping."""
        r_neutral = predict_fu_tissue(logP=2.0, drug_type="neutral", tissue="liver")
        r_base = predict_fu_tissue(logP=2.0, drug_type="strong_base", tissue="liver")
        assert r_base["kp_tissue"] == pytest.approx(r_neutral["kp_tissue"] * 2.0, rel=1e-9)

    def test_distribution_category_extensive(self):
        """High-logP adipose → extensive distribution."""
        r = predict_fu_tissue(logP=5.0, drug_type="neutral", tissue="adipose")
        assert r["distribution_category"] == "extensive"

    def test_distribution_category_low(self):
        """Very low logP, low-lipid tissue → low distribution."""
        r = predict_fu_tissue(logP=-2.0, drug_type="neutral", tissue="muscle")
        # kp = 0.01*0.01 + 0.01^0.3*0.07 + 0.9; P^0.3 at logP=-2 ≈ 0.251
        # ≈ 0.0001 + 0.0176 + 0.9 = 0.917 < 1 → low
        assert r["distribution_category"] == "low"

    def test_distribution_category_moderate(self):
        """Intermediate Kp → moderate distribution."""
        r = predict_fu_tissue(logP=1.0, drug_type="neutral", tissue="brain")
        # From earlier calc ≈ 2.0 (between 1 and 10)
        assert r["distribution_category"] == "moderate"

    def test_invalid_tissue_raises(self):
        with pytest.raises(ValueError, match="Invalid tissue"):
            predict_fu_tissue(logP=1.0, drug_type="neutral", tissue="spleen")

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="Invalid drug_type"):
            predict_fu_tissue(logP=1.0, drug_type="ampholyte", tissue="liver")

    def test_all_tissues_accepted(self):
        valid_tissues = ["muscle", "adipose", "liver", "brain", "kidney", "lung", "heart"]
        for tissue in valid_tissues:
            r = predict_fu_tissue(logP=1.0, drug_type="neutral", tissue=tissue)
            assert r["tissue"] == tissue


# ---------------------------------------------------------------------------
# kp_ratio
# ---------------------------------------------------------------------------


class TestKpRatio:
    def test_accumulation_when_greater_than_1(self):
        # kp=10, fu_plasma=0.5 → ratio=5 > 1
        assert kp_ratio(10.0, 0.5) == pytest.approx(5.0)

    def test_no_accumulation_when_less_than_1(self):
        # kp=1, fu_plasma=0.5 → ratio=0.5 < 1
        assert kp_ratio(1.0, 0.5) == pytest.approx(0.5)

    def test_formula_correctness(self):
        assert kp_ratio(3.0, 0.2) == pytest.approx(0.6, rel=1e-9)

    def test_zero_fu_plasma(self):
        assert kp_ratio(5.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# vd_from_tissue_binding
# ---------------------------------------------------------------------------


class TestVdFromTissueBinding:
    def test_single_tissue_matches_formula(self):
        # Vd = 3 + Vt*Kp = 3 + 10*5 = 53
        vd = vd_from_tissue_binding(fu_plasma=0.5, tissue_kp_list=[(10.0, 5.0)])
        assert vd == pytest.approx(53.0, rel=1e-9)

    def test_no_tissue_returns_plasma_volume(self):
        vd = vd_from_tissue_binding(fu_plasma=0.5, tissue_kp_list=[])
        assert vd == pytest.approx(3.0, rel=1e-9)

    def test_multiple_tissues(self):
        # Vd = 3 + 10*2 + 5*3 = 3 + 20 + 15 = 38
        vd = vd_from_tissue_binding(fu_plasma=0.5, tissue_kp_list=[(10.0, 2.0), (5.0, 3.0)])
        assert vd == pytest.approx(38.0, rel=1e-9)

    def test_large_kp_gives_large_vd(self):
        vd = vd_from_tissue_binding(fu_plasma=0.1, tissue_kp_list=[(40.0, 100.0)])
        assert vd > 1000.0


# ---------------------------------------------------------------------------
# screen_tissue_distribution
# ---------------------------------------------------------------------------


class TestScreenTissueDistribution:
    def test_returns_seven_entries(self):
        result = screen_tissue_distribution(logP=2.0, drug_type="neutral", fu_plasma=0.5)
        assert len(result) == 7

    def test_sorted_descending_by_kp(self):
        result = screen_tissue_distribution(logP=3.0, drug_type="neutral", fu_plasma=0.5)
        kps = [r["kp_tissue"] for r in result]
        assert kps == sorted(kps, reverse=True)

    def test_first_entry_is_adipose_for_lipophilic(self):
        result = screen_tissue_distribution(logP=4.0, drug_type="neutral", fu_plasma=0.5)
        assert result[0]["tissue"] == "adipose"

    def test_accumulation_flag_set(self):
        # logP=5, fu_plasma=0.9 → adipose kp very large → accumulation True
        result = screen_tissue_distribution(logP=5.0, drug_type="neutral", fu_plasma=0.9)
        adipose_entry = next(r for r in result if r["tissue"] == "adipose")
        assert adipose_entry["accumulation"] is True

    def test_accumulation_flag_false_when_low_kp(self):
        # logP=-3, fu_plasma=0.01 → all kps low, ratios < 1
        result = screen_tissue_distribution(logP=-3.0, drug_type="neutral", fu_plasma=0.01)
        for entry in result:
            # kp_ratio = kp * 0.01; even if kp=2, ratio=0.02 < 1
            expected = (entry["kp_tissue"] * 0.01) > 1.0
            assert entry["accumulation"] == expected

    def test_required_keys_in_each_entry(self):
        result = screen_tissue_distribution(logP=1.0, drug_type="neutral", fu_plasma=0.5)
        for entry in result:
            assert set(entry.keys()) == {"tissue", "kp_tissue", "fu_tissue", "accumulation"}
