"""Tests for tissue Kp predictor — Phase 836."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.partition_coefficient_predictor import (
    TISSUES,
    PartitionCoefficientResult,
    predict_kp,
    predict_kp_all_tissues,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> PartitionCoefficientResult:
    defaults = dict(
        drug_name="TestDrug",
        tissue="liver",
        logp=2.0,
        pka=None,
        fu_plasma=0.1,
        ionization_type="neutral",
    )
    defaults.update(kwargs)
    return predict_kp(**defaults)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_partition_coefficient_result(self):
        result = _default_result()
        assert isinstance(result, PartitionCoefficientResult)

    def test_result_is_frozen(self):
        result = _default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = _default_result(drug_name="Atorvastatin")
        assert result.drug_name == "Atorvastatin"

    def test_tissue_preserved(self):
        result = _default_result(tissue="brain")
        assert result.tissue == "brain"

    def test_fu_plasma_preserved(self):
        result = _default_result(fu_plasma=0.05)
        assert result.fu_plasma == pytest.approx(0.05, rel=1e-9)

    def test_method_is_string(self):
        result = _default_result()
        assert isinstance(result.method, str)

    def test_notes_is_string(self):
        result = _default_result()
        assert isinstance(result.notes, str)


# ---------------------------------------------------------------------------
# Value range tests
# ---------------------------------------------------------------------------


class TestValueRanges:
    def test_kp_positive_all_tissues(self):
        for tissue in TISSUES:
            result = predict_kp("Drug", tissue=tissue, logp=2.0, fu_plasma=0.1)
            assert result.kp > 0.0, f"kp <= 0 for {tissue}"

    def test_log_kp_consistent_with_kp(self):
        result = _default_result()
        assert abs(result.log_kp - math.log10(result.kp)) < 1e-9

    def test_fu_tissue_positive(self):
        result = _default_result()
        assert result.fu_tissue > 0.0

    def test_fu_tissue_at_most_one(self):
        result = _default_result()
        assert result.fu_tissue <= 1.0

    def test_kp_uu_positive(self):
        result = _default_result()
        assert result.kp_uu > 0.0

    def test_accumulation_risk_valid_values(self):
        valid = {"low", "moderate", "high"}
        for tissue in TISSUES:
            result = predict_kp("Drug", tissue=tissue, logp=3.0, fu_plasma=0.1)
            assert result.accumulation_risk in valid, (
                f"Unexpected risk for {tissue}: {result.accumulation_risk}"
            )


# ---------------------------------------------------------------------------
# Accumulation risk classification
# ---------------------------------------------------------------------------


class TestAccumulationRisk:
    def test_low_logp_gives_low_or_moderate_risk(self):
        result = predict_kp("Drug", tissue="muscle", logp=-2.0, fu_plasma=0.5)
        assert result.accumulation_risk in ("low", "moderate")

    def test_high_logp_adipose_high_risk(self):
        result = predict_kp("Drug", tissue="adipose", logp=6.0, fu_plasma=0.1)
        assert result.accumulation_risk == "high"

    def test_kp_below_3_is_low(self):
        result = _default_result(tissue="muscle", logp=-1.0)
        if result.kp < 3.0:
            assert result.accumulation_risk == "low"

    def test_kp_above_10_is_high(self):
        result = predict_kp("Drug", tissue="adipose", logp=7.0, fu_plasma=0.1)
        if result.kp >= 10.0:
            assert result.accumulation_risk == "high"


# ---------------------------------------------------------------------------
# logP effect tests
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logp_higher_kp_adipose(self):
        r_low = predict_kp("Drug", tissue="adipose", logp=1.0, fu_plasma=0.1)
        r_high = predict_kp("Drug", tissue="adipose", logp=5.0, fu_plasma=0.1)
        assert r_high.kp > r_low.kp

    def test_adipose_highest_kp_high_logp(self):
        """For a highly lipophilic drug, adipose should have the highest Kp."""
        results = predict_kp_all_tissues("Drug", logp=5.0, fu_plasma=0.1)
        top_tissue = results[0].tissue
        assert top_tissue == "adipose"

    def test_method_for_known_tissue(self):
        result = _default_result(tissue="liver")
        assert result.method == "rodgers_rowland"

    def test_method_for_unknown_tissue(self):
        result = predict_kp("Drug", tissue="placenta", logp=2.0, fu_plasma=0.1)
        assert result.method == "simplified"


# ---------------------------------------------------------------------------
# Ionizable drug tests
# ---------------------------------------------------------------------------


class TestIonizableDrug:
    def test_base_drug_returns_result(self):
        result = predict_kp(
            "BasicDrug", tissue="brain", logp=2.0, pka=8.5, fu_plasma=0.1, ionization_type="base"
        )
        assert isinstance(result, PartitionCoefficientResult)
        assert result.kp > 0.0

    def test_acid_drug_returns_result(self):
        result = predict_kp(
            "AcidDrug", tissue="muscle", logp=1.5, pka=4.5, fu_plasma=0.2, ionization_type="acid"
        )
        assert isinstance(result, PartitionCoefficientResult)
        assert result.kp > 0.0


# ---------------------------------------------------------------------------
# predict_kp_all_tissues tests
# ---------------------------------------------------------------------------


class TestPredictKpAllTissues:
    def test_returns_list(self):
        results = predict_kp_all_tissues("Drug", logp=2.0, fu_plasma=0.1)
        assert isinstance(results, list)

    def test_returns_8_tissues(self):
        results = predict_kp_all_tissues("Drug", logp=2.0, fu_plasma=0.1)
        assert len(results) == 8

    def test_all_are_partition_coefficient_result(self):
        results = predict_kp_all_tissues("Drug", logp=2.0, fu_plasma=0.1)
        for r in results:
            assert isinstance(r, PartitionCoefficientResult)

    def test_sorted_by_kp_descending(self):
        results = predict_kp_all_tissues("Drug", logp=2.0, fu_plasma=0.1)
        for i in range(len(results) - 1):
            assert results[i].kp >= results[i + 1].kp

    def test_all_8_tissues_covered(self):
        results = predict_kp_all_tissues("Drug", logp=2.0, fu_plasma=0.1)
        returned_tissues = {r.tissue for r in results}
        assert returned_tissues == set(TISSUES)

    def test_all_kp_positive(self):
        results = predict_kp_all_tissues("Drug", logp=3.0, fu_plasma=0.05)
        for r in results:
            assert r.kp > 0.0


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_fu_plasma_zero_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_kp("Drug", tissue="liver", logp=2.0, fu_plasma=0.0)

    def test_fu_plasma_negative_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_kp("Drug", tissue="liver", logp=2.0, fu_plasma=-0.1)

    def test_fu_plasma_above_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            predict_kp("Drug", tissue="liver", logp=2.0, fu_plasma=1.5)

    def test_pka_below_zero_raises(self):
        with pytest.raises(ValueError, match="pKa"):
            predict_kp("Drug", tissue="liver", logp=2.0, fu_plasma=0.1, pka=-1.0)

    def test_pka_above_14_raises(self):
        with pytest.raises(ValueError, match="pKa"):
            predict_kp("Drug", tissue="liver", logp=2.0, fu_plasma=0.1, pka=15.0)

    def test_pka_none_does_not_raise(self):
        result = predict_kp("Drug", tissue="liver", logp=2.0, fu_plasma=0.1, pka=None)
        assert result.kp > 0.0
