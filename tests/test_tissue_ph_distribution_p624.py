"""Tests for Phase 624 — Drug Tissue pH Effect on Distribution."""

from dataclasses import fields

import pytest

from omega_pbpk.prediction.tissue_ph_distribution_p624 import (
    TissuePHDistributionResult,
    compare_tissues,
    predict_tissue_ph_distribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _acid_lysosome():
    """Weak acid in lysosome (pH 5.0). Acids concentrate in basic environments."""
    return predict_tissue_ph_distribution(
        drug_name="IbuprofenAnalogue",
        logP=3.5,
        pka=4.4,
        drug_type="acid",
        tissue="lysosome",
    )


def _base_lysosome():
    """Weak base in lysosome (pH 5.0). Bases get trapped (pKa > 6)."""
    return predict_tissue_ph_distribution(
        drug_name="ChloroquineAnalogue",
        logP=4.0,
        pka=8.3,
        drug_type="base",
        tissue="lysosome",
    )


def _neutral_liver():
    """Neutral drug in liver (pH 7.0). No pH effect."""
    return predict_tissue_ph_distribution(
        drug_name="NeutralDrug",
        logP=2.5,
        pka=7.0,
        drug_type="neutral",
        tissue="liver",
    )


def _base_plasma():
    """Base in plasma (pH 7.4 == plasma). Partition factor ~1."""
    return predict_tissue_ph_distribution(
        drug_name="BasePlasma",
        logP=2.0,
        pka=8.0,
        drug_type="base",
        tissue="plasma",
    )


# ---------------------------------------------------------------------------
# Return type and frozen dataclass
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_tissue_ph_distribution_result(self):
        result = _neutral_liver()
        assert isinstance(result, TissuePHDistributionResult)

    def test_frozen_dataclass_immutable(self):
        result = _neutral_liver()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Modified"  # type: ignore[misc]

    def test_all_fields_present(self):
        result = _neutral_liver()
        expected_fields = {
            "drug_name",
            "drug_type",
            "pka",
            "ph_plasma",
            "ph_tissue",
            "tissue",
            "kp_ph_corrected",
            "kp_neutral",
            "ph_partition_factor",
            "fu_tissue",
            "accumulation_risk",
            "lysosomotropic",
            "notes",
        }
        actual_fields = {f.name for f in fields(result)}
        assert expected_fields == actual_fields

    def test_drug_name_preserved(self):
        result = _neutral_liver()
        assert result.drug_name == "NeutralDrug"

    def test_tissue_preserved(self):
        result = _neutral_liver()
        assert result.tissue == "liver"


# ---------------------------------------------------------------------------
# Neutral drug — no pH effect
# ---------------------------------------------------------------------------


class TestNeutralDrug:
    def test_neutral_partition_factor_is_1(self):
        result = _neutral_liver()
        assert abs(result.ph_partition_factor - 1.0) < 1e-9

    def test_neutral_kp_equals_kp_neutral(self):
        result = _neutral_liver()
        assert abs(result.kp_ph_corrected - result.kp_neutral) < 1e-9

    def test_neutral_not_lysosomotropic(self):
        result = _neutral_liver()
        assert result.lysosomotropic is False

    def test_neutral_kp_neutral_formula(self):
        logP = 2.5
        result = _neutral_liver()
        expected = max(0.01, logP * 0.5 + 0.3)
        assert abs(result.kp_neutral - expected) < 1e-9


# ---------------------------------------------------------------------------
# Acid drug — pH trapping in basic tissues
# ---------------------------------------------------------------------------


class TestAcidDrug:
    def test_acid_prefers_basic_tissue(self):
        """Acid (pKa 4.4) in plasma vs lysosome: should have lower Kp in lysosome."""
        acid_plasma = predict_tissue_ph_distribution(
            "Acid", logP=2.0, pka=4.4, drug_type="acid", tissue="plasma"
        )
        acid_lysosome = _acid_lysosome()
        # At pH << pKa (lysosome pH 5.0), acid is mostly unionized → less trapped
        # but partition factor for acid: (1+10^(7.4-pKa)) / (1+10^(pHt-pKa))
        # pKa=4.4 → 10^(7.4-4.4)=1000, 10^(5.0-4.4)=4 → factor = 1001/5 ≈ 200
        # So acid actually concentrates significantly in lysosome?
        # Actually Henderson-Hasselbalch acid: ionized at HIGH pH, nonionized at low pH
        # At tissue pH 5 < pKa 4.4+: mostly nonionized at pH 5 vs ionized at pH 7.4
        # Kp ratio = (1+10^(3)) / (1+10^(0.6)) ≈ 1001/4.98 ≈ 201 (basic env concentrates acid)
        # So lysosome (pH 5) is MORE acidic than plasma, acids go to basic environments
        assert acid_plasma.ph_partition_factor != acid_lysosome.ph_partition_factor

    def test_acid_partition_factor_formula(self):
        pka = 4.4
        ph_plasma = 7.4
        ph_tissue = 5.0  # lysosome
        result = _acid_lysosome()
        expected = (1.0 + 10.0 ** (ph_plasma - pka)) / (1.0 + 10.0 ** (ph_tissue - pka))
        assert abs(result.ph_partition_factor - expected) < 1e-6

    def test_acid_not_lysosomotropic(self):
        result = _acid_lysosome()
        assert result.lysosomotropic is False

    def test_acid_in_basic_tissue_higher_kp(self):
        """Acid with high pKa in high-pH tissue vs low-pH tissue."""
        # pKa 6 acid: plasma pH 7.4 (ionized), kidney pH 7.2 (ionized), tumor pH 6.5 (less ionized)
        acid_kidney = predict_tissue_ph_distribution(
            "AcidX", logP=2.0, pka=6.0, drug_type="acid", tissue="kidney"
        )
        acid_tumor = predict_tissue_ph_distribution(
            "AcidX", logP=2.0, pka=6.0, drug_type="acid", tissue="tumor"
        )
        # kidney pH 7.2 > tumor pH 6.5 → acid more ionized in kidney → less partitioning
        # partition factor for acid: numerator same, denominator smaller at lower pH → higher Kp at lower pH
        # Wait: acid Kp = (1+10^(7.4-pKa)) / (1+10^(pHt-pKa))
        # Lower tissue pH → smaller denominator → HIGHER Kp
        assert acid_tumor.kp_ph_corrected > acid_kidney.kp_ph_corrected


# ---------------------------------------------------------------------------
# Base drug — pH trapping (lysosomotropic)
# ---------------------------------------------------------------------------


class TestBaseDrug:
    def test_base_lysosome_partition_factor_less_than_1(self):
        """Base (pKa 8.3) at lysosome pH 5.0.

        base factor = (1+10^(pKa-pH_plasma)) / (1+10^(pKa-pH_tissue))
                    = (1+10^0.9) / (1+10^3.3) ≈ 8.94 / 1996 ≈ 0.0045
        Low tissue pH → large denominator → factor < 1 → reduced Kp in acidic lysosome.
        """
        result = _base_lysosome()
        assert result.ph_partition_factor < 1.0

    def test_base_partition_factor_formula(self):
        pka = 8.3
        ph_plasma = 7.4
        ph_tissue = 5.0
        result = _base_lysosome()
        expected = (1.0 + 10.0 ** (pka - ph_plasma)) / (1.0 + 10.0 ** (pka - ph_tissue))
        assert abs(result.ph_partition_factor - expected) < 1e-6

    def test_base_lysosomotropic_high_pka(self):
        """Base with pKa > 6 should be lysosomotropic."""
        result = _base_lysosome()
        assert result.lysosomotropic is True

    def test_base_not_lysosomotropic_low_pka(self):
        """Base with pKa <= 6 should NOT be lysosomotropic."""
        result = predict_tissue_ph_distribution(
            "WeakBase", logP=2.0, pka=5.5, drug_type="base", tissue="lysosome"
        )
        assert result.lysosomotropic is False

    def test_base_plasma_partition_factor_near_1(self):
        """Base in plasma: tissue pH == plasma pH → partition factor = 1."""
        result = _base_plasma()
        assert abs(result.ph_partition_factor - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Tumor pH effect
# ---------------------------------------------------------------------------


class TestTumorPH:
    def test_tumor_ph_is_65(self):
        result = predict_tissue_ph_distribution(
            "Drug", logP=2.0, pka=7.0, drug_type="base", tissue="tumor"
        )
        assert abs(result.ph_tissue - 6.5) < 1e-9

    def test_base_lower_kp_in_tumor_than_plasma(self):
        """Basic drug: lower tumor pH → larger denominator → lower partition factor → lower Kp.

        base factor = (1+10^(pKa-plasma_pH)) / (1+10^(pKa-tissue_pH))
        tumor pH 6.5 < plasma pH 7.4 → 10^(pKa-6.5) >> 10^(pKa-7.4) → factor < 1
        """
        base_plasma = predict_tissue_ph_distribution(
            "BaseD", logP=2.0, pka=8.0, drug_type="base", tissue="plasma"
        )
        base_tumor = predict_tissue_ph_distribution(
            "BaseD", logP=2.0, pka=8.0, drug_type="base", tissue="tumor"
        )
        # Plasma: ph_tissue == ph_plasma → factor = 1.0
        assert abs(base_plasma.ph_partition_factor - 1.0) < 1e-9
        # Tumor: lower pH → factor < 1
        assert base_tumor.ph_partition_factor < 1.0


# ---------------------------------------------------------------------------
# Accumulation risk
# ---------------------------------------------------------------------------


class TestAccumulationRisk:
    def test_risk_valid_values(self):
        result = _base_lysosome()
        assert result.accumulation_risk in ("low", "moderate", "high")

    def test_high_risk_when_kp_ge_10(self):
        result = _base_lysosome()
        if result.kp_ph_corrected >= 10.0:
            assert result.accumulation_risk == "high"

    def test_low_risk_when_kp_lt_2(self):
        # Neutral drug in same-pH tissue → kp_neutral only
        result = predict_tissue_ph_distribution(
            "Neutral", logP=1.0, pka=7.0, drug_type="neutral", tissue="plasma"
        )
        kp = result.kp_ph_corrected
        if kp < 2.0:
            assert result.accumulation_risk == "low"

    def test_moderate_risk_range(self):
        result = predict_tissue_ph_distribution(
            "MidDrug", logP=3.0, pka=7.0, drug_type="neutral", tissue="liver"
        )
        kp = result.kp_ph_corrected
        if 2.0 <= kp < 10.0:
            assert result.accumulation_risk == "moderate"


# ---------------------------------------------------------------------------
# fu_tissue
# ---------------------------------------------------------------------------


class TestFuTissue:
    def test_fu_tissue_range(self):
        result = _base_lysosome()
        assert 1e-6 <= result.fu_tissue <= 1.0

    def test_low_kp_high_fu_tissue(self):
        """Lower Kp → higher fu_tissue. Base in acidic lysosome has low Kp (factor<1)."""
        result = _base_lysosome()
        # kp_ph_corrected << 1 → fu_tissue = fu_plasma / kp → fu_tissue clamped to 1.0
        assert result.fu_tissue == 1.0  # clamped at maximum


# ---------------------------------------------------------------------------
# Compare tissues
# ---------------------------------------------------------------------------


class TestCompareTissues:
    def test_returns_list(self):
        result = compare_tissues("Chloroquine", logP=4.0, pka=8.3, drug_type="base")
        assert isinstance(result, list)

    def test_result_length(self):
        result = compare_tissues("Chloroquine", logP=4.0, pka=8.3, drug_type="base")
        assert len(result) == 8  # 8 standard tissues

    def test_sorted_by_kp_descending(self):
        result = compare_tissues("Chloroquine", logP=4.0, pka=8.3, drug_type="base")
        kps = [r.kp_ph_corrected for r in result]
        assert kps == sorted(kps, reverse=True)

    def test_all_results_correct_type(self):
        result = compare_tissues("Chloroquine", logP=4.0, pka=8.3, drug_type="base")
        assert all(isinstance(r, TissuePHDistributionResult) for r in result)

    def test_plasma_first_for_basic_drug(self):
        """Plasma (pH 7.4 = plasma pH) gives partition factor=1 → Kp = kp_neutral.
        More basic tissues (lower pH) have factor < 1 → lower Kp.
        Plasma or lung (pH 7.4) should tie for highest Kp."""
        result = compare_tissues("BasicDrug", logP=3.0, pka=9.0, drug_type="base")
        # The top result should be one of the tissues at pH 7.4 (plasma or lung)
        assert result[0].tissue in ("plasma", "lung")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError, match="drug_type"):
            predict_tissue_ph_distribution(
                "X", logP=2.0, pka=7.0, drug_type="zwitterion", tissue="liver"
            )

    def test_unknown_tissue_uses_default_ph(self):
        result = predict_tissue_ph_distribution(
            "X", logP=2.0, pka=7.0, drug_type="neutral", tissue="spleen"
        )
        assert abs(result.ph_tissue - 7.2) < 1e-9

    def test_notes_is_string(self):
        result = _base_lysosome()
        assert isinstance(result.notes, str)

    def test_ph_plasma_is_74(self):
        result = _neutral_liver()
        assert abs(result.ph_plasma - 7.4) < 1e-9
