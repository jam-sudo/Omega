"""Tests for drug self-association and aggregation prediction (Phase 1044)."""

import pytest

from omega_pbpk.prediction.drug_self_association import (
    SelfAssociationResult,
    predict_self_association,
    screen_concentrations,
)

_VALID_AGGREGATION_RISK = {"none", "low", "moderate", "high"}
_VALID_PK_IMPACT = {"negligible", "minor", "significant"}


def _default_result(**kwargs) -> SelfAssociationResult:
    params = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logp=2.0,
        concentration_mg_mL=1.0,
        pka=7.0,
        ionization_type="neutral",
        ph=7.4,
        ionic_strength_mM=150.0,
    )
    params.update(kwargs)
    return predict_self_association(**params)


class TestReturnType:
    def test_returns_self_association_result(self):
        result = _default_result()
        assert isinstance(result, SelfAssociationResult)

    def test_is_frozen_dataclass(self):
        result = _default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        result = _default_result(drug_name="Ibuprofen")
        assert result.drug_name == "Ibuprofen"


class TestCMC:
    def test_cmc_in_valid_range(self):
        result = _default_result()
        assert 0.01 <= result.cmc_mg_mL <= 200.0

    def test_high_logp_lower_cmc(self):
        """More lipophilic drug has lower CMC (self-associates more readily)."""
        result_low = _default_result(logp=1.0)
        result_high = _default_result(logp=5.0)
        assert result_high.cmc_mg_mL < result_low.cmc_mg_mL

    def test_charged_drug_higher_cmc_than_neutral(self):
        """Ionized acid at physiological pH has higher CMC than neutral drug."""
        result_neutral = _default_result(ionization_type="neutral", logp=2.0)
        result_acid = _default_result(
            ionization_type="acid", pka=4.0, ph=7.4, logp=2.0, ionic_strength_mM=0.0
        )
        # Acid at pH 7.4 is nearly fully ionized (pKa 4) → higher CMC
        assert result_acid.cmc_mg_mL > result_neutral.cmc_mg_mL


class TestFractions:
    def test_fraction_aggregated_in_range(self):
        result = _default_result()
        assert 0.0 <= result.fraction_aggregated <= 1.0

    def test_fraction_monomer_in_range(self):
        result = _default_result()
        assert 0.0 <= result.fraction_monomer <= 1.0

    def test_fractions_sum_to_one(self):
        result = _default_result()
        assert result.fraction_aggregated + result.fraction_monomer == pytest.approx(1.0)

    def test_effective_free_fraction_in_range(self):
        result = _default_result()
        assert 0.0 < result.effective_free_fraction <= 1.0

    def test_below_cmc_no_aggregation(self):
        """At very low concentration (well below CMC), all monomer."""
        result = _default_result(
            logp=1.0,  # high CMC ~= 31 mg/mL
            concentration_mg_mL=0.01,
        )
        assert result.fraction_aggregated == pytest.approx(0.0)
        assert result.effective_free_fraction == pytest.approx(1.0)

    def test_above_cmc_aggregation_present(self):
        """At concentration well above CMC, fraction_aggregated > 0."""
        result = _default_result(
            logp=5.0,  # very low CMC
            concentration_mg_mL=10.0,
        )
        assert result.fraction_aggregated > 0.0


class TestCategories:
    def test_aggregation_risk_valid(self):
        result = _default_result()
        assert result.aggregation_risk in _VALID_AGGREGATION_RISK

    def test_pk_impact_valid(self):
        result = _default_result()
        assert result.pk_impact in _VALID_PK_IMPACT

    def test_no_aggregation_below_cmc_risk_none(self):
        result = _default_result(logp=1.0, concentration_mg_mL=0.01)
        assert result.aggregation_risk == "none"
        assert result.pk_impact == "negligible"


class TestAggregateSize:
    def test_size_positive(self):
        result = _default_result()
        assert result.aggregate_size_nm > 0.0

    def test_monomer_size_small(self):
        """Below CMC, size should reflect monomer (~0.3 nm)."""
        result = _default_result(logp=1.0, concentration_mg_mL=0.01)
        assert result.aggregate_size_nm == pytest.approx(0.3)

    def test_above_cmc_size_larger(self):
        result_low = _default_result(logp=5.0, concentration_mg_mL=0.1)
        result_high = _default_result(logp=5.0, concentration_mg_mL=100.0)
        # Higher concentration above CMC → larger aggregates
        assert result_high.aggregate_size_nm >= result_low.aggregate_size_nm


class TestSolubilityReduction:
    def test_no_aggregation_reduction_fold_one(self):
        result = _default_result(logp=1.0, concentration_mg_mL=0.01)
        assert result.solubility_reduction_fold == pytest.approx(1.0)

    def test_above_cmc_reduction_greater_than_one(self):
        result = _default_result(logp=5.0, concentration_mg_mL=50.0)
        assert result.solubility_reduction_fold > 1.0


class TestScreenConcentrations:
    def test_returns_list(self):
        results = screen_concentrations("Drug", 300.0, 2.0, [0.1, 1.0, 10.0])
        assert isinstance(results, list)

    def test_correct_length(self):
        concs = [0.01, 0.1, 1.0, 10.0, 100.0]
        results = screen_concentrations("Drug", 300.0, 2.0, concs)
        assert len(results) == len(concs)

    def test_all_results_correct_type(self):
        results = screen_concentrations("Drug", 300.0, 2.0, [1.0, 2.0])
        for r in results:
            assert isinstance(r, SelfAssociationResult)

    def test_concentrations_preserved(self):
        concs = [0.5, 5.0, 50.0]
        results = screen_concentrations("Drug", 300.0, 2.0, concs)
        for r, c in zip(results, concs):
            assert r.concentration_mg_mL == pytest.approx(c)


class TestNotes:
    def test_notes_nonempty(self):
        result = _default_result()
        assert isinstance(result.notes, str) and len(result.notes) > 0


class TestValidation:
    def test_zero_concentration_raises(self):
        with pytest.raises(ValueError, match="concentration_mg_mL"):
            predict_self_association("X", mw_Da=300.0, logp=2.0, concentration_mg_mL=0.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match="concentration_mg_mL"):
            predict_self_association("X", mw_Da=300.0, logp=2.0, concentration_mg_mL=-1.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            predict_self_association("X", mw_Da=0.0, logp=2.0, concentration_mg_mL=1.0)

    def test_invalid_ionization_type_raises(self):
        with pytest.raises(ValueError, match="ionization_type"):
            predict_self_association(
                "X",
                mw_Da=300.0,
                logp=2.0,
                concentration_mg_mL=1.0,
                ionization_type="zwitterion",
            )

    def test_negative_pka_raises(self):
        with pytest.raises(ValueError, match="pka"):
            predict_self_association(
                "X",
                mw_Da=300.0,
                logp=2.0,
                concentration_mg_mL=1.0,
                pka=-1.0,
            )

    def test_pka_above_14_raises(self):
        with pytest.raises(ValueError, match="pka"):
            predict_self_association(
                "X",
                mw_Da=300.0,
                logp=2.0,
                concentration_mg_mL=1.0,
                pka=15.0,
            )

    def test_negative_ionic_strength_raises(self):
        with pytest.raises(ValueError, match="ionic_strength_mM"):
            predict_self_association(
                "X",
                mw_Da=300.0,
                logp=2.0,
                concentration_mg_mL=1.0,
                ionic_strength_mM=-10.0,
            )
