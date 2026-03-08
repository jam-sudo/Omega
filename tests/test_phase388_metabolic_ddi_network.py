"""
Tests for Phase 388 — Metabolic Drug Interaction Network
(clinical/metabolic_ddi_network.py)
"""

import pytest

from omega_pbpk.clinical.metabolic_ddi_network import (
    MetabolicDDINetworkResult,
    assess_polypharmacy_ddi,
    predict_metabolic_ddi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inhibition_result(**kwargs) -> MetabolicDDINetworkResult:
    """Return a result for a basic inhibition scenario."""
    params = dict(
        perpetrator_drug="Inhibitor",
        victim_drug="Victim",
        cyp_enzyme="CYP3A4",
        fm_cyp=0.8,
        perpetrator_conc_uM=1.0,
        ki_uM=0.5,
    )
    params.update(kwargs)
    return predict_metabolic_ddi(**params)


def _induction_result(**kwargs) -> MetabolicDDINetworkResult:
    """Return a result for a basic induction scenario."""
    params = dict(
        perpetrator_drug="Inducer",
        victim_drug="Victim",
        cyp_enzyme="CYP3A4",
        fm_cyp=0.8,
        perpetrator_conc_uM=10.0,
        induction_ec50_uM=5.0,
        max_fold_induction=5.0,
    )
    params.update(kwargs)
    return predict_metabolic_ddi(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_metabolic_ddi_network_result(self):
        result = _inhibition_result()
        assert isinstance(result, MetabolicDDINetworkResult)

    def test_result_has_expected_fields(self):
        result = _inhibition_result()
        for field in [
            "perpetrator_drug",
            "victim_drug",
            "cyp_enzyme",
            "interaction_type",
            "ki_uM",
            "ec50_fold_induction",
            "max_induction_fold",
            "perpetrator_conc_uM",
            "aucr",
            "cmax_ratio",
            "risk_category",
            "fda_category",
            "clinical_significance",
            "notes",
        ]:
            assert hasattr(result, field)

    def test_result_is_frozen(self):
        result = _inhibition_result()
        with pytest.raises((AttributeError, TypeError)):
            result.aucr = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Inhibition scenarios
# ---------------------------------------------------------------------------


class TestInhibition:
    def test_inhibition_increases_aucr(self):
        result = _inhibition_result()
        assert result.aucr > 1.0

    def test_strong_inhibitor_classified_strong(self):
        # High [I]/Ki → strong inhibition
        result = _inhibition_result(perpetrator_conc_uM=100.0, ki_uM=0.1, fm_cyp=0.95)
        assert result.risk_category == "strong"

    def test_weak_inhibitor_classified_weak(self):
        # R1 = 1 + 1.0/2.0 = 1.5; AUCR = 1/(0.3 + 0.7/1.5) ≈ 1.30 → "weak"
        result = _inhibition_result(perpetrator_conc_uM=1.0, ki_uM=2.0, fm_cyp=0.7)
        assert result.risk_category == "weak"

    def test_interaction_type_inhibition(self):
        result = _inhibition_result()
        assert result.interaction_type == "inhibition"

    def test_inhibition_aucr_formula(self):
        """Verify AUCR against manual formula: 1/(1 - fm + fm/R1)."""
        fm = 0.8
        conc = 1.0
        ki = 0.5
        r1 = 1.0 + conc / ki
        expected_aucr = 1.0 / (1.0 - fm + fm / r1)
        result = _inhibition_result(fm_cyp=fm, perpetrator_conc_uM=conc, ki_uM=ki)
        assert abs(result.aucr - expected_aucr) < 1e-6

    def test_higher_fm_higher_aucr(self):
        """Sensitive substrates (high fm) show larger DDI."""
        r_low = _inhibition_result(fm_cyp=0.2)
        r_high = _inhibition_result(fm_cyp=0.95)
        assert r_high.aucr > r_low.aucr


# ---------------------------------------------------------------------------
# Induction scenarios
# ---------------------------------------------------------------------------


class TestInduction:
    def test_induction_decreases_aucr(self):
        result = _induction_result()
        assert result.aucr < 1.0

    def test_interaction_type_induction(self):
        result = _induction_result()
        assert result.interaction_type == "induction"

    def test_strong_induction_classified(self):
        # Very strong induction → large decrease in victim AUC
        result = _induction_result(
            fm_cyp=0.95,
            perpetrator_conc_uM=100.0,
            induction_ec50_uM=1.0,
            max_fold_induction=20.0,
        )
        assert result.risk_category in ("moderate", "strong")


# ---------------------------------------------------------------------------
# Mixed interaction
# ---------------------------------------------------------------------------


class TestMixedInteraction:
    def test_mixed_interaction_type(self):
        result = predict_metabolic_ddi(
            perpetrator_drug="MixedPerpetrator",
            victim_drug="Victim",
            cyp_enzyme="CYP2D6",
            fm_cyp=0.7,
            perpetrator_conc_uM=2.0,
            ki_uM=1.0,
            induction_ec50_uM=3.0,
            max_fold_induction=3.0,
        )
        assert result.interaction_type == "mixed"

    def test_mixed_uses_inhibition_aucr(self):
        """When both provided, AUCR should reflect inhibition (increases AUC)."""
        result = predict_metabolic_ddi(
            perpetrator_drug="MixedPerpetrator",
            victim_drug="Victim",
            cyp_enzyme="CYP2D6",
            fm_cyp=0.7,
            perpetrator_conc_uM=2.0,
            ki_uM=1.0,
            induction_ec50_uM=3.0,
            max_fold_induction=3.0,
        )
        assert result.aucr > 1.0  # inhibition dominates


# ---------------------------------------------------------------------------
# No interaction
# ---------------------------------------------------------------------------


class TestNoInteraction:
    def test_no_interaction_aucr_one(self):
        result = predict_metabolic_ddi(
            perpetrator_drug="NoEffect",
            victim_drug="Victim",
            cyp_enzyme="CYP3A4",
            fm_cyp=0.5,
            perpetrator_conc_uM=1.0,
        )
        assert abs(result.aucr - 1.0) < 1e-9

    def test_no_interaction_risk_none(self):
        result = predict_metabolic_ddi(
            perpetrator_drug="NoEffect",
            victim_drug="Victim",
            cyp_enzyme="CYP3A4",
            fm_cyp=0.5,
            perpetrator_conc_uM=1.0,
        )
        assert result.risk_category == "none"

    def test_no_interaction_clinical_no_action(self):
        result = predict_metabolic_ddi(
            perpetrator_drug="NoEffect",
            victim_drug="Victim",
            cyp_enzyme="CYP3A4",
            fm_cyp=0.5,
            perpetrator_conc_uM=1.0,
        )
        assert result.clinical_significance == "no_action"


# ---------------------------------------------------------------------------
# FDA category by fm_cyp
# ---------------------------------------------------------------------------


class TestFDACategory:
    def test_sensitive_substrate(self):
        result = _inhibition_result(fm_cyp=0.95)
        assert result.fda_category == "victim_sensitive_substrate"

    def test_moderate_substrate(self):
        result = _inhibition_result(fm_cyp=0.7)
        assert result.fda_category == "victim_moderate_substrate"

    def test_minor_substrate(self):
        result = _inhibition_result(fm_cyp=0.3)
        assert result.fda_category == "victim_minor_substrate"


# ---------------------------------------------------------------------------
# Clinical significance mapping
# ---------------------------------------------------------------------------


class TestClinicalSignificance:
    def test_none_risk_no_action(self):
        result = predict_metabolic_ddi(
            perpetrator_drug="Weak",
            victim_drug="V",
            cyp_enzyme="CYP3A4",
            fm_cyp=0.5,
            perpetrator_conc_uM=1.0,
        )
        assert result.clinical_significance == "no_action"

    def test_strong_risk_contraindicated(self):
        result = _inhibition_result(perpetrator_conc_uM=100.0, ki_uM=0.1, fm_cyp=0.95)
        assert result.clinical_significance == "contraindicated"

    def test_cmax_ratio_equals_aucr(self):
        result = _inhibition_result()
        assert abs(result.cmax_ratio - result.aucr) < 1e-9


# ---------------------------------------------------------------------------
# Polypharmacy
# ---------------------------------------------------------------------------


class TestPolypharmacy:
    def _make_regimen(self):
        return [
            {"perpetrator": "DrugA", "victim": "V1", "cyp": "CYP3A4", "fm_cyp": 0.9, "ki_uM": 0.2},
            {"perpetrator": "DrugA", "victim": "V2", "cyp": "CYP2D6", "fm_cyp": 0.5, "ki_uM": 5.0},
            {"perpetrator": "DrugA", "victim": "V3", "cyp": "CYP1A2", "fm_cyp": 0.3},
        ]

    def test_returns_correct_count(self):
        results = assess_polypharmacy_ddi(self._make_regimen(), 1.0)
        assert len(results) == 3

    def test_returns_list_of_results(self):
        results = assess_polypharmacy_ddi(self._make_regimen(), 1.0)
        for r in results:
            assert isinstance(r, MetabolicDDINetworkResult)

    def test_sorted_by_aucr_descending(self):
        results = assess_polypharmacy_ddi(self._make_regimen(), 1.0)
        aucs = [r.aucr for r in results]
        assert aucs == sorted(aucs, reverse=True)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_fm_cyp_above_one(self):
        with pytest.raises(ValueError, match="fm_cyp"):
            predict_metabolic_ddi(
                perpetrator_drug="X",
                victim_drug="Y",
                cyp_enzyme="CYP3A4",
                fm_cyp=1.5,
                perpetrator_conc_uM=1.0,
            )

    def test_invalid_fm_cyp_negative(self):
        with pytest.raises(ValueError, match="fm_cyp"):
            predict_metabolic_ddi(
                perpetrator_drug="X",
                victim_drug="Y",
                cyp_enzyme="CYP3A4",
                fm_cyp=-0.1,
                perpetrator_conc_uM=1.0,
            )

    def test_invalid_ki_zero(self):
        with pytest.raises(ValueError, match="ki_uM"):
            predict_metabolic_ddi(
                perpetrator_drug="X",
                victim_drug="Y",
                cyp_enzyme="CYP3A4",
                fm_cyp=0.5,
                perpetrator_conc_uM=1.0,
                ki_uM=0.0,
            )

    def test_invalid_perpetrator_conc_negative(self):
        with pytest.raises(ValueError, match="perpetrator_conc_uM"):
            predict_metabolic_ddi(
                perpetrator_drug="X",
                victim_drug="Y",
                cyp_enzyme="CYP3A4",
                fm_cyp=0.5,
                perpetrator_conc_uM=-1.0,
            )

    def test_invalid_ec50_zero(self):
        with pytest.raises(ValueError, match="induction_ec50_uM"):
            predict_metabolic_ddi(
                perpetrator_drug="X",
                victim_drug="Y",
                cyp_enzyme="CYP3A4",
                fm_cyp=0.5,
                perpetrator_conc_uM=1.0,
                induction_ec50_uM=0.0,
                max_fold_induction=3.0,
            )

    def test_invalid_max_fold_less_than_one(self):
        with pytest.raises(ValueError, match="max_fold_induction"):
            predict_metabolic_ddi(
                perpetrator_drug="X",
                victim_drug="Y",
                cyp_enzyme="CYP3A4",
                fm_cyp=0.5,
                perpetrator_conc_uM=1.0,
                induction_ec50_uM=5.0,
                max_fold_induction=0.5,
            )
