"""Tests for Phase 680 — DDI Severity Classifier."""

import pytest

from omega_pbpk.clinical.ddi_severity_classifier_p680 import (
    DDISeverityResult,
    batch_classify_ddi,
    classify_ddi_severity,
)


def _classify(**overrides):
    defaults = dict(
        drug_a="DrugA",
        drug_b="DrugB",
        interaction_type="CYP3A4_inhibition",
        aucr_estimate=1.0,
        cmaxr_estimate=1.0,
    )
    defaults.update(overrides)
    return classify_ddi_severity(**defaults)


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _classify()
        assert isinstance(r, DDISeverityResult)

    def test_frozen_dataclass(self):
        r = _classify()
        with pytest.raises(AttributeError):
            r.severity_score = 99.0

    def test_drug_names_preserved(self):
        r = _classify(drug_a="Ketoconazole", drug_b="Midazolam")
        assert r.drug_a == "Ketoconazole"
        assert r.drug_b == "Midazolam"


class TestQTProlongation:
    def test_qt_high_aucr_contraindicated(self):
        r = _classify(
            interaction_type="QT_prolongation",
            aucr_estimate=3.0,
        )
        assert r.contraindicated is True

    def test_qt_moderate_aucr_contraindicated(self):
        # QT with score >= 50 is contraindicated; aucr=2.0 → comp=20+30=50
        r = _classify(
            interaction_type="QT_prolongation",
            aucr_estimate=2.0,
        )
        assert r.severity_score >= 50
        assert r.contraindicated is True

    def test_qt_recommendation_avoid(self):
        r = _classify(
            interaction_type="QT_prolongation",
            aucr_estimate=2.0,
        )
        assert r.clinical_recommendation == "avoid combination"


class TestMinorInteraction:
    def test_minor_has_low_score(self):
        r = _classify(
            interaction_type="protein_binding_displacement",
            aucr_estimate=1.0,
            cmaxr_estimate=1.0,
        )
        assert r.severity_score < 20

    def test_minor_class(self):
        r = _classify(
            interaction_type="protein_binding_displacement",
            aucr_estimate=1.0,
            cmaxr_estimate=1.0,
        )
        assert r.severity_class == "minor"

    def test_minor_recommendation(self):
        r = _classify(
            interaction_type="protein_binding_displacement",
            aucr_estimate=1.0,
            cmaxr_estimate=1.0,
        )
        assert r.clinical_recommendation == "standard monitoring"


class TestSeverityClasses:
    def test_contraindicated_class(self):
        r = _classify(
            interaction_type="QT_prolongation",
            aucr_estimate=3.0,
            cmaxr_estimate=3.0,
        )
        assert r.severity_class == "contraindicated"

    def test_major_class(self):
        # additive_toxicity weight=12*2=24, aucr=2.0 → comp=20, total=44
        r = _classify(
            interaction_type="additive_toxicity",
            aucr_estimate=2.0,
        )
        assert r.severity_class == "major"

    def test_moderate_class(self):
        # CYP3A4 weight=8*2=16, aucr=1.3 → comp=6, total=22
        r = _classify(
            interaction_type="CYP3A4_inhibition",
            aucr_estimate=1.3,
        )
        assert r.severity_class == "moderate"

    def test_all_classes_achievable(self):
        classes = set()
        # minor
        classes.add(
            _classify(
                interaction_type="protein_binding_displacement",
            ).severity_class
        )
        # moderate
        classes.add(
            _classify(
                interaction_type="CYP3A4_inhibition",
                aucr_estimate=1.3,
            ).severity_class
        )
        # major
        classes.add(
            _classify(
                interaction_type="additive_toxicity",
                aucr_estimate=2.0,
            ).severity_class
        )
        # contraindicated
        classes.add(
            _classify(
                interaction_type="QT_prolongation",
                aucr_estimate=3.0,
                cmaxr_estimate=3.0,
            ).severity_class
        )
        assert classes == {"minor", "moderate", "major", "contraindicated"}


class TestContraindicated:
    def test_contraindicated_true_for_contraindicated_class(self):
        r = _classify(
            interaction_type="QT_prolongation",
            aucr_estimate=3.0,
            cmaxr_estimate=3.0,
        )
        assert r.contraindicated is True

    def test_not_contraindicated_for_minor(self):
        r = _classify(
            interaction_type="protein_binding_displacement",
        )
        assert r.contraindicated is False


class TestBatchClassify:
    def test_batch_returns_list(self):
        interactions = [
            {"drug_a": "A", "drug_b": "B", "interaction_type": "CYP3A4_inhibition"},
            {
                "drug_a": "C",
                "drug_b": "D",
                "interaction_type": "QT_prolongation",
                "aucr_estimate": 2.0,
            },
        ]
        results = batch_classify_ddi(interactions)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_sorted_descending(self):
        interactions = [
            {"drug_a": "A", "drug_b": "B", "interaction_type": "protein_binding_displacement"},
            {
                "drug_a": "C",
                "drug_b": "D",
                "interaction_type": "QT_prolongation",
                "aucr_estimate": 3.0,
            },
            {"drug_a": "E", "drug_b": "F", "interaction_type": "CYP3A4_inhibition"},
        ]
        results = batch_classify_ddi(interactions)
        for i in range(1, len(results)):
            assert results[i - 1].severity_score >= results[i].severity_score


class TestOnsetDuration:
    def test_onset_estimated_when_none(self):
        r = _classify(interaction_type="QT_prolongation")
        assert r.onset_h == 0.5

    def test_onset_cyp3a4_default(self):
        r = _classify(interaction_type="CYP3A4_inhibition")
        assert r.onset_h == 1.0

    def test_onset_displacement_default(self):
        r = _classify(interaction_type="protein_binding_displacement")
        assert r.onset_h == 0.1

    def test_duration_estimated_cyp(self):
        r = _classify(interaction_type="CYP3A4_inhibition")
        assert r.duration_h == 12.0

    def test_duration_estimated_rapid(self):
        r = _classify(interaction_type="QT_prolongation")
        assert r.duration_h == 2.0

    def test_custom_onset_preserved(self):
        r = _classify(onset_h=5.0)
        assert r.onset_h == 5.0

    def test_custom_duration_preserved(self):
        r = _classify(duration_h=24.0)
        assert r.duration_h == 24.0


class TestValidation:
    def test_negative_aucr_raises(self):
        with pytest.raises(ValueError):
            _classify(aucr_estimate=-1.0)

    def test_zero_aucr_raises(self):
        with pytest.raises(ValueError):
            _classify(aucr_estimate=0.0)

    def test_negative_cmaxr_raises(self):
        with pytest.raises(ValueError):
            _classify(cmaxr_estimate=-1.0)

    def test_invalid_interaction_type_raises(self):
        with pytest.raises(ValueError):
            _classify(interaction_type="invalid_type")
