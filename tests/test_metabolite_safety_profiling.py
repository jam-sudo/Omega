"""Tests for Phase 1048: Drug Metabolite Safety Profiling."""

import pytest

from omega_pbpk.prediction.metabolite_safety_profiling import (
    MetaboliteSafetyResult,
    predict_metabolite_safety,
    screen_metabolites,
)

# ---------------------------------------------------------------------------
# Helper: default call
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="ParentDrug",
        metabolite_name="M1",
        parent_mw_Da=400.0,
        metabolite_mw_Da=380.0,
        metabolite_type="phase1",
        exposure_ratio=0.2,
        has_reactive_group=False,
        has_quinone_methide=False,
        has_arene_oxide=False,
        has_nitroso=False,
        has_acyl_glucuronide=False,
        metabolite_logp=1.0,
        organ_target="liver",
    )
    params.update(kwargs)
    return predict_metabolite_safety(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _default()
    assert isinstance(result, MetaboliteSafetyResult)


# ---------------------------------------------------------------------------
# Result is frozen (immutable)
# ---------------------------------------------------------------------------


def test_frozen():
    result = _default()
    with pytest.raises((AttributeError, TypeError)):
        result.reactive_risk = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# All scores in [0, 100]
# ---------------------------------------------------------------------------


def test_reactive_risk_range():
    result = _default()
    assert 0.0 <= result.reactive_risk <= 100.0


def test_organ_toxicity_risk_range():
    result = _default()
    assert 0.0 <= result.organ_toxicity_risk <= 100.0


def test_genotoxic_risk_range():
    result = _default()
    assert 0.0 <= result.genotoxic_risk <= 100.0


def test_overall_safety_score_range():
    result = _default()
    assert 0.0 <= result.overall_safety_score <= 100.0


# ---------------------------------------------------------------------------
# safety_class in valid set
# ---------------------------------------------------------------------------


def test_safety_class_valid():
    result = _default()
    assert result.safety_class in {"acceptable", "monitor", "concern", "high_risk"}


# ---------------------------------------------------------------------------
# fda_mist_required matches exposure_ratio >= 0.1
# ---------------------------------------------------------------------------


def test_fda_mist_required_true():
    result = _default(exposure_ratio=0.15)
    assert result.fda_mist_required is True


def test_fda_mist_required_false():
    result = _default(exposure_ratio=0.05)
    assert result.fda_mist_required is False


def test_fda_mist_boundary():
    result_at = _default(exposure_ratio=0.1)
    result_below = _default(exposure_ratio=0.09)
    assert result_at.fda_mist_required is True
    assert result_below.fda_mist_required is False


# ---------------------------------------------------------------------------
# Reactive metabolite (has_nitroso=True) → higher reactive_risk
# ---------------------------------------------------------------------------


def test_nitroso_increases_reactive_risk():
    with_nitroso = _default(metabolite_type="reactive", has_nitroso=True)
    without_nitroso = _default(metabolite_type="phase1")
    assert with_nitroso.reactive_risk > without_nitroso.reactive_risk


# ---------------------------------------------------------------------------
# Nitroso → "Nitroso compound" in structural_alerts
# ---------------------------------------------------------------------------


def test_nitroso_in_structural_alerts():
    result = _default(has_nitroso=True)
    assert "Nitroso compound" in result.structural_alerts


# ---------------------------------------------------------------------------
# No alerts → structural_alerts == ("No structural alerts",) or list equivalent
# ---------------------------------------------------------------------------


def test_no_structural_alerts():
    result = _default()
    assert "No structural alerts" in result.structural_alerts


# ---------------------------------------------------------------------------
# genotoxic_risk higher for arene_oxide
# ---------------------------------------------------------------------------


def test_arene_oxide_genotoxic():
    with_ao = _default(has_arene_oxide=True)
    without_ao = _default()
    assert with_ao.genotoxic_risk > without_ao.genotoxic_risk


# ---------------------------------------------------------------------------
# screen_metabolites returns sorted by overall_safety_score desc (safest first)
# ---------------------------------------------------------------------------


def test_screen_metabolites_sorted():
    metabolites = [
        dict(
            drug_name="Drug",
            metabolite_name="M_safe",
            parent_mw_Da=400.0,
            metabolite_mw_Da=380.0,
            metabolite_type="phase1",
            exposure_ratio=0.05,
        ),
        dict(
            drug_name="Drug",
            metabolite_name="M_risky",
            parent_mw_Da=400.0,
            metabolite_mw_Da=380.0,
            metabolite_type="reactive",
            exposure_ratio=0.5,
            has_nitroso=True,
            has_arene_oxide=True,
        ),
        dict(
            drug_name="Drug",
            metabolite_name="M_medium",
            parent_mw_Da=400.0,
            metabolite_mw_Da=380.0,
            metabolite_type="phase2",
            exposure_ratio=0.2,
            has_acyl_glucuronide=True,
        ),
    ]
    results = screen_metabolites(metabolites)
    assert len(results) == 3
    scores = [r.overall_safety_score for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# notes nonempty
# ---------------------------------------------------------------------------


def test_notes_nonempty():
    result = _default()
    assert result.notes and len(result.notes) > 0


# ---------------------------------------------------------------------------
# All organ targets accepted
# ---------------------------------------------------------------------------


def test_organ_targets():
    for organ in ["liver", "kidney", "heart", "bone_marrow"]:
        result = _default(organ_target=organ)
        assert 0.0 <= result.organ_toxicity_risk <= 100.0


# ---------------------------------------------------------------------------
# bone_marrow → higher organ_toxicity_risk than liver
# ---------------------------------------------------------------------------


def test_bone_marrow_higher_organ_risk():
    bm = _default(organ_target="bone_marrow")
    lv = _default(organ_target="liver")
    assert bm.organ_toxicity_risk > lv.organ_toxicity_risk


# ---------------------------------------------------------------------------
# Validation: negative exposure_ratio raises
# ---------------------------------------------------------------------------


def test_validation_negative_exposure():
    with pytest.raises(ValueError, match="exposure_ratio"):
        _default(exposure_ratio=-0.1)


# ---------------------------------------------------------------------------
# Validation: invalid metabolite_type raises
# ---------------------------------------------------------------------------


def test_validation_invalid_type():
    with pytest.raises(ValueError, match="metabolite_type"):
        _default(metabolite_type="phase3")


# ---------------------------------------------------------------------------
# Validation: invalid organ raises
# ---------------------------------------------------------------------------


def test_validation_invalid_organ():
    with pytest.raises(ValueError, match="organ_target"):
        _default(organ_target="lung")


# ---------------------------------------------------------------------------
# Validation: parent_mw_Da <= 0 raises
# ---------------------------------------------------------------------------


def test_validation_parent_mw_zero():
    with pytest.raises(ValueError, match="parent_mw_Da"):
        _default(parent_mw_Da=0.0)
