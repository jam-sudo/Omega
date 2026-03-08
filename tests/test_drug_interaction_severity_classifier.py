"""Tests for Phase 451 — DDI Severity Classification System."""

from omega_pbpk.clinical.drug_interaction_severity_classifier import (
    DDISeverityResult,
    classify_ddi_severity,
    screen_interactions,
)

# ---------------------------------------------------------------------------
# Return type checks
# ---------------------------------------------------------------------------


def test_return_type():
    result = classify_ddi_severity("fluconazole", "warfarin")
    assert isinstance(result, DDISeverityResult)


def test_perpetrator_drug_preserved():
    result = classify_ddi_severity("Fluconazole", "Warfarin")
    assert result.perpetrator_drug == "Fluconazole"


def test_victim_drug_preserved():
    result = classify_ddi_severity("Fluconazole", "Warfarin")
    assert result.victim_drug == "Warfarin"


# ---------------------------------------------------------------------------
# Severity class assertions
# ---------------------------------------------------------------------------


def test_fluconazole_warfarin_major():
    result = classify_ddi_severity("fluconazole", "warfarin")
    assert result.severity_class == "major"


def test_clarithromycin_simvastatin_contraindicated():
    result = classify_ddi_severity("clarithromycin", "simvastatin")
    assert result.severity_class == "contraindicated"


def test_metformin_aspirin_none():
    """metformin + aspirin is not in database — should return 'none'."""
    result = classify_ddi_severity("metformin", "aspirin")
    assert result.severity_class == "none"


def test_omeprazole_clopidogrel_moderate():
    result = classify_ddi_severity("omeprazole", "clopidogrel")
    assert result.severity_class == "moderate"


def test_metformin_alcohol_moderate():
    result = classify_ddi_severity("metformin", "alcohol")
    assert result.severity_class == "moderate"


# ---------------------------------------------------------------------------
# Monitoring and dose adjustment flags
# ---------------------------------------------------------------------------


def test_monitoring_required_for_major():
    result = classify_ddi_severity("fluconazole", "warfarin")
    assert result.monitoring_required is True


def test_monitoring_required_for_contraindicated():
    result = classify_ddi_severity("clarithromycin", "simvastatin")
    assert result.monitoring_required is True


def test_monitoring_not_required_for_none():
    result = classify_ddi_severity("metformin", "aspirin")
    assert result.monitoring_required is False


def test_dose_adjustment_required_for_contraindicated():
    result = classify_ddi_severity("clarithromycin", "simvastatin")
    assert result.dose_adjustment_required is True


def test_dose_adjustment_required_for_major():
    result = classify_ddi_severity("fluconazole", "warfarin")
    assert result.dose_adjustment_required is True


def test_dose_adjustment_not_required_for_moderate():
    result = classify_ddi_severity("omeprazole", "clopidogrel")
    assert result.dose_adjustment_required is False


# ---------------------------------------------------------------------------
# AUCR directional checks
# ---------------------------------------------------------------------------


def test_aucr_gt_1_for_cyp_inhibition():
    """CYP inhibition increases victim exposure: AUCR > 1."""
    result = classify_ddi_severity("fluconazole", "warfarin")
    assert result.predicted_aucr > 1.0


def test_aucr_lt_1_for_cyp_induction():
    """CYP induction decreases victim exposure: AUCR < 1."""
    result = classify_ddi_severity("rifampicin", "warfarin")
    assert result.predicted_aucr < 1.0


def test_aucr_equals_1_for_no_interaction():
    result = classify_ddi_severity("metformin", "aspirin")
    assert result.predicted_aucr == 1.0


# ---------------------------------------------------------------------------
# Bidirectional lookup
# ---------------------------------------------------------------------------


def test_bidirectional_lookup_same_severity():
    """Swapping perpetrator and victim should return the same interaction data."""
    result_fwd = classify_ddi_severity("fluconazole", "warfarin")
    result_rev = classify_ddi_severity("warfarin", "fluconazole")
    assert result_fwd.severity_class == result_rev.severity_class


def test_bidirectional_lookup_same_aucr():
    result_fwd = classify_ddi_severity("clarithromycin", "simvastatin")
    result_rev = classify_ddi_severity("simvastatin", "clarithromycin")
    assert result_fwd.predicted_aucr == result_rev.predicted_aucr


# ---------------------------------------------------------------------------
# FDA guidance category
# ---------------------------------------------------------------------------


def test_fda_category_avoid_for_contraindicated():
    result = classify_ddi_severity("clarithromycin", "simvastatin")
    assert result.fda_guidance_category == "avoid"


def test_fda_category_dose_reduction_for_major():
    result = classify_ddi_severity("fluconazole", "warfarin")
    assert result.fda_guidance_category == "dose_reduction"


def test_fda_category_monitor_for_moderate():
    result = classify_ddi_severity("omeprazole", "clopidogrel")
    assert result.fda_guidance_category == "monitor"


def test_fda_category_no_action_for_none():
    result = classify_ddi_severity("metformin", "aspirin")
    assert result.fda_guidance_category == "no_action"


# ---------------------------------------------------------------------------
# screen_interactions
# ---------------------------------------------------------------------------


def test_screen_interactions_count_for_three_drugs():
    """fluconazole, warfarin, simvastatin: 3 pairs.
    fluconazole-warfarin (major), fluconazole-simvastatin (none), warfarin-simvastatin (none).
    Only 1 non-none pair expected."""
    results = screen_interactions(["fluconazole", "warfarin", "simvastatin"])
    assert len(results) == 1


def test_screen_interactions_sorted_descending():
    """Results should be sorted by predicted_aucr descending."""
    results = screen_interactions(["clarithromycin", "simvastatin", "fluconazole", "warfarin"])
    # Both clarithromycin-simvastatin (10.0) and fluconazole-warfarin (2.7) should be present
    assert len(results) >= 2
    for i in range(1, len(results)):
        assert results[i - 1].predicted_aucr >= results[i].predicted_aucr


def test_screen_interactions_excludes_none():
    """screen_interactions should exclude pairs with severity == 'none'."""
    results = screen_interactions(["metformin", "aspirin", "clarithromycin", "simvastatin"])
    for r in results:
        assert r.severity_class != "none"


def test_screen_interactions_returns_list():
    results = screen_interactions(["fluconazole", "warfarin"])
    assert isinstance(results, list)
