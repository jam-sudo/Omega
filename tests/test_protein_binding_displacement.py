"""
Tests for Phase 876: Drug Protein Binding Displacement Model
"""

import pytest
from omega_pbpk.clinical.protein_binding_displacement import (
    ProteinBindingDisplacementResult,
    predict_protein_binding_displacement,
    screen_displacers,
)


# --- Return type and basic fields ---

def test_returns_correct_type():
    result = predict_protein_binding_displacement(
        "Warfarin", "Aspirin",
        fup_victim=0.01, kd_victim_uM=0.1,
        c_displacer_uM=100.0, kd_displacer_uM=50.0,
    )
    assert isinstance(result, ProteinBindingDisplacementResult)


def test_fields_preserved():
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=10.0, kd_displacer_uM=5.0,
        hepatic_er_victim=0.1, same_binding_site=True,
    )
    assert result.drug_name_victim == "DrugA"
    assert result.drug_name_displacer == "DrugB"
    assert result.fup_victim_baseline == 0.05
    assert result.same_binding_site is True


# --- Displacement increases free fraction ---

def test_fup_displaced_ge_baseline():
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=50.0, kd_displacer_uM=10.0,
    )
    assert result.fup_victim_displaced >= result.fup_victim_baseline


def test_delta_fup_positive():
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.02, kd_victim_uM=0.5,
        c_displacer_uM=100.0, kd_displacer_uM=20.0,
    )
    assert result.delta_fup > 0.0


def test_delta_fup_pct_positive():
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.02, kd_victim_uM=0.5,
        c_displacer_uM=100.0, kd_displacer_uM=20.0,
    )
    assert result.delta_fup_pct > 0.0


# --- Same-site vs different-site ---

def test_same_site_larger_displacement_than_different_site():
    same = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=50.0, kd_displacer_uM=10.0,
        same_binding_site=True,
    )
    diff = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=50.0, kd_displacer_uM=10.0,
        same_binding_site=False,
    )
    assert same.delta_fup > diff.delta_fup


# --- Concentration effect ---

def test_high_c_displacer_larger_displacement():
    low_c = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=1.0, kd_displacer_uM=10.0,
    )
    high_c = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=100.0, kd_displacer_uM=10.0,
    )
    assert high_c.delta_fup > low_c.delta_fup


def test_zero_displacer_no_displacement():
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=0.0, kd_displacer_uM=10.0,
    )
    assert result.delta_fup == 0.0
    assert result.fup_victim_displaced == result.fup_victim_baseline


# --- Hepatic ER effects ---

def test_high_er_auc_change_near_zero():
    result = predict_protein_binding_displacement(
        "HighERDrug", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=50.0, kd_displacer_uM=10.0,
        hepatic_er_victim=0.9,
    )
    assert result.hepatic_er_victim == "high"
    assert abs(result.auc_total_change_pct) < 1.0


def test_low_er_auc_decreases():
    result = predict_protein_binding_displacement(
        "LowERDrug", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=50.0, kd_displacer_uM=10.0,
        hepatic_er_victim=0.1,
    )
    assert result.hepatic_er_victim == "low"
    assert result.auc_total_change_pct < 0.0


def test_intermediate_er_classification():
    result = predict_protein_binding_displacement(
        "MidERDrug", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=50.0, kd_displacer_uM=10.0,
        hepatic_er_victim=0.5,
    )
    assert result.hepatic_er_victim == "intermediate"


# --- cmax_free_change ---

def test_cmax_free_increases_with_displacement():
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=50.0, kd_displacer_uM=10.0,
    )
    assert result.cmax_free_change_pct > 0.0


# --- Clinical significance thresholds ---

def test_clinical_significance_none():
    # With very low displacer concentration → small delta_fup_pct
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.5, kd_victim_uM=1.0,
        c_displacer_uM=0.01, kd_displacer_uM=100.0,
    )
    assert result.clinical_significance == "none"


def test_clinical_significance_significant():
    # Highly bound drug + strong displacer
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.01, kd_victim_uM=0.1,
        c_displacer_uM=1000.0, kd_displacer_uM=5.0,
        same_binding_site=True,
    )
    assert result.clinical_significance == "significant"


def test_notes_is_string():
    result = predict_protein_binding_displacement(
        "DrugA", "DrugB",
        fup_victim=0.05, kd_victim_uM=1.0,
        c_displacer_uM=50.0, kd_displacer_uM=10.0,
    )
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# --- screen_displacers ---

def test_screen_displacers_returns_list():
    displacers = [
        {"name": "DrugB", "c_uM": 10.0, "kd_uM": 5.0, "same_site": True},
        {"name": "DrugC", "c_uM": 50.0, "kd_uM": 20.0, "same_site": True},
        {"name": "DrugD", "c_uM": 1.0, "kd_uM": 100.0, "same_site": False},
    ]
    results = screen_displacers("DrugA", 0.05, 1.0, displacers)
    assert len(results) == 3


def test_screen_displacers_sorted_by_delta_fup_descending():
    displacers = [
        {"name": "Weak", "c_uM": 1.0, "kd_uM": 1000.0, "same_site": True},
        {"name": "Strong", "c_uM": 500.0, "kd_uM": 2.0, "same_site": True},
        {"name": "Medium", "c_uM": 20.0, "kd_uM": 10.0, "same_site": True},
    ]
    results = screen_displacers("DrugA", 0.05, 1.0, displacers)
    delta_fups = [r.delta_fup for r in results]
    assert delta_fups == sorted(delta_fups, reverse=True)


# --- Validation ---

def test_validation_fup_zero():
    with pytest.raises(ValueError, match="fup_victim must be in \\(0, 1\\]"):
        predict_protein_binding_displacement(
            "DrugA", "DrugB",
            fup_victim=0.0, kd_victim_uM=1.0,
            c_displacer_uM=10.0, kd_displacer_uM=5.0,
        )


def test_validation_fup_above_one():
    with pytest.raises(ValueError, match="fup_victim must be in \\(0, 1\\]"):
        predict_protein_binding_displacement(
            "DrugA", "DrugB",
            fup_victim=1.5, kd_victim_uM=1.0,
            c_displacer_uM=10.0, kd_displacer_uM=5.0,
        )


def test_validation_negative_kd_victim():
    with pytest.raises(ValueError, match="kd_victim_uM must be > 0"):
        predict_protein_binding_displacement(
            "DrugA", "DrugB",
            fup_victim=0.05, kd_victim_uM=-1.0,
            c_displacer_uM=10.0, kd_displacer_uM=5.0,
        )


def test_validation_negative_c_displacer():
    with pytest.raises(ValueError, match="c_displacer_uM must be >= 0"):
        predict_protein_binding_displacement(
            "DrugA", "DrugB",
            fup_victim=0.05, kd_victim_uM=1.0,
            c_displacer_uM=-5.0, kd_displacer_uM=5.0,
        )
