"""Tests for Phase 1110 — Formulation Development Index."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.formulation_development_index import (
    FormulationDevelopmentResult,
    calculate_fdi,
    rank_candidates,
)


def test_return_type():
    r = calculate_fdi("DrugA", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert isinstance(r, FormulationDevelopmentResult)


def test_drug_name_preserved():
    r = calculate_fdi("Aspirin", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert r.drug_name == "Aspirin"


def test_fdi_score_in_range():
    r = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert 0.0 <= r.fdi_score <= 100.0


def test_subscores_sum_to_fdi():
    r = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    total = (r.solubility_score + r.permeability_score + r.chemical_stability_score
             + r.physical_stability_score + r.dose_number_score)
    assert abs(total - r.fdi_score) < 1e-9


def test_high_solubility_ideal_logp_excellent():
    r = calculate_fdi("Drug", solubility_mg_mL=500.0, logP=2.0, dose_mg=50.0,
                       melting_point_C=220.0, hygroscopic=False)
    assert r.fdi_class == "excellent"


def test_very_low_solubility_low_score():
    r = calculate_fdi("Drug", solubility_mg_mL=0.001, logP=6.0, dose_mg=500.0,
                       melting_point_C=50.0, hygroscopic=True,
                       has_ester=True, has_aldehyde=True)
    assert r.fdi_score < 40.0


def test_michael_acceptor_parenteral():
    r = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0,
                       has_michael_acceptor=True)
    assert r.recommended_formulation == "parenteral"


def test_michael_acceptor_lowest_chem_score():
    r_ma = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0,
                          has_michael_acceptor=True)
    r_clean = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert r_ma.chemical_stability_score < r_clean.chemical_stability_score


def test_no_reactive_groups_top_chem_score():
    r = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert r.chemical_stability_score == pytest.approx(20.0)


def test_ideal_logp_top_perm_score():
    r = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert r.permeability_score == pytest.approx(20.0)


def test_low_dose_number_top_dn_score():
    r = calculate_fdi("Drug", solubility_mg_mL=1000.0, logP=2.0, dose_mg=10.0)
    assert r.dose_number_score == pytest.approx(20.0)


def test_valid_fdi_class_values():
    valid = {"excellent", "good", "moderate", "challenging", "very_challenging"}
    r = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert r.fdi_class in valid


def test_valid_development_risk():
    valid = {"low", "moderate", "high"}
    r = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert r.development_risk in valid


def test_valid_recommended_formulation():
    valid = {"standard_oral", "solid_dispersion", "nanosuspension",
             "lipid_formulation", "parenteral"}
    r = calculate_fdi("Drug", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0)
    assert r.recommended_formulation in valid


def test_invalid_solubility_raises():
    with pytest.raises(ValueError, match="solubility_mg_mL"):
        calculate_fdi("X", solubility_mg_mL=0.0, logP=2.0, dose_mg=100.0)


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        calculate_fdi("X", solubility_mg_mL=10.0, logP=2.0, dose_mg=0.0)


def test_invalid_logp_raises():
    with pytest.raises(ValueError, match="logP"):
        calculate_fdi("X", solubility_mg_mL=10.0, logP=15.0, dose_mg=100.0)


def test_invalid_melting_point_raises():
    with pytest.raises(ValueError, match="melting_point_C"):
        calculate_fdi("X", solubility_mg_mL=10.0, logP=2.0, dose_mg=100.0,
                       melting_point_C=10.0)


def test_rank_candidates_sorted_descending():
    candidates = [
        {"drug_name": "A", "solubility_mg_mL": 0.01, "logP": 6.0, "dose_mg": 500.0,
         "melting_point_C": 50.0, "hygroscopic": True,
         "has_ester": True, "has_aldehyde": True},
        {"drug_name": "B", "solubility_mg_mL": 500.0, "logP": 2.0, "dose_mg": 50.0,
         "melting_point_C": 220.0, "hygroscopic": False},
        {"drug_name": "C", "solubility_mg_mL": 5.0, "logP": 3.0, "dose_mg": 200.0},
    ]
    results = rank_candidates(candidates)
    scores = [r.fdi_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rank_candidates_returns_correct_count():
    candidates = [
        {"drug_name": "A", "solubility_mg_mL": 10.0, "logP": 2.0, "dose_mg": 100.0},
        {"drug_name": "B", "solubility_mg_mL": 50.0, "logP": 1.5, "dose_mg": 50.0},
    ]
    results = rank_candidates(candidates)
    assert len(results) == 2
