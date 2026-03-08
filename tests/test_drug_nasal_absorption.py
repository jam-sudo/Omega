"""Tests for Phase 1008 — drug_nasal_absorption module."""

import pytest

from omega_pbpk.prediction.drug_nasal_absorption import (
    NasalAbsorptionResult,
    compare_nasal_compounds,
    predict_nasal_absorption,
)

VALID_KWARGS = dict(drug_name="Lidocaine", logp=2.4, mw=234.3, psa=32.0, dose_mg=2.0, oral_f=0.35)


def test_returns_result_type():
    result = predict_nasal_absorption(**VALID_KWARGS)
    assert isinstance(result, NasalAbsorptionResult)


def test_f_nasal_in_range():
    result = predict_nasal_absorption(**VALID_KWARGS)
    assert 0.0 < result.f_nasal < 1.0


def test_ka_nasal_positive():
    result = predict_nasal_absorption(**VALID_KWARGS)
    assert result.ka_nasal_per_min > 0


def test_mucociliary_clearance_fraction_in_range():
    result = predict_nasal_absorption(**VALID_KWARGS)
    assert 0.0 < result.mucociliary_clearance_fraction < 1.0


def test_tmax_nasal_min_positive():
    result = predict_nasal_absorption(**VALID_KWARGS)
    assert result.tmax_nasal_min > 0


def test_nasal_vs_oral_ratio_positive():
    result = predict_nasal_absorption(**VALID_KWARGS)
    assert result.nasal_vs_oral_f_ratio > 0


def test_lipophilic_higher_f_nasal_than_hydrophilic():
    lipophilic = predict_nasal_absorption(drug_name="Lipophilic", logp=3.0, mw=250.0, psa=50.0)
    hydrophilic = predict_nasal_absorption(drug_name="Hydrophilic", logp=-2.0, mw=250.0, psa=50.0)
    assert lipophilic.f_nasal > hydrophilic.f_nasal


def test_high_psa_reduces_f_nasal():
    low_psa = predict_nasal_absorption(drug_name="LowPSA", logp=1.5, mw=300.0, psa=20.0)
    high_psa = predict_nasal_absorption(drug_name="HighPSA", logp=1.5, mw=300.0, psa=200.0)
    assert low_psa.f_nasal > high_psa.f_nasal


def test_large_mw_reduces_f_nasal():
    small = predict_nasal_absorption(drug_name="SmallMW", logp=2.0, mw=200.0, psa=60.0)
    large = predict_nasal_absorption(drug_name="LargeMW", logp=2.0, mw=1000.0, psa=60.0)
    assert small.f_nasal > large.f_nasal


def test_recommended_formulation_valid_values():
    result = predict_nasal_absorption(**VALID_KWARGS)
    assert result.recommended_formulation in {"solution", "powder", "gel", "spray"}


def test_large_mw_recommends_powder():
    result = predict_nasal_absorption(drug_name="BigMol", logp=2.0, mw=600.0, psa=80.0)
    assert result.recommended_formulation == "powder"


def test_hydrophilic_recommends_gel():
    result = predict_nasal_absorption(drug_name="HydroMol", logp=-1.0, mw=300.0, psa=80.0)
    assert result.recommended_formulation == "gel"


def test_small_lipophilic_recommends_spray():
    result = predict_nasal_absorption(drug_name="SprayMol", logp=2.0, mw=150.0, psa=30.0)
    assert result.recommended_formulation == "spray"


def test_compare_nasal_compounds_returns_sorted_list():
    compounds = [
        {"drug_name": "DrugA", "logp": -2.0, "mw": 300.0},
        {"drug_name": "DrugB", "logp": 3.0, "mw": 250.0},
        {"drug_name": "DrugC", "logp": 1.0, "mw": 400.0},
    ]
    results = compare_nasal_compounds(compounds)
    assert len(results) == 3
    f_values = [r.f_nasal for r in results]
    assert f_values == sorted(f_values, reverse=True)


def test_nasal_vs_oral_ratio_greater_than_one_for_high_nasal_low_oral():
    # Drug with good nasal F but very low oral F
    result = predict_nasal_absorption(
        drug_name="NasalDrug", logp=2.5, mw=200.0, psa=40.0, dose_mg=1.0, oral_f=0.05
    )
    assert result.nasal_vs_oral_f_ratio > 1.0


def test_invalid_mw_raises_value_error():
    with pytest.raises(ValueError, match="mw"):
        predict_nasal_absorption(drug_name="Drug", logp=1.0, mw=0.0)


def test_invalid_dose_raises_value_error():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_nasal_absorption(drug_name="Drug", logp=1.0, mw=200.0, dose_mg=-1.0)


def test_invalid_oral_f_raises_value_error():
    with pytest.raises(ValueError, match="oral_f"):
        predict_nasal_absorption(drug_name="Drug", logp=1.0, mw=200.0, oral_f=1.5)
