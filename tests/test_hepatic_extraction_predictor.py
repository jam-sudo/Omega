"""Tests for hepatic extraction ratio predictor."""

import pytest

from omega_pbpk.prediction.hepatic_extraction_predictor import (
    HepExtractionResult,
    predict_hepatic_extraction,
    screen_hepatic_extraction,
)


def test_returns_correct_type():
    result = predict_hepatic_extraction("TestDrug", logP=2.0, mw=300.0)
    assert isinstance(result, HepExtractionResult)


def test_er_in_range():
    result = predict_hepatic_extraction("TestDrug", logP=2.0, mw=300.0)
    assert 0.0 < result.er < 1.0


def test_fh_plus_er_equals_one():
    result = predict_hepatic_extraction("TestDrug", logP=2.0, mw=300.0)
    assert result.fh + result.er == pytest.approx(1.0, abs=1e-10)


def test_extraction_class_valid():
    result = predict_hepatic_extraction("TestDrug", logP=2.0, mw=300.0)
    assert result.extraction_class in {"low", "intermediate", "high"}


def test_high_clint_gives_high_extraction():
    result = predict_hepatic_extraction(
        "HighCLint", logP=2.0, mw=300.0, fup=0.1, cl_int_uL_per_min_per_mg=1000.0
    )
    assert result.extraction_class == "high"


def test_low_clint_gives_low_extraction():
    result = predict_hepatic_extraction(
        "LowCLint", logP=2.0, mw=300.0, fup=0.5, cl_int_uL_per_min_per_mg=1.0
    )
    assert result.extraction_class == "low"


def test_first_pass_loss_pct_equals_er_times_100():
    result = predict_hepatic_extraction("TestDrug", logP=2.0, mw=300.0)
    assert result.first_pass_loss_pct == pytest.approx(result.er * 100.0, abs=1e-6)


def test_high_psa_reduces_er():
    low_psa = predict_hepatic_extraction(
        "Drug", logP=2.0, mw=300.0, psa=50.0, cl_int_uL_per_min_per_mg=100.0
    )
    high_psa = predict_hepatic_extraction(
        "Drug", logP=2.0, mw=300.0, psa=150.0, cl_int_uL_per_min_per_mg=100.0
    )
    assert high_psa.er < low_psa.er


def test_notes_nonempty():
    result = predict_hepatic_extraction("TestDrug", logP=2.0, mw=300.0)
    assert result.notes and len(result.notes) > 0


def test_cl_hepatic_consistent_with_er():
    result = predict_hepatic_extraction("TestDrug", logP=2.0, mw=300.0, qh_L_per_h=90.0)
    assert result.cl_hepatic_L_per_h == pytest.approx(result.er * 90.0, rel=1e-6)


def test_fu_mic_in_range():
    result = predict_hepatic_extraction("TestDrug", logP=2.0, mw=300.0)
    assert 0.01 <= result.fu_mic <= 1.0


def test_fu_mic_lipophilic_lower():
    lipophilic = predict_hepatic_extraction("Lipo", logP=4.0, mw=300.0)
    hydrophilic = predict_hepatic_extraction("Hydro", logP=0.5, mw=300.0)
    assert lipophilic.fu_mic < hydrophilic.fu_mic


def test_screen_sorted_descending_by_er():
    compounds = [
        {"compound_name": "A", "logP": 2.0, "mw": 300.0, "cl_int_uL_per_min_per_mg": 10.0},
        {"compound_name": "B", "logP": 2.0, "mw": 300.0, "cl_int_uL_per_min_per_mg": 500.0},
        {"compound_name": "C", "logP": 2.0, "mw": 300.0, "cl_int_uL_per_min_per_mg": 100.0},
    ]
    results = screen_hepatic_extraction(compounds)
    ers = [r.er for r in results]
    assert ers == sorted(ers, reverse=True)


def test_screen_empty_input():
    results = screen_hepatic_extraction([])
    assert results == []


def test_screen_returns_correct_length():
    compounds = [
        {"compound_name": "X", "logP": 1.0, "mw": 250.0},
        {"compound_name": "Y", "logP": 3.0, "mw": 400.0},
    ]
    results = screen_hepatic_extraction(compounds)
    assert len(results) == 2


def test_validation_fup_greater_than_one():
    with pytest.raises(ValueError, match="fup"):
        predict_hepatic_extraction("Drug", logP=2.0, mw=300.0, fup=1.5)


def test_validation_fup_zero():
    with pytest.raises(ValueError, match="fup"):
        predict_hepatic_extraction("Drug", logP=2.0, mw=300.0, fup=0.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_hepatic_extraction("Drug", logP=2.0, mw=0.0)


def test_validation_mw_negative():
    with pytest.raises(ValueError, match="mw"):
        predict_hepatic_extraction("Drug", logP=2.0, mw=-100.0)


def test_validation_cl_int_zero():
    with pytest.raises(ValueError, match="cl_int"):
        predict_hepatic_extraction("Drug", logP=2.0, mw=300.0, cl_int_uL_per_min_per_mg=0.0)


def test_high_mw_reduces_er():
    normal_mw = predict_hepatic_extraction(
        "Drug", logP=2.0, mw=300.0, cl_int_uL_per_min_per_mg=100.0
    )
    high_mw = predict_hepatic_extraction("Drug", logP=2.0, mw=900.0, cl_int_uL_per_min_per_mg=100.0)
    assert high_mw.er < normal_mw.er


def test_fh_is_one_minus_er():
    result = predict_hepatic_extraction("Drug", logP=2.0, mw=300.0)
    assert result.fh == pytest.approx(1.0 - result.er, abs=1e-12)
