"""Tests for cardiovascular risk assessment module (Phase 701)."""

from __future__ import annotations

import pytest

from omega_pbpk.risk.cardiovascular_risk import (
    CardiovascularRiskResult,
    assess_cardiovascular_risk,
    estimate_herg_pIC50,
    screen_cardiovascular_risk,
)

# ---------------------------------------------------------------------------
# Basic return type and structure
# ---------------------------------------------------------------------------


def test_returns_cardiovascular_risk_result():
    result = assess_cardiovascular_risk("TestDrug", logP=3.0, mw=350.0, psa=60.0)
    assert isinstance(result, CardiovascularRiskResult)


def test_result_fields_present():
    result = assess_cardiovascular_risk("TestDrug", logP=3.0, mw=350.0, psa=60.0)
    assert result.compound_name == "TestDrug"
    assert result.therapeutic_conc_uM == 1.0
    assert result.herg_ic50_uM > 0
    assert result.sr_hERG > 0


def test_notes_nonempty():
    result = assess_cardiovascular_risk("TestDrug", logP=3.0, mw=350.0, psa=60.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# hERG QSAR and safety ratio
# ---------------------------------------------------------------------------


def test_high_logP_basic_drug_low_herg_ic50_high_risk():
    # High logP + basic amine → low hERG IC50 → high cardiac risk
    result = assess_cardiovascular_risk(
        "HighRisk", logP=5.0, mw=350.0, psa=40.0, pka_base=9.0, therapeutic_conc_uM=1.0
    )
    # pIC50 = min(8, 5*2.5 + (10-3.5)*0.5 + 3) = min(8, 12.5+3.25+3) = min(8, 18.75) = 8
    # IC50 = 10^(6-8) = 0.01 µM, SR=0.01 → very_high
    assert result.sr_hERG <= 3.0
    assert result.overall_cardiac_risk == "very_high"


def test_low_logP_neutral_low_risk():
    # Low logP neutral compound → high IC50 → low risk
    result = assess_cardiovascular_risk(
        "LowRisk", logP=0.5, mw=300.0, psa=120.0, therapeutic_conc_uM=0.01
    )
    # base_score = 0.5*2.5 + (10-3)*0.5 - 5 = 1.25 + 3.5 - 5 = -0.25
    # pIC50 = max(3, -0.25) = 3.0, IC50 = 10^3 = 1000 µM, SR = 100000
    assert result.sr_hERG > 30
    assert result.qtc_risk == "low"


def test_sr_herg_equals_ic50_div_conc():
    result = assess_cardiovascular_risk(
        "Drug", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=10.0, therapeutic_conc_uM=2.0
    )
    assert abs(result.sr_hERG - 5.0) < 1e-9


def test_provided_herg_ic50_overrides_qsar():
    result_qsar = assess_cardiovascular_risk("Drug", logP=3.0, mw=300.0, psa=70.0)
    result_exp = assess_cardiovascular_risk("Drug", logP=3.0, mw=300.0, psa=70.0, herg_ic50_uM=0.1)
    assert abs(result_exp.herg_ic50_uM - 0.1) < 1e-9
    # QSAR result should be different from 0.1 µM for this logP
    assert abs(result_qsar.herg_ic50_uM - 0.1) > 0.01


# ---------------------------------------------------------------------------
# QTc risk classification
# ---------------------------------------------------------------------------


def test_qtc_risk_low_when_sr_above_30():
    result = assess_cardiovascular_risk(
        "Safe", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=50.0, therapeutic_conc_uM=1.0
    )
    assert result.qtc_risk == "low"


def test_qtc_risk_moderate_when_sr_between_10_and_30():
    result = assess_cardiovascular_risk(
        "Mod", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=20.0, therapeutic_conc_uM=1.0
    )
    assert result.qtc_risk == "moderate"


def test_qtc_risk_high_when_sr_at_or_below_10():
    result = assess_cardiovascular_risk(
        "High", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=5.0, therapeutic_conc_uM=1.0
    )
    assert result.qtc_risk == "high"


def test_qtc_risk_values():
    result = assess_cardiovascular_risk("X", logP=2.0, mw=300.0, psa=70.0)
    assert result.qtc_risk in {"low", "moderate", "high"}


# ---------------------------------------------------------------------------
# Overall cardiac risk
# ---------------------------------------------------------------------------


def test_overall_cardiac_risk_values():
    result = assess_cardiovascular_risk("X", logP=2.0, mw=300.0, psa=70.0)
    assert result.overall_cardiac_risk in {"low", "moderate", "high", "very_high"}


def test_overall_risk_very_high_when_sr_le_3():
    result = assess_cardiovascular_risk(
        "VH", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=2.0, therapeutic_conc_uM=1.0
    )
    assert result.overall_cardiac_risk == "very_high"


def test_overall_risk_high_when_sr_between_3_and_10():
    result = assess_cardiovascular_risk(
        "H", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=7.0, therapeutic_conc_uM=1.0
    )
    assert result.overall_cardiac_risk == "high"


def test_requires_cardiac_safety_study_when_sr_lt_30():
    result = assess_cardiovascular_risk(
        "X", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=20.0, therapeutic_conc_uM=1.0
    )
    assert result.requires_cardiac_safety_study is True


def test_requires_cardiac_safety_study_false_when_sr_ge_30():
    result = assess_cardiovascular_risk(
        "X", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=50.0, therapeutic_conc_uM=1.0
    )
    assert result.requires_cardiac_safety_study is False


def test_tqt_recommended_when_sr_lt_10():
    result = assess_cardiovascular_risk(
        "X", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=5.0, therapeutic_conc_uM=1.0
    )
    assert result.tqt_study_recommended is True


def test_tqt_not_recommended_when_sr_ge_10():
    result = assess_cardiovascular_risk(
        "X", logP=2.0, mw=300.0, psa=70.0, herg_ic50_uM=15.0, therapeutic_conc_uM=1.0
    )
    assert result.tqt_study_recommended is False


# ---------------------------------------------------------------------------
# Nav1.5
# ---------------------------------------------------------------------------


def test_nav15_sr_computed_when_provided():
    result = assess_cardiovascular_risk(
        "Drug", logP=2.0, mw=300.0, psa=70.0, nav15_ic50_uM=5.0, therapeutic_conc_uM=1.0
    )
    assert result.nav15_ic50_uM == 5.0
    assert result.sr_nav15 == 5.0


def test_nav15_none_when_not_provided():
    result = assess_cardiovascular_risk("Drug", logP=2.0, mw=300.0, psa=70.0)
    assert result.nav15_ic50_uM is None
    assert result.sr_nav15 is None


# ---------------------------------------------------------------------------
# estimate_herg_pIC50
# ---------------------------------------------------------------------------


def test_estimate_herg_pic50_returns_float():
    val = estimate_herg_pIC50(3.0, 350.0, 70.0)
    assert isinstance(val, float)


def test_estimate_herg_pic50_in_range():
    for logp in [-2, 0, 2, 4, 6]:
        val = estimate_herg_pIC50(float(logp), 350.0, 70.0)
        assert 3.0 <= val <= 8.0


def test_estimate_herg_pic50_basic_amine_higher():
    val_neutral = estimate_herg_pIC50(3.0, 350.0, 70.0)
    val_basic = estimate_herg_pIC50(3.0, 350.0, 70.0, pka_base=9.0)
    assert val_basic >= val_neutral


def test_estimate_herg_pic50_high_psa_lower():
    val_low_psa = estimate_herg_pIC50(3.0, 350.0, 50.0)
    val_high_psa = estimate_herg_pIC50(3.0, 350.0, 150.0)
    assert val_high_psa < val_low_psa


# ---------------------------------------------------------------------------
# screen_cardiovascular_risk
# ---------------------------------------------------------------------------


def test_screen_cardiovascular_risk_sorted_ascending_sr():
    compounds = [
        {"compound_name": "A", "logP": 1.0, "mw": 300.0, "psa": 80.0, "herg_ic50_uM": 50.0},
        {"compound_name": "B", "logP": 1.0, "mw": 300.0, "psa": 80.0, "herg_ic50_uM": 5.0},
        {"compound_name": "C", "logP": 1.0, "mw": 300.0, "psa": 80.0, "herg_ic50_uM": 20.0},
    ]
    results = screen_cardiovascular_risk(compounds)
    srs = [r.sr_hERG for r in results]
    assert srs == sorted(srs)


def test_screen_cardiovascular_risk_returns_list():
    compounds = [
        {"compound_name": "A", "logP": 2.0, "mw": 300.0, "psa": 70.0},
        {"compound_name": "B", "logP": 4.0, "mw": 350.0, "psa": 50.0},
    ]
    results = screen_cardiovascular_risk(compounds)
    assert isinstance(results, list)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_therapeutic_conc_zero_raises():
    with pytest.raises(ValueError):
        assess_cardiovascular_risk("X", logP=2.0, mw=300.0, psa=70.0, therapeutic_conc_uM=0.0)


def test_validation_therapeutic_conc_negative_raises():
    with pytest.raises(ValueError):
        assess_cardiovascular_risk("X", logP=2.0, mw=300.0, psa=70.0, therapeutic_conc_uM=-1.0)
