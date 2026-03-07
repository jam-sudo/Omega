"""Tests for drug level monitoring — Phase 726."""

from __future__ import annotations

import pytest

from omega_pbpk.analysis.drug_level_monitoring import (
    DrugLevelAssessment,
    assess_drug_level,
    generate_dose_recommendation,
    monitor_steady_state,
)


def test_returns_assessment():
    result = assess_drug_level(
        compound_name="Vancomycin",
        measured_conc_mg_L=15.0,
        therapeutic_min_mg_L=10.0,
        therapeutic_max_mg_L=20.0,
        sample_time_h=1.0,
        dose_mg=1000.0,
    )
    assert isinstance(result, DrugLevelAssessment)


def test_in_therapeutic_range():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=5.0,
        therapeutic_min_mg_L=4.0,
        therapeutic_max_mg_L=8.0,
        sample_time_h=2.0,
        dose_mg=500.0,
    )
    assert result.in_therapeutic_range is True
    assert result.status == "therapeutic"


def test_subtherapeutic():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=2.0,
        therapeutic_min_mg_L=5.0,
        therapeutic_max_mg_L=10.0,
        sample_time_h=2.0,
        dose_mg=500.0,
    )
    assert result.in_therapeutic_range is False
    assert result.status == "subtherapeutic"


def test_supratherapeutic():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=12.0,
        therapeutic_min_mg_L=5.0,
        therapeutic_max_mg_L=10.0,
        sample_time_h=2.0,
        dose_mg=500.0,
    )
    assert result.status == "supratherapeutic"
    assert result.in_therapeutic_range is False


def test_toxic():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=25.0,
        therapeutic_min_mg_L=5.0,
        therapeutic_max_mg_L=10.0,
        sample_time_h=2.0,
        dose_mg=500.0,
    )
    assert result.status == "toxic"


def test_dose_adjustment_factor_clamped_lower():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=100.0,
        therapeutic_min_mg_L=2.0,
        therapeutic_max_mg_L=4.0,
        sample_time_h=1.0,
        dose_mg=100.0,
    )
    assert result.dose_adjustment_factor >= 0.5


def test_dose_adjustment_factor_clamped_upper():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=0.01,
        therapeutic_min_mg_L=5.0,
        therapeutic_max_mg_L=10.0,
        sample_time_h=1.0,
        dose_mg=100.0,
    )
    assert result.dose_adjustment_factor <= 2.0


def test_percent_of_min():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=10.0,
        therapeutic_min_mg_L=5.0,
        therapeutic_max_mg_L=20.0,
        sample_time_h=1.0,
        dose_mg=500.0,
    )
    assert abs(result.percent_of_min - 200.0) < 0.01


def test_percent_of_max():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=10.0,
        therapeutic_min_mg_L=5.0,
        therapeutic_max_mg_L=20.0,
        sample_time_h=1.0,
        dose_mg=500.0,
    )
    assert abs(result.percent_of_max - 50.0) < 0.01


def test_back_calculated_dose_with_pk_params():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=1.0,
        therapeutic_min_mg_L=0.5,
        therapeutic_max_mg_L=5.0,
        sample_time_h=4.0,
        dose_mg=100.0,
        t_half_h=6.0,
        vd_L=50.0,
        last_dose_time_h=4.0,
    )
    assert result.back_calculated_dose_mg > 0.0


def test_no_back_calc_without_pk_params():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=5.0,
        therapeutic_min_mg_L=4.0,
        therapeutic_max_mg_L=8.0,
        sample_time_h=1.0,
        dose_mg=200.0,
    )
    assert result.back_calculated_dose_mg == 0.0


def test_notes_nonempty():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=5.0,
        therapeutic_min_mg_L=4.0,
        therapeutic_max_mg_L=8.0,
        sample_time_h=1.0,
        dose_mg=200.0,
    )
    assert len(result.notes) > 0


def test_compound_name_stored():
    result = assess_drug_level(
        compound_name="Theophylline",
        measured_conc_mg_L=12.0,
        therapeutic_min_mg_L=10.0,
        therapeutic_max_mg_L=20.0,
        sample_time_h=2.0,
        dose_mg=300.0,
    )
    assert result.compound_name == "Theophylline"


def test_dose_recommendation_in_therapeutic():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=5.0,
        therapeutic_min_mg_L=4.0,
        therapeutic_max_mg_L=8.0,
        sample_time_h=1.0,
        dose_mg=200.0,
    )
    assert "Maintain" in result.dose_recommendation


def test_dose_recommendation_subtherapeutic():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=2.0,
        therapeutic_min_mg_L=5.0,
        therapeutic_max_mg_L=10.0,
        sample_time_h=1.0,
        dose_mg=200.0,
    )
    assert "ncrease" in result.dose_recommendation


def test_dose_recommendation_toxic():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=30.0,
        therapeutic_min_mg_L=5.0,
        therapeutic_max_mg_L=10.0,
        sample_time_h=1.0,
        dose_mg=200.0,
    )
    assert "HOLD" in result.dose_recommendation or "TOXIC" in result.dose_recommendation


def test_validation_negative_conc():
    with pytest.raises(ValueError):
        assess_drug_level(
            compound_name="Drug",
            measured_conc_mg_L=-1.0,
            therapeutic_min_mg_L=5.0,
            therapeutic_max_mg_L=10.0,
            sample_time_h=1.0,
            dose_mg=200.0,
        )


def test_validation_min_ge_max():
    with pytest.raises(ValueError):
        assess_drug_level(
            compound_name="Drug",
            measured_conc_mg_L=5.0,
            therapeutic_min_mg_L=10.0,
            therapeutic_max_mg_L=5.0,
            sample_time_h=1.0,
            dose_mg=200.0,
        )


def test_monitor_steady_state_optimal():
    result = monitor_steady_state(
        compound_name="Vancomycin",
        trough_conc_mg_L=12.0,
        peak_conc_mg_L=35.0,
        therapeutic_trough_min=10.0,
        therapeutic_trough_max=15.0,
        therapeutic_peak_min=30.0,
        therapeutic_peak_max=40.0,
    )
    assert result["overall_status"] == "optimal"
    assert result["trough_in_range"] is True
    assert result["peak_in_range"] is True


def test_monitor_steady_state_returns_dict():
    result = monitor_steady_state(
        compound_name="Drug",
        trough_conc_mg_L=5.0,
        peak_conc_mg_L=20.0,
        therapeutic_trough_min=4.0,
        therapeutic_trough_max=8.0,
        therapeutic_peak_min=15.0,
        therapeutic_peak_max=25.0,
    )
    assert isinstance(result, dict)
    assert "trough_status" in result
    assert "peak_status" in result
    assert "overall_status" in result


def test_monitor_steady_state_subtherapeutic_trough():
    result = monitor_steady_state(
        compound_name="Drug",
        trough_conc_mg_L=2.0,
        peak_conc_mg_L=20.0,
        therapeutic_trough_min=5.0,
        therapeutic_trough_max=10.0,
        therapeutic_peak_min=15.0,
        therapeutic_peak_max=25.0,
    )
    assert result["trough_status"] == "subtherapeutic"
    assert result["trough_in_range"] is False


def test_generate_dose_recommendation_returns_string():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=5.0,
        therapeutic_min_mg_L=4.0,
        therapeutic_max_mg_L=8.0,
        sample_time_h=1.0,
        dose_mg=200.0,
    )
    rec = generate_dose_recommendation(result)
    assert isinstance(rec, str)
    assert len(rec) > 0


def test_generate_dose_recommendation_matches_assessment():
    result = assess_drug_level(
        compound_name="Drug",
        measured_conc_mg_L=5.0,
        therapeutic_min_mg_L=4.0,
        therapeutic_max_mg_L=8.0,
        sample_time_h=1.0,
        dose_mg=200.0,
    )
    rec = generate_dose_recommendation(result)
    assert rec == result.dose_recommendation
