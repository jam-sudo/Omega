"""Tests for Phase 178 — Sick day protocol PK adjustments."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.sick_day_protocol import (
    SickDayPKResult,
    assess_sick_day_pk,
    recommend_sick_day_dose,
)

_BASE = dict(
    drug_name="TestDrug",
    illness_type="fever",
    baseline_dose_mg=100.0,
    baseline_cl_L_per_h=5.0,
    baseline_vd_L=50.0,
    baseline_f=0.8,
    baseline_ka_per_h=1.0,
    dosing_interval_h=24.0,
)


def test_assess_returns_correct_type():
    result = assess_sick_day_pk(**_BASE)
    assert isinstance(result, SickDayPKResult)


def test_recommend_returns_correct_type():
    result = recommend_sick_day_dose(**_BASE)
    assert isinstance(result, SickDayPKResult)


def test_drug_name_preserved():
    result = assess_sick_day_pk(**_BASE)
    assert result.drug_name == "TestDrug"


def test_illness_type_preserved():
    result = assess_sick_day_pk(**_BASE)
    assert result.illness_type == "fever"


def test_baseline_dose_preserved():
    result = assess_sick_day_pk(**_BASE)
    assert result.baseline_dose_mg == pytest.approx(100.0)


def test_fever_increases_cl():
    result = assess_sick_day_pk(**_BASE)
    assert result.cl_sick_L_per_h > _BASE["baseline_cl_L_per_h"]


def test_fever_increases_vd():
    result = assess_sick_day_pk(**_BASE)
    assert result.vd_sick_L > _BASE["baseline_vd_L"]


def test_fever_reduces_f():
    result = assess_sick_day_pk(**_BASE)
    assert result.f_sick < _BASE["baseline_f"]


def test_vomiting_reduces_f_severely():
    r = assess_sick_day_pk(**{**_BASE, "illness_type": "vomiting"})
    assert r.f_sick < _BASE["baseline_f"] * 0.6


def test_dehydration_reduces_cl():
    r = assess_sick_day_pk(**{**_BASE, "illness_type": "dehydration"})
    assert r.cl_sick_L_per_h < _BASE["baseline_cl_L_per_h"]


def test_infection_reduces_cl():
    r = assess_sick_day_pk(**{**_BASE, "illness_type": "infection"})
    assert r.cl_sick_L_per_h < _BASE["baseline_cl_L_per_h"]


def test_css_baseline_positive():
    result = assess_sick_day_pk(**_BASE)
    assert result.css_avg_baseline_mg_L > 0.0


def test_css_sick_positive():
    result = assess_sick_day_pk(**_BASE)
    assert result.css_avg_sick_mg_L > 0.0


def test_auc_ratio_positive():
    result = assess_sick_day_pk(**_BASE)
    assert result.auc_ratio_sick_to_baseline > 0.0


def test_vomiting_triggers_skip_dose():
    r = assess_sick_day_pk(**{**_BASE, "illness_type": "vomiting"})
    assert r.clinical_action == "skip_dose"


def test_dehydration_monitoring_required():
    r = assess_sick_day_pk(**{**_BASE, "illness_type": "dehydration"})
    assert r.monitoring_required is True


def test_recommend_sick_dose_positive():
    result = recommend_sick_day_dose(**_BASE)
    assert result.sick_day_dose_mg > 0.0


def test_recommend_dose_capped_at_2x():
    result = recommend_sick_day_dose(**_BASE)
    assert result.sick_day_dose_mg <= _BASE["baseline_dose_mg"] * 2.0


def test_recommend_dose_floor_at_0p25x():
    result = recommend_sick_day_dose(**_BASE)
    assert result.sick_day_dose_mg >= _BASE["baseline_dose_mg"] * 0.25


def test_recommend_notes_nonempty():
    result = recommend_sick_day_dose(**_BASE)
    assert len(result.notes) > 0


def test_invalid_illness_type():
    with pytest.raises(ValueError, match="illness_type"):
        assess_sick_day_pk(**{**_BASE, "illness_type": "cold"})


def test_zero_dose_raises():
    with pytest.raises(ValueError, match="baseline_dose_mg"):
        assess_sick_day_pk(**{**_BASE, "baseline_dose_mg": 0.0})


def test_zero_cl_raises():
    with pytest.raises(ValueError, match="baseline_cl_L_per_h"):
        assess_sick_day_pk(**{**_BASE, "baseline_cl_L_per_h": 0.0})


def test_invalid_f_raises():
    with pytest.raises(ValueError, match="baseline_f"):
        assess_sick_day_pk(**{**_BASE, "baseline_f": 0.0})


def test_zero_interval_raises():
    with pytest.raises(ValueError, match="dosing_interval_h"):
        assess_sick_day_pk(**{**_BASE, "dosing_interval_h": 0.0})


def test_zero_target_auc_fraction_raises():
    with pytest.raises(ValueError, match="target_auc_fraction"):
        recommend_sick_day_dose(**{**_BASE, "target_auc_fraction": 0.0})
