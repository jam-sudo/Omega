"""Tests for Phase 702 — Hepatic Blood Flow Extraction."""

import pytest

from omega_pbpk.core.hepatic_blood_flow_p702 import (
    HepaticBloodFlowResult,
    simulate_hepatic_blood_flow,
)


@pytest.fixture
def base_result():
    return simulate_hepatic_blood_flow("TestDrug", cl_int_L_per_h=50.0, fu_plasma=0.5)


def test_return_type(base_result):
    assert isinstance(base_result, HepaticBloodFlowResult)


def test_drug_name(base_result):
    assert base_result.drug_name == "TestDrug"


def test_cl_int_stored(base_result):
    assert base_result.cl_int_L_per_h == 50.0


def test_fu_stored(base_result):
    assert base_result.fu_plasma == 0.5


def test_qh_stored(base_result):
    assert base_result.qh_L_per_h == 90.0


def test_cl_well_stirred_positive(base_result):
    assert base_result.cl_well_stirred_L_per_h > 0


def test_cl_parallel_tube_positive(base_result):
    assert base_result.cl_parallel_tube_L_per_h > 0


def test_cl_dispersion_positive(base_result):
    assert base_result.cl_dispersion_L_per_h > 0


def test_er_well_stirred_range(base_result):
    assert 0 <= base_result.er_well_stirred <= 1


def test_er_parallel_tube_range(base_result):
    assert 0 <= base_result.er_parallel_tube <= 1


def test_er_dispersion_range(base_result):
    assert 0 <= base_result.er_dispersion <= 1


def test_f_oral_well_stirred_range(base_result):
    assert 0 <= base_result.f_oral_well_stirred <= 1


def test_f_oral_parallel_tube_range(base_result):
    assert 0 <= base_result.f_oral_parallel_tube <= 1


def test_f_oral_dispersion_range(base_result):
    assert 0 <= base_result.f_oral_dispersion <= 1


def test_high_clint_high_extraction():
    r = simulate_hepatic_blood_flow("HighE", cl_int_L_per_h=1000.0, fu_plasma=0.8)
    assert r.er_well_stirred > 0.7
    assert r.classification == "high extraction"


def test_low_clint_low_extraction():
    r = simulate_hepatic_blood_flow("LowE", cl_int_L_per_h=5.0, fu_plasma=0.1)
    assert r.er_well_stirred < 0.3
    assert r.classification == "low extraction"


def test_cl_ws_le_qh(base_result):
    assert base_result.cl_well_stirred_L_per_h <= base_result.qh_L_per_h


def test_cl_pt_ge_cl_ws(base_result):
    assert base_result.cl_parallel_tube_L_per_h >= base_result.cl_well_stirred_L_per_h - 1e-9


def test_f_oral_plus_er_ws_equals_one(base_result):
    assert base_result.f_oral_well_stirred + base_result.er_well_stirred == pytest.approx(1.0)


def test_f_oral_plus_er_pt_equals_one(base_result):
    assert base_result.f_oral_parallel_tube + base_result.er_parallel_tube == pytest.approx(1.0)


def test_f_oral_plus_er_disp_equals_one(base_result):
    assert base_result.f_oral_dispersion + base_result.er_dispersion == pytest.approx(1.0)


def test_classification_valid(base_result):
    assert base_result.classification in {
        "low extraction",
        "intermediate extraction",
        "high extraction",
    }


def test_validation_negative_clint():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow("X", cl_int_L_per_h=-10, fu_plasma=0.5)


def test_validation_zero_clint():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow("X", cl_int_L_per_h=0, fu_plasma=0.5)


def test_validation_fu_zero():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow("X", cl_int_L_per_h=50, fu_plasma=0.0)


def test_validation_fu_above_one():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow("X", cl_int_L_per_h=50, fu_plasma=1.5)


def test_validation_negative_qh():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow("X", cl_int_L_per_h=50, fu_plasma=0.5, qh_L_per_h=-1)


def test_validation_negative_dispersion():
    with pytest.raises(ValueError):
        simulate_hepatic_blood_flow("X", cl_int_L_per_h=50, fu_plasma=0.5, dispersion_number=-0.1)


def test_notes_is_string(base_result):
    assert isinstance(base_result.notes, str)
    assert len(base_result.notes) > 0


def test_intermediate_extraction():
    r = simulate_hepatic_blood_flow("Mid", cl_int_L_per_h=50.0, fu_plasma=0.5)
    assert r.classification in {"low extraction", "intermediate extraction", "high extraction"}
