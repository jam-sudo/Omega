"""Tests for Phase 482 — Hepatic Sinusoidal Extraction."""

import pytest

from omega_pbpk.core.hepatic_sinusoidal_extraction_p482 import (
    HepaticSinusoidalResult,
    screen_drugs_sinusoidal,
    simulate_hepatic_sinusoidal_extraction,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(**kwargs):
    defaults = dict(drug_name="DrugA", fu_plasma=0.5, cl_int_total_L_per_h=50.0)
    defaults.update(kwargs)
    return simulate_hepatic_sinusoidal_extraction(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _run()
    assert isinstance(result, HepaticSinusoidalResult)


def test_has_all_fields():
    result = _run()
    for attr in [
        "drug_name",
        "fu_plasma",
        "cl_int_total_L_per_h",
        "Q_h_L_per_h",
        "e_zone1",
        "e_zone2",
        "e_zone3",
        "f_h_zone1",
        "f_h_zone2",
        "f_h_zone3",
        "f_h_total",
        "er_total",
        "cl_hepatic_L_per_h",
        "well_stirred_er",
        "zonal_vs_wellstirred_ratio",
        "notes",
    ]:
        assert hasattr(result, attr)


# ---------------------------------------------------------------------------
# Zone ordering: periportal has highest CYP → highest extraction
# ---------------------------------------------------------------------------


def test_e_zone1_greater_than_e_zone3():
    result = _run()
    assert result.e_zone1 > result.e_zone3


def test_e_zone1_greater_than_e_zone2():
    result = _run()
    assert result.e_zone1 > result.e_zone2


def test_e_zone2_greater_than_e_zone3():
    result = _run()
    assert result.e_zone2 > result.e_zone3


# ---------------------------------------------------------------------------
# f_h = 1 - e relationships
# ---------------------------------------------------------------------------


def test_f_h_zone1_equals_one_minus_e_zone1():
    result = _run()
    assert abs(result.f_h_zone1 - (1.0 - result.e_zone1)) < 1e-9


def test_f_h_zone2_equals_one_minus_e_zone2():
    result = _run()
    assert abs(result.f_h_zone2 - (1.0 - result.e_zone2)) < 1e-9


def test_f_h_zone3_equals_one_minus_e_zone3():
    result = _run()
    assert abs(result.f_h_zone3 - (1.0 - result.e_zone3)) < 1e-9


# ---------------------------------------------------------------------------
# Cascade multiplication
# ---------------------------------------------------------------------------


def test_f_h_total_equals_product_of_zones():
    result = _run()
    expected = result.f_h_zone1 * result.f_h_zone2 * result.f_h_zone3
    assert abs(result.f_h_total - expected) < 1e-9


def test_er_total_equals_one_minus_f_h_total():
    result = _run()
    assert abs(result.er_total - (1.0 - result.f_h_total)) < 1e-9


# ---------------------------------------------------------------------------
# Range checks
# ---------------------------------------------------------------------------


def test_f_h_total_in_open_unit_interval():
    result = _run()
    assert 0.0 < result.f_h_total < 1.0


def test_er_total_in_unit_interval():
    result = _run()
    assert 0.0 < result.er_total < 1.0


def test_all_e_zones_in_unit_interval():
    result = _run()
    for e in [result.e_zone1, result.e_zone2, result.e_zone3]:
        assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------
# cl_hepatic
# ---------------------------------------------------------------------------


def test_cl_hepatic_equals_qh_times_er():
    result = _run()
    expected = result.Q_h_L_per_h * result.er_total
    assert abs(result.cl_hepatic_L_per_h - expected) < 1e-6


# ---------------------------------------------------------------------------
# Higher CLint → higher ER
# ---------------------------------------------------------------------------


def test_higher_cl_int_gives_higher_er():
    low = _run(cl_int_total_L_per_h=10.0)
    high = _run(cl_int_total_L_per_h=500.0)
    assert high.er_total > low.er_total


# ---------------------------------------------------------------------------
# Well-stirred comparison
# ---------------------------------------------------------------------------


def test_well_stirred_er_in_unit_interval():
    result = _run()
    assert 0.0 <= result.well_stirred_er <= 1.0


def test_zonal_vs_wellstirred_ratio_positive_and_finite():
    # Ratio should be a positive finite number
    result = _run(fu_plasma=0.01, cl_int_total_L_per_h=1.0)
    assert result.zonal_vs_wellstirred_ratio > 0.0
    assert result.zonal_vs_wellstirred_ratio < 100.0


def test_zonal_ratio_positive():
    result = _run()
    assert result.zonal_vs_wellstirred_ratio > 0.0


# ---------------------------------------------------------------------------
# screen_drugs_sinusoidal
# ---------------------------------------------------------------------------


def test_screen_drugs_returns_list():
    drugs = [
        {"drug_name": "A", "fu_plasma": 0.9, "cl_int_total_L_per_h": 5.0},
        {"drug_name": "B", "fu_plasma": 0.5, "cl_int_total_L_per_h": 200.0},
    ]
    results = screen_drugs_sinusoidal(drugs)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_drugs_sorted_by_f_h_total_descending():
    drugs = [
        {"drug_name": "HighExtraction", "fu_plasma": 0.9, "cl_int_total_L_per_h": 5000.0},
        {"drug_name": "LowExtraction", "fu_plasma": 0.01, "cl_int_total_L_per_h": 0.1},
    ]
    results = screen_drugs_sinusoidal(drugs)
    assert results[0].f_h_total >= results[1].f_h_total


def test_screen_three_drugs_sorted():
    drugs = [
        {"drug_name": "C1", "fu_plasma": 0.5, "cl_int_total_L_per_h": 10.0},
        {"drug_name": "C2", "fu_plasma": 0.5, "cl_int_total_L_per_h": 100.0},
        {"drug_name": "C3", "fu_plasma": 0.5, "cl_int_total_L_per_h": 1000.0},
    ]
    results = screen_drugs_sinusoidal(drugs)
    for i in range(len(results) - 1):
        assert results[i].f_h_total >= results[i + 1].f_h_total


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_fu_plasma_zero_raises():
    with pytest.raises(ValueError):
        _run(fu_plasma=0.0)


def test_fu_plasma_greater_than_one_raises():
    with pytest.raises(ValueError):
        _run(fu_plasma=1.5)


def test_cl_int_zero_raises():
    with pytest.raises(ValueError):
        _run(cl_int_total_L_per_h=0.0)


def test_cl_int_negative_raises():
    with pytest.raises(ValueError):
        _run(cl_int_total_L_per_h=-10.0)


def test_qh_zero_raises():
    with pytest.raises(ValueError):
        _run(Q_h_L_per_h=0.0)
