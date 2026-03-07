"""Tests for Enhanced NCA — Phase 813."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.nca_enhanced import (
    NCAEnhancedResult,
    batch_nca,
    nca_enhanced,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_C0 = 10.0
_KE = 0.1
_DOSE = 100.0
_TIMES = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
_CONCS = [_C0 * math.exp(-_KE * t) for t in _TIMES]


def test_returns_nca_enhanced_result():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert isinstance(result, NCAEnhancedResult)


def test_compound_name_preserved():
    result = nca_enhanced("TestDrug", _TIMES, _CONCS, _DOSE)
    assert result.compound_name == "TestDrug"


def test_dose_preserved():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.dose_mg == _DOSE


def test_route_preserved_iv():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE, route="iv")
    assert result.route == "iv"


def test_route_preserved_oral():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE, route="oral")
    assert result.route == "oral"


def test_notes_nonempty():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert len(result.notes) > 0


def test_cmax_equals_first_point_for_iv():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE, route="iv")
    assert abs(result.cmax - _C0) < 1e-6


def test_tmax_is_zero_for_iv_monoexp():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE, route="iv")
    assert result.tmax_h == 0.0


def test_cmax_positive():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.cmax > 0.0


def test_auc_0_t_positive():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.auc_0_t > 0.0


def test_auc_0_inf_geq_auc_0_t():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.auc_0_inf >= result.auc_0_t


def test_auc_extrapolation_pct_between_0_and_100():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert 0.0 <= result.auc_extrapolation_pct <= 100.0


def test_auc_0_inf_analytical():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE, route="iv")
    expected = _C0 / _KE
    assert abs(result.auc_0_inf - expected) / expected < 0.15


def test_kel_positive():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.kel > 0.0


def test_t_half_analytical():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    expected_t_half = math.log(2) / _KE
    assert abs(result.t_half_h - expected_t_half) / expected_t_half < 0.15


def test_kel_r2_high_for_monoexp():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.kel_r2 > 0.95


def test_n_timepoints_terminal_geq_3():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.n_timepoints_terminal >= 3


def test_aumc_positive():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.aumc > 0.0


def test_mrt_positive():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.mrt_h > 0.0


def test_mrt_analytical():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE, route="iv")
    expected_mrt = 1.0 / _KE
    assert abs(result.mrt_h - expected_mrt) / expected_mrt < 0.20


def test_vd_area_positive():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.vd_area_L > 0.0


def test_cl_obs_positive():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    assert result.cl_obs_L_per_h > 0.0


def test_cl_obs_analytical():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE, route="iv")
    expected_cl = _DOSE / (_C0 / _KE)
    assert abs(result.cl_obs_L_per_h - expected_cl) / expected_cl < 0.15


def test_bioavailability_ratio_1_without_iv_reference():
    result = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE, route="oral")
    assert result.bioavailability_ratio == 1.0


def test_bioavailability_ratio_with_iv_reference():
    iv_auc = _C0 / _KE
    result = nca_enhanced(
        "DrugA",
        _TIMES,
        _CONCS,
        _DOSE,
        route="oral",
        iv_auc_for_bioavailability=iv_auc,
        iv_dose_for_bioavailability=_DOSE,
    )
    assert 0.0 < result.bioavailability_ratio <= 2.0


def test_mismatched_length_raises():
    with pytest.raises(ValueError, match="same length"):
        nca_enhanced("D", [0.0, 1.0, 2.0], [1.0, 0.5], 100.0)


def test_too_few_timepoints_raises():
    with pytest.raises(ValueError, match="At least 2"):
        nca_enhanced("D", [0.0], [1.0], 100.0)


def test_non_increasing_times_raises():
    with pytest.raises(ValueError, match="strictly increasing"):
        nca_enhanced("D", [0.0, 2.0, 1.0], [1.0, 0.5, 0.8], 100.0)


def test_batch_nca_returns_list():
    profiles = [
        {
            "compound_name": "DrugA",
            "times_h": _TIMES,
            "concentrations_mg_L": _CONCS,
            "dose_mg": _DOSE,
        },
        {
            "compound_name": "DrugB",
            "times_h": _TIMES,
            "concentrations_mg_L": _CONCS,
            "dose_mg": 200.0,
        },
    ]
    results = batch_nca(profiles)
    assert isinstance(results, list)
    assert len(results) == 2


def test_batch_nca_types():
    profiles = [
        {
            "compound_name": "DrugA",
            "times_h": _TIMES,
            "concentrations_mg_L": _CONCS,
            "dose_mg": _DOSE,
        },
    ]
    results = batch_nca(profiles)
    assert isinstance(results[0], NCAEnhancedResult)


def test_batch_nca_names():
    profiles = [
        {
            "compound_name": "Alpha",
            "times_h": _TIMES,
            "concentrations_mg_L": _CONCS,
            "dose_mg": 50.0,
        },
        {
            "compound_name": "Beta",
            "times_h": _TIMES,
            "concentrations_mg_L": _CONCS,
            "dose_mg": 100.0,
        },
    ]
    results = batch_nca(profiles)
    names = [r.compound_name for r in results]
    assert "Alpha" in names
    assert "Beta" in names


def test_linear_pk_cl_stable():
    r1 = nca_enhanced("DrugA", _TIMES, _CONCS, _DOSE)
    concs_2x = [c * 2 for c in _CONCS]
    r2 = nca_enhanced("DrugB", _TIMES, concs_2x, _DOSE * 2)
    assert abs(r1.cl_obs_L_per_h - r2.cl_obs_L_per_h) / r1.cl_obs_L_per_h < 0.01
