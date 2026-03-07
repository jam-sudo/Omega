"""Tests for Phase 1117: Gastric-retentive drug delivery system PK model."""

import math
import pytest

from omega_pbpk.core.gastric_retention_pk import (
    GastricRetentionResult,
    _FORMULATION_PARAMS,
    compare_formulations,
    simulate_gastric_retention_pk,
)


# ---------------------------------------------------------------------------
# Basic result structure
# ---------------------------------------------------------------------------


def test_returns_gastric_retention_result():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    assert isinstance(result, GastricRetentionResult)


def test_result_fields_populated():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    assert result.drug_name == "TestDrug"
    assert result.dose_mg == 100.0
    assert result.formulation_type == "floating"
    assert len(result.times_h) > 0
    assert len(result.c_plasma_mg_L) == len(result.times_h)
    assert len(result.a_gastric_mg) == len(result.times_h)
    assert len(result.a_intestine_mg) == len(result.times_h)


def test_time_starts_at_zero():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    assert result.times_h[0] == pytest.approx(0.0)


def test_c_plasma_starts_at_zero():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    assert result.c_plasma_mg_L[0] == pytest.approx(0.0)


def test_a_gastric_starts_at_dose():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=200.0)
    assert result.a_gastric_mg[0] == pytest.approx(200.0)


def test_a_intestine_starts_at_zero():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    assert result.a_intestine_mg[0] == pytest.approx(0.0)


def test_c_plasma_rises_then_falls():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0, t_end_h=24.0)
    c = result.c_plasma_mg_L
    # Must rise from 0 to some peak and then fall
    peak_idx = c.index(max(c))
    assert peak_idx > 0  # rises
    assert c[-1] < c[peak_idx]  # falls after peak


def test_a_gastric_monotonically_decreasing():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    ag = result.a_gastric_mg
    # Should generally decrease (allow small numeric noise)
    for i in range(1, len(ag)):
        assert ag[i] <= ag[i - 1] + 1e-9


def test_cmax_is_max_of_plasma():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    assert result.cmax_mg_L == pytest.approx(max(result.c_plasma_mg_L))


def test_tmax_corresponds_to_cmax():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    peak_idx = result.c_plasma_mg_L.index(result.cmax_mg_L)
    assert result.tmax_h == pytest.approx(result.times_h[peak_idx])


def test_auc_positive():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    assert result.auc_mg_h_per_L > 0.0


def test_bioavailability_between_0_and_100():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0)
    assert 0.0 <= result.bioavailability_estimate_pct <= 100.0


def test_notes_contains_formulation_type():
    result = simulate_gastric_retention_pk("TestDrug", dose_mg=100.0, formulation_type="floating")
    assert "floating" in result.notes


# ---------------------------------------------------------------------------
# Formulation-specific behaviour
# ---------------------------------------------------------------------------


def test_floating_retention_factor():
    result = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="floating")
    assert result.retention_factor == pytest.approx(_FORMULATION_PARAMS["floating"]["retention_factor"])


def test_bioadhesive_retention_factor():
    result = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="bioadhesive")
    assert result.retention_factor == pytest.approx(_FORMULATION_PARAMS["bioadhesive"]["retention_factor"])


def test_expandable_retention_factor():
    result = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="expandable")
    assert result.retention_factor == pytest.approx(_FORMULATION_PARAMS["expandable"]["retention_factor"])


def test_conventional_retention_factor_zero():
    result = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="conventional")
    assert result.retention_factor == pytest.approx(0.0)


def test_gastric_retentive_higher_auc_than_conventional():
    """Gastric-retentive formulations should have higher AUC than conventional."""
    conventional = simulate_gastric_retention_pk(
        "Drug", dose_mg=100.0, formulation_type="conventional"
    )
    for ft in ["floating", "bioadhesive", "expandable"]:
        retentive = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type=ft)
        assert retentive.auc_mg_h_per_L > conventional.auc_mg_h_per_L, (
            f"{ft} should have higher AUC than conventional"
        )


def test_floating_has_longer_tmax_than_conventional():
    conv = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="conventional")
    floating = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="floating")
    assert floating.tmax_h >= conv.tmax_h


def test_expandable_has_longer_tmax_than_conventional():
    conv = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="conventional")
    expandable = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="expandable")
    assert expandable.tmax_h >= conv.tmax_h


def test_gastric_residence_h_from_params():
    for ft, params in _FORMULATION_PARAMS.items():
        result = simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type=ft)
        assert result.gastric_residence_h == pytest.approx(params["gastric_residence_h"])


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


def test_compare_formulations_returns_four():
    results = compare_formulations("Drug", dose_mg=100.0)
    assert len(results) == 4


def test_compare_formulations_sorted_by_auc_descending():
    results = compare_formulations("Drug", dose_mg=100.0)
    aucs = [r.auc_mg_h_per_L for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_formulations_all_types_present():
    results = compare_formulations("Drug", dose_mg=100.0)
    types = {r.formulation_type for r in results}
    assert types == {"floating", "bioadhesive", "expandable", "conventional"}


def test_compare_formulations_conventional_last():
    results = compare_formulations("Drug", dose_mg=100.0)
    assert results[-1].formulation_type == "conventional"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_gastric_retention_pk("Drug", dose_mg=0.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_gastric_retention_pk("Drug", dose_mg=-10.0)


def test_invalid_cl():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        simulate_gastric_retention_pk("Drug", dose_mg=100.0, cl_L_per_h=0.0)


def test_invalid_vd():
    with pytest.raises(ValueError, match="vd_L"):
        simulate_gastric_retention_pk("Drug", dose_mg=100.0, vd_L=-1.0)


def test_invalid_formulation_type():
    with pytest.raises(ValueError, match="formulation_type"):
        simulate_gastric_retention_pk("Drug", dose_mg=100.0, formulation_type="magic_tablet")
