"""Tests for PK parameter sensitivity analysis — Phase 271."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_parameter_sensitivity import (
    PKSensitivityResult,
    compute_pk_sensitivity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> PKSensitivityResult:
    defaults = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ka_per_h=1.5,
        f=0.8,
        route="oral",
        perturbation_pct=10.0,
    )
    defaults.update(kwargs)
    return compute_pk_sensitivity(**defaults)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        compute_pk_sensitivity("Drug", 0.0, 5.0, 50.0)


def test_dose_negative_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        compute_pk_sensitivity("Drug", -1.0, 5.0, 50.0)


def test_cl_zero_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        compute_pk_sensitivity("Drug", 100.0, 0.0, 50.0)


def test_cl_negative_raises():
    with pytest.raises(ValueError, match="cl_L_per_h"):
        compute_pk_sensitivity("Drug", 100.0, -5.0, 50.0)


def test_vd_zero_raises():
    with pytest.raises(ValueError, match="vd_L"):
        compute_pk_sensitivity("Drug", 100.0, 5.0, 0.0)


def test_vd_negative_raises():
    with pytest.raises(ValueError, match="vd_L"):
        compute_pk_sensitivity("Drug", 100.0, 5.0, -10.0)


def test_ka_zero_raises():
    with pytest.raises(ValueError, match="ka_per_h"):
        compute_pk_sensitivity("Drug", 100.0, 5.0, 50.0, ka_per_h=0.0)


def test_f_zero_raises():
    with pytest.raises(ValueError, match="f must be in"):
        compute_pk_sensitivity("Drug", 100.0, 5.0, 50.0, f=0.0)


def test_f_above_one_raises():
    with pytest.raises(ValueError, match="f must be in"):
        compute_pk_sensitivity("Drug", 100.0, 5.0, 50.0, f=1.1)


def test_perturbation_zero_raises():
    with pytest.raises(ValueError, match="perturbation_pct"):
        compute_pk_sensitivity("Drug", 100.0, 5.0, 50.0, perturbation_pct=0.0)


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="route"):
        compute_pk_sensitivity("Drug", 100.0, 5.0, 50.0, route="subcutaneous")


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_coefficient_count():
    result = _default_result()
    # 4 parameters × 3 metrics = 12
    assert len(result.coefficients) == 12


def test_parameter_names():
    result = _default_result()
    params = {c.parameter for c in result.coefficients}
    assert params == {"cl", "vd", "ka", "f"}


def test_metric_names():
    result = _default_result()
    metrics = {c.output_metric for c in result.coefficients}
    assert metrics == {"auc", "cmax", "t_half"}


def test_sensitivity_coefficients_are_finite():
    result = _default_result()
    for c in result.coefficients:
        assert math.isfinite(c.sensitivity_coefficient), (
            f"{c.parameter}/{c.output_metric} SC is not finite"
        )


def test_drug_name_stored():
    result = _default_result(drug_name="Warfarin")
    assert result.drug_name == "Warfarin"


def test_most_sensitive_auc_is_valid_param():
    result = _default_result()
    assert result.most_sensitive_param_auc in {"cl", "vd", "ka", "f"}


def test_most_sensitive_cmax_is_valid_param():
    result = _default_result()
    assert result.most_sensitive_param_cmax in {"cl", "vd", "ka", "f"}


def test_notes_contains_drug_name():
    result = _default_result(drug_name="Aspirin")
    assert "Aspirin" in result.notes


# ---------------------------------------------------------------------------
# Analytical sensitivity correctness
# ---------------------------------------------------------------------------


def test_cl_sensitivity_for_auc_negative():
    """AUC = F*D/CL → CL sensitivity must be negative (AUC decreases as CL increases)."""
    result = _default_result(route="oral")
    cl_auc = next(
        c for c in result.coefficients if c.parameter == "cl" and c.output_metric == "auc"
    )
    assert cl_auc.sensitivity_coefficient < 0.0


def test_cl_sensitivity_for_auc_approx_neg_one():
    """AUC = F*D/CL → forward-diff SC ≈ -1/(1+h). With h=0.10, expected ≈ -0.9091."""
    h = 0.10
    expected = -1.0 / (1.0 + h)
    result = _default_result(route="oral", perturbation_pct=10.0)
    cl_auc = next(
        c for c in result.coefficients if c.parameter == "cl" and c.output_metric == "auc"
    )
    assert abs(cl_auc.sensitivity_coefficient - expected) < 1e-6


def test_f_sensitivity_for_auc_equals_pos_one():
    """AUC = F*D/CL is linear in F → forward-diff SC = exactly +1.0."""
    result = _default_result(route="oral", perturbation_pct=10.0)
    f_auc = next(c for c in result.coefficients if c.parameter == "f" and c.output_metric == "auc")
    assert abs(f_auc.sensitivity_coefficient - 1.0) < 1e-6


def test_vd_sensitivity_for_t_half_positive():
    """t½ = 0.693*Vd/CL → Vd sensitivity must be positive."""
    result = _default_result()
    vd_thalf = next(
        c for c in result.coefficients if c.parameter == "vd" and c.output_metric == "t_half"
    )
    assert vd_thalf.sensitivity_coefficient > 0.0


def test_vd_sensitivity_for_t_half_approx_pos_one():
    """t½ = ln2*Vd/CL is linear in Vd → forward-diff SC = exactly +1.0."""
    result = _default_result(perturbation_pct=10.0)
    vd_thalf = next(
        c for c in result.coefficients if c.parameter == "vd" and c.output_metric == "t_half"
    )
    assert abs(vd_thalf.sensitivity_coefficient - 1.0) < 1e-6


def test_cl_sensitivity_for_t_half_negative():
    """t½ = 0.693*Vd/CL → CL sensitivity must be negative."""
    result = _default_result()
    cl_thalf = next(
        c for c in result.coefficients if c.parameter == "cl" and c.output_metric == "t_half"
    )
    assert cl_thalf.sensitivity_coefficient < 0.0


def test_cl_sensitivity_for_t_half_approx_neg_one():
    """t½ = 0.693*Vd/CL → forward-diff SC ≈ -1/(1+h). With h=0.10, expected ≈ -0.9091."""
    h = 0.10
    expected = -1.0 / (1.0 + h)
    result = _default_result(perturbation_pct=10.0)
    cl_thalf = next(
        c for c in result.coefficients if c.parameter == "cl" and c.output_metric == "t_half"
    )
    assert abs(cl_thalf.sensitivity_coefficient - expected) < 1e-6


def test_ka_sensitivity_for_auc_equals_zero():
    """AUC for oral = F*D/CL, independent of ka → SC = 0."""
    result = _default_result(route="oral")
    ka_auc = next(
        c for c in result.coefficients if c.parameter == "ka" and c.output_metric == "auc"
    )
    assert abs(ka_auc.sensitivity_coefficient) < 1e-6


def test_cl_sensitivity_auc_iv_route_negative():
    """IV route: AUC = D/CL → sensitivity of CL is negative."""
    result = _default_result(route="iv")
    cl_auc = next(
        c for c in result.coefficients if c.parameter == "cl" and c.output_metric == "auc"
    )
    assert cl_auc.sensitivity_coefficient < 0.0


def test_cl_sensitivity_auc_iv_route_approx_neg_one():
    """IV route: AUC = D/CL → forward-diff SC ≈ -1/(1+h) ≈ -0.9091."""
    h = 0.10
    expected = -1.0 / (1.0 + h)
    result = _default_result(route="iv", perturbation_pct=10.0)
    cl_auc = next(
        c for c in result.coefficients if c.parameter == "cl" and c.output_metric == "auc"
    )
    assert abs(cl_auc.sensitivity_coefficient - expected) < 1e-6


# ---------------------------------------------------------------------------
# Perturbation size tests
# ---------------------------------------------------------------------------


def test_small_perturbation_works():
    result = compute_pk_sensitivity("Drug", 100.0, 5.0, 50.0, perturbation_pct=1.0)
    assert len(result.coefficients) == 12


def test_large_perturbation_works():
    result = compute_pk_sensitivity("Drug", 100.0, 5.0, 50.0, perturbation_pct=20.0)
    assert len(result.coefficients) == 12


# ---------------------------------------------------------------------------
# Ranks
# ---------------------------------------------------------------------------


def test_ranks_per_metric_are_1_to_4():
    result = _default_result()
    for metric in ("auc", "cmax", "t_half"):
        ranks = sorted(c.rank for c in result.coefficients if c.output_metric == metric)
        assert ranks == [1, 2, 3, 4]
