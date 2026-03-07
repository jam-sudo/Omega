"""Tests for core/physiological_flow.py — Phase 597."""

import pytest

from omega_pbpk.core.physiological_flow import (
    PhysiologicalParameters,
    compute_organ_clearance,
    get_human_parameters,
    get_mouse_parameters,
    get_rat_parameters,
    scale_to_body_weight,
)


def test_human_returns_result():
    params = get_human_parameters()
    assert isinstance(params, PhysiologicalParameters)


def test_human_co_positive():
    params = get_human_parameters()
    assert params.cardiac_output_L_per_h > 0


def test_human_organ_flows_sum_to_1():
    params = get_human_parameters()
    non_lung = {k: v for k, v in params.organ_flows.items() if k != "lung"}
    total = sum(non_lung.values())
    assert abs(total - 1.0) < 0.05


def test_human_plasma_volume_positive():
    params = get_human_parameters()
    assert params.plasma_volume_L > 0


def test_rat_returns_result():
    params = get_rat_parameters()
    assert isinstance(params, PhysiologicalParameters)


def test_mouse_returns_result():
    params = get_mouse_parameters()
    assert isinstance(params, PhysiologicalParameters)


def test_scale_increases_co_for_heavier():
    base = get_human_parameters(70.0)
    heavy = scale_to_body_weight(base, 100.0)
    assert heavy.cardiac_output_L_per_h > base.cardiac_output_L_per_h


def test_scale_decreases_volume_for_lighter():
    base = get_human_parameters(70.0)
    light = scale_to_body_weight(base, 50.0)
    assert light.organ_volumes_L["muscle"] < base.organ_volumes_L["muscle"]


def test_scale_preserves_organ_flow_fractions():
    base = get_human_parameters(70.0)
    scaled = scale_to_body_weight(base, 100.0)
    for organ in base.organ_flows:
        assert abs(scaled.organ_flows[organ] - base.organ_flows[organ]) < 1e-9


def test_compute_organ_cl_well_stirred_positive():
    cl = compute_organ_clearance("liver", 100.0, 0.5, 81.0, model="well_stirred")
    assert cl > 0


def test_compute_organ_cl_parallel_tube_positive():
    cl = compute_organ_clearance("liver", 100.0, 0.5, 81.0, model="parallel_tube")
    assert cl > 0


def test_compute_organ_cl_max_is_blood_flow():
    blood_flow_L_per_h = 81.0
    q_mL_per_min = blood_flow_L_per_h * 1000.0 / 60.0
    cl = compute_organ_clearance("liver", 1e9, 1.0, blood_flow_L_per_h, model="well_stirred")
    assert cl <= q_mL_per_min + 1e-3


def test_well_stirred_vs_parallel_tube_ordering():
    # Parallel-tube model gives >= clearance compared to well-stirred
    q = 81.0
    clint = 50.0
    fu = 0.5
    cl_ws = compute_organ_clearance("liver", clint, fu, q, model="well_stirred")
    cl_pt = compute_organ_clearance("liver", clint, fu, q, model="parallel_tube")
    assert cl_pt >= cl_ws - 1e-9


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        get_human_parameters(body_weight_kg=0)
    with pytest.raises(ValueError):
        get_human_parameters(body_weight_kg=-10)


def test_invalid_new_weight_raises():
    base = get_human_parameters()
    with pytest.raises(ValueError):
        scale_to_body_weight(base, 0.0)
    with pytest.raises(ValueError):
        scale_to_body_weight(base, -5.0)


def test_invalid_fu_raises():
    with pytest.raises(ValueError):
        compute_organ_clearance("liver", 100.0, 1.5, 81.0)
    with pytest.raises(ValueError):
        compute_organ_clearance("liver", 100.0, 0.0, 81.0)


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        compute_organ_clearance("liver", 100.0, 0.5, 81.0, model="unknown_model")


def test_human_hematocrit_in_range():
    params = get_human_parameters()
    assert 0.3 <= params.hematocrit <= 0.6


def test_liver_flow_fraction():
    params = get_human_parameters()
    assert abs(params.organ_flows["liver"] - 0.215) <= 0.05


def test_kidney_flow_fraction():
    params = get_human_parameters()
    assert abs(params.organ_flows["kidney"] - 0.194) <= 0.05


def test_total_blood_volume_gt_plasma():
    params = get_human_parameters()
    assert params.total_blood_volume_L > params.plasma_volume_L


def test_species_string_human():
    params = get_human_parameters()
    assert params.species == "human"


def test_species_string_rat():
    params = get_rat_parameters()
    assert params.species == "rat"


def test_species_string_mouse():
    params = get_mouse_parameters()
    assert params.species == "mouse"


def test_notes_nonempty():
    params = get_human_parameters()
    assert len(params.notes) > 0
