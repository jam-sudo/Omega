"""Tests for Phase 1063 — Lymph Node Delivery Optimization."""

import math

import pytest

from omega_pbpk.clinical.lymph_node_delivery_optimization import (
    LymphDeliveryOptimizationResult,
    optimize_lymph_node_delivery,
)

# ---------------------------------------------------------------------------
# Basic return type and field sanity
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = optimize_lymph_node_delivery("DrugA", mw_Da=500.0, logp=2.0)
    assert isinstance(result, LymphDeliveryOptimizationResult)


def test_result_is_frozen():
    result = optimize_lymph_node_delivery("DrugA", mw_Da=500.0, logp=2.0)
    with pytest.raises((AttributeError, TypeError)):
        result.f_lymph = 0.5  # type: ignore[misc]


def test_f_lymph_in_range():
    result = optimize_lymph_node_delivery("DrugA", mw_Da=500.0, logp=2.0)
    assert 0.0 < result.f_lymph <= 0.85


def test_targeting_score_in_range():
    result = optimize_lymph_node_delivery("DrugA", mw_Da=500.0, logp=2.0)
    assert 0.0 <= result.targeting_score <= 100.0


def test_lymph_node_auc_ratio_positive():
    result = optimize_lymph_node_delivery("DrugA", mw_Da=500.0, logp=2.0)
    assert result.lymph_node_auc_ratio > 0.0


def test_systemic_exposure_reduction_positive():
    result = optimize_lymph_node_delivery("DrugA", mw_Da=500.0, logp=2.0)
    assert result.systemic_exposure_reduction > 0.0


def test_recommended_flag_matches_targeting_score():
    for route in ["SC", "IM", "intradermal"]:
        for ftype in ["solution", "nanoparticle", "liposome", "emulsion", "depot"]:
            result = optimize_lymph_node_delivery(
                "DrugX",
                mw_Da=400.0,
                logp=1.5,
                formulation_type=ftype,
                injection_route=route,
            )
            assert result.recommended == (result.targeting_score >= 70.0)


def test_formulation_score_positive():
    result = optimize_lymph_node_delivery("DrugA", mw_Da=500.0, logp=2.0)
    assert result.formulation_score > 0.0


def test_notes_nonempty():
    result = optimize_lymph_node_delivery("DrugA", mw_Da=500.0, logp=2.0)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Comparative behaviour
# ---------------------------------------------------------------------------


def test_nanoparticle_higher_f_lymph_than_solution():
    base_kwargs = dict(drug_name="DrugB", mw_Da=300.0, logp=1.0, injection_route="SC")
    result_np = optimize_lymph_node_delivery(**base_kwargs, formulation_type="nanoparticle")
    result_sol = optimize_lymph_node_delivery(**base_kwargs, formulation_type="solution")
    assert result_np.f_lymph > result_sol.f_lymph


def test_intradermal_higher_f_lymph_than_im():
    base_kwargs = dict(drug_name="DrugC", mw_Da=400.0, logp=1.5, formulation_type="nanoparticle")
    result_id = optimize_lymph_node_delivery(**base_kwargs, injection_route="intradermal")
    result_im = optimize_lymph_node_delivery(**base_kwargs, injection_route="IM")
    assert result_id.f_lymph > result_im.f_lymph


def test_peg_coated_higher_f_lymph():
    base_kwargs = dict(
        drug_name="DrugD",
        mw_Da=500.0,
        logp=2.0,
        formulation_type="nanoparticle",
        injection_route="SC",
    )
    result_peg = optimize_lymph_node_delivery(**base_kwargs, peg_coated=True)
    result_no_peg = optimize_lymph_node_delivery(**base_kwargs, peg_coated=False)
    assert result_peg.f_lymph > result_no_peg.f_lymph


def test_optimal_injection_route_nanoparticle_is_intradermal():
    result = optimize_lymph_node_delivery(
        "NanoD", mw_Da=1000.0, logp=0.5, drug_class="nanoparticle"
    )
    assert result.optimal_injection_route == "intradermal"


def test_optimal_injection_route_small_molecule_is_sc():
    result = optimize_lymph_node_delivery(
        "SmallD", mw_Da=400.0, logp=2.0, drug_class="small_molecule"
    )
    assert result.optimal_injection_route == "SC"


def test_particle_size_50nm_gives_high_size_score():
    """50 nm is the optimal size; f_lymph should be high."""
    result = optimize_lymph_node_delivery(
        "NP50",
        mw_Da=5000.0,
        logp=0.0,
        formulation_type="nanoparticle",
        injection_route="intradermal",
        particle_size_nm=50.0,
    )
    # size_score = 100, so f_lymph should be near max
    assert result.f_lymph > 0.5


def test_recommended_particle_size_small_molecule():
    result = optimize_lymph_node_delivery("SM", mw_Da=500.0, logp=1.0, drug_class="small_molecule")
    assert math.isclose(result.recommended_particle_size, max(1.0, 500.0 / 500.0), rel_tol=1e-6)


def test_recommended_particle_size_nanoparticle():
    result = optimize_lymph_node_delivery(
        "NP",
        mw_Da=5000.0,
        logp=0.0,
        drug_class="nanoparticle",
        formulation_type="nanoparticle",
    )
    assert math.isclose(result.recommended_particle_size, 30.0, rel_tol=1e-6)


def test_recommended_particle_size_peptide():
    result = optimize_lymph_node_delivery("Pep", mw_Da=2000.0, logp=-1.0, drug_class="peptide")
    assert math.isclose(result.recommended_particle_size, 5.0, rel_tol=1e-6)


def test_f_lymph_max_cap():
    """Even with best settings, f_lymph <= 0.85."""
    result = optimize_lymph_node_delivery(
        "MaxDrug",
        mw_Da=50000.0,
        logp=0.0,
        drug_class="nanoparticle",
        formulation_type="nanoparticle",
        particle_size_nm=50.0,
        injection_route="intradermal",
        peg_coated=True,
    )
    assert result.f_lymph <= 0.85


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        optimize_lymph_node_delivery("DrugX", mw_Da=0.0, logp=1.0)


def test_invalid_drug_class_raises():
    with pytest.raises(ValueError, match="drug_class"):
        optimize_lymph_node_delivery("DrugX", mw_Da=400.0, logp=1.0, drug_class="alien")


def test_invalid_formulation_raises():
    with pytest.raises(ValueError, match="formulation_type"):
        optimize_lymph_node_delivery("DrugX", mw_Da=400.0, logp=1.0, formulation_type="magic")


def test_invalid_route_raises():
    with pytest.raises(ValueError, match="injection_route"):
        optimize_lymph_node_delivery("DrugX", mw_Da=400.0, logp=1.0, injection_route="oral")


def test_negative_particle_size_raises():
    with pytest.raises(ValueError, match="particle_size_nm"):
        optimize_lymph_node_delivery("DrugX", mw_Da=400.0, logp=1.0, particle_size_nm=-5.0)
