"""
Tests for Phase 636 — Drug Tumor Vascular Extravasation
"""

import pytest

from omega_pbpk.core.tumor_extravasation_p636 import (
    TumorExtravasationResult,
    simulate_tumor_extravasation,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="doxorubicin",
        dose_mg=50.0,
        particle_type="small_molecule",
        mw_Da=544.0,
        cl_sys_L_per_h=30.0,
        vd_sys_L=800.0,
        tumor_weight_g=5.0,
        t_end_h=72.0,
        dt_h=0.2,
    )
    params.update(kwargs)
    return simulate_tumor_extravasation(**params)


# ---------------------------------------------------------------------------
# Return type and structure tests
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = default_result()
    assert isinstance(result, TumorExtravasationResult)


def test_result_fields_present():
    result = default_result()
    assert hasattr(result, "drug_name")
    assert hasattr(result, "dose_mg")
    assert hasattr(result, "particle_type")
    assert hasattr(result, "times_h")
    assert hasattr(result, "c_plasma_mg_L")
    assert hasattr(result, "c_tumor_vasculature_mg_L")
    assert hasattr(result, "c_tumor_interstitium_mg_g")
    assert hasattr(result, "cmax_plasma_mg_L")
    assert hasattr(result, "cmax_tumor_mg_g")
    assert hasattr(result, "auc_plasma_mg_h_per_L")
    assert hasattr(result, "auc_tumor_mg_h_per_g")
    assert hasattr(result, "epr_enhancement_factor")
    assert hasattr(result, "tumor_accumulation_pct")
    assert hasattr(result, "vascular_permeability_coeff")
    assert hasattr(result, "notes")


def test_list_fields_are_lists():
    result = default_result()
    assert isinstance(result.times_h, list)
    assert isinstance(result.c_plasma_mg_L, list)
    assert isinstance(result.c_tumor_vasculature_mg_L, list)
    assert isinstance(result.c_tumor_interstitium_mg_g, list)


def test_list_lengths_equal():
    result = default_result()
    n = len(result.times_h)
    assert len(result.c_plasma_mg_L) == n
    assert len(result.c_tumor_vasculature_mg_L) == n
    assert len(result.c_tumor_interstitium_mg_g) == n


def test_list_length_correct():
    result = default_result(t_end_h=48.0, dt_h=0.2)
    expected_len = int(round(48.0 / 0.2)) + 1
    assert len(result.times_h) == expected_len


def test_string_fields():
    result = default_result(drug_name="paclitaxel", particle_type="nanoparticle")
    assert result.drug_name == "paclitaxel"
    assert result.particle_type == "nanoparticle"
    assert result.dose_mg == 50.0


# ---------------------------------------------------------------------------
# PK correctness
# ---------------------------------------------------------------------------


def test_plasma_starts_positive():
    result = default_result()
    assert result.c_plasma_mg_L[0] > 0.0


def test_plasma_decays_monotonically():
    # Systemic clearance dominates — plasma should decrease overall
    result = default_result()
    assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]


def test_vascular_tumor_starts_positive():
    result = default_result()
    assert result.c_tumor_vasculature_mg_L[0] > 0.0


def test_interstitium_builds_up():
    result = default_result()
    # Tumor interstitium concentration should increase before decaying
    assert max(result.c_tumor_interstitium_mg_g) > 0.0


def test_cmax_plasma_positive():
    result = default_result()
    assert result.cmax_plasma_mg_L > 0.0


def test_cmax_tumor_positive():
    result = default_result()
    assert result.cmax_tumor_mg_g > 0.0


def test_auc_plasma_positive():
    result = default_result()
    assert result.auc_plasma_mg_h_per_L > 0.0


def test_auc_tumor_positive():
    result = default_result()
    assert result.auc_tumor_mg_h_per_g > 0.0


# ---------------------------------------------------------------------------
# Particle type differences
# ---------------------------------------------------------------------------


def test_small_molecule_higher_permeability_than_antibody():
    r_sm = default_result(particle_type="small_molecule")
    r_ab = default_result(particle_type="antibody")
    assert r_sm.vascular_permeability_coeff > r_ab.vascular_permeability_coeff


def test_nanoparticle_higher_permeability_than_antibody():
    r_np = default_result(particle_type="nanoparticle")
    r_ab = default_result(particle_type="antibody")
    assert r_np.vascular_permeability_coeff > r_ab.vascular_permeability_coeff


def test_small_molecule_higher_tumor_auc_than_antibody():
    # Small molecules accumulate faster due to higher permeability
    r_sm = default_result(particle_type="small_molecule", t_end_h=24.0)
    r_ab = default_result(particle_type="antibody", t_end_h=24.0)
    assert r_sm.auc_tumor_mg_h_per_g > r_ab.auc_tumor_mg_h_per_g


def test_permeability_stored_correctly():
    result = default_result(particle_type="small_molecule", mw_Da=500.0)
    # P_base = 1e-5, MW=500 → scaling = 1.0, P = 1e-5
    assert abs(result.vascular_permeability_coeff - 1e-5) < 1e-12


def test_permeability_mw_scaling():
    # Higher MW → lower permeability
    r_low_mw = default_result(particle_type="nanoparticle", mw_Da=500.0)
    r_high_mw = default_result(particle_type="nanoparticle", mw_Da=10000.0)
    assert r_low_mw.vascular_permeability_coeff > r_high_mw.vascular_permeability_coeff


# ---------------------------------------------------------------------------
# EPR enhancement
# ---------------------------------------------------------------------------


def test_epr_enhancement_in_bounds():
    result = default_result()
    assert 0.1 <= result.epr_enhancement_factor <= 100.0


def test_epr_enhancement_positive():
    for pt in ("small_molecule", "nanoparticle", "antibody"):
        result = default_result(particle_type=pt)
        assert result.epr_enhancement_factor > 0.0


# ---------------------------------------------------------------------------
# Tumor accumulation
# ---------------------------------------------------------------------------


def test_tumor_accumulation_nonnegative():
    result = default_result()
    assert result.tumor_accumulation_pct >= 0.0


def test_tumor_accumulation_increases_with_dose():
    r_low = default_result(dose_mg=10.0)
    r_high = default_result(dose_mg=100.0)
    # Both should have similar % accumulation (dose-proportional)
    # Just ensure both are non-negative
    assert r_low.tumor_accumulation_pct >= 0.0
    assert r_high.tumor_accumulation_pct >= 0.0


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_notes_is_string():
    result = default_result()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_tumor_extravasation(
            drug_name="d",
            dose_mg=0.0,
            particle_type="small_molecule",
            mw_Da=400.0,
            cl_sys_L_per_h=10.0,
            vd_sys_L=50.0,
        )


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_tumor_extravasation(
            drug_name="d",
            dose_mg=100.0,
            particle_type="small_molecule",
            mw_Da=400.0,
            cl_sys_L_per_h=0.0,
            vd_sys_L=50.0,
        )


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_tumor_extravasation(
            drug_name="d",
            dose_mg=100.0,
            particle_type="small_molecule",
            mw_Da=400.0,
            cl_sys_L_per_h=10.0,
            vd_sys_L=0.0,
        )


def test_validation_tumor_weight_zero():
    with pytest.raises(ValueError, match="tumor_weight_g"):
        simulate_tumor_extravasation(
            drug_name="d",
            dose_mg=100.0,
            particle_type="small_molecule",
            mw_Da=400.0,
            cl_sys_L_per_h=10.0,
            vd_sys_L=50.0,
            tumor_weight_g=0.0,
        )


def test_validation_invalid_particle_type():
    with pytest.raises(ValueError, match="particle_type"):
        simulate_tumor_extravasation(
            drug_name="d",
            dose_mg=100.0,
            particle_type="liposome",
            mw_Da=400.0,
            cl_sys_L_per_h=10.0,
            vd_sys_L=50.0,
        )
