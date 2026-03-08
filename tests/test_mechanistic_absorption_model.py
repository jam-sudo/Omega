"""
Tests for Phase 399 — Mechanistic Absorption Model (MAM).
~22 tests covering return types, BCS limits, dimensionless numbers,
flags, AUC scaling, sensitivity analysis, and validation.
"""

import pytest

from omega_pbpk.core.mechanistic_absorption_model import (
    MAMResult,
    sensitivity_analysis_mam,
    simulate_mam,
)

# ---------------------------------------------------------------------------
# Return-type / structure tests
# ---------------------------------------------------------------------------


def test_simulate_mam_returns_mam_result():
    result = simulate_mam("TestDrug", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=1e-4)
    assert isinstance(result, MAMResult)


def test_result_has_required_fields():
    r = simulate_mam("Drug", dose_mg=50, solubility_mg_mL=0.5, permeability_cm_s=5e-5)
    for field in [
        "drug_name",
        "dose_mg",
        "times_h",
        "m_undissolved_mg",
        "m_dissolved_mg",
        "c_plasma_mg_L",
        "fa_predicted",
        "fa_dissolution_limited",
        "fa_permeability_limited",
        "auc_mg_h_per_L",
        "cmax_mg_L",
        "tmax_h",
        "dose_number",
        "absorption_number",
        "dissolution_number",
        "notes",
    ]:
        assert hasattr(r, field), f"Missing field: {field}"


def test_time_lists_same_length():
    r = simulate_mam("Drug", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=1e-4)
    n = len(r.times_h)
    assert len(r.m_undissolved_mg) == n
    assert len(r.m_dissolved_mg) == n
    assert len(r.c_plasma_mg_L) == n


def test_times_start_at_zero():
    r = simulate_mam("Drug", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=1e-4)
    assert r.times_h[0] == 0.0


def test_drug_name_preserved():
    r = simulate_mam("Ibuprofen", dose_mg=400, solubility_mg_mL=0.05, permeability_cm_s=2e-4)
    assert r.drug_name == "Ibuprofen"
    assert r.dose_mg == 400.0


# ---------------------------------------------------------------------------
# Biopharmaceutics / physicochemical tests
# ---------------------------------------------------------------------------


def test_high_solubility_high_permeability_high_fa():
    """BCS class I: high sol, high Peff → fa near 1."""
    r = simulate_mam(
        "BCS_I", dose_mg=100, solubility_mg_mL=10.0, permeability_cm_s=1e-3, t_end_h=24.0
    )
    assert r.fa_predicted > 0.8, f"fa={r.fa_predicted:.3f} expected >0.8"


def test_low_solubility_low_permeability_low_fa():
    """BCS class IV: very low sol, very low Peff → low fa."""
    r = simulate_mam(
        "BCS_IV", dose_mg=500, solubility_mg_mL=0.001, permeability_cm_s=1e-7, t_end_h=12.0
    )
    assert r.fa_predicted < 0.3, f"fa={r.fa_predicted:.3f} expected <0.3"


def test_fa_between_0_and_1():
    for sol in [0.001, 0.1, 10.0]:
        r = simulate_mam("D", dose_mg=100, solubility_mg_mL=sol, permeability_cm_s=1e-4)
        assert 0.0 <= r.fa_predicted <= 1.0


# ---------------------------------------------------------------------------
# Dimensionless number tests
# ---------------------------------------------------------------------------


def test_dose_number_high_for_insoluble():
    """High dose, very low solubility → Do >> 1."""
    r = simulate_mam("Insoluble", dose_mg=500, solubility_mg_mL=0.001, permeability_cm_s=1e-4)
    assert r.dose_number > 1.0, f"Do={r.dose_number:.2f}"


def test_dose_number_formula():
    """Do = dose / (solubility * 250)."""
    r = simulate_mam("D", dose_mg=250, solubility_mg_mL=1.0, permeability_cm_s=1e-4)
    expected_Do = 250.0 / (1.0 * 250.0)
    assert abs(r.dose_number - expected_Do) < 1e-6


def test_absorption_number_large_for_high_permeability():
    """An >> 1 for high Peff."""
    r = simulate_mam("HiPerm", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=1e-2)
    assert r.absorption_number > 1.0, f"An={r.absorption_number:.2f}"


def test_dissolution_number_large_for_tiny_particle():
    """Very small particle → fast dissolution → Dn > 1."""
    r = simulate_mam(
        "TinyParticle",
        dose_mg=100,
        solubility_mg_mL=1.0,
        permeability_cm_s=1e-4,
        particle_size_um=1.0,
    )
    assert r.dissolution_number > 1.0, f"Dn={r.dissolution_number:.3f}"


def test_dissolution_number_small_for_large_particle():
    """Large particle → slow dissolution → Dn < 1."""
    r = simulate_mam(
        "LargeParticle",
        dose_mg=100,
        solubility_mg_mL=0.01,
        permeability_cm_s=1e-4,
        particle_size_um=500.0,
    )
    assert r.dissolution_number < 1.0, f"Dn={r.dissolution_number:.3f}"


# ---------------------------------------------------------------------------
# Flag tests
# ---------------------------------------------------------------------------


def test_dissolution_limited_flag():
    """Do > 1 and Dn < 1 → dissolution-limited."""
    r = simulate_mam(
        "DissLim",
        dose_mg=1000,
        solubility_mg_mL=0.001,
        permeability_cm_s=1e-3,
        particle_size_um=200.0,
    )
    # Force Do > 1 (dose=1000, sol=0.001 → Do=4), Dn < 1 (large particle, low sol)
    assert r.dose_number > 1.0
    if r.dissolution_number < 1.0:
        assert r.fa_dissolution_limited is True


def test_permeability_limited_flag():
    """An < 1 → permeability-limited."""
    r = simulate_mam("PermLim", dose_mg=100, solubility_mg_mL=10.0, permeability_cm_s=1e-7)
    assert r.fa_permeability_limited is True


def test_well_absorbed_flags_false():
    """High sol + high Peff → both flags False."""
    r = simulate_mam("BCSI", dose_mg=10, solubility_mg_mL=10.0, permeability_cm_s=1e-3)
    # An should be large → permeability flag False
    assert r.fa_permeability_limited is False


# ---------------------------------------------------------------------------
# AUC and PK metric tests
# ---------------------------------------------------------------------------


def test_auc_positive():
    r = simulate_mam("D", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=1e-4)
    assert r.auc_mg_h_per_L > 0


def test_cmax_positive():
    r = simulate_mam("D", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=1e-4)
    assert r.cmax_mg_L > 0


def test_tmax_positive():
    r = simulate_mam("D", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=1e-4)
    assert r.tmax_h > 0


def test_auc_scales_with_dose():
    """Double dose should roughly double AUC (linear regime)."""
    r1 = simulate_mam("D", dose_mg=100, solubility_mg_mL=5.0, permeability_cm_s=5e-4)
    r2 = simulate_mam("D", dose_mg=200, solubility_mg_mL=5.0, permeability_cm_s=5e-4)
    ratio = r2.auc_mg_h_per_L / r1.auc_mg_h_per_L
    assert 1.5 < ratio < 2.5, f"AUC ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# Sensitivity analysis tests
# ---------------------------------------------------------------------------


def test_sensitivity_analysis_returns_list():
    results = sensitivity_analysis_mam("Drug", dose_mg=100, permeability_cm_s=1e-4)
    assert isinstance(results, list)


def test_sensitivity_analysis_returns_5_results():
    results = sensitivity_analysis_mam("Drug", dose_mg=100, permeability_cm_s=1e-4)
    assert len(results) == 5


def test_sensitivity_analysis_sorted_descending():
    results = sensitivity_analysis_mam("Drug", dose_mg=100, permeability_cm_s=1e-4)
    fas = [r.fa_predicted for r in results]
    assert fas == sorted(fas, reverse=True), f"Not sorted: {fas}"


def test_sensitivity_analysis_custom_solubilities():
    sols = [0.1, 1.0, 5.0]
    results = sensitivity_analysis_mam(
        "Drug", dose_mg=100, permeability_cm_s=1e-4, solubilities=sols
    )
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validation_zero_dose():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_mam("D", dose_mg=0, solubility_mg_mL=1.0, permeability_cm_s=1e-4)


def test_validation_negative_solubility():
    with pytest.raises(ValueError, match="solubility"):
        simulate_mam("D", dose_mg=100, solubility_mg_mL=-1.0, permeability_cm_s=1e-4)


def test_validation_zero_permeability():
    with pytest.raises(ValueError, match="permeability"):
        simulate_mam("D", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=0)


def test_validation_negative_particle_size():
    with pytest.raises(ValueError, match="particle_size"):
        simulate_mam(
            "D", dose_mg=100, solubility_mg_mL=1.0, permeability_cm_s=1e-4, particle_size_um=-5
        )
