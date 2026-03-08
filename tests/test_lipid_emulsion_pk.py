"""Tests for Phase 470 — Intravenous Lipid Emulsion PK."""

from __future__ import annotations

import pytest

from omega_pbpk.core.lipid_emulsion_pk import (
    LipidEmulsionPKResult,
    screen_ile_doses,
    simulate_lipid_emulsion_rescue,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_EFFICACY = {"low", "moderate", "high"}


def _make(
    drug="TestDrug",
    dose_mg=100.0,
    logP=3.0,
    cl=5.0,
    vd=50.0,
    ile_mL=100.0,
    ile_frac=0.20,
    t_end=6.0,
    dt=0.05,
):
    return simulate_lipid_emulsion_rescue(
        drug_name=drug,
        dose_mg=dose_mg,
        logP=logP,
        cl_sys_L_per_h=cl,
        vd_sys_L=vd,
        ile_dose_mL=ile_mL,
        ile_lipid_fraction=ile_frac,
        t_end_h=t_end,
        dt_h=dt,
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = _make()
    assert isinstance(result, LipidEmulsionPKResult)


def test_lists_populated():
    result = _make()
    assert len(result.times_h) > 1
    assert len(result.c_total_mg_L) == len(result.times_h)
    assert len(result.c_free_mg_L) == len(result.times_h)


def test_times_start_at_zero():
    result = _make()
    assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Lipid sink effect
# ---------------------------------------------------------------------------


def test_f_free_less_than_one_for_lipophilic():
    """Lipophilic drug (logP=3) should have f_free < 1."""
    result = _make(logP=3.0)
    assert result.f_free < 1.0


def test_f_free_approximately_one_for_hydrophilic():
    """Very hydrophilic drug (logP=-3) → Kp_ile clamped to 1 → minimal sequestration."""
    result = _make(logP=-3.0)
    # kp_ile = 10^(0.7*-3) = 10^-2.1 ≈ 0.008, clamped to 1
    # Vd_eff = 50 + 0.02*1 = 50.02 → f_free ≈ 0.9996
    assert result.f_free > 0.99


def test_higher_logP_lower_f_free():
    """Higher logP → stronger lipid sink → lower free fraction."""
    r_low = _make(logP=1.0)
    r_high = _make(logP=5.0)
    assert r_high.f_free < r_low.f_free


def test_higher_ile_dose_higher_sequestration():
    """More ILE → more lipid volume → more sequestration."""
    r_low = _make(ile_mL=50.0)
    r_high = _make(ile_mL=500.0)
    assert r_high.sequestration_pct > r_low.sequestration_pct


def test_sequestration_pct_in_range():
    result = _make()
    assert 0.0 <= result.sequestration_pct <= 100.0


def test_f_free_plus_sequestration_pct_100():
    """f_free + sequestration_pct/100 should equal 1."""
    result = _make(logP=4.0)
    assert result.f_free + result.sequestration_pct / 100.0 == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# cmax_free
# ---------------------------------------------------------------------------


def test_cmax_free_is_initial_free():
    """cmax_free = c_free at t=0 (first element)."""
    result = _make()
    assert result.cmax_free_mg_L == pytest.approx(result.c_free_mg_L[0], rel=1e-6)


def test_cmax_free_less_than_dose_over_vd():
    """Sequestration should reduce free Cmax below dose/Vd_sys."""
    result = _make(logP=3.0, dose_mg=100.0, vd=50.0)
    naive_c0 = 100.0 / 50.0  # 2.0 mg/L
    assert result.cmax_free_mg_L < naive_c0


# ---------------------------------------------------------------------------
# Kp_ile clamping
# ---------------------------------------------------------------------------


def test_kp_ile_min_clamped():
    """Very hydrophilic drug → Kp_ile clamped to 1."""
    result = _make(logP=-10.0)
    assert result.kp_ile == pytest.approx(1.0)


def test_kp_ile_max_clamped():
    """Very lipophilic drug → Kp_ile clamped to 1000."""
    result = _make(logP=20.0)
    assert result.kp_ile == pytest.approx(1000.0)


def test_kp_ile_formula():
    """kp_ile = 10^(0.7 * logP) for moderate logP."""
    logP = 2.0
    expected = 10.0 ** (0.7 * logP)  # 10^1.4 ≈ 25.12
    result = _make(logP=logP)
    assert result.kp_ile == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def test_auc_positive():
    result = _make()
    assert result.auc_free_mg_h_per_L > 0.0


def test_auc_decreases_with_more_ile():
    """More ILE → lower free drug → lower AUC_free."""
    r_low = _make(ile_mL=50.0)
    r_high = _make(ile_mL=500.0)
    assert r_high.auc_free_mg_h_per_L < r_low.auc_free_mg_h_per_L


# ---------------------------------------------------------------------------
# Rescue efficacy classification
# ---------------------------------------------------------------------------


def test_rescue_efficacy_valid_string():
    result = _make()
    assert result.rescue_efficacy in _VALID_EFFICACY


def test_rescue_efficacy_low():
    """Hydrophilic drug with small ILE → low sequestration → low efficacy."""
    result = _make(logP=-2.0, ile_mL=10.0)
    assert result.sequestration_pct < 20.0
    assert result.rescue_efficacy == "low"


def test_rescue_efficacy_high():
    """Highly lipophilic drug with large ILE → high sequestration → high efficacy."""
    result = _make(logP=5.0, ile_mL=500.0)
    assert result.sequestration_pct > 60.0
    assert result.rescue_efficacy == "high"


# ---------------------------------------------------------------------------
# screen_ile_doses
# ---------------------------------------------------------------------------


def test_screen_ile_doses_default_length():
    results = screen_ile_doses("Drug", 100.0, 3.0, 5.0)
    assert len(results) == 4


def test_screen_ile_doses_sorted_descending():
    results = screen_ile_doses("Drug", 100.0, 3.0, 5.0)
    seq_pcts = [r.sequestration_pct for r in results]
    assert seq_pcts == sorted(seq_pcts, reverse=True)


def test_screen_ile_doses_custom_doses():
    results = screen_ile_doses("Drug", 100.0, 2.0, 5.0, ile_doses_mL=[100.0, 200.0])
    assert len(results) == 2


def test_screen_ile_doses_all_instances():
    results = screen_ile_doses("Drug", 100.0, 3.0, 5.0)
    for r in results:
        assert isinstance(r, LipidEmulsionPKResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_lipid_emulsion_rescue("Drug", 0.0, 3.0, 5.0)


def test_invalid_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_lipid_emulsion_rescue("Drug", -50.0, 3.0, 5.0)


def test_invalid_cl_zero():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_lipid_emulsion_rescue("Drug", 100.0, 3.0, 0.0)


def test_invalid_cl_negative():
    with pytest.raises(ValueError, match="cl_sys_L_per_h"):
        simulate_lipid_emulsion_rescue("Drug", 100.0, 3.0, -1.0)


def test_invalid_vd_zero():
    with pytest.raises(ValueError, match="vd_sys_L"):
        simulate_lipid_emulsion_rescue("Drug", 100.0, 3.0, 5.0, vd_sys_L=0.0)


def test_invalid_ile_dose_zero():
    with pytest.raises(ValueError, match="ile_dose_mL"):
        simulate_lipid_emulsion_rescue("Drug", 100.0, 3.0, 5.0, ile_dose_mL=0.0)


def test_invalid_ile_fraction_zero():
    with pytest.raises(ValueError, match="ile_lipid_fraction"):
        simulate_lipid_emulsion_rescue("Drug", 100.0, 3.0, 5.0, ile_lipid_fraction=0.0)


def test_invalid_ile_fraction_gt_one():
    with pytest.raises(ValueError, match="ile_lipid_fraction"):
        simulate_lipid_emulsion_rescue("Drug", 100.0, 3.0, 5.0, ile_lipid_fraction=1.5)
