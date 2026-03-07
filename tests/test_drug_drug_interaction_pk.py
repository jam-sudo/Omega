"""Tests for Phase 275: dynamic DDI PK simulation."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.drug_drug_interaction_pk import DDIPKResult, simulate_ddi_pk

# --- shared valid kwargs for convenience ---
BASE = dict(
    victim_name="MidazolAm",
    dose_victim_mg=5.0,
    cl_victim_L_per_h=20.0,
    vd_victim_L=50.0,
    inhibitor_name="Ketoconazole",
    dose_inhibitor_mg=200.0,
    cl_inhibitor_L_per_h=5.0,
    vd_inhibitor_L=30.0,
    ki_mg_L=0.01,
    fm_cyp=1.0,
    inhibition_type="competitive",
    route="oral",
    t_end_h=24.0,
    dt_h=0.1,
)


# ── Input validation ──────────────────────────────────────────────────────────

def test_dose_victim_zero():
    with pytest.raises(ValueError, match="dose_victim_mg"):
        simulate_ddi_pk(**{**BASE, "dose_victim_mg": 0.0})


def test_dose_victim_negative():
    with pytest.raises(ValueError, match="dose_victim_mg"):
        simulate_ddi_pk(**{**BASE, "dose_victim_mg": -1.0})


def test_cl_victim_zero():
    with pytest.raises(ValueError, match="cl_victim_L_per_h"):
        simulate_ddi_pk(**{**BASE, "cl_victim_L_per_h": 0.0})


def test_vd_victim_zero():
    with pytest.raises(ValueError, match="vd_victim_L"):
        simulate_ddi_pk(**{**BASE, "vd_victim_L": 0.0})


def test_dose_inhibitor_zero():
    with pytest.raises(ValueError, match="dose_inhibitor_mg"):
        simulate_ddi_pk(**{**BASE, "dose_inhibitor_mg": 0.0})


def test_cl_inhibitor_zero():
    with pytest.raises(ValueError, match="cl_inhibitor_L_per_h"):
        simulate_ddi_pk(**{**BASE, "cl_inhibitor_L_per_h": 0.0})


def test_vd_inhibitor_zero():
    with pytest.raises(ValueError, match="vd_inhibitor_L"):
        simulate_ddi_pk(**{**BASE, "vd_inhibitor_L": 0.0})


def test_ki_zero():
    with pytest.raises(ValueError, match="ki_mg_L"):
        simulate_ddi_pk(**{**BASE, "ki_mg_L": 0.0})


def test_kdeg_zero():
    with pytest.raises(ValueError, match="kdeg_per_h"):
        simulate_ddi_pk(**{**BASE, "kdeg_per_h": 0.0})


def test_fm_cyp_zero():
    with pytest.raises(ValueError, match="fm_cyp"):
        simulate_ddi_pk(**{**BASE, "fm_cyp": 0.0})


def test_fm_cyp_above_one():
    with pytest.raises(ValueError, match="fm_cyp"):
        simulate_ddi_pk(**{**BASE, "fm_cyp": 1.5})


def test_invalid_inhibition_type():
    with pytest.raises(ValueError, match="inhibition_type"):
        simulate_ddi_pk(**{**BASE, "inhibition_type": "allosteric"})


def test_invalid_route():
    with pytest.raises(ValueError, match="route"):
        simulate_ddi_pk(**{**BASE, "route": "inhalation"})


# ── No inhibition effect ──────────────────────────────────────────────────────

def test_very_large_ki_aucr_near_one():
    """When Ki >> inhibitor concentration, virtually no inhibition."""
    result = simulate_ddi_pk(**{**BASE, "ki_mg_L": 1e9})
    assert abs(result.aucr - 1.0) < 0.05


# ── Competitive inhibition ────────────────────────────────────────────────────

def test_competitive_aucr_greater_than_one():
    """Competitive inhibitor reduces victim CL → AUC increases."""
    result = simulate_ddi_pk(**BASE)
    assert result.aucr > 1.0


def test_auc_victim_ddi_positive():
    result = simulate_ddi_pk(**BASE)
    assert result.auc_victim_ddi > 0.0


def test_cmax_victim_positive():
    result = simulate_ddi_pk(**BASE)
    assert result.cmax_victim > 0.0


def test_array_lengths_match():
    result = simulate_ddi_pk(**BASE)
    assert len(result.c_victim_mg_L) == len(result.times_h)
    assert len(result.c_inhibitor_mg_L) == len(result.times_h)
    assert len(result.cl_inhibited) == len(result.times_h)


def test_names_stored():
    result = simulate_ddi_pk(**BASE)
    assert result.victim_name == "MidazolAm"
    assert result.inhibitor_name == "Ketoconazole"


def test_inhibition_type_stored():
    result = simulate_ddi_pk(**BASE)
    assert result.inhibition_type == "competitive"


# ── MBI inhibition ────────────────────────────────────────────────────────────

def test_mbi_inhibition_type_stored():
    result = simulate_ddi_pk(
        **{**BASE, "inhibition_type": "mbi", "kinact_per_h": 0.5, "ki_mg_L": 0.01}
    )
    assert result.inhibition_type == "mbi"


def test_mbi_aucr_greater_than_one():
    result = simulate_ddi_pk(
        **{**BASE, "inhibition_type": "mbi", "kinact_per_h": 0.5, "ki_mg_L": 0.01}
    )
    assert result.aucr > 1.0


# ── fm_cyp sensitivity ────────────────────────────────────────────────────────

def test_small_fm_cyp_minimal_ddi():
    """Very small fm_cyp → DDI is minimal (AUCR close to 1)."""
    result = simulate_ddi_pk(**{**BASE, "fm_cyp": 0.01})
    assert result.aucr < 1.1


def test_full_fm_cyp_maximal_ddi():
    """fm_cyp=1.0 → full DDI effect; AUCR should be substantially > 1."""
    result = simulate_ddi_pk(**{**BASE, "fm_cyp": 1.0, "ki_mg_L": 0.01})
    assert result.aucr > 2.0


# ── IV route ─────────────────────────────────────────────────────────────────

def test_iv_route_works():
    result = simulate_ddi_pk(**{**BASE, "route": "iv"})
    assert isinstance(result, DDIPKResult)
    assert result.auc_victim_ddi > 0.0
