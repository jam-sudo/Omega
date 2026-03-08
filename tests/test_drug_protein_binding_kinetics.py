"""Tests for Phase 369 — Drug-Protein Binding Kinetics (kon/koff)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.drug_protein_binding_kinetics import (
    DrugProteinBindingKineticsResult,
    simulate_drug_protein_binding,
)

# ── Basic helpers ──────────────────────────────────────────────────────────────


def _run_default(**kwargs):
    """Run simulation with sensible defaults, accepting overrides."""
    defaults = dict(
        drug_name="TestDrug",
        drug_conc_mg_L=10.0,
        mw_Da=500.0,
        protein_conc_uM=600.0,
        kon_per_uM_per_s=1e-3,
        koff_per_s=1e-3,
        n_sites=1,
        t_end_s=60.0,
    )
    defaults.update(kwargs)
    return simulate_drug_protein_binding(**defaults)


# ── Return type and structure ──────────────────────────────────────────────────


def test_return_type():
    result = _run_default()
    assert isinstance(result, DrugProteinBindingKineticsResult)


def test_return_is_plain_dataclass_not_frozen():
    result = _run_default()
    # Plain dataclass: should be mutable
    result.drug_name = "Modified"
    assert result.drug_name == "Modified"


def test_drug_conc_uM_conversion():
    """drug_uM = drug_conc_mg_L / mw_Da * 1e6 / 1000."""
    result = _run_default(drug_conc_mg_L=10.0, mw_Da=500.0)
    expected_uM = 10.0 / 500.0 * 1e6 / 1000.0  # = 20.0 uM
    assert abs(result.drug_conc_uM - expected_uM) < 1e-9


def test_kd_equals_koff_over_kon():
    result = _run_default(kon_per_uM_per_s=2e-3, koff_per_s=4e-3)
    expected_kd = 4e-3 / 2e-3  # = 2.0 uM
    assert abs(result.kd_uM - expected_kd) < 1e-9


# ── Binding physics ────────────────────────────────────────────────────────────


def test_fu_in_valid_range():
    result = _run_default()
    assert 0.0 < result.fu_at_equilibrium <= 1.0


def test_fu_approaches_1_for_very_weak_binding():
    """Very large Kd (large koff, small kon) → almost no binding → fu near 1."""
    result = _run_default(kon_per_uM_per_s=1e-6, koff_per_s=1.0, t_end_s=200.0)
    assert result.fu_at_equilibrium > 0.95


def test_fu_less_than_1_for_strong_binding():
    """Very small Kd → strong binding → fu well below 1."""
    result = _run_default(kon_per_uM_per_s=1.0, koff_per_s=1e-6, t_end_s=100.0)
    assert result.fu_at_equilibrium < 0.5


def test_mass_conservation_free_plus_bound_equals_total():
    """free_drug[i] + bound_drug[i] ≈ drug_conc_uM for all time points."""
    result = _run_default()
    total = result.drug_conc_uM
    for d, dp in zip(result.free_drug_uM, result.bound_drug_uM):
        assert abs(d + dp - total) < 1e-6, f"Mass conservation violated: {d + dp} != {total}"


def test_free_drug_starts_at_drug_conc_uM():
    result = _run_default()
    assert abs(result.free_drug_uM[0] - result.drug_conc_uM) < 1e-9


def test_bound_drug_starts_at_zero():
    result = _run_default()
    assert abs(result.bound_drug_uM[0]) < 1e-9


def test_free_drug_decreases_over_time():
    """Free drug monotonically decreases as complex forms."""
    result = _run_default(kon_per_uM_per_s=0.1, koff_per_s=0.01, t_end_s=30.0)
    free = result.free_drug_uM
    # Check overall trend: final < initial
    assert free[-1] < free[0], "Free drug should decrease as protein binding occurs"


def test_bound_drug_increases_over_time():
    """Bound drug increases from 0 to equilibrium level."""
    result = _run_default(kon_per_uM_per_s=0.1, koff_per_s=0.01, t_end_s=30.0)
    bound = result.bound_drug_uM
    assert bound[-1] > bound[0], "Bound drug should increase as complex forms"


# ── Half-lives ─────────────────────────────────────────────────────────────────


def test_t_half_binding_positive():
    result = _run_default()
    assert result.t_half_binding_s > 0.0


def test_t_half_dissociation_positive():
    result = _run_default()
    assert result.t_half_dissociation_s > 0.0


def test_t_half_dissociation_equals_ln2_over_koff():
    koff = 5e-3
    result = _run_default(koff_per_s=koff)
    expected = math.log(2.0) / koff
    assert abs(result.t_half_dissociation_s - expected) < 1e-9


def test_t_half_binding_formula():
    """t_half_binding = ln(2) / (kon * protein_conc + koff)."""
    kon = 2e-3
    koff = 1e-3
    prot = 400.0
    result = _run_default(kon_per_uM_per_s=kon, koff_per_s=koff, protein_conc_uM=prot)
    expected = math.log(2.0) / (kon * prot + koff)
    assert abs(result.t_half_binding_s - expected) < 1e-9


# ── Comparative behaviour ──────────────────────────────────────────────────────


def test_higher_kon_reaches_equilibrium_faster():
    """Faster association rate → shorter equilibrium time."""
    r_slow = _run_default(kon_per_uM_per_s=1e-4, t_end_s=500.0)
    r_fast = _run_default(kon_per_uM_per_s=1e-2, t_end_s=500.0)
    assert r_fast.equilibrium_time_s < r_slow.equilibrium_time_s


def test_higher_protein_conc_lowers_fu():
    """More protein binding sites → lower free fraction."""
    r_low = _run_default(protein_conc_uM=100.0, t_end_s=200.0)
    r_high = _run_default(protein_conc_uM=800.0, t_end_s=200.0)
    assert r_high.fu_at_equilibrium < r_low.fu_at_equilibrium


# ── Times array ───────────────────────────────────────────────────────────────


def test_times_start_at_zero():
    result = _run_default()
    assert result.times_s[0] == 0.0


def test_times_end_at_t_end_s():
    result = _run_default(t_end_s=120.0)
    assert abs(result.times_s[-1] - 120.0) < 1e-6


def test_array_lengths_consistent():
    result = _run_default()
    n = len(result.times_s)
    assert len(result.free_drug_uM) == n
    assert len(result.bound_drug_uM) == n


# ── Validation errors ──────────────────────────────────────────────────────────


def test_validation_error_drug_conc_zero():
    with pytest.raises(ValueError, match="drug_conc_mg_L"):
        _run_default(drug_conc_mg_L=0.0)


def test_validation_error_drug_conc_negative():
    with pytest.raises(ValueError, match="drug_conc_mg_L"):
        _run_default(drug_conc_mg_L=-1.0)


def test_validation_error_kon_zero():
    with pytest.raises(ValueError, match="kon"):
        _run_default(kon_per_uM_per_s=0.0)


def test_validation_error_koff_zero():
    with pytest.raises(ValueError, match="koff"):
        _run_default(koff_per_s=0.0)


def test_validation_error_protein_conc_zero():
    with pytest.raises(ValueError, match="protein_conc_uM"):
        _run_default(protein_conc_uM=0.0)


def test_validation_error_mw_zero():
    with pytest.raises(ValueError, match="mw_Da"):
        _run_default(mw_Da=0.0)


# ── Misc ───────────────────────────────────────────────────────────────────────


def test_drug_name_preserved():
    result = _run_default(drug_name="Warfarin")
    assert result.drug_name == "Warfarin"


def test_notes_is_string():
    result = _run_default()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_equilibrium_time_within_simulation():
    result = _run_default(kon_per_uM_per_s=0.1, koff_per_s=0.01, t_end_s=30.0)
    assert 0.0 <= result.equilibrium_time_s <= 30.0 + 1e-6
