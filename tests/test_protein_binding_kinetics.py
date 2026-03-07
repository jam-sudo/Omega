"""Tests for protein_binding_kinetics module (Phase 375)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.protein_binding_kinetics import (
    ProteinBindingResult,
    protein_binding_sensitivity,
    simulate_binding_kinetics,
)

# ── helpers ────────────────────────────────────────────────────────────────


def _default_result(**kwargs) -> ProteinBindingResult:
    """Run simulation with sensible defaults, optionally overriding."""
    params = dict(
        drug_name="TestDrug",
        c_total_mg_L=10.0,
        kon_per_h_per_mg_L=0.1,
        koff_per_h=1.0,
        protein_conc_mg_L=40.0,
        t_end_h=5.0,
        dt_h=0.01,
    )
    params.update(kwargs)
    return simulate_binding_kinetics(**params)


# ── Input validation ────────────────────────────────────────────────────────


class TestSimulateBindingKineticsValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            _default_result(drug_name="")

    def test_non_string_drug_name_raises(self):
        with pytest.raises((ValueError, TypeError)):
            simulate_binding_kinetics(
                drug_name=123,  # type: ignore[arg-type]
                c_total_mg_L=10.0,
                kon_per_h_per_mg_L=0.1,
                koff_per_h=1.0,
                protein_conc_mg_L=40.0,
            )

    def test_zero_c_total_raises(self):
        with pytest.raises(ValueError, match="c_total"):
            _default_result(c_total_mg_L=0.0)

    def test_negative_c_total_raises(self):
        with pytest.raises(ValueError, match="c_total"):
            _default_result(c_total_mg_L=-5.0)

    def test_zero_kon_raises(self):
        with pytest.raises(ValueError, match="kon"):
            _default_result(kon_per_h_per_mg_L=0.0)

    def test_negative_kon_raises(self):
        with pytest.raises(ValueError, match="kon"):
            _default_result(kon_per_h_per_mg_L=-1.0)

    def test_zero_koff_raises(self):
        with pytest.raises(ValueError, match="koff"):
            _default_result(koff_per_h=0.0)

    def test_negative_koff_raises(self):
        with pytest.raises(ValueError, match="koff"):
            _default_result(koff_per_h=-0.5)

    def test_zero_protein_conc_raises(self):
        with pytest.raises(ValueError, match="protein"):
            _default_result(protein_conc_mg_L=0.0)

    def test_negative_protein_conc_raises(self):
        with pytest.raises(ValueError, match="protein"):
            _default_result(protein_conc_mg_L=-10.0)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end"):
            _default_result(t_end_h=0.0)

    def test_negative_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end"):
            _default_result(t_end_h=-1.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt"):
            _default_result(dt_h=0.0)

    def test_dt_ge_t_end_raises(self):
        with pytest.raises(ValueError):
            _default_result(t_end_h=1.0, dt_h=1.0)


# ── Basic output shape ──────────────────────────────────────────────────────


class TestSimulateBindingKineticsOutput:
    def test_returns_protein_binding_result(self):
        res = _default_result()
        assert isinstance(res, ProteinBindingResult)

    def test_time_series_same_length(self):
        res = _default_result()
        assert len(res.times_h) == len(res.c_free_mg_L) == len(res.c_bound_mg_L)

    def test_time_starts_at_zero(self):
        res = _default_result()
        assert res.times_h[0] == pytest.approx(0.0)

    def test_time_ends_near_t_end(self):
        res = _default_result(t_end_h=2.0, dt_h=0.01)
        assert res.times_h[-1] == pytest.approx(2.0, abs=0.02)

    def test_field_values_match_inputs(self):
        res = _default_result(c_total_mg_L=5.0, protein_conc_mg_L=20.0)
        assert res.c_total_mg_L == pytest.approx(5.0)
        assert res.protein_conc_mg_L == pytest.approx(20.0)

    def test_drug_name_stored(self):
        res = _default_result(drug_name="Warfarin")
        assert res.drug_name == "Warfarin"


# ── Mass conservation ────────────────────────────────────────────────────────


class TestMassConservation:
    def test_free_plus_bound_equals_total_at_every_step(self):
        res = _default_result()
        total = res.c_total_mg_L
        for free, bound in zip(res.c_free_mg_L, res.c_bound_mg_L, strict=True):
            assert free + bound == pytest.approx(total, abs=1e-6)

    def test_initial_all_free(self):
        res = _default_result()
        assert res.c_free_mg_L[0] == pytest.approx(res.c_total_mg_L, abs=1e-9)
        assert res.c_bound_mg_L[0] == pytest.approx(0.0, abs=1e-9)

    def test_concentrations_non_negative(self):
        res = _default_result()
        assert all(f >= -1e-9 for f in res.c_free_mg_L)
        assert all(b >= -1e-9 for b in res.c_bound_mg_L)


# ── Equilibrium properties ──────────────────────────────────────────────────


class TestEquilibrium:
    def test_kd_equals_koff_over_kon(self):
        res = _default_result(kon_per_h_per_mg_L=0.2, koff_per_h=2.0)
        assert res.kd_mg_L == pytest.approx(2.0 / 0.2)

    def test_fu_equilibrium_formula(self):
        # fu_eq = Kd / (Kd + protein_conc)
        kon = 0.1
        koff = 1.0
        protein = 40.0
        kd = koff / kon
        expected_fu = kd / (kd + protein)
        res = _default_result(kon_per_h_per_mg_L=kon, koff_per_h=koff, protein_conc_mg_L=protein)
        assert res.fu_equilibrium == pytest.approx(expected_fu, rel=1e-6)

    def test_t_half_binding_equals_ln2_over_koff(self):
        koff = 3.0
        res = _default_result(koff_per_h=koff)
        assert res.t_half_binding_h == pytest.approx(math.log(2.0) / koff, rel=1e-6)

    def test_simulation_approaches_equilibrium(self):
        """After many half-lives the simulated fu should be close to fu_eq.

        Use drug << protein (protein excess) so ligand depletion is negligible
        and the simple Kd/(Kd+protein_conc) formula applies.
        """
        kon = 0.05
        koff = 0.5
        protein = 200.0  # large protein excess
        kd = koff / kon
        fu_eq = kd / (kd + protein)
        c_total = 0.1  # drug << protein → ligand-depletion negligible
        # Run for 20 half-lives (well into equilibrium)
        t_half = math.log(2.0) / koff
        res = simulate_binding_kinetics(
            drug_name="X",
            c_total_mg_L=c_total,
            kon_per_h_per_mg_L=kon,
            koff_per_h=koff,
            protein_conc_mg_L=protein,
            t_end_h=20 * t_half,
            dt_h=0.001,
        )
        # fu at last time point
        fu_sim = res.c_free_mg_L[-1] / c_total
        assert fu_sim == pytest.approx(fu_eq, abs=0.01)

    def test_higher_protein_gives_lower_fu(self):
        res_low = _default_result(protein_conc_mg_L=10.0)
        res_high = _default_result(protein_conc_mg_L=80.0)
        assert res_low.fu_equilibrium > res_high.fu_equilibrium

    def test_higher_koff_gives_higher_fu(self):
        """Higher koff → higher Kd → weaker binding → higher fu."""
        res_low = _default_result(koff_per_h=0.5)
        res_high = _default_result(koff_per_h=5.0)
        assert res_low.fu_equilibrium < res_high.fu_equilibrium


# ── Sensitivity function ────────────────────────────────────────────────────


class TestProteinBindingSensitivity:
    def test_empty_koff_values_raises(self):
        with pytest.raises(ValueError):
            protein_binding_sensitivity("Drug", 10.0, 0.1, [], 40.0)

    def test_negative_koff_in_list_raises(self):
        with pytest.raises(ValueError):
            protein_binding_sensitivity("Drug", 10.0, 0.1, [1.0, -0.5], 40.0)

    def test_returns_list_of_dicts(self):
        result = protein_binding_sensitivity("Drug", 10.0, 0.1, [0.5, 1.0, 2.0], 40.0)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_length_matches_koff_values(self):
        koffs = [0.1, 0.5, 1.0, 2.0, 5.0]
        result = protein_binding_sensitivity("Drug", 10.0, 0.1, koffs, 40.0)
        assert len(result) == len(koffs)

    def test_koff_stored_in_result(self):
        koffs = [0.5, 2.0]
        result = protein_binding_sensitivity("Drug", 10.0, 0.1, koffs, 40.0)
        assert result[0]["koff"] == pytest.approx(0.5)
        assert result[1]["koff"] == pytest.approx(2.0)

    def test_fu_increases_with_koff(self):
        """Higher koff → higher Kd → higher fu (monotone)."""
        koffs = [0.1, 0.5, 1.0, 2.0, 5.0]
        result = protein_binding_sensitivity("Drug", 10.0, 0.1, koffs, 40.0)
        fus = [r["fu_equilibrium"] for r in result]
        assert all(fus[i] < fus[i + 1] for i in range(len(fus) - 1))

    def test_kd_equals_koff_over_kon(self):
        kon = 0.2
        koffs = [1.0, 2.0]
        result = protein_binding_sensitivity("Drug", 10.0, kon, koffs, 40.0)
        for r in result:
            assert r["kd_mg_L"] == pytest.approx(r["koff"] / kon, rel=1e-9)

    def test_zero_protein_raises(self):
        with pytest.raises(ValueError, match="protein"):
            protein_binding_sensitivity("Drug", 10.0, 0.1, [1.0], 0.0)

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            protein_binding_sensitivity("", 10.0, 0.1, [1.0], 40.0)
