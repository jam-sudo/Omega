"""
Tests for Phase 387 — Depot Injection PK Model
"""

import math

import pytest

from omega_pbpk.core.depot_injection_pk import (
    DepotInjectionResult,
    compare_formulations,
    simulate_depot_injection,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> DepotInjectionResult:
    """Return a result with sensible defaults, overrideable via kwargs."""
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        formulation_type="suspension",
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
        k_release_per_h=0.05,
        t_end_h=200.0,
        dt_h=0.5,
    )
    params.update(kwargs)
    return simulate_depot_injection(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_depot_injection_result(self):
        result = _default_result()
        assert isinstance(result, DepotInjectionResult)

    def test_result_has_expected_fields(self):
        result = _default_result()
        assert hasattr(result, "drug_name")
        assert hasattr(result, "formulation_type")
        assert hasattr(result, "dose_mg")
        assert hasattr(result, "times_h")
        assert hasattr(result, "c_depot_mg")
        assert hasattr(result, "c_plasma_mg_L")
        assert hasattr(result, "cmax_mg_L")
        assert hasattr(result, "tmax_h")
        assert hasattr(result, "auc_mg_h_per_L")
        assert hasattr(result, "t_half_apparent_h")
        assert hasattr(result, "duration_above_mic_h")
        assert hasattr(result, "depot_t90_h")
        assert hasattr(result, "flip_flop_kinetics")
        assert hasattr(result, "notes")

    def test_list_fields_are_lists(self):
        result = _default_result()
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_depot_mg, list)
        assert isinstance(result.c_plasma_mg_L, list)

    def test_list_fields_same_length(self):
        result = _default_result()
        n = len(result.times_h)
        assert len(result.c_depot_mg) == n
        assert len(result.c_plasma_mg_L) == n


# ---------------------------------------------------------------------------
# Depot dynamics
# ---------------------------------------------------------------------------


class TestDepotDynamics:
    def test_depot_starts_at_dose(self):
        result = _default_result(dose_mg=200.0)
        assert abs(result.c_depot_mg[0] - 200.0) < 1e-6

    def test_depot_decreases_monotonically(self):
        result = _default_result()
        for i in range(1, len(result.c_depot_mg)):
            assert result.c_depot_mg[i] <= result.c_depot_mg[i - 1] + 1e-9

    def test_depot_approaches_zero(self):
        # After very long simulation, almost all drug absorbed
        result = simulate_depot_injection(
            drug_name="X",
            dose_mg=100.0,
            formulation_type="suspension",
            cl_sys_L_per_h=5.0,
            vd_sys_L=10.0,
            k_release_per_h=0.1,
            t_end_h=1000.0,
            dt_h=1.0,
        )
        assert result.c_depot_mg[-1] < 1.0  # < 1 mg remaining

    def test_depot_t90_analytical(self):
        k_rel = 0.05
        result = _default_result(k_release_per_h=k_rel, formulation_type="suspension")
        expected = math.log(10.0) / k_rel
        assert abs(result.depot_t90_h - expected) < 1e-6


# ---------------------------------------------------------------------------
# Plasma dynamics
# ---------------------------------------------------------------------------


class TestPlasmaDynamics:
    def test_plasma_starts_at_zero(self):
        result = _default_result()
        assert result.c_plasma_mg_L[0] == 0.0

    def test_plasma_rises_then_falls(self):
        result = _default_result()
        cmax_idx = result.c_plasma_mg_L.index(result.cmax_mg_L)
        # Must rise before Cmax
        assert result.c_plasma_mg_L[cmax_idx] >= result.c_plasma_mg_L[0]
        # Must fall after Cmax (check well after peak)
        if cmax_idx + 10 < len(result.c_plasma_mg_L):
            assert result.c_plasma_mg_L[-1] < result.cmax_mg_L

    def test_cmax_positive(self):
        result = _default_result()
        assert result.cmax_mg_L > 0.0

    def test_tmax_positive(self):
        result = _default_result()
        assert result.tmax_h > 0.0

    def test_auc_positive(self):
        result = _default_result()
        assert result.auc_mg_h_per_L > 0.0

    def test_higher_dose_higher_cmax(self):
        r100 = _default_result(dose_mg=100.0)
        r200 = _default_result(dose_mg=200.0)
        assert r200.cmax_mg_L > r100.cmax_mg_L

    def test_higher_dose_higher_auc(self):
        r100 = _default_result(dose_mg=100.0)
        r200 = _default_result(dose_mg=200.0)
        assert r200.auc_mg_h_per_L > r100.auc_mg_h_per_L

    def test_linearity_cmax(self):
        r100 = _default_result(dose_mg=100.0)
        r200 = _default_result(dose_mg=200.0)
        ratio = r200.cmax_mg_L / r100.cmax_mg_L
        assert abs(ratio - 2.0) < 0.05  # within 5 % of 2x


# ---------------------------------------------------------------------------
# Formulation effects
# ---------------------------------------------------------------------------


class TestFormulationEffects:
    def test_implant_slower_than_suspension(self):
        r_sus = _default_result(formulation_type="suspension", t_end_h=720.0)
        r_imp = simulate_depot_injection(
            drug_name="TestDrug",
            dose_mg=100.0,
            formulation_type="implant",
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
            k_release_per_h=0.05,
            t_end_h=720.0,
            dt_h=0.5,
        )
        # Implant has longer Tmax (slower absorption)
        assert r_imp.tmax_h > r_sus.tmax_h

    def test_implant_flip_flop(self):
        """Implant should trigger flip-flop kinetics (absorption << elimination)."""
        result = simulate_depot_injection(
            drug_name="TestDrug",
            dose_mg=100.0,
            formulation_type="implant",
            cl_sys_L_per_h=5.0,
            vd_sys_L=10.0,
            k_release_per_h=0.5,
            t_end_h=720.0,
            dt_h=0.5,
        )
        assert result.flip_flop_kinetics is True

    def test_suspension_no_flip_flop(self):
        """Fast release suspension should NOT have flip-flop when k_rel >> k_elim."""
        # CL=1, Vd=100 → k_elim=0.01; k_release=0.5 → eff=0.5 >> 0.01
        result = simulate_depot_injection(
            drug_name="TestDrug",
            dose_mg=100.0,
            formulation_type="suspension",
            cl_sys_L_per_h=1.0,
            vd_sys_L=100.0,
            k_release_per_h=0.5,
            t_end_h=200.0,
            dt_h=0.5,
        )
        assert result.flip_flop_kinetics is False

    def test_depot_t90_implant_longer_than_suspension(self):
        k_rel = 0.05
        r_sus = _default_result(formulation_type="suspension", k_release_per_h=k_rel)
        r_imp = simulate_depot_injection(
            drug_name="TestDrug",
            dose_mg=100.0,
            formulation_type="implant",
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
            k_release_per_h=k_rel,
            t_end_h=200.0,
            dt_h=0.5,
        )
        assert r_imp.depot_t90_h > r_sus.depot_t90_h


# ---------------------------------------------------------------------------
# MIC duration
# ---------------------------------------------------------------------------


class TestMICDuration:
    def test_no_mic_returns_zero(self):
        result = _default_result(mic_mg_L=0.0)
        assert result.duration_above_mic_h == 0.0

    def test_mic_duration_positive_when_set(self):
        result = _default_result(mic_mg_L=0.01)
        assert result.duration_above_mic_h > 0.0

    def test_high_mic_reduces_duration(self):
        r_low = _default_result(mic_mg_L=0.01)
        r_high = _default_result(mic_mg_L=1000.0)
        assert r_low.duration_above_mic_h > r_high.duration_above_mic_h


# ---------------------------------------------------------------------------
# compare_formulations
# ---------------------------------------------------------------------------


class TestCompareFormulations:
    def test_returns_four_results(self):
        results = compare_formulations(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
        )
        assert len(results) == 4

    def test_returns_list_of_depot_injection_results(self):
        results = compare_formulations(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
        )
        for r in results:
            assert isinstance(r, DepotInjectionResult)

    def test_sorted_by_auc_descending(self):
        results = compare_formulations(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
        )
        aucs = [r.auc_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_all_formulation_types_represented(self):
        results = compare_formulations(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
        )
        types = {r.formulation_type for r in results}
        assert types == {"suspension", "microsphere", "implant", "oil_solution"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_depot_injection(
                drug_name="X",
                dose_mg=-10.0,
                formulation_type="suspension",
                cl_sys_L_per_h=1.0,
                vd_sys_L=10.0,
            )

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="cl_sys_L_per_h"):
            simulate_depot_injection(
                drug_name="X",
                dose_mg=100.0,
                formulation_type="suspension",
                cl_sys_L_per_h=0.0,
                vd_sys_L=10.0,
            )

    def test_invalid_vd(self):
        with pytest.raises(ValueError, match="vd_sys_L"):
            simulate_depot_injection(
                drug_name="X",
                dose_mg=100.0,
                formulation_type="suspension",
                cl_sys_L_per_h=1.0,
                vd_sys_L=0.0,
            )

    def test_invalid_k_release(self):
        with pytest.raises(ValueError, match="k_release_per_h"):
            simulate_depot_injection(
                drug_name="X",
                dose_mg=100.0,
                formulation_type="suspension",
                cl_sys_L_per_h=1.0,
                vd_sys_L=10.0,
                k_release_per_h=0.0,
            )

    def test_invalid_formulation_type(self):
        with pytest.raises(ValueError, match="formulation_type"):
            simulate_depot_injection(
                drug_name="X",
                dose_mg=100.0,
                formulation_type="unknown_type",
                cl_sys_L_per_h=1.0,
                vd_sys_L=10.0,
            )
