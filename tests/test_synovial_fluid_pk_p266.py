"""
Tests for Phase 266: Synovial Fluid PK Model (p266)
"""

import pytest

from omega_pbpk.core.synovial_fluid_pk_p266 import (
    SynovialFluidPKResult,
    compare_routes,
    simulate_synovial_fluid_pk,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _sim(drug_name="Ibuprofen", dose_mg=400.0, route="iv_bolus", cl_L_per_h=10.0, **kwargs):
    return simulate_synovial_fluid_pk(drug_name, dose_mg, route, cl_L_per_h, **kwargs)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_synovial_result(self):
        result = _sim()
        assert isinstance(result, SynovialFluidPKResult)

    def test_result_has_list_fields(self):
        result = _sim()
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_plasma_mg_L, list)
        assert isinstance(result.c_synovial_mg_L, list)


# ---------------------------------------------------------------------------
# IV bolus
# ---------------------------------------------------------------------------
class TestIVBolus:
    def test_plasma_starts_high(self):
        result = _sim(route="iv_bolus")
        assert result.c_plasma_mg_L[0] > 0.0

    def test_synovial_starts_at_zero(self):
        result = _sim(route="iv_bolus")
        assert result.c_synovial_mg_L[0] == 0.0

    def test_cmax_plasma_positive(self):
        result = _sim(route="iv_bolus")
        assert result.cmax_plasma_mg_L > 0.0

    def test_cmax_synovial_positive(self):
        result = _sim(route="iv_bolus")
        assert result.cmax_synovial_mg_L > 0.0


# ---------------------------------------------------------------------------
# Oral route
# ---------------------------------------------------------------------------
class TestOral:
    def test_plasma_starts_at_zero(self):
        result = _sim(route="oral")
        assert result.c_plasma_mg_L[0] == 0.0

    def test_synovial_starts_at_zero(self):
        result = _sim(route="oral")
        assert result.c_synovial_mg_L[0] == 0.0

    def test_cmax_synovial_positive_oral(self):
        result = _sim(route="oral")
        assert result.cmax_synovial_mg_L > 0.0

    def test_auc_positive_oral(self):
        result = _sim(route="oral")
        assert result.auc_plasma_mg_h_per_L > 0.0
        assert result.auc_synovial_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Intra-articular route
# ---------------------------------------------------------------------------
class TestIntraArticular:
    def test_synovial_starts_high(self):
        result = _sim(route="intra_articular")
        assert result.c_synovial_mg_L[0] > 0.0

    def test_plasma_starts_near_zero(self):
        result = _sim(route="intra_articular")
        assert result.c_plasma_mg_L[0] == 0.0

    def test_ia_gives_very_high_cmax_synovial(self):
        result = _sim(route="intra_articular", dose_mg=10.0)
        # IA: C_synovial(0) = dose / V_synovial = 10 / 0.004 = 2500 mg/L
        assert result.cmax_synovial_mg_L > 100.0


# ---------------------------------------------------------------------------
# AUC values
# ---------------------------------------------------------------------------
class TestAUC:
    def test_auc_plasma_positive_iv(self):
        result = _sim(route="iv_bolus")
        assert result.auc_plasma_mg_h_per_L > 0.0

    def test_auc_synovial_positive_iv(self):
        result = _sim(route="iv_bolus")
        assert result.auc_synovial_mg_h_per_L > 0.0


# ---------------------------------------------------------------------------
# Synovial/plasma ratio
# ---------------------------------------------------------------------------
class TestSynovialPlasmaRatio:
    def test_ratio_positive_iv(self):
        result = _sim(route="iv_bolus")
        assert result.synovial_plasma_ratio > 0.0


# ---------------------------------------------------------------------------
# Effective half-life
# ---------------------------------------------------------------------------
class TestEffectiveHalfLife:
    def test_half_life_positive(self):
        result = _sim()
        assert result.effective_half_life_synovial_h > 0.0

    def test_half_life_analytical(self):
        import math

        # k_sp = 0.05, k_elim_s = 0.02 -> t_half = ln2/(0.07) ~ 9.9 h
        result = _sim()
        expected = math.log(2) / (0.05 + 0.02)
        assert abs(result.effective_half_life_synovial_h - expected) < 1e-6


# ---------------------------------------------------------------------------
# Compare routes
# ---------------------------------------------------------------------------
class TestCompareRoutes:
    def test_returns_three_results(self):
        results = compare_routes("Drug", 100.0, 5.0)
        assert len(results) == 3

    def test_all_are_synovial_result(self):
        results = compare_routes("Drug", 100.0, 5.0)
        assert all(isinstance(r, SynovialFluidPKResult) for r in results)

    def test_sorted_descending_by_auc_synovial(self):
        results = compare_routes("Drug", 100.0, 5.0)
        aucs = [r.auc_synovial_mg_h_per_L for r in results]
        assert aucs == sorted(aucs, reverse=True)

    def test_ia_has_highest_synovial_auc(self):
        results = compare_routes("Drug", 100.0, 5.0)
        assert results[0].route == "intra_articular"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------
class TestValidation:
    def test_zero_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_synovial_fluid_pk("D", 0.0, "iv_bolus", 5.0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_synovial_fluid_pk("D", -10.0, "iv_bolus", 5.0)

    def test_zero_clearance(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_synovial_fluid_pk("D", 100.0, "iv_bolus", 0.0)

    def test_invalid_route(self):
        with pytest.raises(ValueError, match="route"):
            simulate_synovial_fluid_pk("D", 100.0, "subcutaneous", 5.0)
