"""Tests for Phase 603: Drug permeation through skin layers."""

import pytest

from omega_pbpk.core.skin_layer_permeation_p603 import (
    SkinLayerPermeationResult,
    simulate_skin_permeation,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _default_result(**kwargs):
    """Return a simulation result with sensible defaults, overriding via kwargs."""
    params = dict(
        drug_name="TestDrug",
        dose_mg=10.0,
        patch_area_cm2=10.0,
        logP=2.0,
        mw_Da=300.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
        t_end_h=72.0,
        dt_h=0.5,
    )
    params.update(kwargs)
    return simulate_skin_permeation(**params)


# ---------------------------------------------------------------------------
# Return type & structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        res = _default_result()
        assert isinstance(res, SkinLayerPermeationResult)

    def test_list_fields_are_lists(self):
        res = _default_result()
        assert isinstance(res.times_h, list)
        assert isinstance(res.c_sc_mg_cm3, list)
        assert isinstance(res.c_epidermis_mg_cm3, list)
        assert isinstance(res.c_dermis_mg_cm3, list)
        assert isinstance(res.c_systemic_mg_L, list)

    def test_list_lengths_consistent(self):
        res = _default_result()
        n = len(res.times_h)
        assert len(res.c_sc_mg_cm3) == n
        assert len(res.c_epidermis_mg_cm3) == n
        assert len(res.c_dermis_mg_cm3) == n
        assert len(res.c_systemic_mg_L) == n

    def test_scalar_fields_are_floats(self):
        res = _default_result()
        assert isinstance(res.cmax_systemic_mg_L, float)
        assert isinstance(res.tmax_systemic_h, float)
        assert isinstance(res.auc_systemic_mg_h_per_L, float)
        assert isinstance(res.lag_time_h, float)
        assert isinstance(res.steady_state_flux_mg_cm2_h, float)
        assert isinstance(res.total_absorbed_pct, float)

    def test_notes_is_string(self):
        res = _default_result()
        assert isinstance(res.notes, str)
        assert len(res.notes) > 0

    def test_drug_name_preserved(self):
        res = _default_result(drug_name="Estradiol")
        assert res.drug_name == "Estradiol"

    def test_dose_preserved(self):
        res = _default_result(dose_mg=5.0)
        assert res.dose_mg == 5.0

    def test_patch_area_preserved(self):
        res = _default_result(patch_area_cm2=20.0)
        assert res.patch_area_cm2 == 20.0


# ---------------------------------------------------------------------------
# Time vector
# ---------------------------------------------------------------------------


class TestTimeVector:
    def test_starts_at_zero(self):
        res = _default_result()
        assert res.times_h[0] == pytest.approx(0.0)

    def test_ends_near_t_end(self):
        res = _default_result(t_end_h=48.0, dt_h=1.0)
        assert res.times_h[-1] == pytest.approx(48.0, abs=1.0)

    def test_monotonically_increasing(self):
        res = _default_result()
        for i in range(1, len(res.times_h)):
            assert res.times_h[i] > res.times_h[i - 1]


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_sc_initial_nonzero(self):
        res = _default_result()
        assert res.c_sc_mg_cm3[0] > 0.0

    def test_epi_initial_zero(self):
        res = _default_result()
        assert res.c_epidermis_mg_cm3[0] == pytest.approx(0.0)

    def test_derm_initial_zero(self):
        res = _default_result()
        assert res.c_dermis_mg_cm3[0] == pytest.approx(0.0)

    def test_systemic_initial_zero(self):
        res = _default_result()
        assert res.c_systemic_mg_L[0] == pytest.approx(0.0)

    def test_sc_declines_over_time(self):
        res = _default_result()
        assert res.c_sc_mg_cm3[-1] < res.c_sc_mg_cm3[0]


# ---------------------------------------------------------------------------
# Systemic PK metrics
# ---------------------------------------------------------------------------


class TestSystemicPK:
    def test_auc_positive(self):
        res = _default_result()
        assert res.auc_systemic_mg_h_per_L > 0.0

    def test_cmax_positive(self):
        res = _default_result()
        assert res.cmax_systemic_mg_L > 0.0

    def test_tmax_within_simulation(self):
        res = _default_result(t_end_h=72.0)
        assert 0.0 <= res.tmax_systemic_h <= 72.0

    def test_lag_time_positive(self):
        res = _default_result()
        assert res.lag_time_h >= 0.0

    def test_lag_time_less_than_tmax(self):
        res = _default_result()
        # Lag must come before or at peak
        assert res.lag_time_h <= res.tmax_systemic_h + 1e-6


# ---------------------------------------------------------------------------
# Lipophilicity effect
# ---------------------------------------------------------------------------


class TestLipophilicityEffect:
    def test_higher_logP_faster_sc_permeation(self):
        """Higher logP should give higher stratum corneum permeability."""
        res_low = _default_result(logP=0.5, mw_Da=200.0)
        res_high = _default_result(logP=4.0, mw_Da=200.0)
        # Higher logP → more drug in epidermis/dermis/systemic faster
        assert res_high.auc_systemic_mg_h_per_L >= res_low.auc_systemic_mg_h_per_L

    def test_higher_logP_higher_cmax(self):
        res_low = _default_result(logP=0.5, mw_Da=200.0)
        res_high = _default_result(logP=3.0, mw_Da=200.0)
        assert res_high.cmax_systemic_mg_L >= res_low.cmax_systemic_mg_L


# ---------------------------------------------------------------------------
# Molecular weight effect
# ---------------------------------------------------------------------------


class TestMWEffect:
    def test_smaller_mw_higher_auc(self):
        """Smaller MW → higher permeability coefficients → greater absorption."""
        res_small = _default_result(mw_Da=100.0, logP=2.0)
        res_large = _default_result(mw_Da=600.0, logP=2.0)
        assert res_small.auc_systemic_mg_h_per_L >= res_large.auc_systemic_mg_h_per_L

    def test_larger_mw_reduces_permeation(self):
        res_small = _default_result(mw_Da=150.0)
        res_large = _default_result(mw_Da=800.0)
        assert res_small.total_absorbed_pct >= res_large.total_absorbed_pct


# ---------------------------------------------------------------------------
# Flux and absorption
# ---------------------------------------------------------------------------


class TestFluxAndAbsorption:
    def test_steady_state_flux_nonnegative(self):
        res = _default_result()
        assert res.steady_state_flux_mg_cm2_h >= 0.0

    def test_total_absorbed_pct_in_range(self):
        res = _default_result()
        assert 0.0 <= res.total_absorbed_pct <= 100.0


# ---------------------------------------------------------------------------
# Patch area effect
# ---------------------------------------------------------------------------


class TestPatchAreaEffect:
    def test_larger_area_higher_auc(self):
        res_small = _default_result(patch_area_cm2=5.0)
        res_large = _default_result(patch_area_cm2=20.0)
        # Larger area → more flux → more systemic exposure
        assert res_large.auc_systemic_mg_h_per_L >= res_small.auc_systemic_mg_h_per_L


# ---------------------------------------------------------------------------
# Validation / error cases
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=-1.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=0.0)

    def test_negative_area_raises(self):
        with pytest.raises(ValueError, match="patch_area_cm2"):
            _default_result(patch_area_cm2=-5.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _default_result(cl_sys_L_per_h=0.0)

    def test_negative_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _default_result(vd_sys_L=-10.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default_result(mw_Da=0.0)


# ---------------------------------------------------------------------------
# Concentrations are non-negative
# ---------------------------------------------------------------------------


class TestNonNegativity:
    def test_sc_nonneg(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_sc_mg_cm3)

    def test_epi_nonneg(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_epidermis_mg_cm3)

    def test_derm_nonneg(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_dermis_mg_cm3)

    def test_systemic_nonneg(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_systemic_mg_L)
