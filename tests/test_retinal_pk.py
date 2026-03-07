"""Tests for Phase 176: retinal_pk module."""

import pytest

from omega_pbpk.core.retinal_pk import RetinalPKResult, simulate_retinal_pk

# ---------------------------------------------------------------------------
# Intravitreal route
# ---------------------------------------------------------------------------


class TestIntravitreal:
    def setup_method(self):
        self.result = simulate_retinal_pk(
            drug_name="Ranibizumab",
            dose_mg=0.5,
            route="intravitreal",
            mw_Da=500.0,
            t_end_h=720.0,
            dt_h=1.0,
        )

    def test_returns_retinal_pk_result(self):
        assert isinstance(self.result, RetinalPKResult)

    def test_vitreous_initial_concentration_nonzero(self):
        """Intravitreal: C_vitreous(0) should be dose/V_vit = 0.5/0.004 = 125 mg/L."""
        assert self.result.c_vitreous_mg_L[0] == pytest.approx(125.0, rel=1e-6)

    def test_retina_starts_near_zero(self):
        assert self.result.c_retina_mg_L[0] == pytest.approx(0.0, abs=1e-9)

    def test_systemic_starts_at_zero(self):
        assert self.result.c_systemic_mg_L[0] == pytest.approx(0.0, abs=1e-9)

    def test_retina_rises_after_injection(self):
        """Retinal concentration should increase before peak."""
        max_ret = max(self.result.c_retina_mg_L)
        assert max_ret > 0.0

    def test_vitreous_declines_over_time(self):
        """Vitreous concentration should be lower at end than beginning."""
        assert self.result.c_vitreous_mg_L[-1] < self.result.c_vitreous_mg_L[0]

    def test_cmax_retina_positive(self):
        assert self.result.cmax_retina_mg_L > 0.0

    def test_tmax_retina_positive(self):
        assert self.result.tmax_retina_h > 0.0

    def test_auc_retina_positive(self):
        assert self.result.auc_retina_mg_h_per_L > 0.0

    def test_auc_vitreous_positive(self):
        assert self.result.auc_vitreous_mg_h_per_L > 0.0

    def test_t_half_vitreous_positive(self):
        assert self.result.t_half_vitreous_h > 0.0

    def test_systemic_exposure_pct_is_small(self):
        """Systemic exposure should be a small fraction for intravitreal route."""
        assert 0.0 <= self.result.systemic_exposure_pct < 20.0

    def test_time_array_length(self):
        n_steps = int(720.0 / 1.0)
        assert len(self.result.times_h) == n_steps + 1

    def test_time_starts_at_zero(self):
        assert self.result.times_h[0] == 0.0

    def test_time_ends_at_t_end(self):
        assert self.result.times_h[-1] == pytest.approx(720.0)

    def test_notes_mentions_intravitreal(self):
        assert "ntravitreal" in self.result.notes or "intravitreal" in self.result.notes.lower()


# ---------------------------------------------------------------------------
# Systemic route
# ---------------------------------------------------------------------------


class TestSystemicRoute:
    def setup_method(self):
        self.result = simulate_retinal_pk(
            drug_name="SystemicDrug",
            dose_mg=10.0,
            route="systemic",
            mw_Da=300.0,
            t_end_h=48.0,
            dt_h=0.5,
        )

    def test_systemic_initial_concentration_nonzero(self):
        """Systemic route: C_systemic(0) = dose / Vd_sys = 10/5 = 2 mg/L."""
        assert self.result.c_systemic_mg_L[0] == pytest.approx(2.0, rel=1e-6)

    def test_vitreous_starts_at_zero(self):
        assert self.result.c_vitreous_mg_L[0] == pytest.approx(0.0, abs=1e-9)

    def test_retina_starts_at_zero(self):
        assert self.result.c_retina_mg_L[0] == pytest.approx(0.0, abs=1e-9)

    def test_systemic_clears_over_time(self):
        """Systemic drug should clear (ke_sys=1/h), declining from initial value."""
        assert self.result.c_systemic_mg_L[-1] < self.result.c_systemic_mg_L[0]

    def test_route_stored_correctly(self):
        assert self.result.route == "systemic"


# ---------------------------------------------------------------------------
# MW effect
# ---------------------------------------------------------------------------


class TestMWEffect:
    def test_smaller_mw_higher_retinal_auc(self):
        """Smaller MW → kvr closer to max → faster retinal uptake → higher AUC."""
        result_small = simulate_retinal_pk(
            "Drug", 0.5, "intravitreal", mw_Da=200.0, t_end_h=720.0, dt_h=1.0
        )
        result_large = simulate_retinal_pk(
            "Drug", 0.5, "intravitreal", mw_Da=1500.0, t_end_h=720.0, dt_h=1.0
        )
        assert result_small.auc_retina_mg_h_per_L > result_large.auc_retina_mg_h_per_L

    def test_mw_stored(self):
        result = simulate_retinal_pk("Drug", 0.5, "intravitreal", mw_Da=750.0, t_end_h=100.0)
        assert result.mw_Da == pytest.approx(750.0)

    def test_very_large_mw_clamps_kvr(self):
        """MW=10000 → kvr = base*0.1 (clamped at min)."""
        result = simulate_retinal_pk(
            "BigMolecule", 0.5, "intravitreal", mw_Da=10000.0, t_end_h=720.0, dt_h=1.0
        )
        # Should still run without error and produce valid output
        assert result.auc_retina_mg_h_per_L >= 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_retinal_pk("Drug", -1.0, "intravitreal")

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_retinal_pk("Drug", 0.0, "intravitreal")

    def test_invalid_route_raises(self):
        with pytest.raises(ValueError, match="route"):
            simulate_retinal_pk("Drug", 0.5, "topical")

    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_retinal_pk("Drug", 0.5, "intravitreal", mw_Da=-100.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw_Da"):
            simulate_retinal_pk("Drug", 0.5, "intravitreal", mw_Da=0.0)


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


class TestResultFields:
    def setup_method(self):
        self.result = simulate_retinal_pk(
            "TestDrug", 1.0, "intravitreal", mw_Da=500.0, t_end_h=168.0, dt_h=1.0
        )

    def test_drug_name_stored(self):
        assert self.result.drug_name == "TestDrug"

    def test_dose_stored(self):
        assert self.result.dose_mg == pytest.approx(1.0)

    def test_retina_to_vitreous_ratio_nonnegative(self):
        assert self.result.retina_to_vitreous_auc_ratio >= 0.0

    def test_systemic_exposure_pct_nonnegative(self):
        assert self.result.systemic_exposure_pct >= 0.0

    def test_all_concentration_lists_same_length(self):
        n = len(self.result.times_h)
        assert len(self.result.c_vitreous_mg_L) == n
        assert len(self.result.c_retina_mg_L) == n
        assert len(self.result.c_systemic_mg_L) == n

    def test_all_concentrations_nonnegative(self):
        assert all(c >= 0.0 for c in self.result.c_vitreous_mg_L)
        assert all(c >= 0.0 for c in self.result.c_retina_mg_L)
        assert all(c >= 0.0 for c in self.result.c_systemic_mg_L)
