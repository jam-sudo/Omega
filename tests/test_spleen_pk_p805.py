"""Tests for Phase 805 spleen PK model."""

from __future__ import annotations

import pytest

from omega_pbpk.core.spleen_pk_p805 import SpleenPKResult, simulate_spleen_pk


class TestSpleenPKValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=-10.0)

    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError):
            simulate_spleen_pk(drug_name="", dose_mg=10.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, cl_L_per_h=0.0)

    def test_zero_vd_plasma_raises(self):
        with pytest.raises(ValueError, match="vd_plasma_L"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, vd_plasma_L=0.0)

    def test_zero_v_spleen_raises(self):
        with pytest.raises(ValueError, match="v_spleen_L"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, v_spleen_L=0.0)

    def test_negative_macrophage_uptake_raises(self):
        with pytest.raises(ValueError, match="macrophage_uptake_rate"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, macrophage_uptake_rate=-0.1)

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, t_end_h=0.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, dt_h=0.0)


class TestSpleenPKResult:
    def _run(self, **kwargs):
        defaults = dict(
            drug_name="TestDrug",
            dose_mg=100.0,
            cl_L_per_h=5.0,
            vd_plasma_L=5.0,
            q_spleen_L_per_h=0.2,
            v_spleen_L=0.15,
            kp_spleen=3.0,
            macrophage_uptake_rate=0.1,
            t_end_h=48.0,
            dt_h=0.1,
        )
        defaults.update(kwargs)
        return simulate_spleen_pk(**defaults)

    def test_returns_correct_type(self):
        result = self._run()
        assert isinstance(result, SpleenPKResult)

    def test_list_fields_present(self):
        result = self._run()
        assert isinstance(result.times_h, list)
        assert isinstance(result.c_plasma_mg_L, list)
        assert isinstance(result.c_spleen_mg_L, list)

    def test_list_lengths_match(self):
        result = self._run()
        n = len(result.times_h)
        assert len(result.c_plasma_mg_L) == n
        assert len(result.c_spleen_mg_L) == n

    def test_cmax_plasma_positive(self):
        result = self._run()
        assert result.cmax_plasma > 0

    def test_auc_plasma_positive(self):
        result = self._run()
        assert result.auc_plasma > 0

    def test_auc_spleen_positive(self):
        result = self._run()
        assert result.auc_spleen > 0

    def test_cmax_spleen_positive(self):
        result = self._run()
        assert result.cmax_spleen > 0

    def test_spleen_plasma_kp_positive(self):
        result = self._run()
        assert result.spleen_plasma_kp > 0

    def test_times_start_at_zero(self):
        result = self._run()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = self._run(t_end_h=24.0)
        assert result.times_h[-1] == pytest.approx(24.0, rel=0.05)

    def test_drug_name_stored(self):
        result = self._run(drug_name="Nanothing")
        assert result.drug_name == "Nanothing"

    def test_dose_stored(self):
        result = self._run(dose_mg=50.0)
        assert result.dose_mg == pytest.approx(50.0)

    def test_higher_macrophage_uptake_lower_spleen_auc(self):
        """Higher macrophage_uptake_rate → lower spleen AUC (irreversible sink)."""
        low = self._run(macrophage_uptake_rate=0.01)
        high = self._run(macrophage_uptake_rate=1.0)
        assert high.auc_spleen < low.auc_spleen

    def test_particle_size_200nm_higher_macrophage_uptake_than_10nm(self):
        """200 nm particles → higher phagocytosis than 10 nm particles."""
        large = self._run(particle_size_nm=200.0)
        small = self._run(particle_size_nm=10.0)
        assert large.macrophage_phagocytosis_pct >= small.macrophage_phagocytosis_pct

    def test_particle_size_modulates_notes(self):
        result = self._run(particle_size_nm=200.0)
        assert "200" in result.notes or "nm" in result.notes.lower()

    def test_macrophage_phagocytosis_pct_between_0_and_100(self):
        result = self._run(macrophage_uptake_rate=5.0)
        assert 0.0 <= result.macrophage_phagocytosis_pct <= 100.0

    def test_red_pulp_pct_positive(self):
        result = self._run()
        assert result.red_pulp_uptake_pct > 0

    def test_marginal_zone_pct_positive(self):
        result = self._run()
        assert result.marginal_zone_pct > 0

    def test_t_half_spleen_positive(self):
        result = self._run()
        assert result.t_half_spleen_h > 0

    def test_notes_is_string(self):
        result = self._run()
        assert isinstance(result.notes, str)

    def test_oral_route(self):
        result = self._run(route="oral", f_oral=0.8, ka_per_h=1.0)
        assert result.cmax_plasma > 0
        assert result.auc_plasma > 0

    def test_all_concentrations_nonnegative(self):
        result = self._run(macrophage_uptake_rate=2.0)
        for c in result.c_plasma_mg_L:
            assert c >= 0.0
        for c in result.c_spleen_mg_L:
            assert c >= 0.0
