"""Tests for spleen PK model (Phase 429)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.spleen_pk import (
    SpleenPKResult,
    mps_uptake_analysis,
    simulate_spleen_pk,
)


class TestSpleenPKValidation:
    def test_empty_drug_name_raises(self):
        with pytest.raises(ValueError, match="drug_name"):
            simulate_spleen_pk(drug_name="", dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=0.0, cl_L_per_h=5.0, vd_L=50.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=-1.0, cl_L_per_h=5.0, vd_L=50.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, cl_L_per_h=0.0, vd_L=50.0)

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, cl_L_per_h=5.0, vd_L=0.0)

    def test_zero_blood_flow_raises(self):
        with pytest.raises(ValueError, match="spleen_blood_flow_L_per_h"):
            simulate_spleen_pk(
                drug_name="Drug",
                dose_mg=10.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                spleen_blood_flow_L_per_h=0.0,
            )

    def test_zero_kp_raises(self):
        with pytest.raises(ValueError, match="kp_spleen"):
            simulate_spleen_pk(
                drug_name="Drug",
                dose_mg=10.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                kp_spleen=0.0,
            )

    def test_negative_phago_rate_raises(self):
        with pytest.raises(ValueError, match="phagocytosis_rate_per_h"):
            simulate_spleen_pk(
                drug_name="Drug",
                dose_mg=10.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                phagocytosis_rate_per_h=-0.1,
            )

    def test_zero_t_end_raises(self):
        with pytest.raises(ValueError, match="t_end_h"):
            simulate_spleen_pk(
                drug_name="Drug", dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0, t_end_h=0.0
            )

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt_h"):
            simulate_spleen_pk(drug_name="Drug", dose_mg=10.0, cl_L_per_h=5.0, vd_L=50.0, dt_h=0.0)


class TestSpleenPKResult:
    def _run(self, **kwargs):
        defaults = dict(
            drug_name="TestNano",
            dose_mg=100.0,
            cl_L_per_h=3.0,
            vd_L=50.0,
            spleen_blood_flow_L_per_h=0.9,
            kp_spleen=3.0,
            phagocytosis_rate_per_h=0.1,
            t_end_h=24.0,
            dt_h=0.05,
        )
        defaults.update(kwargs)
        return simulate_spleen_pk(**defaults)

    def test_returns_correct_type(self):
        result = self._run()
        assert isinstance(result, SpleenPKResult)

    def test_drug_name_stored(self):
        result = self._run(drug_name="Nano42")
        assert result.drug_name == "Nano42"

    def test_dose_stored(self):
        result = self._run(dose_mg=50.0)
        assert result.dose_mg == 50.0

    def test_times_start_at_zero(self):
        result = self._run()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self):
        result = self._run(t_end_h=12.0)
        assert result.times_h[-1] == pytest.approx(12.0)

    def test_plasma_cmax_positive(self):
        result = self._run()
        assert result.cmax_plasma > 0.0

    def test_auc_plasma_positive(self):
        result = self._run()
        assert result.auc_plasma > 0.0

    def test_initial_plasma_conc_equals_dose_over_vd_plasma(self):
        result = self._run(dose_mg=100.0, vd_L=50.0, kp_spleen=1.0)
        # vd_spleen = 50 * 0.003 * 1 = 0.15 L; vd_plasma = 50 - 0.15 = 49.85 L
        expected = 100.0 / 49.85
        assert result.c_plasma_mg_L[0] == pytest.approx(expected, rel=0.05)

    def test_plasma_conc_decreases_overall(self):
        result = self._run()
        # plasma should generally decrease from initial bolus
        assert result.c_plasma_mg_L[-1] < result.c_plasma_mg_L[0]

    def test_phagocytosed_mass_monotonically_increases(self):
        result = self._run(phagocytosis_rate_per_h=0.5)
        for i in range(1, len(result.m_phagocytosed_mg)):
            assert result.m_phagocytosed_mg[i] >= result.m_phagocytosed_mg[i - 1] - 1e-9

    def test_no_phagocytosis_zero_sequestration(self):
        result = self._run(phagocytosis_rate_per_h=0.0)
        assert result.f_phagocytosed_24h == pytest.approx(0.0, abs=1e-9)
        assert result.m_phagocytosed_mg[-1] == pytest.approx(0.0, abs=1e-9)

    def test_high_phago_rate_more_sequestration(self):
        result_low = self._run(phagocytosis_rate_per_h=0.01)
        result_high = self._run(phagocytosis_rate_per_h=2.0)
        assert result_high.f_phagocytosed_24h > result_low.f_phagocytosed_24h

    def test_f_phagocytosed_between_0_and_1(self):
        result = self._run(phagocytosis_rate_per_h=1.0)
        assert 0.0 <= result.f_phagocytosed_24h <= 1.0

    def test_higher_kp_increases_spleen_concentration(self):
        result_low = self._run(kp_spleen=1.0)
        result_high = self._run(kp_spleen=10.0)
        max_spleen_low = max(result_low.c_spleen_mg_L)
        max_spleen_high = max(result_high.c_spleen_mg_L)
        assert max_spleen_high > max_spleen_low

    def test_list_lengths_match(self):
        result = self._run()
        n = len(result.times_h)
        assert len(result.c_plasma_mg_L) == n
        assert len(result.c_spleen_mg_L) == n
        assert len(result.m_phagocytosed_mg) == n

    def test_notes_is_string(self):
        result = self._run()
        assert isinstance(result.notes, str)

    def test_nanoparticle_flag_in_notes(self):
        result = self._run(kp_spleen=10.0)
        assert "nanoparticle" in result.notes.lower() or "high kp" in result.notes.lower()

    def test_all_concentrations_nonnegative(self):
        result = self._run(phagocytosis_rate_per_h=2.0)
        for c in result.c_plasma_mg_L:
            assert c >= 0.0
        for c in result.c_spleen_mg_L:
            assert c >= 0.0
        for m in result.m_phagocytosed_mg:
            assert m >= 0.0

    def test_dose_proportionality(self):
        r1 = self._run(dose_mg=10.0)
        r2 = self._run(dose_mg=100.0)
        assert r2.auc_plasma == pytest.approx(r1.auc_plasma * 10.0, rel=0.05)


class TestMPSUptakeAnalysis:
    def _run(self, rates=None):
        kwargs = dict(
            drug_name="NanoParticle",
            dose_mg=100.0,
            cl_L_per_h=3.0,
            vd_L=50.0,
        )
        if rates is not None:
            kwargs["phagocytosis_rates"] = rates
        return mps_uptake_analysis(**kwargs)

    def test_returns_list(self):
        results = self._run()
        assert isinstance(results, list)

    def test_default_five_entries(self):
        results = self._run()
        assert len(results) == 5

    def test_custom_rates_count(self):
        results = self._run(rates=[0.0, 1.0, 3.0])
        assert len(results) == 3

    def test_each_result_has_required_keys(self):
        results = self._run()
        for r in results:
            assert "phagocytosis_rate_per_h" in r
            assert "f_sequestered" in r
            assert "classification" in r
            assert "cmax_plasma_mg_L" in r
            assert "auc_plasma_mg_h_L" in r

    def test_zero_rate_zero_sequestration(self):
        results = self._run(rates=[0.0])
        assert results[0]["f_sequestered"] == pytest.approx(0.0, abs=1e-9)

    def test_higher_rate_more_sequestration(self):
        results = self._run(rates=[0.1, 2.0])
        assert results[1]["f_sequestered"] >= results[0]["f_sequestered"]

    def test_classification_strings(self):
        results = self._run()
        valid = {
            "Low MPS uptake",
            "Moderate MPS uptake",
            "High MPS uptake",
            "Extensive MPS sequestration",
        }
        for r in results:
            assert r["classification"] in valid

    def test_empty_rates_raises(self):
        with pytest.raises(ValueError, match="phagocytosis_rates"):
            mps_uptake_analysis(
                drug_name="Drug",
                dose_mg=10.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                phagocytosis_rates=[],
            )

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError):
            mps_uptake_analysis(
                drug_name="Drug",
                dose_mg=10.0,
                cl_L_per_h=5.0,
                vd_L=50.0,
                phagocytosis_rates=[-0.5],
            )

    def test_f_sequestered_between_0_and_1(self):
        results = self._run(rates=[0.0, 0.5, 1.0, 5.0])
        for r in results:
            assert 0.0 <= r["f_sequestered"] <= 1.0
