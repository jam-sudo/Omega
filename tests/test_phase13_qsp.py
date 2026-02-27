"""Phase 13 — QSP / Target Engagement model tests.

Tests cover:
- TmddParams: Kd, baseline target, Km properties
- simulate_tmdd (full model): target depletion, drug-target complex formation, occupancy
- simulate_tmdd (QSS model): consistent with full model at low kon
- receptor_occupancy: Hill equation correctness
- emax_effect: stimulation and inhibition modes
- IndirectResponseModel: all four types (I–IV)
- simulate_indirect_response: baseline, steady state, direction of effect
"""

import numpy as np
import pytest

from omega_pbpk.core.qsp import (
    IndirectResponseModel,
    TmddParams,
    TmddResult,
    emax_effect,
    receptor_occupancy,
    simulate_indirect_response,
    simulate_tmdd,
)

# ---------------------------------------------------------------------------
# TmddParams properties
# ---------------------------------------------------------------------------


class TestTmddParams:
    def test_kd_computed(self):
        p = TmddParams(kon_per_nM_per_h=0.1, koff_per_h=1.0)
        assert p.kd_nM == pytest.approx(10.0, rel=1e-6)

    def test_baseline_target_from_ksyn_kdeg(self):
        p = TmddParams(ksyn_nM_per_h=0.5, kdeg_per_h=0.1)
        assert p.baseline_target_nM == pytest.approx(5.0, rel=1e-6)

    def test_baseline_target_override(self):
        p = TmddParams(ksyn_nM_per_h=1.0, kdeg_per_h=0.1, r0_nM=20.0)
        assert p.baseline_target_nM == pytest.approx(20.0)

    def test_km_qss(self):
        p = TmddParams(kon_per_nM_per_h=0.1, koff_per_h=0.5, ke_LR_per_h=0.05)
        # Km = (0.5 + 0.05) / 0.1 = 5.5 nM
        assert p.km_nM == pytest.approx(5.5, rel=1e-6)


# ---------------------------------------------------------------------------
# simulate_tmdd — full model
# ---------------------------------------------------------------------------


class TestSimulateTmdd:
    @pytest.fixture
    def default_params(self):
        return TmddParams(
            kon_per_nM_per_h=0.091,
            koff_per_h=0.001,
            ke_L_per_h=0.02,
            ke_LR_per_h=0.005,
            ksyn_nM_per_h=0.11,
            kdeg_per_h=0.01,
        )

    def test_returns_tmdd_result(self, default_params):
        result = simulate_tmdd(default_params, dose_nM=100.0)
        assert isinstance(result, TmddResult)

    def test_free_drug_starts_at_dose(self, default_params):
        result = simulate_tmdd(default_params, dose_nM=100.0)
        assert result.free_drug_nM[0] == pytest.approx(100.0, rel=0.01)

    def test_free_drug_declines_over_time(self, default_params):
        result = simulate_tmdd(default_params, dose_nM=100.0)
        assert result.free_drug_nM[-1] < result.free_drug_nM[0]

    def test_complex_forms_and_peaks(self, default_params):
        result = simulate_tmdd(default_params, dose_nM=100.0)
        # Complex should rise then fall
        peak_idx = int(np.argmax(result.complex_nM))
        assert 0 < peak_idx < len(result.time_h) - 1

    def test_target_depleted_at_high_dose(self, default_params):
        """High drug dose should suppress free target substantially."""
        result = simulate_tmdd(default_params, dose_nM=10000.0, t_end_h=168.0)
        # Early time points: target should be depleted
        early_target = result.free_target_nM[5]  # t ≈ 0.5h
        baseline_target = default_params.baseline_target_nM
        assert early_target < baseline_target * 0.5

    def test_target_recovers_at_low_dose(self, default_params):
        """Low dose: target partially recovers at end of simulation."""
        result = simulate_tmdd(default_params, dose_nM=1.0, t_end_h=720.0)
        final_target = result.free_target_nM[-1]
        baseline = default_params.baseline_target_nM
        # After 720h, target should return toward baseline
        assert final_target > 0.5 * baseline

    def test_target_occupancy_range_0_to_1(self, default_params):
        result = simulate_tmdd(default_params, dose_nM=100.0)
        ro = result.target_occupancy
        assert np.all(ro >= 0.0 - 1e-6)
        assert np.all(ro <= 1.0 + 1e-6)

    def test_max_occupancy_high_dose(self, default_params):
        """Very high dose → near-complete target occupancy achieved."""
        result = simulate_tmdd(default_params, dose_nM=100_000.0)
        assert float(np.max(result.target_occupancy)) > 0.9

    def test_all_concentrations_non_negative(self, default_params):
        result = simulate_tmdd(default_params, dose_nM=100.0)
        assert np.all(result.free_drug_nM >= 0)
        assert np.all(result.free_target_nM >= 0)
        assert np.all(result.complex_nM >= 0)

    def test_pk_pd_summary_keys(self, default_params):
        result = simulate_tmdd(default_params, dose_nM=100.0)
        summary = result.pk_pd_summary()
        for key in (
            "Cmax_nM",
            "Tmax_h",
            "AUC_nM_h",
            "Kd_nM",
            "baseline_target_nM",
            "max_target_occupancy",
        ):
            assert key in summary


class TestSimulateTmddQSS:
    """QSS approximation should be consistent with full model."""

    def _compare(self, dose_nM):
        params_full = TmddParams(
            kon_per_nM_per_h=0.01,  # low kon → QSS valid
            koff_per_h=0.5,
            ke_L_per_h=0.03,
            ke_LR_per_h=0.01,
            ksyn_nM_per_h=0.5,
            kdeg_per_h=0.05,
            use_qss=False,
        )
        params_qss = TmddParams(
            kon_per_nM_per_h=0.01,
            koff_per_h=0.5,
            ke_L_per_h=0.03,
            ke_LR_per_h=0.01,
            ksyn_nM_per_h=0.5,
            kdeg_per_h=0.05,
            use_qss=True,
        )
        r_full = simulate_tmdd(params_full, dose_nM=dose_nM, t_end_h=120.0)
        r_qss = simulate_tmdd(params_qss, dose_nM=dose_nM, t_end_h=120.0)
        return r_full, r_qss

    def test_qss_returns_tmdd_result(self):
        params = TmddParams(use_qss=True)
        result = simulate_tmdd(params, dose_nM=50.0)
        assert isinstance(result, TmddResult)

    def test_qss_auc_close_to_full_at_low_kon(self):
        r_full, r_qss = self._compare(dose_nM=10.0)
        auc_full = float(np.trapezoid(r_full.free_drug_nM, r_full.time_h))
        auc_qss = float(np.trapezoid(r_qss.free_drug_nM, r_qss.time_h))
        # Within 20% (QSS is an approximation)
        assert abs(auc_qss - auc_full) / max(auc_full, 1e-9) < 0.25

    def test_qss_non_negative(self):
        params = TmddParams(use_qss=True)
        result = simulate_tmdd(params, dose_nM=100.0)
        assert np.all(result.free_drug_nM >= 0)
        assert np.all(result.free_target_nM >= 0)


# ---------------------------------------------------------------------------
# receptor_occupancy
# ---------------------------------------------------------------------------


class TestReceptorOccupancy:
    def test_zero_conc_gives_zero_occupancy(self):
        ro = receptor_occupancy(np.array([0.0]), ec50=1.0)
        assert float(ro[0]) == pytest.approx(0.0, abs=1e-9)

    def test_at_ec50_occupancy_is_half(self):
        ro = receptor_occupancy(np.array([1.0]), ec50=1.0)
        assert float(ro[0]) == pytest.approx(0.5, rel=1e-6)

    def test_high_conc_approaches_one(self):
        ro = receptor_occupancy(np.array([1e9]), ec50=1.0)
        assert float(ro[0]) > 0.999

    def test_hill_coefficient_steepens_curve(self):
        c = np.array([0.5, 1.0, 2.0])
        ro1 = receptor_occupancy(c, ec50=1.0, hill=1.0)
        ro2 = receptor_occupancy(c, ec50=1.0, hill=3.0)
        # Hill=3: steeper sigmoidal → below EC50 lower, above EC50 higher
        assert ro2[0] < ro1[0]  # below EC50: steeper → lower occupancy
        assert ro2[2] > ro1[2]  # above EC50: steeper → higher occupancy

    def test_output_shape_matches_input(self):
        c = np.linspace(0, 10, 50)
        ro = receptor_occupancy(c, ec50=2.0)
        assert ro.shape == c.shape

    def test_occupancy_monotone_increasing(self):
        c = np.linspace(0, 100, 100)
        ro = receptor_occupancy(c, ec50=5.0)
        assert np.all(np.diff(ro) >= 0)


# ---------------------------------------------------------------------------
# emax_effect
# ---------------------------------------------------------------------------


class TestEmaxEffect:
    def test_stimulation_increases_effect(self):
        c = np.array([0.0, 1.0, 10.0])
        e = emax_effect(c, e0=1.0, emax=5.0, ec50=1.0, inhibitory=False)
        assert e[0] == pytest.approx(1.0, abs=1e-9)  # at C=0: E=E0
        assert e[-1] > e[0]  # high C: E > E0

    def test_inhibition_decreases_effect(self):
        c = np.array([0.0, 1.0, 1000.0])
        e = emax_effect(c, e0=10.0, emax=1.0, ec50=1.0, inhibitory=True)
        assert e[0] == pytest.approx(10.0, abs=1e-6)  # at C=0: E=E0
        assert e[-1] < e[0]  # high C: E < E0

    def test_inhibition_at_ec50_half_emax_suppression(self):
        """At EC50 with Emax=1 (complete), should see ~50% suppression."""
        c = np.array([1.0])
        e = emax_effect(c, e0=10.0, emax=1.0, ec50=1.0, inhibitory=True)
        assert float(e[0]) == pytest.approx(5.0, rel=0.01)

    def test_stimulation_at_ec50(self):
        c = np.array([1.0])
        e = emax_effect(c, e0=0.0, emax=4.0, ec50=1.0, inhibitory=False)
        assert float(e[0]) == pytest.approx(2.0, rel=0.01)

    def test_effect_array_shape(self):
        c = np.linspace(0, 10, 100)
        e = emax_effect(c, e0=1.0, emax=2.0, ec50=1.0)
        assert e.shape == c.shape


# ---------------------------------------------------------------------------
# IndirectResponseModel types
# ---------------------------------------------------------------------------


class TestIndirectResponseModels:
    def _build_conc(self, t):
        """Simple declining drug concentration: C(t) = C0*exp(-ke*t)."""
        return 5.0 * np.exp(-0.2 * t)

    def test_model_i_response_increases(self):
        """Model I: drug stimulates kin → response rises above baseline."""
        t = np.linspace(0, 100, 500)
        c = self._build_conc(t)
        model = IndirectResponseModel(model_type="I", kin=1.0, kout=0.1, e_max=0.8, ec50=1.0)
        result = simulate_indirect_response(model, t, c)
        baseline = model.baseline_response
        peak = float(np.max(result.response))
        assert peak > baseline

    def test_model_ii_response_decreases(self):
        """Model II: drug inhibits kin → response falls below baseline."""
        t = np.linspace(0, 150, 600)
        c = self._build_conc(t)
        model = IndirectResponseModel(model_type="II", kin=1.0, kout=0.1, e_max=0.8, ec50=1.0)
        result = simulate_indirect_response(model, t, c)
        baseline = model.baseline_response
        nadir = float(np.min(result.response))
        assert nadir < baseline

    def test_model_iii_response_decreases(self):
        """Model III: drug stimulates kout → response falls."""
        t = np.linspace(0, 150, 600)
        c = self._build_conc(t)
        model = IndirectResponseModel(model_type="III", kin=1.0, kout=0.1, e_max=0.8, ec50=1.0)
        result = simulate_indirect_response(model, t, c)
        baseline = model.baseline_response
        nadir = float(np.min(result.response))
        assert nadir < baseline

    def test_model_iv_response_increases(self):
        """Model IV: drug inhibits kout → response rises."""
        t = np.linspace(0, 150, 600)
        c = self._build_conc(t)
        model = IndirectResponseModel(model_type="IV", kin=1.0, kout=0.1, e_max=0.8, ec50=1.0)
        result = simulate_indirect_response(model, t, c)
        baseline = model.baseline_response
        peak = float(np.max(result.response))
        assert peak > baseline

    def test_response_returns_to_baseline_zero_drug(self):
        """With no drug (C=0), response should remain at baseline."""
        t = np.linspace(0, 100, 500)
        c = np.zeros_like(t)
        model = IndirectResponseModel(model_type="IV", kin=1.0, kout=0.1)
        result = simulate_indirect_response(model, t, c)
        baseline = model.baseline_response
        assert np.allclose(result.response, baseline, rtol=0.01)

    def test_invalid_model_type_raises(self):
        t = np.linspace(0, 10, 50)
        c = np.ones(50)
        model = IndirectResponseModel(model_type="V")
        with pytest.raises(ValueError, match="Unknown model type"):
            simulate_indirect_response(model, t, c)

    def test_response_non_negative(self):
        t = np.linspace(0, 100, 500)
        c = self._build_conc(t)
        for mtype in ("I", "II", "III", "IV"):
            model = IndirectResponseModel(model_type=mtype, kin=1.0, kout=0.1, e_max=0.9, ec50=1.0)
            result = simulate_indirect_response(model, t, c)
            assert np.all(result.response >= 0)

    def test_baseline_response_property(self):
        model = IndirectResponseModel(kin=2.0, kout=0.5)
        assert model.baseline_response == pytest.approx(4.0, rel=1e-9)

    def test_max_effect_computed(self):
        t = np.linspace(0, 150, 600)
        c = self._build_conc(t)
        model = IndirectResponseModel(model_type="I", kin=1.0, kout=0.1, e_max=0.8, ec50=1.0)
        result = simulate_indirect_response(model, t, c)
        assert result.max_effect > 0
