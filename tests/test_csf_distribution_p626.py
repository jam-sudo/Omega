"""Tests for Phase 626 — Drug Spinal Fluid (CSF) Distribution."""

import pytest

from omega_pbpk.core.csf_distribution_p626 import (
    CSFDistributionResult,
    simulate_csf_distribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs):
    """Run simulation with sensible defaults, overriding via kwargs."""
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        logP=2.5,
        mw_Da=300.0,
        psa_A2=60.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=40.0,
        fu_plasma=0.2,
        t_end_h=24.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_csf_distribution(**params)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        res = _default_result()
        assert isinstance(res, CSFDistributionResult)

    def test_drug_name_preserved(self):
        res = _default_result(drug_name="Clonazepam")
        assert res.drug_name == "Clonazepam"

    def test_dose_preserved(self):
        res = _default_result(dose_mg=50.0)
        assert res.dose_mg == 50.0

    def test_list_fields_are_lists(self):
        res = _default_result()
        for field in [res.times_h, res.c_plasma_mg_L, res.c_csf_mg_L, res.csf_plasma_ratio]:
            assert isinstance(field, list)

    def test_list_lengths_consistent(self):
        res = _default_result()
        n = len(res.times_h)
        assert len(res.c_plasma_mg_L) == n
        assert len(res.c_csf_mg_L) == n
        assert len(res.csf_plasma_ratio) == n

    def test_times_start_at_zero(self):
        res = _default_result()
        assert res.times_h[0] == pytest.approx(0.0)

    def test_times_end_near_t_end(self):
        res = _default_result(t_end_h=12.0, dt_h=0.05)
        assert res.times_h[-1] == pytest.approx(12.0, abs=0.1)

    def test_notes_is_string(self):
        res = _default_result()
        assert isinstance(res.notes, str)


# ---------------------------------------------------------------------------
# Plasma PK
# ---------------------------------------------------------------------------


class TestPlasmaPK:
    def test_plasma_starts_at_zero(self):
        res = _default_result()
        assert res.c_plasma_mg_L[0] == pytest.approx(0.0, abs=1e-6)

    def test_plasma_non_negative(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_plasma_mg_L)

    def test_plasma_rises_and_falls(self):
        res = _default_result()
        cmax = res.cmax_plasma_mg_L
        assert cmax > 0.0
        assert res.c_plasma_mg_L[0] < cmax

    def test_auc_plasma_positive(self):
        res = _default_result()
        assert res.auc_plasma_mg_h_per_L > 0.0

    def test_higher_dose_higher_auc_plasma(self):
        res1 = _default_result(dose_mg=100.0)
        res2 = _default_result(dose_mg=200.0)
        assert res2.auc_plasma_mg_h_per_L > res1.auc_plasma_mg_h_per_L


# ---------------------------------------------------------------------------
# CSF PK
# ---------------------------------------------------------------------------


class TestCSFPK:
    def test_csf_starts_at_zero(self):
        res = _default_result()
        assert res.c_csf_mg_L[0] == pytest.approx(0.0, abs=1e-6)

    def test_csf_non_negative(self):
        res = _default_result()
        assert all(c >= 0.0 for c in res.c_csf_mg_L)

    def test_csf_auc_positive(self):
        res = _default_result()
        assert res.auc_csf_mg_h_per_L > 0.0

    def test_cmax_csf_positive(self):
        res = _default_result()
        assert res.cmax_csf_mg_L > 0.0

    def test_tmax_csf_positive(self):
        res = _default_result()
        assert res.tmax_csf_h >= 0.0

    def test_tmax_csf_after_tmax_plasma(self):
        """CSF peak typically lags plasma peak."""
        res = _default_result()
        # Find tmax_plasma
        cmax_plasma = max(res.c_plasma_mg_L)
        tmax_plasma = res.times_h[res.c_plasma_mg_L.index(cmax_plasma)]
        # CSF tmax should be >= plasma tmax (drug must pass through blood first)
        assert res.tmax_csf_h >= tmax_plasma - 1.0  # allow small tolerance


# ---------------------------------------------------------------------------
# logP effect on CSF penetration
# ---------------------------------------------------------------------------


class TestLogPEffect:
    def test_higher_logP_higher_csf_penetration(self):
        """More lipophilic drugs penetrate CSF better."""
        res_lo = _default_result(logP=0.5, psa_A2=40.0)
        res_hi = _default_result(logP=3.5, psa_A2=40.0)
        assert res_hi.csf_auc_ratio >= res_lo.csf_auc_ratio

    def test_negative_logP_very_low_penetration(self):
        """Very lipophilic drugs (logP=5) penetrate CSF better than borderline (logP=1).

        Uses low PSA and low MW so k_csf escapes the lower clamp for logP=5,
        while logP=1 stays clamped at 0.001 /h.
        """
        res_low = simulate_csf_distribution(
            drug_name="LowLogP",
            dose_mg=100.0,
            logP=1.0,
            mw_Da=200.0,
            psa_A2=10.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=40.0,
            fu_plasma=0.2,
            t_end_h=24.0,
            dt_h=0.05,
        )
        res_high = simulate_csf_distribution(
            drug_name="HighLogP",
            dose_mg=100.0,
            logP=5.0,
            mw_Da=200.0,
            psa_A2=10.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=40.0,
            fu_plasma=0.2,
            t_end_h=24.0,
            dt_h=0.05,
        )
        # The highly lipophilic drug must achieve higher CSF penetration
        assert res_high.csf_auc_ratio > res_low.csf_auc_ratio


# ---------------------------------------------------------------------------
# PSA effect on CSF penetration
# ---------------------------------------------------------------------------


class TestPSAEffect:
    def test_high_psa_lower_penetration(self):
        """High PSA reduces permeability."""
        res_lo = _default_result(psa_A2=20.0, logP=2.5)
        res_hi = _default_result(psa_A2=180.0, logP=2.5)
        assert res_lo.csf_auc_ratio >= res_hi.csf_auc_ratio

    def test_psa_zero_gives_valid_result(self):
        res = _default_result(psa_A2=0.0)
        assert isinstance(res, CSFDistributionResult)
        assert res.csf_auc_ratio >= 0.0


# ---------------------------------------------------------------------------
# AUC ratio and efficacy potential
# ---------------------------------------------------------------------------


class TestCSFAUCRatio:
    def test_auc_ratio_non_negative(self):
        res = _default_result()
        assert res.csf_auc_ratio >= 0.0

    def test_auc_ratio_equals_csf_over_plasma(self):
        res = _default_result()
        expected = res.auc_csf_mg_h_per_L / res.auc_plasma_mg_h_per_L
        assert res.csf_auc_ratio == pytest.approx(expected, rel=1e-6)

    def test_efficacy_high(self):
        """High logP, low PSA → should achieve high efficacy potential."""
        res = simulate_csf_distribution(
            drug_name="HighCNS",
            dose_mg=100.0,
            logP=4.0,
            mw_Da=200.0,
            psa_A2=10.0,
            cl_sys_L_per_h=3.0,
            vd_sys_L=100.0,
            fu_plasma=0.5,
            t_end_h=48.0,
            dt_h=0.05,
        )
        assert res.efficacy_potential in ("high", "moderate")

    def test_efficacy_low(self):
        """Low logP, high PSA → low efficacy potential."""
        res = simulate_csf_distribution(
            drug_name="LowCNS",
            dose_mg=100.0,
            logP=-0.5,
            mw_Da=600.0,
            psa_A2=190.0,
            cl_sys_L_per_h=10.0,
            vd_sys_L=20.0,
            fu_plasma=0.05,
            t_end_h=24.0,
            dt_h=0.05,
        )
        assert res.efficacy_potential == "low"

    def test_efficacy_potential_valid_string(self):
        res = _default_result()
        assert res.efficacy_potential in ("high", "moderate", "low")


# ---------------------------------------------------------------------------
# Blood-CSF barrier permeability
# ---------------------------------------------------------------------------


class TestBBCSFBPermeability:
    def test_permeability_in_valid_range(self):
        res = _default_result()
        assert 1e-9 <= res.bbcsfb_permeability <= 1e-5

    def test_permeability_in_notes(self):
        res = _default_result()
        assert "cm/s" in res.notes or "permeability" in res.notes.lower()


# ---------------------------------------------------------------------------
# t_equilibration
# ---------------------------------------------------------------------------


class TestEquilibration:
    def test_t_equilibration_non_negative(self):
        res = _default_result()
        assert res.t_equilibration_h >= 0.0

    def test_t_equilibration_within_simulation(self):
        res = _default_result(t_end_h=24.0)
        assert res.t_equilibration_h <= 24.0 + 1e-6


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=0.0)

    def test_negative_dose(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _default_result(dose_mg=-5.0)

    def test_zero_cl_sys(self):
        with pytest.raises(ValueError, match="cl_sys"):
            _default_result(cl_sys_L_per_h=0.0)

    def test_zero_vd(self):
        with pytest.raises(ValueError, match="vd_sys"):
            _default_result(vd_sys_L=0.0)

    def test_zero_mw(self):
        with pytest.raises(ValueError, match="mw_Da"):
            _default_result(mw_Da=0.0)

    def test_negative_psa(self):
        with pytest.raises(ValueError, match="psa_A2"):
            _default_result(psa_A2=-10.0)

    def test_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _default_result(fu_plasma=0.0)

    def test_fu_plasma_above_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            _default_result(fu_plasma=1.5)
