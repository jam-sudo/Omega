"""Tests for Phase 222 — Epidural Drug Delivery PK Model."""

import pytest

from omega_pbpk.core.epidural_pk import (
    EpiduralPKResult,
    compare_epidural_doses,
    simulate_epidural_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs):
    params = dict(
        drug_name="Bupivacaine",
        dose_mg=10.0,
        cl_L_per_h=0.5,
        vd_sys_L=40.0,
        t_end_h=12.0,
        dt_h=0.05,
    )
    params.update(kwargs)
    return simulate_epidural_pk(**params)


# ---------------------------------------------------------------------------
# Return type and field tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_epidural_pk_result(self):
        result = _default_result()
        assert isinstance(result, EpiduralPKResult)

    def test_drug_name_preserved(self):
        result = simulate_epidural_pk("Ropivacaine", 15.0, 0.6, 50.0, t_end_h=8.0, dt_h=0.1)
        assert result.drug_name == "Ropivacaine"

    def test_dose_mg_preserved(self):
        result = _default_result(dose_mg=20.0)
        assert result.dose_mg == 20.0

    def test_times_h_is_list(self):
        result = _default_result()
        assert isinstance(result.times_h, list)
        assert len(result.times_h) > 0

    def test_concentration_lists_same_length(self):
        result = _default_result()
        n = len(result.times_h)
        assert len(result.c_epidural_mg_mL) == n
        assert len(result.c_csf_mg_mL) == n
        assert len(result.c_spinal_mg_g) == n
        assert len(result.c_systemic_mg_L) == n

    def test_notes_non_empty(self):
        result = _default_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Initial condition tests
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_epidural_starts_at_dose_over_v_epi(self):
        """C_epidural(0) = dose / V_epi = 10 / 30 mg/mL."""
        result = _default_result(dose_mg=10.0)
        expected = 10.0 / 30.0
        assert abs(result.c_epidural_mg_mL[0] - expected) < 1e-9

    def test_csf_starts_at_zero(self):
        result = _default_result()
        assert result.c_csf_mg_mL[0] == 0.0

    def test_spinal_starts_at_zero(self):
        result = _default_result()
        assert result.c_spinal_mg_g[0] == 0.0

    def test_systemic_starts_at_zero(self):
        result = _default_result()
        assert result.c_systemic_mg_L[0] == 0.0

    def test_epidural_starts_positive(self):
        result = _default_result()
        assert result.c_epidural_mg_mL[0] > 0.0


# ---------------------------------------------------------------------------
# Dynamics tests
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_epidural_decreases_over_time(self):
        """Epidural compartment should monotonically decrease (net outflow only)."""
        result = _default_result()
        assert result.c_epidural_mg_mL[-1] < result.c_epidural_mg_mL[0]

    def test_csf_rises_then_falls(self):
        """CSF concentration should first rise then fall."""
        result = _default_result(t_end_h=24.0)
        peak_idx = result.c_csf_mg_mL.index(max(result.c_csf_mg_mL))
        assert peak_idx > 0, "CSF should peak after t=0"
        assert result.c_csf_mg_mL[-1] < result.c_csf_mg_mL[peak_idx], (
            "CSF should decline after peak"
        )

    def test_spinal_rises_from_zero(self):
        """Spinal cord concentration should increase from zero."""
        result = _default_result()
        assert max(result.c_spinal_mg_g) > 0.0

    def test_systemic_rises_from_zero(self):
        result = _default_result()
        assert max(result.c_systemic_mg_L) > 0.0

    def test_cmax_csf_positive(self):
        result = _default_result()
        assert result.cmax_csf_mg_mL > 0.0

    def test_tmax_csf_positive(self):
        result = _default_result()
        assert result.tmax_csf_h > 0.0

    def test_auc_csf_positive(self):
        result = _default_result()
        assert result.auc_csf_mg_h_per_mL > 0.0

    def test_auc_spinal_positive(self):
        result = _default_result()
        assert result.auc_spinal_mg_h_per_g > 0.0

    def test_cmax_systemic_positive(self):
        result = _default_result()
        assert result.cmax_systemic_mg_L > 0.0


# ---------------------------------------------------------------------------
# Dose proportionality tests
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_larger_dose_larger_cmax_csf(self):
        r10 = simulate_epidural_pk("Drug", 10.0, 0.5, 40.0, t_end_h=12.0, dt_h=0.05)
        r20 = simulate_epidural_pk("Drug", 20.0, 0.5, 40.0, t_end_h=12.0, dt_h=0.05)
        assert r20.cmax_csf_mg_mL > r10.cmax_csf_mg_mL

    def test_larger_dose_larger_auc_spinal(self):
        r10 = simulate_epidural_pk("Drug", 10.0, 0.5, 40.0, t_end_h=12.0, dt_h=0.05)
        r20 = simulate_epidural_pk("Drug", 20.0, 0.5, 40.0, t_end_h=12.0, dt_h=0.05)
        assert r20.auc_spinal_mg_h_per_g > r10.auc_spinal_mg_h_per_g


# ---------------------------------------------------------------------------
# compare_epidural_doses tests
# ---------------------------------------------------------------------------


class TestCompareEpiduralDoses:
    def test_returns_list(self):
        results = compare_epidural_doses(
            "Drug", [5.0, 10.0, 20.0], 0.5, 40.0, t_end_h=12.0, dt_h=0.1
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_spinal_auc_descending(self):
        results = compare_epidural_doses(
            "Drug", [5.0, 10.0, 20.0], 0.5, 40.0, t_end_h=12.0, dt_h=0.1
        )
        for i in range(len(results) - 1):
            assert results[i].auc_spinal_mg_h_per_g >= results[i + 1].auc_spinal_mg_h_per_g

    def test_highest_dose_first(self):
        results = compare_epidural_doses(
            "Drug", [5.0, 10.0, 20.0], 0.5, 40.0, t_end_h=12.0, dt_h=0.1
        )
        assert results[0].dose_mg == pytest.approx(20.0)

    def test_all_drug_names_preserved(self):
        results = compare_epidural_doses(
            "Lidocaine", [5.0, 10.0], 0.5, 40.0, t_end_h=12.0, dt_h=0.1
        )
        assert all(r.drug_name == "Lidocaine" for r in results)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg must be > 0"):
            simulate_epidural_pk("Drug", 0.0, 0.5, 40.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg must be > 0"):
            simulate_epidural_pk("Drug", -5.0, 0.5, 40.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
            simulate_epidural_pk("Drug", 10.0, 0.0, 40.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
            simulate_epidural_pk("Drug", 10.0, -1.0, 40.0)
