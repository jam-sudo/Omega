"""Tests for Phase 435 — Epidural Drug PK Model (3-compartment)."""

import pytest

from omega_pbpk.core.epidural_pk_p435 import (
    EpiduralPKResult,
    compare_epidural_drugs,
    simulate_epidural_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default(**kwargs):
    params = dict(
        drug_name="Bupivacaine",
        dose_mg=10.0,
        cl_sys_L_per_h=0.5,
        vd_sys_L=20.0,
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
        result = _default()
        assert isinstance(result, EpiduralPKResult)

    def test_drug_name_preserved(self):
        result = simulate_epidural_pk("Ropivacaine", 15.0, 0.6)
        assert result.drug_name == "Ropivacaine"

    def test_dose_mg_preserved(self):
        result = _default(dose_mg=25.0)
        assert result.dose_mg == 25.0

    def test_times_h_is_list(self):
        result = _default()
        assert isinstance(result.times_h, list)
        assert len(result.times_h) > 0

    def test_arrays_consistent_length(self):
        result = _default()
        n = len(result.times_h)
        assert len(result.c_epidural_mg_mL) == n
        assert len(result.c_csf_mg_mL) == n
        assert len(result.c_systemic_mg_L) == n

    def test_times_start_at_zero(self):
        result = _default()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_notes_non_empty(self):
        result = _default()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_epidural_starts_at_dose_over_v_epidural(self):
        """C_epidural(0) = dose_mg / v_epidural_mL."""
        result = simulate_epidural_pk("Drug", 10.0, 0.5, v_epidural_mL=1.0)
        expected = 10.0 / 1.0
        assert result.c_epidural_mg_mL[0] == pytest.approx(expected)

    def test_csf_starts_at_zero(self):
        result = _default()
        assert result.c_csf_mg_mL[0] == pytest.approx(0.0)

    def test_systemic_starts_at_zero(self):
        result = _default()
        assert result.c_systemic_mg_L[0] == pytest.approx(0.0)

    def test_epidural_starts_positive(self):
        result = _default()
        assert result.c_epidural_mg_mL[0] > 0.0


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


class TestSummaryMetrics:
    def test_cmax_csf_positive(self):
        result = _default()
        assert result.cmax_csf_mg_mL > 0.0

    def test_auc_csf_positive(self):
        result = _default()
        assert result.auc_csf_mg_h_per_mL > 0.0

    def test_auc_systemic_positive(self):
        result = _default()
        assert result.auc_systemic_mg_h_per_L > 0.0

    def test_systemic_absorption_pct_between_0_and_100(self):
        result = _default()
        assert 0.0 < result.systemic_absorption_pct < 100.0

    def test_t_half_epidural_positive(self):
        result = _default()
        assert result.t_half_epidural_h > 0.0

    def test_csf_to_epidural_ratio_positive(self):
        result = _default()
        assert result.csf_to_epidural_ratio > 0.0

    def test_cmax_epidural_positive(self):
        result = _default()
        assert result.cmax_epidural_mg_mL > 0.0

    def test_tmax_csf_nonnegative(self):
        result = _default()
        assert result.tmax_csf_h >= 0.0

    def test_t_half_analytical(self):
        """t_half = 0.693 / (k_epi_to_csf + k_epi_to_sys + k_epi_fat)."""
        result = simulate_epidural_pk(
            "Drug", 10.0, 0.5, k_epi_to_csf=0.3, k_epi_to_sys=0.5, k_epi_fat=0.2
        )
        expected = 0.693 / (0.3 + 0.5 + 0.2)
        assert result.t_half_epidural_h == pytest.approx(expected, rel=1e-4)

    def test_systemic_absorption_pct_analytical(self):
        """systemic_absorption_pct = k_epi_to_sys / k_total * 100."""
        result = simulate_epidural_pk(
            "Drug", 10.0, 0.5, k_epi_to_csf=0.3, k_epi_to_sys=0.5, k_epi_fat=0.2
        )
        expected = 0.5 / (0.3 + 0.5 + 0.2) * 100.0
        assert result.systemic_absorption_pct == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------


class TestParameterSensitivity:
    def test_higher_k_epi_to_csf_gives_higher_csf_penetration(self):
        r_low = simulate_epidural_pk(
            "Drug", 10.0, 0.5, k_epi_to_csf=0.1, k_epi_to_sys=0.5, k_epi_fat=0.2
        )
        r_high = simulate_epidural_pk(
            "Drug", 10.0, 0.5, k_epi_to_csf=0.6, k_epi_to_sys=0.5, k_epi_fat=0.2
        )
        assert r_high.auc_csf_mg_h_per_mL > r_low.auc_csf_mg_h_per_mL

    def test_dose_linearity_cmax_epidural(self):
        """2x dose should give 2x cmax_epidural."""
        r1 = simulate_epidural_pk("Drug", 10.0, 0.5)
        r2 = simulate_epidural_pk("Drug", 20.0, 0.5)
        assert r2.cmax_epidural_mg_mL == pytest.approx(2.0 * r1.cmax_epidural_mg_mL, rel=1e-6)

    def test_dose_linearity_cmax_csf(self):
        """2x dose should give 2x cmax_csf (linear system)."""
        r1 = simulate_epidural_pk("Drug", 10.0, 0.5)
        r2 = simulate_epidural_pk("Drug", 20.0, 0.5)
        assert r2.cmax_csf_mg_mL == pytest.approx(2.0 * r1.cmax_csf_mg_mL, rel=1e-4)


# ---------------------------------------------------------------------------
# compare_epidural_drugs
# ---------------------------------------------------------------------------


class TestCompareEpiduralDrugs:
    def _drug_params(self, name, k_csf):
        return dict(
            drug_name=name,
            dose_mg=10.0,
            cl_sys_L_per_h=0.5,
            k_epi_to_csf=k_csf,
        )

    def test_returns_list(self):
        drugs = [self._drug_params("A", 0.2), self._drug_params("B", 0.5)]
        results = compare_epidural_drugs(drugs)
        assert isinstance(results, list)

    def test_correct_length(self):
        drugs = [
            self._drug_params("A", 0.1),
            self._drug_params("B", 0.4),
            self._drug_params("C", 0.7),
        ]
        results = compare_epidural_drugs(drugs)
        assert len(results) == 3

    def test_sorted_descending_by_csf_to_epidural_ratio(self):
        drugs = [
            self._drug_params("LowCSF", 0.1),
            self._drug_params("MidCSF", 0.4),
            self._drug_params("HighCSF", 0.8),
        ]
        results = compare_epidural_drugs(drugs)
        for i in range(len(results) - 1):
            assert results[i].csf_to_epidural_ratio >= results[i + 1].csf_to_epidural_ratio


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises_value_error(self):
        with pytest.raises(ValueError):
            simulate_epidural_pk("Drug", 0.0, 0.5)

    def test_negative_dose_raises_value_error(self):
        with pytest.raises(ValueError):
            simulate_epidural_pk("Drug", -5.0, 0.5)

    def test_zero_cl_sys_raises_value_error(self):
        with pytest.raises(ValueError):
            simulate_epidural_pk("Drug", 10.0, 0.0)

    def test_negative_cl_sys_raises_value_error(self):
        with pytest.raises(ValueError):
            simulate_epidural_pk("Drug", 10.0, -1.0)

    def test_zero_vd_raises_value_error(self):
        with pytest.raises(ValueError):
            simulate_epidural_pk("Drug", 10.0, 0.5, vd_sys_L=0.0)
