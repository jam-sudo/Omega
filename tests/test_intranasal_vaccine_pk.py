"""Tests for Phase 512 — Intranasal Vaccine Delivery PK."""

import pytest

from omega_pbpk.core.intranasal_vaccine_pk import (
    IntranasalVaccinePKResult,
    compare_dose_levels,
    simulate_intranasal_vaccine_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(dose=100.0, k_clear=2.0, k_uptake=0.5, t_end=48.0, dt=0.1):
    return simulate_intranasal_vaccine_pk(
        drug_name="TestVaccine",
        dose_mcg=dose,
        k_clear=k_clear,
        k_uptake=k_uptake,
        t_end_h=t_end,
        dt_h=dt,
    )


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = _sim()
        assert isinstance(result, IntranasalVaccinePKResult)

    def test_all_fields_present(self):
        result = _sim()
        fields = [
            "drug_name",
            "dose_mcg",
            "times_h",
            "c_nasal_mcg_mL",
            "c_mucosal_mcg_mL",
            "c_systemic_mcg_mL",
            "cmax_nasal",
            "cmax_mucosal",
            "cmax_systemic",
            "tmax_nasal_h",
            "tmax_mucosal_h",
            "tmax_systemic_h",
            "auc_mucosal",
            "auc_systemic",
            "mucosal_to_systemic_ratio",
            "nasal_retention_pct",
            "notes",
        ]
        for f in fields:
            assert hasattr(result, f), f"Missing field: {f}"

    def test_drug_name_stored(self):
        result = simulate_intranasal_vaccine_pk("FluVaccine", dose_mcg=50.0)
        assert result.drug_name == "FluVaccine"

    def test_dose_stored(self):
        result = _sim(dose=75.0)
        assert result.dose_mcg == 75.0


# ---------------------------------------------------------------------------
# Time series properties
# ---------------------------------------------------------------------------


class TestTimeSeries:
    def test_times_list_not_empty(self):
        result = _sim()
        assert len(result.times_h) > 0

    def test_all_lists_same_length(self):
        result = _sim()
        n = len(result.times_h)
        assert len(result.c_nasal_mcg_mL) == n
        assert len(result.c_mucosal_mcg_mL) == n
        assert len(result.c_systemic_mcg_mL) == n

    def test_times_starts_at_zero(self):
        result = _sim()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_times_ends_near_t_end(self):
        result = _sim(t_end=24.0)
        assert result.times_h[-1] == pytest.approx(24.0, abs=0.2)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_nasal_starts_at_dose(self):
        """C_nasal at t=0 = dose_mcg / V_nasal (1 mL) = dose_mcg mcg/mL."""
        result = _sim(dose=100.0)
        assert result.c_nasal_mcg_mL[0] == pytest.approx(100.0)

    def test_mucosal_starts_at_zero(self):
        result = _sim()
        assert result.c_mucosal_mcg_mL[0] == pytest.approx(0.0)

    def test_systemic_starts_at_zero(self):
        result = _sim()
        assert result.c_systemic_mcg_mL[0] == pytest.approx(0.0)

    def test_nasal_monotonically_decreasing(self):
        """Nasal depot can only clear/transfer, so it decreases monotonically."""
        result = _sim()
        cn = result.c_nasal_mcg_mL
        # Allow tiny numerical noise
        for i in range(1, len(cn)):
            assert cn[i] <= cn[i - 1] + 1e-10


# ---------------------------------------------------------------------------
# Profile shape properties
# ---------------------------------------------------------------------------


class TestProfileShape:
    def test_mucosal_rises_then_falls(self):
        result = _sim()
        cm = result.c_mucosal_mcg_mL
        # Must reach a peak > 0 and fall below peak
        peak = max(cm)
        assert peak > 0.0
        assert cm[-1] < peak  # falls from peak

    def test_systemic_rises_then_falls(self):
        result = _sim()
        cs = result.c_systemic_mcg_mL
        peak = max(cs)
        assert peak > 0.0
        assert cs[-1] < peak

    def test_cmax_nasal_greater_than_cmax_mucosal(self):
        """Nasal concentration (dose/1mL) should exceed mucosal peak."""
        result = _sim(dose=100.0)
        assert result.cmax_nasal > result.cmax_mucosal

    def test_cmax_mucosal_greater_than_cmax_systemic(self):
        result = _sim()
        assert result.cmax_mucosal > result.cmax_systemic


# ---------------------------------------------------------------------------
# AUC properties
# ---------------------------------------------------------------------------


class TestAUC:
    def test_auc_mucosal_positive(self):
        result = _sim()
        assert result.auc_mucosal > 0.0

    def test_auc_systemic_positive(self):
        result = _sim()
        assert result.auc_systemic > 0.0

    def test_auc_mucosal_greater_than_auc_systemic(self):
        result = _sim()
        assert result.auc_mucosal > result.auc_systemic

    def test_mucosal_to_systemic_ratio_gt_1(self):
        result = _sim()
        assert result.mucosal_to_systemic_ratio > 1.0


# ---------------------------------------------------------------------------
# Dose linearity
# ---------------------------------------------------------------------------


class TestDoseLinearity:
    def test_auc_mucosal_proportional_to_dose(self):
        r1 = simulate_intranasal_vaccine_pk("V", dose_mcg=50.0)
        r2 = simulate_intranasal_vaccine_pk("V", dose_mcg=100.0)
        ratio = r2.auc_mucosal / r1.auc_mucosal
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_cmax_proportional_to_dose(self):
        r1 = simulate_intranasal_vaccine_pk("V", dose_mcg=50.0)
        r2 = simulate_intranasal_vaccine_pk("V", dose_mcg=100.0)
        ratio = r2.cmax_mucosal / r1.cmax_mucosal
        assert ratio == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Nasal retention
# ---------------------------------------------------------------------------


class TestNasalRetention:
    def test_nasal_retention_between_0_and_100(self):
        result = _sim()
        assert 0.0 <= result.nasal_retention_pct <= 100.0

    def test_nasal_retention_near_zero_after_long_time(self):
        result = _sim(t_end=200.0)
        assert result.nasal_retention_pct < 1.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_intranasal_vaccine_pk("V", dose_mcg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_intranasal_vaccine_pk("V", dose_mcg=-10.0)

    def test_zero_k_clear_raises(self):
        with pytest.raises(ValueError):
            simulate_intranasal_vaccine_pk("V", dose_mcg=50.0, k_clear=0.0)

    def test_negative_k_clear_raises(self):
        with pytest.raises(ValueError):
            simulate_intranasal_vaccine_pk("V", dose_mcg=50.0, k_clear=-1.0)

    def test_zero_k_uptake_raises(self):
        with pytest.raises(ValueError):
            simulate_intranasal_vaccine_pk("V", dose_mcg=50.0, k_uptake=0.0)

    def test_negative_k_uptake_raises(self):
        with pytest.raises(ValueError):
            simulate_intranasal_vaccine_pk("V", dose_mcg=50.0, k_uptake=-0.5)


# ---------------------------------------------------------------------------
# compare_dose_levels
# ---------------------------------------------------------------------------


class TestCompareDoseLevels:
    def test_returns_list(self):
        out = compare_dose_levels("V", doses_mcg=[10.0, 50.0, 100.0])
        assert isinstance(out, list)

    def test_sorted_by_auc_mucosal_descending(self):
        doses = [10.0, 100.0, 50.0, 200.0]
        out = compare_dose_levels("V", doses_mcg=doses)
        aucs = [r.auc_mucosal for r in out]
        assert aucs == sorted(aucs, reverse=True)

    def test_length_matches_input(self):
        out = compare_dose_levels("V", doses_mcg=[5.0, 20.0, 80.0])
        assert len(out) == 3

    def test_all_results_correct_type(self):
        out = compare_dose_levels("V", doses_mcg=[25.0, 75.0])
        for r in out:
            assert isinstance(r, IntranasalVaccinePKResult)
