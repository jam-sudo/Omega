"""
Tests for Phase 212 — eye_drop_pk.py
~26 tests covering return type, field values, pharmacokinetics, comparison, and validation.
"""

import pytest

from omega_pbpk.core.eye_drop_pk import (
    EyeDropPKResult,
    compare_drop_volumes,
    simulate_eye_drop_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_default(drug_name="TestDrug", dose_mg=0.5, cpf=1.0, t_end=4.0):
    """Run a short simulation (4 h, coarse dt) for speed."""
    return simulate_eye_drop_pk(
        drug_name=drug_name,
        dose_mg=dose_mg,
        corneal_perm_factor=cpf,
        t_end_h=t_end,
        dt_h=1.0 / 60.0,  # 1-minute steps
    )


# ---------------------------------------------------------------------------
# 1. Return type and fields
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_eye_drop_pk_result(self):
        result = run_default()
        assert isinstance(result, EyeDropPKResult)

    def test_drug_name_preserved(self):
        result = run_default(drug_name="Timolol")
        assert result.drug_name == "Timolol"

    def test_dose_preserved(self):
        result = run_default(dose_mg=0.25)
        assert result.dose_mg == pytest.approx(0.25)

    def test_corneal_perm_factor_preserved(self):
        result = run_default(cpf=1.5)
        assert result.corneal_perm_factor == pytest.approx(1.5)

    def test_times_h_is_list(self):
        result = run_default()
        assert isinstance(result.times_h, list)

    def test_c_tear_is_list(self):
        result = run_default()
        assert isinstance(result.c_tear_mg_mL, list)

    def test_c_cornea_is_list(self):
        result = run_default()
        assert isinstance(result.c_cornea_mg_g, list)

    def test_c_aqueous_is_list(self):
        result = run_default()
        assert isinstance(result.c_aqueous_mg_mL, list)

    def test_lists_same_length(self):
        result = run_default()
        n = len(result.times_h)
        assert len(result.c_tear_mg_mL) == n
        assert len(result.c_cornea_mg_g) == n
        assert len(result.c_aqueous_mg_mL) == n

    def test_time_starts_at_zero(self):
        result = run_default()
        assert result.times_h[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_c_aqueous_starts_at_zero(self):
        result = run_default()
        assert result.c_aqueous_mg_mL[0] == pytest.approx(0.0)

    def test_c_cornea_starts_at_zero(self):
        result = run_default()
        assert result.c_cornea_mg_g[0] == pytest.approx(0.0)

    def test_c_tear_starts_high(self):
        """C_tear(0) = dose_mg / V_tear = 0.5 / 0.007 ≈ 71.4 mg/mL"""
        result = run_default(dose_mg=0.5)
        expected = 0.5 / 0.007
        assert result.c_tear_mg_mL[0] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Pharmacokinetic behaviour
# ---------------------------------------------------------------------------


class TestPharmacokinetics:
    def test_c_tear_decreases_rapidly(self):
        """Tear concentration should drop significantly within first hour."""
        result = run_default(t_end=4.0)
        idx_1h = min(60, len(result.times_h) - 1)  # approx 1-h index (1/min steps)
        assert result.c_tear_mg_mL[idx_1h] < result.c_tear_mg_mL[0] * 0.5

    def test_c_aqueous_rises_then_falls(self):
        """Aqueous should increase from 0, reach a max, then decline."""
        result = run_default(t_end=6.0, cpf=1.0)
        ca = result.c_aqueous_mg_mL
        max_val = max(ca)
        assert max_val > 0.0
        # Peak should not be at time 0
        peak_idx = ca.index(max_val)
        assert peak_idx > 0

    def test_cmax_aqueous_greater_than_zero(self):
        result = run_default()
        assert result.cmax_aqueous_mg_mL > 0.0

    def test_tmax_aqueous_greater_than_zero(self):
        result = run_default()
        assert result.tmax_aqueous_h > 0.0

    def test_auc_aqueous_greater_than_zero(self):
        result = run_default()
        assert result.auc_aqueous_mg_h_per_mL > 0.0

    def test_bioavailability_between_0_and_100(self):
        result = run_default()
        assert 0.0 <= result.bioavailability_corneal_pct <= 100.0

    def test_all_concentrations_nonnegative(self):
        result = run_default(t_end=6.0)
        assert all(c >= 0.0 for c in result.c_tear_mg_mL)
        assert all(c >= 0.0 for c in result.c_cornea_mg_g)
        assert all(c >= 0.0 for c in result.c_aqueous_mg_mL)


# ---------------------------------------------------------------------------
# 4. corneal_perm_factor effect
# ---------------------------------------------------------------------------


class TestCornealPermFactor:
    def test_higher_perm_factor_higher_aqueous_auc(self):
        low = run_default(cpf=0.5)
        high = run_default(cpf=2.0)
        assert high.auc_aqueous_mg_h_per_mL > low.auc_aqueous_mg_h_per_mL

    def test_higher_perm_factor_higher_cmax(self):
        low = run_default(cpf=0.5)
        high = run_default(cpf=2.0)
        assert high.cmax_aqueous_mg_mL > low.cmax_aqueous_mg_mL


# ---------------------------------------------------------------------------
# 5. compare_drop_volumes
# ---------------------------------------------------------------------------


class TestCompareDropVolumes:
    def test_returns_list(self):
        results = compare_drop_volumes(
            "Drug", 1.0, [25.0, 50.0], corneal_perm_factor=1.0, t_end_h=4.0, dt_h=1.0 / 60.0
        )
        assert isinstance(results, list)

    def test_length_matches_volumes(self):
        volumes = [10.0, 25.0, 50.0]
        results = compare_drop_volumes(
            "Drug", 1.0, volumes, corneal_perm_factor=1.0, t_end_h=4.0, dt_h=1.0 / 60.0
        )
        assert len(results) == 3

    def test_sorted_by_auc_ascending(self):
        volumes = [50.0, 10.0, 25.0]
        results = compare_drop_volumes(
            "Drug", 1.0, volumes, corneal_perm_factor=1.0, t_end_h=4.0, dt_h=1.0 / 60.0
        )
        aucs = [r.auc_aqueous_mg_h_per_mL for r in results]
        assert aucs == sorted(aucs)

    def test_larger_volume_higher_auc(self):
        results = compare_drop_volumes(
            "Drug", 1.0, [10.0, 50.0], corneal_perm_factor=1.0, t_end_h=4.0, dt_h=1.0 / 60.0
        )
        # sorted ascending → smaller dose first, larger dose last
        assert results[0].dose_mg < results[1].dose_mg
        assert results[0].auc_aqueous_mg_h_per_mL < results[1].auc_aqueous_mg_h_per_mL


# ---------------------------------------------------------------------------
# 6. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_eye_drop_pk("D", dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_eye_drop_pk("D", dose_mg=-1.0)

    def test_perm_factor_too_low_raises(self):
        with pytest.raises(ValueError, match="corneal_perm_factor"):
            simulate_eye_drop_pk("D", dose_mg=0.5, corneal_perm_factor=0.05)

    def test_perm_factor_too_high_raises(self):
        with pytest.raises(ValueError, match="corneal_perm_factor"):
            simulate_eye_drop_pk("D", dose_mg=0.5, corneal_perm_factor=11.0)
