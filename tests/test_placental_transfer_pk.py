"""Tests for placental transfer PK model (Phase 906)."""

import pytest

from omega_pbpk.core.placental_transfer_pk import (
    PlacentalTransferResult,
    compare_placental_transfer,
    simulate_placental_transfer,
)


def make_default() -> PlacentalTransferResult:
    return simulate_placental_transfer(
        drug_name="TestDrug",
        dose_mg=100.0,
        k_abs_per_h=1.0,
        k_placenta_transfer_per_h=0.1,
        k_fetal_elimination_per_h=0.05,
        cl_maternal_L_per_h=10.0,
        vd_maternal_L=30.0,
        vd_fetal_L=3.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


class TestReturnType:
    def test_returns_correct_type(self):
        result = make_default()
        assert isinstance(result, PlacentalTransferResult)

    def test_drug_name_preserved(self):
        result = make_default()
        assert result.drug_name == "TestDrug"

    def test_dose_mg_preserved(self):
        result = make_default()
        assert result.dose_mg == 100.0


class TestTimesAndConcentrations:
    def test_times_nonempty(self):
        result = make_default()
        assert len(result.times_h) > 0

    def test_concentrations_same_length(self):
        result = make_default()
        assert len(result.c_maternal_mg_L) == len(result.times_h)
        assert len(result.c_fetal_mg_L) == len(result.times_h)

    def test_times_start_at_zero(self):
        result = make_default()
        assert result.times_h[0] == pytest.approx(0.0)

    def test_concentrations_nonnegative(self):
        result = make_default()
        assert all(c >= 0.0 for c in result.c_maternal_mg_L)
        assert all(c >= 0.0 for c in result.c_fetal_mg_L)


class TestMaternalKinetics:
    def test_cmax_maternal_positive(self):
        result = make_default()
        assert result.cmax_maternal > 0.0

    def test_auc_maternal_positive(self):
        result = make_default()
        assert result.auc_maternal > 0.0

    def test_maternal_starts_at_zero(self):
        """Oral absorption: maternal starts at 0, rises after gut absorption."""
        result = make_default()
        assert result.c_maternal_mg_L[0] == pytest.approx(0.0)

    def test_maternal_cmax_occurs_after_t0(self):
        result = make_default()
        assert result.tmax_maternal_h > 0.0


class TestFetalKinetics:
    def test_cmax_fetal_positive(self):
        result = make_default()
        assert result.cmax_fetal > 0.0

    def test_auc_fetal_positive(self):
        result = make_default()
        assert result.auc_fetal > 0.0

    def test_fetal_starts_at_zero(self):
        result = make_default()
        assert result.c_fetal_mg_L[0] == pytest.approx(0.0)

    def test_fetal_tmax_after_maternal_tmax(self):
        """Fetal peak should lag behind maternal peak."""
        result = make_default()
        assert result.tmax_fetal_h >= result.tmax_maternal_h


class TestFetalMaternalRatio:
    def test_ratio_nonnegative(self):
        result = make_default()
        assert result.fetal_maternal_ratio >= 0.0

    def test_ratio_less_than_one_with_fast_fetal_elimination(self):
        """With rapid fetal elimination relative to transfer, ratio < 1."""
        result = simulate_placental_transfer(
            k_placenta_transfer_per_h=0.05,
            k_fetal_elimination_per_h=2.0,
            vd_maternal_L=30.0,
            vd_fetal_L=30.0,
        )
        assert result.fetal_maternal_ratio < 1.0

    def test_higher_transfer_rate_higher_ratio(self):
        result_low = simulate_placental_transfer(k_placenta_transfer_per_h=0.01)
        result_high = simulate_placental_transfer(k_placenta_transfer_per_h=0.5)
        assert result_high.fetal_maternal_ratio > result_low.fetal_maternal_ratio

    def test_ratio_zero_when_no_transfer(self):
        result = simulate_placental_transfer(k_placenta_transfer_per_h=0.0)
        assert result.fetal_maternal_ratio == pytest.approx(0.0, abs=1e-6)


class TestPlacentalClearance:
    def test_clearance_positive(self):
        result = make_default()
        assert result.placental_clearance_mL_min > 0.0

    def test_clearance_clamped_above_minimum(self):
        result = simulate_placental_transfer(k_placenta_transfer_per_h=0.0)
        assert result.placental_clearance_mL_min >= 0.001

    def test_clearance_clamped_below_maximum(self):
        # Very high transfer rate — should not exceed 100 mL/min
        result = simulate_placental_transfer(k_placenta_transfer_per_h=10000.0, vd_maternal_L=30.0)
        assert result.placental_clearance_mL_min <= 100.0


class TestFetalSafetyConcern:
    def test_safety_concern_is_bool(self):
        result = make_default()
        assert isinstance(result.fetal_safety_concern, bool)

    def test_high_transfer_triggers_concern(self):
        result = simulate_placental_transfer(
            k_placenta_transfer_per_h=1.0,
            k_fetal_elimination_per_h=0.001,
        )
        assert result.fetal_safety_concern is True

    def test_no_transfer_no_concern(self):
        result = simulate_placental_transfer(k_placenta_transfer_per_h=0.0)
        assert result.fetal_safety_concern is False


class TestNotes:
    def test_notes_nonempty(self):
        result = make_default()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0

    def test_high_transfer_notes(self):
        result = simulate_placental_transfer(
            k_placenta_transfer_per_h=1.0,
            k_fetal_elimination_per_h=0.001,
        )
        assert "High" in result.notes or "Contraindicated" in result.notes

    def test_low_transfer_notes(self):
        result = simulate_placental_transfer(k_placenta_transfer_per_h=0.0)
        assert "Limited" in result.notes or "safer" in result.notes


class TestValidation:
    def test_zero_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(dose_mg=0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(dose_mg=-100.0)

    def test_zero_cl_maternal_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(cl_maternal_L_per_h=0.0)

    def test_zero_vd_maternal_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(vd_maternal_L=0.0)

    def test_zero_vd_fetal_raises(self):
        with pytest.raises(ValueError):
            simulate_placental_transfer(vd_fetal_L=0.0)


class TestComparePlacentalTransfer:
    def test_returns_list(self):
        scenarios = [
            {"name": "low", "k_placenta_transfer_per_h": 0.01},
            {"name": "high", "k_placenta_transfer_per_h": 0.5},
        ]
        results = compare_placental_transfer("Drug", 100.0, scenarios)
        assert isinstance(results, list)

    def test_correct_length(self):
        scenarios = [
            {"k_placenta_transfer_per_h": 0.01},
            {"k_placenta_transfer_per_h": 0.1},
            {"k_placenta_transfer_per_h": 0.5},
        ]
        results = compare_placental_transfer("Drug", 100.0, scenarios)
        assert len(results) == 3

    def test_sorted_descending_by_ratio(self):
        scenarios = [
            {"k_placenta_transfer_per_h": 0.01},
            {"k_placenta_transfer_per_h": 0.5},
            {"k_placenta_transfer_per_h": 0.1},
        ]
        results = compare_placental_transfer("Drug", 100.0, scenarios)
        ratios = [r.fetal_maternal_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_all_results_are_correct_type(self):
        scenarios = [{"k_placenta_transfer_per_h": 0.1}]
        results = compare_placental_transfer("Drug", 100.0, scenarios)
        for r in results:
            assert isinstance(r, PlacentalTransferResult)

    def test_scenario_name_used(self):
        scenarios = [{"name": "MyScenario", "k_placenta_transfer_per_h": 0.1}]
        results = compare_placental_transfer("Drug", 100.0, scenarios)
        assert results[0].drug_name == "MyScenario"
