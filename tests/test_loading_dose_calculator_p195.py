"""Tests for Phase 195 — Loading Dose Calculator (loading_dose_calculator_p195)."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.loading_dose_calculator_p195 import (
    LoadingDoseResult,
    calculate_loading_dose,
    compare_loading_strategies,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_result(**kwargs) -> LoadingDoseResult:
    """Create a result with sensible defaults, overridable via kwargs."""
    params = dict(
        drug_name="TestDrug",
        target_conc_mg_L=5.0,
        vd_L=50.0,
        fu_plasma=1.0,
        cl_L_per_h=2.0,
        bioavailability=1.0,
        dosing_interval_h=12.0,
        n_loading_doses=1,
        ka_per_h=1.0,
    )
    params.update(kwargs)
    return calculate_loading_dose(**params)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_loading_dose_result(self):
        result = _default_result()
        assert isinstance(result, LoadingDoseResult)

    def test_dataclass_is_frozen(self):
        result = _default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.loading_dose_mg = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = _default_result(drug_name="Warfarin")
        assert result.drug_name == "Warfarin"

    def test_target_conc_preserved(self):
        result = _default_result(target_conc_mg_L=10.0)
        assert result.target_conc_mg_L == pytest.approx(10.0)

    def test_vd_preserved(self):
        result = _default_result(vd_L=30.0)
        assert result.vd_L == pytest.approx(30.0)

    def test_cl_preserved(self):
        result = _default_result(cl_L_per_h=3.0)
        assert result.cl_L_per_h == pytest.approx(3.0)

    def test_fu_plasma_preserved(self):
        result = _default_result(fu_plasma=0.5)
        assert result.fu_plasma == pytest.approx(0.5)

    def test_bioavailability_preserved(self):
        result = _default_result(bioavailability=0.8)
        assert result.bioavailability == pytest.approx(0.8)

    def test_n_loading_doses_preserved(self):
        result = _default_result(n_loading_doses=3)
        assert result.n_loading_doses == 3

    def test_dosing_interval_preserved(self):
        result = _default_result(dosing_interval_h=8.0)
        assert result.dosing_interval_h == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Core physics
# ---------------------------------------------------------------------------


class TestCorePhysics:
    def test_loading_dose_positive(self):
        result = _default_result()
        assert result.loading_dose_mg > 0.0

    def test_maintenance_dose_positive(self):
        result = _default_result()
        assert result.maintenance_dose_mg > 0.0

    def test_loading_ge_maintenance(self):
        """Loading dose should be >= maintenance dose in typical scenarios."""
        result = _default_result(
            target_conc_mg_L=5.0,
            vd_L=50.0,
            cl_L_per_h=2.0,
            dosing_interval_h=12.0,
        )
        assert result.loading_dose_mg >= result.maintenance_dose_mg

    def test_higher_vd_higher_loading_dose(self):
        r_low = _default_result(vd_L=20.0)
        r_high = _default_result(vd_L=100.0)
        assert r_high.loading_dose_mg > r_low.loading_dose_mg

    def test_higher_target_higher_loading(self):
        r_low = _default_result(target_conc_mg_L=2.0)
        r_high = _default_result(target_conc_mg_L=10.0)
        assert r_high.loading_dose_mg > r_low.loading_dose_mg

    def test_time_to_90pct_positive(self):
        result = _default_result()
        assert result.time_to_90pct_target_h > 0.0

    def test_accumulation_ratio_gte_one(self):
        """Accumulation ratio >= 1 for repeated dosing."""
        result = _default_result()
        assert result.accumulation_ratio >= 1.0

    def test_dose_ratio_gt_one(self):
        """Loading dose > maintenance dose → ratio > 1."""
        result = _default_result(vd_L=80.0, cl_L_per_h=2.0, dosing_interval_h=12.0)
        assert result.dose_ratio_loading_to_maintenance >= 1.0

    def test_notes_is_string(self):
        result = _default_result()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Bioavailability effect
# ---------------------------------------------------------------------------


class TestBioavailability:
    def test_lower_f_higher_loading_dose(self):
        r_iv = _default_result(bioavailability=1.0)
        r_oral = _default_result(bioavailability=0.4)
        # Oral route requires larger administered dose
        assert r_oral.loading_dose_mg > r_iv.loading_dose_mg

    def test_lower_f_higher_maintenance(self):
        r_iv = _default_result(bioavailability=1.0)
        r_oral = _default_result(bioavailability=0.5)
        assert r_oral.maintenance_dose_mg > r_iv.maintenance_dose_mg


# ---------------------------------------------------------------------------
# compare_loading_strategies
# ---------------------------------------------------------------------------


class TestCompareStrategies:
    def test_returns_correct_count(self):
        n_options = [1, 2, 3]
        results = compare_loading_strategies(
            drug_name="DrugX",
            target_conc_mg_L=5.0,
            vd_L=50.0,
            cl_L_per_h=2.0,
            n_doses_options=n_options,
        )
        assert len(results) == len(n_options)

    def test_all_results_are_loading_dose_result(self):
        results = compare_loading_strategies(
            drug_name="DrugX",
            target_conc_mg_L=5.0,
            vd_L=50.0,
            cl_L_per_h=2.0,
            n_doses_options=[1, 2],
        )
        for r in results:
            assert isinstance(r, LoadingDoseResult)

    def test_n_loading_doses_stored_correctly(self):
        n_options = [1, 2, 3, 4]
        results = compare_loading_strategies(
            drug_name="DrugX",
            target_conc_mg_L=5.0,
            vd_L=50.0,
            cl_L_per_h=2.0,
            n_doses_options=n_options,
        )
        for result, expected_n in zip(results, n_options):
            assert result.n_loading_doses == expected_n

    def test_single_option_list(self):
        results = compare_loading_strategies(
            drug_name="DrugX",
            target_conc_mg_L=3.0,
            vd_L=30.0,
            cl_L_per_h=1.0,
            n_doses_options=[1],
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_target_conc_zero_raises(self):
        with pytest.raises(ValueError, match="target_conc_mg_L"):
            calculate_loading_dose(
                drug_name="X",
                target_conc_mg_L=0.0,
                vd_L=50.0,
                cl_L_per_h=2.0,
            )

    def test_target_conc_negative_raises(self):
        with pytest.raises(ValueError, match="target_conc_mg_L"):
            calculate_loading_dose(
                drug_name="X",
                target_conc_mg_L=-1.0,
                vd_L=50.0,
                cl_L_per_h=2.0,
            )

    def test_vd_zero_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            calculate_loading_dose(
                drug_name="X",
                target_conc_mg_L=5.0,
                vd_L=0.0,
                cl_L_per_h=2.0,
            )

    def test_vd_negative_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            calculate_loading_dose(
                drug_name="X",
                target_conc_mg_L=5.0,
                vd_L=-10.0,
                cl_L_per_h=2.0,
            )

    def test_cl_zero_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            calculate_loading_dose(
                drug_name="X",
                target_conc_mg_L=5.0,
                vd_L=50.0,
                cl_L_per_h=0.0,
            )

    def test_cl_negative_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            calculate_loading_dose(
                drug_name="X",
                target_conc_mg_L=5.0,
                vd_L=50.0,
                cl_L_per_h=-1.0,
            )

    def test_bioavailability_zero_raises(self):
        with pytest.raises(ValueError, match="bioavailability"):
            calculate_loading_dose(
                drug_name="X",
                target_conc_mg_L=5.0,
                vd_L=50.0,
                cl_L_per_h=2.0,
                bioavailability=0.0,
            )

    def test_bioavailability_above_one_raises(self):
        with pytest.raises(ValueError, match="bioavailability"):
            calculate_loading_dose(
                drug_name="X",
                target_conc_mg_L=5.0,
                vd_L=50.0,
                cl_L_per_h=2.0,
                bioavailability=1.5,
            )
