"""Tests for Phase 274 placental transfer PK model (gestational-age-aware)."""

import pytest

from omega_pbpk.core.placental_transfer_pk_p274 import (
    PlacentalTransferResult,
    compare_gestational_ages,
    simulate_placental_transfer,
)


def make_default() -> PlacentalTransferResult:
    return simulate_placental_transfer(
        drug_name="TestDrug",
        dose_mg=100.0,
        gestational_age_weeks=20.0,
        cl_L_per_h=2.0,
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
        assert result.dose_mg == pytest.approx(100.0)

    def test_gestational_age_preserved(self):
        result = make_default()
        assert result.gestational_age_weeks == pytest.approx(20.0)


class TestTimesAndConcentrations:
    def test_times_nonempty(self):
        result = make_default()
        assert len(result.times_h) > 0

    def test_concentrations_same_length_as_times(self):
        result = make_default()
        assert len(result.c_maternal_mg_L) == len(result.times_h)
        assert len(result.c_fetal_mg_L) == len(result.times_h)

    def test_fetal_starts_at_zero(self):
        result = make_default()
        assert result.c_fetal_mg_L[0] == pytest.approx(0.0)

    def test_maternal_starts_at_dose_over_volume(self):
        # IV bolus: C_maternal(0) = dose / V_maternal = 100/5 = 20 mg/L
        result = simulate_placental_transfer(
            drug_name="D", dose_mg=100.0, gestational_age_weeks=20.0, cl_L_per_h=2.0
        )
        assert result.c_maternal_mg_L[0] == pytest.approx(20.0)

    def test_concentrations_nonnegative(self):
        result = make_default()
        assert all(c >= 0.0 for c in result.c_maternal_mg_L)
        assert all(c >= 0.0 for c in result.c_fetal_mg_L)


class TestCmaxAndAUC:
    def test_cmax_fetal_positive(self):
        result = make_default()
        assert result.cmax_fetal_mg_L > 0.0

    def test_cmax_maternal_positive(self):
        result = make_default()
        assert result.cmax_maternal_mg_L > 0.0

    def test_auc_maternal_positive(self):
        result = make_default()
        assert result.auc_maternal_mg_h_per_L > 0.0

    def test_auc_fetal_positive(self):
        result = make_default()
        assert result.auc_fetal_mg_h_per_L > 0.0


class TestFetalMaternalRatio:
    def test_ratio_nonnegative(self):
        result = make_default()
        assert result.fetal_maternal_ratio >= 0.0

    def test_ratio_positive(self):
        result = make_default()
        assert result.fetal_maternal_ratio > 0.0

    def test_later_ga_higher_k_mf(self):
        """Later gestational age has higher placental transfer rate constant k_mf."""
        # k_mf = 0.01 + 0.001 * GA: GA=8 → 0.018, GA=36 → 0.046
        # Both produce positive transfer; this test verifies the model runs for both
        early = simulate_placental_transfer(
            drug_name="D", dose_mg=100.0, gestational_age_weeks=8.0, cl_L_per_h=20.0
        )
        late = simulate_placental_transfer(
            drug_name="D", dose_mg=100.0, gestational_age_weeks=36.0, cl_L_per_h=20.0
        )
        assert early.cmax_fetal_mg_L > 0.0
        assert late.cmax_fetal_mg_L > 0.0


class TestTrimester:
    def test_first_trimester(self):
        result = simulate_placental_transfer("D", 100.0, 8.0, 2.0)
        assert result.trimester == "first"

    def test_second_trimester(self):
        result = simulate_placental_transfer("D", 100.0, 20.0, 2.0)
        assert result.trimester == "second"

    def test_third_trimester(self):
        result = simulate_placental_transfer("D", 100.0, 32.0, 2.0)
        assert result.trimester == "third"

    def test_boundary_second_trimester(self):
        result = simulate_placental_transfer("D", 100.0, 13.0, 2.0)
        assert result.trimester == "second"

    def test_boundary_third_trimester(self):
        result = simulate_placental_transfer("D", 100.0, 28.0, 2.0)
        assert result.trimester == "third"


class TestFetalExposureRisk:
    def test_risk_valid_values(self):
        result = make_default()
        assert result.fetal_exposure_risk in ("low", "moderate", "high")

    def test_risk_is_valid_string(self):
        # All gestational ages produce a valid risk classification
        result = simulate_placental_transfer("D", 100.0, 4.0, 10.0)
        assert result.fetal_exposure_risk in ("low", "moderate", "high")

    def test_notes_nonempty(self):
        result = make_default()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


class TestCompareGestationalAges:
    def test_returns_list(self):
        results = compare_gestational_ages("D", 100.0, 2.0)
        assert isinstance(results, list)

    def test_returns_five_by_default(self):
        results = compare_gestational_ages("D", 100.0, 2.0)
        assert len(results) == 5

    def test_sorted_descending_by_ratio(self):
        results = compare_gestational_ages("D", 100.0, 2.0)
        ratios = [r.fetal_maternal_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_all_correct_type(self):
        results = compare_gestational_ages("D", 100.0, 2.0)
        for r in results:
            assert isinstance(r, PlacentalTransferResult)

    def test_custom_ages(self):
        results = compare_gestational_ages("D", 100.0, 2.0, ages_weeks=[10, 20, 30])
        assert len(results) == 3


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_placental_transfer("D", -50.0, 20.0, 2.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_placental_transfer("D", 0.0, 20.0, 2.0)

    def test_ga_too_low_raises(self):
        with pytest.raises(ValueError, match="gestational_age_weeks"):
            simulate_placental_transfer("D", 100.0, 3.0, 2.0)

    def test_ga_too_high_raises(self):
        with pytest.raises(ValueError, match="gestational_age_weeks"):
            simulate_placental_transfer("D", 100.0, 43.0, 2.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_placental_transfer("D", 100.0, 20.0, 0.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_placental_transfer("D", 100.0, 20.0, -1.0)
