"""Tests for Phase 1082 — Muscle Compartment Drug Distribution."""
import pytest
from omega_pbpk.core.muscle_distribution_pk import (
    MuscleDistributionResult,
    compare_logp_muscle,
    simulate_muscle_distribution,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def baseline_result():
    return simulate_muscle_distribution("TestStatin", dose_mg=40.0, logP=2.0)


@pytest.fixture
def high_logp_result():
    return simulate_muscle_distribution("LipophilicDrug", dose_mg=40.0, logP=5.0)


@pytest.fixture
def low_logp_result():
    return simulate_muscle_distribution("HydrophilicDrug", dose_mg=40.0, logP=-1.0)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_dataclass(self, baseline_result):
        assert isinstance(baseline_result, MuscleDistributionResult)

    def test_drug_name_stored(self, baseline_result):
        assert baseline_result.drug_name == "TestStatin"

    def test_dose_stored(self, baseline_result):
        assert baseline_result.dose_mg == 40.0

    def test_logP_stored(self, baseline_result):
        assert baseline_result.logP == 2.0

    def test_times_is_list(self, baseline_result):
        assert isinstance(baseline_result.times_h, list)
        assert len(baseline_result.times_h) > 0

    def test_c_plasma_is_list(self, baseline_result):
        assert isinstance(baseline_result.c_plasma_mg_L, list)

    def test_c_skeletal_is_list(self, baseline_result):
        assert isinstance(baseline_result.c_skeletal_mg_L, list)

    def test_c_cardiac_is_list(self, baseline_result):
        assert isinstance(baseline_result.c_cardiac_mg_L, list)


# ---------------------------------------------------------------------------
# Initial conditions (IV bolus)
# ---------------------------------------------------------------------------

class TestInitialConditions:
    def test_c_plasma_at_t0_equals_dose_over_volume(self, baseline_result):
        # default V_plasma = 3.0 L
        expected = 40.0 / 3.0
        assert baseline_result.c_plasma_mg_L[0] == pytest.approx(expected, rel=1e-6)

    def test_c_skeletal_starts_at_zero(self, baseline_result):
        assert baseline_result.c_skeletal_mg_L[0] == pytest.approx(0.0)

    def test_c_cardiac_starts_at_zero(self, baseline_result):
        assert baseline_result.c_cardiac_mg_L[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Physiological plausibility
# ---------------------------------------------------------------------------

class TestPhysiology:
    def test_auc_plasma_positive(self, baseline_result):
        assert baseline_result.auc_plasma > 0.0

    def test_auc_skeletal_positive(self, baseline_result):
        assert baseline_result.auc_skeletal > 0.0

    def test_auc_cardiac_positive(self, baseline_result):
        assert baseline_result.auc_cardiac > 0.0

    def test_cmax_plasma_equals_initial(self, baseline_result):
        # For IV bolus, cmax should be at t=0
        assert baseline_result.cmax_plasma == pytest.approx(40.0 / 3.0, rel=1e-6)

    def test_cardiac_to_plasma_ratio_positive(self, baseline_result):
        assert baseline_result.cardiac_to_plasma_ratio > 0.0

    def test_plasma_declines_over_time(self, baseline_result):
        # Last plasma conc should be lower than first
        assert baseline_result.c_plasma_mg_L[-1] < baseline_result.c_plasma_mg_L[0]


# ---------------------------------------------------------------------------
# logP effects
# ---------------------------------------------------------------------------

class TestLogPEffects:
    def test_higher_logP_gives_higher_cmax_cardiac(self, low_logp_result, high_logp_result):
        assert high_logp_result.cmax_cardiac > low_logp_result.cmax_cardiac

    def test_higher_logP_increases_cardiac_to_plasma_ratio(
        self, low_logp_result, high_logp_result
    ):
        assert (
            high_logp_result.cardiac_to_plasma_ratio
            > low_logp_result.cardiac_to_plasma_ratio
        )

    def test_very_high_logP_gives_high_myopathy_or_cardiac_risk(self):
        result = simulate_muscle_distribution("HighLogP", dose_mg=100.0, logP=8.0)
        risk_high = (
            result.myopathy_risk == "high" or result.cardiac_exposure_risk == "high"
        )
        assert risk_high


# ---------------------------------------------------------------------------
# Risk classifications
# ---------------------------------------------------------------------------

class TestRiskClassifications:
    def test_myopathy_risk_valid(self, baseline_result):
        assert baseline_result.myopathy_risk in {"low", "moderate", "high"}

    def test_cardiac_exposure_risk_valid(self, baseline_result):
        assert baseline_result.cardiac_exposure_risk in {"low", "moderate", "high"}

    def test_myopathy_risk_valid_high_logP(self, high_logp_result):
        assert high_logp_result.myopathy_risk in {"low", "moderate", "high"}

    def test_cardiac_exposure_risk_valid_high_logP(self, high_logp_result):
        assert high_logp_result.cardiac_exposure_risk in {"low", "moderate", "high"}

    def test_low_logP_gives_low_cardiac_risk(self):
        result = simulate_muscle_distribution("Hydrophilic", dose_mg=40.0, logP=-3.0)
        assert result.cardiac_exposure_risk == "low"


# ---------------------------------------------------------------------------
# compare_logp_muscle
# ---------------------------------------------------------------------------

class TestCompareLogpMuscle:
    def test_returns_correct_count(self):
        results = compare_logp_muscle("Drug", 50.0, [1.0, 3.0, 5.0])
        assert len(results) == 3

    def test_sorted_by_cardiac_to_plasma_ratio_descending(self):
        results = compare_logp_muscle("Drug", 50.0, [-1.0, 1.0, 3.0, 5.0])
        ratios = [r.cardiac_to_plasma_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_single_logp(self):
        results = compare_logp_muscle("Drug", 50.0, [2.0])
        assert len(results) == 1

    def test_all_results_are_correct_type(self):
        results = compare_logp_muscle("Drug", 50.0, [1.0, 2.0])
        for r in results:
            assert isinstance(r, MuscleDistributionResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_muscle_distribution("Drug", dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_muscle_distribution("Drug", dose_mg=0.0)

    def test_logP_too_high_raises(self):
        with pytest.raises(ValueError, match="logP"):
            simulate_muscle_distribution("Drug", dose_mg=50.0, logP=15.0)

    def test_logP_too_low_raises(self):
        with pytest.raises(ValueError, match="logP"):
            simulate_muscle_distribution("Drug", dose_mg=50.0, logP=-10.0)

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_total"):
            simulate_muscle_distribution("Drug", dose_mg=50.0, cl_total_L_per_h=-1.0)

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_total"):
            simulate_muscle_distribution("Drug", dose_mg=50.0, cl_total_L_per_h=0.0)

    def test_negative_volume_raises(self):
        with pytest.raises(ValueError, match="v_plasma"):
            simulate_muscle_distribution("Drug", dose_mg=50.0, v_plasma_L=-1.0)

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError, match="v_plasma"):
            simulate_muscle_distribution("Drug", dose_mg=50.0, v_plasma_L=0.0)
