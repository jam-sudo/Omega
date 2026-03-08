"""Tests for Phase 641 — Drug Gastric Retention Formulation."""

import pytest

from omega_pbpk.core.gastric_retention_p641 import (
    _VALID_FORMULATIONS,
    GastricRetentionResult,
    simulate_gastric_retention,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def floating_result():
    return simulate_gastric_retention(
        drug_name="TestDrug",
        dose_mg=100.0,
        formulation_type="floating",
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
        solubility_mg_mL=5.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


@pytest.fixture
def swellable_result():
    return simulate_gastric_retention(
        drug_name="TestDrug",
        dose_mg=100.0,
        formulation_type="swellable",
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
        solubility_mg_mL=5.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


@pytest.fixture
def bioadhesive_result():
    return simulate_gastric_retention(
        drug_name="TestDrug",
        dose_mg=100.0,
        formulation_type="bioadhesive",
        cl_sys_L_per_h=2.0,
        vd_sys_L=20.0,
        solubility_mg_mL=5.0,
        t_end_h=24.0,
        dt_h=0.1,
    )


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_gastric_retention_result(self, floating_result):
        assert isinstance(floating_result, GastricRetentionResult)

    def test_result_has_drug_name(self, floating_result):
        assert floating_result.drug_name == "TestDrug"

    def test_result_has_dose_mg(self, floating_result):
        assert floating_result.dose_mg == 100.0

    def test_result_has_formulation_type(self, floating_result):
        assert floating_result.formulation_type == "floating"

    def test_list_fields_same_length(self, floating_result):
        n = len(floating_result.times_h)
        assert len(floating_result.c_gastric_depot_mg) == n
        assert len(floating_result.c_dissolved_mg_mL) == n
        assert len(floating_result.c_plasma_mg_L) == n
        assert len(floating_result.release_rate_mg_h) == n
        assert len(floating_result.fraction_released) == n

    def test_times_start_at_zero(self, floating_result):
        assert floating_result.times_h[0] == pytest.approx(0.0)

    def test_times_end_at_t_end(self, floating_result):
        assert floating_result.times_h[-1] == pytest.approx(24.0)

    def test_notes_is_string(self, floating_result):
        assert isinstance(floating_result.notes, str)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_depot_starts_at_dose(self, floating_result):
        assert floating_result.c_gastric_depot_mg[0] == pytest.approx(100.0)

    def test_dissolved_starts_at_zero(self, floating_result):
        assert floating_result.c_dissolved_mg_mL[0] == pytest.approx(0.0)

    def test_plasma_starts_at_zero(self, floating_result):
        assert floating_result.c_plasma_mg_L[0] == pytest.approx(0.0)

    def test_fraction_released_starts_at_zero(self, floating_result):
        assert floating_result.fraction_released[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Physical constraints
# ---------------------------------------------------------------------------


class TestPhysicalConstraints:
    def test_depot_non_negative(self, floating_result):
        assert all(x >= 0.0 for x in floating_result.c_gastric_depot_mg)

    def test_dissolved_non_negative(self, floating_result):
        assert all(x >= 0.0 for x in floating_result.c_dissolved_mg_mL)

    def test_plasma_non_negative(self, floating_result):
        assert all(x >= 0.0 for x in floating_result.c_plasma_mg_L)

    def test_release_rate_non_negative(self, floating_result):
        assert all(x >= 0.0 for x in floating_result.release_rate_mg_h)

    def test_fraction_released_in_0_1(self, floating_result):
        assert all(0.0 <= x <= 1.0 for x in floating_result.fraction_released)

    def test_depot_decreases_overall(self, floating_result):
        # Depot should end lower than it started
        assert floating_result.c_gastric_depot_mg[-1] < floating_result.c_gastric_depot_mg[0]


# ---------------------------------------------------------------------------
# Fraction released
# ---------------------------------------------------------------------------


class TestFractionReleased:
    def test_fraction_released_increases(self, floating_result):
        # fraction_released should be monotonically non-decreasing
        fr = floating_result.fraction_released
        assert all(fr[i] >= fr[i - 1] - 1e-9 for i in range(1, len(fr)))

    def test_fraction_released_final_bioadhesive(self, bioadhesive_result):
        # After 24h, should have released a substantial fraction
        assert bioadhesive_result.fraction_released[-1] > 0.3

    def test_fraction_released_final_swellable(self, swellable_result, bioadhesive_result):
        # Higher k_release -> more fraction released
        assert swellable_result.fraction_released[-1] > bioadhesive_result.fraction_released[-1]


# ---------------------------------------------------------------------------
# Formulation differences: k_release ordering
# ---------------------------------------------------------------------------


class TestFormulationDifferences:
    def test_swellable_releases_faster_than_floating(self, swellable_result, floating_result):
        # swellable k_release=0.08 > floating k_release=0.05
        assert swellable_result.fraction_released[-1] > floating_result.fraction_released[-1]

    def test_floating_releases_faster_than_bioadhesive(self, floating_result, bioadhesive_result):
        # floating k_release=0.05 > bioadhesive k_release=0.03
        assert floating_result.fraction_released[-1] > bioadhesive_result.fraction_released[-1]

    def test_bioadhesive_longest_gastric_retention(
        self, floating_result, swellable_result, bioadhesive_result
    ):
        # bioadhesive has lowest k_release → longest retention time
        assert (
            bioadhesive_result.gastric_retention_time_h >= floating_result.gastric_retention_time_h
        )
        assert (
            bioadhesive_result.gastric_retention_time_h >= swellable_result.gastric_retention_time_h
        )

    def test_swellable_shortest_gastric_retention(
        self, floating_result, swellable_result, bioadhesive_result
    ):
        # swellable has highest k_release → shortest retention time
        assert swellable_result.gastric_retention_time_h <= floating_result.gastric_retention_time_h


# ---------------------------------------------------------------------------
# Gastric retention time
# ---------------------------------------------------------------------------


class TestGastricRetentionTime:
    def test_gastric_retention_time_is_positive(self, floating_result):
        assert floating_result.gastric_retention_time_h > 0.0

    def test_gastric_retention_time_within_simulation(self, floating_result):
        assert floating_result.gastric_retention_time_h <= 24.0

    def test_gastric_retention_time_at_90pct_release(self, swellable_result):
        # At gastric_retention_time_h, fraction_released should be >= 0.9 OR it's t_end
        grt = swellable_result.gastric_retention_time_h
        if grt < 24.0:
            # Find the fraction at that time
            idx = min(
                range(len(swellable_result.times_h)),
                key=lambda i: abs(swellable_result.times_h[i] - grt),
            )
            assert swellable_result.fraction_released[idx] >= 0.9 - 1e-6


# ---------------------------------------------------------------------------
# Plasma PK metrics
# ---------------------------------------------------------------------------


class TestPKMetrics:
    def test_cmax_positive(self, floating_result):
        assert floating_result.cmax_mg_L > 0.0

    def test_tmax_in_range(self, floating_result):
        assert 0.0 <= floating_result.tmax_h <= 24.0

    def test_auc_positive(self, floating_result):
        assert floating_result.auc_mg_h_per_L > 0.0

    def test_auc_trapezoidal_consistency(self, floating_result):
        # Verify AUC manually
        times = floating_result.times_h
        plasma = floating_result.c_plasma_mg_L
        expected_auc = sum(
            0.5 * (plasma[i] + plasma[i - 1]) * (times[i] - times[i - 1])
            for i in range(1, len(times))
        )
        assert floating_result.auc_mg_h_per_L == pytest.approx(expected_auc, rel=1e-6)

    def test_cmax_equals_max_plasma(self, floating_result):
        assert floating_result.cmax_mg_L == pytest.approx(max(floating_result.c_plasma_mg_L))


# ---------------------------------------------------------------------------
# Sustained release duration
# ---------------------------------------------------------------------------


class TestSustainedRelease:
    def test_sustained_duration_positive(self, floating_result):
        assert floating_result.dose_release_sustained_h > 0.0

    def test_sustained_duration_less_than_t_end(self, floating_result):
        assert floating_result.dose_release_sustained_h <= 24.0 + 0.1

    def test_higher_dose_longer_sustained(self):
        r_low = simulate_gastric_retention(
            drug_name="D",
            dose_mg=50.0,
            formulation_type="floating",
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
            t_end_h=24.0,
        )
        r_high = simulate_gastric_retention(
            drug_name="D",
            dose_mg=200.0,
            formulation_type="floating",
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
            t_end_h=24.0,
        )
        assert r_high.dose_release_sustained_h >= r_low.dose_release_sustained_h


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_gastric_retention("D", -10.0, "floating", 2.0, 20.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_gastric_retention("D", 0.0, "floating", 2.0, 20.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            simulate_gastric_retention("D", 100.0, "floating", 0.0, 20.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            simulate_gastric_retention("D", 100.0, "floating", 2.0, -5.0)

    def test_invalid_formulation_type_raises(self):
        with pytest.raises(ValueError, match="formulation_type"):
            simulate_gastric_retention("D", 100.0, "expandable", 2.0, 20.0)

    def test_invalid_solubility_raises(self):
        with pytest.raises(ValueError, match="solubility"):
            simulate_gastric_retention("D", 100.0, "floating", 2.0, 20.0, solubility_mg_mL=0.0)

    def test_all_formulation_types_valid(self):
        for ft in _VALID_FORMULATIONS:
            result = simulate_gastric_retention("D", 100.0, ft, 2.0, 20.0, t_end_h=10.0)
            assert isinstance(result, GastricRetentionResult)


# ---------------------------------------------------------------------------
# Release rate
# ---------------------------------------------------------------------------


class TestReleaseRate:
    def test_release_rate_starts_positive(self, floating_result):
        # Initial release rate = k_release * dose_mg
        expected = 0.05 * 100.0
        assert floating_result.release_rate_mg_h[0] == pytest.approx(expected, rel=1e-6)

    def test_release_rate_decays(self, floating_result):
        # Release rate should decrease as depot empties
        assert floating_result.release_rate_mg_h[-1] < floating_result.release_rate_mg_h[0]
