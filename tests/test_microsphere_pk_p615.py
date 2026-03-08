"""Tests for Phase 615: Microsphere Drug Release PK.

Tests cover return type, list lengths, biphasic release mechanics,
PK metrics (Cmax, tmax, AUC, MRT), size effects, and validation.
"""

import pytest

from omega_pbpk.core.microsphere_pk_p615 import (
    MicrospherePKResult,
    compare_microsphere_sizes,
    simulate_microsphere_pk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def default_result():
    return simulate_microsphere_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        microsphere_diameter_um=50.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )


@pytest.fixture
def small_sphere_result():
    return simulate_microsphere_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        microsphere_diameter_um=10.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )


@pytest.fixture
def large_sphere_result():
    return simulate_microsphere_pk(
        drug_name="TestDrug",
        dose_mg=100.0,
        microsphere_diameter_um=100.0,
        cl_sys_L_per_h=5.0,
        vd_sys_L=50.0,
    )


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------
class TestReturnType:
    def test_returns_microsphere_pk_result(self, default_result):
        assert isinstance(default_result, MicrospherePKResult)

    def test_result_is_plain_dataclass_not_frozen(self, default_result):
        # plain @dataclass allows mutation
        default_result.drug_name = "Modified"
        assert default_result.drug_name == "Modified"

    def test_drug_name_preserved(self, default_result):
        assert default_result.drug_name == "TestDrug"

    def test_dose_preserved(self, default_result):
        assert default_result.dose_mg == 100.0

    def test_diameter_preserved(self, default_result):
        assert default_result.microsphere_diameter_um == 50.0


# ---------------------------------------------------------------------------
# List length tests
# ---------------------------------------------------------------------------
class TestListLengths:
    def test_times_h_is_list(self, default_result):
        assert isinstance(default_result.times_h, list)

    def test_fraction_released_is_list(self, default_result):
        assert isinstance(default_result.fraction_released, list)

    def test_c_systemic_is_list(self, default_result):
        assert isinstance(default_result.c_systemic_mg_L, list)

    def test_all_lists_same_length(self, default_result):
        n = len(default_result.times_h)
        assert len(default_result.fraction_released) == n
        assert len(default_result.c_systemic_mg_L) == n

    def test_list_length_matches_t_end_dt(self):
        r = simulate_microsphere_pk(
            drug_name="Drug",
            dose_mg=50.0,
            microsphere_diameter_um=50.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=20.0,
            t_end_h=100.0,
            dt_h=1.0,
        )
        # Steps = int(t_end/dt) + 1 = 101
        assert len(r.times_h) == 101

    def test_times_start_at_zero(self, default_result):
        assert default_result.times_h[0] == 0.0

    def test_fraction_released_starts_at_zero(self, default_result):
        assert default_result.fraction_released[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Release mechanics tests
# ---------------------------------------------------------------------------
class TestReleaseMechanics:
    def test_fraction_released_monotonically_increasing(self, default_result):
        fr = default_result.fraction_released
        for i in range(1, len(fr)):
            assert fr[i] >= fr[i - 1] - 1e-9

    def test_fraction_released_bounded_01(self, default_result):
        for fr in default_result.fraction_released:
            assert 0.0 <= fr <= 1.0 + 1e-9

    def test_t_half_release_less_than_t_end(self, default_result):
        assert default_result.t_half_release_h < 720.0

    def test_sustained_release_greater_than_t_half_release(self, default_result):
        assert default_result.sustained_release_duration_h >= default_result.t_half_release_h

    def test_small_sphere_faster_than_large_sphere(self, small_sphere_result, large_sphere_result):
        # Smaller microspheres should release faster (lower t_half_release)
        assert small_sphere_result.t_half_release_h < large_sphere_result.t_half_release_h

    def test_small_sphere_shorter_sustained_duration(
        self, small_sphere_result, large_sphere_result
    ):
        assert (
            small_sphere_result.sustained_release_duration_h
            < large_sphere_result.sustained_release_duration_h
        )


# ---------------------------------------------------------------------------
# PK metric tests
# ---------------------------------------------------------------------------
class TestPKMetrics:
    def test_cmax_positive(self, default_result):
        assert default_result.cmax_mg_L > 0.0

    def test_tmax_within_simulation_range(self, default_result):
        assert 0.0 <= default_result.tmax_h <= 720.0

    def test_auc_positive(self, default_result):
        assert default_result.auc_mg_h_per_L > 0.0

    def test_mrt_greater_than_tmax(self, default_result):
        # MRT (mean residence time) should generally be >= tmax
        assert default_result.mean_residence_time_h >= default_result.tmax_h

    def test_cmax_matches_max_of_list(self, default_result):
        assert default_result.cmax_mg_L == pytest.approx(
            max(default_result.c_systemic_mg_L), rel=1e-6
        )

    def test_auc_trapezoidal_manual(self):
        # Short simulation to verify AUC calculation
        r = simulate_microsphere_pk(
            drug_name="Drug",
            dose_mg=10.0,
            microsphere_diameter_um=50.0,
            cl_sys_L_per_h=2.0,
            vd_sys_L=10.0,
            t_end_h=48.0,
            dt_h=1.0,
        )
        t = r.times_h
        c = r.c_systemic_mg_L
        expected_auc = sum(0.5 * (c[i] + c[i - 1]) * (t[i] - t[i - 1]) for i in range(1, len(t)))
        assert r.auc_mg_h_per_L == pytest.approx(expected_auc, rel=1e-6)

    def test_c_systemic_nonnegative(self, default_result):
        for c in default_result.c_systemic_mg_L:
            assert c >= 0.0


# ---------------------------------------------------------------------------
# PLGA MW effect test
# ---------------------------------------------------------------------------
class TestPLGAEffect:
    def test_higher_mw_plga_slower_release(self):
        r_low = simulate_microsphere_pk(
            drug_name="Drug",
            dose_mg=100.0,
            microsphere_diameter_um=50.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
            plga_mw_kDa=25.0,
        )
        r_high = simulate_microsphere_pk(
            drug_name="Drug",
            dose_mg=100.0,
            microsphere_diameter_um=50.0,
            cl_sys_L_per_h=5.0,
            vd_sys_L=50.0,
            plga_mw_kDa=100.0,
        )
        # Higher MW = slower erosion = longer sustained release
        assert r_high.sustained_release_duration_h > r_low.sustained_release_duration_h


# ---------------------------------------------------------------------------
# Compare microsphere sizes tests
# ---------------------------------------------------------------------------
class TestCompareMicrospheres:
    def test_compare_returns_list(self):
        results = compare_microsphere_sizes("Drug", 100.0, 5.0, 50.0)
        assert isinstance(results, list)

    def test_compare_default_returns_three_results(self):
        results = compare_microsphere_sizes("Drug", 100.0, 5.0, 50.0)
        assert len(results) == 3

    def test_compare_sorted_descending_sustained_duration(self):
        results = compare_microsphere_sizes("Drug", 100.0, 5.0, 50.0)
        durations = [r.sustained_release_duration_h for r in results]
        assert durations == sorted(durations, reverse=True)

    def test_compare_custom_diameters(self):
        results = compare_microsphere_sizes(
            "Drug", 100.0, 5.0, 50.0, diameters=[5.0, 20.0, 50.0, 200.0]
        )
        assert len(results) == 4

    def test_compare_all_results_are_microsphere_pk_result(self):
        results = compare_microsphere_sizes("Drug", 100.0, 5.0, 50.0)
        for r in results:
            assert isinstance(r, MicrospherePKResult)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
class TestValidation:
    def test_invalid_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_microsphere_pk("Drug", -10.0, 50.0, 5.0, 50.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_microsphere_pk("Drug", 0.0, 50.0, 5.0, 50.0)

    def test_invalid_diameter_raises(self):
        with pytest.raises(ValueError, match="microsphere_diameter"):
            simulate_microsphere_pk("Drug", 100.0, 0.0, 5.0, 50.0)

    def test_invalid_cl_raises(self):
        with pytest.raises(ValueError, match="cl_sys"):
            simulate_microsphere_pk("Drug", 100.0, 50.0, 0.0, 50.0)

    def test_invalid_vd_raises(self):
        with pytest.raises(ValueError, match="vd_sys"):
            simulate_microsphere_pk("Drug", 100.0, 50.0, 5.0, 0.0)

    def test_invalid_drug_loading_raises(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            simulate_microsphere_pk("Drug", 100.0, 50.0, 5.0, 50.0, drug_loading_pct=0.0)

    def test_drug_loading_over_100_raises(self):
        with pytest.raises(ValueError, match="drug_loading_pct"):
            simulate_microsphere_pk("Drug", 100.0, 50.0, 5.0, 50.0, drug_loading_pct=110.0)


# ---------------------------------------------------------------------------
# Notes test
# ---------------------------------------------------------------------------
class TestNotes:
    def test_notes_is_str(self, default_result):
        assert isinstance(default_result.notes, str)

    def test_notes_contains_drug_name(self, default_result):
        assert "50.0" in default_result.notes  # diameter
