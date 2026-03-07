"""Tests for Simeoni tumor growth inhibition (TGI) model."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.tgi_model import (
    TGIResult,
    simulate_tgi,
    tgi_dose_response,
)

# ── Common parameters ───────────────────────────────────────────────

DRUG_NAME = "TestAnticancer"
DOSE_MG = 50.0
CL = 2.0  # L/h
VD = 20.0  # L
K2_ACTIVE = 0.05  # active drug
K2_ZERO = 0.0  # vehicle control


def _base_result(**kwargs) -> TGIResult:
    defaults = dict(
        drug_name=DRUG_NAME,
        dose_mg=DOSE_MG,
        cl_L_per_h=CL,
        vd_L=VD,
        k2_per_mg_per_L_per_h=K2_ACTIVE,
        t_end_h=672.0,
        dt_h=1.0,
    )
    defaults.update(kwargs)
    return simulate_tgi(**defaults)


# ── Input validation ────────────────────────────────────────────────


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_tgi(
                DRUG_NAME, dose_mg=-1.0, cl_L_per_h=CL, vd_L=VD, k2_per_mg_per_L_per_h=K2_ACTIVE
            )

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_tgi(
                DRUG_NAME, dose_mg=0.0, cl_L_per_h=CL, vd_L=VD, k2_per_mg_per_L_per_h=K2_ACTIVE
            )

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_tgi(
                DRUG_NAME, dose_mg=DOSE_MG, cl_L_per_h=0.0, vd_L=VD, k2_per_mg_per_L_per_h=K2_ACTIVE
            )

    def test_negative_cl_raises(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_tgi(
                DRUG_NAME,
                dose_mg=DOSE_MG,
                cl_L_per_h=-1.0,
                vd_L=VD,
                k2_per_mg_per_L_per_h=K2_ACTIVE,
            )

    def test_zero_vd_raises(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_tgi(
                DRUG_NAME, dose_mg=DOSE_MG, cl_L_per_h=CL, vd_L=0.0, k2_per_mg_per_L_per_h=K2_ACTIVE
            )

    def test_negative_k2_raises(self):
        with pytest.raises(ValueError, match="k2"):
            simulate_tgi(
                DRUG_NAME, dose_mg=DOSE_MG, cl_L_per_h=CL, vd_L=VD, k2_per_mg_per_L_per_h=-0.1
            )

    def test_zero_lambda0_raises(self):
        with pytest.raises(ValueError, match="lambda0"):
            simulate_tgi(
                DRUG_NAME,
                dose_mg=DOSE_MG,
                cl_L_per_h=CL,
                vd_L=VD,
                k2_per_mg_per_L_per_h=K2_ACTIVE,
                lambda0_per_h=0.0,
            )

    def test_zero_lambda1_raises(self):
        with pytest.raises(ValueError, match="lambda1"):
            simulate_tgi(
                DRUG_NAME,
                dose_mg=DOSE_MG,
                cl_L_per_h=CL,
                vd_L=VD,
                k2_per_mg_per_L_per_h=K2_ACTIVE,
                lambda1_g_per_h=0.0,
            )

    def test_zero_x0_raises(self):
        with pytest.raises(ValueError, match="x0_g"):
            simulate_tgi(
                DRUG_NAME,
                dose_mg=DOSE_MG,
                cl_L_per_h=CL,
                vd_L=VD,
                k2_per_mg_per_L_per_h=K2_ACTIVE,
                x0_g=0.0,
            )

    def test_zero_n_doses_raises(self):
        with pytest.raises(ValueError, match="n_doses"):
            simulate_tgi(
                DRUG_NAME,
                dose_mg=DOSE_MG,
                cl_L_per_h=CL,
                vd_L=VD,
                k2_per_mg_per_L_per_h=K2_ACTIVE,
                n_doses=0,
            )


# ── Result structure ─────────────────────────────────────────────────


class TestResultStructure:
    def setup_method(self):
        self.result = _base_result()

    def test_returns_tgi_result(self):
        assert isinstance(self.result, TGIResult)

    def test_drug_name_stored(self):
        assert self.result.drug_name == DRUG_NAME

    def test_dose_stored(self):
        assert self.result.dose_mg == pytest.approx(DOSE_MG)

    def test_route_stored(self):
        assert self.result.route == "iv"

    def test_times_h_is_list(self):
        assert isinstance(self.result.times_h, list)

    def test_c_plasma_is_list(self):
        assert isinstance(self.result.c_plasma_mg_L, list)

    def test_tumor_mass_is_list(self):
        assert isinstance(self.result.tumor_mass_g, list)

    def test_array_lengths_equal(self):
        n = len(self.result.times_h)
        assert len(self.result.c_plasma_mg_L) == n
        assert len(self.result.tumor_mass_g) == n

    def test_notes_is_nonempty_string(self):
        assert isinstance(self.result.notes, str)
        assert len(self.result.notes) > 0

    def test_regression_is_bool(self):
        assert isinstance(self.result.regression, bool)


# ── Tumor growth without drug (vehicle control) ──────────────────────


class TestVehicleControl:
    def setup_method(self):
        self.result = simulate_tgi(
            DRUG_NAME,
            dose_mg=DOSE_MG,
            cl_L_per_h=CL,
            vd_L=VD,
            k2_per_mg_per_L_per_h=0.0,  # no drug effect
            x0_g=0.2,
            t_end_h=672.0,
            dt_h=1.0,
        )

    def test_tumor_grows_without_drug(self):
        assert self.result.tumor_mass_final > self.result.tumor_mass_initial

    def test_no_regression_without_drug(self):
        assert not self.result.regression

    def test_initial_mass_near_x0(self):
        assert self.result.tumor_mass_g[0] == pytest.approx(0.2, rel=0.01)


# ── Tumor regression with high drug dose ─────────────────────────────


class TestTumorRegression:
    def setup_method(self):
        # Use very high k2 to ensure regression
        self.result = simulate_tgi(
            DRUG_NAME,
            dose_mg=500.0,
            cl_L_per_h=CL,
            vd_L=VD,
            k2_per_mg_per_L_per_h=0.5,  # high kill rate
            x0_g=0.2,
            t_end_h=672.0,
            dt_h=1.0,
        )

    def test_regression_true_with_high_k2(self):
        assert self.result.regression

    def test_min_tumor_less_than_initial(self):
        assert self.result.min_tumor_mass_g < self.result.tumor_mass_initial

    def test_tumor_mass_all_positive(self):
        assert all(m >= 0 for m in self.result.tumor_mass_g)


# ── Initial conditions ───────────────────────────────────────────────


class TestInitialConditions:
    def test_tumor_mass_initial_equals_x0(self):
        result = simulate_tgi(
            DRUG_NAME,
            dose_mg=DOSE_MG,
            cl_L_per_h=CL,
            vd_L=VD,
            k2_per_mg_per_L_per_h=K2_ACTIVE,
            x0_g=0.3,
            dt_h=1.0,
        )
        assert result.tumor_mass_g[0] == pytest.approx(0.3, rel=0.01)

    def test_tumor_mass_initial_stored(self):
        result = _base_result(x0_g=0.5)
        assert result.tumor_mass_initial == pytest.approx(result.tumor_mass_g[0], rel=0.01)


# ── Plasma PK ────────────────────────────────────────────────────────


class TestPlasmaConcentration:
    def test_plasma_starts_positive_iv(self):
        result = simulate_tgi(
            DRUG_NAME,
            dose_mg=DOSE_MG,
            cl_L_per_h=CL,
            vd_L=VD,
            k2_per_mg_per_L_per_h=K2_ACTIVE,
            route="iv",
            dt_h=1.0,
        )
        # IV: concentration should appear quickly
        assert max(result.c_plasma_mg_L) > 0

    def test_three_doses_multiple_peaks(self):
        result = simulate_tgi(
            DRUG_NAME,
            dose_mg=50.0,
            cl_L_per_h=CL,
            vd_L=VD,
            k2_per_mg_per_L_per_h=K2_ACTIVE,
            route="iv",
            n_doses=3,
            dose_interval_h=168.0,
            t_end_h=600.0,
            dt_h=1.0,
        )
        assert result.n_doses == 3
        c = result.c_plasma_mg_L
        # With 3 IV doses, concentration should show at least 2 rises
        # (it rises again at each dose time)
        # Find local maxima (simple approach: check count of rising edges)
        rising = sum(1 for i in range(1, len(c)) if c[i] > c[i - 1])
        assert rising >= 2  # at least 2 rising events (dose 2 and 3)

    def test_c_plasma_all_nonnegative(self):
        result = _base_result()
        assert all(c >= 0 for c in result.c_plasma_mg_L)


# ── TGI metrics ──────────────────────────────────────────────────────


class TestTGIMetrics:
    def test_min_tumor_mass_le_initial(self):
        result = _base_result()
        assert result.min_tumor_mass_g <= result.tumor_mass_initial + 1e-9

    def test_time_to_min_in_range(self):
        result = _base_result()
        assert 0 <= result.time_to_min_h <= 672.0

    def test_tumor_growth_inhibition_zero_without_control(self):
        result = _base_result()
        # Without _control_final, TGI% defaults to 0
        assert result.tumor_growth_inhibition_pct == pytest.approx(0.0)


# ── tgi_dose_response ────────────────────────────────────────────────


class TestTGIDoseResponse:
    def setup_method(self):
        self.doses = [10.0, 50.0, 200.0]
        self.results = tgi_dose_response(
            DRUG_NAME,
            doses_mg=self.doses,
            cl_L_per_h=CL,
            vd_L=VD,
            k2=K2_ACTIVE,
            t_end_h=672.0,
            dt_h=2.0,
        )

    def test_returns_correct_count(self):
        assert len(self.results) == 3

    def test_all_tgi_result_type(self):
        assert all(isinstance(r, TGIResult) for r in self.results)

    def test_sorted_by_min_tumor_ascending(self):
        mins = [r.min_tumor_mass_g for r in self.results]
        assert mins == sorted(mins)

    def test_all_tumor_masses_positive(self):
        for r in self.results:
            assert all(m >= 0 for m in r.tumor_mass_g)

    def test_tgi_pct_computed_with_control(self):
        # With control available, TGI% should be computed (may not be 0)
        # At least one result should have nonzero TGI%
        tgi_values = [r.tumor_growth_inhibition_pct for r in self.results]
        # Not all should be zero
        assert any(v != 0.0 for v in tgi_values)
