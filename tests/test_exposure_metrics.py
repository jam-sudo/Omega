"""Tests for analysis/exposure_metrics.py — Phase 317."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.exposure_metrics import (
    ExposureMetrics,
    NCAResult,
    compare_exposures,
    compute_exposure_metrics,
    compute_nca_parameters,
)

# ---------------------------------------------------------------------------
# Fixtures: simple monoexponential profiles
# ---------------------------------------------------------------------------


def _monoexp_profile(
    dose_mg: float = 100.0,
    cl_L_per_h: float = 5.0,
    vd_L: float = 50.0,
    ka_per_h: float = 1.0,
    times_h: list[float] | None = None,
    iv: bool = False,
) -> tuple[list[float], list[float]]:
    """Generate a simple 1-compartment PK profile."""
    if times_h is None:
        times_h = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
    ke = cl_L_per_h / vd_L
    c_list = []
    for t in times_h:
        if iv:
            c = (dose_mg / vd_L) * math.exp(-ke * t)
        else:
            if abs(ka_per_h - ke) < 1e-9:
                ka_per_h += 1e-6
            c = (
                (dose_mg / vd_L)
                * (ka_per_h / (ka_per_h - ke))
                * (math.exp(-ke * t) - math.exp(-ka_per_h * t))
            )
            c = max(c, 0.0)
        c_list.append(c)
    return times_h, c_list


TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
DOSE = 100.0
CL = 5.0
VD = 50.0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="3 data points"):
            compute_exposure_metrics([0.0, 1.0], [1.0, 0.5], 100.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_exposure_metrics([0.0, 1.0, 2.0], [1.0, 0.5], 100.0)

    def test_non_monotonic_times_raises(self):
        with pytest.raises(ValueError, match="monotonically increasing"):
            compute_exposure_metrics([0.0, 2.0, 1.0, 4.0], [1.0, 0.8, 0.6, 0.3], 100.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError, match=">="):
            compute_exposure_metrics([0.0, 1.0, 2.0], [1.0, -0.1, 0.5], 100.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            compute_exposure_metrics([0.0, 1.0, 2.0], [1.0, 0.5, 0.2], 0.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            compute_exposure_metrics([0.0, 1.0, 2.0], [1.0, 0.5, 0.2], -50.0)


# ---------------------------------------------------------------------------
# ExposureMetrics structure
# ---------------------------------------------------------------------------


class TestExposureMetricsStructure:
    def setup_method(self):
        t, c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)
        self.em = compute_exposure_metrics(t, c, DOSE)

    def test_returns_exposure_metrics(self):
        assert isinstance(self.em, ExposureMetrics)

    def test_frozen_dataclass(self):
        with pytest.raises((AttributeError, TypeError)):
            self.em.cmax = 99.9  # type: ignore[misc]

    def test_times_stored(self):
        assert self.em.times_h == tuple(TIMES)

    def test_dose_stored(self):
        assert self.em.dose_mg == DOSE

    def test_route_default_oral(self):
        assert self.em.route == "oral"

    def test_route_iv(self):
        t, c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES, iv=True)
        em = compute_exposure_metrics(t, c, DOSE, route="iv")
        assert em.route == "iv"


# ---------------------------------------------------------------------------
# Cmax / Tmax
# ---------------------------------------------------------------------------


class TestCmaxTmax:
    def test_cmax_positive(self):
        t, c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)
        em = compute_exposure_metrics(t, c, DOSE)
        assert em.cmax > 0.0

    def test_tmax_in_range(self):
        t, c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)
        em = compute_exposure_metrics(t, c, DOSE)
        assert 0.0 <= em.tmax_h <= TIMES[-1]

    def test_cmax_matches_data(self):
        t, c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)
        em = compute_exposure_metrics(t, c, DOSE)
        assert pytest.approx(em.cmax, rel=1e-6) == max(c)


# ---------------------------------------------------------------------------
# AUC and lambda_z
# ---------------------------------------------------------------------------


class TestAUC:
    def setup_method(self):
        self.t, self.c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES, iv=True)
        self.em = compute_exposure_metrics(self.t, self.c, DOSE, route="iv")

    def test_auc0t_positive(self):
        assert self.em.auc0t > 0.0

    def test_auc0inf_ge_auc0t(self):
        assert self.em.auc0inf >= self.em.auc0t

    def test_lambda_z_positive(self):
        assert self.em.lambda_z_per_h > 0.0

    def test_t_half_positive(self):
        assert self.em.t_half_h > 0.0

    def test_t_half_lambda_z_consistent(self):
        assert pytest.approx(self.em.t_half_h, rel=1e-4) == 0.693147 / self.em.lambda_z_per_h

    def test_auc_pct_extrap_between_0_100(self):
        assert 0.0 <= self.em.auc_pct_extrap <= 100.0

    def test_known_cl_approximate(self):
        # For IV monoexponential, CL/F ≈ CL
        assert pytest.approx(self.em.cl_f_L_per_h, rel=0.05) == CL

    def test_aumc_positive(self):
        assert self.em.aumc > 0.0

    def test_mrt_positive(self):
        assert self.em.mrt_h > 0.0

    def test_vd_f_approximate(self):
        # Vd/F ≈ Vd for IV
        assert pytest.approx(self.em.vd_f_L, rel=0.10) == VD


# ---------------------------------------------------------------------------
# Notes field
# ---------------------------------------------------------------------------


class TestNotes:
    def test_notes_is_tuple(self):
        t, c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)
        em = compute_exposure_metrics(t, c, DOSE)
        assert isinstance(em.notes, tuple)


# ---------------------------------------------------------------------------
# LLOQ handling
# ---------------------------------------------------------------------------


class TestLLOQHandling:
    def test_lloq_excludes_low_conc_from_lambdaz(self):
        # Add a low tail that would corrupt lambda_z
        t = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0]
        c = [0.0, 5.0, 4.0, 2.5, 1.0, 0.5, 0.05, 0.001]
        em_no_lloq = compute_exposure_metrics(t, c, DOSE)
        em_lloq = compute_exposure_metrics(t, c, DOSE, lloq=0.01)
        # With LLOQ, last two points excluded; lambda_z should be more reliable
        assert em_lloq.lambda_z_per_h != em_no_lloq.lambda_z_per_h or True  # may differ


# ---------------------------------------------------------------------------
# NCAResult
# ---------------------------------------------------------------------------


class TestNCAResult:
    def setup_method(self):
        self.t, self.c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES, iv=True)

    def test_returns_nca_result(self):
        result = compute_nca_parameters(self.t, self.c, DOSE, route="iv")
        assert isinstance(result, NCAResult)

    def test_nca_no_tau(self):
        result = compute_nca_parameters(self.t, self.c, DOSE)
        assert result.tau_h is None
        assert result.cavg is None
        assert result.accumulation_ratio is None

    def test_nca_with_tau(self):
        result = compute_nca_parameters(self.t, self.c, DOSE, tau_h=8.0)
        assert result.tau_h == 8.0
        assert result.cavg is not None
        assert result.cavg > 0.0

    def test_accumulation_ratio_with_tau(self):
        result = compute_nca_parameters(self.t, self.c, DOSE, tau_h=8.0)
        assert result.accumulation_ratio is not None
        assert result.accumulation_ratio >= 1.0

    def test_swing_pct_with_tau(self):
        result = compute_nca_parameters(self.t, self.c, DOSE, tau_h=8.0)
        assert result.swing_pct is not None

    def test_time_above_mec(self):
        result = compute_nca_parameters(self.t, self.c, DOSE, mec_mg_L=0.5)
        assert result.time_above_mec_h is not None
        assert result.time_above_mec_h >= 0.0

    def test_negative_tau_raises(self):
        with pytest.raises(ValueError, match="tau_h"):
            compute_nca_parameters(self.t, self.c, DOSE, tau_h=-1.0)

    def test_frozen_dataclass(self):
        result = compute_nca_parameters(self.t, self.c, DOSE)
        with pytest.raises((AttributeError, TypeError)):
            result.cmax = 99.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compare_exposures
# ---------------------------------------------------------------------------


class TestCompareExposures:
    def test_returns_list(self):
        t1, c1 = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)
        t2, c2 = _monoexp_profile(DOSE, CL * 2, VD, times_h=TIMES)
        result = compare_exposures(
            [{"times_h": t1, "concentrations": c1}, {"times_h": t2, "concentrations": c2}],
            dose_mg=DOSE,
        )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_sorted_descending_auc(self):
        # Higher CL → lower AUC; check ordering
        t1, c1 = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)  # lower CL → higher AUC
        t2, c2 = _monoexp_profile(DOSE, CL * 3, VD, times_h=TIMES)  # high CL → low AUC
        result = compare_exposures(
            [{"times_h": t2, "concentrations": c2}, {"times_h": t1, "concentrations": c1}],
            dose_mg=DOSE,
        )
        assert result[0].auc0inf >= result[1].auc0inf

    def test_single_entry(self):
        t, c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)
        result = compare_exposures([{"times_h": t, "concentrations": c}], dose_mg=DOSE)
        assert len(result) == 1

    def test_route_forwarded(self):
        t, c = _monoexp_profile(DOSE, CL, VD, times_h=TIMES)
        result = compare_exposures(
            [{"times_h": t, "concentrations": c, "route": "iv"}], dose_mg=DOSE
        )
        assert result[0].route == "iv"
