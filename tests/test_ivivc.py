"""Tests for IVIVC (in vitro–in vivo correlation) module."""

import numpy as np
import pytest

from omega_pbpk.biopharmaceutics.ivivc import (
    IVIVCCorrelation,
    IVIVCResult,
    compute_ivivc,
    wagner_nelson,
)

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------
DISS_TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
DISS_FRACS = [0.0, 0.15, 0.28, 0.48, 0.72, 0.88, 0.96, 1.00]

KE = 0.15  # 1/h
VD = 30.0  # L
DOSE = 100.0  # mg
KA = 0.8  # 1/h
PLASMA_TIMES = np.linspace(0, 24, 97)  # 0 to 24h, 97 points
PLASMA_CONC = (DOSE * KA / (VD * (KA - KE))) * (
    np.exp(-KE * PLASMA_TIMES) - np.exp(-KA * PLASMA_TIMES)
)
PLASMA_CONC = np.maximum(PLASMA_CONC, 0.0)


# ---------------------------------------------------------------------------
# TestWagnerNelson
# ---------------------------------------------------------------------------
class TestWagnerNelson:
    def test_returns_two_arrays(self):
        result = wagner_nelson(PLASMA_TIMES, PLASMA_CONC, KE, VD, DOSE)
        assert isinstance(result, tuple) and len(result) == 2
        t, fa = result
        assert isinstance(t, np.ndarray)
        assert isinstance(fa, np.ndarray)
        assert len(t) == len(fa)

    def test_fa_clipped_0_to_1(self):
        _, fa = wagner_nelson(PLASMA_TIMES, PLASMA_CONC, KE, VD, DOSE)
        assert np.all(fa >= 0.0)
        assert np.all(fa <= 1.0)

    def test_fa_nonnegative(self):
        _, fa = wagner_nelson(PLASMA_TIMES, PLASMA_CONC, KE, VD, DOSE)
        assert np.all(fa >= 0)

    def test_fa_late_approaches_1(self):
        _, fa = wagner_nelson(PLASMA_TIMES, PLASMA_CONC, KE, VD, DOSE)
        assert fa[-1] > 0.8

    def test_invalid_ke_raises(self):
        with pytest.raises(ValueError):
            wagner_nelson(PLASMA_TIMES, PLASMA_CONC, 0, VD, DOSE)
        with pytest.raises(ValueError):
            wagner_nelson(PLASMA_TIMES, PLASMA_CONC, -0.1, VD, DOSE)


# ---------------------------------------------------------------------------
# TestIVIVCCorrelation
# ---------------------------------------------------------------------------
class TestIVIVCCorrelation:
    def test_fields_exist(self):
        c = IVIVCCorrelation(
            level="A",
            r_squared=0.95,
            slope=1.0,
            intercept=0.0,
            passes_criteria=True,
        )
        assert hasattr(c, "level")
        assert hasattr(c, "r_squared")
        assert hasattr(c, "slope")
        assert hasattr(c, "intercept")
        assert hasattr(c, "passes_criteria")

    def test_passes_criteria_a_high_r2(self):
        c = IVIVCCorrelation(
            level="A",
            r_squared=0.95,
            slope=1.0,
            intercept=0.0,
            passes_criteria=True,
        )
        assert c.passes_criteria is True

    def test_fails_criteria_a_low_r2(self):
        c = IVIVCCorrelation(
            level="A",
            r_squared=0.85,
            slope=1.0,
            intercept=0.0,
            passes_criteria=False,
        )
        assert c.passes_criteria is False


# ---------------------------------------------------------------------------
# TestComputeIVIVC (integration)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestComputeIVIVC:
    @pytest.fixture(autouse=True)
    def _result(self):
        self.result = compute_ivivc(
            DISS_TIMES,
            DISS_FRACS,
            list(PLASMA_TIMES),
            list(PLASMA_CONC),
            KE,
            VD,
            DOSE,
        )

    def test_returns_ivivc_result(self):
        assert isinstance(self.result, IVIVCResult)

    def test_level_a_not_none(self):
        assert self.result.level_a is not None

    def test_level_b_not_none(self):
        assert self.result.level_b is not None

    def test_level_c_not_none(self):
        assert self.result.level_c is not None

    def test_f_absorbed_length(self):
        assert len(self.result.f_absorbed) == len(DISS_TIMES)

    def test_f_dissolved_matches_input(self):
        assert self.result.f_dissolved == DISS_FRACS

    def test_mdt_positive(self):
        assert self.result.mdt_in_vitro_h > 0

    def test_mrt_positive(self):
        assert self.result.mrt_in_vivo_h > 0

    def test_recommended_level_valid(self):
        assert self.result.recommended_level in ("A", "B", "C", "none")

    def test_levels_filter_excludes_a(self):
        result_bc = compute_ivivc(
            DISS_TIMES,
            DISS_FRACS,
            list(PLASMA_TIMES),
            list(PLASMA_CONC),
            KE,
            VD,
            DOSE,
            levels=["B", "C"],
        )
        assert result_bc.level_a is None

    def test_level_a_r2_between_0_and_1(self):
        assert self.result.level_a is not None
        assert 0.0 <= self.result.level_a.r_squared <= 1.0

    def test_perfect_ivivc_passes_a(self):
        t_p = np.linspace(0, 12, 25)
        cp_p = (DOSE * KA / (VD * (KA - KE))) * (np.exp(-KE * t_p) - np.exp(-KA * t_p))
        cp_p = np.maximum(cp_p, 0.0)
        _, fa_perfect = wagner_nelson(t_p, cp_p, KE, VD, DOSE)
        perf_times = list(t_p)
        perf_fracs = [float(f) for f in fa_perfect]
        result_perf = compute_ivivc(
            perf_times,
            perf_fracs,
            list(t_p),
            list(cp_p),
            KE,
            VD,
            DOSE,
            levels=["A"],
        )
        assert result_perf.level_a.passes_criteria
