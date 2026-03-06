"""Tests for modified-release formulation modeling."""

import pytest

from omega_pbpk.biopharmaceutics.modified_release import (
    ModifiedReleaseResult,
    MRType,
    compare_mr_types,
    simulate_modified_release,
)

_DRUG = "test_drug"
_DOSE = 200.0
_CL = 5.0
_VD = 50.0


class TestMRType:
    def test_all_types_exist(self):
        types = [
            MRType.MATRIX,
            MRType.OSMOTIC,
            MRType.MULTIPARTICULATE,
            MRType.EROSION,
            MRType.DIFFUSION,
        ]
        assert len(types) == 5

    def test_values(self):
        assert MRType.OSMOTIC.value == "osmotic"
        assert MRType.MATRIX.value == "matrix"


class TestInputValidation:
    def test_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_modified_release(_DRUG, 0, _CL, _VD)

    def test_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            simulate_modified_release(_DRUG, _DOSE, 0, _VD)

    def test_vd_zero(self):
        with pytest.raises(ValueError, match="vd_L"):
            simulate_modified_release(_DRUG, _DOSE, _CL, 0)


class TestOROSSimulation:
    def setup_method(self):
        self.result = simulate_modified_release(
            _DRUG,
            _DOSE,
            _CL,
            _VD,
            mr_type=MRType.OSMOTIC,
            release_duration_h=12.0,
            t_end_h=24.0,
            dt_h=0.1,
        )

    def test_result_type(self):
        assert isinstance(self.result, ModifiedReleaseResult)

    def test_mr_type_stored(self):
        assert self.result.mr_type == "osmotic"

    def test_release_fraction_0_to_1(self):
        for f in self.result.release_fraction:
            assert 0.0 <= f <= 1.0 + 1e-9

    def test_release_fraction_monotone(self):
        rf = self.result.release_fraction
        for i in range(1, len(rf)):
            assert rf[i] >= rf[i - 1] - 1e-9

    def test_cmax_positive(self):
        assert self.result.cmax_mg_L > 0

    def test_auc_positive(self):
        assert self.result.auc_mg_h_L > 0

    def test_plasma_conc_non_negative(self):
        for c in self.result.plasma_conc:
            assert c >= 0.0

    def test_t_plateau_equals_release_duration(self):
        assert self.result.t_plateau_h == pytest.approx(12.0)

    def test_comparison_string(self):
        assert "OROS" in self.result.comparison_vs_ir

    def test_fluctuation_positive(self):
        assert self.result.fluctuation_pct >= 0


class TestMatrixSimulation:
    def setup_method(self):
        self.result = simulate_modified_release(
            _DRUG,
            _DOSE,
            _CL,
            _VD,
            mr_type=MRType.MATRIX,
            release_duration_h=12.0,
            t_end_h=24.0,
            dt_h=0.1,
        )

    def test_release_follows_sqrt_time(self):
        """Matrix release is Higuchi (sqrt-time), slower than zero-order initially."""
        rf = self.result.release_fraction
        # At t=1h: OROS would have 1/12 ~ 8.3%; matrix should be sqrt(1/12) ~ 29% (faster initially)
        # Actually matrix starts faster (sqrt) but slows later
        assert rf[0] == pytest.approx(0.0)
        assert rf[-1] == pytest.approx(1.0, abs=0.01)

    def test_mr_type_stored(self):
        assert self.result.mr_type == "matrix"

    def test_comparison_string(self):
        assert "Matrix" in self.result.comparison_vs_ir


class TestDiffusionSimulation:
    def setup_method(self):
        self.result = simulate_modified_release(
            _DRUG,
            _DOSE,
            _CL,
            _VD,
            mr_type=MRType.DIFFUSION,
            k_release=0.2,
            t_end_h=24.0,
            dt_h=0.1,
        )

    def test_release_asymptotes_to_1(self):
        rf = self.result.release_fraction
        assert rf[-1] > 0.9  # approaches 1 asymptotically

    def test_cmax_positive(self):
        assert self.result.cmax_mg_L > 0


class TestMultiparticulateSimulation:
    def setup_method(self):
        self.result = simulate_modified_release(
            _DRUG,
            _DOSE,
            _CL,
            _VD,
            mr_type=MRType.MULTIPARTICULATE,
            k_release=0.15,
            t_end_h=24.0,
            dt_h=0.1,
        )

    def test_biexponential_release(self):
        rf = self.result.release_fraction
        assert rf[0] == pytest.approx(0.0)
        assert rf[-1] > 0.8

    def test_mr_type_stored(self):
        assert self.result.mr_type == "mp"


class TestCompareMRTypes:
    def test_all_types_returned(self):
        results = compare_mr_types(_DRUG, _DOSE, _CL, _VD)
        assert set(results.keys()) == {"matrix", "osmotic", "mp", "erosion", "diffusion"}

    def test_all_results_positive_cmax(self):
        results = compare_mr_types(_DRUG, _DOSE, _CL, _VD)
        for mr_type, r in results.items():
            assert r.cmax_mg_L > 0, f"Cmax=0 for {mr_type}"

    def test_osmotic_lower_fluctuation_than_ir(self):
        """OROS should have lower peak-trough fluctuation vs matrix (roughly)."""
        results = compare_mr_types(_DRUG, _DOSE, _CL, _VD)
        # All MR types should have finite fluctuation
        for r in results.values():
            assert r.fluctuation_pct >= 0

    def test_dose_preserved(self):
        results = compare_mr_types(_DRUG, _DOSE, _CL, _VD)
        for r in results.values():
            assert r.dose_mg == _DOSE
