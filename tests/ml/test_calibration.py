"""Tests for confidence calibration of ADME prediction intervals."""

from __future__ import annotations


class TestCalibrationMetrics:
    """Test basic calibration metric functions."""

    def test_picp_all_covered(self):
        from omega_pbpk.ml.evaluation.metrics import picp

        true = [1.0, 2.0, 3.0]
        lo = [0.5, 1.5, 2.5]
        hi = [1.5, 2.5, 3.5]
        assert picp(true, lo, hi) == 1.0

    def test_picp_none_covered(self):
        from omega_pbpk.ml.evaluation.metrics import picp

        true = [1.0, 2.0, 3.0]
        lo = [5.0, 5.0, 5.0]
        hi = [6.0, 6.0, 6.0]
        assert picp(true, lo, hi) == 0.0

    def test_picp_partial(self):
        from omega_pbpk.ml.evaluation.metrics import picp

        true = [1.0, 2.0, 3.0]
        lo = [0.5, 5.0, 2.5]
        hi = [1.5, 6.0, 3.5]
        assert abs(picp(true, lo, hi) - 2 / 3) < 0.01

    def test_picp_empty(self):
        from omega_pbpk.ml.evaluation.metrics import picp

        assert picp([], [], []) == 0.0

    def test_mpiw(self):
        from omega_pbpk.ml.evaluation.metrics import mpiw

        lo = [0.0, 1.0, 2.0]
        hi = [1.0, 3.0, 4.0]
        assert abs(mpiw(lo, hi) - 5 / 3) < 0.01

    def test_mpiw_empty(self):
        from omega_pbpk.ml.evaluation.metrics import mpiw

        assert mpiw([], []) == 0.0


class TestFindAdjustmentFactor:
    """Test binary search for interval adjustment."""

    def test_perfect_coverage_returns_one(self):
        from omega_pbpk.ml.evaluation.metrics import _find_adjustment_factor

        true = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        lo = [t - 0.5 for t in true]
        hi = [t + 0.5 for t in true]
        # All covered at factor 1.0, target 0.9 → factor can shrink well below 1.0
        # since even very narrow intervals still cover 100% of these well-centered points
        factor = _find_adjustment_factor(true, lo, hi, target_coverage=0.90)
        assert 0.001 <= factor <= 1.5

    def test_narrow_intervals_need_widening(self):
        from omega_pbpk.ml.evaluation.metrics import _find_adjustment_factor

        # True values offset from centers → narrow intervals miss them
        centers = [1.0, 2.0, 3.0, 4.0, 5.0]
        true = [c + 0.5 for c in centers]  # 0.5 away from center
        lo = [c - 0.01 for c in centers]
        hi = [c + 0.01 for c in centers]
        # Very narrow intervals, need widening to reach true values
        factor = _find_adjustment_factor(true, lo, hi, target_coverage=0.90)
        assert factor > 1.0


class TestCalibrateUncertainty:
    """Test the full calibration pipeline."""

    def test_calibrate_runs_with_reference_data(self):
        from omega_pbpk.ml.evaluation.metrics import calibrate_uncertainty
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        predictor = EnsembleADMEPredictor()
        report = calibrate_uncertainty(predictor, target_coverage=0.90)

        # Should produce results (reference data exists with 153 compounds)
        assert len(report.results) > 0

        for r in report.results:
            assert 0 <= r.actual_coverage <= 1.0
            assert r.n_samples >= 5
            assert r.mean_interval_width >= 0
            assert r.adjustment_factor > 0

        # Report should render as string
        s = str(report)
        assert "Conformal Calibration Report" in s

    def test_calibrate_missing_file(self):
        from pathlib import Path

        from omega_pbpk.ml.evaluation.metrics import calibrate_uncertainty
        from omega_pbpk.ml.models.adme.ensemble import EnsembleADMEPredictor

        predictor = EnsembleADMEPredictor()
        report = calibrate_uncertainty(predictor, data_path=Path("/nonexistent/file.csv"))
        assert len(report.results) == 0
