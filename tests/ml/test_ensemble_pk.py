"""Tests for PBPK + ML ensemble."""

import numpy as np


class TestEnsemblePK:
    def test_ensemble_returns_positive(self):
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import ensemble_cmax

        result = ensemble_cmax(cmax_pbpk=5.0, cmax_ml=3.0, confidence="high")
        assert result > 0

    def test_high_confidence_between_inputs(self):
        """Result should be between the two inputs (geometric mean)."""
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import ensemble_cmax

        result = ensemble_cmax(cmax_pbpk=5.0, cmax_ml=3.0, confidence="high")
        assert 3.0 <= result <= 5.0

    def test_low_confidence_favors_ml(self):
        """Low confidence should weight ML more → result closer to ML."""
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import ensemble_cmax

        result_low = ensemble_cmax(cmax_pbpk=10.0, cmax_ml=2.0, confidence="low")
        result_high = ensemble_cmax(cmax_pbpk=10.0, cmax_ml=2.0, confidence="high")
        assert result_low < result_high

    def test_scale_ct_curve(self):
        """C(t) curve should be scaled to match target Cmax."""
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import scale_ct_curve

        time_h = np.array([0, 1, 2, 3, 4])
        cp = np.array([0.0, 5.0, 3.0, 2.0, 1.0])
        target_cmax = 10.0

        scaled = scale_ct_curve(time_h, cp, target_cmax)
        assert np.isclose(np.max(scaled), target_cmax)
        assert np.argmax(scaled) == np.argmax(cp)

    def test_scale_preserves_shape(self):
        """Scaling should preserve relative concentrations."""
        from omega_pbpk.ml.models.direct_pk.ensemble_pk import scale_ct_curve

        time_h = np.array([0, 1, 2, 3])
        cp = np.array([0.0, 4.0, 2.0, 1.0])
        scaled = scale_ct_curve(time_h, cp, 8.0)
        # All values should double
        np.testing.assert_allclose(scaled, cp * 2.0)
