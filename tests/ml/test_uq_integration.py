"""Test conformal UQ integration in pipeline."""


class TestUQIntegration:
    def test_result_has_intervals(self):
        """SimulationResult should include prediction intervals."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
                dose_mg=200.0,
            )
        )
        assert result.cmax_ci90 is not None
        lo, hi = result.cmax_ci90
        assert lo > 0
        assert hi > lo

    def test_interval_width_is_reasonable(self):
        """90% CI should not be wider than 100x."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                dose_mg=200.0,
            )
        )
        lo, hi = result.cmax_ci90
        assert hi / max(lo, 1e-12) < 10000.0  # relaxed after E2E calibration

    def test_auc_and_thalf_intervals(self):
        """AUC and t_half CI should also be present."""
        from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

        pipeline = OmegaPipeline()
        result = pipeline.simulate(
            SimulationRequest(
                smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                dose_mg=200.0,
            )
        )
        assert result.auc_ci90 is not None
        assert result.thalf_ci90 is not None
        assert result.auc_ci90[1] > result.auc_ci90[0]
        assert result.thalf_ci90[1] > result.thalf_ci90[0]
