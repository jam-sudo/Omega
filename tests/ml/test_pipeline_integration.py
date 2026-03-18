"""Pipeline integration tests for ML corrections."""

import pytest


@pytest.mark.slow
def test_pipeline_with_ml_corrections():
    """Pipeline should use ML corrections when available."""
    from omega_pbpk.pipeline import OmegaPipeline, SimulationRequest

    pipeline = OmegaPipeline()
    pipeline.use_ml_corrections = True
    result = pipeline.simulate(
        SimulationRequest(
            smiles="c1ccc2c(c1)c(=O)c1c(n2C)cc(Cl)c(F)c1",  # midazolam
            dose_mg=7.5,
        )
    )
    assert result.cmax_mg_L > 0
    assert result.auc0t_mg_h_L > 0


def test_pipeline_ml_corrections_disabled_by_default():
    """ML corrections should be opt-in to preserve backward compatibility."""
    from omega_pbpk.pipeline import OmegaPipeline

    pipeline = OmegaPipeline()
    assert not getattr(pipeline, "use_ml_corrections", True)
