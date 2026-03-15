"""Tests for Bayesian individual PK parameter estimation."""
import numpy as np


def test_fit_individual_recovers_params():
    """Given simulated C(t) data, fitting should recover approximate CL/Vd."""
    from omega_pbpk.ml.models.foundation.individual_estimation import fit_individual

    dose = 100.0
    vd_true = 50.0
    cl_true = 5.0
    ka = 1.0
    ke = cl_true / vd_true
    F = 0.8

    times = [0.5, 1.0, 2.0, 4.0, 8.0]
    concs = []
    for t in times:
        c = (F * dose / vd_true) * (ka / (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
        concs.append(max(c, 0.0))

    observations = list(zip(times, concs))
    result = fit_individual(
        observations=observations,
        dose_mg=dose,
        base_cl=10.0,  # start from wrong value
        base_vd=100.0,  # start from wrong value
    )

    assert 0.3 < result["cl_scale"] < 2.0
    assert 0.3 < result["vd_scale"] < 2.0
    assert "cl_individual" in result
    assert "vd_individual" in result
    assert "residual" in result


def test_fit_individual_single_observation():
    """Even with 1 observation, fitting should not crash."""
    from omega_pbpk.ml.models.foundation.individual_estimation import fit_individual

    result = fit_individual(
        observations=[(2.0, 1.5)],
        dose_mg=100.0,
        base_cl=5.0,
        base_vd=50.0,
    )
    assert "cl_scale" in result
    assert "vd_scale" in result
    assert result["cl_scale"] > 0
    assert result["vd_scale"] > 0
