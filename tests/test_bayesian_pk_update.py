"""Tests for analysis/bayesian_pk_update.py (Phase 608)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.bayesian_pk_update import (
    BayesianPKResult,
    bayesian_update,
    predict_1cpt,
)


def _true_concs(times, dose=100.0, cl=5.0, vd=20.0, ka=1.0, f=1.0):
    ke = cl / vd
    concs = []
    for t in times:
        c = f * dose * ka / (vd * (ka - ke)) * (math.exp(-ke * t) - math.exp(-ka * t))
        concs.append(max(0.0, c))
    return concs


OBS_TIMES = [0.5, 1.0, 2.0, 4.0]
OBS_CONCS = _true_concs(OBS_TIMES)

BASE_KWARGS = dict(
    drug_name="TestDrug",
    obs_times_h=OBS_TIMES,
    obs_conc_mg_L=OBS_CONCS,
    prior_cl_L_per_h=5.0,
    prior_vd_L=20.0,
    dose_mg=100.0,
    route="oral",
    prior_ka_per_h=1.0,
    f_oral=1.0,
)


def test_returns_result():
    result = bayesian_update(**BASE_KWARGS)
    assert isinstance(result, BayesianPKResult)


def test_n_observations_correct():
    times = [1.0, 2.0, 4.0, 8.0]
    concs = _true_concs(times)
    result = bayesian_update(**{**BASE_KWARGS, "obs_times_h": times, "obs_conc_mg_L": concs})
    assert result.n_observations == 4


def test_posterior_cl_positive():
    result = bayesian_update(**BASE_KWARGS)
    assert result.posterior_cl_L_per_h > 0.0


def test_posterior_vd_positive():
    result = bayesian_update(**BASE_KWARGS)
    assert result.posterior_vd_L > 0.0


def test_posterior_ka_none_for_iv():
    times = [0.5, 1.0, 2.0, 4.0]
    concs = [100.0 / 20.0 * math.exp(-5.0 / 20.0 * t) for t in times]
    result = bayesian_update(
        **{**BASE_KWARGS, "route": "iv", "obs_times_h": times, "obs_conc_mg_L": concs}
    )
    assert result.posterior_ka_per_h is None


def test_posterior_ka_positive_for_oral():
    result = bayesian_update(**BASE_KWARGS)
    assert result.posterior_ka_per_h is not None
    assert result.posterior_ka_per_h > 0.0


def test_shrinkage_cl_nonnegative():
    result = bayesian_update(**BASE_KWARGS)
    assert result.cl_shrinkage_pct >= 0.0


def test_shrinkage_vd_nonnegative():
    result = bayesian_update(**BASE_KWARGS)
    assert result.vd_shrinkage_pct >= 0.0


def test_rmse_nonnegative():
    result = bayesian_update(**BASE_KWARGS)
    assert result.rmse >= 0.0


def test_predicted_len_matches_obs():
    result = bayesian_update(**BASE_KWARGS)
    assert len(result.predicted_concentrations) == len(OBS_TIMES)


def test_residuals_len_matches_obs():
    result = bayesian_update(**BASE_KWARGS)
    assert len(result.residuals) == len(OBS_TIMES)


def test_auc_positive():
    result = bayesian_update(**BASE_KWARGS)
    assert result.individual_auc_mg_h_L > 0.0


def test_t_half_positive():
    result = bayesian_update(**BASE_KWARGS)
    assert result.individual_t_half_h > 0.0


def test_posterior_near_true_cl():
    """With true CL=5, prior CL=10: posterior should be closer to 5."""
    times = [0.5, 1.0, 2.0, 4.0, 6.0]
    concs = _true_concs(times, cl=5.0, vd=20.0)
    result = bayesian_update(
        drug_name="Test",
        obs_times_h=times,
        obs_conc_mg_L=concs,
        prior_cl_L_per_h=10.0,
        prior_vd_L=20.0,
        dose_mg=100.0,
        route="oral",
        prior_ka_per_h=1.0,
    )
    assert abs(result.posterior_cl_L_per_h - 5.0) < abs(result.prior_cl_L_per_h - 5.0)


def test_posterior_near_true_vd():
    """With true Vd=20, prior Vd=40: posterior should be closer to 20."""
    times = [0.5, 1.0, 2.0, 4.0, 6.0]
    concs = _true_concs(times, cl=5.0, vd=20.0)
    result = bayesian_update(
        drug_name="Test",
        obs_times_h=times,
        obs_conc_mg_L=concs,
        prior_cl_L_per_h=5.0,
        prior_vd_L=40.0,
        dose_mg=100.0,
        route="oral",
        prior_ka_per_h=1.0,
    )
    assert abs(result.posterior_vd_L - 20.0) < abs(result.prior_vd_L - 20.0)


def test_predict_1cpt_iv_at_t0():
    dose = 100.0
    vd = 20.0
    concs = predict_1cpt([0.0], dose, cl_L_per_h=5.0, vd_L=vd, route="iv")
    assert abs(concs[0] - dose / vd) < 1e-9


def test_predict_1cpt_oral_positive():
    concs = predict_1cpt([1.0, 2.0, 4.0], 100.0, cl_L_per_h=5.0, vd_L=20.0, route="oral")
    assert all(c >= 0.0 for c in concs)


def test_predict_1cpt_oral_rises_then_falls():
    times = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    concs = predict_1cpt(times, 100.0, cl_L_per_h=5.0, vd_L=20.0, route="oral", ka_per_h=1.5)
    # Find peak index
    peak_idx = concs.index(max(concs))
    # Must rise before peak and fall after
    assert peak_idx > 0
    assert peak_idx < len(concs) - 1
    assert concs[peak_idx] > concs[0]
    assert concs[-1] < concs[peak_idx]


def test_low_sigma_tighter_posterior():
    """Lower sigma means observations get more weight -> posterior closer to data-generating params."""
    times = [0.5, 1.0, 2.0, 4.0]
    concs = _true_concs(times, cl=5.0, vd=20.0)
    r_low = bayesian_update(
        **{**BASE_KWARGS, "prior_cl_L_per_h": 8.0, "obs_conc_mg_L": concs, "sigma_prop": 0.05}
    )
    r_high = bayesian_update(
        **{**BASE_KWARGS, "prior_cl_L_per_h": 8.0, "obs_conc_mg_L": concs, "sigma_prop": 0.5}
    )
    # Low sigma should pull posterior closer to true CL=5
    assert abs(r_low.posterior_cl_L_per_h - 5.0) <= abs(r_high.posterior_cl_L_per_h - 5.0)


def test_invalid_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        bayesian_update(
            drug_name="X",
            obs_times_h=[1.0, 2.0],
            obs_conc_mg_L=[1.0],
            prior_cl_L_per_h=5.0,
            prior_vd_L=20.0,
            dose_mg=100.0,
        )


def test_invalid_dose_raises():
    with pytest.raises(ValueError):
        bayesian_update(**{**BASE_KWARGS, "dose_mg": -10.0})


def test_invalid_prior_cl_raises():
    with pytest.raises(ValueError):
        bayesian_update(**{**BASE_KWARGS, "prior_cl_L_per_h": 0.0})


def test_notes_nonempty():
    result = bayesian_update(**BASE_KWARGS)
    assert len(result.notes) > 0
