"""Tests for clinical/concentration_effect_modeling.py — Phase 598."""

import pytest

from omega_pbpk.clinical.concentration_effect_modeling import (
    ConcentrationEffectResult,
    effect_site_model,
    emax_model,
    fit_emax_model,
    linear_model,
    sigmoid_emax_model,
)


def test_emax_model_returns_list():
    result = emax_model([1.0, 5.0, 10.0], emax=100.0, ec50=5.0)
    assert isinstance(result, list)
    assert len(result) == 3


def test_emax_at_ec50_is_half_max():
    ec50 = 5.0
    emax = 100.0
    result = emax_model([ec50], emax=emax, ec50=ec50)
    assert abs(result[0] - emax / 2.0) < 1e-6


def test_emax_at_zero_is_e0():
    result = emax_model([0.0001], emax=100.0, ec50=5.0, e0=5.0)
    assert abs(result[0] - 5.0) < 0.1


def test_emax_at_high_conc_approaches_max():
    emax = 100.0
    result = emax_model([1e9], emax=emax, ec50=5.0)
    assert abs(result[0] - emax) < 0.001


def test_sigmoid_emax_same_as_emax_at_hill1():
    concs = [1.0, 5.0, 10.0, 50.0]
    emax = 80.0
    ec50 = 10.0
    e_plain = emax_model(concs, emax=emax, ec50=ec50)
    e_sigmoid = sigmoid_emax_model(concs, emax=emax, ec50=ec50, hill_n=1.0)
    for a, b in zip(e_plain, e_sigmoid):
        assert abs(a - b) < 1e-9


def test_sigmoid_steeper_with_higher_hill():
    # At concentration = EC50/2 (below EC50), hill=3 gives less effect than hill=1
    # At concentration = 2*EC50 (above EC50), hill=3 gives more effect than hill=1
    ec50 = 10.0
    emax = 100.0
    c_above = [2 * ec50]
    e_n1 = sigmoid_emax_model(c_above, emax=emax, ec50=ec50, hill_n=1.0)
    e_n3 = sigmoid_emax_model(c_above, emax=emax, ec50=ec50, hill_n=3.0)
    assert e_n3[0] > e_n1[0]


def test_linear_model_proportional():
    concs = [1.0, 2.0, 3.0]
    result = linear_model(concs, slope=2.0)
    assert abs(result[0] - 2.0) < 1e-9
    assert abs(result[1] - 4.0) < 1e-9
    assert abs(result[2] - 6.0) < 1e-9


def test_linear_model_with_intercept():
    result = linear_model([5.0], slope=1.0, intercept=10.0)
    assert abs(result[0] - 15.0) < 1e-9


def test_fit_returns_result():
    concs = [1.0, 2.0, 5.0, 10.0, 20.0]
    # Generate data from true Emax=100, EC50=5, n=1
    obs = [100.0 * c / (5.0 + c) for c in concs]
    result = fit_emax_model(concs, obs)
    assert isinstance(result, ConcentrationEffectResult)


def test_fit_ec50_close_to_true():
    true_ec50 = 5.0
    concs = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    obs = [100.0 * c / (true_ec50 + c) for c in concs]
    result = fit_emax_model(concs, obs)
    assert abs(result.ec50 - true_ec50) / true_ec50 < 0.5


def test_fit_emax_close_to_true():
    true_emax = 100.0
    true_ec50 = 5.0
    concs = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    obs = [true_emax * c / (true_ec50 + c) for c in concs]
    result = fit_emax_model(concs, obs)
    assert abs(result.emax - true_emax) / true_emax < 0.30


def test_fit_r_squared_positive():
    concs = [1.0, 5.0, 10.0, 50.0]
    obs = [20.0, 50.0, 66.7, 90.9]
    result = fit_emax_model(concs, obs)
    assert result.r_squared >= 0.0


def test_fit_r_squared_max_1():
    concs = [1.0, 5.0, 10.0, 50.0]
    obs = [20.0, 50.0, 66.7, 90.9]
    result = fit_emax_model(concs, obs)
    assert result.r_squared <= 1.0


def test_effect_site_model_returns_tuple():
    times = [0.0, 1.0, 2.0, 4.0, 8.0]
    cp = [0.0, 10.0, 8.0, 4.0, 1.0]
    result = effect_site_model(times, cp, ke0_per_h=1.0, emax=100.0, ec50=5.0)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_effect_site_ce_positive_after_time():
    times = [0.0, 1.0, 2.0, 4.0, 8.0]
    cp = [0.0, 10.0, 8.0, 4.0, 1.0]
    c_effect, effects = effect_site_model(times, cp, ke0_per_h=2.0, emax=100.0, ec50=5.0)
    # After some time with drug, effect site should accumulate
    assert c_effect[-1] > 0 or c_effect[2] > 0


def test_effect_site_ce_tracks_plasma():
    times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    # Constant plasma concentration
    cp = [10.0] * len(times)
    c_effect, effects = effect_site_model(times, cp, ke0_per_h=5.0, emax=100.0, ec50=5.0, dt_h=0.05)
    # Ce should rise toward Cp = 10
    assert c_effect[-1] > c_effect[0]


def test_effect_site_effect_positive():
    times = [0.0, 1.0, 2.0, 4.0]
    cp = [0.0, 20.0, 20.0, 20.0]
    _, effects = effect_site_model(times, cp, ke0_per_h=2.0, emax=100.0, ec50=5.0)
    # At least one effect value should be positive
    assert max(effects) > 0


def test_effect_at_ec50_is_half_emax():
    concs = [1.0, 5.0, 10.0, 50.0]
    obs = [20.0, 50.0, 66.7, 90.9]
    result = fit_emax_model(concs, obs)
    expected = result.e0 + result.emax / 2.0
    assert abs(result.effect_at_ec50 - expected) < 1e-9


def test_effect_at_2x_ec50_gt_ec50_effect():
    concs = [1.0, 5.0, 10.0, 50.0]
    obs = [20.0, 50.0, 66.7, 90.9]
    result = fit_emax_model(concs, obs)
    assert result.effect_at_2x_ec50 > result.effect_at_ec50


def test_invalid_empty_concentrations_raises():
    with pytest.raises(ValueError):
        emax_model([], emax=100.0, ec50=5.0)
    with pytest.raises(ValueError):
        sigmoid_emax_model([], emax=100.0, ec50=5.0)
    with pytest.raises(ValueError):
        fit_emax_model([], [])


def test_invalid_ec50_zero_raises():
    with pytest.raises(ValueError):
        emax_model([1.0, 2.0], emax=100.0, ec50=0.0)
    with pytest.raises(ValueError):
        sigmoid_emax_model([1.0], emax=100.0, ec50=0.0)


def test_invalid_emax_zero_raises():
    with pytest.raises(ValueError):
        emax_model([1.0], emax=0.0, ec50=5.0)
    with pytest.raises(ValueError):
        sigmoid_emax_model([1.0], emax=0.0, ec50=5.0)


def test_invalid_ke0_zero_raises():
    with pytest.raises(ValueError):
        effect_site_model([0.0, 1.0], [0.0, 10.0], ke0_per_h=0.0, emax=100.0, ec50=5.0)


def test_sigmoid_monotone_increasing():
    concs = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    emax = 100.0
    ec50 = 10.0
    effects = sigmoid_emax_model(concs, emax=emax, ec50=ec50, hill_n=2.0)
    for i in range(1, len(effects)):
        assert effects[i] >= effects[i - 1]


def test_notes_nonempty():
    concs = [1.0, 5.0, 10.0, 50.0]
    obs = [20.0, 50.0, 66.7, 90.9]
    result = fit_emax_model(concs, obs)
    assert len(result.notes) > 0
