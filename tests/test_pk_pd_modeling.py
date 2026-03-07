"""Tests for PK/PD model fitting utilities (Phase 386)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.analysis.pk_pd_modeling import (
    EmaxFitResult,
    IndirectResponseResult,
    fit_emax_model,
    fit_indirect_response_model,
    pdp_index,
)

# ── Emax model fitting ────────────────────────────────────────────────────────


def _generate_emax_data(
    emax: float = 100.0, ec50: float = 10.0, n: float = 1.0
) -> tuple[list[float], list[float]]:
    """Generate synthetic Emax/Hill data."""
    concs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    effects = [emax * c**n / (ec50**n + c**n) for c in concs]
    return concs, effects


def test_fit_emax_returns_result_type():
    concs, effs = _generate_emax_data()
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    assert isinstance(result, EmaxFitResult)


def test_fit_emax_high_r_squared_on_known_data():
    concs, effs = _generate_emax_data(emax=80.0, ec50=5.0, n=1.0)
    result = fit_emax_model(concs, effs, emax_guess=80.0, ec50_guess=5.0)
    assert result.r_squared > 0.95


def test_fit_emax_positive_emax():
    concs, effs = _generate_emax_data()
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    assert result.emax > 0.0


def test_fit_emax_positive_ec50():
    concs, effs = _generate_emax_data()
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    assert result.ec50 > 0.0


def test_fit_emax_hill_n_in_valid_range():
    concs, effs = _generate_emax_data()
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    assert result.hill_n in [0.5, 1.0, 1.5, 2.0]


def test_fit_emax_residuals_length_matches_data():
    concs, effs = _generate_emax_data()
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    assert len(result.residuals) == len(concs)


def test_fit_emax_r_squared_between_0_and_1():
    concs, effs = _generate_emax_data()
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    assert 0.0 <= result.r_squared <= 1.0


def test_fit_emax_notes_not_empty():
    concs, effs = _generate_emax_data()
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    assert len(result.notes) > 0


def test_fit_emax_hill_n2_data():
    """Hill n=2 data should fit reasonably well."""
    concs, effs = _generate_emax_data(emax=100.0, ec50=10.0, n=2.0)
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    assert result.r_squared > 0.90
    assert result.hill_n == 2.0


def test_fit_emax_higher_conc_higher_effect():
    concs, effs = _generate_emax_data()
    result = fit_emax_model(concs, effs, emax_guess=100.0, ec50_guess=10.0)
    # Verify monotonicity: predicted effect increases with concentration
    preds = [
        result.emax * c**result.hill_n / (result.ec50**result.hill_n + c**result.hill_n)
        for c in sorted(concs)
    ]
    for i in range(1, len(preds)):
        assert preds[i] >= preds[i - 1]


# ── Emax validation errors ────────────────────────────────────────────────────


def test_fit_emax_empty_concentrations_raises():
    with pytest.raises(ValueError, match="concentrations"):
        fit_emax_model([], [1.0, 2.0], emax_guess=100.0, ec50_guess=10.0)


def test_fit_emax_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="length"):
        fit_emax_model([1.0, 2.0], [10.0], emax_guess=100.0, ec50_guess=10.0)


def test_fit_emax_negative_emax_guess_raises():
    with pytest.raises(ValueError, match="emax_guess"):
        fit_emax_model([1.0, 2.0], [10.0, 20.0], emax_guess=-100.0, ec50_guess=10.0)


def test_fit_emax_zero_ec50_guess_raises():
    with pytest.raises(ValueError, match="ec50_guess"):
        fit_emax_model([1.0, 2.0], [10.0, 20.0], emax_guess=100.0, ec50_guess=0.0)


def test_fit_emax_negative_concentration_raises():
    with pytest.raises(ValueError, match="concentrations"):
        fit_emax_model([-1.0, 2.0], [10.0, 20.0], emax_guess=100.0, ec50_guess=10.0)


# ── Indirect response model fitting ──────────────────────────────────────────


def _generate_ir_data(
    effect_type: str = "inhibit_kin",
    kin: float = 10.0,
    kout: float = 0.5,
    imax: float = 0.8,
    ic50: float = 5.0,
    n_pts: int = 12,
) -> tuple[list[float], list[float]]:
    """Generate synthetic indirect response data."""
    t_end = 24.0
    dt = t_end / (n_pts - 1)
    times = [i * dt for i in range(n_pts)]

    # Simulate response
    r0 = kin / kout
    r = r0
    obs_dt = dt / 10
    sim_t = 0.0
    captured = []

    all_sim_r = []
    all_sim_t = []

    n_sim = int(t_end / obs_dt)
    for _step in range(n_sim + 1):
        all_sim_t.append(sim_t)
        all_sim_r.append(r)

        # Advance
        c = 10.0 * math.exp(-0.3 * sim_t)
        inh = imax * c / (ic50 + c)
        if effect_type == "inhibit_kin":
            dr = kin * (1.0 - inh) - kout * r
        else:
            dr = kin - kout * (1.0 - inh) * r
        r = max(0.0, r + dr * obs_dt)
        sim_t += obs_dt

    # Sample at observation times
    for t_obs in times:
        # Find closest simulated time
        idx = min(range(len(all_sim_t)), key=lambda i: abs(all_sim_t[i] - t_obs))
        captured.append(all_sim_r[idx])

    return times, captured


def test_fit_ir_returns_result_type():
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    assert isinstance(result, IndirectResponseResult)


def test_fit_ir_baseline_r0_equals_kin_over_kout():
    """R0 should equal kin / kout."""
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    expected_r0 = 10.0 / result.kout
    assert result.r0 == pytest.approx(expected_r0, rel=1e-4)


def test_fit_ir_r_predicted_length_matches_times():
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    assert len(result.r_predicted) == len(times)


def test_fit_ir_r_squared_at_most_1():
    """R² is at most 1.0 (can be negative for poor fits)."""
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    assert result.r_squared <= 1.0


def test_fit_ir_inhibit_kout_type():
    times, responses = _generate_ir_data(effect_type="inhibit_kout")
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kout",
    )
    assert result.effect_type == "inhibit_kout"


def test_fit_ir_effect_type_stored_correctly():
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    assert result.effect_type == "inhibit_kin"


def test_fit_ir_kout_positive():
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    assert result.kout > 0.0


def test_fit_ir_imax_in_valid_range():
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    assert 0.0 < result.imax <= 1.0


def test_fit_ir_notes_not_empty():
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    assert len(result.notes) > 0


def test_fit_ir_times_stored_correctly():
    times, responses = _generate_ir_data()
    result = fit_indirect_response_model(
        times,
        responses,
        kin=10.0,
        kout_guess=0.5,
        imax_guess=0.8,
        ic50_guess=5.0,
        effect_type="inhibit_kin",
    )
    assert result.times == times


# ── Indirect response validation errors ──────────────────────────────────────


def test_fit_ir_empty_times_raises():
    with pytest.raises(ValueError, match="times"):
        fit_indirect_response_model(
            [],
            [1.0],
            kin=10.0,
            kout_guess=0.5,
            imax_guess=0.8,
            ic50_guess=5.0,
            effect_type="inhibit_kin",
        )


def test_fit_ir_invalid_effect_type_raises():
    times, responses = _generate_ir_data()
    with pytest.raises(ValueError, match="effect_type"):
        fit_indirect_response_model(
            times,
            responses,
            kin=10.0,
            kout_guess=0.5,
            imax_guess=0.8,
            ic50_guess=5.0,
            effect_type="stimulate_kin",
        )


def test_fit_ir_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="length"):
        fit_indirect_response_model(
            [1.0, 2.0],
            [1.0],
            kin=10.0,
            kout_guess=0.5,
            imax_guess=0.8,
            ic50_guess=5.0,
            effect_type="inhibit_kin",
        )


def test_fit_ir_negative_kout_raises():
    times, responses = _generate_ir_data()
    with pytest.raises(ValueError, match="kout_guess"):
        fit_indirect_response_model(
            times,
            responses,
            kin=10.0,
            kout_guess=-0.5,
            imax_guess=0.8,
            ic50_guess=5.0,
            effect_type="inhibit_kin",
        )


def test_fit_ir_imax_above_one_raises():
    times, responses = _generate_ir_data()
    with pytest.raises(ValueError, match="imax_guess"):
        fit_indirect_response_model(
            times,
            responses,
            kin=10.0,
            kout_guess=0.5,
            imax_guess=1.5,
            ic50_guess=5.0,
            effect_type="inhibit_kin",
        )


# ── pdp_index ─────────────────────────────────────────────────────────────────


def test_pdp_index_zero_if_effect_auc_zero():
    assert pdp_index(auc=10.0, effect_auc=0.0) == 0.0


def test_pdp_index_correct_calculation():
    result = pdp_index(auc=50.0, effect_auc=25.0)
    assert result == pytest.approx(0.5, rel=1e-6)


def test_pdp_index_raises_on_zero_auc():
    with pytest.raises(ValueError, match="auc"):
        pdp_index(auc=0.0, effect_auc=10.0)


def test_pdp_index_raises_on_negative_auc():
    with pytest.raises(ValueError, match="auc"):
        pdp_index(auc=-5.0, effect_auc=10.0)


def test_pdp_index_raises_on_negative_effect_auc():
    with pytest.raises(ValueError, match="effect_auc"):
        pdp_index(auc=10.0, effect_auc=-1.0)


def test_pdp_index_greater_effect_greater_index():
    idx_lo = pdp_index(auc=100.0, effect_auc=10.0)
    idx_hi = pdp_index(auc=100.0, effect_auc=50.0)
    assert idx_hi > idx_lo


def test_pdp_index_units_correct():
    """pdp_index = effect_auc / auc."""
    auc = 200.0
    effect_auc = 80.0
    assert pdp_index(auc, effect_auc) == pytest.approx(effect_auc / auc)
