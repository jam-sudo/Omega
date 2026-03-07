"""Tests for biphasic (hormetic) PD model — Phase 255."""

from __future__ import annotations

import pytest

from omega_pbpk.clinical.biphasic_pd import (
    BiphasicPDResult,
    biphasic_effect,
    evaluate_biphasic_pd,
)

# ---------------------------------------------------------------------------
# biphasic_effect unit tests
# ---------------------------------------------------------------------------


def test_baseline_at_zero_concentration():
    """At concentration=0, effect must equal 1.0 (baseline)."""
    assert biphasic_effect(0.0) == 1.0


def test_low_conc_gives_stimulation():
    """At very low concentration (ec50_stim/10), stimulation dominates -> effect > 1."""
    ec50_stim = 0.5
    low_conc = ec50_stim / 10.0
    effect = biphasic_effect(low_conc, e_stim=1.5, ec50_stim=ec50_stim)
    assert effect > 1.0


def test_high_conc_inhibition_dominates():
    """At very high concentration (ec50_inhib * 10), net effect below stimulation plateau."""
    ec50_inhib = 5.0
    high_conc = ec50_inhib * 10.0
    effect = biphasic_effect(high_conc, e_stim=1.5, e_inhib=0.2, ec50_inhib=ec50_inhib)
    # At very high conc, both stim and inhib near max; net < 1 + e_stim
    assert effect < 1.0 + 1.5


def test_very_high_conc_approaches_asymptote():
    """At very high concentration, effect approaches 1 + e_stim - e_inhib."""
    e_stim = 1.5
    e_inhib = 0.5
    ec50_stim = 0.5
    ec50_inhib = 5.0
    high_effect = biphasic_effect(1e6, e_stim=e_stim, ec50_stim=ec50_stim,
                                  e_inhib=e_inhib, ec50_inhib=ec50_inhib)
    asymptote = 1.0 + e_stim - e_inhib  # both Hill fractions approach 1
    assert high_effect == pytest.approx(asymptote, rel=1e-3)


def test_invalid_ec50_stim_zero():
    with pytest.raises(ValueError, match="ec50_stim"):
        biphasic_effect(1.0, ec50_stim=0.0)


def test_invalid_ec50_stim_negative():
    with pytest.raises(ValueError, match="ec50_stim"):
        biphasic_effect(1.0, ec50_stim=-1.0)


def test_invalid_ec50_inhib_zero():
    with pytest.raises(ValueError, match="ec50_inhib"):
        biphasic_effect(1.0, ec50_inhib=0.0)


def test_invalid_ec50_inhib_negative():
    with pytest.raises(ValueError, match="ec50_inhib"):
        biphasic_effect(1.0, ec50_inhib=-2.0)


def test_invalid_negative_concentration():
    with pytest.raises(ValueError, match="concentration"):
        biphasic_effect(-0.1)


def test_no_stimulation_no_inhibition_gives_baseline():
    """With e_stim=0, e_inhib=0, effect stays at 1.0 everywhere."""
    for c in [0.0, 0.1, 1.0, 10.0]:
        assert biphasic_effect(c, e_stim=0.0, e_inhib=0.0) == pytest.approx(1.0)


def test_positive_hill_coefficients():
    """Hill coefficients > 1 create steeper dose-response. Check function runs."""
    effect = biphasic_effect(1.0, hill_stim=2.0, hill_inhib=3.0)
    assert isinstance(effect, float)


# ---------------------------------------------------------------------------
# evaluate_biphasic_pd unit tests
# ---------------------------------------------------------------------------


def test_concentration_count():
    result = evaluate_biphasic_pd("TestDrug", c_max=10.0, n_points=30)
    assert len(result.concentrations) == 30


def test_effects_count():
    result = evaluate_biphasic_pd("TestDrug", c_max=10.0, n_points=30)
    assert len(result.effects) == 30


def test_peak_stimulation_greater_than_one():
    """With e_stim > 0, peak_stimulation must be > 1.0."""
    result = evaluate_biphasic_pd("Drug", c_max=20.0, e_stim=1.5)
    assert result.peak_stimulation > 1.0


def test_peak_stimulation_conc_in_range():
    """Peak stimulation concentration must be within evaluated range."""
    c_max = 15.0
    result = evaluate_biphasic_pd("Drug", c_max=c_max)
    assert 0.0 < result.peak_stimulation_conc <= c_max


def test_tipping_point_positive():
    result = evaluate_biphasic_pd("Drug", c_max=20.0, e_stim=1.5, e_inhib=1.0)
    assert result.tipping_point_conc > 0.0


def test_invalid_c_max_zero():
    with pytest.raises(ValueError, match="c_max"):
        evaluate_biphasic_pd("Drug", c_max=0.0)


def test_invalid_c_max_negative():
    with pytest.raises(ValueError, match="c_max"):
        evaluate_biphasic_pd("Drug", c_max=-5.0)


def test_invalid_n_points_one():
    with pytest.raises(ValueError, match="n_points"):
        evaluate_biphasic_pd("Drug", c_max=10.0, n_points=1)


def test_effect_at_cmax_matches_biphasic_effect():
    """effect_at_cmax should match direct call to biphasic_effect at c_max."""
    c_max = 10.0
    e_stim, ec50_stim, e_inhib, ec50_inhib = 1.5, 0.5, 0.2, 5.0
    result = evaluate_biphasic_pd(
        "Drug", c_max=c_max, n_points=100,
        e_stim=e_stim, ec50_stim=ec50_stim,
        e_inhib=e_inhib, ec50_inhib=ec50_inhib,
    )
    expected = biphasic_effect(c_max, e_stim=e_stim, ec50_stim=ec50_stim,
                               e_inhib=e_inhib, ec50_inhib=ec50_inhib)
    assert result.effect_at_cmax == pytest.approx(expected, rel=1e-4)


def test_no_stimulation_all_effects_le_one():
    """With e_stim=0, all effects should be <= 1.0 (no stimulation)."""
    result = evaluate_biphasic_pd("Drug", c_max=10.0, e_stim=0.0, e_inhib=0.3)
    assert all(e <= 1.0 + 1e-9 for e in result.effects)


def test_drug_name_stored():
    result = evaluate_biphasic_pd("Aspirin", c_max=5.0)
    assert result.drug_name == "Aspirin"


def test_model_type_bell():
    result = evaluate_biphasic_pd("Drug", c_max=10.0, e_stim=1.5, e_inhib=0.2)
    assert result.model_type == "bell"


def test_notes_is_string():
    result = evaluate_biphasic_pd("Drug", c_max=10.0)
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


def test_result_is_frozen_dataclass():
    result = evaluate_biphasic_pd("Drug", c_max=10.0)
    assert isinstance(result, BiphasicPDResult)
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Other"  # type: ignore[misc]
