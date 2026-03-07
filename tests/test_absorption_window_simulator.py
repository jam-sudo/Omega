"""Tests for Phase 1057 — Drug Absorption Window Simulator."""

import pytest

from omega_pbpk.core.absorption_window_simulator import (
    AbsorptionWindowResult,
    simulate_absorption_window,
)


def test_return_type():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert isinstance(result, AbsorptionWindowResult)


def test_times_h_nonempty():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert len(result.times_h) > 0


def test_fraction_absorbed_starts_at_zero():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert result.fraction_absorbed[0] == pytest.approx(0.0, abs=1e-9)


def test_fraction_absorbed_monotonically_nondecreasing():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    for i in range(1, len(result.fraction_absorbed)):
        assert result.fraction_absorbed[i] >= result.fraction_absorbed[i - 1] - 1e-9


def test_total_fa_in_valid_range():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert 0.0 <= result.total_fa <= 1.0


def test_fa_si_plus_fa_colon_approx_total_fa():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert abs(result.fa_small_intestine + result.fa_colon - result.total_fa) < 0.02


def test_absorption_rate_positive_during_transit():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    # At t=0, drug is in SI and absorption rate should be positive
    assert result.absorption_rate_mg_h[0] > 0


def test_c_portal_positive_during_absorption():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    # At t=0, portal concentration driven by absorption rate should be positive
    assert result.c_portal_mg_L[0] > 0


def test_high_peff_higher_total_fa_than_low_peff():
    """High Peff drug absorbs more than low Peff drug (peff injected directly to avoid cap)."""
    result_high = simulate_absorption_window(
        "HighPeff", dose_mg=100.0, logp=3.0, mw_Da=200.0, peff_cm_s=0.001, t_end_h=30.0
    )
    result_low = simulate_absorption_window(
        "LowPeff", dose_mg=100.0, logp=-2.0, mw_Da=500.0, peff_cm_s=1e-6, t_end_h=30.0
    )
    assert result_high.total_fa > result_low.total_fa


def test_fed_state_slower_transit():
    result_fasted = simulate_absorption_window(
        "TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0, fed_state=False
    )
    result_fed = simulate_absorption_window(
        "TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0, fed_state=True
    )
    # Fed state extends the absorption window
    assert result_fed.absorption_window_h > result_fasted.absorption_window_h


def test_tmax_absorption_h_nonnegative():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert result.tmax_absorption_h >= 0.0


def test_notes_nonempty():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert len(result.notes) > 0


def test_drug_name_preserved():
    result = simulate_absorption_window("Metformin", dose_mg=500.0, logp=-1.4, mw_Da=129.0)
    assert result.drug_name == "Metformin"


def test_dose_mg_preserved():
    result = simulate_absorption_window("TestDrug", dose_mg=250.0, logp=2.0, mw_Da=300.0)
    assert result.dose_mg == 250.0


def test_absorption_window_h_equals_si_transit():
    result = simulate_absorption_window(
        "TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0, si_transit_h=4.0
    )
    assert result.absorption_window_h == pytest.approx(4.0, rel=1e-3)


def test_custom_peff_used():
    result = simulate_absorption_window(
        "TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0, peff_cm_s=0.01
    )
    assert result.total_fa > 0


def test_acid_ionization_reduces_peff_at_low_pka():
    """Acid drug (pKa=5) is mostly ionized at intestinal pH 6.8 → lower absorption."""
    result_neutral = simulate_absorption_window(
        "Neutral",
        dose_mg=100.0,
        logp=2.0,
        mw_Da=300.0,
        ionization_type="neutral",
        peff_cm_s=5e-5,
    )
    result_acid = simulate_absorption_window(
        "Acid",
        dose_mg=100.0,
        logp=2.0,
        mw_Da=300.0,
        ionization_type="acid",
        pka=5.0,
        peff_cm_s=5e-5,
    )
    # Acid with pka 5 → mostly ionized at pH 6.8 → lower effective Peff → lower absorption
    assert result_neutral.total_fa >= result_acid.total_fa


def test_fraction_absorbed_length_matches_times():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert len(result.fraction_absorbed) == len(result.times_h)


def test_position_cm_length_matches_times():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    assert len(result.position_cm) == len(result.times_h)


def test_position_monotonically_nondecreasing():
    result = simulate_absorption_window("TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0)
    for i in range(1, len(result.position_cm)):
        assert result.position_cm[i] >= result.position_cm[i - 1] - 1e-9


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_absorption_window("TestDrug", dose_mg=0.0, logp=2.0, mw_Da=300.0)


def test_validation_si_transit_zero_raises():
    with pytest.raises(ValueError, match="si_transit_h"):
        simulate_absorption_window(
            "TestDrug", dose_mg=100.0, logp=2.0, mw_Da=300.0, si_transit_h=0.0
        )


def test_validation_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        simulate_absorption_window(
            "TestDrug",
            dose_mg=100.0,
            logp=2.0,
            mw_Da=300.0,
            ionization_type="unknown",
        )
