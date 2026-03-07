"""Tests for Phase 1017: extended-release ocular drug delivery PK model."""

import math
import pytest

from omega_pbpk.core.ocular_drug_delivery_pk import (
    OcularDrugDeliveryResult,
    simulate_ocular_drug_delivery,
)


# ---------------------------------------------------------------------------
# Basic return-type and field checks
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    res = simulate_ocular_drug_delivery("TestDrug", 1.0)
    assert isinstance(res, OcularDrugDeliveryResult)


def test_cmax_aqueous_positive():
    res = simulate_ocular_drug_delivery("TestDrug", 1.0)
    assert res.cmax_aqueous_mg_L > 0.0


def test_implant_initial_drug_load():
    """a_implant_mg[0] should equal dose_mg."""
    res = simulate_ocular_drug_delivery("Drug", 5.0)
    assert math.isclose(res.a_implant_mg[0], 5.0, rel_tol=1e-9)


def test_a_implant_monotonically_decreases():
    res = simulate_ocular_drug_delivery("Drug", 2.0)
    for i in range(1, len(res.a_implant_mg)):
        assert res.a_implant_mg[i] <= res.a_implant_mg[i - 1], (
            f"a_implant increased at step {i}"
        )


def test_steady_state_aqueous_positive():
    res = simulate_ocular_drug_delivery("Drug", 1.0)
    assert res.steady_state_aqueous_mg_L > 0.0


def test_duration_effective_positive():
    res = simulate_ocular_drug_delivery("Drug", 1.0)
    assert res.duration_effective_h > 0.0


def test_drug_load_remaining_in_range():
    res = simulate_ocular_drug_delivery("Drug", 1.0)
    assert 0.0 <= res.drug_load_remaining_pct_at_30d <= 100.0


def test_c_aqueous_starts_at_zero():
    """Initial aqueous concentration should be 0."""
    res = simulate_ocular_drug_delivery("Drug", 1.0)
    assert math.isclose(res.c_aqueous_mg_L[0], 0.0, abs_tol=1e-12)


def test_device_type_preserved():
    for device in ("implant", "punctal_plug", "contact_lens", "insert"):
        res = simulate_ocular_drug_delivery("Drug", 1.0, device_type=device)
        assert res.device_type == device


# ---------------------------------------------------------------------------
# Device-type comparison tests
# ---------------------------------------------------------------------------


def test_implant_longer_duration_than_contact_lens():
    """Implant releases slower → drug persists longer than contact lens."""
    implant = simulate_ocular_drug_delivery("Drug", 1.0, device_type="implant")
    cl = simulate_ocular_drug_delivery("Drug", 1.0, device_type="contact_lens")
    assert implant.duration_effective_h >= cl.duration_effective_h


def test_contact_lens_drug_load_depleted_faster():
    """Contact lens (k=0.05/h) should deplete > 50% within 30 days."""
    res = simulate_ocular_drug_delivery(
        "Drug", 1.0, device_type="contact_lens", t_end_h=720.0
    )
    assert res.drug_load_remaining_pct_at_30d < 50.0


def test_implant_drug_load_higher_than_contact_lens_at_30d():
    """Implant releases slower → more drug remaining at 30 days."""
    implant = simulate_ocular_drug_delivery("Drug", 1.0, device_type="implant")
    cl = simulate_ocular_drug_delivery("Drug", 1.0, device_type="contact_lens")
    assert implant.drug_load_remaining_pct_at_30d > cl.drug_load_remaining_pct_at_30d


# ---------------------------------------------------------------------------
# logP effect
# ---------------------------------------------------------------------------


def test_high_logp_increases_cmax_aqueous():
    """Higher logP → larger k_cornea_factor → higher Cmax in aqueous."""
    low = simulate_ocular_drug_delivery("Drug", 1.0, logp=0.0)
    high = simulate_ocular_drug_delivery("Drug", 1.0, logp=4.0)
    assert high.cmax_aqueous_mg_L >= low.cmax_aqueous_mg_L


# ---------------------------------------------------------------------------
# Systemic compartment
# ---------------------------------------------------------------------------


def test_cmax_systemic_positive():
    res = simulate_ocular_drug_delivery("Drug", 1.0)
    assert res.cmax_systemic_mg_L > 0.0


def test_systemic_lower_than_aqueous():
    """Systemic Cmax should generally be much lower than aqueous Cmax."""
    res = simulate_ocular_drug_delivery("Drug", 1.0)
    assert res.cmax_systemic_mg_L < res.cmax_aqueous_mg_L


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ocular_drug_delivery("Drug", 0.0)


def test_negative_dose_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        simulate_ocular_drug_delivery("Drug", -1.0)


def test_invalid_device_type_raises():
    with pytest.raises(ValueError, match="device_type"):
        simulate_ocular_drug_delivery("Drug", 1.0, device_type="eye_drop")
