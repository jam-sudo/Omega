"""Tests for carrier-mediated transport model (Phase 833)."""

import pytest

from omega_pbpk.core.carrier_mediated_transport import (
    CarrierTransportResult,
    screen_transporters,
    simulate_carrier_transport,
)

# ── Return type ───────────────────────────────────────────────────────────────


def test_return_type():
    result = simulate_carrier_transport("TestDrug", "OATP1B1", c_donor_initial_mg_L=5.0)
    assert isinstance(result, CarrierTransportResult)


def test_result_fields_present():
    result = simulate_carrier_transport("TestDrug", "OATP1B1", c_donor_initial_mg_L=5.0)
    assert result.drug_name == "TestDrug"
    assert result.transporter_name == "OATP1B1"
    assert result.direction == "influx"


# ── Time course correctness (influx) ─────────────────────────────────────────


def test_c_donor_decreases_influx():
    """Donor concentration should decrease over time in influx mode."""
    result = simulate_carrier_transport(
        "DrugA", "OATP1B1", c_donor_initial_mg_L=10.0, direction="influx"
    )
    assert result.c_donor_mg_L[0] > result.c_donor_mg_L[-1]


def test_c_acceptor_increases_influx():
    """Acceptor concentration should increase over time in influx mode."""
    result = simulate_carrier_transport(
        "DrugA", "OATP1B1", c_donor_initial_mg_L=10.0, direction="influx"
    )
    assert result.c_acceptor_mg_L[-1] > result.c_acceptor_mg_L[0]


def test_c_acceptor_starts_at_zero():
    result = simulate_carrier_transport("DrugA", "P-gp", c_donor_initial_mg_L=5.0)
    assert result.c_acceptor_mg_L[0] == pytest.approx(0.0)


def test_times_are_monotonically_increasing():
    result = simulate_carrier_transport("DrugA", "OATP1B1", c_donor_initial_mg_L=5.0)
    for i in range(1, len(result.times_h)):
        assert result.times_h[i] > result.times_h[i - 1]


def test_times_start_at_zero():
    result = simulate_carrier_transport("DrugA", "OATP1B1", c_donor_initial_mg_L=5.0)
    assert result.times_h[0] == pytest.approx(0.0)


# ── Transport efficiency ──────────────────────────────────────────────────────


def test_transport_efficiency_in_range():
    result = simulate_carrier_transport("DrugA", "OATP1B1", c_donor_initial_mg_L=5.0)
    assert 0.0 < result.transport_efficiency < 1.0


def test_higher_vmax_gives_higher_efficiency():
    low_vmax = simulate_carrier_transport(
        "DrugA", "T1", c_donor_initial_mg_L=5.0, vmax_mg_per_h=0.01
    )
    high_vmax = simulate_carrier_transport(
        "DrugA", "T1", c_donor_initial_mg_L=5.0, vmax_mg_per_h=1.0
    )
    assert high_vmax.transport_efficiency > low_vmax.transport_efficiency


# ── Saturation fraction ───────────────────────────────────────────────────────


def test_saturation_fraction_formula():
    c0 = 3.0
    km = 2.0
    result = simulate_carrier_transport("DrugA", "T1", c_donor_initial_mg_L=c0, km_mg_per_L=km)
    expected = c0 / (km + c0)
    assert result.saturation_fraction == pytest.approx(expected, rel=1e-6)


def test_saturation_fraction_high_conc():
    """At very high concentration relative to Km, saturation_fraction approaches 1."""
    result = simulate_carrier_transport("DrugA", "T1", c_donor_initial_mg_L=1000.0, km_mg_per_L=1.0)
    assert result.saturation_fraction > 0.99


def test_saturation_fraction_low_conc():
    """At very low concentration relative to Km, saturation_fraction approaches 0."""
    result = simulate_carrier_transport("DrugA", "T1", c_donor_initial_mg_L=0.001, km_mg_per_L=10.0)
    assert result.saturation_fraction < 0.01


# ── Efflux direction ──────────────────────────────────────────────────────────


def test_efflux_direction_stored():
    result = simulate_carrier_transport(
        "DrugA", "P-gp", c_donor_initial_mg_L=5.0, direction="efflux"
    )
    assert result.direction == "efflux"


def test_efflux_donor_stays_higher_than_influx():
    """With efflux transporter active, donor concentration stays higher than pure influx."""
    influx = simulate_carrier_transport(
        "DrugA", "T1", c_donor_initial_mg_L=5.0, direction="influx", vmax_mg_per_h=0.5
    )
    efflux = simulate_carrier_transport(
        "DrugA", "T1", c_donor_initial_mg_L=5.0, direction="efflux", vmax_mg_per_h=0.5
    )
    # Efflux pump resists net flux to acceptor → donor should be depleted more slowly
    assert efflux.c_donor_mg_L[-1] >= influx.c_donor_mg_L[-1]


# ── AUC values ────────────────────────────────────────────────────────────────


def test_influx_auc_positive():
    result = simulate_carrier_transport("DrugA", "OATP1B1", c_donor_initial_mg_L=5.0)
    assert result.influx_auc > 0.0


def test_efflux_auc_positive():
    result = simulate_carrier_transport("DrugA", "OATP1B1", c_donor_initial_mg_L=5.0)
    assert result.efflux_auc > 0.0


# ── Net transport classification ─────────────────────────────────────────────


def test_net_transport_is_valid_string():
    result = simulate_carrier_transport("DrugA", "OATP1B1", c_donor_initial_mg_L=5.0)
    assert result.net_transport in ("influx_dominant", "efflux_dominant", "balanced")


# ── screen_transporters ───────────────────────────────────────────────────────


def test_screen_transporters_sorted_by_efficiency():
    transporter_data = [
        {"name": "Low_T", "vmax": 0.01, "km": 1.0, "direction": "influx"},
        {"name": "High_T", "vmax": 2.0, "km": 0.5, "direction": "influx"},
        {"name": "Mid_T", "vmax": 0.5, "km": 1.0, "direction": "influx"},
    ]
    results = screen_transporters("DrugA", 5.0, transporter_data)
    efficiencies = [r.transport_efficiency for r in results]
    assert efficiencies == sorted(efficiencies, reverse=True)


def test_screen_transporters_returns_correct_count():
    transporter_data = [
        {"name": "T1", "vmax": 0.1, "km": 1.0, "direction": "influx"},
        {"name": "T2", "vmax": 0.5, "km": 1.0, "direction": "efflux"},
    ]
    results = screen_transporters("DrugA", 5.0, transporter_data)
    assert len(results) == 2


def test_screen_transporters_result_type():
    transporter_data = [
        {"name": "T1", "vmax": 0.1, "km": 1.0, "direction": "influx"},
    ]
    results = screen_transporters("DrugA", 5.0, transporter_data)
    assert isinstance(results[0], CarrierTransportResult)


# ── Validation errors ─────────────────────────────────────────────────────────


def test_negative_c_donor_raises():
    with pytest.raises(ValueError, match="c_donor_initial"):
        simulate_carrier_transport("DrugA", "T1", c_donor_initial_mg_L=-1.0)


def test_zero_vmax_raises():
    with pytest.raises(ValueError, match="vmax"):
        simulate_carrier_transport("DrugA", "T1", c_donor_initial_mg_L=5.0, vmax_mg_per_h=0.0)


def test_negative_vmax_raises():
    with pytest.raises(ValueError, match="vmax"):
        simulate_carrier_transport("DrugA", "T1", c_donor_initial_mg_L=5.0, vmax_mg_per_h=-1.0)


def test_zero_km_raises():
    with pytest.raises(ValueError, match="km"):
        simulate_carrier_transport("DrugA", "T1", c_donor_initial_mg_L=5.0, km_mg_per_L=0.0)


def test_zero_membrane_area_raises():
    with pytest.raises(ValueError, match="membrane_area"):
        simulate_carrier_transport("DrugA", "T1", c_donor_initial_mg_L=5.0, membrane_area_cm2=0.0)
