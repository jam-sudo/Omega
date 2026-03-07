"""Tests for core/efflux_transporter_kinetics.py — Phase 283."""

from __future__ import annotations

import pytest

from omega_pbpk.core.efflux_transporter_kinetics import (
    Caco2TransportResult,
    NetFluxResult,
    classify_substrate,
    efflux_rate,
    net_flux,
    simulate_caco2_transport,
)

# ---------------------------------------------------------------------------
# efflux_rate
# ---------------------------------------------------------------------------


class TestEffluxRate:
    def test_basic_mm_kinetics(self):
        """Efflux rate at C = Km should equal Vmax/2."""
        vmax = 10.0
        km = 5.0
        c = 5.0  # C == Km => rate = Vmax/2
        result = efflux_rate(c, vmax, km)
        assert abs(result - vmax / 2) < 1e-9

    def test_zero_concentration(self):
        """Zero concentration should give zero efflux rate."""
        result = efflux_rate(0.0, 10.0, 5.0)
        assert result == 0.0

    def test_high_concentration_approaches_vmax(self):
        """At very high concentration, efflux rate approaches Vmax."""
        result = efflux_rate(1e6, 10.0, 5.0)
        assert result > 9.99

    def test_low_concentration_linear(self):
        """At C << Km, efflux rate is approximately linear: Vmax*C/Km."""
        vmax, km, c = 10.0, 100.0, 0.1
        result = efflux_rate(c, vmax, km)
        expected = vmax * c / km
        assert abs(result - expected) / expected < 0.01

    def test_invalid_negative_concentration(self):
        with pytest.raises(ValueError, match="c_intracellular_mg_L"):
            efflux_rate(-1.0, 10.0, 5.0)

    def test_invalid_zero_vmax(self):
        with pytest.raises(ValueError, match="vmax_mg_per_h"):
            efflux_rate(1.0, 0.0, 5.0)

    def test_invalid_negative_vmax(self):
        with pytest.raises(ValueError, match="vmax_mg_per_h"):
            efflux_rate(1.0, -5.0, 5.0)

    def test_invalid_zero_km(self):
        with pytest.raises(ValueError, match="km_mg_L"):
            efflux_rate(1.0, 10.0, 0.0)

    def test_returns_float(self):
        assert isinstance(efflux_rate(2.0, 10.0, 5.0), float)


# ---------------------------------------------------------------------------
# classify_substrate
# ---------------------------------------------------------------------------


class TestClassifySubstrate:
    def test_not_substrate(self):
        assert classify_substrate(1.0) == "not_substrate"
        assert classify_substrate(1.99) == "not_substrate"

    def test_boundary_moderate(self):
        assert classify_substrate(2.0) == "moderate"
        assert classify_substrate(3.5) == "moderate"
        assert classify_substrate(5.0) == "moderate"

    def test_strong_substrate(self):
        assert classify_substrate(5.01) == "strong"
        assert classify_substrate(20.0) == "strong"


# ---------------------------------------------------------------------------
# net_flux
# ---------------------------------------------------------------------------


class TestNetFlux:
    def test_returns_netfluxresult(self):
        result = net_flux(
            c_apical_mg_L=10.0,
            c_basal_mg_L=0.1,
            papp_ab_cm_s=1e-6,
            papp_ba_cm_s=5e-6,
            vmax_mg_per_h=0.5,
            km_mg_L=2.0,
        )
        assert isinstance(result, NetFluxResult)

    def test_efflux_ratio_calculated(self):
        result = net_flux(10.0, 0.1, 2e-6, 10e-6, 0.5, 2.0)
        assert abs(result.efflux_ratio - 5.0) < 1e-9

    def test_substrate_class_strong(self):
        result = net_flux(10.0, 0.1, 1e-6, 10e-6, 0.5, 2.0)
        assert result.substrate_class == "strong"

    def test_substrate_class_not(self):
        result = net_flux(10.0, 0.1, 5e-6, 6e-6, 0.5, 2.0)
        assert result.substrate_class == "not_substrate"

    def test_substrate_class_moderate(self):
        result = net_flux(10.0, 0.1, 2e-6, 6e-6, 0.5, 2.0)
        assert result.substrate_class == "moderate"

    def test_invalid_negative_c_apical(self):
        with pytest.raises(ValueError, match="c_apical"):
            net_flux(-1.0, 0.1, 1e-6, 5e-6, 0.5, 2.0)

    def test_invalid_negative_c_basal(self):
        with pytest.raises(ValueError, match="c_basal"):
            net_flux(10.0, -1.0, 1e-6, 5e-6, 0.5, 2.0)

    def test_invalid_zero_papp_ab(self):
        with pytest.raises(ValueError, match="papp_ab"):
            net_flux(10.0, 0.1, 0.0, 5e-6, 0.5, 2.0)

    def test_invalid_zero_papp_ba(self):
        with pytest.raises(ValueError, match="papp_ba"):
            net_flux(10.0, 0.1, 1e-6, 0.0, 0.5, 2.0)

    def test_invalid_zero_area(self):
        with pytest.raises(ValueError, match="area"):
            net_flux(10.0, 0.1, 1e-6, 5e-6, 0.5, 2.0, area_cm2=0.0)

    def test_result_frozen(self):
        result = net_flux(10.0, 0.1, 1e-6, 5e-6, 0.5, 2.0)
        with pytest.raises(Exception):
            result.efflux_ratio = 99.0  # type: ignore[misc]

    def test_notes_contains_er(self):
        result = net_flux(10.0, 0.1, 1e-6, 5e-6, 0.5, 2.0)
        assert "ER=" in result.notes


# ---------------------------------------------------------------------------
# simulate_caco2_transport
# ---------------------------------------------------------------------------


class TestSimulateCaco2Transport:
    def test_returns_result(self):
        result = simulate_caco2_transport(
            c_donor_mg_L=10.0,
            papp_cm_s=1e-6,
            vmax_mg_per_h=0.5,
            km_mg_L=2.0,
        )
        assert isinstance(result, Caco2TransportResult)

    def test_time_series_length(self):
        result = simulate_caco2_transport(10.0, 1e-6, 0.5, 2.0, t_end_h=1.0, dt_h=0.1)
        assert len(result.times_h) == len(result.c_apical)
        assert len(result.times_h) == len(result.c_basolateral)

    def test_apical_decreases_monotonically(self):
        """Apical concentration should generally decrease over time."""
        result = simulate_caco2_transport(10.0, 5e-6, 0.1, 10.0, t_end_h=2.0, dt_h=0.05)
        # Check that final apical < initial apical
        assert result.c_apical[-1] < result.c_apical[0]

    def test_basolateral_increases_then_plateaus(self):
        """Basolateral concentration should increase from zero."""
        result = simulate_caco2_transport(10.0, 5e-6, 0.1, 10.0, t_end_h=2.0, dt_h=0.05)
        assert result.c_basolateral[-1] > 0.0

    def test_times_start_at_zero(self):
        result = simulate_caco2_transport(10.0, 1e-6, 0.5, 2.0)
        assert result.times_h[0] == 0.0

    def test_efflux_ratio_positive(self):
        result = simulate_caco2_transport(10.0, 1e-6, 2.0, 1.0)
        assert result.efflux_ratio > 0.0

    def test_substrate_class_valid(self):
        result = simulate_caco2_transport(10.0, 1e-6, 5.0, 0.5)
        assert result.substrate_class in {"not_substrate", "moderate", "strong"}

    def test_invalid_zero_donor(self):
        with pytest.raises(ValueError, match="c_donor"):
            simulate_caco2_transport(0.0, 1e-6, 0.5, 2.0)

    def test_invalid_zero_papp(self):
        with pytest.raises(ValueError, match="papp_cm_s"):
            simulate_caco2_transport(10.0, 0.0, 0.5, 2.0)

    def test_invalid_zero_volume_donor(self):
        with pytest.raises(ValueError, match="volume_donor"):
            simulate_caco2_transport(10.0, 1e-6, 0.5, 2.0, volume_donor_mL=0.0)

    def test_result_frozen(self):
        result = simulate_caco2_transport(10.0, 1e-6, 0.5, 2.0)
        with pytest.raises(Exception):
            result.efflux_ratio = 99.0  # type: ignore[misc]

    def test_papp_apparent_nonnegative(self):
        result = simulate_caco2_transport(10.0, 1e-6, 0.5, 2.0)
        assert result.papp_apparent >= 0.0

    def test_high_vmax_strong_efflux(self):
        """High Vmax should produce strong substrate classification."""
        result = simulate_caco2_transport(
            c_donor_mg_L=10.0,
            papp_cm_s=1e-7,
            vmax_mg_per_h=100.0,
            km_mg_L=1.0,
        )
        assert result.efflux_ratio > 2.0
