"""Tests for core/drug_transport_model.py — Phase 505."""

from __future__ import annotations

import pytest

from omega_pbpk.core.drug_transport_model import (
    TransporterKineticsResult,
    calculate_net_transport,
    efflux_transport_rate,
    inhibition_constant,
    simulate_transporter_kinetics,
    transport_rate,
)

# ---------------------------------------------------------------------------
# transport_rate
# ---------------------------------------------------------------------------


class TestTransportRate:
    def test_zero_substrate_returns_zero(self):
        assert transport_rate(0.0, 100.0, 10.0) == 0.0

    def test_low_concentration_linear(self):
        """At C << Km, rate ≈ Vmax * C / Km (first-order)."""
        c = 0.001  # µM, much less than Km=10
        km = 10.0
        vmax = 100.0
        expected = vmax * c / km  # ≈ 0.01
        result = transport_rate(c, vmax, km)
        assert abs(result - expected) / expected < 0.05

    def test_high_concentration_saturates(self):
        """At C >> Km, rate ≈ Vmax."""
        c = 10000.0  # µM, much greater than Km=10
        km = 10.0
        vmax = 100.0
        result = transport_rate(c, vmax, km)
        assert result > 0.99 * vmax

    def test_at_km_half_vmax(self):
        """At C == Km, rate = Vmax / 2."""
        km = 5.0
        vmax = 80.0
        result = transport_rate(km, vmax, km)
        assert abs(result - vmax / 2) < 1e-9

    def test_inhibitor_reduces_rate(self):
        """Adding a competitive inhibitor decreases transport rate."""
        c = 5.0
        km = 5.0
        vmax = 100.0
        rate_no_inhib = transport_rate(c, vmax, km)
        rate_with_inhib = transport_rate(c, vmax, km, inhibitor_conc_uM=10.0, ki_uM=5.0)
        assert rate_with_inhib < rate_no_inhib

    def test_inhibitor_competitive_formula(self):
        """Competitive inhibition: apparent Km = Km * (1 + I/Ki)."""
        c = 4.0
        km = 4.0
        vmax = 50.0
        i = 4.0
        ki = 4.0
        km_app = km * (1 + i / ki)  # = 8
        expected = vmax * c / (km_app + c)
        result = transport_rate(c, vmax, km, inhibitor_conc_uM=i, ki_uM=ki)
        assert abs(result - expected) < 1e-9

    def test_higher_vmax_gives_higher_rate(self):
        assert transport_rate(5.0, 200.0, 10.0) > transport_rate(5.0, 100.0, 10.0)

    def test_negative_substrate_raises(self):
        with pytest.raises(ValueError, match="substrate"):
            transport_rate(-1.0, 100.0, 10.0)

    def test_zero_km_raises(self):
        with pytest.raises(ValueError, match="km_uM"):
            transport_rate(5.0, 100.0, 0.0)

    def test_negative_vmax_raises(self):
        with pytest.raises(ValueError, match="vmax"):
            transport_rate(5.0, -10.0, 5.0)

    def test_inhibitor_without_ki_raises(self):
        with pytest.raises(ValueError, match="ki_uM"):
            transport_rate(5.0, 100.0, 5.0, inhibitor_conc_uM=2.0)

    def test_zero_inhibitor_no_change(self):
        """Zero inhibitor concentration should give same result as no inhibitor."""
        r1 = transport_rate(5.0, 100.0, 10.0)
        r2 = transport_rate(5.0, 100.0, 10.0, inhibitor_conc_uM=0.0, ki_uM=5.0)
        assert abs(r1 - r2) < 1e-9


# ---------------------------------------------------------------------------
# efflux_transport_rate
# ---------------------------------------------------------------------------


class TestEffluxTransportRate:
    def test_ba_direction_uses_basal_conc(self):
        """B→A efflux driven by basolateral concentration."""
        rate = efflux_transport_rate(
            basal_conc_uM=10.0, apical_conc_uM=1.0, vmax=100.0, km=5.0, direction="ba"
        )
        expected = 100.0 * 10.0 / (5.0 + 10.0)
        assert abs(rate - expected) < 1e-9

    def test_ab_direction_uses_apical_conc(self):
        """A→B uptake driven by apical concentration."""
        rate = efflux_transport_rate(
            basal_conc_uM=1.0, apical_conc_uM=8.0, vmax=100.0, km=4.0, direction="ab"
        )
        expected = 100.0 * 8.0 / (4.0 + 8.0)
        assert abs(rate - expected) < 1e-9

    def test_zero_driving_concentration_returns_zero(self):
        rate = efflux_transport_rate(0.0, 5.0, 100.0, 5.0, direction="ba")
        assert rate == 0.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            efflux_transport_rate(5.0, 5.0, 100.0, 5.0, direction="invalid")

    def test_zero_km_raises(self):
        with pytest.raises(ValueError, match="km"):
            efflux_transport_rate(5.0, 5.0, 100.0, 0.0)


# ---------------------------------------------------------------------------
# simulate_transporter_kinetics
# ---------------------------------------------------------------------------


class TestSimulateTransporterKinetics:
    def test_returns_correct_type(self):
        result = simulate_transporter_kinetics(
            "TestDrug",
            initial_conc_uM=10.0,
            vmax_uptake=50.0,
            km_uptake=5.0,
        )
        assert isinstance(result, TransporterKineticsResult)

    def test_higher_vmax_uptake_higher_ss(self):
        """Higher uptake Vmax should lead to higher steady-state intracellular concentration."""
        r_low = simulate_transporter_kinetics("Drug", 10.0, vmax_uptake=20.0, km_uptake=5.0)
        r_high = simulate_transporter_kinetics("Drug", 10.0, vmax_uptake=100.0, km_uptake=5.0)
        assert r_high.steady_state_conc_uM > r_low.steady_state_conc_uM

    def test_with_efflux_lower_ss_than_without(self):
        """Active efflux should decrease steady-state intracellular concentration."""
        r_no_efflux = simulate_transporter_kinetics(
            "Drug", 10.0, vmax_uptake=50.0, km_uptake=5.0, vmax_efflux=0.0
        )
        r_efflux = simulate_transporter_kinetics(
            "Drug", 10.0, vmax_uptake=50.0, km_uptake=5.0, vmax_efflux=40.0, km_efflux=5.0
        )
        assert r_efflux.steady_state_conc_uM < r_no_efflux.steady_state_conc_uM

    def test_net_accumulation_ratio_uptake_dominated(self):
        """When uptake >> efflux, accumulation ratio should exceed 1."""
        result = simulate_transporter_kinetics(
            "Drug",
            10.0,
            vmax_uptake=200.0,
            km_uptake=1.0,
            vmax_efflux=0.0,
            cl_passive_per_h=0.01,
            cell_volume_uL=0.5,
            t_end_h=8.0,
        )
        assert result.net_accumulation_ratio > 1.0

    def test_time_array_length_correct(self):
        result = simulate_transporter_kinetics(
            "Drug", 5.0, vmax_uptake=30.0, km_uptake=3.0, t_end_h=2.0, dt_h=0.1
        )
        # n_steps + 1 time points
        assert len(result.times_h) == len(result.c_intracellular_uM)
        assert result.times_h[0] == 0.0

    def test_drug_name_preserved(self):
        result = simulate_transporter_kinetics(
            "Methotrexate", 10.0, vmax_uptake=50.0, km_uptake=5.0
        )
        assert result.drug_name == "Methotrexate"

    def test_initial_intracellular_zero(self):
        result = simulate_transporter_kinetics("Drug", 10.0, vmax_uptake=50.0, km_uptake=5.0)
        assert result.c_intracellular_uM[0] == 0.0

    def test_negative_initial_conc_raises(self):
        with pytest.raises(ValueError, match="initial_conc_uM"):
            simulate_transporter_kinetics("Drug", -1.0, 50.0, 5.0)

    def test_zero_km_uptake_raises(self):
        with pytest.raises(ValueError, match="km_uptake"):
            simulate_transporter_kinetics("Drug", 5.0, 50.0, 0.0)

    def test_efflux_without_km_raises(self):
        with pytest.raises(ValueError, match="km_efflux"):
            simulate_transporter_kinetics("Drug", 5.0, 50.0, 5.0, vmax_efflux=30.0, km_efflux=None)

    def test_zero_cell_volume_raises(self):
        with pytest.raises(ValueError, match="cell_volume_uL"):
            simulate_transporter_kinetics("Drug", 5.0, 50.0, 5.0, cell_volume_uL=0.0)

    def test_notes_contains_efflux_info(self):
        r_no = simulate_transporter_kinetics("D", 5.0, 50.0, 5.0, vmax_efflux=0.0)
        r_yes = simulate_transporter_kinetics("D", 5.0, 50.0, 5.0, vmax_efflux=20.0, km_efflux=5.0)
        assert "efflux" in r_no.notes
        assert "efflux" in r_yes.notes


# ---------------------------------------------------------------------------
# calculate_net_transport
# ---------------------------------------------------------------------------


class TestCalculateNetTransport:
    def test_basic_calculation(self):
        result = calculate_net_transport(10.0, 3.0, 2.0)
        assert abs(result - 5.0) < 1e-9

    def test_all_zero(self):
        assert calculate_net_transport(0.0, 0.0, 0.0) == 0.0

    def test_efflux_dominated_negative(self):
        """When efflux + passive > uptake, net transport is negative."""
        result = calculate_net_transport(5.0, 8.0, 3.0)
        assert result < 0


# ---------------------------------------------------------------------------
# inhibition_constant
# ---------------------------------------------------------------------------


class TestInhibitionConstant:
    def test_ic50_at_km_gives_ki_half(self):
        """Cheng-Prusoff: IC50 at [S]=Km → Ki = IC50 / 2."""
        ki = inhibition_constant(ic50_uM=10.0, substrate_conc_uM=5.0, km_uM=5.0)
        assert abs(ki - 5.0) < 1e-9

    def test_no_substrate_returns_ic50(self):
        """Without substrate/Km context, Ki = IC50."""
        ki = inhibition_constant(ic50_uM=8.0)
        assert abs(ki - 8.0) < 1e-9

    def test_cheng_prusoff_formula(self):
        """Ki = IC50 / (1 + [S]/Km)."""
        ic50 = 12.0
        s = 6.0
        km = 3.0
        expected = ic50 / (1 + s / km)
        ki = inhibition_constant(ic50_uM=ic50, substrate_conc_uM=s, km_uM=km)
        assert abs(ki - expected) < 1e-9

    def test_zero_ic50_raises(self):
        with pytest.raises(ValueError, match="ic50_uM"):
            inhibition_constant(ic50_uM=0.0)

    def test_negative_ic50_raises(self):
        with pytest.raises(ValueError, match="ic50_uM"):
            inhibition_constant(ic50_uM=-1.0)

    def test_zero_km_raises(self):
        with pytest.raises(ValueError, match="km_uM"):
            inhibition_constant(ic50_uM=10.0, substrate_conc_uM=5.0, km_uM=0.0)

    def test_ki_less_than_ic50_with_substrate(self):
        """Ki is always ≤ IC50 when [S] > 0."""
        ki = inhibition_constant(ic50_uM=20.0, substrate_conc_uM=10.0, km_uM=5.0)
        assert ki < 20.0
