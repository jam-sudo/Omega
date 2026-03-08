"""Tests for drug efflux pump kinetics model (phase 662)."""

import pytest

from omega_pbpk.core.efflux_pump_kinetics_p662 import (
    EffluxPumpKineticsResult,
    simulate_efflux_pump,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = dict(
    drug_name="TestDrug",
    pump_name="Pgp",
    initial_apical_mg_L=100.0,
    vmax_mg_h=0.01,
    km_mg_L=10.0,
    cell_volume_mL=0.5,
)


def _run(**overrides):
    params = {**BASE, **overrides}
    return simulate_efflux_pump(**params)


# ---------------------------------------------------------------------------
# Basic return-type & structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_result_dataclass(self):
        r = _run()
        assert isinstance(r, EffluxPumpKineticsResult)

    def test_drug_name_stored(self):
        r = _run()
        assert r.drug_name == "TestDrug"

    def test_pump_name_stored(self):
        r = _run()
        assert r.pump_name == "Pgp"

    def test_times_is_list(self):
        r = _run()
        assert isinstance(r.times_h, list)

    def test_list_lengths_match(self):
        r = _run()
        n = len(r.times_h)
        assert len(r.c_intracellular_mg_L) == n
        assert len(r.c_extracellular_mg_L) == n
        assert len(r.c_basolateral_mg_L) == n
        assert len(r.efflux_rate_mg_h) == n


# ---------------------------------------------------------------------------
# Non-negativity
# ---------------------------------------------------------------------------


class TestNonNegativity:
    def test_c_intracellular_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_intracellular_mg_L)

    def test_c_extracellular_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_extracellular_mg_L)

    def test_c_basolateral_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.c_basolateral_mg_L)

    def test_efflux_rate_nonneg(self):
        r = _run()
        assert all(v >= 0 for v in r.efflux_rate_mg_h)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


class TestDynamics:
    def test_intracellular_initially_increases(self):
        r = _run()
        # Should increase from 0 at start
        assert r.c_intracellular_mg_L[0] == 0.0
        mid = len(r.times_h) // 4
        assert r.c_intracellular_mg_L[mid] > 0.0

    def test_efflux_rate_positive_when_drug_in_cell(self):
        r = _run()
        mid = len(r.times_h) // 2
        assert r.efflux_rate_mg_h[mid] > 0.0

    def test_basolateral_accumulates(self):
        r = _run()
        assert r.c_basolateral_mg_L[-1] > r.c_basolateral_mg_L[0]


# ---------------------------------------------------------------------------
# Efflux properties
# ---------------------------------------------------------------------------


class TestEffluxProperties:
    def test_efflux_ratio_positive(self):
        r = _run()
        assert r.efflux_ratio > 0

    def test_permeability_ba_greater_than_ab(self):
        r = _run()
        assert r.permeability_ba_cm_s > r.permeability_ab_cm_s

    def test_efflux_efficiency_in_range(self):
        r = _run()
        assert 0.0 <= r.efflux_efficiency_pct <= 100.0

    def test_transporter_saturation_in_range(self):
        r = _run()
        assert 0.0 <= r.transporter_saturation_pct <= 100.0

    def test_ic50_is_twice_km(self):
        r = _run(km_mg_L=25.0)
        assert abs(r.ic50_inhibition_mg_L - 50.0) < 1e-10


# ---------------------------------------------------------------------------
# Vmax effects
# ---------------------------------------------------------------------------


class TestVmaxEffects:
    def test_higher_vmax_higher_efflux_efficiency(self):
        r_low = _run(vmax_mg_h=0.001)
        r_high = _run(vmax_mg_h=0.1)
        assert r_high.efflux_efficiency_pct > r_low.efflux_efficiency_pct

    def test_higher_vmax_lower_intracellular(self):
        r_low = _run(vmax_mg_h=0.001)
        r_high = _run(vmax_mg_h=0.1)
        assert r_high.steady_state_intracellular_mg_L <= r_low.steady_state_intracellular_mg_L


# ---------------------------------------------------------------------------
# Pump name validation
# ---------------------------------------------------------------------------


class TestPumpValidation:
    def test_valid_pgp(self):
        r = _run(pump_name="Pgp")
        assert r.pump_name == "Pgp"

    def test_valid_bcrp(self):
        r = _run(pump_name="BCRP")
        assert r.pump_name == "BCRP"

    def test_valid_mrp2(self):
        r = _run(pump_name="MRP2")
        assert r.pump_name == "MRP2"

    def test_invalid_pump_raises(self):
        with pytest.raises(ValueError, match="pump_name"):
            _run(pump_name="INVALID")


# ---------------------------------------------------------------------------
# Dose proportionality
# ---------------------------------------------------------------------------


class TestDoseProportionality:
    def test_higher_dose_more_basolateral(self):
        r_low = _run(initial_apical_mg_L=50.0)
        r_high = _run(initial_apical_mg_L=200.0)
        assert r_high.c_basolateral_mg_L[-1] > r_low.c_basolateral_mg_L[-1]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_zero_apical_raises(self):
        with pytest.raises(ValueError, match="initial_apical"):
            _run(initial_apical_mg_L=0.0)

    def test_negative_apical_raises(self):
        with pytest.raises(ValueError, match="initial_apical"):
            _run(initial_apical_mg_L=-10.0)

    def test_zero_vmax_raises(self):
        with pytest.raises(ValueError, match="vmax"):
            _run(vmax_mg_h=0.0)

    def test_negative_km_raises(self):
        with pytest.raises(ValueError, match="km"):
            _run(km_mg_L=-1.0)
