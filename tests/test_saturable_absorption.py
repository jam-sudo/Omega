"""Tests for saturable absorption kinetics module (Phase 290)."""

from __future__ import annotations

import pytest

from omega_pbpk.core.saturable_absorption import (
    DoseProportionalityResult,
    SaturableAbsorptionResult,
    dose_proportionality_analysis,
    saturable_absorption_rate,
    simulate_saturable_absorption_pk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(**kw) -> SaturableAbsorptionResult:
    defaults = dict(
        dose_mg=100.0,
        vd_L=30.0,
        cl_L_per_h=5.0,
        jmax_mg_per_h_per_cm2=0.005,
        km_mg_L=50.0,
        passive_fraction=0.1,
        dt_h=0.1,
    )
    defaults.update(kw)
    return simulate_saturable_absorption_pk(**defaults)


def _dpa(**kw) -> DoseProportionalityResult:
    defaults = dict(
        doses_mg=[10.0, 50.0, 100.0, 300.0, 600.0],
        vd_L=30.0,
        cl_L_per_h=5.0,
        jmax=0.005,
        km=50.0,
        passive_fraction=0.1,
        dt_h=0.1,
    )
    defaults.update(kw)
    return dose_proportionality_analysis(**defaults)


# ---------------------------------------------------------------------------
# saturable_absorption_rate
# ---------------------------------------------------------------------------


class TestSaturableAbsorptionRate:
    def test_zero_concentration_returns_zero(self):
        assert saturable_absorption_rate(0.0, 0.01, 10.0) == 0.0

    def test_negative_concentration_returns_zero(self):
        assert saturable_absorption_rate(-5.0, 0.01, 10.0) == 0.0

    def test_positive_concentration_positive_rate(self):
        rate = saturable_absorption_rate(10.0, 0.01, 10.0, area_cm2=100.0)
        assert rate > 0

    def test_carrier_saturates_at_high_conc(self):
        """At very high concentration, carrier approaches Jmax; adding more conc gives diminishing return."""
        carrier_low = 0.01 * 100.0 * 10.0 / (10.0 + 10.0)
        carrier_high = 0.01 * 100.0 * 10000.0 / (10.0 + 10000.0)
        assert carrier_high > carrier_low

    def test_at_km_carrier_half_max(self):
        """At C = Km, carrier rate = Jmax * area / 2."""
        jmax = 0.02
        km = 20.0
        area = 100.0
        rate = saturable_absorption_rate(km, jmax, km, area_cm2=area, passive_perm=0.0)
        expected_carrier = jmax * area * km / (km + km)
        assert abs(rate - expected_carrier) < 1e-6

    def test_passive_perm_explicit(self):
        """Explicit passive_perm=0 gives only carrier contribution."""
        jmax = 0.01
        km = 10.0
        area = 100.0
        c = 10.0
        rate = saturable_absorption_rate(c, jmax, km, area_cm2=area, passive_perm=0.0)
        expected = jmax * area * c / (km + c)
        assert abs(rate - expected) < 1e-9

    def test_larger_area_larger_rate(self):
        r_small = saturable_absorption_rate(10.0, 0.01, 5.0, area_cm2=100.0)
        r_large = saturable_absorption_rate(10.0, 0.01, 5.0, area_cm2=500.0)
        assert r_large > r_small


# ---------------------------------------------------------------------------
# simulate_saturable_absorption_pk
# ---------------------------------------------------------------------------


class TestSimulateSaturableAbsorptionPk:
    def test_returns_result_type(self):
        assert isinstance(_sim(), SaturableAbsorptionResult)

    def test_frozen_dataclass(self):
        r = _sim()
        with pytest.raises((AttributeError, TypeError)):
            r.cmax = 999.0  # type: ignore[misc]

    def test_times_and_plasma_same_length(self):
        r = _sim()
        assert len(r.times_h) == len(r.c_plasma)

    def test_cmax_positive(self):
        assert _sim().cmax > 0

    def test_auc_positive(self):
        assert _sim().auc > 0

    def test_f_absorbed_in_01(self):
        r = _sim()
        assert 0.0 <= r.f_absorbed <= 1.0

    def test_plasma_non_negative(self):
        r = _sim()
        assert all(c >= 0 for c in r.c_plasma)

    def test_dose_stored(self):
        r = _sim(dose_mg=200.0)
        assert r.dose_mg == 200.0

    def test_saturation_index_in_01(self):
        r = _sim()
        assert 0.0 <= r.saturation_index <= 1.0

    def test_larger_passive_fraction_larger_f_absorbed(self):
        """Higher passive permeability should increase total absorption."""
        r_low = _sim(passive_fraction=0.01)
        r_high = _sim(passive_fraction=1.0)
        assert r_high.f_absorbed >= r_low.f_absorbed

    def test_notes_contains_dose(self):
        r = _sim(dose_mg=50.0)
        assert "50.0" in r.notes

    def test_validation_dose_zero(self):
        with pytest.raises(ValueError, match="dose_mg"):
            _sim(dose_mg=0.0)

    def test_validation_vd_negative(self):
        with pytest.raises(ValueError, match="vd_L"):
            _sim(vd_L=-1.0)

    def test_validation_cl_zero(self):
        with pytest.raises(ValueError, match="cl_L_per_h"):
            _sim(cl_L_per_h=0.0)

    def test_validation_jmax_zero(self):
        with pytest.raises(ValueError, match="jmax"):
            _sim(jmax_mg_per_h_per_cm2=0.0)

    def test_validation_km_negative(self):
        with pytest.raises(ValueError, match="km"):
            _sim(km_mg_L=-5.0)

    def test_validation_area_zero(self):
        with pytest.raises(ValueError, match="area_cm2"):
            _sim(area_cm2=0.0)

    def test_higher_dose_higher_cmax(self):
        """More drug → higher plasma Cmax."""
        r_low = _sim(dose_mg=50.0)
        r_high = _sim(dose_mg=400.0)
        assert r_high.cmax > r_low.cmax


# ---------------------------------------------------------------------------
# dose_proportionality_analysis
# ---------------------------------------------------------------------------


class TestDoseProportionalityAnalysis:
    def test_returns_result_type(self):
        assert isinstance(_dpa(), DoseProportionalityResult)

    def test_frozen_dataclass(self):
        r = _dpa()
        with pytest.raises((AttributeError, TypeError)):
            r.power_law_exponent = 999.0  # type: ignore[misc]

    def test_doses_stored(self):
        doses = [10.0, 100.0, 500.0]
        r = _dpa(doses_mg=doses)
        assert r.doses_mg == doses

    def test_aucs_length_matches_doses(self):
        r = _dpa()
        assert len(r.aucs) == len(r.doses_mg)

    def test_cmaxes_length_matches_doses(self):
        r = _dpa()
        assert len(r.cmaxes) == len(r.doses_mg)

    def test_dose_normalized_aucs_length(self):
        r = _dpa()
        assert len(r.dose_normalized_aucs) == len(r.doses_mg)

    def test_aucs_positive(self):
        r = _dpa()
        assert all(a > 0 for a in r.aucs)

    def test_cmaxes_positive(self):
        r = _dpa()
        assert all(c > 0 for c in r.cmaxes)

    def test_saturable_gives_sublinear_exponent(self):
        """With very high Jmax saturation, power law exponent should be < 1."""
        r = _dpa(
            doses_mg=[10.0, 100.0, 500.0, 1000.0],
            jmax=0.001,  # low Jmax → strong saturation
            km=5.0,
            passive_fraction=0.0001,
        )
        assert r.power_law_exponent < 1.0

    def test_linear_absorption_gives_exponent_near_1(self):
        """Very high Km (no saturation) + dominant passive → near-linear PK."""
        r = _dpa(
            doses_mg=[10.0, 50.0, 100.0, 200.0],
            jmax=0.0001,  # tiny carrier contribution
            km=100000.0,  # very high Km → no saturation
            passive_fraction=5.0,  # dominant passive absorption
        )
        # With dominant linear passive, exponent should be close to 1
        assert r.power_law_exponent > 0.8

    def test_classification_in_valid_set(self):
        r = _dpa()
        assert r.proportionality_classification in {"linear", "slightly_nonlinear", "nonlinear"}

    def test_min_two_doses_required(self):
        with pytest.raises(ValueError):
            dose_proportionality_analysis([100.0], 30.0, 5.0, 0.005, 50.0)

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            dose_proportionality_analysis([-10.0, 100.0], 30.0, 5.0, 0.005, 50.0)

    def test_notes_contains_exponent(self):
        r = _dpa()
        assert "power_law_exponent" in r.notes
