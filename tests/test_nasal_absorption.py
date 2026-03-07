"""Tests for Phase 571 — nasal_absorption.py"""

import pytest

from omega_pbpk.biopharmaceutics.nasal_absorption import (
    compare_nasal_vs_oral,
    intranasal_bioavailability,
    nasal_absorption_model,
    nasal_deposition_fraction,
)

# ---------------------------------------------------------------------------
# nasal_deposition_fraction
# ---------------------------------------------------------------------------


class TestNasalDepositionFraction:
    def test_large_particle_high_nasal_deposition(self):
        """Particles > 10 um should have high nasal deposition fraction."""
        result = nasal_deposition_fraction(particle_size_um=20.0)
        assert result["deposit_frac_nasal"] > 0.7

    def test_small_particle_low_nasal_deposition(self):
        """Particles < 2 um should have low nasal deposition."""
        result = nasal_deposition_fraction(particle_size_um=1.0)
        assert result["deposit_frac_nasal"] < 0.5

    def test_small_particle_higher_lung_deposition(self):
        """Small particles should deposit more in lungs than large ones."""
        small = nasal_deposition_fraction(particle_size_um=0.5)
        large = nasal_deposition_fraction(particle_size_um=20.0)
        assert small["deposit_frac_lung"] > large["deposit_frac_lung"]

    def test_fractions_sum_to_one(self):
        """All deposition fractions must sum to 1."""
        for ps in [1.0, 5.0, 10.0, 20.0]:
            r = nasal_deposition_fraction(particle_size_um=ps)
            total = r["deposit_frac_nasal"] + r["deposit_frac_lung"] + r["exhaled_frac"]
            assert abs(total - 1.0) < 1e-9, f"Sum={total} for ps={ps}"

    def test_exhaled_frac_fixed(self):
        """Exhaled fraction should always be 0.05."""
        for ps in [0.5, 5.0, 50.0]:
            r = nasal_deposition_fraction(particle_size_um=ps)
            assert abs(r["exhaled_frac"] - 0.05) < 1e-9

    def test_higher_flow_increases_nasal_deposition(self):
        """Higher flow rate → larger Stk → more nasal deposition."""
        low_flow = nasal_deposition_fraction(particle_size_um=5.0, flow_rate_L_min=10.0)
        high_flow = nasal_deposition_fraction(particle_size_um=5.0, flow_rate_L_min=30.0)
        assert high_flow["deposit_frac_nasal"] > low_flow["deposit_frac_nasal"]

    def test_returns_notes_string(self):
        r = nasal_deposition_fraction(particle_size_um=5.0)
        assert isinstance(r["notes"], str) and len(r["notes"]) > 0

    def test_invalid_particle_size(self):
        with pytest.raises(ValueError):
            nasal_deposition_fraction(particle_size_um=0.0)

    def test_invalid_flow_rate(self):
        with pytest.raises(ValueError):
            nasal_deposition_fraction(particle_size_um=5.0, flow_rate_L_min=-1.0)

    def test_fractions_nonnegative(self):
        for ps in [0.1, 1.0, 100.0]:
            r = nasal_deposition_fraction(particle_size_um=ps)
            assert r["deposit_frac_nasal"] >= 0.0
            assert r["deposit_frac_lung"] >= 0.0
            assert r["exhaled_frac"] >= 0.0


# ---------------------------------------------------------------------------
# nasal_absorption_model
# ---------------------------------------------------------------------------


class TestNasalAbsorptionModel:
    def _run(self, peff=1e-4, mc_half=0.25):
        return nasal_absorption_model(
            drug_name="TestDrug",
            deposited_dose_mg=10.0,
            peff_cm_s=peff,
            mucociliary_t_half_h=mc_half,
        )

    def test_higher_peff_higher_f_absorbed(self):
        low = self._run(peff=1e-5)
        high = self._run(peff=1e-3)
        assert high["f_absorbed"] > low["f_absorbed"]

    def test_lower_mc_half_lower_f_absorbed(self):
        """Faster clearance (lower t½) means less time for absorption."""
        fast = self._run(mc_half=0.1)
        slow = self._run(mc_half=1.0)
        assert slow["f_absorbed"] > fast["f_absorbed"]

    def test_f_absorbed_in_unit_interval(self):
        r = self._run()
        assert 0.0 <= r["f_absorbed"] <= 1.0

    def test_amount_decreasing_over_time(self):
        r = self._run()
        amounts = r["amounts_on_mucosa_mg"]
        # Not monotone due to floating point, but should trend down
        assert amounts[0] >= amounts[-1]

    def test_initial_amount_equals_dose(self):
        r = self._run()
        assert abs(r["amounts_on_mucosa_mg"][0] - 10.0) < 1e-6

    def test_times_start_at_zero(self):
        r = self._run()
        assert r["times_h"][0] == pytest.approx(0.0)

    def test_cmax_positive_for_positive_peff(self):
        r = self._run(peff=1e-4)
        assert r["cmax_approx_mg_L"] > 0.0

    def test_returns_drug_name(self):
        r = self._run()
        assert r["drug_name"] == "TestDrug"

    def test_invalid_peff(self):
        with pytest.raises(ValueError):
            nasal_absorption_model("D", 10.0, peff_cm_s=-1.0)

    def test_invalid_mc_half(self):
        with pytest.raises(ValueError):
            nasal_absorption_model("D", 10.0, peff_cm_s=1e-4, mucociliary_t_half_h=0.0)


# ---------------------------------------------------------------------------
# intranasal_bioavailability
# ---------------------------------------------------------------------------


class TestIntranasalBioavailability:
    def test_no_first_pass(self):
        F = intranasal_bioavailability(f_nasal=0.8)
        assert abs(F - 0.8) < 1e-9

    def test_with_first_pass(self):
        F = intranasal_bioavailability(f_nasal=0.8, f_first_pass=0.5)
        assert abs(F - 0.4) < 1e-9

    def test_result_le_one(self):
        F = intranasal_bioavailability(f_nasal=1.0, f_first_pass=0.0)
        assert F <= 1.0

    def test_result_ge_zero(self):
        F = intranasal_bioavailability(f_nasal=0.0)
        assert F >= 0.0

    def test_invalid_f_nasal(self):
        with pytest.raises(ValueError):
            intranasal_bioavailability(f_nasal=1.5)

    def test_invalid_f_first_pass(self):
        with pytest.raises(ValueError):
            intranasal_bioavailability(f_nasal=0.5, f_first_pass=-0.1)


# ---------------------------------------------------------------------------
# compare_nasal_vs_oral
# ---------------------------------------------------------------------------


class TestCompareNasalVsOral:
    def test_nasal_advantage_true(self):
        r = compare_nasal_vs_oral(0.9, 0.5, 0.5, 1.5)
        assert r["nasal_advantage"] is True

    def test_nasal_advantage_false(self):
        r = compare_nasal_vs_oral(0.4, 0.8, 0.5, 1.5)
        assert r["nasal_advantage"] is False

    def test_bioavailability_ratio(self):
        r = compare_nasal_vs_oral(0.8, 0.4, 1.0, 2.0)
        assert abs(r["bioavailability_ratio"] - 2.0) < 1e-9

    def test_tmax_ratio(self):
        r = compare_nasal_vs_oral(0.5, 0.5, 1.0, 2.0)
        assert abs(r["tmax_ratio"] - 0.5) < 1e-9

    def test_notes_returned(self):
        r = compare_nasal_vs_oral(0.6, 0.4, 0.5, 1.5)
        assert isinstance(r["notes"], str) and len(r["notes"]) > 0

    def test_invalid_nasal_f(self):
        with pytest.raises(ValueError):
            compare_nasal_vs_oral(-0.1, 0.5, 1.0, 2.0)

    def test_invalid_oral_f(self):
        with pytest.raises(ValueError):
            compare_nasal_vs_oral(0.5, 1.5, 1.0, 2.0)

    def test_invalid_tmax(self):
        with pytest.raises(ValueError):
            compare_nasal_vs_oral(0.5, 0.5, 0.0, 1.0)
