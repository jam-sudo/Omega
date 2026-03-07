"""Tests for Phase 705 — BBB transport pharmacokinetics."""

from __future__ import annotations

import pytest

from omega_pbpk.core.bbb_transport_pk import (
    BBBTransportResult,
    compare_pgp_effect,
    simulate_bbb_transport,
)


class TestSimulateBBBTransport:
    """Tests for simulate_bbb_transport function."""

    def test_returns_bbb_transport_result(self):
        result = simulate_bbb_transport("TestDrug", dose_mg=10.0)
        assert isinstance(result, BBBTransportResult)

    def test_initial_plasma_concentration(self):
        """IV bolus: C_plasma[0] should equal dose_mg / vd_plasma_L."""
        dose = 10.0
        vd = 10.0
        result = simulate_bbb_transport("Drug", dose_mg=dose, vd_plasma_L=vd)
        expected = dose / vd
        assert abs(result.c_plasma_mg_L[0] - expected) < 1e-9

    def test_initial_brain_concentration_zero(self):
        """Brain concentration starts at zero for IV bolus."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0)
        assert result.c_brain_mg_L[0] == 0.0

    def test_cmax_brain_positive(self):
        """Brain Cmax should be positive after simulation."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0, logP=3.0, psa=50.0, t_end_h=12.0)
        assert result.cmax_brain > 0.0

    def test_auc_brain_positive(self):
        """AUC in brain compartment should be positive."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0, t_end_h=12.0)
        assert result.auc_brain > 0.0

    def test_auc_plasma_positive(self):
        """AUC in plasma compartment should be positive."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0, t_end_h=12.0)
        assert result.auc_plasma > 0.0

    def test_brain_to_plasma_ratio_positive(self):
        """Brain-to-plasma ratio should be positive."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0, t_end_h=12.0)
        assert result.brain_to_plasma_ratio > 0.0

    def test_high_logp_higher_brain_penetration(self):
        """High logP compound should have higher brain penetration than low logP."""
        result_high = simulate_bbb_transport("HighLogP", dose_mg=10.0, logP=4.0, psa=50.0)
        result_low = simulate_bbb_transport("LowLogP", dose_mg=10.0, logP=-1.0, psa=50.0)
        assert result_high.brain_to_plasma_ratio > result_low.brain_to_plasma_ratio

    def test_high_psa_lower_brain_penetration(self):
        """High PSA compound should have lower brain penetration than low PSA."""
        result_high_psa = simulate_bbb_transport("HighPSA", dose_mg=10.0, logP=2.0, psa=150.0)
        result_low_psa = simulate_bbb_transport("LowPSA", dose_mg=10.0, logP=2.0, psa=30.0)
        assert result_low_psa.brain_to_plasma_ratio > result_high_psa.brain_to_plasma_ratio

    def test_higher_fpgp_lower_brain_ratio(self):
        """Higher P-gp efflux factor should reduce brain-to-plasma ratio."""
        result_no_pgp = simulate_bbb_transport("Drug", dose_mg=10.0, logP=2.0, f_pgp=0.0)
        result_with_pgp = simulate_bbb_transport("Drug", dose_mg=10.0, logP=2.0, f_pgp=2.0)
        assert result_no_pgp.brain_to_plasma_ratio > result_with_pgp.brain_to_plasma_ratio

    def test_bbb_penetration_class_valid_values(self):
        """BBB penetration class should be one of the expected strings."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0)
        assert result.bbb_penetration_class in {"high", "moderate", "low"}

    def test_high_logp_classified_high_penetration(self):
        """Very high logP should give high BBB penetration class."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0, logP=5.0, psa=20.0, mw=200.0)
        assert result.bbb_penetration_class == "high"

    def test_low_logp_classified_low_penetration(self):
        """Very low logP + high PSA should give low BBB penetration class."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0, logP=-2.0, psa=180.0, mw=600.0)
        assert result.bbb_penetration_class == "low"

    def test_notes_nonempty(self):
        """Notes field should not be empty."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0)
        assert len(result.notes) > 0

    def test_ps_in_positive(self):
        """PS_in should be positive."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0)
        assert result.ps_in_L_per_h > 0.0

    def test_ps_out_positive(self):
        """PS_out (baseline) should be positive."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0)
        assert result.ps_out_L_per_h > 0.0

    def test_ps_out_equals_ps_in_no_pgp(self):
        """Without P-gp (f_pgp=0), PS_out should equal PS_in."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0, f_pgp=0.0)
        assert abs(result.ps_out_L_per_h - result.ps_in_L_per_h) < 1e-9

    def test_times_h_length_matches_concentrations(self):
        """Time and concentration lists should have equal length."""
        result = simulate_bbb_transport("Drug", dose_mg=10.0, t_end_h=5.0, dt_h=0.1)
        assert len(result.times_h) == len(result.c_plasma_mg_L) == len(result.c_brain_mg_L)

    def test_drug_name_stored(self):
        """Drug name should be stored in result."""
        result = simulate_bbb_transport("Aspirin", dose_mg=10.0)
        assert result.drug_name == "Aspirin"

    def test_validation_dose_zero_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bbb_transport("Drug", dose_mg=0.0)

    def test_validation_dose_negative_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_bbb_transport("Drug", dose_mg=-5.0)

    def test_validation_psa_negative_raises(self):
        with pytest.raises(ValueError, match="psa"):
            simulate_bbb_transport("Drug", dose_mg=10.0, psa=-10.0)

    def test_validation_fpgp_negative_raises(self):
        with pytest.raises(ValueError, match="f_pgp"):
            simulate_bbb_transport("Drug", dose_mg=10.0, f_pgp=-0.1)


class TestComparePgpEffect:
    """Tests for compare_pgp_effect function."""

    def test_returns_list_of_four_by_default(self):
        """Default call should return list of 4 results."""
        results = compare_pgp_effect("Drug", dose_mg=10.0)
        assert len(results) == 4

    def test_sorted_by_fpgp_ascending(self):
        """Results should be sorted by f_pgp ascending."""
        results = compare_pgp_effect("Drug", dose_mg=10.0)
        f_pgp_values = [r.f_pgp for r in results]
        assert f_pgp_values == sorted(f_pgp_values)

    def test_custom_fpgp_values(self):
        """Custom f_pgp values list length should match output."""
        results = compare_pgp_effect("Drug", dose_mg=10.0, f_pgp_values=[0.0, 1.0, 3.0])
        assert len(results) == 3

    def test_brain_ratio_decreases_with_pgp(self):
        """Brain-to-plasma ratio should decrease as P-gp efflux increases."""
        results = compare_pgp_effect("Drug", dose_mg=10.0, logP=3.0)
        ratios = [r.brain_to_plasma_ratio for r in results]
        # Each ratio should be <= the previous one
        for i in range(1, len(ratios)):
            assert ratios[i] <= ratios[i - 1]
