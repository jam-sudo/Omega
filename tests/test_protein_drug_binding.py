"""Tests for protein-drug binding kinetics (Phase 328)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.protein_drug_binding import (
    BindingKineticsResult,
    DisplacementResult,
    drug_displacement_kinetics,
    plasma_protein_dissociation_half_life,
    simulate_binding_kinetics,
)


# ---------------------------------------------------------------------------
# plasma_protein_dissociation_half_life tests
# ---------------------------------------------------------------------------


class TestDissociationHalfLife:
    def test_basic_half_life(self):
        t_half = plasma_protein_dissociation_half_life(koff_per_h=1.0)
        assert t_half == pytest.approx(math.log(2.0), rel=1e-6)

    def test_fast_koff_short_half_life(self):
        t_half = plasma_protein_dissociation_half_life(koff_per_h=10.0)
        assert t_half == pytest.approx(math.log(2.0) / 10.0, rel=1e-6)

    def test_slow_koff_long_half_life(self):
        t_half_slow = plasma_protein_dissociation_half_life(koff_per_h=0.1)
        t_half_fast = plasma_protein_dissociation_half_life(koff_per_h=1.0)
        assert t_half_slow > t_half_fast

    def test_invalid_koff_zero(self):
        with pytest.raises(ValueError, match="koff_per_h must be > 0"):
            plasma_protein_dissociation_half_life(0.0)

    def test_invalid_koff_negative(self):
        with pytest.raises(ValueError, match="koff_per_h must be > 0"):
            plasma_protein_dissociation_half_life(-1.0)


# ---------------------------------------------------------------------------
# simulate_binding_kinetics tests
# ---------------------------------------------------------------------------


class TestSimulateBindingKinetics:
    def test_returns_binding_kinetics_result(self):
        result = simulate_binding_kinetics("HSA", 100.0, 600000.0)
        assert isinstance(result, BindingKineticsResult)

    def test_result_is_frozen(self):
        result = simulate_binding_kinetics("HSA", 100.0, 600000.0)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Changed"  # type: ignore[misc]

    def test_time_array_starts_at_zero(self):
        result = simulate_binding_kinetics("Drug", 100.0, 500.0)
        assert result.times_h[0] == pytest.approx(0.0)

    def test_time_array_ends_at_t_end(self):
        result = simulate_binding_kinetics("Drug", 100.0, 500.0, t_end_h=3.0)
        assert result.times_h[-1] == pytest.approx(3.0, abs=0.01)

    def test_dp_starts_at_zero(self):
        result = simulate_binding_kinetics("Drug", 100.0, 500.0)
        assert result.dp_complex_nM[0] == pytest.approx(0.0, abs=1e-9)

    def test_dp_approaches_equilibrium(self):
        result = simulate_binding_kinetics(
            "Drug", 100.0, 1000.0, kon_per_nM_per_h=0.1, koff_per_h=1.0, t_end_h=10.0
        )
        final_dp = result.dp_complex_nM[-1]
        assert final_dp == pytest.approx(result.equilibrium_dp_nM, rel=0.05)

    def test_equilibrium_dp_bounded_by_min_conc(self):
        result = simulate_binding_kinetics("Drug", 50.0, 100.0)
        assert result.equilibrium_dp_nM <= min(50.0, 100.0)

    def test_kd_computed_correctly(self):
        result = simulate_binding_kinetics(
            "Drug", 100.0, 500.0, kon_per_nM_per_h=0.2, koff_per_h=2.0
        )
        assert result.kd_nM == pytest.approx(2.0 / 0.2, rel=1e-6)

    def test_t_to_equilibrium_95pct_is_set(self):
        result = simulate_binding_kinetics(
            "Drug", 100.0, 500.0, kon_per_nM_per_h=0.1, koff_per_h=1.0, t_end_h=10.0
        )
        assert result.t_to_equilibrium_95pct_h is not None
        assert result.t_to_equilibrium_95pct_h > 0.0

    def test_d_free_plus_dp_equals_total(self):
        result = simulate_binding_kinetics("Drug", 100.0, 500.0)
        dp = result.dp_complex_nM[-1]
        d_free = result.d_free_nM[-1]
        assert dp + d_free == pytest.approx(100.0, rel=1e-4)

    def test_invalid_drug_conc_zero(self):
        with pytest.raises(ValueError, match="c_drug_nM must be > 0"):
            simulate_binding_kinetics("Drug", 0.0, 500.0)

    def test_invalid_protein_conc_negative(self):
        with pytest.raises(ValueError, match="c_protein_nM must be > 0"):
            simulate_binding_kinetics("Drug", 100.0, -50.0)

    def test_invalid_kon_zero(self):
        with pytest.raises(ValueError, match="kon_per_nM_per_h must be > 0"):
            simulate_binding_kinetics("Drug", 100.0, 500.0, kon_per_nM_per_h=0.0)

    def test_invalid_koff_zero(self):
        with pytest.raises(ValueError, match="koff_per_h must be > 0"):
            simulate_binding_kinetics("Drug", 100.0, 500.0, koff_per_h=0.0)

    def test_invalid_t_end_zero(self):
        with pytest.raises(ValueError, match="t_end_h must be > 0"):
            simulate_binding_kinetics("Drug", 100.0, 500.0, t_end_h=0.0)

    def test_dp_never_exceeds_drug_total(self):
        result = simulate_binding_kinetics("Drug", 100.0, 500.0, t_end_h=5.0)
        assert max(result.dp_complex_nM) <= 100.0 + 1e-9

    def test_dp_never_exceeds_protein_total(self):
        result = simulate_binding_kinetics("Drug", 100.0, 500.0, t_end_h=5.0)
        assert max(result.dp_complex_nM) <= 500.0 + 1e-9

    def test_notes_not_empty(self):
        result = simulate_binding_kinetics("Drug", 100.0, 500.0)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# drug_displacement_kinetics tests
# ---------------------------------------------------------------------------


class TestDrugDisplacementKinetics:
    def test_returns_displacement_result(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=100.0,
            c_drug2_nM=50.0,
            c_protein_nM=600.0,
            kon1=0.1,
            koff1=1.0,
            kon2=0.2,
            koff2=0.5,
        )
        assert isinstance(result, DisplacementResult)

    def test_result_is_frozen(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
            kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.c_protein_nM = 999.0  # type: ignore[misc]

    def test_time_starts_at_zero(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
            kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
        )
        assert result.times_h[0] == pytest.approx(0.0)

    def test_dp_starts_at_zero(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
            kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
        )
        assert result.dp1_nM[0] == pytest.approx(0.0, abs=1e-9)
        assert result.dp2_nM[0] == pytest.approx(0.0, abs=1e-9)

    def test_total_bound_does_not_exceed_protein(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=500.0, c_drug2_nM=500.0, c_protein_nM=200.0,
            kon1=1.0, koff1=0.1, kon2=1.0, koff2=0.1, t_end_h=5.0,
        )
        for dp1, dp2 in zip(result.dp1_nM, result.dp2_nM):
            assert dp1 + dp2 <= 200.0 + 1e-6

    def test_final_fu_drug1_in_0_1(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
            kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
        )
        assert 0.0 <= result.final_fu_drug1 <= 1.0

    def test_final_fu_drug2_in_0_1(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
            kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
        )
        assert 0.0 <= result.final_fu_drug2 <= 1.0

    def test_competitive_displacer_reduces_drug1_binding(self):
        # Without competitor: only drug1
        result_alone = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=1e-6, c_protein_nM=600.0,
            kon1=0.5, koff1=0.5, kon2=0.001, koff2=10.0, t_end_h=6.0,
        )
        # With strong competitor
        result_compete = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=1000.0, c_protein_nM=600.0,
            kon1=0.5, koff1=0.5, kon2=1.0, koff2=0.1, t_end_h=6.0,
        )
        # Drug1 should have less bound complex with strong competition
        assert result_compete.dp1_nM[-1] < result_alone.dp1_nM[-1]

    def test_invalid_drug1_conc_zero(self):
        with pytest.raises(ValueError, match="c_drug1_nM must be > 0"):
            drug_displacement_kinetics(
                c_drug1_nM=0.0, c_drug2_nM=50.0, c_protein_nM=600.0,
                kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
            )

    def test_invalid_drug2_conc_negative(self):
        with pytest.raises(ValueError, match="c_drug2_nM must be > 0"):
            drug_displacement_kinetics(
                c_drug1_nM=100.0, c_drug2_nM=-1.0, c_protein_nM=600.0,
                kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
            )

    def test_invalid_protein_conc_zero(self):
        with pytest.raises(ValueError, match="c_protein_nM must be > 0"):
            drug_displacement_kinetics(
                c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=0.0,
                kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
            )

    def test_invalid_kon1_zero(self):
        with pytest.raises(ValueError, match="kon1 must be > 0"):
            drug_displacement_kinetics(
                c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
                kon1=0.0, koff1=1.0, kon2=0.2, koff2=0.5,
            )

    def test_invalid_koff2_zero(self):
        with pytest.raises(ValueError, match="koff2 must be > 0"):
            drug_displacement_kinetics(
                c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
                kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.0,
            )

    def test_p_free_non_negative(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
            kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5, t_end_h=4.0,
        )
        assert all(p >= -1e-9 for p in result.p_free_nM)

    def test_notes_not_empty(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=100.0, c_drug2_nM=50.0, c_protein_nM=600.0,
            kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
        )
        assert len(result.notes) > 0

    def test_stored_concentrations_correct(self):
        result = drug_displacement_kinetics(
            c_drug1_nM=123.0, c_drug2_nM=456.0, c_protein_nM=789.0,
            kon1=0.1, koff1=1.0, kon2=0.2, koff2=0.5,
        )
        assert result.c_drug1_total == pytest.approx(123.0)
        assert result.c_drug2_total == pytest.approx(456.0)
        assert result.c_protein_nM == pytest.approx(789.0)
