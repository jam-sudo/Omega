"""Tests for Phase 219 — Drug Permeability pH Profile."""

import pytest

from omega_pbpk.prediction.permeability_ph import (
    PhPermeabilityResult,
    effective_permeability,
    fraction_unionized,
    optimal_absorption_ph,
    ph_permeability_profile,
)

# ---------------------------------------------------------------------------
# fraction_unionized
# ---------------------------------------------------------------------------


class TestFractionUnionized:
    def test_acid_at_pka_is_half(self):
        fu = fraction_unionized(pka=4.5, ph=4.5, compound_type="acid")
        assert abs(fu - 0.5) < 1e-6

    def test_acid_below_pka_mostly_unionized(self):
        # pH << pKa → fu ≈ 1
        fu = fraction_unionized(pka=8.0, ph=1.0, compound_type="acid")
        assert fu > 0.99

    def test_acid_above_pka_mostly_ionized(self):
        # pH >> pKa → fu ≈ 0
        fu = fraction_unionized(pka=2.0, ph=8.0, compound_type="acid")
        assert fu < 0.01

    def test_base_at_pka_is_half(self):
        fu = fraction_unionized(pka=8.0, ph=8.0, compound_type="base")
        assert abs(fu - 0.5) < 1e-6

    def test_base_above_pka_mostly_unionized(self):
        # pH >> pKa → fu ≈ 1
        fu = fraction_unionized(pka=4.0, ph=8.0, compound_type="base")
        assert fu > 0.99

    def test_base_below_pka_mostly_ionized(self):
        # pH << pKa → fu ≈ 0
        fu = fraction_unionized(pka=8.0, ph=1.0, compound_type="base")
        assert fu < 0.01

    def test_case_insensitive_compound_type(self):
        fu1 = fraction_unionized(4.5, 4.5, "ACID")
        fu2 = fraction_unionized(4.5, 4.5, "acid")
        assert abs(fu1 - fu2) < 1e-9

    def test_invalid_compound_type_raises(self):
        with pytest.raises(ValueError):
            fraction_unionized(4.5, 7.0, "neutral")

    def test_returns_float_in_0_1(self):
        for ph in [1.0, 4.0, 7.0, 8.0]:
            fu = fraction_unionized(4.5, ph, "acid")
            assert 0.0 <= fu <= 1.0


# ---------------------------------------------------------------------------
# effective_permeability
# ---------------------------------------------------------------------------


class TestEffectivePermeability:
    def test_never_exceeds_papp(self):
        papp = 5e-5
        for ph in [1.0, 4.0, 7.4, 8.0]:
            peff = effective_permeability(papp, 4.5, ph, "acid")
            assert peff <= papp + 1e-15

    def test_equals_papp_when_fully_unionized(self):
        # acid at very low pH → fu ≈ 1 → Peff ≈ Papp
        papp = 3e-5
        peff = effective_permeability(papp, 10.0, 1.0, "acid")
        assert abs(peff - papp) / papp < 0.01

    def test_zero_concentration_region(self):
        # base at very low pH (pKa=9) → fu ≈ 0 → Peff ≈ 0
        peff = effective_permeability(1e-4, 9.0, 1.0, "base")
        assert peff < 1e-10

    def test_invalid_papp_raises(self):
        with pytest.raises(ValueError):
            effective_permeability(-1e-5, 4.5, 7.0, "acid")

    def test_invalid_papp_zero_raises(self):
        with pytest.raises(ValueError):
            effective_permeability(0.0, 4.5, 7.0, "acid")


# ---------------------------------------------------------------------------
# ph_permeability_profile
# ---------------------------------------------------------------------------


class TestPhPermeabilityProfile:
    def test_returns_correct_type(self):
        result = ph_permeability_profile("aspirin", 5e-5, 3.5, "acid")
        assert isinstance(result, PhPermeabilityResult)

    def test_list_lengths_match_ph_range(self):
        result = ph_permeability_profile("aspirin", 5e-5, 3.5, "acid")
        n = len(result.ph_values)
        assert len(result.fraction_unionized_values) == n
        assert len(result.effective_permeability_cm_s) == n

    def test_default_ph_range_length(self):
        result = ph_permeability_profile("aspirin", 5e-5, 3.5, "acid")
        assert len(result.ph_values) == 5

    def test_weak_acid_max_absorption_stomach(self):
        # Weak acid (pKa=3.5) → mostly unionized at low pH → max Peff in stomach
        result = ph_permeability_profile("weak_acid", 5e-5, 3.5, "acid")
        assert result.ph_at_max_peff <= 2.0

    def test_weak_base_max_absorption_intestine(self):
        # Weak base (pKa=8.0) → mostly unionized at higher pH
        result = ph_permeability_profile("weak_base", 5e-5, 8.0, "base")
        assert result.ph_at_max_peff >= 6.0

    def test_bcs_high_permeability(self):
        result = ph_permeability_profile("high_perm", 5e-4, 4.5, "acid")
        assert result.bcs_class_permeability == "high"

    def test_bcs_low_permeability(self):
        result = ph_permeability_profile("low_perm", 1e-6, 4.5, "acid")
        assert result.bcs_class_permeability == "low"

    def test_absorption_window_not_empty(self):
        result = ph_permeability_profile("drug", 5e-5, 4.5, "acid")
        assert result.absorption_window != ""

    def test_max_peff_equals_max_of_list(self):
        result = ph_permeability_profile("drug", 5e-5, 4.5, "acid")
        assert abs(result.max_peff_cm_s - max(result.effective_permeability_cm_s)) < 1e-15

    def test_custom_ph_range(self):
        ph_range = [2.0, 5.0, 7.5]
        result = ph_permeability_profile("drug", 5e-5, 4.5, "acid", ph_range=ph_range)
        assert result.ph_values == ph_range

    def test_invalid_papp_raises(self):
        with pytest.raises(ValueError):
            ph_permeability_profile("drug", 0.0, 4.5, "acid")

    def test_invalid_pka_raises(self):
        with pytest.raises(ValueError):
            ph_permeability_profile("drug", 5e-5, 15.0, "acid")

    def test_too_few_ph_values_raises(self):
        with pytest.raises(ValueError):
            ph_permeability_profile("drug", 5e-5, 4.5, "acid", ph_range=[7.0])

    def test_notes_not_empty(self):
        result = ph_permeability_profile("drug", 5e-5, 4.5, "acid")
        assert result.notes != ""

    def test_drug_name_stored(self):
        result = ph_permeability_profile("ibuprofen", 5e-5, 4.5, "acid")
        assert result.drug_name == "ibuprofen"


# ---------------------------------------------------------------------------
# optimal_absorption_ph
# ---------------------------------------------------------------------------


class TestOptimalAbsorptionPh:
    def test_acid_optimal_ph_low(self):
        # Weak acid: best at low pH
        ph_opt = optimal_absorption_ph("aspirin", 5e-5, 3.5, "acid")
        assert ph_opt <= 2.0

    def test_base_optimal_ph_high(self):
        # Weak base (pKa=8): best at high pH
        ph_opt = optimal_absorption_ph("propranolol", 5e-5, 8.0, "base")
        assert ph_opt >= 7.0

    def test_returns_float(self):
        ph_opt = optimal_absorption_ph("drug", 5e-5, 5.0, "acid")
        assert isinstance(ph_opt, float)

    def test_ph_in_valid_range(self):
        ph_opt = optimal_absorption_ph("drug", 5e-5, 5.0, "acid")
        assert 1.0 <= ph_opt <= 8.0

    def test_invalid_compound_type_raises(self):
        with pytest.raises(ValueError):
            optimal_absorption_ph("drug", 5e-5, 5.0, "zwitterion")
