"""Tests for ionization_calculator — Phase 485."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.ionization_calculator import (
    GIChargeResult,
    effective_charge,
    fraction_ionized,
    fraction_neutral,
    ionization_profile,
    predict_gi_charge_states,
    solubility_from_ionization,
)

# ---------------------------------------------------------------------------
# fraction_ionized — acid
# ---------------------------------------------------------------------------


class TestFractionIonizedAcid:
    def test_acid_at_pka_half_ionized(self):
        fi = fraction_ionized(ph=7.4, pka=7.4, drug_type="acid")
        assert abs(fi - 0.5) < 1e-9

    def test_acid_ph_much_greater_than_pka_fully_ionized(self):
        fi = fraction_ionized(ph=14.0, pka=4.0, drug_type="acid")
        assert fi > 0.999

    def test_acid_ph_much_less_than_pka_mostly_neutral(self):
        fi = fraction_ionized(ph=1.0, pka=7.0, drug_type="acid")
        assert fi < 0.001

    def test_acid_aspirin_pka_stomach(self):
        # Aspirin pKa=3.5; at stomach pH=1.5 mostly neutral
        fi = fraction_ionized(ph=1.5, pka=3.5, drug_type="acid")
        assert fi < 0.01

    def test_acid_aspirin_intestine(self):
        # At pH=7.4 mostly ionized
        fi = fraction_ionized(ph=7.4, pka=3.5, drug_type="acid")
        assert fi > 0.99


# ---------------------------------------------------------------------------
# fraction_ionized — base
# ---------------------------------------------------------------------------


class TestFractionIonizedBase:
    def test_base_at_pka_half_ionized(self):
        fi = fraction_ionized(ph=9.0, pka=9.0, drug_type="base")
        assert abs(fi - 0.5) < 1e-9

    def test_base_ph_much_less_than_pka_fully_ionized(self):
        fi = fraction_ionized(ph=1.0, pka=9.0, drug_type="base")
        assert fi > 0.999

    def test_base_ph_much_greater_than_pka_mostly_neutral(self):
        fi = fraction_ionized(ph=14.0, pka=7.0, drug_type="base")
        assert fi < 0.001

    def test_base_amitriptyline_stomach(self):
        # pKa ~9.4; at pH=1.5 fully ionized
        fi = fraction_ionized(ph=1.5, pka=9.4, drug_type="base")
        assert fi > 0.999


# ---------------------------------------------------------------------------
# fraction_neutral
# ---------------------------------------------------------------------------


class TestFractionNeutral:
    def test_sum_to_one_acid(self):
        for ph in [1.5, 4.0, 7.4, 10.0]:
            fi = fraction_ionized(ph, pka=5.0, drug_type="acid")
            fn = fraction_neutral(ph, pka=5.0, drug_type="acid")
            assert abs(fi + fn - 1.0) < 1e-9

    def test_sum_to_one_base(self):
        for ph in [2.0, 7.0, 9.0, 12.0]:
            fi = fraction_ionized(ph, pka=8.0, drug_type="base")
            fn = fraction_neutral(ph, pka=8.0, drug_type="base")
            assert abs(fi + fn - 1.0) < 1e-9

    def test_neutral_drug_fully_neutral(self):
        fn = fraction_neutral(ph=7.4, pka=7.4, drug_type="neutral")
        assert fn == 1.0

    def test_neutral_drug_fi_zero(self):
        fi = fraction_ionized(ph=5.0, pka=8.0, drug_type="neutral")
        assert fi == 0.0


# ---------------------------------------------------------------------------
# effective_charge
# ---------------------------------------------------------------------------


class TestEffectiveCharge:
    def test_fully_ionized_acid_charge_minus_one(self):
        # Acid pKa=4 at pH=10: fi ≈ 1, charge ≈ -1
        charge = effective_charge(ph=10.0, pka_values=[4.0], drug_types=["acid"])
        assert charge < -0.99

    def test_fully_ionized_base_charge_plus_one(self):
        # Base pKa=9 at pH=1: fi ≈ 1, charge ≈ +1
        charge = effective_charge(ph=1.0, pka_values=[9.0], drug_types=["base"])
        assert charge > 0.99

    def test_neutral_drug_zero_charge(self):
        charge = effective_charge(ph=7.4, pka_values=[7.0], drug_types=["neutral"])
        assert charge == 0.0

    def test_zwitterion_net_charge(self):
        # Zwitterion: base pKa=9 and acid pKa=3; at pH=7.4 net ≈ 0 nominally
        charge = effective_charge(ph=7.4, pka_values=[9.0, 3.0], drug_types=["base", "acid"])
        # Both are charged at pH 7.4 → net approximately 0 (±1)
        assert -2.0 <= charge <= 2.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            effective_charge(ph=7.4, pka_values=[5.0, 8.0], drug_types=["acid"])

    def test_empty_pka_raises(self):
        with pytest.raises(ValueError):
            effective_charge(ph=7.4, pka_values=[], drug_types=[])


# ---------------------------------------------------------------------------
# ionization_profile
# ---------------------------------------------------------------------------


class TestIonizationProfile:
    def test_returns_correct_keys(self):
        profile = ionization_profile(pka=5.0, drug_type="acid")
        assert "ph_values" in profile
        assert "fraction_ionized" in profile
        assert "fraction_neutral" in profile

    def test_n_points_respected(self):
        profile = ionization_profile(pka=5.0, n_points=20)
        assert len(profile["ph_values"]) == 20

    def test_sum_to_one_at_each_point(self):
        profile = ionization_profile(pka=7.4, drug_type="acid", n_points=10)
        for fi, fn in zip(profile["fraction_ionized"], profile["fraction_neutral"]):
            assert abs(fi + fn - 1.0) < 1e-9

    def test_acid_monotone_increase_with_ph(self):
        profile = ionization_profile(pka=5.0, drug_type="acid", n_points=30)
        fi = profile["fraction_ionized"]
        assert all(fi[i] <= fi[i + 1] + 1e-9 for i in range(len(fi) - 1))

    def test_invalid_ph_range_raises(self):
        with pytest.raises(ValueError):
            ionization_profile(pka=5.0, ph_range=(10.0, 5.0))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            ionization_profile(pka=5.0, n_points=1)


# ---------------------------------------------------------------------------
# predict_gi_charge_states
# ---------------------------------------------------------------------------


class TestPredictGIChargeStates:
    def test_acid_pka4_mostly_neutral_in_stomach(self):
        result = predict_gi_charge_states(pka=4.0, drug_type="acid")
        assert isinstance(result, GIChargeResult)
        # pKa=4, stomach pH=1.5 → 10^(4-1.5)=316 → fn = 316/317 ≈ 0.997
        assert result.stomach_fraction_neutral > 0.99

    def test_base_pka9_mostly_ionized_in_stomach(self):
        result = predict_gi_charge_states(pka=9.0, drug_type="base")
        # pH=1.5, pKa=9 → fi = 1/(1+10^(1.5-9)) ≈ 1 → fn ≈ 0
        assert result.stomach_fraction_neutral < 0.01

    def test_neutral_drug_all_segments_neutral(self):
        result = predict_gi_charge_states(pka=7.4, drug_type="neutral")
        assert result.stomach_fraction_neutral == 1.0
        assert result.duodenum_fraction_neutral == 1.0
        assert result.jejunum_fraction_neutral == 1.0
        assert result.ileum_fraction_neutral == 1.0
        assert result.colon_fraction_neutral == 1.0

    def test_absorption_favorable_segment_is_string(self):
        result = predict_gi_charge_states(pka=5.0, drug_type="acid")
        assert isinstance(result.absorption_favorable_segment, str)
        assert result.absorption_favorable_segment in (
            "stomach",
            "duodenum",
            "jejunum",
            "ileum",
            "colon",
        )

    def test_result_is_frozen_dataclass(self):
        result = predict_gi_charge_states(pka=5.0, drug_type="acid")
        with pytest.raises((AttributeError, TypeError)):
            result.pka = 999.0  # type: ignore[misc]

    def test_invalid_drug_type_raises(self):
        with pytest.raises(ValueError):
            predict_gi_charge_states(pka=5.0, drug_type="amphoteric_unknown")


# ---------------------------------------------------------------------------
# solubility_from_ionization
# ---------------------------------------------------------------------------


class TestSolubilityFromIonization:
    def test_acid_high_ph_higher_solubility(self):
        s_low = solubility_from_ionization(
            intrinsic_sol_mg_mL=0.01, ph=3.0, pka=5.0, drug_type="acid"
        )
        s_high = solubility_from_ionization(
            intrinsic_sol_mg_mL=0.01, ph=8.0, pka=5.0, drug_type="acid"
        )
        assert s_high > s_low

    def test_base_low_ph_higher_solubility(self):
        s_low_ph = solubility_from_ionization(
            intrinsic_sol_mg_mL=0.01, ph=3.0, pka=8.0, drug_type="base"
        )
        s_high_ph = solubility_from_ionization(
            intrinsic_sol_mg_mL=0.01, ph=10.0, pka=8.0, drug_type="base"
        )
        assert s_low_ph > s_high_ph

    def test_neutral_solubility_unchanged(self):
        s = solubility_from_ionization(
            intrinsic_sol_mg_mL=1.0, ph=7.4, pka=5.0, drug_type="neutral"
        )
        assert s == 1.0

    def test_at_pka_solubility_doubled(self):
        # At pH = pKa: factor = 1 + 10^0 = 2
        s = solubility_from_ionization(intrinsic_sol_mg_mL=1.0, ph=5.0, pka=5.0, drug_type="acid")
        assert abs(s - 2.0) < 1e-6

    def test_negative_intrinsic_solubility_raises(self):
        with pytest.raises(ValueError):
            solubility_from_ionization(intrinsic_sol_mg_mL=-1.0, ph=7.4, pka=5.0)

    def test_invalid_ph_raises(self):
        with pytest.raises(ValueError):
            solubility_from_ionization(intrinsic_sol_mg_mL=1.0, ph=15.0, pka=5.0)
