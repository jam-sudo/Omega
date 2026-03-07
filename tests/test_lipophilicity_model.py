"""Tests for prediction/lipophilicity_model.py — logD and membrane partitioning."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.lipophilicity_model import (
    LogDProfileResult,
    aqueous_solubility_from_logd,
    logd_from_logp,
    membrane_partition_coefficient,
    predict_logd_profile,
)


# ---------------------------------------------------------------------------
# logd_from_logp
# ---------------------------------------------------------------------------


class TestLogdFromLogp:
    def test_neutral_compound_returns_logp(self):
        assert math.isclose(logd_from_logp(3.0, None, None, 7.4), 3.0)

    def test_neutral_any_ph(self):
        assert math.isclose(logd_from_logp(2.5, None, None, 5.0), 2.5)

    def test_acid_at_high_ph_decreases(self):
        # Acid pKa=5, pH=7.4 → fully ionised → logD << logP
        logD = logd_from_logp(2.0, pka_acid=5.0, pka_base=None, ph=7.4)
        assert logD < 2.0

    def test_acid_below_pka_approaches_logp(self):
        # Acid pKa=5, pH=1 → mostly unionised → logD ≈ logP
        logD = logd_from_logp(2.0, pka_acid=5.0, pka_base=None, ph=1.0)
        # At pH=1, ionisation = 10^(1-5) = 0.0001 → logD ≈ logP - log(1.0001)
        assert math.isclose(logD, 2.0, abs_tol=0.01)

    def test_base_at_low_ph_decreases(self):
        # Base pKa=9, pH=5 → protonated → logD < logP
        logD = logd_from_logp(2.0, pka_acid=None, pka_base=9.0, ph=5.0)
        assert logD < 2.0

    def test_base_above_pka_approaches_logp(self):
        # Base pKa=5, pH=9 → mostly neutral → logD ≈ logP
        logD = logd_from_logp(2.0, pka_acid=None, pka_base=5.0, ph=9.0)
        assert math.isclose(logD, 2.0, abs_tol=0.01)

    def test_zwitterion_both_corrections(self):
        logD_z = logd_from_logp(3.0, pka_acid=4.0, pka_base=9.0, ph=7.4)
        logD_a = logd_from_logp(3.0, pka_acid=4.0, pka_base=None, ph=7.4)
        logD_b = logd_from_logp(3.0, pka_acid=None, pka_base=9.0, ph=7.4)
        # Zwitterion more hydrophilic than either alone
        assert logD_z <= logD_a
        assert logD_z <= logD_b

    def test_acid_at_pka_logd_equals_logp_minus_log2(self):
        # At pH=pKa: 50% ionised → logD = logP - log10(2)
        logD = logd_from_logp(3.0, pka_acid=7.4, pka_base=None, ph=7.4)
        expected = 3.0 - math.log10(2.0)
        assert math.isclose(logD, expected, rel_tol=1e-9)

    def test_returns_float(self):
        result = logd_from_logp(2.0, None, None, 7.4)
        assert isinstance(result, float)

    def test_negative_logp_preserved_for_neutral(self):
        assert math.isclose(logd_from_logp(-1.0, None, None, 7.4), -1.0)


# ---------------------------------------------------------------------------
# membrane_partition_coefficient
# ---------------------------------------------------------------------------


class TestMembranePartitionCoefficient:
    def test_neutral_positive_logd(self):
        kp = membrane_partition_coefficient(3.0, logD74=2.0, mw=300.0, charge="neutral")
        expected = 10 ** (0.9 * 2.0 + 0.2)
        assert math.isclose(kp, expected, rel_tol=1e-9)

    def test_cationic_formula(self):
        kp = membrane_partition_coefficient(3.0, logD74=2.0, mw=300.0, charge="cationic")
        expected = 10 ** (0.7 * 2.0 + 0.5)
        assert math.isclose(kp, expected, rel_tol=1e-9)

    def test_anionic_formula(self):
        kp = membrane_partition_coefficient(3.0, logD74=2.0, mw=300.0, charge="anionic")
        expected = 10 ** (0.8 * 2.0 - 0.3)
        assert math.isclose(kp, expected, rel_tol=1e-9)

    def test_zwitterion_formula(self):
        kp = membrane_partition_coefficient(3.0, logD74=2.0, mw=300.0, charge="zwitterion")
        expected = 10 ** (0.85 * 2.0)
        assert math.isclose(kp, expected, rel_tol=1e-9)

    def test_invalid_charge_raises(self):
        with pytest.raises(ValueError, match="charge"):
            membrane_partition_coefficient(3.0, 2.0, 300.0, charge="amphoteric")

    def test_returns_float(self):
        result = membrane_partition_coefficient(3.0, 2.0, 300.0)
        assert isinstance(result, float)

    def test_kp_positive(self):
        kp = membrane_partition_coefficient(3.0, 2.0, 300.0)
        assert kp > 0.0

    def test_cationic_higher_than_anionic_positive_logd(self):
        # Cationic slope (0.7) + offset 0.5 vs anionic (0.8) - 0.3
        # At logD=2: cationic = 10^1.9; anionic = 10^1.3 → cationic higher
        kp_c = membrane_partition_coefficient(3.0, 2.0, 300.0, "cationic")
        kp_a = membrane_partition_coefficient(3.0, 2.0, 300.0, "anionic")
        assert kp_c > kp_a

    def test_case_insensitive_charge(self):
        kp1 = membrane_partition_coefficient(3.0, 2.0, 300.0, "Neutral")
        kp2 = membrane_partition_coefficient(3.0, 2.0, 300.0, "neutral")
        assert math.isclose(kp1, kp2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# predict_logd_profile
# ---------------------------------------------------------------------------


class TestPredictLogdProfile:
    def test_returns_logd_profile_result(self):
        result = predict_logd_profile(3.0, None, None)
        assert isinstance(result, LogDProfileResult)

    def test_frozen_dataclass(self):
        result = predict_logd_profile(3.0, None, None)
        with pytest.raises((AttributeError, TypeError)):
            result.logP = 4.0  # type: ignore[misc]

    def test_n_points(self):
        result = predict_logd_profile(3.0, None, None, n_points=20)
        assert len(result.ph_values) == 20
        assert len(result.logd_values) == 20

    def test_neutral_all_logd_equal_logp(self):
        result = predict_logd_profile(2.5, None, None)
        for ld in result.logd_values:
            assert math.isclose(ld, 2.5, rel_tol=1e-9)

    def test_logd_at_74_matches_function(self):
        result = predict_logd_profile(3.0, pka_acid=5.0, pka_base=None)
        expected = logd_from_logp(3.0, pka_acid=5.0, pka_base=None, ph=7.4)
        assert math.isclose(result.logd_at_74, expected, rel_tol=1e-9)

    def test_ph_range_ascending(self):
        result = predict_logd_profile(3.0, None, None, ph_range=(2.0, 9.0))
        assert result.ph_values[0] < result.ph_values[-1]

    def test_raises_on_invalid_ph_range(self):
        with pytest.raises(ValueError, match="ph_range"):
            predict_logd_profile(3.0, None, None, ph_range=(9.0, 2.0))

    def test_raises_on_equal_ph_range(self):
        with pytest.raises(ValueError, match="ph_range"):
            predict_logd_profile(3.0, None, None, ph_range=(7.0, 7.0))

    def test_raises_on_too_few_points(self):
        with pytest.raises(ValueError, match="n_points"):
            predict_logd_profile(3.0, None, None, n_points=1)

    def test_acid_min_logd_at_high_ph(self):
        # Acid fully ionised at high pH → minimum logD at high pH
        result = predict_logd_profile(3.0, pka_acid=4.0, pka_base=None)
        # min logD should be toward the higher pH end
        assert result.ph_of_min_logd > 5.0

    def test_base_min_logd_at_low_ph(self):
        # Base fully protonated at low pH → minimum logD at low pH
        result = predict_logd_profile(3.0, pka_acid=None, pka_base=9.0)
        assert result.ph_of_min_logd < 5.0

    def test_notes_nonempty(self):
        result = predict_logd_profile(3.0, None, None)
        assert len(result.notes) > 0

    def test_logp_stored(self):
        result = predict_logd_profile(2.7, None, None)
        assert math.isclose(result.logP, 2.7)


# ---------------------------------------------------------------------------
# aqueous_solubility_from_logd
# ---------------------------------------------------------------------------


class TestAqueousSolubilityFromLogd:
    def test_neutral_compound(self):
        # logP=logD=2, Tm=150 → log S = 0.5 - 2 - 0.01*(150-25) = 0.5-2-1.25=-2.75
        # ionisation_factor = 10^(2-2) = 1 → S = 10^-2.75 * 2
        log_s = 0.5 - 2.0 - 0.01 * (150.0 - 25.0)
        s_intr = 10**log_s
        expected = s_intr * (1.0 + 10 ** (2.0 - 2.0))
        result = aqueous_solubility_from_logd(logP=2.0, logD74=2.0, melting_point_C=150.0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_ionised_compound_higher_solubility(self):
        # logD74 < logP means ionised → higher solubility vs neutral
        s_neutral = aqueous_solubility_from_logd(2.0, 2.0, 150.0)
        s_ionised = aqueous_solubility_from_logd(2.0, 0.0, 150.0)  # logD=0 < logP=2
        assert s_ionised > s_neutral

    def test_higher_logp_lower_solubility(self):
        s1 = aqueous_solubility_from_logd(2.0, 2.0, 150.0)
        s2 = aqueous_solubility_from_logd(4.0, 4.0, 150.0)
        assert s2 < s1

    def test_higher_melting_point_lower_solubility(self):
        s1 = aqueous_solubility_from_logd(2.0, 2.0, 100.0)
        s2 = aqueous_solubility_from_logd(2.0, 2.0, 200.0)
        assert s2 < s1

    def test_returns_float(self):
        result = aqueous_solubility_from_logd(2.0, 2.0)
        assert isinstance(result, float)

    def test_positive_solubility(self):
        result = aqueous_solubility_from_logd(2.0, 2.0)
        assert result > 0.0
