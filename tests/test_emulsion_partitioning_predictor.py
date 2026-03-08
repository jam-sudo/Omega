"""Tests for Phase 311 — Emulsion Drug Partitioning Predictor."""

import pytest

from omega_pbpk.prediction.emulsion_partitioning_predictor import (
    EmulsionPartitioningResult,
    predict_emulsion_partitioning,
    screen_oil_fractions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_basic(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        logp=2.0,
        pka=7.0,
        ionization_type="neutral",
        oil_fraction=0.3,
        emulsion_type="o/w",
        ph_aqueous=7.4,
    )
    defaults.update(kwargs)
    return predict_emulsion_partitioning(**defaults)


# ---------------------------------------------------------------------------
# Return type and field preservation
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_correct_type(self):
        result = make_basic()
        assert isinstance(result, EmulsionPartitioningResult)

    def test_drug_name_preserved(self):
        result = make_basic(drug_name="Ibuprofen")
        assert result.drug_name == "Ibuprofen"

    def test_logp_preserved(self):
        result = make_basic(logp=3.5)
        assert result.logp == 3.5

    def test_pka_preserved(self):
        result = make_basic(pka=4.5)
        assert result.pka == 4.5

    def test_ionization_type_preserved(self):
        result = make_basic(ionization_type="acid")
        assert result.ionization_type == "acid"

    def test_oil_fraction_preserved(self):
        result = make_basic(oil_fraction=0.4)
        assert result.oil_fraction == 0.4

    def test_emulsion_type_preserved(self):
        result = make_basic(emulsion_type="w/o")
        assert result.emulsion_type == "w/o"

    def test_ph_preserved(self):
        result = make_basic(ph_aqueous=6.8)
        assert result.ph_aqueous == 6.8


# ---------------------------------------------------------------------------
# Phase fraction sum to 1.0
# ---------------------------------------------------------------------------


class TestFractionSumToOne:
    def test_fractions_sum_to_one_ow(self):
        result = make_basic(emulsion_type="o/w", logp=3.0)
        assert abs(result.oil_phase_fraction_drug + result.water_phase_fraction_drug - 1.0) < 1e-10

    def test_fractions_sum_to_one_wo(self):
        result = make_basic(emulsion_type="w/o", logp=1.0)
        assert abs(result.oil_phase_fraction_drug + result.water_phase_fraction_drug - 1.0) < 1e-10

    def test_fractions_sum_to_one_neutral_high_logp(self):
        result = make_basic(logp=5.0, ionization_type="neutral")
        assert abs(result.oil_phase_fraction_drug + result.water_phase_fraction_drug - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# High logP → high oil fraction
# ---------------------------------------------------------------------------


class TestHighLogP:
    def test_high_logp_high_oil_fraction(self):
        result = make_basic(logp=6.0, ionization_type="neutral")
        assert result.oil_phase_fraction_drug > 0.9

    def test_high_logp_low_water_fraction(self):
        result = make_basic(logp=6.0, ionization_type="neutral")
        assert result.water_phase_fraction_drug < 0.1


# ---------------------------------------------------------------------------
# Low logP → high water fraction
# ---------------------------------------------------------------------------


class TestLowLogP:
    def test_low_logp_high_water_fraction(self):
        result = make_basic(logp=-2.0, ionization_type="neutral")
        assert result.water_phase_fraction_drug > 0.9

    def test_low_logp_low_oil_fraction(self):
        result = make_basic(logp=-2.0, ionization_type="neutral")
        assert result.oil_phase_fraction_drug < 0.1


# ---------------------------------------------------------------------------
# Ionization effects on logD
# ---------------------------------------------------------------------------


class TestIonizationEffect:
    def test_acid_at_low_ph_higher_logd(self):
        """Acid at low pH → mostly un-ionized → logD closer to logP."""
        low_ph = make_basic(ionization_type="acid", pka=6.0, ph_aqueous=3.0, logp=2.0)
        high_ph = make_basic(ionization_type="acid", pka=6.0, ph_aqueous=9.0, logp=2.0)
        assert low_ph.log_d_at_ph > high_ph.log_d_at_ph

    def test_acid_at_low_ph_higher_oil_fraction(self):
        """Acid at low pH → higher logD → more drug in oil phase."""
        low_ph = make_basic(ionization_type="acid", pka=6.0, ph_aqueous=3.0, logp=2.0)
        high_ph = make_basic(ionization_type="acid", pka=6.0, ph_aqueous=9.0, logp=2.0)
        assert low_ph.oil_phase_fraction_drug > high_ph.oil_phase_fraction_drug

    def test_base_at_high_ph_higher_logd(self):
        """Base at high pH → mostly un-ionized → logD closer to logP."""
        high_ph = make_basic(ionization_type="base", pka=8.0, ph_aqueous=11.0, logp=2.0)
        low_ph = make_basic(ionization_type="base", pka=8.0, ph_aqueous=4.0, logp=2.0)
        assert high_ph.log_d_at_ph > low_ph.log_d_at_ph

    def test_neutral_logd_equals_logp(self):
        result = make_basic(ionization_type="neutral", logp=3.0)
        assert abs(result.log_d_at_ph - 3.0) < 1e-10


# ---------------------------------------------------------------------------
# Release rate classification
# ---------------------------------------------------------------------------


class TestReleaseRate:
    def test_high_d_slow_release(self):
        # logD must produce D > 100; neutral with logp=3 → D=1000
        result = make_basic(logp=3.0, ionization_type="neutral")
        assert result.release_rate_relative == "slow"

    def test_moderate_d_moderate_release(self):
        # logp=1.5 → D≈31.6
        result = make_basic(logp=1.5, ionization_type="neutral")
        assert result.release_rate_relative == "moderate"

    def test_low_d_fast_release(self):
        # logp=-1 → D≈0.1
        result = make_basic(logp=-1.0, ionization_type="neutral")
        assert result.release_rate_relative == "fast"


# ---------------------------------------------------------------------------
# Bioavailability factor
# ---------------------------------------------------------------------------


class TestBioavailabilityFactor:
    def test_ow_bioavailability_is_water_fraction(self):
        result = make_basic(emulsion_type="o/w", logp=2.0, ionization_type="neutral")
        assert abs(result.bioavailability_factor - result.water_phase_fraction_drug) < 1e-10

    def test_wo_bioavailability_is_oil_fraction_times_07(self):
        result = make_basic(emulsion_type="w/o", logp=2.0, ionization_type="neutral")
        expected = result.oil_phase_fraction_drug * 0.7
        assert abs(result.bioavailability_factor - expected) < 1e-10


# ---------------------------------------------------------------------------
# Stability risk
# ---------------------------------------------------------------------------


class TestStabilityRisk:
    def test_high_stability_risk_when_mostly_in_oil(self):
        # Very high logp → f_oil > 0.9 → high risk
        result = make_basic(logp=7.0, ionization_type="neutral", oil_fraction=0.5)
        assert result.stability_risk == "high"

    def test_low_stability_risk_when_mostly_in_water(self):
        result = make_basic(logp=-2.0, ionization_type="neutral")
        assert result.stability_risk == "low"


# ---------------------------------------------------------------------------
# screen_oil_fractions
# ---------------------------------------------------------------------------


class TestScreenOilFractions:
    def test_returns_list(self):
        results = screen_oil_fractions("Drug", 2.0, 7.0, "neutral", [0.1, 0.3, 0.5])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_descending_by_oil_phase_fraction(self):
        results = screen_oil_fractions("Drug", 2.0, 7.0, "neutral", [0.1, 0.3, 0.5, 0.7])
        fractions = [r.oil_phase_fraction_drug for r in results]
        assert fractions == sorted(fractions, reverse=True)

    def test_all_results_are_correct_type(self):
        results = screen_oil_fractions("Drug", 2.0, 7.0, "neutral", [0.2, 0.5])
        for r in results:
            assert isinstance(r, EmulsionPartitioningResult)

    def test_screen_single_oil_fraction(self):
        results = screen_oil_fractions("Drug", 2.0, 7.0, "neutral", [0.3])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_logp_too_low(self):
        with pytest.raises(ValueError, match="logp"):
            make_basic(logp=-6.0)

    def test_invalid_logp_too_high(self):
        with pytest.raises(ValueError, match="logp"):
            make_basic(logp=11.0)

    def test_invalid_pka_zero(self):
        with pytest.raises(ValueError, match="pka"):
            make_basic(pka=0.0)

    def test_invalid_pka_too_high(self):
        with pytest.raises(ValueError, match="pka"):
            make_basic(pka=15.0)

    def test_invalid_oil_fraction_zero(self):
        with pytest.raises(ValueError, match="oil_fraction"):
            make_basic(oil_fraction=0.0)

    def test_invalid_oil_fraction_one(self):
        with pytest.raises(ValueError, match="oil_fraction"):
            make_basic(oil_fraction=1.0)

    def test_invalid_emulsion_type(self):
        with pytest.raises(ValueError, match="emulsion_type"):
            make_basic(emulsion_type="invalid")

    def test_invalid_ph_zero(self):
        with pytest.raises(ValueError, match="ph_aqueous"):
            make_basic(ph_aqueous=0.0)

    def test_invalid_ph_14(self):
        with pytest.raises(ValueError, match="ph_aqueous"):
            make_basic(ph_aqueous=14.0)

    def test_invalid_ionization_type(self):
        with pytest.raises(ValueError, match="ionization_type"):
            make_basic(ionization_type="zwitterion")
