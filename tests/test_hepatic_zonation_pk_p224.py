"""Tests for Phase 224 — hepatic_zonation_pk_p224."""

import pytest

from omega_pbpk.core.hepatic_zonation_pk_p224 import (
    HepaticZonationResult,
    assess_zone_selectivity,
    simulate_hepatic_zonation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_DRUG = "TestDrug"
DEFAULT_CLINT = 5.0  # L/h
DEFAULT_FU = 0.1
DEFAULT_Q = 1.5  # L/h


def _sim(cl=DEFAULT_CLINT, fu=DEFAULT_FU, q=DEFAULT_Q):
    return simulate_hepatic_zonation(DEFAULT_DRUG, cl, fu, q)


# ---------------------------------------------------------------------------
# Return type / structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_hepatic_zonation_result(self):
        result = _sim()
        assert isinstance(result, HepaticZonationResult)

    def test_result_is_frozen(self):
        result = _sim()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "x"  # type: ignore[misc]

    def test_notes_non_empty(self):
        result = _sim()
        assert isinstance(result.notes, str)
        assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = simulate_hepatic_zonation("Ibuprofen", 3.0, 0.01, 1.5)
        assert result.drug_name == "Ibuprofen"

    def test_cl_intrinsic_preserved(self):
        result = _sim(cl=7.3)
        assert result.cl_intrinsic_L_per_h == pytest.approx(7.3)

    def test_fu_plasma_preserved(self):
        result = _sim(fu=0.25)
        assert result.fu_plasma == pytest.approx(0.25)

    def test_q_hepatic_preserved(self):
        result = _sim(q=2.0)
        assert result.q_hepatic_L_per_h == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Extraction ratio bounds [0, 1]
# ---------------------------------------------------------------------------


class TestExtractionBounds:
    def test_e_zone1_in_0_1(self):
        result = _sim()
        assert 0.0 <= result.e_zone1 <= 1.0

    def test_e_zone2_in_0_1(self):
        result = _sim()
        assert 0.0 <= result.e_zone2 <= 1.0

    def test_e_zone3_in_0_1(self):
        result = _sim()
        assert 0.0 <= result.e_zone3 <= 1.0

    def test_e_hepatic_uniform_in_0_1(self):
        result = _sim()
        assert 0.0 <= result.e_hepatic_uniform <= 1.0

    def test_e_hepatic_zonal_in_0_1(self):
        result = _sim()
        assert 0.0 <= result.e_hepatic_zonal <= 1.0


# ---------------------------------------------------------------------------
# Fh bounds [0, 1]
# ---------------------------------------------------------------------------


class TestFhBounds:
    def test_fh_uniform_in_0_1(self):
        result = _sim()
        assert 0.0 <= result.fh_uniform <= 1.0

    def test_fh_zonal_in_0_1(self):
        result = _sim()
        assert 0.0 <= result.fh_zonal <= 1.0

    def test_fh_plus_e_equals_one_uniform(self):
        result = _sim()
        assert result.fh_uniform + result.e_hepatic_uniform == pytest.approx(1.0, abs=1e-9)

    def test_fh_plus_e_equals_one_zonal(self):
        result = _sim()
        assert result.fh_zonal + result.e_hepatic_zonal == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# High / Low CLint behaviour
# ---------------------------------------------------------------------------


class TestClintEffect:
    def test_high_clint_gives_high_e_uniform(self):
        result_high = _sim(cl=100.0)
        result_low = _sim(cl=0.1)
        assert result_high.e_hepatic_uniform > result_low.e_hepatic_uniform

    def test_high_clint_gives_high_e_zonal(self):
        result_high = _sim(cl=100.0)
        result_low = _sim(cl=0.1)
        assert result_high.e_hepatic_zonal > result_low.e_hepatic_zonal

    def test_zero_clint_gives_zero_e(self):
        result = _sim(cl=0.0)
        assert result.e_hepatic_uniform == pytest.approx(0.0, abs=1e-9)
        assert result.e_hepatic_zonal == pytest.approx(0.0, abs=1e-9)

    def test_zero_clint_fh_equals_one(self):
        result = _sim(cl=0.0)
        assert result.fh_uniform == pytest.approx(1.0, abs=1e-9)
        assert result.fh_zonal == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Zonation vs Uniform difference
# ---------------------------------------------------------------------------


class TestZonationDifference:
    def test_fh_zonal_differs_from_fh_uniform(self):
        """With heterogeneous zone factors (0.8, 1.2, 1.0), zonal != uniform."""
        result = _sim(cl=5.0, fu=0.1)
        # The zonal and uniform models differ because zone factors are not all 1.0
        assert abs(result.fh_zonal - result.fh_uniform) > 1e-6

    def test_cl_hepatic_zonal_positive(self):
        result = _sim(cl=5.0)
        assert result.cl_hepatic_zonal_L_per_h >= 0.0

    def test_cl_hepatic_zonal_matches_q_times_e(self):
        result = _sim(cl=5.0, fu=0.1, q=1.5)
        expected = result.q_hepatic_L_per_h * result.e_hepatic_zonal
        assert result.cl_hepatic_zonal_L_per_h == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# assess_zone_selectivity tests
# ---------------------------------------------------------------------------


class TestAssessZoneSelectivity:
    def test_returns_dict(self):
        profile = {"zone1": 0.5, "zone2": 1.5, "zone3": 1.0}
        result = assess_zone_selectivity(DEFAULT_DRUG, 5.0, 0.1, profile)
        assert isinstance(result, dict)

    def test_expected_keys_present(self):
        profile = {"zone1": 0.5, "zone2": 1.5, "zone3": 1.0}
        result = assess_zone_selectivity(DEFAULT_DRUG, 5.0, 0.1, profile)
        for key in [
            "drug_name",
            "zone1_e",
            "zone2_e",
            "zone3_e",
            "dominant_zone",
            "fh_zonal",
            "cl_hepatic_L_per_h",
            "profile_used",
        ]:
            assert key in result

    def test_all_zone_e_in_0_1(self):
        profile = {"zone1": 0.5, "zone2": 1.5, "zone3": 1.0}
        result = assess_zone_selectivity(DEFAULT_DRUG, 5.0, 0.1, profile)
        for zone_key in ["zone1_e", "zone2_e", "zone3_e"]:
            assert 0.0 <= result[zone_key] <= 1.0

    def test_dominant_zone_high_factor_wins(self):
        profile = {"zone1": 0.1, "zone2": 3.0, "zone3": 0.1}
        result = assess_zone_selectivity(DEFAULT_DRUG, 5.0, 0.1, profile)
        assert result["dominant_zone"] == "zone2"

    def test_fh_zonal_in_0_1(self):
        profile = {"zone1": 1.0, "zone2": 1.0, "zone3": 1.0}
        result = assess_zone_selectivity(DEFAULT_DRUG, 5.0, 0.1, profile)
        assert 0.0 <= result["fh_zonal"] <= 1.0

    def test_drug_name_preserved(self):
        profile = {"zone1": 1.0, "zone2": 1.0, "zone3": 1.0}
        result = assess_zone_selectivity("Warfarin", 2.0, 0.01, profile)
        assert result["drug_name"] == "Warfarin"

    def test_profile_used_preserved(self):
        profile = {"zone1": 0.5, "zone2": 2.0, "zone3": 0.8}
        result = assess_zone_selectivity(DEFAULT_DRUG, 5.0, 0.1, profile)
        assert result["profile_used"] == profile


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_clint_raises(self):
        with pytest.raises(ValueError, match="cl_intrinsic"):
            simulate_hepatic_zonation("Drug", -1.0, 0.1, 1.5)

    def test_zero_fu_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_hepatic_zonation("Drug", 5.0, 0.0, 1.5)

    def test_fu_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_hepatic_zonation("Drug", 5.0, 1.5, 1.5)

    def test_zero_q_raises(self):
        with pytest.raises(ValueError, match="q_hepatic"):
            simulate_hepatic_zonation("Drug", 5.0, 0.1, 0.0)

    def test_negative_q_raises(self):
        with pytest.raises(ValueError, match="q_hepatic"):
            simulate_hepatic_zonation("Drug", 5.0, 0.1, -1.0)

    def test_assess_negative_clint_raises(self):
        with pytest.raises(ValueError):
            assess_zone_selectivity("Drug", -1.0, 0.1, {"zone1": 1.0})

    def test_assess_invalid_fu_raises(self):
        with pytest.raises(ValueError):
            assess_zone_selectivity("Drug", 5.0, 0.0, {"zone1": 1.0})

    def test_assess_invalid_q_raises(self):
        with pytest.raises(ValueError):
            assess_zone_selectivity("Drug", 5.0, 0.1, {"zone1": 1.0}, q_hepatic_L_per_h=0.0)
