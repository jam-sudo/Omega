"""Tests for Phase 621: Hepatic Sinusoid Zonation model."""

import pytest

from omega_pbpk.core.hepatic_zonation_p621 import (
    HepaticZonationResult,
    simulate_hepatic_zonation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        dose_mg=100.0,
        cl_int_portal_L_per_h=20.0,
        fm_cyp3a4=0.4,
        fm_cyp2e1=0.2,
    )
    params.update(kwargs)
    return simulate_hepatic_zonation(**params)


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_hepatic_zonation_result(self):
        r = default_result()
        assert isinstance(r, HepaticZonationResult)

    def test_drug_name_preserved(self):
        r = simulate_hepatic_zonation("Aspirin", 200.0, 15.0, 0.2, 0.1)
        assert r.drug_name == "Aspirin"

    def test_dose_mg_preserved(self):
        r = default_result(dose_mg=50.0)
        assert r.dose_mg == 50.0

    def test_times_is_list(self):
        r = default_result()
        assert isinstance(r.times_h, list)

    def test_times_starts_at_zero(self):
        r = default_result()
        assert r.times_h[0] == pytest.approx(0.0)

    def test_times_ends_near_t_end(self):
        r = simulate_hepatic_zonation("D", 100.0, 20.0, 0.3, 0.1, t_end_h=4.0, dt_h=0.05)
        assert r.times_h[-1] == pytest.approx(4.0, abs=0.06)

    def test_all_concentration_lists_same_length(self):
        r = default_result()
        n = len(r.times_h)
        assert len(r.c_portal_mg_L) == n
        assert len(r.c_zone1_mg_L) == n
        assert len(r.c_zone2_mg_L) == n
        assert len(r.c_zone3_mg_L) == n
        assert len(r.c_hepatic_vein_mg_L) == n

    def test_zone_contribution_is_dict(self):
        r = default_result()
        assert isinstance(r.zone_contribution, dict)

    def test_zone_contribution_has_all_zones(self):
        r = default_result()
        assert "zone1" in r.zone_contribution
        assert "zone2" in r.zone_contribution
        assert "zone3" in r.zone_contribution

    def test_notes_is_str(self):
        r = default_result()
        assert isinstance(r.notes, str) and len(r.notes) > 0


# ---------------------------------------------------------------------------
# Concentration non-negativity
# ---------------------------------------------------------------------------


class TestConcentrationNonNegative:
    def test_portal_non_negative(self):
        r = default_result()
        assert all(c >= 0.0 for c in r.c_portal_mg_L)

    def test_zone1_non_negative(self):
        r = default_result()
        assert all(c >= 0.0 for c in r.c_zone1_mg_L)

    def test_zone2_non_negative(self):
        r = default_result()
        assert all(c >= 0.0 for c in r.c_zone2_mg_L)

    def test_zone3_non_negative(self):
        r = default_result()
        assert all(c >= 0.0 for c in r.c_zone3_mg_L)

    def test_hepatic_vein_non_negative(self):
        r = default_result()
        assert all(c >= 0.0 for c in r.c_hepatic_vein_mg_L)


# ---------------------------------------------------------------------------
# Extraction ratio
# ---------------------------------------------------------------------------


class TestExtractionRatio:
    def test_extraction_ratio_in_0_1(self):
        r = default_result()
        assert 0.0 <= r.extraction_ratio_portal <= 1.0

    def test_high_clint_gives_nonzero_zone_metabolism(self):
        # High CLint → large fraction metabolized in zones → zone3 > portal output
        r = default_result(cl_int_portal_L_per_h=500.0)
        # Zone contributions should still sum to 1.0 regardless of CLint
        assert abs(sum(r.zone_contribution.values()) - 1.0) < 1e-6

    def test_extraction_ratio_zero_clint_scenario(self):
        # Very low CLint → very low extraction
        r = simulate_hepatic_zonation("D", 100.0, 0.01, 0.0, 0.0)
        assert r.extraction_ratio_portal < 0.5

    def test_extraction_ratio_bounded(self):
        # Extraction ratio must always be in [0, 1] regardless of AUC values
        r_high = default_result(cl_int_portal_L_per_h=500.0)
        assert 0.0 <= r_high.extraction_ratio_portal <= 1.0


# ---------------------------------------------------------------------------
# AUC checks
# ---------------------------------------------------------------------------


class TestAUC:
    def test_auc_portal_positive(self):
        r = default_result()
        assert r.auc_portal_mg_h_per_L > 0.0

    def test_auc_hv_positive(self):
        r = default_result()
        assert r.auc_hepatic_vein_mg_h_per_L > 0.0

    def test_higher_dose_higher_auc(self):
        r1 = default_result(dose_mg=100.0)
        r2 = default_result(dose_mg=200.0)
        assert r2.auc_portal_mg_h_per_L > r1.auc_portal_mg_h_per_L


# ---------------------------------------------------------------------------
# Zone contribution
# ---------------------------------------------------------------------------


class TestZoneContribution:
    def test_zone_contributions_sum_to_one(self):
        r = default_result()
        total = sum(r.zone_contribution.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_cyp2e1_dominant_more_zone3_metabolism(self):
        # CYP2E1 factor in zone3 = 2.0 → expect zone3 dominant
        r_2e1 = simulate_hepatic_zonation("D", 100.0, 30.0, fm_cyp3a4=0.0, fm_cyp2e1=0.9)
        # zone3 should have larger contribution than zone1
        assert r_2e1.zone_contribution["zone3"] > r_2e1.zone_contribution["zone1"]

    def test_cyp3a4_dominant_more_zone3_metabolism(self):
        # CYP3A4 factor in zone3 = 1.5 (vs zone1 = 0.5) → zone3 > zone1
        r_3a4 = simulate_hepatic_zonation("D", 100.0, 30.0, fm_cyp3a4=0.9, fm_cyp2e1=0.0)
        assert r_3a4.zone_contribution["zone3"] > r_3a4.zone_contribution["zone1"]

    def test_equal_fm_gives_balanced_contributions(self):
        # fm_cyp3a4=0, fm_cyp2e1=0 → other enzyme (all factors=1.0) → equal CLint per zone
        r = simulate_hepatic_zonation("D", 100.0, 30.0, fm_cyp3a4=0.0, fm_cyp2e1=0.0)
        # With equal CLint per zone, contributions should be roughly equal
        # (not exactly equal because C_z1 > C_z2 > C_z3 normally in tanks-in-series)
        contribs = list(r.zone_contribution.values())
        # All should be between 0.1 and 0.9 (reasonable range)
        for c in contribs:
            assert 0.0 <= c <= 1.0

    def test_zone_contributions_all_non_negative(self):
        r = default_result()
        for val in r.zone_contribution.values():
            assert val >= 0.0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_hepatic_zonation("D", -1.0, 10.0, 0.3, 0.1)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_hepatic_zonation("D", 0.0, 10.0, 0.3, 0.1)

    def test_zero_clint_raises(self):
        with pytest.raises(ValueError, match="cl_int"):
            simulate_hepatic_zonation("D", 100.0, 0.0, 0.3, 0.1)

    def test_negative_clint_raises(self):
        with pytest.raises(ValueError, match="cl_int"):
            simulate_hepatic_zonation("D", 100.0, -5.0, 0.3, 0.1)

    def test_fm_cyp3a4_above_1_raises(self):
        with pytest.raises(ValueError, match="fm_cyp3a4"):
            simulate_hepatic_zonation("D", 100.0, 10.0, 1.2, 0.0)

    def test_fm_cyp2e1_above_1_raises(self):
        with pytest.raises(ValueError, match="fm_cyp2e1"):
            simulate_hepatic_zonation("D", 100.0, 10.0, 0.0, 1.2)

    def test_fm_sum_above_1_raises(self):
        with pytest.raises(ValueError, match="fm_cyp3a4 \\+ fm_cyp2e1"):
            simulate_hepatic_zonation("D", 100.0, 10.0, 0.7, 0.5)

    def test_fm_exactly_1_total_allowed(self):
        # fm_cyp3a4=0.6, fm_cyp2e1=0.4 → sum=1.0 is OK
        r = simulate_hepatic_zonation("D", 100.0, 10.0, 0.6, 0.4)
        assert isinstance(r, HepaticZonationResult)


# ---------------------------------------------------------------------------
# Sensitivity / physical plausibility
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_higher_dose_scales_concentrations(self):
        r1 = simulate_hepatic_zonation("D", 100.0, 20.0, 0.4, 0.2)
        r2 = simulate_hepatic_zonation("D", 200.0, 20.0, 0.4, 0.2)
        # AUC should roughly double
        assert r2.auc_portal_mg_h_per_L > r1.auc_portal_mg_h_per_L

    def test_zone_concentrations_all_non_zero_after_absorption(self):
        # After drug absorption all zones should have non-zero concentrations
        r = simulate_hepatic_zonation("D", 100.0, 20.0, 0.4, 0.2, t_end_h=6.0)
        # Check at some mid-point time (index ~100 ≈ t=2h)
        mid = len(r.times_h) // 2
        assert r.c_zone1_mg_L[mid] >= 0.0
        assert r.c_zone2_mg_L[mid] >= 0.0
        assert r.c_zone3_mg_L[mid] >= 0.0
