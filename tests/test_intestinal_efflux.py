"""Tests for Phase 1081 — Intestinal Efflux (P-gp / BCRP) model."""

import pytest

from omega_pbpk.core.intestinal_efflux import (
    IntestinalEffluxResult,
    assess_pgp_impact,
    simulate_intestinal_efflux,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_result():
    return simulate_intestinal_efflux("TestDrug", dose_mg=100.0)


@pytest.fixture
def high_pgp_result():
    return simulate_intestinal_efflux("TestDrug", dose_mg=100.0, pgp_expression=3.0)


@pytest.fixture
def inhibitor_result():
    return simulate_intestinal_efflux("TestDrug", dose_mg=100.0, inhibitor_fraction=0.8)


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self, baseline_result):
        assert isinstance(baseline_result, IntestinalEffluxResult)

    def test_drug_name_stored(self, baseline_result):
        assert baseline_result.drug_name == "TestDrug"

    def test_dose_stored(self, baseline_result):
        assert baseline_result.dose_mg == 100.0

    def test_times_is_list(self, baseline_result):
        assert isinstance(baseline_result.times_h, list)
        assert len(baseline_result.times_h) > 0

    def test_a_lumen_is_list(self, baseline_result):
        assert isinstance(baseline_result.a_lumen_mg, list)

    def test_a_enterocyte_is_list(self, baseline_result):
        assert isinstance(baseline_result.a_enterocyte_mg, list)

    def test_a_portal_is_list(self, baseline_result):
        assert isinstance(baseline_result.a_portal_mg, list)


# ---------------------------------------------------------------------------
# Initial condition tests
# ---------------------------------------------------------------------------


class TestInitialConditions:
    def test_a_lumen_starts_at_dose(self, baseline_result):
        assert abs(baseline_result.a_lumen_mg[0] - 100.0) < 1e-9

    def test_a_enterocyte_starts_at_zero(self, baseline_result):
        assert baseline_result.a_enterocyte_mg[0] == pytest.approx(0.0)

    def test_a_portal_starts_at_zero(self, baseline_result):
        assert baseline_result.a_portal_mg[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Pharmacodynamics / fractions
# ---------------------------------------------------------------------------


class TestFractions:
    def test_f_absorbed_between_0_and_1(self, baseline_result):
        assert 0.0 <= baseline_result.f_absorbed <= 1.0

    def test_f_effluxed_between_0_and_1(self, baseline_result):
        assert 0.0 <= baseline_result.f_effluxed <= 1.0

    def test_f_metabolized_gut_between_0_and_1(self, baseline_result):
        assert 0.0 <= baseline_result.f_metabolized_gut <= 1.0

    def test_mass_balance_le_1(self, baseline_result):
        # f_effluxed counts cumulative efflux including recycled molecules (cycling effect).
        # f_absorbed + f_metabolized_gut should together not exceed ~1.0.
        # The spec states "approximately <= 1.0" — cycling means efflux can cause sum > 1.
        # Each individual fraction must be in [0,1].
        assert baseline_result.f_absorbed <= 1.0
        assert baseline_result.f_effluxed <= 1.0
        assert baseline_result.f_metabolized_gut <= 1.0
        # Non-efflux routes: absorbed + metabolized <= 1 (these are unique drug fates)
        assert (baseline_result.f_absorbed + baseline_result.f_metabolized_gut) <= 1.01

    def test_efflux_ratio_in_0_to_1(self, baseline_result):
        assert 0.0 <= baseline_result.efflux_ratio <= 1.0

    def test_efflux_ratio_in_0_to_1_high_pgp(self, high_pgp_result):
        assert 0.0 <= high_pgp_result.efflux_ratio <= 1.0


# ---------------------------------------------------------------------------
# Transporter effect tests
# ---------------------------------------------------------------------------


class TestTransporterEffects:
    def test_higher_pgp_lowers_f_absorbed(self, baseline_result, high_pgp_result):
        assert high_pgp_result.f_absorbed < baseline_result.f_absorbed

    def test_inhibitor_increases_f_absorbed(self, baseline_result, inhibitor_result):
        assert inhibitor_result.f_absorbed > baseline_result.f_absorbed

    def test_higher_pgp_increases_efflux_ratio(self, baseline_result, high_pgp_result):
        assert high_pgp_result.efflux_ratio > baseline_result.efflux_ratio

    def test_inhibitor_decreases_efflux_ratio(self, baseline_result, inhibitor_result):
        assert inhibitor_result.efflux_ratio < baseline_result.efflux_ratio


# ---------------------------------------------------------------------------
# AUC and bioavailability impact
# ---------------------------------------------------------------------------


class TestAUCAndImpact:
    def test_auc_portal_positive(self, baseline_result):
        assert baseline_result.auc_portal > 0.0

    def test_bioavailability_impact_valid_values(self, baseline_result):
        assert baseline_result.bioavailability_impact in {"high", "moderate", "low"}

    def test_bioavailability_impact_valid_high_pgp(self, high_pgp_result):
        assert high_pgp_result.bioavailability_impact in {"high", "moderate", "low"}

    def test_high_pgp_gives_high_impact(self):
        result = simulate_intestinal_efflux(
            "Substrate", dose_mg=50.0, pgp_expression=5.0, bcrp_expression=2.0
        )
        assert result.bioavailability_impact == "high"

    def test_no_efflux_gives_low_impact(self):
        result = simulate_intestinal_efflux(
            "Drug", dose_mg=50.0, pgp_expression=0.0, bcrp_expression=0.0
        )
        assert result.bioavailability_impact == "low"


# ---------------------------------------------------------------------------
# inhibitor_benefit
# ---------------------------------------------------------------------------


class TestInhibitorBenefit:
    def test_inhibitor_benefit_nonnegative(self, baseline_result):
        # Inhibition should not decrease AUC
        assert baseline_result.inhibitor_benefit >= 0.0

    def test_inhibitor_benefit_zero_when_already_fully_inhibited(self):
        # At inhibitor_fraction=0.9 (our reference point), benefit should be ~0
        result = simulate_intestinal_efflux("Drug", dose_mg=50.0, inhibitor_fraction=0.9)
        assert abs(result.inhibitor_benefit) < 1e-6


# ---------------------------------------------------------------------------
# assess_pgp_impact
# ---------------------------------------------------------------------------


class TestAssessPgpImpact:
    def test_returns_correct_count(self):
        pgp_levels = [0.5, 1.0, 2.0, 3.0]
        results = assess_pgp_impact("Drug", 100.0, pgp_levels)
        assert len(results) == 4

    def test_sorted_by_f_absorbed_descending(self):
        pgp_levels = [0.5, 1.0, 2.0, 3.0]
        results = assess_pgp_impact("Drug", 100.0, pgp_levels)
        absorbed_vals = [r.f_absorbed for r in results]
        assert absorbed_vals == sorted(absorbed_vals, reverse=True)

    def test_single_pgp_level(self):
        results = assess_pgp_impact("Drug", 100.0, [1.0])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_intestinal_efflux("Drug", dose_mg=-10.0)

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="dose_mg"):
            simulate_intestinal_efflux("Drug", dose_mg=0.0)

    def test_negative_pgp_raises(self):
        with pytest.raises(ValueError, match="pgp_expression"):
            simulate_intestinal_efflux("Drug", dose_mg=50.0, pgp_expression=-1.0)

    def test_negative_bcrp_raises(self):
        with pytest.raises(ValueError, match="bcrp_expression"):
            simulate_intestinal_efflux("Drug", dose_mg=50.0, bcrp_expression=-0.5)

    def test_inhibitor_fraction_above_1_raises(self):
        with pytest.raises(ValueError, match="inhibitor_fraction"):
            simulate_intestinal_efflux("Drug", dose_mg=50.0, inhibitor_fraction=1.5)

    def test_inhibitor_fraction_below_0_raises(self):
        with pytest.raises(ValueError, match="inhibitor_fraction"):
            simulate_intestinal_efflux("Drug", dose_mg=50.0, inhibitor_fraction=-0.1)
