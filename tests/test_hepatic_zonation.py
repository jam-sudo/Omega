"""Tests for Phase 349 — Hepatic Zonal Drug Metabolism.

Coverage:
  - Equal CLint zones produce equal f_metabolized fractions
  - Higher Zone 3 CLint raises f_metabolized_zone3 above zone1
  - f_metabolized fractions sum to 1.0
  - reactive_metabolite flag controls reactive_metabolite_zone3_fraction
  - extraction_ratio in [0, 1]
  - zone3_burden_ratio definition
  - Hepatotoxicity risk classification
  - compare_enzyme_inducers: length and sort order
  - Frozen dataclass field access
  - notes contains drug_name
  - Input validation
"""

from __future__ import annotations

import math

import pytest

from omega_pbpk.core.hepatic_zonation import (
    HepaticZonationResult,
    compare_enzyme_inducers,
    simulate_hepatic_zonation,
)

# ---------------------------------------------------------------------------
# Helper constants
# ---------------------------------------------------------------------------
_DRUG = "TestDrug"
_DOSE = 100.0


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHepaticZonationBasic:
    def test_returns_result_type(self):
        result = simulate_hepatic_zonation(_DRUG, _DOSE)
        assert isinstance(result, HepaticZonationResult)

    def test_frozen_dataclass(self):
        result = simulate_hepatic_zonation(_DRUG, _DOSE)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_stored(self):
        result = simulate_hepatic_zonation("Midazolam", _DOSE)
        assert result.drug_name == "Midazolam"

    def test_dose_stored(self):
        result = simulate_hepatic_zonation(_DRUG, 50.0)
        assert result.dose_mg == 50.0

    def test_notes_contains_drug_name(self):
        name = "Acetaminophen"
        result = simulate_hepatic_zonation(name, _DOSE)
        assert name in result.notes


# ---------------------------------------------------------------------------
# Fraction metabolized tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFractionMetabolized:
    def test_fractions_sum_to_one(self):
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE, clint_zone1_mL_per_min_per_g=1.0, clint_zone3_mL_per_min_per_g=2.0
        )
        total = result.f_metabolized_zone1 + result.f_metabolized_zone2 + result.f_metabolized_zone3
        assert abs(total - 1.0) < 1e-9

    def test_equal_clint_equal_fractions(self):
        """When zone1 == zone3 CLint, fractions should reflect zone mass distribution."""
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=1.0,
            clint_zone3_mL_per_min_per_g=1.0,
        )
        # With equal clint per gram: CLint_zone1 = clint * mass_z1; CLint_zone3 = clint * mass_z3
        # So f_zone3 / f_zone1 = mass_z3 / mass_z1 = 0.45 / 0.35 ≈ 1.286
        ratio = result.f_metabolized_zone3 / result.f_metabolized_zone1
        assert abs(ratio - (0.45 / 0.35)) < 0.01

    def test_higher_zone3_clint_higher_zone3_fraction(self):
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=0.5,
            clint_zone3_mL_per_min_per_g=4.0,
        )
        assert result.f_metabolized_zone3 > result.f_metabolized_zone1

    def test_higher_zone1_clint_higher_zone1_fraction(self):
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=5.0,
            clint_zone3_mL_per_min_per_g=0.1,
        )
        # Zone3 has more mass (45%) but much lower CLint; zone1 fraction should be dominant
        # zone1 CLint_total = 5.0 * 525 = 2625, zone3 = 0.1 * 675 = 67.5
        assert result.f_metabolized_zone1 > result.f_metabolized_zone3

    def test_all_fractions_non_negative(self):
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=0.1,
            clint_zone3_mL_per_min_per_g=3.0,
        )
        assert result.f_metabolized_zone1 >= 0
        assert result.f_metabolized_zone2 >= 0
        assert result.f_metabolized_zone3 >= 0

    def test_zero_clint_distributes_uniformly(self):
        """Zero CLint: no metabolism, fractions fall back to zone mass fractions."""
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=0.0,
            clint_zone3_mL_per_min_per_g=0.0,
        )
        total = result.f_metabolized_zone1 + result.f_metabolized_zone2 + result.f_metabolized_zone3
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# PK metric tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPKMetrics:
    def test_extraction_ratio_in_bounds(self):
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=5.0,
            clint_zone3_mL_per_min_per_g=10.0,
        )
        assert 0.0 <= result.extraction_ratio <= 1.0

    def test_cl_total_positive(self):
        result = simulate_hepatic_zonation(_DRUG, _DOSE)
        assert result.cl_total_L_per_h >= 0.0

    def test_cl_intrinsic_zone1_positive(self):
        result = simulate_hepatic_zonation(_DRUG, _DOSE)
        assert result.cl_intrinsic_zone1_L_per_h >= 0.0

    def test_cl_intrinsic_zone3_positive(self):
        result = simulate_hepatic_zonation(_DRUG, _DOSE)
        assert result.cl_intrinsic_zone3_L_per_h >= 0.0

    def test_zone3_burden_ratio_definition(self):
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=1.0,
            clint_zone3_mL_per_min_per_g=2.0,
        )
        expected = result.f_metabolized_zone3 / result.f_metabolized_zone1
        assert abs(result.zone3_burden_ratio - expected) < 1e-9

    def test_higher_clint_gives_higher_er(self):
        low = simulate_hepatic_zonation(
            _DRUG, _DOSE, clint_zone1_mL_per_min_per_g=0.1, clint_zone3_mL_per_min_per_g=0.1
        )
        high = simulate_hepatic_zonation(
            _DRUG, _DOSE, clint_zone1_mL_per_min_per_g=5.0, clint_zone3_mL_per_min_per_g=10.0
        )
        assert high.extraction_ratio > low.extraction_ratio


# ---------------------------------------------------------------------------
# Reactive metabolite tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReactiveMetabolite:
    def test_reactive_metabolite_true(self):
        result = simulate_hepatic_zonation(_DRUG, _DOSE, reactive_metabolite=True)
        assert result.reactive_metabolite_zone3_fraction == 0.7

    def test_reactive_metabolite_false(self):
        result = simulate_hepatic_zonation(_DRUG, _DOSE, reactive_metabolite=False)
        assert result.reactive_metabolite_zone3_fraction == 0.3


# ---------------------------------------------------------------------------
# Hepatotoxicity risk classification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHepatotoxicityRisk:
    def test_low_risk_default(self):
        """Default low CLint ratio → low risk."""
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=1.0,
            clint_zone3_mL_per_min_per_g=1.0,
            reactive_metabolite=False,
        )
        assert result.hepatotoxicity_risk == "low"

    def test_high_risk_high_burden_and_reactive(self):
        """zone3_burden_ratio > 3 AND reactive_metabolite → high."""
        # Very high zone3 CLint will push burden ratio > 3
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=0.1,
            clint_zone3_mL_per_min_per_g=5.0,
            reactive_metabolite=True,
        )
        assert result.hepatotoxicity_risk == "high"

    def test_moderate_risk_high_burden_no_reactive(self):
        """zone3_burden_ratio > 2 WITHOUT reactive metabolite → moderate."""
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=0.1,
            clint_zone3_mL_per_min_per_g=3.0,
            reactive_metabolite=False,
        )
        assert result.hepatotoxicity_risk in ("moderate", "high")

    def test_moderate_risk_reactive_low_burden(self):
        """reactive_metabolite AND zone3_burden_ratio > 1 → at least moderate."""
        result = simulate_hepatic_zonation(
            _DRUG, _DOSE,
            clint_zone1_mL_per_min_per_g=1.0,
            clint_zone3_mL_per_min_per_g=2.0,
            reactive_metabolite=True,
        )
        assert result.hepatotoxicity_risk in ("moderate", "high")

    def test_risk_in_valid_set(self):
        for clint3 in [0.5, 2.0, 5.0, 10.0]:
            result = simulate_hepatic_zonation(
                _DRUG, _DOSE,
                clint_zone1_mL_per_min_per_g=0.5,
                clint_zone3_mL_per_min_per_g=clint3,
                reactive_metabolite=True,
            )
            assert result.hepatotoxicity_risk in ("low", "moderate", "high")


# ---------------------------------------------------------------------------
# compare_enzyme_inducers tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompareEnzymeInducers:
    def test_returns_correct_length(self):
        fold_vals = [1, 2, 5, 10]
        results = compare_enzyme_inducers(
            _DRUG, _DOSE, fu_plasma=0.5, clint_z1_baseline=1.0,
            fold_induction_z3_values=fold_vals,
        )
        assert len(results) == len(fold_vals)

    def test_returns_default_length(self):
        results = compare_enzyme_inducers(
            _DRUG, _DOSE, fu_plasma=0.5, clint_z1_baseline=1.0
        )
        assert len(results) == 4  # default [1, 2, 5, 10]

    def test_sorted_descending(self):
        results = compare_enzyme_inducers(
            _DRUG, _DOSE, fu_plasma=0.5, clint_z1_baseline=1.0,
            fold_induction_z3_values=[1, 2, 5, 10],
        )
        ratios = [r.zone3_burden_ratio for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_higher_fold_induction_higher_burden(self):
        results = compare_enzyme_inducers(
            _DRUG, _DOSE, fu_plasma=0.5, clint_z1_baseline=1.0,
            fold_induction_z3_values=[1, 10],
        )
        # After sorting descending, first result should have higher burden
        assert results[0].zone3_burden_ratio >= results[1].zone3_burden_ratio

    def test_all_results_are_hepatic_zonation_result(self):
        results = compare_enzyme_inducers(
            _DRUG, _DOSE, fu_plasma=0.5, clint_z1_baseline=0.5,
        )
        assert all(isinstance(r, HepaticZonationResult) for r in results)

    def test_single_fold_value(self):
        results = compare_enzyme_inducers(
            _DRUG, _DOSE, fu_plasma=0.5, clint_z1_baseline=1.0,
            fold_induction_z3_values=[3],
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInputValidation:
    def test_invalid_blood_flow(self):
        with pytest.raises(ValueError, match="hepatic_blood_flow"):
            simulate_hepatic_zonation(_DRUG, _DOSE, hepatic_blood_flow_L_per_h=0.0)

    def test_invalid_negative_clint_zone1(self):
        with pytest.raises(ValueError, match="clint_zone1"):
            simulate_hepatic_zonation(_DRUG, _DOSE, clint_zone1_mL_per_min_per_g=-1.0)

    def test_invalid_negative_clint_zone3(self):
        with pytest.raises(ValueError, match="clint_zone3"):
            simulate_hepatic_zonation(_DRUG, _DOSE, clint_zone3_mL_per_min_per_g=-0.1)

    def test_invalid_liver_mass(self):
        with pytest.raises(ValueError, match="liver_mass_g"):
            simulate_hepatic_zonation(_DRUG, _DOSE, liver_mass_g=0.0)

    def test_invalid_fu_plasma_zero(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_hepatic_zonation(_DRUG, _DOSE, fu_plasma=0.0)

    def test_invalid_fu_plasma_over_one(self):
        with pytest.raises(ValueError, match="fu_plasma"):
            simulate_hepatic_zonation(_DRUG, _DOSE, fu_plasma=1.5)
