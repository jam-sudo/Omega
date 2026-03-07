"""Tests for Phase 572 — salt_form_prediction.py"""

import pytest

from omega_pbpk.prediction.salt_form_prediction import (
    common_ion_effect,
    salt_solubility_enhancement,
    salt_stability_flag,
    select_optimal_salt,
)

# ---------------------------------------------------------------------------
# salt_solubility_enhancement
# ---------------------------------------------------------------------------


class TestSaltSolubilityEnhancement:
    def test_acid_high_ph_enhancement(self):
        """Acid drug at pH >> pKa should show large solubility enhancement."""
        r = salt_solubility_enhancement(
            pka_drug=4.0,
            pka_counterion=14.0,
            intrinsic_solubility_mg_L=10.0,
            ph=7.4,
            drug_type="acid",
        )
        assert r["enhancement_fold"] > 100

    def test_s_salt_greater_than_intrinsic_when_ionized(self):
        """S_salt should exceed intrinsic solubility when drug is ionised."""
        r = salt_solubility_enhancement(
            pka_drug=5.0,
            pka_counterion=14.0,
            intrinsic_solubility_mg_L=50.0,
            ph=7.4,
            drug_type="acid",
        )
        assert r["s_salt_mg_L"] > r["s_intrinsic_mg_L"]

    def test_base_drug_enhancement(self):
        """Base drug at pH < pKa should show enhancement."""
        r = salt_solubility_enhancement(
            pka_drug=9.0,
            pka_counterion=-7.0,
            intrinsic_solubility_mg_L=20.0,
            ph=7.4,
            drug_type="base",
        )
        assert r["enhancement_fold"] > 1.0

    def test_enhancement_fold_ge_one(self):
        """Enhancement fold must always be >= 1."""
        for ph in [5.0, 7.4, 9.0]:
            r = salt_solubility_enhancement(4.0, 14.0, 10.0, ph=ph, drug_type="acid")
            assert r["enhancement_fold"] >= 1.0

    def test_returns_recommendation_string(self):
        r = salt_solubility_enhancement(4.0, 14.0, 10.0, ph=7.4, drug_type="acid")
        assert isinstance(r["recommendation"], str) and len(r["recommendation"]) > 0

    def test_invalid_solubility(self):
        with pytest.raises(ValueError):
            salt_solubility_enhancement(4.0, 14.0, -10.0, ph=7.4, drug_type="acid")

    def test_invalid_drug_type(self):
        with pytest.raises(ValueError):
            salt_solubility_enhancement(4.0, 14.0, 10.0, ph=7.4, drug_type="neutral")

    def test_acid_at_low_ph_minimal_enhancement(self):
        """Acid drug at pH << pKa: ionisation is low → small enhancement."""
        r = salt_solubility_enhancement(
            pka_drug=8.0,
            pka_counterion=14.0,
            intrinsic_solubility_mg_L=100.0,
            ph=4.0,
            drug_type="acid",
        )
        # pH=4, pKa=8: 10^(4-8) = 0.0001 → S_salt ~ 100.01; enhancement ≈ 1.0001
        assert r["enhancement_fold"] < 2.0


# ---------------------------------------------------------------------------
# select_optimal_salt
# ---------------------------------------------------------------------------


class TestSelectOptimalSalt:
    def test_acid_returns_four_counterions(self):
        results = select_optimal_salt(
            "Drug_A", pka=4.0, drug_type="acid", intrinsic_solubility_mg_L=10.0
        )
        assert len(results) == 4

    def test_base_returns_four_counterions(self):
        results = select_optimal_salt(
            "Drug_B", pka=9.0, drug_type="base", intrinsic_solubility_mg_L=10.0
        )
        assert len(results) == 4

    def test_sorted_by_s_salt_descending(self):
        results = select_optimal_salt(
            "Drug_A", pka=4.0, drug_type="acid", intrinsic_solubility_mg_L=10.0
        )
        s_values = [r["s_salt_mg_L"] for r in results]
        assert s_values == sorted(s_values, reverse=True)

    def test_contains_drug_name(self):
        results = select_optimal_salt(
            "MyDrug", pka=4.0, drug_type="acid", intrinsic_solubility_mg_L=10.0
        )
        assert all(r["drug_name"] == "MyDrug" for r in results)

    def test_each_has_stability_flag(self):
        results = select_optimal_salt(
            "D", pka=5.0, drug_type="acid", intrinsic_solubility_mg_L=20.0
        )
        for r in results:
            assert "stability" in r
            assert isinstance(r["stability"], str)

    def test_invalid_drug_type(self):
        with pytest.raises(ValueError):
            select_optimal_salt(
                "D", pka=5.0, drug_type="zwitterion", intrinsic_solubility_mg_L=10.0
            )


# ---------------------------------------------------------------------------
# common_ion_effect
# ---------------------------------------------------------------------------


class TestCommonIonEffect:
    def test_high_counterion_reduces_solubility(self):
        r = common_ion_effect(salt_solubility_mg_L=1000.0, counterion_conc_mM=500.0)
        assert r["s_effective_mg_L"] < 1000.0

    def test_zero_counterion_no_reduction(self):
        r = common_ion_effect(salt_solubility_mg_L=500.0, counterion_conc_mM=0.0)
        assert abs(r["s_effective_mg_L"] - 500.0) < 1e-9

    def test_reduction_factor_increases_with_conc(self):
        low = common_ion_effect(500.0, 50.0)
        high = common_ion_effect(500.0, 200.0)
        assert high["reduction_factor"] > low["reduction_factor"]

    def test_returns_notes(self):
        r = common_ion_effect(500.0, 100.0)
        assert isinstance(r["notes"], str) and len(r["notes"]) > 0

    def test_invalid_solubility(self):
        with pytest.raises(ValueError):
            common_ion_effect(-100.0, 50.0)

    def test_invalid_conc(self):
        with pytest.raises(ValueError):
            common_ion_effect(500.0, -10.0)

    def test_s_effective_positive(self):
        r = common_ion_effect(100.0, 900.0)
        assert r["s_effective_mg_L"] > 0.0


# ---------------------------------------------------------------------------
# salt_stability_flag
# ---------------------------------------------------------------------------


class TestSaltStabilityFlag:
    def test_stable_large_delta_pka(self):
        result = salt_stability_flag(pka_drug=4.0, pka_counterion=14.0, drug_type="acid")
        assert result == "stable"

    def test_moderate_delta_pka(self):
        result = salt_stability_flag(pka_drug=4.0, pka_counterion=5.5, drug_type="acid")
        assert result == "moderate stability"

    def test_unstable_small_delta_pka(self):
        result = salt_stability_flag(pka_drug=4.0, pka_counterion=4.3, drug_type="acid")
        assert "unstable" in result

    def test_stable_for_hcl_salt_of_base(self):
        # Base pKa=9, HCl pKa=-7 → delta=16 → stable
        result = salt_stability_flag(pka_drug=9.0, pka_counterion=-7.0, drug_type="base")
        assert result == "stable"

    def test_symmetry(self):
        """Swapping pKa values should give the same result."""
        r1 = salt_stability_flag(pka_drug=4.0, pka_counterion=14.0, drug_type="acid")
        r2 = salt_stability_flag(pka_drug=14.0, pka_counterion=4.0, drug_type="acid")
        assert r1 == r2

    def test_invalid_drug_type(self):
        with pytest.raises(ValueError):
            salt_stability_flag(4.0, 14.0, "amphoteric")

    def test_exactly_three_is_stable(self):
        result = salt_stability_flag(pka_drug=4.0, pka_counterion=7.0, drug_type="acid")
        assert result == "stable"

    def test_exactly_one_is_moderate(self):
        result = salt_stability_flag(pka_drug=4.0, pka_counterion=5.0, drug_type="acid")
        assert result == "moderate stability"
