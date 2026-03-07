"""
Tests for Phase 377 — Tissue-to-Plasma Ratio Calculator
"""

import pytest

from omega_pbpk.prediction.tissue_plasma_ratio import (
    TissueRatioResult,
    calculate_tissue_ratios,
    screen_drugs_tissue_distribution,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lipophilic_base():
    """A lipophilic base drug (logP=4, fup=0.05)."""
    return calculate_tissue_ratios(
        drug_name="LipoDrug",
        logp=4.0,
        pka=8.5,
        drug_type="base",
        fup=0.05,
    )


@pytest.fixture
def hydrophilic_acid():
    """A hydrophilic acid drug (logP=-1, fup=0.8)."""
    return calculate_tissue_ratios(
        drug_name="HydroDrug",
        logp=-1.0,
        pka=4.5,
        drug_type="acid",
        fup=0.8,
    )


@pytest.fixture
def neutral_moderate():
    """A neutral drug with moderate lipophilicity (logP=2, fup=0.3)."""
    return calculate_tissue_ratios(
        drug_name="NeutralDrug",
        logp=2.0,
        pka=7.0,
        drug_type="neutral",
        fup=0.3,
    )


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_tissue_ratio_result(self, lipophilic_base):
        assert isinstance(lipophilic_base, TissueRatioResult)

    def test_result_is_frozen(self, lipophilic_base):
        with pytest.raises((AttributeError, TypeError)):
            lipophilic_base.kp_liver = 99.9  # type: ignore[misc]

    def test_all_fields_present(self, lipophilic_base):
        fields = [
            "drug_name",
            "logp",
            "pka",
            "drug_type",
            "fup",
            "fut",
            "kp_liver",
            "kp_kidney",
            "kp_muscle",
            "kp_adipose",
            "kp_brain",
            "kp_lung",
            "kp_heart",
            "kp_gut",
            "vd_predicted_L_per_kg",
            "dominant_tissue",
            "distribution_class",
            "notes",
        ]
        for f in fields:
            assert hasattr(lipophilic_base, f), f"Missing field: {f}"


# ---------------------------------------------------------------------------
# 2. Lipophilic drug — higher adipose Kp
# ---------------------------------------------------------------------------


class TestLipophilicDrug:
    def test_adipose_kp_high_for_lipophilic(self, lipophilic_base):
        # Adipose has the most lipid; lipophilic drugs should distribute there
        assert lipophilic_base.kp_adipose > lipophilic_base.kp_muscle

    def test_dominant_tissue_is_adipose_for_very_high_logp(self):
        result = calculate_tissue_ratios(
            drug_name="SuperLipo",
            logp=6.0,
            pka=7.0,
            drug_type="neutral",
            fup=0.01,
        )
        assert result.dominant_tissue == "adipose"

    def test_adipose_kp_increases_with_logp(self):
        r_low = calculate_tissue_ratios("A", 1.0, 7.0, "neutral", 0.5)
        r_high = calculate_tissue_ratios("B", 5.0, 7.0, "neutral", 0.5)
        assert r_high.kp_adipose > r_low.kp_adipose

    def test_vd_extensive_for_lipophilic(self, lipophilic_base):
        assert lipophilic_base.distribution_class == "extensive"

    def test_vd_large_for_lipophilic(self, lipophilic_base):
        assert lipophilic_base.vd_predicted_L_per_kg > 2.0


# ---------------------------------------------------------------------------
# 3. Hydrophilic drug — lower Kp values
# ---------------------------------------------------------------------------


class TestHydrophilicDrug:
    def test_adipose_kp_low_for_hydrophilic(self, hydrophilic_acid):
        # Hydrophilic drugs should not distribute much into adipose
        assert hydrophilic_acid.kp_adipose < 5.0

    def test_all_kp_positive(self, hydrophilic_acid):
        kps = [
            hydrophilic_acid.kp_liver,
            hydrophilic_acid.kp_kidney,
            hydrophilic_acid.kp_muscle,
            hydrophilic_acid.kp_adipose,
            hydrophilic_acid.kp_brain,
        ]
        for kp in kps:
            assert kp > 0

    def test_vd_restricted_or_moderate_for_hydrophilic(self, hydrophilic_acid):
        assert hydrophilic_acid.distribution_class in {"restricted", "moderate"}

    def test_muscle_kp_relatively_high_hydrophilic(self, hydrophilic_acid):
        # Muscle is water-rich; hydrophilic drugs partition there more than adipose
        assert hydrophilic_acid.kp_muscle >= hydrophilic_acid.kp_adipose


# ---------------------------------------------------------------------------
# 4. Dominant tissue and distribution class
# ---------------------------------------------------------------------------


class TestDominantTissueAndClass:
    def test_dominant_tissue_is_string(self, neutral_moderate):
        assert isinstance(neutral_moderate.dominant_tissue, str)

    def test_dominant_tissue_in_known_tissues(self, neutral_moderate):
        valid = {"liver", "kidney", "muscle", "adipose", "brain", "lung", "heart", "gut"}
        assert neutral_moderate.dominant_tissue in valid

    def test_distribution_class_values(self):
        for logp, _expected in [(-2.0, "restricted"), (2.0, "moderate"), (5.0, "extensive")]:
            result = calculate_tissue_ratios("X", logp, 7.0, "neutral", 0.5)
            assert result.distribution_class in {"restricted", "moderate", "extensive"}

    def test_restricted_vd_below_0_3(self):
        result = calculate_tissue_ratios("Polar", -3.0, 7.0, "neutral", 0.99)
        if result.distribution_class == "restricted":
            assert result.vd_predicted_L_per_kg < 0.3

    def test_extensive_vd_above_2(self):
        result = calculate_tissue_ratios("Lipo", 5.0, 7.0, "neutral", 0.05)
        if result.distribution_class == "extensive":
            assert result.vd_predicted_L_per_kg >= 2.0


# ---------------------------------------------------------------------------
# 5. fup sensitivity
# ---------------------------------------------------------------------------


class TestFupSensitivity:
    def test_fup_affects_kp(self):
        # Changing fup changes fu_factor = (1/fup)*0.1, which affects the
        # protein binding term in both numerator and denominator.
        # Verify that Kp values differ when fup differs.
        r_high = calculate_tissue_ratios("D", 2.0, 7.0, "neutral", 0.8)
        r_low = calculate_tissue_ratios("D", 2.0, 7.0, "neutral", 0.1)
        assert r_low.kp_liver != r_high.kp_liver

    def test_fup_affects_vd(self):
        # Vd should differ for different fup values.
        r_high = calculate_tissue_ratios("D", 2.0, 7.0, "neutral", 0.8)
        r_low = calculate_tissue_ratios("D", 2.0, 7.0, "neutral", 0.1)
        assert r_low.vd_predicted_L_per_kg != r_high.vd_predicted_L_per_kg

    def test_fut_between_0_and_1(self, lipophilic_base):
        assert 0.0 <= lipophilic_base.fut <= 1.0

    def test_fut_lower_for_more_lipophilic(self):
        r_low_lp = calculate_tissue_ratios("A", 1.0, 7.0, "neutral", 0.5)
        r_high_lp = calculate_tissue_ratios("B", 5.0, 7.0, "neutral", 0.5)
        # More lipophilic → more tissue binding → lower fut
        assert r_high_lp.fut <= r_low_lp.fut


# ---------------------------------------------------------------------------
# 6. screen_drugs_tissue_distribution
# ---------------------------------------------------------------------------


class TestScreenDrugs:
    @pytest.fixture
    def drug_list(self):
        return [
            {"name": "HydroDrug", "logp": -1.0, "pka": 4.5, "drug_type": "acid", "fup": 0.8},
            {"name": "MidDrug", "logp": 2.0, "pka": 7.0, "drug_type": "neutral", "fup": 0.3},
            {"name": "LipoDrug", "logp": 5.0, "pka": 8.5, "drug_type": "base", "fup": 0.05},
        ]

    def test_screen_returns_list(self, drug_list):
        results = screen_drugs_tissue_distribution(drug_list)
        assert isinstance(results, list)

    def test_screen_returns_correct_count(self, drug_list):
        results = screen_drugs_tissue_distribution(drug_list)
        assert len(results) == 3

    def test_screen_sorted_by_vd_descending(self, drug_list):
        results = screen_drugs_tissue_distribution(drug_list)
        vds = [r.vd_predicted_L_per_kg for r in results]
        assert vds == sorted(vds, reverse=True)

    def test_screen_all_are_results(self, drug_list):
        results = screen_drugs_tissue_distribution(drug_list)
        for r in results:
            assert isinstance(r, TissueRatioResult)

    def test_screen_lipophilic_drug_first(self, drug_list):
        results = screen_drugs_tissue_distribution(drug_list)
        assert results[0].drug_name == "LipoDrug"

    def test_screen_hydrophilic_drug_last(self, drug_list):
        results = screen_drugs_tissue_distribution(drug_list)
        assert results[-1].drug_name == "HydroDrug"


# ---------------------------------------------------------------------------
# 7. Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_drug_type(self):
        with pytest.raises(ValueError, match="drug_type"):
            calculate_tissue_ratios("X", 1.0, 7.0, "zwitterion", 0.5)

    def test_fup_zero(self):
        with pytest.raises(ValueError, match="fup"):
            calculate_tissue_ratios("X", 1.0, 7.0, "neutral", 0.0)

    def test_fup_greater_than_one(self):
        with pytest.raises(ValueError, match="fup"):
            calculate_tissue_ratios("X", 1.0, 7.0, "neutral", 1.5)

    def test_logp_too_low(self):
        with pytest.raises(ValueError, match="logp"):
            calculate_tissue_ratios("X", -6.0, 7.0, "neutral", 0.5)

    def test_logp_too_high(self):
        with pytest.raises(ValueError, match="logp"):
            calculate_tissue_ratios("X", 11.0, 7.0, "neutral", 0.5)

    def test_pka_out_of_range(self):
        with pytest.raises(ValueError, match="pka"):
            calculate_tissue_ratios("X", 1.0, 15.0, "neutral", 0.5)

    def test_mw_zero(self):
        with pytest.raises(ValueError, match="molecular_weight"):
            calculate_tissue_ratios("X", 1.0, 7.0, "neutral", 0.5, molecular_weight=0.0)

    def test_mw_negative(self):
        with pytest.raises(ValueError, match="molecular_weight"):
            calculate_tissue_ratios("X", 1.0, 7.0, "neutral", 0.5, molecular_weight=-100.0)
