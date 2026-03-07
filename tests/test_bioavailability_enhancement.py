"""Tests for biopharmaceutics/bioavailability_enhancement.py — Phase 319."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.biopharmaceutics.bioavailability_enhancement import (
    BioavailEnhancementResult,
    amorphous_enhancement,
    compare_enhancement_strategies_ba,
    lipid_formulation_enhancement,
    nanosizing_enhancement,
)


# ---------------------------------------------------------------------------
# nanosizing_enhancement
# ---------------------------------------------------------------------------

class TestNanosingEnhancement:
    def test_basic_fold_increase(self):
        # 100 µm → 200 nm: (100*1000/200)^0.3 = 500^0.3
        expected = 500 ** 0.3
        result = nanosizing_enhancement(0.01, 100.0)
        assert abs(result - expected) < 1e-6

    def test_fold_clamped_at_10(self):
        # Very large particle → large ratio, should clamp to 10
        result = nanosizing_enhancement(0.01, 10_000.0, target_size_nm=1.0)
        assert result == pytest.approx(10.0)

    def test_fold_clamped_at_1_when_no_improvement(self):
        # Particle already smaller than target → fold < 1 → clamp to 1.0
        result = nanosizing_enhancement(0.01, 0.0001, target_size_nm=200.0)
        assert result == pytest.approx(1.0)

    def test_same_size_gives_fold_1(self):
        # particle_size_um * 1000 == target_size_nm → ratio=1 → fold=1.0
        result = nanosizing_enhancement(0.01, 0.2, target_size_nm=200.0)
        assert result == pytest.approx(1.0)

    def test_custom_target_size(self):
        # 50 µm → 100 nm: (50*1000/100)^0.3 = 500^0.3
        result = nanosizing_enhancement(0.05, 50.0, target_size_nm=100.0)
        expected = min(max((50_000 / 100) ** 0.3, 1.0), 10.0)
        assert abs(result - expected) < 1e-6

    def test_invalid_solubility(self):
        with pytest.raises(ValueError, match="solubility_bulk_mg_mL"):
            nanosizing_enhancement(0.0, 100.0)

    def test_invalid_particle_size(self):
        with pytest.raises(ValueError, match="particle_size_um"):
            nanosizing_enhancement(0.01, -5.0)

    def test_invalid_target_size(self):
        with pytest.raises(ValueError, match="target_size_nm"):
            nanosizing_enhancement(0.01, 100.0, target_size_nm=0.0)

    def test_drug_density_ignored_but_validated(self):
        # drug_density_g_mL parameter accepted; must be > 0
        result = nanosizing_enhancement(0.01, 100.0, drug_density_g_mL=1.5)
        assert result > 1.0

    def test_drug_density_zero_raises(self):
        with pytest.raises(ValueError, match="drug_density_g_mL"):
            nanosizing_enhancement(0.01, 100.0, drug_density_g_mL=0.0)


# ---------------------------------------------------------------------------
# amorphous_enhancement
# ---------------------------------------------------------------------------

class TestAmorphousEnhancement:
    def test_basic_fold(self):
        # fold = exp(0.005 * (180 - 80)) = exp(0.5)
        result = amorphous_enhancement(tm_C=180.0, tg_C=80.0)
        assert abs(result - math.exp(0.5)) < 1e-6

    def test_small_delta_gives_near_1(self):
        # tm-tg=2 → exp(0.01) ≈ 1.01
        result = amorphous_enhancement(tm_C=82.0, tg_C=80.0)
        assert result == pytest.approx(math.exp(0.01), rel=1e-6)

    def test_large_delta_clamped_at_50(self):
        # tm-tg = 2000 → exp(10) >> 50 → clamp
        result = amorphous_enhancement(tm_C=2010.0, tg_C=10.0)
        assert result == pytest.approx(50.0)

    def test_fold_lower_clamped_at_1(self):
        # Even tiny delta → exp(0.005 * tiny) ≥ 1
        result = amorphous_enhancement(tm_C=80.1, tg_C=80.0)
        assert result >= 1.0

    def test_invalid_tg_zero(self):
        with pytest.raises(ValueError, match="tg_C"):
            amorphous_enhancement(tm_C=180.0, tg_C=0.0)

    def test_invalid_tm_le_tg(self):
        with pytest.raises(ValueError, match="tm_C"):
            amorphous_enhancement(tm_C=80.0, tg_C=80.0)

    def test_invalid_tm_less_than_tg(self):
        with pytest.raises(ValueError, match="tm_C"):
            amorphous_enhancement(tm_C=50.0, tg_C=80.0)

    def test_temperature_c_accepted(self):
        # temperature_C is a parameter; does not affect result in current model
        r1 = amorphous_enhancement(tm_C=180.0, tg_C=80.0, temperature_C=25.0)
        r2 = amorphous_enhancement(tm_C=180.0, tg_C=80.0, temperature_C=37.0)
        assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# lipid_formulation_enhancement
# ---------------------------------------------------------------------------

class TestLipidFormulationEnhancement:
    def test_sedds_default(self):
        logP = 3.0
        meal_fat_g = 20.0
        ff = 2.0  # SEDDS
        expected = math.exp(0.3 * logP) * (meal_fat_g / 10.0) ** 0.5 * ff
        result = lipid_formulation_enhancement(logP=logP, meal_fat_g=meal_fat_g, formulation_type="SEDDS")
        assert abs(result - min(max(expected, 1.0), 20.0)) < 1e-6

    def test_smedds_higher_than_sedds(self):
        r_sedds = lipid_formulation_enhancement(logP=3.0, formulation_type="SEDDS")
        r_smedds = lipid_formulation_enhancement(logP=3.0, formulation_type="SMEDDS")
        assert r_smedds > r_sedds

    def test_snedds_highest(self):
        r_smedds = lipid_formulation_enhancement(logP=3.0, formulation_type="SMEDDS")
        r_snedds = lipid_formulation_enhancement(logP=3.0, formulation_type="SNEDDS")
        assert r_snedds >= r_smedds

    def test_solution_formulation(self):
        result = lipid_formulation_enhancement(logP=2.0, formulation_type="solution")
        assert result >= 1.0

    def test_high_logP_clamped_at_20(self):
        result = lipid_formulation_enhancement(logP=10.0, meal_fat_g=40.0, formulation_type="SNEDDS")
        assert result == pytest.approx(20.0)

    def test_zero_meal_fat_gives_fold_1(self):
        # meal_fat_g=0 → fat_factor=0 → fold=0 → clamp to 1.0
        result = lipid_formulation_enhancement(logP=3.0, meal_fat_g=0.0)
        assert result == pytest.approx(1.0)

    def test_low_logP_small_fold(self):
        # logP=0 → exp(0)=1; small fat → small fold; may stay near 1
        result = lipid_formulation_enhancement(logP=0.0, meal_fat_g=10.0, formulation_type="SEDDS")
        assert result >= 1.0

    def test_invalid_formulation_type(self):
        with pytest.raises(ValueError, match="formulation_type"):
            lipid_formulation_enhancement(logP=3.0, formulation_type="UNKNOWN")

    def test_negative_meal_fat_raises(self):
        with pytest.raises(ValueError, match="meal_fat_g"):
            lipid_formulation_enhancement(logP=3.0, meal_fat_g=-1.0)

    def test_case_insensitive_formulation_type(self):
        # formulation_type is uppercased internally
        r1 = lipid_formulation_enhancement(logP=3.0, formulation_type="sedds")
        r2 = lipid_formulation_enhancement(logP=3.0, formulation_type="SEDDS")
        assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# compare_enhancement_strategies_ba
# ---------------------------------------------------------------------------

class TestCompareEnhancementStrategies:
    def _default_result(self) -> BioavailEnhancementResult:
        return compare_enhancement_strategies_ba(
            drug_name="TestDrug",
            bcs_class=2,
            logP=3.5,
            solubility_bulk_mg_mL=0.02,
            tm_C=180.0,
            tg_C=80.0,
        )

    def test_returns_dataclass(self):
        result = self._default_result()
        assert isinstance(result, BioavailEnhancementResult)

    def test_frozen(self):
        result = self._default_result()
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "Other"  # type: ignore[misc]

    def test_bcs_class_2_base_f(self):
        result = self._default_result()
        assert result.base_f_estimated == pytest.approx(0.20)

    def test_bcs_class_1_base_f(self):
        result = compare_enhancement_strategies_ba(
            "DrugA", bcs_class=1, logP=1.0, solubility_bulk_mg_mL=1.0, tm_C=150.0, tg_C=60.0
        )
        assert result.base_f_estimated == pytest.approx(0.90)

    def test_bcs_class_3_base_f(self):
        result = compare_enhancement_strategies_ba(
            "DrugB", bcs_class=3, logP=1.0, solubility_bulk_mg_mL=1.0, tm_C=150.0, tg_C=60.0
        )
        assert result.base_f_estimated == pytest.approx(0.50)

    def test_bcs_class_4_base_f(self):
        result = compare_enhancement_strategies_ba(
            "DrugC", bcs_class=4, logP=4.0, solubility_bulk_mg_mL=0.01, tm_C=200.0, tg_C=90.0
        )
        assert result.base_f_estimated == pytest.approx(0.10)

    def test_predicted_f_clamped_to_1(self):
        # BCS I + high folds → predicted F should not exceed 1.0
        result = compare_enhancement_strategies_ba(
            "DrugD", bcs_class=1, logP=5.0, solubility_bulk_mg_mL=0.001, tm_C=300.0, tg_C=100.0
        )
        for val in (result.predicted_f_nanosized, result.predicted_f_asd,
                    result.predicted_f_sedds, result.predicted_f_smedds):
            assert val <= 1.0 + 1e-9

    def test_predicted_f_geq_base_f(self):
        result = self._default_result()
        for val in (result.predicted_f_nanosized, result.predicted_f_asd,
                    result.predicted_f_sedds, result.predicted_f_smedds):
            assert val >= result.base_f_estimated - 1e-9

    def test_recommended_strategy_is_best(self):
        result = self._default_result()
        scores = {
            "nanosizing": result.predicted_f_nanosized,
            "amorphous_solid_dispersion": result.predicted_f_asd,
            "SEDDS": result.predicted_f_sedds,
            "SMEDDS": result.predicted_f_smedds,
        }
        best = max(scores, key=lambda k: scores[k])
        assert result.recommended_strategy == best

    def test_notes_not_empty(self):
        result = self._default_result()
        assert len(result.notes) > 10

    def test_invalid_bcs_class(self):
        with pytest.raises(ValueError, match="bcs_class"):
            compare_enhancement_strategies_ba(
                "X", bcs_class=5, logP=2.0, solubility_bulk_mg_mL=0.1, tm_C=150.0, tg_C=80.0
            )

    def test_invalid_solubility(self):
        with pytest.raises(ValueError, match="solubility_bulk_mg_mL"):
            compare_enhancement_strategies_ba(
                "X", bcs_class=2, logP=2.0, solubility_bulk_mg_mL=0.0, tm_C=150.0, tg_C=80.0
            )

    def test_invalid_tg_zero(self):
        with pytest.raises(ValueError, match="tg_C"):
            compare_enhancement_strategies_ba(
                "X", bcs_class=2, logP=2.0, solubility_bulk_mg_mL=0.1, tm_C=150.0, tg_C=0.0
            )

    def test_invalid_tm_le_tg(self):
        with pytest.raises(ValueError, match="tm_C"):
            compare_enhancement_strategies_ba(
                "X", bcs_class=2, logP=2.0, solubility_bulk_mg_mL=0.1, tm_C=80.0, tg_C=80.0
            )

    def test_logp_field_preserved(self):
        result = compare_enhancement_strategies_ba(
            "DrugE", bcs_class=2, logP=2.5, solubility_bulk_mg_mL=0.05, tm_C=160.0, tg_C=70.0
        )
        assert result.logP == pytest.approx(2.5)

    def test_drug_name_preserved(self):
        result = compare_enhancement_strategies_ba(
            "Ibuprofen", bcs_class=2, logP=3.97, solubility_bulk_mg_mL=0.021, tm_C=76.0, tg_C=40.0
        )
        assert result.drug_name == "Ibuprofen"
