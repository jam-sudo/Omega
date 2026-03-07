"""Tests for Phase 310 — Drug Stability in Biological Matrices."""

import pytest

from omega_pbpk.prediction.matrix_stability_predictor import (
    MatrixStabilityResult,
    assess_freeze_thaw_stability,
    predict_matrix_stability,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

BASE_PARAMS = dict(
    drug_name="DrugA",
    matrix_type="plasma",
    temperature_C=4.0,
    duration_h=24.0,
    logp=2.0,
    pka=7.0,
    ionization_type="neutral",
    is_ester=False,
    is_amide=False,
    has_oxidizable_group=False,
)

VALID_CLASSES = {"stable", "acceptable", "unstable"}
VALID_STORAGES = {"4C", "-20C", "-80C"}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_predict_returns_correct_type(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert isinstance(result, MatrixStabilityResult)

    def test_freeze_thaw_returns_correct_type(self):
        result = assess_freeze_thaw_stability(drug_name="DrugA", n_cycles=3)
        assert isinstance(result, MatrixStabilityResult)


# ---------------------------------------------------------------------------
# Field preservation
# ---------------------------------------------------------------------------


class TestFieldPreservation:
    def test_drug_name_preserved(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert result.drug_name == "DrugA"

    def test_matrix_type_preserved(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert result.matrix_type == "plasma"

    def test_temperature_preserved(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert result.temperature_C == 4.0

    def test_duration_preserved(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert result.duration_h == 24.0


# ---------------------------------------------------------------------------
# stability_pct in (0, 100]
# ---------------------------------------------------------------------------


class TestStabilityRange:
    def test_stability_pct_in_range_cold_stable(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert 0.0 < result.stability_pct <= 100.0

    def test_stability_pct_in_range_warm(self):
        result = predict_matrix_stability(
            **{**BASE_PARAMS, "temperature_C": 37.0, "duration_h": 72.0}
        )
        assert 0.0 < result.stability_pct <= 100.0

    def test_analyte_recovery_equals_stability(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert result.analyte_recovery_pct == pytest.approx(result.stability_pct)


# ---------------------------------------------------------------------------
# Higher temperature → lower stability_pct
# ---------------------------------------------------------------------------


class TestTemperatureEffect:
    def test_higher_temp_lower_stability(self):
        cold = predict_matrix_stability(**{**BASE_PARAMS, "temperature_C": -20.0})
        warm = predict_matrix_stability(**{**BASE_PARAMS, "temperature_C": 37.0})
        assert warm.stability_pct < cold.stability_pct

    def test_4c_more_stable_than_37c(self):
        at_4 = predict_matrix_stability(**BASE_PARAMS)
        at_37 = predict_matrix_stability(**{**BASE_PARAMS, "temperature_C": 37.0})
        assert at_4.stability_pct > at_37.stability_pct


# ---------------------------------------------------------------------------
# Ester → lower stability
# ---------------------------------------------------------------------------


class TestEsterEffect:
    def test_ester_lower_stability_than_non_ester(self):
        no_ester = predict_matrix_stability(**BASE_PARAMS)
        ester = predict_matrix_stability(**{**BASE_PARAMS, "is_ester": True})
        assert ester.stability_pct < no_ester.stability_pct

    def test_ester_raises_k_deg(self):
        no_ester = predict_matrix_stability(**BASE_PARAMS)
        ester = predict_matrix_stability(**{**BASE_PARAMS, "is_ester": True})
        assert ester.k_deg_per_h > no_ester.k_deg_per_h


# ---------------------------------------------------------------------------
# tissue_homogenate < csf stability
# ---------------------------------------------------------------------------


class TestMatrixComparison:
    def test_tissue_homogenate_less_stable_than_csf(self):
        th = predict_matrix_stability(**{**BASE_PARAMS, "matrix_type": "tissue_homogenate"})
        csf = predict_matrix_stability(**{**BASE_PARAMS, "matrix_type": "csf"})
        assert th.stability_pct < csf.stability_pct

    def test_whole_blood_more_degraded_than_plasma(self):
        blood = predict_matrix_stability(**{**BASE_PARAMS, "matrix_type": "whole_blood"})
        plasma = predict_matrix_stability(**{**BASE_PARAMS, "matrix_type": "plasma"})
        assert blood.stability_pct < plasma.stability_pct


# ---------------------------------------------------------------------------
# stability_class in valid set
# ---------------------------------------------------------------------------


class TestStabilityClass:
    def test_stability_class_valid(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert result.stability_class in VALID_CLASSES

    def test_cold_stable_class_is_stable(self):
        # At -80°C for a short time, should be stable
        result = predict_matrix_stability(
            **{**BASE_PARAMS, "temperature_C": -80.0, "duration_h": 1.0}
        )
        assert result.stability_class == "stable"

    def test_very_unstable_class(self):
        # tissue_homogenate + ester + oxidizable at 37°C for long time → unstable
        result = predict_matrix_stability(
            drug_name="Unstable",
            matrix_type="tissue_homogenate",
            temperature_C=37.0,
            duration_h=96.0,
            logp=4.0,
            is_ester=True,
            is_amide=False,
            has_oxidizable_group=True,
        )
        assert result.stability_class == "unstable"

    def test_recommended_storage_valid(self):
        result = predict_matrix_stability(**BASE_PARAMS)
        assert result.recommended_storage in VALID_STORAGES

    def test_unstable_storage_is_minus80(self):
        result = predict_matrix_stability(
            drug_name="Unstable",
            matrix_type="tissue_homogenate",
            temperature_C=37.0,
            duration_h=96.0,
            is_ester=True,
            has_oxidizable_group=True,
        )
        assert result.recommended_storage == "-80C"

    def test_stable_storage_is_4c(self):
        result = predict_matrix_stability(
            **{**BASE_PARAMS, "temperature_C": -80.0, "duration_h": 1.0}
        )
        assert result.recommended_storage == "4C"


# ---------------------------------------------------------------------------
# Freeze-thaw: more cycles → lower stability
# ---------------------------------------------------------------------------


class TestFreezeThaw:
    def test_more_cycles_lower_stability(self):
        few = assess_freeze_thaw_stability(drug_name="Drug", n_cycles=1)
        many = assess_freeze_thaw_stability(drug_name="Drug", n_cycles=5)
        assert many.stability_pct < few.stability_pct

    def test_ester_less_stable_in_freeze_thaw(self):
        no_ester = assess_freeze_thaw_stability(drug_name="Drug", n_cycles=3)
        ester = assess_freeze_thaw_stability(drug_name="Drug", n_cycles=3, is_ester=True)
        assert ester.stability_pct < no_ester.stability_pct

    def test_freeze_thaw_stability_in_range(self):
        result = assess_freeze_thaw_stability(drug_name="Drug", n_cycles=3)
        assert 0.0 < result.stability_pct <= 100.0

    def test_freeze_thaw_stability_class_valid(self):
        result = assess_freeze_thaw_stability(drug_name="Drug", n_cycles=3)
        assert result.stability_class in VALID_CLASSES

    def test_freeze_thaw_n_cycles_one(self):
        result = assess_freeze_thaw_stability(drug_name="Drug", n_cycles=1)
        assert isinstance(result, MatrixStabilityResult)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_matrix_type_raises(self):
        with pytest.raises(ValueError, match="matrix_type"):
            predict_matrix_stability(**{**BASE_PARAMS, "matrix_type": "saliva"})

    def test_temperature_too_low_raises(self):
        with pytest.raises(ValueError, match="temperature_C"):
            predict_matrix_stability(**{**BASE_PARAMS, "temperature_C": -201.0})

    def test_temperature_too_high_raises(self):
        with pytest.raises(ValueError, match="temperature_C"):
            predict_matrix_stability(**{**BASE_PARAMS, "temperature_C": 100.0})

    def test_duration_zero_raises(self):
        with pytest.raises(ValueError, match="duration_h"):
            predict_matrix_stability(**{**BASE_PARAMS, "duration_h": 0.0})

    def test_duration_negative_raises(self):
        with pytest.raises(ValueError, match="duration_h"):
            predict_matrix_stability(**{**BASE_PARAMS, "duration_h": -1.0})

    def test_freeze_thaw_zero_cycles_raises(self):
        with pytest.raises(ValueError, match="n_cycles"):
            assess_freeze_thaw_stability(drug_name="Drug", n_cycles=0)

    def test_freeze_thaw_negative_cycles_raises(self):
        with pytest.raises(ValueError, match="n_cycles"):
            assess_freeze_thaw_stability(drug_name="Drug", n_cycles=-1)
