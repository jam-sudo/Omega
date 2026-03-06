"""Tests for omega_pbpk.prediction.clearance_prediction module."""

import pytest

from omega_pbpk.prediction.clearance_prediction import (
    ClearancePredictionResult,
    predict_clearance,
    batch_clearance_predict,
)

VALID_DRUG_LIKE = dict(drug_name="DrugA", logP=2.0, MW=300.0, fup=0.1)
VALID_EXTREME = dict(drug_name="BigMol", logP=6.0, MW=700.0, fup=0.05)

VALID_ROUTES = {"renal", "hepatic", "mixed"}


# ---------------------------------------------------------------------------
# Result type and structure
# ---------------------------------------------------------------------------

class TestResultType:
    def test_returns_result_instance(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert isinstance(result, ClearancePredictionResult)

    def test_result_has_drug_name(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.drug_name == "DrugA"

    def test_model_used_string(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.model_used == "linear_physicochemical_v1"

    def test_all_cl_fields_present(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        for attr in ("cl_predicted_L_per_h", "cl_renal_L_per_h",
                     "cl_hepatic_L_per_h", "cl_total_L_per_h"):
            assert hasattr(result, attr)

    def test_vd_field_present(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert hasattr(result, "vd_predicted_L")

    def test_t_half_field_present(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert hasattr(result, "t_half_predicted_h")

    def test_elimination_route_field_present(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert hasattr(result, "elimination_route")

    def test_confidence_field_present(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert hasattr(result, "confidence")


# ---------------------------------------------------------------------------
# Numerical constraints
# ---------------------------------------------------------------------------

class TestNumericalConstraints:
    def test_cl_total_equals_hepatic_plus_renal(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.cl_total_L_per_h == pytest.approx(
            result.cl_hepatic_L_per_h + result.cl_renal_L_per_h, rel=1e-6
        )

    def test_t_half_positive(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.t_half_predicted_h > 0

    def test_t_half_formula(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        expected = 0.693 * result.vd_predicted_L / result.cl_total_L_per_h
        assert result.t_half_predicted_h == pytest.approx(expected, rel=1e-6)

    def test_vd_positive(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.vd_predicted_L > 0

    def test_cl_hepatic_non_negative(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.cl_hepatic_L_per_h >= 0

    def test_cl_renal_non_negative(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.cl_renal_L_per_h >= 0

    def test_cl_total_non_negative(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.cl_total_L_per_h >= 0

    def test_cl_predicted_non_negative(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.cl_predicted_L_per_h >= 0


# ---------------------------------------------------------------------------
# Elimination route
# ---------------------------------------------------------------------------

class TestEliminationRoute:
    def test_route_in_valid_set(self):
        result = predict_clearance(**VALID_DRUG_LIKE)
        assert result.elimination_route in VALID_ROUTES

    def test_route_in_valid_set_extreme(self):
        result = predict_clearance(**VALID_EXTREME)
        assert result.elimination_route in VALID_ROUTES


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_drug_like_confidence_0_9(self):
        # drug-like: 100 <= MW <= 500 and 0 <= logP <= 5
        result = predict_clearance(drug_name="DL", logP=2.0, MW=300.0, fup=0.1)
        assert result.confidence == pytest.approx(0.9)

    def test_extreme_mw_confidence_0_6(self):
        result = predict_clearance(drug_name="Big", logP=2.0, MW=700.0, fup=0.05)
        assert result.confidence == pytest.approx(0.6)

    def test_extreme_logp_confidence_0_6(self):
        result = predict_clearance(drug_name="Lipophilic", logP=6.0, MW=300.0, fup=0.05)
        assert result.confidence == pytest.approx(0.6)

    def test_low_mw_confidence_0_6(self):
        result = predict_clearance(drug_name="Tiny", logP=2.0, MW=50.0, fup=0.3)
        assert result.confidence == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Physicochemical relationships
# ---------------------------------------------------------------------------

class TestPhysicochemicalRelationships:
    def test_high_fup_higher_cl_hepatic_than_low_fup(self):
        low_fup = predict_clearance(drug_name="Low", logP=2.0, MW=300.0, fup=0.01)
        high_fup = predict_clearance(drug_name="High", logP=2.0, MW=300.0, fup=0.9)
        assert high_fup.cl_hepatic_L_per_h > low_fup.cl_hepatic_L_per_h

    def test_high_mw_lower_cl_renal_than_low_mw(self):
        low_mw = predict_clearance(drug_name="Small", logP=1.0, MW=150.0, fup=0.5)
        high_mw = predict_clearance(drug_name="Large", logP=1.0, MW=450.0, fup=0.5)
        assert high_mw.cl_renal_L_per_h < low_mw.cl_renal_L_per_h


# ---------------------------------------------------------------------------
# ValueError inputs
# ---------------------------------------------------------------------------

class TestValueErrors:
    def test_fup_zero_raises(self):
        with pytest.raises(ValueError):
            predict_clearance(drug_name="X", logP=2.0, MW=300.0, fup=0.0)

    def test_fup_negative_raises(self):
        with pytest.raises(ValueError):
            predict_clearance(drug_name="X", logP=2.0, MW=300.0, fup=-0.1)

    def test_fup_greater_than_one_raises(self):
        with pytest.raises(ValueError):
            predict_clearance(drug_name="X", logP=2.0, MW=300.0, fup=1.1)

    def test_mw_zero_raises(self):
        with pytest.raises(ValueError):
            predict_clearance(drug_name="X", logP=2.0, MW=0.0, fup=0.1)

    def test_mw_negative_raises(self):
        with pytest.raises(ValueError):
            predict_clearance(drug_name="X", logP=2.0, MW=-100.0, fup=0.1)


# ---------------------------------------------------------------------------
# Optional parameters
# ---------------------------------------------------------------------------

class TestOptionalParameters:
    def test_default_pka_and_psa(self):
        result = predict_clearance(drug_name="Default", logP=2.0, MW=300.0, fup=0.1)
        assert isinstance(result, ClearancePredictionResult)
        assert result.pka == pytest.approx(7.0)
        assert result.PSA == pytest.approx(60.0)

    def test_custom_pka_stored(self):
        result = predict_clearance(drug_name="Acid", logP=2.0, MW=300.0, fup=0.1, pka=4.5)
        assert result.pka == pytest.approx(4.5)

    def test_drug_type_accepted(self):
        result = predict_clearance(
            drug_name="Acid", logP=2.0, MW=300.0, fup=0.1, drug_type="acid"
        )
        assert isinstance(result, ClearancePredictionResult)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

class TestBatchClearancePredict:
    def _compounds(self):
        return [
            {"drug_name": "Fast", "logP": 1.0, "MW": 200.0, "fup": 0.9},
            {"drug_name": "Slow", "logP": 2.0, "MW": 400.0, "fup": 0.05},
            {"drug_name": "Mid",  "logP": 2.0, "MW": 300.0, "fup": 0.3},
        ]

    def test_batch_returns_list(self):
        results = batch_clearance_predict(self._compounds())
        assert isinstance(results, list)

    def test_batch_length_matches_valid_inputs(self):
        results = batch_clearance_predict(self._compounds())
        assert len(results) == 3

    def test_batch_sorted_by_cl_total_desc(self):
        results = batch_clearance_predict(self._compounds())
        cl_totals = [r.cl_total_L_per_h for r in results]
        assert cl_totals == sorted(cl_totals, reverse=True)

    def test_batch_skips_invalid_entries(self):
        compounds = self._compounds() + [
            {"drug_name": "Bad", "logP": 1.0, "MW": 300.0, "fup": -0.5},
        ]
        results = batch_clearance_predict(compounds)
        assert len(results) == 3

    def test_batch_all_results_correct_type(self):
        results = batch_clearance_predict(self._compounds())
        for r in results:
            assert isinstance(r, ClearancePredictionResult)
