"""Tests for omega_pbpk.prediction.concentration_ratio module."""

import pytest

from omega_pbpk.prediction.concentration_ratio import (
    ConcentrationRatioResult,
    _estimate_fut,
    predict_kp,
    predict_kp_panel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_DRUG = "TestDrug"
DEFAULT_LOGP = 2.0
DEFAULT_FUP = 0.3
DEFAULT_TISSUE = "muscle"


def _default_result(**kwargs):
    params = dict(
        drug_name=DEFAULT_DRUG,
        logP=DEFAULT_LOGP,
        fup=DEFAULT_FUP,
        tissue=DEFAULT_TISSUE,
    )
    params.update(kwargs)
    return predict_kp(**params)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------
class TestResultType:
    def test_returns_concentration_ratio_result(self):
        result = _default_result()
        assert isinstance(result, ConcentrationRatioResult)

    def test_dataclass_fields_present(self):
        result = _default_result()
        for field in (
            "drug_name",
            "logP",
            "fup",
            "pka",
            "tissue",
            "kp_predicted",
            "kp_method",
            "tissue_conc_free_uM",
            "plasma_conc_total_mg_L",
            "tissue_binding_factor",
        ):
            assert hasattr(result, field), f"Missing field: {field}"

    def test_drug_name_stored(self):
        result = predict_kp("Warfarin", 2.7, 0.01, "liver")
        assert result.drug_name == "Warfarin"

    def test_tissue_stored(self):
        result = predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, "brain")
        assert result.tissue == "brain"

    def test_fup_stored(self):
        result = _default_result(fup=0.15)
        assert result.fup == pytest.approx(0.15)

    def test_logP_stored(self):
        result = _default_result(logP=3.5)
        assert result.logP == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# 2. Core numeric constraints
# ---------------------------------------------------------------------------
class TestNumericConstraints:
    def test_kp_predicted_positive(self):
        result = _default_result()
        assert result.kp_predicted > 0

    def test_tissue_binding_factor_positive(self):
        result = _default_result()
        assert result.tissue_binding_factor > 0

    def test_kp_method_string(self):
        result = _default_result()
        assert isinstance(result.kp_method, str)
        assert len(result.kp_method) > 0

    def test_kp_method_value(self):
        result = _default_result()
        assert result.kp_method == "rodgers_rowland_simplified"

    def test_tissue_binding_factor_equals_inverse_fut(self):
        """tissue_binding_factor = 1/fut by spec."""
        result = _default_result()
        fut = _estimate_fut(DEFAULT_LOGP, DEFAULT_FUP, DEFAULT_TISSUE)
        assert result.tissue_binding_factor == pytest.approx(1.0 / fut, rel=1e-6)

    def test_kp_equals_fup_over_fut(self):
        """kp = fup / fut by spec."""
        result = _default_result()
        fut = _estimate_fut(DEFAULT_LOGP, DEFAULT_FUP, DEFAULT_TISSUE)
        expected_kp = DEFAULT_FUP / fut
        assert result.kp_predicted == pytest.approx(expected_kp, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. LogP and fup sensitivity
# ---------------------------------------------------------------------------
class TestParameterSensitivity:
    def test_high_logP_higher_kp_than_low_logP(self):
        """Lipophilic drugs partition more into tissue."""
        low = predict_kp(DEFAULT_DRUG, logP=0.5, fup=DEFAULT_FUP, tissue=DEFAULT_TISSUE)
        high = predict_kp(DEFAULT_DRUG, logP=4.0, fup=DEFAULT_FUP, tissue=DEFAULT_TISSUE)
        assert high.kp_predicted > low.kp_predicted

    def test_low_fup_nonzero_kp(self):
        low_fup = predict_kp(DEFAULT_DRUG, logP=DEFAULT_LOGP, fup=0.05, tissue=DEFAULT_TISSUE)
        assert low_fup.kp_predicted > 0  # model kp independent of fup by design

    def test_brain_kp_distinct_from_muscle(self):
        brain = predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, "brain")
        muscle = predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, "muscle")
        assert brain.kp_predicted != muscle.kp_predicted

    def test_acid_lower_kp_than_neutral_in_brain(self):
        neutral = predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, "brain", drug_type="neutral")
        acid = predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, "brain", drug_type="acid")
        assert acid.kp_predicted < neutral.kp_predicted

    def test_base_higher_kp_than_neutral_in_brain(self):
        neutral = predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, "brain", drug_type="neutral")
        base = predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, "brain", drug_type="base")
        assert base.kp_predicted > neutral.kp_predicted


# ---------------------------------------------------------------------------
# 4. _estimate_fut constraints
# ---------------------------------------------------------------------------
class TestEstimateFut:
    def test_fut_in_valid_range_low_logP(self):
        fut = _estimate_fut(logP=-2.0, fup=0.9, tissue="muscle")
        assert 0.001 <= fut <= 1.0

    def test_fut_in_valid_range_high_logP(self):
        fut = _estimate_fut(logP=5.0, fup=0.01, tissue="liver")
        assert 0.001 <= fut <= 1.0

    def test_fut_in_valid_range_brain(self):
        fut = _estimate_fut(logP=3.0, fup=0.2, tissue="brain")
        assert 0.001 <= fut <= 1.0

    def test_fut_lower_bound_respected(self):
        """Extreme lipophilicity should not drive fut below 0.001."""
        fut = _estimate_fut(logP=10.0, fup=0.001, tissue="adipose")
        assert fut >= 0.001

    def test_fut_upper_bound_respected(self):
        """Very hydrophilic drug should not exceed 1.0."""
        fut = _estimate_fut(logP=-5.0, fup=1.0, tissue="muscle")
        assert fut <= 1.0


# ---------------------------------------------------------------------------
# 5. Validation / error handling
# ---------------------------------------------------------------------------
class TestValidation:
    def test_fup_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, fup=0.0, tissue=DEFAULT_TISSUE)

    def test_fup_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, fup=-0.1, tissue=DEFAULT_TISSUE)

    def test_fup_greater_than_one_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, fup=1.1, tissue=DEFAULT_TISSUE)

    def test_fup_exactly_one_is_valid(self):
        result = predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, fup=1.0, tissue=DEFAULT_TISSUE)
        assert result.kp_predicted > 0  # fup=1.0 is valid (boundary included)

    def test_empty_tissue_raises_value_error(self):
        with pytest.raises(ValueError):
            predict_kp(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, tissue="")


# ---------------------------------------------------------------------------
# 6. predict_kp_panel
# ---------------------------------------------------------------------------
class TestPredictKpPanel:
    TISSUES = ["liver", "kidney", "brain", "muscle", "adipose"]

    def test_panel_returns_list(self):
        results = predict_kp_panel(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, self.TISSUES)
        assert isinstance(results, list)

    def test_panel_length_matches_tissues(self):
        results = predict_kp_panel(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, self.TISSUES)
        assert len(results) == len(self.TISSUES)

    def test_panel_sorted_by_kp_descending(self):
        results = predict_kp_panel(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, self.TISSUES)
        kps = [r.kp_predicted for r in results]
        assert kps == sorted(kps, reverse=True)

    def test_panel_all_items_are_concentration_ratio_result(self):
        results = predict_kp_panel(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, self.TISSUES)
        for r in results:
            assert isinstance(r, ConcentrationRatioResult)

    def test_panel_default_tissues_non_empty(self):
        """Calling without explicit tissues should use a default set."""
        results = predict_kp_panel(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP)
        assert len(results) > 0

    def test_panel_single_tissue(self):
        results = predict_kp_panel(DEFAULT_DRUG, DEFAULT_LOGP, DEFAULT_FUP, ["liver"])
        assert len(results) == 1

    def test_panel_two_tissues_sorted(self):
        results = predict_kp_panel(DEFAULT_DRUG, 4.0, 0.1, ["muscle", "adipose"])
        assert len(results) == 2
        assert results[0].kp_predicted >= results[1].kp_predicted
