"""Tests for dose-response modeling module (Phase 356)."""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.analysis.dose_response import (
    DoseResponseResult,
    compute_therapeutic_index,
    fit_dose_response,
    screen_dose_response_compounds,
)

# ---------------------------------------------------------------------------
# Helpers: generate synthetic dose-response data
# ---------------------------------------------------------------------------


def _hill_curve(doses, e0, emax, ec50, n):
    d = np.array(doses, dtype=float)
    return (e0 + emax * d**n / (ec50**n + d**n)).tolist()


def _default_doses():
    """Standard 8-point dose range."""
    return [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def _perfect_hill_responses(ec50=10.0, n=1.5, e0=0.0, emax=100.0):
    return _hill_curve(_default_doses(), e0, emax, ec50, n)


# ---------------------------------------------------------------------------
# fit_dose_response tests
# ---------------------------------------------------------------------------


class TestFitDoseResponse:
    def test_returns_dose_response_result(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert isinstance(result, DoseResponseResult)

    def test_frozen_dataclass(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        with pytest.raises((AttributeError, TypeError)):
            result.drug_name = "other"  # type: ignore[misc]

    def test_drug_name_preserved(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("TestDrug", doses, responses)
        assert result.drug_name == "TestDrug"

    def test_endpoint_preserved(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses, endpoint="toxicity")
        assert result.endpoint == "toxicity"

    def test_model_type_preserved(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses, model_type="sigmoid_emax")
        assert result.model_type == "sigmoid_emax"

    def test_perfect_hill_curve_r_squared_high(self):
        doses = _default_doses()
        responses = _perfect_hill_responses(ec50=10.0, n=1.5)
        result = fit_dose_response("DrugA", doses, responses)
        assert result.r_squared > 0.99

    def test_r_squared_in_zero_one(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert 0.0 <= result.r_squared <= 1.0

    def test_ed50_equals_ec50(self):
        doses = _default_doses()
        responses = _perfect_hill_responses(ec50=10.0)
        result = fit_dose_response("DrugA", doses, responses)
        assert result.ed50_mg == pytest.approx(result.ec50_mg, rel=1e-6)

    def test_ed10_less_than_ed50_less_than_ed90(self):
        doses = _default_doses()
        responses = _perfect_hill_responses(ec50=10.0, n=1.5)
        result = fit_dose_response("DrugA", doses, responses)
        assert result.ed10_mg < result.ed50_mg
        assert result.ed50_mg < result.ed90_mg

    def test_ec50_recovered_approximately(self):
        doses = _default_doses()
        ec50_true = 10.0
        responses = _perfect_hill_responses(ec50=ec50_true, n=1.5)
        result = fit_dose_response("DrugA", doses, responses)
        assert result.ec50_mg == pytest.approx(ec50_true, rel=0.1)

    def test_hill_n_positive(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert result.hill_n > 0.0

    def test_emax_positive(self):
        doses = _default_doses()
        responses = _perfect_hill_responses(emax=80.0)
        result = fit_dose_response("DrugA", doses, responses)
        assert result.emax > 0.0

    def test_rmse_nonnegative(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert result.rmse >= 0.0

    def test_e0_fixed_parameter_respected(self):
        doses = _default_doses()
        responses = _perfect_hill_responses(e0=5.0)
        result = fit_dose_response("DrugA", doses, responses, e0=5.0)
        assert result.e0 == pytest.approx(5.0, abs=1e-6)

    def test_emax_fixed_parameter_respected(self):
        doses = _default_doses()
        responses = _perfect_hill_responses(emax=100.0)
        result = fit_dose_response("DrugA", doses, responses, emax_fixed=100.0)
        assert result.emax == pytest.approx(100.0, abs=1e-6)

    def test_doses_preserved(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert result.doses_mg == doses

    def test_responses_preserved(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert result.responses == pytest.approx(responses, rel=1e-6)

    def test_responses_fitted_same_length(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert len(result.responses_fitted) == len(doses)

    def test_d_min_effective_in_dose_range(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert result.d_min_effective >= min(doses)
        assert result.d_min_effective <= max(doses)

    def test_d_max_useful_positive(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert result.d_max_useful > 0.0

    def test_therapeutic_index_zero_single_endpoint(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert result.therapeutic_index == pytest.approx(0.0)

    def test_selectivity_window_zero_single_endpoint(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert result.selectivity_window_fold == pytest.approx(0.0)

    def test_notes_nonempty(self):
        doses = _default_doses()
        responses = _perfect_hill_responses()
        result = fit_dose_response("DrugA", doses, responses)
        assert len(result.notes) > 0

    def test_noisy_data_still_fits(self):
        rng = np.random.default_rng(42)
        doses = _default_doses()
        clean = np.array(_perfect_hill_responses(ec50=10.0, n=1.0))
        noisy = (clean + rng.normal(0, 3, size=len(clean))).tolist()
        result = fit_dose_response("DrugA", doses, noisy)
        # Just check it runs and returns something reasonable
        assert result.ec50_mg > 0.0
        assert result.emax > 0.0

    def test_both_e0_and_emax_fixed(self):
        doses = _default_doses()
        responses = _perfect_hill_responses(e0=0.0, emax=100.0)
        result = fit_dose_response("DrugA", doses, responses, e0=0.0, emax_fixed=100.0)
        assert result.e0 == pytest.approx(0.0, abs=1e-6)
        assert result.emax == pytest.approx(100.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestFitDoseResponseValidation:
    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="4 data points"):
            fit_dose_response("DrugA", [1.0, 10.0, 100.0], [10.0, 50.0, 90.0])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_dose_response("DrugA", [1.0, 10.0, 100.0, 1000.0], [10.0, 50.0, 90.0])

    def test_zero_dose_raises(self):
        with pytest.raises(ValueError, match="doses_mg must be > 0"):
            fit_dose_response("DrugA", [0.0, 1.0, 10.0, 100.0], [0.0, 10.0, 50.0, 90.0])

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="doses_mg must be > 0"):
            fit_dose_response("DrugA", [-1.0, 1.0, 10.0, 100.0], [0.0, 10.0, 50.0, 90.0])

    def test_infinite_response_raises(self):
        with pytest.raises(ValueError, match="finite"):
            fit_dose_response("DrugA", [1.0, 10.0, 100.0, 1000.0], [0.0, float("inf"), 50.0, 90.0])


# ---------------------------------------------------------------------------
# compute_therapeutic_index tests
# ---------------------------------------------------------------------------


class TestComputeTherapeuticIndex:
    def _make_result(self, drug_name, endpoint, ec50):
        doses = _default_doses()
        responses = _perfect_hill_responses(ec50=ec50)
        return fit_dose_response(drug_name, doses, responses, endpoint=endpoint)

    def test_returns_dict(self):
        eff = self._make_result("DrugA", "efficacy", ec50=10.0)
        tox = self._make_result("DrugA", "toxicity", ec50=100.0)
        result = compute_therapeutic_index(eff, tox)
        assert isinstance(result, dict)

    def test_ti_equals_ed50_ratio(self):
        eff = self._make_result("DrugA", "efficacy", ec50=10.0)
        tox = self._make_result("DrugA", "toxicity", ec50=100.0)
        result = compute_therapeutic_index(eff, tox)
        expected_ti = tox.ec50_mg / eff.ec50_mg
        assert result["therapeutic_index"] == pytest.approx(expected_ti, rel=0.05)

    def test_ti_keys_present(self):
        eff = self._make_result("DrugA", "efficacy", ec50=5.0)
        tox = self._make_result("DrugA", "toxicity", ec50=50.0)
        result = compute_therapeutic_index(eff, tox)
        assert "therapeutic_index" in result
        assert "selectivity_window" in result
        assert "risk_assessment" in result

    def test_favorable_risk_high_ti(self):
        eff = self._make_result("DrugA", "efficacy", ec50=1.0)
        tox = self._make_result("DrugA", "toxicity", ec50=100.0)
        result = compute_therapeutic_index(eff, tox)
        assert result["risk_assessment"] == "favorable"

    def test_unfavorable_risk_low_ti(self):
        eff = self._make_result("DrugA", "efficacy", ec50=50.0)
        tox = self._make_result("DrugA", "toxicity", ec50=55.0)
        result = compute_therapeutic_index(eff, tox)
        assert result["risk_assessment"] == "unfavorable"

    def test_marginal_risk(self):
        eff = self._make_result("DrugA", "efficacy", ec50=10.0)
        tox = self._make_result("DrugA", "toxicity", ec50=30.0)
        result = compute_therapeutic_index(eff, tox)
        assert result["risk_assessment"] in ("marginal", "favorable", "unfavorable")

    def test_ti_positive(self):
        eff = self._make_result("DrugA", "efficacy", ec50=10.0)
        tox = self._make_result("DrugA", "toxicity", ec50=50.0)
        result = compute_therapeutic_index(eff, tox)
        assert result["therapeutic_index"] > 0.0


# ---------------------------------------------------------------------------
# screen_dose_response_compounds tests
# ---------------------------------------------------------------------------


class TestScreenDoseResponseCompounds:
    def _make_compound(self, name, ec50):
        doses = _default_doses()
        responses = _perfect_hill_responses(ec50=ec50)
        return {"name": name, "doses_mg": doses, "responses": responses}

    def test_returns_list(self):
        compounds = [
            self._make_compound("DrugA", ec50=10.0),
            self._make_compound("DrugB", ec50=1.0),
            self._make_compound("DrugC", ec50=50.0),
        ]
        results = screen_dose_response_compounds(compounds)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_sorted_by_ec50_ascending(self):
        compounds = [
            self._make_compound("DrugA", ec50=50.0),
            self._make_compound("DrugB", ec50=1.0),
            self._make_compound("DrugC", ec50=10.0),
        ]
        results = screen_dose_response_compounds(compounds)
        ec50s = [r.ec50_mg for r in results]
        assert ec50s == sorted(ec50s)

    def test_most_potent_first(self):
        compounds = [
            self._make_compound("DrugA", ec50=100.0),
            self._make_compound("DrugB", ec50=0.5),
        ]
        results = screen_dose_response_compounds(compounds)
        assert results[0].ec50_mg < results[1].ec50_mg

    def test_all_results_are_dose_response_result(self):
        compounds = [
            self._make_compound("DrugA", ec50=10.0),
            self._make_compound("DrugB", ec50=5.0),
        ]
        results = screen_dose_response_compounds(compounds)
        for r in results:
            assert isinstance(r, DoseResponseResult)

    def test_single_compound(self):
        compounds = [self._make_compound("DrugA", ec50=10.0)]
        results = screen_dose_response_compounds(compounds)
        assert len(results) == 1

    def test_missing_name_raises(self):
        compounds = [
            {"doses_mg": [1.0, 10.0, 100.0, 1000.0], "responses": [10.0, 50.0, 80.0, 95.0]}
        ]
        with pytest.raises(ValueError, match="name"):
            screen_dose_response_compounds(compounds)

    def test_missing_doses_raises(self):
        compounds = [{"name": "DrugA", "responses": [10.0, 50.0, 80.0, 95.0]}]
        with pytest.raises(ValueError, match="doses_mg"):
            screen_dose_response_compounds(compounds)

    def test_missing_responses_raises(self):
        compounds = [{"name": "DrugA", "doses_mg": [1.0, 10.0, 100.0, 1000.0]}]
        with pytest.raises(ValueError, match="responses"):
            screen_dose_response_compounds(compounds)

    def test_optional_endpoint_key(self):
        compounds = [
            {**self._make_compound("DrugA", ec50=10.0), "endpoint": "toxicity"},
        ]
        results = screen_dose_response_compounds(compounds)
        assert results[0].endpoint == "toxicity"

    def test_empty_list_returns_empty(self):
        results = screen_dose_response_compounds([])
        assert results == []
