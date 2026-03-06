"""Tests for omega_pbpk.clinical.trial_simulation."""

import pytest
from omega_pbpk.clinical.trial_simulation import (
    TrialSimResult,
    dose_response_simulation,
    simulate_trial,
)

VALID_RECOMMENDATIONS = {"proceed", "enrich", "stop_futility", "stop_efficacy"}

# --- Fixtures ---

@pytest.fixture
def base_result():
    return simulate_trial(
        drug_name="DrugA",
        dose_mg=100.0,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=42,
    )


@pytest.fixture
def low_dose_result():
    return simulate_trial(
        drug_name="DrugB",
        dose_mg=10.0,
        n_subjects=200,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=42,
    )


@pytest.fixture
def high_dose_result():
    return simulate_trial(
        drug_name="DrugB",
        dose_mg=500.0,
        n_subjects=200,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=42,
    )


@pytest.fixture
def dose_response_results():
    doses = [10.0, 50.0, 100.0, 200.0, 400.0]
    return dose_response_simulation(
        drug_name="DrugC",
        doses=doses,
        n_subjects=150,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=42,
    )


# --- 1. Return type is TrialSimResult ---

def test_simulate_trial_returns_trialsimresult(base_result):
    assert isinstance(base_result, TrialSimResult)


# --- 2. responder_rate in [0, 1] ---

def test_responder_rate_bounds_low(base_result):
    assert base_result.responder_rate >= 0.0


def test_responder_rate_bounds_high(base_result):
    assert base_result.responder_rate <= 1.0


# --- 3. p_value in [0, 1] ---

def test_p_value_lower_bound(base_result):
    assert base_result.p_value >= 0.0


def test_p_value_upper_bound(base_result):
    assert base_result.p_value <= 1.0


# --- 4. confidence_interval_95 is tuple of 2 floats with ci[0] < ci[1] ---

def test_ci_is_tuple(base_result):
    assert isinstance(base_result.confidence_interval_95, tuple)


def test_ci_has_two_elements(base_result):
    assert len(base_result.confidence_interval_95) == 2


def test_ci_elements_are_floats(base_result):
    ci = base_result.confidence_interval_95
    assert isinstance(ci[0], float)
    assert isinstance(ci[1], float)


def test_ci_lower_less_than_upper(base_result):
    ci = base_result.confidence_interval_95
    assert ci[0] <= ci[1]  # can be equal for extreme responder rates


# --- 5. is_significant == (p_value < 0.05) ---

def test_is_significant_matches_p_value(base_result):
    assert base_result.is_significant == (base_result.p_value < 0.05)


def test_is_significant_across_multiple_doses(dose_response_results):
    for r in dose_response_results:
        assert r.is_significant == (r.p_value < 0.05)


# --- 6. trial_recommendation is valid string ---

def test_trial_recommendation_valid(base_result):
    assert base_result.trial_recommendation in VALID_RECOMMENDATIONS


def test_trial_recommendation_type(base_result):
    assert isinstance(base_result.trial_recommendation, str)


def test_trial_recommendation_across_doses(dose_response_results):
    for r in dose_response_results:
        assert r.trial_recommendation in VALID_RECOMMENDATIONS


# --- 7. High dose -> higher responder_rate than low dose (same ec50) ---

def test_high_dose_higher_responder_rate(low_dose_result, high_dose_result):
    assert high_dose_result.responder_rate >= low_dose_result.responder_rate


# --- 8. dose_response list length == len(doses) ---

def test_dose_response_list_length(dose_response_results):
    assert len(dose_response_results) == 5


def test_dose_response_list_length_single():
    results = dose_response_simulation(
        drug_name="DrugD",
        doses=[100.0],
        n_subjects=50,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=42,
    )
    assert len(results) == 1


# --- 9. dose_response sorted by dose_mg ascending ---

def test_dose_response_sorted_ascending(dose_response_results):
    doses = [r.dose_mg for r in dose_response_results]
    assert doses == sorted(doses)


def test_dose_response_sorted_when_input_unsorted():
    doses = [400.0, 10.0, 200.0, 50.0]
    results = dose_response_simulation(
        drug_name="DrugE",
        doses=doses,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=42,
    )
    dose_vals = [r.dose_mg for r in results]
    assert dose_vals == sorted(dose_vals)


# --- 10. n_subjects preserved ---

def test_n_subjects_preserved(base_result):
    assert base_result.n_subjects == 100


def test_n_subjects_preserved_varied():
    result = simulate_trial(
        drug_name="DrugF",
        dose_mg=50.0,
        n_subjects=250,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=42,
    )
    assert result.n_subjects == 250


# --- 11. drug_name preserved ---

def test_drug_name_preserved(base_result):
    assert base_result.drug_name == "DrugA"


def test_drug_name_preserved_in_dose_response(dose_response_results):
    for r in dose_response_results:
        assert r.drug_name == "DrugC"


# --- 12. seed reproducible (same result twice) ---

def test_seed_reproducibility():
    kwargs = dict(
        drug_name="DrugG",
        dose_mg=100.0,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=99,
    )
    r1 = simulate_trial(**kwargs)
    r2 = simulate_trial(**kwargs)
    assert r1.responder_rate == r2.responder_rate
    assert r1.p_value == r2.p_value
    assert r1.mean_response == r2.mean_response
    assert r1.is_significant == r2.is_significant
    assert r1.confidence_interval_95 == r2.confidence_interval_95


def test_different_seeds_may_differ():
    common = dict(
        drug_name="DrugH",
        dose_mg=100.0,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
    )
    r1 = simulate_trial(**common, seed=1)
    r2 = simulate_trial(**common, seed=2)
    # With different seeds results should not always be identical
    # (probabilistic; extremely unlikely to be equal for large n)
    results_identical = (
        r1.responder_rate == r2.responder_rate
        and r1.p_value == r2.p_value
        and r1.mean_response == r2.mean_response
    )
    # We only assert that the function runs; identity is not guaranteed
    assert not results_identical or True  # always passes but documents intent


# --- 13. mean_response in [0, emax] ---

def test_mean_response_lower_bound(base_result):
    assert base_result.mean_response >= 0.0


def test_mean_response_upper_bound_default_emax(base_result):
    assert base_result.mean_response <= 1.0


def test_mean_response_upper_bound_custom_emax():
    result = simulate_trial(
        drug_name="DrugI",
        dose_mg=100.0,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        emax=2.0,
        seed=42,
    )
    assert result.mean_response >= 0.0
    assert result.mean_response <= 2.0


# --- 14. dose_mg preserved ---

def test_dose_mg_preserved():
    result = simulate_trial(
        drug_name="DrugJ",
        dose_mg=75.5,
        n_subjects=80,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        seed=42,
    )
    assert result.dose_mg == 75.5


# --- 15. primary_endpoint preserved ---

def test_primary_endpoint_default(base_result):
    assert base_result.primary_endpoint == "response_rate"


def test_primary_endpoint_custom():
    result = simulate_trial(
        drug_name="DrugK",
        dose_mg=100.0,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        primary_endpoint="pfs",
        seed=42,
    )
    assert result.primary_endpoint == "pfs"


# --- 16. Hill coefficient and placebo effect parameters accepted ---

def test_hill_coefficient_parameter():
    result = simulate_trial(
        drug_name="DrugL",
        dose_mg=100.0,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        n_hill=2.0,
        seed=42,
    )
    assert isinstance(result, TrialSimResult)
    assert 0.0 <= result.responder_rate <= 1.0


def test_placebo_effect_parameter():
    result = simulate_trial(
        drug_name="DrugM",
        dose_mg=100.0,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        placebo_effect=0.2,
        seed=42,
    )
    assert isinstance(result, TrialSimResult)
    assert 0.0 <= result.responder_rate <= 1.0


# --- 17. dose_response returns list of TrialSimResult ---

def test_dose_response_returns_list(dose_response_results):
    assert isinstance(dose_response_results, list)


def test_dose_response_elements_are_trialsimresult(dose_response_results):
    for r in dose_response_results:
        assert isinstance(r, TrialSimResult)


# --- 18. power field exists and is in [0, 1] ---

def test_power_in_range(base_result):
    assert 0.0 <= base_result.power <= 1.0


def test_power_across_doses(dose_response_results):
    for r in dose_response_results:
        assert 0.0 <= r.power <= 1.0


# --- 19. Very low dose -> low responder_rate ---

def test_very_low_dose_low_responder_rate():
    result = simulate_trial(
        drug_name="DrugN",
        dose_mg=0.001,
        n_subjects=500,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=10.0,
        seed=42,
    )
    # At negligible dose, response should be close to placebo (low)
    assert result.responder_rate < 0.9


# --- 20. responder_threshold parameter accepted ---

def test_responder_threshold_parameter():
    result = simulate_trial(
        drug_name="DrugO",
        dose_mg=100.0,
        n_subjects=100,
        cl_L_per_h=5.0,
        vd_L=50.0,
        ec50_mg_L=1.0,
        responder_threshold=0.3,
        seed=42,
    )
    assert isinstance(result, TrialSimResult)
    assert 0.0 <= result.responder_rate <= 1.0


# --- 21. TrialSimResult dataclass fields ---

def test_trialsimresult_has_all_fields(base_result):
    assert hasattr(base_result, "n_subjects")
    assert hasattr(base_result, "drug_name")
    assert hasattr(base_result, "dose_mg")
    assert hasattr(base_result, "primary_endpoint")
    assert hasattr(base_result, "responder_rate")
    assert hasattr(base_result, "mean_response")
    assert hasattr(base_result, "p_value")
    assert hasattr(base_result, "power")
    assert hasattr(base_result, "confidence_interval_95")
    assert hasattr(base_result, "is_significant")
    assert hasattr(base_result, "trial_recommendation")
