"""Tests for Phase 668 — dose_response_toxicity."""

import pytest

from omega_pbpk.risk.dose_response_toxicity import (
    ToxicityDoseResponseResult,
    compute_dose_response_curve,
    model_toxicity_dose_response,
    screen_toxicity_endpoints,
)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

NOAEL = 10.0  # mg/kg
LOAEL = 20.0  # mg/kg
TD50 = 100.0  # mg/kg
HUMAN_DOSE = 1.0  # mg/kg — large margin → acceptable


def make_result(human_dose: float = HUMAN_DOSE, hill_n: float = 2.0) -> ToxicityDoseResponseResult:
    return model_toxicity_dose_response(
        compound_name="TestCompound",
        toxic_endpoint="hepatotoxicity",
        noael_mg_kg=NOAEL,
        loael_mg_kg=LOAEL,
        td50_mg_kg=TD50,
        human_dose_mg_kg=human_dose,
        hill_n=hill_n,
    )


# ---------------------------------------------------------------------------
# Basic return type and field checks
# ---------------------------------------------------------------------------


def test_returns_toxicity_dose_response_result():
    assert isinstance(make_result(), ToxicityDoseResponseResult)


def test_noael_positive():
    assert make_result().noael_mg_kg > 0


def test_td50_greater_than_noael():
    r = make_result()
    assert r.td50_mg_kg > r.noael_mg_kg


def test_predicted_incidence_in_range():
    r = make_result()
    assert 0.0 <= r.predicted_incidence_at_dose <= 1.0


def test_hill_n_positive():
    assert make_result().hill_n > 0


def test_dose_margin_equals_noael_over_dose():
    r = make_result(human_dose=HUMAN_DOSE)
    assert abs(r.dose_margin - NOAEL / HUMAN_DOSE) < 1e-9


def test_benchmark_dose_10_positive():
    assert make_result().benchmark_dose_10_mg_kg > 0


def test_risk_category_valid_values():
    valid = {"acceptable", "marginal", "concerning", "unacceptable"}
    assert make_result().risk_category in valid


def test_notes_nonempty():
    assert len(make_result().notes) > 0


# ---------------------------------------------------------------------------
# Risk category thresholds
# ---------------------------------------------------------------------------


def test_large_dose_margin_acceptable():
    """NOAEL/human_dose > 10 → acceptable."""
    r = make_result(human_dose=0.5)  # margin = 10/0.5 = 20
    assert r.risk_category == "acceptable"


def test_small_dose_margin_unacceptable():
    """NOAEL/human_dose < 2 → unacceptable."""
    r = make_result(human_dose=8.0)  # margin = 10/8 = 1.25
    assert r.risk_category == "unacceptable"


def test_dose_margin_3_concerning():
    """NOAEL/human_dose ≈ 3 → concerning."""
    r = make_result(human_dose=NOAEL / 3.0)  # margin = 3
    assert r.risk_category == "concerning"


def test_dose_margin_7_marginal():
    """NOAEL/human_dose ≈ 7 → marginal."""
    r = make_result(human_dose=NOAEL / 7.0)  # margin = 7
    assert r.risk_category == "marginal"


# ---------------------------------------------------------------------------
# compute_dose_response_curve
# ---------------------------------------------------------------------------


def test_curve_length_matches_doses():
    doses = [0.1, 1.0, 10.0, 100.0, 1000.0]
    curve = compute_dose_response_curve(TD50, 2.0, doses)
    assert len(curve) == len(doses)


def test_curve_all_values_in_0_1():
    doses = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
    curve = compute_dose_response_curve(TD50, 2.0, doses)
    for v in curve:
        assert 0.0 <= v <= 1.0


def test_curve_response_at_td50_approx_half():
    """At dose=TD50, Hill response ≈ background + (1-background)*0.5 ≈ 0.505."""
    background = 0.01
    curve = compute_dose_response_curve(TD50, 2.0, [TD50], background)
    expected = background + (1.0 - background) * 0.5
    assert abs(curve[0] - expected) < 1e-9


def test_curve_response_increases_with_dose():
    doses = [1.0, 10.0, 50.0, 100.0, 500.0]
    curve = compute_dose_response_curve(TD50, 2.0, doses)
    for i in range(len(curve) - 1):
        assert curve[i + 1] >= curve[i]


# ---------------------------------------------------------------------------
# screen_toxicity_endpoints
# ---------------------------------------------------------------------------

ENDPOINTS = [
    {"endpoint": "hepatotoxicity", "noael": 10.0, "loael": 20.0, "td50": 100.0},
    {"endpoint": "nephrotoxicity", "noael": 5.0, "loael": 10.0, "td50": 50.0},
    {"endpoint": "cardiotoxicity", "noael": 50.0, "loael": 80.0, "td50": 200.0},
]


def test_screen_returns_list():
    results = screen_toxicity_endpoints("TestDrug", ENDPOINTS, human_dose_mg_kg=1.0)
    assert isinstance(results, list)
    assert len(results) == len(ENDPOINTS)


def test_screen_sorted_by_dose_margin_ascending():
    results = screen_toxicity_endpoints("TestDrug", ENDPOINTS, human_dose_mg_kg=1.0)
    margins = [r.dose_margin for r in results]
    assert margins == sorted(margins)


def test_screen_most_concerning_first():
    """Lowest NOAEL (nephrotoxicity, NOAEL=5) should be first."""
    results = screen_toxicity_endpoints("TestDrug", ENDPOINTS, human_dose_mg_kg=1.0)
    assert results[0].toxic_endpoint == "nephrotoxicity"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_noael_zero_raises():
    with pytest.raises(ValueError):
        model_toxicity_dose_response("X", "liver", 0.0, 5.0, 50.0)


def test_validation_loael_le_noael_raises():
    with pytest.raises(ValueError):
        model_toxicity_dose_response("X", "liver", 10.0, 10.0, 100.0)


def test_validation_loael_gt_td50_raises():
    with pytest.raises(ValueError):
        model_toxicity_dose_response("X", "liver", 5.0, 200.0, 100.0)


def test_validation_background_ge_1_raises():
    with pytest.raises(ValueError):
        model_toxicity_dose_response("X", "liver", 5.0, 10.0, 100.0, background_incidence=1.0)


# ---------------------------------------------------------------------------
# Edge case: predicted incidence near background at very low dose
# ---------------------------------------------------------------------------


def test_predicted_incidence_near_background_at_low_dose():
    """At very low human dose, incidence ≈ background."""
    background = 0.01
    r = model_toxicity_dose_response(
        "X",
        "liver",
        noael_mg_kg=100.0,
        loael_mg_kg=200.0,
        td50_mg_kg=1000.0,
        human_dose_mg_kg=0.0001,
        background_incidence=background,
    )
    assert abs(r.predicted_incidence_at_dose - background) < 0.001
