"""Tests for Phase 459 — Dose-Dependent Bioavailability."""

import pytest

from omega_pbpk.prediction.dose_dependent_bioavailability import (
    DoseBioavailabilityResult,
    predict_dose_bioavailability,
    screen_doses,
)

# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


def test_return_type_basic():
    result = predict_dose_bioavailability("DrugA", dose_mg=10.0)
    assert isinstance(result, DoseBioavailabilityResult)


def test_return_type_all_fields_present():
    result = predict_dose_bioavailability("DrugB", dose_mg=50.0)
    assert hasattr(result, "drug_name")
    assert hasattr(result, "dose_mg")
    assert hasattr(result, "fa")
    assert hasattr(result, "f_hepatic")
    assert hasattr(result, "f_oral")
    assert hasattr(result, "mechanism")
    assert hasattr(result, "dose_proportionality_index")
    assert hasattr(result, "notes")


# ---------------------------------------------------------------------------
# f_oral range tests
# ---------------------------------------------------------------------------


def test_f_oral_between_0_and_1_saturable_absorption():
    result = predict_dose_bioavailability("DrugC", dose_mg=100.0, saturable_absorption=True)
    assert 0.0 <= result.f_oral <= 1.0


def test_f_oral_between_0_and_1_linear():
    result = predict_dose_bioavailability(
        "DrugD", dose_mg=50.0, saturable_absorption=False, saturable_firstpass=False
    )
    assert 0.0 <= result.f_oral <= 1.0


def test_fa_between_0_and_1():
    result = predict_dose_bioavailability("DrugE", dose_mg=200.0, saturable_absorption=True)
    assert 0.0 <= result.fa <= 1.0


def test_f_hepatic_between_0_and_1():
    result = predict_dose_bioavailability("DrugF", dose_mg=50.0)
    assert 0.0 <= result.f_hepatic <= 1.0


# ---------------------------------------------------------------------------
# Absorption saturation behaviour
# ---------------------------------------------------------------------------


def test_low_dose_fa_linear_regime():
    """At dose << Ka_mg, fa ≈ fa_max * dose / Ka_mg (linear regime)."""
    fa_max = 0.95
    Ka_mg = 500.0
    dose_mg = 5.0  # << Ka_mg
    result = predict_dose_bioavailability(
        "DrugG", dose_mg=dose_mg, fa_max=fa_max, Ka_mg=Ka_mg, saturable_absorption=True
    )
    expected_fa = fa_max * dose_mg / (Ka_mg + dose_mg)
    assert abs(result.fa - expected_fa) < 1e-9


def test_high_dose_fa_approaches_fa_max():
    """At dose >> Ka_mg, fa approaches fa_max."""
    fa_max = 0.90
    Ka_mg = 10.0
    dose_mg = 10000.0  # >> Ka_mg
    result = predict_dose_bioavailability(
        "DrugH", dose_mg=dose_mg, fa_max=fa_max, Ka_mg=Ka_mg, saturable_absorption=True
    )
    assert abs(result.fa - fa_max) < 0.01


def test_fa_increases_with_dose():
    r1 = predict_dose_bioavailability("Drug", dose_mg=10.0, saturable_absorption=True, Ka_mg=100.0)
    r2 = predict_dose_bioavailability("Drug", dose_mg=100.0, saturable_absorption=True, Ka_mg=100.0)
    assert r2.fa > r1.fa


def test_linear_fa_equals_fa_max():
    """Without saturable absorption, fa == fa_max (capped at 1)."""
    fa_max = 0.80
    result = predict_dose_bioavailability(
        "DrugI", dose_mg=50.0, fa_max=fa_max, saturable_absorption=False
    )
    assert abs(result.fa - fa_max) < 1e-9


# ---------------------------------------------------------------------------
# Dose-proportionality index
# ---------------------------------------------------------------------------


def test_dpi_greater_than_1_for_saturable_absorption_low_dose():
    """Saturable absorption at low dose → DPI > 1 (supra-proportional).

    When dose << Ka_mg, doubling the dose more than doubles AUC because
    fa grows from dose/(Ka+dose) to 2*dose/(Ka+2*dose), which is a larger
    relative gain than the 2x dose increase.
    """
    result = predict_dose_bioavailability(
        "DrugJ",
        dose_mg=10.0,  # well below Ka_mg=100
        Ka_mg=100.0,
        saturable_absorption=True,
        saturable_firstpass=False,
        cl_int_L_per_h=1.0,  # minimal hepatic extraction
        Q_h_L_per_h=90.0,
        fu_plasma=0.01,
    )
    # At low dose relative to Ka, DPI > 1 (supra-proportional)
    assert result.dose_proportionality_index > 1.0


def test_dpi_approx_1_for_linear_model():
    """Linear model → DPI ≈ 1."""
    result = predict_dose_bioavailability(
        "DrugK",
        dose_mg=50.0,
        saturable_absorption=False,
        saturable_firstpass=False,
    )
    assert abs(result.dose_proportionality_index - 1.0) < 0.01


def test_dpi_positive():
    result = predict_dose_bioavailability("DrugL", dose_mg=100.0)
    assert result.dose_proportionality_index > 0.0


# ---------------------------------------------------------------------------
# Mechanism string
# ---------------------------------------------------------------------------


def test_mechanism_linear():
    result = predict_dose_bioavailability(
        "Drug", dose_mg=10.0, saturable_absorption=False, saturable_firstpass=False
    )
    assert result.mechanism == "linear"


def test_mechanism_saturable_absorption():
    result = predict_dose_bioavailability(
        "Drug", dose_mg=10.0, saturable_absorption=True, saturable_firstpass=False
    )
    assert result.mechanism == "saturable_absorption"


def test_mechanism_saturable_firstpass():
    result = predict_dose_bioavailability(
        "Drug", dose_mg=10.0, saturable_absorption=False, saturable_firstpass=True
    )
    assert result.mechanism == "saturable_firstpass"


def test_mechanism_both():
    result = predict_dose_bioavailability(
        "Drug", dose_mg=10.0, saturable_absorption=True, saturable_firstpass=True
    )
    assert result.mechanism == "both"


# ---------------------------------------------------------------------------
# screen_doses
# ---------------------------------------------------------------------------


def test_screen_doses_returns_list():
    results = screen_doses("DrugM", [100.0, 10.0, 50.0])
    assert isinstance(results, list)
    assert len(results) == 3


def test_screen_doses_sorted_ascending():
    results = screen_doses("DrugN", [300.0, 10.0, 100.0, 50.0])
    doses = [r.dose_mg for r in results]
    assert doses == sorted(doses)


def test_screen_doses_all_results_correct_type():
    results = screen_doses("DrugO", [10.0, 20.0, 50.0])
    for r in results:
        assert isinstance(r, DoseBioavailabilityResult)


def test_screen_doses_kwargs_forwarded():
    results = screen_doses(
        "DrugP", [10.0, 50.0], saturable_absorption=False, saturable_firstpass=False
    )
    for r in results:
        assert r.mechanism == "linear"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_dose_bioavailability("Drug", dose_mg=0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg"):
        predict_dose_bioavailability("Drug", dose_mg=-10.0)


def test_validation_fa_max_zero():
    with pytest.raises(ValueError, match="fa_max"):
        predict_dose_bioavailability("Drug", dose_mg=10.0, fa_max=0.0)


def test_validation_fa_max_above_1():
    with pytest.raises(ValueError, match="fa_max"):
        predict_dose_bioavailability("Drug", dose_mg=10.0, fa_max=1.5)


def test_validation_Ka_mg_zero():
    with pytest.raises(ValueError, match="Ka_mg"):
        predict_dose_bioavailability("Drug", dose_mg=10.0, Ka_mg=0.0)


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        predict_dose_bioavailability("Drug", dose_mg=10.0, fu_plasma=0.0)


def test_validation_cl_int_zero():
    with pytest.raises(ValueError, match="cl_int_L_per_h"):
        predict_dose_bioavailability("Drug", dose_mg=10.0, cl_int_L_per_h=0.0)


def test_validation_Qh_zero():
    with pytest.raises(ValueError, match="Q_h_L_per_h"):
        predict_dose_bioavailability("Drug", dose_mg=10.0, Q_h_L_per_h=0.0)
