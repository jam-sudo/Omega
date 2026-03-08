"""Tests for Phase 1033 — Drug Nasal Bioavailability (nasal_bioavailability_p1033.py)."""

import pytest

from omega_pbpk.prediction.nasal_bioavailability_p1033 import (
    NasalBioavailabilityResult,
    compare_formulations,
    predict_nasal_bioavailability,
)


def _make_result(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        mw_Da=300.0,
        logp=2.0,
        pka=7.0,
        ionization_type="neutral",
        formulation="solution",
        dose_mg=1.0,
        fu_plasma=0.3,
        cl_hepatic_L_per_h=10.0,
        q_hepatic_L_per_h=90.0,
    )
    defaults.update(kwargs)
    return predict_nasal_bioavailability(**defaults)


# --- Return type ---


def test_return_type():
    result = _make_result()
    assert isinstance(result, NasalBioavailabilityResult)


def test_frozen_dataclass():
    result = _make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.fa_nasal = 0.5  # type: ignore[misc]


# --- Value ranges ---


def test_fa_nasal_in_range():
    result = _make_result()
    assert 0 < result.fa_nasal <= 1.0


def test_f_systemic_in_range():
    result = _make_result()
    assert 0 <= result.f_systemic <= 1.0


def test_deposition_fraction_in_range():
    result = _make_result()
    assert 0 < result.deposition_fraction < 1.0


def test_t_max_absorption_positive():
    result = _make_result()
    assert result.t_max_absorption_h > 0


def test_cl_mucociliary_is_30():
    result = _make_result()
    assert result.cl_mucociliary_per_h == 30.0


def test_bioavailability_class_valid():
    result = _make_result()
    assert result.bioavailability_class in {"high", "moderate", "low"}


# --- Physics checks ---


def test_low_mw_high_logp_higher_fa_than_high_mw_low_logp():
    """Low MW + high logP should give higher fa_nasal than high MW + low logP."""
    low_mw_high_logp = _make_result(mw_Da=150.0, logp=3.0)
    high_mw_low_logp = _make_result(mw_Da=800.0, logp=-1.0)
    assert low_mw_high_logp.fa_nasal > high_mw_low_logp.fa_nasal


def test_solution_higher_f_systemic_than_powder():
    """Solution has higher deposition fraction than powder, so higher f_systemic."""
    sol = _make_result(formulation="solution")
    pwd = _make_result(formulation="powder")
    assert sol.f_systemic > pwd.f_systemic


def test_high_hepatic_cl_lower_f_systemic():
    """High hepatic CL → higher E_h → lower f_hepatic → lower f_systemic."""
    low_cl = _make_result(cl_hepatic_L_per_h=5.0)
    high_cl = _make_result(cl_hepatic_L_per_h=80.0)
    assert high_cl.f_systemic < low_cl.f_systemic


def test_bioavailability_class_best_case_moderate():
    """Best-case scenario (low MW, high logP, solution, near-zero hepatic CL)
    gives moderate class — model cap of k_abs=30=/h limits fa_nasal to 0.5 max,
    so f_systemic cannot exceed 0.5*0.85=0.425 (below 'high' threshold of 0.5)."""
    result = predict_nasal_bioavailability(
        drug_name="Ideal",
        mw_Da=200.0,
        logp=3.0,
        ionization_type="neutral",
        formulation="solution",
        cl_hepatic_L_per_h=1.0,
        q_hepatic_L_per_h=90.0,
    )
    assert result.bioavailability_class == "moderate"


def test_bioavailability_class_low():
    """Very unfavorable properties should yield 'low' class."""
    result = predict_nasal_bioavailability(
        drug_name="Poor",
        mw_Da=1500.0,
        logp=-2.0,
        ionization_type="acid",
        pka=4.0,
        formulation="powder",
        cl_hepatic_L_per_h=85.0,
        q_hepatic_L_per_h=90.0,
    )
    assert result.bioavailability_class == "low"


# --- Formulation factors ---


def test_formulation_factors():
    """Each formulation has expected formulation_factor."""
    expected = {"solution": 1.0, "spray": 0.95, "powder": 0.85, "gel": 0.90}
    for form, expected_ff in expected.items():
        result = _make_result(formulation=form)
        assert abs(result.formulation_factor - expected_ff) < 1e-9, (
            f"{form}: {result.formulation_factor}"
        )


# --- compare_formulations ---


def test_compare_formulations_returns_4():
    results = compare_formulations("TestDrug", mw_Da=300.0, logp=2.0)
    assert len(results) == 4


def test_compare_formulations_sorted_descending():
    results = compare_formulations("TestDrug", mw_Da=300.0, logp=2.0)
    f_systemics = [r.f_systemic for r in results]
    assert f_systemics == sorted(f_systemics, reverse=True)


def test_compare_formulations_all_nasal_bioavailability_result():
    results = compare_formulations("TestDrug", mw_Da=300.0, logp=2.0)
    for r in results:
        assert isinstance(r, NasalBioavailabilityResult)


# --- Notes ---


def test_notes_nonempty():
    result = _make_result()
    assert len(result.notes) > 0


# --- Validation ---


def test_validation_mw_zero_raises():
    with pytest.raises(ValueError, match="mw_Da"):
        _make_result(mw_Da=0.0)


def test_validation_dose_zero_raises():
    with pytest.raises(ValueError, match="dose_mg"):
        _make_result(dose_mg=0.0)


def test_validation_invalid_formulation_raises():
    with pytest.raises(ValueError, match="formulation"):
        _make_result(formulation="inhaler")


def test_validation_invalid_ionization_type_raises():
    with pytest.raises(ValueError, match="ionization_type"):
        _make_result(ionization_type="zwitterion")


def test_validation_fu_plasma_zero_raises():
    with pytest.raises(ValueError, match="fu_plasma"):
        _make_result(fu_plasma=0.0)


def test_validation_cl_hepatic_zero_raises():
    with pytest.raises(ValueError, match="cl_hepatic"):
        _make_result(cl_hepatic_L_per_h=0.0)


def test_validation_q_hepatic_zero_raises():
    with pytest.raises(ValueError, match="q_hepatic"):
        _make_result(q_hepatic_L_per_h=0.0)
