"""Tests for Phase 1050 — Drug Solubility Enhancement Techniques."""

import pytest

from omega_pbpk.prediction.solubility_enhancement_p1050 import (
    SolubilityEnhancementResult,
    compare_enhancement_techniques,
    predict_solubility_enhancement,
)


def default_result(**kwargs):
    params = dict(
        drug_name="TestDrug",
        mw_Da=350.0,
        logp=3.0,
        melting_point_C=200.0,
        pka=7.0,
        ionization_type="neutral",
        technique="amorphous_dispersion",
    )
    params.update(kwargs)
    return predict_solubility_enhancement(**params)


def test_return_type_frozen():
    result = default_result()
    assert isinstance(result, SolubilityEnhancementResult)
    # frozen dataclass: cannot assign
    with pytest.raises((AttributeError, TypeError)):
        result.enhancement_fold = 999.0  # type: ignore[misc]


def test_enhanced_gte_baseline():
    result = default_result()
    assert result.enhanced_solubility_mg_mL >= result.baseline_solubility_mg_mL


def test_enhancement_fold_gte_1():
    result = default_result()
    assert result.enhancement_fold >= 1.0


def test_mechanism_valid():
    valid = {"amorphous", "complexation", "particle_size", "salt_form", "cosolvency"}
    for tech in [
        "amorphous_dispersion",
        "cyclodextrin_complexation",
        "particle_size_reduction",
        "salt_form",
        "cosolvency",
    ]:
        r = default_result(technique=tech)
        assert r.mechanism in valid


def test_stability_risk_valid():
    result = default_result()
    assert result.stability_risk in {"low", "moderate", "high"}


def test_feasibility_in_range():
    result = default_result()
    assert 0.0 <= result.feasibility_score <= 100.0


def test_recommended_logic():
    result = default_result()
    expected = result.enhancement_fold >= 10.0 and result.feasibility_score >= 50.0
    assert result.recommended == expected


def test_amorphous_high_tm_higher_enhancement():
    r_high = default_result(melting_point_C=300.0)
    r_low = default_result(melting_point_C=110.0)
    assert r_high.enhancement_fold > r_low.enhancement_fold


def test_salt_form_ionizable_higher_than_neutral():
    r_ion = predict_solubility_enhancement(
        drug_name="X",
        mw_Da=300.0,
        logp=2.0,
        melting_point_C=200.0,
        pka=4.0,
        ionization_type="acid",
        technique="salt_form",
    )
    r_neu = predict_solubility_enhancement(
        drug_name="X",
        mw_Da=300.0,
        logp=2.0,
        melting_point_C=200.0,
        pka=4.0,
        ionization_type="neutral",
        technique="salt_form",
    )
    assert r_ion.enhancement_fold > r_neu.enhancement_fold


def test_cosolvency_higher_logp_higher_enhancement():
    r_high = predict_solubility_enhancement(
        drug_name="X",
        mw_Da=300.0,
        logp=5.0,
        melting_point_C=200.0,
        technique="cosolvency",
        cosolvent_pct=20.0,
    )
    r_low = predict_solubility_enhancement(
        drug_name="X",
        mw_Da=300.0,
        logp=1.0,
        melting_point_C=200.0,
        technique="cosolvency",
        cosolvent_pct=20.0,
    )
    assert r_high.enhancement_fold >= r_low.enhancement_fold


def test_compare_returns_5():
    results = compare_enhancement_techniques(
        drug_name="Drug", mw_Da=350.0, logp=3.0, melting_point_C=200.0
    )
    assert len(results) == 5


def test_compare_sorted_desc():
    results = compare_enhancement_techniques(
        drug_name="Drug", mw_Da=350.0, logp=3.0, melting_point_C=200.0
    )
    folds = [r.enhancement_fold for r in results]
    assert folds == sorted(folds, reverse=True)


def test_predicted_bioavailability_gain_range():
    result = default_result()
    assert 0.0 <= result.predicted_bioavailability_gain <= 0.5


def test_notes_nonempty():
    result = default_result()
    assert len(result.notes) > 0


def test_technique_stored():
    result = default_result()
    assert result.technique == "amorphous_dispersion"


def test_drug_name_stored():
    result = default_result()
    assert result.drug_name == "TestDrug"


def test_cyclodextrin_returns_complexation_mechanism():
    r = default_result(technique="cyclodextrin_complexation")
    assert r.mechanism == "complexation"


def test_particle_size_returns_particle_size_mechanism():
    r = default_result(technique="particle_size_reduction")
    assert r.mechanism == "particle_size"


def test_cosolvency_returns_cosolvency_mechanism():
    r = default_result(technique="cosolvency")
    assert r.mechanism == "cosolvency"


def test_validation_mw_zero():
    with pytest.raises(ValueError):
        predict_solubility_enhancement(drug_name="X", mw_Da=0.0, logp=2.0, melting_point_C=200.0)


def test_validation_invalid_technique():
    with pytest.raises(ValueError):
        predict_solubility_enhancement(
            drug_name="X", mw_Da=300.0, logp=2.0, melting_point_C=200.0, technique="magic_powder"
        )


def test_validation_cosolvent_out_of_range():
    with pytest.raises(ValueError):
        predict_solubility_enhancement(
            drug_name="X",
            mw_Da=300.0,
            logp=2.0,
            melting_point_C=200.0,
            technique="cosolvency",
            cosolvent_pct=150.0,
        )
