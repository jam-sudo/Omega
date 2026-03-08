"""Tests for Phase 598 — Drug Salivary Excretion."""

import pytest

from omega_pbpk.prediction.salivary_drug_excretion_p598 import (
    SalivaryDrugExcretionResult,
    predict_salivary_excretion,
    saliva_monitoring_assessment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(**kwargs):
    params = dict(
        drug_name="TestDrug",
        plasma_conc_mg_L=10.0,
        fu_plasma=0.5,
        pka=None,
        drug_type="neutral",
        salivary_ph=6.8,
        plasma_ph=7.4,
        salivary_flow_mL_per_min=0.5,
    )
    params.update(kwargs)
    return predict_salivary_excretion(**params)


# ---------------------------------------------------------------------------
# 1. Return type
# ---------------------------------------------------------------------------


def test_returns_correct_type():
    result = _run()
    assert isinstance(result, SalivaryDrugExcretionResult)


# ---------------------------------------------------------------------------
# 2. Neutral drug: S/P ratio = fu_plasma
# ---------------------------------------------------------------------------


def test_neutral_drug_sp_ratio_equals_fu_plasma():
    fu = 0.4
    result = _run(fu_plasma=fu, drug_type="neutral")
    assert result.saliva_plasma_ratio == pytest.approx(fu)


def test_neutral_drug_ionization_correction_is_one():
    result = _run(drug_type="neutral")
    assert result.ionization_correction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Saliva concentration formula
# ---------------------------------------------------------------------------


def test_saliva_concentration_formula():
    plasma = 20.0
    fu = 0.6
    result = _run(plasma_conc_mg_L=plasma, fu_plasma=fu, drug_type="neutral")
    assert result.saliva_conc_mg_L == pytest.approx(plasma * fu)


# ---------------------------------------------------------------------------
# 4. fu_saliva is always 1.0
# ---------------------------------------------------------------------------


def test_fu_saliva_is_one():
    result = _run()
    assert result.fu_saliva == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Salivary clearance formula
# ---------------------------------------------------------------------------


def test_salivary_clearance_formula():
    flow = 0.8
    fu = 0.5
    result = _run(fu_plasma=fu, salivary_flow_mL_per_min=flow, drug_type="neutral")
    assert result.salivary_clearance_mL_per_min == pytest.approx(flow * fu)


# ---------------------------------------------------------------------------
# 6. Salivary flow stored correctly
# ---------------------------------------------------------------------------


def test_salivary_flow_stored():
    result = _run(salivary_flow_mL_per_min=0.7)
    assert result.salivary_flow_mL_per_min == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 7. Monitoring viability: S/P in [0.5, 2.0] → True
# ---------------------------------------------------------------------------


def test_monitoring_viable_when_ratio_in_range():
    # fu=1.0, neutral → S/P=1.0 → viable
    result = _run(fu_plasma=1.0, drug_type="neutral")
    assert result.is_salivary_monitor_viable is True


def test_monitoring_not_viable_when_ratio_too_low():
    # fu=0.1, neutral → S/P=0.1 → not viable
    result = _run(fu_plasma=0.1, drug_type="neutral")
    assert result.is_salivary_monitor_viable is False


def test_monitoring_not_viable_when_ratio_too_high():
    # Need S/P > 2.0. Use base with pka.
    # base: (1+10^(pka-7.4))/(1+10^(pka-6.8)) * fu
    # Choose pka=5.0, fu=1.0
    # numerator = 1 + 10^(5-7.4) = 1 + 10^(-2.4) ≈ 1.004
    # denominator = 1 + 10^(5-6.8) = 1 + 10^(-1.8) ≈ 1.016
    # ion = 1.004/1.016 < 1 → ratio < 1 → won't be > 2
    # Try acid with pka close to plasma_ph:
    # acid: (1+10^(7.4-pka))/(1+10^(6.8-pka)) * fu
    # pka=9.0, fu=1.0:
    # num = 1 + 10^(7.4-9)=1+10^-1.6 ≈ 1.025
    # den = 1 + 10^(6.8-9)=1+10^-2.2 ≈ 1.006
    # ion ≈ 1.019 → S/P ≈ 1.019 → still viable
    # For > 2.0 with acid: pka in range where ion_factor > 2
    # pka=7.0: num = 1+10^(7.4-7)=1+10^0.4 ≈ 3.512; den=1+10^(6.8-7)=1+10^-0.2 ≈ 1.631
    # ion ≈ 2.15 → S/P = fu*2.15; with fu=1 → S/P ≈ 2.15 → concentrated
    result = _run(fu_plasma=1.0, drug_type="acid", pka=7.0)
    assert result.is_salivary_monitor_viable is False
    assert result.excretion_class == "concentrated"


# ---------------------------------------------------------------------------
# 8. Excretion class boundaries
# ---------------------------------------------------------------------------


def test_excretion_class_low():
    result = _run(fu_plasma=0.3, drug_type="neutral")
    assert result.excretion_class == "low"


def test_excretion_class_similar():
    result = _run(fu_plasma=0.8, drug_type="neutral")
    assert result.excretion_class == "similar"


def test_excretion_class_concentrated():
    # acid drug, pka=7.0 gives ion_factor > 2
    result = _run(fu_plasma=1.0, drug_type="acid", pka=7.0)
    assert result.excretion_class == "concentrated"


# ---------------------------------------------------------------------------
# 9. Acid drug with pka < plasma_ph: ion_factor > 1 (less ionized in saliva)
# ---------------------------------------------------------------------------


def test_acid_drug_ionization_formula():
    pka = 6.0
    plasma_ph = 7.4
    salivary_ph = 6.8
    fu = 0.5
    result = _run(
        fu_plasma=fu, drug_type="acid", pka=pka, plasma_ph=plasma_ph, salivary_ph=salivary_ph
    )
    expected_ion = (1 + 10 ** (plasma_ph - pka)) / (1 + 10 ** (salivary_ph - pka))
    assert result.ionization_correction == pytest.approx(expected_ion, rel=1e-5)
    assert result.saliva_plasma_ratio == pytest.approx(fu * expected_ion, rel=1e-5)


# ---------------------------------------------------------------------------
# 10. Base drug ionization formula
# ---------------------------------------------------------------------------


def test_base_drug_ionization_formula():
    pka = 9.0
    plasma_ph = 7.4
    salivary_ph = 6.8
    fu = 0.7
    result = _run(
        fu_plasma=fu, drug_type="base", pka=pka, plasma_ph=plasma_ph, salivary_ph=salivary_ph
    )
    expected_ion = (1 + 10 ** (pka - plasma_ph)) / (1 + 10 ** (pka - salivary_ph))
    assert result.ionization_correction == pytest.approx(expected_ion, rel=1e-5)
    assert result.saliva_plasma_ratio == pytest.approx(fu * expected_ion, rel=1e-5)


# ---------------------------------------------------------------------------
# 11. pka=None for acid/base gives ion_factor=1.0
# ---------------------------------------------------------------------------


def test_acid_no_pka_gives_ion_factor_one():
    result = _run(drug_type="acid", pka=None)
    assert result.ionization_correction == pytest.approx(1.0)


def test_base_no_pka_gives_ion_factor_one():
    result = _run(drug_type="base", pka=None)
    assert result.ionization_correction == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 12. Validation: plasma_conc <= 0
# ---------------------------------------------------------------------------


def test_validation_plasma_conc_zero():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        _run(plasma_conc_mg_L=0.0)


def test_validation_plasma_conc_negative():
    with pytest.raises(ValueError, match="plasma_conc_mg_L"):
        _run(plasma_conc_mg_L=-5.0)


# ---------------------------------------------------------------------------
# 13. Validation: fu_plasma out of range
# ---------------------------------------------------------------------------


def test_validation_fu_plasma_zero():
    with pytest.raises(ValueError, match="fu_plasma"):
        _run(fu_plasma=0.0)


def test_validation_fu_plasma_greater_than_one():
    with pytest.raises(ValueError, match="fu_plasma"):
        _run(fu_plasma=1.5)


# ---------------------------------------------------------------------------
# 14. Validation: salivary_flow <= 0
# ---------------------------------------------------------------------------


def test_validation_salivary_flow_zero():
    with pytest.raises(ValueError, match="salivary_flow_mL_per_min"):
        _run(salivary_flow_mL_per_min=0.0)


# ---------------------------------------------------------------------------
# 15. Validation: invalid drug_type
# ---------------------------------------------------------------------------


def test_validation_invalid_drug_type():
    with pytest.raises(ValueError, match="drug_type"):
        _run(drug_type="zwitterion")


# ---------------------------------------------------------------------------
# 16. Drug name stored correctly
# ---------------------------------------------------------------------------


def test_drug_name_stored():
    result = _run(drug_name="Metformin")
    assert result.drug_name == "Metformin"


# ---------------------------------------------------------------------------
# 17. Plasma concentration stored correctly
# ---------------------------------------------------------------------------


def test_plasma_conc_stored():
    result = _run(plasma_conc_mg_L=15.0)
    assert result.plasma_conc_mg_L == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# 18. fu_plasma stored correctly
# ---------------------------------------------------------------------------


def test_fu_plasma_stored():
    result = _run(fu_plasma=0.65)
    assert result.fu_plasma == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# 19. Notes is a non-empty string
# ---------------------------------------------------------------------------


def test_notes_non_empty():
    result = _run()
    assert isinstance(result.notes, str)
    assert len(result.notes) > 0


# ---------------------------------------------------------------------------
# 20. Result is frozen (immutable)
# ---------------------------------------------------------------------------


def test_result_is_frozen():
    result = _run()
    with pytest.raises((AttributeError, TypeError)):
        result.drug_name = "Modified"


# ---------------------------------------------------------------------------
# 21. saliva_monitoring_assessment returns dict with correct keys
# ---------------------------------------------------------------------------


def test_monitoring_assessment_returns_dict():
    assessment = saliva_monitoring_assessment(
        drug_name="TestDrug",
        plasma_conc_mg_L=10.0,
        fu_plasma=0.8,
    )
    assert isinstance(assessment, dict)
    assert "viable" in assessment
    assert "ratio" in assessment
    assert "recommendation" in assessment


# ---------------------------------------------------------------------------
# 22. saliva_monitoring_assessment viable=True for neutral drug fu=0.8
# ---------------------------------------------------------------------------


def test_monitoring_assessment_viable():
    assessment = saliva_monitoring_assessment(
        drug_name="Caffeine",
        plasma_conc_mg_L=5.0,
        fu_plasma=0.8,
        drug_type="neutral",
    )
    assert assessment["viable"] is True
    assert (
        "viable" in assessment["recommendation"].lower()
        or "non-invasive" in assessment["recommendation"].lower()
    )


# ---------------------------------------------------------------------------
# 23. saliva_monitoring_assessment not viable for fu=0.1
# ---------------------------------------------------------------------------


def test_monitoring_assessment_not_viable():
    assessment = saliva_monitoring_assessment(
        drug_name="HighBinding",
        plasma_conc_mg_L=5.0,
        fu_plasma=0.1,
        drug_type="neutral",
    )
    assert assessment["viable"] is False
    assert "NOT" in assessment["recommendation"] or "not" in assessment["recommendation"].lower()


# ---------------------------------------------------------------------------
# 24. Boundary: fu_plasma=1.0 neutral → S/P=1.0, class="similar", viable
# ---------------------------------------------------------------------------


def test_full_unbound_neutral():
    result = _run(fu_plasma=1.0, drug_type="neutral")
    assert result.saliva_plasma_ratio == pytest.approx(1.0)
    assert result.excretion_class == "similar"
    assert result.is_salivary_monitor_viable is True


# ---------------------------------------------------------------------------
# 25. Salivary clearance proportional to flow rate
# ---------------------------------------------------------------------------


def test_salivary_clearance_proportional_to_flow():
    r1 = _run(salivary_flow_mL_per_min=0.5)
    r2 = _run(salivary_flow_mL_per_min=1.0)
    assert r2.salivary_clearance_mL_per_min == pytest.approx(2.0 * r1.salivary_clearance_mL_per_min)
