"""
Tests for Phase 384 — Drug Degradation in Biological Matrices
"""

import pytest

from omega_pbpk.prediction.biological_matrix_degradation import (
    MatrixDegradationResult,
    compare_matrices,
    predict_matrix_degradation,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_result(
    drug_name="TestDrug",
    matrix="plasma",
    logp=2.0,
    mw_da=300.0,
    n_ester_bonds=0,
    n_amide_bonds=0,
    temperature_C=37.0,
):
    return predict_matrix_degradation(
        drug_name=drug_name,
        matrix=matrix,
        logp=logp,
        mw_da=mw_da,
        n_ester_bonds=n_ester_bonds,
        n_amide_bonds=n_amide_bonds,
        temperature_C=temperature_C,
    )


# ---------------------------------------------------------------------------
# Test 1: Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, MatrixDegradationResult)


# ---------------------------------------------------------------------------
# Test 2: Result is frozen dataclass (immutable)
# ---------------------------------------------------------------------------


def test_frozen_dataclass():
    result = make_result()
    with pytest.raises((AttributeError, TypeError)):
        result.t50_min = 999.0


# ---------------------------------------------------------------------------
# Test 3: Ester bonds increase degradation rate
# ---------------------------------------------------------------------------


def test_esters_increase_degradation():
    no_ester = make_result(n_ester_bonds=0)
    with_ester = make_result(n_ester_bonds=3)
    assert with_ester.degradation_rate_per_min > no_ester.degradation_rate_per_min


# ---------------------------------------------------------------------------
# Test 4: Ester bonds decrease t50 (faster degradation)
# ---------------------------------------------------------------------------


def test_esters_decrease_t50():
    no_ester = make_result(n_ester_bonds=0)
    with_ester = make_result(n_ester_bonds=3)
    assert with_ester.t50_min < no_ester.t50_min


# ---------------------------------------------------------------------------
# Test 5: Microsomes degrade faster than plasma
# ---------------------------------------------------------------------------


def test_microsomes_faster_than_plasma():
    plasma_result = make_result(matrix="plasma")
    micro_result = make_result(matrix="microsomes")
    assert micro_result.degradation_rate_per_min > plasma_result.degradation_rate_per_min
    assert micro_result.t50_min < plasma_result.t50_min


# ---------------------------------------------------------------------------
# Test 6: Liver homogenate faster than whole blood
# ---------------------------------------------------------------------------


def test_liver_faster_than_blood():
    blood_result = make_result(matrix="whole_blood")
    liver_result = make_result(matrix="liver_homogenate")
    assert liver_result.degradation_rate_per_min > blood_result.degradation_rate_per_min


# ---------------------------------------------------------------------------
# Test 7: stability_class "stable" for t50 > 60
# ---------------------------------------------------------------------------


def test_stability_class_stable():
    # CSF has very low base rate => t50 should be very large
    result = make_result(matrix="cerebrospinal_fluid")
    assert result.stability_class == "stable"
    assert result.t50_min > 60.0


# ---------------------------------------------------------------------------
# Test 8: stability_class "labile" for microsomes with multiple esters
# ---------------------------------------------------------------------------


def test_stability_class_labile():
    result = predict_matrix_degradation(
        "LabileDrug",
        "microsomes",
        logp=3.0,
        mw_da=400.0,
        n_ester_bonds=4,
        n_amide_bonds=0,
        temperature_C=37.0,
    )
    assert result.stability_class == "labile"
    assert result.t50_min < 10.0


# ---------------------------------------------------------------------------
# Test 9: stability_class "moderate" for mid-range t50
# ---------------------------------------------------------------------------


def test_stability_class_moderate():
    from omega_pbpk.prediction.biological_matrix_degradation import _classify_stability

    # moderate: t50 >= 10 and t50 <= 60
    assert _classify_stability(30.0) == "moderate"
    assert _classify_stability(10.0) == "moderate"
    # boundary: exactly 60.0 falls to "moderate" (spec uses strict >60 for stable)
    assert _classify_stability(60.0) == "moderate"
    assert _classify_stability(60.1) == "stable"


# ---------------------------------------------------------------------------
# Test 10: Hydrolysis pathway for ester bonds in plasma
# ---------------------------------------------------------------------------


def test_hydrolysis_pathway_esters_plasma():
    result = predict_matrix_degradation(
        "EsterDrug",
        "plasma",
        logp=1.0,
        mw_da=300.0,
        n_ester_bonds=2,
        n_amide_bonds=0,
    )
    assert result.primary_degradation_pathway == "hydrolysis"


# ---------------------------------------------------------------------------
# Test 11: Oxidation pathway for microsomes (lipophilic, no esters)
# ---------------------------------------------------------------------------


def test_oxidation_pathway_microsomes():
    result = predict_matrix_degradation(
        "LipophilicDrug",
        "microsomes",
        logp=3.5,
        mw_da=400.0,
        n_ester_bonds=0,
        n_amide_bonds=1,
    )
    assert result.primary_degradation_pathway == "oxidation"


# ---------------------------------------------------------------------------
# Test 12: Enzymatic pathway for liver homogenate with no esters
# ---------------------------------------------------------------------------


def test_enzymatic_pathway_liver():
    result = predict_matrix_degradation(
        "EnzymaticDrug",
        "liver_homogenate",
        logp=0.5,
        mw_da=300.0,
        n_ester_bonds=0,
        n_amide_bonds=2,
    )
    # logp <= 1, no esters → enzymatic
    assert result.primary_degradation_pathway == "enzymatic"


# ---------------------------------------------------------------------------
# Test 13: Temperature effect — higher temp increases rate
# ---------------------------------------------------------------------------


def test_temperature_effect():
    result_37 = make_result(temperature_C=37.0)
    result_50 = make_result(temperature_C=50.0)
    assert result_50.degradation_rate_per_min > result_37.degradation_rate_per_min


# ---------------------------------------------------------------------------
# Test 14: Lower temperature decreases rate
# ---------------------------------------------------------------------------


def test_low_temperature_slows_degradation():
    result_37 = make_result(temperature_C=37.0)
    result_4 = make_result(temperature_C=4.0)
    assert result_4.degradation_rate_per_min < result_37.degradation_rate_per_min


# ---------------------------------------------------------------------------
# Test 15: q10_factor is physically reasonable (1.5 to 4.0)
# ---------------------------------------------------------------------------


def test_q10_factor_range():
    result = make_result()
    assert 1.5 <= result.q10_factor <= 4.0


# ---------------------------------------------------------------------------
# Test 16: 60-min remaining for stable drug > 50%
# ---------------------------------------------------------------------------


def test_percent_remaining_stable():
    result = make_result(matrix="cerebrospinal_fluid")
    assert result.percent_remaining_60min > 50.0


# ---------------------------------------------------------------------------
# Test 17: 60-min remaining for labile drug < 50%
# ---------------------------------------------------------------------------


def test_percent_remaining_labile():
    result = predict_matrix_degradation(
        "LabileDrug",
        "microsomes",
        logp=3.0,
        mw_da=400.0,
        n_ester_bonds=4,
        n_amide_bonds=0,
    )
    assert result.percent_remaining_60min < 50.0


# ---------------------------------------------------------------------------
# Test 18: t90 > t50
# ---------------------------------------------------------------------------


def test_t90_greater_than_t50():
    result = make_result()
    assert result.t90_min > result.t50_min


# ---------------------------------------------------------------------------
# Test 19: compare_matrices returns 7 results
# ---------------------------------------------------------------------------


def test_compare_returns_all_matrices():
    results = compare_matrices("Drug", logp=2.0, mw_da=300.0, n_ester_bonds=0, n_amide_bonds=0)
    assert len(results) == 7


# ---------------------------------------------------------------------------
# Test 20: compare_matrices returns sorted by t50 descending
# ---------------------------------------------------------------------------


def test_compare_sorted_descending():
    results = compare_matrices("Drug", logp=2.0, mw_da=300.0, n_ester_bonds=0, n_amide_bonds=0)
    t50s = [r.t50_min for r in results]
    assert t50s == sorted(t50s, reverse=True)


# ---------------------------------------------------------------------------
# Test 21: Validation — invalid matrix
# ---------------------------------------------------------------------------


def test_validation_invalid_matrix():
    with pytest.raises(ValueError, match="matrix"):
        predict_matrix_degradation("D", "urine", 2.0, 300.0, 0, 0)


# ---------------------------------------------------------------------------
# Test 22: Validation — temperature out of range (too low)
# ---------------------------------------------------------------------------


def test_validation_temperature_low():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_matrix_degradation("D", "plasma", 2.0, 300.0, 0, 0, temperature_C=3.0)


# ---------------------------------------------------------------------------
# Test 23: Validation — temperature out of range (too high)
# ---------------------------------------------------------------------------


def test_validation_temperature_high():
    with pytest.raises(ValueError, match="temperature_C"):
        predict_matrix_degradation("D", "plasma", 2.0, 300.0, 0, 0, temperature_C=61.0)


# ---------------------------------------------------------------------------
# Test 24: Validation — negative ester bonds
# ---------------------------------------------------------------------------


def test_validation_negative_esters():
    with pytest.raises(ValueError, match="n_ester_bonds"):
        predict_matrix_degradation("D", "plasma", 2.0, 300.0, -1, 0)


# ---------------------------------------------------------------------------
# Test 25: Validation — negative amide bonds
# ---------------------------------------------------------------------------


def test_validation_negative_amides():
    with pytest.raises(ValueError, match="n_amide_bonds"):
        predict_matrix_degradation("D", "plasma", 2.0, 300.0, 0, -1)


# ---------------------------------------------------------------------------
# Test 26: Validation — mw <= 0
# ---------------------------------------------------------------------------


def test_validation_mw():
    with pytest.raises(ValueError, match="mw_da"):
        predict_matrix_degradation("D", "plasma", 2.0, 0.0, 0, 0)


# ---------------------------------------------------------------------------
# Test 27: drug_name and matrix stored correctly
# ---------------------------------------------------------------------------


def test_fields_stored():
    result = predict_matrix_degradation(
        "Aspirin", "plasma", logp=1.2, mw_da=180.0, n_ester_bonds=1, n_amide_bonds=0
    )
    assert result.drug_name == "Aspirin"
    assert result.matrix == "plasma"
    assert result.activation_energy_kJ_per_mol == 50.0


# ---------------------------------------------------------------------------
# Test 28: Amide bonds contribute to degradation rate
# ---------------------------------------------------------------------------


def test_amides_contribute_to_rate():
    no_amide = make_result(n_amide_bonds=0)
    with_amide = make_result(n_amide_bonds=3)
    assert with_amide.degradation_rate_per_min > no_amide.degradation_rate_per_min


# ---------------------------------------------------------------------------
# Test 29: Lipophilicity contribution (logP > 2 increases rate)
# ---------------------------------------------------------------------------


def test_lipophilicity_contribution():
    low_logp = make_result(logp=0.0)
    high_logp = make_result(logp=5.0)
    assert high_logp.degradation_rate_per_min > low_logp.degradation_rate_per_min


# ---------------------------------------------------------------------------
# Test 30: combined pathway for gastric_fluid with no esters
# ---------------------------------------------------------------------------


def test_combined_pathway_gastric():
    result = predict_matrix_degradation(
        "GastricDrug",
        "gastric_fluid",
        logp=1.5,
        mw_da=300.0,
        n_ester_bonds=0,
        n_amide_bonds=1,
    )
    assert result.primary_degradation_pathway == "combined"
