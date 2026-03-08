"""Tests for Phase 347 — Cyclodextrin Complexation Solubility Enhancement."""

import pytest

from omega_pbpk.prediction.cyclodextrin_complexation import (
    CyclodextrinComplexationResult,
    predict_cyclodextrin_complexation,
    screen_cd_types,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_result(**kwargs) -> CyclodextrinComplexationResult:
    defaults = dict(
        drug_name="TestDrug",
        intrinsic_solubility_mg_mL=0.1,
        logP=2.5,
        pKa=7.0,
        drug_type="neutral",
        cd_type="beta",
        cd_conc_mM=10.0,
        K11_per_mM=10.0,
        dose_mg=100.0,
    )
    defaults.update(kwargs)
    return predict_cyclodextrin_complexation(**defaults)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    r = _base_result()
    assert isinstance(r, CyclodextrinComplexationResult)


def test_result_is_frozen():
    r = _base_result()
    with pytest.raises((AttributeError, TypeError)):
        r.enhancement_ratio = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enhancement ratio ≥ 1.0
# ---------------------------------------------------------------------------


def test_enhancement_ratio_at_least_one():
    r = _base_result()
    assert r.enhancement_ratio >= 1.0


def test_solubility_with_cd_ge_intrinsic():
    r = _base_result()
    assert r.solubility_with_cd_mg_mL >= r.intrinsic_solubility_mg_mL


# ---------------------------------------------------------------------------
# Monotonicity: higher CD conc → higher enhancement
# ---------------------------------------------------------------------------


def test_higher_cd_conc_higher_enhancement():
    r_low = _base_result(cd_conc_mM=5.0)
    r_high = _base_result(cd_conc_mM=20.0)
    assert r_high.enhancement_ratio > r_low.enhancement_ratio


def test_higher_cd_conc_higher_solubility():
    r_low = _base_result(cd_conc_mM=5.0)
    r_high = _base_result(cd_conc_mM=20.0)
    assert r_high.solubility_with_cd_mg_mL > r_low.solubility_with_cd_mg_mL


# ---------------------------------------------------------------------------
# Monotonicity: higher K11 → higher enhancement
# ---------------------------------------------------------------------------


def test_higher_k11_higher_enhancement():
    r_low = _base_result(K11_per_mM=5.0)
    r_high = _base_result(K11_per_mM=20.0)
    assert r_high.enhancement_ratio > r_low.enhancement_ratio


# ---------------------------------------------------------------------------
# CD type corrections: beta > gamma
# ---------------------------------------------------------------------------


def test_beta_cd_higher_than_gamma():
    r_beta = _base_result(cd_type="beta")
    r_gamma = _base_result(cd_type="gamma")
    assert r_beta.k11_effective_per_mM > r_gamma.k11_effective_per_mM


def test_hp_beta_cd_higher_than_alpha():
    r_hp = _base_result(cd_type="HP_beta")
    r_alpha = _base_result(cd_type="alpha")
    assert r_hp.k11_effective_per_mM > r_alpha.k11_effective_per_mM


# ---------------------------------------------------------------------------
# dose_number > 0
# ---------------------------------------------------------------------------


def test_dose_number_positive():
    r = _base_result()
    assert r.dose_number_with_cd > 0.0


def test_dose_number_formula():
    r = _base_result(dose_mg=100.0)
    expected = 100.0 / (r.solubility_with_cd_mg_mL * 250.0)
    assert abs(r.dose_number_with_cd - expected) < 1e-9


# ---------------------------------------------------------------------------
# BCS dissolution class
# ---------------------------------------------------------------------------


def test_bcs_class_complete_dissolution_when_do_lt_1():
    # Very high solubility → Do < 1
    r = _base_result(intrinsic_solubility_mg_mL=10.0, cd_conc_mM=50.0, dose_mg=10.0)
    assert r.bcs_dissolution_class == "complete_dissolution"


def test_bcs_class_dissolution_limited_when_do_ge_1():
    # Very low solubility → Do ≥ 1
    r = _base_result(intrinsic_solubility_mg_mL=0.001, cd_conc_mM=1.0, K11_per_mM=1.0)
    assert r.bcs_dissolution_class == "dissolution_limited"


def test_bcs_dissolution_class_in_valid_set():
    r = _base_result()
    assert r.bcs_dissolution_class in ("complete_dissolution", "dissolution_limited")


# ---------------------------------------------------------------------------
# Formulation feasibility
# ---------------------------------------------------------------------------


def test_formulation_feasibility_feasible_high_enhancement():
    # Enhancement > 2 → feasible
    r = _base_result(cd_conc_mM=10.0, K11_per_mM=20.0)
    assert r.formulation_feasibility == "feasible"


def test_formulation_feasibility_marginal_moderate():
    # Enhancement between 1 and 2
    r = _base_result(cd_conc_mM=0.05, K11_per_mM=10.0)
    assert r.formulation_feasibility == "marginal"


def test_formulation_feasibility_in_valid_set():
    r = _base_result()
    assert r.formulation_feasibility in ("feasible", "marginal", "not_beneficial")


# ---------------------------------------------------------------------------
# Optimal CD concentration
# ---------------------------------------------------------------------------


def test_optimal_cd_conc_formula():
    r = _base_result()
    # optimal = 9 / K11_eff
    expected = 9.0 / r.k11_effective_per_mM
    assert abs(r.optimal_cd_conc_mM - expected) < 1e-9


# ---------------------------------------------------------------------------
# logP correction
# ---------------------------------------------------------------------------


def test_logP_outside_range_halves_k11():
    r_in = _base_result(logP=3.0)
    r_out = _base_result(logP=7.0)
    assert abs(r_out.k11_effective_per_mM - r_in.k11_effective_per_mM * 0.5) < 1e-9


# ---------------------------------------------------------------------------
# screen_cd_types
# ---------------------------------------------------------------------------


def test_screen_cd_types_returns_four_results():
    results = screen_cd_types(
        drug_name="TestDrug",
        intrinsic_sol=0.1,
        logP=2.5,
        pKa=7.0,
        drug_type="neutral",
        cd_conc_mM=10.0,
    )
    assert len(results) == 4


def test_screen_cd_types_sorted_descending():
    results = screen_cd_types(
        drug_name="TestDrug",
        intrinsic_sol=0.1,
        logP=2.5,
        pKa=7.0,
        drug_type="neutral",
        cd_conc_mM=10.0,
    )
    ratios = [r.enhancement_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_screen_cd_types_all_cd_types_present():
    results = screen_cd_types(
        drug_name="TestDrug",
        intrinsic_sol=0.1,
        logP=2.5,
        pKa=7.0,
        drug_type="neutral",
        cd_conc_mM=10.0,
    )
    cd_types = {r.cd_type for r in results}
    assert cd_types == {"alpha", "beta", "gamma", "HP_beta"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_intrinsic_sol_zero():
    with pytest.raises(ValueError, match="intrinsic_solubility_mg_mL"):
        _base_result(intrinsic_solubility_mg_mL=0.0)


def test_validation_intrinsic_sol_negative():
    with pytest.raises(ValueError):
        _base_result(intrinsic_solubility_mg_mL=-1.0)


def test_validation_cd_conc_zero():
    with pytest.raises(ValueError, match="cd_conc_mM"):
        _base_result(cd_conc_mM=0.0)


def test_validation_k11_zero():
    with pytest.raises(ValueError, match="K11_per_mM"):
        _base_result(K11_per_mM=0.0)


def test_validation_invalid_cd_type():
    with pytest.raises(ValueError, match="cd_type"):
        _base_result(cd_type="delta")


def test_validation_invalid_drug_type():
    with pytest.raises(ValueError, match="drug_type"):
        _base_result(drug_type="zwitterion")
