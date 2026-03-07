"""Tests for Phase 867: Intracellular Drug Accumulation."""

import pytest

from omega_pbpk.core.intracellular_accumulation import (
    IntracellularAccumulationResult,
    compare_ionization_types,
    simulate_intracellular_accumulation,
)

# --- Type and field checks ---


def test_returns_correct_type():
    result = simulate_intracellular_accumulation("DrugA", 100.0)
    assert isinstance(result, IntracellularAccumulationResult)


def test_fields_preserved():
    result = simulate_intracellular_accumulation(
        "DrugB", 50.0, ionization_type="base", pka_base=9.0
    )
    assert result.drug_name == "DrugB"
    assert result.ionization_type == "base"
    assert result.pka_base == 9.0


def test_times_list_length():
    result = simulate_intracellular_accumulation("DrugA", 100.0, t_end_h=10.0, dt_h=0.5)
    assert len(result.times_h) == len(result.c_plasma_mg_L)
    assert len(result.times_h) == len(result.c_cytosol_mg_L)
    assert len(result.times_h) == len(result.c_lysosome_mg_L)


def test_times_start_at_zero():
    result = simulate_intracellular_accumulation("DrugA", 100.0)
    assert result.times_h[0] == pytest.approx(0.0)


# --- pH trapping / Kp_lysosome ---


def test_neutral_drug_kp_lysosome_equals_1():
    result = simulate_intracellular_accumulation("NeutralDrug", 100.0, ionization_type="neutral")
    assert result.kp_lysosome == pytest.approx(1.0)


def test_acid_drug_kp_lysosome_equals_1():
    result = simulate_intracellular_accumulation(
        "AcidDrug", 100.0, ionization_type="acid", pka_base=4.0
    )
    assert result.kp_lysosome == pytest.approx(1.0)


def test_basic_drug_high_pka_high_kp():
    # Basic drug with high pKa → strong pH trapping in lysosomes
    result = simulate_intracellular_accumulation(
        "HighBaseDrug", 100.0, ionization_type="base", pka_base=10.0
    )
    assert result.kp_lysosome > 10.0


def test_basic_drug_kp_greater_than_neutral():
    r_base = simulate_intracellular_accumulation(
        "Drug", 100.0, ionization_type="base", pka_base=9.0
    )
    r_neutral = simulate_intracellular_accumulation("Drug", 100.0, ionization_type="neutral")
    assert r_base.kp_lysosome > r_neutral.kp_lysosome


# --- Cmax checks ---


def test_cmax_plasma_positive():
    result = simulate_intracellular_accumulation("DrugA", 100.0)
    assert result.cmax_plasma > 0.0


def test_cmax_cytosol_positive():
    result = simulate_intracellular_accumulation("DrugA", 100.0)
    assert result.cmax_cytosol > 0.0


def test_cmax_lysosome_positive_for_base():
    result = simulate_intracellular_accumulation(
        "DrugA", 100.0, ionization_type="base", pka_base=9.0
    )
    assert result.cmax_lysosome > 0.0


# --- AUC and ratio ---


def test_auc_plasma_positive():
    result = simulate_intracellular_accumulation("DrugA", 100.0)
    assert result.auc_plasma > 0.0


def test_lysosome_to_plasma_ratio_positive():
    result = simulate_intracellular_accumulation(
        "DrugA", 100.0, ionization_type="base", pka_base=9.0
    )
    assert result.lysosome_to_plasma_ratio > 0.0


def test_base_drug_higher_lyso_ratio_than_neutral():
    r_base = simulate_intracellular_accumulation(
        "Drug", 100.0, ionization_type="base", pka_base=9.0
    )
    r_neutral = simulate_intracellular_accumulation("Drug", 100.0, ionization_type="neutral")
    assert r_base.lysosome_to_plasma_ratio > r_neutral.lysosome_to_plasma_ratio


# --- Accumulation class ---


def test_accumulation_class_negligible_for_neutral():
    result = simulate_intracellular_accumulation("Neutral", 100.0, ionization_type="neutral")
    assert result.accumulation_class == "negligible"


def test_accumulation_class_extensive_for_high_pka_base():
    result = simulate_intracellular_accumulation(
        "HighBase", 100.0, ionization_type="base", pka_base=10.0
    )
    assert result.accumulation_class == "extensive"


def test_accumulation_class_values():
    # kp < 2 → negligible
    r1 = simulate_intracellular_accumulation("D", 100.0, ionization_type="neutral")
    assert r1.accumulation_class in {"negligible", "moderate", "extensive"}


# --- compare_ionization_types ---


def test_compare_ionization_types_returns_list():
    results = compare_ionization_types("Drug", 100.0, pka_base=9.0)
    assert isinstance(results, list)
    assert len(results) == 3


def test_compare_ionization_types_contains_all_types():
    results = compare_ionization_types("Drug", 100.0, pka_base=9.0)
    types = {r.ionization_type for r in results}
    assert types == {"base", "acid", "neutral"}


def test_compare_ionization_types_sorted_descending():
    results = compare_ionization_types("Drug", 100.0, pka_base=9.0)
    ratios = [r.lysosome_to_plasma_ratio for r in results]
    assert ratios == sorted(ratios, reverse=True)


def test_compare_ionization_types_base_has_highest_ratio():
    results = compare_ionization_types("Drug", 100.0, pka_base=9.0)
    assert results[0].ionization_type == "base"


# --- Validation ---


def test_validation_dose_zero():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_intracellular_accumulation("D", 0.0)


def test_validation_dose_negative():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        simulate_intracellular_accumulation("D", -10.0)


def test_validation_invalid_ionization_type():
    with pytest.raises(ValueError):
        simulate_intracellular_accumulation("D", 100.0, ionization_type="zwitterion")


def test_validation_vd_zero():
    with pytest.raises(ValueError, match="vd_L must be > 0"):
        simulate_intracellular_accumulation("D", 100.0, vd_L=0.0)


def test_validation_cl_zero():
    with pytest.raises(ValueError, match="cl_L_per_h must be > 0"):
        simulate_intracellular_accumulation("D", 100.0, cl_L_per_h=0.0)
