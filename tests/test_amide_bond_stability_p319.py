"""Tests for Phase 319 — Amide Bond Stability Predictor."""

import pytest

from omega_pbpk.prediction.amide_bond_stability_p319 import (
    AmideStabilityResult,
    compare_conditions,
    predict_amide_stability,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_KWARGS = dict(
    drug_name="TestDrug",
    n_amide_bonds=2,
    logp=1.0,
    mw=350.0,
    is_peptide=False,
    steric_protection_score=0,
    ph_condition="neutral",
    temperature_C=37.0,
    incubation_h=24.0,
)


def make_result(**overrides) -> AmideStabilityResult:
    kwargs = {**BASE_KWARGS, **overrides}
    return predict_amide_stability(**kwargs)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_return_type():
    result = make_result()
    assert isinstance(result, AmideStabilityResult)


# ---------------------------------------------------------------------------
# stability_pct in [0, 100]
# ---------------------------------------------------------------------------


def test_stability_pct_range_neutral():
    r = make_result(ph_condition="neutral")
    assert 0.0 <= r.stability_pct <= 100.0


def test_stability_pct_range_acidic():
    r = make_result(ph_condition="acidic")
    assert 0.0 <= r.stability_pct <= 100.0


def test_stability_pct_range_basic():
    r = make_result(ph_condition="basic")
    assert 0.0 <= r.stability_pct <= 100.0


def test_stability_pct_range_enzymatic():
    r = make_result(ph_condition="enzymatic")
    assert 0.0 <= r.stability_pct <= 100.0


# ---------------------------------------------------------------------------
# Condition ordering: basic > neutral in degradation rate
# ---------------------------------------------------------------------------


def test_basic_less_stable_than_neutral():
    r_neutral = make_result(ph_condition="neutral", incubation_h=48.0)
    r_basic = make_result(ph_condition="basic", incubation_h=48.0)
    assert r_basic.stability_pct < r_neutral.stability_pct


# ---------------------------------------------------------------------------
# Enzymatic → highest k (least stable)
# ---------------------------------------------------------------------------


def test_enzymatic_least_stable():
    conditions = ["acidic", "neutral", "basic", "enzymatic"]
    results = {c: make_result(ph_condition=c, incubation_h=48.0) for c in conditions}
    assert results["enzymatic"].effective_k_per_h >= results["acidic"].effective_k_per_h
    assert results["enzymatic"].effective_k_per_h >= results["neutral"].effective_k_per_h
    assert results["enzymatic"].effective_k_per_h >= results["basic"].effective_k_per_h


# ---------------------------------------------------------------------------
# More amide bonds → lower stability
# ---------------------------------------------------------------------------


def test_more_amide_bonds_less_stable():
    r1 = make_result(n_amide_bonds=1, incubation_h=48.0)
    r4 = make_result(n_amide_bonds=4, incubation_h=48.0)
    assert r4.stability_pct < r1.stability_pct


# ---------------------------------------------------------------------------
# Steric protection → higher stability
# ---------------------------------------------------------------------------


def test_steric_protection_increases_stability():
    r0 = make_result(steric_protection_score=0, incubation_h=24.0)
    r3 = make_result(steric_protection_score=3, incubation_h=24.0)
    assert r3.stability_pct > r0.stability_pct


# ---------------------------------------------------------------------------
# Peptide → lower stability
# ---------------------------------------------------------------------------


def test_peptide_less_stable():
    r_nopep = make_result(is_peptide=False, incubation_h=24.0)
    r_pep = make_result(is_peptide=True, incubation_h=24.0)
    assert r_pep.stability_pct < r_nopep.stability_pct


# ---------------------------------------------------------------------------
# half_life_h > 0
# ---------------------------------------------------------------------------


def test_half_life_positive():
    r = make_result()
    assert r.half_life_h > 0.0


# ---------------------------------------------------------------------------
# stability_class in valid set
# ---------------------------------------------------------------------------


def test_stability_class_valid():
    valid = {"stable", "moderately_stable", "labile"}
    for cond in ["acidic", "neutral", "basic", "enzymatic"]:
        r = make_result(ph_condition=cond)
        assert r.stability_class in valid, f"Unexpected class '{r.stability_class}' for {cond}"


# ---------------------------------------------------------------------------
# stability_class boundaries
# ---------------------------------------------------------------------------


def test_stability_class_stable():
    # neutral, 1 amide bond, max steric protection, short incubation → should be stable
    r = make_result(
        ph_condition="neutral", n_amide_bonds=1, steric_protection_score=3, incubation_h=1.0
    )
    assert r.stability_class == "stable"


def test_stability_class_labile():
    # enzymatic, many amide bonds, peptide, long incubation → labile
    r = make_result(ph_condition="enzymatic", n_amide_bonds=5, is_peptide=True, incubation_h=200.0)
    assert r.stability_class == "labile"


# ---------------------------------------------------------------------------
# Lipophilicity protection
# ---------------------------------------------------------------------------


def test_logp_above_2_increases_stability():
    r_hydrophilic = make_result(logp=0.0, incubation_h=24.0)
    r_lipophilic = make_result(logp=3.0, incubation_h=24.0)
    assert r_lipophilic.stability_pct > r_hydrophilic.stability_pct


# ---------------------------------------------------------------------------
# Temperature effect: lower temp → more stable (less degradation)
# ---------------------------------------------------------------------------


def test_lower_temperature_more_stable():
    r_cold = make_result(temperature_C=4.0, incubation_h=24.0)
    r_warm = make_result(temperature_C=37.0, incubation_h=24.0)
    assert r_cold.stability_pct > r_warm.stability_pct


# ---------------------------------------------------------------------------
# compare_conditions sorted descending
# ---------------------------------------------------------------------------


def test_compare_conditions_returns_list():
    results = compare_conditions(
        drug_name="DrugX",
        n_amide_bonds=2,
        logp=1.0,
        mw=300.0,
        is_peptide=False,
        steric_protection_score=1,
        incubation_h=24.0,
    )
    assert isinstance(results, list)
    assert len(results) == 4


def test_compare_conditions_sorted_descending():
    results = compare_conditions(
        drug_name="DrugX",
        n_amide_bonds=2,
        logp=1.0,
        mw=300.0,
        is_peptide=False,
        steric_protection_score=1,
        incubation_h=24.0,
    )
    for i in range(1, len(results)):
        assert results[i - 1].stability_pct >= results[i].stability_pct


def test_compare_conditions_all_types():
    results = compare_conditions(
        drug_name="Peptide",
        n_amide_bonds=3,
        logp=0.5,
        mw=500.0,
        is_peptide=True,
        steric_protection_score=0,
        incubation_h=48.0,
    )
    conditions_found = {r.ph_condition for r in results}
    assert conditions_found == {"acidic", "neutral", "basic", "enzymatic"}


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_n_amide_bonds_zero():
    with pytest.raises(ValueError, match="n_amide_bonds"):
        make_result(n_amide_bonds=0)


def test_validation_logp_out_of_range():
    with pytest.raises(ValueError, match="logp"):
        make_result(logp=15.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        make_result(mw=0.0)


def test_validation_temperature_out_of_range():
    with pytest.raises(ValueError, match="temperature_C"):
        make_result(temperature_C=150.0)


def test_validation_incubation_h_zero():
    with pytest.raises(ValueError, match="incubation_h"):
        make_result(incubation_h=0.0)


def test_validation_invalid_ph_condition():
    with pytest.raises(ValueError, match="ph_condition"):
        make_result(ph_condition="plasma")


def test_validation_invalid_steric_score():
    with pytest.raises(ValueError, match="steric_protection_score"):
        make_result(steric_protection_score=5)
