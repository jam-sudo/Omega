"""Tests for ASD drug release and supersaturation kinetics — Phase 1136."""

import pytest

from omega_pbpk.prediction.amorphous_solid_dispersion_p1136 import (
    _POLYMER_PARAMS,
    _VALID_POLYMERS,
    ASDReleaseResult,
    compare_polymers,
    predict_asd_release,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(**kwargs):
    defaults = dict(
        drug_name="TestDrug",
        intrinsic_solubility_mg_mL=0.1,
        polymer_type="HPMC-AS",
        logP=3.0,
        drug_loading_pct=20.0,
        t_end_h=4.0,
        n_points=40,
    )
    defaults.update(kwargs)
    return predict_asd_release(**defaults)


# ---------------------------------------------------------------------------
# Return type and fields
# ---------------------------------------------------------------------------


def test_returns_asd_release_result():
    assert isinstance(_run(), ASDReleaseResult)


def test_drug_name_preserved():
    result = _run(drug_name="Itraconazole")
    assert result.drug_name == "Itraconazole"


def test_polymer_type_preserved():
    result = _run(polymer_type="PVP")
    assert result.polymer_type == "PVP"


def test_intrinsic_solubility_preserved():
    result = _run(intrinsic_solubility_mg_mL=0.5)
    assert result.intrinsic_solubility_mg_mL == pytest.approx(0.5)


def test_logp_preserved():
    result = _run(logP=2.5)
    assert result.logP == pytest.approx(2.5)


def test_drug_loading_preserved():
    result = _run(drug_loading_pct=25.0)
    assert result.drug_loading_pct == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Profile properties
# ---------------------------------------------------------------------------


def test_times_is_tuple():
    result = _run()
    assert isinstance(result.times_h, tuple)


def test_concentration_is_tuple():
    result = _run()
    assert isinstance(result.concentration_mg_mL, tuple)


def test_times_and_concs_same_length():
    result = _run(n_points=40)
    assert len(result.times_h) == len(result.concentration_mg_mL) == 40


def test_times_start_at_zero():
    result = _run()
    assert result.times_h[0] == pytest.approx(0.0)


def test_concentration_at_t0_near_zero():
    # At t=0, spring term = 1 - exp(0) = 0
    result = _run()
    assert result.concentration_mg_mL[0] == pytest.approx(0.0, abs=1e-9)


def test_concentration_positive_after_start():
    result = _run()
    assert max(result.concentration_mg_mL) > 0.0


# ---------------------------------------------------------------------------
# Supersaturation
# ---------------------------------------------------------------------------


def test_supersaturation_ratio_above_one():
    result = _run()
    assert result.supersaturation_ratio > 1.0


def test_max_supersaturation_above_intrinsic():
    result = _run(intrinsic_solubility_mg_mL=0.1)
    assert result.max_supersaturation_mg_mL > 0.1


def test_supersaturation_ratio_consistent():
    result = _run(intrinsic_solubility_mg_mL=0.2)
    expected = result.max_supersaturation_mg_mL / 0.2
    assert result.supersaturation_ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Polymer comparison
# ---------------------------------------------------------------------------


def test_pvp_higher_msr_than_hpmc():
    # PVP msr_factor=5.0 > HPMC msr_factor=3.0
    pvp = _POLYMER_PARAMS["PVP"]["msr_factor"]
    hpmc = _POLYMER_PARAMS["HPMC"]["msr_factor"]
    assert pvp > hpmc


def test_hpmc_as_lowest_crystallization_rate():
    # k_cryst for HPMC-AS = 0.05, lowest of all
    kcryst_hpmc_as = _POLYMER_PARAMS["HPMC-AS"]["k_cryst"]
    for polymer, params in _POLYMER_PARAMS.items():
        if polymer != "HPMC-AS":
            assert kcryst_hpmc_as < params["k_cryst"]


def test_hpmc_as_longer_parachute_than_pvp():
    r_hpmc_as = _run(polymer_type="HPMC-AS", drug_loading_pct=20.0)
    r_pvp = _run(polymer_type="PVP", drug_loading_pct=20.0)
    # HPMC-AS has lower k_cryst → longer parachute duration
    assert r_hpmc_as.parachute_duration_h >= r_pvp.parachute_duration_h


# ---------------------------------------------------------------------------
# Drug loading penalty
# ---------------------------------------------------------------------------


def test_high_loading_reduces_supersaturation():
    r_low = _run(drug_loading_pct=20.0)
    r_high = _run(drug_loading_pct=50.0)
    # High loading (>30%) applies 0.8 penalty on MSR
    assert r_high.max_supersaturation_mg_mL < r_low.max_supersaturation_mg_mL


def test_loading_above_30_applies_penalty():
    # At exactly 31%, penalty should apply
    r_below = _run(drug_loading_pct=30.0)
    r_above = _run(drug_loading_pct=31.0)
    assert r_above.max_supersaturation_mg_mL < r_below.max_supersaturation_mg_mL


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def test_auc_positive():
    result = _run()
    assert result.auc_supersaturation_mg_h_per_mL > 0.0


def test_auc_increases_with_higher_solubility():
    r_low = _run(intrinsic_solubility_mg_mL=0.01)
    r_high = _run(intrinsic_solubility_mg_mL=1.0)
    assert r_high.auc_supersaturation_mg_h_per_mL > r_low.auc_supersaturation_mg_h_per_mL


# ---------------------------------------------------------------------------
# compare_polymers
# ---------------------------------------------------------------------------


def test_compare_returns_four_results():
    results = compare_polymers("DrugX", 0.1, 3.0, 20.0)
    assert len(results) == 4


def test_compare_sorted_by_auc_descending():
    results = compare_polymers("DrugX", 0.1, 3.0, 20.0)
    aucs = [r.auc_supersaturation_mg_h_per_mL for r in results]
    assert aucs == sorted(aucs, reverse=True)


def test_compare_all_are_asd_release_results():
    results = compare_polymers("DrugX", 0.1, 3.0, 20.0)
    for r in results:
        assert isinstance(r, ASDReleaseResult)


def test_compare_covers_all_polymers():
    results = compare_polymers("DrugX", 0.1, 3.0, 20.0)
    polymer_types = {r.polymer_type for r in results}
    assert polymer_types == set(_VALID_POLYMERS)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_invalid_solubility_zero_raises():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        _run(intrinsic_solubility_mg_mL=0.0)


def test_invalid_solubility_negative_raises():
    with pytest.raises(ValueError, match="intrinsic_solubility"):
        _run(intrinsic_solubility_mg_mL=-1.0)


def test_invalid_polymer_raises():
    with pytest.raises(ValueError, match="polymer_type"):
        _run(polymer_type="UnknownPolymer")


def test_invalid_loading_zero_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        _run(drug_loading_pct=0.0)


def test_invalid_loading_above_100_raises():
    with pytest.raises(ValueError, match="drug_loading_pct"):
        _run(drug_loading_pct=101.0)


def test_invalid_logp_too_low_raises():
    with pytest.raises(ValueError, match="logP"):
        _run(logP=-6.0)


def test_invalid_logp_too_high_raises():
    with pytest.raises(ValueError, match="logP"):
        _run(logP=11.0)
