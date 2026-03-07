"""Tests for Phase 1096 — Drug Serum Albumin Binding Simulation."""

import pytest

from omega_pbpk.prediction.albumin_binding_simulation import (
    AlbuminBindingResult,
    compare_affinities,
    fu_concentration_curve,
    predict_albumin_binding,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def default_result() -> AlbuminBindingResult:
    return predict_albumin_binding("Warfarin", drug_conc_uM=1.0)


# ── Return type ───────────────────────────────────────────────────────────────

def test_return_type():
    res = default_result()
    assert isinstance(res, AlbuminBindingResult)


def test_frozen_dataclass():
    """AlbuminBindingResult must be immutable."""
    res = default_result()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        res.fu_predicted = 0.5  # type: ignore[misc]


# ── fu range ─────────────────────────────────────────────────────────────────

def test_fu_in_valid_range():
    res = default_result()
    assert 0.0 < res.fu_predicted <= 1.0


def test_fu_no_albumin_is_one():
    """When albumin = 0, all drug is free: fu = 1.0."""
    res = predict_albumin_binding("Drug", drug_conc_uM=1.0, albumin_conc_g_per_L=0.0)
    assert res.fu_predicted == pytest.approx(1.0)


def test_fu_high_conc_approaches_one():
    """At saturating drug concentrations fu approaches 1.0."""
    res_low = predict_albumin_binding("Drug", drug_conc_uM=0.1)
    res_high = predict_albumin_binding("Drug", drug_conc_uM=1e6)
    assert res_high.fu_predicted > res_low.fu_predicted


# ── Higher Ka → lower fu ──────────────────────────────────────────────────────

def test_higher_ka_lower_fu():
    """Stronger binding (higher Ka) should give lower free fraction."""
    res_weak = predict_albumin_binding("Drug", drug_conc_uM=1.0, ka_site1_L_per_uM=0.1)
    res_strong = predict_albumin_binding("Drug", drug_conc_uM=1.0, ka_site1_L_per_uM=10.0)
    assert res_strong.fu_predicted < res_weak.fu_predicted


# ── Binding amounts ───────────────────────────────────────────────────────────

def test_bound_positive_with_albumin():
    res = default_result()
    assert res.b_site1_uM + res.b_site2_uM > 0.0


def test_b_site1_zero_when_site_II_only():
    res = predict_albumin_binding("Drug", drug_conc_uM=1.0, primary_site="site_II")
    assert res.b_site1_uM == pytest.approx(0.0)


def test_b_site2_zero_when_site_I_only():
    res = predict_albumin_binding("Drug", drug_conc_uM=1.0, primary_site="site_I")
    assert res.b_site2_uM == pytest.approx(0.0)


def test_fraction_site1_in_range():
    res = default_result()
    assert 0.0 <= res.fraction_site1 <= 1.0


def test_fraction_site1_is_one_when_only_site_I():
    res = predict_albumin_binding("Drug", drug_conc_uM=1.0, primary_site="site_I")
    assert res.fraction_site1 == pytest.approx(1.0)


def test_fraction_site1_is_zero_when_only_site_II():
    res = predict_albumin_binding("Drug", drug_conc_uM=1.0, primary_site="site_II")
    assert res.fraction_site1 == pytest.approx(0.0)


# ── Nonlinear binding detection ───────────────────────────────────────────────

def test_nonlinear_binding_high_Ka():
    """High-affinity drug at concentration comparable to albumin capacity shows nonlinear binding.

    Albumin ≈ 526 µM. At 100 µM drug with Ka=20 the binding sites are significantly
    occupied, so at 10× (1000 µM) fu changes >10% — nonlinear.
    """
    res = predict_albumin_binding("Warfarin", drug_conc_uM=100.0, ka_site1_L_per_uM=20.0)
    assert res.nonlinear_binding is True


def test_nonlinear_binding_low_Ka():
    """Very low affinity drug: binding barely changes with concentration."""
    res = predict_albumin_binding("Drug", drug_conc_uM=1.0, ka_site1_L_per_uM=0.001, ka_site2_L_per_uM=0.001)
    assert res.nonlinear_binding is False


def test_concentration_dependent_fu_string_valid():
    res = default_result()
    assert res.concentration_dependent_fu in ("linear", "nonlinear")


def test_concentration_dependent_fu_matches_nonlinear_flag():
    res = default_result()
    expected = "nonlinear" if res.nonlinear_binding else "linear"
    assert res.concentration_dependent_fu == expected


# ── fu_concentration_curve ────────────────────────────────────────────────────

def test_fu_concentration_curve_count():
    concs = [0.1, 1.0, 10.0, 100.0]
    results = fu_concentration_curve("Drug", conc_range_uM=concs)
    assert len(results) == 4


def test_fu_concentration_curve_sorted_ascending():
    concs = [100.0, 0.1, 10.0, 1.0]
    results = fu_concentration_curve("Drug", conc_range_uM=concs)
    concs_out = [r.drug_conc_uM for r in results]
    assert concs_out == sorted(concs_out)


def test_fu_concentration_curve_all_valid():
    results = fu_concentration_curve("Drug", conc_range_uM=[0.5, 5.0, 50.0])
    for r in results:
        assert isinstance(r, AlbuminBindingResult)
        assert 0.0 < r.fu_predicted <= 1.0


# ── compare_affinities ────────────────────────────────────────────────────────

def test_compare_affinities_count():
    results = compare_affinities("Drug", drug_conc_uM=1.0, ka1_list=[0.1, 1.0, 5.0])
    assert len(results) == 3


def test_compare_affinities_sorted_fu_ascending():
    results = compare_affinities("Drug", drug_conc_uM=1.0, ka1_list=[0.1, 1.0, 5.0, 20.0])
    fus = [r.fu_predicted for r in results]
    assert fus == sorted(fus)


def test_compare_affinities_all_fu_valid():
    results = compare_affinities("Drug", drug_conc_uM=1.0, ka1_list=[0.5, 2.0])
    for r in results:
        assert 0.0 < r.fu_predicted <= 1.0


# ── Validation errors ─────────────────────────────────────────────────────────

def test_validation_drug_conc_zero():
    with pytest.raises(ValueError, match="drug_conc_uM"):
        predict_albumin_binding("Drug", drug_conc_uM=0.0)


def test_validation_drug_conc_negative():
    with pytest.raises(ValueError, match="drug_conc_uM"):
        predict_albumin_binding("Drug", drug_conc_uM=-1.0)


def test_validation_albumin_negative():
    with pytest.raises(ValueError, match="albumin_conc_g_per_L"):
        predict_albumin_binding("Drug", drug_conc_uM=1.0, albumin_conc_g_per_L=-1.0)


def test_validation_ka1_zero():
    with pytest.raises(ValueError, match="ka_site1"):
        predict_albumin_binding("Drug", drug_conc_uM=1.0, ka_site1_L_per_uM=0.0)


def test_validation_ka2_negative():
    with pytest.raises(ValueError, match="ka_site2"):
        predict_albumin_binding("Drug", drug_conc_uM=1.0, ka_site2_L_per_uM=-0.1)


def test_validation_primary_site_invalid():
    with pytest.raises(ValueError, match="primary_site"):
        predict_albumin_binding("Drug", drug_conc_uM=1.0, primary_site="site_III")
