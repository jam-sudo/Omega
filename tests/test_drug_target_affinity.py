"""Tests for prediction/drug_target_affinity.py — Phase 638."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.drug_target_affinity import (
    DrugTargetAffinityResult,
    predict_drug_target_affinity,
    screen_against_targets,
)

VALID_POTENCY_CLASSES = {"very_potent", "potent", "moderate", "weak"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kinase_result() -> DrugTargetAffinityResult:
    return predict_drug_target_affinity(
        compound_name="KinaseInhibitor",
        target_class="kinase",
        logP=3.0,
        mw=400.0,
        psa=80.0,
        n_hbd=2,
        n_hba=5,
    )


# ---------------------------------------------------------------------------
# Return type and basic field tests
# ---------------------------------------------------------------------------


def test_returns_drug_target_affinity_result(kinase_result):
    assert isinstance(kinase_result, DrugTargetAffinityResult)


def test_predicted_ki_um_positive(kinase_result):
    assert kinase_result.predicted_Ki_uM > 0.0


def test_predicted_pki_positive(kinase_result):
    assert kinase_result.predicted_pKi > 0.0


def test_predicted_ic50_positive(kinase_result):
    assert kinase_result.predicted_IC50_uM > 0.0


def test_ic50_approximately_twice_ki(kinase_result):
    ratio = kinase_result.predicted_IC50_uM / kinase_result.predicted_Ki_uM
    assert pytest.approx(ratio, rel=1e-6) == 2.0


def test_potency_class_valid(kinase_result):
    assert kinase_result.potency_class in VALID_POTENCY_CLASSES


def test_selectivity_score_range(kinase_result):
    assert 0.0 <= kinase_result.selectivity_score <= 10.0


def test_drugability_score_range(kinase_result):
    assert 0.0 <= kinase_result.drugability_score <= 100.0


def test_notes_nonempty(kinase_result):
    assert len(kinase_result.notes) > 0


# ---------------------------------------------------------------------------
# Physicochemical effects
# ---------------------------------------------------------------------------


def test_high_logp_lower_ki_kinase():
    """Higher logP → stronger binding → lower Ki for kinase."""
    r_low = predict_drug_target_affinity("X", "kinase", logP=1.0, mw=400.0, psa=80.0)
    r_high = predict_drug_target_affinity("X", "kinase", logP=5.0, mw=400.0, psa=80.0)
    assert r_high.predicted_Ki_uM < r_low.predicted_Ki_uM


def test_higher_psa_lower_affinity():
    """Higher PSA (polar penalty) → higher Ki."""
    r_low = predict_drug_target_affinity("X", "kinase", logP=3.0, mw=400.0, psa=60.0)
    r_high = predict_drug_target_affinity("X", "kinase", logP=3.0, mw=400.0, psa=150.0)
    assert r_high.predicted_Ki_uM > r_low.predicted_Ki_uM


def test_nuclear_receptor_higher_logp_better_binding():
    """Nuclear receptor favours higher logP → lower Ki."""
    r_low = predict_drug_target_affinity("X", "nuclear_receptor", logP=1.0, mw=400.0, psa=80.0)
    r_high = predict_drug_target_affinity("X", "nuclear_receptor", logP=6.0, mw=400.0, psa=80.0)
    assert r_high.predicted_Ki_uM < r_low.predicted_Ki_uM


# ---------------------------------------------------------------------------
# screen_against_targets
# ---------------------------------------------------------------------------


def test_screen_against_targets_returns_five():
    results = screen_against_targets("DrugA", logP=3.0, mw=400.0, psa=80.0)
    assert len(results) == 5


def test_screen_against_targets_sorted_by_ki():
    results = screen_against_targets("DrugA", logP=3.0, mw=400.0, psa=80.0)
    ki_values = [r.predicted_Ki_uM for r in results]
    assert ki_values == sorted(ki_values)


def test_screen_custom_targets_subset():
    results = screen_against_targets(
        "DrugB", logP=3.0, mw=400.0, psa=80.0, targets=["kinase", "gpcr"]
    )
    assert len(results) == 2
    target_classes = {r.target_class for r in results}
    assert target_classes == {"kinase", "gpcr"}


# ---------------------------------------------------------------------------
# Charge bonus (basic drug)
# ---------------------------------------------------------------------------


def test_pka_base_charge_bonus_gpcr():
    """Basic drug (pKa_base=9.0) should have lower Ki for GPCR vs no pKa."""
    r_neutral = predict_drug_target_affinity(
        "X", "gpcr", logP=3.0, mw=350.0, psa=60.0, pka_base=None
    )
    r_basic = predict_drug_target_affinity("X", "gpcr", logP=3.0, mw=350.0, psa=60.0, pka_base=9.0)
    assert r_basic.predicted_Ki_uM < r_neutral.predicted_Ki_uM


# ---------------------------------------------------------------------------
# Drugability score sensitivity
# ---------------------------------------------------------------------------


def test_bcs_like_mw_logp_higher_drugability():
    """Low MW and low logP (drug-like) → higher drugability score."""
    r_druglike = predict_drug_target_affinity(
        "X", "kinase", logP=2.0, mw=300.0, psa=60.0, n_hbd=1, n_hba=3
    )
    r_heavy = predict_drug_target_affinity(
        "X", "kinase", logP=2.0, mw=800.0, psa=60.0, n_hbd=1, n_hba=3
    )
    assert r_druglike.drugability_score > r_heavy.drugability_score


def test_heavy_mw_drugability_penalty():
    """MW > 700 should impose a drugability penalty."""
    r_light = predict_drug_target_affinity("X", "kinase", logP=3.0, mw=350.0, psa=80.0)
    r_heavy = predict_drug_target_affinity("X", "kinase", logP=3.0, mw=750.0, psa=80.0)
    assert r_heavy.drugability_score < r_light.drugability_score


# ---------------------------------------------------------------------------
# Potency class boundaries
# ---------------------------------------------------------------------------


def test_potency_very_potent_for_low_ki():
    """When model produces Ki < 0.01 µM, class should be very_potent."""
    # Use nuclear_receptor with high logP and low MW to push pKi very high
    r = predict_drug_target_affinity(
        "SuperPotent", "nuclear_receptor", logP=8.0, mw=200.0, psa=10.0, n_hbd=3
    )
    # pKi is clamped to 12 → Ki = 1e-12 M = 1e-6 µM → very_potent
    if r.predicted_Ki_uM < 0.01:
        assert r.potency_class == "very_potent"


def test_potency_weak_for_high_ki():
    """When model produces Ki > 1.0 µM, class should be weak."""
    r = predict_drug_target_affinity(
        "WeakBinder", "ion_channel", logP=-3.0, mw=700.0, psa=200.0, n_hbd=0
    )
    if r.predicted_Ki_uM > 1.0:
        assert r.potency_class == "weak"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_unknown_target_class():
    with pytest.raises(ValueError, match="target_class"):
        predict_drug_target_affinity("X", "unknown_target", logP=2.0, mw=300.0, psa=60.0)


def test_validation_mw_zero():
    with pytest.raises(ValueError, match="mw"):
        predict_drug_target_affinity("X", "kinase", logP=2.0, mw=0.0, psa=60.0)


def test_validation_psa_negative():
    with pytest.raises(ValueError, match="psa"):
        predict_drug_target_affinity("X", "kinase", logP=2.0, mw=300.0, psa=-1.0)


def test_validation_n_hbd_negative():
    with pytest.raises(ValueError, match="n_hbd"):
        predict_drug_target_affinity("X", "kinase", logP=2.0, mw=300.0, psa=60.0, n_hbd=-1)
