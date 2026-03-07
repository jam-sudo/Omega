"""Tests for Phase 618 — predict_metabolic_stability (from measured microsomal t½)."""

import pytest

from omega_pbpk.prediction.metabolic_stability import (
    MetabolicStabilityResult,
    classify_stability,
    predict_metabolic_stability,
    predict_primary_pathway,
)

DRUG = "TestDrug618"


def make_result(**kwargs):
    defaults = dict(
        drug_name=DRUG,
        t_half_microsomal_min=30.0,
        protein_mg_per_mL=0.5,
        fu_mic=1.0,
        logP=2.0,
        mw=400.0,
        pka_base=None,
        n_aromatic_rings=0,
        vd_L=50.0,
        fu_plasma=0.5,
        mppgl=45.0,
        liver_weight_g=1500.0,
    )
    defaults.update(kwargs)
    return predict_metabolic_stability(**defaults)


def test_returns_result():
    r = make_result()
    assert isinstance(r, MetabolicStabilityResult)


def test_clint_positive():
    r = make_result()
    assert r.clint_mL_per_min_per_mg > 0


def test_scaled_clint_positive():
    r = make_result()
    assert r.clint_scaled_mL_per_min > 0


def test_hepatic_cl_positive():
    r = make_result()
    assert r.hepatic_cl_L_per_h > 0


def test_er_in_0_1():
    r = make_result()
    assert 0.0 <= r.extraction_ratio <= 1.0


def test_stability_class_valid():
    r = make_result()
    assert r.stability_class in {"unstable", "moderate", "stable", "very_stable"}


def test_short_t_half_unstable():
    r = make_result(t_half_microsomal_min=2.0)
    assert r.clint_mL_per_min_per_mg > 30 or r.stability_class in {"unstable", "moderate"}


def test_long_t_half_stable():
    r = make_result(t_half_microsomal_min=100.0)
    assert r.stability_class in {"stable", "very_stable"}


def test_high_logP_metabolite_risk():
    r = make_result(logP=5.0)
    assert r.metabolite_toxicity_risk is True


def test_low_logP_low_risk():
    r = make_result(logP=0.0, n_aromatic_rings=0)
    assert r.metabolite_toxicity_risk is False


def test_primary_pathway_cyp3a4():
    r = make_result(logP=3.0, mw=400.0, pka_base=None)
    # logP>2 and mw>300 → CYP3A4 (if not very_stable)
    if r.stability_class != "very_stable":
        assert r.primary_metabolic_pathway == "CYP3A4"


def test_primary_pathway_cyp2d6():
    r = make_result(pka_base=9.0, t_half_microsomal_min=10.0)
    if r.stability_class != "very_stable":
        assert r.primary_metabolic_pathway == "CYP2D6"


def test_primary_pathway_cyp2c9():
    r = make_result(logP=1.0, mw=300.0, pka_base=None, t_half_microsomal_min=20.0)
    if r.stability_class != "very_stable":
        assert r.primary_metabolic_pathway == "CYP2C9"


def test_plasma_t_half_positive():
    r = make_result()
    assert r.plasma_t_half_estimate_h > 0


def test_in_vitro_to_in_vivo_factor():
    r = make_result(mppgl=45.0, liver_weight_g=1500.0)
    assert abs(r.in_vitro_to_in_vivo_factor - 67.5) < 1e-6


def test_fu_mic_correction():
    r_full = make_result(fu_mic=1.0)
    r_half = make_result(fu_mic=0.5)
    # fu_mic=0.5 → CLint * 2 compared to fu_mic=1.0
    assert r_half.clint_mL_per_min_per_mg > r_full.clint_mL_per_min_per_mg


def test_classify_stability_unstable():
    assert classify_stability(200.0) == "unstable"


def test_classify_stability_very_stable():
    assert classify_stability(5.0) == "very_stable"


def test_predict_pathway_cyp3a4():
    assert predict_primary_pathway(3.0, None, 400.0) == "CYP3A4"


def test_invalid_t_half_raises():
    with pytest.raises(ValueError):
        make_result(t_half_microsomal_min=0.0)


def test_invalid_protein_raises():
    with pytest.raises(ValueError):
        make_result(protein_mg_per_mL=0.0)


def test_notes_nonempty():
    r = make_result()
    assert len(r.notes) > 0
