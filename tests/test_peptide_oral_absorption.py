"""Tests for Phase 853 — Peptide/Protein Oral Absorption."""

import pytest

from omega_pbpk.prediction.peptide_oral_absorption import (
    PeptideOralAbsorptionResult,
    predict_peptide_oral_absorption,
    screen_peptide_modifications,
)

VALID_ABSORPTION_CLASSES = {"negligible", "poor", "moderate", "good"}


# ─── Basic return type ───────────────────────────────────────────────────────


def test_returns_correct_type():
    result = predict_peptide_oral_absorption(
        drug_name="TestPep",
        mw_da=500.0,
        n_hbond_donors=5,
        logp=1.5,
        dose_mg=10.0,
    )
    assert isinstance(result, PeptideOralAbsorptionResult)


def test_fields_preserved():
    result = predict_peptide_oral_absorption(
        drug_name="MyPep",
        mw_da=800.0,
        n_hbond_donors=3,
        logp=1.0,
        dose_mg=50.0,
        gi_stability_pct=60.0,
        p_gp_substrate=False,
        peptide_type="small_peptide",
    )
    assert result.drug_name == "MyPep"
    assert result.mw_da == 800.0
    assert result.dose_mg == 50.0
    assert result.gi_stability_pct == 60.0
    assert result.peptide_type == "small_peptide"


# ─── Component factors ────────────────────────────────────────────────────────


def test_f_size_decreases_with_mw():
    small = predict_peptide_oral_absorption("A", 200.0, 2, 1.5, 10.0)
    large = predict_peptide_oral_absorption("B", 5000.0, 2, 1.5, 10.0)
    assert small.f_size > large.f_size


def test_f_size_cap_at_one():
    result = predict_peptide_oral_absorption("A", 100.0, 1, 1.5, 10.0)
    assert result.f_size <= 1.0


def test_p_gp_substrate_reduces_fa():
    no_pgp = predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, p_gp_substrate=False)
    pgp = predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, p_gp_substrate=True)
    assert pgp.fa_oral < no_pgp.fa_oral


def test_f_pgp_values():
    no_pgp = predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, p_gp_substrate=False)
    pgp = predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, p_gp_substrate=True)
    assert no_pgp.f_pgp == 1.0
    assert pgp.f_pgp == 0.3


def test_gi_stability_scales_f_stable():
    low = predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, gi_stability_pct=10.0)
    high = predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, gi_stability_pct=90.0)
    assert high.f_stable > low.f_stable


# ─── Peptide type effects ─────────────────────────────────────────────────────


def test_protein_type_lower_fa_than_small_peptide():
    small_pep = predict_peptide_oral_absorption(
        "A",
        500.0,
        3,
        1.5,
        10.0,
        gi_stability_pct=80.0,
        peptide_type="small_peptide",
    )
    protein = predict_peptide_oral_absorption(
        "A",
        500.0,
        3,
        1.5,
        10.0,
        gi_stability_pct=80.0,
        peptide_type="protein",
    )
    assert protein.fa_oral < small_pep.fa_oral


def test_large_peptide_between_small_and_protein():
    sp = predict_peptide_oral_absorption(
        "A", 500.0, 3, 1.5, 10.0, gi_stability_pct=80.0, peptide_type="small_peptide"
    )
    lp = predict_peptide_oral_absorption(
        "A", 500.0, 3, 1.5, 10.0, gi_stability_pct=80.0, peptide_type="large_peptide"
    )
    pr = predict_peptide_oral_absorption(
        "A", 500.0, 3, 1.5, 10.0, gi_stability_pct=80.0, peptide_type="protein"
    )
    assert sp.fa_oral >= lp.fa_oral >= pr.fa_oral


# ─── Absorbed dose relationship ───────────────────────────────────────────────


def test_absorbed_dose_equals_f_times_dose():
    result = predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0)
    assert abs(result.absorbed_dose_mg - result.bioavailability_f * result.dose_mg) < 1e-9


def test_absorbed_dose_non_negative():
    result = predict_peptide_oral_absorption("A", 50000.0, 10, -5.0, 100.0, gi_stability_pct=1.0)
    assert result.absorbed_dose_mg >= 0.0


# ─── Absorption class ─────────────────────────────────────────────────────────


def test_absorption_class_valid():
    result = predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0)
    assert result.absorption_class in VALID_ABSORPTION_CLASSES


def test_absorption_class_negligible_for_zero_stability():
    result = predict_peptide_oral_absorption(
        "A",
        500.0,
        3,
        1.5,
        10.0,
        gi_stability_pct=0.0,
    )
    assert result.absorption_class == "negligible"
    assert result.fa_oral == 0.0


def test_absorption_class_boundaries():
    # Force F near boundaries
    # good: F >= 0.3 — use small MW, stable compound
    good = predict_peptide_oral_absorption("A", 100.0, 0, 1.5, 10.0, gi_stability_pct=100.0)
    assert good.absorption_class in {"good", "moderate"}  # depends on computed f_size

    # negligible: very large protein, unstable
    neg = predict_peptide_oral_absorption(
        "A",
        50000.0,
        20,
        -3.0,
        10.0,
        gi_stability_pct=1.0,
        peptide_type="protein",
    )
    assert neg.absorption_class == "negligible"


# ─── fa_oral bounds ───────────────────────────────────────────────────────────


def test_fa_oral_in_zero_one():
    cases = [
        (100.0, 1.5, 100.0, False, "small_peptide"),
        (50000.0, -5.0, 0.0, True, "protein"),
        (800.0, 2.0, 50.0, False, "large_peptide"),
    ]
    for mw, lp, gi, pgp, pt in cases:
        r = predict_peptide_oral_absorption(
            "X", mw, 3, lp, 10.0, gi_stability_pct=gi, p_gp_substrate=pgp, peptide_type=pt
        )
        assert 0.0 <= r.fa_oral <= 1.0


# ─── screen_peptide_modifications ────────────────────────────────────────────


def test_screen_returns_list():
    mods = [
        {"name": "Mod-A", "gi_stability_pct": 80.0, "p_gp_substrate": False, "logp_delta": 0.0},
        {"name": "Mod-B", "gi_stability_pct": 40.0, "p_gp_substrate": True, "logp_delta": -1.0},
    ]
    results = screen_peptide_modifications("BasePep", 600.0, 1.5, 10.0, mods)
    assert isinstance(results, list)
    assert len(results) == 2


def test_screen_returns_correct_types():
    mods = [
        {"name": "Mod-A", "gi_stability_pct": 80.0, "p_gp_substrate": False, "logp_delta": 0.0},
    ]
    results = screen_peptide_modifications("BasePep", 600.0, 1.5, 10.0, mods)
    assert all(isinstance(r, PeptideOralAbsorptionResult) for r in results)


def test_screen_sorted_descending():
    mods = [
        {"name": "Low", "gi_stability_pct": 10.0, "p_gp_substrate": True, "logp_delta": -2.0},
        {"name": "High", "gi_stability_pct": 90.0, "p_gp_substrate": False, "logp_delta": 0.5},
        {"name": "Mid", "gi_stability_pct": 50.0, "p_gp_substrate": False, "logp_delta": 0.0},
    ]
    results = screen_peptide_modifications("BasePep", 500.0, 1.5, 10.0, mods)
    for i in range(len(results) - 1):
        assert results[i].bioavailability_f >= results[i + 1].bioavailability_f


def test_screen_logp_delta_applied():
    """Modification with logp_delta should differ from base."""
    mods = [
        {"name": "Base", "gi_stability_pct": 60.0, "p_gp_substrate": False, "logp_delta": 0.0},
        {"name": "Shifted", "gi_stability_pct": 60.0, "p_gp_substrate": False, "logp_delta": 5.0},
    ]
    results = screen_peptide_modifications("BasePep", 500.0, 1.5, 10.0, mods)
    base_r = next(r for r in results if r.drug_name == "Base")
    shifted_r = next(r for r in results if r.drug_name == "Shifted")
    # logp 6.5 should give lower f_lip than 1.5
    assert base_r.f_lip != shifted_r.f_lip


# ─── Validation errors ────────────────────────────────────────────────────────


def test_invalid_mw_da():
    with pytest.raises(ValueError, match="mw_da must be > 0"):
        predict_peptide_oral_absorption("A", 0.0, 3, 1.5, 10.0)


def test_invalid_mw_da_negative():
    with pytest.raises(ValueError, match="mw_da must be > 0"):
        predict_peptide_oral_absorption("A", -100.0, 3, 1.5, 10.0)


def test_invalid_dose_mg():
    with pytest.raises(ValueError, match="dose_mg must be > 0"):
        predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 0.0)


def test_invalid_gi_stability_low():
    with pytest.raises(ValueError, match="gi_stability_pct must be in"):
        predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, gi_stability_pct=-1.0)


def test_invalid_gi_stability_high():
    with pytest.raises(ValueError, match="gi_stability_pct must be in"):
        predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, gi_stability_pct=101.0)


def test_invalid_peptide_type():
    with pytest.raises(ValueError, match="peptide_type must be one of"):
        predict_peptide_oral_absorption("A", 500.0, 3, 1.5, 10.0, peptide_type="macro")
