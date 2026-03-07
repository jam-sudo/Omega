"""Tests for prediction/insilico_admet.py — Phase 196."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.insilico_admet import (
    ADMETProfile,
    compare_admet_profiles,
    predict_admet_profile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ASPIRIN_LIKE = dict(
    compound_name="Aspirin",
    smiles="CC(=O)Oc1ccccc1C(=O)O",
    mw=180.0,
    logP=1.2,
    psa=63.6,
    n_hbd=1,
    n_hba=4,
)

LIPOPHILIC = dict(
    compound_name="LipoDrug",
    smiles="CCCCCCCCCC",
    mw=142.0,
    logP=5.5,
    psa=0.0,
    n_hbd=0,
    n_hba=0,
)


def _predict(**kwargs) -> ADMETProfile:
    defaults = dict(ASPIRIN_LIKE)
    defaults.update(kwargs)
    return predict_admet_profile(**defaults)


# ---------------------------------------------------------------------------
# Return type / structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dataclass(self):
        r = _predict()
        assert isinstance(r, ADMETProfile)

    def test_compound_name_preserved(self):
        r = _predict(compound_name="MyDrug")
        assert r.compound_name == "MyDrug"

    def test_smiles_preserved(self):
        smiles = "CCO"
        r = _predict(smiles=smiles)
        assert r.smiles == smiles

    def test_notes_non_empty(self):
        r = _predict()
        assert isinstance(r.notes, str)
        assert len(r.notes) > 0


# ---------------------------------------------------------------------------
# Absorption
# ---------------------------------------------------------------------------


class TestAbsorption:
    def test_high_logP_gives_higher_hia(self):
        r_high = _predict(logP=4.0)
        r_low = _predict(logP=-1.0)
        assert r_high.hia_pct > r_low.hia_pct

    def test_hia_between_0_and_100(self):
        for lp in [-3.0, 0.0, 3.0, 6.0]:
            r = _predict(logP=lp)
            assert 0.0 <= r.hia_pct <= 100.0

    def test_bioavailability_between_0_and_100(self):
        for lp in [-2.0, 2.0, 5.0]:
            r = _predict(logP=lp)
            assert 0.0 <= r.oral_bioavailability_pct <= 100.0

    def test_high_psa_reduces_bioavailability(self):
        r_high_psa = _predict(psa=180.0)
        r_low_psa = _predict(psa=20.0)
        assert r_high_psa.oral_bioavailability_pct <= r_low_psa.oral_bioavailability_pct


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------


class TestDistribution:
    def test_fup_between_0_and_1(self):
        for lp in [-2.0, 0.0, 3.0, 7.0]:
            r = _predict(logP=lp)
            assert 0.0 < r.fup <= 1.0

    def test_bbb_penetration_true_for_cns_like(self):
        # logP=2, mw=300, psa=50 → BBB = True
        r = predict_admet_profile("CNSDrug", "CCNCC", mw=300.0, logP=2.0, psa=50.0)
        assert r.bbb_penetration is True

    def test_bbb_penetration_false_for_high_psa(self):
        r = predict_admet_profile("PolarDrug", "CCNCC", mw=300.0, logP=2.0, psa=120.0)
        assert r.bbb_penetration is False

    def test_bbb_penetration_false_for_large_mw(self):
        r = predict_admet_profile("BigDrug", "CCNCC", mw=600.0, logP=2.0, psa=50.0)
        assert r.bbb_penetration is False

    def test_vss_positive(self):
        r = _predict()
        assert r.vss_L_per_kg > 0.0


# ---------------------------------------------------------------------------
# Metabolism
# ---------------------------------------------------------------------------


class TestMetabolism:
    def test_cyp3a4_substrate_for_lipophilic_large(self):
        # logP > 2 and mw > 300 → True
        r = predict_admet_profile("Big", "CCNCC", mw=400.0, logP=3.0)
        assert r.cyp3a4_substrate is True

    def test_cyp3a4_substrate_false_for_small_hydrophilic(self):
        r = predict_admet_profile("Small", "CCO", mw=100.0, logP=1.0)
        assert r.cyp3a4_substrate is False

    def test_cyp2d6_substrate_with_N_and_small_mw(self):
        # "N" in smiles and mw < 400
        r = predict_admet_profile("Amine", "CCN", mw=200.0, logP=1.5)
        assert r.cyp2d6_substrate is True

    def test_cyp2d6_substrate_false_no_nitrogen(self):
        r = predict_admet_profile("NoN", "CCCC", mw=200.0, logP=1.5)
        assert r.cyp2d6_substrate is False

    def test_t_half_positive(self):
        r = _predict()
        assert r.t_half_h > 0.0


# ---------------------------------------------------------------------------
# Toxicity
# ---------------------------------------------------------------------------


class TestToxicity:
    def test_herg_risk_lipophilic_low_hba(self):
        # logP > 3, n_hba < 3, mw > 300 → True
        r = predict_admet_profile("hERG", "CCNCC", mw=350.0, logP=4.0, n_hba=2)
        assert r.herg_risk is True

    def test_herg_risk_false_polar(self):
        # n_hba >= 3 → False
        r = predict_admet_profile("Safe", "CCNCC", mw=350.0, logP=4.0, n_hba=5)
        assert r.herg_risk is False

    def test_ames_risk_nitro_aromatic(self):
        smiles = "c1ccc([N+](=O)[O-])cc1"  # nitrobenzene
        r = predict_admet_profile("NitroBenz", smiles, mw=123.0, logP=1.8)
        assert r.ames_risk is True

    def test_ames_risk_false_clean(self):
        r = predict_admet_profile("Clean", "CCCO", mw=74.0, logP=0.8)
        assert r.ames_risk is False

    def test_pains_flag_nitroaromatic(self):
        # "[N+](=O)[O-]" in smiles AND "c" in smiles
        smiles = "c1ccc([N+](=O)[O-])cc1"
        r = predict_admet_profile("NitroBenz", smiles, mw=123.0, logP=1.8)
        assert r.pains_flag is True

    def test_pains_flag_false_clean(self):
        r = predict_admet_profile("Clean", "CCCO", mw=74.0, logP=0.8)
        assert r.pains_flag is False


# ---------------------------------------------------------------------------
# Drug-likeness
# ---------------------------------------------------------------------------


class TestDrugLikeness:
    def test_lipinski_pass_typical_oral_drug(self):
        r = predict_admet_profile("OralDrug", "CCO", mw=300.0, logP=2.0, n_hbd=2, n_hba=4)
        assert r.lipinski_pass is True

    def test_lipinski_fail_high_mw(self):
        r = predict_admet_profile("Big", "CCO", mw=600.0, logP=2.0, n_hbd=2, n_hba=4)
        assert r.lipinski_pass is False

    def test_lipinski_fail_high_logP(self):
        r = predict_admet_profile("Lipo", "CCCCCC", mw=300.0, logP=6.0, n_hbd=1, n_hba=2)
        assert r.lipinski_pass is False

    def test_lipinski_fail_high_hbd(self):
        r = predict_admet_profile("Polar", "CCO", mw=200.0, logP=1.0, n_hbd=7, n_hba=4)
        assert r.lipinski_pass is False


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_score_between_0_and_100(self):
        for lp in [-2.0, 0.0, 2.0, 5.0, 8.0]:
            r = _predict(logP=lp)
            assert 0.0 <= r.composite_score <= 100.0

    def test_clean_drug_like_compound_has_high_score(self):
        r = predict_admet_profile(
            "CleanDrug",
            "CCO",
            mw=200.0,
            logP=2.0,
            psa=50.0,
            n_hbd=1,
            n_hba=3,
        )
        assert r.composite_score > 40.0

    def test_toxic_compound_has_lower_score(self):
        # Nitroaromatic, high logP, low hba → ames + herg + hepatotox
        r_toxic = predict_admet_profile(
            "Toxic",
            "c1ccc([N+](=O)[O-])cc1",
            mw=350.0,
            logP=4.5,
            n_hba=1,
        )
        r_clean = predict_admet_profile(
            "Clean",
            "CCCO",
            mw=200.0,
            logP=1.5,
            n_hba=5,
        )
        assert r_toxic.composite_score <= r_clean.composite_score


# ---------------------------------------------------------------------------
# compare_admet_profiles
# ---------------------------------------------------------------------------


class TestCompareAdmetProfiles:
    def test_returns_list(self):
        compounds = [
            {"name": "A", "smiles": "CCO", "mw": 200.0, "logP": 2.0},
            {"name": "B", "smiles": "CCCC", "mw": 300.0, "logP": 4.0},
        ]
        results = compare_admet_profiles(compounds)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_sorted_descending_by_score(self):
        compounds = [
            {"name": "C1", "smiles": "CCO", "mw": 200.0, "logP": 1.0, "n_hba": 8},
            {"name": "C2", "smiles": "CCO", "mw": 300.0, "logP": 2.5, "n_hba": 5},
            {
                "name": "C3",
                "smiles": "c1ccc([N+](=O)[O-])cc1",
                "mw": 400.0,
                "logP": 4.5,
                "n_hba": 1,
            },
        ]
        results = compare_admet_profiles(compounds)
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_profiles_returned(self):
        compounds = [
            {"name": f"Drug{i}", "smiles": "CCO", "mw": 200.0 + i * 50, "logP": float(i)}
            for i in range(5)
        ]
        results = compare_admet_profiles(compounds)
        assert len(results) == 5

    def test_single_compound(self):
        compounds = [{"name": "Solo", "smiles": "CCO", "mw": 100.0, "logP": 1.0}]
        results = compare_admet_profiles(compounds)
        assert len(results) == 1
        assert results[0].compound_name == "Solo"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_admet_profile("X", "CCO", mw=-1.0, logP=2.0)

    def test_zero_mw_raises(self):
        with pytest.raises(ValueError, match="mw"):
            predict_admet_profile("X", "CCO", mw=0.0, logP=2.0)

    def test_empty_smiles_raises(self):
        with pytest.raises(ValueError, match="smiles"):
            predict_admet_profile("X", "", mw=200.0, logP=2.0)

    def test_whitespace_smiles_raises(self):
        with pytest.raises(ValueError, match="smiles"):
            predict_admet_profile("X", "   ", mw=200.0, logP=2.0)
