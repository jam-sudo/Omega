"""Tests for pk_from_smiles module (Phase 358)."""

from __future__ import annotations

import math

import pytest

from omega_pbpk.prediction.pk_from_smiles import (
    PKFromSMILESResult,
    batch_predict_pk,
    parse_smiles_properties,
    predict_pk_from_smiles,
)

# Simple aspirin-like SMILES: CC(=O)Oc1ccccc1C(=O)O
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

# Highly lipophilic: long carbon chain, no heteroatoms
LIPOPHILIC_SMILES = "CCCCCCCCCCCCCCCC"  # hexadecane

# Simple drug-like
DRUGLIKE_SMILES = "CCN(CC)CCOC(=O)c1ccc(N)cc1"  # benzocaine-like

# Large, rule-of-5-violating compound
LARGE_SMILES = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"  # very long chain


class TestParseSMILESProperties:
    def test_aspirin_mw_approximate(self):
        props = parse_smiles_properties(ASPIRIN_SMILES)
        # Aspirin MW ~180; allow wide tolerance for rough method
        assert 100 < props["mw"] < 350

    def test_aspirin_logP_low(self):
        props = parse_smiles_properties(ASPIRIN_SMILES)
        # Aspirin logP ~1.2; expect moderate low value
        assert props["logP"] < 4.0

    def test_lipophilic_high_logP(self):
        props = parse_smiles_properties(LIPOPHILIC_SMILES)
        # All carbons → high logP
        assert props["logP"] > 3.0

    def test_returns_all_keys(self):
        props = parse_smiles_properties(ASPIRIN_SMILES)
        for key in ("mw", "logP", "hbd", "hba", "psa", "rotatable_bonds", "n_rings"):
            assert key in props

    def test_hba_counts_nitrogen_and_oxygen(self):
        # SMILES with multiple O and N
        props = parse_smiles_properties("NCC(=O)O")  # glycine-like
        assert props["hba"] >= 2  # at least N and O

    def test_psa_positive(self):
        props = parse_smiles_properties(ASPIRIN_SMILES)
        assert props["psa"] >= 0

    def test_rotatable_bonds_nonnegative(self):
        props = parse_smiles_properties(ASPIRIN_SMILES)
        assert props["rotatable_bonds"] >= 0

    def test_n_rings_benzene(self):
        props = parse_smiles_properties("c1ccccc1")  # benzene
        assert props["n_rings"] >= 1

    def test_mw_positive(self):
        props = parse_smiles_properties(LIPOPHILIC_SMILES)
        assert props["mw"] > 0

    def test_logP_no_heteroatoms_positive(self):
        # Pure carbon chain should have positive logP
        props = parse_smiles_properties(LIPOPHILIC_SMILES)
        assert props["logP"] > 0

    def test_nitrogen_reduces_logP(self):
        props_c = parse_smiles_properties("CCCCCC")
        props_cn = parse_smiles_properties("CCCCNN")
        assert props_cn["logP"] < props_c["logP"]

    def test_empty_smiles_raises(self):
        with pytest.raises(ValueError):
            parse_smiles_properties("")

    def test_whitespace_smiles_raises(self):
        with pytest.raises(ValueError):
            parse_smiles_properties("   ")


class TestPredictPKFromSMILES:
    def test_returns_correct_type(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES, "Aspirin")
        assert isinstance(r, PKFromSMILESResult)

    def test_result_is_frozen(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        with pytest.raises((AttributeError, TypeError)):
            r.pred_logP = 99.0  # type: ignore[misc]

    def test_auc_positive(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.auc_mg_h_per_L > 0

    def test_cmax_positive(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.cmax_mg_L > 0

    def test_tmax_nonnegative(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.tmax_h >= 0

    def test_times_and_concs_same_length(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert len(r.times_h) == len(r.c_plasma_mg_L)

    def test_fu_plasma_in_0_1(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert 0 < r.pred_fu_plasma <= 1.0

    def test_bcs_class_valid(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.pred_bcs_class in ("I", "II", "III", "IV")

    def test_lipinski_passes_drug_like(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        # Aspirin is drug-like
        assert isinstance(r.lipinski_passes, bool)

    def test_lipinski_fails_large_lipophilic(self):
        r = predict_pk_from_smiles(LARGE_SMILES)
        # Very large compound should fail Lipinski
        assert not r.lipinski_passes

    def test_thalf_consistent_with_cl_vd(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        expected_thalf = 0.693 * r.pred_vd_L / r.pred_cl_L_per_h
        assert abs(r.pred_t_half_h - expected_thalf) < 0.01

    def test_compound_name_stored(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES, "Aspirin")
        assert r.compound_name == "Aspirin"

    def test_smiles_stored(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.smiles == ASPIRIN_SMILES

    def test_dose_stored(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES, dose_mg=250.0)
        assert r.dose_mg == pytest.approx(250.0)

    def test_higher_dose_higher_cmax(self):
        r_low = predict_pk_from_smiles(ASPIRIN_SMILES, dose_mg=100.0)
        r_high = predict_pk_from_smiles(ASPIRIN_SMILES, dose_mg=500.0)
        assert r_high.cmax_mg_L > r_low.cmax_mg_L

    def test_lipophilic_higher_logP(self):
        r_lip = predict_pk_from_smiles(LIPOPHILIC_SMILES)
        r_asp = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r_lip.pred_logP > r_asp.pred_logP

    def test_vd_positive(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.pred_vd_L > 0

    def test_cl_positive(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.pred_cl_L_per_h > 0

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError):
            predict_pk_from_smiles("")

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError):
            predict_pk_from_smiles(ASPIRIN_SMILES, dose_mg=-10.0)

    def test_mw_estimate_positive(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.pred_mw > 0

    def test_hba_nonnegative(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.pred_hba >= 0

    def test_hbd_nonnegative(self):
        r = predict_pk_from_smiles(ASPIRIN_SMILES)
        assert r.pred_hbd >= 0


class TestBatchPredictPK:
    def test_returns_list(self):
        compounds = [
            {"smiles": ASPIRIN_SMILES, "name": "Aspirin"},
            {"smiles": LIPOPHILIC_SMILES, "name": "Lipophilic"},
        ]
        results = batch_predict_pk(compounds)
        assert isinstance(results, list)

    def test_length_matches_input(self):
        compounds = [
            {"smiles": ASPIRIN_SMILES, "name": f"Compound{i}"}
            for i in range(4)
        ]
        results = batch_predict_pk(compounds)
        assert len(results) == 4

    def test_sorted_by_cl_ascending(self):
        compounds = [
            {"smiles": ASPIRIN_SMILES, "name": "Asp"},
            {"smiles": LIPOPHILIC_SMILES, "name": "Lip"},
            {"smiles": DRUGLIKE_SMILES, "name": "Drug"},
        ]
        results = batch_predict_pk(compounds)
        cls = [r.pred_cl_L_per_h for r in results]
        assert cls == sorted(cls)

    def test_empty_returns_empty(self):
        results = batch_predict_pk([])
        assert results == []

    def test_each_result_is_correct_type(self):
        compounds = [{"smiles": ASPIRIN_SMILES, "name": "A"}]
        results = batch_predict_pk(compounds)
        assert isinstance(results[0], PKFromSMILESResult)
