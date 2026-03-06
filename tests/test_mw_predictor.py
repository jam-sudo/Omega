"""Tests for MW predictor (Phase 113)."""

from __future__ import annotations

import pytest

from omega_pbpk.prediction.mw_predictor import MWResult, predict_mw, screen_mw


class TestPredictMW:
    def test_returns_result_type(self):
        r = predict_mw("Drug", n_C=20)
        assert isinstance(r, MWResult)

    def test_mw_positive(self):
        r = predict_mw("Drug", n_C=20, n_N=2, n_O=3)
        assert r.MW_g_mol > 0.0

    def test_more_carbons_higher_mw(self):
        r_low = predict_mw("Drug", n_C=10)
        r_high = predict_mw("Drug", n_C=30)
        assert r_high.MW_g_mol > r_low.MW_g_mol

    def test_bromine_increases_mw(self):
        r_no_br = predict_mw("Drug", n_C=15)
        r_br = predict_mw("Drug", n_C=15, n_Br=1)
        assert r_br.MW_g_mol > r_no_br.MW_g_mol

    def test_formula_contains_C(self):
        r = predict_mw("Drug", n_C=10)
        assert "C" in r.molecular_formula

    def test_formula_contains_N_if_nonzero(self):
        r = predict_mw("Drug", n_C=10, n_N=2)
        assert "N" in r.molecular_formula

    def test_lipinski_mw_ok_small_mol(self):
        r = predict_mw("Drug", n_C=15, n_O=2)
        assert r.lipinski_mw_ok is True

    def test_lipinski_mw_fails_large_mol(self):
        r = predict_mw("Drug", n_C=50, n_N=10, n_O=10)
        assert r.lipinski_mw_ok is False

    def test_heavy_atom_count_correct(self):
        r = predict_mw("Drug", n_C=10, n_N=2, n_O=3, n_S=1)
        assert r.n_heavy_atoms == 16

    def test_drug_like_flag(self):
        r = predict_mw("Drug", n_C=20, n_N=2, n_O=3)
        assert isinstance(r.drug_like, bool)

    def test_explicit_H_override(self):
        r = predict_mw("Drug", n_C=10, explicit_H=20)
        assert r.n_H == 20


class TestScreenMW:
    def test_returns_list(self):
        compounds = [{"drug_name": "A", "n_C": 10}, {"drug_name": "B", "n_C": 20, "n_N": 2}]
        results = screen_mw(compounds)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_empty_returns_empty(self):
        assert screen_mw([]) == []
