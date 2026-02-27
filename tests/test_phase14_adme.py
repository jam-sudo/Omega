"""Phase 14: GNN ADME Predictor tests.

Tests cover:
  - SmilesFeaturizer: 20-feature extraction, atom counting, ring detection,
    H-bond donor/acceptor, aromaticity, carbonyl detection.
  - NumpyADMEPredictor: endpoint range, qualitative trends (lipophilic > polar,
    high TPSA → low Peff, high logP → low fup), batch prediction.
  - ADMEPredictor integration: numpy GNN used as fallback, confidence=medium.
"""

from __future__ import annotations

import numpy as np
import pytest

from omega_pbpk.features.smiles_featurizer import (
    FEATURE_NAMES,
    N_FEATURES,
    SmilesFeaturizer,
    _count_carbonyl,
    _count_hbd,
    _count_rings,
    _parse_atoms,
)
from omega_pbpk.ml_models.numpy_adme import NumpyADMEPredictor
from omega_pbpk.prediction.adme_predictor import ADMEPredictor

# ---------------------------------------------------------------------------
# Reference molecules (all valid SMILES)
# ---------------------------------------------------------------------------
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"  # logP ~1.2, MW ~180, 1 ring
CAFFEINE = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"  # logP ~-0.07, MW ~194, 3 rings (2 fused)
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"  # logP ~0.46, MW ~151, 1 ring, HBD=2
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # logP ~3.5, MW ~206, 1 ring
METHANE = "C"
ETHANOL = "CCO"
ANILINE = "Nc1ccccc1"  # aromatic amine
PYRIDINE = "c1ccncc1"  # aromatic N
CHLOROBENZENE = "Clc1ccccc1"


# ============================================================================
# SmilesFeaturizer
# ============================================================================


class TestSmilesFeaturizer:
    @pytest.fixture
    def featurizer(self):
        return SmilesFeaturizer()

    def test_feature_vector_length(self, featurizer):
        fv = featurizer.featurize(ASPIRIN)
        assert fv.descriptors.shape == (N_FEATURES,)
        assert len(fv.feature_names) == N_FEATURES

    def test_feature_names_match_constant(self, featurizer):
        fv = featurizer.featurize(ASPIRIN)
        assert fv.feature_names == FEATURE_NAMES

    def test_all_features_finite(self, featurizer):
        for smi in [ASPIRIN, CAFFEINE, PARACETAMOL, IBUPROFEN, METHANE, ETHANOL]:
            fv = featurizer.featurize(smi)
            assert np.all(np.isfinite(fv.descriptors)), f"Non-finite feature in {smi}"

    def test_all_features_non_negative(self, featurizer):
        for smi in [ASPIRIN, CAFFEINE, PARACETAMOL, IBUPROFEN]:
            fv = featurizer.featurize(smi)
            # logp_contrib (index 16) can be negative; all counts should be ≥ 0
            counts = fv.descriptors[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]
            ]
            assert np.all(counts >= 0), f"Negative count in {smi}"

    def test_methane_mostly_zero(self, featurizer):
        fv = featurizer.featurize(METHANE)
        x = fv.descriptors
        assert x[2] == 1.0  # 1 aliphatic C
        assert x[3] == 0.0  # 0 aromatic C
        assert x[13] == 0.0  # 0 rings
        assert x[14] == 0.0  # 0 aromatic atoms

    def test_aspirin_mw_approximate(self, featurizer):
        fv = featurizer.featurize(ASPIRIN)
        mw = fv.descriptors[0]
        # Actual MW 180.16; our estimate with H correction ~183-190
        assert 170 <= mw <= 200

    def test_caffeine_mw_approximate(self, featurizer):
        fv = featurizer.featurize(CAFFEINE)
        mw = fv.descriptors[0]
        assert 180 <= mw <= 215

    def test_aspirin_ring_count(self, featurizer):
        fv = featurizer.featurize(ASPIRIN)
        assert fv.descriptors[13] == 1.0  # 1 benzene ring

    def test_caffeine_ring_count(self, featurizer):
        fv = featurizer.featurize(CAFFEINE)
        # Caffeine has 2 fused rings (imidazole + pyrimidine) → 3 ring closures
        assert fv.descriptors[13] >= 2

    def test_aromatic_carbon_benzene(self, featurizer):
        benzene = "c1ccccc1"
        fv = featurizer.featurize(benzene)
        assert fv.descriptors[3] == 6.0  # 6 aromatic C
        assert fv.descriptors[2] == 0.0  # 0 aliphatic C

    def test_aromatic_N_pyridine(self, featurizer):
        fv = featurizer.featurize(PYRIDINE)
        assert fv.descriptors[5] == 1.0  # 1 aromatic N
        assert fv.descriptors[4] == 0.0  # 0 aliphatic N

    def test_chlorobenzene_Cl_count(self, featurizer):
        fv = featurizer.featurize(CHLOROBENZENE)
        assert fv.descriptors[9] == 1.0  # 1 Cl

    def test_ibuprofen_has_carbonyl(self, featurizer):
        fv = featurizer.featurize(IBUPROFEN)
        assert fv.descriptors[15] >= 1  # C=O in carboxylic acid

    def test_paracetamol_hbd_positive(self, featurizer):
        """Paracetamol has NH and OH → at least 1 HBD."""
        fv = featurizer.featurize(PARACETAMOL)
        assert fv.descriptors[11] >= 1  # HBD ≥ 1

    def test_hba_counts_N_O(self, featurizer):
        """HBA = N + O + F atoms."""
        fv = featurizer.featurize(ASPIRIN)
        n_O = fv.descriptors[6]
        n_hba = fv.descriptors[12]
        # aspirin has 4 O and 0 N, 0 F → HBA = 4
        assert n_hba == n_O

    def test_ethanol_oxygen_count(self, featurizer):
        fv = featurizer.featurize(ETHANOL)
        assert fv.descriptors[6] == 1.0  # 1 O

    def test_frac_sp3_benzene_is_zero(self, featurizer):
        benzene = "c1ccccc1"
        fv = featurizer.featurize(benzene)
        assert fv.descriptors[18] == 0.0  # all C are aromatic

    def test_frac_sp3_ethanol_is_one(self, featurizer):
        fv = featurizer.featurize(ETHANOL)
        assert fv.descriptors[18] == 1.0  # all C are aliphatic

    def test_halogen_score_zero_no_halogen(self, featurizer):
        fv = featurizer.featurize(ASPIRIN)
        assert fv.descriptors[19] == 0.0  # no halogens in aspirin

    def test_halogen_score_positive_chlorobenzene(self, featurizer):
        fv = featurizer.featurize(CHLOROBENZENE)
        assert fv.descriptors[19] > 0.0

    def test_logp_contrib_positive_lipophilic(self, featurizer):
        """All-carbon ring should have positive logP contribution."""
        fv = featurizer.featurize("c1ccccc1")  # benzene
        assert fv.descriptors[16] > 0.0

    def test_logp_contrib_negative_polar(self, featurizer):
        """Many heteroatoms → negative logP contribution."""
        fv = featurizer.featurize("NCCO")  # ethanolamine
        assert fv.descriptors[16] < 0.0

    def test_tpsa_nonzero_with_heteroatoms(self, featurizer):
        fv = featurizer.featurize(PARACETAMOL)
        assert fv.descriptors[17] > 0.0

    def test_tpsa_zero_benzene(self, featurizer):
        """Benzene has no polar atoms → TPSA ≈ 0 (only HBD addition, and HBD=0)."""
        fv = featurizer.featurize("c1ccccc1")
        assert fv.descriptors[17] == 0.0

    def test_batch_shape(self, featurizer):
        smiles_list = [ASPIRIN, CAFFEINE, PARACETAMOL]
        mat = featurizer.featurize_batch(smiles_list)
        assert mat.shape == (3, N_FEATURES)

    def test_batch_consistent_with_single(self, featurizer):
        smiles_list = [ASPIRIN, IBUPROFEN]
        mat = featurizer.featurize_batch(smiles_list)
        for i, smi in enumerate(smiles_list):
            single = featurizer.featurize(smi).descriptors
            np.testing.assert_array_equal(mat[i], single)


# ============================================================================
# Low-level parsing helpers
# ============================================================================


class TestParseAtoms:
    def test_methane_one_C(self):
        atoms = _parse_atoms("C")
        assert atoms == [("C", False)]

    def test_ethanol_two_C_one_O(self):
        atoms = _parse_atoms("CCO")
        syms = [s for s, _ in atoms]
        assert syms.count("C") == 2
        assert syms.count("O") == 1

    def test_benzene_six_aromatic_C(self):
        atoms = _parse_atoms("c1ccccc1")
        assert all(is_arom for _, is_arom in atoms)
        assert len(atoms) == 6

    def test_chlorobenzene_Cl_detected(self):
        atoms = _parse_atoms("Clc1ccccc1")
        syms = [s for s, _ in atoms]
        assert "Cl" in syms

    def test_bracket_nh_nitrogen(self):
        atoms = _parse_atoms("C[NH]C")
        syms = [s for s, _ in atoms]
        assert "N" in syms

    def test_bracket_oh_oxygen(self):
        atoms = _parse_atoms("[OH]CC")
        syms = [s for s, _ in atoms]
        assert "O" in syms

    def test_pyridine_aromatic_N(self):
        atoms = _parse_atoms("c1ccncc1")
        n_arom = [(s, a) for s, a in atoms if s == "N"]
        assert n_arom == [("N", True)]


class TestCountRings:
    def test_benzene_one_ring(self):
        assert _count_rings("c1ccccc1") == 1

    def test_naphthalene_two_rings(self):
        assert _count_rings("c1ccc2ccccc2c1") == 2

    def test_linear_no_rings(self):
        assert _count_rings("CCCC") == 0

    def test_spiro_two_rings(self):
        assert _count_rings("C1CCC2(CC1)CCCC2") == 2


class TestCountCarbonyl:
    def test_ketone(self):
        assert _count_carbonyl("CC(=O)C") >= 1

    def test_ester(self):
        assert _count_carbonyl("CC(=O)OC") >= 1

    def test_no_carbonyl(self):
        assert _count_carbonyl("CCO") == 0


class TestCountHBD:
    def test_bracket_nh(self):
        assert _count_hbd("C[NH]C") >= 1

    def test_bracket_oh(self):
        assert _count_hbd("[OH]CC") >= 1

    def test_no_donors_benzene(self):
        assert _count_hbd("c1ccccc1") == 0


# ============================================================================
# NumpyADMEPredictor
# ============================================================================


class TestNumpyADMEPredictor:
    @pytest.fixture
    def predictor(self):
        return NumpyADMEPredictor()

    def test_returns_gnn_result(self, predictor):
        from omega_pbpk.ml_models.numpy_adme import GNNADMEResult

        res = predictor.predict(ASPIRIN)
        assert isinstance(res, GNNADMEResult)

    def test_logP_range(self, predictor):
        for smi in [ASPIRIN, CAFFEINE, IBUPROFEN]:
            res = predictor.predict(smi)
            assert -6.0 <= res.logP <= 8.0, f"logP out of range for {smi}"

    def test_logS_range(self, predictor):
        for smi in [ASPIRIN, CAFFEINE, IBUPROFEN]:
            res = predictor.predict(smi)
            assert -10.0 <= res.logS <= 2.0

    def test_logPeff_range(self, predictor):
        for smi in [ASPIRIN, CAFFEINE, PARACETAMOL]:
            res = predictor.predict(smi)
            assert -3.0 <= res.logPeff <= 2.0

    def test_logfup_range(self, predictor):
        for smi in [ASPIRIN, CAFFEINE, IBUPROFEN]:
            res = predictor.predict(smi)
            assert -4.0 <= res.logfup <= 0.0

    def test_logCLint_range(self, predictor):
        for smi in [ASPIRIN, IBUPROFEN]:
            res = predictor.predict(smi)
            assert -2.0 <= res.logCLint_3A4 <= 2.0

    def test_loghERG_range(self, predictor):
        for smi in [ASPIRIN, CAFFEINE]:
            res = predictor.predict(smi)
            assert -2.0 <= res.loghERG_IC50 <= 3.0

    def test_peff_positive(self, predictor):
        res = predictor.predict(ASPIRIN)
        assert res.peff > 0.0

    def test_fup_between_0_and_1(self, predictor):
        for smi in [ASPIRIN, CAFFEINE, IBUPROFEN]:
            res = predictor.predict(smi)
            assert 0.0 < res.fup <= 1.0

    def test_clint_positive(self, predictor):
        res = predictor.predict(IBUPROFEN)
        assert res.clint_3a4 > 0.0

    def test_herg_ic50_positive(self, predictor):
        res = predictor.predict(ASPIRIN)
        assert res.herg_ic50_uM > 0.0

    def test_lipophilic_has_higher_logP_than_polar(self, predictor):
        """Ibuprofen (logP ~3.5) should have higher logP than paracetamol (logP ~0.46)."""
        lipophilic = predictor.predict(IBUPROFEN)
        polar = predictor.predict(PARACETAMOL)
        assert lipophilic.logP > polar.logP

    def test_lipophilic_has_lower_logS_than_polar(self, predictor):
        """Higher logP → lower solubility (ESOL model)."""
        lipophilic = predictor.predict(IBUPROFEN)
        polar = predictor.predict(PARACETAMOL)
        assert lipophilic.logS < polar.logS

    def test_lipophilic_has_lower_fup_than_polar(self, predictor):
        """More lipophilic → more protein bound → lower fup."""
        lipophilic = predictor.predict(IBUPROFEN)
        polar = predictor.predict(CAFFEINE)
        assert lipophilic.fup < polar.fup

    def test_halogenated_higher_logP(self, predictor):
        """Chlorobenzene should be more lipophilic than benzene."""
        halogened = predictor.predict(CHLOROBENZENE)
        parent = predictor.predict("c1ccccc1")
        assert halogened.logP > parent.logP

    def test_to_dict_keys(self, predictor):
        d = predictor.predict(ASPIRIN).to_dict()
        assert set(d.keys()) == {
            "logP",
            "logS",
            "logPeff",
            "logfup",
            "logCLint_3A4",
            "loghERG_IC50",
            "peff",
            "fup",
            "clint_3a4",
            "herg_ic50_uM",
        }

    def test_batch_prediction_length(self, predictor):
        smiles_list = [ASPIRIN, CAFFEINE, PARACETAMOL, IBUPROFEN]
        results = predictor.predict_batch(smiles_list)
        assert len(results) == 4

    def test_batch_consistent_with_single(self, predictor):
        smiles_list = [ASPIRIN, IBUPROFEN]
        batch = predictor.predict_batch(smiles_list)
        for i, smi in enumerate(smiles_list):
            single = predictor.predict(smi)
            assert batch[i].logP == single.logP
            assert batch[i].logS == single.logS

    def test_logP_consistent_with_logS(self, predictor):
        """Across molecules, higher logP → lower logS (ESOL correlation)."""
        mols = [METHANE, ETHANOL, PARACETAMOL, IBUPROFEN]
        results = [predictor.predict(s) for s in mols]
        logPs = [r.logP for r in results]
        logSs = [r.logS for r in results]
        # Pearson-like: sort by logP and check logS goes the other way overall
        sorted_by_logp = sorted(zip(logPs, logSs, strict=False))
        # The most lipophilic should have lower logS than the most hydrophilic
        assert sorted_by_logp[-1][1] < sorted_by_logp[0][1]

    def test_high_tpsa_low_peff(self, predictor):
        """Paracetamol (higher TPSA via NH, OH) should have lower Peff than ibuprofen."""
        polar = predictor.predict(PARACETAMOL)
        lipophilic = predictor.predict(IBUPROFEN)
        assert polar.logPeff < lipophilic.logPeff

    def test_result_deterministic(self, predictor):
        """Same SMILES always gives same result."""
        r1 = predictor.predict(ASPIRIN)
        r2 = predictor.predict(ASPIRIN)
        assert r1.logP == r2.logP
        assert r1.fup == r2.fup


# ============================================================================
# ADMEPredictor integration — numpy GNN used when no RDKit
# ============================================================================


class TestADMEPredictorNumpyFallback:
    @pytest.fixture
    def adme(self):
        return ADMEPredictor()

    def test_predict_returns_adme_properties(self, adme):
        from omega_pbpk.prediction.adme_predictor import ADMEProperties

        result = adme.predict(ASPIRIN)
        assert isinstance(result, ADMEProperties)

    def test_mw_reasonable(self, adme):
        result = adme.predict(ASPIRIN)
        assert 150 <= result.mw <= 220

    def test_fup_between_0_and_1(self, adme):
        for smi in [ASPIRIN, IBUPROFEN, CAFFEINE]:
            result = adme.predict(smi)
            assert 0.0 < result.fup <= 1.0

    def test_peff_positive(self, adme):
        result = adme.predict(ASPIRIN)
        assert result.peff > 0.0

    def test_clint_3a4_positive(self, adme):
        result = adme.predict(IBUPROFEN)
        assert result.clint_3a4 > 0.0

    def test_herg_ic50_positive(self, adme):
        result = adme.predict(ASPIRIN)
        assert result.herg_ic50_uM > 0.0

    def test_numpy_gnn_method_confidence_is_medium(self, adme):
        """_predict_numpy_gnn directly returns 'medium' (better than 'low' heuristic)."""
        from omega_pbpk.prediction.adme_predictor import _NUMPY_PREDICTOR, HAS_RDKIT

        if HAS_RDKIT or _NUMPY_PREDICTOR is None:
            pytest.skip("numpy GNN only tested when no RDKit")
        result = adme._predict_numpy_gnn(ASPIRIN)
        assert result.confidence == "medium"

    def test_lipophilic_higher_logP_than_polar(self, adme):
        lip = adme.predict(IBUPROFEN)
        pol = adme.predict(PARACETAMOL)
        assert lip.logP > pol.logP

    def test_predict_from_dict(self, adme):
        from omega_pbpk.prediction.adme_predictor import ADMEProperties

        result = adme.predict_from_dict({"logP": 2.5, "mw": 250.0, "fup": 0.3})
        assert isinstance(result, ADMEProperties)
        assert result.logP == 2.5
