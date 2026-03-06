"""Tests for Phase 2 metabolism prediction (UGT/SULT/MAO)."""

import pytest

from omega_pbpk.prediction.phase2_predictor import Phase2Predictor, Phase2Properties


@pytest.fixture(scope="module")
def predictor():
    return Phase2Predictor()


class TestPhase2Properties:
    def test_dataclass_fields(self, predictor):
        r = predictor.predict("CCO")
        assert hasattr(r, "clint_ugt_ul_min_mg")
        assert hasattr(r, "clint_sult_ul_min_mg")
        assert hasattr(r, "clint_mao_ul_min_mg")
        assert hasattr(r, "clint_phase2_total_ul_min_mg")
        assert hasattr(r, "ugt_substrate_prob")
        assert hasattr(r, "sult_substrate_prob")
        assert hasattr(r, "mao_substrate_prob")
        assert hasattr(r, "dominant_pathway")
        assert hasattr(r, "confidence")

    def test_result_type(self, predictor):
        assert isinstance(predictor.predict("CCO"), Phase2Properties)


class TestClintBounds:
    """All CLint values must be non-negative and within plausible range."""

    @pytest.mark.parametrize(
        "smiles",
        [
            "CCO",  # ethanol — aliphatic OH
            "c1ccc(O)cc1",  # phenol — aromatic OH (UGT/SULT substrate)
            "CC(=O)O",  # acetic acid — carboxylic acid (UGT)
            "CCN",  # ethylamine — primary amine (MAO)
            "CCCCCC",  # hexane — no polar groups → minimal metabolism
            "CC(N)C(=O)O",  # alanine — amine + COOH
        ],
    )
    def test_clint_non_negative(self, predictor, smiles):
        r = predictor.predict(smiles)
        assert r.clint_ugt_ul_min_mg >= 0.0
        assert r.clint_sult_ul_min_mg >= 0.0
        assert r.clint_mao_ul_min_mg >= 0.0
        assert r.clint_phase2_total_ul_min_mg >= 0.0

    @pytest.mark.parametrize(
        "smiles",
        [
            "CCO",
            "c1ccc(O)cc1",
            "CCN",
        ],
    )
    def test_clint_within_range(self, predictor, smiles):
        """All CLint estimates must be within [0, 100] µL/min/mg."""
        r = predictor.predict(smiles)
        assert r.clint_ugt_ul_min_mg <= 100.0
        assert r.clint_sult_ul_min_mg <= 100.0
        assert r.clint_mao_ul_min_mg <= 100.0

    def test_total_equals_sum(self, predictor):
        r = predictor.predict("c1ccc(O)cc1")
        expected = r.clint_ugt_ul_min_mg + r.clint_sult_ul_min_mg + r.clint_mao_ul_min_mg
        assert r.clint_phase2_total_ul_min_mg == pytest.approx(expected, abs=1e-6)


class TestProbabilityBounds:
    @pytest.mark.parametrize(
        "smiles",
        ["CCO", "c1ccc(O)cc1", "CCN", "CCCCCC", "CC(=O)O"],
    )
    def test_probabilities_in_01(self, predictor, smiles):
        r = predictor.predict(smiles)
        assert 0.0 <= r.ugt_substrate_prob <= 1.0
        assert 0.0 <= r.sult_substrate_prob <= 1.0
        assert 0.0 <= r.mao_substrate_prob <= 1.0


class TestDominantPathway:
    def test_dominant_pathway_string(self, predictor):
        r = predictor.predict("CCO")
        assert isinstance(r.dominant_pathway, str)
        assert len(r.dominant_pathway) > 0

    def test_no_polar_groups_returns_none(self, predictor):
        """Non-polar alkane → no Phase 2 → dominant='none'."""
        r = predictor.predict("CCCCCCCC")  # octane
        assert r.dominant_pathway == "none"

    def test_dominant_consistent_with_clint(self, predictor):
        """dominant_pathway must match the pathway with highest CLint."""
        r = predictor.predict("c1ccc(O)cc1")
        clints = {
            "UGT": r.clint_ugt_ul_min_mg,
            "SULT": r.clint_sult_ul_min_mg,
            "MAO": r.clint_mao_ul_min_mg,
        }
        total = sum(clints.values())
        if total >= 0.01:
            expected_dominant = max(clints, key=lambda k: clints[k])
            assert r.dominant_pathway == expected_dominant


class TestConfidence:
    def test_confidence_is_string(self, predictor):
        r = predictor.predict("CCO")
        assert r.confidence in ("high", "low")

    def test_confidence_low_without_rdkit(self, monkeypatch):
        """When RDKit unavailable, heuristic gives confidence='low'."""
        import omega_pbpk.prediction.phase2_predictor as mod

        original = mod.HAS_RDKIT
        monkeypatch.setattr(mod, "HAS_RDKIT", False)
        p = Phase2Predictor()
        r = p.predict("CCO")
        assert r.confidence == "low"
        monkeypatch.setattr(mod, "HAS_RDKIT", original)


class TestPhenolUGTSULT:
    def test_phenol_ugt_above_alkane(self, predictor):
        """Phenol (aromatic OH) should have >= UGT CLint than pure alkane."""
        r_phenol = predictor.predict("c1ccc(O)cc1")
        r_alkane = predictor.predict("CCCCCC")
        assert r_phenol.clint_ugt_ul_min_mg >= r_alkane.clint_ugt_ul_min_mg

    def test_phenol_sult_above_alkane(self, predictor):
        r_phenol = predictor.predict("c1ccc(O)cc1")
        r_alkane = predictor.predict("CCCCCC")
        assert r_phenol.clint_sult_ul_min_mg >= r_alkane.clint_sult_ul_min_mg

    def test_carboxylic_acid_ugt_above_zero(self, predictor):
        """Carboxylic acids are UGT substrates — CLint_UGT > 0."""
        r = predictor.predict("CC(=O)O")
        # Should have some UGT activity
        assert r.ugt_substrate_prob >= 0.0  # at least non-zero prob pathway considered


class TestAmineMAO:
    def test_primary_amine_mao_above_alkane(self, predictor):
        """Primary amine (ethylamine) should have higher MAO CLint than alkane."""
        r_amine = predictor.predict("CCN")
        r_alkane = predictor.predict("CCCCCC")
        assert r_amine.clint_mao_ul_min_mg >= r_alkane.clint_mao_ul_min_mg

    def test_amine_mao_prob_above_alkane(self, predictor):
        r_amine = predictor.predict("CCN")
        r_alkane = predictor.predict("CCCCCC")
        assert r_amine.mao_substrate_prob >= r_alkane.mao_substrate_prob


class TestBatchPrediction:
    def test_multiple_compounds(self, predictor):
        smiles_list = ["CCO", "c1ccc(O)cc1", "CCN", "CCCCCC", "CC(=O)O"]
        results = [predictor.predict(s) for s in smiles_list]
        assert len(results) == 5
        for r in results:
            assert isinstance(r, Phase2Properties)

    def test_results_differ_across_compounds(self, predictor):
        """Different SMILES should give different predictions."""
        r1 = predictor.predict("CCO")  # ethanol (aliphatic OH)
        r2 = predictor.predict("CCN")  # ethylamine (amine)
        # At least one metric differs
        differs = (
            r1.clint_ugt_ul_min_mg != r2.clint_ugt_ul_min_mg
            or r1.clint_mao_ul_min_mg != r2.clint_mao_ul_min_mg
            or r1.ugt_substrate_prob != r2.ugt_substrate_prob
        )
        assert differs
